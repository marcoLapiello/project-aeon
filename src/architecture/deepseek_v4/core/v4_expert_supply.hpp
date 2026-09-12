#pragma once

#include "infrastructure/core/aeon_loader.hpp"
#include "infrastructure/core/expert_payload_pool.hpp"
#include "infrastructure/core/tiered_expert_supply.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aeon::core {

class V4ExpertSupplyCoordinator {
public:
    static constexpr uint64_t DEFAULT_DEMOTION_QUEUE_CAPACITY =
        TieredExpertSupply::DEFAULT_DEMOTION_QUEUE_CAPACITY;
    using PendingRegistryTransfer = TieredExpertSupply::PendingTransfer;

    struct LayerPrefetchState {
        std::array<int32_t, 6> vram_slots{-1, -1, -1, -1, -1, -1};
        std::array<uint32_t, 6> global_expert_ids{0, 0, 0, 0, 0, 0};
        std::array<uint64_t, 6> operation_ids{0, 0, 0, 0, 0, 0};
        std::array<bool, 6> is_prefetched{false, false, false, false, false, false};
        std::array<uint32_t, 6> staging_indices{0, 0, 0, 0, 0, 0};
        std::array<bool, 6> io_pending{false, false, false, false, false, false};
        std::array<uint64_t, 6> io_user_data{0, 0, 0, 0, 0, 0};
        std::array<uint32_t, 6> io_request_counts{0, 0, 0, 0, 0, 0};
        TieredExpertSupply::PayloadBatch supply_batch;
    };

    V4ExpertSupplyCoordinator() = default;

    V4ExpertSupplyCoordinator(const V4ExpertSupplyCoordinator&) = delete;
    V4ExpertSupplyCoordinator& operator=(const V4ExpertSupplyCoordinator&) = delete;

    void configure(
        AeonModelLoader* aeon_loader,
        ExpertPayloadPool* payload_pool,
        HostExpertPool* host_pool,
        ExpertRegistry* expert_registry,
        PrefetchStagingArena* prefetch_staging,
        SupplyTelemetry* supply_telemetry,
        aeon::io::DirectIOReader* direct_io_reader,
        std::unordered_map<uint64_t, aeon::io::DirectIOCompletion>* direct_io_completions,
        uint64_t* next_direct_io_id,
        hipStream_t compute_stream,
        hipStream_t sdma_stream,
        hipStream_t sdma_cold_stream,
        hipStream_t demotion_stream,
        size_t expert_payload_bytes,
        uint64_t demotion_queue_capacity
    ) {
        if (aeon_loader == nullptr || expert_registry == nullptr) {
            throw std::invalid_argument("V4ExpertSupplyCoordinator: model source and registry are required");
        }

        const bool has_mapped_experts = aeon_loader->total_dense_tensors() > 0;
        const uint32_t experts_per_layer = aeon_loader->experts_per_layer();
        TieredExpertSupply::PayloadSource source;
        if (has_mapped_experts) {
            source.direct_fd = aeon_loader->expert_direct_fd();
        }
        source.locate = [aeon_loader, experts_per_layer](uint32_t global_expert_id) {
            if (experts_per_layer == 0) {
                throw std::runtime_error("V4ExpertSupplyCoordinator: invalid expert catalog width");
            }
            const uint32_t layer_id = global_expert_id / experts_per_layer;
            const uint32_t expert_id = global_expert_id % experts_per_layer;
            const auto location = aeon_loader->get_expert_location(layer_id, expert_id);
            return TieredExpertSupply::PayloadLocation{
                location.file_offset,
                location.byte_length
            };
        };
        source.host_payload = [aeon_loader, has_mapped_experts, experts_per_layer](uint32_t global_expert_id) {
            if (!has_mapped_experts || experts_per_layer == 0) {
                return static_cast<const uint8_t*>(nullptr);
            }
            return aeon_loader->get_expert_data(
                global_expert_id / experts_per_layer,
                global_expert_id % experts_per_layer);
        };

        expert_registry_ = expert_registry;
        supply_.configure(
            std::move(source),
            payload_pool,
            host_pool,
            expert_registry,
            prefetch_staging,
            supply_telemetry,
            direct_io_reader,
            direct_io_completions,
            next_direct_io_id,
            compute_stream,
            sdma_stream,
            sdma_cold_stream,
            demotion_stream,
            expert_payload_bytes,
            demotion_queue_capacity
        );
    }

    size_t expert_payload_bytes() const noexcept {
        return supply_.expert_payload_bytes();
    }

    uint64_t demotion_queue_capacity() const noexcept {
        return supply_.demotion_queue_capacity();
    }

    LayerPrefetchState dispatch_layer_prefetch(
        uint32_t target_l,
        uint32_t pos,
        const std::vector<int32_t>& topk_experts,
        std::vector<uint32_t>& leased_experts
    ) {
        constexpr size_t ROUTED_EXPERTS = 6;
        if (topk_experts.size() != ROUTED_EXPERTS) {
            throw std::invalid_argument(
                "V4ExpertSupplyCoordinator: expected exactly six routed experts");
        }
        if (expert_registry_ == nullptr) {
            throw std::logic_error("V4ExpertSupplyCoordinator: coordinator is not configured");
        }

        std::vector<TieredExpertSupply::PayloadRequest> requests;
        requests.reserve(ROUTED_EXPERTS);
        const uint32_t staging_offset = (target_l % 2) * ROUTED_EXPERTS;
        for (size_t index = 0; index < ROUTED_EXPERTS; ++index) {
            if (topk_experts[index] < 0) {
                throw std::invalid_argument(
                    "V4ExpertSupplyCoordinator: routed expert ID cannot be negative");
            }
            requests.push_back(TieredExpertSupply::PayloadRequest{
                expert_registry_->get_global_id(
                    target_l, static_cast<uint32_t>(topk_experts[index])),
                staging_offset + static_cast<uint32_t>(index)
            });
        }

        LayerPrefetchState state;
        state.supply_batch = supply_.dispatch(requests, pos, leased_experts);
        if (state.supply_batch.transfers.size() != ROUTED_EXPERTS) {
            throw std::logic_error(
                "V4ExpertSupplyCoordinator: supply returned an incomplete routed batch");
        }
        sync_state(state);
        return state;
    }

    void materialize_layer_prefetch(LayerPrefetchState& state) {
        supply_.materialize(state.supply_batch);
        sync_state(state);
    }

    PendingRegistryTransfer& ensure_registry_transfer(uint64_t operation_id, uint32_t gid) {
        return supply_.ensure_registry_transfer(operation_id, gid);
    }

    PendingRegistryTransfer* find_registry_transfer(uint64_t operation_id) {
        return supply_.find_registry_transfer(operation_id);
    }

    const PendingRegistryTransfer* find_registry_transfer(uint64_t operation_id) const {
        return supply_.find_registry_transfer(operation_id);
    }

    void schedule_demotion(const ExpertRequestReservation& request) {
        supply_.schedule_demotion(request);
    }

    void wait_for_demotion_dependency(uint64_t operation_id, hipStream_t stream) {
        supply_.wait_for_demotion_dependency(operation_id, stream);
    }

    void record_h2d_event(
        uint64_t operation_id,
        hipStream_t stream,
        uint32_t staging_idx,
        bool has_staging,
        int32_t source_slot,
        int32_t destination_slot
    ) {
        supply_.record_h2d_event(
            operation_id, stream, staging_idx, has_staging, source_slot, destination_slot);
    }

    void bind_staging(uint64_t operation_id, uint32_t staging_idx) {
        supply_.bind_staging(operation_id, staging_idx);
    }

    void mark_gpu_readiness_wait_start(uint64_t operation_id) {
        supply_.mark_gpu_readiness_wait_start(operation_id);
    }

    void mark_registry_request_failed(uint64_t operation_id, const std::string& reason) {
        supply_.mark_registry_request_failed(operation_id, reason);
    }

    void reap_registry_transfers() {
        supply_.reap_registry_transfers();
    }

    void record_supply_request(const ExpertRequestReservation& request) {
        supply_.record_supply_request(request);
    }

    void observe_supply_occupancy(ExpertTier source_tier) {
        supply_.observe_supply_occupancy(source_tier);
    }

    void clear() noexcept {
        supply_.clear();
        expert_registry_ = nullptr;
    }

private:
    void sync_state(LayerPrefetchState& state) const {
        for (size_t index = 0; index < state.supply_batch.transfers.size(); ++index) {
            const auto& transfer = state.supply_batch.transfers[index];
            state.vram_slots[index] = transfer.vram_slot;
            state.global_expert_ids[index] = transfer.global_expert_id;
            state.operation_ids[index] = transfer.operation_id;
            state.is_prefetched[index] = transfer.is_prefetched;
            state.staging_indices[index] = transfer.staging_idx;
            state.io_pending[index] = transfer.io_pending;
            state.io_user_data[index] = transfer.io_user_data;
            state.io_request_counts[index] = transfer.io_request_count;
        }
    }

    TieredExpertSupply supply_;
    ExpertRegistry* expert_registry_{nullptr};
};

} // namespace aeon::core