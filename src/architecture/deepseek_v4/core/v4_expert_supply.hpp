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

    // One routed expert per token, the `k` of the top-k selection. An expert is
    // selected at most once per token (the router's top-k is a set), so a token's
    // six entries are distinct; only across tokens does an expert repeat.
    static constexpr uint32_t ROUTED_EXPERTS = 6;

    // The state of one dispatch. It is **the layer's deduplicated set**, not one
    // token's six experts: the `C = 1` case (decode) deduplicates to exactly six,
    // and a chunk of `C` tokens collapses its `6C` requests to the layer's union
    // (Step 6 D2). The parallel per-expert arrays are indexed by position in that
    // distinct set; `token_indices[t][k]` is the distinct index of token `t`'s
    // `k`-th expert, so a slot is stage-once and read by every token that chose it
    // without ever permuting the slot-sum order (step §7).
    struct LayerPrefetchState {
        std::vector<int32_t> vram_slots;
        std::vector<uint32_t> global_expert_ids;
        std::vector<uint64_t> operation_ids;
        std::vector<uint8_t> is_prefetched;
        std::vector<uint32_t> staging_indices;
        std::vector<uint8_t> io_pending;
        std::vector<uint64_t> io_user_data;
        std::vector<uint32_t> io_request_counts;
        TieredExpertSupply::PayloadBatch supply_batch;

        // The batch shape. Decode leaves this as one token whose six indices are
        // `0..5`; a chunk fills `token_indices` with one row per token.
        uint32_t first_position{0};
        uint32_t token_count{0};
        std::vector<std::array<uint32_t, ROUTED_EXPERTS>> token_indices;

        size_t expert_count() const noexcept { return supply_batch.transfers.size(); }
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

    // io_uring submission cost, split out from the rest of the dispatch path.
    uint64_t direct_io_submit_ns() const noexcept { return supply_.direct_io_submit_ns(); }
    uint64_t direct_io_requests_submitted() const noexcept {
        return supply_.direct_io_requests_submitted();
    }
    uint64_t direct_io_submit_calls() const noexcept {
        return supply_.direct_io_submit_calls();
    }

    // The transfer split (see `TieredExpertSupply`): host time blocked on NVMe
    // completions, CPU time to submit H2D, and host time blocked on the H2D
    // copies landing. Both phases drive the same supply, so one set serves the
    // prefill and decode attribution alike.
    uint64_t io_wait_ns() const noexcept { return supply_.io_wait_ns(); }
    uint64_t h2d_enqueue_ns() const noexcept { return supply_.h2d_enqueue_ns(); }
    uint64_t h2d_drain_ns() const noexcept { return supply_.h2d_drain_ns(); }
    uint64_t h2d_drain_calls() const noexcept { return supply_.h2d_drain_calls(); }
    uint64_t dispatch_cpu_ns() const noexcept { return supply_.dispatch_cpu_ns(); }
    // Staging slots freed by the completion path instead of a boundary block (P2.2).
    uint64_t staging_released_on_completion() const noexcept {
        return supply_.staging_released_on_completion();
    }
    uint64_t copies_pumped() const noexcept { return supply_.copies_pumped(); }
    void reset_transfer_counters() noexcept { supply_.reset_transfer_counters(); }

    LayerPrefetchState dispatch_layer_prefetch(
        uint32_t target_l,
        uint32_t pos,
        const std::vector<int32_t>& topk_experts,
        std::vector<uint32_t>& leased_experts
    ) {
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
        state.first_position = pos;
        state.token_count = 1;
        state.token_indices.assign(1, std::array<uint32_t, ROUTED_EXPERTS>{0, 1, 2, 3, 4, 5});
        sync_state(state);
        return state;
    }

    // The layer-wide dispatch (Step 6 item 4): a chunk's `6C` requests issued as
    // **one set**. The requests are deduplicated to the layer's distinct experts,
    // each distinct expert is staged **once** and read by every token that selected
    // it, and the per-token index map is what lets the caller rebuild each token's
    // six expert slots without disturbing the slot-sum order.
    //
    // Staging indices are the distinct-set positions `0..D-1`. `D <= 6C`, and the
    // host sizes the arena to at least that (Step 6 D4), so no transfer waits for a
    // slot; a caller that under-sizes the arena fails loudly here rather than
    // silently colliding two experts on one slot.
    LayerPrefetchState dispatch_layer_prefetch_batch(
        uint32_t target_l,
        uint32_t first_position,
        const std::vector<std::vector<int32_t>>& topk_experts,
        std::vector<uint32_t>& leased_experts
    ) {
        if (topk_experts.empty()) {
            throw std::invalid_argument(
                "V4ExpertSupplyCoordinator: a layer-wide dispatch needs at least one token");
        }
        if (expert_registry_ == nullptr) {
            throw std::logic_error("V4ExpertSupplyCoordinator: coordinator is not configured");
        }

        std::vector<uint32_t> distinct;
        std::unordered_map<uint32_t, uint32_t> distinct_index;
        std::vector<std::array<uint32_t, ROUTED_EXPERTS>> token_indices(topk_experts.size());

        for (size_t token = 0; token < topk_experts.size(); ++token) {
            if (topk_experts[token].size() != ROUTED_EXPERTS) {
                throw std::invalid_argument(
                    "V4ExpertSupplyCoordinator: every token must route to six experts");
            }
            for (uint32_t k = 0; k < ROUTED_EXPERTS; ++k) {
                if (topk_experts[token][k] < 0) {
                    throw std::invalid_argument(
                        "V4ExpertSupplyCoordinator: routed expert ID cannot be negative");
                }
                const uint32_t gid = expert_registry_->get_global_id(
                    target_l, static_cast<uint32_t>(topk_experts[token][k]));
                auto found = distinct_index.find(gid);
                if (found == distinct_index.end()) {
                    found = distinct_index.emplace(gid, static_cast<uint32_t>(distinct.size())).first;
                    distinct.push_back(gid);
                }
                token_indices[token][k] = found->second;
            }
        }

        std::vector<TieredExpertSupply::PayloadRequest> requests;
        requests.reserve(distinct.size());
        for (uint32_t index = 0; index < distinct.size(); ++index) {
            requests.push_back(TieredExpertSupply::PayloadRequest{distinct[index], index});
        }

        LayerPrefetchState state;
        state.supply_batch = supply_.dispatch(requests, first_position, leased_experts);
        if (state.supply_batch.transfers.size() != distinct.size()) {
            throw std::logic_error(
                "V4ExpertSupplyCoordinator: supply returned an incomplete layer-wide batch");
        }
        state.first_position = first_position;
        state.token_count = static_cast<uint32_t>(topk_experts.size());
        state.token_indices = std::move(token_indices);
        sync_state(state);
        return state;
    }

    void materialize_layer_prefetch(LayerPrefetchState& state) {
        supply_.materialize(state.supply_batch);
        sync_state(state);
    }

    // Streaming prefill (Step 6 item 6): load a whole layer's expert set. Unlike
    // `dispatch_layer_prefetch_batch` this is not driven by a routing result — the
    // layer is loaded whole, because a prefill chunk touches ~255 of 256 experts and
    // the set is therefore **known rather than guessed** (plan §2). Staging indices
    // are `staging_base + position` in `local_expert_ids`, so the caller sizes the
    // arena to the layer and picks the bank: the sweep alternates banks by layer
    // parity, which is what lets one layer's copies stay in flight through its body
    // while the next layer's reads fill the other bank.
    LayerPrefetchState dispatch_layer_stream(
        uint32_t layer,
        const std::vector<uint32_t>& local_expert_ids,
        std::vector<uint32_t>& leased_experts,
        uint32_t staging_base = 0
    ) {
        if (expert_registry_ == nullptr) {
            throw std::logic_error("V4ExpertSupplyCoordinator: coordinator is not configured");
        }
        std::vector<TieredExpertSupply::PayloadRequest> requests;
        requests.reserve(local_expert_ids.size());
        for (uint32_t index = 0; index < local_expert_ids.size(); ++index) {
            if (local_expert_ids[index] >= expert_registry_->experts_per_layer) {
                throw std::out_of_range(
                    "V4ExpertSupplyCoordinator: streamed expert id is outside its layer");
            }
            requests.push_back(TieredExpertSupply::PayloadRequest{
                expert_registry_->get_global_id(layer, local_expert_ids[index]),
                staging_base + index
            });
        }
        LayerPrefetchState state;
        state.supply_batch = supply_.dispatch(requests, layer, leased_experts);
        if (state.supply_batch.transfers.size() != local_expert_ids.size()) {
            throw std::logic_error(
                "V4ExpertSupplyCoordinator: supply returned an incomplete layer stream");
        }
        state.first_position = layer;
        state.token_count = 0;
        sync_state(state);
        return state;
    }

    // Hands the staging slots a streamed batch borrowed back to the arena. The
    // executor releases its own in `on_routed_consumed`; the sweep has no such hook.
    //
    // The sweep defers this past the layer body (see `V4PrefillSweep`): the copy is
    // left in flight with no host wait, the body's MoE dispatch joins the pending
    // transfer and the executor orders the weights behind it per expert, and this runs
    // at the layer boundary — where the driver has synchronized the compute stream, so
    // the per-slot event sync inside is instant and charges ~nothing to
    // `h2d_drain_ns_`.
    void finish_streamed_batch(LayerPrefetchState& state) {
        supply_.release_streamed_staging(state.supply_batch);
        sync_state(state);
    }

    // The non-blocking half of `materialize_layer_prefetch` — enqueue a copy for each
    // expert of `state` whose reads have all landed, and return without waiting (P2.3).
    // The per-token pump calls this on the **lookahead** layer, so its VRAM block is
    // filled during the previous layer's body instead of at the next boundary.
    size_t pump_layer_prefetch(LayerPrefetchState& state) {
        const size_t enqueued = supply_.materialize_available(state.supply_batch);
        if (enqueued > 0) sync_state(state);
        return enqueued;
    }

    void reap_registry_transfers() {
        supply_.reap_registry_transfers();
    }

    // The staging arena's occupancy by pipeline stage — the corridor's fill, as
    // opposed to its size. Read at a layer boundary by the sweep's readout.
    PrefetchStagingArena::StateCounts staging_state_counts() const {
        return supply_.staging_state_counts();
    }

    uint32_t staging_in_use_slots() const noexcept { return supply_.staging_in_use_slots(); }

    void mark_gpu_readiness_wait_start(uint64_t operation_id) {
        supply_.mark_gpu_readiness_wait_start(operation_id);
    }

    void clear() noexcept {
        supply_.clear();
        expert_registry_ = nullptr;
    }

private:
    void sync_state(LayerPrefetchState& state) const {
        const size_t count = state.supply_batch.transfers.size();
        state.vram_slots.assign(count, -1);
        state.global_expert_ids.assign(count, 0);
        state.operation_ids.assign(count, 0);
        state.is_prefetched.assign(count, 0);
        state.staging_indices.assign(count, 0);
        state.io_pending.assign(count, 0);
        state.io_user_data.assign(count, 0);
        state.io_request_counts.assign(count, 0);
        for (size_t index = 0; index < count; ++index) {
            const auto& transfer = state.supply_batch.transfers[index];
            state.vram_slots[index] = transfer.vram_slot;
            state.global_expert_ids[index] = transfer.global_expert_id;
            state.operation_ids[index] = transfer.operation_id;
            state.is_prefetched[index] = transfer.is_prefetched ? 1u : 0u;
            state.staging_indices[index] = transfer.staging_idx;
            state.io_pending[index] = transfer.io_pending ? 1u : 0u;
            state.io_user_data[index] = transfer.io_user_data;
            state.io_request_counts[index] = transfer.io_request_count;
        }
    }

    TieredExpertSupply supply_;
    ExpertRegistry* expert_registry_{nullptr};
};

} // namespace aeon::core