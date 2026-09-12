#pragma once

#include "infrastructure/core/expert_payload_pool.hpp"
#include "infrastructure/core/expert_registry.hpp"
#include "infrastructure/core/host_expert_pool.hpp"
#include "infrastructure/core/prefetch_staging.hpp"
#include "infrastructure/core/supply_telemetry.hpp"
#include "infrastructure/io/direct_io_reader.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aeon::core {

class TieredExpertSupply {
public:
    static constexpr uint64_t DEFAULT_DEMOTION_QUEUE_CAPACITY = 2;

    struct PayloadLocation {
        uint64_t file_offset{0};
        size_t byte_length{0};
    };

    struct PayloadSource {
        int direct_fd{-1};
        std::function<PayloadLocation(uint32_t)> locate;
        std::function<const uint8_t*(uint32_t)> host_payload;

        bool supports_direct_io() const noexcept {
            return direct_fd >= 0 && static_cast<bool>(locate);
        }
    };

    struct PayloadRequest {
        uint32_t global_expert_id{0};
        uint32_t staging_idx{0};
    };

    struct PayloadTransfer {
        uint32_t global_expert_id{0};
        uint64_t operation_id{0};
        int32_t vram_slot{-1};
        bool is_prefetched{false};
        uint32_t staging_idx{0};
        bool io_pending{false};
        uint64_t io_user_data{0};
        uint32_t io_request_count{0};
    };

    struct PayloadBatch {
        std::vector<PayloadTransfer> transfers;
    };

    struct PendingTransfer {
        uint64_t operation_id{0};
        uint32_t global_expert_id{0};
        uint32_t demoted_expert_id{0};
        ExpertTier source_tier{ExpertTier::COLD_NVME};
        SupplyTelemetryPhase phase{SupplyTelemetryPhase::Decode};
        std::chrono::steady_clock::time_point h2d_enqueued_at{};
        hipEvent_t demotion_event{nullptr};
        hipEvent_t h2d_event{nullptr};
        uint32_t staging_idx{0};
        bool has_staging{false};
        bool h2d_submitted{false};
        int32_t demotion_source_slot{-1};
        int32_t demotion_destination_slot{-1};
        int32_t demotion_staging_idx{-1};
        bool demotion_uses_staging{false};
        bool demotion_submitted{false};
        int32_t h2d_source_slot{-1};
        int32_t h2d_destination_slot{-1};
        std::chrono::steady_clock::time_point io_submitted_at{};
        uint64_t nvme_read_service_ns{0};
        uint64_t nvme_completion_wait_ns{0};
        uint64_t staging_wait_ns{0};
        uint64_t staging_reuse_wait_ns{0};
        std::chrono::steady_clock::time_point staging_acquired_at{};
        std::chrono::steady_clock::time_point gpu_wait_started_at{};
        bool request_failed{false};
        std::string failure_reason;
    };

    TieredExpertSupply() = default;

    ~TieredExpertSupply() {
        clear();
    }

    TieredExpertSupply(const TieredExpertSupply&) = delete;
    TieredExpertSupply& operator=(const TieredExpertSupply&) = delete;

    void configure(
        PayloadSource source,
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
        clear();
        source_ = std::move(source);
        payload_pool_ = payload_pool;
        host_pool_ = host_pool;
        expert_registry_ = expert_registry;
        prefetch_staging_ = prefetch_staging;
        supply_telemetry_ = supply_telemetry;
        direct_io_reader_ = direct_io_reader;
        direct_io_completions_ = direct_io_completions;
        next_direct_io_id_ = next_direct_io_id;
        compute_stream_ = compute_stream;
        sdma_stream_ = sdma_stream;
        sdma_cold_stream_ = sdma_cold_stream;
        demotion_stream_ = demotion_stream;
        expert_payload_bytes_ = expert_payload_bytes;
        demotion_queue_capacity_ = demotion_queue_capacity;
    }

    size_t expert_payload_bytes() const noexcept {
        return expert_payload_bytes_;
    }

    uint64_t demotion_queue_capacity() const noexcept {
        return demotion_queue_capacity_;
    }

    PayloadBatch dispatch(
        const std::vector<PayloadRequest>& requests,
        uint64_t current_step,
        std::vector<uint32_t>& leased_experts
    ) {
        PayloadBatch batch;
        batch.transfers.resize(requests.size());
        bool submitted_direct_io = false;

        std::vector<size_t> request_order(requests.size());
        for (size_t index = 0; index < requests.size(); ++index) {
            request_order[index] = index;
        }
        auto reservation_priority = [&](size_t request_index) {
            const uint32_t gid = requests[request_index].global_expert_id;
            const auto& entry = expert_registry_->catalog[gid];
            if (entry.owner == ExpertTier::HOT_VRAM &&
                entry.operation == ExpertOperation::NONE &&
                entry.publication == ExpertPublication::PUBLISHED) {
                return 0;
            }
            if (entry.owner == ExpertTier::HOT_VRAM) {
                return 1;
            }
            return 2;
        };
        std::stable_sort(request_order.begin(), request_order.end(), [&](size_t left, size_t right) {
            return reservation_priority(left) < reservation_priority(right);
        });

        for (const size_t request_index : request_order) {
            const auto& payload_request = requests[request_index];
            const uint32_t expert_id = payload_request.global_expert_id;
            ExpertRequestReservation request;
            for (;;) {
                request = expert_registry_->reserve_request(
                    expert_id, current_step, demotion_queue_capacity_);
                if (request.kind != ExpertRequestKind::PENDING) {
                    break;
                }

                const auto* pending = find_registry_transfer(request.operation_id);
                const auto& pending_entry = expert_registry_->catalog[request.global_expert_id];
                if (pending_entry.operation != ExpertOperation::DEMOTION_PENDING) {
                    break;
                }
                const hipError_t h2d_status = pending != nullptr && pending->h2d_event != nullptr
                    ? hipEventQuery(pending->h2d_event)
                    : hipErrorNotReady;
                if (h2d_status == hipSuccess) {
                    expert_registry_->release_lease(request.global_expert_id);
                    reap_registry_transfers();
                    continue;
                }
                if (h2d_status != hipErrorNotReady) {
                    throw std::runtime_error(
                        "TieredExpertSupply: failed to query an optional demotion dependency "
                        "(expert=" + std::to_string(expert_id) +
                        ", operation=" + std::to_string(request.operation_id) + ")"
                    );
                }
                expert_registry_->release_lease(request.global_expert_id);
                throw std::runtime_error(
                    "TieredExpertSupply: request encountered an in-flight optional demotion; "
                    "request-path CPU synchronization is forbidden "
                    "(expert=" + std::to_string(expert_id) +
                    ", operation=" + std::to_string(request.operation_id) + ")"
                );
            }

            auto& state = batch.transfers[request_index];
            state.global_expert_id = request.global_expert_id;
            state.operation_id = request.operation_id;
            state.vram_slot = request.vram_slot;

            record_supply_request(request);
            observe_supply_occupancy(request.source_tier);
            leased_experts.push_back(request.global_expert_id);

            if (request.kind == ExpertRequestKind::HOT_HIT) {
                continue;
            }

            if (request.kind == ExpertRequestKind::PENDING) {
                const auto* pending = find_registry_transfer(request.operation_id);
                if (pending == nullptr || !pending->h2d_submitted) {
                    throw std::runtime_error(
                        "TieredExpertSupply: duplicate request joined before its transfer was submitted "
                        "(expert=" + std::to_string(expert_id) +
                        ", gid=" + std::to_string(request.global_expert_id) +
                        ", operation=" + std::to_string(request.operation_id) + ")"
                    );
                }
                state.is_prefetched = pending->has_staging;
                state.staging_idx = pending->staging_idx;
                continue;
            }

            try {
                schedule_demotion(request);
                const bool source_is_warm = request.source_tier == ExpertTier::WARM_HOST;
                const int32_t source_slot = request.source_host_slot;
                const bool can_direct_read = request.source_tier == ExpertTier::COLD_NVME &&
                                             direct_io_reader_ != nullptr &&
                                             source_.supports_direct_io() &&
                                             prefetch_staging_ != nullptr;
                ensure_registry_transfer(request.operation_id, request.global_expert_id);

                if (can_direct_read) {
                    bind_staging(request.operation_id, payload_request.staging_idx);
                    const auto location = source_.locate(request.global_expert_id);
                    if (location.byte_length != expert_payload_bytes_) {
                        throw std::runtime_error(
                            "TieredExpertSupply: payload source returned an incompatible payload size");
                    }
                    const uint64_t request_id = *next_direct_io_id_;

                    prefetch_staging_->begin_io(payload_request.staging_idx);
                    const size_t request_count = direct_io_reader_->submit_read_chunks(
                        source_.direct_fd,
                        prefetch_staging_->get_slot_ptr(payload_request.staging_idx),
                        location.byte_length,
                        location.file_offset,
                        request_id
                    );
                    *next_direct_io_id_ += request_count;

                    state.staging_idx = payload_request.staging_idx;
                    state.io_pending = true;
                    state.io_user_data = request_id;
                    state.io_request_count = static_cast<uint32_t>(request_count);
                    submitted_direct_io = true;
                } else if (source_is_warm && host_pool_ && source_slot >= 0 &&
                           host_pool_->is_slot_pinned(static_cast<uint32_t>(source_slot))) {
                    bind_staging(request.operation_id, payload_request.staging_idx);
                    prefetch_staging_->begin_direct_transfer(payload_request.staging_idx);
                    wait_for_demotion_dependency(request.operation_id, sdma_stream_);
                    payload_pool_->upload_from_host_expert(
                        static_cast<uint32_t>(request.vram_slot),
                        host_pool_->get_expert_slot_ptr(static_cast<uint32_t>(source_slot)),
                        sdma_stream_
                    );
                    check_hip(
                        hipEventRecord(prefetch_staging_->events[payload_request.staging_idx], sdma_stream_),
                        "hipEventRecord(Warm H2D)");
                    record_h2d_event(
                        request.operation_id, sdma_stream_, payload_request.staging_idx, true,
                        source_slot, request.vram_slot);

                    state.is_prefetched = true;
                    state.staging_idx = payload_request.staging_idx;
                } else if (source_is_warm) {
                    auto* transfer = find_registry_transfer(request.operation_id);
                    transfer->staging_idx = payload_request.staging_idx;
                    transfer->has_staging = true;
                    prefetch_staging_->stage_payload(
                        payload_request.staging_idx,
                        host_pool_->get_expert_slot_ptr(static_cast<uint32_t>(source_slot))
                    );
                    const uint8_t* pinned_payload =
                        prefetch_staging_->get_slot_ptr(payload_request.staging_idx);
                    prefetch_staging_->begin_gpu_transfer(payload_request.staging_idx);
                    wait_for_demotion_dependency(request.operation_id, sdma_stream_);
                    payload_pool_->upload_from_host_expert(
                        static_cast<uint32_t>(request.vram_slot), pinned_payload, sdma_stream_);
                    check_hip(
                        hipEventRecord(
                            prefetch_staging_->events[payload_request.staging_idx], sdma_stream_),
                        "hipEventRecord(Warm staged H2D)");
                    record_h2d_event(
                        request.operation_id, sdma_stream_, payload_request.staging_idx, true,
                        static_cast<int32_t>(payload_request.staging_idx), request.vram_slot);

                    state.is_prefetched = true;
                    state.staging_idx = payload_request.staging_idx;
                } else {
                    const uint8_t* src_ptr = source_.host_payload
                        ? source_.host_payload(request.global_expert_id)
                        : nullptr;

                    if (src_ptr && prefetch_staging_) {
                        bind_staging(request.operation_id, payload_request.staging_idx);
                        prefetch_staging_->stage_payload(payload_request.staging_idx, src_ptr);
                        const uint8_t* pinned_payload =
                            prefetch_staging_->get_slot_ptr(payload_request.staging_idx);
                        prefetch_staging_->begin_gpu_transfer(payload_request.staging_idx);
                        wait_for_demotion_dependency(request.operation_id, sdma_stream_);
                        payload_pool_->upload_from_host_expert(
                            static_cast<uint32_t>(request.vram_slot), pinned_payload, sdma_stream_);
                        check_hip(
                            hipEventRecord(
                                prefetch_staging_->events[payload_request.staging_idx], sdma_stream_),
                            "hipEventRecord(mapped H2D)");
                        record_h2d_event(
                            request.operation_id, sdma_stream_, payload_request.staging_idx, true,
                            static_cast<int32_t>(payload_request.staging_idx), request.vram_slot);

                        state.is_prefetched = true;
                        state.staging_idx = payload_request.staging_idx;
                    } else if (src_ptr) {
                        wait_for_demotion_dependency(request.operation_id, compute_stream_);
                        payload_pool_->upload_from_host_expert(
                            static_cast<uint32_t>(request.vram_slot), src_ptr, compute_stream_);
                        record_h2d_event(
                            request.operation_id, compute_stream_, 0, false,
                            -1, request.vram_slot);
                    } else {
                        throw std::runtime_error(
                            "TieredExpertSupply: host payload unavailable for requested expert");
                    }
                }
            } catch (...) {
                mark_registry_request_failed(request.operation_id, "transfer_submission_failure");
                reap_registry_transfers();
                throw;
            }
        }

        if (submitted_direct_io) {
            try {
                direct_io_reader_->submit_pending_reads();
            } catch (...) {
                for (auto& state : batch.transfers) {
                    if (state.io_pending) {
                        mark_registry_request_failed(
                            state.operation_id, "nvme_submit_failure");
                    }
                }
                throw;
            }
            const auto submitted_at = std::chrono::steady_clock::now();
            for (const auto& state : batch.transfers) {
                if (!state.io_pending) continue;
                auto* transfer = find_registry_transfer(state.operation_id);
                if (transfer != nullptr) {
                    transfer->io_submitted_at = submitted_at;
                }
            }
        }
        return batch;
    }

    void materialize(PayloadBatch& batch) {
        for (auto& state : batch.transfers) {
            if (!state.io_pending) {
                continue;
            }

            const size_t request_count = state.io_request_count;
            for (size_t chunk = 0; chunk < request_count; ++chunk) {
                const uint64_t request_id = state.io_user_data + chunk;
                if (direct_io_completions_ == nullptr) {
                    mark_registry_request_failed(state.operation_id, "nvme_completion_store_unavailable");
                    throw std::runtime_error(
                        "TieredExpertSupply: direct I/O completion store is unavailable");
                }
                auto completion_it = direct_io_completions_->find(request_id);
                const bool waited_for_completion = completion_it == direct_io_completions_->end();
                const auto wait_started_at = std::chrono::steady_clock::now();
                while (completion_it == direct_io_completions_->end()) {
                    if (!direct_io_reader_) {
                        mark_registry_request_failed(
                            state.operation_id, "nvme_reader_unavailable");
                        throw std::runtime_error("TieredExpertSupply: direct I/O request has no reader");
                    }
                    aeon::io::DirectIOCompletion completion;
                    try {
                        completion = direct_io_reader_->wait_for_completion();
                    } catch (...) {
                        mark_registry_request_failed(
                            state.operation_id, "nvme_completion_wait_failure");
                        throw;
                    }
                    direct_io_completions_->emplace(completion.user_data, completion);
                    completion_it = direct_io_completions_->find(request_id);
                }

                const auto completion = completion_it->second;
                direct_io_completions_->erase(completion_it);
                const auto completed_at = std::chrono::steady_clock::now();
                if (auto* transfer = find_registry_transfer(state.operation_id)) {
                    if (transfer->io_submitted_at.time_since_epoch().count() != 0) {
                        transfer->nvme_read_service_ns += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                completed_at - transfer->io_submitted_at).count());
                    }
                    if (waited_for_completion) {
                        transfer->nvme_completion_wait_ns += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                completed_at - wait_started_at).count());
                    }
                }
                if (completion.result < 0) {
                    mark_registry_request_failed(state.operation_id, "nvme_read_failure");
                    throw std::runtime_error(
                        "TieredExpertSupply: direct expert read failed: " +
                        std::string(strerror(-completion.result)));
                }
                const size_t chunk_offset = chunk * aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
                if (chunk_offset >= expert_payload_bytes_) {
                    mark_registry_request_failed(state.operation_id, "nvme_invalid_chunk");
                    throw std::runtime_error(
                        "TieredExpertSupply: direct expert read returned an invalid chunk index");
                }
                const size_t expected_bytes = std::min(
                    aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES,
                    expert_payload_bytes_ - chunk_offset
                );
                if (completion.result != static_cast<int32_t>(expected_bytes)) {
                    mark_registry_request_failed(state.operation_id, "nvme_short_read");
                    throw std::runtime_error(
                        "TieredExpertSupply: direct expert read returned a short payload");
                }
            }

            const uint32_t staging_idx = state.staging_idx;
            prefetch_staging_->complete_io(staging_idx);
            prefetch_staging_->begin_gpu_transfer(staging_idx);
            wait_for_demotion_dependency(state.operation_id, sdma_cold_stream_);
            payload_pool_->upload_from_host_expert(
                static_cast<uint32_t>(state.vram_slot),
                prefetch_staging_->get_slot_ptr(staging_idx),
                sdma_cold_stream_
            );
            check_hip(
                hipEventRecord(prefetch_staging_->events[staging_idx], sdma_cold_stream_),
                "hipEventRecord(cold H2D)");
            record_h2d_event(
                state.operation_id, sdma_cold_stream_, staging_idx, true,
                static_cast<int32_t>(staging_idx), state.vram_slot);

            state.is_prefetched = true;
            state.io_pending = false;
        }
    }

    PendingTransfer& ensure_registry_transfer(uint64_t operation_id, uint32_t gid) {
        for (auto& transfer : registry_transfers_) {
            if (transfer.operation_id == operation_id) {
                return transfer;
            }
        }
        registry_transfers_.push_back(PendingTransfer{});
        auto& transfer = registry_transfers_.back();
        transfer.operation_id = operation_id;
        transfer.global_expert_id = gid;
        transfer.source_tier = expert_registry_->catalog[gid].owner;
        transfer.phase = supply_telemetry_->current_phase();
        return transfer;
    }

    PendingTransfer* find_registry_transfer(uint64_t operation_id) {
        for (auto& transfer : registry_transfers_) {
            if (transfer.operation_id == operation_id) {
                return &transfer;
            }
        }
        return nullptr;
    }

    const PendingTransfer* find_registry_transfer(uint64_t operation_id) const {
        for (const auto& transfer : registry_transfers_) {
            if (transfer.operation_id == operation_id) {
                return &transfer;
            }
        }
        return nullptr;
    }

    void schedule_demotion(const ExpertRequestReservation& request) {
        if (request.demotion) {
            auto& transfer = ensure_registry_transfer(
                request.operation_id, request.global_expert_id);
            transfer.demoted_expert_id = request.demotion->victim_gid;
            transfer.demotion_source_slot = static_cast<int32_t>(
                request.demotion->source_vram_slot);
            transfer.demotion_destination_slot = static_cast<int32_t>(
                request.demotion->destination_host_slot);
            supply_telemetry_->record_demotion_attempt(transfer.source_tier);
            uint8_t* destination = host_pool_->get_expert_slot_ptr(
                request.demotion->destination_host_slot);
            if (!host_pool_->is_slot_pinned(request.demotion->destination_host_slot)) {
                uint32_t staging_idx = 0;
                if (prefetch_staging_ == nullptr ||
                    !prefetch_staging_->try_begin_direct_transfer(staging_idx)) {
                    expert_registry_->drop_demotion(request.operation_id);
                    supply_telemetry_->record_demotion_drop(
                        transfer.source_tier, "unpinned_fallback_unavailable");
                    supply_telemetry_->record_transfer_event(
                        request.operation_id,
                        transfer.demoted_expert_id,
                        "d2h",
                        transfer.demotion_source_slot,
                        transfer.demotion_destination_slot,
                        "drop",
                        "unpinned_fallback_unavailable");
                    return;
                }
                transfer.demotion_staging_idx = static_cast<int32_t>(staging_idx);
                transfer.demotion_uses_staging = true;
                destination = prefetch_staging_->get_slot_ptr(staging_idx);
            }
            try {
                check_hip(
                    hipEventCreateWithFlags(&transfer.demotion_event, hipEventDisableTiming),
                    "hipEventCreateWithFlags(demotion)");
                payload_pool_->download_to_host_expert(
                    request.demotion->source_vram_slot,
                    destination,
                    demotion_stream_
                );
                transfer.demotion_submitted = true;
                check_hip(
                    hipEventRecord(transfer.demotion_event, demotion_stream_),
                    "hipEventRecord(demotion)");
            } catch (...) {
                if (!transfer.demotion_submitted) {
                    expert_registry_->drop_demotion(request.operation_id);
                    if (transfer.demotion_staging_idx >= 0 && prefetch_staging_) {
                        prefetch_staging_->release_after_failure(
                            static_cast<uint32_t>(transfer.demotion_staging_idx));
                    }
                    if (transfer.demotion_event != nullptr) {
                        (void)hipEventDestroy(transfer.demotion_event);
                        transfer.demotion_event = nullptr;
                    }
                    supply_telemetry_->record_demotion_drop(
                        transfer.source_tier, "d2h_submission_failure");
                    supply_telemetry_->record_transfer_event(
                        request.operation_id,
                        transfer.demoted_expert_id,
                        "d2h",
                        transfer.demotion_source_slot,
                        transfer.demotion_destination_slot,
                        "failed",
                        "d2h_submission_failure");
                }
                throw;
            }
            supply_telemetry_->record_transfer_event(
                request.operation_id,
                transfer.demoted_expert_id,
                "d2h",
                transfer.demotion_source_slot,
                transfer.demotion_destination_slot,
                "submitted");
            return;
        }

        if (expert_registry_->host_capacity == 0) {
            return;
        }

        auto& transfer = ensure_registry_transfer(
            request.operation_id, request.global_expert_id);
        for (const auto& entry : expert_registry_->catalog) {
            if (entry.operation_id == request.operation_id &&
                entry.operation == ExpertOperation::DEMOTION_PENDING) {
                const char* drop_reason = expert_demotion_drop_reason_name(
                    entry.demotion_drop_reason);
                transfer.demoted_expert_id = entry.global_expert_id;
                transfer.demotion_source_slot = entry.slot_idx;
                transfer.demotion_destination_slot = -1;
                supply_telemetry_->record_demotion_attempt(request.source_tier);
                supply_telemetry_->record_demotion_drop(
                    request.source_tier, drop_reason);
                supply_telemetry_->record_transfer_event(
                    request.operation_id,
                    entry.global_expert_id,
                    "d2h",
                    entry.slot_idx,
                    -1,
                    "drop",
                    drop_reason);
                break;
            }
        }
    }

    void wait_for_demotion_dependency(uint64_t operation_id, hipStream_t stream) {
        const auto* transfer = find_registry_transfer(operation_id);
        if (transfer != nullptr && transfer->demotion_event != nullptr) {
            check_hip(
                hipStreamWaitEvent(stream, transfer->demotion_event, 0),
                "hipStreamWaitEvent(demotion)");
        }
    }

    void record_h2d_event(
        uint64_t operation_id,
        hipStream_t stream,
        uint32_t staging_idx,
        bool has_staging,
        int32_t source_slot,
        int32_t destination_slot
    ) {
        auto* transfer = find_registry_transfer(operation_id);
        if (transfer == nullptr) {
            throw std::logic_error("TieredExpertSupply: H2D completion has no registry transfer");
        }
        if (transfer->h2d_event != nullptr) {
            throw std::logic_error("TieredExpertSupply: duplicate H2D submission for one expert operation");
        }
        check_hip(
            hipEventCreateWithFlags(&transfer->h2d_event, hipEventDisableTiming),
            "hipEventCreateWithFlags(H2D)");
        check_hip(hipEventRecord(transfer->h2d_event, stream), "hipEventRecord(H2D)");
        transfer->staging_idx = staging_idx;
        transfer->has_staging = has_staging;
        transfer->h2d_source_slot = source_slot;
        transfer->h2d_destination_slot = destination_slot;
        transfer->h2d_enqueued_at = std::chrono::steady_clock::now();
        if (has_staging && transfer->staging_acquired_at.time_since_epoch().count() != 0) {
            transfer->staging_wait_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    transfer->h2d_enqueued_at - transfer->staging_acquired_at).count());
        }
        transfer->h2d_submitted = true;
    }

    void bind_staging(uint64_t operation_id, uint32_t staging_idx) {
        auto* transfer = find_registry_transfer(operation_id);
        if (transfer == nullptr) {
            throw std::logic_error("TieredExpertSupply: staging binding has no registry transfer");
        }
        transfer->staging_idx = staging_idx;
        transfer->has_staging = true;
        transfer->staging_acquired_at = std::chrono::steady_clock::now();
        transfer->staging_reuse_wait_ns = prefetch_staging_->take_reuse_delay_ns(staging_idx);
    }

    void mark_gpu_readiness_wait_start(uint64_t operation_id) {
        auto* transfer = find_registry_transfer(operation_id);
        if (transfer != nullptr && transfer->gpu_wait_started_at.time_since_epoch().count() == 0) {
            transfer->gpu_wait_started_at = std::chrono::steady_clock::now();
        }
    }

    void mark_registry_request_failed(uint64_t operation_id, const std::string& reason) {
        auto* transfer = find_registry_transfer(operation_id);
        if (transfer == nullptr) return;
        transfer->request_failed = true;
        transfer->failure_reason = reason;
        if (transfer->has_staging && !transfer->h2d_submitted && prefetch_staging_) {
            prefetch_staging_->release_after_failure(transfer->staging_idx);
        }
    }

    void reap_registry_transfers() {
        for (size_t index = 0; index < registry_transfers_.size();) {
            auto& transfer = registry_transfers_[index];
            bool demotion_ready = transfer.demotion_event == nullptr;
            if (transfer.demotion_event != nullptr) {
                const hipError_t result = hipEventQuery(transfer.demotion_event);
                if (result == hipSuccess) {
                    if (transfer.demotion_uses_staging && prefetch_staging_) {
                        std::memcpy(
                            host_pool_->get_expert_slot_ptr(
                                static_cast<uint32_t>(transfer.demotion_destination_slot)),
                            prefetch_staging_->get_slot_ptr(
                                static_cast<uint32_t>(transfer.demotion_staging_idx)),
                            expert_payload_bytes());
                        prefetch_staging_->release_after_gpu_transfer(
                            static_cast<uint32_t>(transfer.demotion_staging_idx));
                    }
                    expert_registry_->complete_demotion(transfer.operation_id);
                    supply_telemetry_->record_demotion_completion(
                        transfer.source_tier, expert_payload_bytes());
                    supply_telemetry_->record_transfer_event(
                        transfer.operation_id,
                        transfer.demoted_expert_id,
                        "d2h",
                        transfer.demotion_source_slot,
                        transfer.demotion_destination_slot,
                        "complete");
                    (void)hipEventDestroy(transfer.demotion_event);
                    transfer.demotion_event = nullptr;
                    demotion_ready = true;
                } else if (result != hipErrorNotReady) {
                    if (transfer.demotion_uses_staging && prefetch_staging_) {
                        prefetch_staging_->release_after_failure(
                            static_cast<uint32_t>(transfer.demotion_staging_idx));
                    }
                    expert_registry_->fail_demotion(transfer.operation_id);
                    supply_telemetry_->record_demotion_drop(
                        transfer.source_tier, "d2h_failure");
                    supply_telemetry_->record_transfer_event(
                        transfer.operation_id,
                        transfer.demoted_expert_id,
                        "d2h",
                        transfer.demotion_source_slot,
                        transfer.demotion_destination_slot,
                        "failed",
                        "d2h_failure");
                    (void)hipEventDestroy(transfer.demotion_event);
                    transfer.demotion_event = nullptr;
                    demotion_ready = true;
                }
            }

            if (transfer.request_failed && !transfer.h2d_submitted) {
                if (!demotion_ready) {
                    ++index;
                    continue;
                }
                expert_registry_->fail_request(transfer.operation_id);
                supply_telemetry_->record_transfer_event(
                    transfer.operation_id,
                    transfer.global_expert_id,
                    "h2d",
                    transfer.h2d_source_slot,
                    transfer.h2d_destination_slot,
                    "failed",
                    transfer.failure_reason.c_str());
                registry_transfers_.erase(
                    registry_transfers_.begin() + static_cast<std::ptrdiff_t>(index));
                continue;
            }

            if (!transfer.h2d_submitted || transfer.h2d_event == nullptr || !demotion_ready) {
                ++index;
                continue;
            }

            const hipError_t result = hipEventQuery(transfer.h2d_event);
            if (result == hipErrorNotReady) {
                ++index;
                continue;
            }
            if (result != hipSuccess) {
                if (transfer.has_staging && prefetch_staging_) {
                    prefetch_staging_->release_after_failure(transfer.staging_idx);
                }
                expert_registry_->fail_request(transfer.operation_id);
                supply_telemetry_->record_transfer_event(
                    transfer.operation_id,
                    transfer.global_expert_id,
                    "h2d",
                    transfer.h2d_source_slot,
                    transfer.h2d_destination_slot,
                    "failed",
                    "h2d_failure");
                (void)hipEventDestroy(transfer.h2d_event);
                registry_transfers_.erase(
                    registry_transfers_.begin() + static_cast<std::ptrdiff_t>(index));
                continue;
            }

            expert_registry_->complete_request(transfer.operation_id);
            const auto ready_at = std::chrono::steady_clock::now();
            const auto h2d_ns = transfer.h2d_enqueued_at.time_since_epoch().count() == 0
                ? uint64_t{0}
                : static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    ready_at - transfer.h2d_enqueued_at).count());
            const auto gpu_wait_ns = transfer.gpu_wait_started_at.time_since_epoch().count() == 0
                ? uint64_t{0}
                : static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    ready_at - transfer.gpu_wait_started_at).count());
            supply_telemetry_->record_timing(
                transfer.phase,
                transfer.source_tier,
                transfer.nvme_read_service_ns,
                transfer.nvme_completion_wait_ns,
                h2d_ns,
                gpu_wait_ns,
                transfer.staging_wait_ns,
                transfer.staging_reuse_wait_ns);
            supply_telemetry_->record_transfer_event(
                transfer.operation_id,
                transfer.global_expert_id,
                "h2d",
                transfer.h2d_source_slot,
                transfer.h2d_destination_slot,
                "complete");
            (void)hipEventDestroy(transfer.h2d_event);
            registry_transfers_.erase(
                registry_transfers_.begin() + static_cast<std::ptrdiff_t>(index));
        }
    }

    void record_supply_request(const ExpertRequestReservation& request) {
        const bool physical_transfer = request.kind != ExpertRequestKind::HOT_HIT &&
                                       request.kind != ExpertRequestKind::PENDING;
        const uint64_t logical_bytes = request.source_tier == ExpertTier::HOT_VRAM
            ? 0
            : expert_payload_bytes();
        supply_telemetry_->record_request(
            supply_telemetry_->current_phase(),
            request.source_tier,
            logical_bytes,
            physical_transfer ? expert_payload_bytes() : 0,
            request.source_tier == ExpertTier::COLD_NVME && physical_transfer
                ? expert_payload_bytes() : 0,
            request.source_tier == ExpertTier::WARM_HOST && physical_transfer
                ? expert_payload_bytes() : 0,
            request.source_tier == ExpertTier::COLD_NVME && physical_transfer
                ? expert_payload_bytes() : 0
        );
    }

    void observe_supply_occupancy(ExpertTier source_tier) {
        const uint64_t warm_pinned_bytes = host_pool_
            ? static_cast<uint64_t>(host_pool_->pinned_slot_count()) * host_pool_->payload_bytes()
            : 0;
        const uint64_t warm_unpinned_bytes = host_pool_
            ? static_cast<uint64_t>(host_pool_->unpinned_slot_count()) * host_pool_->payload_bytes()
            : 0;
        supply_telemetry_->observe_occupancy(
            expert_registry_->published_hot_slots(),
            expert_registry_->published_warm_slots(),
            expert_registry_->pending_transfer_count(),
            expert_registry_->pending_demotion_count,
            warm_pinned_bytes,
            warm_unpinned_bytes,
            source_tier
        );
    }

    void clear() noexcept {
        for (auto& transfer : registry_transfers_) {
            if (transfer.demotion_event != nullptr) {
                (void)hipEventDestroy(transfer.demotion_event);
                transfer.demotion_event = nullptr;
            }
            if (transfer.h2d_event != nullptr) {
                (void)hipEventDestroy(transfer.h2d_event);
                transfer.h2d_event = nullptr;
            }
        }
        registry_transfers_.clear();
    }

private:
    static void check_hip(hipError_t error, const char* operation) {
        if (error != hipSuccess) {
            throw std::runtime_error(
                std::string("TieredExpertSupply: ") + operation + ": " + hipGetErrorString(error));
        }
    }

    PayloadSource source_;
    ExpertPayloadPool* payload_pool_{nullptr};
    HostExpertPool* host_pool_{nullptr};
    ExpertRegistry* expert_registry_{nullptr};
    PrefetchStagingArena* prefetch_staging_{nullptr};
    SupplyTelemetry* supply_telemetry_{nullptr};
    aeon::io::DirectIOReader* direct_io_reader_{nullptr};
    std::unordered_map<uint64_t, aeon::io::DirectIOCompletion>* direct_io_completions_{nullptr};
    uint64_t* next_direct_io_id_{nullptr};
    hipStream_t compute_stream_{nullptr};
    hipStream_t sdma_stream_{nullptr};
    hipStream_t sdma_cold_stream_{nullptr};
    hipStream_t demotion_stream_{nullptr};
    size_t expert_payload_bytes_{0};
    uint64_t demotion_queue_capacity_{0};
    std::vector<PendingTransfer> registry_transfers_;
};

} // namespace aeon::core