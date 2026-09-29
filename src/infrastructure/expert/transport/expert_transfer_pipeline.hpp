#pragma once

// -----------------------------------------------------------------------------
// The expert transfer pipeline: the mechanisms that move an expert's bytes and
// keep its transfer record — the read-completion leg, the H2D copy leg, and the
// small bookkeeping (staging binding, event recording, demotion dependency,
// failure marking) they share.
//
// It is deliberately *not* the policy: `TieredExpertSupply` decides what to
// reserve and when to demote, and drives this pipeline; the pipeline does not
// decide which expert moves. The read leg and the copy leg are decoupled so reads
// are bounded by staging and copies by VRAM.
//
// Collaborators are bound in one place, so the pipeline can be read without the
// assembly around it.
// -----------------------------------------------------------------------------

#include "infrastructure/expert/storage/expert_payload_pool.hpp"
#include "infrastructure/expert/residency/expert_registry.hpp"
#include "infrastructure/expert/storage/host_expert_pool.hpp"
#include "infrastructure/expert/transport/pending_transfer_registry.hpp"
#include "infrastructure/expert/transport/prefetch_staging.hpp"
#include "infrastructure/expert/transport/supply_transfer_counters.hpp"
#include "infrastructure/expert/transport/tiered_expert_supply_types.hpp"
#include "infrastructure/io/direct_io_reader.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace aeon::core {

class ExpertTransferPipeline {
public:
    struct Services {
        PrefetchStagingArena* prefetch_staging{nullptr};
        ExpertPayloadPool* payload_pool{nullptr};
        HostExpertPool* host_pool{nullptr};
        ExpertRegistry* expert_registry{nullptr};
        PendingTransferRegistry* transfers{nullptr};
        SupplyTransferCounters* counters{nullptr};
        aeon::io::DirectIOReader* direct_io_reader{nullptr};
        std::unordered_map<uint64_t, aeon::io::DirectIOCompletion>* direct_io_completions{nullptr};
        hipStream_t sdma_cold_stream{nullptr};
        size_t expert_payload_bytes{0};
        uint64_t demotion_queue_capacity{0};
    };

    void bind(const Services& services) {
        prefetch_staging_ = services.prefetch_staging;
        payload_pool_ = services.payload_pool;
        host_pool_ = services.host_pool;
        expert_registry_ = services.expert_registry;
        transfers_ = services.transfers;
        counters_ = services.counters;
        direct_io_reader_ = services.direct_io_reader;
        direct_io_completions_ = services.direct_io_completions;
        sdma_cold_stream_ = services.sdma_cold_stream;
        expert_payload_bytes_ = services.expert_payload_bytes;
        demotion_queue_capacity_ = services.demotion_queue_capacity;
    }

    // Releases the staging slots a **streamed** batch borrowed, once its uploads
    // have landed. The executor does this in `on_routed_consumed`; the prefill sweep
    // has no such hook, so it calls this as soon as its batch is materialized — the
    // sweep's loads are the only traffic on the arena at that moment, which is what
    // makes an immediate release safe.
    void release_streamed_staging(PayloadBatch& batch) {
        if (prefetch_staging_ == nullptr) return;
        const auto drain_started = std::chrono::steady_clock::now();
        for (auto& state : batch.transfers) {
            if (!state.is_prefetched) continue;
            const uint32_t staging_idx = state.staging_idx;
            if (prefetch_staging_->slot_state(staging_idx) !=
                PrefetchStagingArena::SlotState::GPU_TRANSFER_PENDING) {
                state.is_prefetched = false;
                continue;
            }
            check_hip(
                hipEventSynchronize(prefetch_staging_->events[staging_idx]),
                "hipEventSynchronize(stream staging)");
            ++counters_->h2d_drain_calls;
            prefetch_staging_->release_after_gpu_transfer(staging_idx);
            state.is_prefetched = false;
        }
        counters_->h2d_drain_ns += elapsed_ns(drain_started);
    }

    void materialize(PayloadBatch& batch) {
        // Enqueue the copy for every expert whose reads have **already** landed but
        // whose copy was deferred by the non-blocking pump: a staged-only expert that
        // found no free VRAM slot at the moment its read completed. This pass makes
        // the blocking materialize a *join* of the two legs rather than a read-only
        // loop: without it an expert sits `io_complete` and unsubmitted, and the layer
        // body's MoE dispatch — which joins a transfer only after its copy was
        // submitted — surfaces it as
        // "duplicate request joined before its transfer was submitted".
        for (auto& state : batch.transfers) {
            if (state.io_complete && !enqueue_expert_copy(state, state.staging_idx)) {
                // The blocking path is used where a VRAM slot is guaranteed (the
                // sweep's `materialize_entry`, after the previous layer released its
                // block), so a refusal here is a real defect rather than the
                // staged-only deferral.
                throw std::runtime_error(
                    "TieredExpertSupply: materialize found no VRAM slot for a staged-only "
                    "expert — the lookahead over-committed its copy budget");
            }
        }

        for (auto& state : batch.transfers) {
            if (!state.io_pending) {
                continue;
            }

            const auto io_wait_started = std::chrono::steady_clock::now();
            const size_t request_count = state.io_request_count;
            for (size_t chunk = 0; chunk < request_count; ++chunk) {
                const uint64_t request_id = state.io_user_data + chunk;
                if (direct_io_completions_ == nullptr) {
                    mark_request_failed(state.operation_id, "nvme_completion_store_unavailable");
                    throw std::runtime_error(
                        "TieredExpertSupply: direct I/O completion store is unavailable");
                }
                auto completion_it = direct_io_completions_->find(request_id);
                const bool waited_for_completion = completion_it == direct_io_completions_->end();
                const auto wait_started_at = std::chrono::steady_clock::now();
                while (completion_it == direct_io_completions_->end()) {
                    if (!direct_io_reader_) {
                        mark_request_failed(
                            state.operation_id, "nvme_reader_unavailable");
                        throw std::runtime_error("TieredExpertSupply: direct I/O request has no reader");
                    }
                    aeon::io::DirectIOCompletion completion;
                    try {
                        completion = direct_io_reader_->wait_for_completion();
                    } catch (...) {
                        mark_request_failed(
                            state.operation_id, "nvme_completion_wait_failure");
                        throw;
                    }
                    direct_io_completions_->emplace(completion.user_data, completion);
                    completion_it = direct_io_completions_->find(request_id);
                }

                const auto completion = completion_it->second;
                direct_io_completions_->erase(completion_it);
                const auto completed_at = std::chrono::steady_clock::now();
                if (auto* transfer = transfers_->find(state.operation_id)) {
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
                    mark_request_failed(state.operation_id, "nvme_read_failure");
                    throw std::runtime_error(
                        "TieredExpertSupply: direct expert read failed: " +
                        std::string(strerror(-completion.result)));
                }
                const size_t chunk_offset = chunk * aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
                if (chunk_offset >= expert_payload_bytes_) {
                    mark_request_failed(state.operation_id, "nvme_invalid_chunk");
                    throw std::runtime_error(
                        "TieredExpertSupply: direct expert read returned an invalid chunk index");
                }
                const size_t expected_bytes = std::min(
                    aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES,
                    expert_payload_bytes_ - chunk_offset
                );
                if (completion.result != static_cast<int32_t>(expected_bytes)) {
                    mark_request_failed(state.operation_id, "nvme_short_read");
                    throw std::runtime_error(
                        "TieredExpertSupply: direct expert read returned a short payload");
                }
            }

            const uint32_t staging_idx = state.staging_idx;
            counters_->io_wait_ns += elapsed_ns(io_wait_started);
            // The blocking path is used where a VRAM slot is guaranteed (the sweep's
            // `materialize_entry`, after the previous layer released its block), so a
            // refusal here is a real defect rather than the staged-only deferral.
            if (!enqueue_expert_copy(state, staging_idx)) {
                throw std::runtime_error(
                    "TieredExpertSupply: materialize found no VRAM slot for a staged-only "
                    "expert — the lookahead over-committed its copy budget");
            }
        }
    }

    // Enqueue one expert's H2D copy out of its (already landed) staging slot, and
    // mark the transfer done. Shared by the blocking `materialize` and the
    // non-blocking `materialize_available` so the two cannot diverge — the copy's
    // stream, its gate event, and the registry record are identical either way.
    //
    // Returns `false` **without enqueuing** when the operation is staged-only and no
    // VRAM slot is free: the expert stays in staging and the caller retries later —
    // the decoupling of the read leg from the copy leg, since reads are bounded by
    // staging and copies by VRAM.
    bool enqueue_expert_copy(PayloadTransfer& state, uint32_t staging_idx) {
        if (state.vram_slot < 0) {
            // No destination yet: take it now, when the copy can actually run. Covers
            // both a staged-only cold read and a deferred Warm hand-off.
            if (expert_registry_->free_vram_slot_count() == 0) {
                return false;
            }
            state.vram_slot = expert_registry_->attach_vram_destination(
                state.operation_id, demotion_queue_capacity_);
        }
        const auto h2d_enqueue_started = std::chrono::steady_clock::now();
        // A Warm hand-off marked its slot `IO_COMPLETE` at dispatch, so the read leg is
        // already accounted for; only a cold read still needs its completion recorded.
        if (!state.staging_ready) {
            prefetch_staging_->complete_io(staging_idx);
        }
        prefetch_staging_->begin_gpu_transfer(staging_idx);
        wait_for_demotion_dependency(state.operation_id, sdma_cold_stream_);
        // The upload source: the pinned Warm slot when there is one (no arena copy
        // was made), else the staging slot the read landed in.
        const uint8_t* source_ptr = state.warm_host_slot >= 0
            ? host_pool_->get_expert_slot_ptr(static_cast<uint32_t>(state.warm_host_slot))
            : prefetch_staging_->get_slot_ptr(staging_idx);
        payload_pool_->upload_from_host_expert(
            static_cast<uint32_t>(state.vram_slot),
            source_ptr,
            sdma_cold_stream_
        );
        check_hip(
            hipEventRecord(prefetch_staging_->events[staging_idx], sdma_cold_stream_),
            "hipEventRecord(cold H2D)");
        record_h2d_event(
            state.operation_id, sdma_cold_stream_, staging_idx, true,
            state.warm_host_slot >= 0 ? state.warm_host_slot
                                      : static_cast<int32_t>(staging_idx),
            state.vram_slot);
        counters_->h2d_enqueue_ns += elapsed_ns(h2d_enqueue_started);

        state.is_prefetched = true;
        state.io_pending = false;
        state.io_complete = false;
        state.staging_ready = false;
        state.warm_host_slot = -1;
        return true;
    }

    // The **non-blocking** materialize: move every completion the CQ already holds
    // into the store, then enqueue the copy for each expert whose reads have *all*
    // landed, and return how many were enqueued. Experts whose reads are still in
    // flight are left for the next call; nothing is waited on.
    //
    // This is what lets a layer's copies be issued as its reads land — during the
    // previous layer's body — instead of in one block at the boundary. It is called
    // from the per-token pump, which is the only host activity inside a body.
    size_t materialize_available(PayloadBatch& batch) {
        if (direct_io_completions_ == nullptr) return 0;
        if (direct_io_reader_ != nullptr) {
            aeon::io::DirectIOCompletion completion;
            while (direct_io_reader_->try_completion(completion)) {
                direct_io_completions_->emplace(completion.user_data, completion);
            }
        }

        size_t enqueued = 0;
        for (auto& state : batch.transfers) {
            // Phase 1: move this expert's landed reads out of the CQ, once. A read
            // that is already marked complete skips straight to the copy below.
            if (state.io_pending) {
                bool complete = true;
                for (size_t chunk = 0; chunk < state.io_request_count; ++chunk) {
                    if (direct_io_completions_->find(state.io_user_data + chunk) ==
                        direct_io_completions_->end()) {
                        complete = false;
                        break;
                    }
                }
                if (!complete) continue;

                const auto io_wait_started = std::chrono::steady_clock::now();
                for (size_t chunk = 0; chunk < state.io_request_count; ++chunk) {
                    const uint64_t request_id = state.io_user_data + chunk;
                    auto completion_it = direct_io_completions_->find(request_id);
                    const auto completed_at = std::chrono::steady_clock::now();
                    if (auto* transfer = transfers_->find(state.operation_id)) {
                        if (transfer->io_submitted_at.time_since_epoch().count() != 0) {
                            transfer->nvme_read_service_ns += static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    completed_at - transfer->io_submitted_at).count());
                        }
                    }
                    const auto completion = completion_it->second;
                    direct_io_completions_->erase(completion_it);
                    if (completion.result < 0) {
                        mark_request_failed(state.operation_id, "nvme_read_failure");
                        throw std::runtime_error(
                            "TieredExpertSupply: direct expert read failed: " +
                            std::string(strerror(-completion.result)));
                    }
                    const size_t chunk_offset =
                        chunk * aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
                    const size_t expected_bytes = std::min(
                        aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES,
                        expert_payload_bytes_ - chunk_offset);
                    if (completion.result != static_cast<int32_t>(expected_bytes)) {
                        mark_request_failed(state.operation_id, "nvme_short_read");
                        throw std::runtime_error(
                            "TieredExpertSupply: direct expert read returned a short payload");
                    }
                }
                counters_->io_wait_ns += elapsed_ns(io_wait_started);
                state.io_pending = false;
                state.io_complete = true;
            }

            // Phase 2: the copy. A staged-only expert whose VRAM is not free yet is
            // left `io_complete` and retried on a later call — the deferral that
            // decouples the copy leg from the read leg.
            if (state.io_complete && enqueue_expert_copy(state, state.staging_idx)) {
                ++enqueued;
            }
        }
        counters_->copies_pumped += static_cast<uint64_t>(enqueued);
        return enqueued;
    }

    void wait_for_demotion_dependency(uint64_t operation_id, hipStream_t stream) {
        const auto* transfer = transfers_->find(operation_id);
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
        auto* transfer = transfers_->find(operation_id);
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
        auto* transfer = transfers_->find(operation_id);
        if (transfer == nullptr) {
            throw std::logic_error("TieredExpertSupply: staging binding has no registry transfer");
        }
        transfer->staging_idx = staging_idx;
        transfer->has_staging = true;
        transfer->staging_acquired_at = std::chrono::steady_clock::now();
        transfer->staging_reuse_wait_ns = prefetch_staging_->take_reuse_delay_ns(staging_idx);
    }

    void mark_gpu_readiness_wait_start(uint64_t operation_id) {
        auto* transfer = transfers_->find(operation_id);
        if (transfer != nullptr && transfer->gpu_wait_started_at.time_since_epoch().count() == 0) {
            transfer->gpu_wait_started_at = std::chrono::steady_clock::now();
        }
    }

    void mark_request_failed(uint64_t operation_id, const std::string& reason) {
        auto* transfer = transfers_->find(operation_id);
        if (transfer == nullptr) return;
        transfer->request_failed = true;
        transfer->failure_reason = reason;
        if (transfer->has_staging && !transfer->h2d_submitted && prefetch_staging_) {
            prefetch_staging_->release_after_failure(transfer->staging_idx);
        }
    }

private:
    static void check_hip(hipError_t error, const char* operation) {
        if (error != hipSuccess) {
            throw std::runtime_error(
                std::string("TieredExpertSupply: ") + operation + ": " + hipGetErrorString(error));
        }
    }

    static uint64_t elapsed_ns(std::chrono::steady_clock::time_point started) noexcept {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
    }

    PrefetchStagingArena* prefetch_staging_{nullptr};
    ExpertPayloadPool* payload_pool_{nullptr};
    HostExpertPool* host_pool_{nullptr};
    ExpertRegistry* expert_registry_{nullptr};
    PendingTransferRegistry* transfers_{nullptr};
    SupplyTransferCounters* counters_{nullptr};
    aeon::io::DirectIOReader* direct_io_reader_{nullptr};
    std::unordered_map<uint64_t, aeon::io::DirectIOCompletion>* direct_io_completions_{nullptr};
    hipStream_t sdma_cold_stream_{nullptr};
    size_t expert_payload_bytes_{0};
    uint64_t demotion_queue_capacity_{0};
};

} // namespace aeon::core
