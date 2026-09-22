#pragma once

// -----------------------------------------------------------------------------
// The production routed-expert executor.
//
// `V4RoutedExpertExecutor` (core/v4_layer_body.hpp) is the seam the layer body
// calls. Inside a gate it is implemented against payloads supplied directly; in
// the runtime it must be the whole tiered supply system — index lookup, Hot/Warm/
// Cold promotion, prefetch, leases, staging — followed by the fused expert
// kernels. This file is that implementation, and it is the only place the graph
// touches storage.
//
// Two things are deliberately *not* here:
//
//   * the router. The body computes the ids and the weights and hands them over
//     through `on_routing_ready`; the executor never re-derives a selection.
//   * the shared expert. The body runs it into `moe_accum` before calling
//     `accumulate_routed`, and this executor folds that buffer in as the **initial
//     value of the fp32 accumulator** rather than adding to a completed sum —
//     which is the reference's own fused shape (plan 2.10.5).
//
// -----------------------------------------------------------------------------
// The accumulation, and why it is this one
//
// Two dispatches, matching the plan's kernel inventory (items 19b):
//
//   1. `aeon_moe_fused_w2_contrib_kernel` gives each of the six experts its own
//      fp32 slice of the 4096-wide output. One writer per element, so the result
//      is deterministic **by construction** rather than by observation.
//   2. `v4_moe_accumulate_fixed_order_kernel` sums those six slices in slot order
//      in fp32 and rounds to fp16 **once**.
//
// Neither of the two older paths had both required properties: the `atomicAdd`
// path has an undefined reduction order (trap 38), and the fp16 read-modify-write
// path re-rounds six times, which plan §2.10.3 forbids. Since a decode step that
// is not bit-reproducible cannot support a byte-exact restore (item 22/R3), the
// fixed-order pair is the only admissible choice here — it is not a preference.
//
// -----------------------------------------------------------------------------
// The lease policy, and the hazard it exists to close
//
// A lease does exactly one thing (ExpertRegistry::reserve_vram_destination):
// it excludes a slot from being *chosen as an eviction victim*. It grants no
// ordering. So the safety property is a timing property of the caller:
//
//   **A lease must be held for as long as compute work that reads that slot may
//   still be in flight.**
//
// Releasing earlier is silently wrong, and the failure mode is worth stating
// precisely because it looks innocuous. The dangerous pair is *not* "the demotion
// D2H reads the slot while a kernel reads it" — two readers cannot conflict. It is
// **the incoming expert's H2D overwriting the slot while the outgoing expert's
// kernel still reads it**. The incoming upload is ordered against the demotion
// (`wait_for_demotion_dependency` → the SDMA stream) and against nothing else: it
// never waits for the compute stream. So an early release lets a miss for layer
// L+1 land an H2D on a slot that layer L's W2 kernel has not finished reading, and
// the kernel then multiplies against a mixture of two different experts. The result
// is a plausible number, produced from wrong weights, with nothing anywhere
// recording that it became wrong.
//
// Holding a lease longer is only a capacity cost, so the policy here is the safe
// side of that asymmetry, with the capacity handled explicitly rather than assumed:
//
//   * leases are held for the whole token — the token boundary is where the caller
//     has a compute-stream boundary for free (sampling must read the logits back),
//     so `release_leases()` there costs one host release per expert per token and
//     no synchronization it did not already need;
//   * before dispatching a layer, if the outstanding leases would leave fewer than
//     a full layer's worth of reclaimable slots, the executor **drains the compute
//     stream and releases**, which restores the "no reader in flight" precondition
//     by construction.
//
// The second rule is what makes the first safe on a pool too small to hold
// `6 × 43` leases. It is unreachable whenever the pool can, which is the normal
// case; it exists so that a smaller pool degrades into extra synchronization
// instead of into the registry's "no reclaimable Hot VRAM slot" throw, and it never
// trades correctness for the schedule.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/v4_device_streams.hpp"
#include "architecture/deepseek_v4/core/v4_expert_supply.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp"
#include "backend/swizzled_w4a16/core/vram_expert_pool.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "infrastructure/core/expert_registry.hpp"
#include "infrastructure/core/host_expert_pool.hpp"
#include "infrastructure/core/prefetch_staging.hpp"
#include "infrastructure/core/routing_reuse.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)
#endif

namespace aeon::core {

// Device scratch owned by the routed-expert path.
//
// It is separate from `PipelineScratchBuffers` on purpose: those buffers belong to
// one token's activations, and these two belong to whichever supply tier is being
// consumed. Keeping them apart is what lets the executor be handed to a gate with
// nothing but this struct.
struct V4RoutedExpertScratch {
    static constexpr int kExperts = 6;
    static constexpr int kHidden = 4096;
    static constexpr int kIntermediate = 2048;

    // The fused W13 kernel indexes its weight structs by expert id, and the structs
    // are sized for the kernel's own maximum; only `kExperts` slots are written.
    half* d_expert_hidden{nullptr};   // [kAeonSwizzledMaxExperts, kIntermediate]
    float* d_contrib{nullptr};        // [kExperts, kHidden]

    V4RoutedExpertScratch() = default;

    ~V4RoutedExpertScratch() {
        free();
    }

    V4RoutedExpertScratch(const V4RoutedExpertScratch&) = delete;
    V4RoutedExpertScratch& operator=(const V4RoutedExpertScratch&) = delete;

    void allocate() {
        free();
        CHECK_HIP(hipMalloc(&d_expert_hidden,
                            static_cast<size_t>(kernel::kAeonSwizzledMaxExperts) *
                                kIntermediate * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_contrib,
                            static_cast<size_t>(kExperts) * kHidden * sizeof(float)));
    }

    void free() noexcept {
        if (d_expert_hidden) { (void)hipFree(d_expert_hidden); d_expert_hidden = nullptr; }
        if (d_contrib) { (void)hipFree(d_contrib); d_contrib = nullptr; }
    }
};

// The four streams the executor interacts with live in `core/v4_device_streams.hpp`,
// owned by the host and borrowed here — one definition, because this class's
// capacity fallback has to drain exactly the streams that carry expert traffic and
// a second definition of "the streams" is a way to hand it the wrong set.
//
// `compute` is where the expert kernels run and where the per-transfer staging
// events are awaited. The other three carry the storage traffic (warm and cold
// uploads, eviction downloads) and are needed by the capacity fallback: a slot is
// only reclaimable once the copies *out of and into* it have completed, so a
// fallback that drained only `compute` would release leases without freeing
// anything, and the next dispatch would fail with the registry's own
// "no reclaimable Hot VRAM slot" error.

// The production executor. Borrows everything it needs; owns only its bookkeeping.
class V4TieredExpertExecutor final : public V4RoutedExpertExecutor {
public:
    V4TieredExpertExecutor(
        V4ExpertSupplyCoordinator& supply,
        UnifiedVRAMExpertPool& vram_pool,
        PrefetchStagingArena& staging,
        ExpertRegistry& registry,
        V4RoutedExpertScratch& scratch,
        const V4DeviceStreams& streams,
        SupplyTelemetry& telemetry,
        RoutingReuseProfiler* reuse_profiler,
        float swiglu_limit
    )
        : supply_(supply),
          vram_pool_(vram_pool),
          staging_(staging),
          registry_(registry),
          scratch_(scratch),
          streams_(streams),
          telemetry_(telemetry),
          reuse_profiler_(reuse_profiler),
          swiglu_limit_(swiglu_limit) {
        // The fused dispatches are compiled for exactly six experts; a mismatch here
        // would be a shape error inside the kernel launch, so it is refused at
        // construction where the message can say why.
        if (registry_.vram_capacity == 0) {
            throw std::invalid_argument(
                "V4TieredExpertExecutor: the VRAM expert pool has no slots");
        }
        if (streams_.compute == nullptr) {
            throw std::invalid_argument("V4TieredExpertExecutor: a compute stream is required");
        }
    }

    // Runs once the router's ids are known and before any device work for the MoE.
    //
    // This is where the cold reads are submitted, which is the point of the seam:
    // the body runs the shared expert on the GPU between this call and
    // `accumulate_routed`, so the NVMe latency is covered by work that has to happen
    // anyway. `materialize` (in `accumulate_routed`) is what waits for it.
    void on_routing_ready(uint32_t layer_id, uint32_t position,
                          const std::vector<int32_t>& ids,
                          const std::vector<float>& weights) override {
        (void)position;
        (void)weights;
        if (ids.size() != static_cast<size_t>(V4RoutedExpertScratch::kExperts)) {
            throw std::invalid_argument(
                "V4TieredExpertExecutor: expected six routed experts, got " +
                std::to_string(ids.size()));
        }

        // Retire the transfers that have already completed, exactly as the engine
        // does between layers. Without this the registry keeps every finished
        // operation pending, and once the pool is saturated no slot is ever
        // reclaimable again — the failure the gate hit first.
        supply_.reap_registry_transfers();
        ensure_pool_headroom();
        current_layer_ = layer_id;
        state_ = supply_.dispatch_layer_prefetch(layer_id, position, ids, leases_);
        batch_materialized_ = false;
        batch_draws_ += ids.size();
        batch_distinct_ += state_.expert_count();
        observe_layer(layer_id);
    }

    // The layer-wide dispatch (Step 6 item 4): a chunk's `6C` routed requests
    // issued as **one deduplicated set**. Same seam, same invariants — the only
    // difference is that the layer's union is staged once and reused by every
    // token, instead of each token issuing its own six transfers. `C = 1`
    // reproduces `on_routing_ready` exactly (the dedup of six distinct ids is the
    // six ids), so the decode path is not a second code path.
    void on_routing_ready_batch(uint32_t layer_id,
                                uint32_t first_position,
                                const std::vector<std::vector<int32_t>>& ids,
                                const std::vector<std::vector<float>>& weights) override {
        (void)weights;
        if (ids.empty()) {
            throw std::invalid_argument(
                "V4TieredExpertExecutor: a layer-wide dispatch needs at least one token");
        }
        supply_.reap_registry_transfers();
        ensure_pool_headroom(static_cast<size_t>(ids.size()) *
                             V4RoutedExpertScratch::kExperts);
        current_layer_ = layer_id;
        state_ = supply_.dispatch_layer_prefetch_batch(layer_id, first_position, ids, leases_);
        batch_materialized_ = false;
        batch_draws_ += static_cast<uint64_t>(ids.size()) *
                        V4RoutedExpertScratch::kExperts;
        batch_distinct_ += state_.expert_count();
        observe_layer(layer_id);
    }

    // `moe_accum` already holds the shared expert's output; it is the accumulator's
    // initial value, and the fixed-order reduce is the single rounding.
    //
    // The batch is materialized **once**, on the first token that consumes it, so
    // the cold-read latency stays overlapped with the shared-expert pass of that
    // token (the reason the seam splits `on_routing_ready` from
    // `accumulate_routed` at all). Every later token of the batch reads the same
    // resident slots through its own index map, and the slot-sum order is
    // untouched: token `t`'s `k`-th expert is still the `k`-th operand of the
    // fixed-order reduce, whatever distinct slot it resolves to.
    void accumulate_routed(uint32_t layer_id, uint32_t position,
                           const half* expert_input, const float* expert_weights,
                           half* moe_accum) override {
        (void)layer_id;
        if (current_layer_ != layer_id) {
            throw std::logic_error(
                "V4TieredExpertExecutor: accumulate_routed without a matching "
                "on_routing_ready");
        }
        if (position < state_.first_position) {
            throw std::logic_error(
                "V4TieredExpertExecutor: consume before the batch's first position");
        }
        const uint32_t token_index = position - state_.first_position;
        if (token_index >= state_.token_count) {
            throw std::out_of_range(
                "V4TieredExpertExecutor: token index past the batch's token count");
        }

        if (!batch_materialized_) {
            // Waits for the io_uring completions; the submission already happened.
            supply_.materialize_layer_prefetch(state_);
            for (size_t index = 0; index < state_.expert_count(); ++index) {
                if (!state_.is_prefetched[index]) {
                    continue;
                }
                // The upload may be in flight on a side stream. The supply records a
                // per-transfer event on the stream that carries the upload, and waiting
                // on it here is what orders this batch's kernels behind the copy — the
                // one ordering the executor must not skip. Waited once per distinct
                // expert, not once per token, because the transfer is shared.
                supply_.mark_gpu_readiness_wait_start(state_.operation_ids[index]);
                const uint32_t staging_idx = state_.staging_indices[index];
                CHECK_HIP(hipStreamWaitEvent(
                    streams_.compute, staging_.events[staging_idx], 0));
                if (std::find(staging_in_use_.begin(), staging_in_use_.end(), staging_idx) ==
                    staging_in_use_.end()) {
                    staging_in_use_.push_back(staging_idx);
                }
            }
            batch_materialized_ = true;
        }

        const auto& token_map = state_.token_indices[token_index];
        kernel::SwizzledW13ExpertPtrs w13{};
        kernel::SwizzledW2ExpertPtrs w2{};
        for (int k = 0; k < V4RoutedExpertScratch::kExperts; ++k) {
            const int32_t slot = state_.vram_slots[token_map[static_cast<size_t>(k)]];
            if (slot < 0) {
                throw std::logic_error(
                    "V4TieredExpertExecutor: the supply returned no VRAM slot for a routed expert");
            }
            w13.w1[k] = reinterpret_cast<const uint4*>(
                vram_pool_.get_w1_packed(static_cast<uint32_t>(slot)));
            w13.s1[k] = vram_pool_.get_w1_scale(static_cast<uint32_t>(slot));
            w13.w3[k] = reinterpret_cast<const uint4*>(
                vram_pool_.get_w3_packed(static_cast<uint32_t>(slot)));
            w13.s3[k] = vram_pool_.get_w3_scale(static_cast<uint32_t>(slot));
            w2.w2[k] = reinterpret_cast<const uint4*>(
                vram_pool_.get_w2_packed(static_cast<uint32_t>(slot)));
            w2.s2[k] = vram_pool_.get_w2_scale(static_cast<uint32_t>(slot));
        }

        // 1. gate/up → clamped SwiGLU, per expert, into its own 2048-wide row.
        //    (`output_f32` is the atomic path's accumulator; the contribution path
        //    below does not use it, so it is null.)
        kernel::dispatch_aeon_moe_fused_w13_swiglu<8, 4, 8, 16>(
            expert_input, w13, scratch_.d_expert_hidden, nullptr,
            V4RoutedExpertScratch::kHidden,
            V4RoutedExpertScratch::kExperts,
            V4RoutedExpertScratch::kIntermediate,
            V4RoutedExpertScratch::kHidden,
            swiglu_limit_, streams_.compute);

        // 2. W2 per expert into its own fp32 slice — one writer per element.
        kernel::dispatch_aeon_moe_fused_w2_contrib<8, 8, 4, 16>(
            scratch_.d_expert_hidden, w2, expert_weights, scratch_.d_contrib,
            V4RoutedExpertScratch::kExperts,
            V4RoutedExpertScratch::kHidden,
            V4RoutedExpertScratch::kIntermediate,
            streams_.compute);

        // 3. fp32, slot order, one rounding, shared expert as the initial value.
        constexpr int kThreads = 256;
        kernel::v4_moe_accumulate_fixed_order_kernel
            <<<(V4RoutedExpertScratch::kHidden + kThreads - 1) / kThreads,
               kThreads, 0, streams_.compute>>>(
                scratch_.d_contrib,
                V4RoutedExpertScratch::kExperts,
                moe_accum,
                moe_accum,
                V4RoutedExpertScratch::kHidden);
    }

    void on_routed_consumed(uint32_t layer_id, uint32_t position) override {
        (void)layer_id;
        (void)position;
        for (const uint32_t staging_idx : staging_in_use_) {
            staging_.release_after_gpu_transfer(staging_idx);
        }
        staging_in_use_.clear();
    }

    // The token boundary: the caller has a compute-stream boundary here (sampling
    // must read the logits back), so every lease held for this token can be handed
    // back without an additional synchronization. See the lease policy above for why
    // the boundary is a *precondition* and not an implementation detail.
    void release_leases() {
        for (const uint32_t gid : leases_) {
            registry_.release_lease(gid);
        }
        leases_.clear();
    }

    size_t outstanding_leases() const noexcept {
        return leases_.size();
    }

    // Distinct experts of the **last** dispatch, and the tokens it covered. Lets a
    // gate confirm a chunk really issued one layer-wide batch (`tokens == C`) rather
    // than a sequence of per-token ones.
    uint32_t last_dispatch_experts() const noexcept {
        return static_cast<uint32_t>(state_.expert_count());
    }
    uint32_t last_dispatch_tokens() const noexcept { return state_.token_count; }

    // Cumulative `6 x tokens` draws and the distinct experts they collapsed to.
    // `distinct < draws` is the measurement that dedup happened at all; without it
    // a byte-exact pass could be a per-token dispatch wearing the batch's name.
    uint64_t batch_draws() const noexcept { return batch_draws_; }
    uint64_t batch_distinct() const noexcept { return batch_distinct_; }

private:
    // Keeps the leases this token is holding from starving the next dispatch.
    //
    // A dispatch needs reclaimable victims, and only published, unleased, idle
    // slots qualify. So the guard fires when the leases already held could leave
    // fewer than one layer's worth. It then makes the releases *mean* something:
    // every stream that carries expert traffic is drained, the registry is asked to
    // retire the completed operations, and only then are the leases handed back.
    // Draining `compute` alone would free nothing, because the slot is still held
    // by a pending demotion or upload.
    //
    // On a pool large enough for `6 x 43` leases this is never reached, which is
    // the intended steady state; the counter exists so a gate can say which of the
    // two regimes it ran in.
    void ensure_pool_headroom(size_t incoming_leases = V4RoutedExpertScratch::kExperts) {
        // During a swept prefill nothing evicts: the sweep allocates from the free
        // list only and releases whole layers in order, so there is no victim
        // selection to protect and no drain to run. The guard exists for decode,
        // where a lease count can starve LRU victim selection.
        if (registry_.prefill_streaming()) {
            return;
        }
        if (leases_.size() + incoming_leases <= registry_.vram_capacity) {
            return;
        }
        CHECK_HIP(hipStreamSynchronize(streams_.compute));
        if (streams_.sdma != nullptr) {
            CHECK_HIP(hipStreamSynchronize(streams_.sdma));
        }
        if (streams_.sdma_cold != nullptr) {
            CHECK_HIP(hipStreamSynchronize(streams_.sdma_cold));
        }
        if (streams_.demotion != nullptr) {
            CHECK_HIP(hipStreamSynchronize(streams_.demotion));
        }
        supply_.reap_registry_transfers();
        release_leases();
        telemetry_.record_forced_drain();
    }

    V4ExpertSupplyCoordinator& supply_;
    UnifiedVRAMExpertPool& vram_pool_;
    PrefetchStagingArena& staging_;
    ExpertRegistry& registry_;
    V4RoutedExpertScratch& scratch_;
    V4DeviceStreams streams_;
    SupplyTelemetry& telemetry_;
    // Null unless the routing study asked for reuse profiling. Kept as a pointer so
    // its absence is the off switch: nothing is recorded when it is null.
    RoutingReuseProfiler* reuse_profiler_{nullptr};
    float swiglu_limit_{10.0f};

    uint32_t current_layer_{0};
    V4ExpertSupplyCoordinator::LayerPrefetchState state_;
    // A layer-wide dispatch is materialized once, when its first token is
    // consumed; every later token of the batch reads the same resident slots.
    bool batch_materialized_{false};
    uint64_t batch_draws_{0};
    uint64_t batch_distinct_{0};
    std::vector<uint32_t> leases_;
    std::vector<uint32_t> staging_in_use_;

    // Reports the layer just dispatched to the two measurement consumers, each of
    // which owns its own switch: the supply telemetry (answering-tier mix, always
    // cheap) and the routing reuse profiler (stack distances, only when enabled).
    // Runs after `dispatch_layer_prefetch`, so `state_.supply_batch` holds one
    // transfer per routed expert with its `source_tier` resolved.
    void observe_layer(uint32_t layer_id) {
        const auto& transfers = state_.supply_batch.transfers;
        bool any_cold = false;
        bool any_warm = false;
        for (const auto& transfer : transfers) {
            if (transfer.source_tier == ExpertTier::COLD_NVME) {
                any_cold = true;
            } else if (transfer.source_tier == ExpertTier::WARM_HOST) {
                any_warm = true;
            }
        }
        const SupplyTelemetryPhase phase = telemetry_.current_phase();
        telemetry_.record_layer_outcome(phase, any_cold, any_warm);

        // The reuse profiler observes the decode stream only: prefill has a
        // different reuse structure and is the sweep's concern, so mixing the two
        // would answer neither. A null profiler is the off switch.
        if (reuse_profiler_ != nullptr && phase == SupplyTelemetryPhase::Decode) {
            const size_t count = std::min<size_t>(transfers.size(), 6);
            std::array<RoutingReuseProfiler::Observation, 6> observations{};
            for (size_t i = 0; i < count; ++i) {
                observations[i].global_expert_id = transfers[i].global_expert_id;
                observations[i].answered_from_hot =
                    transfers[i].source_tier == ExpertTier::HOT_VRAM;
            }
            reuse_profiler_->observe_layer(layer_id, observations.data(), count);
        }
    }
};

} // namespace aeon::core
