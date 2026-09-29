#pragma once

// -----------------------------------------------------------------------------
// The model host — what is resident, in the order the assembly has to happen.
//
// Deliberately *not* the forward pass: the graph (`runtime/v4_graph.hpp`) owns the order
// of operations, and this owns the objects they read and write. The split is what
// keeps the graph small enough to read — nothing in this file knows what a token is,
// and nothing in the graph knows how a tensor got into VRAM.
//
// It builds the whole assembly:
//
//    1. select the compute device and create the four shared streams
//    2. open the `.aeon` container and resolve its weight backend
//    3. load the architecture contract (`config.json`)
//    4. resolve the 43 layer specs from the compression schedule
//    5. validate every required dense tensor against the contract
//    6. evaluate the memory budget and refuse an infeasible configuration
//    7. build the RoPE tables and upload the model-level tensors
//    8. allocate the activation scratch
//    9. construct the 43 layers (dense weights + attention state in VRAM)
//   10. allocate the Hot VRAM expert pool and the staging arena
//   11. initialise the expert registry (Hot residents, Warm capacity)
//   12. preload the Hot slots, then the Warm pool, with batched `O_DIRECT` reads
//   13. configure the tiered supply on the four streams
//   14. construct the routed-expert scratch and the production executor
//   15. allocate the prefill workspace
//
// The **streaming system enters the graph only here**. `V4Graph` never learns which
// tier answered a request: it calls the `V4RoutedExpertExecutor` the body declares,
// and that executor is this object's. That is the whole reason the storage half can
// be correct while the graph is numerics-only.
//
// Every object below already exists and is used as it stands: `AeonModelLoader`,
// `DeepSeekV4Config`, `V4ModelSpec`, `V4ModelContract`, `MemoryBudgetEngine`,
// `V4ModelResources`, `V4Layer`, `V4ActivationScratch`, `DeviceStreams`,
// `V4LayerBodyTables`.
//
// A host is not an op, so there is no arithmetic oracle for it. Its gate is that the
// assembly's own invariants hold and that something above it produces oracle-checked
// numbers: the layers and the head stage are read back and compared, and the budget's
// own accounting is cross-checked against what was actually allocated.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/spec/config.hpp"
#include "infrastructure/core/memory_budget.hpp"
#include "infrastructure/core/device_streams.hpp"
#include "architecture/deepseek_v4/moe/v4_expert_executor.hpp"
#include "architecture/deepseek_v4/moe/v4_expert_supply.hpp"
#include "architecture/deepseek_v4/layer/v4_layer.hpp"
#include "architecture/deepseek_v4/layer/v4_layer_body.hpp"
#include "architecture/deepseek_v4/layer/v4_layer_body_batch.hpp"
#include "architecture/deepseek_v4/runtime/v4_prefill_workspace.hpp"
#include "infrastructure/core/host_partition.hpp"
#include "infrastructure/core/prefill_controller.hpp"
#include "architecture/deepseek_v4/spec/v4_model_contract.hpp"
#include "architecture/deepseek_v4/runtime/v4_model_resources.hpp"
#include "architecture/deepseek_v4/spec/v4_model_spec.hpp"
#include "architecture/deepseek_v4/spec/v4_memory_geometry.hpp"
#include "architecture/deepseek_v4/layer/v4_activation_scratch.hpp"
#include "backend/swizzled_w4a16/core/vram_expert_pool.hpp"
#include "infrastructure/backend_registry/expert_backend.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "infrastructure/core/expert_direct_io.hpp"
#include "infrastructure/core/expert_registry.hpp"
#include "infrastructure/core/expert_tier_state.hpp"
#include "infrastructure/core/expert_tier_loader.hpp"
#include "infrastructure/core/host_expert_pool.hpp"
#include "infrastructure/core/expert_host_region.hpp"
#include "infrastructure/core/prefetch_staging.hpp"
#include "infrastructure/core/prefill_sweep.hpp"
#include "infrastructure/core/supply_telemetry.hpp"
#include "infrastructure/hip_check.hpp"
#include "platform/rdna3/device.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aeon::core {

class V4ModelHost {
public:
    V4ModelHost() = default;

    ~V4ModelHost() {
        free();
    }

    V4ModelHost(const V4ModelHost&) = delete;
    V4ModelHost& operator=(const V4ModelHost&) = delete;

    // The whole assembly, in order, refusing at the first step that cannot hold.
    //
    // The budget refusal is load-bearing rather than informative: a configuration that
    // does not fit is rejected before anything is uploaded, because the alternative —
    // uploading a backbone that does not fit and discovering it as an allocation
    // failure three layers in — is the failure mode the budget engine exists to prevent.
    void initialize(const std::string& model_dir,
                    const AeonRuntimeConfig& runtime_cfg,
                    bool verbose = false) {
        free();
        verbose_ = verbose;

        select_compute_device(verbose_);
        streams_ = DeviceStreams::create();

        loader_.open_model(model_dir);
        const auto& expert_format = loader_.expert_format();
        const auto& backend = ExpertBackendRegistry::resolve(expert_format);
        if (!backend.supports_fused_moe_experts) {
            throw std::runtime_error(
                "V4ModelHost: the selected artifact requires a different weight backend");
        }

        config_ = DeepSeekV4Config::load_from_json(model_dir + "/config.json");
        layer_specs_ = V4ModelSpec::resolve_layers(config_);
        V4ModelContract::validate(config_, loader_);

        // The budget is sized against the bytes the graph actually **uploads**, not
        // the container's file size: `embed.weight` stays host-side and the unused
        // `mtp.*` draft head is never read, together 1.957 GiB a file-size reservation
        // would over-count, costing 148 Hot expert slots (675 instead of 823 at context
        // 256). The contract is the authority on the uploaded set — the same table that
        // validates the artifact — so the budget cannot drift from what reaches VRAM.
        budget_ = MemoryBudgetEngine::evaluate(
            runtime_cfg, make_v4_memory_geometry(config_),
            V4ModelContract::uploaded_dense_bytes(config_), expert_format);
        if (!budget_.is_feasible) {
            throw std::runtime_error(
                "V4ModelHost: the memory budget rejected this configuration: " +
                budget_.rejection_reason);
        }

        context_capacity_ = runtime_cfg.context_size;
        if (context_capacity_ == 0) {
            throw std::invalid_argument("V4ModelHost: the context cannot be zero tokens");
        }

        // The RoPE tables and the model-level tensors are built for the declared
        // context, so `context_capacity_` is also every layer's `max_seq_len`.
        // Growing the context later means rebuilding both, which is why the layer
        // state's own refusal is the thing a caller meets rather than a silently
        // clamped position.
        resources_.initialize(loader_, context_capacity_, config_);
        scratch_.allocate();

        const int32_t num_layers = config_.num_hidden_layers;
        layers_.resize(static_cast<size_t>(num_layers));
        for (int32_t layer = 0; layer < num_layers; ++layer) {
            layers_[static_cast<size_t>(layer)].init_with_loader(
                layer_specs_[static_cast<size_t>(layer)], loader_, context_capacity_);
        }

        // Every dense tensor the contract enumerates has been uploaded by this
        // point — `resources_` binds the model-level ones and the loop above binds
        // the per-layer ones — and the only dense read left on the request path is
        // `embed.weight` in `embed_token`. The pages the uploads faulted in are
        // therefore dead weight that competes for host RAM with the **pinned** Warm
        // pool, which the kernel cannot reclaim or swap.
        //
        // Released here, before the Warm preload rather than after it, because the
        // preload is where host pressure peaks: it fills up to `warm_host_bytes` of
        // unevictable memory while the released pages would still be resident. The
        // step touches only the experts container, so nothing below re-reads dense.
        if (runtime_cfg.release_dense_pages_after_upload) {
            last_released_dense_bytes_ = loader_.release_dense_pages_except("embed.weight");
            // Reported here rather than with the residency summary below, so the
            // figure appears before the preload instead of after it.
            if (verbose_ && last_released_dense_bytes_ > 0) {
                std::printf(
                    "[Host] Released %.2f GiB of dense-container page cache after upload "
                    "(only embed.weight stays resident)\n",
                    static_cast<double>(last_released_dense_bytes_) / (1024.0 * 1024.0 * 1024.0));
            }
        }

        // The telemetry sink is opened before the supply is built, so a request
        // recorded during initialization cannot be dropped. (`enable_jsonl` opens
        // in truncate mode and clears the summaries, so "open early" means "start
        // from a clean slate and capture everything after it" rather than "start
        // clean and hope nothing has happened yet".)
        if (!runtime_cfg.supply_telemetry_path.empty()) {
            tier_.telemetry.enable_jsonl(runtime_cfg.supply_telemetry_path, runtime_cfg.run_id);
            if (verbose_) {
                std::printf("[Host] Supply telemetry -> %s (run_id=%s)\n",
                            runtime_cfg.supply_telemetry_path.c_str(),
                            runtime_cfg.run_id.c_str());
            }
        }

        initialize_experts(runtime_cfg);

        if (verbose_) {
            std::printf(
                "[Host] %d layers (%u Sliding, %u CSA, %u HCA), context %u tokens, "
                "%u hot + %u warm expert slots, demotion queue %llu%s\n",
                num_layers, count_kind(V4AttentionKind::Sliding),
                count_kind(V4AttentionKind::CSA), count_kind(V4AttentionKind::HCA),
                context_capacity_, budget_.hot_vram_slots, budget_.warm_host_slots,
                static_cast<unsigned long long>(tier_.demotion_queue_capacity),
                executor_ ? "" : " (expert tier not built: no Hot VRAM slot)");
        }
    }

    void free() noexcept {
        // Flush the telemetry sink first, while the pools it observes still exist,
        // so the final summary captures the peak occupancy of the run being torn
        // down rather than a post-free zero.
        tier_.telemetry.disable();

        // Teardown is the construction order reversed, and the two references that
        // matter are dropped first: the executor borrows the supply, the pool, the
        // staging arena and the registry, and the supply borrows the reader and the
        // pools. Freeing in any other order would leave a live object pointing at
        // freed storage.
        executor_.reset();
        supply_.clear();
        expert_scratch_.free();
        tier_.free();

        // The layers free their own dense weights and attention state, the
        // resources and the scratch free theirs, and every `free()` nulls what it
        // released — so this is safe to call twice, once explicitly and once from
        // the destructor.
        layers_.clear();
        resources_.free();
        scratch_.free();
        prefill_workspace_.free();
        streams_.destroy();
        loader_.close_all();
        layer_specs_.clear();
        budget_ = MemoryBudgetReport{};
        context_capacity_ = 0;
    }

    // --- the parts the graph drives -----------------------------------------

    uint32_t num_layers() const noexcept {
        return static_cast<uint32_t>(layers_.size());
    }

    uint32_t context_capacity() const noexcept { return context_capacity_; }

    const DeepSeekV4Config& config() const noexcept { return config_; }

    V4ModelResources& resources() noexcept { return resources_; }
    const V4ModelResources& resources() const noexcept { return resources_; }

    V4ActivationScratch& scratch() noexcept { return scratch_; }
    const V4ActivationScratch& scratch() const noexcept { return scratch_; }

    // --- the layer-major prefill working set --------------------------------
    //
    // Two buffers, and the distinction between them is the whole reason they are
    // separate: the **chunk workspace** holds per-op temporaries for the rows in
    // flight and is recycled as layers advance, while the **carry** holds the
    // residual being transformed and must survive all 43 layers of a pass.

    // The per-layer chunk workspace. `V4LayerBodyBatchScratch` sizes itself from a
    // layer's own capacities (the indexer's candidate scores and top-k), so left to
    // itself it would be re-allocated whenever the layer changes. When the window
    // workspace was allocated at load for the worst case across the layers
    // (`allocate_prefill_workspace`), that one buffer already covers every layer and
    // this is a no-op — which is what makes the layer-major pass allocation-free.
    void ensure_batch_scratch(uint32_t layer_id, uint32_t count) {
        prefill_workspace_.ensure_batch_scratch(layer(layer_id), layer_id, count);
    }

    V4LayerBodyBatchScratch& batch_scratch() noexcept { return prefill_workspace_.batch_scratch(); }
    const V4LayerBodyBatchScratch& batch_scratch() const noexcept {
        return prefill_workspace_.batch_scratch();
    }

    // ---- the prefill workspace, derived from the knobs -----------------------
    //
    // The residual carry and the batch scratch are functions of the configured
    // window `W` and chunk `C`, so both are **derived and allocated once, at load**,
    // rather than grown lazily on the first window. Two reasons, and the second is
    // the one that matters:
    //
    //   * a window is a known size, so an allocation inside it can only fail after
    //     work has begun — and a mid-prefill failure has no clean recovery;
    //   * the budget has to know the figure up front (`vram_prefill_carry_bytes`),
    //     or the Hot pool is sized against VRAM the workspace then takes.
    //
    // The batch scratch covers all 43 layers, so its allocation is the worst-case
    // layout, not any one layer's. Its exact size is checked against the budget's
    // allowance here, so a configuration that would overrun fails with a named
    // message instead of quietly shrinking the expert pool.
    void allocate_prefill_workspace(uint32_t window_tokens, uint32_t chunk_tokens) {
        prefill_workspace_.allocate(window_tokens, chunk_tokens, layers_, config_);
    }

    // What was allocated, for the report and the gates.
    uint32_t prefill_window_tokens() const noexcept {
        return prefill_workspace_.prefill_window_tokens();
    }
    uint32_t prefill_chunk_tokens() const noexcept {
        return prefill_workspace_.prefill_chunk_tokens();
    }
    size_t prefill_carry_bytes() const noexcept {
        return prefill_workspace_.prefill_carry_bytes();
    }
    size_t prefill_batch_scratch_bytes() const noexcept {
        return prefill_workspace_.prefill_batch_scratch_bytes();
    }
    // The decode workspace's real size (`V4ActivationScratch` + the routed-expert
    // scratch), for the report and the load-time check against the budget's allowance.
    size_t decode_scratch_bytes() const noexcept {
        return scratch_.bytes() + expert_scratch_.bytes();
    }
    // The pinned host staging arena's footprint, so all three prefill buffers can be
    // reported side by side rather than only the VRAM pair.
    size_t staging_bytes() const noexcept {
        if (!tier_.staging || !experts_ready()) return 0;
        return static_cast<size_t>(tier_.staging->slot_count()) *
               loader_.expert_format().payload_bytes;
    }

    // The residual carry: one window's worth of per-token residual, both fp16 and
    // fp32, held in VRAM for the whole layer-major pass. Grows only, so a window
    // of a given size is allocated once and reused by every later pass that fits.
    void ensure_prefill_carry(uint32_t tokens) {
        prefill_workspace_.ensure_prefill_carry(tokens, config_);
    }

    half* prefill_carry_half() noexcept { return prefill_workspace_.prefill_carry_half(); }
    float* prefill_carry() noexcept { return prefill_workspace_.prefill_carry(); }
    uint32_t prefill_carry_tokens() const noexcept {
        return prefill_workspace_.prefill_carry_tokens();
    }

    const DeviceStreams& streams() const noexcept { return streams_; }

    V4Layer& layer(uint32_t layer_id) { return layers_.at(layer_id); }
    const V4Layer& layer(uint32_t layer_id) const { return layers_.at(layer_id); }

    const std::vector<V4LayerSpec>& layer_specs() const noexcept { return layer_specs_; }

    const MemoryBudgetReport& budget() const noexcept { return budget_; }

    // The routed-expert supply the layer body drives. It is the *interface* type
    // on purpose: the graph must not be able to tell which tier answered, and it
    // must not acquire a dependency on the storage system to run a layer.
    V4RoutedExpertExecutor& executor() {
        if (!executor_) {
            throw std::logic_error(
                "V4ModelHost: the expert tier was not built — the memory budget left no "
                "Hot VRAM slot, so the graph cannot run");
        }
        return *executor_;
    }

    bool experts_ready() const noexcept { return executor_ != nullptr; }

    // The token boundary. The graph calls this once per token, after the head and
    // before the caller reads the logits back — a lease grants no ordering, so it must
    // be held for as long as compute reading that slot may be in flight, and the logits
    // readback is the compute-stream boundary that makes the release safe. Exposed here
    // rather than on the seam because releasing is a property of the concrete tiered
    // executor, not of the interface the body sees.
    void release_expert_leases() {
        if (executor_) executor_->release_leases();
    }

    // Leases still held by the tiered executor. A gate asserts this is 0 at the
    // end of a run: every lease must be handed back by the token boundary, or a
    // slot stays frozen and the next dispatch's victim selection starves.
    size_t outstanding_expert_leases() const noexcept {
        return executor_ ? executor_->outstanding_leases() : 0;
    }

    // Times the emergency drain in `ensure_pool_headroom` fired, from the supply
    // telemetry. Zero whenever the pool can hold a token's `6 x 43` leases, which is
    // the intended steady state; a non-zero value is how a gate says it ran the
    // starved regime.
    uint64_t forced_drains() const noexcept {
        return tier_.telemetry.forced_drains();
    }

    // Staging slots not AVAILABLE. A gate asserts this returns to 0 at the end of a
    // run — the arena must not leak. Reads the concrete executor's arena, so it is
    // safe only after `initialize_experts`; returns 0 when the arena does not exist.
    uint32_t staging_in_use_slots() const noexcept {
        return tier_.staging ? tier_.staging->in_use_slots() : 0;
    }

    // The staging arena's slot count. A batch dispatcher must not assign more
    // distinct staging indices than this, and the layer-major window refuses a
    // chunk whose `6C` requests would exceed it.
    uint32_t staging_slot_count() const noexcept {
        return tier_.staging ? tier_.staging->slot_count() : 0;
    }

    // Re-size the staging arena at runtime: a **depth** change, not a format change.
    // The contract, its preconditions and the region's fixed-total refusal live in
    // `HostPartition::resize` (`host_partition.hpp`); this is the host's entry
    // point to it.
    bool resize_staging_slots(uint32_t slots) {
        return tier_.host_partition.resize(slots, experts_per_layer(), outstanding_expert_leases());
    }

    // ---- the Warm/staging partition -----------------------------------------
    //
    // Warm and the corridor are one pinned region cut by a boundary the phases move.
    // The partition's contract — the three dispatch requirements, why the phases are
    // cut separately, and why moving the boundary is cheap — lives in
    // `HostPartition` (`host_partition.hpp`). This is the host's entry point.
    void apply_host_partition(uint32_t warm_slots, uint32_t staging_slots) {
        tier_.host_partition.apply(warm_slots, staging_slots, experts_per_layer(),
                              outstanding_expert_leases());
    }

    // The corridor capacity a **chunked** window will have, which is what a caller's
    // chunk has to fit — not the live arena, which is cut to decode's smaller shape
    // between windows.
    uint32_t batch_staging_capacity() const noexcept {
        return tier_.host_partition.batch_staging_capacity();
    }

    // Layer-sized staging banks the sweep's arena holds, derived from the arena's
    // actual depth (`slots / experts_per_layer`). The default is `2`; the derivation
    // exists so a runtime resize moves the arena and the sweep's bank indexing
    // together, with no second place for the two to disagree.
    uint32_t sweep_staging_banks() const noexcept {
        return tier_.host_partition.sweep_staging_banks();
    }

    // The shape of the last expert dispatch: tokens covered, and the distinct
    // experts they resolved to. A gate reads these to show a chunk issued one
    // layer-wide batch (`tokens == C`) and that dedup collapsed its `6C` draws.
    uint32_t last_expert_dispatch_tokens() const noexcept {
        return executor_ ? executor_->last_dispatch_tokens() : 0;
    }
    uint32_t last_expert_dispatch_experts() const noexcept {
        return executor_ ? executor_->last_dispatch_experts() : 0;
    }
    uint64_t expert_batch_draws() const noexcept {
        return executor_ ? executor_->batch_draws() : 0;
    }
    uint64_t expert_batch_distinct() const noexcept {
        return executor_ ? executor_->batch_distinct() : 0;
    }

    // The demotion-queue capacity the supply was configured with, after the derived
    // default and the `demotion_queue_capacity` override are resolved.
    uint64_t demotion_queue_capacity() const noexcept { return tier_.demotion_queue_capacity; }

    // Diagnostics, for an assembly gate: the registry's residency claims and the
    // pool it made them against. Not used by the graph.
    const ExpertRegistry& registry() const noexcept { return tier_.registry; }
    UnifiedVRAMExpertPool& vram_pool() noexcept {
        return static_cast<UnifiedVRAMExpertPool&>(*tier_.payload_pool);
    }
    const SupplyTelemetry& telemetry() const noexcept { return tier_.telemetry; }

    // Tell the telemetry which phase the next dispatch belongs to. The caller is
    // the generation loop, which already knows whether the token it is about to
    // advance is a prompt token (prefill) or a generated one (decode); this only
    // forwards that fact, it does not derive it. A no-op when the sink is off.
    //
    // It also drives the frozen-Warm policy when `freeze_warm_during_prefill` is set:
    // prefill enters the frozen mode, decode leaves it. Leaving settles any in-flight
    // shadow copy first (`reap` is event-query only, no CPU synchronization) so the idle
    // residencies are visible before they are released; a copy that is still genuinely
    // in flight is reclaimed by eviction when it settles.
    void set_supply_phase(bool prefill) {
        tier_.telemetry.set_phase(prefill ? RoutingPhase::Prefill : RoutingPhase::Decode);
        if (!freeze_warm_during_prefill_ || !experts_ready()) return;
        // Only the **transition** matters, and only on the way down does work have to
        // be done: leaving frozen must be total, because decode admits single
        // ownership and the invariant refuses a shadow left behind. So the boundary
        // drains the expert streams (a deliberate drain, not on the request path) and
        // reaps, which completes every in-flight copy before the idle shadows are
        // released. The swept prefill owns this flag while it is running.
        if (tier_.registry.prefill_streaming() || prefill == supply_phase_prefill_) return;
        supply_phase_prefill_ = prefill;
        if (prefill) {
            tier_.registry.set_warm_frozen(true);
        } else {
            drain_expert_streams();
            supply_.reap_registry_transfers();
            tier_.registry.set_warm_frozen(false);
        }
    }

    // Frozen-prefill state, for a gate: whether the registry is in frozen mode and
    // how many Warm-owned experts currently hold an extra VRAM copy.
    bool warm_frozen() const noexcept { return tier_.registry.warm_frozen(); }
    uint32_t shadow_resident_count() const noexcept {
        return tier_.registry.shadow_resident_count();
    }
    int32_t shadow_slot_of(uint32_t gid) const { return tier_.registry.shadow_slot_of(gid); }
    uint64_t shadow_copies() const noexcept { return tier_.registry.shadow_copies; }

    // Logical Warm bytes the supply served in a phase. Requires the telemetry sink to
    // have been enabled.
    uint64_t supply_logical_bytes_from_warm(bool prefill) const noexcept {
        return tier_.telemetry.logical_bytes_from_warm(
            prefill ? SupplyTelemetryPhase::Prefill : SupplyTelemetryPhase::Decode);
    }

    // Per-layer outcome counts (thesis-1 measurement), from the supply telemetry:
    // how many dispatches in the given phase were answered entirely from Hot, from
    // Hot+Warm with no Cold, and with at least one Cold. `outcome` is 0/1/2 for
    // AllHot/WarmNoCold/HasCold.
    uint64_t layer_outcome_count(bool prefill, uint32_t outcome) const noexcept {
        if (outcome > 2) return 0;
        return tier_.telemetry.layer_outcome_count(
            prefill ? SupplyTelemetryPhase::Prefill : SupplyTelemetryPhase::Decode,
            static_cast<SupplyTelemetry::LayerOutcome>(outcome));
    }

    uint64_t layer_dispatches(bool prefill) const noexcept {
        return tier_.telemetry.layer_dispatches(
            prefill ? SupplyTelemetryPhase::Prefill : SupplyTelemetryPhase::Decode);
    }

    // Lifetime supply traffic, for a benchmark. Never reset, independent of the
    // JSONL sink, so a throughput run can state bytes read without opening one.
    uint64_t supply_requests() const noexcept { return tier_.telemetry.lifetime_requests(); }
    uint64_t supply_bytes_from_nvme() const noexcept {
        return tier_.telemetry.lifetime_nvme_bytes();
    }
    uint64_t supply_bytes_from_host() const noexcept {
        return tier_.telemetry.lifetime_host_bytes();
    }
    uint64_t supply_h2d_bytes() const noexcept { return tier_.telemetry.lifetime_h2d_bytes(); }

    // Routing reuse-distance profiling (Phase 1 of the routing study): a separate
    // module with its own switch, not part of the supply telemetry.
    bool routing_reuse_enabled() const noexcept {
        return tier_.reuse_profiler.enabled();
    }

    RoutingReuseProfiler::Curve routing_reuse_curve() const {
        return tier_.reuse_profiler.curve();
    }

    // One generated token's worth of decode accounting. A no-op when the sink is off.
    void record_supply_decode_token() {
        tier_.telemetry.record_decode_token();
    }

    // Host read access, for building an oracle: `embed.weight` is reached through
    // `resources().host_embed_table` (a pointer into the container), while
    // `head.weight`, `norm.weight` and the three `hc_head` tensors are device-only
    // and have to come from the container's own host mapping.
    const AeonModelLoader& loader() const noexcept { return loader_; }

    // The two RoPE bases, in the shape the layer body consumes. Sliding layers use
    // the plain base (theta 10000) and compressed layers the YaRN-on-compressed one
    // (theta 160000, factor 16). Handed over as a plain struct so the body keeps no
    // dependency on the resources object.
    V4LayerBodyTables tables() const noexcept {
        V4LayerBodyTables result;
        result.sliding_cos = resources_.d_cos_cache;
        result.sliding_sin = resources_.d_sin_cache;
        result.compressed_cos = resources_.d_compressed_cos_cache;
        result.compressed_sin = resources_.d_compressed_sin_cache;
        return result;
    }

    // ---- the prefill sweep / routed bank -------------------------------------
    //
    // `forward_window` drives the window lifecycle below; the strategy, the
    // prompt-length gate and the sweep itself live in `PrefillController`
    // (`prefill_controller.hpp`). These are the host's entry points to it.
    bool prefill_sweep_enabled() const noexcept { return prefill_controller_.enabled(); }

    // The prompt-length gate, resolved at load (`E / 4` unless configured). Reported
    // so a gate can name the switch point.
    uint32_t prefill_sweep_min_tokens() const noexcept { return prefill_controller_.min_tokens(); }

    // Whether a window of `window_tokens` runs the sweep: enabled and feasible, and
    // at least the gate long. This is the **only** condition on the switch.
    bool prefill_sweep_engaged_for(uint32_t window_tokens) const noexcept {
        return prefill_controller_.engaged_for(window_tokens);
    }

    // The strategy the last window began with, chosen from its length: the sweep at
    // or above the gate, the route-aware cached supply below it. Recorded rather than
    // inferred so a gate can read it.
    bool prefill_sweep_engaged() const noexcept { return prefill_controller_.engaged(); }

    // `window_tokens` is the window length `W`, which the driver is the only one to
    // know at this point — the gate is read here and nowhere else. Both strategies
    // are prefill supplies; `prefill_sweep` enables them, the length picks which.
    void prefill_begin(uint32_t window_tokens) { prefill_controller_.begin(window_tokens); }

    void prefill_before_layer(uint32_t layer) { prefill_controller_.before_layer(layer); }

    void prefill_after_layer(uint32_t layer) { prefill_controller_.after_layer(layer); }

    void prefill_end() { prefill_controller_.end(); }

    // Re-admit the Warm residents a boundary move surrendered. `release_host_tail`
    // recorded them; each is read back into a free Warm slot through the same blocking
    // `O_DIRECT` path the startup preload uses, so the tier ends the window as it began
    // it. The cost is the re-read and nothing else — the corridor's rent, which a
    // smaller `--staging-blocks` declines to pay.
    void restore_prefill_warm(const ExpertFormatDescriptor& format) {
        const std::vector<uint32_t> restore = tier_.registry.host_restore_set();
        if (restore.empty()) return;
        const uint32_t per_layer = tier_.registry.experts_per_layer;
        const size_t batch = std::max<size_t>(
            1, tier_.direct_io.submission_capacity() / ExpertDirectIO::requests_per_fragment(format));
        // Advance a single cursor rather than stepping `start` by a whole batch: a batch
        // that runs out of room must **stop**, not skip the experts it could not place.
        // (Stepping the outer loop by `batch` after an inner break silently dropped the
        // remainder of that batch, which is a restoration gap, not a capacity limit.)
        size_t cursor = 0;
        while (cursor < restore.size()) {
            std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
            std::vector<uint8_t*> destinations;
            while (cursor < restore.size() && expert_ids.size() < batch) {
                const uint32_t gid = restore[cursor++];
                const int32_t slot = tier_.registry.take_free_host_slot();
                if (slot < 0) {
                    // The tier is full. Concurrent decode residency took the room the
                    // borrowed expert would need — the pool is a fixed size, so an
                    // admission during the window displaces a restoration. The
                    // remainder stay Cold and are re-read on demand, which is the same
                    // cost as the borrow itself.
                    cursor = restore.size();
                    break;
                }
                tier_.registry.admit_warm(gid, static_cast<uint32_t>(slot));
                expert_ids.emplace_back(gid / per_layer, gid % per_layer);
                destinations.push_back(
                    tier_.host_pool.get_expert_slot_ptr(static_cast<uint32_t>(slot)));
            }
            if (!expert_ids.empty()) {
                tier_.direct_io.read_blocking(loader_, expert_ids, destinations);
            }
        }
        tier_.registry.clear_host_restore_set();
    }

    // Sweep counters, for the gate: how many layer loads ran, how many experts they
    // streamed, and the deepest frontier the lookahead reached.
    const PrefillSweep& prefill_sweep() const noexcept { return prefill_controller_.sweep(); }

    // The corridor's **fill** per layer — staging slots reading, staging slots
    // copying, and the lookahead layer's reserved-but-not-yet-arrived VRAM experts.
    // A healthy two-block corridor shows `reading` and `copying` both non-zero while a
    // body runs; one pinned at `E` with the other at zero is the parking lot. Empty
    // unless a sweep ran.
    const std::vector<PrefillSweep::BlockOccupancy>& sweep_occupancy() const noexcept {
        return prefill_controller_.occupancy();
    }

    // Nanoseconds the sweep spent inside its layer loads (`dispatch` + `materialize`
    // + release), against the wall clock of the window. The split is what says
    // whether a swept prefill is transfer-bound or compute-bound, and how much a
    // load/compute overlap could recover.
    uint64_t sweep_load_ns() const noexcept { return prefill_controller_.load_ns(); }
    // The dispatch half of the same accounting: time spent *submitting* reads, which
    // is what the double buffer pays to keep the drive busy across the compute.
    uint64_t sweep_io_ns() const noexcept { return prefill_controller_.io_ns(); }
    // Inside `io_uring_enter` alone, and the SQE count it submitted. When this is
    // large the cost is the drive's queue, not the CPU.
    uint64_t direct_io_submit_ns() const noexcept { return supply_.direct_io_submit_ns(); }
    uint64_t direct_io_requests_submitted() const noexcept {
        return supply_.direct_io_requests_submitted();
    }
    uint64_t direct_io_submit_calls() const noexcept {
        return supply_.direct_io_submit_calls();
    }
    // The transfer split (see `TieredExpertSupply`): host time blocked on NVMe
    // completions (`io_wait`), CPU time to submit the H2D copies (`h2d_enqueue`), and
    // host time blocked on those copies landing (`h2d_drain`, plus `h2d_drain_calls`
    // event syncs). Unlike the sweep-scoped `sweep_load_ns`, these accumulate across
    // **both** phases, which is what lets one bench attribute prefill and decode from
    // the same counters. `reset_supply_transfer_counters` slices between them.
    uint64_t supply_io_wait_ns() const noexcept { return supply_.io_wait_ns(); }
    uint64_t supply_h2d_enqueue_ns() const noexcept { return supply_.h2d_enqueue_ns(); }
    uint64_t supply_h2d_drain_ns() const noexcept { return supply_.h2d_drain_ns(); }
    uint64_t supply_h2d_drain_calls() const noexcept { return supply_.h2d_drain_calls(); }
    // CPU time in `dispatch`'s per-request loop (the registry reservation and the two
    // `O(catalog)` scans). Reported for both phases, since the loop is shared.
    uint64_t supply_dispatch_cpu_ns() const noexcept { return supply_.dispatch_cpu_ns(); }
    // Staging slots freed by the completion path instead of a boundary block.
    uint64_t supply_staging_released_on_completion() const noexcept {
        return supply_.staging_released_on_completion();
    }
    // Copies the mid-body pump issued.
    uint64_t supply_copies_pumped() const noexcept { return supply_.copies_pumped(); }
    void reset_supply_transfer_counters() noexcept { supply_.reset_transfer_counters(); }
    // Layers whose reads were in flight when a body started (1 = the double buffer
    // is engaged; 0 = the pool is too small for two layers and loads are serial).
    uint32_t sweep_lookahead_depth() const noexcept {
        return prefill_controller_.lookahead_depth();
    }
    // The lookahead length the **free blocks** allow at this instant — the smaller of
    // the free VRAM blocks and the free staging blocks. Derived, not configured: it
    // grows on a larger pool and shrinks to 0 when either runs out.
    uint32_t sweep_derived_ahead_capacity() const noexcept {
        return prefill_controller_.derived_ahead_capacity();
    }
    // Test instrument: cap the sweep's read lookahead (0 = staging-bounded only).
    void set_sweep_read_ahead_max(uint32_t max_depth) noexcept {
        prefill_controller_.set_read_ahead_max(max_depth);
    }

    // The start of a new sequence. Every layer's ring sentinels, counters and
    // committed-entry positions go back to what a freshly allocated layer holds,
    // so a second conversation cannot read the first one's context.
    void reset_generation_state() {
        for (auto& layer : layers_) {
            layer.reset_generation_state();
        }
    }

private:
    // Reaches a compute-stream boundary on every stream that can carry expert
    // traffic. Used by the prefill sweep's switch: entering it frees the whole Hot
    // pool, so no upload or demotion may still be reading or writing a slot.
    void drain_expert_streams() {
        CHECK_HIP(hipStreamSynchronize(streams_.compute));
        if (streams_.sdma != nullptr) { CHECK_HIP(hipStreamSynchronize(streams_.sdma)); }
        if (streams_.sdma_cold != nullptr) { CHECK_HIP(hipStreamSynchronize(streams_.sdma_cold)); }
        if (streams_.demotion != nullptr) { CHECK_HIP(hipStreamSynchronize(streams_.demotion)); }
    }

    uint32_t count_kind(V4AttentionKind kind) const noexcept {
        uint32_t count = 0;
        for (const auto& spec : layer_specs_) {
            if (spec.attention_kind == kind) ++count;
        }
        return count;
    }

    // The expert tier, in the order the loop needs it.
    //
    // Skipped, not faked, when the budget left no Hot VRAM slot. The dense graph is
    // still constructible (the head stage reads no expert), but the graph is not
    // runnable, and `executor()` refuses rather than hand back an executor whose pool
    // is empty and whose first dispatch would throw.
    void initialize_experts(const AeonRuntimeConfig& runtime_cfg) {
        if (budget_.hot_vram_slots == 0) return;

        const auto& format = loader_.expert_format();
        const uint32_t warm_slots = budget_.warm_host_slots;

        // The neutral expert tier, built by `ExpertTierState::initialize`: the Hot
        // pool, the pinned region and staging corridor, the Warm/staging partition, the
        // residency registry, the direct reader, and the batched Hot/Warm preload. The
        // concrete payload pool is the one backend-shaped piece, so it is built here
        // (the composition root) and sized by the tier; everything else is engine.
        tier_.payload_pool = std::make_unique<UnifiedVRAMExpertPool>();
        tier_.initialize(ExpertTierState::Params{
            &format, &runtime_cfg, &budget_, &loader_,
            static_cast<uint32_t>(config_.num_hidden_layers),
            static_cast<uint32_t>(config_.n_routed_experts),
            static_cast<uint32_t>(config_.num_experts_per_tok),
            streams_.compute, &prefill_controller_.sweep(), &supply_});

        // The tiered supply, on the four shared streams.
        supply_.configure(
            &loader_,
            tier_.payload_pool.get(),
            warm_slots > 0 ? &tier_.host_pool : nullptr,
            &tier_.registry,
            tier_.staging.get(),
            &tier_.telemetry,
            tier_.direct_io.reader(),
            tier_.direct_io.completions(),
            tier_.direct_io.next_id(),
            streams_.compute, streams_.sdma, streams_.sdma_cold, streams_.demotion,
            format.payload_bytes,
            runtime_cfg.demotion_queue_capacity > 0
                ? runtime_cfg.demotion_queue_capacity
                : (runtime_cfg.enable_warm_refill
                    ? static_cast<uint64_t>(config_.num_experts_per_tok) : 0));
        tier_.demotion_queue_capacity = supply_.demotion_queue_capacity();
        freeze_warm_during_prefill_ = runtime_cfg.freeze_warm_during_prefill;

        // The routed-expert scratch and the production executor. The executor borrows
        // the four streams the host owns, so its capacity fallback drains exactly the
        // set that carries expert traffic.
        expert_scratch_.allocate();

        // The decode workspace's real allocations against the budget's allowance:
        // `scratch_` is allocated far above in this function, so the two together are
        // what the report's decode term stands for. Checked, not trusted, in the same
        // way the batch scratch is.
        const size_t decode_scratch_bytes = scratch_.bytes() + expert_scratch_.bytes();
        if (decode_scratch_bytes > decode_scratch_allowance_bytes()) {
            throw std::runtime_error(
                "V4ModelHost: the decode workspace is " +
                std::to_string(decode_scratch_bytes / (1024 * 1024)) +
                " MiB but the budget allows " +
                std::to_string(decode_scratch_allowance_bytes() / (1024 * 1024)) +
                " MiB — raise DECODE_SCRATCH_ALLOWANCE_BYTES");
        }
        // The routing reuse profiler is a separate concern from the supply
        // telemetry and has its own switch: it is reset (which enables it) only
        // when the routing study asks for it, and a null pointer is what the
        // executor sees as "off".
        if (runtime_cfg.profile_routing_reuse) {
            tier_.reuse_profiler.reset(tier_.registry.total_experts);
        }
        executor_ = std::make_unique<V4TieredExpertExecutor>(
            supply_, vram_pool(), *tier_.staging, tier_.registry, expert_scratch_, streams_,
            tier_.telemetry, runtime_cfg.profile_routing_reuse ? &tier_.reuse_profiler : nullptr,
            config_.swiglu_limit);

        // The prefill controller: the sweep and the strategy switch. It borrows the
        // same two components the executor does, and its sweep's bank count was bound by
        // the partition where the arena was built; this resolves the switch and the
        // prompt-length gate. The two host-side restore steps are injected, since both
        // use the host's blocking-read path.
        //
        // The prompt-length gate is resolved once from the layer width, and **placed at
        // the measured crossover** rather than derived from theory: the A/B
        // (`scripts/prefill_ab.sh`) puts the routed bank ahead of the sweep up to about
        // `0.7 E` tokens and the sweep ahead from `0.75 E`, so the default is `3 E / 4`.
        // It is a visible, overridable setting, and the round fraction keeps it
        // expressed in the model's own terms.
        const uint32_t sweep_min_tokens = runtime_cfg.prefill_sweep_min_tokens > 0
            ? runtime_cfg.prefill_sweep_min_tokens
            : std::max<uint32_t>(
                  1, (static_cast<uint32_t>(config_.n_routed_experts) * 3u) / 4u);
        prefill_controller_.bind(PrefillController::Services{
            &streams_, &supply_, executor_.get(), &tier_.registry, &tier_.host_partition,
            [this] { restore_prefill_warm(loader_.expert_format()); },
            [this] { tier_.bulk_loader.restore_residents(loader_.expert_format()); }});
        prefill_controller_.configure(runtime_cfg.prefill_sweep, sweep_min_tokens);
        // Let the per-token hook pump the swept lookahead's copies. The executor does
        // not know about the sweep, so it gets a callback; the sweep ignores the call
        // when it is not driving a window.
        executor_->set_supply_pump([this] { prefill_controller_.pump(); });

        // The prefill workspace, derived from the configured window and chunk and
        // allocated **once, here**, so no window can fail mid-prompt on an allocation
        // and the budget's carry term is realised rather than merely reserved.
        allocate_prefill_workspace(
            runtime_cfg.prefill_window == 0
                ? runtime_cfg.context_size
                : std::min(runtime_cfg.prefill_window, runtime_cfg.context_size),
            runtime_cfg.prefill_chunk > 0 ? runtime_cfg.prefill_chunk : 1);
    }

    uint32_t experts_per_layer() const noexcept {
        return static_cast<uint32_t>(config_.n_routed_experts);
    }

    bool verbose_{false};
    AeonModelLoader loader_;
    DeepSeekV4Config config_;
    std::vector<V4LayerSpec> layer_specs_;
    MemoryBudgetReport budget_;
    uint32_t context_capacity_{0};
    size_t last_released_dense_bytes_{0};
    bool freeze_warm_during_prefill_{false};
    bool supply_phase_prefill_{false};

    DeviceStreams streams_;
    V4ModelResources resources_;
    V4ActivationScratch scratch_;
    std::vector<V4Layer> layers_;

    // The layer-major prefill working set: the residual carry and the worst-case
    // batch scratch, allocated once at load. Owned by the workspace (see
    // `v4_prefill_workspace.hpp`).
    V4PrefillWorkspace prefill_workspace_;

    // The engine-owned expert tier: the neutral state the tiered supply operates on,
    // including the Hot payload pool (the concrete, format-specific pool is
    // constructed here and moved in). See `expert_tier_state.hpp`.
    ExpertTierState tier_;
    V4ExpertSupplyCoordinator supply_;
    V4RoutedExpertScratch expert_scratch_;
    std::unique_ptr<V4TieredExpertExecutor> executor_;
    // The layer-major prefill lifecycle and the sweep: the strategy switch, the
    // prompt-length gate, and the window begin/end that move the partition. It
    // reaches its collaborators as bound services (see `prefill_controller.hpp`);
    // the host still owns the arena, region and sweep storage lifetime.
    PrefillController prefill_controller_;
};

} // namespace aeon::core
