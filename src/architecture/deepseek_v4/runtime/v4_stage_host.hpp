#pragma once

// -----------------------------------------------------------------------------
// The stage host — everything one pipeline stage owns.
//
// A stage is a device and a contiguous range of layers, with all the VRAM and
// host state those layers need: the model resources it shares, its activation and
// routed-expert scratch, its layers, its prefill workspace, and its own expert
// tier (registry, pools, corridor, supply, executor, sweep). None of it is
// model-graph code: the stage knows what is resident and how a token's experts
// reach it, not what a token is.
//
// It is separate from `V4ModelHost` because the host owns the things that are the
// **same for every stage** — the artifact, the resolved configuration and specs,
// the budget, the topology — while the stage owns the things that **differ per
// device**. A single-device run is one stage; a pipeline is several, each built on
// its own device over its own layer range.
//
// The split is also what keeps the expert tier device-local: the registry never
// learns about devices, because there is one registry per stage and one stage per
// device.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/spec/config.hpp"
#include "architecture/deepseek_v4/spec/v4_model_spec.hpp"
#include "architecture/deepseek_v4/layer/v4_layer.hpp"
#include "architecture/deepseek_v4/layer/v4_layer_body.hpp"
#include "architecture/deepseek_v4/layer/v4_layer_body_batch.hpp"
#include "architecture/deepseek_v4/layer/v4_activation_scratch.hpp"
#include "architecture/deepseek_v4/runtime/v4_model_resources.hpp"
#include "architecture/deepseek_v4/runtime/v4_prefill_workspace.hpp"
#include "architecture/deepseek_v4/moe/v4_expert_executor.hpp"
#include "architecture/deepseek_v4/moe/v4_expert_supply.hpp"
#include "infrastructure/expert/expert_tier_state.hpp"
#include "infrastructure/expert/transport/supply_telemetry.hpp"
#include "infrastructure/prefill/prefill_controller.hpp"
#include "infrastructure/prefill/prefill_sweep.hpp"
#include "infrastructure/parallel/device_context.hpp"
#include "infrastructure/parallel/parallel_topology.hpp"
#include "infrastructure/memory/memory_budget_report.hpp"
#include "infrastructure/memory/runtime_config.hpp"
#include "infrastructure/artifact/aeon_loader.hpp"
#include "infrastructure/hip_check.hpp"
#include "backend/swizzled_w4a16/core/vram_expert_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::core {

// What the stage needs from the host that owns it. The pointers are borrowed and
// outlive the stage: the host frees its stages before it tears any of these down.
struct V4StageParams {
    AeonModelLoader* loader{nullptr};
    const DeepSeekV4Config* config{nullptr};
    const std::vector<V4LayerSpec>* layer_specs{nullptr};
    const MemoryBudgetReport* budget{nullptr};
    const AeonRuntimeConfig* runtime_cfg{nullptr};
    uint32_t context_capacity{0};
    bool verbose{false};
};

class V4StageHost {
public:
    V4StageHost() = default;

    ~V4StageHost() { free(); }

    V4StageHost(const V4StageHost&) = delete;
    V4StageHost& operator=(const V4StageHost&) = delete;

    // Build the stage on `device` over `range`: select the device, create its
    // streams, and upload the dense weights for its layers. The expert tier is a
    // separate step (`initialize_experts`) so the host can release the dense
    // container's page cache between the uploads and the Warm preload.
    void initialize(const LayerRange& range, int device, const V4StageParams& params) {
        free();
        range_ = range;
        params_ = params;
        verbose_ = params.verbose;
        context_ = DeviceContext::create(device);

        const AeonModelLoader& loader = *params.loader;
        const DeepSeekV4Config& config = *params.config;
        const std::vector<V4LayerSpec>& layer_specs = *params.layer_specs;
        // The RoPE tables and the model-level tensors are built for the declared
        // context, so `context_capacity` is also every layer's `max_seq_len`.
        resources_.initialize(loader, params.context_capacity, config);
        scratch_.allocate();

        layers_.resize(range.count);
        for (uint32_t i = 0; i < range.count; ++i) {
            layers_[i].init_with_loader(layer_specs[range.first + i], loader,
                                        params.context_capacity);
        }

        // The telemetry sink is opened before the supply is built, so a request
        // recorded during initialization cannot be dropped. (`enable_jsonl` opens in
        // truncate mode and clears the summaries, so "open early" means "start from a
        // clean slate and capture everything after it" rather than "start clean and
        // hope nothing has happened yet".)
        if (!params.runtime_cfg->supply_telemetry_path.empty()) {
            tier_.telemetry.enable_jsonl(params.runtime_cfg->supply_telemetry_path,
                                         params.runtime_cfg->run_id);
            if (verbose_) {
                std::printf("[Stage %u] Supply telemetry -> %s (run_id=%s)\n", range.first,
                            params.runtime_cfg->supply_telemetry_path.c_str(),
                            params.runtime_cfg->run_id.c_str());
            }
        }
    }

    void free() noexcept {
        // Flush the telemetry sink first, while the pools it observes still exist,
        // so the final summary captures the peak occupancy of the run being torn down
        // rather than a post-free zero.
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

        // The layers free their own dense weights and attention state, the resources
        // and the scratch free theirs, and every `free()` nulls what it released — so
        // this is safe to call twice, once explicitly and once from the destructor.
        layers_.clear();
        resources_.free();
        scratch_.free();
        prefill_workspace_.free();
        context_.destroy();
        range_ = LayerRange{};
        params_ = V4StageParams{};
    }

    // --- the layer range ------------------------------------------------------

    const LayerRange& range() const noexcept { return range_; }

    // A global layer id resolves to this stage's local slot. The caller is
    // responsible for handing this stage a layer it owns.
    V4Layer& layer(uint32_t global_layer_id) {
        return layers_.at(global_layer_id - range_.first);
    }
    const V4Layer& layer(uint32_t global_layer_id) const {
        return layers_.at(global_layer_id - range_.first);
    }

    uint32_t num_layers() const noexcept { return static_cast<uint32_t>(layers_.size()); }

    // Build the expert tier: the Hot/Warm/Cold pools, the registry, the supply and
    // the routed-expert executor. Skipped, not faked, when the budget left no Hot
    // VRAM slot (then `executor()` refuses, `experts_ready()` is false). Defined out
    // of line below the class.
    void initialize_experts();

    // --- the parts the graph drives -------------------------------------------

    V4ModelResources& resources() noexcept { return resources_; }
    const V4ModelResources& resources() const noexcept { return resources_; }

    V4ActivationScratch& scratch() noexcept { return scratch_; }
    const V4ActivationScratch& scratch() const noexcept { return scratch_; }

    const DeviceStreams& streams() const noexcept { return context_.streams; }

    // --- the layer-major prefill working set ----------------------------------
    //
    // Two buffers, and the distinction between them is the whole reason they are
    // separate: the **chunk workspace** holds per-op temporaries for the rows in
    // flight and is recycled as layers advance, while the **carry** holds the
    // residual being transformed and must survive all of a pass's layers.

    void ensure_batch_scratch(uint32_t layer_id, uint32_t count) {
        prefill_workspace_.ensure_batch_scratch(layer(layer_id), layer_id, count);
    }

    V4LayerBodyBatchScratch& batch_scratch() noexcept { return prefill_workspace_.batch_scratch(); }
    const V4LayerBodyBatchScratch& batch_scratch() const noexcept {
        return prefill_workspace_.batch_scratch();
    }

    void allocate_prefill_workspace(uint32_t window_tokens, uint32_t chunk_tokens) {
        prefill_workspace_.allocate(window_tokens, chunk_tokens, layers_, *params_.config);
    }

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
    size_t decode_scratch_bytes() const noexcept {
        return scratch_.bytes() + expert_scratch_.bytes();
    }
    size_t staging_bytes() const noexcept {
        if (!tier_.staging || !experts_ready()) return 0;
        return static_cast<size_t>(tier_.staging->slot_count()) *
               params_.loader->expert_format().payload_bytes;
    }

    void ensure_prefill_carry(uint32_t tokens) {
        prefill_workspace_.ensure_prefill_carry(tokens, *params_.config);
    }

    half* prefill_carry_half() noexcept { return prefill_workspace_.prefill_carry_half(); }
    float* prefill_carry() noexcept { return prefill_workspace_.prefill_carry(); }
    uint32_t prefill_carry_tokens() const noexcept {
        return prefill_workspace_.prefill_carry_tokens();
    }

    // The routed-expert supply the layer body drives. It is the *interface* type on
    // purpose: the graph must not be able to tell which tier answered, and it must
    // not acquire a dependency on the storage system to run a layer.
    V4RoutedExpertExecutor& executor() {
        if (!executor_) {
            throw std::logic_error(
                "V4StageHost: the expert tier was not built — the memory budget left no "
                "Hot VRAM slot, so the graph cannot run");
        }
        return *executor_;
    }

    bool experts_ready() const noexcept { return executor_ != nullptr; }

    void release_expert_leases() {
        if (executor_) executor_->release_leases();
    }

    size_t outstanding_expert_leases() const noexcept {
        return executor_ ? executor_->outstanding_leases() : 0;
    }

    uint64_t forced_drains() const noexcept {
        return tier_.telemetry.forced_drains();
    }

    uint32_t staging_in_use_slots() const noexcept {
        return tier_.staging ? tier_.staging->in_use_slots() : 0;
    }

    uint32_t staging_slot_count() const noexcept {
        return tier_.staging ? tier_.staging->slot_count() : 0;
    }

    bool resize_staging_slots(uint32_t slots) {
        return tier_.host_partition.resize(slots, experts_per_layer(), outstanding_expert_leases());
    }

    void apply_host_partition(uint32_t warm_slots, uint32_t staging_slots) {
        tier_.host_partition.apply(warm_slots, staging_slots, experts_per_layer(),
                                   outstanding_expert_leases());
    }

    uint32_t batch_staging_capacity() const noexcept {
        return tier_.host_partition.batch_staging_capacity();
    }

    uint32_t sweep_staging_banks() const noexcept {
        return tier_.host_partition.sweep_staging_banks();
    }

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

    uint64_t demotion_queue_capacity() const noexcept { return tier_.demotion_queue_capacity; }

    const ExpertRegistry& registry() const noexcept { return tier_.registry; }
    UnifiedVRAMExpertPool& vram_pool() noexcept {
        return static_cast<UnifiedVRAMExpertPool&>(*tier_.payload_pool);
    }
    const SupplyTelemetry& telemetry() const noexcept { return tier_.telemetry; }

    // Tell the telemetry which phase the next dispatch belongs to. The caller is the
    // generation loop, which already knows whether the token it is about to advance is
    // a prompt token (prefill) or a generated one (decode); this only forwards that
    // fact, it does not derive it. A no-op when the sink is off.
    //
    // It also drives the frozen-Warm policy when `freeze_warm_during_prefill` is set:
    // prefill enters the frozen mode, decode leaves it. Leaving settles any in-flight
    // shadow copy first (`reap` is event-query only, no CPU synchronization) so the
    // idle residencies are visible before they are released; a copy that is still
    // genuinely in flight is reclaimed by eviction when it settles.
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

    bool warm_frozen() const noexcept { return tier_.registry.warm_frozen(); }
    uint32_t shadow_resident_count() const noexcept {
        return tier_.registry.shadow_resident_count();
    }
    int32_t shadow_slot_of(uint32_t gid) const { return tier_.registry.shadow_slot_of(gid); }
    uint64_t shadow_copies() const noexcept { return tier_.registry.shadow_copies; }

    uint64_t supply_logical_bytes_from_warm(bool prefill) const noexcept {
        return tier_.telemetry.logical_bytes_from_warm(
            prefill ? SupplyTelemetryPhase::Prefill : SupplyTelemetryPhase::Decode);
    }

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

    uint64_t supply_requests() const noexcept { return tier_.telemetry.lifetime_requests(); }
    uint64_t supply_bytes_from_nvme() const noexcept {
        return tier_.telemetry.lifetime_nvme_bytes();
    }
    uint64_t supply_bytes_from_host() const noexcept {
        return tier_.telemetry.lifetime_host_bytes();
    }
    uint64_t supply_h2d_bytes() const noexcept { return tier_.telemetry.lifetime_h2d_bytes(); }

    bool routing_reuse_enabled() const noexcept { return tier_.reuse_profiler.enabled(); }

    RoutingReuseProfiler::Curve routing_reuse_curve() const {
        return tier_.reuse_profiler.curve();
    }

    void record_supply_decode_token() { tier_.telemetry.record_decode_token(); }

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

    bool prefill_sweep_enabled() const noexcept { return prefill_controller_.enabled(); }

    uint32_t prefill_sweep_min_tokens() const noexcept { return prefill_controller_.min_tokens(); }

    bool prefill_sweep_engaged_for(uint32_t window_tokens) const noexcept {
        return prefill_controller_.engaged_for(window_tokens);
    }

    bool prefill_sweep_engaged() const noexcept { return prefill_controller_.engaged(); }

    void prefill_begin(uint32_t window_tokens) { prefill_controller_.begin(window_tokens); }

    void prefill_before_layer(uint32_t layer) { prefill_controller_.before_layer(layer); }

    void prefill_after_layer(uint32_t layer) { prefill_controller_.after_layer(layer); }

    void prefill_end() { prefill_controller_.end(); }

    const PrefillSweep& prefill_sweep() const noexcept { return prefill_controller_.sweep(); }

    const std::vector<PrefillSweep::BlockOccupancy>& sweep_occupancy() const noexcept {
        return prefill_controller_.occupancy();
    }

    uint64_t sweep_load_ns() const noexcept { return prefill_controller_.load_ns(); }
    uint64_t sweep_io_ns() const noexcept { return prefill_controller_.io_ns(); }
    uint64_t direct_io_submit_ns() const noexcept { return supply_.direct_io_submit_ns(); }
    uint64_t direct_io_requests_submitted() const noexcept {
        return supply_.direct_io_requests_submitted();
    }
    uint64_t direct_io_submit_calls() const noexcept {
        return supply_.direct_io_submit_calls();
    }
    uint64_t supply_io_wait_ns() const noexcept { return supply_.io_wait_ns(); }
    uint64_t supply_h2d_enqueue_ns() const noexcept { return supply_.h2d_enqueue_ns(); }
    uint64_t supply_h2d_drain_ns() const noexcept { return supply_.h2d_drain_ns(); }
    uint64_t supply_h2d_drain_calls() const noexcept { return supply_.h2d_drain_calls(); }
    uint64_t supply_dispatch_cpu_ns() const noexcept { return supply_.dispatch_cpu_ns(); }
    uint64_t supply_staging_released_on_completion() const noexcept {
        return supply_.staging_released_on_completion();
    }
    uint64_t supply_copies_pumped() const noexcept { return supply_.copies_pumped(); }
    void reset_supply_transfer_counters() noexcept { supply_.reset_transfer_counters(); }
    uint32_t sweep_lookahead_depth() const noexcept {
        return prefill_controller_.lookahead_depth();
    }
    uint32_t sweep_derived_ahead_capacity() const noexcept {
        return prefill_controller_.derived_ahead_capacity();
    }
    void set_sweep_read_ahead_max(uint32_t max_depth) noexcept {
        prefill_controller_.set_read_ahead_max(max_depth);
    }

    // The start of a new sequence. Every layer's ring sentinels, counters and
    // committed-entry positions go back to what a freshly allocated layer holds, so a
    // second conversation cannot read the first one's context.
    void reset_generation_state() {
        for (auto& layer : layers_) {
            layer.reset_generation_state();
        }
    }

private:
    // A compute-stream boundary on every stream that can carry expert traffic. Used
    // by the prefill sweep's switch: entering it frees the whole Hot pool, so no
    // upload or demotion may still be reading or writing a slot.
    void drain_expert_streams() {
        CHECK_HIP(hipStreamSynchronize(context_.streams.compute));
        if (context_.streams.sdma != nullptr) {
            CHECK_HIP(hipStreamSynchronize(context_.streams.sdma));
        }
        if (context_.streams.sdma_cold != nullptr) {
            CHECK_HIP(hipStreamSynchronize(context_.streams.sdma_cold));
        }
        if (context_.streams.demotion != nullptr) {
            CHECK_HIP(hipStreamSynchronize(context_.streams.demotion));
        }
    }

    uint32_t experts_per_layer() const noexcept {
        return static_cast<uint32_t>(params_.config->n_routed_experts);
    }

    LayerRange range_;
    V4StageParams params_;
    bool verbose_{false};
    bool freeze_warm_during_prefill_{false};
    bool supply_phase_prefill_{false};

    DeviceContext context_;
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
    // the stage still owns the arena, region and sweep storage lifetime.
    PrefillController prefill_controller_;
};

// The expert tier, in the order the loop needs it. Defined out of line so the
// declaration above is the only one in the class (a member function cannot be
// declared and then defined in-class).
//
// Skipped, not faked, when the budget left no Hot VRAM slot. The dense graph is
// still constructible (the head stage reads no expert), but the graph is not
// runnable, and `executor()` refuses rather than hand back an executor whose pool
// is empty and whose first dispatch would throw.
inline void V4StageHost::initialize_experts() {
    const MemoryBudgetReport& budget = *params_.budget;
    AeonModelLoader& loader = *params_.loader;
    const DeepSeekV4Config& config = *params_.config;
    const AeonRuntimeConfig& runtime_cfg = *params_.runtime_cfg;
    if (budget.hot_vram_slots == 0) return;

    const auto& format = loader.expert_format();
    const uint32_t warm_slots = budget.warm_host_slots;

    // The neutral expert tier, built by `ExpertTierState::initialize`: the Hot pool,
    // the pinned region and staging corridor, the Warm/staging partition, the
    // residency registry, the direct reader, and the batched Hot/Warm preload. The
    // concrete payload pool is the one backend-shaped piece, so it is built here (the
    // composition root) and sized by the tier; everything else is engine.
    tier_.payload_pool = std::make_unique<UnifiedVRAMExpertPool>();
    tier_.initialize(ExpertTierState::Params{
        &format, &runtime_cfg, &budget, &loader,
        range_.count,
        static_cast<uint32_t>(config.n_routed_experts),
        static_cast<uint32_t>(config.num_experts_per_tok),
        range_.first,
        context_.streams.compute, &prefill_controller_.sweep(), &supply_});

    // The tiered supply, on the four shared streams.
    supply_.configure(
        &loader,
        tier_.payload_pool.get(),
        warm_slots > 0 ? &tier_.host_pool : nullptr,
        &tier_.registry,
        tier_.staging.get(),
        &tier_.telemetry,
        tier_.direct_io.reader(),
        tier_.direct_io.completions(),
        tier_.direct_io.next_id(),
        context_.streams.compute, context_.streams.sdma, context_.streams.sdma_cold,
        context_.streams.demotion,
        format.payload_bytes,
        runtime_cfg.demotion_queue_capacity > 0
            ? runtime_cfg.demotion_queue_capacity
            : (runtime_cfg.enable_warm_refill
                ? static_cast<uint64_t>(config.num_experts_per_tok) : 0),
        range_.first);
    tier_.demotion_queue_capacity = supply_.demotion_queue_capacity();
    freeze_warm_during_prefill_ = runtime_cfg.freeze_warm_during_prefill;

    // The routed-expert scratch and the production executor. The executor borrows the
    // four streams the stage owns, so its capacity fallback drains exactly the set
    // that carries expert traffic.
    expert_scratch_.allocate();

    // The decode workspace's real allocations against the budget's allowance:
    // `scratch_` is allocated in `initialize`, so the two together are what the
    // report's decode term stands for. Checked, not trusted, in the same way the batch
    // scratch is.
    const size_t decode_scratch = scratch_.bytes() + expert_scratch_.bytes();
    if (decode_scratch > decode_scratch_allowance_bytes()) {
        throw std::runtime_error(
            "V4StageHost: the decode workspace is " +
            std::to_string(decode_scratch / (1024 * 1024)) +
            " MiB but the budget allows " +
            std::to_string(decode_scratch_allowance_bytes() / (1024 * 1024)) +
            " MiB — raise DECODE_SCRATCH_ALLOWANCE_BYTES");
    }
    // The routing reuse profiler is a separate concern from the supply telemetry and
    // has its own switch: it is reset (which enables it) only when the routing study
    // asks for it, and a null pointer is what the executor sees as "off".
    if (runtime_cfg.profile_routing_reuse) {
        tier_.reuse_profiler.reset(tier_.registry.total_experts);
    }
    executor_ = std::make_unique<V4TieredExpertExecutor>(
        supply_, vram_pool(), *tier_.staging, tier_.registry, expert_scratch_,
        context_.streams, tier_.telemetry,
        runtime_cfg.profile_routing_reuse ? &tier_.reuse_profiler : nullptr,
        config.swiglu_limit);

    // The prefill controller: the sweep and the strategy switch. It borrows the same
    // two components the executor does, and its sweep's bank count was bound by the
    // partition where the arena was built; this resolves the switch and the
    // prompt-length gate.
    //
    // The prompt-length gate is resolved once from the layer width, and **placed at the
    // measured crossover** rather than derived from theory: the A/B
    // (`scripts/prefill_ab.sh`) puts the routed bank ahead of the sweep up to about
    // `0.7 E` tokens and the sweep ahead from `0.75 E`, so the default is `3 E / 4`. It
    // is a visible, overridable setting, and the round fraction keeps it expressed in
    // the model's own terms.
    const uint32_t sweep_min_tokens = runtime_cfg.prefill_sweep_min_tokens > 0
        ? runtime_cfg.prefill_sweep_min_tokens
        : std::max<uint32_t>(
              1, (static_cast<uint32_t>(config.n_routed_experts) * 3u) / 4u);
    prefill_controller_.bind(PrefillController::Services{
        &context_.streams, &supply_, executor_.get(), &tier_.registry, &tier_.host_partition});
    prefill_controller_.configure(runtime_cfg.prefill_sweep, sweep_min_tokens);
    // Let the per-token hook pump the swept lookahead's copies. The executor does not
    // know about the sweep, so it gets a callback; the sweep ignores the call when it
    // is not driving a window.
    executor_->set_supply_pump([this] { prefill_controller_.pump(); });

    // The prefill workspace, derived from the configured window and chunk and
    // allocated **once, here**, so no window can fail mid-prompt on an allocation and
    // the budget's carry term is realised rather than merely reserved.
    allocate_prefill_workspace(
        runtime_cfg.prefill_window == 0
            ? runtime_cfg.context_size
            : std::min(runtime_cfg.prefill_window, runtime_cfg.context_size),
        runtime_cfg.prefill_chunk > 0 ? runtime_cfg.prefill_chunk : 1);
}

} // namespace aeon::core
