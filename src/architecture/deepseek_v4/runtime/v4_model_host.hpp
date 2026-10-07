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
#include "infrastructure/parallel/device_context.hpp"
#include "infrastructure/parallel/parallel_topology.hpp"
#include "infrastructure/parallel/peer_access.hpp"
#include "infrastructure/memory/memory_budget.hpp"
#include "infrastructure/device_streams.hpp"
#include "architecture/deepseek_v4/runtime/v4_stage_host.hpp"
#include "architecture/deepseek_v4/spec/v4_model_contract.hpp"
#include "architecture/deepseek_v4/spec/v4_model_spec.hpp"
#include "architecture/deepseek_v4/spec/v4_memory_geometry.hpp"
#include "infrastructure/backend_registry/expert_backend.hpp"
#include "infrastructure/artifact/aeon_loader.hpp"
#include "infrastructure/hip_check.hpp"
#include "platform/device.hpp"

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

        // Tensor parallelism is not wired yet — the rank-sharded artifact, the
        // collective and the split layer body do not exist — so it is a named refusal
        // rather than a silent single-rank run. Pipeline parallelism is supported: one
        // stage per device, a peer copy of the residual at the boundary.
        if (runtime_cfg.parallel.tensor_parallel > 1) {
            throw std::runtime_error("V4ModelHost: tensor parallelism not yet supported");
        }

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

        // The topology is resolved once, here, and every device the host touches is
        // reached through it rather than by a heuristic. `artifact_max_tp` is 1 until
        // the artifact declares otherwise; at one rank and one stage this validates
        // the device id and selects stage 0 / rank 0's device.
        int visible_devices = 0;
        CHECK_HIP(hipGetDeviceCount(&visible_devices));
        topology_ = ParallelTopology::resolve(
            runtime_cfg.parallel, visible_devices, /*artifact_max_tp=*/1u,
            static_cast<uint32_t>(config_.num_hidden_layers));

        // A pipeline copies a token's residual from one stage's device to the next. Peer
        // access makes that copy direct where the hardware allows it; where it does not,
        // `hipMemcpyPeerAsync` still produces the same bytes through a staged path, so
        // this is best-effort and never a refusal.
        if (topology_.pp() > 1) {
            std::vector<int> stage_devices;
            stage_devices.reserve(topology_.pp());
            for (uint32_t stage = 0; stage < topology_.pp(); ++stage) {
                stage_devices.push_back(topology_.device(stage, 0));
            }
            enable_peer_access(stage_devices, verbose_);
        }

        // The budget is sized against the bytes the graph actually **uploads**, not
        // the container's file size: `embed.weight` stays host-side and the unused
        // `mtp.*` draft head is never read, together 1.957 GiB a file-size reservation
        // would over-count, costing 148 Hot expert slots (675 instead of 823 at context
        // 256). The contract is the authority on the uploaded set — the same table that
        // validates the artifact — so the budget cannot drift from what reaches VRAM.
        //
        // Each stage is evaluated on **its own** device and against **its own** layer
        // range: its dense and attention bytes, plus its proportional share of the host
        // budget. At one stage the range is the whole model and the share is the whole
        // host budget, so the single-device figure is unchanged.
        budgets_.clear();
        budgets_.reserve(topology_.pp());
        const uint32_t model_layers = static_cast<uint32_t>(config_.num_hidden_layers);
        std::vector<uint32_t> stage_layer_counts(topology_.pp());
        for (uint32_t stage = 0; stage < topology_.pp(); ++stage) {
            stage_layer_counts[stage] = topology_.stage_layers(stage).count;
        }
        const std::vector<size_t> stage_warm_shares = MemoryBudgetEngine::split_host_budget(
            runtime_cfg.warm_host_bytes, stage_layer_counts);
        for (uint32_t stage = 0; stage < topology_.pp(); ++stage) {
            const LayerRange range = topology_.stage_layers(stage);
            const bool has_head = (range.first + range.count == model_layers);

            // Each stage's host budget is its proportional share, which is what makes
            // the total host RAM across a pipeline the number the user set rather than
            // a multiple of it.
            AeonRuntimeConfig stage_cfg = runtime_cfg;
            stage_cfg.warm_host_bytes = stage_warm_shares[stage];

            // The artifact's expert catalog covers the whole model; a stage's budget
            // only ever touches its own layers, so the format is narrowed to the
            // stage's layer count. `payload_bytes` and `experts_per_layer` are
            // unchanged.
            ExpertFormatDescriptor stage_format = expert_format;
            stage_format.num_layers = range.count;

            MemoryBudgetReport report;
            {
                // Evaluate against the stage's own device, so per-device refusals (a
                // display-loaded card, a fraction with no room) name that device.
                DeviceScope scope(topology_.device(stage, 0));
                report = MemoryBudgetEngine::evaluate(
                    stage_cfg, make_v4_memory_geometry(config_, range),
                    V4ModelContract::uploaded_dense_bytes(config_, range, has_head),
                    stage_format);
            }
            if (!report.is_feasible) {
                throw std::runtime_error(
                    "stage " + std::to_string(stage) + ": " + report.rejection_reason);
            }
            budgets_.push_back(std::move(report));
        }

        context_capacity_ = runtime_cfg.context_size;
        if (context_capacity_ == 0) {
            throw std::invalid_argument("V4ModelHost: the context cannot be zero tokens");
        }

        // The stages upload the dense weights; at one rank and one stage this is the
        // single device, and a pipeline builds one stage per layer range. Each stage's
        // RoPE tables and model-level tensors are built for the declared context, so
        // `context_capacity_` is also every layer's `max_seq_len`. Growing the context
        // later means rebuilding both, which is why the layer state's own refusal is
        // the thing a caller meets rather than a silently clamped position.
        stages_.clear();
        stages_.reserve(topology_.pp());
        for (uint32_t stage = 0; stage < topology_.pp(); ++stage) {
            stages_.push_back(std::make_unique<V4StageHost>());
            stages_.back()->initialize(
                topology_.stage_layers(stage), topology_.device(stage, 0),
                V4StageParams{&loader_, &config_, &layer_specs_, &budgets_[stage], &runtime_cfg,
                              context_capacity_, verbose_, stage});
        }

        // Every dense tensor the contract enumerates has been uploaded by this point,
        // and the only dense read left on the request path is `embed.weight` in
        // `embed_token`. The pages the uploads faulted in are therefore dead weight
        // that competes for host RAM with the **pinned** Warm pool, which the kernel
        // cannot reclaim or swap.
        //
        // Released here, after the dense uploads and before the Warm preload (the stage
        // expert build below), because the preload is where host pressure peaks: it
        // fills up to `warm_host_bytes` of unevictable memory while the released pages
        // would still be resident. The step touches only the experts container, so
        // nothing below re-reads dense.
        if (runtime_cfg.release_dense_pages_after_upload) {
            last_released_dense_bytes_ = loader_.release_dense_pages_except("embed.weight");
            // Reported here rather than with the residency summary below, so the figure
            // appears before the preload instead of after it.
            if (verbose_ && last_released_dense_bytes_ > 0) {
                std::printf(
                    "[Host] Released %.2f GiB of dense-container page cache after upload "
                    "(only embed.weight stays resident)\n",
                    static_cast<double>(last_released_dense_bytes_) / (1024.0 * 1024.0 * 1024.0));
            }
        }

        // Now the expert tier, per stage: the Hot/Warm pools, the corridor and the Warm
        // preload all live here, on the stage's own device.
        for (auto& stage : stages_) {
            stage->initialize_experts();
        }

        if (verbose_) {
            std::printf(
                "[Host] %u stages over %d layers (%u Sliding, %u CSA, %u HCA), context %u tokens%s\n",
                static_cast<uint32_t>(stages_.size()), config_.num_hidden_layers,
                count_kind(V4AttentionKind::Sliding), count_kind(V4AttentionKind::CSA),
                count_kind(V4AttentionKind::HCA), context_capacity_,
                experts_ready() ? "" : " (expert tier not built: no Hot VRAM slot)");
            // One report per stage: each names its device, its layer range, its own
            // Hot/Warm capacity and its share of VRAM and host RAM.
            for (uint32_t stage = 0; stage < budgets_.size(); ++stage) {
                const MemoryBudgetReport& report = budgets_[stage];
                const LayerRange range = topology_.stage_layers(stage);
                std::printf(
                    "[Stage %u] device %d, layers [%u, %u): %u hot + %u warm expert slots, "
                    "%.2f GiB dense, %.2f GiB attention, %.2f GiB host\n",
                    stage, report.device_index, range.first, range.first + range.count,
                    report.hot_vram_slots, report.warm_host_slots,
                    static_cast<double>(report.vram_dense_bytes) / (1024.0 * 1024.0 * 1024.0),
                    static_cast<double>(report.vram_kv_bytes) / (1024.0 * 1024.0 * 1024.0),
                    static_cast<double>(report.configured_host_budget_bytes) /
                        (1024.0 * 1024.0 * 1024.0));
            }
        }
    }

    void free() noexcept {
        // Each stage tears down its own expert tier and device in construction order
        // reversed; the host owns the artifact, the config and the budget and releases
        // them last.
        stages_.clear();
        loader_.close_all();
        layer_specs_.clear();
        budgets_.clear();
        context_capacity_ = 0;
    }

    // --- the parts the graph drives -----------------------------------------

    uint32_t num_layers() const noexcept {
        return static_cast<uint32_t>(config_.num_hidden_layers);
    }

    uint32_t context_capacity() const noexcept { return context_capacity_; }

    const DeepSeekV4Config& config() const noexcept { return config_; }

    V4ModelResources& resources() noexcept { return stage().resources(); }
    const V4ModelResources& resources() const noexcept { return stage().resources(); }

    V4ActivationScratch& scratch() noexcept { return stage().scratch(); }
    const V4ActivationScratch& scratch() const noexcept { return stage().scratch(); }

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
        stage_of_layer(layer_id).ensure_batch_scratch(layer_id, count);
    }

    V4LayerBodyBatchScratch& batch_scratch() noexcept { return stage().batch_scratch(); }
    const V4LayerBodyBatchScratch& batch_scratch() const noexcept {
        return stage().batch_scratch();
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
        stage().allocate_prefill_workspace(window_tokens, chunk_tokens);
    }

    // What was allocated, for the report and the gates.
    uint32_t prefill_window_tokens() const noexcept { return stage().prefill_window_tokens(); }
    uint32_t prefill_chunk_tokens() const noexcept { return stage().prefill_chunk_tokens(); }
    size_t prefill_carry_bytes() const noexcept { return stage().prefill_carry_bytes(); }
    size_t prefill_batch_scratch_bytes() const noexcept {
        return stage().prefill_batch_scratch_bytes();
    }
    // The decode workspace's real size (`V4ActivationScratch` + the routed-expert
    // scratch), for the report and the load-time check against the budget's allowance.
    size_t decode_scratch_bytes() const noexcept { return stage().decode_scratch_bytes(); }
    // The pinned host staging arena's footprint, so all three prefill buffers can be
    // reported side by side rather than only the VRAM pair.
    size_t staging_bytes() const noexcept { return stage().staging_bytes(); }

    // The residual carry: one window's worth of per-token residual, both fp16 and
    // fp32, held in VRAM for the whole layer-major pass. Grows only, so a window
    // of a given size is allocated once and reused by every later pass that fits.
    void ensure_prefill_carry(uint32_t tokens) { stage().ensure_prefill_carry(tokens); }

    half* prefill_carry_half() noexcept { return stage().prefill_carry_half(); }
    float* prefill_carry() noexcept { return stage().prefill_carry(); }
    uint32_t prefill_carry_tokens() const noexcept { return stage().prefill_carry_tokens(); }

    const DeviceStreams& streams() const noexcept { return stage().streams(); }

    // The stream the embedding and stage 0 run on, and the stream the head and the
    // logits read back on. They are the same stream at one stage; a pipeline puts the
    // head on the last device, so a sampler must read the last stage's stream.
    hipStream_t embed_stream() const noexcept { return stages_.front()->streams().compute; }
    hipStream_t head_stream() const noexcept { return stages_.back()->streams().compute; }
    // The device the head runs on — the last stage's. The sampler's device workspace
    // and logits readback must live there, not on the first stage's device.
    int head_device() const noexcept { return stages_.back()->device_index(); }

    // The stages, by index. A stage is one device and one contiguous layer range.
    uint32_t stage_count() const noexcept { return static_cast<uint32_t>(stages_.size()); }
    V4StageHost& stage(uint32_t index) { return *stages_.at(index); }
    const V4StageHost& stage(uint32_t index) const { return *stages_.at(index); }
    V4StageHost& stage_of(uint32_t layer_id) { return stage_of_layer(layer_id); }
    const V4StageHost& stage_of(uint32_t layer_id) const { return stage_of_layer(layer_id); }

    // The resolved parallel topology: one device in the degenerate case. The host
    // owns every device and layer-range decision through it.
    const ParallelTopology& topology() const noexcept { return topology_; }

    V4Layer& layer(uint32_t layer_id) { return stage_of_layer(layer_id).layer(layer_id); }
    const V4Layer& layer(uint32_t layer_id) const {
        return stage_of_layer(layer_id).layer(layer_id);
    }

    const std::vector<V4LayerSpec>& layer_specs() const noexcept { return layer_specs_; }

    // The whole-model budget, for pp = 1: the single stage's report. A pipeline has
    // one report per stage (`budgets()` / `stage_budget`).
    const MemoryBudgetReport& budget() const noexcept { return budgets_.front(); }

    // Every stage's budget report, in stage order.
    const std::vector<MemoryBudgetReport>& budgets() const noexcept { return budgets_; }
    const MemoryBudgetReport& stage_budget(uint32_t stage) const { return budgets_.at(stage); }

    // The routed-expert supply the layer body drives. It is the *interface* type
    // on purpose: the graph must not be able to tell which tier answered, and it
    // must not acquire a dependency on the storage system to run a layer.
    V4RoutedExpertExecutor& executor() { return stage().executor(); }

    bool experts_ready() const noexcept {
        for (const auto& stage : stages_) {
            if (!stage->experts_ready()) return false;
        }
        return !stages_.empty();
    }

    // The token boundary. The graph calls this once per token, after the head and
    // before the caller reads the logits back — a lease grants no ordering, so it must
    // be held for as long as compute reading that slot may be in flight, and the logits
    // readback is the compute-stream boundary that makes the release safe. Exposed here
    // rather than on the seam because releasing is a property of the concrete tiered
    // executor, not of the interface the body sees.
    void release_expert_leases() {
        // Every stage holds its own leases; releasing only the first would leave a
        // later stage's slot frozen for the next dispatch.
        for (auto& stage : stages_) {
            stage->release_expert_leases();
        }
    }

    // Leases still held by the tiered executor. A gate asserts this is 0 at the
    // end of a run: every lease must be handed back by the token boundary, or a
    // slot stays frozen and the next dispatch's victim selection starves.
    size_t outstanding_expert_leases() const noexcept {
        size_t total = 0;
        for (const auto& stage : stages_) {
            total += stage->outstanding_expert_leases();
        }
        return total;
    }

    // Times the emergency drain in `ensure_pool_headroom` fired, from the supply
    // telemetry. Zero whenever the pool can hold a token's `6 x 43` leases, which is
    // the intended steady state; a non-zero value is how a gate says it ran the
    // starved regime.
    uint64_t forced_drains() const noexcept { return stage().forced_drains(); }

    // Staging slots not AVAILABLE. A gate asserts this returns to 0 at the end of a
    // run — the arena must not leak. Reads the concrete executor's arena, so it is
    // safe only after `initialize_experts`; returns 0 when the arena does not exist.
    uint32_t staging_in_use_slots() const noexcept { return stage().staging_in_use_slots(); }

    // The staging arena's slot count. A batch dispatcher must not assign more
    // distinct staging indices than this, and the layer-major window refuses a
    // chunk whose `6C` requests would exceed it.
    uint32_t staging_slot_count() const noexcept { return stage().staging_slot_count(); }

    // Re-size the staging arena at runtime: a **depth** change, not a format change.
    // The contract, its preconditions and the region's fixed-total refusal live in
    // `HostPartition::resize` (`host_partition.hpp`); this is the host's entry
    // point to it.
    bool resize_staging_slots(uint32_t slots) { return stage().resize_staging_slots(slots); }

    // ---- the Warm/staging partition -----------------------------------------
    //
    // Warm and the corridor are one pinned region cut by a boundary the phases move.
    // The partition's contract — the three dispatch requirements, why the phases are
    // cut separately, and why moving the boundary is cheap — lives in
    // `HostPartition` (`host_partition.hpp`). This is the host's entry point.
    void apply_host_partition(uint32_t warm_slots, uint32_t staging_slots) {
        stage().apply_host_partition(warm_slots, staging_slots);
    }

    // The corridor capacity a **chunked** window will have, which is what a caller's
    // chunk has to fit — not the live arena, which is cut to decode's smaller shape
    // between windows.
    uint32_t batch_staging_capacity() const noexcept { return stage().batch_staging_capacity(); }

    // Layer-sized staging banks the sweep's arena holds, derived from the arena's
    // actual depth (`slots / experts_per_layer`). The default is `2`; the derivation
    // exists so a runtime resize moves the arena and the sweep's bank indexing
    // together, with no second place for the two to disagree.
    uint32_t sweep_staging_banks() const noexcept { return stage().sweep_staging_banks(); }

    // The shape of the last expert dispatch: tokens covered, and the distinct
    // experts they resolved to. A gate reads these to show a chunk issued one
    // layer-wide batch (`tokens == C`) and that dedup collapsed its `6C` draws.
    uint32_t last_expert_dispatch_tokens() const noexcept {
        return stage().last_expert_dispatch_tokens();
    }
    uint32_t last_expert_dispatch_experts() const noexcept {
        return stage().last_expert_dispatch_experts();
    }
    uint64_t expert_batch_draws() const noexcept { return stage().expert_batch_draws(); }
    uint64_t expert_batch_distinct() const noexcept { return stage().expert_batch_distinct(); }

    // The demotion-queue capacity the supply was configured with, after the derived
    // default and the `demotion_queue_capacity` override are resolved.
    uint64_t demotion_queue_capacity() const noexcept { return stage().demotion_queue_capacity(); }

    // Diagnostics, for an assembly gate: the registry's residency claims and the
    // pool it made them against. Not used by the graph.
    const ExpertRegistry& registry() const noexcept { return stage().registry(); }
    UnifiedVRAMExpertPool& vram_pool() noexcept { return stage().vram_pool(); }
    const SupplyTelemetry& telemetry() const noexcept { return stage().telemetry(); }

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
        // The phase transition drains each stage's own streams and moves its own
        // partition, so every stage must see it, not just the first.
        for (auto& stage : stages_) {
            stage->set_supply_phase(prefill);
        }
    }

    // Frozen-prefill state, for a gate: whether the registry is in frozen mode and
    // how many Warm-owned experts currently hold an extra VRAM copy.
    bool warm_frozen() const noexcept { return stage().warm_frozen(); }
    uint32_t shadow_resident_count() const noexcept { return stage().shadow_resident_count(); }
    int32_t shadow_slot_of(uint32_t gid) const { return stage().shadow_slot_of(gid); }
    uint64_t shadow_copies() const noexcept { return stage().shadow_copies(); }

    // Logical Warm bytes the supply served in a phase. Requires the telemetry sink to
    // have been enabled.
    uint64_t supply_logical_bytes_from_warm(bool prefill) const noexcept {
        return stage().supply_logical_bytes_from_warm(prefill);
    }

    // Per-layer outcome counts (thesis-1 measurement), from the supply telemetry:
    // how many dispatches in the given phase were answered entirely from Hot, from
    // Hot+Warm with no Cold, and with at least one Cold. `outcome` is 0/1/2 for
    // AllHot/WarmNoCold/HasCold.
    uint64_t layer_outcome_count(bool prefill, uint32_t outcome) const noexcept {
        return stage().layer_outcome_count(prefill, outcome);
    }

    uint64_t layer_dispatches(bool prefill) const noexcept {
        return stage().layer_dispatches(prefill);
    }

    // Lifetime supply traffic, for a benchmark. Never reset, independent of the
    // JSONL sink, so a throughput run can state bytes read without opening one.
    uint64_t supply_requests() const noexcept { return stage().supply_requests(); }
    uint64_t supply_bytes_from_nvme() const noexcept { return stage().supply_bytes_from_nvme(); }
    uint64_t supply_bytes_from_host() const noexcept { return stage().supply_bytes_from_host(); }
    uint64_t supply_h2d_bytes() const noexcept { return stage().supply_h2d_bytes(); }

    // Routing reuse-distance profiling: a separate
    // module with its own switch, not part of the supply telemetry.
    bool routing_reuse_enabled() const noexcept { return stage().routing_reuse_enabled(); }

    RoutingReuseProfiler::Curve routing_reuse_curve() const {
        return stage().routing_reuse_curve();
    }

    // One generated token's worth of decode accounting. A no-op when the sink is off.
    void record_supply_decode_token() {
        for (auto& stage : stages_) {
            stage->record_supply_decode_token();
        }
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
    V4LayerBodyTables tables() const noexcept { return stage().tables(); }

    // ---- the prefill sweep / routed bank -------------------------------------
    //
    // `forward_window` drives the window lifecycle below; the strategy, the
    // prompt-length gate and the sweep itself live in `PrefillController`
    // (`prefill_controller.hpp`). These are the host's entry points to it.
    bool prefill_sweep_enabled() const noexcept { return stage().prefill_sweep_enabled(); }

    // The prompt-length gate, resolved at load (`E / 4` unless configured). Reported
    // so a gate can name the switch point.
    uint32_t prefill_sweep_min_tokens() const noexcept { return stage().prefill_sweep_min_tokens(); }

    // Whether a window of `window_tokens` runs the sweep: enabled and feasible, and
    // at least the gate long. This is the **only** condition on the switch.
    bool prefill_sweep_engaged_for(uint32_t window_tokens) const noexcept {
        return stage().prefill_sweep_engaged_for(window_tokens);
    }

    // The strategy the last window began with, chosen from its length: the sweep at
    // or above the gate, the route-aware cached supply below it. Recorded rather than
    // inferred so a gate can read it.
    bool prefill_sweep_engaged() const noexcept { return stage().prefill_sweep_engaged(); }

    // `window_tokens` is the window length `W`, which the driver is the only one to
    // know at this point — the gate is read here and nowhere else. Both strategies
    // are prefill supplies; `prefill_sweep` enables them, the length picks which.
    void prefill_begin(uint32_t window_tokens) { stage().prefill_begin(window_tokens); }

    void prefill_before_layer(uint32_t layer) { stage_of_layer(layer).prefill_before_layer(layer); }

    void prefill_after_layer(uint32_t layer) { stage_of_layer(layer).prefill_after_layer(layer); }

    void prefill_end() { stage().prefill_end(); }

    // Sweep counters, for the gate: how many layer loads ran, how many experts they
    // streamed, and the deepest frontier the lookahead reached.
    const PrefillSweep& prefill_sweep() const noexcept { return stage().prefill_sweep(); }

    // The corridor's **fill** per layer — staging slots reading, staging slots
    // copying, and the lookahead layer's reserved-but-not-yet-arrived VRAM experts.
    // A healthy two-block corridor shows `reading` and `copying` both non-zero while a
    // body runs; one pinned at `E` with the other at zero is the parking lot. Empty
    // unless a sweep ran.
    const std::vector<PrefillSweep::BlockOccupancy>& sweep_occupancy() const noexcept {
        return stage().sweep_occupancy();
    }

    // Nanoseconds the sweep spent inside its layer loads (`dispatch` + `materialize`
    // + release), against the wall clock of the window. The split is what says
    // whether a swept prefill is transfer-bound or compute-bound, and how much a
    // load/compute overlap could recover.
    uint64_t sweep_load_ns() const noexcept { return stage().sweep_load_ns(); }
    // The dispatch half of the same accounting: time spent *submitting* reads, which
    // is what the double buffer pays to keep the drive busy across the compute.
    uint64_t sweep_io_ns() const noexcept { return stage().sweep_io_ns(); }
    // Inside `io_uring_enter` alone, and the SQE count it submitted. When this is
    // large the cost is the drive's queue, not the CPU.
    uint64_t direct_io_submit_ns() const noexcept { return stage().direct_io_submit_ns(); }
    uint64_t direct_io_requests_submitted() const noexcept {
        return stage().direct_io_requests_submitted();
    }
    uint64_t direct_io_submit_calls() const noexcept {
        return stage().direct_io_submit_calls();
    }
    // The transfer split (see `TieredExpertSupply`): host time blocked on NVMe
    // completions (`io_wait`), CPU time to submit the H2D copies (`h2d_enqueue`), and
    // host time blocked on those copies landing (`h2d_drain`, plus `h2d_drain_calls`
    // event syncs). Unlike the sweep-scoped `sweep_load_ns`, these accumulate across
    // **both** phases, which is what lets one bench attribute prefill and decode from
    // the same counters. `reset_supply_transfer_counters` slices between them.
    uint64_t supply_io_wait_ns() const noexcept { return stage().supply_io_wait_ns(); }
    uint64_t supply_h2d_enqueue_ns() const noexcept { return stage().supply_h2d_enqueue_ns(); }
    uint64_t supply_h2d_drain_ns() const noexcept { return stage().supply_h2d_drain_ns(); }
    uint64_t supply_h2d_drain_calls() const noexcept { return stage().supply_h2d_drain_calls(); }
    // CPU time in `dispatch`'s per-request loop (the registry reservation and the two
    // `O(catalog)` scans). Reported for both phases, since the loop is shared.
    uint64_t supply_dispatch_cpu_ns() const noexcept { return stage().supply_dispatch_cpu_ns(); }
    // Staging slots freed by the completion path instead of a boundary block.
    uint64_t supply_staging_released_on_completion() const noexcept {
        return stage().supply_staging_released_on_completion();
    }
    // Copies the mid-body pump issued.
    uint64_t supply_copies_pumped() const noexcept { return stage().supply_copies_pumped(); }
    void reset_supply_transfer_counters() noexcept { stage().reset_supply_transfer_counters(); }
    // Layers whose reads were in flight when a body started (1 = the double buffer
    // is engaged; 0 = the pool is too small for two layers and loads are serial).
    uint32_t sweep_lookahead_depth() const noexcept { return stage().sweep_lookahead_depth(); }
    // The lookahead length the **free blocks** allow at this instant — the smaller of
    // the free VRAM blocks and the free staging blocks. Derived, not configured: it
    // grows on a larger pool and shrinks to 0 when either runs out.
    uint32_t sweep_derived_ahead_capacity() const noexcept {
        return stage().sweep_derived_ahead_capacity();
    }
    // Test instrument: cap the sweep's read lookahead (0 = staging-bounded only).
    void set_sweep_read_ahead_max(uint32_t max_depth) noexcept {
        stage().set_sweep_read_ahead_max(max_depth);
    }

    // The start of a new sequence. Every layer's ring sentinels, counters and
    // committed-entry positions go back to what a freshly allocated layer holds,
    // so a second conversation cannot read the first one's context.
    //
    // The epoch is bumped here so a caller that keeps a description of the state
    // (the engine's prefix record) can tell that the state it described is gone.
    void reset_generation_state() {
        for (auto& stage : stages_) {
            stage->reset_generation_state();
        }
        ++state_epoch_;
    }

    // How many times the generation state has been reset. A monotonically
    // increasing counter rather than a flag, because the interesting question is
    // "is this the same generation the record was built against", not "has it ever
    // been reset".
    uint64_t state_epoch() const noexcept { return state_epoch_; }

private:
    // The owning stage of a layer, and the sole stage in the degenerate case. A
    // single-device run has one stage — `stage()` is the whole model; a pipeline
    // maps a layer id to its stage through the topology.
    V4StageHost& stage() { return *stages_.front(); }
    const V4StageHost& stage() const { return *stages_.front(); }
    V4StageHost& stage_of_layer(uint32_t layer_id) {
        return *stages_.at(topology_.stage_of(layer_id));
    }
    const V4StageHost& stage_of_layer(uint32_t layer_id) const {
        return *stages_.at(topology_.stage_of(layer_id));
    }

    uint32_t count_kind(V4AttentionKind kind) const noexcept {
        uint32_t count = 0;
        for (const auto& spec : layer_specs_) {
            if (spec.attention_kind == kind) ++count;
        }
        return count;
    }

    // The expert tier is built per stage (`V4StageHost::initialize_experts`), not
    // here: the stage owns the device, the pools and the supply.

    bool verbose_{false};
    AeonModelLoader loader_;
    DeepSeekV4Config config_;
    std::vector<V4LayerSpec> layer_specs_;
    // One report per stage, in stage order. A single-device run holds exactly one,
    // and `budget()` returns it; a pipeline holds one per stage.
    std::vector<MemoryBudgetReport> budgets_;
    uint32_t context_capacity_{0};
    size_t last_released_dense_bytes_{0};
    // Bumped by `reset_generation_state`; the engine's prefix record compares it to
    // tell whether the state it described has been reset underneath it.
    uint64_t state_epoch_{0};

    ParallelTopology topology_;
    // The stages: one device and one contiguous layer range each. A single-device
    // run is one stage; a pipeline is several, built in stage order. Held by pointer
    // because a stage owns pinned host memory and device handles and so is neither
    // copyable nor movable.
    std::vector<std::unique_ptr<V4StageHost>> stages_;
};

} // namespace aeon::core
