#pragma once

// -----------------------------------------------------------------------------
// The model host — what is resident, in the order the assembly has to happen.
//
// This is the component the composition plan calls **G1**, and it is the largest
// thing the graph rewrite was missing. It is deliberately *not* the forward pass:
// the graph (`core/v4_graph.hpp`) owns the order of operations, and this owns the
// objects they read and write. The split is what keeps the graph small enough to
// read — nothing in this file knows what a token is, and nothing in the graph
// knows how a tensor got into VRAM.
//
// It builds all fifteen steps of the plan's assembly (composition plan §5.1):
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
//   15. (nothing left — the graph above this file drives it)
//
// The expert half is a **lift**, not a second design: `V4Pipeline::initialize`
// performs the same sequence (composition plan §6.1, G1). The difference is that
// the historical version built and ran its own graph inline, while this hands the
// same objects to whichever graph the caller drives.
//
// The **streaming system enters the graph only here**. `V4Graph` never learns
// which tier answered a request: it calls the `V4RoutedExpertExecutor` the body
// declares, and that executor is this object's. That is the whole reason the
// storage half can be correct while the graph is numerics-only.
//
// What is *not* re-implemented here. Every object below already exists and is
// used as it stands: `AeonModelLoader`, `DeepSeekV4Config`, `V4ModelSpec`,
// `V4ModelContract`, `MemoryBudgetEngine`, `V4ModelResources`, `V4Layer`,
// `PipelineScratchBuffers`, `V4DeviceStreams`, `V4LayerBodyTables`. This is that
// assembly lifted into the rewrite rather than a second design of it.
//
// Verification note (plan Part V, second rule). A host is not an op, so there is
// no arithmetic oracle for it. Its gate is that the assembly's own invariants
// hold and that something above it produces oracle-checked numbers: the layers
// and the head stage are read back and compared, and the budget's own accounting
// is cross-checked against what was actually allocated. "It constructed without
// throwing" is not the evidence.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_device_streams.hpp"
#include "architecture/deepseek_v4/core/v4_expert_executor.hpp"
#include "architecture/deepseek_v4/core/v4_expert_supply.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body_batch.hpp"
#include "architecture/deepseek_v4/core/v4_model_contract.hpp"
#include "architecture/deepseek_v4/core/v4_model_resources.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline_scratch.hpp"
#include "backend/swizzled_w4a16/core/vram_expert_pool.hpp"
#include "infrastructure/backend_registry/expert_backend.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "infrastructure/core/expert_registry.hpp"
#include "infrastructure/core/host_expert_pool.hpp"
#include "infrastructure/core/expert_host_region.hpp"
#include "infrastructure/core/prefetch_staging.hpp"
#include "architecture/deepseek_v4/core/v4_prefill_sweep.hpp"
#include "infrastructure/core/supply_telemetry.hpp"
#include "infrastructure/hip_check.hpp"
#include "infrastructure/io/aligned_allocator.hpp"
#include "infrastructure/io/direct_io_reader.hpp"
#include "platform/rdna3/device.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
    // The refusal at step 6 is load-bearing rather than informative: a
    // configuration that does not fit is rejected before anything is uploaded,
    // because the alternative — uploading a backbone that does not fit and
    // discovering it as an allocation failure three layers in — is the failure
    // mode the budget engine exists to prevent.
    void initialize(const std::string& model_dir,
                    const AeonRuntimeConfig& runtime_cfg,
                    bool verbose = false) {
        free();
        verbose_ = verbose;

        select_compute_device(verbose_);
        streams_ = V4DeviceStreams::create();

        loader_.open_model(model_dir);
        const auto& expert_format = loader_.expert_format();
        const auto& backend = ExpertBackendRegistry::resolve(expert_format);
        if (!backend.supports_v4_pipeline) {
            throw std::runtime_error(
                "V4ModelHost: the selected artifact requires a different weight backend");
        }

        config_ = DeepSeekV4Config::load_from_json(model_dir + "/config.json");
        layer_specs_ = V4ModelSpec::resolve_layers(config_);
        V4ModelContract::validate(config_, loader_);

        // The budget is sized against the bytes the graph actually **uploads**, not
        // the container's file size: `embed.weight` stays host-side and the unused
        // `mtp.*` draft head is never read, and together those are 1.957 GiB the
        // old file-size reservation over-counted. Reserving them cost 148 Hot
        // expert slots (675 instead of 823 at context 256). The contract is the
        // authority on the uploaded set — the same table that validates the
        // artifact — so the budget cannot drift from what is placed on the device.
        budget_ = MemoryBudgetEngine::evaluate(
            runtime_cfg, config_, V4ModelContract::uploaded_dense_bytes(config_), expert_format);
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
        // state's own refusal (trap 40) is the thing a caller meets rather than a
        // silently clamped position.
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
            telemetry_.enable_jsonl(runtime_cfg.supply_telemetry_path, runtime_cfg.run_id);
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
                static_cast<unsigned long long>(demotion_queue_capacity_),
                executor_ ? "" : " (expert tier not built: no Hot VRAM slot)");
        }
    }

    void free() noexcept {
        // Flush the telemetry sink first, while the pools it observes still exist,
        // so the final summary captures the peak occupancy of the run being torn
        // down rather than a post-free zero.
        telemetry_.disable();

        // Teardown is the construction order reversed, and the two references that
        // matter are dropped first: the executor borrows the supply, the pool, the
        // staging arena and the registry, and the supply borrows the reader and the
        // pools. Freeing in any other order would leave a live object pointing at
        // freed storage.
        executor_.reset();
        supply_.clear();
        expert_scratch_.free();
        vram_pool_.free();
        host_pool_.free();
        staging_.reset();
        io_reader_.reset();
        completions_.clear();
        next_io_id_ = 1;

        // The layers free their own dense weights and attention state, the
        // resources and the scratch free theirs, and every `free()` nulls what it
        // released — so this is safe to call twice, once explicitly and once from
        // the destructor.
        layers_.clear();
        resources_.free();
        scratch_.free();
        batch_scratch_.free();
        if (d_prefill_carry_half_) { (void)hipFree(d_prefill_carry_half_); d_prefill_carry_half_ = nullptr; }
        if (d_prefill_carry_) { (void)hipFree(d_prefill_carry_); d_prefill_carry_ = nullptr; }
        prefill_carry_tokens_ = 0;
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

    PipelineScratchBuffers& scratch() noexcept { return scratch_; }
    const PipelineScratchBuffers& scratch() const noexcept { return scratch_; }

    // --- the layer-major prefill working set (Step 6) ------------------------
    //
    // Two buffers, and the distinction between them is the whole reason they are
    // separate (§6.4): the **chunk workspace** holds per-op temporaries for the
    // rows in flight and is recycled as layers advance, while the **carry** holds
    // the residual being transformed and must survive all 43 layers of a pass.

    // The per-layer chunk workspace. `V4LayerBodyBatchScratch` sizes itself from a
    // layer's own capacities (the indexer's candidate scores and top-k), so left to
    // itself it would be re-allocated whenever the layer changes. When the window
    // workspace was allocated at load for the worst case across the layers
    // (`allocate_prefill_workspace`), that one buffer already covers every layer and
    // this is a no-op — which is what makes the layer-major pass allocation-free.
    void ensure_batch_scratch(uint32_t layer_id, uint32_t count) {
        if (prefill_workspace_ready_ && count <= batch_scratch_.token_count()) return;
        if (batch_scratch_count_ == count && batch_scratch_layer_ == layer_id) return;
        batch_scratch_.allocate(layer(layer_id), count);
        batch_scratch_layer_ = layer_id;
        batch_scratch_count_ = count;
    }

    V4LayerBodyBatchScratch& batch_scratch() noexcept { return batch_scratch_; }
    const V4LayerBodyBatchScratch& batch_scratch() const noexcept { return batch_scratch_; }

    // ---- Step 6 item 7: the prefill workspace, derived from the knobs ---------
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
        if (window_tokens == 0 || chunk_tokens == 0) {
            throw std::invalid_argument(
                "V4ModelHost: the prefill window and chunk must be positive");
        }
        if (chunk_tokens > V4LayerBodyBatchScratch::kMaxTokens) {
            throw std::invalid_argument(
                "V4ModelHost: prefill chunk " + std::to_string(chunk_tokens) +
                " exceeds the body's row cap of " +
                std::to_string(V4LayerBodyBatchScratch::kMaxTokens));
        }

        ensure_prefill_carry(window_tokens);

        // Worst case across the layers: the composed row-set, the indexer's
        // candidate scores and its top-k are each the maximum any layer needs.
        uint32_t max_compressed_capacity = 0;
        uint32_t max_index_topk = 0;
        uint32_t max_local_capacity = 0;
        for (const auto& layer : layers_) {
            const auto& layout = layer.state_layout();
            max_compressed_capacity = std::max(max_compressed_capacity, layout.compressed_capacity);
            max_index_topk = std::max(max_index_topk, layout.index_topk);
            max_local_capacity = std::max(max_local_capacity, layout.local_capacity);
        }
        batch_scratch_.allocate_capacity(max_compressed_capacity, max_index_topk,
                                         max_local_capacity, chunk_tokens);
        const size_t allowance = batch_scratch_allowance_bytes(chunk_tokens);
        if (batch_scratch_.bytes() > allowance) {
            throw std::runtime_error(
                "V4ModelHost: the prefill batch scratch is " +
                std::to_string(batch_scratch_.bytes() / (1024 * 1024)) +
                " MiB but the budget allows " +
                std::to_string(allowance / (1024 * 1024)) +
                " MiB for chunk " + std::to_string(chunk_tokens) +
                " — raise BATCH_SCRATCH_BYTES_PER_ROW or lower prefill_chunk");
        }

        prefill_window_tokens_ = window_tokens;
        prefill_chunk_tokens_ = chunk_tokens;
        prefill_workspace_ready_ = true;
    }

    // What was allocated, for the report and the gates.
    uint32_t prefill_window_tokens() const noexcept { return prefill_window_tokens_; }
    uint32_t prefill_chunk_tokens() const noexcept { return prefill_chunk_tokens_; }
    size_t prefill_carry_bytes() const noexcept {
        const size_t hc_dim = static_cast<size_t>(config_.hc_mult) *
                              static_cast<size_t>(config_.hidden_size);
        return static_cast<size_t>(prefill_carry_tokens_) * hc_dim *
               (sizeof(uint16_t) + sizeof(float));
    }
    size_t prefill_batch_scratch_bytes() const noexcept { return batch_scratch_.bytes(); }
    // The decode workspace's real size (`V4PipelineScratchBuffers` + the routed-expert
    // scratch), for the report and the load-time check against the budget's allowance.
    size_t decode_scratch_bytes() const noexcept {
        return scratch_.bytes() + expert_scratch_.bytes();
    }
    // The pinned host staging arena's footprint, so all three prefill buffers can be
    // reported side by side rather than only the VRAM pair.
    size_t staging_bytes() const noexcept {
        if (!staging_ || !experts_ready()) return 0;
        return static_cast<size_t>(staging_->slot_count()) *
               loader_.expert_format().payload_bytes;
    }

    // The residual carry: one window's worth of per-token residual, both fp16 and
    // fp32, held in VRAM for the whole layer-major pass. Grows only, so a window
    // of a given size is allocated once and reused by every later pass that fits.
    void ensure_prefill_carry(uint32_t tokens) {
        if (tokens == 0) {
            throw std::invalid_argument("V4ModelHost: a prefill carry cannot be zero tokens");
        }
        if (tokens <= prefill_carry_tokens_) return;

        if (d_prefill_carry_half_) { (void)hipFree(d_prefill_carry_half_); d_prefill_carry_half_ = nullptr; }
        if (d_prefill_carry_) { (void)hipFree(d_prefill_carry_); d_prefill_carry_ = nullptr; }
        prefill_carry_tokens_ = 0;

        const uint32_t hc_dim = static_cast<uint32_t>(config_.hc_mult) *
                                static_cast<uint32_t>(config_.hidden_size);
        CHECK_HIP(hipMalloc(&d_prefill_carry_half_,
                            static_cast<size_t>(tokens) * hc_dim * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_prefill_carry_,
                            static_cast<size_t>(tokens) * hc_dim * sizeof(float)));
        prefill_carry_tokens_ = tokens;
    }

    half* prefill_carry_half() noexcept { return d_prefill_carry_half_; }
    float* prefill_carry() noexcept { return d_prefill_carry_; }
    uint32_t prefill_carry_tokens() const noexcept { return prefill_carry_tokens_; }

    const V4DeviceStreams& streams() const noexcept { return streams_; }

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
    // before the caller reads the logits back — a lease grants no ordering, so it
    // must be held for as long as compute reading that slot may be in flight
    // (trap 41), and the logits readback is the compute-stream boundary that makes
    // the release safe. Exposed here rather than on the seam because releasing is a
    // property of the concrete tiered executor, not of the interface the body sees.
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
    // telemetry. Zero whenever the pool can hold a token's `6 x 43` leases, which
    // is the intended steady state; a non-zero value is how a gate says it ran the
    // starved regime (Step 4).
    uint64_t forced_drains() const noexcept {
        return telemetry_.forced_drains();
    }

    // Staging slots not AVAILABLE. A gate asserts this returns to 0 at the end of a
    // run — the arena must not leak. Reads the concrete executor's arena, so it is
    // safe only after `initialize_experts`; returns 0 when the arena does not exist.
    uint32_t staging_in_use_slots() const noexcept {
        return staging_ ? staging_->in_use_slots() : 0;
    }

    // The staging arena's slot count. A batch dispatcher must not assign more
    // distinct staging indices than this, and the layer-major window refuses a
    // chunk whose `6C` requests would exceed it.
    uint32_t staging_slot_count() const noexcept {
        return staging_ ? staging_->slot_count() : 0;
    }

    // Re-size the staging arena at runtime: a **depth** change, not a format change.
    // The depth is a resource budget the transfer corridor consumes, not a behaviour
    // constant — above the minimum the design needs, more slots only give the
    // lookahead more room to run ahead, and the throughput must not change. This is
    // what lets a host size the corridor from its remaining RAM instead of from a
    // figure tuned on one machine, and it is what the depth-sensitivity gate varies.
    //
    // Preconditions, refused rather than assumed: every staging slot free and no
    // expert lease outstanding. The resize destroys and recreates the per-slot
    // events, so nothing may still reference one — which is exactly the state a
    // window boundary leaves behind. `prefill_active_` is deliberately **not** the
    // guard: it records the strategy the last window chose and stays set afterwards
    // (gates read it), so it does not mean "a window is running now". The two state
    // conditions below are the exact ones, and an in-progress window cannot satisfy
    // both while a caller outside the engine is on the stack.
    bool resize_staging_slots(uint32_t slots) {
        if (staging_ == nullptr) return false;
        if (staging_->in_use_slots() != 0) return false;
        if (outstanding_expert_leases() != 0) return false;
        if (slots < PrefetchStagingArena::TOTAL_STAGING_SLOTS) return false;
        // A shared region has a **fixed** total: its corridor can only be resized by
        // moving the boundary, which is what the phase transitions do. Letting a
        // caller resize the arena alone would silently over-commit the region, so the
        // two directions are different operations and this one refuses.
        if (host_region_active_) return false;

        staging_->resize(slots);
        const uint32_t per_layer = static_cast<uint32_t>(config_.n_routed_experts);
        sweep_staging_banks_ = per_layer == 0
            ? 1u
            : std::max<uint32_t>(1u, slots / per_layer);
        prefill_sweep_.configure(&supply_, &registry_, sweep_staging_banks_);
        // The budget report is the plan's figure; keep it equal to the allocation so
        // the two cannot drift after a resize.
        budget_.transient_staging_bytes =
            static_cast<size_t>(slots) * staging_->payload_bytes();
        return true;
    }

    // ---- the Warm/staging partition -----------------------------------------
    //
    // Warm and the corridor are one pinned region cut by a boundary
    // (`ExpertHostRegion`). Both phases move it, and the two moves are exact
    // inverses because the region's total is fixed:
    //
    //   * decode, and a **chunked** prefill, need only the corridor's base
    //     (`max(12, min(6C, E))` — the deduplicated set a batch can bind, with the
    //     decode double-buffer as the floor). Everything else is Warm residency,
    //     which is where decode's NVMe hits are decided;
    //   * a **swept** prefill grows the corridor to `blocks x E` for the length of the
    //     window and hands the difference back at `prefill_end`.
    //
    // Moving it is a pointer and a free-list edit: the storage is allocated once, so
    // no page is committed or pinned by the move. The only real cost is that a slot
    // surrendered by Warm must be **drained**, and a Warm expert that is dropped is
    // re-read from NVMe on demand — which is why the move happens only when a swept
    // window actually needs the room, and why the corridor's base rather than its
    // maximum is what decode gets.
    //
    // `warm_slots` is the whole description of the split: the Warm head is
    // `[0, warm_slots)` and the corridor is `[warm_slots, region_slots)`. Every step
    // is idempotent, so a caller may apply the same partition repeatedly.
    void apply_host_partition(uint32_t warm_slots, uint32_t staging_slots) {
        if (!host_region_active_ || staging_ == nullptr) return;
        // **Idempotent by construction.** `prefill_begin` is not the only caller that
        // enters a window — the driver enters one itself when it begins a window, so
        // the same partition is requested twice for the same window. Re-applying it
        // would re-base the arena and destroy its per-slot events while the sweep's
        // own lookahead is using them, so an already-current partition is a no-op.
        if (warm_slots == current_warm_slots_ &&
            staging_slots == staging_->slot_count()) {
            return;
        }
        // A boundary move re-bases the arena and re-creates its per-slot events, so
        // nothing may still reference one. This is the same precondition
        // `resize_staging_slots` enforces, and it is what a phase boundary leaves.
        if (staging_->in_use_slots() != 0 || outstanding_expert_leases() != 0) {
            throw std::logic_error(
                "V4ModelHost: cannot move the host partition with a transfer or lease live "
                "(staging_in_use=" + std::to_string(staging_->in_use_slots()) +
                " of " + std::to_string(staging_->slot_count()) +
                ", leases=" + std::to_string(outstanding_expert_leases()) + ")");
        }
        if (warm_slots + staging_slots != host_region_.slot_count()) {
            throw std::logic_error(
                "V4ModelHost: the host partition does not cover the region exactly");
        }

        // Shrink: demote the Warm tail **before** the capacity moves under it, so the
        // free-list edit finds every surrendered slot empty.
        if (warm_slots < registry_.usable_host_capacity()) {
            registry_.release_host_tail(warm_slots);
            registry_.shrink_host_capacity(warm_slots);
        }
        host_pool_.bind(host_region_.base(), warm_slots, host_region_.format(),
                        host_region_.is_pinned());
        if (warm_slots > registry_.usable_host_capacity()) {
            registry_.grow_host_capacity(warm_slots);
        }
        staging_->bind(host_region_.slot_ptr(warm_slots), staging_slots,
                       host_region_.format());

        const uint32_t per_layer = static_cast<uint32_t>(config_.n_routed_experts);
        sweep_staging_banks_ = per_layer == 0
            ? 1u
            : std::max<uint32_t>(1u, staging_slots / per_layer);
        prefill_sweep_.configure(&supply_, &registry_, sweep_staging_banks_);
        budget_.transient_staging_bytes =
            static_cast<size_t>(staging_slots) * staging_->payload_bytes();
        current_warm_slots_ = warm_slots;
    }

    // The partition right now: Warm slots resident, and the corridor's slots.
    uint32_t warm_host_slots_current() const noexcept { return current_warm_slots_; }
    bool staging_shares_warm_region() const noexcept { return host_region_active_; }

    // Layer-sized staging banks the sweep's arena holds, derived from the arena's
    // actual depth (`slots / experts_per_layer`). The default is `2` (R2); the
    // derivation exists so a runtime resize moves the arena and the sweep's bank
    // indexing together, with no second place for the two to disagree.
    uint32_t sweep_staging_banks() const noexcept { return sweep_staging_banks_; }

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
    // default and the `demotion_queue_capacity` override are resolved. Reported so a
    // gate can name which arm of the Step 5 A/B it ran.
    uint64_t demotion_queue_capacity() const noexcept { return demotion_queue_capacity_; }

    // Diagnostics, for an assembly gate: the registry's residency claims and the
    // pool it made them against. Not used by the graph.
    const ExpertRegistry& registry() const noexcept { return registry_; }
    UnifiedVRAMExpertPool& vram_pool() noexcept { return vram_pool_; }
    const SupplyTelemetry& telemetry() const noexcept { return telemetry_; }

    // Tell the telemetry which phase the next dispatch belongs to. The caller is
    // the generation loop, which already knows whether the token it is about to
    // advance is a prompt token (prefill) or a generated one (decode); this only
    // forwards that fact, it does not derive it. A no-op when the sink is off.
    //
    // It also drives the frozen-Warm policy (Step 6 D-b) when
    // `freeze_warm_during_prefill` is set: prefill enters the frozen mode, decode
    // leaves it. Leaving settles any in-flight shadow copy first (`reap` is
    // event-query only, no CPU synchronization) so the idle residencies are visible
    // before they are released; a copy that is still genuinely in flight is
    // reclaimed by eviction when it settles.
    void set_supply_phase(bool prefill) {
        telemetry_.set_phase(prefill ? RoutingPhase::Prefill : RoutingPhase::Decode);
        if (!freeze_warm_during_prefill_ || !experts_ready()) return;
        // Only the **transition** matters, and only on the way down does work have to
        // be done: leaving frozen must be total, because decode admits single
        // ownership and the invariant refuses a shadow left behind. So the boundary
        // drains the expert streams (a legate boundary, not the request path) and
        // reaps, which completes every in-flight copy before the idle shadows are
        // released. The swept prefill owns this flag while it is running.
        if (registry_.prefill_streaming() || prefill == supply_phase_prefill_) return;
        supply_phase_prefill_ = prefill;
        if (prefill) {
            registry_.set_warm_frozen(true);
        } else {
            drain_expert_streams();
            supply_.reap_registry_transfers();
            registry_.set_warm_frozen(false);
        }
    }

    // Frozen-prefill state, for a gate: whether the registry is in frozen mode and
    // how many Warm-owned experts currently hold an extra VRAM copy (Step 6 D-b).
    bool warm_frozen() const noexcept { return registry_.warm_frozen(); }
    uint32_t shadow_resident_count() const noexcept {
        return registry_.shadow_resident_count();
    }
    int32_t shadow_slot_of(uint32_t gid) const { return registry_.shadow_slot_of(gid); }
    uint64_t shadow_copies() const noexcept { return registry_.shadow_copies; }

    // Logical Warm bytes the supply served in a phase (Step 6 outcome 3). Requires
    // the telemetry sink to have been enabled.
    uint64_t supply_logical_bytes_from_warm(bool prefill) const noexcept {
        return telemetry_.logical_bytes_from_warm(
            prefill ? SupplyTelemetryPhase::Prefill : SupplyTelemetryPhase::Decode);
    }

    // Per-layer outcome counts (thesis-1 measurement), from the supply telemetry:
    // how many dispatches in the given phase were answered entirely from Hot, from
    // Hot+Warm with no Cold, and with at least one Cold. `outcome` is 0/1/2 for
    // AllHot/WarmNoCold/HasCold.
    uint64_t layer_outcome_count(bool prefill, uint32_t outcome) const noexcept {
        if (outcome > 2) return 0;
        return telemetry_.layer_outcome_count(
            prefill ? SupplyTelemetryPhase::Prefill : SupplyTelemetryPhase::Decode,
            static_cast<SupplyTelemetry::LayerOutcome>(outcome));
    }

    uint64_t layer_dispatches(bool prefill) const noexcept {
        return telemetry_.layer_dispatches(
            prefill ? SupplyTelemetryPhase::Prefill : SupplyTelemetryPhase::Decode);
    }

    // Lifetime supply traffic, for a benchmark. Never reset, independent of the
    // JSONL sink, so a throughput run can state bytes read without opening one.
    uint64_t supply_requests() const noexcept { return telemetry_.lifetime_requests(); }
    uint64_t supply_bytes_from_nvme() const noexcept {
        return telemetry_.lifetime_nvme_bytes();
    }
    uint64_t supply_bytes_from_host() const noexcept {
        return telemetry_.lifetime_host_bytes();
    }
    uint64_t supply_h2d_bytes() const noexcept { return telemetry_.lifetime_h2d_bytes(); }

    // Routing reuse-distance profiling (Phase 1 of the routing study): a separate
    // module with its own switch, not part of the supply telemetry.
    bool routing_reuse_enabled() const noexcept {
        return reuse_profiler_.enabled();
    }

    RoutingReuseProfiler::Curve routing_reuse_curve() const {
        return reuse_profiler_.curve();
    }

    // One generated token's worth of decode accounting. A no-op when the sink is off.
    void record_supply_decode_token() {
        telemetry_.record_decode_token();
    }

    // Dense-container page-cache residency dropped after the uploads, in bytes (0
    // when the runtime was configured not to release). Reported rather than
    // inferred: `madvise` is a hint, so the only figure worth printing is this one.
    size_t released_dense_bytes() const noexcept { return last_released_dense_bytes_; }

    // Host read access, for building an oracle: `embed.weight` is reached through
    // `resources().host_embed_table` (a pointer into the container), while
    // `head.weight`, `norm.weight` and the three `hc_head` tensors are device-only
    // and have to come from the container's own host mapping.
    const AeonModelLoader& loader() const noexcept { return loader_; }

    // The two RoPE bases, in the shape the layer body consumes. Sliding layers use
    // the plain base (theta 10000) and compressed layers the YaRN-on-compressed one
    // (theta 160000, factor 16) — plan 2.3, trap 7. Handed over as a plain struct
    // so the body keeps no dependency on the resources object.
    V4LayerBodyTables tables() const noexcept {
        V4LayerBodyTables result;
        result.sliding_cos = resources_.d_cos_cache;
        result.sliding_sin = resources_.d_sin_cache;
        result.compressed_cos = resources_.d_compressed_cos_cache;
        result.compressed_sin = resources_.d_compressed_sin_cache;
        return result;
    }

    // ---- Step 6 item 6: the prefill sweep ------------------------------------
    //
    // `forward_window` drives these. The whole switch is local to a window: the
    // sweep frees only what the pass needs on entry, streams the layers in order, and
    // leaves the residents that were present at entry (plus the restored drain set)
    // on exit, so a window is self-contained and decode can resume the moment it ends
    // on the set it had before.
    //
    // Enabled only when `AeonRuntimeConfig::prefill_sweep` is set **and** the Hot pool
    // can hold a whole layer (the sweep has no victim to evict). This is the
    // configuration-level feasibility; whether a given window *uses* the sweep also
    // depends on the prompt-length gate, which `prefill_sweep_engaged_for` applies.
    // Otherwise the window drives the per-token dispatch, which is what the engine's
    // `forward_token` path uses.
    bool prefill_sweep_enabled() const noexcept {
        return prefill_sweep_requested_ && prefill_sweep_.is_feasible();
    }

    // The prompt-length gate, resolved at load (`E / 4` unless configured). Reported
    // so a gate can name the switch point.
    uint32_t prefill_sweep_min_tokens() const noexcept { return sweep_min_tokens_; }

    // Whether a window of `window_tokens` runs the sweep: enabled and feasible, and
    // at least the gate long. This is the **only** condition on the switch.
    bool prefill_sweep_engaged_for(uint32_t window_tokens) const noexcept {
        return prefill_sweep_enabled() && window_tokens >= sweep_min_tokens_;
    }

    // The strategy the last window began with, chosen from its length: the sweep at
    // or above the gate, the route-aware cached supply below it. Recorded rather than
    // inferred so a gate can read it.
    bool prefill_sweep_engaged() const noexcept { return sweep_active_; }

    // `window_tokens` is the window length `W`, which the driver is the only one to
    // know at this point — the gate is read here and nowhere else. Both strategies
    // are prefill supplies; `prefill_sweep` enables them, the length picks which.
    void prefill_begin(uint32_t window_tokens) {
        prefill_active_ = prefill_sweep_enabled();
        sweep_active_ = prefill_active_ && window_tokens >= sweep_min_tokens_;
        if (!prefill_active_) return;
        drain_expert_streams();
        supply_.reap_registry_transfers();
        executor_->release_leases();
        // The window's shape decides the partition: a **swept** window grows the
        // corridor to its configured size and takes it out of Warm (restored in
        // `prefill_end`); a routed window needs only the corridor's base and leaves
        // Warm at its maximum, which is where decode's hits come from. Placed here,
        // while nothing is in flight, because a move re-bases the arena and re-creates
        // its per-slot events.
        if (host_region_active_ && staging_prefill_slots_ > staging_base_slots_) {
            apply_host_partition(
                sweep_active_ ? warm_slots_prefill_ : warm_slots_base_,
                sweep_active_ ? staging_prefill_slots_ : staging_base_slots_);
        }
        if (sweep_active_) {
            prefill_sweep_.begin();
        } else {
            // The routed bank (Step 4): drain one layer's worth of the worst-LRU
            // residents, keep the rest resident, and admit route-aware. The layer's
            // union grows into the freed `E` slots and is leased until the layer
            // retires, so it persists across the layer's chunks without a whole-layer
            // pre-load.
            registry_.begin_prefill_stream(
                registry_.experts_per_layer,
                ExpertRegistry::PrefillAlloc::BoundedEvict);
        }
    }

    void prefill_before_layer(uint32_t layer) {
        if (!prefill_active_) return;
        if (sweep_active_) prefill_sweep_.before_layer(layer);
        // The routed path needs no pre-load: the body's router drives admission, and
        // the union is held resident by its leases until the layer retires.
    }

    void prefill_after_layer(uint32_t layer) {
        if (!prefill_active_) return;
        if (sweep_active_) {
            prefill_sweep_.after_layer(layer);
            return;
        }
        // The layer is dead the moment it retires, so its prefill-admitted set is
        // released while the residents present at entry are spared (Step 1). Unlike
        // the sweep — which loads a layer in one batch and settles it before the body
        // — the routed path's **last chunk** may have left an upload in flight, and
        // the release refuses a pending transfer, so settle and reap it first.
        drain_expert_streams();
        supply_.reap_registry_transfers();
        registry_.release_layer(layer);
    }

    void prefill_end() {
        if (!prefill_active_) return;
        supply_.reap_registry_transfers();
        if (sweep_active_) {
            prefill_sweep_.end();
        } else {
            // Settle the routed path's in-flight uploads before the mode change: the
            // end refuses a pending transfer.
            drain_expert_streams();
            supply_.reap_registry_transfers();
            registry_.end_prefill_stream();
        }
        // Hand the corridor's prefill borrow back to Warm. The window is over and
        // everything is drained, so the surrendered slots return to the Warm free list
        // and decode resumes with the larger residency — which is the whole reason the
        // boundary moved rather than the arena simply being sized for the worst case.
        // A restored expert is re-read on demand, so this costs nothing until one is
        // actually wanted.
        if (host_region_active_ && warm_slots_base_ > current_warm_slots_) {
            apply_host_partition(warm_slots_base_, staging_base_slots_);
        }
        // Reload whatever the drain freed, through the normal cold path, so decode
        // resumes on the set the pool held before the pass (the plan's restore
        // requirement). Shared by both strategies.
        restore_prefill_residents(loader_.expert_format());
        // `prefill_active_`/`sweep_active_` deliberately stay set: they record the
        // strategy the window chose, for the gates that read the choice afterwards.
        // `prefill_begin` re-decides both for the next window.
    }

    // Sweep counters, for the gate: how many layer loads ran, how many experts they
    // streamed, and the deepest frontier the lookahead reached.
    const V4PrefillSweep& prefill_sweep() const noexcept { return prefill_sweep_; }

    // The corridor's **fill** per layer — staging slots reading, staging slots
    // copying, and the lookahead layer's reserved-but-not-yet-arrived VRAM experts.
    // The pipeline model (supply-chain plan §0) says a healthy two-block corridor
    // shows `reading` and `copying` both non-zero while a body runs; one pinned at `E`
    // with the other at zero is the parking lot. Empty unless a sweep ran.
    const std::vector<V4PrefillSweep::BlockOccupancy>& sweep_occupancy() const noexcept {
        return prefill_sweep_.occupancy_samples();
    }

    // Nanoseconds the sweep spent inside its layer loads (`dispatch` + `materialize`
    // + release), against the wall clock of the window. The split is what says
    // whether a swept prefill is transfer-bound or compute-bound, and how much a
    // load/compute overlap could recover.
    uint64_t sweep_load_ns() const noexcept { return prefill_sweep_.load_ns(); }
    // The dispatch half of the same accounting: time spent *submitting* reads, which
    // is what the double buffer pays to keep the drive busy across the compute.
    uint64_t sweep_io_ns() const noexcept { return prefill_sweep_.io_ns(); }
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
    // Staging slots freed by the completion path instead of a boundary block (P2.2).
    uint64_t supply_staging_released_on_completion() const noexcept {
        return supply_.staging_released_on_completion();
    }
    // Copies the mid-body pump issued (P2.3).
    uint64_t supply_copies_pumped() const noexcept { return supply_.copies_pumped(); }
    void reset_supply_transfer_counters() noexcept { supply_.reset_transfer_counters(); }
    // Layers whose reads were in flight when a body started (1 = the double buffer
    // is engaged; 0 = the pool is too small for two layers and loads are serial).
    uint32_t sweep_lookahead_depth() const noexcept {
        return prefill_sweep_.lookahead_depth();
    }
    // The lookahead length the **free blocks** allow at this instant (plan R5) — the
    // smaller of the free VRAM blocks and the free staging blocks. Derived, not
    // configured: it grows on a larger pool and shrinks to 0 when either runs out.
    uint32_t sweep_derived_ahead_capacity() const noexcept {
        return prefill_sweep_.derived_lookahead_capacity();
    }
    // Test instrument: cap the sweep's read lookahead (0 = staging-bounded only).
    void set_sweep_read_ahead_max(uint32_t max_depth) noexcept {
        prefill_sweep_.set_read_ahead_max(max_depth);
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

    // Steps 10–15 of §5.1 — the expert tier, in the order the loop needs it.
    //
    // Skipped, not faked, when the budget left no Hot VRAM slot. The dense graph
    // is still constructible (the head stage reads no expert), but the graph is
    // not runnable, and `executor()` refuses rather than hand back an executor
    // whose pool is empty and whose first dispatch would throw.
    void initialize_experts(const AeonRuntimeConfig& runtime_cfg) {
        if (budget_.hot_vram_slots == 0) return;

        const auto& format = loader_.expert_format();
        const uint32_t warm_slots = budget_.warm_host_slots;

        // 10 — the Hot VRAM pool, the staging arena, and the direct reader. The
        // arena and the reader are built for the *artifact's* format, not the
        // backend's default, so a payload's staging slot is the artifact's own
        // `payload_bytes` wide.
        //
        // The arena's slot count is the decode shape unless a prefill chunk is
        // configured (Step 6 D4): a chunk issues up to `6C` deduplicated transfers
        // as one set, and each distinct expert needs its own slot in transit. Sized
        // to the **ceiling** `6C` here, so no transfer ever waits for a slot; the
        // Step 7 sweep picks the smaller concurrency depth.
        vram_pool_.allocate(budget_.hot_vram_slots, format);
        // A chunk's deduplicated distinct set can never exceed the **layer's** expert
        // count: dedup collapses `6C` requests onto at most `n_routed_experts`
        // experts. Sizing to `6C` alone asks for 384 slots at `C = 64` (5.1 GiB
        // pinned) where 256 will do, so the term is capped by the layer here — and
        // this is the same ceiling the graph's guard checks against, which is why a
        // legal wide chunk is no longer refused.
        //
        // The prefill sweep loads a **whole layer** in one batch, and with the deferred
        // drain it holds one layer's uploads in flight while the lookahead reads the
        // next, so it needs `banks * experts_per_layer` slots (Step 6 item 6; Phase 1 of
        // the supply-chain hot-path plan). `staging_slot_count` is shared with the
        // budget report, so the figure the plan prints is the figure allocated here.
        const uint32_t experts_per_layer = static_cast<uint32_t>(config_.n_routed_experts);
        const uint32_t dedup_ceiling = std::min<uint32_t>(
            PrefetchStagingArena::EXPERTS_PER_HORIZON *
                std::max<uint32_t>(1, runtime_cfg.prefill_chunk),
            experts_per_layer);
        const auto staging = aeon::core::staging_slot_counts(runtime_cfg, experts_per_layer);
        const uint32_t staging_slots = staging.prefill;

        // The host region: **one** pinned allocation shared by the Warm tier and the
        // corridor, cut by a boundary the phases move (`apply_host_partition`). Its
        // total is the configured host budget, so the RAM a run holds is the number
        // the user set — Warm and staging are no longer two settings to add and
        // overshoot.
        //
        // With no host budget there is no region and the corridor allocates its own
        // memory, which is the decode-only and gate shape.
        const uint32_t region_slots = budget_.host_region_slots;
        host_region_active_ = region_slots > 0;
        staging_base_slots_ = staging.base;
        staging_prefill_slots_ = staging.prefill;
        if (host_region_active_) {
            if (region_slots < staging.base) {
                throw std::runtime_error(
                    "V4ModelHost: the host region (" + std::to_string(region_slots) +
                    " slots) cannot hold the corridor's base (" +
                    std::to_string(staging.base) + " slots)");
            }
            host_region_.allocate(region_slots, format);
            warm_slots_base_ = region_slots - staging.base;
            warm_slots_prefill_ = region_slots - staging.prefill;
            current_warm_slots_ = warm_slots_base_;
            // Start at the base partition: decode and a chunked prefill are the
            // opening state, and a swept window moves the boundary in `prefill_begin`.
            staging_ = std::make_unique<PrefetchStagingArena>(
                host_region_.slot_ptr(warm_slots_base_), staging.base, format);
        } else {
            staging_ = std::make_unique<PrefetchStagingArena>(format, staging_slots);
        }
        // The sweep's bank count is **derived from the arena it actually has**, not
        // from a configured constant: `staging_slots / experts_per_layer` whole
        // layer-sized banks. A depth change moves the arena and this derivation
        // together, so there is no second place for the two to disagree (see
        // `apply_host_partition`).
        sweep_staging_banks_ = experts_per_layer == 0
            ? 1u
            : std::max<uint32_t>(1u, staging_->slot_count() / experts_per_layer);

        // The direct reader's ring must hold **every read that can be outstanding at
        // once**, not one layer's worth. `dispatch()` queues a batch's cold requests
        // before it submits once, and with the decoupled corridor (plan P2.6) more
        // than one layer's reads can be in flight simultaneously — bounded by the
        // staging arena, since a read needs a staging slot. Sizing this to a single
        // layer while two are outstanding over-subscribes the completion queue (it is
        // `2 x SQ`); the kernel then cannot post completions and `wait_for_completion`
        // stalls, which measured as `io_wait 1.7 -> 12.8 s`. So the ring follows the
        // staging depth.
        size_t batch_requests = direct_requests_per_expert(format) * dedup_ceiling;
        if (runtime_cfg.prefill_sweep) {
            batch_requests = std::max<size_t>(
                batch_requests,
                direct_requests_per_expert(format) * static_cast<size_t>(staging_slots));
        }
        const uint32_t io_queue_depth = static_cast<uint32_t>(
            std::max<size_t>(64, batch_requests));
        io_reader_ = std::make_unique<aeon::io::DirectIOReader>(
            io_queue_depth, true, format.sector_size);

        // 11 — the registry. It is what decides residency for every request, and
        // it saturates VRAM at construction: every Hot slot is owned from the
        // first token on, so the production steady state (a cold miss must evict a
        // resident) is the only state that exists.
        registry_.init(static_cast<uint32_t>(config_.num_hidden_layers),
                       static_cast<uint32_t>(config_.n_routed_experts),
                       budget_.hot_vram_slots, warm_slots,
                       runtime_cfg.preload_warm_host);
        // The per-request audit is a debugging instrument (see
        // `AeonRuntimeConfig::validate_registry_each_request`); the boundary audits
        // and `invariants_hold()` run regardless of it.
        registry_.set_validate_each_request(runtime_cfg.validate_registry_each_request);

        // 12 — Hot, then Warm, filled by batched `O_DIRECT` reads. This is the
        // only step with real mass; the byte-exactness of every route it uses is
        // the item-21 gate's subject, so nothing here re-checks it.
        preload_hot_experts(format);
        if (warm_slots > 0) {
            // The Warm pool is a **view** over the region's head when one is shared,
            // and an allocation of its own otherwise. Either way it is `warm_slots`
            // wide, which is the capacity the registry was initialised with.
            if (host_region_active_) {
                host_pool_.bind(host_region_.base(), warm_slots, host_region_.format(),
                                host_region_.is_pinned());
            } else {
                host_pool_.allocate(warm_slots, format);
            }
            if (runtime_cfg.preload_warm_host) {
                preload_warm_experts(format);
            }
        }

        // 13 — the tiered supply, on the four shared streams.
        supply_.configure(
            &loader_,
            &vram_pool_,
            warm_slots > 0 ? &host_pool_ : nullptr,
            &registry_,
            staging_.get(),
            &telemetry_,
            io_reader_.get(),
            &completions_,
            &next_io_id_,
            streams_.compute, streams_.sdma, streams_.sdma_cold, streams_.demotion,
            format.payload_bytes,
            runtime_cfg.demotion_queue_capacity > 0
                ? runtime_cfg.demotion_queue_capacity
                : (runtime_cfg.enable_warm_refill
                    ? V4ExpertSupplyCoordinator::DEFAULT_DEMOTION_QUEUE_CAPACITY : 0));
        demotion_queue_capacity_ = supply_.demotion_queue_capacity();
        freeze_warm_during_prefill_ = runtime_cfg.freeze_warm_during_prefill;

        // 14 — the routed-expert scratch and the production executor. The executor
        // borrows the four streams the host owns, so its capacity fallback drains
        // exactly the set that carries expert traffic.
        expert_scratch_.allocate();

        // The decode workspace's real allocations against the budget's allowance
        // (Step 6 item 7): `scratch_` is allocated far above in this function, so the
        // two together are what the report's decode term stands for. Checked, not
        // trusted, in the same way the batch scratch is.
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
            reuse_profiler_.reset(registry_.total_experts);
        }
        executor_ = std::make_unique<V4TieredExpertExecutor>(
            supply_, vram_pool_, *staging_, registry_, expert_scratch_, streams_,
            telemetry_, runtime_cfg.profile_routing_reuse ? &reuse_profiler_ : nullptr,
            config_.swiglu_limit);

        // 15 — the prefill sweep (Step 6 item 6). Borrows the same two components the
        // executor does; it runs only between `prefill_begin` and `prefill_end`.
        prefill_sweep_requested_ = runtime_cfg.prefill_sweep;
        // The prompt-length gate (Step 3), resolved once from the layer width, and
        // **placed at the measured crossover** rather than derived from theory: the
        // A/B (`scripts/prefill_ab.sh`, ledger M44) puts the routed bank ahead of the
        // sweep up to about `0.7 E` tokens and the sweep ahead from `0.75 E`, so the
        // default is `3 E / 4`. It is a visible, overridable setting, and the round
        // fraction keeps it expressed in the model's own terms.
        sweep_min_tokens_ = runtime_cfg.prefill_sweep_min_tokens > 0
            ? runtime_cfg.prefill_sweep_min_tokens
            : std::max<uint32_t>(
                  1, (static_cast<uint32_t>(config_.n_routed_experts) * 3u) / 4u);
        // The sweep's bank count must match the arena's (`sweep_staging_banks_`, set
        // where the arena is built), or a layer's reads would collide with the bank
        // still in flight.
        prefill_sweep_.configure(&supply_, &registry_, sweep_staging_banks_);
        // P2.3: let the per-token hook pump the swept lookahead's copies. The executor
        // does not know about the sweep, so it gets a callback; the sweep ignores the
        // call when it is not driving a window.
        executor_->set_supply_pump([this] { (void)prefill_sweep_.pump(); });

        // 16 — the prefill workspace (Step 6 item 7), derived from the configured
        // window and chunk and allocated **once, here**, so no window can fail
        // mid-prompt on an allocation and the budget's carry term is realised rather
        // than merely reserved.
        allocate_prefill_workspace(
            runtime_cfg.prefill_window == 0
                ? runtime_cfg.context_size
                : std::min(runtime_cfg.prefill_window, runtime_cfg.context_size),
            runtime_cfg.prefill_chunk > 0 ? runtime_cfg.prefill_chunk : 1);
    }

    size_t direct_requests_per_expert(const ExpertFormatDescriptor& format) const noexcept {
        return (format.payload_bytes + aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES - 1) /
               aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
    }

    // Every Hot resident, read straight from the artifact and uploaded once. The
    // physical slots come from the registry's own `vram_slots` rather than being
    // re-derived from the round-robin order.
    void preload_hot_experts(const ExpertFormatDescriptor& format) {
        const size_t batch = std::max<size_t>(
            1, io_reader_->submission_capacity() / direct_requests_per_expert(format));
        for (uint32_t start = 0; start < budget_.hot_vram_slots;
             start += static_cast<uint32_t>(batch)) {
            const uint32_t end = std::min<uint32_t>(
                budget_.hot_vram_slots, start + static_cast<uint32_t>(batch));

            std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
            std::vector<aeon::io::AlignedBuffer> buffers;
            for (uint32_t slot = start; slot < end; ++slot) {
                const int32_t gid = registry_.vram_slots[slot];
                if (gid < 0) continue;
                const auto& entry = registry_.catalog[static_cast<size_t>(gid)];
                expert_ids.emplace_back(entry.layer_id, entry.expert_id);
                buffers.emplace_back(format.payload_bytes, format.sector_size);
            }

            std::vector<uint8_t*> destinations;
            destinations.reserve(buffers.size());
            for (auto& buffer : buffers) {
                destinations.push_back(static_cast<uint8_t*>(buffer.data()));
            }
            read_experts_direct_blocking(expert_ids, destinations);

            size_t index = 0;
            for (uint32_t slot = start; slot < end; ++slot) {
                if (registry_.vram_slots[slot] < 0) continue;
                vram_pool_.upload_from_host_expert(
                    slot, destinations[index++], streams_.compute);
            }
            CHECK_HIP(hipStreamSynchronize(streams_.compute));
        }
    }

    // The Warm pool is filled in place — the registry chose the owning experts at
    // construction, so this writes their payloads into the pinned slots it
    // reserved rather than deciding residency again.
    void preload_warm_experts(const ExpertFormatDescriptor& format) {
        const size_t batch = std::max<size_t>(
            1, io_reader_->submission_capacity() / direct_requests_per_expert(format));
        for (uint32_t start = 0; start < budget_.warm_host_slots;
             start += static_cast<uint32_t>(batch)) {
            const uint32_t end = std::min<uint32_t>(
                budget_.warm_host_slots, start + static_cast<uint32_t>(batch));

            std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
            std::vector<uint8_t*> destinations;
            for (uint32_t slot = start; slot < end; ++slot) {
                const int32_t gid = registry_.host_slots[slot];
                if (gid < 0) continue;
                const auto& entry = registry_.catalog[static_cast<size_t>(gid)];
                expert_ids.emplace_back(entry.layer_id, entry.expert_id);
                destinations.push_back(host_pool_.get_expert_slot_ptr(slot));
            }
            read_experts_direct_blocking(expert_ids, destinations);
        }
    }

    // Reloads the Hot residents a prefill drained, so the pool returns to the set it
    // held before the pass (the plan's restore requirement). The gids come from the
    // registry's `restore_set()`; each is admitted through the normal cold path —
    // reserve a slot, read the payload, publish it — which is the same machinery
    // decode uses, so nothing here is a second code path. It runs at a boundary (the
    // prefill has already ended), so its blocking reads are off the hot path, and it
    // is batched exactly like `preload_hot_experts`.
    void restore_prefill_residents(const ExpertFormatDescriptor& format) {
        const std::vector<uint32_t> restore = registry_.restore_set();
        if (restore.empty()) return;
        const uint32_t per_layer = registry_.experts_per_layer;
        const size_t batch = std::max<size_t>(
            1, io_reader_->submission_capacity() / direct_requests_per_expert(format));
        for (size_t start = 0; start < restore.size(); start += batch) {
            const size_t end = std::min(restore.size(), start + batch);
            std::vector<uint32_t> operation_ids;
            std::vector<int32_t> slots;
            std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
            std::vector<aeon::io::AlignedBuffer> buffers;
            operation_ids.reserve(end - start);
            for (size_t i = start; i < end; ++i) {
                const uint32_t gid = restore[i];
                const auto request = registry_.reserve_request(
                    gid, 0, demotion_queue_capacity_);
                if (request.kind != ExpertRequestKind::COLD_MISS || request.vram_slot < 0) {
                    throw std::runtime_error(
                        "V4ModelHost: a drained prefill resident was not cold at restore");
                }
                operation_ids.push_back(request.operation_id);
                slots.push_back(request.vram_slot);
                expert_ids.emplace_back(gid / per_layer, gid % per_layer);
                buffers.emplace_back(format.payload_bytes, format.sector_size);
            }
            std::vector<uint8_t*> destinations;
            destinations.reserve(buffers.size());
            for (auto& buffer : buffers) {
                destinations.push_back(static_cast<uint8_t*>(buffer.data()));
            }
            read_experts_direct_blocking(expert_ids, destinations);
            for (size_t i = 0; i < slots.size(); ++i) {
                vram_pool_.upload_from_host_expert(
                    static_cast<uint32_t>(slots[i]), destinations[i], streams_.compute);
            }
            CHECK_HIP(hipStreamSynchronize(streams_.compute));
            for (size_t i = 0; i < operation_ids.size(); ++i) {
                registry_.complete_request(operation_ids[i]);
                registry_.release_lease(restore[start + i]);
            }
        }
    }

    // A batched, blocking `O_DIRECT` read of whole expert payloads. The batch is
    // sized to the reader's own submission capacity, so the chunk count per
    // request is the artifact's (a payload is 3.375 of the 4 MiB chunks) and the
    // short final request is expected rather than an error.
    void read_experts_direct_blocking(
        const std::vector<std::pair<uint32_t, uint32_t>>& expert_ids,
        const std::vector<uint8_t*>& destinations) {
        if (expert_ids.size() != destinations.size()) {
            throw std::invalid_argument("V4ModelHost: a direct expert read batch is mismatched");
        }
        if (expert_ids.empty()) return;

        const size_t requests_per_expert = direct_requests_per_expert(loader_.expert_format());
        const size_t max_batch = std::max<size_t>(
            1, io_reader_->submission_capacity() / requests_per_expert);

        struct ReadJob {
            uint64_t first_user_data{0};
            size_t request_count{0};
        };

        for (size_t start = 0; start < expert_ids.size(); start += max_batch) {
            const size_t end = std::min(expert_ids.size(), start + max_batch);
            std::vector<ReadJob> jobs;
            size_t total_requests = 0;

            for (size_t i = start; i < end; ++i) {
                const auto location = loader_.get_expert_location(
                    expert_ids[i].first, expert_ids[i].second);
                const uint64_t first_user_data = next_io_id_;
                const size_t request_count = io_reader_->submit_read_chunks(
                    loader_.expert_direct_fd(), destinations[i], location.byte_length,
                    location.file_offset, first_user_data);
                next_io_id_ += request_count;
                total_requests += request_count;
                jobs.push_back(ReadJob{first_user_data, request_count});
            }

            if (io_reader_->submit_pending_reads() != total_requests) {
                throw std::runtime_error(
                    "V4ModelHost: a direct expert batch submitted an unexpected request count");
            }

            std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> completions;
            completions.reserve(total_requests);
            for (size_t i = 0; i < total_requests; ++i) {
                const auto completion = io_reader_->wait_for_completion();
                completions.emplace(completion.user_data, completion);
            }

            for (const auto& job : jobs) {
                for (size_t chunk = 0; chunk < job.request_count; ++chunk) {
                    const auto it = completions.find(job.first_user_data + chunk);
                    if (it == completions.end() || it->second.result < 0) {
                        throw std::runtime_error(
                            "V4ModelHost: a direct expert read failed");
                    }
                    const size_t chunk_offset =
                        chunk * aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
                    const size_t expected = std::min(
                        aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES,
                        loader_.expert_format().payload_bytes - chunk_offset);
                    if (it->second.result != static_cast<int32_t>(expected)) {
                        throw std::runtime_error(
                            "V4ModelHost: a direct expert read returned a short payload");
                    }
                }
            }
        }
    }

    bool verbose_{false};
    AeonModelLoader loader_;
    DeepSeekV4Config config_;
    std::vector<V4LayerSpec> layer_specs_;
    MemoryBudgetReport budget_;
    uint32_t context_capacity_{0};
    size_t last_released_dense_bytes_{0};
    uint64_t demotion_queue_capacity_{0};
    bool freeze_warm_during_prefill_{false};
    bool supply_phase_prefill_{false};

    V4DeviceStreams streams_;
    V4ModelResources resources_;
    PipelineScratchBuffers scratch_;
    std::vector<V4Layer> layers_;

    // The layer-major prefill working set (Step 6). The workspace is re-allocated
    // when the layer changes; the carry grows and is kept.
    V4LayerBodyBatchScratch batch_scratch_;
    uint32_t batch_scratch_layer_{UINT32_MAX};
    uint32_t batch_scratch_count_{0};
    half* d_prefill_carry_half_{nullptr};
    float* d_prefill_carry_{nullptr};
    uint32_t prefill_carry_tokens_{0};
    // Step 6 item 7: the configured knobs and whether the workspace was allocated at
    // load for them.
    uint32_t prefill_window_tokens_{0};
    uint32_t prefill_chunk_tokens_{0};
    bool prefill_workspace_ready_{false};

    UnifiedVRAMExpertPool vram_pool_;
    HostExpertPool host_pool_;
    ExpertRegistry registry_;
    std::unique_ptr<PrefetchStagingArena> staging_;
    // The pinned region the Warm pool and the corridor share, when a host budget is
    // configured. Empty otherwise, and the corridor allocates its own memory.
    ExpertHostRegion host_region_;
    bool host_region_active_{false};
    // The two partitions the phases move between: the corridor's base
    // (`max(12, min(6C, E))`) with Warm at its maximum, and the swept prefill's
    // `blocks x E` with Warm at its minimum. Both sum to the region exactly.
    uint32_t staging_base_slots_{0};
    uint32_t staging_prefill_slots_{0};
    uint32_t warm_slots_base_{0};
    uint32_t warm_slots_prefill_{0};
    uint32_t current_warm_slots_{0};
    SupplyTelemetry telemetry_;
    RoutingReuseProfiler reuse_profiler_;
    std::unique_ptr<aeon::io::DirectIOReader> io_reader_;
    std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> completions_;
    uint64_t next_io_id_{1};
    V4ExpertSupplyCoordinator supply_;
    V4RoutedExpertScratch expert_scratch_;
    std::unique_ptr<V4TieredExpertExecutor> executor_;
    V4PrefillSweep prefill_sweep_;
    bool prefill_sweep_requested_{false};
    // Layer-sized staging banks the sweep's arena holds, derived from the arena's
    // depth (`slots / experts_per_layer`). The default is `2` — two layer-blocks, one
    // the read destination and one the copy source — and it is deliberately **not** a
    // config knob, because a settable depth lets a resource select the algorithm (R6).
    uint32_t sweep_staging_banks_{2};
    // Whether the layer-major prefill supply is active at all this window (either
    // strategy). The length picks the strategy; this says one was chosen.
    bool prefill_active_{false};
    // The prompt-length gate, resolved at load from `prefill_sweep_min_tokens`
    // (`E / 4` unless configured). Below it a window runs the route-aware cached
    // supply. See `prefill_sweep_min_tokens()`.
    uint32_t sweep_min_tokens_{0};
    // Set by `prefill_begin` for the window it opens, so the per-layer hooks and
    // `prefill_end` act on the strategy that was chosen for *this* window rather
    // than re-deciding it (and so a gate can read the choice).
    bool sweep_active_{false};
};

} // namespace aeon::core
