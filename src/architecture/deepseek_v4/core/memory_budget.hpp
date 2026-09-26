#pragma once

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer_state.hpp"
#include "infrastructure/core/expert_format.hpp"
#include "infrastructure/core/prefetch_staging.hpp"

#include <hip/hip_runtime.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace aeon::core {

// Constant safety margins & architectural parameters
constexpr size_t VRAM_HEADROOM_SAFETY_BYTES = 300ULL * 1024ULL * 1024ULL; // 300 MB
// Host RAM the engine will never plan to use, held back for the OS and for this
// process's own non-expert footprint (the graph, the tokenizer, the pinned
// transport staging arena, and whatever the dense container still holds in page
// cache).
//
// It replaces a percentage cap (`0.70 * total`, i.e. 43.84 GiB here), which had
// two defects: it never accounted for memory *already in use*, so a 43.84 GiB
// Warm allocation on a machine with 30 GiB resident would swap; and it moved with
// the machine's RAM size rather than with what the process needs. A fixed reserve
// is the honest statement of "the engine may have everything else".
//
// A previous revision of this comment justified the reserve with "the ~13 GiB of
// resident dense weights that the embedding lookup and every oracle read touch
// through the mmap". That was wrong twice over. Only `embed.weight` (0.99 GiB) is
// touched per token — every other dense tensor is uploaded once and never read
// again — and those pages are now released outright after the uploads
// (`AeonModelLoader::release_dense_pages_except`, driven by
// `release_dense_pages_after_upload`). The reserve is not sized around dense
// residency, and it never was.
constexpr size_t HOST_RAM_RESERVED_BYTES = 10ULL * 1024ULL * 1024ULL * 1024ULL; // 10 GiB

// VRAM allowance for the body's batch scratch (`V4LayerBodyBatchScratch`), derived
// from the configured prefill chunk `C`.
//
// It is an **allowance**, not a computed size: the exact figure depends on each
// layer's state layout (the indexer's candidate scores and top-k, the composed
// row-set), so the host allocates the real buffer at load and checks it against this
// number, failing with a named message rather than silently eating into the Hot
// pool. The per-row figure is `1 MiB`, against a measured `~0.69 MiB/row` (a `C = 64`
// scratch is `44 MiB`), which leaves the allowance ~45% above the real cost and is
// what makes it safe to state without walking the layer layouts here.
constexpr size_t BATCH_SCRATCH_BYTES_PER_ROW = 1ULL * 1024ULL * 1024ULL;

inline size_t batch_scratch_allowance_bytes(uint32_t chunk_tokens) {
    return static_cast<size_t>(chunk_tokens) * BATCH_SCRATCH_BYTES_PER_ROW;
}

// VRAM allowance for the **decode** path's fixed workspace: `V4PipelineScratchBuffers`
// (the per-op temporaries, sized by the kernel shapes) plus `V4RoutedExpertScratch`
// (six experts' accumulators). Neither is a function of a knob — both are fixed by the
// model — so this is an allowance in the same sense as the batch scratch: the host
// allocates the real buffers at load and checks them against it, so the figure is
// verified rather than trusted. Measured, the two together are ≈`5 MiB`; the margin
// covers a kernel whose padded row count grows.
constexpr size_t DECODE_SCRATCH_ALLOWANCE_BYTES = 16ULL * 1024ULL * 1024ULL;

inline size_t decode_scratch_allowance_bytes() {
    return DECODE_SCRATCH_ALLOWANCE_BYTES;
}


struct AeonRuntimeConfig {
    // User-configurable: target context sequence length (tokens)
    uint32_t context_size{4096};

    // Maximum host budget for persistent Warm payloads plus runtime transport.
    // Zero disables persistent Warm ownership and D2H refill.
    size_t warm_host_bytes{0};

    // Allocate the configured Warm capacity without requiring a synchronous
    // startup fill. This remains enabled by default for compatibility.
    bool preload_warm_host{true};

    // After the dense uploads finish, release this process's residency of the
    // container's pages, keeping only `embed.weight`. The pages are clean and
    // file-backed, so the kernel may reclaim them anyway; releasing them
    // deterministically keeps the host footprint from depending on when the
    // reclaim happens to run, which matters because the Warm pool is pinned and
    // cannot be reclaimed at all. Zero VRAM cost. Set false to keep every dense
    // page resident, which is what a caller reading the container host-side after
    // initialization wants.
    bool release_dense_pages_after_upload{true};

    // Diagnostic A/B control. The production default keeps asynchronous refill enabled.
    bool enable_warm_refill{true};

    // Correctness-mode control for deterministic routed-expert accumulation.
    // The production default retains the fused atomic accumulation path.
    bool deterministic_expert_accumulation{false};

    // Supply telemetry sink. Empty path disables recording entirely (the default);
    // a non-empty path opens a JSONL stream that `TieredExpertSupply` writes its
    // request, timing, occupancy, and demotion records into. `run_id` is stamped on
    // every row so several runs can share or be told apart in one directory.
    //
    // Only supply-mediated traffic is captured: the Hot and Warm startup preloads
    // read through `read_experts_direct_blocking`, not through the supply, so the
    // Warmup phase stays empty by construction.
    std::string supply_telemetry_path;
    std::string run_id{"unnamed"};

    // Diagnostic pressure knob. Zero (the default) leaves the Hot pool at the
    // derived size. A non-zero value caps it at `min(derived, value)`, floored at
    // 6 (one layer's routed experts) so the graph stays runnable. It exists so a
    // gate can force the starved-pool path — the emergency drain in
    // `ensure_pool_headroom` and eviction-under-lease-pressure — which the derived
    // size never reaches on this GPU (779 slots against an ≈264-slot arm point).
    uint32_t max_hot_vram_slots{0};

    // Demotion-queue capacity override. Zero (the default) derives it from
    // `enable_warm_refill` — the default capacity when refill is on, 0 when off.
    // A positive value sets it directly, which is how the Step 5 A/B varies the
    // number of evictions allowed in flight before the rest are dropped with
    // `queue_pressure`.
    uint64_t demotion_queue_capacity{0};

    // Phase 1 of the routing study: profile decode routing reuse distances. Off by
    // default; when on, the host enables the profiler and `aeon_chat` prints an
    // ideal-LRU hit-rate curve next to the measured Hot hit rate.
    bool profile_routing_reuse{false};

    // Run the registry's full invariant audit after **every** expert reservation and
    // completion, not only at the phase boundaries. Off by default: the audit is a
    // whole-registry scan, so per-request it dominates a batch (measured 10.2 s of a
    // 338-token swept prefill, ledger M43). On, it localises a bookkeeping defect to
    // the individual operation that caused it. The boundaries and `invariants_hold()`
    // always audit regardless.
    bool validate_registry_each_request{false};

    // The body chunk `C` the layer-major prefill window runs with — the rows in
    // flight per body invocation (Step 6 §6b). User-configurable; validated against
    // the body's own row cap (`V4LayerBodyBatchScratch::kMaxTokens`) and against the
    // window. It bounds the batch scratch (allocated at load) and the staging arena
    // (`min(6C, experts_per_layer)`). Larger = fewer body invocations; measured
    // nearly flat in throughput, so the default is on the wide side.
    uint32_t prefill_chunk{64};

    // The layer-major prefill **window** `W` (Step 6 §6b): how many prompt tokens
    // one layer-major pass carries in its residual, and therefore how many sweeps a
    // prompt of `N` tokens pays (`⌈N/W⌉`). User-configurable. `0` means the whole
    // prompt, bounded by the context — one pass, but a carry allocated for the whole
    // context. The default matches the reference implementation's segment size
    // (colibri `V4_PREFILL_SEGMENT = 4096`), which bounds the carry at ~393 MiB
    // regardless of context and keeps a long prompt at `⌈N/4096⌉` passes.
    uint32_t prefill_window{4096};

    // Step 6 D-b (policy A): freeze the Warm tier during prefill. A prefill touches
    // every expert, so letting the sweep promote from Warm would **move** each
    // Warm-resident expert into VRAM and empty the tier — destroying exactly the
    // "natural selection" decode's Warm hits depend on. With this on, a prefill
    // copies a Warm expert into VRAM **without** transferring ownership (a shadow
    // residency) and evicts by **release** rather than demotion, so Warm's resident
    // set is identical before and after the prefill. On by default, per D-b; a gate
    // A/Bs it against `false`.
    bool freeze_warm_during_prefill{true};

    // Step 6 item 6: drive the layer-major prefill window with the **expert sweep**
    // instead of the per-token dispatch. The sweep frees only what it needs on
    // entry, holds a sliding window of whole layer sets (`L, L+1, L+2, …` up to
    // capacity), releases each layer's prefill-admitted set as it retires, and
    // leaves the residents preserved at entry resident on exit — so Warm and its LRU
    // ranking are untouched across the whole prefill and decode resumes on them. On
    // by default; it is the prefill strategy the plan decided (D-a). Falls back to
    // the per-token path when the Hot pool cannot hold a whole layer.
    bool prefill_sweep{true};

    // The **prompt-length gate** for the sweep (Prefill Supply Strategy plan, Step
    // 3). Below this window length the layer-major window runs the route-aware
    // cached supply instead of the sweep: a whole-layer load over-reads a short
    // prompt's small distinct set, so the sweep is the right strategy only for long
    // ones. Zero (the default) derives the gate from the layer width (`3 E / 4`),
    // placed at the measured crossover, so it is expressed in the model's own terms
    // rather than as a constant tuned for one GPU. This is the **only** condition on
    // the switch besides feasibility — a hidden threshold is exactly what the plan
    // forbids.
    uint32_t prefill_sweep_min_tokens{0};

    // The layer-sized staging **banks** the sweep's arena holds when `prefill_sweep`
    // is on, in units of one layer's payloads (`E`). **A memory budget, not a tuning
    // figure**: the prefetch depth is *derived* from the arena (`blanks - 1` blocks of
    // free staging, bounded to one read wave at a time), so the count decides how much
    // pinned host memory the corridor spends and nothing else. Measured across `2E…
    // 6E`: `29.83–30.39 s`, Gate A spread `1.009x`, `42/43` layer-bodies overlapped at
    // every size (SUPPLY_CHAIN_HOT_PATH_ANALYSIS §17.4).
    //
    // This is why the knob is legitimate where an earlier revision removed it: the
    // objection then was that depth selected the *algorithm* (Gate A failed at
    // `1.38x`, `1E` ran a serial read/copy pipeline). It does not any more, so the
    // arena can be sized from the host's remaining RAM, which is exactly what a
    // portability knob should express.
    //
    // `1` is the minimum and is supported (a single bank, the blocking drain); below
    // one layer the swept dispatch cannot bind a whole layer's set, so smaller values
    // are refused at load. Each block is `E x payload_bytes` of **non-reclaimable
    // pinned** host memory — `3.44 GiB` per block at `E = 256`.
    uint32_t prefill_sweep_staging_blocks{2};

    // Hardware target device index
    int device_id{0};

};

// The staging arena's slot count, shared by the budget report and the arena's own
// construction so the reported `transient_staging_bytes` is exactly what is
// allocated. Two terms:
//
//   * decode/a-chunk's deduplicated distinct set, at most the layer width (`6C`
//     capped by `E`), with `TOTAL_STAGING_SLOTS` as the floor; and
//   * when the sweep is on, `prefill_sweep_staging_blocks` layer-blocks — at least
//     one to bind a whole layer's set, and by default two so one bank is the read
//     destination and one the copy source (see `AeonRuntimeConfig::prefill_sweep`).
inline uint32_t staging_slot_count(const AeonRuntimeConfig& cfg, uint32_t experts_per_layer) {
    const uint32_t chunk_ceiling = std::min<uint32_t>(
        6u * std::max<uint32_t>(1, cfg.prefill_chunk), experts_per_layer);
    const uint32_t base = std::max<uint32_t>(
        PrefetchStagingArena::TOTAL_STAGING_SLOTS, chunk_ceiling);
    if (!cfg.prefill_sweep) return base;
    const uint32_t blocks = std::max<uint32_t>(1, cfg.prefill_sweep_staging_blocks);
    return std::max<uint32_t>(base, blocks * experts_per_layer);
}

struct MemoryBudgetReport {
    bool is_feasible{false};
    std::string rejection_reason;
    // Hardware limits
    size_t total_vram_bytes{0};
    size_t free_vram_bytes{0};

    // What the plan is actually sized against: `min(free_vram, total_vram)`.
    // Planning against `total_vram` ignores VRAM another process already holds,
    // which on a card running a display server (or shared with another job) is
    // memory the engine cannot have.
    size_t usable_vram_bytes{0};
    size_t total_host_ram_bytes{0};
    size_t max_allowed_host_ram_bytes{0};

    // VRAM allocations
    size_t vram_dense_bytes{0};
    size_t vram_kv_bytes{0};
    size_t vram_local_kv_bytes{0};
    size_t vram_compressed_kv_bytes{0};
    size_t vram_compressor_state_bytes{0};
    size_t vram_indexer_state_bytes{0};
    size_t vram_attention_metadata_bytes{0};
    size_t vram_attention_state_bytes{0};
    size_t vram_rope_bytes{0};
    size_t vram_scratch_bytes{0};
    // The three terms of `vram_scratch_bytes`, reported separately because each has
    // its own owner: the decode workspace is fixed by the model's kernel shapes, the
    // batch scratch is a function of the chunk `C`, and the residual carry of the
    // window `W`.
    size_t vram_decode_scratch_bytes{0};
    size_t vram_batch_scratch_bytes{0};
    // The residual carry alone, derived from the configured prefill window. Reported
    // separately from the batch-scratch allowance it is summed with, so the window's
    // cost is visible rather than folded into a single scratch figure.
    size_t vram_prefill_carry_bytes{0};
    size_t vram_headroom_bytes{VRAM_HEADROOM_SAFETY_BYTES};
    size_t vram_min_active_bytes{0};
    size_t vram_available_for_experts{0};

    // Expert slot allocations
    uint32_t hot_vram_slots{0};
    size_t hot_vram_bytes{0};
    uint32_t warm_host_slots{0};
    size_t warm_host_bytes{0};
    size_t expert_payload_bytes{0};
    size_t configured_host_budget_bytes{0};
    size_t persistent_warm_host_budget_bytes{0};
    size_t transient_staging_bytes{0};
    uint32_t cold_nvme_slots{0};

    // Diagnostics / recommendations
    uint32_t max_viable_context_size{0};

    std::string to_string() const {
        std::ostringstream oss;
        oss << "\n================================================================================\n"
            << "                      Project Aeon: Memory Budget Report                        \n"
            << "================================================================================\n"
            << std::fixed << std::setprecision(2)
            << "  Feasibility Status       : " << (is_feasible ? "[FEASIBLE / APPROVED]" : "[REJECTED]") << "\n";

        if (!is_feasible) {
            oss << "  Rejection Reason         : " << rejection_reason << "\n"
                << "  Max Viable Context Size  : " << max_viable_context_size << " tokens\n";
        }

        oss << "--------------------------------------------------------------------------------\n"
            << "  Hardware Environment:\n"
            << "    - Total VRAM           : " << (double)total_vram_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Free VRAM (at init)  : " << (double)free_vram_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Usable VRAM (planned): " << (double)usable_vram_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Total Host RAM       : " << (double)total_host_ram_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Max Allowed Host RAM : " << (double)max_allowed_host_ram_bytes / (1024 * 1024 * 1024) << " GB (total - 10 GiB reserved)\n"
            << "--------------------------------------------------------------------------------\n"
            << "  VRAM Allocation Breakdown (dense is what is uploaded, not the container):\n"
            << "    - Dense (uploaded)     : " << (double)vram_dense_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Local K/V state      : " << (double)vram_local_kv_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Compressed K/V state : " << (double)vram_compressed_kv_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Compressor state     : " << (double)vram_compressor_state_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Indexer state        : " << (double)vram_indexer_state_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Attention metadata   : " << (double)vram_attention_metadata_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - RoPE tables          : " << (double)vram_rope_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Attention state total: " << (double)vram_attention_state_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Compute Scratch      : " << (double)vram_scratch_bytes / (1024 * 1024) << " MB"
            << "  (decode allowance " << (double)vram_decode_scratch_bytes / (1024 * 1024)
            << " MB + batch allowance " << (double)vram_batch_scratch_bytes / (1024 * 1024)
            << " MB + residual carry " << (double)vram_prefill_carry_bytes / (1024 * 1024)
            << " MB; the two allowances are checked against the real allocations at load)\n"
            << "    - Safety Headroom      : " << (double)vram_headroom_bytes / (1024 * 1024) << " MB (Fixed OS/GTT buffer)\n"
            << "    - Active Experts Min   : " << (double)vram_min_active_bytes / (1024 * 1024) << " MB\n"
            << "    - Available for Hot Pool: " << (double)vram_available_for_experts / (1024 * 1024 * 1024) << " GB\n"
            << "--------------------------------------------------------------------------------\n"
            << "  3-Tier Expert Hierarchy Distribution (Total: "
            << (hot_vram_slots + warm_host_slots + cold_nvme_slots) << " experts):\n"
            << "    - Tier 1: Hot VRAM     : " << hot_vram_slots << " slots ("
            << (double)hot_vram_bytes / (1024 * 1024 * 1024) << " GB)\n"
            << "    - Tier 2: Warm Host DDR: " << warm_host_slots << " slots ("
            << (double)warm_host_bytes / (1024 * 1024 * 1024) << " GB)\n"
            << "    - Expert Payload Size : " << expert_payload_bytes << " bytes\n"
            << "    - Configured Host Budget : " << (double)configured_host_budget_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Persistent Warm Budget : " << (double)persistent_warm_host_budget_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Transient Staging    : " << (double)transient_staging_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Tier 3: Cold NVMe SSD: " << cold_nvme_slots << " slots\n"
            << "================================================================================\n";
        return oss.str();
    }
};

// The residual carry's bytes for the configured prefill window: the fp16 broadcast
// and the fp32 copy, per token, over the `hc_mult * hidden_size` stream width.
// Derived rather than assumed, because it scales with the configured window and at
// a whole-context window it is gigabytes — not a rounding error on the expert pool.
inline size_t prefill_carry_bytes(const AeonRuntimeConfig& runtime_cfg,
                                  const DeepSeekV4Config& model_cfg) {
    const uint32_t ctx = runtime_cfg.context_size;
    const uint32_t window = runtime_cfg.prefill_window == 0
        ? ctx
        : std::min(runtime_cfg.prefill_window, ctx);
    const size_t hc_dim = static_cast<size_t>(model_cfg.hc_mult) *
                          static_cast<size_t>(model_cfg.hidden_size);
    return static_cast<size_t>(window) * hc_dim * (sizeof(uint16_t) + sizeof(float));
}

class MemoryBudgetEngine {
public:
    struct AttentionStateMemory {
        size_t local_kv_bytes{0};
        size_t compressed_kv_bytes{0};
        size_t compressor_state_bytes{0};
        size_t indexer_state_bytes{0};
        size_t metadata_bytes{0};
        size_t layer_state_bytes{0};
        size_t rope_bytes{0};

        size_t total_bytes() const {
            return layer_state_bytes + rope_bytes;
        }
    };

    static AttentionStateMemory attention_state_memory(
        const DeepSeekV4Config& model_cfg,
        uint32_t context_size
    ) {
        if (context_size == 0) {
            throw std::invalid_argument("MemoryBudgetEngine: attention context cannot be 0");
        }
        const auto layer_specs = V4ModelSpec::resolve_layers(model_cfg);
        AttentionStateMemory memory;
        for (const auto& layer_spec : layer_specs) {
            const auto layout = V4LayerStateLayout::from_spec(layer_spec, context_size);
            memory.local_kv_bytes += layout.local_cache_bytes();
            memory.compressed_kv_bytes += layout.compressed_cache_bytes();
            memory.compressor_state_bytes += layout.compressor_state_bytes();
            memory.indexer_state_bytes += layout.indexer_cache_bytes() + layout.indexer_workspace_bytes();
            memory.metadata_bytes += layout.local_metadata_bytes() + layout.compressed_metadata_bytes();
            memory.layer_state_bytes += layout.total_device_bytes();
        }

        const size_t rope_half = static_cast<size_t>(model_cfg.qk_rope_head_dim / 2);
        memory.rope_bytes = static_cast<size_t>(context_size) * rope_half * sizeof(float) * 4;
        return memory;
    }

    static MemoryBudgetReport evaluate(
        const AeonRuntimeConfig& runtime_cfg,
        const DeepSeekV4Config& model_cfg,
        size_t dense_weights_bytes,
        const ExpertFormatDescriptor& expert_format
    ) {
        MemoryBudgetReport report;

        try {
            expert_format.validate_catalog();
        } catch (const std::exception& error) {
            report.rejection_reason = error.what();
            return report;
        }

        if (model_cfg.num_hidden_layers < 0 || model_cfg.n_routed_experts < 0 ||
            expert_format.num_layers != static_cast<uint32_t>(model_cfg.num_hidden_layers) ||
            expert_format.experts_per_layer != static_cast<uint32_t>(model_cfg.n_routed_experts)) {
            report.rejection_reason =
                "Expert format catalog does not match the model configuration";
            return report;
        }
        if (expert_format.total_experts() > std::numeric_limits<uint32_t>::max()) {
            report.rejection_reason = "Expert format catalog exceeds the runtime ID range";
            return report;
        }
        report.expert_payload_bytes = expert_format.payload_bytes;

        // 1. Query physical GPU memory
        size_t free_vram = 0, total_vram = 0;
        hipError_t err = hipMemGetInfo(&free_vram, &total_vram);
        if (err != hipSuccess) {
            report.is_feasible = false;
            report.rejection_reason = std::string("hipMemGetInfo failed: ") + hipGetErrorString(err);
            return report;
        }

        report.total_vram_bytes = total_vram;
        report.free_vram_bytes  = free_vram;
        // Plan against what this process can actually allocate. `total_vram` is the
        // card's nominal capacity; `free_vram` is what is left for us, and the
        // smaller of the two is the only defensible planning basis.
        report.usable_vram_bytes = std::min(free_vram, total_vram);

        // 2. Query physical Host memory
        struct sysinfo si;
        if (sysinfo(&si) != 0) {
            report.is_feasible = false;
            report.rejection_reason = "Failed to query system host memory via sysinfo()";
            return report;
        }

        report.total_host_ram_bytes       = static_cast<size_t>(si.totalram) * si.mem_unit;
        report.max_allowed_host_ram_bytes = report.total_host_ram_bytes > HOST_RAM_RESERVED_BYTES
            ? report.total_host_ram_bytes - HOST_RAM_RESERVED_BYTES
            : 0;

        // 3. Validate Context Length against model architecture
        if (runtime_cfg.context_size == 0) {
            report.is_feasible = false;
            report.rejection_reason = "Context size cannot be 0";
            return report;
        }

        if (runtime_cfg.context_size > static_cast<uint32_t>(model_cfg.max_position_embeddings)) {
            report.is_feasible = false;
            report.rejection_reason = "Requested context size (" + std::to_string(runtime_cfg.context_size) +
                ") exceeds model max_position_embeddings (" +
                std::to_string(model_cfg.max_position_embeddings) + ")";
            return report;
        }

        AttentionStateMemory attention_memory;
        try {
            attention_memory = attention_state_memory(model_cfg, runtime_cfg.context_size);
        } catch (const std::exception& error) {
            report.rejection_reason = error.what();
            return report;
        }

        // 4. Calculate exact VRAM requirements from the resolved layer classes.
        report.vram_dense_bytes   = dense_weights_bytes;
        report.vram_local_kv_bytes = attention_memory.local_kv_bytes;
        report.vram_compressed_kv_bytes = attention_memory.compressed_kv_bytes;
        report.vram_compressor_state_bytes = attention_memory.compressor_state_bytes;
        report.vram_indexer_state_bytes = attention_memory.indexer_state_bytes;
        report.vram_attention_metadata_bytes = attention_memory.metadata_bytes;
        report.vram_attention_state_bytes = attention_memory.layer_state_bytes;
        report.vram_rope_bytes = attention_memory.rope_bytes;
        report.vram_kv_bytes = attention_memory.total_bytes();
        // The prefill workspace is **derived from the configured knobs** (Step 6 item
        // 7): the residual carry is a pure function of the window, and the batch
        // scratch is an allowance the host checks its real allocation against at load.
        // The `100 MiB` literal this replaces was wrong in both directions — it
        // over-counted decode scratch several-fold and did not cover the batch scratch
        // at all.
        const size_t carry_bytes = prefill_carry_bytes(runtime_cfg, model_cfg);
        report.vram_prefill_carry_bytes = runtime_cfg.prefill_sweep ? carry_bytes : 0;
        report.vram_batch_scratch_bytes = runtime_cfg.prefill_sweep
            ? batch_scratch_allowance_bytes(std::max<uint32_t>(1, runtime_cfg.prefill_chunk))
            : 0;
        // The decode workspace is **not** a knob: `V4PipelineScratchBuffers` and
        // `V4RoutedExpertScratch` are fixed by the model's kernel shapes, and both
        // report their real allocation size. The old `100 MiB` literal that stood for
        // this was wrong in both directions — it over-counted by ~an order of
        // magnitude *and* ignored the batch scratch entirely.
        report.vram_decode_scratch_bytes = decode_scratch_allowance_bytes();
        report.vram_scratch_bytes = report.vram_decode_scratch_bytes +
                                    report.vram_batch_scratch_bytes +
                                    report.vram_prefill_carry_bytes;
        report.vram_headroom_bytes = VRAM_HEADROOM_SAFETY_BYTES;

        // Minimum active experts needed for execution:
        // 2 * num_experts_per_tok to guarantee compute + prefetch buffering without stalling
        uint32_t min_active_slots = static_cast<uint32_t>(2 * model_cfg.num_experts_per_tok);
        report.vram_min_active_bytes = static_cast<size_t>(min_active_slots) *
                           expert_format.payload_bytes;

        // 5. Total essential baseline VRAM required
        size_t baseline_vram_needed = report.vram_dense_bytes +
                                      report.vram_kv_bytes +
                                      report.vram_scratch_bytes +
                                      report.vram_headroom_bytes +
                                      report.vram_min_active_bytes;

        // Calculate max viable context size for this GPU using the same layout.
        const size_t non_attention_required = report.vram_dense_bytes + report.vram_scratch_bytes +
                                               report.vram_headroom_bytes + report.vram_min_active_bytes;
        if (report.usable_vram_bytes > non_attention_required) {
            size_t low = 0;
            size_t high = static_cast<size_t>(model_cfg.max_position_embeddings);
            while (low < high) {
                const size_t midpoint = low + (high - low + 1) / 2;
                const auto candidate = attention_state_memory(model_cfg, static_cast<uint32_t>(midpoint));
                if (non_attention_required + candidate.total_bytes() <= report.usable_vram_bytes) {
                    low = midpoint;
                } else {
                    high = midpoint - 1;
                }
            }
            report.max_viable_context_size = static_cast<uint32_t>(low);
        }

        // 6. Hard Feasibility Gate Evaluation
        if (baseline_vram_needed > report.usable_vram_bytes) {
            report.is_feasible = false;
            std::ostringstream err_oss;
            err_oss << "VRAM capacity exceeded: Required baseline "
                    << (double)baseline_vram_needed / (1024 * 1024 * 1024) << " GB, but only "
                    << (double)report.usable_vram_bytes / (1024 * 1024 * 1024) << " GB is usable"
                    << " (of " << (double)total_vram / (1024 * 1024 * 1024)
                    << " GB total, " << (double)free_vram / (1024 * 1024 * 1024) << " GB free).";
            report.rejection_reason = err_oss.str();
            return report;
        }

        // 7. Calculate Hot VRAM Expert Pool capacity
        size_t remaining_for_experts = report.usable_vram_bytes - (report.vram_dense_bytes +
                                                     report.vram_kv_bytes +
                                                     report.vram_scratch_bytes +
                                                     report.vram_headroom_bytes);
        report.vram_available_for_experts = remaining_for_experts;
        report.hot_vram_slots = static_cast<uint32_t>(
            remaining_for_experts / expert_format.payload_bytes);

        // Diagnostic cap (see `AeonRuntimeConfig::max_hot_vram_slots`). The cap is
        // itself floored at one layer's routed experts, because a smaller pool
        // cannot hold a single dispatch's working set and the executor would refuse
        // — a knob that makes the graph unrunnable is not a useful pressure knob.
        if (runtime_cfg.max_hot_vram_slots > 0) {
            const uint32_t cap = std::max<uint32_t>(runtime_cfg.max_hot_vram_slots, 6);
            report.hot_vram_slots = std::min(report.hot_vram_slots, cap);
        }
        report.hot_vram_bytes = static_cast<size_t>(report.hot_vram_slots) *
                                expert_format.payload_bytes;

        // 8. Calculate persistent Warm capacity.
        //
        // Warm and the transport staging arena are **two independent budgets**, not
        // one pool with a deduction (Step 0 D5): staging is a transit corridor whose
        // size follows the chunk `C`, and Warm is expert residency. So the requested
        // `warm_host_bytes` is the Warm budget *in full* — it is not reduced by the
        // staging figure. What is checked is the **total**: Warm plus staging plus the
        // fixed reserve must fit the host, or the plan would swap.
        //
        // Staging follows the chunk, and the figure matches what the host actually
        // allocates: `max(12, min(6C, experts_per_layer))` slots, plus the sweep's
        // layer-sized banks. The literal `12` that used to stand here was decode's
        // shape and understated a `C = 256` arena (256 slots, 3456 MiB) by ~21x, and
        // the sweep's over-allocation would understate it again. Both the report and
        // the arena call `staging_slot_count`, so the two cannot drift.
        const uint32_t experts_per_layer =
            static_cast<uint32_t>(expert_format.experts_per_layer);
        const uint32_t staging_slots = staging_slot_count(runtime_cfg, experts_per_layer);
        report.transient_staging_bytes =
            static_cast<size_t>(staging_slots) * expert_format.payload_bytes;
        report.configured_host_budget_bytes = runtime_cfg.warm_host_bytes == 0
            ? report.transient_staging_bytes
            : std::min(runtime_cfg.warm_host_bytes, report.max_allowed_host_ram_bytes);
        const size_t persistent_host_budget = runtime_cfg.warm_host_bytes == 0
            ? 0
            : report.configured_host_budget_bytes;
        report.persistent_warm_host_budget_bytes = persistent_host_budget;

        // The total, which is where the two budgets meet. `max_allowed_host_ram_bytes`
        // is already the reserve-subtracted ceiling.
        const size_t host_total = persistent_host_budget + report.transient_staging_bytes;
        if (host_total > report.max_allowed_host_ram_bytes) {
            report.is_feasible = false;
            report.rejection_reason =
                "Host plan exceeds the RAM ceiling: Warm " +
                std::to_string(persistent_host_budget / (1024ULL * 1024 * 1024)) +
                " GiB + staging " +
                std::to_string(report.transient_staging_bytes / (1024 * 1024)) +
                " MiB > " +
                std::to_string(report.max_allowed_host_ram_bytes / (1024ULL * 1024 * 1024)) +
                " GiB allowed";
            return report;
        }
        if (runtime_cfg.warm_host_bytes > 0 &&
            persistent_host_budget < expert_format.payload_bytes) {
            report.is_feasible = false;
            report.rejection_reason = "Warm host budget cannot hold one complete expert";
            return report;
        }

        const uint32_t total_experts = static_cast<uint32_t>(expert_format.total_experts());
        uint32_t remaining_after_vram = (total_experts > report.hot_vram_slots)
            ? (total_experts - report.hot_vram_slots)
            : 0;

        uint32_t host_slots_budgeted = static_cast<uint32_t>(
            persistent_host_budget / expert_format.payload_bytes);
        report.warm_host_slots = std::min(remaining_after_vram, host_slots_budgeted);
        report.warm_host_bytes = static_cast<size_t>(report.warm_host_slots) *
                                 expert_format.payload_bytes;

        // 9. Cold NVMe pool gets the rest
        report.cold_nvme_slots = total_experts - (report.hot_vram_slots + report.warm_host_slots);

        report.is_feasible = true;
        return report;
    }

    static MemoryBudgetReport evaluate(
        const AeonRuntimeConfig& runtime_cfg,
        const DeepSeekV4Config& model_cfg,
        size_t dense_weights_bytes
    ) {
        return evaluate(
            runtime_cfg,
            model_cfg,
            dense_weights_bytes,
            make_current_swizzled_expert_format(
                model_cfg.num_hidden_layers,
                model_cfg.n_routed_experts));
    }
};

} // namespace aeon::core
