#pragma once

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer_state.hpp"
#include "infrastructure/core/expert_format.hpp"

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

constexpr size_t PIPELINE_SCRATCH_BYTES      = 100ULL * 1024ULL * 1024ULL; // ~100 MB activation scratch

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

    // The body chunk `C` the layer-major prefill window runs with — the rows in
    // flight per body invocation (Step 6 §6b). It is a memory decision: it bounds
    // the batch scratch and, because a chunk dispatches its `6C` routed-expert
    // requests as one deduplicated set (Step 6 D1/D4), it bounds the staging arena,
    // which is sized `6 * prefill_chunk` slots. The default `1` reproduces decode's
    // `C = 1` shape exactly, so nothing on the certified path moves. Step 7 exposes
    // it as a setting and sweeps it.
    uint32_t prefill_chunk{1};

    // The layer-major prefill **window** `W` (Step 6 §6b): how many prompt tokens
    // one layer-major pass carries in its residual, and therefore how many sweeps a
    // prompt of `N` tokens pays (`⌈N/W⌉`). `0` means the whole prompt — one pass
    // over the model, which is §6b's `156 GB` floor and the engine's default. The
    // carry is `W × 64 KB` of VRAM (the bf16 broadcast and the fp32 residual, ~96 KB
    // per token with the allocation as built), so a very long prompt is a VRAM
    // decision and not a free one; Step 7 derives a default bound and sweeps it.
    uint32_t prefill_window{0};

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
    // instead of the per-token dispatch. The sweep drains Hot on entry, holds a
    // sliding window of whole layer sets (`L, L+1, L+2, …` up to capacity), releases
    // each layer's whole set as it retires, and leaves Hot empty on exit — so Warm
    // and its LRU ranking are untouched across the whole prefill and decode resumes
    // on them. On by default; it is the prefill strategy the plan decided (D-a).
    // Falls back to the per-token path when the Hot pool cannot hold a whole layer.
    bool prefill_sweep{true};

    // Hardware target device index
    int device_id{0};

};

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
            << "    - Compute Scratch      : " << (double)vram_scratch_bytes / (1024 * 1024) << " MB\n"
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
        report.vram_scratch_bytes = PIPELINE_SCRATCH_BYTES;
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

        // 8. Calculate persistent Warm capacity. Transport staging is allocated
        // for Cold requests even when Warm is disabled, but it never counts as a
        // persistent Warm slot.
        report.transient_staging_bytes = static_cast<size_t>(12) *
                         expert_format.payload_bytes;
        report.configured_host_budget_bytes = runtime_cfg.warm_host_bytes == 0
            ? report.transient_staging_bytes
            : std::min(runtime_cfg.warm_host_bytes, report.max_allowed_host_ram_bytes);
        const size_t persistent_host_budget = runtime_cfg.warm_host_bytes == 0
            ? 0
            : report.configured_host_budget_bytes > report.transient_staging_bytes
                ? report.configured_host_budget_bytes - report.transient_staging_bytes
                : 0;
        report.persistent_warm_host_budget_bytes = persistent_host_budget;
        if (runtime_cfg.warm_host_bytes > 0 &&
            persistent_host_budget < expert_format.payload_bytes) {
            report.is_feasible = false;
            report.rejection_reason = "Warm host budget cannot hold one complete expert after reserving transient staging";
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
