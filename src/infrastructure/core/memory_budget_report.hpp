#pragma once

// -----------------------------------------------------------------------------
// The memory budget's *output*: the report a caller inspects, plus the derived
// quantity that belongs to it.
//
// `MemoryBudgetReport` is pure data plus a `to_string()` dump — no hardware query
// and no policy. `AttentionStateMemory` (the per-context attention-state breakdown)
// and `ModelMemoryGeometry` the engine consumes live in
// `model_memory_geometry.hpp`; this header holds only the output and the residual
// carry derived from it.
//
// Splitting the report from the engine keeps the data a caller reads independent
// of the code that queries the device, which is what lets a test construct a
// report without a GPU. Splitting it from the model config is what lets it live in
// `infrastructure/`.
// -----------------------------------------------------------------------------

#include "infrastructure/core/model_memory_geometry.hpp"
#include "infrastructure/core/runtime_config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace aeon::core {

struct MemoryBudgetReport {
    bool is_feasible{false};
    std::string rejection_reason;
    // Hardware limits
    size_t total_vram_bytes{0};
    size_t free_vram_bytes{0};

    // What the budget is actually sized against: `min(free_vram, total_vram)`.
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
    // The same figure at the other two partitions: what Warm holds while a routed
    // prefill (one layer's distinct set) or a swept prefill (`blocks` layers) has
    // taken its share of the corridor.
    uint32_t warm_host_slots_routed{0};
    uint32_t warm_host_slots_min{0};
    // The shared pinned region: Warm slots followed by the corridor's slots, cut by a
    // boundary that moves at the phase transitions.
    uint32_t host_region_slots{0};
    size_t host_region_bytes{0};
    size_t expert_payload_bytes{0};
    size_t configured_host_budget_bytes{0};
    size_t persistent_warm_host_budget_bytes{0};
    size_t transient_staging_bytes{0};
    // The corridor's **decode** requirement (`2 x 6`, double-buffered) and its
    // **chunked-prefill** requirement (the layer's deduplicated distinct set). The
    // allocation is the largest requirement (`transient_staging_bytes`); these two are
    // what each phase actually binds, and so what each phase hands back to Warm.
    size_t staging_decode_bytes{0};
    size_t staging_batch_bytes{0};
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
            << (double)warm_host_bytes / (1024 * 1024 * 1024) << " GB)"
            << "  [decode; " << warm_host_slots_routed << " at a routed prefill, "
            << warm_host_slots_min << " at a swept one]\n"
            << "    - Expert Payload Size : " << expert_payload_bytes << " bytes\n"
            << "    - Host Region (Warm + staging, one pinned allocation): "
            << (double)host_region_bytes / (1024 * 1024 * 1024) << " GB in "
            << host_region_slots << " slots\n"
            << "    - Configured Host Budget : " << (double)configured_host_budget_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Persistent Warm Budget : " << (double)persistent_warm_host_budget_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Transient Staging    : " << (double)transient_staging_bytes / (1024 * 1024 * 1024) << " GB"
            << "  (peak; decode " << (double)staging_decode_bytes / (1024 * 1024)
            << " MiB, routed prefill " << (double)staging_batch_bytes / (1024 * 1024)
            << " MiB — the boundary hands the difference to Warm)\n"
            << "    - Tier 3: Cold NVMe SSD: " << cold_nvme_slots << " slots\n"
            << "================================================================================\n";
        return oss.str();
    }
};

// The residual carry's bytes for the configured prefill window: the fp16 broadcast
// and the fp32 copy, per token, over the residual stream width. The per-token
// figure comes from the model's geometry; the window comes from the knobs. Derived
// rather than assumed, because it scales with the configured window and at a
// whole-context window it is gigabytes — not a rounding error on the expert pool.
inline size_t prefill_carry_bytes(const AeonRuntimeConfig& runtime_cfg,
                                  const ModelMemoryGeometry& geometry) {
    const uint32_t ctx = runtime_cfg.context_size;
    const uint32_t window = runtime_cfg.prefill_window == 0
        ? ctx
        : std::min(runtime_cfg.prefill_window, ctx);
    return static_cast<size_t>(window) * geometry.prefill_carry_bytes_per_token;
}

} // namespace aeon::core
