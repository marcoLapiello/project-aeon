#pragma once

// -----------------------------------------------------------------------------
// The memory budget engine: the code that queries the device and host, evaluates
// feasibility, and derives the tier partition from the runtime knobs.
//
// It reads `AeonRuntimeConfig` (the knobs) and `ModelMemoryGeometry` (what only the
// architecture knows), produces `MemoryBudgetReport` (the output), and is the only
// part of the budget that touches HIP or `sysinfo`. The report and the geometry are
// separate headers so a caller can read the report's shape without pulling in the
// device query, and so the engine reads no model type at all.
// -----------------------------------------------------------------------------

#include "infrastructure/core/model_memory_geometry.hpp"
#include "infrastructure/core/memory_budget_report.hpp"
#include "infrastructure/core/runtime_config.hpp"
#include "infrastructure/core/expert_format.hpp"

#include <hip/hip_runtime.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace aeon::core {

class MemoryBudgetEngine {
public:
    static MemoryBudgetReport evaluate(
        const AeonRuntimeConfig& runtime_cfg,
        const ModelMemoryGeometry& geometry,
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

        if (geometry.num_hidden_layers < 0 || geometry.routed_experts < 0 ||
            expert_format.num_layers != static_cast<uint32_t>(geometry.num_hidden_layers) ||
            expert_format.experts_per_layer != static_cast<uint32_t>(geometry.routed_experts)) {
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

        if (runtime_cfg.context_size > static_cast<uint32_t>(geometry.max_position_embeddings)) {
            report.is_feasible = false;
            report.rejection_reason = "Requested context size (" + std::to_string(runtime_cfg.context_size) +
                ") exceeds model max_position_embeddings (" +
                std::to_string(geometry.max_position_embeddings) + ")";
            return report;
        }

        AttentionStateMemory attention_memory;
        try {
            attention_memory = geometry.attention_state_memory(runtime_cfg.context_size);
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
        // The prefill workspace is **derived from the configured knobs**: the residual
        // carry is a pure function of the window, and the batch scratch is an allowance
        // the host checks its real allocation against at load. The `100 MiB` literal
        // this replaces was wrong in both directions — it over-counted decode scratch
        // several-fold and did not cover the batch scratch at all.
        const size_t carry_bytes = prefill_carry_bytes(runtime_cfg, geometry);
        report.vram_prefill_carry_bytes = runtime_cfg.prefill_sweep ? carry_bytes : 0;
        report.vram_batch_scratch_bytes = runtime_cfg.prefill_sweep
            ? batch_scratch_allowance_bytes(std::max<uint32_t>(1, runtime_cfg.prefill_chunk))
            : 0;
        // The decode workspace is **not** a knob: `V4ActivationScratch` and
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
        uint32_t min_active_slots = static_cast<uint32_t>(2 * geometry.experts_per_tok);
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
            size_t high = static_cast<size_t>(geometry.max_position_embeddings);
            while (low < high) {
                const size_t midpoint = low + (high - low + 1) / 2;
                const auto candidate = geometry.attention_state_memory(static_cast<uint32_t>(midpoint));
                if (non_attention_required + candidate.total_bytes() <= report.usable_vram_bytes) {
                    low = midpoint;
                } else {
                    high = midpoint - 1;
                }
            }
            report.max_viable_context_size = static_cast<uint32_t>(low);
        }

        // Hard feasibility gate: reject before anything is uploaded.
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

        // Hot VRAM expert-pool capacity from what remains.
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

        // 8. Calculate the host region: **one** pinned budget that Warm and the
        // transport corridor share.
        //
        // `warm_host_bytes` is the **total** host RAM for expert payloads, and the
        // partition inside it moves at runtime. Warm and staging were two separate
        // allocations, so the RAM a run held was `warm + staging` — a number the user
        // could only reach by adding two settings, and one they could overshoot
        // without noticing. They are also physically the same thing (expert payloads
        // of the artifact's width, 4 KiB-aligned, same layout); the difference is
        // policy, not storage. One number that *is* the total is what makes the
        // setting a guard rather than a hint.
        //
        // The corridor's three requirements, and the phase-precise partitions they
        // imply. Each phase cuts the boundary so Warm keeps everything the corridor is
        // not using **for that phase**: the differences are real residency, and decode
        // is the phase that wants the most of it.
        const uint32_t experts_per_layer =
            static_cast<uint32_t>(expert_format.experts_per_layer);
        const auto staging = aeon::core::staging_slot_counts(
            runtime_cfg, experts_per_layer, static_cast<uint32_t>(geometry.experts_per_tok));
        report.transient_staging_bytes =
            static_cast<size_t>(staging.peak) * expert_format.payload_bytes;
        report.staging_decode_bytes =
            static_cast<size_t>(staging.decode) * expert_format.payload_bytes;
        report.staging_batch_bytes =
            static_cast<size_t>(staging.batch) * expert_format.payload_bytes;

        report.configured_host_budget_bytes = runtime_cfg.warm_host_bytes;
        const size_t host_region_bytes = runtime_cfg.warm_host_bytes;
        // The ceiling is already reserve-subtracted, so this is the whole check.
        if (host_region_bytes > report.max_allowed_host_ram_bytes) {
            report.is_feasible = false;
            report.rejection_reason =
                "Host budget " +
                std::to_string(host_region_bytes / (1024ULL * 1024 * 1024)) +
                " GiB exceeds the " +
                std::to_string(report.max_allowed_host_ram_bytes / (1024ULL * 1024 * 1024)) +
                " GiB allowed";
            return report;
        }
        if (host_region_bytes != 0 && host_region_bytes < report.transient_staging_bytes) {
            report.is_feasible = false;
            report.rejection_reason =
                "Host budget " +
                std::to_string(host_region_bytes / (1024 * 1024)) +
                " MiB cannot hold the " +
                std::to_string(report.transient_staging_bytes / (1024 * 1024)) +
                " MiB the corridor needs at its largest (a swept prefill) — raise it or "
                "lower prefill_sweep_staging_blocks";
            return report;
        }

        const uint32_t region_slots = static_cast<uint32_t>(
            host_region_bytes / expert_format.payload_bytes);
        report.host_region_slots = region_slots;
        report.host_region_bytes =
            static_cast<size_t>(region_slots) * expert_format.payload_bytes;
        // Warm keeps what each phase's corridor does not use. Decode gives the corridor
        // the least, so it is where Warm is largest — which is the point: decode is
        // where the residency pays.
        const auto warm_for = [region_slots](uint32_t staging_slots) {
            return region_slots > staging_slots ? region_slots - staging_slots : 0u;
        };
        const uint32_t warm_slots_max = warm_for(staging.decode);
        report.warm_host_slots_routed = warm_for(staging.batch);
        report.warm_host_slots_min = warm_for(staging.prefill);

        const uint32_t total_experts = static_cast<uint32_t>(expert_format.total_experts());
        uint32_t remaining_after_vram = (total_experts > report.hot_vram_slots)
            ? (total_experts - report.hot_vram_slots)
            : 0;

        report.warm_host_slots = std::min(remaining_after_vram, warm_slots_max);
        report.warm_host_bytes = static_cast<size_t>(report.warm_host_slots) *
                                 expert_format.payload_bytes;
        report.persistent_warm_host_budget_bytes = report.warm_host_bytes;

        // 9. Cold NVMe pool gets the rest
        report.cold_nvme_slots = total_experts - (report.hot_vram_slots + report.warm_host_slots);

        report.is_feasible = true;
        return report;
    }

    static MemoryBudgetReport evaluate(
        const AeonRuntimeConfig& runtime_cfg,
        const ModelMemoryGeometry& geometry,
        size_t dense_weights_bytes
    ) {
        return evaluate(
            runtime_cfg,
            geometry,
            dense_weights_bytes,
            make_current_swizzled_expert_format(
                static_cast<uint32_t>(geometry.num_hidden_layers),
                static_cast<uint32_t>(geometry.routed_experts)));
    }
};

} // namespace aeon::core
