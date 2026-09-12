#pragma once

#include "core/config.hpp"
#include "core/expert_format.hpp"

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
constexpr double HOST_RAM_MAX_RATIO         = 0.70;                       // 70% cap (~43-45 GB) to leave comfortable room for OS
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

    // Diagnostic A/B control. The production default keeps asynchronous refill enabled.
    bool enable_warm_refill{true};

    // Hardware target device index
    int device_id{0};

};

struct MemoryBudgetReport {
    bool is_feasible{false};
    std::string rejection_reason;

    // Hardware limits
    size_t total_vram_bytes{0};
    size_t free_vram_bytes{0};
    size_t total_host_ram_bytes{0};
    size_t max_allowed_host_ram_bytes{0};

    // VRAM allocations
    size_t vram_dense_bytes{0};
    size_t vram_kv_bytes{0};
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
            << "    - Total Host RAM       : " << (double)total_host_ram_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - Max Allowed Host RAM : " << (double)max_allowed_host_ram_bytes / (1024 * 1024 * 1024) << " GB (70% safety cap)\n"
            << "--------------------------------------------------------------------------------\n"
            << "  VRAM Allocation Breakdown:\n"
            << "    - Dense Model Weights  : " << (double)vram_dense_bytes / (1024 * 1024 * 1024) << " GB\n"
            << "    - KV Cache Buffer      : " << (double)vram_kv_bytes / (1024 * 1024 * 1024) << " GB\n"
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

        // 2. Query physical Host memory
        struct sysinfo si;
        if (sysinfo(&si) != 0) {
            report.is_feasible = false;
            report.rejection_reason = "Failed to query system host memory via sysinfo()";
            return report;
        }

        report.total_host_ram_bytes       = static_cast<size_t>(si.totalram) * si.mem_unit;
        report.max_allowed_host_ram_bytes = static_cast<size_t>(report.total_host_ram_bytes * HOST_RAM_MAX_RATIO);

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

        // 4. Calculate exact VRAM requirements
        // KV Cache for DeepSeek-V4 MLA: layers * context_size * kv_dim * sizeof(half)
        // kv_dim is head_dim (512) for num_key_value_heads (1)
        report.vram_dense_bytes   = dense_weights_bytes;
        report.vram_kv_bytes      = static_cast<size_t>(model_cfg.num_hidden_layers) *
                                    runtime_cfg.context_size *
                                    model_cfg.head_dim *
                                    sizeof(uint16_t);
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

        // Calculate max viable context size for this GPU
        size_t non_kv_required = report.vram_dense_bytes + report.vram_scratch_bytes +
                                 report.vram_headroom_bytes + report.vram_min_active_bytes;
        if (total_vram > non_kv_required) {
            size_t max_kv_bytes = total_vram - non_kv_required;
            size_t bytes_per_token = static_cast<size_t>(model_cfg.num_hidden_layers) *
                                     model_cfg.head_dim * sizeof(uint16_t);
            report.max_viable_context_size = static_cast<uint32_t>(max_kv_bytes / bytes_per_token);
        } else {
            report.max_viable_context_size = 0;
        }

        // 6. Hard Feasibility Gate Evaluation
        if (baseline_vram_needed > total_vram) {
            report.is_feasible = false;
            std::ostringstream err_oss;
            err_oss << "VRAM capacity exceeded: Required baseline "
                    << (double)baseline_vram_needed / (1024 * 1024 * 1024) << " GB, but device has only "
                    << (double)total_vram / (1024 * 1024 * 1024) << " GB.";
            report.rejection_reason = err_oss.str();
            return report;
        }

        // 7. Calculate Hot VRAM Expert Pool capacity
        size_t remaining_for_experts = total_vram - (report.vram_dense_bytes +
                                                     report.vram_kv_bytes +
                                                     report.vram_scratch_bytes +
                                                     report.vram_headroom_bytes);
        report.vram_available_for_experts = remaining_for_experts;
        report.hot_vram_slots = static_cast<uint32_t>(
            remaining_for_experts / expert_format.payload_bytes);
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
