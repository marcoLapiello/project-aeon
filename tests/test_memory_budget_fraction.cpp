// -----------------------------------------------------------------------------
// Gate — the VRAM utilization fraction and the per-device refusal.
//
// Pure and host-only: the device's free/total memory is injected as
// `DeviceMemoryInfo`, so every accept and refuse boundary is exercised without a
// GPU. The gate pins the property that replaced the fixed headroom — the planning
// allowance is `floor(fraction * total)`, the Hot pool follows the fraction, and a
// device whose free memory is below the allowance is refused by name rather than
// silently shrunk.
// -----------------------------------------------------------------------------

#include "infrastructure/memory/memory_budget.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

using aeon::core::AeonRuntimeConfig;
using aeon::core::AttentionStateMemory;
using aeon::core::DeviceMemoryInfo;
using aeon::core::ExpertFormatDescriptor;
using aeon::core::MemoryBudgetEngine;
using aeon::core::MemoryBudgetReport;
using aeon::core::ModelMemoryGeometry;

namespace {

uint32_t checks = 0;
uint32_t failures = 0;

void check(const char* label, bool ok, const std::string& detail = "") {
    std::printf("  %-58s %-30s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    ++checks;
    if (!ok) ++failures;
}

constexpr size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr int kDevice = 2;

// A small but valid geometry: two layers of eight experts, fp16-sized attention
// state that grows with the context. Nothing here needs an artifact.
ModelMemoryGeometry make_geometry() {
    ModelMemoryGeometry geometry;
    geometry.num_hidden_layers = 2;
    geometry.routed_experts = 8;
    geometry.experts_per_tok = 2;
    geometry.max_position_embeddings = 4096;
    geometry.prefill_carry_bytes_per_token = 128;
    geometry.attention_state_memory = [](uint32_t context_size) {
        AttentionStateMemory memory;
        memory.layer_state_bytes = static_cast<size_t>(context_size) * 1024;
        memory.rope_bytes = 4096;
        return memory;
    };
    return geometry;
}

MemoryBudgetReport evaluate(double fraction, size_t free_bytes, size_t total_bytes,
                            size_t dense_bytes = 1ULL * kGiB) {
    AeonRuntimeConfig cfg;
    cfg.context_size = 4096;
    cfg.warm_host_bytes = 0;
    cfg.gpu_memory_utilization = fraction;
    const ExpertFormatDescriptor format =
        aeon::core::make_current_swizzled_expert_format(2, 8);
    return MemoryBudgetEngine::evaluate(cfg, make_geometry(), dense_bytes, format,
                                        DeviceMemoryInfo{kDevice, free_bytes, total_bytes});
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  the VRAM utilization fraction: allowance, refusal, and Hot-pool scaling\n");
    std::printf("================================================================================\n");

    const size_t total = 24ULL * kGiB;

    // --- the record carries the device and the fraction ----------------------
    std::printf("\n[report record]\n");
    {
        const MemoryBudgetReport report = evaluate(0.9, total, total);
        check("feasible at full free memory", report.is_feasible, report.rejection_reason);
        check("device index recorded", report.device_index == kDevice);
        check("fraction recorded", report.gpu_memory_utilization > 0.89 &&
                                       report.gpu_memory_utilization < 0.91);
        const size_t expected = static_cast<size_t>(0.9 * static_cast<double>(total));
        check("usable == floor(fraction * total)", report.usable_vram_bytes == expected,
              std::to_string(report.usable_vram_bytes));
    }

    // --- the accept / refuse boundary ----------------------------------------
    std::printf("\n[allowance vs free]\n");
    {
        const size_t allowance = static_cast<size_t>(0.95 * static_cast<double>(total));
        // Free exactly at the allowance: accepted.
        const MemoryBudgetReport at = evaluate(0.95, allowance, total);
        check("free == allowance accepted", at.is_feasible, at.rejection_reason);
        check("usable == allowance at boundary", at.usable_vram_bytes == allowance);

        // One byte less: refused, by name.
        const MemoryBudgetReport below = evaluate(0.95, allowance - 1, total);
        check("free < allowance refused", !below.is_feasible);
        check("refusal names the remediation",
              below.rejection_reason.find("lower --gpu-memory-utilization") != std::string::npos,
              below.rejection_reason);

        // Plenty free: accepted even at a high fraction.
        const MemoryBudgetReport plenty = evaluate(0.99, total, total);
        check("free == total accepted", plenty.is_feasible, plenty.rejection_reason);
    }

    // --- a display-loaded device (free well below the allowance) -------------
    std::printf("\n[display-loaded device]\n");
    {
        // 10 GiB free of a 24 GiB card, at 0.95: the allowance is 22.8 GiB, far above
        // what is free, so the device is refused rather than silently shrunk.
        const MemoryBudgetReport report = evaluate(0.95, 10ULL * kGiB, total);
        check("display-loaded device refused", !report.is_feasible);
        check("refusal is a named device error",
              report.rejection_reason.find("device 2") != std::string::npos,
              report.rejection_reason);
    }

    // --- the Hot pool follows the fraction -----------------------------------
    std::printf("\n[Hot pool scales with the fraction]\n");
    {
        const MemoryBudgetReport low = evaluate(0.50, total, total);
        const MemoryBudgetReport high = evaluate(0.90, total, total);
        check("both feasible", low.is_feasible && high.is_feasible);
        check("a larger fraction yields more Hot slots",
              high.hot_vram_slots > low.hot_vram_slots,
              std::to_string(low.hot_vram_slots) + " -> " + std::to_string(high.hot_vram_slots));
        check("allowance grows with the fraction",
              high.usable_vram_bytes > low.usable_vram_bytes);
    }

    std::printf("\n--------------------------------------------------------------------------------\n");
    std::printf("  %u checks, %u failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
