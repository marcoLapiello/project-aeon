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
#include "architecture/deepseek_v4/spec/v4_memory_geometry.hpp"
#include "architecture/deepseek_v4/spec/v4_model_contract.hpp"
#include "infrastructure/parallel/parallel_topology.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using aeon::core::AeonRuntimeConfig;
using aeon::core::AttentionStateMemory;
using aeon::core::DeviceMemoryInfo;
using aeon::core::ExpertFormatDescriptor;
using aeon::core::LayerRange;
using aeon::core::MemoryBudgetEngine;
using aeon::core::MemoryBudgetReport;
using aeon::core::ModelMemoryGeometry;
using aeon::core::ParallelTopology;
using aeon::core::ParallelTopologyConfig;

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

    // --- the host budget splits across stages with no byte lost ----------------
    std::printf("\n[host-budget split]\n");
    {
        const std::vector<uint32_t> counts{22, 21};
        const auto shares = MemoryBudgetEngine::split_host_budget(41ULL * kGiB, counts);
        check("the shares sum to the whole",
              shares.size() == 2 && shares[0] + shares[1] == 41ULL * kGiB,
              std::to_string(shares[0]) + " + " + std::to_string(shares[1]));
        check("the larger stage gets the larger share", shares[0] > shares[1]);
        const auto one = MemoryBudgetEngine::split_host_budget(41ULL * kGiB, {43});
        check("one stage takes the whole budget", one.size() == 1 && one[0] == 41ULL * kGiB);
    }

    // --- the host region is one allocation shared by every stage ---------------
    //
    // A pipeline must hold the host RAM the user set, not a multiple of it: the stages'
    // Warm lanes tile the head of one region, the corridor sits at the tail, and the
    // lanes at the end are the ones a larger corridor eats.
    std::printf("\n[shared host region]\n");
    {
        const ExpertFormatDescriptor format =
            aeon::core::make_current_swizzled_expert_format(2, 8);
        AeonRuntimeConfig cfg;
        cfg.context_size = 4096;
        // decode corridor 4 slots, swept peak 16 slots at 2 experts per token.
        cfg.warm_host_bytes = 40ULL * format.payload_bytes;
        const std::vector<uint32_t> counts{1, 1};
        const auto plan = MemoryBudgetEngine::plan_shared_host_region(cfg, format, 2, counts);
        check("every stage sees the same region",
              plan.size() == 2 && plan[0].region_slots == 40 && plan[1].region_slots == 40);
        check("lanes tile the region minus the decode corridor",
              plan[0].lane_slots + plan[1].lane_slots == 36 && plan[0].lane_base_slots == 0 &&
                  plan[1].lane_base_slots == plan[0].lane_slots,
              std::to_string(plan[0].lane_slots) + " + " + std::to_string(plan[1].lane_slots));

        // The last stage's lane is what a swept corridor takes first.
        cfg.max_hot_vram_slots = 6;
        const auto last = MemoryBudgetEngine::evaluate(
            cfg, make_geometry(), 1ULL * kGiB, format,
            DeviceMemoryInfo{kDevice, total, total}, &plan[1]);
        const auto first = MemoryBudgetEngine::evaluate(
            cfg, make_geometry(), 1ULL * kGiB, format,
            DeviceMemoryInfo{kDevice, total, total}, &plan[0]);
        check("both lanes are feasible", first.is_feasible && last.is_feasible,
              first.rejection_reason + last.rejection_reason);
        check("the report carries the lane base", last.host_lane_base_slots == 18);
        check("the first lane is untouched by a swept corridor",
              first.warm_host_slots_min == plan[0].lane_slots,
              std::to_string(first.warm_host_slots_min));
        check("the last lane shrinks under a swept corridor",
              last.warm_host_slots_min < plan[1].lane_slots,
              std::to_string(last.warm_host_slots_min));

        // Without a Warm budget a pipeline still shares one corridor-only region.
        AeonRuntimeConfig no_warm = cfg;
        no_warm.warm_host_bytes = 0;
        const auto corridor_only =
            MemoryBudgetEngine::plan_shared_host_region(no_warm, format, 2, counts);
        check("no Warm, two stages: one corridor-sized region and no lanes",
              corridor_only[0].region_slots == 16 && corridor_only[0].lane_slots == 0 &&
                  corridor_only[1].lane_slots == 0,
              std::to_string(corridor_only[0].region_slots));
        const auto single = MemoryBudgetEngine::plan_shared_host_region(no_warm, format, 2, {2});
        check("no Warm, one stage: no region at all", single[0].region_slots == 0);

        // A region below the corridor's peak is refused, whatever the stage count.
        AeonRuntimeConfig tiny_region = cfg;
        tiny_region.warm_host_bytes = 8ULL * format.payload_bytes;
        const auto tiny_plan =
            MemoryBudgetEngine::plan_shared_host_region(tiny_region, format, 2, counts);
        const auto tiny_report = MemoryBudgetEngine::evaluate(
            tiny_region, make_geometry(), 1ULL * kGiB, format,
            DeviceMemoryInfo{kDevice, total, total}, &tiny_plan[0]);
        check("a region below the corridor peak is refused", !tiny_report.is_feasible,
              tiny_report.rejection_reason);
    }

    // --- per-stage geometry and dense bytes sum to the whole -------------------
    //
    // A pipeline splits the model into contiguous layer ranges, and each stage is
    // budgeted against its own range. The split must be lossless: the per-stage
    // attention state (and the per-stage dense upload) must sum to the single-device
    // figures, or a pipeline would silently change the memory the model needs.
    const std::string config_path =
        "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/config.json";
    if (std::filesystem::exists(config_path)) {
        std::printf("\n[per-stage geometry / dense sums to the whole]\n");
        const auto config = aeon::core::DeepSeekV4Config::load_from_json(config_path);
        const auto whole = aeon::core::make_v4_memory_geometry(config);
        const size_t whole_dense = aeon::core::V4ModelContract::uploaded_dense_bytes(config);
        const uint32_t layers = static_cast<uint32_t>(config.num_hidden_layers);

        for (uint32_t pp = 1; pp <= 4; ++pp) {
            ParallelTopologyConfig topo_cfg;
            topo_cfg.pipeline_parallel = pp;
            const auto topology =
                ParallelTopology::resolve(topo_cfg, static_cast<int>(pp), 1u, layers);

            size_t sum_layer_state = 0;
            size_t sum_dense = 0;
            size_t sum_rope = 0;
            bool ranges_ok = true;
            for (uint32_t stage = 0; stage < topology.pp(); ++stage) {
                const LayerRange range = topology.stage_layers(stage);
                const auto geometry = aeon::core::make_v4_memory_geometry(config, range);
                const auto attention = geometry.attention_state_memory(4096);
                sum_layer_state += attention.layer_state_bytes;
                sum_rope += attention.rope_bytes;
                const bool has_head = (range.first + range.count == layers);
                sum_dense += aeon::core::V4ModelContract::uploaded_dense_bytes(
                    config, range, has_head);
                ranges_ok = ranges_ok && (range.first + range.count <= layers);
            }

            const auto whole_attention = whole.attention_state_memory(4096);
            check(("pp=" + std::to_string(pp) + ": layer state sums to the whole").c_str(),
                  ranges_ok && sum_layer_state == whole_attention.layer_state_bytes,
                  std::to_string(sum_layer_state) + " vs " +
                      std::to_string(whole_attention.layer_state_bytes));
            check(("pp=" + std::to_string(pp) + ": RoPE is paid on every stage").c_str(),
                  sum_rope == static_cast<size_t>(pp) * whole_attention.rope_bytes,
                  std::to_string(sum_rope));
            check(("pp=" + std::to_string(pp) + ": dense upload sums to the whole").c_str(),
                  sum_dense == whole_dense,
                  std::to_string(sum_dense) + " vs " + std::to_string(whole_dense));
        }

        // A per-stage budget evaluated on a stage's own device is feasible when the
        // device has the room, and refuses when a stage's host share cannot hold its
        // corridor peak.
        ParallelTopologyConfig two_cfg;
        two_cfg.pipeline_parallel = 2;
        const auto two = ParallelTopology::resolve(two_cfg, 2, 1u, layers);
        const LayerRange first = two.stage_layers(0);
        AeonRuntimeConfig cfg;
        cfg.context_size = 4096;
        cfg.warm_host_bytes = 8ULL * kGiB;
        const ExpertFormatDescriptor stage_format = [&] {
            // The artifact declares the whole model; a stage sees its own layers.
            auto format = aeon::core::make_current_swizzled_expert_format(
                layers, static_cast<uint32_t>(config.n_routed_experts));
            format.num_layers = first.count;
            return format;
        }();
        const auto stage_report = MemoryBudgetEngine::evaluate(
            cfg, aeon::core::make_v4_memory_geometry(config, first),
            aeon::core::V4ModelContract::uploaded_dense_bytes(
                config, first, first.first + first.count == layers),
            stage_format, DeviceMemoryInfo{kDevice, total, total});
        check("a stage budget is feasible on a device with room", stage_report.is_feasible,
              stage_report.rejection_reason);

        // A host share below one swept corridor is refused (the engine's corridor
        // check), which is what stops a pipeline from overcommitting host RAM.
        AeonRuntimeConfig tiny = cfg;
        tiny.warm_host_bytes = 1ULL * 1024ULL * 1024ULL;  // 1 MiB
        const auto refused = MemoryBudgetEngine::evaluate(
            tiny, aeon::core::make_v4_memory_geometry(config, first),
            aeon::core::V4ModelContract::uploaded_dense_bytes(
                config, first, first.first + first.count == layers),
            stage_format, DeviceMemoryInfo{kDevice, total, total});
        check("a stage share below its corridor peak is refused", !refused.is_feasible,
              refused.rejection_reason);
    } else {
        std::printf("\n[per-stage geometry] skipped: %s is absent\n", config_path.c_str());
    }

    std::printf("\n--------------------------------------------------------------------------------\n");
    std::printf("  %u checks, %u failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
