#include "core/device.hpp"
#include "core/v4_pipeline.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << " Global VRAM Expert Pool End-to-End Generation Benchmark                      " << std::endl;
    std::cout << " Target: AMD Radeon RX 7900 XTX (Navi 31 / gfx1100) — Bare-Metal Wave32 Engine " << std::endl;
    std::cout << "================================================================================" << std::endl;

    aeon::core::select_compute_device(true);

    std::string aeon_model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

    // 1. Initialize pipeline with dynamic memory budgeting & global unified pool
    aeon::core::V4Pipeline pipeline;
    aeon::core::AeonRuntimeConfig pipeline_cfg;
    pipeline_cfg.context_size = 4096;
    pipeline_cfg.host_ram_bytes = 35ULL * 1024ULL * 1024ULL * 1024ULL;
    pipeline_cfg.preload_warm_host = false;

    // Benchmark across 2 active layers (matching Milestone 3 test conditions for direct comparison)
    pipeline.init_dynamic_global(aeon_model_dir, pipeline_cfg, 2);

    // Warmup run
    std::cout << "\n[Warmup] Executing 3 warmup generation steps..." << std::endl;
    std::vector<uint32_t> warmup_prompt = {1, 42, 100};
    pipeline.generate(warmup_prompt, 3);

    // Benchmark Suite matching Milestone 3 exactly:
    // - Short Prompt: 4 prompt tokens -> 16 generated tokens
    // - Medium Prompt: 8 prompt tokens -> 32 generated tokens
    struct BenchConfig {
        std::string name;
        std::vector<uint32_t> prompt;
        uint32_t new_tokens;
    };

    std::vector<BenchConfig> configs = {
        {"Short Prompt (4 tok) -> 16 gen", {1, 101, 2054, 300}, 16},
        {"Medium Prompt (8 tok) -> 32 gen", {1, 15, 230, 4000, 501, 882, 1092, 9999}, 32},
    };

    std::cout << "\n--------------------------------------------------------------------------------" << std::endl;
    std::cout << "                         Benchmark Execution Results                            " << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    for (const auto& cfg : configs) {
        // Reset registry hit/miss counters
        pipeline.expert_registry_->hits_hot = 0;
        pipeline.expert_registry_->hits_warm = 0;
        pipeline.expert_registry_->misses_cold = 0;

        double ttft_ms = 0.0;
        double tok_sec = 0.0;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto tokens = pipeline.generate(cfg.prompt, cfg.new_tokens, &ttft_ms, &tok_sec);
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_per_token = (cfg.new_tokens > 0) ? (total_ms / (cfg.prompt.size() + cfg.new_tokens)) : 0.0;

        uint64_t hits = pipeline.expert_registry_->hits_hot;
        uint64_t misses = pipeline.expert_registry_->hits_warm + pipeline.expert_registry_->misses_cold;
        double hit_rate = (hits + misses > 0) ? (100.0 * (double)hits / (double)(hits + misses)) : 0.0;

        std::cout << "\nScenario: " << cfg.name << std::endl;
        std::cout << "  > Total Prompt Tokens     : " << cfg.prompt.size() << std::endl;
        std::cout << "  > Total Generated Tokens  : " << tokens.size() << std::endl;
        std::cout << "  > Total Latency           : " << std::fixed << std::setprecision(2) << total_ms << " ms" << std::endl;
        std::cout << "  > TTFT (Prompt Prefill)   : " << std::fixed << std::setprecision(2) << ttft_ms << " ms ("
                  << std::setprecision(2) << (ttft_ms / cfg.prompt.size()) << " ms/tok)" << std::endl;
        std::cout << "  > Decode Throughput       : " << std::fixed << std::setprecision(2) << tok_sec << " tokens/sec" << std::endl;
        std::cout << "  > Decode Step Latency     : " << std::fixed << std::setprecision(2) << (1000.0 / tok_sec) << " ms/token" << std::endl;
        std::cout << "  > Overall Average Latency : " << std::fixed << std::setprecision(2) << ms_per_token << " ms/token" << std::endl;
        std::cout << "  > Tier 1 VRAM Cache Hits  : " << hits << std::endl;
        std::cout << "  > Tier 1 VRAM Cache Misses: " << misses << std::endl;
        std::cout << "  > Tier 1 VRAM Hit Rate    : " << std::fixed << std::setprecision(1) << hit_rate << " %" << std::endl;
    }

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  [SUCCESS] Global VRAM pool benchmark completed on physical silicon!          " << std::endl;
    std::cout << "================================================================================" << std::endl;
    return 0;
}
