#include "core/device.hpp"
#include "core/v4_pipeline.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "  Project Aeon: 43-Layer Full-Model Autoregressive Inference Benchmark          " << std::endl;
    std::cout << "  Model: DeepSeek-V4-Flash-0731 (INT4-W4A16, 43 Layers, 11,008 Routed Experts)  " << std::endl;
    std::cout << "  Target: AMD Radeon RX 7900 XTX (Navi 31 / gfx1100) — Bare-Metal Wave32 Engine " << std::endl;
    std::cout << "================================================================================" << std::endl;

    aeon::core::select_compute_device(true);

    std::string aeon_model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

    // 1. Configure runtime for full 43-layer model
    aeon::core::V4Pipeline pipeline;
    aeon::core::AeonRuntimeConfig pipeline_cfg;
    const uint64_t warm_host_gib = argc > 1 ? std::stoull(argv[1]) : 35;
    pipeline_cfg.context_size = 4096; // 4096 tokens context
    pipeline_cfg.host_ram_bytes = warm_host_gib * 1024ULL * 1024ULL * 1024ULL;
    pipeline_cfg.preload_warm_host = warm_host_gib > 0;
    std::cout << "[Profile] Warm Host budget: " << warm_host_gib << " GiB"
              << (pipeline_cfg.preload_warm_host ? " (enabled)" : " (disabled)") << std::endl;

    std::cout << "\n[Step 1] Initializing complete 43-layer pipeline..." << std::endl;
    auto t_init_start = std::chrono::high_resolution_clock::now();
    pipeline.init_dynamic_global(aeon_model_dir, pipeline_cfg, 43);
    auto t_init_end = std::chrono::high_resolution_clock::now();
    double init_sec = std::chrono::duration<double>(t_init_end - t_init_start).count();
    std::cout << "\n  > Complete 43-layer initialization completed in " << std::fixed << std::setprecision(2)
              << init_sec << " seconds." << std::endl;

    // 2. Warmup run (single token prompt, 2 tokens generated)
    std::cout << "\n[Step 2] Executing single-pass warmup across all 43 layers..." << std::endl;
    std::vector<uint32_t> warmup_prompt = {1, 100};
    pipeline.generate(warmup_prompt, 2);
    std::cout << "  > Warmup pass complete across all 43 layers!" << std::endl;

    // 3. Autoregressive Performance Benchmark
    std::cout << "\n[Step 3] Running full 43-layer generation benchmark..." << std::endl;
    std::vector<uint32_t> prompt = {1, 101, 2054, 300}; // 4 tokens
    uint32_t gen_tokens = 8;                             // 8 generated tokens

    // Reset hit/miss counters
    pipeline.expert_registry_->hits_hot = 0;
    pipeline.expert_registry_->hits_warm = 0;
    pipeline.expert_registry_->misses_cold = 0;
    pipeline.reset_routing_locality_stats();

    double ttft_ms = 0.0;
    double tok_sec = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tokens = pipeline.generate(prompt, gen_tokens, &ttft_ms, &tok_sec);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_per_token = (gen_tokens > 0) ? (total_ms / (prompt.size() + gen_tokens)) : 0.0;

    uint64_t hits = pipeline.expert_registry_->hits_hot;
    uint64_t misses = pipeline.expert_registry_->hits_warm + pipeline.expert_registry_->misses_cold;
    double hit_rate = (hits + misses > 0) ? (100.0 * (double)hits / (double)(hits + misses)) : 0.0;

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "                  43-Layer Full-Model Benchmark Results                         " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "  > Total Prompt Tokens     : " << prompt.size() << std::endl;
    std::cout << "  > Total Generated Tokens  : " << tokens.size() << std::endl;
    std::cout << "  > Generated Token IDs     : [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i] << (i + 1 < tokens.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  > Total Execution Time    : " << std::fixed << std::setprecision(2) << total_ms << " ms" << std::endl;
    std::cout << "  > TTFT (Prompt Prefill)   : " << std::fixed << std::setprecision(2) << ttft_ms << " ms ("
              << std::setprecision(2) << (ttft_ms / prompt.size()) << " ms/tok across 43 layers)" << std::endl;
    std::cout << "  > Decode Throughput       : " << std::fixed << std::setprecision(2) << tok_sec << " tokens/sec" << std::endl;
    std::cout << "  > Decode Step Latency     : " << std::fixed << std::setprecision(2) << (1000.0 / tok_sec) << " ms/token" << std::endl;
    std::cout << "  > Overall Average Latency : " << std::fixed << std::setprecision(2) << ms_per_token << " ms/token" << std::endl;
    std::cout << "  > Tier 1 VRAM Cache Hits  : " << hits << std::endl;
    std::cout << "  > Tier 1 VRAM Cache Misses: " << misses << std::endl;
    std::cout << "  > Tier 1 VRAM Hit Rate    : " << std::fixed << std::setprecision(1) << hit_rate << " %" << std::endl;
    std::cout << "  > Tier 2 Warm Host Hits   : " << pipeline.expert_registry_->hits_warm << std::endl;
    std::cout << "  > Tier 3 Cold NVMe Misses : " << pipeline.expert_registry_->misses_cold << std::endl;
    std::cout << "================================================================================" << std::endl;

    pipeline.print_routing_locality_report();

    return 0;
}
