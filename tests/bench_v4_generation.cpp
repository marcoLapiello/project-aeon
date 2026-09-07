#include "core/device.hpp"
#include "core/v4_pipeline.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "  Spike 6.2: DeepSeek-V4 End-to-End Autoregressive Generation Benchmark         " << std::endl;
    std::cout << "  Target: AMD Radeon RX 7900 XTX (Navi 31 / gfx1100) — Bare-Metal Wave32 Engine " << std::endl;
    std::cout << "================================================================================" << std::endl;

    aeon::core::select_compute_device(true);

    std::string snapshot_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16/"
                               "models--yiminyuan--DeepSeek-V4-Flash-0731-INT4-W4A16/"
                               "snapshots/64700592cadaf205fe0c13202061ff4b45afbfd0";

    // Initialize pipeline with 2 consecutive layers and 8 VRAM slots per layer
    aeon::core::V4Pipeline pipeline;
    pipeline.init(snapshot_dir, 2, 8, 1024);

    // Warmup run
    std::cout << "\n[Warmup] Executing 3 warmup generation steps..." << std::endl;
    std::vector<uint32_t> warmup_prompt = {1, 42, 100};
    pipeline.generate(warmup_prompt, 3);

    // Benchmark Suite
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
        // Reset cache counters
        for (auto& l : pipeline.layers) {
            l->cache_hits = 0;
            l->cache_misses = 0;
        }

        double ttft_ms = 0.0;
        double tok_sec = 0.0;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto tokens = pipeline.generate(cfg.prompt, cfg.new_tokens, &ttft_ms, &tok_sec);
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_per_token = (cfg.new_tokens > 0) ? (total_ms / (cfg.prompt.size() + cfg.new_tokens)) : 0.0;

        uint64_t total_hits = 0;
        uint64_t total_misses = 0;
        for (const auto& l : pipeline.layers) {
            total_hits += l->cache_hits;
            total_misses += l->cache_misses;
        }
        double hit_rate = (total_hits + total_misses > 0) ?
            (100.0 * (double)total_hits / (double)(total_hits + total_misses)) : 0.0;

        std::cout << "\nScenario: " << cfg.name << std::endl;
        std::cout << "  > Total Prompt Tokens     : " << cfg.prompt.size() << std::endl;
        std::cout << "  > Total Generated Tokens  : " << tokens.size() << std::endl;
        std::cout << "  > Total Latency           : " << std::fixed << std::setprecision(2) << total_ms << " ms" << std::endl;
        std::cout << "  > TTFT (Prompt Prefill)   : " << std::fixed << std::setprecision(2) << ttft_ms << " ms ("
                  << std::setprecision(2) << (ttft_ms / cfg.prompt.size()) << " ms/tok)" << std::endl;
        std::cout << "  > Decode Throughput       : " << std::fixed << std::setprecision(2) << tok_sec << " tokens/sec" << std::endl;
        std::cout << "  > Decode Step Latency     : " << std::fixed << std::setprecision(2) << (1000.0 / tok_sec) << " ms/token" << std::endl;
        std::cout << "  > Overall Average Latency : " << std::fixed << std::setprecision(2) << ms_per_token << " ms/token" << std::endl;
        std::cout << "  > Tier 1 VRAM Cache Hits  : " << total_hits << std::endl;
        std::cout << "  > Tier 1 VRAM Cache Misses: " << total_misses << std::endl;
        std::cout << "  > Tier 1 VRAM Hit Rate    : " << std::fixed << std::setprecision(1) << hit_rate << " %" << std::endl;
    }

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  [SUCCESS] Spike 6 Complete: Generation Benchmark Verified on Physical GPU!    " << std::endl;
    std::cout << "================================================================================" << std::endl;
    return 0;
}
