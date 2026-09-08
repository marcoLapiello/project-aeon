#include "core/device.hpp"
#include "core/v4_pipeline.hpp"

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Result {
    std::string label;
    std::vector<uint32_t> generated;
    double total_ms{0.0};
    double ttft_ms{0.0};
    double tok_per_sec{0.0};
};

Result run_case(const std::string& model_dir, const std::string& label, bool enable_direct_io) {
    std::cout << "\n[" << label << "] Initializing 12-slot native pipeline..." << std::endl;
    aeon::core::V4Pipeline pipeline;
    pipeline.init_aeon(model_dir, 2, 12, 512, enable_direct_io);

    const uint32_t warmup_token = pipeline.step(1, 0);
    assert(warmup_token == 69146);

    const std::vector<uint32_t> prompt = {1, 100, 256};
    constexpr uint32_t GENERATED_TOKENS = 16;
    double ttft_ms = 0.0;
    double tok_per_sec = 0.0;

    const auto start = std::chrono::steady_clock::now();
    const auto generated = pipeline.generate(prompt, GENERATED_TOKENS, &ttft_ms, &tok_per_sec);
    const auto end = std::chrono::steady_clock::now();

    for (uint32_t token : generated) {
        assert(token < 129280);
    }

    Result result{
        .label = label,
        .generated = generated,
        .total_ms = std::chrono::duration<double, std::milli>(end - start).count(),
        .ttft_ms = ttft_ms,
        .tok_per_sec = tok_per_sec
    };

    std::cout << std::fixed << std::setprecision(2)
              << "  Total latency       : " << result.total_ms << " ms\n"
              << "  TTFT                : " << result.ttft_ms << " ms\n"
              << "  Decode throughput   : " << result.tok_per_sec << " tok/s\n"
              << "  Layer 0 hits/misses : " << pipeline.layers[0]->cache_hits << "/"
              << pipeline.layers[0]->cache_misses << '\n'
              << "  Layer 1 hits/misses : " << pipeline.layers[1]->cache_hits << "/"
              << pipeline.layers[1]->cache_misses << std::endl;

    return result;
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

    std::cout << "Native cold-miss source A/B benchmark\n"
              << "The direct-I/O case runs first so the mmap case is not warmed by it.\n";

    const Result direct = run_case(model_dir, "Direct O_DIRECT + io_uring", true);
    const Result mmap = run_case(model_dir, "mmap + memcpy source", false);

    assert(direct.generated == mmap.generated);

    std::cout << "\nA/B comparison\n"
              << std::fixed << std::setprecision(2)
              << "  Direct TTFT delta  : " << (direct.ttft_ms - mmap.ttft_ms) << " ms\n"
              << "  Direct decode delta: " << (direct.tok_per_sec - mmap.tok_per_sec) << " tok/s\n"
              << "  Direct total delta : " << (direct.total_ms - mmap.total_ms) << " ms\n"
              << "  Generated tokens   : " << direct.generated.size() << " (identical)" << std::endl;

    return 0;
}
