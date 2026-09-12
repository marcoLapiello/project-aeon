#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline.hpp"

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
    double prefill_routed_section_ms{0.0};
    double prefill_expert_compute_ms{0.0};
    double decode_routed_section_ms{0.0};
    double decode_expert_compute_ms{0.0};
};

Result run_case(
    const std::string& model_dir,
    const std::string& label,
    bool enable_direct_io,
    const std::string& telemetry_path
) {
    std::cout << "\n[" << label << "] Initializing 12-slot version-2 swizzled pipeline..."
              << std::endl;
    aeon::core::V4Pipeline pipeline;
    pipeline.init_aeon(model_dir, 2, 12, 512, enable_direct_io);
    pipeline.enable_expert_timing();
    pipeline.enable_supply_telemetry(telemetry_path, label);

    const uint32_t warmup_token = pipeline.step(1, 0);
    assert(warmup_token == 69146);
    pipeline.reset_supply_telemetry();
    pipeline.reset_expert_timing();

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
        .tok_per_sec = tok_per_sec,
        .prefill_routed_section_ms = pipeline.expert_timing(
            aeon::core::RoutingPhase::Prefill).routed_section_ms,
        .prefill_expert_compute_ms = pipeline.expert_timing(
            aeon::core::RoutingPhase::Prefill).expert_compute_ms,
        .decode_routed_section_ms = pipeline.expert_timing(
            aeon::core::RoutingPhase::Decode).routed_section_ms,
        .decode_expert_compute_ms = pipeline.expert_timing(
            aeon::core::RoutingPhase::Decode).expert_compute_ms
    };

    std::cout << std::fixed << std::setprecision(2)
              << "  Total latency       : " << result.total_ms << " ms\n"
              << "  TTFT                : " << result.ttft_ms << " ms\n"
              << "  Decode throughput   : " << result.tok_per_sec << " tok/s\n"
              << "  Prefill routed GPU  : " << result.prefill_routed_section_ms
              << " ms (expert compute " << result.prefill_expert_compute_ms << " ms)\n"
              << "  Decode routed GPU   : " << result.decode_routed_section_ms
              << " ms (expert compute " << result.decode_expert_compute_ms << " ms)\n"
              << "  Layer 0 hits/misses : " << pipeline.layers[0]->cache_hits << "/"
              << pipeline.layers[0]->cache_misses << '\n'
              << "  Layer 1 hits/misses : " << pipeline.layers[1]->cache_hits << "/"
              << pipeline.layers[1]->cache_misses << '\n'
              << "  Telemetry output    : " << telemetry_path << std::endl;

    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << std::endl;
        return 2;
    }

    aeon::core::select_compute_device(true);
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

    std::cout << "Version-2 swizzled cold-miss source A/B benchmark\n"
              << "The direct-I/O case runs first so the mmap case is not warmed by it.\n";

    const Result direct = run_case(
        model_dir, "Direct O_DIRECT + io_uring", true,
        "/tmp/aeon_io_modes_v2_direct.jsonl");
    const Result mmap = run_case(
        model_dir, "mmap + memcpy source", false,
        "/tmp/aeon_io_modes_v2_mmap.jsonl");

    assert(direct.generated == mmap.generated);

    std::cout << "\nA/B comparison\n"
              << std::fixed << std::setprecision(2)
              << "  Direct TTFT delta  : " << (direct.ttft_ms - mmap.ttft_ms) << " ms\n"
              << "  Direct decode delta: " << (direct.tok_per_sec - mmap.tok_per_sec) << " tok/s\n"
              << "  Direct total delta : " << (direct.total_ms - mmap.total_ms) << " ms\n"
              << "  Generated tokens   : " << direct.generated.size() << " (identical)" << std::endl;

    return 0;
}
