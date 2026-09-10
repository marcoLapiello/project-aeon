#include "core/device.hpp"
#include "kernel/aeon_moe_fused_w13.hpp"
#include "kernel/aeon_w4a16_swizzle.hpp"
#include "kernel/aeon_w4a16_swizzled_gemv.hpp"
#include "kernel/v4_pipeline_ops.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

namespace {

constexpr int EXPERTS = 6;
constexpr int N = 2048;
constexpr int K = 4096;
constexpr int GROUPS = K / 32;
constexpr int WAVES = 8;
constexpr int RPW = 4;
constexpr int LPR = 8;
constexpr int ITERS = 16;
constexpr int BENCHMARK_ITERATIONS = 100;
constexpr int BENCHMARK_TRIALS = 5;
constexpr float SWIGLU_LIMIT = 10.0f;

uint32_t make_source_word(std::size_t word_index, uint32_t seed) {
    uint32_t word = 0;
    for (int nibble = 0; nibble < 8; ++nibble) {
        const uint32_t value = static_cast<uint32_t>(
            (word_index * 3 + nibble * 5 + seed) % 16);
        word |= value << (4 * nibble);
    }
    return word;
}

void make_swizzled_weights(
    int expert,
    std::vector<uint32_t>& swizzled_packed,
    std::vector<half>& swizzled_scale
) {
    const std::size_t packed_words = static_cast<std::size_t>(N) * K / 8;
    const std::size_t scale_count = static_cast<std::size_t>(N) * GROUPS;
    std::vector<uint32_t> source_packed(packed_words);
    std::vector<half> source_scale(scale_count);
    const uint32_t seed = static_cast<uint32_t>(1 + expert * 7);

    for (std::size_t index = 0; index < packed_words; ++index) {
        source_packed[index] = make_source_word(index, seed);
    }
    for (std::size_t index = 0; index < scale_count; ++index) {
        source_scale[index] = __float2half(
            0.00390625f * static_cast<float>(1 + (index + seed) % 29));
    }
    aeon::swizzle_w4a16(source_packed.data(), source_scale.data(),
                        swizzled_packed.data(), swizzled_scale.data(),
                        N, K, aeon::kCfgW13);
}

double elapsed_us(hipEvent_t start, hipEvent_t stop, int iterations) {
    float elapsed_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
    return static_cast<double>(elapsed_ms) * 1000.0 / iterations;
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);

    const std::size_t packed_words = static_cast<std::size_t>(N) * K / 8;
    const std::size_t scale_count = static_cast<std::size_t>(N) * GROUPS;
    std::vector<half> host_activation(K);
    std::array<std::vector<uint32_t>, EXPERTS> host_w1;
    std::array<std::vector<uint32_t>, EXPERTS> host_w3;
    std::array<std::vector<half>, EXPERTS> host_s1;
    std::array<std::vector<half>, EXPERTS> host_s3;
    for (int expert = 0; expert < EXPERTS; ++expert) {
        host_w1[expert].resize(packed_words);
        host_w3[expert].resize(packed_words);
        host_s1[expert].resize(scale_count);
        host_s3[expert].resize(scale_count);
        make_swizzled_weights(expert, host_w1[expert], host_s1[expert]);
        make_swizzled_weights(expert + EXPERTS, host_w3[expert], host_s3[expert]);
    }
    for (int k = 0; k < K; ++k) {
        host_activation[k] = __float2half(static_cast<float>((k % 19) - 9) * 0.03125f);
    }

    half* device_activation = nullptr;
    std::array<uint32_t*, EXPERTS> device_w1{};
    std::array<uint32_t*, EXPERTS> device_w3{};
    std::array<half*, EXPERTS> device_s1{};
    std::array<half*, EXPERTS> device_s3{};
    half* device_gate = nullptr;
    half* device_up = nullptr;
    half* device_baseline_hidden = nullptr;
    half* device_fused_hidden = nullptr;
    half* device_staged_hidden = nullptr;
    CHECK_HIP(hipMalloc(&device_activation, K * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_gate, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_up, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_baseline_hidden, EXPERTS * N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_fused_hidden, EXPERTS * N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_staged_hidden, EXPERTS * N * sizeof(half)));
    CHECK_HIP(hipMemcpy(device_activation, host_activation.data(), K * sizeof(half), hipMemcpyHostToDevice));

    aeon::kernel::SwizzledW13ExpertPtrs device_weights{};
    for (int expert = 0; expert < EXPERTS; ++expert) {
        CHECK_HIP(hipMalloc(&device_w1[expert], packed_words * sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&device_w3[expert], packed_words * sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&device_s1[expert], scale_count * sizeof(half)));
        CHECK_HIP(hipMalloc(&device_s3[expert], scale_count * sizeof(half)));
        CHECK_HIP(hipMemcpy(device_w1[expert], host_w1[expert].data(),
                            packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_w3[expert], host_w3[expert].data(),
                            packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_s1[expert], host_s1[expert].data(),
                            scale_count * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_s3[expert], host_s3[expert].data(),
                            scale_count * sizeof(half), hipMemcpyHostToDevice));
        device_weights.w1[expert] = reinterpret_cast<const uint4*>(device_w1[expert]);
        device_weights.w3[expert] = reinterpret_cast<const uint4*>(device_w3[expert]);
        device_weights.s1[expert] = device_s1[expert];
        device_weights.s3[expert] = device_s3[expert];
    }

    for (int expert = 0; expert < EXPERTS; ++expert) {
        aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<WAVES, RPW, LPR, ITERS>(
            device_activation, device_w1[expert], device_s1[expert],
            device_gate, N, K);
        aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<WAVES, RPW, LPR, ITERS>(
            device_activation, device_w3[expert], device_s3[expert],
            device_up, N, K);
        aeon::kernel::v4_pipeline_swiglu_clamp_kernel<<<(N + 255) / 256, 256>>>(
            device_gate, device_up,
            device_baseline_hidden + static_cast<size_t>(expert) * N,
            N, SWIGLU_LIMIT);
    }
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<WAVES, RPW, LPR, ITERS>(
        device_activation, device_weights, device_fused_hidden,
        nullptr, 0,
        EXPERTS, N, K, SWIGLU_LIMIT);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<WAVES, RPW, LPR, ITERS, true>(
        device_activation, device_weights, device_staged_hidden,
        nullptr, 0,
        EXPERTS, N, K, SWIGLU_LIMIT);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<half> baseline(EXPERTS * N);
    std::vector<half> fused(EXPERTS * N);
    std::vector<half> staged(EXPERTS * N);
    CHECK_HIP(hipMemcpy(baseline.data(), device_baseline_hidden,
                        baseline.size() * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(fused.data(), device_fused_hidden,
                        fused.size() * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(staged.data(), device_staged_hidden,
                        staged.size() * sizeof(half), hipMemcpyDeviceToHost));
    float max_error = 0.0f;
    float max_staged_difference = 0.0f;
    for (std::size_t index = 0; index < baseline.size(); ++index) {
        max_error = std::max(max_error,
            std::abs(__half2float(baseline[index]) - __half2float(fused[index])));
        max_staged_difference = std::max(max_staged_difference,
            std::abs(__half2float(fused[index]) - __half2float(staged[index])));
    }

    hipEvent_t baseline_start, baseline_stop, fused_start, fused_stop;
    CHECK_HIP(hipEventCreate(&baseline_start));
    CHECK_HIP(hipEventCreate(&baseline_stop));
    CHECK_HIP(hipEventCreate(&fused_start));
    CHECK_HIP(hipEventCreate(&fused_stop));
    hipEvent_t staged_start, staged_stop;
    CHECK_HIP(hipEventCreate(&staged_start));
    CHECK_HIP(hipEventCreate(&staged_stop));

    CHECK_HIP(hipEventRecord(baseline_start, 0));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration) {
        for (int expert = 0; expert < EXPERTS; ++expert) {
            aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<WAVES, RPW, LPR, ITERS>(
                device_activation, device_w1[expert], device_s1[expert],
                device_gate, N, K);
            aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<WAVES, RPW, LPR, ITERS>(
                device_activation, device_w3[expert], device_s3[expert],
                device_up, N, K);
            aeon::kernel::v4_pipeline_swiglu_clamp_kernel<<<(N + 255) / 256, 256>>>(
                device_gate, device_up,
                device_baseline_hidden + static_cast<size_t>(expert) * N,
                N, SWIGLU_LIMIT);
        }
    }
    CHECK_HIP(hipEventRecord(baseline_stop, 0));
    CHECK_HIP(hipEventSynchronize(baseline_stop));

    std::array<double, BENCHMARK_TRIALS> direct_samples{};
    std::array<double, BENCHMARK_TRIALS> staged_samples{};
    for (int trial = 0; trial < BENCHMARK_TRIALS; ++trial) {
        const bool direct_first = (trial % 2) == 0;
        for (int pass = 0; pass < 2; ++pass) {
            const bool run_direct = direct_first ? pass == 0 : pass == 1;
            hipEvent_t start = run_direct ? fused_start : staged_start;
            hipEvent_t stop = run_direct ? fused_stop : staged_stop;
            CHECK_HIP(hipEventRecord(start, 0));
            for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration) {
                if (run_direct) {
                    aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<WAVES, RPW, LPR, ITERS>(
                        device_activation, device_weights, device_fused_hidden,
                        nullptr, 0,
                        EXPERTS, N, K, SWIGLU_LIMIT);
                } else {
                    aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<WAVES, RPW, LPR, ITERS, true>(
                        device_activation, device_weights, device_staged_hidden,
                        nullptr, 0,
                        EXPERTS, N, K, SWIGLU_LIMIT);
                }
            }
            CHECK_HIP(hipEventRecord(stop, 0));
            CHECK_HIP(hipEventSynchronize(stop));
            const double sample_us = elapsed_us(start, stop, BENCHMARK_ITERATIONS);
            if (run_direct) {
                direct_samples[trial] = sample_us;
            } else {
                staged_samples[trial] = sample_us;
            }
        }
    }

    auto median = [](std::array<double, BENCHMARK_TRIALS> samples) {
        std::sort(samples.begin(), samples.end());
        return samples[BENCHMARK_TRIALS / 2];
    };
    const double baseline_us = elapsed_us(baseline_start, baseline_stop, BENCHMARK_ITERATIONS);
    const double fused_us = median(direct_samples);
    const double staged_us = median(staged_samples);
    const auto direct_range = std::minmax_element(direct_samples.begin(), direct_samples.end());
    const auto staged_range = std::minmax_element(staged_samples.begin(), staged_samples.end());
    std::cout << std::fixed << std::setprecision(3)
              << "W1/W3 + SwiGLU: current 18 launches=" << baseline_us
              << " us, fused direct median=" << fused_us
              << " us [" << *direct_range.first << "," << *direct_range.second << "]"
              << ", fused staged median=" << staged_us
              << " us [" << *staged_range.first << "," << *staged_range.second << "]"
              << " us, direct/staged=" << (fused_us / staged_us)
              << "x, baseline/direct max diff=" << max_error
              << ", direct/staged max diff=" << max_staged_difference << std::endl;

    CHECK_HIP(hipEventDestroy(baseline_start));
    CHECK_HIP(hipEventDestroy(baseline_stop));
    CHECK_HIP(hipEventDestroy(fused_start));
    CHECK_HIP(hipEventDestroy(fused_stop));
    CHECK_HIP(hipEventDestroy(staged_start));
    CHECK_HIP(hipEventDestroy(staged_stop));
    CHECK_HIP(hipFree(device_activation));
    CHECK_HIP(hipFree(device_gate));
    CHECK_HIP(hipFree(device_up));
    CHECK_HIP(hipFree(device_baseline_hidden));
    CHECK_HIP(hipFree(device_fused_hidden));
    CHECK_HIP(hipFree(device_staged_hidden));
    for (int expert = 0; expert < EXPERTS; ++expert) {
        CHECK_HIP(hipFree(device_w1[expert]));
        CHECK_HIP(hipFree(device_w3[expert]));
        CHECK_HIP(hipFree(device_s1[expert]));
        CHECK_HIP(hipFree(device_s3[expert]));
    }
    return 0;
}