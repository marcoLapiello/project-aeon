#include "core/device.hpp"
#include "kernel/aeon_moe_fused_w2.hpp"
#include "kernel/aeon_w4a16_swizzle.hpp"
#include "kernel/aeon_w4a16_swizzled_gemv.hpp"
#include "kernel/v4_pipeline_ops.hpp"
#include "kernel/w4a16_gemm.hpp"

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
constexpr int N = 4096;
constexpr int K = 2048;
constexpr int GROUPS = K / 32;
constexpr int WAVES = 8;
constexpr int RPW = 8;
constexpr int LPR = 4;
constexpr int ITERS = 16;
constexpr int BENCHMARK_ITERATIONS = 100;
constexpr std::size_t MALL_BYTES = 96ull << 20;
constexpr std::size_t COLD_WORKING_SET_BYTES = 4 * MALL_BYTES;

struct ColdW2Set {
    uint8_t* storage = nullptr;
    std::array<uint32_t*, EXPERTS> source_packed{};
    std::array<half*, EXPERTS> source_scale{};
    std::array<uint32_t*, EXPERTS> swizzled_packed{};
    std::array<half*, EXPERTS> swizzled_scale{};
    aeon::kernel::SwizzledW2ExpertPtrs fused{};
};

uint32_t make_source_word(std::size_t word_index, uint32_t seed) {
    uint32_t word = 0;
    for (int nibble = 0; nibble < 8; ++nibble) {
        const uint32_t value = static_cast<uint32_t>(
            (word_index * 3 + nibble * 5 + seed) % 16);
        word |= value << (4 * nibble);
    }
    return word;
}

void make_weights(
    int expert,
    std::vector<uint32_t>& source_packed,
    std::vector<half>& source_scale,
    std::vector<uint32_t>& swizzled_packed,
    std::vector<half>& swizzled_scale
) {
    const uint32_t seed = static_cast<uint32_t>(1 + expert * 7);
    for (std::size_t index = 0; index < source_packed.size(); ++index) {
        source_packed[index] = make_source_word(index, seed);
    }
    for (std::size_t index = 0; index < source_scale.size(); ++index) {
        source_scale[index] = __float2half(
            0.00390625f * static_cast<float>(1 + (index + seed) % 29));
    }
    aeon::swizzle_w4a16(source_packed.data(), source_scale.data(),
                        swizzled_packed.data(), swizzled_scale.data(),
                        N, K, aeon::kCfgW2);
}

double elapsed_us(hipEvent_t start, hipEvent_t stop, int iterations) {
    float elapsed_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
    return static_cast<double>(elapsed_ms) * 1000.0 / iterations;
}

void initialize_cold_set(
    ColdW2Set& set,
    const std::array<uint32_t*, EXPERTS>& base_source_packed,
    const std::array<half*, EXPERTS>& base_source_scale,
    const std::array<uint32_t*, EXPERTS>& base_swizzled_packed,
    const std::array<half*, EXPERTS>& base_swizzled_scale,
    std::size_t packed_bytes,
    std::size_t scale_bytes,
    std::size_t set_bytes
) {
    CHECK_HIP(hipMalloc(&set.storage, set_bytes));
    std::size_t offset = 0;
    for (int expert = 0; expert < EXPERTS; ++expert) {
        set.source_packed[expert] = reinterpret_cast<uint32_t*>(set.storage + offset);
        offset += packed_bytes;
        set.source_scale[expert] = reinterpret_cast<half*>(set.storage + offset);
        offset += scale_bytes;
        set.swizzled_packed[expert] = reinterpret_cast<uint32_t*>(set.storage + offset);
        offset += packed_bytes;
        set.swizzled_scale[expert] = reinterpret_cast<half*>(set.storage + offset);
        offset += scale_bytes;
        set.fused.w2[expert] = reinterpret_cast<const uint4*>(set.swizzled_packed[expert]);
        set.fused.s2[expert] = set.swizzled_scale[expert];
        CHECK_HIP(hipMemcpy(set.source_packed[expert], base_source_packed[expert],
                            packed_bytes, hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(set.source_scale[expert], base_source_scale[expert],
                            scale_bytes, hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(set.swizzled_packed[expert], base_swizzled_packed[expert],
                            packed_bytes, hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(set.swizzled_scale[expert], base_swizzled_scale[expert],
                            scale_bytes, hipMemcpyDeviceToDevice));
    }
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);
    const std::size_t packed_words = static_cast<std::size_t>(N) * K / 8;
    const std::size_t scale_count = static_cast<std::size_t>(N) * GROUPS;
    std::vector<half> host_hidden(EXPERTS * K);
    std::array<std::vector<uint32_t>, EXPERTS> source_packed;
    std::array<std::vector<half>, EXPERTS> source_scale;
    std::array<std::vector<uint32_t>, EXPERTS> swizzled_packed;
    std::array<std::vector<half>, EXPERTS> swizzled_scale;
    for (int expert = 0; expert < EXPERTS; ++expert) {
        source_packed[expert].resize(packed_words);
        source_scale[expert].resize(scale_count);
        swizzled_packed[expert].resize(packed_words);
        swizzled_scale[expert].resize(scale_count);
        make_weights(expert, source_packed[expert], source_scale[expert],
                     swizzled_packed[expert], swizzled_scale[expert]);
        for (int k = 0; k < K; ++k) {
            host_hidden[expert * K + k] =
                __float2half(static_cast<float>(((k + expert * 3) % 19) - 9) * 0.03125f);
        }
    }
    const std::array<float, EXPERTS> host_topk_weights = {0.05f, 0.10f, 0.15f,
                                                          0.20f, 0.25f, 0.25f};

    half* device_hidden = nullptr;
    float* device_topk_weights = nullptr;
    half* device_baseline = nullptr;
    half* device_down = nullptr;
    float* device_fused_f32 = nullptr;
    half* device_fused_f16 = nullptr;
    int* device_counters = nullptr;
    CHECK_HIP(hipMalloc(&device_hidden, host_hidden.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_topk_weights, EXPERTS * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_baseline, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_down, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_fused_f32, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_fused_f16, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_counters, (N / (WAVES * RPW)) * sizeof(int)));
    CHECK_HIP(hipMemcpy(device_hidden, host_hidden.data(), host_hidden.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_topk_weights, host_topk_weights.data(), EXPERTS * sizeof(float), hipMemcpyHostToDevice));

    std::array<uint32_t*, EXPERTS> device_source_packed{};
    std::array<half*, EXPERTS> device_source_scale{};
    std::array<uint32_t*, EXPERTS> device_swizzled_packed{};
    std::array<half*, EXPERTS> device_swizzled_scale{};
    aeon::kernel::SwizzledW2ExpertPtrs device_weights{};
    for (int expert = 0; expert < EXPERTS; ++expert) {
        CHECK_HIP(hipMalloc(&device_source_packed[expert], packed_words * sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&device_source_scale[expert], scale_count * sizeof(half)));
        CHECK_HIP(hipMalloc(&device_swizzled_packed[expert], packed_words * sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&device_swizzled_scale[expert], scale_count * sizeof(half)));
        CHECK_HIP(hipMemcpy(device_source_packed[expert], source_packed[expert].data(), packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_source_scale[expert], source_scale[expert].data(), scale_count * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_swizzled_packed[expert], swizzled_packed[expert].data(), packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_swizzled_scale[expert], swizzled_scale[expert].data(), scale_count * sizeof(half), hipMemcpyHostToDevice));
        device_weights.w2[expert] = reinterpret_cast<const uint4*>(device_swizzled_packed[expert]);
        device_weights.s2[expert] = device_swizzled_scale[expert];
    }

    const std::size_t packed_bytes = packed_words * sizeof(uint32_t);
    const std::size_t scale_bytes = scale_count * sizeof(half);
    const std::size_t fused_weight_bytes = EXPERTS *
        (packed_bytes + scale_bytes);
    const std::size_t expert_set_bytes = EXPERTS *
        (2 * packed_bytes + 2 * scale_bytes);
    const std::size_t cold_set_count =
        (COLD_WORKING_SET_BYTES + expert_set_bytes - 1) / expert_set_bytes + 1;
    std::vector<ColdW2Set> cold_sets(cold_set_count);
    for (ColdW2Set& cold_set : cold_sets) {
        initialize_cold_set(cold_set,
                            device_source_packed, device_source_scale,
                            device_swizzled_packed, device_swizzled_scale,
                            packed_bytes, scale_bytes, expert_set_bytes);
    }
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemset(device_baseline, 0, N * sizeof(half)));
    for (int expert = 0; expert < EXPERTS; ++expert) {
        aeon::kernel::dispatch_w4a16_gemm(
            device_hidden + static_cast<size_t>(expert) * K,
            device_source_packed[expert], device_source_scale[expert],
            device_down, 1, N, K);
        aeon::kernel::v4_pipeline_accumulate_expert_kernel<<<(N + 255) / 256, 256>>>(
            device_baseline, device_down, host_topk_weights[expert], N);
    }
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemset(device_fused_f32, 0, N * sizeof(float)));
    CHECK_HIP(hipMemset(device_fused_f16, 0, N * sizeof(half)));
    CHECK_HIP(hipMemset(device_counters, 0, (N / (WAVES * RPW)) * sizeof(int)));
    aeon::kernel::dispatch_aeon_moe_fused_w2_accum<WAVES, RPW, LPR, ITERS>(
        device_hidden, device_weights, device_topk_weights,
        nullptr,
        device_fused_f32, device_fused_f16, device_counters,
        EXPERTS, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<half> baseline(N);
    std::vector<half> fused(N);
    CHECK_HIP(hipMemcpy(baseline.data(), device_baseline, N * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(fused.data(), device_fused_f16, N * sizeof(half), hipMemcpyDeviceToHost));
    float max_difference = 0.0f;
    for (int row = 0; row < N; ++row) {
        max_difference = std::max(max_difference,
            std::abs(__half2float(baseline[row]) - __half2float(fused[row])));
    }

    hipEvent_t baseline_start, baseline_stop, fused_start, fused_stop;
    hipEvent_t cold_baseline_start, cold_baseline_stop;
    hipEvent_t cold_fused_start, cold_fused_stop;
    CHECK_HIP(hipEventCreate(&baseline_start));
    CHECK_HIP(hipEventCreate(&baseline_stop));
    CHECK_HIP(hipEventCreate(&fused_start));
    CHECK_HIP(hipEventCreate(&fused_stop));
    CHECK_HIP(hipEventCreate(&cold_baseline_start));
    CHECK_HIP(hipEventCreate(&cold_baseline_stop));
    CHECK_HIP(hipEventCreate(&cold_fused_start));
    CHECK_HIP(hipEventCreate(&cold_fused_stop));

    CHECK_HIP(hipEventRecord(baseline_start, 0));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration) {
        for (int expert = 0; expert < EXPERTS; ++expert) {
            aeon::kernel::dispatch_w4a16_gemm(
                device_hidden + static_cast<size_t>(expert) * K,
                device_source_packed[expert], device_source_scale[expert],
                device_down, 1, N, K);
            aeon::kernel::v4_pipeline_accumulate_expert_kernel<<<(N + 255) / 256, 256>>>(
                device_baseline, device_down, host_topk_weights[expert], N);
        }
    }
    CHECK_HIP(hipEventRecord(baseline_stop, 0));
    CHECK_HIP(hipEventSynchronize(baseline_stop));

    CHECK_HIP(hipEventRecord(fused_start, 0));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration) {
        aeon::kernel::dispatch_aeon_moe_fused_w2_accum<WAVES, RPW, LPR, ITERS>(
            device_hidden, device_weights, device_topk_weights,
            nullptr,
            device_fused_f32, device_fused_f16, device_counters,
            EXPERTS, N, K);
    }
    CHECK_HIP(hipEventRecord(fused_stop, 0));
    CHECK_HIP(hipEventSynchronize(fused_stop));

    const double baseline_us = elapsed_us(baseline_start, baseline_stop, BENCHMARK_ITERATIONS);
    const double fused_us = elapsed_us(fused_start, fused_stop, BENCHMARK_ITERATIONS);
    std::cout << std::fixed << std::setprecision(3)
              << "W2 + accumulation: current 12 launches=" << baseline_us
              << " us, fused 1 launch=" << fused_us
              << " us, speedup=" << (baseline_us / fused_us)
              << "x, output max diff=" << max_difference << std::endl;

    CHECK_HIP(hipMemset(device_baseline, 0, N * sizeof(half)));
    CHECK_HIP(hipMemset(device_fused_f32, 0, N * sizeof(float)));
    CHECK_HIP(hipMemset(device_fused_f16, 0, N * sizeof(half)));
    CHECK_HIP(hipMemset(device_counters, 0, (N / (WAVES * RPW)) * sizeof(int)));

    CHECK_HIP(hipEventRecord(cold_baseline_start, 0));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration) {
        const ColdW2Set& cold_set =
            cold_sets[static_cast<std::size_t>(iteration) % cold_sets.size()];
        for (int expert = 0; expert < EXPERTS; ++expert) {
            aeon::kernel::dispatch_w4a16_gemm(
                device_hidden + static_cast<size_t>(expert) * K,
                cold_set.source_packed[expert], cold_set.source_scale[expert],
                device_down, 1, N, K);
            aeon::kernel::v4_pipeline_accumulate_expert_kernel<<<(N + 255) / 256, 256>>>(
                device_baseline, device_down, host_topk_weights[expert], N);
        }
    }
    CHECK_HIP(hipEventRecord(cold_baseline_stop, 0));
    CHECK_HIP(hipEventSynchronize(cold_baseline_stop));

    CHECK_HIP(hipEventRecord(cold_fused_start, 0));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration) {
        const ColdW2Set& cold_set =
            cold_sets[static_cast<std::size_t>(iteration) % cold_sets.size()];
        aeon::kernel::dispatch_aeon_moe_fused_w2_accum<WAVES, RPW, LPR, ITERS>(
            device_hidden, cold_set.fused, device_topk_weights,
            nullptr,
            device_fused_f32, device_fused_f16, device_counters,
            EXPERTS, N, K);
    }
    CHECK_HIP(hipEventRecord(cold_fused_stop, 0));
    CHECK_HIP(hipEventSynchronize(cold_fused_stop));

    const double cold_baseline_us = elapsed_us(
        cold_baseline_start, cold_baseline_stop, BENCHMARK_ITERATIONS);
    const double cold_fused_us = elapsed_us(
        cold_fused_start, cold_fused_stop, BENCHMARK_ITERATIONS);
    const double cold_fused_gbps =
        static_cast<double>(fused_weight_bytes) / cold_fused_us / 1000.0;
    std::cout << std::fixed << std::setprecision(3)
              << "W2 cold rotation: " << cold_set_count << " sets / "
              << (static_cast<double>(cold_set_count * expert_set_bytes) /
                  (1024.0 * 1024.0))
              << " MiB working set, current 12 launches=" << cold_baseline_us
              << " us, fused 1 launch=" << cold_fused_us
              << " us, speedup=" << (cold_baseline_us / cold_fused_us)
              << "x, fused weight BW=" << cold_fused_gbps << " GB/s"
              << std::endl;

    CHECK_HIP(hipEventDestroy(baseline_start));
    CHECK_HIP(hipEventDestroy(baseline_stop));
    CHECK_HIP(hipEventDestroy(fused_start));
    CHECK_HIP(hipEventDestroy(fused_stop));
    CHECK_HIP(hipEventDestroy(cold_baseline_start));
    CHECK_HIP(hipEventDestroy(cold_baseline_stop));
    CHECK_HIP(hipEventDestroy(cold_fused_start));
    CHECK_HIP(hipEventDestroy(cold_fused_stop));
    CHECK_HIP(hipFree(device_hidden));
    CHECK_HIP(hipFree(device_topk_weights));
    CHECK_HIP(hipFree(device_baseline));
    CHECK_HIP(hipFree(device_down));
    CHECK_HIP(hipFree(device_fused_f32));
    CHECK_HIP(hipFree(device_fused_f16));
    CHECK_HIP(hipFree(device_counters));
    for (int expert = 0; expert < EXPERTS; ++expert) {
        CHECK_HIP(hipFree(device_source_packed[expert]));
        CHECK_HIP(hipFree(device_source_scale[expert]));
        CHECK_HIP(hipFree(device_swizzled_packed[expert]));
        CHECK_HIP(hipFree(device_swizzled_scale[expert]));
    }
    for (ColdW2Set& cold_set : cold_sets) {
        CHECK_HIP(hipFree(cold_set.storage));
    }
    return 0;
}