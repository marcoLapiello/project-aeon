#include "core/device.hpp"
#include "kernel/aeon_w4a16_swizzle.hpp"
#include "kernel/aeon_w4a16_swizzled_gemv.hpp"
#include "kernel/w4a16_gemm.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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

constexpr int N = 2048;
constexpr int K = 4096;
constexpr int GROUPS = K / 32;
constexpr int WAVES = 8;
constexpr int LPR = 8;
constexpr int RPW = 4;
constexpr int ITERS = 16;
constexpr int BENCHMARK_ITERATIONS = 200;

uint32_t make_source_word(std::size_t word_index, uint32_t seed) {
    uint32_t word = 0;
    for (int nibble = 0; nibble < 8; ++nibble) {
        const uint32_t value = static_cast<uint32_t>(
            (word_index * 3 + nibble * 5 + seed) % 16);
        word |= value << (4 * nibble);
    }
    return word;
}

void fill_weights(
    uint32_t seed,
    std::vector<uint32_t>& packed,
    std::vector<half>& scale
) {
    for (std::size_t index = 0; index < packed.size(); ++index) {
        packed[index] = make_source_word(index, seed);
    }
    for (std::size_t index = 0; index < scale.size(); ++index) {
        scale[index] = __float2half(
            0.00390625f * static_cast<float>(1 + (index + seed) % 29));
    }
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
    std::vector<uint32_t> source_packed_a(packed_words);
    std::vector<uint32_t> source_packed_b(packed_words);
    std::vector<half> source_scale_a(scale_count);
    std::vector<half> source_scale_b(scale_count);
    std::vector<uint32_t> swizzled_packed_a(packed_words);
    std::vector<uint32_t> swizzled_packed_b(packed_words);
    std::vector<half> swizzled_scale_a(scale_count);
    std::vector<half> swizzled_scale_b(scale_count);
    std::vector<half> source_output_a(N);
    std::vector<half> source_output_b(N);
    std::vector<half> swizzled_output_a(N);
    std::vector<half> swizzled_output_b(N);

    for (int k = 0; k < K; ++k) {
        host_activation[k] = __float2half(static_cast<float>((k % 19) - 9) * 0.03125f);
    }
    fill_weights(1, source_packed_a, source_scale_a);
    fill_weights(7, source_packed_b, source_scale_b);
    aeon::swizzle_w4a16(source_packed_a.data(), source_scale_a.data(),
                        swizzled_packed_a.data(), swizzled_scale_a.data(),
                        N, K, aeon::kCfgW13);
    aeon::swizzle_w4a16(source_packed_b.data(), source_scale_b.data(),
                        swizzled_packed_b.data(), swizzled_scale_b.data(),
                        N, K, aeon::kCfgW13);

    half* device_activation = nullptr;
    uint32_t* device_source_packed_a = nullptr;
    uint32_t* device_source_packed_b = nullptr;
    half* device_source_scale_a = nullptr;
    half* device_source_scale_b = nullptr;
    uint32_t* device_swizzled_packed_a = nullptr;
    uint32_t* device_swizzled_packed_b = nullptr;
    half* device_swizzled_scale_a = nullptr;
    half* device_swizzled_scale_b = nullptr;
    half* device_source_output_a = nullptr;
    half* device_source_output_b = nullptr;
    half* device_swizzled_output_a = nullptr;
    half* device_swizzled_output_b = nullptr;

    CHECK_HIP(hipMalloc(&device_activation, K * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_source_packed_a, packed_words * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_source_packed_b, packed_words * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_source_scale_a, scale_count * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_source_scale_b, scale_count * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_swizzled_packed_a, packed_words * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_swizzled_packed_b, packed_words * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_swizzled_scale_a, scale_count * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_swizzled_scale_b, scale_count * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_source_output_a, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_source_output_b, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_swizzled_output_a, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_swizzled_output_b, N * sizeof(half)));

    CHECK_HIP(hipMemcpy(device_activation, host_activation.data(), K * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_source_packed_a, source_packed_a.data(), packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_source_packed_b, source_packed_b.data(), packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_source_scale_a, source_scale_a.data(), scale_count * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_source_scale_b, source_scale_b.data(), scale_count * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_swizzled_packed_a, swizzled_packed_a.data(), packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_swizzled_packed_b, swizzled_packed_b.data(), packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_swizzled_scale_a, swizzled_scale_a.data(), scale_count * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_swizzled_scale_b, swizzled_scale_b.data(), scale_count * sizeof(half), hipMemcpyHostToDevice));

    aeon::kernel::dispatch_w4a16_gemm(
        device_activation, device_source_packed_a, device_source_scale_a,
        device_source_output_a, 1, N, K);
    aeon::kernel::dispatch_w4a16_gemm(
        device_activation, device_source_packed_b, device_source_scale_b,
        device_source_output_b, 1, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    aeon::kernel::dispatch_aeon_w4a16_swizzled_dual_gemv<WAVES, RPW, LPR, ITERS>(
        device_activation,
        device_swizzled_packed_a, device_swizzled_scale_a,
        device_swizzled_packed_b, device_swizzled_scale_b,
        device_swizzled_output_a, device_swizzled_output_b, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(source_output_a.data(), device_source_output_a, N * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(source_output_b.data(), device_source_output_b, N * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(swizzled_output_a.data(), device_swizzled_output_a, N * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(swizzled_output_b.data(), device_swizzled_output_b, N * sizeof(half), hipMemcpyDeviceToHost));

    float max_difference = 0.0f;
    for (int row = 0; row < N; ++row) {
        max_difference = std::max(max_difference,
            std::abs(__half2float(source_output_a[row]) - __half2float(swizzled_output_a[row])));
        max_difference = std::max(max_difference,
            std::abs(__half2float(source_output_b[row]) - __half2float(swizzled_output_b[row])));
    }

    hipEvent_t source_start, source_stop, swizzled_start, swizzled_stop;
    CHECK_HIP(hipEventCreate(&source_start));
    CHECK_HIP(hipEventCreate(&source_stop));
    CHECK_HIP(hipEventCreate(&swizzled_start));
    CHECK_HIP(hipEventCreate(&swizzled_stop));

    CHECK_HIP(hipEventRecord(source_start, 0));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration) {
        aeon::kernel::dispatch_w4a16_gemm(
            device_activation, device_source_packed_a, device_source_scale_a,
            device_source_output_a, 1, N, K);
        aeon::kernel::dispatch_w4a16_gemm(
            device_activation, device_source_packed_b, device_source_scale_b,
            device_source_output_b, 1, N, K);
    }
    CHECK_HIP(hipEventRecord(source_stop, 0));
    CHECK_HIP(hipEventSynchronize(source_stop));

    CHECK_HIP(hipEventRecord(swizzled_start, 0));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration) {
        aeon::kernel::dispatch_aeon_w4a16_swizzled_dual_gemv<WAVES, RPW, LPR, ITERS>(
            device_activation,
            device_swizzled_packed_a, device_swizzled_scale_a,
            device_swizzled_packed_b, device_swizzled_scale_b,
            device_swizzled_output_a, device_swizzled_output_b, N, K);
    }
    CHECK_HIP(hipEventRecord(swizzled_stop, 0));
    CHECK_HIP(hipEventSynchronize(swizzled_stop));

    const double source_us = elapsed_us(source_start, source_stop, BENCHMARK_ITERATIONS);
    const double swizzled_us = elapsed_us(swizzled_start, swizzled_stop, BENCHMARK_ITERATIONS);
    std::cout << std::fixed << std::setprecision(3)
              << "W1/W3 pair: current two launches=" << source_us
              << " us, swizzled dual=" << swizzled_us
              << " us, speedup=" << (source_us / swizzled_us)
              << "x, output max diff=" << max_difference << std::endl;

    CHECK_HIP(hipEventDestroy(source_start));
    CHECK_HIP(hipEventDestroy(source_stop));
    CHECK_HIP(hipEventDestroy(swizzled_start));
    CHECK_HIP(hipEventDestroy(swizzled_stop));
    CHECK_HIP(hipFree(device_activation));
    CHECK_HIP(hipFree(device_source_packed_a));
    CHECK_HIP(hipFree(device_source_packed_b));
    CHECK_HIP(hipFree(device_source_scale_a));
    CHECK_HIP(hipFree(device_source_scale_b));
    CHECK_HIP(hipFree(device_swizzled_packed_a));
    CHECK_HIP(hipFree(device_swizzled_packed_b));
    CHECK_HIP(hipFree(device_swizzled_scale_a));
    CHECK_HIP(hipFree(device_swizzled_scale_b));
    CHECK_HIP(hipFree(device_source_output_a));
    CHECK_HIP(hipFree(device_source_output_b));
    CHECK_HIP(hipFree(device_swizzled_output_a));
    CHECK_HIP(hipFree(device_swizzled_output_b));
    return 0;
}