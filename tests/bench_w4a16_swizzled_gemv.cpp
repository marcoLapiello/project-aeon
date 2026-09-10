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

constexpr int kWaves = 8;
constexpr int kIterations = 16;
constexpr int kBenchmarkIterations = 200;

uint32_t make_source_word(std::size_t word_index) {
    uint32_t word = 0;
    for (int nibble = 0; nibble < 8; ++nibble) {
        const uint32_t value = static_cast<uint32_t>((word_index * 3 + nibble * 5 + 1) % 16);
        word |= value << (4 * nibble);
    }
    return word;
}

template <int RPW, int LPR>
void benchmark_shape(const char* name, int N, int K) {
    const int groups = K / 32;
    const std::size_t packed_words = static_cast<std::size_t>(N) * K / 8;
    const std::size_t scale_count = static_cast<std::size_t>(N) * groups;

    std::vector<half> host_activation(K);
    std::vector<uint32_t> host_source_packed(packed_words);
    std::vector<half> host_source_scale(scale_count);
    std::vector<uint32_t> host_swizzled_packed(packed_words);
    std::vector<half> host_swizzled_scale(scale_count);
    std::vector<half> host_source_output(N);
    std::vector<half> host_swizzled_output(N);

    for (int k = 0; k < K; ++k) {
        host_activation[k] = __float2half(static_cast<float>((k % 19) - 9) * 0.03125f);
    }
    for (std::size_t index = 0; index < host_source_packed.size(); ++index) {
        host_source_packed[index] = make_source_word(index);
    }
    for (std::size_t index = 0; index < host_source_scale.size(); ++index) {
        host_source_scale[index] = __float2half(
            0.00390625f * static_cast<float>(1 + index % 29));
    }

    aeon::swizzle_w4a16(
        host_source_packed.data(), host_source_scale.data(),
        host_swizzled_packed.data(), host_swizzled_scale.data(),
        N, K, RPW == 4 ? aeon::kCfgW13 : aeon::kCfgW2);

    half* device_activation = nullptr;
    uint32_t* device_source_packed = nullptr;
    half* device_source_scale = nullptr;
    uint32_t* device_swizzled_packed = nullptr;
    half* device_swizzled_scale = nullptr;
    half* device_source_output = nullptr;
    half* device_swizzled_output = nullptr;

    CHECK_HIP(hipMalloc(&device_activation, host_activation.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_source_packed, host_source_packed.size() * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_source_scale, host_source_scale.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_swizzled_packed, host_swizzled_packed.size() * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_swizzled_scale, host_swizzled_scale.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_source_output, host_source_output.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_swizzled_output, host_swizzled_output.size() * sizeof(half)));

    CHECK_HIP(hipMemcpy(device_activation, host_activation.data(),
                        host_activation.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_source_packed, host_source_packed.data(),
                        host_source_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_source_scale, host_source_scale.data(),
                        host_source_scale.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_swizzled_packed, host_swizzled_packed.data(),
                        host_swizzled_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_swizzled_scale, host_swizzled_scale.data(),
                        host_swizzled_scale.size() * sizeof(half), hipMemcpyHostToDevice));

    aeon::kernel::dispatch_w4a16_gemm(
        device_activation, device_source_packed, device_source_scale,
        device_source_output, 1, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<kWaves, RPW, LPR, kIterations>(
        device_activation, device_swizzled_packed, device_swizzled_scale,
        device_swizzled_output, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(host_source_output.data(), device_source_output,
                        host_source_output.size() * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(host_swizzled_output.data(), device_swizzled_output,
                        host_swizzled_output.size() * sizeof(half), hipMemcpyDeviceToHost));

    float max_difference = 0.0f;
    for (int row = 0; row < N; ++row) {
        max_difference = std::max(
            max_difference,
            std::abs(__half2float(host_source_output[row]) -
                     __half2float(host_swizzled_output[row])));
    }

    hipEvent_t source_start, source_stop, swizzled_start, swizzled_stop;
    CHECK_HIP(hipEventCreate(&source_start));
    CHECK_HIP(hipEventCreate(&source_stop));
    CHECK_HIP(hipEventCreate(&swizzled_start));
    CHECK_HIP(hipEventCreate(&swizzled_stop));

    CHECK_HIP(hipEventRecord(source_start, 0));
    for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration) {
        aeon::kernel::dispatch_w4a16_gemm(
            device_activation, device_source_packed, device_source_scale,
            device_source_output, 1, N, K);
    }
    CHECK_HIP(hipEventRecord(source_stop, 0));
    CHECK_HIP(hipEventSynchronize(source_stop));

    CHECK_HIP(hipEventRecord(swizzled_start, 0));
    for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration) {
        aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<kWaves, RPW, LPR, kIterations>(
            device_activation, device_swizzled_packed, device_swizzled_scale,
            device_swizzled_output, N, K);
    }
    CHECK_HIP(hipEventRecord(swizzled_stop, 0));
    CHECK_HIP(hipEventSynchronize(swizzled_stop));

    float source_total_ms = 0.0f;
    float swizzled_total_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&source_total_ms, source_start, source_stop));
    CHECK_HIP(hipEventElapsedTime(&swizzled_total_ms, swizzled_start, swizzled_stop));

    const double source_us = source_total_ms * 1000.0 / kBenchmarkIterations;
    const double swizzled_us = swizzled_total_ms * 1000.0 / kBenchmarkIterations;
    std::cout << std::fixed << std::setprecision(3)
              << name << ": current=" << source_us << " us, swizzled=" << swizzled_us
              << " us, speedup=" << (source_us / swizzled_us)
              << "x, output max diff=" << max_difference << std::endl;

    CHECK_HIP(hipEventDestroy(source_start));
    CHECK_HIP(hipEventDestroy(source_stop));
    CHECK_HIP(hipEventDestroy(swizzled_start));
    CHECK_HIP(hipEventDestroy(swizzled_stop));
    CHECK_HIP(hipFree(device_activation));
    CHECK_HIP(hipFree(device_source_packed));
    CHECK_HIP(hipFree(device_source_scale));
    CHECK_HIP(hipFree(device_swizzled_packed));
    CHECK_HIP(hipFree(device_swizzled_scale));
    CHECK_HIP(hipFree(device_source_output));
    CHECK_HIP(hipFree(device_swizzled_output));
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);
    benchmark_shape<4, 8>("W1/W3", 2048, 4096);
    benchmark_shape<8, 4>("W2", 4096, 2048);
    return 0;
}