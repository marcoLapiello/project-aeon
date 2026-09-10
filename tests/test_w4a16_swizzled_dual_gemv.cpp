#include "core/device.hpp"
#include "kernel/aeon_w4a16_swizzle.hpp"
#include "kernel/aeon_w4a16_swizzled_gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
constexpr int ITERS = 16;

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
    std::vector<uint32_t>& source_packed,
    std::vector<half>& source_scale
) {
    for (std::size_t index = 0; index < source_packed.size(); ++index) {
        source_packed[index] = make_source_word(index, seed);
    }
    for (std::size_t index = 0; index < source_scale.size(); ++index) {
        source_scale[index] = __float2half(
            0.00390625f * static_cast<float>(1 + (index + seed) % 29));
    }
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
    std::vector<half> separate_a(N);
    std::vector<half> separate_b(N);
    std::vector<half> dual_a(N);
    std::vector<half> dual_b(N);

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
    uint32_t* device_packed_a = nullptr;
    uint32_t* device_packed_b = nullptr;
    half* device_scale_a = nullptr;
    half* device_scale_b = nullptr;
    half* device_separate_a = nullptr;
    half* device_separate_b = nullptr;
    half* device_dual_a = nullptr;
    half* device_dual_b = nullptr;
    CHECK_HIP(hipMalloc(&device_activation, K * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_packed_a, packed_words * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_packed_b, packed_words * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_scale_a, scale_count * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_scale_b, scale_count * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_separate_a, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_separate_b, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_dual_a, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_dual_b, N * sizeof(half)));

    CHECK_HIP(hipMemcpy(device_activation, host_activation.data(), K * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_packed_a, swizzled_packed_a.data(), packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_packed_b, swizzled_packed_b.data(), packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_scale_a, swizzled_scale_a.data(), scale_count * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_scale_b, swizzled_scale_b.data(), scale_count * sizeof(half), hipMemcpyHostToDevice));

    aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<WAVES, 4, 8, ITERS>(
        device_activation, device_packed_a, device_scale_a, device_separate_a, N, K);
    aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<WAVES, 4, 8, ITERS>(
        device_activation, device_packed_b, device_scale_b, device_separate_b, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(separate_a.data(), device_separate_a, N * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(separate_b.data(), device_separate_b, N * sizeof(half), hipMemcpyDeviceToHost));

    aeon::kernel::dispatch_aeon_w4a16_swizzled_dual_gemv<WAVES, 4, 8, ITERS>(
        device_activation,
        device_packed_a, device_scale_a,
        device_packed_b, device_scale_b,
        device_dual_a, device_dual_b, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(dual_a.data(), device_dual_a, N * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(dual_b.data(), device_dual_b, N * sizeof(half), hipMemcpyDeviceToHost));

    float max_error_a = 0.0f;
    float max_error_b = 0.0f;
    for (int row = 0; row < N; ++row) {
        max_error_a = std::max(max_error_a,
                               std::abs(__half2float(separate_a[row]) - __half2float(dual_a[row])));
        max_error_b = std::max(max_error_b,
                               std::abs(__half2float(separate_b[row]) - __half2float(dual_b[row])));
    }
    std::cout << "Dual W1 max difference: " << max_error_a << std::endl;
    std::cout << "Dual W3 max difference: " << max_error_b << std::endl;
    assert(max_error_a < 0.05f);
    assert(max_error_b < 0.05f);

    CHECK_HIP(hipFree(device_activation));
    CHECK_HIP(hipFree(device_packed_a));
    CHECK_HIP(hipFree(device_packed_b));
    CHECK_HIP(hipFree(device_scale_a));
    CHECK_HIP(hipFree(device_scale_b));
    CHECK_HIP(hipFree(device_separate_a));
    CHECK_HIP(hipFree(device_separate_b));
    CHECK_HIP(hipFree(device_dual_a));
    CHECK_HIP(hipFree(device_dual_b));
    std::cout << "[PASS] Dual swizzled W1/W3 GEMV matches two separate launches" << std::endl;
    return 0;
}