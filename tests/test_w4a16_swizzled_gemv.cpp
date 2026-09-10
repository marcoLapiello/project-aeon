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

struct ShapeCase {
    const char* name;
    int N;
    int K;
    aeon::SwizzleCfg cfg;
};

uint32_t make_source_word(std::size_t word_index) {
    uint32_t word = 0;
    for (int nibble = 0; nibble < 8; ++nibble) {
        const uint32_t value = static_cast<uint32_t>((word_index * 3 + nibble * 5 + 1) % 16);
        word |= value << (4 * nibble);
    }
    return word;
}

template <int RPW, int LPR>
void verify_shape(const ShapeCase& shape) {
    constexpr int WAVES = 8;
    constexpr int ITERS = 16;
    const int groups = shape.K / 32;
    const std::size_t packed_words = static_cast<std::size_t>(shape.N) * shape.K / 8;
    const std::size_t scale_count = static_cast<std::size_t>(shape.N) * groups;

    std::vector<half> host_activation(shape.K);
    std::vector<uint32_t> host_source_packed(packed_words);
    std::vector<half> host_source_scale(scale_count);
    std::vector<uint32_t> host_swizzled_packed(packed_words);
    std::vector<half> host_swizzled_scale(scale_count);
    std::vector<half> host_output(shape.N);
    std::vector<float> host_reference(shape.N, 0.0f);

    for (int k = 0; k < shape.K; ++k) {
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
        shape.N, shape.K, shape.cfg);

    const int source_words_per_row = shape.K / 8;
    for (int row = 0; row < shape.N; ++row) {
        float sum = 0.0f;
        for (int k = 0; k < shape.K; ++k) {
            const uint32_t word = host_source_packed[
                static_cast<std::size_t>(row) * source_words_per_row + k / 8];
            const int nibble = static_cast<int>((word >> ((k % 8) * 4)) & 0xFu);
            const float scale = __half2float(
                host_source_scale[static_cast<std::size_t>(row) * groups + k / 32]);
            sum += __half2float(host_activation[k]) * static_cast<float>(nibble - 8) * scale;
        }
        host_reference[row] = sum;
    }

    half* device_activation = nullptr;
    uint32_t* device_packed = nullptr;
    half* device_scale = nullptr;
    half* device_output = nullptr;
    CHECK_HIP(hipMalloc(&device_activation, host_activation.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_packed, host_swizzled_packed.size() * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_scale, host_swizzled_scale.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_output, host_output.size() * sizeof(half)));

    CHECK_HIP(hipMemcpy(device_activation, host_activation.data(),
                        host_activation.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_packed, host_swizzled_packed.data(),
                        host_swizzled_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_scale, host_swizzled_scale.data(),
                        host_swizzled_scale.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(device_output, 0, host_output.size() * sizeof(half)));

    aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<WAVES, RPW, LPR, ITERS>(
        device_activation, device_packed, device_scale, device_output,
        shape.N, shape.K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(host_output.data(), device_output,
                        host_output.size() * sizeof(half), hipMemcpyDeviceToHost));

    float max_error = 0.0f;
    for (int row = 0; row < shape.N; ++row) {
        max_error = std::max(max_error,
                             std::abs(__half2float(host_output[row]) - host_reference[row]));
    }
    std::cout << shape.name << " swizzled GEMV max error: " << max_error << std::endl;
    assert(max_error < 0.05f);

    CHECK_HIP(hipFree(device_activation));
    CHECK_HIP(hipFree(device_packed));
    CHECK_HIP(hipFree(device_scale));
    CHECK_HIP(hipFree(device_output));
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);
    verify_shape<4, 8>({"W1/W3", 2048, 4096, aeon::kCfgW13});
    verify_shape<8, 4>({"W2", 4096, 2048, aeon::kCfgW2});
    std::cout << "[PASS] Swizzled W4A16 GEMV matches the source-layout CPU reference" << std::endl;
    return 0;
}