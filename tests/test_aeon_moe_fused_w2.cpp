#include "platform/rdna3/device.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_w4a16_swizzle.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
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

constexpr int EXPERTS = 6;
constexpr int N = 4096;
constexpr int K = 2048;
constexpr int GROUPS = K / 32;
constexpr int WAVES = 8;
constexpr int RPW = 8;
constexpr int LPR = 4;
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
    std::vector<float> host_reference(N, 0.0f);
    for (int row = 0; row < N; ++row) {
        for (int expert = 0; expert < EXPERTS; ++expert) {
            float dot = 0.0f;
            for (int k = 0; k < K; ++k) {
                const uint32_t word = source_packed[expert][
                    static_cast<std::size_t>(row) * (K / 8) + k / 8];
                const int nibble = static_cast<int>((word >> ((k % 8) * 4)) & 0xFu);
                const float scale = __half2float(
                    source_scale[expert][static_cast<std::size_t>(row) * GROUPS + k / 32]);
                dot += __half2float(host_hidden[expert * K + k]) *
                       static_cast<float>(nibble - 8) * scale;
            }
            host_reference[row] += host_topk_weights[expert] * dot;
        }
    }

    half* device_hidden = nullptr;
    float* device_topk_weights = nullptr;
    float* device_output_f32 = nullptr;
    half* device_output_f16 = nullptr;
    int* device_counters = nullptr;
    CHECK_HIP(hipMalloc(&device_hidden, host_hidden.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_topk_weights, EXPERTS * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_output_f32, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_output_f16, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_counters, (N / (WAVES * RPW)) * sizeof(int)));
    CHECK_HIP(hipMemcpy(device_hidden, host_hidden.data(), host_hidden.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_topk_weights, host_topk_weights.data(), EXPERTS * sizeof(float), hipMemcpyHostToDevice));

    aeon::kernel::SwizzledW2ExpertPtrs device_weights{};
    std::array<uint32_t*, EXPERTS> device_packed{};
    std::array<half*, EXPERTS> device_scale{};
    for (int expert = 0; expert < EXPERTS; ++expert) {
        CHECK_HIP(hipMalloc(&device_packed[expert], packed_words * sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&device_scale[expert], scale_count * sizeof(half)));
        CHECK_HIP(hipMemcpy(device_packed[expert], swizzled_packed[expert].data(),
                            packed_words * sizeof(uint32_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_scale[expert], swizzled_scale[expert].data(),
                            scale_count * sizeof(half), hipMemcpyHostToDevice));
        device_weights.w2[expert] = reinterpret_cast<const uint4*>(device_packed[expert]);
        device_weights.s2[expert] = device_scale[expert];
    }

    CHECK_HIP(hipMemset(device_output_f32, 0, N * sizeof(float)));
    CHECK_HIP(hipMemset(device_output_f16, 0, N * sizeof(half)));
    CHECK_HIP(hipMemset(device_counters, 0, (N / (WAVES * RPW)) * sizeof(int)));
    aeon::kernel::dispatch_aeon_moe_fused_w2_accum<WAVES, RPW, LPR, ITERS>(
        device_hidden, device_weights, device_topk_weights,
        nullptr,
        device_output_f32, device_output_f16, device_counters,
        EXPERTS, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<half> actual(N);
    std::vector<int> counters(N / (WAVES * RPW));
    CHECK_HIP(hipMemcpy(actual.data(), device_output_f16, N * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(counters.data(), device_counters,
                        counters.size() * sizeof(int), hipMemcpyDeviceToHost));
    float max_error = 0.0f;
    for (int row = 0; row < N; ++row) {
        max_error = std::max(max_error,
            std::abs(__half2float(actual[row]) - host_reference[row]));
    }
    std::cout << "Six-expert fused W2 accumulation max error: " << max_error << std::endl;
    assert(max_error < 0.1f);
    for (int counter : counters) {
        assert(counter == 0);
    }

    CHECK_HIP(hipFree(device_hidden));
    CHECK_HIP(hipFree(device_topk_weights));
    CHECK_HIP(hipFree(device_output_f32));
    CHECK_HIP(hipFree(device_output_f16));
    CHECK_HIP(hipFree(device_counters));
    for (int expert = 0; expert < EXPERTS; ++expert) {
        CHECK_HIP(hipFree(device_packed[expert]));
        CHECK_HIP(hipFree(device_scale[expert]));
    }
    std::cout << "[PASS] Six-expert fused W2 weighted accumulation" << std::endl;
    return 0;
}