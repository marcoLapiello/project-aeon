#include "core/device.hpp"
#include "kernel/aeon_moe_fused_w13.hpp"
#include "kernel/aeon_w4a16_swizzle.hpp"
#include "kernel/aeon_w4a16_swizzled_gemv.hpp"
#include "kernel/v4_pipeline_ops.hpp"

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
constexpr int N = 2048;
constexpr int K = 4096;
constexpr int GROUPS = K / 32;
constexpr int WAVES = 8;
constexpr int RPW = 4;
constexpr int LPR = 8;
constexpr int ITERS = 16;
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
    CHECK_HIP(hipMalloc(&device_activation, K * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_gate, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_up, N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_baseline_hidden, EXPERTS * N * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_fused_hidden, EXPERTS * N * sizeof(half)));
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

    CHECK_HIP(hipMemset(device_baseline_hidden, 0, EXPERTS * N * sizeof(half)));
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

    std::vector<half> baseline(EXPERTS * N);
    std::vector<half> fused(EXPERTS * N);
    CHECK_HIP(hipMemcpy(baseline.data(), device_baseline_hidden,
                        baseline.size() * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(fused.data(), device_fused_hidden,
                        fused.size() * sizeof(half), hipMemcpyDeviceToHost));

    float max_error = 0.0f;
    for (std::size_t index = 0; index < baseline.size(); ++index) {
        max_error = std::max(max_error,
            std::abs(__half2float(baseline[index]) - __half2float(fused[index])));
    }
    std::cout << "Six-expert fused W1/W3 SwiGLU max difference: " << max_error << std::endl;
    assert(max_error < 0.05f);

    CHECK_HIP(hipFree(device_activation));
    CHECK_HIP(hipFree(device_gate));
    CHECK_HIP(hipFree(device_up));
    CHECK_HIP(hipFree(device_baseline_hidden));
    CHECK_HIP(hipFree(device_fused_hidden));
    for (int expert = 0; expert < EXPERTS; ++expert) {
        CHECK_HIP(hipFree(device_w1[expert]));
        CHECK_HIP(hipFree(device_w3[expert]));
        CHECK_HIP(hipFree(device_s1[expert]));
        CHECK_HIP(hipFree(device_s3[expert]));
    }
    std::cout << "[PASS] Six-expert fused W1/W3 plus clamped SwiGLU" << std::endl;
    return 0;
}