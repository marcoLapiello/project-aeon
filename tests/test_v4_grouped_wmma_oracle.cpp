// -----------------------------------------------------------------------------
// Tier-1 gate: the grouped (multi-token) W4A16 expert GEMM built on Wave32 WMMA,
// versus the independent fp64 expert reference.
//
// The single-token expert gate (`test_v4_expert_oracle`) already proves the
// dequantization format, the clamp rule and the composed FFN. What it cannot
// prove is the part that only exists here:
//
//   * the permutation is honoured — expert `e` must consume exactly the tokens
//     `token_indices[expert_offsets[e] .. expert_offsets[e+1])`, in that order,
//     and no others;
//   * the LDS staging transposes the swizzled layout into the WMMA B fragment's
//     `[K][N]` order. Getting that wrong is not a crash — it silently multiplies
//     the wrong weight rows, so the payload is decoded here from the format
//     definition with `swizzled_decode`, sharing no code with the staging;
//   * a group whose token count is not a multiple of the 16-row M tile pads
//     rather than reading past the end.
//
// So the fixture is deliberately lumpy: one expert with more than one M tile and
// one with fewer than one, and a permutation that is neither identity nor sorted,
// so an implementation that ignored it would have to be wrong.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_grouped_wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_HIP(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

namespace {

namespace kernel = aeon::kernel;
namespace reference = aeon::reference;

constexpr int kW1Rows = 2048;
constexpr int kW1Columns = 4096;
constexpr double kLimit = 10.0;

// The device dequantizes the weights to fp16 (one rounding per weight) and stores
// the intermediate as fp16; the reference keeps both in double. The tolerance is
// therefore a rounding bound on the kernel, not on the oracle.
constexpr double kHiddenTol = 8e-3;

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got, double tol) {
    const reference::ErrorStats stats = reference::compare(want, got, 1.0);
    if (stats.size_mismatch) {
        std::printf("  %-52s SIZE MISMATCH                 FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(stats.max_rel) && stats.max_rel <= tol;
    std::printf("  %-52s max_abs=%.3e max_rel=%.3e  %s\n", label, stats.max_abs,
                stats.max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1: grouped W4A16 WMMA expert GEMM (permuted, batched)\n";
    aeon::core::select_compute_device(true);

    reference::Rng rng(0x9A17ED0Bull);

    // ---------------------------------------------------------------------
    // Permutation. Expert 0 takes 20 tokens (two M tiles, the second partial);
    // expert 1 takes 5 (a single partial M tile). Tokens are interleaved so a
    // kernel that walked the chunk in order would produce a different answer.
    // ---------------------------------------------------------------------
    std::vector<int> token_indices;
    for (int t = 0; t < 20; ++t) token_indices.push_back(2 * t);          // even tokens
    for (int t = 0; t < 5; ++t) token_indices.push_back(2 * t + 1);       // odd tokens
    const std::vector<int> expert_offsets = {0, 20, 25};
    const int expert_count = 2;
    const int token_count = 40;   // every index the permutation names must exist
    const int hidden_tokens = 20;   // capacity per expert block

    // ---------------------------------------------------------------------
    // Two synthetic experts in the real swizzled format, encoded by the oracle
    // and decoded back by the oracle. The kernel never sees these doubles.
    // ---------------------------------------------------------------------
    std::vector<std::vector<uint8_t>> payloads(
        expert_count, std::vector<uint8_t>(aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
    kernel::SwizzledW13ExpertPtrs weights{};
    std::vector<reference::DecodedExpertWeights> decoded(expert_count);
    // The kernel reads the weights from device memory, so each payload gets its
    // own upload and the pointer table points at the copies, exactly as the
    // runtime's VRAM pool slots do.
    std::vector<uint8_t*> device_payloads(expert_count, nullptr);

    for (int e = 0; e < expert_count; ++e) {
        const std::vector<double> w1 = rng.vector_filled(kW1Rows * kW1Columns, 0.25);
        const std::vector<double> w3 = rng.vector_filled(kW1Rows * kW1Columns, 0.25);
        reference::swizzled_encode(payloads[e].data(), reference::SwizzledKind::W1, w1);
        reference::swizzled_encode(payloads[e].data(), reference::SwizzledKind::W3, w3);

        // W2 is part of the payload layout; this gate does not consume it, so it
        // is left as the zeroed region the caller allocated.
        decoded[e] = reference::decode_expert_weights(payloads[e].data());

        CHECK_HIP(hipMalloc(&device_payloads[e], payloads[e].size()));
        CHECK_HIP(hipMemcpy(device_payloads[e], payloads[e].data(), payloads[e].size(),
                            hipMemcpyHostToDevice));

        const uint8_t* base = device_payloads[e];
        weights.w1[e] = reinterpret_cast<const uint4*>(
            base + aeon::core::AEON_W1_PACKED_OFFSET);
        weights.s1[e] = reinterpret_cast<const half*>(
            base + aeon::core::AEON_W1_SCALE_OFFSET);
        weights.w3[e] = reinterpret_cast<const uint4*>(
            base + aeon::core::AEON_W3_PACKED_OFFSET);
        weights.s3[e] = reinterpret_cast<const half*>(
            base + aeon::core::AEON_W3_SCALE_OFFSET);
    }

    // ---------------------------------------------------------------------
    // Activations, one row per chunk token.
    // ---------------------------------------------------------------------
    std::vector<half> host_activation(
        static_cast<size_t>(token_count) * kW1Columns);
    for (int t = 0; t < token_count; ++t) {
        for (int k = 0; k < kW1Columns; ++k) {
            host_activation[static_cast<size_t>(t) * kW1Columns + k] =
                __float2half(static_cast<float>(rng.symmetric(0.5)));
        }
    }

    // ---------------------------------------------------------------------
    // Reference: gate/up and the clamped SwiGLU, per (expert, permutation row).
    // ---------------------------------------------------------------------
    std::vector<std::vector<double>> want(
        expert_count, std::vector<double>(
                          static_cast<size_t>(hidden_tokens) * kW1Rows, 0.0));
    long clamped_count = 0;
    for (int e = 0; e < expert_count; ++e) {
        for (size_t j = 0; j < static_cast<size_t>(expert_offsets[e + 1] - expert_offsets[e]); ++j) {
            const int token = token_indices[expert_offsets[e] + j];
            std::vector<double> activation(kW1Columns);
            for (int k = 0; k < kW1Columns; ++k) {
                activation[k] = static_cast<double>(
                    __half2float(host_activation[static_cast<size_t>(token) * kW1Columns + k]));
            }

            const std::vector<double> gate = reference::matvec(
                kW1Rows, kW1Columns, activation,
                [&](size_t o, size_t i) { return decoded[e].w1[o * kW1Columns + i]; });
            const std::vector<double> up = reference::matvec(
                kW1Rows, kW1Columns, activation,
                [&](size_t o, size_t i) { return decoded[e].w3[o * kW1Columns + i]; });

            for (int n = 0; n < kW1Rows; ++n) {
                if (std::fabs(gate[n]) > kLimit ||
                    std::fabs(up[n]) > kLimit) {
                    ++clamped_count;
                }
                want[e][j * kW1Rows + n] =
                    reference::clamped_swiglu(gate[n], up[n], kLimit);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Device run.
    // ---------------------------------------------------------------------
    const size_t hidden_elements =
        static_cast<size_t>(expert_count) * hidden_tokens * kW1Rows;

    half* d_activation = nullptr;
    half* d_hidden = nullptr;
    int* d_offsets = nullptr;
    int* d_indices = nullptr;
    CHECK_HIP(hipMalloc(&d_activation, host_activation.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_hidden, hidden_elements * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_offsets, expert_offsets.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_indices, token_indices.size() * sizeof(int)));
    CHECK_HIP(hipMemcpy(d_activation, host_activation.data(),
                        host_activation.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_offsets, expert_offsets.data(),
                        expert_offsets.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_indices, token_indices.data(),
                        token_indices.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_hidden, 0, hidden_elements * sizeof(half)));

    kernel::dispatch_aeon_moe_grouped_w13_swiglu_wmma<4, 4, 8>(
        d_activation, d_offsets, d_indices, weights, d_hidden, expert_count,
        hidden_tokens, kW1Rows, kW1Columns, static_cast<float>(kLimit));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<half> host_hidden(hidden_elements);
    CHECK_HIP(hipMemcpy(host_hidden.data(), d_hidden, hidden_elements * sizeof(half),
                        hipMemcpyDeviceToHost));

    bool ok = true;
    for (int e = 0; e < expert_count; ++e) {
        std::vector<double> got(static_cast<size_t>(hidden_tokens) * kW1Rows);
        for (size_t i = 0; i < got.size(); ++i) {
            got[i] = static_cast<double>(__half2float(host_hidden[
                (static_cast<size_t>(e) * hidden_tokens * kW1Rows) + i]));
        }
        char label[96];
        std::snprintf(label, sizeof(label),
                      "expert %d, %d tokens (hidden, clamped SwiGLU)",
                      e, expert_offsets[e + 1] - expert_offsets[e]);
        ok &= report(label, want[e], got, kHiddenTol);
    }

    std::printf("  %-52s %-22s %s\n", "clamp fires in this fixture",
                (std::to_string(clamped_count) + " values").c_str(),
                clamped_count > 0 ? "PASS" : "FAIL");
    ok &= clamped_count > 0;

    CHECK_HIP(hipFree(d_activation));
    CHECK_HIP(hipFree(d_hidden));
    CHECK_HIP(hipFree(d_offsets));
    CHECK_HIP(hipFree(d_indices));
    for (uint8_t* payload : device_payloads) {
        CHECK_HIP(hipFree(payload));
    }

    if (!ok) {
        std::cout << "[FAIL] grouped W4A16 WMMA expert GEMM disagrees with the oracle\n";
        return 1;
    }
    std::cout << "[PASS] grouped W4A16 WMMA expert GEMM matches the oracle\n";
    return 0;
}
