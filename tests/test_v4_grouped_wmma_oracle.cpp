// -----------------------------------------------------------------------------
// Tier-1 gate: the grouped (multi-token) W4A16 expert pair built on Wave32 WMMA,
// versus the independent fp64 expert reference.
//
// The single-token expert gate (`test_v4_expert_oracle`) already proves the
// dequantization format, the clamp rule and the composed FFN. What it cannot
// prove is the part that only exists here:
//
//   * the permutation is honoured — expert `e` must consume exactly the tokens
//     `token_indices[expert_offsets[e] .. expert_offsets[e+1])`, in that order,
//     and no others;
//   * token and draw are separate indexings. A draw is a (token, slot) pair, so a
//     chunk token and its output row are different numbers; a kernel that used one
//     where the other belongs scatters contributions across draws;
//   * the LDS staging transposes the swizzled layout into the WMMA B fragment's
//     `[K][N]` order. Getting that wrong is not a crash — it silently multiplies
//     the wrong weight rows, so the payload is decoded here from the format
//     definition with `swizzled_decode`, sharing no code with the staging;
//   * a group whose token count is not a multiple of the 16-row M tile pads
//     rather than reading past the end.
//
// So the fixture is deliberately lumpy: one expert with more than one M tile and
// one with fewer than one, a permutation that is neither identity nor sorted, and
// two routed slots per token so the two indexings cannot coincide.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"
#include "architecture/deepseek_v4/kernels/moe_grouped_dispatch.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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
constexpr int kW2Rows = 4096;
constexpr int kW2Columns = 2048;
constexpr double kLimit = 10.0;

// The device dequantizes the weights to fp16 and stores the intermediate as
// fp16; the reference keeps both in double. The tolerances are therefore rounding
// bounds on the kernel, not on the oracle.
//
// Every dequantized weight is a 4-bit integer times a power-of-two scale, so it is
// exactly representable in fp16 — the *only* device rounding in the whole pair is
// the `hidden` store. That is why the composed bound can sit at the stated ε
// instead of being loosened to absorb a chain of roundings: a value materially
// above it means the arithmetic is wrong, not that it drifted.
constexpr double kHiddenTol = 8e-3;
constexpr double kFfnTol = 1e-3;

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
    std::cout << "[Gate] Tier-1: grouped W4A16 WMMA expert pair (permuted, batched)\n";
    aeon::core::select_compute_device(true);

    reference::Rng rng(0x9A17ED0Bull);

    // ---------------------------------------------------------------------
    // Permutation. Two routed slots per token, so a chunk token index and its
    // draw row `token * kSlots + slot` are different numbers — a kernel that used
    // one where the other belongs scatters contributions across draws.
    //
    // Expert 0 takes 20 tokens at slot 0 (one full M tile and a partial one);
    // expert 1 takes 5 tokens at slot 1 (a single partial tile), listed in reverse
    // so an order-only walk of the chunk produces a different answer.
    // ---------------------------------------------------------------------
    constexpr int kSlots = 2;
    std::vector<int> token_indices;
    std::vector<int> draw_indices;
    std::vector<float> draw_weights;
    for (int t = 0; t < 20; ++t) {                       // expert 0's group
        token_indices.push_back(t);
        draw_indices.push_back(t * kSlots + 0);
        draw_weights.push_back(0.5f + 0.25f * static_cast<float>(t % 3));
    }
    for (int t = 0; t < 5; ++t) {                        // expert 1's group
        token_indices.push_back(4 - t);
        draw_indices.push_back((4 - t) * kSlots + 1);
        draw_weights.push_back(0.75f - 0.25f * static_cast<float>(t % 2));
    }
    const std::vector<int> expert_offsets = {0, 20, 25};
    const int expert_count = 2;
    const int token_count = 24;
    const int draw_count = token_count * kSlots;
    const int hidden_tokens = 20;   // capacity per expert block

    // ---------------------------------------------------------------------
    // Two synthetic experts in the real swizzled format, encoded by the oracle
    // and decoded back by the oracle. The kernel never sees these doubles.
    // ---------------------------------------------------------------------
    std::vector<std::vector<uint8_t>> payloads(
        expert_count, std::vector<uint8_t>(aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
    kernel::SwizzledW13ExpertPtrs w13_weights{};
    kernel::SwizzledW2ExpertPtrs w2_weights{};
    std::vector<reference::DecodedExpertWeights> decoded(expert_count);
    // The kernel reads the weights from device memory, so each payload gets its
    // own upload and the pointer table points at the copies, exactly as the
    // runtime's VRAM pool slots do.
    std::vector<uint8_t*> device_payloads(expert_count, nullptr);

    for (int e = 0; e < expert_count; ++e) {
        const std::vector<double> w1 = rng.vector_filled(kW1Rows * kW1Columns, 0.25);
        const std::vector<double> w3 = rng.vector_filled(kW1Rows * kW1Columns, 0.25);
        const std::vector<double> w2 = rng.vector_filled(kW2Rows * kW2Columns, 0.25);
        reference::swizzled_encode(payloads[e].data(), reference::SwizzledKind::W1, w1);
        reference::swizzled_encode(payloads[e].data(), reference::SwizzledKind::W3, w3);
        reference::swizzled_encode(payloads[e].data(), reference::SwizzledKind::W2, w2);

        decoded[e] = reference::decode_expert_weights(payloads[e].data());

        CHECK_HIP(hipMalloc(&device_payloads[e], payloads[e].size()));
        CHECK_HIP(hipMemcpy(device_payloads[e], payloads[e].data(), payloads[e].size(),
                            hipMemcpyHostToDevice));

        const uint8_t* base = device_payloads[e];
        w13_weights.w1[e] = reinterpret_cast<const uint4*>(
            base + aeon::core::AEON_W1_PACKED_OFFSET);
        w13_weights.s1[e] = reinterpret_cast<const half*>(
            base + aeon::core::AEON_W1_SCALE_OFFSET);
        w13_weights.w3[e] = reinterpret_cast<const uint4*>(
            base + aeon::core::AEON_W3_PACKED_OFFSET);
        w13_weights.s3[e] = reinterpret_cast<const half*>(
            base + aeon::core::AEON_W3_SCALE_OFFSET);
        w2_weights.w2[e] = reinterpret_cast<const uint4*>(
            base + aeon::core::AEON_W2_PACKED_OFFSET);
        w2_weights.s2[e] = reinterpret_cast<const half*>(
            base + aeon::core::AEON_W2_SCALE_OFFSET);
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
    // Reference: the composed routed FFN per permutation position. `want_hidden`
    // is checked at the gate kernel's output so a failure localizes to one half of
    // the pair; `want_draw` is the weighted W2 result at the draw's own row.
    // ---------------------------------------------------------------------
    std::vector<std::vector<double>> want_hidden(
        expert_count, std::vector<double>(
                          static_cast<size_t>(hidden_tokens) * kW1Rows, 0.0));
    std::vector<std::vector<double>> want_draw(
        static_cast<size_t>(draw_count), std::vector<double>(kW2Rows, 0.0));

    long clamped_count = 0;
    for (int e = 0; e < expert_count; ++e) {
        const int first = expert_offsets[e];
        const int count = expert_offsets[e + 1] - first;
        for (int j = 0; j < count; ++j) {
            const int position = first + j;
            const int token = token_indices[position];
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
                if (std::fabs(gate[n]) > kLimit || std::fabs(up[n]) > kLimit) {
                    ++clamped_count;
                }
                want_hidden[e][j * kW1Rows + n] =
                    reference::clamped_swiglu(gate[n], up[n], kLimit);
            }

            // W2 consumes the fp16-rounded intermediate, because that is exactly
            // what the gate kernel stores. The oracle deliberately keeps `hidden`
            // in double, so the rounding is applied here, in the gate, where it is
            // visible — the alternative of folding it into the oracle would hide a
            // real discrepancy behind the reference.
            std::vector<double> hidden_fp16(kW1Rows);
            for (int n = 0; n < kW1Rows; ++n) {
                hidden_fp16[n] = static_cast<double>(__half2float(__float2half(
                    static_cast<float>(want_hidden[e][j * kW1Rows + n]))));
            }
            const std::vector<double> out = reference::matvec(
                kW2Rows, kW2Columns, hidden_fp16,
                [&](size_t o, size_t i) { return decoded[e].w2[o * kW2Columns + i]; });

            const size_t draw = static_cast<size_t>(draw_indices[position]);
            for (int n = 0; n < kW2Rows; ++n) {
                want_draw[draw][n] = static_cast<double>(draw_weights[position]) * out[n];
            }
        }
    }

    // ---------------------------------------------------------------------
    // Device run.
    // ---------------------------------------------------------------------
    const size_t hidden_elements =
        static_cast<size_t>(expert_count) * hidden_tokens * kW1Rows;
    const size_t contrib_elements = static_cast<size_t>(draw_count) * kW2Rows;

    half* d_activation = nullptr;
    half* d_hidden = nullptr;
    float* d_contrib = nullptr;
    int* d_offsets = nullptr;
    int* d_indices = nullptr;
    int* d_draws = nullptr;
    float* d_draw_weights = nullptr;
    CHECK_HIP(hipMalloc(&d_activation, host_activation.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_hidden, hidden_elements * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_contrib, contrib_elements * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_offsets, expert_offsets.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_indices, token_indices.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_draws, draw_indices.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_draw_weights, draw_weights.size() * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_activation, host_activation.data(),
                        host_activation.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_offsets, expert_offsets.data(),
                        expert_offsets.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_indices, token_indices.data(),
                        token_indices.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_draws, draw_indices.data(),
                        draw_indices.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_draw_weights, draw_weights.data(),
                        draw_weights.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_hidden, 0, hidden_elements * sizeof(half)));
    CHECK_HIP(hipMemset(d_contrib, 0, contrib_elements * sizeof(float)));

    kernel::dispatch_aeon_moe_grouped_w13_swiglu_wmma<4, 4, 8, kernel::kMoeGroupedMTiles>(
        d_activation, d_offsets, d_indices, w13_weights, d_hidden, expert_count,
        hidden_tokens, kW1Rows, kW1Columns, static_cast<float>(kLimit));
    kernel::dispatch_aeon_moe_grouped_w2_wmma<4, 8, 4, kernel::kMoeGroupedMTiles>(
        d_hidden, d_offsets, d_draws, d_draw_weights, w2_weights, d_contrib,
        expert_count, hidden_tokens, kW2Rows, kW2Columns);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<half> host_hidden(hidden_elements);
    CHECK_HIP(hipMemcpy(host_hidden.data(), d_hidden, hidden_elements * sizeof(half),
                        hipMemcpyDeviceToHost));
    std::vector<float> host_contrib(contrib_elements);
    CHECK_HIP(hipMemcpy(host_contrib.data(), d_contrib, contrib_elements * sizeof(float),
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
        ok &= report(label, want_hidden[e], got, kHiddenTol);
    }

    // The draw rows are compared as one flat block: every draw carries its own
    // routing weight and its own expert, so a mixed-up index shows as a moved row
    // rather than as a scale error.
    std::vector<double> want_flat;
    want_flat.reserve(contrib_elements);
    for (const auto& row : want_draw) {
        want_flat.insert(want_flat.end(), row.begin(), row.end());
    }
    ok &= report("routed output, every draw (weighted fp32 contribution)", want_flat,
                 std::vector<double>(host_contrib.begin(), host_contrib.end()), kFfnTol);

    std::printf("  %-52s %-22s %s\n", "clamp fires in this fixture",
                (std::to_string(clamped_count) + " values").c_str(),
                clamped_count > 0 ? "PASS" : "FAIL");
    ok &= clamped_count > 0;

    // ---------------------------------------------------------------------
    // Strided A-read. The same activation rows, each padded out to a wider pitch
    // and read with `activation_stride`, must produce the identical hidden output.
    // The batch workspace pads every token to its own 16-row tile, so the grouped
    // gate is called with a tile pitch rather than a compact `[T, K]` batch; this
    // proves the A-read honours that pitch instead of assuming compactness.
    // ---------------------------------------------------------------------
    {
        const int stride = kW1Columns + 64;   // +128 B keeps each row 16-byte aligned
        std::vector<half> padded(static_cast<size_t>(token_count) * stride,
                                 __float2half(0.0f));
        for (int t = 0; t < token_count; ++t) {
            std::copy(host_activation.begin() + static_cast<size_t>(t) * kW1Columns,
                      host_activation.begin() + static_cast<size_t>(t + 1) * kW1Columns,
                      padded.begin() + static_cast<size_t>(t) * stride);
        }
        half* d_padded = nullptr;
        half* d_hidden_strided = nullptr;
        CHECK_HIP(hipMalloc(&d_padded, padded.size() * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_hidden_strided, hidden_elements * sizeof(half)));
        CHECK_HIP(hipMemcpy(d_padded, padded.data(), padded.size() * sizeof(half),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(d_hidden_strided, 0, hidden_elements * sizeof(half)));

        kernel::dispatch_aeon_moe_grouped_w13_swiglu_wmma<4, 4, 8, kernel::kMoeGroupedMTiles>(
            d_padded, d_offsets, d_indices, w13_weights, d_hidden_strided, expert_count,
            hidden_tokens, kW1Rows, kW1Columns, static_cast<float>(kLimit), stride);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<half> host_strided(hidden_elements);
        CHECK_HIP(hipMemcpy(host_strided.data(), d_hidden_strided,
                            hidden_elements * sizeof(half), hipMemcpyDeviceToHost));
        int differing = 0;
        for (size_t i = 0; i < hidden_elements; ++i) {
            if (__half2float(host_strided[i]) != __half2float(host_hidden[i])) ++differing;
        }
        std::printf("  %-52s %-22s %s\n", "strided A-read equals the compact A-read",
                    (differing == 0 ? "bit-identical" : std::to_string(differing) + " differ")
                        .c_str(),
                    differing == 0 ? "PASS" : "FAIL");
        ok &= differing == 0;

        CHECK_HIP(hipFree(d_padded));
        CHECK_HIP(hipFree(d_hidden_strided));
    }

    CHECK_HIP(hipFree(d_activation));
    CHECK_HIP(hipFree(d_hidden));
    CHECK_HIP(hipFree(d_contrib));
    CHECK_HIP(hipFree(d_offsets));
    CHECK_HIP(hipFree(d_indices));
    CHECK_HIP(hipFree(d_draws));
    CHECK_HIP(hipFree(d_draw_weights));
    for (uint8_t* payload : device_payloads) {
        CHECK_HIP(hipFree(payload));
    }

    if (!ok) {
        std::cout << "[FAIL] grouped W4A16 WMMA expert pair disagrees with the oracle\n";
        return 1;
    }
    std::cout << "[PASS] grouped W4A16 WMMA expert pair matches the oracle\n";
    return 0;
}
