// -----------------------------------------------------------------------------
// Tier-1 gate: routed expert — fused INT4 dequantization, matmul, and clamped
// SwiGLU (Steps 2.10.2 / 2.10.3) — versus an independent fp64 oracle.
//
// This is the largest op in the graph and the one with the most ways to be
// silently wrong, so the gate is split by *rule* rather than by kernel:
//
//   A. the dequantization FORMAT   — zero point, nibble permutation, scale layout
//   B. the clamped SwiGLU RULE     — the clamp is asymmetric
//   C. the composed routed FFN     — fused W13 + SwiGLU + W2
//   D. a real artifact payload     — anchors A/B/C to bytes the converter wrote
//   E. accumulation order          — a measurement, not a certification
//
// Sections A and B compute the deliberately-wrong variants (`SwizzledDecodeOptions`,
// `ClampMode`) and require both that the wrong variant differs materially and that
// the device follows the right one. Without that, a pass would show only that two
// implementations of the same (possibly wrong) rule agree.
//
// Section D matters because sections A-C build their payloads with an encoder
// written from the same format description as the decoder. That pair is
// self-consistent by construction; only real artifact bytes prove the description
// matches what the converter actually wrote. (The legacy `test_v4_real_expert_parity`
// does that check too, against `v4_int4_reference.hpp`; it is gated off by default
// and does not test the clamp rule at all, so this gate does not rely on it.)
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_w4a16_swizzled_gemv.hpp"
#include "infrastructure/core/aeon_loader.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
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

using aeon::reference::ClampMode;
using aeon::reference::ErrorStats;
using aeon::reference::SwizzledDecodeOptions;
using aeon::reference::SwizzledKind;

constexpr int kWaves = 8;
constexpr int kW1Rows = 2048, kW1Columns = 4096, kW1Rpw = 4, kW1Lpr = 8;
constexpr int kW2Rows = 4096, kW2Columns = 2048, kW2Rpw = 8, kW2Lpr = 4;
constexpr int kIterations = 16;
constexpr double kLimit = 10.0;

// Fraction-of-peak tolerances. The device accumulates in fp32 in a different
// order and stores `hidden` as fp16 between the two halves, so these are
// rounding bounds, not the oracle's own error.
constexpr double kGemvTol = 4e-3;
constexpr double kHiddenTol = 6e-3;
constexpr double kFfnTol = 6e-3;

std::vector<double> widen(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(__half2float(v[i]));
    return out;
}

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got, double tol) {
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-54s SIZE MISMATCH                 FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(s.max_rel) && s.max_rel <= tol;
    std::printf("  %-54s max_abs=%.3e max_rel=%.3e  %s\n",
                label, s.max_abs, s.max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-54s %-22s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

double peak_of(const std::vector<double>& v) {
    double peak = 0.0;
    for (double x : v) peak = std::fmax(peak, std::fabs(x));
    return peak;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: routed expert (dequant + matmul + clamped SwiGLU)\n";
    aeon::core::select_compute_device(true);

    bool ok = true;
    aeon::reference::Rng gen(0x0E7E1217ull);

    // ---------------------------------------------------------------------
    // Fixture: a synthetic expert payload in the real swizzled format.
    // ---------------------------------------------------------------------
    const size_t kPayloadBytes = aeon::core::AEON_SWIZZLED_EXPERT_BYTES;
    std::vector<uint8_t> payload(kPayloadBytes, 0);

    // Source weights, kept in double so section A can measure how faithfully the
    // 4-bit encoding represents them.
    std::vector<double> src_w1(kW1Rows * kW1Columns), src_w2(kW2Rows * kW2Columns),
        src_w3(kW1Rows * kW1Columns);
    for (double& v : src_w1) v = gen.symmetric(0.06);
    for (double& v : src_w3) v = gen.symmetric(0.06);
    for (double& v : src_w2) v = gen.symmetric(0.06);

    aeon::reference::swizzled_encode(payload.data(), SwizzledKind::W1, src_w1);
    aeon::reference::swizzled_encode(payload.data(), SwizzledKind::W2, src_w2);
    aeon::reference::swizzled_encode(payload.data(), SwizzledKind::W3, src_w3);

    // =====================================================================
    // 0. Oracle self-check
    // =====================================================================
    // The oracle is a reference only if it is what it claims to be. Encoding a
    // matrix and decoding it back must reproduce the matrix to within half a
    // quantization step — the guarantee of round-to-nearest. A wrong oracle
    // would fail here rather than silently certifying a wrong kernel.
    std::cout << "\n--- 0. oracle self-check (encode -> decode) ---\n";
    {
        const std::vector<double> decoded =
            aeon::reference::swizzled_decode(payload.data(), SwizzledKind::W1);
        double worst = 0.0;
        for (size_t i = 0; i < src_w1.size(); ++i) {
            worst = std::fmax(worst, std::fabs(decoded[i] - src_w1[i]));
        }
        // Half a step: the scale is the smallest power of two >= peak/7, so one
        // step is `scale` and round-to-nearest errs by at most `scale/2`. The
        // bound used here is the looser `peak/7`, which is what the encoder
        // guarantees for every column in the row.
        const double source_peak = peak_of(src_w1);
        ok &= check("decoded weights reproduce the source within one step",
                    worst <= source_peak / 7.0,
                    "max |w - decode(encode(w))| = " + std::to_string(worst) +
                        " vs step " + std::to_string(source_peak / 7.0));
    }

    // ---------------------------------------------------------------------
    // Device buffers.
    // ---------------------------------------------------------------------
    __half* d_payload = nullptr;   // W1/W2/W3 regions, as the kernel sees them
    __half* d_x = nullptr;         // [4096] activation
    __half* d_gemv_out = nullptr;  // [2048]
    __half* d_hidden = nullptr;    // [2048] post-SwiGLU
    __half* d_ffn_out = nullptr;   // [4096]
    float* d_accum = nullptr;      // [4096]
    float* d_topk = nullptr;
    int* d_counters = nullptr;
    __half* d_gate = nullptr;
    __half* d_up = nullptr;
    __half* d_swiglu = nullptr;

    const size_t kAccumBlocks = static_cast<size_t>(kW2Rows / (kWaves * kW2Rpw));
    CHECK_HIP(hipMalloc(&d_payload, kPayloadBytes));
    CHECK_HIP(hipMalloc(&d_x, kW1Columns * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_gemv_out, kW1Rows * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_hidden, kW1Rows * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_ffn_out, kW2Rows * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_accum, kW2Rows * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_topk, aeon::kernel::kAeonSwizzledMaxExperts * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_counters, kAccumBlocks * sizeof(int)));

    const size_t kSwigluCount = 64;
    CHECK_HIP(hipMalloc(&d_gate, kSwigluCount * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_up, kSwigluCount * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_swiglu, kSwigluCount * sizeof(__half)));

    CHECK_HIP(hipMemcpy(d_payload, payload.data(), kPayloadBytes, hipMemcpyHostToDevice));

    // Activation magnitudes: a moderate one where the clamp is mostly inert, and
    // a large one that drives the pre-activations past the limit, so section C
    // exercises the clamp rather than merely passing through it.
    const std::vector<double> x_moderate = gen.vector_filled(kW1Columns, 1.0);
    std::vector<double> x_large(kW1Columns);
    for (size_t i = 0; i < x_large.size(); ++i) x_large[i] = x_moderate[i] * 8.0;

    auto to_half = [](const std::vector<double>& v) {
        std::vector<__half> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = __float2half(static_cast<float>(v[i]));
        return out;
    };

    // Device W1 GEMV: `W1 @ x -> [2048]`, straight through the swizzled format.
    auto run_w1 = [&](const std::vector<__half>& x) {
        CHECK_HIP(hipMemcpy(d_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice));
        aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<kWaves, kW1Rpw, kW1Lpr, kIterations>(
            d_x,
            reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(d_payload) +
                                              aeon::core::AEON_W1_PACKED_OFFSET),
            reinterpret_cast<const half*>(reinterpret_cast<const uint8_t*>(d_payload) +
                                          aeon::core::AEON_W1_SCALE_OFFSET),
            d_gemv_out, kW1Rows, kW1Columns);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<__half> out(kW1Rows);
        CHECK_HIP(hipMemcpy(out.data(), d_gemv_out, out.size() * sizeof(__half),
                            hipMemcpyDeviceToHost));
        return out;
    };

    // =====================================================================
    // A. The dequantization format
    // =====================================================================
    std::cout << "\n--- A. dequantization: zero point, nibble order, scale layout ---\n";
    {
        const std::vector<double> decoded =
            aeon::reference::swizzled_decode(payload.data(), SwizzledKind::W1);
        const std::vector<double> x_d = [&] {
            const std::vector<__half> h = to_half(x_moderate);
            return widen(h);
        }();

        const std::vector<double> want = aeon::reference::matvec(
            kW1Rows, kW1Columns, x_d,
            [&](size_t o, size_t i) { return decoded[o * kW1Columns + i]; });

        const std::vector<double> got = widen(run_w1(to_half(x_moderate)));
        ok &= report("W1 GEMV vs decoder (signed zero point, swizzled nibbles)",
                     want, got, kGemvTol);

        // (1) The zero point must be signed. A reader that treats the nibble as
        //     an unsigned magnitude still computes a plausible dot product.
        {
            SwizzledDecodeOptions wrong;
            wrong.signed_zero_point = false;
            const std::vector<double> bad =
                aeon::reference::swizzled_decode(payload.data(), SwizzledKind::W1, wrong);
            const std::vector<double> bad_out = aeon::reference::matvec(
                kW1Rows, kW1Columns, x_d,
                [&](size_t o, size_t i) { return bad[o * kW1Columns + i]; });
            const double d = aeon::reference::compare(want, bad_out, 1.0).max_rel;
            const double to_wrong = aeon::reference::compare(bad_out, got, 1.0).max_rel;
            ok &= check("the -8 zero point is load-bearing", d > 1e-1 && to_wrong > 1e-1,
                        "unsigned read differs by " + std::to_string(d));
        }

        // (2) The nibbles are permuted within each word. Reading them in
        //     column order is the obvious implementation and is wrong.
        {
            SwizzledDecodeOptions wrong;
            wrong.permute_nibbles = false;
            const std::vector<double> bad =
                aeon::reference::swizzled_decode(payload.data(), SwizzledKind::W1, wrong);
            const std::vector<double> bad_out = aeon::reference::matvec(
                kW1Rows, kW1Columns, x_d,
                [&](size_t o, size_t i) { return bad[o * kW1Columns + i]; });
            const double d = aeon::reference::compare(want, bad_out, 1.0).max_rel;
            const double to_wrong = aeon::reference::compare(bad_out, got, 1.0).max_rel;
            ok &= check("the nibble permutation is load-bearing", d > 1e-1 && to_wrong > 1e-1,
                        "unpermuted read differs by " + std::to_string(d));
        }

        // (3) The scale layout: one fp16 scale per 32-column group, shared by
        //     4096/32 = 128 groups per row. Confirmed structurally, because a
        //     wrong grouping would already have failed (1) or (2) numerically.
        const aeon::reference::SwizzledShape shape =
            aeon::reference::swizzled_shape(SwizzledKind::W1);
        ok &= check("one scale per 32-column group",
                    shape.weight_count() / 32 == 2048u * 128u,
                    "2048 rows x 128 groups = " + std::to_string(shape.weight_count() / 32));
        ok &= check("W1 shape is 2048 x 4096 with 4x8 rows/lanes per wave",
                    shape.rows == kW1Rows && shape.columns == kW1Columns &&
                        shape.rows_per_wave == kW1Rpw && shape.lanes_per_row == kW1Lpr,
                    "2048 x 4096, RPW 4, LPR 8, ITERS " +
                        std::to_string(shape.iterations()));
    }

    // =====================================================================
    // B. The clamped SwiGLU rule — the clamp is asymmetric
    // =====================================================================
    std::cout << "\n--- B. clamped SwiGLU: gate max-only, up both sides ---\n";
    {
        // Values chosen to straddle +-limit on both branches, including the case
        // that separates the asymmetric rule from the symmetric one: a gate far
        // BELOW -limit. Asymmetric leaves it alone (silu(-40) ~ -1.7e-16);
        // symmetric clamps it to -10 (silu(-10) = -4.5e-4). Both are tiny, but
        // the ratio is 2.6e12, so the difference is unambiguous.
        const double kProbe[8] = {-40.0, -12.0, -5.0, 0.0, 5.0, 12.0, 40.0, 3.0};
        std::vector<double> gate_d(kSwigluCount), up_d(kSwigluCount);
        for (size_t i = 0; i < kSwigluCount; ++i) {
            gate_d[i] = kProbe[i % 8];
            up_d[i] = kProbe[(i / 8 + 3) % 8];
        }

        std::vector<__half> gate_h(kSwigluCount), up_h(kSwigluCount), out_h(kSwigluCount);
        for (size_t i = 0; i < kSwigluCount; ++i) {
            gate_h[i] = __float2half(static_cast<float>(gate_d[i]));
            up_h[i] = __float2half(static_cast<float>(up_d[i]));
        }
        CHECK_HIP(hipMemcpy(d_gate, gate_h.data(), kSwigluCount * sizeof(__half),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_up, up_h.data(), kSwigluCount * sizeof(__half),
                            hipMemcpyHostToDevice));
        aeon::kernel::v4_pipeline_swiglu_clamp_kernel<<<
            dim3(static_cast<unsigned>((kSwigluCount + 63) / 64)), dim3(64)>>>(
            d_gate, d_up, d_swiglu, static_cast<int>(kSwigluCount),
            static_cast<float>(kLimit));
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(out_h.data(), d_swiglu, kSwigluCount * sizeof(__half),
                            hipMemcpyDeviceToHost));

        // The oracle takes the widened probe values, so the comparison isolates
        // the activation rule rather than the fp16 input rounding.
        std::vector<double> want(kSwigluCount);
        for (size_t i = 0; i < kSwigluCount; ++i) {
            want[i] = aeon::reference::clamped_swiglu(
                widen(gate_h)[i], widen(up_h)[i], kLimit, ClampMode::Asymmetric);
        }
        ok &= report("device SwiGLU vs oracle (asymmetric clamp)",
                     want, widen(out_h), 2e-2);

        // (1) Symmetric clamping of the gate is the plausible wrong reading.
        //
        //     The difference is measured ABSOLUTELY, not relatively: the probes
        //     that separate the two rules produce values of order 4.5e-4 against
        //     a vector peak of order 1500, so a relative bound would be dominated
        //     by probes the two rules agree on and could never fail.
        {
            std::vector<double> symmetric(kSwigluCount);
            for (size_t i = 0; i < kSwigluCount; ++i) {
                symmetric[i] = aeon::reference::clamped_swiglu(
                    widen(gate_h)[i], widen(up_h)[i], kLimit, ClampMode::Symmetric);
            }
            const double rule_delta =
                aeon::reference::compare(want, symmetric, 1.0).max_abs;
            const double device_delta =
                aeon::reference::compare(symmetric, widen(out_h), 1.0).max_abs;
            ok &= check("symmetric gate clamp is a different activation",
                        rule_delta > 1e-4 && device_delta > 1e-5,
                        "max_abs vs asymmetric = " + std::to_string(rule_delta) +
                            "; vs device = " + std::to_string(device_delta));
        }

        // (2) No clamp at all differs too, so the clamp is not decorative.
        {
            std::vector<double> none(kSwigluCount);
            for (size_t i = 0; i < kSwigluCount; ++i) {
                none[i] = aeon::reference::clamped_swiglu(
                    widen(gate_h)[i], widen(up_h)[i], kLimit, ClampMode::None);
            }
            const ErrorStats s = aeon::reference::compare(want, none, 1.0);
            ok &= check("removing the clamp is a different activation",
                        s.max_abs > 1e-2,
                        "max_abs = " + std::to_string(s.max_abs));
        }

        // (2b) THE DEVICE must be on the asymmetric side of that fork.
        //
        //      Found by mutation testing. Every check above is either
        //      oracle-vs-oracle (1, 3) or a relative comparison whose floor is
        //      the probe's peak of ~1600 (2), and the whole asymmetric/symmetric
        //      difference is at most `silu(-limit)*limit` = 4.5e-3. So a kernel
        //      that clamps the gate symmetrically passed this gate completely.
        //
        //      The fix is a TARGETED comparison: restrict to the entries where
        //      the two rules actually disagree — gate below -limit — because a
        //      max_abs over the whole probe is dominated by the ~1600-magnitude
        //      entries that both rules treat identically.
        {
            size_t differing = 0;
            double worst_vs_right = 0.0, worst_vs_wrong = 0.0;
            const std::vector<double> device_out = widen(out_h);
            for (size_t i = 0; i < kSwigluCount; ++i) {
                if (!(gate_d[i] < -kLimit)) continue;
                ++differing;
                const double wrong = aeon::reference::clamped_swiglu(
                    widen(gate_h)[i], widen(up_h)[i], kLimit, ClampMode::Symmetric);
                worst_vs_right = std::fmax(worst_vs_right, std::fabs(device_out[i] - want[i]));
                worst_vs_wrong = std::fmax(worst_vs_wrong, std::fabs(device_out[i] - wrong));
            }
            // The tolerance is tied to the magnitude of the affected entries
            // (~4.5e-3), not to the probe peak, so fp16 rounding on the large
            // entries cannot mask the difference.
            const double tol = 1e-5;
            ok &= check("device follows the ASYMMETRIC rule at gate < -limit",
                        differing > 0 && worst_vs_right < tol &&
                            worst_vs_wrong > 100.0 * tol,
                        std::to_string(differing) + " entries; vs asymmetric = " +
                            std::to_string(worst_vs_right) + ", vs symmetric = " +
                            std::to_string(worst_vs_wrong));
        }

        // (3) State the asymmetry itself as a measurable, not just "the outputs
        //     differ": gate = -40 must NOT be clamped, while up = -40 must be.
        {
            const double below = aeon::reference::clamped_swiglu(-40.0, 1.0, kLimit,
                                                                 ClampMode::Asymmetric);
            const double below_sym = aeon::reference::clamped_swiglu(-40.0, 1.0, kLimit,
                                                                     ClampMode::Symmetric);
            const double up_clamped = aeon::reference::clamped_swiglu(0.0, -40.0, kLimit,
                                                                      ClampMode::Asymmetric);
            ok &= check("gate below -limit is NOT clamped; up below -limit IS",
                        std::fabs(below) < 1e-15 && std::fabs(up_clamped) < 1e-15 &&
                            std::fabs(below_sym) > 1e-5,
                        "silu(-40)=" + std::to_string(below) + " symmetric=" +
                            std::to_string(below_sym));
        }

        // (4) The clamp must actually fire on this data, or (1) and (2) are
        //     vacuous. Counted directly.
        {
            size_t gate_above = 0, up_outside = 0, gate_below = 0;
            for (size_t i = 0; i < kSwigluCount; ++i) {
                if (gate_d[i] > kLimit) ++gate_above;
                if (std::fabs(up_d[i]) > kLimit) ++up_outside;
                if (gate_d[i] < -kLimit) ++gate_below;
            }
            ok &= check("probe data exercises all three clamp edges",
                        gate_above > 0 && up_outside > 0 && gate_below > 0,
                        std::to_string(gate_above) + " gate>+10, " +
                            std::to_string(up_outside) + " |up|>10, " +
                            std::to_string(gate_below) + " gate<-10");
        }
    }

    // =====================================================================
    // C. The composed routed FFN
    // =====================================================================
    std::cout << "\n--- C. composed routed FFN (fused W13 + SwiGLU + W2) ---\n";

    // Device expert pointers, one expert, pointing at this fixture's payload.
    aeon::kernel::SwizzledW13ExpertPtrs w13{};
    aeon::kernel::SwizzledW2ExpertPtrs w2{};
    {
        const uint8_t* base = reinterpret_cast<const uint8_t*>(d_payload);
        w13.w1[0] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W1_PACKED_OFFSET);
        w13.s1[0] = reinterpret_cast<const half*>(base + aeon::core::AEON_W1_SCALE_OFFSET);
        w13.w3[0] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W3_PACKED_OFFSET);
        w13.s3[0] = reinterpret_cast<const half*>(base + aeon::core::AEON_W3_SCALE_OFFSET);
        w2.w2[0] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W2_PACKED_OFFSET);
        w2.s2[0] = reinterpret_cast<const half*>(base + aeon::core::AEON_W2_SCALE_OFFSET);
    }

    const float kUnitWeight[1] = {1.0f};
    CHECK_HIP(hipMemcpy(d_topk, kUnitWeight, sizeof(kUnitWeight), hipMemcpyHostToDevice));

    // Runs the whole expert on the device and returns (hidden, ffn_output).
    auto run_expert_real = [&](const std::vector<__half>& x,
                               std::vector<__half>& hidden_out,
                               std::vector<__half>& ffn_out) {
        CHECK_HIP(hipMemcpy(d_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice));
        aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<kWaves, kW1Rpw, kW1Lpr, kIterations>(
            d_x, w13, d_hidden, nullptr, 0, 1, kW1Rows, kW1Columns,
            static_cast<float>(kLimit));
        CHECK_HIP(hipGetLastError());

        CHECK_HIP(hipMemset(d_accum, 0, kW2Rows * sizeof(float)));
        CHECK_HIP(hipMemset(d_counters, 0, kAccumBlocks * sizeof(int)));
        aeon::kernel::dispatch_aeon_moe_fused_w2_accum<kWaves, kW2Rpw, kW2Lpr, kIterations>(
            d_hidden, w2, d_topk, nullptr, d_accum, d_ffn_out, d_counters,
            1, kW2Rows, kW2Columns);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());

        hidden_out.assign(kW1Rows, __half{});
        ffn_out.assign(kW2Rows, __half{});
        CHECK_HIP(hipMemcpy(hidden_out.data(), d_hidden, kW1Rows * sizeof(__half),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(ffn_out.data(), d_ffn_out, kW2Rows * sizeof(__half),
                            hipMemcpyDeviceToHost));
    };

    for (int scale_case = 0; scale_case < 2; ++scale_case) {
        const std::vector<double>& x_src = scale_case == 0 ? x_moderate : x_large;
        const std::vector<__half> x_h = to_half(x_src);
        const std::vector<double> x_d = widen(x_h);

        std::vector<__half> hidden_h, ffn_h;
        run_expert_real(x_h, hidden_h, ffn_h);

        const aeon::reference::ExpertGateUp gu =
            aeon::reference::expert_gate_up(payload.data(), x_d);

        std::vector<double> want_hidden(kW1Rows);
        for (int i = 0; i < kW1Rows; ++i) {
            want_hidden[i] = aeon::reference::clamped_swiglu(
                gu.gate[i], gu.up[i], kLimit, ClampMode::Asymmetric);
        }
        ok &= report(scale_case == 0 ? "hidden vs oracle (moderate activation)"
                                     : "hidden vs oracle (large activation)",
                     want_hidden, widen(hidden_h), kHiddenTol);

        const std::vector<double> want_ffn =
            aeon::reference::expert_ffn(payload.data(), x_d, kLimit, ClampMode::Asymmetric);
        ok &= report(scale_case == 0 ? "FFN output vs oracle (moderate activation)"
                                     : "FFN output vs oracle (large activation)",
                     want_ffn, widen(ffn_h), kFfnTol);

        // How far the clamp actually moves this case. Two separate questions,
        // reported separately because the first run of this gate conflated them:
        //
        //   * does the clamp FIRE here (some pre-activation outside the limit)?
        //   * is the ASYMMETRY observable here (some gate below -limit)?
        //
        // The second is strictly stronger and does not follow from the first.
        // In the large-activation case 360 gate values exceed +10 and 662 up
        // values are outside +-10, yet the symmetric and asymmetric rules still
        // agree — because every out-of-range gate is *above* the limit, and that is
        // the edge the two rules share. The asymmetry is certified in section B
        // against a constructed probe (gate = -40); here it is measured and only
        // asserted when the data actually reaches that edge.
        std::vector<double> sym_hidden(kW1Rows);
        for (int i = 0; i < kW1Rows; ++i) {
            sym_hidden[i] = aeon::reference::clamped_swiglu(
                gu.gate[i], gu.up[i], kLimit, ClampMode::Symmetric);
        }
        size_t gate_above = 0, up_outside = 0, gate_below = 0;
        for (int i = 0; i < kW1Rows; ++i) {
            if (gu.gate[i] > kLimit) ++gate_above;
            if (std::fabs(gu.up[i]) > kLimit) ++up_outside;
            if (gu.gate[i] < -kLimit) ++gate_below;
        }
        const double sym_delta = aeon::reference::compare(
            want_hidden, sym_hidden, 1.0).max_abs;
        const std::string engagement =
            std::to_string(gate_above) + " gate>+10, " +
            std::to_string(up_outside) + " |up|>10, " +
            std::to_string(gate_below) + " gate<-10; sym delta " +
            std::to_string(sym_delta);

        if (scale_case == 0) {
            std::printf("  %-54s %-22s %s\n", "moderate: clamp engagement measured",
                        gate_above + up_outside == 0 ? "clamp inert" : "clamp fires", engagement.c_str());
        } else {
            ok &= check("large: the clamp fires on this data",
                        gate_above + up_outside > 0, engagement);
            // The asymmetry, when the data reaches it.
            if (gate_below > 0) {
                ok &= check("large: the asymmetric edge is observable",
                            sym_delta > 1e-5, engagement);

                // ...and the DEVICE is on the asymmetric side of it. Same
                // targeted comparison as section B, for the same reason: a
                // max_abs over `hidden` is dominated by the ~40-magnitude
                // entries, where the two rules agree, so it cannot see a
                // symmetrically-clamped kernel. Restricting to gate < -limit
                // isolates the entries that carry the information.
                size_t differing = 0;
                double worst_vs_right = 0.0, worst_vs_wrong = 0.0;
                const std::vector<double> device_hidden = widen(hidden_h);
                for (int i = 0; i < kW1Rows; ++i) {
                    if (!(gu.gate[i] < -kLimit)) continue;
                    ++differing;
                    worst_vs_right = std::fmax(worst_vs_right,
                                               std::fabs(device_hidden[i] - want_hidden[i]));
                    worst_vs_wrong = std::fmax(worst_vs_wrong,
                                               std::fabs(device_hidden[i] - sym_hidden[i]));
                }
                const double tol = 1e-4;
                ok &= check("device follows the ASYMMETRIC rule (fused W13 kernel)",
                            differing > 0 && worst_vs_right < tol &&
                                worst_vs_wrong > 10.0 * tol,
                            std::to_string(differing) + " entries; vs asymmetric = " +
                                std::to_string(worst_vs_right) + ", vs symmetric = " +
                                std::to_string(worst_vs_wrong));
            } else {
                std::printf("  %-54s %-22s %s\n",
                            "large: asymmetric edge not reached (section B covers it)",
                            "n/a", engagement.c_str());
            }
        }
    }

    // =====================================================================
    // D. A real artifact payload
    // =====================================================================
    std::cout << "\n--- D. real artifact expert payload ---\n";
    {
        const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
        aeon::core::AeonModelLoader loader;
        loader.open_model(model_dir);

        constexpr uint32_t kLayer = 3; // first non-hash MoE layer
        constexpr uint32_t kExpert = 17;
        const uint8_t* real = loader.get_expert_data(kLayer, kExpert);

        std::vector<uint8_t> real_payload(kPayloadBytes);
        std::memcpy(real_payload.data(), real, kPayloadBytes);

        // Real weights are larger than the synthetic fixture, so the activation
        // is kept at moderate scale; the clamp rule is section B's job.
        const std::vector<__half> x_h = to_half(x_moderate);
        const std::vector<double> x_d = widen(x_h);

        const std::vector<double> decoded_w1 =
            aeon::reference::swizzled_decode(real_payload.data(), SwizzledKind::W1);

        // (1) W1 through the format, on real bytes.
        CHECK_HIP(hipMemcpy(d_payload, real_payload.data(), kPayloadBytes,
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_x, x_h.data(), x_h.size() * sizeof(__half),
                            hipMemcpyHostToDevice));
        aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<kWaves, kW1Rpw, kW1Lpr, kIterations>(
            d_x,
            reinterpret_cast<const uint32_t*>(
                reinterpret_cast<const uint8_t*>(d_payload) + aeon::core::AEON_W1_PACKED_OFFSET),
            reinterpret_cast<const half*>(
                reinterpret_cast<const uint8_t*>(d_payload) + aeon::core::AEON_W1_SCALE_OFFSET),
            d_gemv_out, kW1Rows, kW1Columns);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<__half> w1_out(kW1Rows);
        CHECK_HIP(hipMemcpy(w1_out.data(), d_gemv_out, kW1Rows * sizeof(__half),
                            hipMemcpyDeviceToHost));

        const std::vector<double> want_w1 = aeon::reference::matvec(
            kW1Rows, kW1Columns, x_d,
            [&](size_t o, size_t i) { return decoded_w1[o * kW1Columns + i]; });
        ok &= report("L3 E17: W1 GEMV vs decoder (real payload)",
                     want_w1, widen(w1_out), kGemvTol);

        // (2) The composed FFN on real bytes. Pointers must be refreshed: they
        //     still point at the synthetic payload's device copy otherwise.
        aeon::kernel::SwizzledW13ExpertPtrs real_w13{};
        aeon::kernel::SwizzledW2ExpertPtrs real_w2{};
        const uint8_t* base = reinterpret_cast<const uint8_t*>(d_payload);
        real_w13.w1[0] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W1_PACKED_OFFSET);
        real_w13.s1[0] = reinterpret_cast<const half*>(base + aeon::core::AEON_W1_SCALE_OFFSET);
        real_w13.w3[0] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W3_PACKED_OFFSET);
        real_w13.s3[0] = reinterpret_cast<const half*>(base + aeon::core::AEON_W3_SCALE_OFFSET);
        real_w2.w2[0] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W2_PACKED_OFFSET);
        real_w2.s2[0] = reinterpret_cast<const half*>(base + aeon::core::AEON_W2_SCALE_OFFSET);

        aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<kWaves, kW1Rpw, kW1Lpr, kIterations>(
            d_x, real_w13, d_hidden, nullptr, 0, 1, kW1Rows, kW1Columns,
            static_cast<float>(kLimit));
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipMemset(d_accum, 0, kW2Rows * sizeof(float)));
        CHECK_HIP(hipMemset(d_counters, 0, kAccumBlocks * sizeof(int)));
        aeon::kernel::dispatch_aeon_moe_fused_w2_accum<kWaves, kW2Rpw, kW2Lpr, kIterations>(
            d_hidden, real_w2, d_topk, nullptr, d_accum, d_ffn_out, d_counters,
            1, kW2Rows, kW2Columns);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<__half> hidden_h(kW1Rows), ffn_h(kW2Rows);
        CHECK_HIP(hipMemcpy(hidden_h.data(), d_hidden, kW1Rows * sizeof(__half),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(ffn_h.data(), d_ffn_out, kW2Rows * sizeof(__half),
                            hipMemcpyDeviceToHost));

        const aeon::reference::ExpertGateUp gu =
            aeon::reference::expert_gate_up(real_payload.data(), x_d);
        std::vector<double> want_hidden(kW1Rows);
        for (int i = 0; i < kW1Rows; ++i) {
            want_hidden[i] = aeon::reference::clamped_swiglu(
                gu.gate[i], gu.up[i], kLimit, ClampMode::Asymmetric);
        }
        ok &= report("L3 E17: hidden vs oracle (real payload)",
                     want_hidden, widen(hidden_h), kHiddenTol);

        const std::vector<double> want_ffn = aeon::reference::expert_ffn(
            real_payload.data(), x_d, kLimit, ClampMode::Asymmetric);
        ok &= report("L3 E17: FFN output vs oracle (real payload)",
                     want_ffn, widen(ffn_h), kFfnTol);

        // The real payload must be a genuine INT4 encoding, not a degenerate one
        // (all zeros would pass every comparison above trivially).
        size_t nonzero = 0;
        for (double v : decoded_w1) {
            if (v != 0.0) ++nonzero;
        }
        ok &= check("real payload is non-degenerate",
                    nonzero > decoded_w1.size() / 2,
                    std::to_string(nonzero * 100 / decoded_w1.size()) +
                        "% of W1 weights are non-zero");
    }

    // =====================================================================
    // E. Accumulation order — a MEASUREMENT
    // =====================================================================
    std::cout << "\n--- E. accumulation order (measurement, not certification) ---\n";
    {
        // The W2 path accumulates with `atomicAdd`, so the summation order is a
        // property of the scheduler, not of the source. Gate 14 asks whether that
        // order is stable and how much it moves the result. Repeat identical work
        // and compare.
        constexpr int kExperts = 6;
        const std::vector<__half> x_h = to_half(x_moderate);
        CHECK_HIP(hipMemcpy(d_x, x_h.data(), x_h.size() * sizeof(__half),
                            hipMemcpyHostToDevice));

        aeon::kernel::SwizzledW13ExpertPtrs many_w13{};
        aeon::kernel::SwizzledW2ExpertPtrs many_w2{};
        const uint8_t* base = reinterpret_cast<const uint8_t*>(d_payload);
        for (int e = 0; e < kExperts; ++e) {
            many_w13.w1[e] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W1_PACKED_OFFSET);
            many_w13.s1[e] = reinterpret_cast<const half*>(base + aeon::core::AEON_W1_SCALE_OFFSET);
            many_w13.w3[e] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W3_PACKED_OFFSET);
            many_w13.s3[e] = reinterpret_cast<const half*>(base + aeon::core::AEON_W3_SCALE_OFFSET);
            many_w2.w2[e] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W2_PACKED_OFFSET);
            many_w2.s2[e] = reinterpret_cast<const half*>(base + aeon::core::AEON_W2_SCALE_OFFSET);
        }

        __half* d_hidden_many = nullptr;
        CHECK_HIP(hipMalloc(&d_hidden_many,
                            aeon::kernel::kAeonSwizzledMaxExperts * kW1Rows * sizeof(__half)));
        aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<kWaves, kW1Rpw, kW1Lpr, kIterations>(
            d_x, many_w13, d_hidden_many, nullptr, 0, kExperts, kW1Rows, kW1Columns,
            static_cast<float>(kLimit));
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());

        float weights[kExperts] = {0.31f, 0.27f, 0.19f, 0.11f, 0.07f, 0.05f};
        std::vector<float> weights_host(aeon::kernel::kAeonSwizzledMaxExperts, 0.0f);
        for (int e = 0; e < kExperts; ++e) weights_host[static_cast<size_t>(e)] = weights[e];
        CHECK_HIP(hipMemcpy(d_topk, weights_host.data(),
                            weights_host.size() * sizeof(float), hipMemcpyHostToDevice));

        constexpr int kRepeats = 32;
        std::vector<__half> first(kW2Rows);
        size_t differing_runs = 0;
        double worst_abs = 0.0;
        for (int r = 0; r < kRepeats; ++r) {
            CHECK_HIP(hipMemset(d_accum, 0, kW2Rows * sizeof(float)));
            CHECK_HIP(hipMemset(d_counters, 0, kAccumBlocks * sizeof(int)));
            aeon::kernel::dispatch_aeon_moe_fused_w2_accum<kWaves, kW2Rpw, kW2Lpr, kIterations>(
                d_hidden_many, many_w2, d_topk, nullptr, d_accum, d_ffn_out, d_counters,
                kExperts, kW2Rows, kW2Columns);
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipDeviceSynchronize());

            std::vector<__half> run(kW2Rows);
            CHECK_HIP(hipMemcpy(run.data(), d_ffn_out, kW2Rows * sizeof(__half),
                                hipMemcpyDeviceToHost));
            if (r == 0) {
                first = run;
                continue;
            }
            if (run != first) {
                ++differing_runs;
                for (int i = 0; i < kW2Rows; ++i) {
                    worst_abs = std::fmax(worst_abs, std::fabs(
                        static_cast<double>(__half2float(run[static_cast<size_t>(i)])) -
                        static_cast<double>(__half2float(first[static_cast<size_t>(i)]))));
                }
            }
        }

        std::printf("  %-54s %-22s %s\n",
                    "32 identical runs of the 6-expert atomic accumulation",
                    differing_runs == 0 ? "bit-identical" : "NOT identical",
                    differing_runs == 0 ? "PASS" : "MEASURED");
        std::printf("      (%zu of %d runs differ after fp16 rounding; worst |delta| = %.3e)\n",
                    differing_runs, kRepeats - 1, worst_abs);

        // Whatever the answer, the total must still equal the weight-weighted sum
        // of the per-expert outputs to within the fp32 accumulation bound. That is
        // the property Gate 14 actually cares about.
        {
            std::vector<double> reference(kW2Rows, 0.0);
            for (int e = 0; e < kExperts; ++e) {
                // Hidden rows for expert e, from the device's own buffer.
                std::vector<__half> hidden_e(kW1Rows);
                CHECK_HIP(hipMemcpy(hidden_e.data(),
                                    d_hidden_many + static_cast<size_t>(e) * kW1Rows,
                                    kW1Rows * sizeof(__half), hipMemcpyDeviceToHost));
                std::vector<__half> out_e(kW2Rows);
                // A single-expert run of the same kernel, so the comparison is
                // between two device paths and the oracle is not involved.
                aeon::kernel::SwizzledW2ExpertPtrs one{};
                one.w2[0] = many_w2.w2[e];
                one.s2[0] = many_w2.s2[e];
                __half* d_hidden_one = d_hidden_many + static_cast<size_t>(e) * kW1Rows;
                float unit[aeon::kernel::kAeonSwizzledMaxExperts] = {};
                unit[0] = 1.0f;
                float* d_unit = nullptr;
                CHECK_HIP(hipMalloc(&d_unit, sizeof(unit)));
                CHECK_HIP(hipMemcpy(d_unit, unit, sizeof(unit), hipMemcpyHostToDevice));
                CHECK_HIP(hipMemset(d_accum, 0, kW2Rows * sizeof(float)));
                CHECK_HIP(hipMemset(d_counters, 0, kAccumBlocks * sizeof(int)));
                aeon::kernel::dispatch_aeon_moe_fused_w2_accum<
                    kWaves, kW2Rpw, kW2Lpr, kIterations>(
                    d_hidden_one, one, d_unit, nullptr, d_accum, d_ffn_out, d_counters,
                    1, kW2Rows, kW2Columns);
                CHECK_HIP(hipGetLastError());
                CHECK_HIP(hipDeviceSynchronize());
                CHECK_HIP(hipMemcpy(out_e.data(), d_ffn_out, kW2Rows * sizeof(__half),
                                    hipMemcpyDeviceToHost));
                CHECK_HIP(hipFree(d_unit));
                for (int i = 0; i < kW2Rows; ++i) {
                    reference[static_cast<size_t>(i)] +=
                        static_cast<double>(weights[e]) *
                        static_cast<double>(__half2float(out_e[static_cast<size_t>(i)]));
                }
            }
            const double peak = peak_of(reference);
            double worst = 0.0;
            for (int i = 0; i < kW2Rows; ++i) {
                worst = std::fmax(worst, std::fabs(
                    static_cast<double>(__half2float(first[static_cast<size_t>(i)])) -
                    reference[static_cast<size_t>(i)]));
            }
            const double rel = peak > 0.0 ? worst / peak : worst;
            ok &= check("6-expert sum equals the weighted per-expert sum",
                        rel < 2e-3,
                        "max_rel = " + std::to_string(rel) +
                            " (fp16 output, fp32 accumulation)");
        }

        CHECK_HIP(hipFree(d_hidden_many));
    }

    CHECK_HIP(hipFree(d_payload));
    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_gemv_out));
    CHECK_HIP(hipFree(d_hidden));
    CHECK_HIP(hipFree(d_ffn_out));
    CHECK_HIP(hipFree(d_accum));
    CHECK_HIP(hipFree(d_topk));
    CHECK_HIP(hipFree(d_counters));
    CHECK_HIP(hipFree(d_gate));
    CHECK_HIP(hipFree(d_up));
    CHECK_HIP(hipFree(d_swiglu));

    std::cout << (ok ? "\n[SUCCESS] Routed expert gate passed.\n"
                     : "\n[FAILURE] Routed expert gate failed.\n");
    return ok ? 0 : 1;
}
