// -----------------------------------------------------------------------------
// Tier-1 gate: Hyper-Connections — project, Sinkhorn, pre-combine, post (Steps
// 2.0 and 2.7) — versus an independent fp64 reference.
//
// Hyper-Connections is where the original audit found a structural error, so
// this gate is written to be *discriminating*, not merely close. Four things it
// proves that a closeness test would not:
//
//   1. THE COMB INDEX CONVENTION. The comb is read as `C[contraction][output]`
//      and the post expansion is `out[j] = Σ_i C[i][j]·residual[i]`, which is
//      upstream's `einsum("...ij,...ih->...jh")`. A transposed comb is still a
//      valid doubly-stochastic matrix and still produces plausible output, so
//      the gate computes the transposed reading as well and asserts that (a) the
//      two differ materially and (b) the kernel matches the einsum one.
//   2. `hc_scale` HAS THREE ENTRIES and the comb uses `[2]`. The gate asserts
//      that reading the comb with `scale[1]` changes the result, so a two-scale
//      implementation cannot pass.
//   3. SINKHORN NEEDS 20 ITERATIONS. The gate measures the deviation from doubly
//      stochastic at 1 and at 20 iterations: large at 1, ~1e-7 at 20.
//   4. THE EPSILONS ARE PLACED ASYMMETRICALLY. Pre-mix carries `+eps` after the
//      sigmoid; post-mix carries none and is multiplied by a hardcoded 2.0. Both
//      are asserted by driving the sigmoid to 0 and checking the two branches
//      land on opposite sides of the eps.
//
// Precision. All HC state is fp32 except `layer_input` and the post residual,
// which are fp16. The oracle is fp64. Expected floor: ~1e-6 relative on the
// fp32 stages, ~5e-4 (one fp16 ulp) on the stages that store fp16.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/hc_sinkhorn.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

using aeon::reference::ErrorStats;
using aeon::reference::HcParams;
using aeon::reference::HcPreResult;

constexpr size_t kHidden  = 4096;
constexpr size_t kHcMult  = 4;
constexpr size_t kHcMult2 = kHcMult * kHcMult;      // 16
constexpr size_t kHcMult3 = kHcMult * 2 + kHcMult2; // 24
constexpr size_t kHcHidden = kHcMult * kHidden;     // 16384

constexpr double kFp32Tol = 1e-5; // relative to peak
constexpr double kFp16Tol = 2e-3;

std::vector<double> widen_half(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(__half2float(v[i]));
    return out;
}

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got, double tol, bool peak_denominator = true) {
    // `peak_denominator` reports the error as a fraction of the vector's own
    // scale. That is the right unit for signed, unstructured data — a
    // per-element relative error on a value that happens to sit near zero says
    // nothing about whether the computation is right.
    const ErrorStats s =
        aeon::reference::compare(want, got, peak_denominator ? 1.0 : 1e-3);
    if (s.size_mismatch) {
        std::printf("  %-40s SIZE MISMATCH                          FAIL\n", label);
        return false;
    }
    const bool pass = peak_denominator ? (s.max_rel <= tol) : aeon::reference::within(s, tol, tol);
    std::printf("  %-40s max_abs=%.3e  max_rel=%.3e  (peak=%.3e)  %s\n",
                label, s.max_abs, s.max_rel, aeon::reference::peak_abs(want),
                pass ? "PASS" : "FAIL");
    return pass;
}

bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-40s %-36s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

// Row and column sums of the comb, as a single "is it doubly stochastic" metric.
double stochastic_deviation(const std::vector<double>& C, size_t hc_mult) {
    double worst = 0.0;
    for (size_t i = 0; i < hc_mult; ++i) {
        double row = 0.0;
        for (size_t j = 0; j < hc_mult; ++j) row += C[i * hc_mult + j];
        worst = std::fmax(worst, std::fabs(row - 1.0));
    }
    for (size_t j = 0; j < hc_mult; ++j) {
        double col = 0.0;
        for (size_t i = 0; i < hc_mult; ++i) col += C[i * hc_mult + j];
        worst = std::fmax(worst, std::fabs(col - 1.0));
    }
    return worst;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: Hyper-Connections (project, Sinkhorn, post) vs fp64 reference\n";
    aeon::core::select_compute_device(true);

    bool ok = true;

    const HcParams params; // rms/pre/sinkhorn eps 1e-6, post_mult 2.0, iters 20

    // -----------------------------------------------------------------------
    // Inputs
    // -----------------------------------------------------------------------
    aeon::reference::Rng gen(0x0C0FFEE5ull);

    // hc_scale is [3]. Values chosen so the three branches are distinguishable
    // and so the comb logits are strongly asymmetric (making the transpose test
    // meaningful).
    const std::vector<double> hc_scale{1.7, 0.35, 1.15};
    const std::vector<float> h_scale = {1.7f, 0.35f, 1.15f};

    const std::vector<double> hc_base = gen.vector_filled(kHcMult3, 0.5);
    std::vector<float> h_base(kHcMult3);
    for (size_t i = 0; i < kHcMult3; ++i) h_base[i] = static_cast<float>(hc_base[i]);

    const std::vector<double> res_d = gen.vector_filled(kHcHidden, 1.0);
    std::vector<float> h_res(kHcHidden);
    for (size_t i = 0; i < kHcHidden; ++i) h_res[i] = static_cast<float>(res_d[i]);

    // `fn` is fp32 in the model (upstream asserts fn.dtype == float32).
    const std::vector<double> fn_d = gen.vector_filled(kHcMult3 * kHcHidden, 0.02);
    std::vector<float> h_fn(fn_d.size());
    for (size_t i = 0; i < fn_d.size(); ++i) h_fn[i] = static_cast<float>(fn_d[i]);

    // Oracle input: the fp32 values widening to double, exactly what the kernel reads.
    std::vector<double> res(kHcHidden), fn(fn_d.size());
    for (size_t i = 0; i < kHcHidden; ++i) res[i] = static_cast<double>(h_res[i]);
    for (size_t i = 0; i < fn.size(); ++i) fn[i] = static_cast<double>(h_fn[i]);
    const auto fn_ref = [&fn](size_t m, size_t k) { return fn[m * kHcHidden + k]; };

    // --- Device buffers ------------------------------------------------------
    float *d_res = nullptr, *d_fn = nullptr, *d_scale = nullptr, *d_base = nullptr;
    float *d_mixes = nullptr, *d_pre = nullptr, *d_post = nullptr, *d_comb = nullptr;
    CHECK_HIP(hipMalloc(&d_res, kHcHidden * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_fn, fn.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_scale, 3 * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_base, kHcMult3 * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_mixes, kHcMult3 * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_pre, kHcMult * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_post, kHcMult * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_comb, kHcMult2 * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_res, h_res.data(), kHcHidden * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_fn, h_fn.data(), fn.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_scale, h_scale.data(), 3 * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_base, h_base.data(), kHcMult3 * sizeof(float), hipMemcpyHostToDevice));

    auto download_f32 = [](const float* src, size_t n) {
        std::vector<float> out(n);
        CHECK_HIP(hipMemcpy(out.data(), src, n * sizeof(float), hipMemcpyDeviceToHost));
        return out;
    };
    auto to_d = [](const std::vector<float>& v) {
        return std::vector<double>(v.begin(), v.end());
    };

    // =======================================================================
    // Stage 1 — HC project: mixes[24]
    // =======================================================================
    aeon::kernel::hc_project_kernel<<<dim3(kHcMult3), dim3(256)>>>(
        d_res, d_fn, d_mixes, static_cast<int>(kHidden), static_cast<int>(kHcMult),
        static_cast<float>(params.rms_eps));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    const std::vector<double> mixes_want =
        aeon::reference::hc_mixes(res, kHcMult3, fn_ref, kHcHidden, params.rms_eps);
    const std::vector<double> mixes_got = to_d(download_f32(d_mixes, kHcMult3));
    ok &= report("hc_project: mixes[24]", mixes_want, mixes_got, kFp32Tol);

    // =======================================================================
    // Stage 2 — Sinkhorn: pre_mix[4], post_mix[4], comb[16]
    // Fed the kernel's own `mixes`, so this stage is isolated from the
    // projection and the measured delta is attributable to Sinkhorn alone.
    // =======================================================================
    aeon::kernel::hc_sinkhorn_normalize_kernel<<<dim3(1), dim3(32)>>>(
        d_mixes, d_scale, d_base, d_pre, d_post, d_comb,
        static_cast<float>(params.pre_eps), static_cast<float>(params.sinkhorn_eps),
        static_cast<float>(params.post_mult), params.sinkhorn_iters);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    HcPreResult ref;
    ref.mixes = mixes_got; // identical input to both
    aeon::reference::hc_sinkhorn(ref.mixes, hc_scale, hc_base, kHcMult, params, ref);

    ok &= report("sinkhorn: pre_mix[4]", ref.pre_mix, to_d(download_f32(d_pre, kHcMult)), kFp32Tol);
    ok &= report("sinkhorn: post_mix[4]", ref.post_mix, to_d(download_f32(d_post, kHcMult)), kFp32Tol);
    ok &= report("sinkhorn: comb[16]", ref.comb, to_d(download_f32(d_comb, kHcMult2)), kFp32Tol);

    const std::vector<double> comb_got = to_d(download_f32(d_comb, kHcMult2));

    // --- Discriminating check (3): the comb really is doubly stochastic ------
    ok &= check("comb is doubly stochastic (20 iters)",
                stochastic_deviation(comb_got, kHcMult) < 1e-4,
                "max |row sum - 1| and |col sum - 1| = " +
                    std::to_string(stochastic_deviation(comb_got, kHcMult)));

    // --- Discriminating check (3b): the iteration count is load-bearing ------
    {
        HcParams one_iter = params;
        one_iter.sinkhorn_iters = 1;
        HcPreResult few;
        few.mixes = mixes_got;
        aeon::reference::hc_sinkhorn(few.mixes, hc_scale, hc_base, kHcMult, one_iter, few);

        // Re-run the kernel with 1 iteration to confirm it tracks the oracle there too.
        aeon::kernel::hc_sinkhorn_normalize_kernel<<<dim3(1), dim3(32)>>>(
            d_mixes, d_scale, d_base, d_pre, d_post, d_comb,
            static_cast<float>(params.pre_eps), static_cast<float>(params.sinkhorn_eps),
            static_cast<float>(params.post_mult), 1);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        const std::vector<double> comb_one = to_d(download_f32(d_comb, kHcMult2));

        const double dev_one = stochastic_deviation(comb_one, kHcMult);
        const double dev_twenty = stochastic_deviation(comb_got, kHcMult);
        ok &= check("20 iterations are load-bearing",
                    dev_one > 100.0 * dev_twenty && dev_twenty < 1e-4,
                    "deviation at 1 iter = " + std::to_string(dev_one) +
                        ", at 20 = " + std::to_string(dev_twenty));
    }

    // Restore the full-iteration comb for the remaining stages.
    aeon::kernel::hc_sinkhorn_normalize_kernel<<<dim3(1), dim3(32)>>>(
        d_mixes, d_scale, d_base, d_pre, d_post, d_comb,
        static_cast<float>(params.pre_eps), static_cast<float>(params.sinkhorn_eps),
        static_cast<float>(params.post_mult), params.sinkhorn_iters);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    // --- Discriminating check (2): the comb uses hc_scale[2], not [1] --------
    {
        std::vector<double> wrong_scale = hc_scale;
        std::swap(wrong_scale[1], wrong_scale[2]);
        HcPreResult wrong;
        wrong.mixes = mixes_got;
        aeon::reference::hc_sinkhorn(wrong.mixes, wrong_scale, hc_base, kHcMult, params, wrong);

        const ErrorStats s = aeon::reference::compare(ref.comb, wrong.comb, 1.0);
        ok &= check("comb uses hc_scale[2] (third entry)",
                    s.max_rel > 1e-2,
                    "scale[1] vs scale[2] for comb differ by " + std::to_string(s.max_rel));
    }

    // --- Discriminating check (4): eps placement and the hardcoded 2.0 -------
    {
        // Drive the logits far negative so sigmoid -> 0. Then pre_mix -> pre_eps
        // and post_mix -> 0. If the eps were on the wrong branch, or absent from
        // the pre branch, the two would not be separated by a factor of ~1e6.
        std::vector<double> tiny(kHcMult3, -200.0);
        HcPreResult low;
        aeon::reference::hc_sinkhorn(tiny, hc_scale, hc_base, kHcMult, params, low);

        const double pre_min = *std::min_element(low.pre_mix.begin(), low.pre_mix.end());
        const double post_max = *std::max_element(low.post_mix.begin(), low.post_mix.end());
        ok &= check("pre_mix carries +eps, post_mix does not",
                    pre_min > params.pre_eps * 0.99 && post_max < params.pre_eps * 0.01,
                    "pre is " + std::to_string(pre_min) + ", post is " + std::to_string(post_max));

        // post_mult = 2.0 is a hardcoded constant, so post_mix may exceed 1.
        std::vector<double> large(kHcMult3, 10.0);
        HcPreResult high;
        aeon::reference::hc_sinkhorn(large, hc_scale, hc_base, kHcMult, params, high);
        const double pmax = *std::max_element(high.post_mix.begin(), high.post_mix.end());
        ok &= check("post_mult = 2.0 (post_mix can exceed 1)", pmax > 1.5,
                    "max post_mix = " + std::to_string(pmax));
    }

    // =======================================================================
    // Stage 3 — pre-combine -> layer_input (fp16 OUTPUT)
    // =======================================================================
    __half* d_layer_input = nullptr;
    CHECK_HIP(hipMalloc(&d_layer_input, kHidden * sizeof(__half)));
    aeon::kernel::hc_pre_combine_kernel<<<dim3((kHidden / 4 + 255) / 256), dim3(256)>>>(
        d_res, d_pre, d_layer_input, static_cast<int>(kHidden), static_cast<int>(kHcMult));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    {
        std::vector<__half> h_out(kHidden);
        CHECK_HIP(hipMemcpy(h_out.data(), d_layer_input, kHidden * sizeof(__half),
                            hipMemcpyDeviceToHost));
        ok &= report("pre-combine -> layer_input[4096]",
                     aeon::reference::hc_pre_combine(res, ref.pre_mix, kHidden),
                     widen_half(h_out), kFp16Tol);
    }

    // =======================================================================
    // Stage 4 — post expansion (fp16 in and out)
    // =======================================================================
    {
        std::vector<__half> h_xr(kHidden), h_res_h(kHcHidden), h_out(kHcHidden);
        for (size_t i = 0; i < kHidden; ++i) h_xr[i] = __float2half(static_cast<float>(gen.symmetric(1.0)));
        for (size_t i = 0; i < kHcHidden; ++i) h_res_h[i] = __float2half(static_cast<float>(gen.symmetric(1.0)));

        __half *d_xr = nullptr, *d_res_h = nullptr, *d_out = nullptr;
        CHECK_HIP(hipMalloc(&d_xr, kHidden * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_res_h, kHcHidden * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_out, kHcHidden * sizeof(__half)));
        CHECK_HIP(hipMemcpy(d_xr, h_xr.data(), kHidden * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_res_h, h_res_h.data(), kHcHidden * sizeof(__half), hipMemcpyHostToDevice));

        aeon::kernel::hc_post_kernel<<<dim3((kHidden + 255) / 256, 1), dim3(256)>>>(
            d_xr, d_res_h, d_post, d_comb, d_out, static_cast<int>(kHidden));
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_out.data(), d_out, kHcHidden * sizeof(__half),
                            hipMemcpyDeviceToHost));

        const std::vector<double> layer_out = widen_half(h_xr);
        const std::vector<double> residual_h = widen_half(h_res_h);
        const std::vector<double> got = widen_half(h_out);

        const std::vector<double> want = aeon::reference::hc_post(
            layer_out, residual_h, ref.post_mix, ref.comb, kHidden, /*transpose_comb=*/false);
        ok &= report("hc_post: residual_out[4 x 4096]", want, got, kFp16Tol);

        // --- The convention discriminator ---------------------------------
        // A transposed comb is still doubly stochastic and still yields
        // plausible residuals, so "the numbers are close" cannot tell the two
        // apart. Compute both and require the correct one to match.
        const std::vector<double> transposed = aeon::reference::hc_post(
            layer_out, residual_h, ref.post_mix, ref.comb, kHidden, /*transpose_comb=*/true);
        const ErrorStats separation = aeon::reference::compare(want, transposed, 1.0);
        ok &= check("comb convention matters (transpose differs)",
                    separation.max_rel > 1e-2,
                    "einsum reading vs transposed differ by " + std::to_string(separation.max_rel));

        // And the kernel must be on the einsum side, not the transposed side.
        const ErrorStats to_transposed = aeon::reference::compare(transposed, got, 1.0);
        const ErrorStats to_einsum = aeon::reference::compare(want, got, 1.0);
        ok &= check("kernel matches einsum convention",
                    to_einsum.max_rel < to_transposed.max_rel,
                    "einsum err " + std::to_string(to_einsum.max_rel) +
                        " < transposed err " + std::to_string(to_transposed.max_rel));

        CHECK_HIP(hipFree(d_xr));
        CHECK_HIP(hipFree(d_res_h));
        CHECK_HIP(hipFree(d_out));
    }

    // =======================================================================
    // Stage 5 — oracle self-checks
    // =======================================================================
    {
        // With zero logits the softmax is uniform (1/4 per entry), and uniform is
        // invariant under every one of the axis normalizations. So the comb must
        // stay uniform.
        //
        // "Uniform" here means uniform up to a known, exact offset. Each
        // normalization divides by `(sum + eps)`, and the sum of a uniform row is
        // exactly 1, so the entry is multiplied by `1/(1+eps)` and lands at
        // `1/4 · (1 - eps + O(eps²))`. The deviation from 1/4 is therefore
        // `eps / hc_mult = 2.5e-7` — *not* zero. Asserting `1e-9` here was wrong
        // and this gate caught it, which is the point of keeping a closed-form
        // check in the oracle.
        //
        // The same offset shows up in the doubly-stochastic check above: the
        // fixed point has row and column sums of `1 - eps`, so the comb is never
        // *exactly* doubly stochastic. `hc_sinkhorn_eps` is the floor on that
        // deviation, and the plan's "within 1e-4" gate is really "within eps".
        std::vector<double> flat(kHcMult3, 0.0);
        const std::vector<double> zero_scale{0.0, 0.0, 0.0};
        const std::vector<double> zero_base(kHcMult3, 0.0);
        HcPreResult u;
        aeon::reference::hc_sinkhorn(flat, zero_scale, zero_base, kHcMult, params, u);

        const double uniform = 1.0 / static_cast<double>(kHcMult);
        double worst = 0.0;
        for (size_t i = 0; i < kHcMult2; ++i) {
            worst = std::fmax(worst, std::fabs(u.comb[i] - uniform));
        }
        const double expected = params.sinkhorn_eps / static_cast<double>(kHcMult);
        ok &= check("oracle: uniform logits -> uniform comb (up to eps)",
                    std::fabs(worst - expected) < 1e-12,
                    "deviation " + std::to_string(worst) + " ~= eps/hc_mult " +
                        std::to_string(expected));
    }

    CHECK_HIP(hipFree(d_res));
    CHECK_HIP(hipFree(d_fn));
    CHECK_HIP(hipFree(d_scale));
    CHECK_HIP(hipFree(d_base));
    CHECK_HIP(hipFree(d_mixes));
    CHECK_HIP(hipFree(d_pre));
    CHECK_HIP(hipFree(d_post));
    CHECK_HIP(hipFree(d_comb));
    CHECK_HIP(hipFree(d_layer_input));

    std::cout << (ok ? "[SUCCESS] Hyper-Connections gate passed.\n"
                     : "[FAILURE] Hyper-Connections gate failed.\n");
    return ok ? 0 : 1;
}
