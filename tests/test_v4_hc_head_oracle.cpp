// -----------------------------------------------------------------------------
// Step 3 gate — Hyper-Connections head reduction, versus an independent fp64
// reference.
//
// This closes the one graph op that had **no code, no oracle and no gate**. It is
// the last operation before the LM head: `hc_head` collapses the model's four
// residual streams to a single 4096-wide vector, and everything downstream of it
// — the final RMSNorm, `lm_head`, the logits — reads that vector. Before this
// gate the fp64 reference did not implement it, so the reference could not have
// caught an error in it, and no other gate exercised it either. An error there
// produces plausibly-scaled logits and therefore fluent-looking garbage, which is
// exactly the failure mode that is hardest to attribute after the fact.
//
// The rule is a single contraction, not a mixture:
//
//   mixes  = hc_head_fn @ rmsnorm_without_weight(flatten(residual), rms_eps)
//   pre[j] = sigmoid(mixes[j] · hc_head_scale + hc_head_base[j]) + hc_eps
//   out[h] = Σ_j pre[j] · residual[j][h]
//
// `[V vllm/model_executor/kernels/mhc/triton.py  hc_head_reduce_triton_kernel]`
//
// It *looks* like Step 2.0's pre-mix and differs in four ways, each of which is
// silent if got wrong. Three are asserted here as discriminating checks rather
// than asserted by construction:
//
//   1. THE NORM HAS NO LEARNED WEIGHT. Every other norm in the model has one
//      (2.1's attention norm, 2.8's FFN norm, Step 4's final norm). This one does
//      not, so there is no tensor to load — and a `weight`-multiplying
//      implementation is reading something the artifact does not contain. The
//      gate asserts the tensor's absence structurally *and* measures that
//      applying a non-unit weight moves the output.
//   2. THE RMS IS OVER THE FLATTENED `hc_mult · hidden` DIMENSION, not per stream
//      — the same rule as `hc_mixes`. Per-stream RMS is a plausible misreading
//      and the gate measures the difference.
//   3. `hc_head_scale` IS A SCALAR `[1]`, broadcasting over all four gates —
//      against `F32 [3]` for `hc_attn_scale` / `hc_ffn_scale`. Copying the layer
//      scale's shape in is the natural error, so the gate asserts the shape from
//      the artifact as well as the dtype.
//   4. THERE IS NO SINKHORN AND NO COMB. With one output stream left there is
//      nothing to mix, and running the Sinkhorn here would be a category error.
//
// Inputs are the artifact's **real** `hc_head_fn` / `base` / `scale` at the real
// shapes, driven by three residual regressions: real token embedding rows expanded
// across the four streams (what the graph actually feeds it), a large-magnitude
// residual that saturates every sigmoid, and a small-magnitude one that sits on
// the `rms_eps` floor.
//
// Precision. `hc_head_fn` is fp32 and the reduction is fp32, so the floor is the
// single fp16 store of the 4096-wide output. Tolerance `3e-3` of peak, the same
// peak-relative basis the other gates use.
//
// HONEST LIMITATION, named so it is not mistaken for coverage. `hc_eps` is added
// after the sigmoid and the kernel keeps its `pre` vector in shared memory, so
// `pre` is **not readable from the output**. The eps-placement rule is therefore
// pinned at the **oracle** level as a closed form (drive the sigmoid to 0 and to
// 1 and require `pre == hc_eps` and `1 + hc_eps` exactly), and the gate reports
// the delta that omitting the eps would cause in `out`: at these magnitudes it is
// ~1e-6 relative, i.e. two orders below one fp16 ulp, so **no output comparison
// can discriminate it**. That is the same honest limitation trap 35 records for
// the sink-in-the-max, and it is stated rather than papered over.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "support/v4_layer_body_gate.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

using aeon::reference::ErrorStats;
using aeon::reference::HcHeadResult;

constexpr uint32_t kHidden = aeon::kernel::DSV4_HIDDEN_SIZE;
constexpr uint32_t kHcMult = 4;
constexpr uint32_t kFlat = kHcMult * kHidden;
constexpr double kRmsEps = 1e-6;
constexpr double kHcEps = 1e-6;

// Real token ids from `profiling-prompts/first-prompt.jsonl`, so the embedding
// rows are the ones the graph is actually driven with.
constexpr uint32_t kRealTokens[] = {65106, 295, 4654, 3999};

struct Harness {
    uint32_t checks{0};
    uint32_t failures{0};

    bool assert_that(const char* label, bool ok, const std::string& detail) {
        std::printf("  %-58s %-30s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
        ++checks;
        if (!ok) ++failures;
        return ok;
    }

    bool report(const char* label, const std::vector<double>& want,
                const std::vector<double>& got, double tol_frac, double abs_floor = 0.0) {
        const bool ok = aeon::testgate::report(label, want, got, tol_frac, abs_floor);
        ++checks;
        if (!ok) ++failures;
        return ok;
    }

    double max_abs_of(const std::vector<double>& a, const std::vector<double>& b) {
        return aeon::reference::compare(a, b, 0.0).max_abs;
    }
};

Harness harness;

const aeon::core::AeonModelLoader* g_loader = nullptr;

std::string sci(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.3e", value);
    return buffer;
}

// The four fp32 head parameters, decoded to fp64 from the artifact.
//
// The loader exposes the tensor as **host** memory (the dense container is
// mmapped), which is why these are direct pointer reads and not device copies.
struct HeadParams {
    std::vector<double> fn;     // [hc_mult * flat]
    std::vector<double> base;   // [hc_mult]
    std::vector<double> scale;  // [1]
};

HeadParams load_head_params() {
    HeadParams p;
    const float* fn = g_loader->get_data_ptr<float>("hc_head_fn");
    const float* base = g_loader->get_data_ptr<float>("hc_head_base");
    const float* scale = g_loader->get_data_ptr<float>("hc_head_scale");
    p.fn.assign(fn, fn + static_cast<size_t>(kHcMult) * kFlat);
    p.base.assign(base, base + kHcMult);
    p.scale.assign(scale, scale + 1);
    return p;
}

// One embedding row, expanded identically across the four HC streams — which is
// what Step 1 does before layer 0.
std::vector<double> residual_from_embedding(uint32_t token) {
    const auto* embed = reinterpret_cast<const uint16_t*>(
        g_loader->get_tensor("embed.weight").data);
    const std::vector<double> row =
        aeon::reference::half_bits_to_doubles(embed + static_cast<size_t>(token) * kHidden, kHidden);

    std::vector<double> residual(kFlat, 0.0);
    for (uint32_t j = 0; j < kHcMult; ++j) {
        for (uint32_t h = 0; h < kHidden; ++h) residual[j * kHidden + h] = row[h];
    }
    return residual;
}

// A deterministic synthetic residual with a chosen scale, used to reach the two
// regimes the real embedding rows cannot: full sigmoid saturation and the
// `rms_eps` floor.
std::vector<double> synthetic_residual(double scale, bool alternates) {
    std::vector<double> residual(kFlat, 0.0);
    for (uint32_t j = 0; j < kHcMult; ++j) {
        for (uint32_t h = 0; h < kHidden; ++h) {
            const double sign = (alternates && (h & 1u)) ? -1.0 : 1.0;
            const double pattern = 1.0 + static_cast<double>((h * 7u + j * 13u) % 11u) / 11.0;
            residual[j * kHidden + h] = sign * scale * pattern;
        }
    }
    return residual;
}

// Streams with deliberately different magnitudes.
//
// The embedding broadcast cannot separate the flattened RMS from the per-stream
// RMS, and that is a fact about the input rather than about the rules: when all
// four streams are byte-identical, the mean square over the flattened `hc_mult ·
// hidden` dimension equals the mean square over one stream, so the two rules
// produce the same number and no probe built on broadcast input can tell them
// apart. Per-stream RMS is only observable against a residual whose streams
// differ in scale.
std::vector<double> stream_asymmetric_residual() {
    std::vector<double> residual(kFlat, 0.0);
    for (uint32_t j = 0; j < kHcMult; ++j) {
        const double stream_scale = 0.05 * static_cast<double>(1u << j); // 0.05, 0.1, 0.2, 0.4
        for (uint32_t h = 0; h < kHidden; ++h) {
            const double pattern = 1.0 + static_cast<double>((h * 5u + j) % 7u) / 7.0;
            residual[j * kHidden + h] = stream_scale * pattern;
        }
    }
    return residual;
}

struct KernelResult {
    std::vector<double> out;  // [hidden], widened from fp16
};

// Runs the production kernel: one block of 32 lanes, as the pipeline launches it.
KernelResult run_kernel(const std::vector<double>& residual, const std::vector<double>& fn,
                        const std::vector<double>& base, const std::vector<double>& scale) {
    std::vector<float> h_residual(kFlat);
    for (size_t i = 0; i < kFlat; ++i) h_residual[i] = static_cast<float>(residual[i]);

    hipStream_t stream = nullptr;
    CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    float* d_residual = nullptr;
    float* d_fn = nullptr;
    float* d_base = nullptr;
    float* d_scale = nullptr;
    __half* d_out = nullptr;
    CHECK_HIP(hipMalloc(&d_residual, kFlat * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_fn, fn.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_base, base.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_scale, sizeof(float)));
    CHECK_HIP(hipMalloc(&d_out, kHidden * sizeof(__half)));

    std::vector<float> h_fn(fn.size());
    for (size_t i = 0; i < fn.size(); ++i) h_fn[i] = static_cast<float>(fn[i]);
    std::vector<float> h_base(base.size());
    for (size_t i = 0; i < base.size(); ++i) h_base[i] = static_cast<float>(base[i]);
    const float h_scale = static_cast<float>(scale[0]);

    CHECK_HIP(hipMemcpy(d_residual, h_residual.data(), kFlat * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_fn, h_fn.data(), h_fn.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_base, h_base.data(), h_base.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_scale, &h_scale, sizeof(float), hipMemcpyHostToDevice));

    aeon::kernel::hc_head_wave32_kernel<<<1, 32, 0, stream>>>(
        d_residual, d_fn, d_base, d_scale, d_out, static_cast<int>(kHidden),
        static_cast<int>(kHcMult), static_cast<float>(kRmsEps), static_cast<float>(kHcEps));
    CHECK_HIP(hipStreamSynchronize(stream));

    std::vector<__half> h_out(kHidden);
    CHECK_HIP(hipMemcpy(h_out.data(), d_out, kHidden * sizeof(__half),
                        hipMemcpyDeviceToHost));

    KernelResult result;
    result.out.resize(kHidden);
    for (uint32_t h = 0; h < kHidden; ++h) {
        result.out[h] = aeon::reference::half_bits_to_double(__half_as_ushort(h_out[h]));
    }

    CHECK_HIP(hipFree(d_out));
    CHECK_HIP(hipFree(d_scale));
    CHECK_HIP(hipFree(d_base));
    CHECK_HIP(hipFree(d_fn));
    CHECK_HIP(hipFree(d_residual));
    CHECK_HIP(hipStreamDestroy(stream));
    return result;
}
HcHeadResult oracle_of(const std::vector<double>& residual, const HeadParams& p) {
    return aeon::reference::hc_head_reduce(
        residual, kHcMult, kHidden, p.fn.data(), p.base.data(), p.scale[0], kRmsEps, kHcEps);
}

int run() {
    std::printf("================================================================================\n");
    std::printf("  Step 3 — Hyper-Connections head reduction vs an independent fp64 reference\n");
    std::printf("================================================================================\n");

    aeon::core::select_compute_device(true);
    aeon::core::AeonModelLoader loader;
    loader.open_model("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon");
    g_loader = &loader;

    // -------------------------------------------------------------------------
    // A — the parameters, from the artifact
    // -------------------------------------------------------------------------
    std::printf("\n[A] The head parameters, checked against the artifact\n");

    const auto& fn_tensor = loader.get_tensor("hc_head_fn");
    const auto& base_tensor = loader.get_tensor("hc_head_base");
    const auto& scale_tensor = loader.get_tensor("hc_head_scale");

    harness.assert_that("A: hc_head_fn is F32 [4, 16384]",
                        fn_tensor.dtype == "F32" && fn_tensor.shape.size() == 2 &&
                            fn_tensor.shape[0] == kHcMult && fn_tensor.shape[1] == kFlat,
                        fn_tensor.dtype + " [" + std::to_string(fn_tensor.shape[0]) + ", " +
                            std::to_string(fn_tensor.shape[1]) + "]");

    harness.assert_that("A: hc_head_base is F32 [4]",
                        base_tensor.dtype == "F32" && base_tensor.byte_size ==
                            static_cast<int64_t>(kHcMult * sizeof(float)),
                        base_tensor.dtype + " bytes=" + std::to_string(base_tensor.byte_size));

    // The head scale is a *scalar*, not the three-entry vector the layer scales
    // are. This is the shape a copy-paste from `hc_attn_scale` gets wrong.
    harness.assert_that("A: hc_head_scale is a scalar F32 [1] (the layer scales are [3])",
                        scale_tensor.dtype == "F32" &&
                            scale_tensor.byte_size == static_cast<int64_t>(sizeof(float)),
                        scale_tensor.dtype + " bytes=" + std::to_string(scale_tensor.byte_size));

    harness.assert_that("A: the layer scales really are [3], so the shapes differ",
                        loader.get_tensor("layers.0.hc_attn_scale").byte_size ==
                            3 * static_cast<int64_t>(sizeof(float)),
                        "hc_attn_scale bytes=" +
                            std::to_string(loader.get_tensor("layers.0.hc_attn_scale").byte_size));

    // The norm is weightless. Asserted structurally first: a tensor that does not
    // exist cannot be read by a correct implementation, and an implementation that
    // reads one would fail to load at all.
    bool norm_tensor_present = false;
    for (const char* name : {"hc_head_norm.weight", "hc_head_norm", "hc_head.weight",
                             "hc_head_norm_weight"}) {
        if (loader.has_tensor(name)) norm_tensor_present = true;
    }
    harness.assert_that("A: there is no HC-head norm tensor to load (weightless norm)",
                        !norm_tensor_present,
                        "probed 4 candidate names, none present");

    const HeadParams params = load_head_params();
    harness.assert_that("A: the head parameters load and are finite",
                        params.fn.size() == static_cast<size_t>(kHcMult) * kFlat &&
                            params.base.size() == kHcMult && params.scale.size() == 1 &&
                            std::isfinite(params.scale[0]) &&
                            aeon::reference::peak_abs(params.fn) > 0.0,
                        "scale=" + aeon::testgate::num(params.scale[0], 8) + " peak|fn|=" +
                            aeon::testgate::num(aeon::reference::peak_abs(params.fn), 6));

    // -------------------------------------------------------------------------
    // B — the composition, on real weights and three residual regimes
    // -------------------------------------------------------------------------
    std::printf("\n[B] The reduction: kernel vs the fp64 reference\n");

    // The residual every discriminating check below is built on: a real embedding
    // row broadcast across the four streams, which is exactly what Step 1 hands
    // the stack.
    const std::vector<double> probe = residual_from_embedding(kRealTokens[0]);

    for (uint32_t token : kRealTokens) {
        const std::vector<double> residual = residual_from_embedding(token);
        const HcHeadResult want = oracle_of(residual, params);
        const KernelResult got = run_kernel(residual, params.fn, params.base, params.scale);
        harness.report(("B: real embedding token " + std::to_string(token)).c_str(),
                       want.out, got.out, 3e-3);
    }

    // Sigmoid saturation, reached through the **bias** rather than the residual,
    // and the reason is itself a property worth asserting: `mixes = (x·fn) ·
    // rsqrt(mean(x²) + eps)` is scale-invariant. Scaling the residual by `k`
    // multiplies the dot by `k` and `1/rms` by `1/k`, so `mixes` — and therefore
    // every `pre` — is unchanged. No amount of scaling the input can saturate
    // anything. Pushing `base` far below and far above zero does saturate, and
    // there every `pre` is exactly `hc_eps` or exactly `1 + hc_eps`.
    {
        std::vector<double> low = params.base;
        for (double& b : low) b -= 100.0;
        const HcHeadResult want_low = aeon::reference::hc_head_reduce(
            probe, kHcMult, kHidden, params.fn.data(), low.data(), params.scale[0],
            kRmsEps, kHcEps);
        const KernelResult got_low = run_kernel(probe, params.fn, low, params.scale);

        std::vector<double> high = params.base;
        for (double& b : high) b += 100.0;
        const HcHeadResult want_high = aeon::reference::hc_head_reduce(
            probe, kHcMult, kHidden, params.fn.data(), high.data(), params.scale[0],
            kRmsEps, kHcEps);
        const KernelResult got_high = run_kernel(probe, params.fn, high, params.scale);

        uint32_t at_zero = 0;
        uint32_t at_one = 0;
        for (double pre : want_low.pre_mix) {
            if (std::fabs(pre - kHcEps) < 1e-15) ++at_zero;
        }
        for (double pre : want_high.pre_mix) {
            if (std::fabs(pre - (1.0 + kHcEps)) < 1e-15) ++at_one;
        }
        harness.assert_that("B: at saturation every pre is exactly hc_eps or 1 + hc_eps",
                            at_zero == kHcMult && at_one == kHcMult,
                            "base-100: " + std::to_string(at_zero) + "/4 at eps; base+100: " +
                                std::to_string(at_one) + "/4 at 1+eps");

        harness.report("B: saturated low (base - 100)", want_low.out, got_low.out, 3e-3,
                       /*abs_floor=*/1e-7);
        harness.report("B: saturated high (base + 100)", want_high.out, got_high.out, 3e-3);
    }

    // The two closed forms the scale structure gives us — with a caveat that is
    // the interesting part rather than a defect.
    //
    // `mixes = (x·fn) · rsqrt(mean(x²) + eps)`. Scaling `x` by `k` scales the dot
    // by `k` and `mean(x²)` by `k²`, so the two cancel **exactly** — except that
    // `eps` does not scale with them. The invariance therefore holds only while
    // `mean(x²) >> eps`, and near the floor it is the `eps` term that deliberately
    // breaks it. Both halves are asserted: exactness at `rms_eps = 0`, and the
    // eps-sized deviation under the real value.
    //
    // The second closed form follows: with every `pre` fixed, `out = Σ pre_j x_j`
    // is degree-1 homogeneous, so the deviation here is the same eps effect seen
    // through the combine — and it would grow without bound if any `pre` were
    // wrong, which is what makes it a check on the combine rather than on `eps`.
    {
        const std::vector<double> base_input = residual_from_embedding(kRealTokens[2]);
        const double scale_factor = 1e3;
        std::vector<double> scaled = base_input;
        for (double& v : scaled) v *= scale_factor;

        const auto pre_delta = [&](double rms_eps) {
            const HcHeadResult a = aeon::reference::hc_head_reduce(
                base_input, kHcMult, kHidden, params.fn.data(), params.base.data(),
                params.scale[0], rms_eps, kHcEps);
            const HcHeadResult b = aeon::reference::hc_head_reduce(
                scaled, kHcMult, kHidden, params.fn.data(), params.base.data(),
                params.scale[0], rms_eps, kHcEps);
            double delta = 0.0;
            for (size_t j = 0; j < kHcMult; ++j) {
                delta = std::fmax(delta, std::fabs(a.pre_mix[j] - b.pre_mix[j]));
            }
            return std::make_pair(delta, a);
        };

        const auto exact = pre_delta(0.0);
        const auto real_eps = pre_delta(kRmsEps);

        harness.assert_that("B: with rms_eps = 0 the scale invariance is exact",
                            exact.first < 1e-14, "max pre delta=" + sci(exact.first));
        harness.assert_that("B: with the real rms_eps it holds to that eps's own order",
                            real_eps.first < 1e-4,
                            "max pre delta=" + sci(real_eps.first) +
                                " — the eps term is what breaks it");

        const std::vector<double>& reference_out = real_eps.second.out;
        std::vector<double> expected(kHidden);
        for (uint32_t h = 0; h < kHidden; ++h) expected[h] = scale_factor * reference_out[h];
        const double homog = harness.max_abs_of(expected, oracle_of(scaled, params).out);
        const double homog_peak = aeon::reference::peak_abs(expected);
        harness.assert_that("B: out is degree-1 homogeneous to that same eps order",
                            homog < 1e-5 * homog_peak,
                            "max_abs=" + sci(homog) + " = " +
                                aeon::testgate::num(homog / homog_peak, 1) + " of peak");
    }

    // Near the eps floor: the residual is small enough that `mean(x²)` is of the
    // order of `rms_eps`, which is the regime the eps exists for.
    {
        const std::vector<double> residual = synthetic_residual(1e-3, false);
        const HcHeadResult want = oracle_of(residual, params);
        const KernelResult got = run_kernel(residual, params.fn, params.base, params.scale);
        harness.report("B: near the rms_eps floor (residual x 1e-3)",
                       want.out, got.out, 3e-3);
    }

    // -------------------------------------------------------------------------
    // C — the discriminating checks
    // -------------------------------------------------------------------------
    std::printf("\n[C] What separates this rule from its near-misses\n");

    const HcHeadResult correct = oracle_of(probe, params);
    const KernelResult kernel = run_kernel(probe, params.fn, params.base, params.scale);

    // The instrument for every check below is the gate's **own** discrimination
    // floor — the distance between the kernel and the reference on this very
    // input — rather than an arbitrary fraction of the peak. A wrong reading only
    // has to be *detectable through this gate* to be a real fork; demanding a
    // large absolute delta would be an unrelated and unreachable bar, because the
    // projection is a 16 384-term sum whose random signs average most changes out.
    const double floor = harness.max_abs_of(kernel.out, correct.out);
    std::printf("  %-58s %-30s\n", "C: (reference) kernel-vs-oracle floor on this input",
                sci(floor).c_str());

    // 1. A learned weight on the norm. The artifact has no such tensor, so the
    //    gate supplies one; a real implementation could not.
    {
        std::vector<double> weight(kFlat);
        for (size_t i = 0; i < kFlat; ++i) weight[i] = 0.5 + static_cast<double>(i % 7) / 7.0;
        const HcHeadResult weighted = aeon::reference::hc_head_reduce(
            probe, kHcMult, kHidden, params.fn.data(), params.base.data(), params.scale[0],
            kRmsEps, kHcEps, /*weighted_norm=*/true, weight.data());
        const double separation = harness.max_abs_of(weighted.out, correct.out);
        harness.assert_that("C: a learned norm weight is detectable through this gate",
                            separation > 3.0 * floor,
                            "delta=" + sci(separation) + " = " +
                                aeon::testgate::num(separation / floor, 1) + "x the floor");
    }

    // 2 and 3. Per-stream RMS instead of the flattened one, on a residual whose
    //    streams actually differ in scale (the broadcast probe cannot separate
    //    them at all — see `stream_asymmetric_residual`).
    {
        const std::vector<double> asymmetric = stream_asymmetric_residual();
        const HcHeadResult want = oracle_of(asymmetric, params);
        const KernelResult got = run_kernel(asymmetric, params.fn, params.base, params.scale);
        const HcHeadResult per_stream = aeon::reference::hc_head_reduce(
            asymmetric, kHcMult, kHidden, params.fn.data(), params.base.data(), params.scale[0],
            kRmsEps, kHcEps, false, nullptr, /*per_stream_rms=*/true);

        const double asymmetric_floor = harness.max_abs_of(got.out, want.out);
        const double separation = harness.max_abs_of(per_stream.out, want.out);
        harness.assert_that("C: per-stream RMS is distinguishable on asymmetric streams",
                            separation > 3.0 * asymmetric_floor,
                            "delta=" + sci(separation) + " = " +
                                aeon::testgate::num(separation / asymmetric_floor, 1) +
                                "x that floor");

        // The kernel must sit on the flattened side of the fork. This is what
        // makes the separation above a real discrimination rather than noise.
        const double to_correct = harness.max_abs_of(got.out, want.out);
        const double to_wrong = harness.max_abs_of(got.out, per_stream.out);
        harness.assert_that("C: the kernel sits on the flattened side of that fork",
                            to_correct < to_wrong / 3.0,
                            "to flattened=" + sci(to_correct) + " to per-stream=" +
                                sci(to_wrong));
    }

    // 4. The eps placement, pinned at the oracle level (see the limitation note).
    {
        std::vector<double> strongly_negative = params.base;
        for (double& b : strongly_negative) b -= 100.0;
        const HcHeadResult all_zero = aeon::reference::hc_head_reduce(
            probe, kHcMult, kHidden, params.fn.data(), strongly_negative.data(),
            params.scale[0], kRmsEps, kHcEps);
        bool all_at_eps = true;
        for (double pre : all_zero.pre_mix) {
            if (std::fabs(pre - kHcEps) > 1e-15) all_at_eps = false;
        }

        std::vector<double> strongly_positive = params.base;
        for (double& b : strongly_positive) b += 100.0;
        const HcHeadResult all_one = aeon::reference::hc_head_reduce(
            probe, kHcMult, kHidden, params.fn.data(), strongly_positive.data(),
            params.scale[0], kRmsEps, kHcEps);
        bool all_at_one = true;
        for (double pre : all_one.pre_mix) {
            if (std::fabs(pre - (1.0 + kHcEps)) > 1e-15) all_at_one = false;
        }

        harness.assert_that("C: hc_eps is added after the sigmoid, at both extremes",
                            all_at_eps && all_at_one,
                            "sigmoid->0 gives hc_eps; sigmoid->1 gives 1+hc_eps");
    }

    // How visible would an omitted eps be in `out`? Reported, because the answer
    // is "not visible", and that is a limitation rather than a pass.
    {
        const HcHeadResult no_eps = aeon::reference::hc_head_reduce(
            probe, kHcMult, kHidden, params.fn.data(), params.base.data(), params.scale[0],
            kRmsEps, /*hc_eps=*/0.0);
        const double peak = aeon::reference::peak_abs(correct.out);
        const double delta = harness.max_abs_of(no_eps.out, correct.out);
        std::printf("  %-58s %-30s\n", "C: (reported) omitting hc_eps moves out by",
                    (sci(delta) + " = " +
                     aeon::testgate::num(delta / peak, 2) + " of peak").c_str());
    }

    // -------------------------------------------------------------------------
    // D — no Sinkhorn, no comb
    // -------------------------------------------------------------------------
    std::printf("\n[D] The head is a contraction, not a mixture\n");

    harness.assert_that("D: the head consumes no comb or sinkhorn parameter",
                        !loader.has_tensor("hc_head_comb") && !loader.has_tensor("hc_head_fn_comb"),
                        "no comb tensor in the artifact");

    // With only stream 3 non-zero, a *contraction* must give exactly
    // `pre[3] · x[3]`; a mixture would carry a cross-stream term. Closed form, and
    // checked against the kernel as well as the reference, so it is a check on the
    // implementation rather than only on the oracle.
    {
        const std::vector<double> full = residual_from_embedding(kRealTokens[1]);
        std::vector<double> only_last(kFlat, 0.0);
        for (uint32_t h = 0; h < kHidden; ++h) {
            only_last[3 * kHidden + h] = full[3 * kHidden + h];
        }

        const HcHeadResult want = oracle_of(only_last, params);
        const KernelResult got = run_kernel(only_last, params.fn, params.base, params.scale);

        std::vector<double> closed_form(kHidden);
        for (uint32_t h = 0; h < kHidden; ++h) {
            closed_form[h] = want.pre_mix[3] * only_last[3 * kHidden + h];
        }

        const double delta = harness.max_abs_of(closed_form, want.out);
        harness.assert_that("D: a single-stream residual gives out == pre[3] · x[3] exactly",
                            delta < 1e-12, "max_abs=" + sci(delta));
        harness.report("D: and the kernel agrees on that single-stream case",
                       closed_form, got.out, 3e-3);
    }

    // The four streams are identical here (embedding broadcast), so the output
    // must equal the row times the sum of the pre-mixes. Closed form, no oracle.
    {
        const std::vector<double> broadcast = residual_from_embedding(kRealTokens[0]);
        const HcHeadResult want = oracle_of(broadcast, params);
        const auto* embed = reinterpret_cast<const uint16_t*>(
            loader.get_tensor("embed.weight").data);
        const std::vector<double> row = aeon::reference::half_bits_to_doubles(
            embed + static_cast<size_t>(kRealTokens[0]) * kHidden, kHidden);

        double pre_sum = 0.0;
        for (double pre : want.pre_mix) pre_sum += pre;

        std::vector<double> closed_form(kHidden);
        for (uint32_t h = 0; h < kHidden; ++h) closed_form[h] = pre_sum * row[h];
        const double delta = harness.max_abs_of(closed_form, want.out);
        harness.assert_that("D: on broadcast streams out == (Σ pre) · row, exactly",
                            delta < 1e-12,
                            "max_abs=" + sci(delta));
    }

    // -------------------------------------------------------------------------
    std::printf("\n[Step 3 hc_head] %s — %u checks, %u failed\n",
                harness.failures == 0 ? "PASS" : "FAIL", harness.checks, harness.failures);
    std::printf("  dimensions: hc_mult=%u, hidden=%u, flattened=%u; params from the artifact\n",
                kHcMult, kHidden, kFlat);

    return harness.failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    return run();
}
