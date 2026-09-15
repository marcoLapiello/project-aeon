// -----------------------------------------------------------------------------
// Tier-1 gate: attention score + sink + softmax (Step 2.4.1) — versus an
// independent fp64 reference.
//
// The kernel under test is `v4_sliding_window_attn_wave32_kernel`, which is the
// one place the local path is self-contained: scores, sink, softmax, and the
// value combination all happen in one launch, with V = K.
//
// What this gate proves beyond closeness:
//
//   * THE LOCAL WINDOW IS EXACTLY `min(pos+1, 128)`. Not "at most 128", and not
//     "all keys up to pos". Perturbing a key that is outside the window must
//     change the output by *zero*, and perturbing one inside must change it.
//   * THE SINK OCCUPIES THE DENOMINATOR. A large positive sink shrinks the
//     output; a large negative one leaves it untouched. Both are checked against
//     the oracle, and the negative case is the discriminating one.
//   * `scale = 1/sqrt(512)`, the FULL head — not 1/sqrt(128) (index head) and
//     not 1/sqrt(64) (rope part).
//   * V = K: there is no value tensor to pass, and the output is a convex
//     combination of the in-window key rows.
//
// HONEST LIMITATION. The plan says the sink must be included in the max. That is
// a numerical-robustness property and it is **not observable in the output**:
// when the sink dominates, all mass sits on the sink and the output is zero
// whether or not the sink was in the max. This gate therefore does *not* claim
// to test it; it asserts finiteness instead, which is the only thing the output
// can show. See the note in the oracle header.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
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

constexpr size_t kHeads     = 64;
constexpr size_t kHeadDim   = 512;
constexpr size_t kWindow    = 128;
constexpr size_t kTokens    = 150;      // > window, so a full-window position exists
constexpr double kScale     = 0.04419417382415922; // 1 / sqrt(512), the full head
constexpr double kTol       = 2e-3;     // fraction of peak; output is fp16

// Positions chosen to cover the plan's three regimes.
constexpr size_t kPosZero     = 0;      // 1 key
constexpr size_t kPosShort    = 5;      // 6 keys, < window
constexpr size_t kPosAtWindow = 127;    // exactly 128 keys — first full window
constexpr size_t kPosPast     = 149;    // full window, oldest keys excluded

std::vector<double> widen(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(__half2float(v[i]));
    return out;
}

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got) {
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-44s SIZE MISMATCH                       FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(s.max_rel) && s.max_rel <= kTol;
    std::printf("  %-44s max_abs=%.3e  max_rel=%.3e  (peak=%.3e)  %s\n",
                label, s.max_abs, s.max_rel, aeon::reference::peak_abs(want),
                pass ? "PASS" : "FAIL");
    return pass;
}

// For the sink-dominated case, where both the reference and the kernel produce
// values at the noise floor. A relative comparison there compares two zeroes;
// the only meaningful statement is an absolute one, against the scale the output
// *would* have had without the sink. `reference_scale` is that scale.
bool report_abs(const char* label, const std::vector<double>& want,
                const std::vector<double>& got, double reference_scale) {
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-44s SIZE MISMATCH                       FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(s.max_abs) && s.max_abs <= 1e-10 * reference_scale;
    std::printf("  %-44s max_abs=%.3e (ref scale %.3e)           %s\n",
                label, s.max_abs, reference_scale, pass ? "PASS" : "FAIL");
    return pass;
}

bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-44s %-32s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

// The window the kernel should use at `position`: `min(pos+1, window)` keys
// ending at `pos`.
size_t window_start(size_t position) {
    return position >= kWindow - 1 ? position - (kWindow - 1) : 0;
}
size_t window_count(size_t position) {
    return position - window_start(position) + 1;
}

// Slices rows [start, start+count) of a [rows, head_dim] buffer.
std::vector<double> slice_rows(const std::vector<double>& v, size_t start,
                               size_t count, size_t head_dim) {
    return std::vector<double>(v.begin() + start * head_dim,
                               v.begin() + (start + count) * head_dim);
}

// Slices one token's [heads, head_dim] q or out.
std::vector<double> slice_token(const std::vector<double>& v, size_t token,
                                size_t heads, size_t head_dim) {
    return std::vector<double>(v.begin() + token * heads * head_dim,
                               v.begin() + (token + 1) * heads * head_dim);
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: attention score + sink + softmax vs fp64 reference\n";
    aeon::core::select_compute_device(true);

    bool ok = true;

    // -----------------------------------------------------------------------
    // Inputs
    // -----------------------------------------------------------------------
    aeon::reference::Rng gen(0x0A77E171ull);

    // q has to differ per token for the window test to mean anything.
    std::vector<__half> h_q(kTokens * kHeads * kHeadDim);
    std::vector<__half> h_k(kTokens * kHeadDim);
    for (size_t i = 0; i < h_q.size(); ++i) {
        h_q[i] = __float2half(static_cast<float>(gen.symmetric(1.0)));
    }
    for (size_t i = 0; i < h_k.size(); ++i) {
        h_k[i] = __float2half(static_cast<float>(gen.symmetric(1.0)));
    }

    // Per-head sink logits, both signs, magnitudes up to ~3.
    std::vector<float> h_sink(kHeads);
    for (size_t h = 0; h < kHeads; ++h) {
        h_sink[h] = static_cast<float>((static_cast<int>(h % 7) - 3) * 0.9 + 0.25);
    }

    // --- Device buffers ------------------------------------------------------
    __half *d_q = nullptr, *d_k = nullptr, *d_out = nullptr;
    float* d_sink = nullptr;
    CHECK_HIP(hipMalloc(&d_q, h_q.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_k, h_k.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_out, h_q.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_sink, kHeads * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sink, h_sink.data(), kHeads * sizeof(float), hipMemcpyHostToDevice));

    auto run = [&](const std::vector<__half>& k, const std::vector<float>& sink) {
        CHECK_HIP(hipMemcpy(d_k, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_sink, sink.data(), sink.size() * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(d_out, 0, h_q.size() * sizeof(__half)));
        aeon::kernel::v4_sliding_window_attn_wave32_kernel<<<dim3(kHeads, kTokens), 32>>>(
            d_q, d_k, d_sink, d_out, static_cast<int>(kTokens), static_cast<int>(kWindow),
            static_cast<float>(kScale));
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<__half> out(h_q.size());
        CHECK_HIP(hipMemcpy(out.data(), d_out, out.size() * sizeof(__half), hipMemcpyDeviceToHost));
        return widen(out);
    };

    // Oracle for one position: build the participating key rows, run the
    // reference, return the token's [heads, head_dim] block.
    const std::vector<double> q_d = widen(h_q);
    auto oracle_at = [&](const std::vector<__half>& k, const std::vector<float>& sink,
                         size_t token) {
        const std::vector<double> k_d = widen(k);
        const size_t start = window_start(token);
        const size_t count = window_count(token);
        std::vector<double> sink_d(kHeads);
        for (size_t h = 0; h < kHeads; ++h) sink_d[h] = static_cast<double>(sink[h]);

        return aeon::reference::attention_scores_sink(
            slice_token(q_d, token, kHeads, kHeadDim), kHeads, kHeadDim,
            slice_rows(k_d, start, count, kHeadDim), count, sink_d, kScale);
    };

    // =======================================================================
    // Stage 1 — the three regimes the plan names
    // =======================================================================
    const std::vector<double> out = run(h_k, h_sink);

    ok &= report("attention @pos0 (1 key)", oracle_at(h_k, h_sink, kPosZero),
                 slice_token(out, kPosZero, kHeads, kHeadDim));
    ok &= report("attention @pos5 (6 keys, < window)", oracle_at(h_k, h_sink, kPosShort),
                 slice_token(out, kPosShort, kHeads, kHeadDim));
    ok &= report("attention @pos127 (exactly 128 keys)", oracle_at(h_k, h_sink, kPosAtWindow),
                 slice_token(out, kPosAtWindow, kHeads, kHeadDim));
    ok &= report("attention @pos149 (> window)", oracle_at(h_k, h_sink, kPosPast),
                 slice_token(out, kPosPast, kHeads, kHeadDim));

    // =======================================================================
    // Stage 2 — the window boundary is exact
    // =======================================================================
    {
        // Perturb a key that is inside the window at pos 5 and pos 127, and
        // outside it at pos 149.
        std::vector<__half> k_prime = h_k;
        for (size_t d = 0; d < kHeadDim; ++d) {
            k_prime[0 * kHeadDim + d] = __float2half(50.0f); // key row 0, huge
        }
        const std::vector<double> out_prime = run(k_prime, h_sink);

        auto block_diff = [&](size_t token) {
            const std::vector<double> a = slice_token(out, token, kHeads, kHeadDim);
            const std::vector<double> b = slice_token(out_prime, token, kHeads, kHeadDim);
            return aeon::reference::compare(a, b, 1.0).max_rel;
        };

        const double d5 = block_diff(kPosShort);
        const double d127 = block_diff(kPosAtWindow);
        const double d149 = block_diff(kPosPast);

        ok &= check("key 0 in window at pos5 -> changes", d5 > 1e-2,
                    "max_rel = " + std::to_string(d5));
        ok &= check("key 0 in window at pos127 -> changes", d127 > 1e-2,
                    "max_rel = " + std::to_string(d127));
        // pos 149 keeps keys 22..149, so key 0 must have zero influence. This is
        // the assertion that the window is `min(pos+1,128)` and not "everything
        // up to pos".
        ok &= check("key 0 out of window at pos149 -> no effect", d149 == 0.0,
                    "max_rel = " + std::to_string(d149) + " (must be exactly 0)");

        // And the perturbed run still matches the oracle at pos 149.
        ok &= report("attention @pos149 after out-of-window change",
                     oracle_at(k_prime, h_sink, kPosPast),
                     slice_token(out_prime, kPosPast, kHeads, kHeadDim));
    }

    // =======================================================================
    // Stage 3 — the sink lives in the denominator
    // =======================================================================
    {
        // Case A: a large NEGATIVE sink must have no effect — its weight
        // exp(sink - m) underflows to zero. A wrong implementation that let the
        // sink drive the max would instead put m at the sink and scale every
        // real weight by exp(score - sink) >> 1, saturating the denominator.
        std::vector<float> sink_neg(kHeads, -60.0f);
        const std::vector<double> out_neg = run(h_k, sink_neg);
        ok &= report("sink = -60 (inactive) vs oracle",
                     oracle_at(h_k, sink_neg, kPosPast),
                     slice_token(out_neg, kPosPast, kHeads, kHeadDim));

        // Compare against the no-sink-at-all output: they must coincide, because
        // exp(-60 - m) ~ 1e-26 contributes nothing.
        std::vector<float> sink_off(kHeads, -std::numeric_limits<float>::infinity());
        const std::vector<double> out_off = run(h_k, sink_off);
        const ErrorStats s = aeon::reference::compare(
            slice_token(out_off, kPosPast, kHeads, kHeadDim),
            slice_token(out_neg, kPosPast, kHeads, kHeadDim), 1.0);
        ok &= check("large negative sink == no sink", s.max_abs < 1e-6,
                    "max_abs = " + std::to_string(s.max_abs));

        // Case B: a large POSITIVE sink absorbs the mass and shrinks the output
        // to ~1e-25. This is the observable consequence of the sink being in the
        // denominator; a no-sink implementation would not shrink at all.
        //
        // Judged on ABSOLUTE error against the un-sunk scale. Both the kernel
        // and the reference produce values at the fp32 noise floor here, so a
        // relative comparison would be dividing one zero by another and would
        // report a meaningless max_rel = 1.0 — as this gate did on its first
        // run. The honest statement is "both are zero to within 1e-10 of the
        // scale the answer would otherwise have had".
        std::vector<float> sink_pos(kHeads, 60.0f);
        const std::vector<double> out_pos = run(h_k, sink_pos);

        std::vector<float> sink_off_tmp(kHeads, -std::numeric_limits<float>::infinity());
        const std::vector<double> out_off_tmp = run(h_k, sink_off_tmp);
        const double unsunk_scale = aeon::reference::peak_abs(
            slice_token(out_off_tmp, kPosPast, kHeads, kHeadDim));

        ok &= report_abs("sink = +60 (dominating) vs oracle",
                         oracle_at(h_k, sink_pos, kPosPast),
                         slice_token(out_pos, kPosPast, kHeads, kHeadDim),
                         unsunk_scale);

        const double peak_on = aeon::reference::peak_abs(slice_token(out_pos, kPosPast, kHeads, kHeadDim));
        const double peak_off = aeon::reference::peak_abs(slice_token(out_off, kPosPast, kHeads, kHeadDim));
        ok &= check("positive sink shrinks the output", peak_on < 1e-3 * peak_off,
                    "peak " + std::to_string(peak_on) + " << " + std::to_string(peak_off));

        // Case C: finite throughout — no NaN or inf from either extreme.
        auto all_finite = [](const std::vector<double>& v) {
            for (double x : v) if (!std::isfinite(x)) return false;
            return true;
        };
        ok &= check("output finite at both sink extremes",
                    all_finite(out_pos) && all_finite(out_neg),
                    "no inf/NaN at sink = +/-60");
    }

    // =======================================================================
    // Stage 4 — the scale is 1/sqrt(512)
    // =======================================================================
    {
        // An output built with the wrong scale must differ materially from ours.
        const std::vector<double> k_d = widen(h_k);
        const size_t start = window_start(kPosPast);
        const size_t count = window_count(kPosPast);
        std::vector<double> sink_d(kHeads);
        for (size_t h = 0; h < kHeads; ++h) sink_d[h] = static_cast<double>(h_sink[h]);

        const std::vector<double> wrong_index_scale = aeon::reference::attention_scores_sink(
            slice_token(q_d, kPosPast, kHeads, kHeadDim), kHeads, kHeadDim,
            slice_rows(k_d, start, count, kHeadDim), count, sink_d,
            1.0 / std::sqrt(128.0));
        const std::vector<double> wrong_rope_scale = aeon::reference::attention_scores_sink(
            slice_token(q_d, kPosPast, kHeads, kHeadDim), kHeads, kHeadDim,
            slice_rows(k_d, start, count, kHeadDim), count, sink_d,
            1.0 / std::sqrt(64.0));

        const std::vector<double> got = slice_token(out, kPosPast, kHeads, kHeadDim);
        const double d_index = aeon::reference::compare(wrong_index_scale, got, 1.0).max_rel;
        const double d_rope = aeon::reference::compare(wrong_rope_scale, got, 1.0).max_rel;

        ok &= check("scale is 1/sqrt(512), not 1/sqrt(128)", d_index > 1e-2,
                    "index-head scale would differ by " + std::to_string(d_index));
        ok &= check("scale is 1/sqrt(512), not 1/sqrt(64)", d_rope > 1e-2,
                    "rope-part scale would differ by " + std::to_string(d_rope));
    }

    // =======================================================================
    // Stage 5 — V = K, and the output is a bounded combination of key rows
    // =======================================================================
    {
        const std::vector<double> got = slice_token(out, kPosPast, kHeads, kHeadDim);
        const std::vector<double> k_d = widen(h_k);
        const size_t start = window_start(kPosPast);
        const size_t count = window_count(kPosPast);

        // Every output element must lie within the range spanned by the
        // in-window key rows at that dimension. Scores are normalised, so an
        // implementation that skipped the division would blow straight past it.
        double worst_excess = 0.0;
        for (size_t h = 0; h < kHeads; ++h) {
            for (size_t d = 0; d < kHeadDim; ++d) {
                double lo = 0.0, hi = 0.0; // sink: the caller may leave mass unallocated
                for (size_t j = 0; j < count; ++j) {
                    const double kv = k_d[(start + j) * kHeadDim + d];
                    lo = std::fmin(lo, kv);
                    hi = std::fmax(hi, kv);
                }
                const double v = got[h * kHeadDim + d];
                worst_excess = std::fmax(worst_excess, std::fmax(v - hi, lo - v));
            }
        }
        ok &= check("output inside the key-row hull (normalised)", worst_excess < 1e-3,
                    "max excursion beyond hull = " + std::to_string(worst_excess));
    }

    // =======================================================================
    // Stage 6 — oracle self-checks
    // =======================================================================
    {
        const std::vector<double> k_d = widen(h_k);
        const size_t start = window_start(kPosPast);
        const size_t count = window_count(kPosPast);
        std::vector<double> sink_d(kHeads);
        for (size_t h = 0; h < kHeads; ++h) sink_d[h] = static_cast<double>(h_sink[h]);
        const std::vector<double> q_t = slice_token(q_d, kPosPast, kHeads, kHeadDim);
        const std::vector<double> k_win = slice_rows(k_d, start, count, kHeadDim);

        // A: the two readings of the sink — "a term in the denominator" vs "one
        //    more key whose value row is zero" — must agree, because the plan
        //    asserts they are numerically identical.
        {
            const std::vector<double> a = aeon::reference::attention_scores_sink(
                q_t, kHeads, kHeadDim, k_win, count, sink_d, kScale);
            const std::vector<double> b = aeon::reference::attention_sink_as_zero_value_key(
                q_t, kHeads, kHeadDim, k_win, count, sink_d, kScale);
            const ErrorStats s = aeon::reference::compare(a, b, 1e-9);
            ok &= check("oracle: sink reading == zero-value key row", s.max_rel < 1e-12,
                        "max_rel between the two readings = " + std::to_string(s.max_rel));
        }

        // B: THE SINK IS A PURE RESCALING. Because it contributes no value, the
        //    sink output must equal the no-sink output multiplied by one scalar,
        //    elementwise. Asserting that the ratio is *constant across every
        //    element* is exactly what "denominator only, no value" means, and it
        //    is the strongest form available: a value contribution would make
        //    the ratio vary per element, and a dropped sink would make it 1.
        //
        //    Requires `sink <= max_j score_j` so both computations share a frame;
        //    -20 is far below the ~+2..3 max score of 128 random 512-dim dots.
        {
            const std::vector<double> sink_low(kHeads, -20.0);
            const std::vector<double> with_sink = aeon::reference::attention_scores_sink(
                q_t, kHeads, kHeadDim, k_win, count, sink_low, kScale);
            const std::vector<double> no_sink = aeon::reference::attention_sink_as_zero_value_key(
                q_t, kHeads, kHeadDim, k_win, count,
                std::vector<double>(kHeads, -std::numeric_limits<double>::infinity()), kScale);

            double ratio_min = std::numeric_limits<double>::infinity();
            double ratio_max = 0.0;
            for (size_t i = 0; i < with_sink.size(); ++i) {
                if (std::fabs(no_sink[i]) < 1e-9) continue; // ratio undefined at zero
                const double r = with_sink[i] / no_sink[i];
                ratio_min = std::fmin(ratio_min, r);
                ratio_max = std::fmax(ratio_max, r);
            }
            const double spread = ratio_max - ratio_min;
            ok &= check("oracle: sink is a single scalar rescale", spread < 1e-9,
                        "ratio spread over all elements = " + std::to_string(spread));
        }

        // C: with no keys at all, every weight is zero, so the output must be
        //    exactly zero — not NaN, and not a division blow-up.
        {
            const std::vector<double> no_keys = aeon::reference::attention_scores_sink(
                q_t, kHeads, kHeadDim, {}, 0, sink_d, kScale);
            double worst = 0.0;
            for (double v : no_keys) worst = std::fmax(worst, std::fabs(v));
            ok &= check("oracle: empty key set -> exactly zero", worst == 0.0,
                        "max |out| with zero keys = " + std::to_string(worst));
        }
    }

    CHECK_HIP(hipFree(d_q));
    CHECK_HIP(hipFree(d_k));
    CHECK_HIP(hipFree(d_out));
    CHECK_HIP(hipFree(d_sink));

    std::cout << (ok ? "[SUCCESS] Attention gate passed.\n"
                     : "[FAILURE] Attention gate failed.\n");
    return ok ? 0 : 1;
}
