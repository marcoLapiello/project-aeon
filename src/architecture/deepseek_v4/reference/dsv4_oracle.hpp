#pragma once

// -----------------------------------------------------------------------------
// DeepSeek-V4 oracle: independent, host-only, double-precision references.
//
// These functions are the *specification written as arithmetic*. They exist so a
// kernel can be compared against something that shares no code with it — see the
// inference pipeline plan, Part V, "Anti-circularity rule".
//
// Binding rules for this header:
//
//   1. HOST ONLY. No HIP/device include and no kernel include. It must compile
//      with a plain host compiler, so it can be read and audited as an oracle.
//
//   2. INDEPENDENT. Every function is written from the specification and the
//      cited upstream references, never from the kernel under test. If a kernel
//      helper and an oracle helper are the same code, the comparison proves
//      self-consistency and nothing else.
//
//   3. ACCUMULATE IN DOUBLE. The kernels accumulate in float; using double here
//      makes the measured delta a statement about the kernel's rounding rather
//      than about the reference's.
//
// Each reference states the plan step it implements. The plan is authoritative
// for semantics; if this file and the plan disagree, the plan wins and this file
// is the bug.
// -----------------------------------------------------------------------------

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aeon::reference {

// ---------------------------------------------------------------------------
// Deterministic inputs
// ---------------------------------------------------------------------------

// Small, allocation-free PRNG so every gate runs on reproducible data. It is not
// cryptographic and is not meant to be; it exists so a failing gate can be
// replayed byte-for-byte from its seed.
class Rng {
public:
    explicit Rng(uint64_t seed) noexcept
        : state_(seed != 0 ? seed : 0x9E3779B97F4A7C15ull) {}

    // SplitMix64: one multiply-xor round per draw, good distribution.
    uint64_t next_u64() noexcept {
        state_ += 0x9E3779B97F4A7C15ull;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // Uniform in [0, 1).
    double uniform01() noexcept {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }

    // Uniform in [-scale, scale).
    double symmetric(double scale) noexcept {
        return (uniform01() * 2.0 - 1.0) * scale;
    }

    std::vector<double> vector_filled(size_t n, double scale) {
        std::vector<double> v(n);
        for (double& x : v) x = symmetric(scale);
        return v;
    }

private:
    uint64_t state_;
};

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

struct ErrorStats {
    double max_abs{0.0};
    double max_rel{0.0};
    double rms_abs{0.0};
    size_t max_abs_index{0};
    size_t max_rel_index{0};
    bool   size_mismatch{false};
};

// Compares an oracle vector against a kernel result, elementwise.
//
// The relative-error denominator is floored at `rel_floor_fraction * max|want|`.
// That is a *reporting* choice, not a pass/fail threshold: a relative error is
// meaningless for an element whose magnitude is at the noise floor, so those
// elements are judged against a fraction of the data's scale instead. The gate
// decides the actual tolerance from the returned stats.
inline ErrorStats compare(const std::vector<double>& want,
                          const std::vector<double>& got,
                          double rel_floor_fraction = 1e-3) {
    ErrorStats stats;
    if (want.size() != got.size()) {
        stats.size_mismatch = true;
        return stats;
    }

    double scale = 0.0;
    for (double w : want) scale = std::fmax(scale, std::fabs(w));
    const double floor = rel_floor_fraction * scale;

    double sum_sq = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = std::fabs(want[i] - got[i]);
        if (d > stats.max_abs) {
            stats.max_abs = d;
            stats.max_abs_index = i;
        }
        const double denom = std::fmax(std::fabs(want[i]), floor);
        const double rel = d / denom;
        if (rel > stats.max_rel) {
            stats.max_rel = rel;
            stats.max_rel_index = i;
        }
        sum_sq += d * d;
    }
    stats.rms_abs = want.empty()
        ? 0.0
        : std::sqrt(sum_sq / static_cast<double>(want.size()));
    return stats;
}

// A step is certified when every element is inside both the absolute and the
// relative tolerance. Stated as one predicate so the gate's decision is a single
// expression rather than a scattered `if`.
inline bool within(const ErrorStats& s, double tol_abs, double tol_rel) {
    return !s.size_mismatch && s.max_abs <= tol_abs && s.max_rel <= tol_rel;
}

// ---------------------------------------------------------------------------
// RMSNorm — Step 2.1 (attention norm), 2.6 (FFN norm), 2.10 (pre-norm),
//          Step 3 (final norm), and the Q/KV path norms in 2.2
// ---------------------------------------------------------------------------

// out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i], accumulated in double.
//
// The graph applies the weighted form at least to the attention RMSNorm (2.1),
// whose spec is literally `x / sqrt(mean(x^2) + eps) * weight` over 4096 with
// fp32 accumulate. `rmsnorm_unit` is the weightless form; whether a given site
// carries a learned scale is a per-site decision recorded in the plan, not
// something this header assumes.
inline std::vector<double> rmsnorm(const std::vector<double>& x,
                                   const std::vector<double>& weight,
                                   double eps) {
    const size_t dim = x.size();
    double sum_sq = 0.0;
    for (double v : x) sum_sq += v * v;
    const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + eps);

    std::vector<double> out(dim);
    for (size_t i = 0; i < dim; ++i) out[i] = x[i] * inv_rms * weight[i];
    return out;
}

// Weightless variant: out[i] = x[i] * rsqrt(mean(x^2) + eps).
inline std::vector<double> rmsnorm_unit(const std::vector<double>& x, double eps) {
    const size_t dim = x.size();
    double sum_sq = 0.0;
    for (double v : x) sum_sq += v * v;
    const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + eps);

    std::vector<double> out(dim);
    for (size_t i = 0; i < dim; ++i) out[i] = x[i] * inv_rms;
    return out;
}

} // namespace aeon::reference
