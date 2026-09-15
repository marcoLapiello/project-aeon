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

#include <algorithm>
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

// ---------------------------------------------------------------------------
// RoPE — Step 2.3
//
// Deliberately *not* parameterised by "which of the two upstream classes am I".
// The only knob is the spec, and there are exactly two specs in the graph, so a
// caller cannot silently invent a third. (trap 7)
// ---------------------------------------------------------------------------

// The two RoPE classes the graph actually has, keyed on `compress_ratio`:
//   ratio <= 1 -> Sliding:    theta = rope_theta          = 10000,  plain RoPE
//   ratio >  1 -> Compressed: theta = compress_rope_theta = 160000, YaRN factor 16
enum class RopeClass { Sliding, Compressed };

inline RopeClass rope_class_for_ratio(int64_t compress_ratio) noexcept {
    // Upstream keys on `compress_ratio > 1` [rope.py:28-30], not `!= 0` — the two
    // agree for our {0, 4, 128}, and `> 1` is the one the reference uses.
    return compress_ratio > 1 ? RopeClass::Compressed : RopeClass::Sliding;
}

struct RopeSpec {
    uint32_t head_dim{512};
    uint32_t rotary_dim{64};   // only the tail rotates
    double   theta{10000.0};
    double   factor{1.0};      // 1.0 disables YaRN entirely
    double   beta_fast{32.0};
    double   beta_slow{1.0};
    uint32_t original_max_position{65536};
};

inline RopeSpec rope_spec_for(RopeClass cls) noexcept {
    RopeSpec spec;
    if (cls == RopeClass::Compressed) {
        spec.theta = 160000.0;
        spec.factor = 16.0;
    }
    // Sliding keeps factor = 1.0. Upstream still routes it through
    // `deepseek_yarn`, but with factor 1.0 the interpolation and extrapolation
    // frequencies coincide, so the ramp cancels and the result is plain RoPE
    // [V vllm/models/deepseek_v4/common/rope.py:31-44].
    return spec;
}

// Inverse frequencies, length `rotary_dim / 2`.
//
// Plain form: `1 / theta^(2k/rotary_dim)` — note the exponent step is `2k/dim`,
// not `k/(dim/2)`, which is what makes the pair stride 2.
//
// YaRN form [V vllm/.../rotary_embedding/deepseek_scaling_rope.py:78-107,
//           V vllm/.../rotary_embedding/common.py:25-70]:
//   low  = floor(correction_dim(beta_fast))          clamp >= 0
//   high = ceil (correction_dim(beta_slow))          clamp <= rotary_dim - 1
//   w_k  = clamp((k - low) / (high - low), 0, 1)
//   inv_k = (1 - w_k) * inv_k + w_k * (inv_k / factor)
// with correction_dim(r) = rotary_dim * ln(orig_max_pos / (r * 2pi)) / (2 ln theta).
// Low k (high frequency) keeps the original; high k is interpolated.
//
// Note on the clamp: the reference clamps `high` to `rotary_dim - 1` = 63, while
// the 32-element ramp it feeds is indexed only to 31. For this model's
// parameters (compressed: low 15, high 25; sliding: no ramp) the clamp never
// binds, so the two agree. It is called out because it *would* diverge for a
// checkpoint whose correction range extended past `rotary_dim / 2`.
inline std::vector<double> rope_inv_freq(const RopeSpec& spec) {
    const size_t half = spec.rotary_dim / 2;
    std::vector<double> inv(half);
    for (size_t k = 0; k < half; ++k) {
        inv[k] = 1.0 / std::pow(spec.theta, (2.0 * static_cast<double>(k)) / spec.rotary_dim);
    }
    if (spec.factor <= 1.0) return inv;

    constexpr double kPi = 3.14159265358979323846;
    const auto correction_dim = [&spec](double rotations) {
        return static_cast<double>(spec.rotary_dim) *
               std::log(static_cast<double>(spec.original_max_position) / (rotations * 2.0 * kPi)) /
               (2.0 * std::log(spec.theta));
    };

    const double low = std::fmax(0.0, std::floor(correction_dim(spec.beta_fast)));
    const double high = std::fmin(static_cast<double>(spec.rotary_dim) - 1.0,
                                  std::ceil(correction_dim(spec.beta_slow)));

    for (size_t k = 0; k < half; ++k) {
        double w = 0.0;
        if (low >= high) {
            w = (static_cast<double>(k) < low) ? 0.0 : 1.0;
        } else {
            w = std::clamp((static_cast<double>(k) - low) / (high - low), 0.0, 1.0);
        }
        inv[k] = (1.0 - w) * inv[k] + w * (inv[k] / spec.factor);
    }
    return inv;
}

// Reference cos/sin tables, laid out `[max_position][rotary_dim / 2]` so a gate
// can compare them directly against the kernel's flat caches.
struct RopeTableRef {
    RopeSpec spec;
    std::vector<double> cos;
    std::vector<double> sin;

    uint32_t half() const noexcept { return spec.rotary_dim / 2; }
};

inline RopeTableRef rope_table(const RopeSpec& spec, uint32_t max_position) {
    RopeTableRef table;
    table.spec = spec;
    const std::vector<double> inv = rope_inv_freq(spec);
    const uint32_t half = spec.rotary_dim / 2;

    table.cos.assign(static_cast<size_t>(max_position) * half, 0.0);
    table.sin.assign(static_cast<size_t>(max_position) * half, 0.0);
    for (uint32_t p = 0; p < max_position; ++p) {
        for (uint32_t k = 0; k < half; ++k) {
            // Accumulated in double. The kernel computes this in float; at large
            // positions that difference is real and is expected — see the gate.
            const double angle = static_cast<double>(p) * inv[k];
            const size_t at = static_cast<size_t>(p) * half + k;
            table.cos[at] = std::cos(angle);
            table.sin[at] = std::sin(angle);
        }
    }
    return table;
}

// Rotates the tail `rotary_dim` of a single head row **in place**. The row is
// `head_dim` wide; the leading `head_dim - rotary_dim` (= 448) entries are the
// nope part and are never touched. (trap 27)
//
// GPT-J interleave: the pairs are adjacent, `(rotary + 2k, rotary + 2k + 1)`.
//   forward: out[2k]   = x[2k]*cos - x[2k+1]*sin
//            out[2k+1] = x[2k]*sin + x[2k+1]*cos
//   inverse: out[2k]   = x[2k]*cos + x[2k+1]*sin
//            out[2k+1] = x[2k+1]*cos - x[2k]*sin
// i.e. the inverse is the transpose of the forward rotation, equivalently the
// forward rotation with `sin` negated [V deepseek_scaling_rope.py:281-284].
inline void rope_apply_tail(std::vector<double>& row,
                            const RopeTableRef& table,
                            uint32_t position,
                            bool inverse) {
    const uint32_t half = table.half();
    const uint32_t rope_offset = table.spec.head_dim - table.spec.rotary_dim;

    for (uint32_t k = 0; k < half; ++k) {
        const size_t pair = static_cast<size_t>(position) * half + k;
        const double c = table.cos[pair];
        const double s = table.sin[pair];

        const size_t i0 = rope_offset + 2 * k;
        const size_t i1 = i0 + 1;
        const double x0 = row[i0];
        const double x1 = row[i1];

        if (inverse) {
            row[i0] = x0 * c + x1 * s;
            row[i1] = x1 * c - x0 * s;
        } else {
            row[i0] = x0 * c - x1 * s;
            row[i1] = x0 * s + x1 * c;
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience
// ---------------------------------------------------------------------------

// Peak magnitude of a vector. Used as a comparison denominator and as a way to
// state a tolerance as "a fraction of the data's own scale".
inline double peak_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::fmax(m, std::fabs(x));
    return m;
}

// ---------------------------------------------------------------------------
// Dense projections — Step 2.2 (MLA), and every other `y = W @ x` in the graph
// ---------------------------------------------------------------------------

// `y[o] = Σ_i W[o, i] · x[i]`, with `W` given as `[out_dim, in_dim]` **row
// major** — the checkpoint's `nn.Linear.weight` layout, which the GEMV kernel
// consumes directly with no transpose.
//
// The element accessor is a callable `double(size_t o, size_t i)` rather than a
// materialised matrix, so a gate can feed the fp16 weights through without first
// widening 33 million values into doubles. The arithmetic — the accumulation
// order, and the fact that it happens in double — stays here, in the oracle,
// not in the test.
template <class Accessor>
std::vector<double> matvec(size_t out_dim, size_t in_dim,
                           const std::vector<double>& x,
                           Accessor w_at) {
    std::vector<double> y(out_dim, 0.0);
    for (size_t o = 0; o < out_dim; ++o) {
        double acc = 0.0;
        for (size_t i = 0; i < in_dim; ++i) acc += w_at(o, i) * x[i];
        y[o] = acc;
    }
    return y;
}

// ---------------------------------------------------------------------------
// MLA Q/KV paths — Step 2.2
//
// Multi-head Latent Attention with a low-rank Q path and a single shared KV
// head. Two things this encodes that a "collapse the matmuls" implementation
// loses (trap 5), and one it must not invent (trap 6):
//
//   Q:  x -> wq_a [4096 -> 1024] -> q_norm (weighted, over 1024)
//          -> wq_b [1024 -> 64*512] -> per-head norm (WEIGHTLESS, over 512)
//   KV: x -> wkv  [4096 -> 512]  -> kv_norm (weighted, over 512)
//
// The per-head Q norm is weightless. Cited two ways: upstream's
// `fused_q_norm_rope(q_input, q_output, eps, freqs_cis, positions)` takes no
// weight argument, and the checkpoint declares exactly
// `attn.wq_a` / `attn.q_norm` / `attn.wq_b` / `attn.wkv` / `attn.kv_norm` —
// there is no per-head norm tensor to load. The pipeline already uses the
// weightless kernel, so kernel and contract agree.
//
// There is no separate V: `kv` is a single 512-wide row used as both key and
// value (trap 6). Nothing here produces a second tensor, and a gate should fail
// if one appears.
// ---------------------------------------------------------------------------

struct MlaQPath {
    std::vector<double> q_lora;      // [q_lora_rank]  — after wq_a
    std::vector<double> q_lora_norm; // [q_lora_rank]  — after the weighted norm
    std::vector<double> q;           // [num_heads * head_dim] — after per-head norm
};

// `weights_are_fp16_rounded` is not a flag — the accessors are expected to yield
// exactly the values the kernel reads. Feed this the fp16-rounded intermediates
// so the delta measures the kernel, not the input quantization.
template <class WqA, class WqB>
MlaQPath mla_q_path(const std::vector<double>& x_norm,
                    const std::vector<double>& q_norm_weight,
                    size_t q_lora_rank, size_t num_heads, size_t head_dim,
                    double eps,
                    WqA wq_a, WqB wq_b) {
    const size_t hidden = x_norm.size();
    MlaQPath out;

    out.q_lora = matvec(q_lora_rank, hidden, x_norm, wq_a);
    out.q_lora_norm = rmsnorm(out.q_lora, q_norm_weight, eps);
    out.q = matvec(num_heads * head_dim, q_lora_rank, out.q_lora_norm, wq_b);

    // Per-head weightless RMSNorm over head_dim. Applied in place, head by head.
    for (size_t h = 0; h < num_heads; ++h) {
        const size_t base = h * head_dim;
        double sum_sq = 0.0;
        for (size_t d = 0; d < head_dim; ++d) {
            const double v = out.q[base + d];
            sum_sq += v * v;
        }
        const double inv = 1.0 / std::sqrt(sum_sq / static_cast<double>(head_dim) + eps);
        for (size_t d = 0; d < head_dim; ++d) out.q[base + d] *= inv;
    }
    return out;
}

// KV: `x -> wkv -> weighted norm over head_dim`. One row, used as both K and V.
template <class Wkv>
std::vector<double> mla_kv_path(const std::vector<double>& x_norm,
                                const std::vector<double>& kv_norm_weight,
                                size_t head_dim, double eps, Wkv wkv) {
    return rmsnorm(matvec(head_dim, x_norm.size(), x_norm, wkv), kv_norm_weight, eps);
}

} // namespace aeon::reference
