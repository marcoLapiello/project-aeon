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
#include <cstring>
#include <limits>
#include <stdexcept>
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

// ---------------------------------------------------------------------------
// Hyper-Connections — Step 2.0 (pre-mix + Sinkhorn), 2.7 (post expansion)
//
// Written directly from the upstream reference, which is the best arbiter here
// because the comb's index convention is easy to get backwards and a transposed
// comb is still a plausible-looking doubly-stochastic matrix.
//
//   [V vllm/model_executor/kernels/mhc/torch.py:6-93  `mhc_pre_torch`]
//   [V vllm/model_executor/kernels/mhc/torch.py:96-108 `mhc_post_torch`]
//
// THE COMB INDEX CONVENTION, stated once and unambiguously, because the plan's
// own 2.0 prose gets it backwards (see the correction note in the plan):
//
//   Let `C` be the 4x4 comb, flattened `C[i * hc_mult + j]`. Then
//
//     * `i` is the CONTRACTION index — the incoming residual stream being read.
//     * `j` is the OUTPUT index — the outgoing residual stream being written.
//
//   and the post expansion is `out[j][h] = Σ_i C[i][j] · residual[i][h]`.
//
// That is exactly upstream's `torch.einsum("...ij,...ih->...jh", comb, residual)`:
// the comb's first axis is contracted against the residual's stream axis, and the
// comb's second axis becomes the output stream. The logits, the softmax axis and
// the normalization axis all follow from it:
//
//   logits  C[i][j] = mixes[2·hc + i·hc + j] · scale[2] + base[2·hc + i·hc + j]
//   C ← softmax(C, axis = j) + sinkhorn_eps     <- the OUTPUT axis (upstream dim=-1)
//   C ← C / (Σ_i C[i][j] + sinkhorn_eps)        <- the CONTRACTION axis (dim=-2)
//   then (iters − 1) × ( normalize over j, then normalize over i )
//
// Every denominator carries `hc_sinkhorn_eps`, not just the row ones.
// ---------------------------------------------------------------------------

inline double sigmoid(double x) noexcept {
    return 1.0 / (1.0 + std::exp(-x));
}

struct HcParams {
    double rms_eps{1e-6};
    double pre_eps{1e-6};
    double sinkhorn_eps{1e-6};
    double post_mult{2.0};   // hc_post_mult_value: a hardcoded constant, not a config key
    int sinkhorn_iters{20};
};

struct HcPreResult {
    std::vector<double> mixes;       // [hc_mults3] = 24
    std::vector<double> pre_mix;     // [hc_mult]  = 4
    std::vector<double> post_mix;    // [hc_mult]  = 4
    std::vector<double> comb;        // [hc_mult * hc_mult] flat, C[contraction * hc + output]
    std::vector<double> layer_input; // [hidden]
};

// `mixes[m] = (x · fn[m]) · rsqrt(mean(x²) + rms_eps)` over the flattened
// `hc_mult × hidden` residual. The RMS is over the *flattened* dimension, not per
// stream, and it scales the projection output rather than the input — algebraically
// the same thing, and the form upstream uses.
template <class Fn>
std::vector<double> hc_mixes(const std::vector<double>& residual,
                             size_t hc_mult3, Fn fn_at, size_t hc_hidden,
                             double rms_eps) {
    double sum_sq = 0.0;
    for (double v : residual) sum_sq += v * v;
    const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(hc_hidden) + rms_eps);

    std::vector<double> mixes(hc_mult3, 0.0);
    for (size_t m = 0; m < hc_mult3; ++m) {
        double dot = 0.0;
        for (size_t k = 0; k < hc_hidden; ++k) dot += residual[k] * fn_at(m, k);
        mixes[m] = dot * inv_rms;
    }
    return mixes;
}

// Applies the scale/base transform, the axis-wise softmax, and the Sinkhorn
// iterations. Split out from `hc_mixes` so a gate can feed it the kernel's own
// `mixes` and isolate this step from the projection.
inline void hc_sinkhorn(const std::vector<double>& mixes,
                        const std::vector<double>& hc_scale, // [3]
                        const std::vector<double>& hc_base,  // [hc_mult3]
                        size_t hc_mult, const HcParams& p, HcPreResult& out) {
    const size_t hc_mult2 = hc_mult * hc_mult;

    out.pre_mix.resize(hc_mult);
    for (size_t j = 0; j < hc_mult; ++j) {
        out.pre_mix[j] = sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + p.pre_eps;
    }

    out.post_mix.resize(hc_mult);
    for (size_t j = 0; j < hc_mult; ++j) {
        // No eps here — only the pre-mix carries `hc_pre_eps`.
        out.post_mix[j] = sigmoid(mixes[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]) * p.post_mult;
    }

    // C[i][j]: i contracts the residual streams, j is the output stream.
    std::vector<double>& C = out.comb;
    C.assign(hc_mult2, 0.0);
    for (size_t i = 0; i < hc_mult; ++i) {
        for (size_t j = 0; j < hc_mult; ++j) {
            const size_t idx = 2 * hc_mult + i * hc_mult + j;
            C[i * hc_mult + j] = mixes[idx] * hc_scale[2] + hc_base[idx];
        }
    }

    // softmax over j (the OUTPUT axis) — upstream `dim=-1`.
    for (size_t i = 0; i < hc_mult; ++i) {
        double max_v = C[i * hc_mult];
        for (size_t j = 1; j < hc_mult; ++j) max_v = std::fmax(max_v, C[i * hc_mult + j]);
        double sum_e = 0.0;
        for (size_t j = 0; j < hc_mult; ++j) {
            const double e = std::exp(C[i * hc_mult + j] - max_v);
            C[i * hc_mult + j] = e;
            sum_e += e;
        }
        for (size_t j = 0; j < hc_mult; ++j) C[i * hc_mult + j] = C[i * hc_mult + j] / sum_e + p.sinkhorn_eps;
    }

    // normalize over i (the CONTRACTION axis) — upstream `dim=-2`.
    auto normalize_over_i = [&]() {
        for (size_t j = 0; j < hc_mult; ++j) {
            double col_sum = 0.0;
            for (size_t i = 0; i < hc_mult; ++i) col_sum += C[i * hc_mult + j];
            const double inv = 1.0 / (col_sum + p.sinkhorn_eps);
            for (size_t i = 0; i < hc_mult; ++i) C[i * hc_mult + j] *= inv;
        }
    };
    auto normalize_over_j = [&]() {
        for (size_t i = 0; i < hc_mult; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < hc_mult; ++j) row_sum += C[i * hc_mult + j];
            const double inv = 1.0 / (row_sum + p.sinkhorn_eps);
            for (size_t j = 0; j < hc_mult; ++j) C[i * hc_mult + j] *= inv;
        }
    };

    normalize_over_i();
    for (int it = 0; it < p.sinkhorn_iters - 1; ++it) {
        normalize_over_j();
        normalize_over_i();
    }
}

// `layer_input[h] = Σ_j pre_mix[j] · residual[j][h]` — the pre-combine.
inline std::vector<double> hc_pre_combine(const std::vector<double>& residual,
                                          const std::vector<double>& pre_mix,
                                          size_t hidden) {
    const size_t hc_mult = pre_mix.size();
    std::vector<double> out(hidden, 0.0);
    for (size_t h = 0; h < hidden; ++h) {
        double acc = 0.0;
        for (size_t j = 0; j < hc_mult; ++j) acc += pre_mix[j] * residual[j * hidden + h];
        out[h] = acc;
    }
    return out;
}

// `out[j][h] = Σ_i C[i][j] · residual[i][h] + post_mix[j] · layer_out[h]`.
//
// `transpose_comb` is not a feature — it exists so a gate can *demonstrate* that
// the convention matters by computing the wrong reading deliberately. Nothing in
// the graph should ever pass `true`.
inline std::vector<double> hc_post(const std::vector<double>& layer_out,
                                   const std::vector<double>& residual,
                                   const std::vector<double>& post_mix,
                                   const std::vector<double>& C,
                                   size_t hidden,
                                   bool transpose_comb = false) {
    const size_t hc_mult = post_mix.size();
    std::vector<double> out(hc_mult * hidden, 0.0);
    for (size_t j = 0; j < hc_mult; ++j) {
        for (size_t h = 0; h < hidden; ++h) {
            double acc = post_mix[j] * layer_out[h];
            for (size_t i = 0; i < hc_mult; ++i) {
                const double c = transpose_comb ? C[j * hc_mult + i] : C[i * hc_mult + j];
                acc += c * residual[i * hidden + h];
            }
            out[j * hidden + h] = acc;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Attention score + sink + softmax — Step 2.4.1
//
// `out[h][d] = Σ_j p_j · k[j][d] / l` with
//
//   s_j = (q[h] · k[j]) · scale          scale = 1/sqrt(head_dim) = 1/sqrt(512)
//   m   = max( max_j s_j , sink[h] )     <- the sink enters the MAX
//   p_j = exp(s_j - m)
//   l   = Σ_j p_j + exp(sink[h] - m)     <- the sink enters the DENOMINATOR
//
// The sink contributes **no value**: there is no `(p_sink · value)` term, so it
// absorbs probability mass and nothing else. Upstream describes it as *"a virtual
// extra K with V=0"* `[V sglang .../dsv4/unified_kv_kernels/paged_prefill.py:194-203]`.
//
// There is no separate value tensor: the same `k` rows are used as values
// (trap 6).
//
// Note on the max. Including the sink in `m` is what keeps every exponent in
// `exp(·) ≤ 0` when the sink dominates. It is a numerical-robustness property and
// it is **not observable in the output**: in the regime where it matters (sink
// far above every score) all mass sits on the sink and the output is zero whether
// or not the sink was included in the max. A gate therefore cannot discriminate
// on this by comparing outputs, and should not pretend to — see the note in the
// attention gate.
// ---------------------------------------------------------------------------

inline std::vector<double> attention_scores_sink(
    const std::vector<double>& q, size_t num_heads, size_t head_dim,
    const std::vector<double>& k, size_t num_keys,
    const std::vector<double>& sink, double scale) {
    std::vector<double> out(num_heads * head_dim, 0.0);

    for (size_t h = 0; h < num_heads; ++h) {
        const double* qh = q.data() + h * head_dim;

        std::vector<double> score(num_keys);
        double m = sink[h];
        for (size_t j = 0; j < num_keys; ++j) {
            const double* kj = k.data() + j * head_dim;
            double dot = 0.0;
            for (size_t d = 0; d < head_dim; ++d) dot += qh[d] * kj[d];
            score[j] = dot * scale;
            m = std::fmax(m, score[j]);
        }

        double l = std::exp(sink[h] - m);
        std::vector<double> p(num_keys);
        for (size_t j = 0; j < num_keys; ++j) {
            p[j] = std::exp(score[j] - m);
            l += p[j];
        }
        const double denom = std::fmax(l, 1e-30);

        double* oh = out.data() + h * head_dim;
        for (size_t d = 0; d < head_dim; ++d) {
            double acc = 0.0;
            for (size_t j = 0; j < num_keys; ++j) acc += p[j] * k[j * head_dim + d];
            oh[d] = acc / denom;
        }
    }
    return out;
}

// The identical quantity written the other way: the sink is simply one more key
// whose **value row is zero**. The plan states these two readings are numerically
// identical; this function exists so a gate can confirm that rather than assume
// it. (They are: the extra term contributes `p_sink · 0 = 0` to every numerator,
// and `p_sink` to the denominator, under the same max.)
inline std::vector<double> attention_sink_as_zero_value_key(
    const std::vector<double>& q, size_t num_heads, size_t head_dim,
    const std::vector<double>& k, size_t num_keys,
    const std::vector<double>& sink, double scale) {
    std::vector<double> out(num_heads * head_dim, 0.0);

    for (size_t h = 0; h < num_heads; ++h) {
        const double* qh = q.data() + h * head_dim;

        // num_keys real scores, then the sink as an extra entry.
        std::vector<double> score(num_keys + 1);
        double m = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < num_keys; ++j) {
            const double* kj = k.data() + j * head_dim;
            double dot = 0.0;
            for (size_t d = 0; d < head_dim; ++d) dot += qh[d] * kj[d];
            score[j] = dot * scale;
            m = std::fmax(m, score[j]);
        }
        score[num_keys] = sink[h];
        m = std::fmax(m, score[num_keys]);

        double l = 0.0;
        std::vector<double> p(num_keys + 1);
        for (size_t j = 0; j <= num_keys; ++j) {
            p[j] = std::exp(score[j] - m);
            l += p[j];
        }
        const double denom = std::fmax(l, 1e-30);

        double* oh = out.data() + h * head_dim;
        for (size_t d = 0; d < head_dim; ++d) {
            double acc = 0.0;
            for (size_t j = 0; j < num_keys; ++j) acc += p[j] * k[j * head_dim + d];
            // The sink row's value is zero, so it adds nothing to `acc`.
            oh[d] = acc / denom;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Compressor — Step 2.4.2
//
// Reference: `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`
// (`compress_norm_rope_store_triton` → the softmax + weighted sum, then RMSNorm)
// and `.../ops/save_partial_states.py` (the APE add).
//
// The compressor turns every `ratio` tokens into one compressed row. The pieces
// that are easy to get wrong, in the order they bite:
//
//   1. THE APE GOES ON `score` AND ONLY `score`, indexed by `position % ratio`,
//      in `[ratio, coeff·head_dim]` row-major order. It is added per token at
//      *store* time, before the window softmax. Upstream:
//        `ape_row = position % COMPRESS_RATIO; store(score + ape)`
//      `[V save_partial_states.py:80-89]`. The checkpoint ships it in the same
//      order (`[ratio, width]`); `ds4` stores it transposed and must not be
//      copied.
//   2. THE SOFTMAX IS PER DIMENSION, over the window. For each output `d`, the
//      weights are `softmax_over_offsets(score[segment(o), d])` — the score is
//      dimension-dependent, so each dimension has its own weight vector. The
//      `kv` read is at the *same* `segment(o)·head_dim + d` offset.
//   3. THE WINDOW IS `(1 + overlap)·ratio` long and ends at the boundary token,
//      where `overlap = (ratio == 4)`. With overlap the row is twice as wide and
//      the second half holds the newer `ratio` tokens: `coeff = 1 + overlap`,
//      `segment = o / ratio`, offset `segment · head_dim + d`.
//   4. THE ENTRY IS EMITTED ONLY WHEN `(pos + 1) % ratio == 0`, and its RoPE
//      position is the window start `(pos / ratio)·ratio`, which equals the
//      plan's `pos + 1 − ratio` exactly at those positions.
// ---------------------------------------------------------------------------

// Floor-mod, because Python's `%` (which the reference uses) and C++'s differ
// for negative operands. The graph only calls this with non-negative positions,
// so the two agree in practice; making it explicit removes the question.
inline int64_t floor_mod(int64_t value, int64_t modulus) noexcept {
    const int64_t r = value % modulus;
    return r < 0 ? r + modulus : r;
}

// The APE row for a token, applied to its score row.
inline std::vector<double> compressor_ape_apply(const std::vector<double>& score_row,
                                                const std::vector<double>& ape,
                                                int64_t position, int64_t ratio,
                                                size_t width) {
    const int64_t row = floor_mod(position, ratio);
    std::vector<double> out(width);
    for (size_t i = 0; i < width; ++i) out[i] = score_row[i] + ape[row * width + i];
    return out;
}

// The window reduction: for each dimension `d`, a softmax over the window of the
// (already APE-adjusted) scores at dimension `d`, used to combine the `kv`
// values at dimension `d`.
//
// `window_kv` and `window_score` are the window entries in chronological order,
// each of width `coeff · head_dim`. The segment of offset `o` is `o / ratio`.
//
// `valid`, when given, marks which offsets actually hold state. A window that
// starts before position 0 has leading offsets that were never written, and the
// kernel skips them (`if (source_position < 0) continue`). They must be skipped
// here too — otherwise they would wrongly enter the max and the denominator.
// Passing no `valid` means every offset is populated. If no offset is
// populated the raw result is zero, matching the kernel's `has_value` guard
// rather than dividing by an empty denominator.
inline std::vector<double> compressor_raw(const std::vector<std::vector<double>>& window_kv,
                                          const std::vector<std::vector<double>>& window_score,
                                          size_t head_dim, int64_t ratio,
                                          const std::vector<bool>* valid = nullptr) {
    const size_t window = window_kv.size();
    std::vector<double> raw(head_dim, 0.0);
    if (window == 0) return raw;

    const auto populated = [&](size_t o) { return valid == nullptr || (*valid)[o]; };

    for (size_t d = 0; d < head_dim; ++d) {
        // Scores for this dimension, one per populated window offset, using the
        // segment mapping `segment = offset / ratio`.
        std::vector<double> s(window, 0.0);
        double m = -std::numeric_limits<double>::infinity();
        size_t populated_count = 0;
        for (size_t o = 0; o < window; ++o) {
            if (!populated(o)) continue;
            const size_t segment = o / static_cast<size_t>(ratio);
            const size_t off = segment * head_dim + d;
            s[o] = window_score[o][off];
            m = std::fmax(m, s[o]);
            ++populated_count;
        }
        if (populated_count == 0) continue;

        double denom = 0.0;
        std::vector<double> w(window, 0.0);
        for (size_t o = 0; o < window; ++o) {
            if (!populated(o)) continue;
            w[o] = std::exp(s[o] - m);
            denom += w[o];
        }
        if (denom <= 0.0) continue;

        double acc = 0.0;
        for (size_t o = 0; o < window; ++o) {
            if (!populated(o)) continue;
            const size_t segment = o / static_cast<size_t>(ratio);
            const size_t off = segment * head_dim + d;
            acc += (w[o] / denom) * window_kv[o][off];
        }
        raw[d] = acc;
    }
    return raw;
}

// The compressed entry's RoPE position: the start of its own window. Upstream
// writes this as `(positions // compress_ratio) * compress_ratio`
// `[V compressor.py; V fused_compress_quant_cache.py]`; the plan writes the
// equivalent `pos + 1 − ratio`. Provided once so a gate can assert they agree
// rather than assume it.
inline int64_t compressor_rope_position(int64_t boundary_position, int64_t ratio) noexcept {
    return (boundary_position / ratio) * ratio;
}

// ---------------------------------------------------------------------------
// Lightning indexer — Step 2.4.3
//
// Reference: `sglang/.../srt/layers/attention/dsv4/indexer.py`
//   `fp8_paged_mqa_logits_torch` / the SM120 variant, lines 100-126 / 245-267.
// The whole scoring path there is five readable lines:
//
//     score  = bmm(kv_value, q.transpose)      # [B, candidates, n_heads]
//     score  = F.relu(score)                   # <-- per HEAD, before weighting
//     score  = score * weight.unsqueeze(1)     # per-head weight, broadcast
//     score  = score.sum(dim=2)                # sum over heads
//     score  = score * kv_scale                # per-key scale (fp8 store)
//
// In closed form
//
//     score[c] = kv_scale[c] · Σ_h w[h] · relu( q[h] · k[c] )
//
// THE RELU IS ON THE PER-HEAD DOT, BEFORE THE WEIGHTING — not on the sum, and
// not after the weight. This is trap 11, and it is easy to omit precisely
// because the result still looks like a plausible attention score.
//
// Scales, from `C4Indexer.__init__` `[V indexer.py:1034, 1075]`:
//   softmax_scale = head_dim**-0.5        = 1/sqrt(128)   (the INDEX head)
//   weight_scale  = softmax_scale·n_heads**-0.5 = 1/sqrt(128·64)
// The pipeline passes the two factors separately rather than their product,
// which is algebraically identical; the oracle takes them separately too so the
// gate exercises the same decomposition.
//
// HADAMARD. Upstream applies a `1/sqrt(n)` Hadamard rotation to indexer Q and K
// before quantization in some paths, and explicitly **drops it in the fused
// path**, labelled *"(logit-preserving)"* `[V dsa_indexer.py:395-398]`. It is a
// conditioning step for the fp8 round-trip, not a graph semantic: because the
// rotation is orthogonal, rotating *both* Q and K leaves every dot product
// unchanged. `hadamard_rotated` below exists so a gate can *measure* that claim
// instead of repeating it — and can show that a one-sided application, which
// would silently change every score, is the failure mode to avoid.
// ---------------------------------------------------------------------------

// Normalized Sylvester–Hadamard transform, in place, `n` a power of two.
// Orthogonal: `H·Hᵀ = I`, which is exactly why a two-sided rotation is
// score-preserving.
inline std::vector<double> hadamard_rotated(std::vector<double> x) {
    const size_t n = x.size();
    for (size_t step = 1; step < n; step *= 2) {
        for (size_t i = 0; i < n; i += 2 * step) {
            for (size_t j = 0; j < step; ++j) {
                const double a = x[i + j];
                const double b = x[i + j + step];
                x[i + j] = a + b;
                x[i + j + step] = a - b;
            }
        }
    }
    const double norm = 1.0 / std::sqrt(static_cast<double>(n));
    for (double& v : x) v *= norm;
    return x;
}

// Rotates each `head_dim`-wide head of a `[heads, head_dim]` row independently.
inline std::vector<double> hadamard_rotate_heads(const std::vector<double>& x,
                                                 size_t num_heads, size_t head_dim) {
    std::vector<double> out(x.size());
    for (size_t h = 0; h < num_heads; ++h) {
        const std::vector<double> head(x.begin() + h * head_dim,
                                       x.begin() + (h + 1) * head_dim);
        const std::vector<double> rotated = hadamard_rotated(head);
        for (size_t d = 0; d < head_dim; ++d) out[h * head_dim + d] = rotated[d];
    }
    return out;
}

// `apply_relu` is not a feature — like `transpose_comb` in `hc_post`, it exists
// so a gate can compute the *wrong* variant deliberately and demonstrate that
// the difference is real. Nothing in the graph passes `false`.
inline std::vector<double> indexer_scores(const std::vector<double>& query,
                                          const std::vector<double>& key,
                                          const std::vector<double>& weights,
                                          size_t num_heads, size_t head_dim,
                                          double softmax_scale, double head_scale,
                                          bool apply_relu = true) {
    const size_t candidates = key.size() / head_dim;
    std::vector<double> scores(candidates, 0.0);
    for (size_t c = 0; c < candidates; ++c) {
        double acc = 0.0;
        for (size_t h = 0; h < num_heads; ++h) {
            double dot = 0.0;
            for (size_t d = 0; d < head_dim; ++d) {
                dot += query[h * head_dim + d] * key[c * head_dim + d];
            }
            const double rectified = apply_relu ? std::fmax(dot, 0.0) : dot;
            acc += rectified * weights[h] * softmax_scale * head_scale;
        }
        scores[c] = acc;
    }
    return scores;
}

// Top-k by descending score, ties broken to the **lower index** (trap 18). Ties
// are broken by index rather than left to the sort, so the selection is
// deterministic — the same rule the plan states for the router.
inline std::vector<int32_t> topk_indices(const std::vector<double>& scores, size_t k) {
    std::vector<int32_t> order(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) order[i] = static_cast<int32_t>(i);

    std::stable_sort(order.begin(), order.end(),
                     [&scores](int32_t a, int32_t b) {
                         const double sa = scores[static_cast<size_t>(a)];
                         const double sb = scores[static_cast<size_t>(b)];
                         if (sa != sb) return sa > sb;
                         return a < b;
                     });

    const size_t take = std::min(k, order.size());
    order.resize(take);
    return order;
}

// ---------------------------------------------------------------------------
// Grouped low-rank output projection — Step 2.5
//
//     z[t, g, r] = Σ_d o[t, g, d] · wo_a[g, r, d]
//
// with `o` `[T, G, D]` and `wo_a` `[G, R, D]`, flattened to `[T, G·R]`
// `[V sglang models/deepseek_v4.py:1718-1738 einsum("tgd,grd->tgr", o, wo_a)]`.
//
// Two structural facts this encodes, both of which a plausible implementation
// can get wrong:
//
//  * The group axis of `o` is a plain `view(T, G, -1)` of the token-major
//    `[T, heads, head_dim]` attention output `[V sglang :1785
//    o.view(o.shape[0], self.n_local_groups, -1)]`, so group g is the **8
//    contiguous heads** [8g, 8g+8) — not an interleaved slice of heads.
//  * The checkpoint tensor is stored `[G·R, D]` = `[8192, 4096]` and is a view
//    of `[G, R, D]` `[V sglang :3346 weight.view(G * R, D)]`, so the group
//    blocks are contiguous in the row-major layout: flat index `(g·R + r)·D + d`.
//
// `contiguous_blocks=false` exists for the same reason `transpose_comb` and
// `apply_relu` do elsewhere in this file: so a gate can compute the *wrong*
// layout deliberately and demonstrate that the difference is material. Nothing
// in the graph passes `false`.
template <class Accessor>
std::vector<double> grouped_wo_a(size_t tokens, size_t groups, size_t rank,
                                 size_t group_dim, const std::vector<double>& o,
                                 Accessor w_at, bool contiguous_blocks = true) {
    std::vector<double> z(tokens * groups * rank, 0.0);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t g = 0; g < groups; ++g) {
            for (size_t r = 0; r < rank; ++r) {
                double acc = 0.0;
                for (size_t d = 0; d < group_dim; ++d) {
                    const size_t wi = contiguous_blocks
                        ? g * (rank * group_dim) + r * group_dim + d
                        : r * (groups * group_dim) + g * group_dim + d;
                    acc += w_at(wi) * o[(t * groups + g) * group_dim + d];
                }
                z[(t * groups + g) * rank + r] = acc;
            }
        }
    }
    return z;
}

// ---------------------------------------------------------------------------
// MoE router — Step 2.9
//
//     scores[e] = sqrt(softplus(logits[e])),   softplus threshold 20
//     selection[e] = scores[e] + bias[e]
//     ids = top-6(selection)   |  ids = tid2eid[token]     (layers 0..2)
//     w[k] = scores[ids[k]] / Σ_k scores[ids[k]] · 1.5
//
// Written from the **naive reference**
// `vllm/tests/kernels/moe/test_topk_softplus_sqrt.py::_torch_topk_softplus_sqrt`,
// which is the arbiter for semantics and tie-break; the fused kernel and `ds4`
// were used only as cross-checks. Three things this encodes that are easy to get
// wrong and that the plan calls out explicitly:
//
//  * the bias enters **after** softplus and sqrt, on the score — not on the
//    logit. `ffn.gate.bias` is renamed `e_score_correction_bias` on load
//    `[V nvidia/model.py:1704]`, and a loader that treats it as a linear bias
//    is silently wrong;
//  * selection is **flat** — a top-6 over all 256 experts. `n_group` and
//    `topk_group` are absent from the config, so no group pre-filter exists;
//  * the stored weight is the **unbiased** score, normalized by its own sum and
//    only then scaled (trap 29).
//
// `bias_before_softplus` is the deliberately-wrong variant that trap names, in
// the same spirit as `transpose_comb` and `apply_relu`: a gate computes the
// wrong answer on purpose to show the difference is material. Nothing in the
// graph passes `true`.
inline double softplus(double x) {
    // `F.softplus` with the default threshold: above 20 the identity is used,
    // because `log1p(exp(x))` has no precision left there.
    return x > 20.0 ? x : std::log1p(std::exp(x));
}

inline double router_score(double logit) {
    return std::sqrt(softplus(logit));
}

struct RouterSelection {
    std::vector<int32_t> ids;
    std::vector<double> weights;
};

// Layers >= num_hash_layers. `bias` is required here (it is `[n_routed_experts]`
// F32 in the checkpoint); an empty bias means "behave as if it were zero", which
// the gate uses to show the bias is load-bearing.
inline RouterSelection router_topk(const std::vector<double>& logits,
                                   const std::vector<double>& bias,
                                   size_t top_k, double scaling,
                                   bool bias_before_softplus = false) {
    const size_t experts = logits.size();
    std::vector<double> scores(experts, 0.0);
    std::vector<double> selection(experts, 0.0);
    for (size_t e = 0; e < experts; ++e) {
        const double b = bias.empty() ? 0.0 : bias[e];
        if (bias_before_softplus) {
            scores[e] = router_score(logits[e] + b);
        } else {
            scores[e] = router_score(logits[e]);
        }
        selection[e] = bias.empty() ? scores[e] : scores[e] + b;
    }

    RouterSelection out;
    out.ids = topk_indices(selection, top_k);
    out.weights.resize(top_k, 0.0);

    double sum = 0.0;
    for (size_t k = 0; k < top_k; ++k) {
        out.weights[k] = scores[static_cast<size_t>(out.ids[k])];
        sum += out.weights[k];
    }
    // Order matters for reproducibility and matches the kernel: divide by the
    // sum, then scale. The naive reference carries no guard; the fused kernel
    // uses `Σ > 0 ? Σ : 1` and our kernel adds `1e-20`. Since
    // `sqrt(softplus(x)) > 0` for every finite `x`, none of the three can fire
    // for a logit reachable with fp16 weights — the divergence needs a score
    // below ~1.6e-21, i.e. a logit below about -96.
    for (size_t k = 0; k < top_k; ++k) {
        out.weights[k] /= sum;
        out.weights[k] *= scaling;
    }
    return out;
}

// Layers < num_hash_layers. The ids come **directly** from the table, in the
// table's own column order — they are not sorted by score, so a positional
// comparison must not assume ordering. There is no bias and no comparison.
inline RouterSelection router_hash(const std::vector<double>& logits,
                                   const std::vector<int64_t>& table_row,
                                   double scaling) {
    RouterSelection out;
    out.weights.resize(table_row.size(), 0.0);

    double sum = 0.0;
    for (size_t k = 0; k < table_row.size(); ++k) {
        out.ids.push_back(static_cast<int32_t>(table_row[k]));
        out.weights[k] = router_score(logits[static_cast<size_t>(table_row[k])]);
        sum += out.weights[k];
    }
    for (size_t k = 0; k < table_row.size(); ++k) {
        out.weights[k] /= sum;
        out.weights[k] *= scaling;
    }
    return out;
}

// ===========================================================================
// Swizzled W4A16 expert format — Steps 2.10.2 / 2.10.3
// ===========================================================================
//
// The artifact stores each routed expert as W1, W2, W3 in a *swizzled* W4A16
// layout: 4-bit symmetric weights (signed, zero point 8), one fp16 scale per
// 32-column group, and a storage permutation chosen so that a Wave32 reads a
// row as `uint4` words with the activation in `half2` pairs.
//
// Nothing here is derived from a kernel. The address arithmetic is written out
// scalar-by-scalar from the format definition, and every element is read back
// through it, so the device and this file agree only if both understand the same
// byte layout.
//
// Two facts the format fixes, both of which a plausible reader can get wrong and
// both of which the gate below measures rather than assumes:
//
//  * the zero point is **signed**: `w = (q − 8) · scale`, so a stored nibble of
//    8 means zero;
//  * the eight nibbles of a word are **not** in column order — the column whose
//    offset within the 8-column slice is `s` lives at nibble position
//    `kNibbleSlot[s]`.
//
// `SwizzledDecodeOptions` carries the two deliberately-wrong readings, in the
// same spirit as `transpose_comb`, `apply_relu` and `bias_before_softplus`
// elsewhere in this file. Nothing in the graph passes a non-default option.

enum class SwizzledKind { W1, W2, W3 };

struct SwizzledShape {
    size_t packed_offset;
    size_t scale_offset;
    int rows;
    int columns;
    int rows_per_wave;
    int lanes_per_row;

    int iterations() const { return (columns / 32) / lanes_per_row; }
    size_t weight_count() const { return static_cast<size_t>(rows) * columns; }
};

// Offsets and shapes from the artifact's own format constants.
inline SwizzledShape swizzled_shape(SwizzledKind kind) {
    switch (kind) {
        case SwizzledKind::W1: return {0,        4194304,  2048, 4096, 4, 8};
        case SwizzledKind::W2: return {4718592,  8912896,  4096, 2048, 8, 4};
        case SwizzledKind::W3: return {9437184,  13631488, 2048, 4096, 4, 8};
    }
    throw std::invalid_argument("dsv4_oracle: unknown swizzled matrix kind");
}

// Nibble position inside its 4-byte word for a column's offset `s` within the
// 8-column slice, derived from the Wave32 `half2` read order: the first `half2`
// pairs activation columns 0 and 1 with the nibbles that sit in the low and high
// halves of the word, which are nibbles 0 and 4 — hence `{0,4,1,5,2,6,3,7}`.
inline int swizzled_nibble_slot(int s) {
    static constexpr int kSlot[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    return kSlot[s & 7];
}

struct SwizzledAddress {
    size_t storage_slot; // index of the fp16 scale shared by 32 weights
    size_t packed_word;  // index of the uint32 holding this weight
    int nibble;          // nibble position inside that word
};

inline SwizzledAddress swizzled_address(const SwizzledShape& shape, int row, int column) {
    const int group = column / 32;
    const int word_in_group = (column % 32) / 8;
    const int source_nibble = column % 8;
    const int iteration = group / shape.lanes_per_row;
    const int slice = group % shape.lanes_per_row;
    const int row_block = row / shape.rows_per_wave;
    const int row_in_block = row % shape.rows_per_wave;
    const int lane = row_in_block * shape.lanes_per_row + slice;

    const size_t storage = (static_cast<size_t>(row_block) * shape.iterations() +
                            static_cast<size_t>(iteration)) * 32u +
                           static_cast<size_t>(lane);
    SwizzledAddress address;
    address.storage_slot = storage;
    address.packed_word = storage * 4u + static_cast<size_t>(word_in_group);
    address.nibble = swizzled_nibble_slot(source_nibble);
    return address;
}

inline uint16_t load_le_u16(const uint8_t* address) {
    uint16_t value = 0;
    std::memcpy(&value, address, sizeof(value));
    return value;
}

inline uint32_t load_le_u32(const uint8_t* address) {
    uint32_t value = 0;
    std::memcpy(&value, address, sizeof(value));
    return value;
}

inline void store_le_u16(uint8_t* address, uint16_t value) {
    std::memcpy(address, &value, sizeof(value));
}

inline void store_le_u32(uint8_t* address, uint32_t value) {
    std::memcpy(address, &value, sizeof(value));
}

inline double half_bits_to_double(uint16_t bits) {
    const int sign = (bits >> 15) & 1;
    const int exponent = (bits >> 10) & 0x1F;
    const int mantissa = bits & 0x3FF;
    double value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<double>(mantissa), -24);
    } else if (exponent == 0x1F) {
        value = std::numeric_limits<double>::quiet_NaN();
    } else {
        value = std::ldexp(1.0 + static_cast<double>(mantissa) / 1024.0, exponent - 15);
    }
    return sign != 0 ? -value : value;
}

// Widens a run of fp16 **bit patterns** to doubles. This is not the same
// operation as constructing `std::vector<double>` from them: that converts each
// `uint16_t` arithmetically, so a weight of ~0.045 becomes 15360 (its raw bit
// pattern 0x3C00 read as a number). Anywhere this file consumes an fp16 tensor
// it must come through here.
inline std::vector<double> half_bits_to_doubles(const uint16_t* bits, size_t count) {
    std::vector<double> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = half_bits_to_double(bits[i]);
    return out;
}

// Exact fp16 bit pattern for 2^exponent. Scales in the synthetic fixtures are
// restricted to powers of two so the encoder needs no fp32->fp16 rounding step:
// a power of two is representable exactly, and its bits are trivial.
inline uint16_t half_bits_of_power_of_two(int exponent) {
    if (exponent < -14) exponent = -14;
    if (exponent > 15) exponent = 15;
    return static_cast<uint16_t>((exponent + 15) << 10);
}

struct SwizzledDecodeOptions {
    bool signed_zero_point = true; // false: read the nibble as an unsigned magnitude
    bool permute_nibbles = true;   // false: take the nibble straight from the column offset
};

// Reads the payload back into a row-major `[rows, columns]` double matrix.
inline std::vector<double> swizzled_decode(const uint8_t* payload, SwizzledKind kind,
                                           SwizzledDecodeOptions options = {}) {
    if (payload == nullptr) {
        throw std::invalid_argument("dsv4_oracle: null swizzled payload");
    }
    const SwizzledShape shape = swizzled_shape(kind);
    std::vector<double> weights(shape.weight_count(), 0.0);

    for (int row = 0; row < shape.rows; ++row) {
        for (int column = 0; column < shape.columns; ++column) {
            SwizzledAddress address = swizzled_address(shape, row, column);
            if (!options.permute_nibbles) {
                address.nibble = column % 8;
            }
            const uint32_t word = load_le_u32(payload + shape.packed_offset +
                                              address.packed_word * sizeof(uint32_t));
            const int nibble = static_cast<int>((word >> (4 * address.nibble)) & 0xFu);

            const double scale = half_bits_to_double(load_le_u16(
                payload + shape.scale_offset + address.storage_slot * sizeof(uint16_t)));

            const double quantized = options.signed_zero_point
                ? static_cast<double>(nibble - 8)
                : static_cast<double>(nibble);
            weights[static_cast<size_t>(row) * shape.columns + column] = quantized * scale;
        }
    }
    return weights;
}

// Inverse of `swizzled_decode`, used only to build synthetic fixtures. `w` must
// be `[rows * columns]` row-major.
//
// The per-(row, group-of-32) scale is the smallest **power of two** that is at
// least `max|w| / 7`, so every quantized magnitude stays inside the signed 4-bit
// range `[-8, 7]` and the scale is exactly representable in fp16. That trades up
// to one bit of precision for an encoder with no rounding step to get wrong.
inline void swizzled_encode(uint8_t* payload, SwizzledKind kind,
                            const std::vector<double>& w) {
    if (payload == nullptr) {
        throw std::invalid_argument("dsv4_oracle: null swizzled payload");
    }
    const SwizzledShape shape = swizzled_shape(kind);
    if (w.size() != shape.weight_count()) {
        throw std::invalid_argument("dsv4_oracle: swizzled_encode size mismatch");
    }

    // Zero the packed and scale regions of this matrix.
    std::memset(payload + shape.packed_offset, 0,
                static_cast<size_t>(shape.rows) * static_cast<size_t>(shape.columns) / 2);
    std::memset(payload + shape.scale_offset, 0,
                shape.weight_count() / 32 * sizeof(uint16_t));

    for (int row = 0; row < shape.rows; ++row) {
        for (int group = 0; group < shape.columns / 32; ++group) {
            double peak = 0.0;
            for (int j = 0; j < 32; ++j) {
                peak = std::fmax(peak, std::fabs(w[static_cast<size_t>(row) * shape.columns +
                                                    group * 32 + j]));
            }
            const int exponent = peak > 0.0
                ? static_cast<int>(std::ceil(std::log2(peak / 7.0)))
                : -14;
            const uint16_t scale_bits = half_bits_of_power_of_two(exponent);
            const double scale = half_bits_to_double(scale_bits);

            for (int j = 0; j < 32; ++j) {
                const int column = group * 32 + j;
                const double value = w[static_cast<size_t>(row) * shape.columns + column];
                int quantized = static_cast<int>(std::lround(value / scale));
                quantized = std::max(-8, std::min(7, quantized));

                const SwizzledAddress address = swizzled_address(shape, row, column);
                store_le_u16(payload + shape.scale_offset +
                                 address.storage_slot * sizeof(uint16_t),
                             scale_bits);
                uint8_t* word_address = payload + shape.packed_offset +
                                        address.packed_word * sizeof(uint32_t);
                uint32_t word = load_le_u32(word_address);
                word &= ~(0xFu << (4 * address.nibble));
                word |= static_cast<uint32_t>(quantized + 8) << (4 * address.nibble);
                store_le_u32(word_address, word);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Clamped SwiGLU — Step 2.10.3
//
//     hidden = silu(clamp(gate, max = limit)) * clamp(up, min = -limit, max = +limit)
//
// The clamp is **asymmetric** and re-cited from `vllm/.../activation.py:241-242`
// (`SiluAndMulWithClamp`): the gate is bounded only above, the up branch both
// sides. A symmetric clamp looks like the obvious reading and still produces
// plausible activations, so the mode is a parameter here and the gate measures
// the difference.
enum class ClampMode {
    Asymmetric, // the reference: gate max-only, up both sides
    Symmetric,  // gate clamped both sides — the plausible-but-wrong reading
    None        // no clamp at all
};

inline double clamped_swiglu(double gate, double up, double limit,
                             ClampMode mode = ClampMode::Asymmetric) {
    switch (mode) {
        case ClampMode::Asymmetric:
            gate = std::fmin(gate, limit);
            up = std::fmin(std::fmax(up, -limit), limit);
            break;
        case ClampMode::Symmetric:
            gate = std::fmin(std::fmax(gate, -limit), limit);
            up = std::fmin(std::fmax(up, -limit), limit);
            break;
        case ClampMode::None:
            break;
    }
    return (gate / (1.0 + std::exp(-gate))) * up;
}

// The whole routed expert body: `gate = W1·x`, `up = W3·x`, clamped SwiGLU, then
// `out = W2·hidden` — Step 2.10.3 exactly, in double.
//
// Note that the device writes `hidden` to fp16 between the two halves (the fused
// kernel's output is a `half*`), so the gate must allow for that one rounding;
// this function keeps `hidden` in double. The alternative — rounding here too —
// would hide a rounding question behind the oracle.
inline std::vector<double> expert_ffn(const uint8_t* payload,
                                      const std::vector<double>& activation,
                                      double limit = 10.0,
                                      ClampMode mode = ClampMode::Asymmetric) {
    const SwizzledShape w1_shape = swizzled_shape(SwizzledKind::W1);
    const SwizzledShape w2_shape = swizzled_shape(SwizzledKind::W2);
    const SwizzledShape w3_shape = swizzled_shape(SwizzledKind::W3);
    // `x` is the 4096-wide hidden state; W1/W3 map it to the 2048-wide
    // intermediate, which W2 maps back to 4096.
    if (activation.size() != static_cast<size_t>(w1_shape.columns) ||
        activation.size() != static_cast<size_t>(w3_shape.columns) ||
        w1_shape.rows != w2_shape.columns ||
        w2_shape.rows != w1_shape.columns) {
        throw std::invalid_argument("dsv4_oracle: expert_ffn shape mismatch");
    }

    const std::vector<double> w1 = swizzled_decode(payload, SwizzledKind::W1);
    const std::vector<double> w2 = swizzled_decode(payload, SwizzledKind::W2);
    const std::vector<double> w3 = swizzled_decode(payload, SwizzledKind::W3);

    const std::vector<double> gate = matvec(
        static_cast<size_t>(w1_shape.rows), static_cast<size_t>(w1_shape.columns),
        activation, [&](size_t o, size_t i) { return w1[o * w1_shape.columns + i]; });
    const std::vector<double> up = matvec(
        static_cast<size_t>(w3_shape.rows), static_cast<size_t>(w3_shape.columns),
        activation, [&](size_t o, size_t i) { return w3[o * w3_shape.columns + i]; });

    std::vector<double> hidden(gate.size(), 0.0);
    for (size_t i = 0; i < gate.size(); ++i) {
        hidden[i] = clamped_swiglu(gate[i], up[i], limit, mode);
    }

    return matvec(static_cast<size_t>(w2_shape.rows),
                  static_cast<size_t>(w2_shape.columns), hidden,
                  [&](size_t o, size_t i) { return w2[o * w2_shape.columns + i]; });
}

// The two pre-activations, so a gate can check that the clamp actually fires in
// the data it is testing rather than only comparing the composed result.
struct ExpertGateUp {
    std::vector<double> gate;
    std::vector<double> up;
};

inline ExpertGateUp expert_gate_up(const uint8_t* payload,
                                   const std::vector<double>& activation) {
    const std::vector<double> w1 = swizzled_decode(payload, SwizzledKind::W1);
    const std::vector<double> w3 = swizzled_decode(payload, SwizzledKind::W3);
    const SwizzledShape w1_shape = swizzled_shape(SwizzledKind::W1);
    const SwizzledShape w3_shape = swizzled_shape(SwizzledKind::W3);

    ExpertGateUp out;
    out.gate = matvec(static_cast<size_t>(w1_shape.rows),
                      static_cast<size_t>(w1_shape.columns), activation,
                      [&](size_t o, size_t i) { return w1[o * w1_shape.columns + i]; });
    out.up = matvec(static_cast<size_t>(w3_shape.rows),
                    static_cast<size_t>(w3_shape.columns), activation,
                    [&](size_t o, size_t i) { return w3[o * w3_shape.columns + i]; });
    return out;
}

// ---------------------------------------------------------------------------
// Dense (unquantized) FFN — Step 2.10.4, the shared expert
// ---------------------------------------------------------------------------
//
// Structurally the same op as 2.10.3 and **deliberately a separate function**,
// because the two differ in exactly the ways a shared helper would have hidden:
//
//   * the weights are fp16 and stored row-major `[out, in]` with no
//     quantization — this is not the swizzled W4A16 format;
//   * there is no routing at all. The shared expert fires on every token
//     unconditionally `[V config n_shared_experts == 1; V vllm model.py:1024-1031
//     shared_output = self.shared_experts(hidden_states)]`;
//   * the activation rule is the *same* one — `activation_clamp` is passed to
//     both the routed and the shared path in the reference `[V model.py:1016-1021]`.
//
// So this uses `clamped_swiglu` rather than re-deriving it: the rule is shared on
// purpose, and the gate for it lives with 2.10.3.
//
// `w1`, `w3` are `[intermediate, hidden]` and `w2` is `[hidden, intermediate]`,
// all row-major, as the checkpoint stores `nn.Linear.weight`.
inline std::vector<double> dense_ffn(size_t intermediate, size_t hidden,
                                     const std::vector<double>& activation,
                                     const std::vector<double>& w1,
                                     const std::vector<double>& w3,
                                     const std::vector<double>& w2,
                                     double limit = 10.0,
                                     ClampMode mode = ClampMode::Asymmetric) {
    if (activation.size() != hidden || w1.size() != intermediate * hidden ||
        w3.size() != intermediate * hidden || w2.size() != hidden * intermediate) {
        throw std::invalid_argument("dsv4_oracle: dense_ffn shape mismatch");
    }

    const std::vector<double> gate = matvec(intermediate, hidden, activation,
        [&](size_t o, size_t i) { return w1[o * hidden + i]; });
    const std::vector<double> up = matvec(intermediate, hidden, activation,
        [&](size_t o, size_t i) { return w3[o * hidden + i]; });

    std::vector<double> hidden_act(intermediate, 0.0);
    for (size_t i = 0; i < intermediate; ++i) {
        hidden_act[i] = clamped_swiglu(gate[i], up[i], limit, mode);
    }

    return matvec(hidden, intermediate, hidden_act,
                  [&](size_t o, size_t i) { return w2[o * intermediate + i]; });
}

// ===========================================================================
// Tier 2 — the Sliding-class layer body (Steps 2.0 … 2.11)
// ===========================================================================
//
// This is not a new primitive. It is the *composition* the Tier-2 gate exists to
// certify: Tier 1 proved each piece, and the failure mode of a layer is the
// wiring between them, not the pieces.
//
// Written from the plan's Step 2 for a ratio-0 (Sliding) layer, in the order the
// plan states, and built entirely from this file's primitives so that the
// composition is the only new thing under test:
//
//   2.0    HC attention pre-mix + Sinkhorn            -> x_pre
//   2.1    attention RMSNorm                          -> x_norm
//   2.2    MLA Q path (q_lora -> q_norm -> wq_b -> per-head norm), KV path
//   2.3    RoPE forward on q and kv, *sliding* base (theta 10000, plain)
//   2.4.1  local sliding-window attention + sink
//   2.3    inverse RoPE on the attention-output tail
//   2.5    grouped wo_a [8,1024,4096] then wo_b [4096,8192]
//   2.6    HC attention post-mix                      -> res_mid
//   2.7    HC FFN pre-mix + Sinkhorn
//   2.8    FFN RMSNorm
//   2.9    router (hash on layers < 3, biased flat top-6 otherwise)
//   2.10   routed experts + shared expert, clamped SwiGLU
//   2.10.5 combine: routed sum first, then `+= shared`
//   2.11   HC FFN post-mix                            -> res_out
//
// Deliberately absent, and that absence is itself a property the gate asserts:
// the compressor, the indexer, and the compressed row-set. A Sliding layer must
// never read those tensors (traps 4 and 33) — layers 0 and 1 have no compressor
// and no indexer at all, and running one on them would read tensors the artifact
// does not contain. There is no state for them in `SlidingKvRing` and no code
// path here that could consult them.
//
// Ordering note, because it is the one place this composition is a *choice*.
// Step 2.10.5 fixes the term set — `Σ_k w_k·down_k` plus the shared expert — but
// the reference has two orderings of it: an unfused `final += shared` and a fused
// form that passes the shared weights into the MoE kernel. Our engine mirrors the
// fused form; this oracle writes the unfused `routed_sum + shared`. The term set
// is identical and only the fp rounding order differs, so a gate comparing them
// must allow the reassociation — it is not evidence of a defect.

struct LayerBodyShape {
    uint32_t hidden{4096};
    uint32_t hc_mult{4};
    uint32_t q_lora_rank{1024};
    uint32_t num_heads{64};
    uint32_t head_dim{512};
    uint32_t rotary_dim{64};
    uint32_t o_groups{8};
    uint32_t o_lora_rank{1024};
    uint32_t intermediate{2048};
    uint32_t num_experts{256};
    uint32_t top_k{6};
    uint32_t local_capacity{128};

    double eps{1e-6};            // every RMSNorm site in this graph
    double routed_scaling{1.5};
    double swiglu_limit{10.0};

    uint32_t hc_dim() const noexcept { return hc_mult * hidden; }
    uint32_t hc_mult3() const noexcept { return hc_mult * (2 + hc_mult); }
    uint32_t total_q() const noexcept { return num_heads * head_dim; }
    uint32_t group_dim() const noexcept { return (num_heads / o_groups) * head_dim; }
    uint32_t total_o_lora() const noexcept { return o_groups * o_lora_rank; }
    double attn_scale() const noexcept {
        return 1.0 / std::sqrt(static_cast<double>(head_dim));
    }
};

// One Sliding layer's tensors, exactly as the artifact stores them. fp16 tensors
// are passed as their raw 16-bit patterns so the oracle reads the same bits the
// kernel does, with no widening step that could hide an input rounding question.
struct SlidingLayerWeights {
    // Hyper-Connections — fp32 in the checkpoint.
    const float* hc_attn_fn{nullptr};     // [hc_mult3, hc_dim]
    const float* hc_attn_base{nullptr};   // [hc_mult3]
    const float* hc_attn_scale{nullptr};  // [3]
    const float* hc_ffn_fn{nullptr};      // [hc_mult3, hc_dim]
    const float* hc_ffn_base{nullptr};    // [hc_mult3]
    const float* hc_ffn_scale{nullptr};   // [3]

    // Attention — fp16 (raw bits), except the sink which is fp32.
    const uint16_t* attn_norm{nullptr};   // [hidden]
    const uint16_t* wq_a{nullptr};        // [q_lora_rank, hidden]
    const uint16_t* q_norm{nullptr};      // [q_lora_rank]
    const uint16_t* wq_b{nullptr};        // [total_q, q_lora_rank]
    const uint16_t* wkv{nullptr};         // [head_dim, hidden]
    const uint16_t* kv_norm{nullptr};     // [head_dim]
    const float* attn_sink{nullptr};      // [num_heads]
    const uint16_t* wo_a{nullptr};        // [o_groups * o_lora_rank, group_dim]
    const uint16_t* wo_b{nullptr};        // [hidden, total_o_lora]

    // FFN — fp16, plus fp32 bias / an int64 hash table row.
    const uint16_t* ffn_norm{nullptr};      // [hidden]
    const uint16_t* gate_weight{nullptr};   // [num_experts, hidden]
    const float* gate_bias{nullptr};        // [num_experts], null on hash layers
    const int64_t* tid2eid_row{nullptr};    // [top_k], null on biased layers
    const uint16_t* shared_w1{nullptr};     // [intermediate, hidden]
    const uint16_t* shared_w3{nullptr};     // [intermediate, hidden]
    const uint16_t* shared_w2{nullptr};     // [hidden, intermediate]

    // The six selected routed experts, swizzled W4A16 payloads.
    const uint8_t* routed_payloads[8]{};
};

// The local ring. Key and value are the **same** row (trap 6), but the graph
// keeps two caches, so the oracle does too: an implementation that only wrote one
// of them would otherwise pass unnoticed.
struct SlidingKvRing {
    uint32_t capacity{0};
    uint32_t head_dim{0};
    std::vector<double> keys;       // [capacity * head_dim], rotated
    std::vector<double> values;     // [capacity * head_dim]
    std::vector<int64_t> positions; // [capacity], -1 = never written

    void reset(uint32_t cap, uint32_t dim) {
        capacity = cap;
        head_dim = dim;
        keys.assign(static_cast<size_t>(cap) * dim, 0.0);
        values.assign(static_cast<size_t>(cap) * dim, 0.0);
        positions.assign(cap, -1);
    }

    void store(uint32_t slot, int64_t position, const std::vector<double>& row) {
        const size_t at = static_cast<size_t>(slot) * head_dim;
        for (uint32_t d = 0; d < head_dim; ++d) {
            keys[at + d] = row[d];
            values[at + d] = row[d];
        }
        positions[slot] = position;
    }

    // The rows attention reads: every written slot whose position lies in
    // `[current_pos - (capacity - 1), current_pos]`, i.e. exactly
    // `min(current_pos + 1, capacity)` keys. Slot order is *not* meaningful —
    // the online softmax is order-invariant — so the rows are returned in slot
    // order, which is also the order the kernel's loop uses.
    std::vector<size_t> gather(int64_t current_pos) const {
        std::vector<size_t> slots;
        const int64_t first = std::max<int64_t>(
            0, current_pos - static_cast<int64_t>(capacity) + 1);
        for (uint32_t slot = 0; slot < capacity; ++slot) {
            const int64_t p = positions[slot];
            if (p >= first && p <= current_pos) slots.push_back(slot);
        }
        return slots;
    }
};

struct LayerBodyResult {
    // Hyper-Connections (attention sublayer)
    std::vector<double> mixes_a;    // [hc_mult3]
    std::vector<double> pre_a;      // [hc_mult]
    std::vector<double> post_a;     // [hc_mult]
    std::vector<double> comb_a;     // [hc_mult * hc_mult]
    std::vector<double> x_pre;      // [hidden]

    // MLA + attention
    std::vector<double> x_norm;     // [hidden]
    std::vector<double> q_lora;     // [q_lora_rank]
    std::vector<double> q_lora_norm;// [q_lora_rank]
    std::vector<double> q;          // [total_q] after the per-head norm
    std::vector<double> q_rot;      // [total_q] after RoPE
    std::vector<double> kv_norm;    // [head_dim]
    std::vector<double> kv_rot;     // [head_dim] — the row written to the ring
    std::vector<double> attn_out;   // [total_q] before inverse RoPE
    std::vector<double> attn_inv;   // [total_q] after inverse RoPE
    std::vector<double> z;          // [total_o_lora]
    std::vector<double> attn_proj;  // [hidden]

    // Hyper-Connections (FFN sublayer) + residual
    std::vector<double> res_mid;    // [hc_dim]
    std::vector<double> mixes_f;    // [hc_mult3]
    std::vector<double> pre_f;      // [hc_mult]
    std::vector<double> post_f;     // [hc_mult]
    std::vector<double> comb_f;     // [hc_mult * hc_mult]
    std::vector<double> ffn_pre;    // [hidden]

    // FFN
    std::vector<double> ffn_norm;   // [hidden]
    std::vector<double> router_logits;   // [num_experts]
    std::vector<int32_t> routed_ids;     // [top_k]
    std::vector<double> routed_weights;  // [top_k]
    std::vector<std::vector<double>> routed_expert_outputs; // [top_k][hidden]
    std::vector<double> routed_sum;      // [hidden]
    std::vector<double> shared_out;      // [hidden]
    std::vector<double> moe_out;         // [hidden]

    // Layer output
    std::vector<double> res_out;    // [hc_dim]
    uint32_t local_keys_read{0};    // how many ring rows attention actually read
};

// One full Sliding layer for one token. `residual` is `[hc_mult * hidden]` (the
// four HC streams); `ring` is updated in place with the token's own rotated key
// **before** attention reads it, exactly as step D does in the pipeline.
inline LayerBodyResult layer_sliding_body(
    const LayerBodyShape& shape,
    const SlidingLayerWeights& w,
    const RopeTableRef& rope,
    uint32_t position,
    const std::vector<double>& residual,
    SlidingKvRing& ring) {
    if (residual.size() != shape.hc_dim()) {
        throw std::invalid_argument("dsv4_oracle: layer body residual has the wrong width");
    }
    if (ring.capacity != shape.local_capacity || ring.head_dim != shape.head_dim) {
        throw std::invalid_argument("dsv4_oracle: layer body ring does not match the shape");
    }

    const size_t hidden = shape.hidden;
    const size_t head_dim = shape.head_dim;
    const size_t num_heads = shape.num_heads;
    const auto f16 = [](const uint16_t* p, size_t index) {
        return half_bits_to_double(p[index]);
    };

    const HcParams hc_params;

    // -------------------------------------------------------------------
    // 2.0 — HC attention pre-mix + Sinkhorn -> x_pre
    // -------------------------------------------------------------------
    LayerBodyResult out;
    out.mixes_a = hc_mixes(
        residual, shape.hc_mult3(),
        [&](size_t m, size_t k) {
            return static_cast<double>(w.hc_attn_fn[m * shape.hc_dim() + k]);
        },
        shape.hc_dim(), hc_params.rms_eps);

    {
        HcPreResult hc;
        std::vector<double> scale(w.hc_attn_scale, w.hc_attn_scale + 3);
        std::vector<double> base(
            w.hc_attn_base, w.hc_attn_base + shape.hc_mult3());
        hc_sinkhorn(out.mixes_a, scale, base, shape.hc_mult, hc_params, hc);
        out.pre_a = std::move(hc.pre_mix);
        out.post_a = std::move(hc.post_mix);
        out.comb_a = std::move(hc.comb);
    }
    out.x_pre = hc_pre_combine(residual, out.pre_a, hidden);

    // -------------------------------------------------------------------
    // 2.1 — attention RMSNorm
    // -------------------------------------------------------------------
    out.x_norm = rmsnorm(out.x_pre, half_bits_to_doubles(w.attn_norm, hidden), shape.eps);

    // -------------------------------------------------------------------
    // 2.2 — MLA Q and KV paths
    // -------------------------------------------------------------------
    {
        MlaQPath q = mla_q_path(
            out.x_norm,
            half_bits_to_doubles(w.q_norm, shape.q_lora_rank),
            shape.q_lora_rank, num_heads, head_dim, shape.eps,
            [&](size_t o, size_t i) { return f16(w.wq_a, o * hidden + i); },
            [&](size_t o, size_t i) {
                return f16(w.wq_b, o * shape.q_lora_rank + i);
            });
        out.q_lora = std::move(q.q_lora);
        out.q_lora_norm = std::move(q.q_lora_norm);
        out.q = std::move(q.q);

        out.kv_norm = mla_kv_path(
            out.x_norm,
            half_bits_to_doubles(w.kv_norm, head_dim),
            head_dim, shape.eps,
            [&](size_t o, size_t i) { return f16(w.wkv, o * hidden + i); });
    }

    // -------------------------------------------------------------------
    // 2.3 — RoPE forward (sliding base) on q (per head) and kv, then the ring
    //       write. The token's own key is in the cache before attention runs.
    // -------------------------------------------------------------------
    out.q_rot = out.q;
    for (size_t h = 0; h < num_heads; ++h) {
        std::vector<double> row(out.q_rot.begin() + h * head_dim,
                                out.q_rot.begin() + (h + 1) * head_dim);
        rope_apply_tail(row, rope, position, /*inverse=*/false);
        std::copy(row.begin(), row.end(), out.q_rot.begin() + h * head_dim);
    }

    out.kv_rot = out.kv_norm;
    rope_apply_tail(out.kv_rot, rope, position, /*inverse=*/false);
    ring.store(static_cast<uint32_t>(position) % shape.local_capacity,
               static_cast<int64_t>(position), out.kv_rot);

    // -------------------------------------------------------------------
    // 2.4.1 — local sliding-window attention + sink. Ratio 0: local rows only.
    // -------------------------------------------------------------------
    const std::vector<double> sink(w.attn_sink, w.attn_sink + num_heads);

    {
        const std::vector<size_t> slots = ring.gather(static_cast<int64_t>(position));
        out.local_keys_read = static_cast<uint32_t>(slots.size());

        std::vector<double> keys(slots.size() * head_dim, 0.0);
        for (size_t j = 0; j < slots.size(); ++j) {
            std::copy(ring.keys.begin() + slots[j] * head_dim,
                      ring.keys.begin() + (slots[j] + 1) * head_dim,
                      keys.begin() + j * head_dim);
        }
        out.attn_out = attention_scores_sink(
            out.q_rot, num_heads, head_dim, keys, slots.size(), sink,
            shape.attn_scale());
    }

    // -------------------------------------------------------------------
    // 2.3 (inverse) — rotate the attention output tail back, before 2.5.
    // -------------------------------------------------------------------
    out.attn_inv = out.attn_out;
    for (size_t h = 0; h < num_heads; ++h) {
        std::vector<double> row(out.attn_inv.begin() + h * head_dim,
                                out.attn_inv.begin() + (h + 1) * head_dim);
        rope_apply_tail(row, rope, position, /*inverse=*/true);
        std::copy(row.begin(), row.end(), out.attn_inv.begin() + h * head_dim);
    }

    // -------------------------------------------------------------------
    // 2.5 — grouped low-rank output projection, then wo_b
    // -------------------------------------------------------------------
    out.z = grouped_wo_a(
        1, shape.o_groups, shape.o_lora_rank, shape.group_dim(), out.attn_inv,
        [&](size_t wi) { return f16(w.wo_a, wi); });

    out.attn_proj = matvec(
        hidden, shape.total_o_lora(), out.z,
        [&](size_t o, size_t i) { return f16(w.wo_b, o * shape.total_o_lora() + i); });

    // -------------------------------------------------------------------
    // 2.6 — HC attention post-mix -> res_mid
    // -------------------------------------------------------------------
    out.res_mid = hc_post(out.attn_proj, residual, out.post_a, out.comb_a, hidden);

    // -------------------------------------------------------------------
    // 2.7 — HC FFN pre-mix + Sinkhorn
    // -------------------------------------------------------------------
    out.mixes_f = hc_mixes(
        out.res_mid, shape.hc_mult3(),
        [&](size_t m, size_t k) {
            return static_cast<double>(w.hc_ffn_fn[m * shape.hc_dim() + k]);
        },
        shape.hc_dim(), hc_params.rms_eps);
    {
        HcPreResult hc;
        std::vector<double> scale(w.hc_ffn_scale, w.hc_ffn_scale + 3);
        std::vector<double> base(
            w.hc_ffn_base, w.hc_ffn_base + shape.hc_mult3());
        hc_sinkhorn(out.mixes_f, scale, base, shape.hc_mult, hc_params, hc);
        out.pre_f = std::move(hc.pre_mix);
        out.post_f = std::move(hc.post_mix);
        out.comb_f = std::move(hc.comb);
    }
    out.ffn_pre = hc_pre_combine(out.res_mid, out.pre_f, hidden);

    // -------------------------------------------------------------------
    // 2.8 — FFN RMSNorm
    // -------------------------------------------------------------------
    out.ffn_norm = rmsnorm(out.ffn_pre, half_bits_to_doubles(w.ffn_norm, hidden), shape.eps);

    // -------------------------------------------------------------------
    // 2.9 — router. Hash table on layers < 3 (no bias, no top-k); biased flat
    //       top-6 otherwise.
    // -------------------------------------------------------------------
    out.router_logits = matvec(
        shape.num_experts, hidden, out.ffn_norm,
        [&](size_t o, size_t i) { return f16(w.gate_weight, o * hidden + i); });

    if (w.tid2eid_row != nullptr) {
        const std::vector<int64_t> row(
            w.tid2eid_row, w.tid2eid_row + shape.top_k);
        const RouterSelection sel =
            router_hash(out.router_logits, row, shape.routed_scaling);
        out.routed_ids = sel.ids;
        out.routed_weights = sel.weights;
    } else {
        std::vector<double> bias;
        if (w.gate_bias != nullptr) {
            bias.assign(w.gate_bias, w.gate_bias + shape.num_experts);
        }
        const RouterSelection sel =
            router_topk(out.router_logits, bias, shape.top_k, shape.routed_scaling);
        out.routed_ids = sel.ids;
        out.routed_weights = sel.weights;
    }

    // -------------------------------------------------------------------
    // 2.10 — routed experts, then the shared expert.
    // -------------------------------------------------------------------
    out.routed_expert_outputs.resize(out.routed_ids.size());
    out.routed_sum.assign(hidden, 0.0);
    for (size_t k = 0; k < out.routed_ids.size(); ++k) {
        const uint8_t* payload = w.routed_payloads[k];
        if (payload == nullptr) {
            throw std::invalid_argument(
                "dsv4_oracle: layer body is missing a routed expert payload");
        }
        out.routed_expert_outputs[k] = expert_ffn(
            payload, out.ffn_norm, shape.swiglu_limit);
        for (size_t i = 0; i < hidden; ++i) {
            out.routed_sum[i] += out.routed_weights[k] * out.routed_expert_outputs[k][i];
        }
    }

    out.shared_out = dense_ffn(
        shape.intermediate, hidden, out.ffn_norm,
        half_bits_to_doubles(w.shared_w1,
            static_cast<size_t>(shape.intermediate) * hidden),
        half_bits_to_doubles(w.shared_w3,
            static_cast<size_t>(shape.intermediate) * hidden),
        half_bits_to_doubles(w.shared_w2,
            static_cast<size_t>(hidden) * shape.intermediate),
        shape.swiglu_limit);

    // 2.10.5 — the term set is `routed_sum + shared`. See the ordering note.
    out.moe_out.assign(hidden, 0.0);
    for (size_t i = 0; i < hidden; ++i) {
        out.moe_out[i] = out.routed_sum[i] + out.shared_out[i];
    }

    // -------------------------------------------------------------------
    // 2.11 — HC FFN post-mix -> res_out, which is the next layer's res_in.
    // -------------------------------------------------------------------
    out.res_out = hc_post(out.moe_out, out.res_mid, out.post_f, out.comb_f, hidden);
    return out;
}

} // namespace aeon::reference
