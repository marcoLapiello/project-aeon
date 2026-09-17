#pragma once

// -----------------------------------------------------------------------------
// The sampler — the logit-processor seam, and the decision that follows it.
//
// This is the composition plan's **G3** and its phase **P3**, the last op of the
// token path (§2.1, step B6). It is deliberately *not* part of `V4Graph`: the
// graph owns what happens in the model, and sampling is not a model operation.
// The graph hands over fp16 logits; this decides a token.
//
// Plan Step 5 puts three requirements on it, and only the first is ordinary:
//
//   1. temperature, top-k and top-p, with the softmax in fp32;
//   2. a **logit-processor seam** — a hook that may mask or bias the logits
//      *before* the decision. The plan calls this **not deferrable** (§6.4),
//      because constrained/structured output — tool-call JSON — *is* a logit
//      mask, and a closed argmax with no hook forces a pipeline change later;
//   3. argmax first, which for the untruncated defaults is also the model's own
//      decision.
//
// The seam mutates the logits in place rather than returning them, because that
// is what a mask is, and because the sampled probability of an excluded token
// must be **exactly zero** rather than small — a `-inf` logit is what makes that
// true instead of approximately true.
//
// Order: `processor -> temperature -> top-k -> top-p -> softmax -> decide`. The
// seam is first so a mask can make a token impossible; temperature precedes the
// truncations because the standard ordering is what a reference implementation's
// numbers mean. The gate asserts the order rather than assuming it (a `+inf`
// logit must survive `top_k = 1`).
//
// Determinism, and why the generator is written out rather than taken from
// `<random>`. The gate's first clause is "seeded replay is bit-identical", and a
// replay is only bit-identical *by construction* if both the bit stream and the
// mapping to `[0, 1)` are specified here. `V4SplitMix64` is a handful of
// documented lines, so the token sequence is a property of this file.
//
// Precision. The decision runs on **host fp32** logits: the fp32 softmax and the
// seam both live here, so widening on the device would only add a second copy and
// 259 KB more traffic per token. Two things follow:
//
//   * the **greedy** path with no seam installed never widens at all. There is
//     nothing for the host to do but pick the largest, and the certified device
//     argmax pair does that with a 4-byte readback — which is the plan's
//     "device + 4 B host" (§2.2) and the pre-rewrite graph's own design;
//   * widening fp16 -> fp32 is **exact**, so the device argmax and the host
//     argmax are the same function on the same values. The gate asserts equality
//     rather than agreement-within-a-tolerance, and that is why that assertion is
//     strong rather than decorative.
//
// Two argmaxes, on purpose. The device pair is a parallel reduction over fp16;
// `sampler_ops::argmax_of` is the scalar fp32 rule the sampler falls back to when
// a seam is installed or the decision is sampled. They are two implementations of
// one *rule* — ties to the lower index, which is exactly what the certified
// kernel reduces with — and the gate asserts they agree. That agreement is the
// point, not a redundancy.
//
// What is not here. The engine that binds the tokenizer, the prompt encoder and
// the generation loop is P4 (`core/v4_engine.hpp`); the artifact's sampling
// policy is a `config.json` fact the engine reads and passes in. This file knows
// nothing about prompts, turns or stop conditions.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aeon::core {

// -----------------------------------------------------------------------------
// The seam
// -----------------------------------------------------------------------------

// The logit-processor hook: it receives the **host fp32 logits** and the
// vocabulary width and mutates them in place. A mask writes `-INFINITY`, a bias
// adds, a grammar does both.
using V4LogitProcessor = std::function<void(float* logits, uint32_t vocab)>;

// -----------------------------------------------------------------------------
// The configuration
// -----------------------------------------------------------------------------

struct V4SamplerConfig {
    // `<= 0` is the greedy path — the `T -> 0` limit, and the plan's "argmax
    // first" for the untruncated defaults. The artifact's own generation policy
    // (`do_sample = true, temperature = 1, top_p = 1` — plan Step 5) is a *policy*
    // the engine reads and passes in, not a property of this type, which is why
    // the shipped default is the deterministic one.
    float temperature{0.0f};

    // `0` disables truncation; `>= vocab` is equivalent to disabled.
    uint32_t top_k{0};

    // `>= 1` disables truncation. The artifact's own value is exactly `1.0`, so
    // the shipped path sorts nothing at all.
    float top_p{1.0f};

    // The generator's seed. The same seed and the same logits give the same
    // tokens, bit for bit, by construction (see `V4SplitMix64`).
    uint64_t seed{0};
};

// -----------------------------------------------------------------------------
// The generator
// -----------------------------------------------------------------------------

// SplitMix64, written out rather than taken from `<random>`.
//
// A replay is only bit-identical *by construction* if both the bit stream and the
// mapping to `[0, 1)` are specified here, rather than being whatever the standard
// library happens to do this decade. This is the whole generator: no state to get
// wrong and no distribution object whose algorithm may differ between
// implementations.
class V4SplitMix64 {
public:
    explicit V4SplitMix64(uint64_t seed = 0) noexcept : state_(seed) {}

    void seed(uint64_t value) noexcept { state_ = value; }
    uint64_t seed() const noexcept { return state_; }

    uint64_t next_u64() noexcept {
        state_ += 0x9E3779B97F4A7C15ull;  // the golden-ratio odd increment
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // The high 53 bits scaled by `2^-53`: the result is in `[0, 1)` and can never
    // round *up* to 1, which is what makes a categorical walk terminate.
    double next_unit() noexcept {
        return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
    }

private:
    uint64_t state_{0};
};

// -----------------------------------------------------------------------------
// The arithmetic — pure, host-only, and independently testable
// -----------------------------------------------------------------------------

// Each of these is one step of the plan's Step 5 and each is a total function on
// a logit vector, so the gate can pin it against a hand-built vector with no
// device and no model in the loop. They are in a `detail`-style namespace rather
// than private members so that they can be tested on their own, which is what
// makes a failing gate say *which* step is wrong.
namespace sampler_ops {

// The rule the certified device kernel reduces with: strictly-greater wins, so a
// tie is decided in favour of the **lower index**. Two implementations of one
// rule, and the gate asserts they agree on the same values.
inline uint32_t argmax_of(const float* logits, uint32_t n) noexcept {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

// Whether the seam left a decidable distribution: at least one **finite** logit.
//
// Both ways of leaving none are refused rather than answered. "Everything was
// excluded" has no largest logit to name and no distribution to draw from. "A
// token was promoted to `+inf`" has no probability either: `exp` of it is
// not-a-number, not a certainty — a promotion must be a large *finite* bias.
inline bool has_finite_logit(const float* logits, uint32_t n) noexcept {
    for (uint32_t i = 0; i < n; ++i) {
        if (std::isfinite(logits[i])) return true;
    }
    return false;
}

// `T=1` is the identity and returns without touching the vector. `-inf` stays
// `-inf` under a positive scale, so a masked token survives a temperature change.
inline void apply_temperature(float* logits, uint32_t n, float temperature) noexcept {
    if (temperature == 1.0f) return;
    const float inverse = 1.0f / temperature;
    for (uint32_t i = 0; i < n; ++i) logits[i] *= inverse;
}

// Keeps the `k` largest and sends the rest to `-inf`.
//
// The threshold is the k-th largest, and *every* element at or above it is kept:
// a tie at the boundary is kept whole rather than broken silently, which is the
// same "a tie is not an implementation detail" rule the argmax uses. So at least
// `k` survive, and exactly `k` when the threshold is unique.
inline void apply_top_k(float* logits, uint32_t n, uint32_t k) {
    if (k == 0 || k >= n) return;

    std::vector<uint32_t> order(n);
    for (uint32_t i = 0; i < n; ++i) order[i] = i;
    std::nth_element(order.begin(), order.begin() + (k - 1), order.end(),
                     [logits](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });

    const float threshold = logits[order[k - 1]];
    for (uint32_t i = 0; i < n; ++i) {
        if (logits[i] < threshold) logits[i] = -std::numeric_limits<float>::infinity();
    }
}

// The nucleus: the smallest set of the largest logits whose mass reaches `top_p`.
//
// The boundary element is **included**, so the kept mass is `>= top_p` and never
// just short of it, and at least one element always survives. The mass is
// measured against the *whole* support — masking the logits and letting the final
// softmax renormalise gives the correct conditional distribution, but the cut
// itself has to be chosen on the full set or the nucleus drifts.
//
// No-op at `top_p >= 1`, which is both the plan's default and the artifact's own
// value: the untruncated path sorts nothing and allocates nothing.
inline void apply_top_p(float* logits, uint32_t n, float top_p) {
    if (top_p >= 1.0f) return;

    float max_logit = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < n; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    if (!std::isfinite(max_logit)) return;  // nothing left to truncate

    // fp64 accumulation, matching the fixed-order discipline the rest of the
    // rewrite uses: the mass is a sum over 129280 terms and its order must not be
    // a property of the vector's layout.
    double total = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        total += std::exp(static_cast<double>(logits[i]) - max_logit);
    }
    if (!(total > 0.0)) return;

    std::vector<uint32_t> order(n);
    for (uint32_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [logits](uint32_t a, uint32_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;  // a tie is the lower index first, as everywhere else
    });

    double accumulated = 0.0;
    uint32_t keep = 0;
    for (uint32_t rank = 0; rank < n; ++rank) {
        accumulated += std::exp(static_cast<double>(logits[order[rank]]) - max_logit) / total;
        ++keep;  // the boundary element is kept, and `keep >= 1` always
        if (accumulated >= static_cast<double>(top_p)) break;
    }
    for (uint32_t rank = keep; rank < n; ++rank) {
        logits[order[rank]] = -std::numeric_limits<float>::infinity();
    }
}

// fp32 probabilities, accumulated in fp64. The max is subtracted: the logits here
// span their own range and an unshifted `exp` overflows above ~88.
inline void softmax_in_place(float* logits, uint32_t n) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < n; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    double total = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        total += std::exp(static_cast<double>(logits[i]) - max_logit);
    }
    const double inverse = 1.0 / total;
    for (uint32_t i = 0; i < n; ++i) {
        logits[i] = static_cast<float>(
            std::exp(static_cast<double>(logits[i]) - max_logit) * inverse);
    }
}

// The categorical draw.
//
// The walk is in **index order**, so the sequence a seed produces is a property
// of the seed and not of a sort. `next_unit()` is in `[0, 1)` and fp32
// probabilities may sum to a hair below 1, so the last index with positive
// probability is returned as the fallback: a `u` just under 1 must not fall off
// the end of the vector.
inline uint32_t sample_from_probs(const float* probs, uint32_t n, V4SplitMix64& rng) noexcept {
    const double u = rng.next_unit();
    double accumulated = 0.0;
    uint32_t last_positive = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (probs[i] > 0.0f) last_positive = i;
        accumulated += static_cast<double>(probs[i]);
        if (u < accumulated) return i;
    }
    return last_positive;
}

}  // namespace sampler_ops

// -----------------------------------------------------------------------------
// The sampler
// -----------------------------------------------------------------------------

// Owns the seam, the configuration, the generator, and the device workspace for
// the certified argmax pair. Sized from the vocabulary and nothing else, so it
// needs no host and no model to construct — which is what keeps the gate cheap.
class V4Sampler {
public:
    explicit V4Sampler(uint32_t vocab) : vocab_(vocab) {
        if (vocab_ == 0) {
            throw std::invalid_argument("V4Sampler: the vocabulary must be non-empty");
        }
        // One partial per block of 256 lanes, and 505 of those is exactly the
        // model's 129280 logits — but the count is computed, so a different
        // vocabulary sizes its own workspace instead of overrunning this one.
        argmax_blocks_ = (vocab_ + 255) / 256;

        CHECK_HIP(hipMalloc(&d_partial_vals_, argmax_blocks_ * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_partial_idx_, argmax_blocks_ * sizeof(int32_t)));
        CHECK_HIP(hipMalloc(&d_argmax_result_, sizeof(int32_t)));

        host_logits_.resize(vocab_);
        staging_half_.resize(vocab_);
        rng_.seed(config_.seed);
    }

    ~V4Sampler() {
        free();
    }

    V4Sampler(const V4Sampler&) = delete;
    V4Sampler& operator=(const V4Sampler&) = delete;
    V4Sampler(V4Sampler&&) = delete;
    V4Sampler& operator=(V4Sampler&&) = delete;

    uint32_t vocab() const noexcept { return vocab_; }
    uint32_t argmax_blocks() const noexcept { return argmax_blocks_; }

    const V4SamplerConfig& config() const noexcept { return config_; }

    // Validated, not clamped (trap 40's discipline applied to configuration): a
    // negative, NaN or infinite temperature, or a `top_p` outside `(0, 1]`, is
    // refused. The alternative — a silently different distribution — is a
    // behaviour change no downstream comparison could attribute.
    void set_config(const V4SamplerConfig& config) {
        if (!std::isfinite(config.temperature) || config.temperature < 0.0f) {
            throw std::invalid_argument(
                "V4Sampler: temperature must be finite and non-negative, got " +
                std::to_string(config.temperature));
        }
        if (!(config.top_p > 0.0f) || config.top_p > 1.0f) {
            throw std::invalid_argument(
                "V4Sampler: top_p must lie in (0, 1], got " + std::to_string(config.top_p));
        }
        config_ = config;
        rng_.seed(config_.seed);
    }

    void reseed(uint64_t seed) {
        config_.seed = seed;
        rng_.seed(seed);
    }

    // --- the seam ------------------------------------------------------------

    // Installing a processor forces the host path: the device argmax cannot see a
    // mask, so a constrained decision costs the widening. That is the cost of a
    // constraint, not a defect.
    void set_logit_processor(V4LogitProcessor processor) { processor_ = std::move(processor); }
    void clear_logit_processor() { processor_ = nullptr; }
    bool has_logit_processor() const noexcept { return static_cast<bool>(processor_); }

    // --- the decisions -------------------------------------------------------

    // The certified device argmax, on its own: the two-phase reduction, then a
    // 4-byte readback and a stream sync at the token boundary.
    uint32_t device_argmax(const half* d_logits, hipStream_t stream) const {
        kernel::v4_argmax_fp16_partial_kernel<<<argmax_blocks_, 256, 0, stream>>>(
            d_logits, static_cast<int>(vocab_), d_partial_vals_, d_partial_idx_);
        kernel::v4_argmax_partial_reduce_kernel<<<1, 256, 0, stream>>>(
            d_partial_vals_, d_partial_idx_, static_cast<int>(argmax_blocks_), d_argmax_result_);

        int32_t host_id = 0;
        CHECK_HIP(hipMemcpyAsync(&host_id, d_argmax_result_, sizeof(int32_t),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipStreamSynchronize(stream));
        return static_cast<uint32_t>(host_id);
    }

    // The decision, on host fp32 logits, **in place**: seam, temperature, top-k,
    // top-p, then argmax or a draw. No device, no allocation, no widening — which
    // is what lets the gate pin every step of plan Step 5 exactly.
    uint32_t decide(float* logits) {
        const uint32_t n = vocab_;

        if (processor_) processor_(logits, n);
        if (config_.temperature > 0.0f) {
            sampler_ops::apply_temperature(logits, n, config_.temperature);
        }
        sampler_ops::apply_top_k(logits, n, config_.top_k);
        sampler_ops::apply_top_p(logits, n, config_.top_p);

        // A seam may legitimately exclude *almost* everything, but leaving nothing
        // decidable is not a decision. Refused rather than answered with index 0,
        // which is what `argmax_of` would do and what a saturated softmax would
        // turn into a silent uniform draw.
        if (!sampler_ops::has_finite_logit(logits, n)) {
            throw std::runtime_error(
                "V4Sampler: the logit processor left no finite logit — there is no "
                "distribution to decide from (a promotion must be a finite bias)");
        }

        // The greedy path is the `T -> 0` limit. The truncations cannot move it:
        // they keep the largest, so the argmax of the processed logits is the
        // answer whether or not top-k and top-p ran.
        if (!(config_.temperature > 0.0f)) {
            return sampler_ops::argmax_of(logits, n);
        }

        sampler_ops::softmax_in_place(logits, n);
        return sampler_ops::sample_from_probs(logits, n, rng_);
    }

    // The one entry point. `d_logits` is fp16 `[vocab]` on the device, exactly as
    // `V4Graph::forward_token` leaves it.
    //
    // Greedy with no seam takes the device path and reads back four bytes; the
    // widening happens only when the host actually has work to do.
    uint32_t select(const half* d_logits, hipStream_t stream) {
        last_decision_used_device_argmax_ = (config_.temperature <= 0.0f) && !processor_;
        if (last_decision_used_device_argmax_) {
            return device_argmax(d_logits, stream);
        }

        CHECK_HIP(hipMemcpyAsync(staging_half_.data(), d_logits,
                                 static_cast<size_t>(vocab_) * sizeof(half),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipStreamSynchronize(stream));
        for (uint32_t i = 0; i < vocab_; ++i) {
            host_logits_[i] = __half2float(staging_half_[i]);
        }
        return decide(host_logits_.data());
    }

    // The fp32 buffer from the last host-path `select`: the widened logits, and
    // then whatever the seam, the temperature and the truncations did to them (and
    // the probabilities, if a draw was taken). Empty until the first such call.
    const std::vector<float>& host_logits() const noexcept { return host_logits_; }

    // Whether the last `select` took the 4-byte device path. Exposed because an
    // implementation that always widened would be correct and 259 KB slower per
    // token with no other symptom — so "the fast path is the fast path" has to be
    // observable to be assertable.
    bool last_decision_used_device_argmax() const noexcept {
        return last_decision_used_device_argmax_;
    }

    void free() {
        if (d_partial_vals_) { (void)hipFree(d_partial_vals_); d_partial_vals_ = nullptr; }
        if (d_partial_idx_) { (void)hipFree(d_partial_idx_); d_partial_idx_ = nullptr; }
        if (d_argmax_result_) { (void)hipFree(d_argmax_result_); d_argmax_result_ = nullptr; }
    }

private:
    uint32_t vocab_{0};
    uint32_t argmax_blocks_{0};

    V4SamplerConfig config_{};
    V4SplitMix64 rng_{};
    V4LogitProcessor processor_{};

    // Device workspace for the certified argmax pair only. The logits themselves
    // are never copied whole to the device: the host path widens on the way back.
    float* d_partial_vals_{nullptr};
    int32_t* d_partial_idx_{nullptr};
    int32_t* d_argmax_result_{nullptr};

    std::vector<float> host_logits_{};
    std::vector<half> staging_half_{};

    bool last_decision_used_device_argmax_{false};
};

}  // namespace aeon::core
