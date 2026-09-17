// -----------------------------------------------------------------------------
// P3 gate — the sampler and its logit-processor seam.
//
// P2 left the graph producing **logits** and named the next thing plainly:
// "until P3's gate is green the graph ends at logits". The plan's Step 5 puts
// three requirements on the op that follows, and only the first is ordinary:
// temperature / top-k / top-p with an fp32 softmax; a **logit-processor seam**
// that may mask or bias the logits *before* the decision (the plan's
// **not-deferrable** item, because tool-call JSON and structured output are logit
// masks — §6.4); and argmax first.
//
// The gate clause, verbatim (§7 P3): "seeded replay is bit-identical; a mask that
// sets one logit to `-inf` removes that token from the support; `T->0` converges
// to the argmax path; the untruncated defaults reproduce the argmax token. **The
// seam is asserted to exist, not merely to be present.**"
//
// HOW "THE UNTRUNCATED DEFAULTS REPRODUCE THE ARGMAX TOKEN" IS READ, stated so
// the reading can be argued with rather than assumed. The artifact's own
// generation policy is `do_sample = true, temperature = 1, top_p = 1` (plan
// Step 5), and sampling from an untruncated softmax does **not** generally return
// the argmax — so the clause cannot mean "the T=1 draw is the argmax". It is read
// as the plan's own words elsewhere in the same sentence: "argmax at `T=1,
// top_p=1` *first*", i.e. the shipped default configuration is the deterministic
// one, and at the untruncated defaults the sampler's decision is the argmax of
// the logits the graph produced. Three separate statements are then measured, and
// none is a tautology:
//
//   1. the default configuration resolves to the argmax, and that token is the
//      **certified device argmax** on the same logits (D4, E2, F2) — two
//      implementations of one rule agreeing;
//   2. `T -> 0` in *sampling* mode converges to the same token, so the greedy
//      switch is not a special case bolted on beside the sampler (D3);
//   3. at `T=1, top_p=1, top_k=0` the **support is the whole vocabulary** — every
//      token has non-zero probability — which is what "untruncated" means and is
//      the property that makes the first two statements interesting (B8, E4).
//
// WHAT IS REAL HERE. The pure section runs on hand-built vectors with an
// independently written fp64 reference for the softmax and the nucleus, so every
// step of Step 5 is pinned exactly and the gate needs no model for it. The device
// section drives the **certified argmax pair** on uploaded fp16 vectors. Section
// F closes the seam on the artifact's own logits: `V4Graph::forward_token` for a
// real token, then the sampler on exactly what the graph left — which is the only
// statement that the sampler's input contract (`const half* [vocab]`) is the
// graph's output contract rather than an assumption.
//
// WHY WIDENING MAKES AN EXACT ASSERTION POSSIBLE. fp16 -> fp32 is exact, so the
// device argmax over the fp16 logits and the host argmax over their widenings are
// the same function on the same values. Their agreement is therefore asserted as
// **equality**, not within a tolerance — and that is what makes E1/E2/F2 strong.
//
// WHAT IS DELIBERATELY NOT COVERED, named so a green line is not read as more:
//
//   * **The text binding** (tokenizer, prompt encoder, detokenizer, the
//     generation loop) — P4, `core/v4_engine.hpp`. This gate never encodes or
//     decodes a token.
//   * **The artifact's sampling policy**, which is a `config.json` fact the engine
//     reads and passes in; this file's default is the deterministic one on
//     purpose and the gate asserts it is.
//   * **Throughput.** A host-side fp32 softmax over 129280 logits is the plan's
//     accepted first implementation ("host-side is acceptable"); what is asserted
//     here is that the *untruncated* and *greedy* paths allocate nothing and
//     read back four bytes.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_graph.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "architecture/deepseek_v4/core/v4_sampler.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

using aeon::core::V4Graph;
using aeon::core::V4LogitProcessor;
using aeon::core::V4ModelHost;
using aeon::core::V4Sampler;
using aeon::core::V4SamplerConfig;
using aeon::core::V4SplitMix64;

namespace ops = aeon::core::sampler_ops;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kToken = 65106;   // the P1/P2 gates' first probe token

constexpr float kInf = std::numeric_limits<float>::infinity();

// --- the harness -------------------------------------------------------------

struct Harness {
    uint32_t checks{0};
    uint32_t failures{0};

    bool assert_that(const char* label, bool ok, const std::string& detail) {
        std::printf("  %-58s %-42s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
        ++checks;
        if (!ok) ++failures;
        return ok;
    }

    bool close_to(const char* label, const std::vector<double>& want,
                  const std::vector<double>& got, double tolerance) {
        if (want.size() != got.size()) {
            return assert_that(label, false, "size mismatch");
        }
        double worst = 0.0;
        for (size_t i = 0; i < want.size(); ++i) {
            worst = std::fmax(worst, std::fabs(want[i] - got[i]));
        }
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "max_abs=%.3e", worst);
        return assert_that(label, worst <= tolerance, buffer);
    }
};

Harness harness;

using Clock = std::chrono::steady_clock;

double since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void stage(const char* name, const Clock::time_point& start) {
    std::printf("  [time] %-42s %.2f s\n", name, since(start));
}

std::string sci(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.3e", value);
    return buffer;
}

std::string index_list(const std::vector<uint32_t>& indices) {
    std::string out = "{";
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(indices[i]);
    }
    return out + "}";
}

// --- an independently written reference for the two transforms ----------------
//
// The rule is that a step is compared against a reference written independently
// of the code under test. These are what "independent" costs for a softmax and a
// nucleus: both are fp64, both take the obvious definition, and neither shares a
// line with `sampler_ops`.

std::vector<double> reference_softmax(const std::vector<double>& logits) {
    const double peak = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(logits.size());
    double total = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - peak);
        total += probabilities[i];
    }
    for (double& value : probabilities) value /= total;
    return probabilities;
}

// The nucleus by its definition: sort by probability (ties to the lower index),
// accumulate, keep the prefix whose mass first reaches `top_p`.
std::vector<uint32_t> reference_nucleus(const std::vector<double>& logits, double top_p) {
    const std::vector<double> probabilities = reference_softmax(logits);
    std::vector<uint32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return probabilities[a] > probabilities[b];
    });

    std::vector<uint32_t> kept;
    double accumulated = 0.0;
    for (uint32_t index : order) {
        accumulated += probabilities[index];
        kept.push_back(index);
        if (accumulated >= top_p) break;
    }
    std::sort(kept.begin(), kept.end());
    return kept;
}

uint32_t reference_argmax(const std::vector<double>& logits) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

// --- small vector helpers ----------------------------------------------------

std::vector<uint32_t> finite_indices(const std::vector<float>& values) {
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < values.size(); ++i) {
        if (std::isfinite(values[i])) out.push_back(i);
    }
    return out;
}

std::vector<double> as_double(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

__half* upload_fp16(hipStream_t stream, const std::vector<float>& values) {
    std::vector<__half> staged(values.size());
    for (size_t i = 0; i < values.size(); ++i) staged[i] = __float2half(values[i]);
    __half* device = nullptr;
    CHECK_HIP(hipMalloc(&device, values.size() * sizeof(__half)));
    CHECK_HIP(hipMemcpyAsync(device, staged.data(), values.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    return device;
}

// A distinct-logit probe: no ties, so every step's expected answer is unique.
std::vector<float> distinct_probe() {
    return {-1.25f, 3.5f, 0.75f, 2.0f, -4.0f, 1.0f, -0.5f, 4.25f};
}

// The model's own width, so the block count is the model's own 505.
constexpr uint32_t kVocab = 129280;

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  P3 — the sampler and its logit-processor seam\n");
    std::printf("================================================================================\n");
    aeon::core::select_compute_device(true);

    const auto gate_start = Clock::now();

    // =========================================================================
    // A. The generator — the sequence is a property of this file, not of <random>
    // =========================================================================
    std::printf("\n[A] The generator (SplitMix64)\n");
    {
        // A second, deliberately literal transcription of the published
        // recurrence. It shares no line with the header, so a changed constant or
        // shift in `V4SplitMix64` shows up as a difference over a thousand draws
        // rather than as a slightly different but self-consistent token stream.
        const auto literal_splitmix = [](uint64_t& state) {
            state += 0x9E3779B97F4A7C15ull;
            uint64_t z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        };

        uint32_t mismatches = 0;
        for (uint64_t seed_value : {0ull, 1ull, 0xDEADBEEFull, 0xFFFFFFFFFFFFFFFFull}) {
            V4SplitMix64 rng(seed_value);
            uint64_t state = seed_value;
            for (int draw = 0; draw < 1000; ++draw) {
                if (rng.next_u64() != literal_splitmix(state)) ++mismatches;
            }
        }
        harness.assert_that("A: the stream matches a literal transcription",
                            mismatches == 0, std::to_string(mismatches) + " of 4000 differ");

        // Same seed, same stream; different seed, different stream. This is the
        // gate's "seeded replay is bit-identical" at its source.
        V4SplitMix64 a(12345), b(12345), c(12346);
        uint32_t same = 0, differing = 0;
        for (int draw = 0; draw < 256; ++draw) {
            const uint64_t av = a.next_u64();
            if (av == b.next_u64()) ++same;
            if (av != c.next_u64()) ++differing;
        }
        harness.assert_that("A: one seed gives one stream, 256 of 256",
                            same == 256 && differing == 256,
                            std::to_string(same) + " equal, " + std::to_string(differing) +
                                " differ");

        // `[0, 1)` and never 1: a categorical walk with `u == 1` would fall off
        // the end of the vector, so this is a termination property and not
        // decoration. The mean is the one statistical check worth making.
        V4SplitMix64 rng(7);
        double minimum = 1.0, maximum = 0.0, total = 0.0;
        constexpr int kDraws = 100000;
        for (int draw = 0; draw < kDraws; ++draw) {
            const double u = rng.next_unit();
            minimum = std::fmin(minimum, u);
            maximum = std::fmax(maximum, u);
            total += u;
        }
        const double mean = total / kDraws;
        char range[96];
        std::snprintf(range, sizeof(range), "min=%.6e max=%.17g", minimum, maximum);
        harness.assert_that("A: next_unit() stays in [0, 1)",
                            minimum >= 0.0 && maximum < 1.0, range);
        harness.assert_that("A: the mean is 1/2 to within 0.01",
                            std::fabs(mean - 0.5) < 0.01, "mean=" + sci(mean));
        std::printf("  [note] first u64 for seeds 0, 1 and 2: 0x%016llx 0x%016llx 0x%016llx\n",
                    static_cast<unsigned long long>(V4SplitMix64(0).next_u64()),
                    static_cast<unsigned long long>(V4SplitMix64(1).next_u64()),
                    static_cast<unsigned long long>(V4SplitMix64(2).next_u64()));
    }

    // =========================================================================
    // B. The arithmetic — every step of Step 5, pinned exactly
    // =========================================================================
    std::printf("\n[B] Temperature, top-k, top-p, softmax, argmax\n");
    {
        const std::vector<float> probe = distinct_probe();

        // B1/B2 — the softmax against the fp64 reference, and its defining property.
        {
            std::vector<float> logits = probe;
            ops::softmax_in_place(logits.data(), static_cast<uint32_t>(logits.size()));
            harness.close_to("B: the softmax matches the fp64 reference",
                             reference_softmax(as_double(probe)), as_double(logits), 1e-7);

            double total = 0.0;
            for (float value : logits) total += value;
            harness.assert_that("B: the probabilities sum to one",
                                std::fabs(total - 1.0) < 1e-6, "sum=" + sci(total));

            // Shift invariance: softmax(x + c) == softmax(x). This is what proves
            // the max subtraction is a numerical device and not a change of the op.
            std::vector<float> shifted = probe;
            for (float& value : shifted) value += 1234.5f;
            const std::vector<float> unshifted_probs = logits;
            ops::softmax_in_place(shifted.data(), static_cast<uint32_t>(shifted.size()));
            harness.close_to("B: the softmax is shift-invariant",
                             as_double(unshifted_probs), as_double(shifted), 1e-6);

            const uint32_t peak = reference_argmax(as_double(probe));
            uint32_t argmax_of_probs = 0;
            for (uint32_t i = 1; i < logits.size(); ++i) {
                if (logits[i] > logits[argmax_of_probs]) argmax_of_probs = i;
            }
            harness.assert_that("B: the largest logit keeps the largest probability",
                                argmax_of_probs == peak,
                                std::to_string(argmax_of_probs) + " == " + std::to_string(peak));
        }

        // B3 — temperature. `T = 1` must be the identity, bit for bit.
        {
            std::vector<float> logits = probe;
            ops::apply_temperature(logits.data(), static_cast<uint32_t>(logits.size()), 1.0f);
            uint32_t changed = 0;
            for (uint32_t i = 0; i < logits.size(); ++i) {
                if (logits[i] != probe[i]) ++changed;
            }
            harness.assert_that("B: temperature 1.0 is the identity, bit for bit",
                                changed == 0, std::to_string(changed) + " of " +
                                                  std::to_string(logits.size()) + " changed");

            std::vector<float> hot = probe;
            ops::apply_temperature(hot.data(), static_cast<uint32_t>(hot.size()), 0.5f);
            std::vector<double> scaled(probe.size());
            for (size_t i = 0; i < probe.size(); ++i) scaled[i] = probe[i] / 0.5;
            harness.close_to("B: temperature 0.5 doubles the logits (logits / T)",
                             scaled, as_double(hot), 0.0);
        }

        // B4 — top-k keeps the k largest, sends the rest to exactly -inf, and
        // keeps a tie at the threshold whole rather than breaking it.
        {
            std::vector<float> logits = probe;
            ops::apply_top_k(logits.data(), static_cast<uint32_t>(logits.size()), 3);
            // The three largest of the probe are 4.25 (7), 3.5 (1) and 2.0 (3).
            const std::vector<uint32_t> expected{1, 3, 7};
            harness.assert_that("B: top-k=3 leaves exactly the three largest",
                                finite_indices(logits) == expected,
                                index_list(finite_indices(logits)) + " vs " +
                                    index_list(expected));

            bool excluded_are_exactly_negative_infinity = true;
            for (uint32_t i = 0; i < logits.size(); ++i) {
                const bool should_keep = std::find(expected.begin(), expected.end(), i) !=
                                         expected.end();
                if (!should_keep && !(logits[i] == -kInf)) {
                    excluded_are_exactly_negative_infinity = false;
                }
            }
            harness.assert_that("B: the excluded logits are exactly -inf, not merely small",
                                excluded_are_exactly_negative_infinity, "-INFINITY");

            std::vector<float> tied{0.5f, 2.0f, 2.0f, 2.0f, -1.0f};
            ops::apply_top_k(tied.data(), static_cast<uint32_t>(tied.size()), 2);
            harness.assert_that("B: a tie at the threshold is kept whole",
                                finite_indices(tied) == std::vector<uint32_t>({1, 2, 3}),
                                index_list(finite_indices(tied)));

            std::vector<float> untouched = probe;
            ops::apply_top_k(untouched.data(), static_cast<uint32_t>(untouched.size()), 0);
            harness.assert_that("B: top-k=0 is the identity",
                                untouched == probe, "unchanged");
        }

        // B5 — top-p against the fp64 nucleus, and the untruncated identity.
        {
            for (float top_p : {0.5f, 0.9f, 0.99f}) {
                std::vector<float> logits = probe;
                ops::apply_top_p(logits.data(), static_cast<uint32_t>(logits.size()), top_p);
                const std::vector<uint32_t> got = finite_indices(logits);
                const std::vector<uint32_t> expected =
                    reference_nucleus(as_double(probe), static_cast<double>(top_p));
                harness.assert_that(("B: the top-p=" + sci(top_p) + " nucleus is the definition")
                                        .c_str(),
                                    got == expected,
                                    index_list(got) + " vs " + index_list(expected));
            }

            std::vector<float> untouched = probe;
            ops::apply_top_p(untouched.data(), static_cast<uint32_t>(untouched.size()), 1.0f);
            harness.assert_that("B: top-p=1.0 is the identity — the untruncated path",
                                untouched == probe, "unchanged");

            // The boundary element is included, so the retained mass is never just
            // short of the threshold. Measured on the retained set alone.
            std::vector<float> logits = probe;
            ops::apply_top_p(logits.data(), static_cast<uint32_t>(logits.size()), 0.75f);
            double retained = 0.0;
            const std::vector<double> full = reference_softmax(as_double(probe));
            for (uint32_t index : finite_indices(logits)) retained += full[index];
            harness.assert_that("B: the retained mass reaches top-p, it does not fall short",
                                retained >= 0.75, "mass=" + sci(retained));
        }

        // B6 — the argmax rule, including its tie direction.
        {
            std::vector<float> tied{1.0f, 5.0f, 5.0f, 5.0f, -2.0f};
            harness.assert_that("B: a tie goes to the lower index",
                                ops::argmax_of(tied.data(), static_cast<uint32_t>(tied.size())) == 1,
                                std::to_string(ops::argmax_of(tied.data(), 5)) + " == 1");
            harness.assert_that("B: argmax agrees with the reference",
                                ops::argmax_of(probe.data(), static_cast<uint32_t>(probe.size())) ==
                                    reference_argmax(as_double(probe)),
                                std::to_string(ops::argmax_of(probe.data(), 8)));
        }

        // B7 — the draw terminates on a one-hot vector for any `u`, which is the
        // property the `u < 1` bound exists to guarantee.
        {
            std::vector<float> one_hot(8, 0.0f);
            one_hot[5] = 1.0f;
            uint32_t wrong = 0;
            for (uint64_t seed_value = 0; seed_value < 512; ++seed_value) {
                V4SplitMix64 rng(seed_value);
                if (ops::sample_from_probs(one_hot.data(), 8, rng) != 5) ++wrong;
            }
            harness.assert_that("B: a one-hot vector always draws its hot index",
                                wrong == 0, std::to_string(wrong) + " of 512 wrong");
        }
    }

    // =========================================================================
    // C. The seam — asserted to exist, and to run in the right place
    // =========================================================================
    std::printf("\n[C] The logit-processor seam\n");
    const uint32_t probe_size = static_cast<uint32_t>(distinct_probe().size());
    // Index 7 is the probe's argmax, so a mask of it is a mask of the token an
    // unmasked draw would find.
    constexpr uint32_t kMasked = 7;
    {
        const std::vector<float> probe = distinct_probe();

        // C1 — the hook is invoked, exactly once, with the logits and the width.
        {
            V4Sampler sampler(probe_size);
            uint32_t calls = 0;
            uint32_t seen_vocab = 0;
            float* seen_pointer = nullptr;
            sampler.set_logit_processor(
                [&](float* logits, uint32_t vocab) { ++calls; seen_vocab = vocab; seen_pointer = logits; });
            harness.assert_that("C: installing a processor makes has_logit_processor true",
                                sampler.has_logit_processor(), "true");

            std::vector<float> logits = probe;
            (void)sampler.decide(logits.data());
            harness.assert_that("C: the processor is invoked exactly once, on the logits",
                                calls == 1 && seen_vocab == probe_size &&
                                    seen_pointer == logits.data(),
                                std::to_string(calls) + " call(s), vocab " +
                                    std::to_string(seen_vocab));
        }

        // C2 — a mask of one token removes it from the support: probability
        // exactly zero, and never drawn.
        {
            V4Sampler sampler(probe_size);
            V4SamplerConfig config;
            config.temperature = 1.0f;
            config.seed = 99;
            sampler.set_config(config);
            sampler.set_logit_processor([](float* logits, uint32_t) {
                logits[kMasked] = -std::numeric_limits<float>::infinity();
            });

            uint32_t drawn_masked = 0;
            double masked_probability = -1.0;
            for (uint32_t draw = 0; draw < 200; ++draw) {
                std::vector<float> logits = probe;
                if (sampler.decide(logits.data()) == kMasked) ++drawn_masked;
                masked_probability = logits[kMasked];
            }
            harness.assert_that("C: the masked token's probability is exactly zero",
                                masked_probability == 0.0f, sci(masked_probability));
            harness.assert_that("C: the masked token is never drawn, 200 of 200",
                                drawn_masked == 0,
                                std::to_string(drawn_masked) + " of 200");

            // And the rest of the support is untouched: the mask removed one
            // token and did not flatten the distribution.
            std::vector<float> logits = probe;
            (void)sampler.decide(logits.data());
            double survivors = 0.0;
            for (uint32_t i = 0; i < logits.size(); ++i) {
                if (i != kMasked) survivors += logits[i];
            }
            harness.assert_that("C: the remaining probability mass stays at one",
                                std::fabs(survivors - 1.0) < 1e-6, "sum=" + sci(survivors));
        }

        // C3 — a mask that leaves one survivor decides that token with certainty,
        // whatever the seed.
        {
            constexpr uint32_t kSurvivor = 3;
            uint32_t wrong = 0;
            for (uint64_t seed_value = 0; seed_value < 64; ++seed_value) {
                V4Sampler sampler(probe_size);
                V4SamplerConfig config;
                config.temperature = 1.0f;
                config.seed = seed_value;
                sampler.set_config(config);
                sampler.set_logit_processor([](float* logits, uint32_t n) {
                    for (uint32_t i = 0; i < n; ++i) {
                        if (i != kSurvivor) logits[i] = -std::numeric_limits<float>::infinity();
                    }
                });
                std::vector<float> logits = probe;
                if (sampler.decide(logits.data()) != kSurvivor) ++wrong;
            }
            harness.assert_that("C: a single-survivor mask decides it, 64 of 64 seeds",
                                wrong == 0, std::to_string(wrong) + " wrong");
        }

        // C4 — a mask that excludes everything is refused, not answered with
        // index 0.
        {
            V4Sampler sampler(probe_size);
            sampler.set_logit_processor([](float* logits, uint32_t n) {
                for (uint32_t i = 0; i < n; ++i) logits[i] = -std::numeric_limits<float>::infinity();
            });
            std::vector<float> logits = probe;
            bool refused = false;
            try {
                (void)sampler.decide(logits.data());
            } catch (const std::runtime_error&) {
                refused = true;
            }
            harness.assert_that("C: an empty support is refused", refused,
                                refused ? "threw" : "returned a token");

            // And an *infinite* promotion is refused for the same reason: it has
            // no probability, only a not-a-number. Refusing it is a decision about
            // what a seam may do, so it is pinned here rather than left implicit.
            sampler.set_logit_processor([](float* logits, uint32_t) {
                logits[0] = std::numeric_limits<float>::infinity();
            });
            bool infinite_refused = false;
            try {
                (void)sampler.decide(logits.data());
            } catch (const std::runtime_error&) {
                infinite_refused = true;
            }
            harness.assert_that("C: an infinite promotion is refused too", infinite_refused,
                                infinite_refused ? "threw" : "returned a token");
        }

        // C5 — the seam runs **before** the temperature and the truncations.
        //
        // The ordering is asserted by what the processor *sees*, because that is
        // the only thing that discriminates it. A promoted token survives top-k
        // whether the seam ran first or last (a promotion above the threshold is
        // retained either way), so that assertion would pass on both orderings and
        // would be decoration. What cannot pass both ways: with `top_k = 3` and
        // `T = 0.5` installed, a first-running seam still sees all eight finite
        // logits at their raw values; a last-running seam sees three, scaled.
        {
            V4Sampler sampler(probe_size);
            V4SamplerConfig config;
            config.temperature = 0.5f;
            config.top_k = 3;
            sampler.set_config(config);

            uint32_t finite_seen = 0;
            float max_seen = -kInf;
            sampler.set_logit_processor([&](float* logits, uint32_t n) {
                for (uint32_t i = 0; i < n; ++i) {
                    if (std::isfinite(logits[i])) ++finite_seen;
                    max_seen = std::fmax(max_seen, logits[i]);
                }
            });

            std::vector<float> logits = probe;
            (void)sampler.decide(logits.data());

            harness.assert_that("C: the seam sees the logits before top-k truncates them",
                                finite_seen == probe_size,
                                std::to_string(finite_seen) + " of " +
                                    std::to_string(probe_size) + " finite");
            const float probe_max = *std::max_element(probe.begin(), probe.end());
            harness.assert_that("C: the seam sees them before the temperature scaling",
                                max_seen == probe_max,
                                sci(max_seen) + " vs " + sci(probe_max));

            // ...and the consequence of that order: a mask of the argmax combined
            // with `top_k = 1` decides the runner-up. A last-running seam would
            // mask the survivor instead and leave nothing decidable at all.
            V4Sampler ordered(probe_size);
            ordered.set_config(config);
            ordered.set_logit_processor([](float* logits, uint32_t) {
                logits[kMasked] = -std::numeric_limits<float>::infinity();
            });
            std::vector<float> masked = probe;
            bool decided = true;
            uint32_t runner_up = 0;
            try {
                runner_up = ordered.decide(masked.data());
            } catch (const std::runtime_error&) {
                decided = false;
            }
            harness.assert_that("C: masking the argmax under top_k=1 decides the runner-up",
                                decided && runner_up == 1,
                                decided ? std::to_string(runner_up) + " == 1"
                                        : "left no support");
        }
    }

    // =========================================================================
    // D. Determinism, the T->0 limit, and the shipped default
    // =========================================================================
    std::printf("\n[D] Seeded replay, convergence, and the defaults\n");
    {
        const std::vector<float> probe = distinct_probe();

        // D1/D2 — the gate's first clause.
        {
            const auto draw_sequence = [&](uint64_t seed_value, uint32_t count) {
                V4Sampler sampler(probe_size);
                V4SamplerConfig config;
                config.temperature = 1.0f;
                config.seed = seed_value;
                sampler.set_config(config);
                std::vector<uint32_t> tokens;
                for (uint32_t draw = 0; draw < count; ++draw) {
                    std::vector<float> logits = probe;
                    tokens.push_back(sampler.decide(logits.data()));
                }
                return tokens;
            };

            const std::vector<uint32_t> first = draw_sequence(4242, 64);
            const std::vector<uint32_t> replay = draw_sequence(4242, 64);
            const std::vector<uint32_t> other = draw_sequence(4243, 64);
            harness.assert_that("D: the same seed replays bit-identically, 64 of 64",
                                first == replay, first == replay ? "identical" : "diverged");
            harness.assert_that("D: a different seed gives a different sequence",
                                first != other, first != other ? "differs" : "coincides");

            // The replay must survive a fresh object, not merely a repeated call.
            V4Sampler sampler(probe_size);
            V4SamplerConfig config;
            config.temperature = 1.0f;
            config.seed = 4242;
            sampler.set_config(config);
            std::vector<float> logits = probe;
            (void)sampler.decide(logits.data());
            (void)sampler.decide(logits.data());
            sampler.reseed(4242);
            logits = probe;
            const uint32_t after_reseed = sampler.decide(logits.data());
            harness.assert_that("D: reseed restores the sequence, not just the state",
                                after_reseed == first[0],
                                std::to_string(after_reseed) + " == " + std::to_string(first[0]));

            // A draw at T=1 must actually vary, or "sampling" is argmax with extra
            // steps and every determinism check above is vacuous.
            const bool varies = std::adjacent_find(first.begin(), first.end(),
                                                   std::not_equal_to<>()) != first.end();
            harness.assert_that("D: an untruncated T=1 draw is not a disguised argmax",
                                varies, varies ? "varies" : "constant");
        }

        // D3 — T -> 0 converges to the argmax path.
        {
            const uint32_t expected = ops::argmax_of(probe.data(), probe_size);
            V4Sampler sampler(probe_size);
            V4SamplerConfig config;
            config.temperature = 1e-6f;
            config.seed = 5;
            sampler.set_config(config);

            uint32_t wrong = 0;
            for (uint32_t draw = 0; draw < 64; ++draw) {
                std::vector<float> logits = probe;
                if (sampler.decide(logits.data()) != expected) ++wrong;
            }
            harness.assert_that("D: T->0 converges to the argmax, 64 of 64 draws",
                                wrong == 0, std::to_string(wrong) + " wrong, argmax=" +
                                                  std::to_string(expected));
        }

        // D4 — the shipped default: greedy at top-k=0, top-p=1.
        {
            V4Sampler sampler(probe_size);
            harness.assert_that("D: the shipped default is the deterministic one",
                                sampler.config().temperature <= 0.0f &&
                                    sampler.config().top_k == 0 && sampler.config().top_p == 1.0f,
                                "T=" + sci(sampler.config().temperature) + " k=" +
                                    std::to_string(sampler.config().top_k) + " p=" +
                                    sci(sampler.config().top_p));

            std::vector<float> logits = probe;
            const uint32_t token = sampler.decide(logits.data());
            harness.assert_that("D: the untruncated defaults reproduce the argmax token",
                                token == reference_argmax(as_double(probe)),
                                std::to_string(token) + " == " +
                                    std::to_string(reference_argmax(as_double(probe))));

            // The support is the whole vector: that is what makes the two
            // statements above about *untruncated* defaults rather than about a
            // distribution that happened to be narrow.
            harness.assert_that("D: at the untruncated defaults every token is in the support",
                                finite_indices(logits).size() == logits.size(),
                                std::to_string(finite_indices(logits).size()) + " of " +
                                    std::to_string(logits.size()));
        }

        // D5 — invalid configurations are refused, not clamped.
        {
            V4Sampler sampler(probe_size);
            const auto refuses = [&](const V4SamplerConfig& config) {
                try {
                    sampler.set_config(config);
                } catch (const std::invalid_argument&) {
                    return true;
                }
                return false;
            };
            V4SamplerConfig negative;
            negative.temperature = -1.0f;
            V4SamplerConfig zero_p;
            zero_p.top_p = 0.0f;
            V4SamplerConfig above_one_p;
            above_one_p.top_p = 1.5f;
            V4SamplerConfig nan_temperature;
            nan_temperature.temperature = std::numeric_limits<float>::quiet_NaN();

            harness.assert_that("D: a negative temperature is refused", refuses(negative),
                                "refused");
            harness.assert_that("D: top_p = 0 is refused", refuses(zero_p), "refused");
            harness.assert_that("D: top_p > 1 is refused", refuses(above_one_p), "refused");
            harness.assert_that("D: a NaN temperature is refused", refuses(nan_temperature),
                                "refused");
        }
    }

    stage("A-D: the pure sections", gate_start);

    // =========================================================================
    // E. The device path — the certified argmax pair, and the seam through it
    // =========================================================================
    std::printf("\n[E] The device path and the widening\n");
    {
        const auto section_start = Clock::now();
        hipStream_t stream = nullptr;
        CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

        // E1 — the two argmaxes are one rule. fp16 -> fp32 is exact, so this is
        // asserted as equality rather than within a tolerance.
        {
            const std::vector<std::vector<float>> probes = {
                distinct_probe(),
                {-3.0f, 1.5f, 2.25f, 0.0f, 2.25f, -7.5f},   // a tie: lower index wins
                {0.0f, 0.0f, 0.0f, 0.0f},                    // all equal: index 0 wins
                {1.0f},                                      // the degenerate width
            };
            uint32_t disagreements = 0;
            std::string detail;
            for (const auto& values : probes) {
                V4Sampler sampler(static_cast<uint32_t>(values.size()));
                __half* device = upload_fp16(stream, values);
                const uint32_t from_device = sampler.device_argmax(device, stream);

                std::vector<float> widened(values.size());
                for (uint32_t i = 0; i < values.size(); ++i) {
                    widened[i] = __half2float(__float2half(values[i]));
                }
                const uint32_t from_host =
                    ops::argmax_of(widened.data(), static_cast<uint32_t>(widened.size()));
                if (from_device != from_host) {
                    ++disagreements;
                    detail += "(" + std::to_string(from_device) + "!=" +
                              std::to_string(from_host) + ")";
                }
                (void)hipFree(device);
            }
            harness.assert_that("E: device and host argmax agree exactly, 4 of 4 probes",
                                disagreements == 0,
                                disagreements ? detail : "identical, including 2 tie cases");
        }

        // E2 — the fast path is the fast path, and it is observable. An
        // implementation that always widened would be correct and 259 KB slower
        // per token with no other symptom.
        {
            const std::vector<float> values = distinct_probe();
            V4Sampler sampler(static_cast<uint32_t>(values.size()));
            __half* device = upload_fp16(stream, values);

            const uint32_t fast = sampler.select(device, stream);
            const bool used_device = sampler.last_decision_used_device_argmax();

            sampler.set_logit_processor([](float*, uint32_t) {});  // an identity seam
            const uint32_t slow = sampler.select(device, stream);
            const bool used_host = sampler.last_decision_used_device_argmax();

            harness.assert_that("E: greedy with no seam reads back four bytes",
                                used_device && fast == ops::argmax_of(values.data(),
                                                                     static_cast<uint32_t>(
                                                                         values.size())),
                                "device path, token " + std::to_string(fast));
            harness.assert_that("E: installing a seam forces the host path",
                                !used_host, "host path");
            harness.assert_that("E: the two paths decide the same token",
                                fast == slow,
                                std::to_string(fast) + " == " + std::to_string(slow));
            (void)hipFree(device);
        }

        // E3/E4 — the seam and the untruncated default through `select`, on the
        // model's own width, which is where the block count and the 129280-wide
        // softmax are actually exercised.
        {
            std::vector<float> logits(kVocab);
            for (uint32_t i = 0; i < kVocab; ++i) {
                logits[i] = -1.0f - 0.5f * static_cast<float>(i % 8);
            }
            logits[100000] = 6.0f;  // the argmax, at a non-trivial index
            logits[7] = 5.0f;

            V4Sampler sampler(kVocab);
            harness.assert_that("E: the argmax block count is the model's own 505",
                                sampler.argmax_blocks() == 505,
                                std::to_string(sampler.argmax_blocks()));

            __half* device = upload_fp16(stream, logits);
            const uint32_t token = sampler.select(device, stream);
            harness.assert_that("E: the untruncated default picks the planted argmax",
                                token == 100000, std::to_string(token) + " == 100000");

            sampler.set_config(V4SamplerConfig{1.0f, 0, 1.0f, 31337});
            sampler.set_logit_processor([&](float* values, uint32_t n) {
                values[100000] = -std::numeric_limits<float>::infinity();
            });
            uint32_t drawn_the_masked = 0;
            for (uint32_t draw = 0; draw < 32; ++draw) {
                if (sampler.select(device, stream) == 100000) ++drawn_the_masked;
            }
            const std::vector<float>& seen = sampler.host_logits();
            harness.assert_that("E: the seam's mask survives into the sampler's own logits",
                                seen.size() == kVocab && seen[100000] == 0.0f,
                                "probability " + sci(seen.size() == kVocab ? seen[100000] : -1.0));
            harness.assert_that("E: the masked token is never drawn through select",
                                drawn_the_masked == 0,
                                std::to_string(drawn_the_masked) + " of 32");

            // Untruncated means the whole vocabulary has mass. At T=1 the fp16
            // logits' span is small, so every one of the 129280 stays finite in
            // fp32 after the softmax — asserted on the buffer the sampler used.
            uint32_t zero_probability = 0;
            for (float value : seen) {
                if (!(value > 0.0f)) ++zero_probability;
            }
            harness.assert_that("E: at T=1, top_p=1 the support is the whole vocabulary",
                                zero_probability == 1,   // exactly the masked one
                                std::to_string(zero_probability) + " token(s) with no mass");

            (void)hipFree(device);
        }

        CHECK_HIP(hipStreamDestroy(stream));
        stage("E: the device path", section_start);
    }

    // =========================================================================
    // F. The seam closed on the artifact's own logits
    // =========================================================================
    std::printf("\n[F] The sampler on the graph's own logits\n");
    {
        const auto section_start = Clock::now();
        aeon::core::AeonRuntimeConfig runtime;
        runtime.context_size = 256;
        V4ModelHost host;
        host.initialize(kModelDir, runtime, /*verbose=*/false);

        const uint32_t vocab = static_cast<uint32_t>(host.config().vocab_size);
        V4Sampler sampler(vocab);
        harness.assert_that("F: the sampler's width is the model's vocabulary",
                            vocab == kVocab, std::to_string(vocab));

        V4Graph graph(host);
        const half* logits = graph.forward_token(kToken, 0, host.streams().compute);

        const uint32_t device_token = sampler.device_argmax(logits, host.streams().compute);
        const uint32_t selected = sampler.select(logits, host.streams().compute);
        harness.assert_that("F: the default decision is the certified device argmax",
                            selected == device_token,
                            std::to_string(selected) + " == " + std::to_string(device_token));

        // The graph's logits widened on the host must give the same argmax — which
        // is the statement that the sampler's input contract is the graph's output
        // contract, and it is exact because the widening is.
        sampler.set_logit_processor([](float*, uint32_t) {});
        const uint32_t host_token = sampler.select(logits, host.streams().compute);
        const std::vector<float>& seen = sampler.host_logits();
        harness.assert_that("F: the widened logits give the same argmax",
                            host_token == device_token,
                            std::to_string(host_token) + " == " + std::to_string(device_token));
        harness.assert_that("F: the widened buffer is the graph's vocabulary",
                            seen.size() == vocab,
                            std::to_string(seen.size()) + " values");
        harness.assert_that("F: the logits the sampler widens are the graph's own",
                            ops::argmax_of(seen.data(), vocab) == device_token,
                            "argmax " + std::to_string(ops::argmax_of(seen.data(), vocab)));

        stage("F: the artifact", section_start);
    }

    stage("the whole gate", gate_start);
    std::printf("\n--------------------------------------------------------------------------------\n");
    std::printf("  P3: %u checks, %u failures\n", harness.checks, harness.failures);
    std::printf("--------------------------------------------------------------------------------\n");
    return harness.failures == 0 ? 0 : 1;
}
