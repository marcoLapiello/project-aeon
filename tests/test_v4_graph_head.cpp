// -----------------------------------------------------------------------------
// P1 gate — the graph's head end: a token id in, logits out.
//
// The composition plan's P1 builds the three ops at the tail of the forward pass
// plus the minimal host that owns their inputs. This gate is what makes it
// evidence rather than construction.
//
// WHAT THIS GATE OWNS, and why each is genuinely uncovered:
//
//  1. THE DEVICE-SIDE EMBEDDING EXPANSION (plan Step 1). The state entering the
//     layer stack is `hc_mult x 4096` per token — the embedding row broadcast
//     across the four streams. The Step-3 gate already builds that residual, but
//     it builds it **on the host** in fp64 to feed the kernel; nothing has ever
//     exercised the device path that has to produce it (`hc_mult` H2D copies plus
//     `v4_half_to_float_kernel`), and nothing has ever asserted the four streams
//     are identical to each other and to the checkpoint row.
//
//  2. THE COMPOSITION `hc_head -> final RMSNorm -> LM head`. Nothing in the tree
//     has run the final norm or the LM head at all: `head.weight` appears in the
//     model contract's tensor list and in the pre-rewrite `step()`, and in no gate
//     and no rewrite code path. `norm.weight` likewise.
//
// WHAT THIS GATE DOES NOT DO, stated so a green line here is not read as more
// than it is:
//
//  * It does **not** re-certify `hc_head_wave32_kernel`. `tests/test_v4_hc_head_oracle.cpp`
//    owns that, at every discriminating property (weightless norm, flattened RMS,
//    scalar scale, saturated and eps-floor regimes, 6 of 6 mutations killed), and
//    re-deriving it here would be the duplication this rewrite exists to remove.
//    Its coverage is *consumed*: this gate compares the device's `hc_head_out`
//    against `reference::hc_head_reduce`, which is the same instrument that gate
//    uses, pointed at a residual the *device* produced.
//  * It does **not** certify a forward pass. There are no layers in P1; the
//    residual the head consumes is an embedding, not a 43-layer trajectory. The
//    graph driver and its oracle are P2's.
//
// THE ORACLE IS COMPOSED, NOT REUSED WHOLESALE. The three ops are certified
// separately *and* as a chain — the same "isolate, then chain" structure the
// grouped-output gate used, because a fault in one op and a fault in the wiring
// between them are different defects and a chain-only comparison cannot say which
// one moved:
//
//   B2  `hc_head_out`   vs  hc_head_reduce(embed_row)            — isolates hc_head,
//                                                                 fed the oracle's
//                                                                 own residual
//   B3  `head_norm`     vs  rmsnorm(the device's hc_head_out)    — isolates the norm,
//                                                                 fed the kernel's
//                                                                 own output
//   B4  `logits`        vs  matvec(head, the device's head_norm) — isolates the head,
//                                                                 fed the kernel's
//                                                                 own input
//   B5  `logits`        vs  the full chain from the embed row      — the composition
//
// The instrument for the logits. They are the one output here whose elements span
// decades, so a peak-relative bound alone would hide every small logit. The strong
// check is therefore elementwise against the **fp16-rounded oracle**: a correct
// stage should produce, for each of the 129280 logits, the same fp16 value the
// true number rounds to, except where the fp32 the kernel accumulates with and the
// fp64 the oracle accumulates with disagree across a rounding boundary. That
// expected band is `2^-24 / 2^-11 ~ 0.2%`, and the gate asserts a 2% ceiling (10x
// the floor) while *reporting* the measured fraction — so a wrong norm or a
// transposed head, which move most of the vector, cannot pass, and neither can a
// defect that moves only the tail.
//
// Sections:
//   A. the fixture is the real artifact (names, dtypes, shapes, the head is not
//      the embedding, the tokens are real)
//   B. the device stage versus the composed oracle
//   C. non-vacuity — the composition is shown load-bearing before B is trusted
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/v4_graph.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "support/v4_layer_body_gate.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace {

using aeon::core::AeonRuntimeConfig;
using aeon::core::V4Graph;
using aeon::core::V4ModelHost;

constexpr uint32_t kHidden = aeon::kernel::DSV4_HIDDEN_SIZE; // 4096
constexpr uint32_t kHcMult = 4;
constexpr uint32_t kFlat = kHcMult * kHidden; // 16384
constexpr double kRmsEps = 1e-6;
constexpr double kHcEps = 1e-6;

// Real token ids from `profiling-prompts/first-prompt.jsonl`, the same four the
// Step-3 gate uses, so the embedding rows are ones the model is actually driven
// with rather than convenient small integers.
constexpr uint32_t kRealTokens[] = {65106, 295, 4654, 3999};
constexpr size_t kTokenCount = sizeof(kRealTokens) / sizeof(kRealTokens[0]);

struct Harness {
    uint32_t checks{0};
    uint32_t failures{0};

    bool assert_that(const char* label, bool ok, const std::string& detail) {
        std::printf("  %-58s %-34s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
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
};

Harness harness;
const aeon::core::AeonModelLoader* g_loader = nullptr;

std::string sci(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.3e", value);
    return buffer;
}

std::string percent(double fraction) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.3f%%", fraction * 100.0);
    return buffer;
}

// --- the artifact's own tensors, decoded to fp64 ------------------------------
//
// `embed.weight` is reached through the resources' pointer into the container;
// the other four are device-only, so they come from the container's host mapping.
// Both routes are the loader's, not a copy made for the test.

std::vector<double> load_f16(const std::string& name, size_t count) {
    const auto& tensor = g_loader->get_tensor(name);
    const auto* bits = reinterpret_cast<const uint16_t*>(tensor.data);
    return aeon::reference::half_bits_to_doubles(bits, count);
}

std::vector<double> load_f32(const std::string& name, size_t count) {
    const auto& tensor = g_loader->get_tensor(name);
    const auto* source = reinterpret_cast<const float*>(tensor.data);
    return std::vector<double>(source, source + count);
}

// The embedding row, expanded identically across the four streams — the fp64
// statement of what `embed_token` has to produce on the device.
std::vector<double> residual_from_embedding(uint32_t token) {
    const auto& tensor = g_loader->get_tensor("embed.weight");
    const auto* bits = reinterpret_cast<const uint16_t*>(tensor.data);
    const std::vector<double> embed =
        aeon::reference::half_bits_to_doubles(bits + static_cast<size_t>(token) * kHidden, kHidden);

    std::vector<double> flat(kFlat);
    for (uint32_t stream_index = 0; stream_index < kHcMult; ++stream_index) {
        std::copy(embed.begin(), embed.end(),
                  flat.begin() + static_cast<size_t>(stream_index) * kHidden);
    }
    return flat;
}

// --- reading a device buffer back as fp64 ------------------------------------

std::vector<__half> read_half_raw(hipStream_t stream, const half* device, size_t count) {
    std::vector<__half> host(count);
    CHECK_HIP(hipMemcpyAsync(host.data(), device, count * sizeof(half),
                             hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    return host;
}

// One fp16 ulp at the magnitude of `value`: fp16 carries 10 explicit mantissa
// bits, so the spacing is `2^(e-10)` for `2^e <= |value| < 2^(e+1)`. Used to decide
// whether two logits that disagree are a genuine tie rather than a rule error.
double half_ulp(double value) {
    if (value == 0.0) return std::ldexp(1.0, -24);
    return std::ldexp(1.0, std::ilogb(value) - 10);
}

// The top `count` values, descending. Values rather than indices, because the
// logits are fp16 and a 129280-wide fp16 vector has many exact ties: comparing
// which *index* won would be flaky for reasons unrelated to the stage under test.
std::vector<double> top_values(const std::vector<double>& values, size_t count) {
    std::vector<double> sorted = values;
    std::partial_sort(sorted.begin(), sorted.begin() + static_cast<ptrdiff_t>(count),
                      sorted.end(), std::greater<double>());
    sorted.resize(count);
    return sorted;
}

// --- the oracle's head parameters, loaded once --------------------------------

struct HeadParams {
    std::vector<double> fn;      // [hc_mult, hc_mult*hidden]
    std::vector<double> base;    // [hc_mult]
    double scale{0.0};           // scalar
    std::vector<double> norm;    // [hidden]  — the final RMSNorm's learned weight
    std::vector<double> head;    // [vocab, hidden]
    std::vector<double> embed;   // [vocab, hidden]
    uint32_t vocab{0};
};

HeadParams load_head_params() {
    HeadParams params;
    params.vocab = static_cast<uint32_t>(g_loader->get_tensor("head.weight").shape[0]);
    params.fn = load_f32("hc_head_fn", kHcMult * kFlat);
    params.base = load_f32("hc_head_base", kHcMult);
    params.scale = load_f32("hc_head_scale", 1)[0];
    params.norm = load_f16("norm.weight", kHidden);
    params.head = load_f16("head.weight", static_cast<size_t>(params.vocab) * kHidden);
    params.embed = load_f16("embed.weight", static_cast<size_t>(params.vocab) * kHidden);
    return params;
}

// hc_head with whatever substitutions a probe asks for. The oracle takes the
// weight itself rather than reading the global, so a probe cannot accidentally
// measure the artifact twice.
std::vector<double> oracle_hc_head(const std::vector<double>& residual,
                                   const HeadParams& params) {
    return aeon::reference::hc_head_reduce(residual, kHcMult, kHidden,
                                           params.fn.data(), params.base.data(),
                                           params.scale, kRmsEps, kHcEps)
        .out;
}

std::vector<double> oracle_logits(const std::vector<double>& head_norm,
                                  const HeadParams& params,
                                  const std::vector<double>& weight) {
    return aeon::reference::matvec(
        params.vocab, kHidden, head_norm,
        [&weight](size_t o, size_t i) { return weight[o * kHidden + i]; });
}

} // namespace

int main() {
    std::printf(
        "================================================================================\n");
    std::printf("  P1 — the graph's head end: embedding -> hc_head -> final norm -> LM head\n");
    std::printf(
        "================================================================================\n");

    aeon::core::select_compute_device(false);

    aeon::core::AeonModelLoader loader;
    loader.open_model("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon");
    g_loader = &loader;

    V4ModelHost host;
    AeonRuntimeConfig runtime;
    runtime.context_size = 256;
    host.initialize("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon", runtime);

    V4Graph graph(host);
    const hipStream_t stream = host.streams().compute;

    // ---------------------------------------------------------------- A: fixture
    std::printf("\n[A] The fixture is the real artifact\n");

    const auto& embed_t = loader.get_tensor("embed.weight");
    const auto& head_t = loader.get_tensor("head.weight");
    const auto& norm_t = loader.get_tensor("norm.weight");
    const HeadParams params = load_head_params();

    harness.assert_that("A: embed.weight is F16 [129280, 4096]",
                        embed_t.dtype == "F16" && embed_t.shape.size() == 2 &&
                            embed_t.shape[0] == 129280 && embed_t.shape[1] == 4096,
                        std::to_string(embed_t.shape[0]) + "x" + std::to_string(embed_t.shape[1]));
    harness.assert_that("A: head.weight is F16 [129280, 4096]",
                        head_t.dtype == "F16" && head_t.shape.size() == 2 &&
                            head_t.shape[0] == 129280 && head_t.shape[1] == 4096,
                        std::to_string(head_t.shape[0]) + "x" + std::to_string(head_t.shape[1]));
    harness.assert_that("A: norm.weight is F16 [4096]",
                        norm_t.dtype == "F16" && norm_t.shape.size() == 1 &&
                            norm_t.shape[0] == 4096,
                        std::to_string(norm_t.shape[0]));

    // The head is a separate matrix, not the embedding: `tie_word_embeddings = False`.
    // If they were the same tensor this gate would pass while certifying the wrong
    // graph, so the distinction is asserted from the bytes rather than from the
    // config.
    {
        size_t differing = 0;
        const size_t sample = 4096;
        for (size_t i = 0; i < sample; ++i) {
            if (params.embed[i] != params.head[i]) ++differing;
        }
        // Not "all of them differ": two independent fp16 tensors share values by
        // coincidence, and demanding every sampled element differ would fail on a
        // correct artifact. The claim being asserted is that the head is not
        // literally the embedding, so a large majority is the right instrument.
        harness.assert_that("A: head.weight is not embed.weight (untied)",
                            differing > sample / 2,
                            std::to_string(differing) + " of " + std::to_string(sample) +
                                " sampled elements differ (" +
                                percent(static_cast<double>(differing) / sample) + ")");
    }
    harness.assert_that("A: hc_head_fn is F32 [4, 16384] and scale is a scalar",
                        params.fn.size() == kHcMult * kFlat && params.base.size() == kHcMult,
                        "fn=" + std::to_string(params.fn.size()) +
                            " base=" + std::to_string(params.base.size()));
    {
        bool in_range = true;
        for (const uint32_t token : kRealTokens) in_range = in_range && token < params.vocab;
        harness.assert_that("A: the probe token ids are inside the vocabulary", in_range,
                            std::to_string(params.vocab) + " tokens");
    }
    {
        // A token id outside the table is refused, not wrapped into it. The same
        // rule as trap 40: an out-of-range index must not produce finite numbers.
        bool refused = false;
        try {
            graph.embed_token(params.vocab, stream);
        } catch (const std::out_of_range&) {
            refused = true;
        } catch (const std::exception&) {
            refused = false;
        }
        harness.assert_that("A: an out-of-vocabulary token id is refused", refused,
                            "embed_token(vocab) throws");
    }

    // ---------------------------------------------- B: the device stage vs oracle
    std::printf("\n[B] The head stage versus the composed fp64 oracle\n");

    // Per token: the device buffers, and the three oracle views of them.
    struct Step {
        uint32_t token{0};
        std::vector<double> residual;      // [16384] device readback, fp64
        std::vector<double> hc_head_out;   // [4096]  device readback
        std::vector<double> head_norm;     // [4096]  device readback
        std::vector<double> logits;        // [vocab] device readback
        std::vector<__half> logits_half;   // [vocab] device readback, bit-exact
        std::vector<double> want_hc_head;  // oracle hc_head, from the oracle residual
        std::vector<double> want_norm;     // oracle norm, from the device's hc_head_out
        std::vector<double> want_logits_isolated; // oracle head, from the device's head_norm
        std::vector<double> want_logits_chain;    // oracle chain, from the embed row
    };
    std::vector<Step> steps;

    for (size_t index = 0; index < kTokenCount; ++index) {
        const uint32_t token = kRealTokens[index];
        graph.forward_head(token, stream);

        Step step;
        step.token = token;
        step.residual = aeon::testgate::read_float(stream, graph.residual(), kFlat);
        step.hc_head_out = aeon::testgate::upload_and_read(stream, graph.hc_head_output(), kHidden);
        step.head_norm = aeon::testgate::upload_and_read(stream, graph.head_norm(), kHidden);
        step.logits = aeon::testgate::upload_and_read(stream, graph.logits(), params.vocab);
        step.logits_half = read_half_raw(stream, graph.logits(), params.vocab);

        // The oracle's residual is built from the artifact on the host, not from
        // the device's readback: a broken broadcast must fail B1 *and* B2, and an
        // oracle fed the device's own residual could not see it at all (trap 37).
        const std::vector<double> oracle_residual = residual_from_embedding(token);
        step.want_hc_head = oracle_hc_head(oracle_residual, params);
        step.want_norm = aeon::reference::rmsnorm(step.hc_head_out, params.norm, kRmsEps);
        step.want_logits_isolated = oracle_logits(step.head_norm, params, params.head);
        step.want_logits_chain =
            oracle_logits(step.want_norm, params, params.head);

        steps.push_back(std::move(step));
    }

    // B1 — the four HC streams are a broadcast. Asserted against the checkpoint row
    // *and* against each other, because an implementation could produce four
    // identical-but-wrong streams and pass the second test alone.
    {
        uint32_t worst_stream_differences = 0;
        uint32_t worst_row_differences = 0;
        for (const auto& step : steps) {
            const auto& tensor = loader.get_tensor("embed.weight");
            const auto* bits = reinterpret_cast<const uint16_t*>(tensor.data);
            const std::vector<double> row = aeon::reference::half_bits_to_doubles(
                bits + static_cast<size_t>(step.token) * kHidden, kHidden);

            for (uint32_t stream_index = 1; stream_index < kHcMult; ++stream_index) {
                for (uint32_t i = 0; i < kHidden; ++i) {
                    const double base = step.residual[i];
                    const double other = step.residual[stream_index * kHidden + i];
                    if (base != other) ++worst_stream_differences;
                }
            }
            for (uint32_t i = 0; i < kHidden; ++i) {
                if (step.residual[i] != row[i]) ++worst_row_differences;
            }
        }
        harness.assert_that("B1: the four HC streams are byte-identical",
                            worst_stream_differences == 0,
                            std::to_string(worst_stream_differences) + " differing values");
        harness.assert_that("B1: and they are the checkpoint's own embedding row",
                            worst_row_differences == 0,
                            std::to_string(worst_row_differences) + " differing values");
    }

    // B2/B3/B4/B5 — the three checkpoints and the chain.
    for (const auto& step : steps) {
        const std::string tag = " token " + std::to_string(step.token);
        harness.report(("B2: hc_head_out vs oracle" + tag).c_str(),
                       step.want_hc_head, step.hc_head_out, 3e-3);
        harness.report(("B3: head_norm vs oracle (fed the kernel's own output)" + tag).c_str(),
                       step.want_norm, step.head_norm, 3e-3);
        harness.report(("B4: logits vs oracle (isolates the LM head)" + tag).c_str(),
                       step.want_logits_isolated, step.logits, 3e-3);
        harness.report(("B5: logits vs the full oracle chain" + tag).c_str(),
                       step.want_logits_chain, step.logits, 3e-3);
    }

    // B6 — the *mechanism* of the residual error, which is the one thing the
    // peak-relative bound cannot explain.
    //
    // The logits span two decades and the LM head is a 4096-term dot whose inputs
    // are fp16 and whose accumulate is fp32. An accumulation error is proportional
    // to the magnitude of the **terms**, not to the magnitude of the **result**, so
    // for a row that cancels the absolute error stays large while the value is
    // small — and the fp16 store's ulp shrinks with the value. A per-element
    // comparison against each logit's own fp16 quantum is therefore the wrong
    // instrument here: it would report a large "disagreement" on a correct stage,
    // which is what this gate's first run did (42.9% of 517120 elements).
    //
    // The right instruments are two, and together they decide whether the residual
    // is precision or a defect:
    //   (a) the device logit must sit within a couple of fp16 ulps of an
    //       independent **fp32** accumulation of the same row — a different
    //       summation order, so bit identity is not expected, but a wiring error is
    //       orders of magnitude out;
    //   (b) the fraction of rows whose fp16 logit differs from the fp64 oracle's
    //       rounding must be what the fp32 accumulation predicts. If the two agree,
    //       the discrepancy is the instrument's, not the kernel's.
    {
        constexpr size_t kSampleRows = 512;
        const auto& step = steps.front();

        // The natural scale of an accumulation is the sum of the magnitudes of its
        // terms, not the magnitude of its result: for a row that cancels, the result
        // can be far smaller than the terms that produced it, and the fp16 quantum
        // shrinks with the result while the accumulation error does not. `gamma` is
        // the standard bound for `K` fp32 roundings, `K * u / (1 - K * u)`.
        constexpr size_t kTerms = kHidden;
        const double u = std::ldexp(1.0, -24);
        const double gamma = static_cast<double>(kTerms) * u / (1.0 - static_cast<double>(kTerms) * u);

        double worst_budget = 0.0;
        double worst_ulp = 0.0;
        size_t measured_differ = 0;
        size_t predicted_differ = 0;

        for (size_t row = 0; row < kSampleRows; ++row) {
            // An independent fp32 accumulation in ascending k, with the term-magnitude
            // sum taken alongside it. The operands are exact: both were fp16, and
            // fp16 widens into fp32 without loss.
            float accumulator = 0.0f;
            double term_magnitude = 0.0;
            const double* weight_row = params.head.data() + row * kHidden;
            for (uint32_t k = 0; k < kTerms; ++k) {
                const float x = static_cast<float>(step.head_norm[k]);
                const float w = static_cast<float>(weight_row[k]);
                accumulator = std::fma(x, w, accumulator);
                term_magnitude += std::fabs(static_cast<double>(x) * static_cast<double>(w));
            }

            const double got = step.logits[row];
            const double delta = std::fabs(got - static_cast<double>(accumulator));
            worst_budget = std::max(worst_budget, delta / (gamma * term_magnitude));
            worst_ulp = std::max(worst_ulp, delta / half_ulp(got));

            // Does the fp32 accumulation round to a different fp16 than the fp64
            // oracle does, and does the device follow the same way?
            const __half want_fp32 = __float2half(accumulator);
            const __half want_fp64 = __float2half(static_cast<float>(step.want_logits_chain[row]));
            const __half got_half = step.logits_half[row];
            if (__half_as_ushort(want_fp32) != __half_as_ushort(want_fp64)) ++predicted_differ;
            if (__half_as_ushort(got_half) != __half_as_ushort(want_fp64)) ++measured_differ;
        }

        harness.assert_that(
            "B6: each logit matches an fp32 dot within its accumulation budget",
            worst_budget <= 3.0,
            "worst " + aeon::testgate::num(worst_budget, 3) + "x the bound, " +
                aeon::testgate::num(worst_ulp, 1) + " fp16 ulp over " +
                std::to_string(kSampleRows) + " rows");

        const double measured = static_cast<double>(measured_differ) / kSampleRows;
        const double predicted = static_cast<double>(predicted_differ) / kSampleRows;
        harness.assert_that(
            "B6: the fp16 disagreement with the fp64 oracle is the accumulation",
            measured <= predicted * 1.5 + 0.02,
            "measured " + percent(measured) + " vs predicted by fp32 accumulation " +
                percent(predicted));
    }

    // B7 — the argmax and the top-8 *values*, which is what actually feeds the
    // sampler. The indices are deliberately not compared: the logits are fp16, so a
    // 129280-wide vector has many exact ties and an index comparison would be flaky
    // for reasons that have nothing to do with the stage. Values are compared
    // instead — a wrong head moves them wholesale — and a differing argmax is
    // accepted only when the two candidates agree to within one fp16 ulp, the same
    // allowance item 17's gate arrived at for a discrete selection (trap 37).
    {
        size_t argmax_mismatches = 0;
        size_t tied_mismatches = 0;
        double worst_top8 = 0.0;
        double worst_top8_ulps = 0.0;
        std::string worst;

        for (const auto& step : steps) {
            const size_t got_argmax = static_cast<size_t>(
                std::max_element(step.logits.begin(), step.logits.end()) - step.logits.begin());
            const size_t want_argmax = static_cast<size_t>(
                std::max_element(step.want_logits_chain.begin(), step.want_logits_chain.end()) -
                step.want_logits_chain.begin());

            if (got_argmax != want_argmax) {
                ++argmax_mismatches;
                const double got_value = step.logits[got_argmax];
                const double want_value = step.want_logits_chain[want_argmax];
                const double allowance =
                    std::max(half_ulp(got_value), half_ulp(want_value));
                if (std::fabs(got_value - want_value) <= allowance) {
                    ++tied_mismatches;
                } else {
                    worst = "token " + std::to_string(step.token) + ": " +
                            std::to_string(got_argmax) + " vs " +
                            std::to_string(want_argmax) + " by " +
                            sci(std::fabs(got_value - want_value));
                }
            }

            const std::vector<double> got_top = top_values(step.logits, 8);
            const std::vector<double> want_top = top_values(step.want_logits_chain, 8);
            for (size_t i = 0; i < got_top.size(); ++i) {
                const double delta = std::fabs(got_top[i] - want_top[i]);
                worst_top8 = std::max(worst_top8, delta);
                worst_top8_ulps =
                    std::max(worst_top8_ulps, delta / half_ulp(want_top[i]));
            }
        }

        harness.assert_that(
            "B7: the argmax matches the oracle, or is a one-ulp tie",
            argmax_mismatches == tied_mismatches,
            argmax_mismatches == 0
                ? "all " + std::to_string(kTokenCount) + " tokens agree"
                : (argmax_mismatches == tied_mismatches
                       ? std::to_string(argmax_mismatches) + " mismatches, all within one ulp"
                       : worst));
        harness.assert_that("B7: the top-8 values agree within two fp16 ulps",
                            worst_top8_ulps <= 2.0,
                            "worst " + sci(worst_top8) + " = " +
                                aeon::testgate::num(worst_top8_ulps, 3) + " ulp");
    }

    // ---------------------------------------------------- C: non-vacuity
    std::printf("\n[C] The composition is load-bearing\n");

    // C1 — the final RMSNorm's weight is real. If `norm.weight` were uniform the
    // norm would be a pure rescale of a vector that the LM head's row-wise dot
    // product is *also* invariant to up to the logit's own scale, and B3/B4 would
    // be weaker than they look. Measured, not assumed: the weight's spread is
    // reported alongside the delta a unit weight would cause.
    {
        const auto& step = steps.front();
        std::vector<double> unit(kHidden, 1.0);
        const std::vector<double> want_unit =
            oracle_logits(aeon::reference::rmsnorm(step.hc_head_out, unit, kRmsEps), params,
                          params.head);
        double worst = 0.0;
        for (size_t i = 0; i < want_unit.size(); ++i) {
            worst = std::max(worst, std::fabs(want_unit[i] - step.want_logits_chain[i]));
        }
        const double peak = aeon::reference::peak_abs(step.want_logits_chain);
        const double minimum = *std::min_element(params.norm.begin(), params.norm.end());
        const double maximum = *std::max_element(params.norm.begin(), params.norm.end());
        harness.assert_that("C1: the final norm's learned weight moves the logits",
                            worst > 3e-3 * peak,
                            "unit-weight delta " + sci(worst) + " = " +
                                sci(worst / peak) + "*peak, weight spans [" +
                                sci(minimum) + ", " + sci(maximum) + "]");
    }

    // C2 — the head is not the embedding. This is the `tie_word_embeddings` error
    // in its most likely form: pointing the LM head at `embed.weight`. Measured
    // through the oracle so the number is the *effect*, not a kernel artifact.
    {
        const auto& step = steps.front();
        const std::vector<double> want_tied =
            oracle_logits(step.want_norm, params, params.embed);
        double worst = 0.0;
        for (size_t i = 0; i < want_tied.size(); ++i) {
            worst = std::max(worst, std::fabs(want_tied[i] - step.logits[i]));
        }
        const double peak = aeon::reference::peak_abs(step.logits);
        harness.assert_that("C2: reading embed.weight as the head is detectable",
                            worst > 3e-3 * peak,
                            "tied-reading delta " + sci(worst) + " = " + sci(worst / peak) +
                                "*peak");
    }

    // C3 — the stage is not a constant. Distinct tokens must give distinct
    // residuals, distinct head outputs and distinct logits; if the embedding row
    // were not offset by the token id, every token would agree and every check
    // above would pass vacuously.
    {
        size_t distinct_residuals = 0;
        size_t distinct_logits = 0;
        size_t distinct_argmax = 0;
        std::vector<size_t> argmaxes;
        for (size_t i = 0; i < steps.size(); ++i) {
            argmaxes.push_back(static_cast<size_t>(
                std::max_element(steps[i].logits.begin(), steps[i].logits.end()) -
                steps[i].logits.begin()));
            for (size_t j = i + 1; j < steps.size(); ++j) {
                if (steps[i].residual != steps[j].residual) ++distinct_residuals;
                if (steps[i].logits != steps[j].logits) ++distinct_logits;
            }
        }
        std::sort(argmaxes.begin(), argmaxes.end());
        distinct_argmax = static_cast<size_t>(
            std::unique(argmaxes.begin(), argmaxes.end()) - argmaxes.begin());

        const size_t pairs = steps.size() * (steps.size() - 1) / 2;
        harness.assert_that("C3: distinct tokens give distinct residuals and logits",
                            distinct_residuals == pairs && distinct_logits == pairs,
                            std::to_string(distinct_logits) + " of " + std::to_string(pairs) +
                                " token pairs differ in the logits");
        harness.assert_that("C3: and they do not all predict the same token",
                            distinct_argmax > 1,
                            std::to_string(distinct_argmax) + " distinct argmaxes over " +
                                std::to_string(steps.size()) + " tokens");
    }

    // C4 — the norm is applied at all. A skipped final norm leaves the LM head
    // reading `hc_head_out`; that is the wiring error B3/B4 are shaped to catch,
    // and the measurement makes the size of the signal explicit.
    {
        const auto& step = steps.front();
        const std::vector<double> want_unnormed =
            oracle_logits(step.hc_head_out, params, params.head);
        double worst = 0.0;
        for (size_t i = 0; i < want_unnormed.size(); ++i) {
            worst = std::max(worst, std::fabs(want_unnormed[i] - step.logits[i]));
        }
        const double peak = aeon::reference::peak_abs(step.logits);
        harness.assert_that("C4: omitting the final norm is detectable",
                            worst > 3e-3 * peak,
                            "un-normed delta " + sci(worst) + " = " + sci(worst / peak) +
                                "*peak");
    }

    host.free();

    std::printf(
        "--------------------------------------------------------------------------------\n");
    if (harness.failures == 0) {
        std::printf("[P1 — graph head stage] PASS — %u checks, 0 failed\n", harness.checks);
        return 0;
    }
    std::printf("[P1 — graph head stage] FAIL — %u checks, %u failed\n", harness.checks,
                harness.failures);
    return 1;
}
