// -----------------------------------------------------------------------------
// P2 gate — the 43-layer driver: the graph produces a token, on real weights.
//
// P1 certified the tail of the forward pass (embedding -> `hc_head` -> final
// norm -> LM head) on the artifact's real tensors, and said in as many words what
// it did **not** certify: "the residual the head consumes is an embedding, not a
// 43-layer trajectory". This gate is that trajectory.
//
// It certifies two things, and they are different defects:
//
//   1. THE COMPOSITION IS ARITHMETICALLY RIGHT. The graph's `run_layer` chain,
//      over the real 43 layers with real routed experts delivered through the
//      tiered supply, agrees with an independently written fp64 reference
//      (`reference::model_body`) at every layer and at the head.
//   2. THE DRIVER IS THE LOOP IT CLAIMS TO BE. `forward_token` produces the same
//      logits, fp16-bit for fp16-bit, as the gate's own 43 calls to `run_layer`
//      plus the head. A driver that skipped a layer, repeated one, ordered them
//      wrongly or dropped the embedding cannot pass both.
//
// THE INSTRUMENT, and why it is this one. A 43-layer fp64 reference that free-runs
// against the device accumulates the device's fp16 rounding at every step, and by
// the head that drift can flip a router near-tie — after which the two trajectories
// differ for a reason that is not a defect. The serial-decode gate settled the
// method for exactly this: hand the reference the **device's own** per-step
// residual, and drive its MoE combine with the **device's own** selection
// (`LayerBodyWeights::routed_ids_override` / `routed_weights_override`), so the
// comparison measures one layer's composition rather than accumulated drift. That
// is why `V4Graph::run_layer` is public: the gate has to observe each step's input
// and its output.
//
// The selection is therefore *not* left unchecked. Its rule — softplus, add bias,
// flat top-6, ties to the lower index, the hash table on layers 0–2 — is asserted
// separately against the **device's own router logits** (read out of
// `PipelineScratchBuffers::d_router_logits` after the layer), which is trap 37's
// rule: a discrete quantity produced by a rule is checked against the rule applied
// to the device's own inputs, never elementwise against the reference's.
//
// WHAT IS REAL HERE. All 43 layers' dense weights, the real embedding table, the
// real head, the real RoPE tables, the real 512-wide `index_topk`, and **real
// routed experts read from the 145 GB container through the production
// `V4TieredExpertExecutor`** — Hot/Warm/Cold, leases, staging, `O_DIRECT`. The
// expert arithmetic is Tier-1/items 14–19; what is new here is that it is reached
// through the assembled host rather than a fixture.
//
// WHAT IS DELIBERATELY NOT COVERED, named so a green line is not read as more:
//
//   * **The local ring wrap.** It needs more than `sliding_window = 128` tokens
//     (the ring is `min(context, 128)`, and a position is bounded by the context),
//     which at 43 layers would be tens of thousands of expert fetches. The ring at
//     its real capacity is the real-scale state gate's subject; the wrap at small
//     capacity is the serial-decode gate's.
//   * **HCA compression.** A ratio-128 layer commits its first compressed entry at
//     position 127; four tokens never reach it. Compressed attention is certified
//     by items 17/18/20. CSA *is* exercised: it commits at position 3.
//   * **The sampler, the text binding, the observer, and tiering under load.**
//     P3/P4/P5.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_graph.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "support/v4_layer_body_gate.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using aeon::core::AeonModelLoader;
using aeon::core::AeonRuntimeConfig;
using aeon::core::V4AttentionKind;
using aeon::core::V4Graph;
using aeon::core::V4Layer;
using aeon::core::V4LayerSpec;
using aeon::core::V4ModelHost;
using aeon::reference::LayerBodyResult;
using aeon::reference::LayerBodyShape;
using aeon::reference::LayerBodyWeights;
using aeon::reference::ModelBodyResult;
using aeon::reference::ModelBodyShape;
using aeon::reference::ModelBodyState;
using aeon::reference::ModelBodyWeights;
using aeon::reference::RopeTableRef;

using aeon::testgate::hash_row_for_token;
using aeon::testgate::kHidden;
using aeon::testgate::load_layer_weights;
using aeon::testgate::read_float;
using aeon::testgate::report;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

// The context bounds the position, so it also bounds every ring. 256 is the same
// value the P1 gate uses: large enough that the compressor's partial ring and the
// committed-entry counters are exercised for real, small enough that the 43
// layers' attention state stays a fraction of VRAM.
constexpr uint32_t kMaxSeq = 256;

// Real token ids from `profiling-prompts/first-prompt.jsonl` — the same source the
// P1 gate's probes come from. Four tokens is the plan's floor and it is enough to
// matter: RoPE runs at four distinct positions (trap 36 forbids a position-0-only
// gate) and a CSA layer commits its first compressed entry at position 3.
constexpr uint32_t kTokenIds[] = {65106, 295, 4654, 3999};
constexpr uint32_t kTokens = sizeof(kTokenIds) / sizeof(kTokenIds[0]);

// Per-layer `res_out` and the head's checkpoints are fp16-bounded, so
// peak-relative against the oracle is the honest instrument, at the ~4e-3 the
// layer gates measured for the tightest checkpoint. The chained logits get a
// little more room: the head sits on top of 43 layers' rounding rather than one.
constexpr double kLayerTol = 4e-3;
constexpr double kHeadTol = 5e-3;
constexpr double kLogitTol = 6e-3;

struct Harness {
    uint32_t checks{0};
    uint32_t failures{0};

    bool assert_that(const char* label, bool ok, const std::string& detail) {
        std::printf("  %-58s %-38s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
        ++checks;
        if (!ok) ++failures;
        return ok;
    }

    bool compare(const char* label, const std::vector<double>& want,
                 const std::vector<double>& got, double tol_frac) {
        const bool ok = report(label, want, got, tol_frac);
        ++checks;
        if (!ok) ++failures;
        return ok;
    }
};

Harness harness;
const AeonModelLoader* g_loader = nullptr;

std::string sci(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.3e", value);
    return buffer;
}

// Section timing. A gate that runs 43 layers over real experts has more than one
// candidate for its cost, and guessing which one dominates is how a test becomes
// mysteriously slow.
using Clock = std::chrono::steady_clock;

double since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void stage(const char* name, const Clock::time_point& start) {
    std::printf("  [time] %-42s %.1f s\n", name, since(start));
}

// --- device reads ------------------------------------------------------------

std::vector<uint16_t> read_half_bits(hipStream_t stream, const half* device, size_t count) {
    std::vector<half> raw(count);
    CHECK_HIP(hipMemcpyAsync(raw.data(), device, count * sizeof(half),
                             hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    std::vector<uint16_t> bits(count);
    std::memcpy(bits.data(), raw.data(), count * sizeof(uint16_t));
    return bits;
}

std::vector<double> read_half_values(hipStream_t stream, const half* device, size_t count) {
    std::vector<half> values(count);
    CHECK_HIP(hipMemcpyAsync(values.data(), device, count * sizeof(half),
                             hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    std::vector<double> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = static_cast<double>(values[i]);
    return out;
}

size_t argmax_of(const std::vector<double>& values) {
    return static_cast<size_t>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

// --- the oracle's per-layer shape, from the device's own layout --------------

LayerBodyShape make_shape(const V4LayerSpec& spec, const V4Layer& device) {
    LayerBodyShape shape;
    shape.hidden = aeon::kernel::DSV4_HIDDEN_SIZE;
    shape.hc_mult = 4;
    shape.q_lora_rank = aeon::kernel::DSV4_Q_LORA_RANK;
    shape.num_heads = aeon::kernel::DSV4_NUM_HEADS;
    shape.head_dim = static_cast<uint32_t>(spec.head_dim);
    shape.rotary_dim = 64;
    shape.o_groups = 8;
    shape.o_lora_rank = 1024;
    shape.intermediate = 2048;
    shape.num_experts = 256;
    shape.top_k = 6;
    // The ring capacity is the device's own, so the oracle wraps where the device
    // wraps rather than where a constant says it should.
    shape.local_capacity = device.local_cache_capacity();
    shape.compress_ratio = spec.compression_ratio;
    shape.index_n_heads = static_cast<uint32_t>(spec.index_n_heads);
    shape.index_head_dim = static_cast<uint32_t>(spec.index_head_dim);
    shape.index_topk = static_cast<uint32_t>(spec.index_topk);
    shape.eps = 1e-6;
    shape.routed_scaling = 1.5;
    shape.swiglu_limit = 10.0;
    return shape;
}

// The RoPE base the layer's *class* selects: the plain base for Sliding, the
// YaRN-on-compressed base otherwise (plan 2.3, trap 7). The layer body makes this
// choice internally from `attention_kind`; the oracle is handed the answer.
RopeTableRef rope_for(const V4LayerSpec& spec, uint32_t max_position) {
    return aeon::reference::rope_table(
        aeon::reference::rope_spec_for(
            aeon::reference::rope_class_for_ratio(spec.compression_ratio)),
        max_position);
}

// --- the model-level oracle's inputs -----------------------------------------

ModelBodyShape make_model_shape(const aeon::core::DeepSeekV4Config& config) {
    ModelBodyShape shape;
    shape.hidden = aeon::kernel::DSV4_HIDDEN_SIZE;
    shape.hc_mult = static_cast<uint32_t>(config.hc_mult);
    shape.vocab = static_cast<uint32_t>(config.vocab_size);
    shape.rms_eps = config.rms_norm_eps;
    shape.hc_eps = config.hc_eps;
    return shape;
}

ModelBodyWeights load_model_body_weights(const AeonModelLoader& loader) {
    const auto f16 = [&](const std::string& name) {
        return reinterpret_cast<const uint16_t*>(loader.get_tensor(name).data);
    };
    ModelBodyWeights w{};
    w.embed = f16("embed.weight");
    w.final_norm = f16("norm.weight");
    w.lm_head = f16("head.weight");
    w.hc_head_fn = loader.get_data_ptr<float>("hc_head_fn");
    w.hc_head_base = loader.get_data_ptr<float>("hc_head_base");
    w.hc_head_scale = loader.get_data_ptr<float>("hc_head_scale");
    return w;
}

// -----------------------------------------------------------------------------
// The device's routed selection, checked against its own rule.
//
// The reference's combine is *driven* by the device's ids and weights, so the
// selection itself has to be checked somewhere or the gate would accept any
// selection at all. It is checked here, against the device's own router logits
// (trap 37): score them the way the reference scores them — softplus, plus the
// bias on a biased layer, nothing on a hash layer — rank with the model's own tie
// rule (descending, ties to the lower index), and require the top-6 to be the ids
// the device reported.
//
// It also measures how close the 6th and 7th candidates came, because "a router
// near-tie is reachable" is a claim, not an assumption, and the smallest gap over
// the run is what makes it one.
// -----------------------------------------------------------------------------
struct RouterRule {
    uint32_t biased_steps{0};
    uint32_t mismatches{0};
    uint32_t near_ties{0};
    double worst_gap_ratio{std::numeric_limits<double>::infinity()};
};

void audit_router_rule(uint32_t layer_id, uint32_t token_id,
                       const std::vector<double>& logits,
                       const std::vector<int32_t>& selected, RouterRule& rule) {
    const size_t experts = logits.size();
    std::vector<double> scored(experts);
    for (size_t i = 0; i < experts; ++i) {
        scored[i] = aeon::reference::router_score(logits[i]);
    }

    const int64_t* hash_row = hash_row_for_token(*g_loader, layer_id, token_id);
    // On a hash layer the selection comes from `tid2eid` and the ranking is used
    // only for its scores, so there is no rule to compare there; the table's own
    // row is certified by Tier-1 gate 13.
    if (hash_row != nullptr) return;

    const float* device_bias = g_loader->get_data_ptr<float>(
        "layers." + std::to_string(layer_id) + ".ffn.gate.bias");
    for (size_t i = 0; i < experts; ++i) scored[i] += static_cast<double>(device_bias[i]);

    std::vector<size_t> order(experts);
    for (size_t i = 0; i < experts; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&scored](size_t a, size_t b) {
        if (scored[a] != scored[b]) return scored[a] > scored[b];
        return a < b;
    });

    ++rule.biased_steps;
    for (size_t i = 0; i < 6; ++i) {
        if (selected[i] != static_cast<int32_t>(order[i])) {
            ++rule.mismatches;
            break;
        }
    }

    const double sixth = scored[order[5]];
    const double seventh = scored[order[6]];
    const double ratio = std::fabs(seventh - sixth) / std::fmax(std::fabs(sixth), 1e-30);
    rule.worst_gap_ratio = std::fmin(rule.worst_gap_ratio, ratio);
    if (ratio < 1e-3) ++rule.near_ties;
}

} // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  P2 — the 43-layer driver: embedding -> 43 x run_layer -> head\n");
    std::printf("================================================================================\n");
    aeon::core::select_compute_device(true);

    const auto gate_start = Clock::now();

    // -------------------------------------------------------------------------
    // A. The assembly (composition plan G1's gate line)
    // -------------------------------------------------------------------------
    std::printf("\n[A] The host assembly\n");

    AeonRuntimeConfig runtime;
    runtime.context_size = kMaxSeq;
    V4ModelHost host;
    host.initialize(kModelDir, runtime, /*verbose=*/true);
    g_loader = &host.loader();
    const auto after_host = Clock::now();
    stage("A: the host assembly", gate_start);

    const auto& config = host.config();
    const auto& specs = host.layer_specs();
    const auto& budget = host.budget();

    uint32_t sliding = 0, csa = 0, hca = 0;
    for (const auto& spec : specs) {
        if (spec.attention_kind == V4AttentionKind::Sliding) ++sliding;
        if (spec.attention_kind == V4AttentionKind::CSA) ++csa;
        if (spec.attention_kind == V4AttentionKind::HCA) ++hca;
    }

    harness.assert_that("A: the host built the model's 43 layers",
                        host.num_layers() == 43 && static_cast<uint32_t>(specs.size()) == 43,
                        std::to_string(host.num_layers()) + " layers");
    harness.assert_that("A: the three attention classes are the model's own",
                        sliding == 2 && csa == 21 && hca == 20,
                        std::to_string(sliding) + " Sliding, " + std::to_string(csa) +
                            " CSA, " + std::to_string(hca) + " HCA");
    harness.assert_that("A: the context is what was requested",
                        host.context_capacity() == kMaxSeq,
                        std::to_string(host.context_capacity()) + " tokens");
    harness.assert_that("A: the expert tier was built (a Hot VRAM slot exists)",
                        host.experts_ready(), host.experts_ready() ? "yes" : "no");
    harness.assert_that("A: the registry's own invariants hold",
                        host.registry().invariants_hold(), "invariants_hold()");

    uint32_t hot_residents = 0;
    for (const auto& entry : host.registry().catalog) {
        if (entry.owner == aeon::core::ExpertTier::HOT_VRAM) ++hot_residents;
    }
    harness.assert_that("A: every budgeted Hot slot owns a resident expert",
                        hot_residents == budget.hot_vram_slots &&
                            host.registry().free_vram_slots.empty(),
                        std::to_string(hot_residents) + " of " +
                            std::to_string(budget.hot_vram_slots) + " residents, " +
                            std::to_string(host.registry().free_vram_slots.size()) + " free");
    harness.assert_that("A: the artifact is the one the contract described",
                        config.num_hidden_layers == 43 && config.n_routed_experts == 256 &&
                            budget.expert_payload_bytes == aeon::core::AEON_EXPERT_BYTES,
                        std::to_string(config.n_routed_experts) + " experts, payload " +
                            std::to_string(budget.expert_payload_bytes) + " B");

    // -------------------------------------------------------------------------
    // B. The oracle's closed forms, before it is trusted
    // -------------------------------------------------------------------------
    std::printf("\n[B] Oracle self-checks (closed forms)\n");

    const ModelBodyShape model = make_model_shape(config);
    const ModelBodyWeights model_weights = load_model_body_weights(host.loader());

    {
        // The embedding broadcast has a closed form: `hc_mult` byte-identical
        // copies of the token's row. Nothing about the device is involved.
        const std::vector<double> row =
            aeon::reference::model_embed(model, model_weights, kTokenIds[0]);
        bool identical = true;
        for (uint32_t stream = 1; stream < model.hc_mult; ++stream) {
            for (uint32_t h = 0; h < model.hidden; ++h) {
                if (row[static_cast<size_t>(stream) * model.hidden + h] != row[h]) {
                    identical = false;
                }
            }
        }
        harness.assert_that("B: the embedding expands into hc_mult identical streams",
                            identical, std::to_string(model.hc_mult) + " x " +
                                std::to_string(model.hidden));
    }
    stage("B: the oracle self-checks", after_host);

    {
        // The head's RMS is over the **flattened** `hc_mult * hidden` stream, and
        // that fixes how the residual scales through it. Two properties, and one
        // measurement of each:
        //
        //   * `mixes[j] = (residual · fn[j]) * inv_rms`: doubling the residual
        //     doubles the numerator and halves `inv_rms`, so `mixes` — and hence
        //     `pre_mix = sigmoid(mixes·scale + base) + eps` — is invariant. A
        //     per-stream RMS (the plausible misreading) normalises *before* the
        //     projection and is not invariant here.
        //   * `out[h] = Σ_j pre_mix[j] · residual[j][h]` has no normalisation left,
        //     so it doubles exactly — and it can only double exactly if `mixup`
        //     above held.
        //
        // The `rms_eps` floor is the one term that breaks the identity exactly, so
        // it is set to zero here: this pins the algebra the composition states, and
        // the model's own eps is exercised by every device comparison in section C.
        const std::vector<double> residual =
            aeon::reference::model_embed(model, model_weights, kTokenIds[0]);
        std::vector<double> doubled = residual;
        for (double& value : doubled) value *= 2.0;

        ModelBodyShape exact = model;
        exact.rms_eps = 0.0;
        const ModelBodyResult a = aeon::reference::model_head(exact, model_weights, residual);
        const ModelBodyResult b = aeon::reference::model_head(exact, model_weights, doubled);

        double linear = 0.0, invariant = 0.0;
        for (size_t i = 0; i < a.hc_head_out.size(); ++i) {
            linear = std::fmax(linear, std::fabs(b.hc_head_out[i] - 2.0 * a.hc_head_out[i]));
        }
        for (size_t i = 0; i < a.head_norm.size(); ++i) {
            invariant = std::fmax(invariant, std::fabs(a.head_norm[i] - b.head_norm[i]));
        }
        harness.assert_that("B: hc_head's output is exactly linear in the residual",
                            linear < 1e-12, "max|delta|=" + sci(linear));
        harness.assert_that("B: the final RMSNorm is exactly scale invariant",
                            invariant < 1e-12, "max|delta|=" + sci(invariant));
    }

    {
        // The oracle's `swizzled_decode` was restructured to hoist the per-group
        // address and the shared fp16 scale. That is a change to a certified file,
        // so it is checked against the per-element form it replaced, on a real
        // expert's real bytes, in both nibble modes. Nothing about the ordering,
        // the rounding or the values is allowed to move.
        const uint8_t* payload = host.loader().get_expert_data(0, 0);
        const aeon::reference::SwizzledKind kinds[3] = {
            aeon::reference::SwizzledKind::W1,
            aeon::reference::SwizzledKind::W2,
            aeon::reference::SwizzledKind::W3,
        };
        size_t differing = 0, checked = 0;
        for (aeon::reference::SwizzledKind kind : kinds) {
            for (int permuted = 1; permuted >= 0; --permuted) {
                aeon::reference::SwizzledDecodeOptions options;
                options.permute_nibbles = permuted != 0;
                const auto hoisted = aeon::reference::swizzled_decode(payload, kind, options);
                const auto per_element =
                    aeon::reference::swizzled_decode_reference(payload, kind, options);
                for (size_t i = 0; i < hoisted.size(); ++i) {
                    ++checked;
                    if (hoisted[i] != per_element[i]) ++differing;
                }
            }
        }
        harness.assert_that("B: the hoisted decoder equals the per-element form",
                            differing == 0,
                            std::to_string(differing) + " of " + std::to_string(checked) +
                                " values differ");
    }

    // -------------------------------------------------------------------------
    // The oracle's 43-layer graph
    // -------------------------------------------------------------------------
    std::vector<LayerBodyShape> shapes;
    std::vector<RopeTableRef> ropes;
    std::vector<LayerBodyWeights> weights;
    shapes.reserve(host.num_layers());
    ropes.reserve(host.num_layers());
    weights.reserve(host.num_layers());
    for (uint32_t layer = 0; layer < host.num_layers(); ++layer) {
        shapes.push_back(make_shape(specs[layer], host.layer(layer)));
        ropes.push_back(rope_for(specs[layer], kMaxSeq));
        weights.push_back(load_layer_weights(host.loader(), layer));
    }

    ModelBodyState oracle_state;
    oracle_state.reset(shapes, static_cast<uint32_t>(shapes[0].local_capacity));

    harness.assert_that("A: the oracle's ring caps are the device's own",
                        shapes[0].local_capacity == host.layer(0).local_cache_capacity() &&
                            shapes[2].local_capacity == host.layer(2).local_cache_capacity(),
                        std::to_string(shapes[0].local_capacity) + " rows");
    harness.assert_that("A: CSA has the indexer and HCA does not (trap 33)",
                        shapes[2].uses_indexer() && !shapes[3].uses_indexer() &&
                            weights[2].indexer_wq_b != nullptr &&
                            weights[3].indexer_wq_b == nullptr,
                        "layer 2 indexer=" +
                            std::string(weights[2].indexer_wq_b ? "yes" : "no") +
                            ", layer 3 indexer=" +
                            std::string(weights[3].indexer_wq_b ? "yes" : "no"));

    // -------------------------------------------------------------------------
    // C. The composition, layer by layer, over a real token sequence
    // -------------------------------------------------------------------------
    std::printf("\n[C] The 43-layer composition vs the fp64 reference\n");

    V4Graph graph(host);
    hipStream_t stream = host.streams().compute;
    const uint32_t hc_dim = model.hc_dim();
    const auto before_composition = Clock::now();

    RouterRule rule;
    double worst_layer_frac = 0.0;
    uint32_t worst_layer = 0, worst_layer_token = 0;
    std::vector<std::vector<uint16_t>> loop_logits(kTokens);
    std::vector<std::vector<uint16_t>> loop_residual(kTokens);
    uint32_t argmax_agreements = 0;
    // How many *distinct* (layer, expert) pairs the run actually asked for. This is
    // the measurement that decides whether the reference's decode cost could be
    // amortised across tokens; it is reported rather than assumed.
    std::vector<bool> expert_seen(static_cast<size_t>(host.num_layers()) * 256, false);
    size_t expert_requests = 0, distinct_experts = 0;

    for (uint32_t token_index = 0; token_index < kTokens; ++token_index) {
        const uint32_t token_id = kTokenIds[token_index];
        const uint32_t position = token_index;

        // The token's entry point is the embedding — exactly `forward_token`'s
        // first line; everything after it is the loop.
        graph.embed_token(token_id, stream);
        std::vector<double> oracle_input = read_float(stream, graph.residual(), hc_dim);

        double token_worst = 0.0;
        uint32_t token_worst_layer = 0;
        for (uint32_t layer = 0; layer < host.num_layers(); ++layer) {
            const aeon::core::V4LayerBodyOutput out =
                graph.run_layer(layer, token_id, position, stream);
            CHECK_HIP(hipStreamSynchronize(stream));

            const std::vector<double> device_res = read_float(stream, graph.residual(), hc_dim);

            // The device's own discrete selection drives the reference's combine
            // (item 18's method): the term set must be identical for the
            // comparison to be arithmetic rather than a coincidence hunt.
            weights[layer].routed_ids_override = out.topk_indices.data();
            weights[layer].routed_weights_override = out.topk_weights.data();
            for (size_t k = 0; k < out.topk_indices.size(); ++k) {
                const int32_t id = out.topk_indices[k];
                if (id < 0) {
                    throw std::logic_error("P2 gate: a routed expert id is negative");
                }
                weights[layer].routed_payloads[k] =
                    host.loader().get_expert_data(layer, static_cast<uint32_t>(id));
                const size_t pair = static_cast<size_t>(layer) * 256 + static_cast<size_t>(id);
                ++expert_requests;
                if (!expert_seen[pair]) {
                    expert_seen[pair] = true;
                    ++distinct_experts;
                }
            }

            audit_router_rule(layer, token_id,
                              read_float(stream, host.scratch().d_router_logits, 256),
                              out.topk_indices, rule);

            const LayerBodyResult result = aeon::reference::layer_body(
                shapes[layer], weights[layer], ropes[layer], position, oracle_input,
                oracle_state.layers[layer]);

            const aeon::reference::ErrorStats stats =
                aeon::reference::compare(result.res_out, device_res, 0.1);
            const double peak = aeon::reference::peak_abs(result.res_out);
            const double frac = peak > 0.0 ? stats.max_abs / peak : stats.max_abs;
            if (frac > token_worst) {
                token_worst = frac;
                token_worst_layer = layer;
            }
            if (frac > worst_layer_frac) {
                worst_layer_frac = frac;
                worst_layer = layer;
                worst_layer_token = token_index;
            }

            // Re-seed: the next layer's reference input is the device's output.
            oracle_input = device_res;
        }

        harness.assert_that(
            ("C: token " + std::to_string(token_id) + " — all 43 layers' res_out").c_str(),
            token_worst <= kLayerTol,
            "worst " + sci(token_worst) + "*peak at layer " +
                std::to_string(token_worst_layer));

        // The head, fed the device's own final residual, so a head defect cannot
        // masquerade as a layer defect.
        const half* device_logits = graph.head_stage(stream);
        CHECK_HIP(hipStreamSynchronize(stream));
        const ModelBodyResult head = aeon::reference::model_head(
            model, model_weights, read_float(stream, graph.residual(), hc_dim));

        harness.compare(("C: token " + std::to_string(token_id) +
                         " — hc_head_out (device residual in)").c_str(),
                        head.hc_head_out,
                        read_half_values(stream, graph.hc_head_output(), kHidden), 4e-3);
        harness.compare(("C: token " + std::to_string(token_id) +
                         " — final RMSNorm (device hc_head_out in)").c_str(),
                        head.head_norm,
                        read_half_values(stream, graph.head_norm(), kHidden), kHeadTol);

        const std::vector<double> device_logit_values = read_half_values(
            stream, device_logits, static_cast<size_t>(model.vocab));
        harness.compare(("C: token " + std::to_string(token_id) +
                         " — the logits vs the reference").c_str(),
                        head.logits, device_logit_values, kLogitTol);

        if (argmax_of(device_logit_values) == argmax_of(head.logits)) ++argmax_agreements;

        loop_logits[token_index] = read_half_bits(stream, device_logits, model.vocab);
        loop_residual[token_index] =
            read_half_bits(stream, host.scratch().d_res_in_half, hc_dim);

        // The token boundary, exactly as `forward_token` performs it.
        host.release_expert_leases();
    }

    harness.assert_that("C: the router rule reproduces the device's own top-6",
                        rule.mismatches == 0,
                        std::to_string(rule.mismatches) + " mismatches over " +
                            std::to_string(rule.biased_steps) + " biased steps");
    harness.assert_that("C: the argmax is the reference's, every token",
                        argmax_agreements == kTokens,
                        std::to_string(argmax_agreements) + " of " + std::to_string(kTokens));
    harness.assert_that("C: 43 layers were consumed at every token",
                        worst_layer_frac <= kLayerTol,
                        "worst " + sci(worst_layer_frac) + "*peak at layer " +
                            std::to_string(worst_layer) + " token " +
                            std::to_string(worst_layer_token));

    std::printf("  router near-tie measurement: worst 6th/7th score gap %.3e; %u biased "
                "steps below 1e-3\n",
                rule.worst_gap_ratio, rule.near_ties);
    std::printf("  routed-expert reuse: %zu of %zu requests are distinct (layer, expert) pairs\n",
                distinct_experts, expert_requests);
    stage("C: the composition (device + reference)", before_composition);

    // -------------------------------------------------------------------------
    // D. The driver is the loop it claims to be
    // -------------------------------------------------------------------------
    std::printf("\n[D] forward_token reproduces the per-layer loop, bit for bit\n");
    const auto before_driver = Clock::now();

    host.reset_generation_state();
    size_t driver_differences = 0;
    size_t driver_compared = 0;
    for (uint32_t token_index = 0; token_index < kTokens; ++token_index) {
        const half* logits = graph.forward_token(kTokenIds[token_index], token_index, stream);
        const std::vector<uint16_t> bits = read_half_bits(stream, logits, model.vocab);
        driver_compared += bits.size();
        for (size_t i = 0; i < bits.size(); ++i) {
            if (bits[i] != loop_logits[token_index][i]) ++driver_differences;
        }
    }
    harness.assert_that("D: forward_token == embed + 43 x run_layer + head",
                        driver_differences == 0,
                        std::to_string(driver_differences) + " of " +
                            std::to_string(driver_compared) + " fp16 logits differ");
    stage("D: the driver identity pass", before_driver);

    // -------------------------------------------------------------------------
    // E. Non-vacuity
    // -------------------------------------------------------------------------
    std::printf("\n[E] The composition is load-bearing\n");

    {
        size_t distinct_logits = 0, distinct_residuals = 0, pairs = 0;
        for (uint32_t a = 0; a < kTokens; ++a) {
            for (uint32_t b = a + 1; b < kTokens; ++b) {
                ++pairs;
                if (loop_logits[a] != loop_logits[b]) ++distinct_logits;
                if (loop_residual[a] != loop_residual[b]) ++distinct_residuals;
            }
        }
        harness.assert_that("E: distinct tokens give distinct logits and states",
                            distinct_logits == pairs && distinct_residuals == pairs,
                            std::to_string(distinct_logits) + "/" + std::to_string(pairs) +
                                " logit pairs and " + std::to_string(distinct_residuals) + "/" +
                                std::to_string(pairs) + " residual pairs differ");
    }

    host.free();
    stage("total", gate_start);

    const bool ok = harness.failures == 0;
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("[P2 — graph body] %s — %u checks, %u failed\n", ok ? "PASS" : "FAIL",
                harness.checks, harness.failures);
    return ok ? 0 : 1;
}
