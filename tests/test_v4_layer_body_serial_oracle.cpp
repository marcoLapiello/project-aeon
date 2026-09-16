// -----------------------------------------------------------------------------
// Tier-2 gate, item 18 — serial multi-token decode across compressor boundaries,
// with the residual carried by the device itself.
//
// Items 16 and 17 each certify *one layer's composition* for one token at a time.
// Both of them make a deliberate simplification that this gate removes: after
// every step they overwrite the device's residual with `to_half(oracle.res_out)`,
// so the device's own output never feeds back into its own state. That is the
// right instrument for measuring a layer's arithmetic, and it is blind to the
// thing a *loop* does.
//
// This gate therefore runs a three-layer stack — Sliding (layer 0), CSA (layer 2),
// HCA (layer 3) — for 136 tokens and removes that simplification:
//
//   * the device's residual chain is entirely its own. Nothing writes
//     `scratch.d_res_in` (or `d_res_in_half`) from the oracle. Layer L's output is
//     layer L+1's input, and the last layer's output is the next token's input.
//   * the oracle is still the reference, but it is driven by the *device's*
//     residual trajectory: the input it is handed at each step is read out of the
//     device's own buffers. This is trap 37's principle applied to state — a
//     reference must be driven by the device's actual inputs, or the comparison
//     measures input divergence and calls it a defect. Compared this way, every
//     per-step tolerance stays tight while the device's *state* — the local ring,
//     the compressor partial ring, the committed compressed entries and their
//     positions — accumulates its own error across 408 layer-steps, 34 CSA
//     boundaries and one HCA boundary.
//   * the reference's MoE *combine* is driven by the device's own discrete
//     selection (ids and weights), for the same reason and by the same principle.
//     The selection is not thereby unchecked: the rule that produces it — bias
//     after softplus, flat top-6, ties to the lower index, hash table order — is
//     asserted against the device's **own logits**, and the logits themselves are
//     compared on the peak-relative basis. See the finding below for why an
//     elementwise selection comparison is the wrong instrument here.
//
// FINDING (item 18): the device's serial decode is **not bit-reproducible**. The
// default routed-expert path accumulates with `atomicAdd`, whose order across the
// six experts is undefined, so `moe_out` differs between two runs of the same
// binary by ~1e-7. Over a 408-step loop that drift compounds, and any router step
// whose 6th and 7th candidates sit inside the drift lands on either side — this
// gate measured 0–2 such steps per run, with a worst selection-value gap of
// 8.6e-5, varying between runs, and a resulting `moe_out` difference up to
// 7e-3 of peak (above the 4e-3 tolerance; it is what motivated the seam above).
// Items 16 and 17 could not have seen this: both re-seed the residual from the
// oracle, so their trajectory never accumulates.
//
// **Consequence taken, and why it is a decision rather than a workaround.** Driving
// the atomic path made this gate fail roughly one run in ten, and not on the
// near-tie step: once the device's trajectory and the oracle's diverge, the
// oracle's *state* — the ring keys written in earlier steps — is no longer the
// device's state, so a later `attn_proj` disagreement is a consequence of the
// divergence rather than of a defect. A gate that is red one run in ten is not a
// usable signal, and trap 38 says in as many words that every loop gate must state
// which accumulation it requires. **This gate therefore requires the deterministic
// accumulation** (`executor.deterministic = true`), the same choice item 19 made,
// and reports the nondeterminism as a measurement — `near_tie`,
// `selection_mismatch_steps`, `worst_selection_gap`, `drift` — instead of
// inferring it from an intermittently red line. The pipeline's own
// `deterministic_expert_accumulation_` removes the amplification at its source.
// Item 18's *finding* is unchanged and is why the seam and the counters exist.
//
// 136 tokens is not a round number: HCA (ratio 128) commits its first compressed
// entry at position 127, so a run that crosses an HCA boundary cannot be shorter
// than 129 tokens, and eight more are needed before the boundary is behind the
// loop rather than at its end.
//
// **Why the stack is {0, 2, 3} and not layer 1.** It is one layer per branch the
// body can *take*, not a sample of the model. The artifact's classes are
// `compress_ratios = [0, 0, 4, 128, 4, 128, …]` with `num_hash_layers = 3`, so
// layer 0 is Sliding+hash, layer 2 is CSA+hash and layer 3 is HCA+biased. Layer 1
// is the *second* Sliding layer — same attention class as layer 0 and also a hash
// layer — so it adds no branch on either axis: same code, different weights. The
// pair 2/3 is chosen because it straddles both boundaries at once (2 = last hash
// and first CSA; 3 = first biased and first HCA), which is what lets three layers
// cover three classes and both router branches.
//
// What is asserted:
//
//   A. ORACLE SELF-CHECK — a closed form for the residual hand-back and the
//      position mapping of the compressor ring, so a wrong oracle cannot certify
//      a wrong kernel.
//   B. THE SERIAL LOOP — every step of the stack, device versus oracle: the
//      attention norm, the local ring row, the compressor's APE-adjusted partial
//      row, `attn_proj`, `moe_out`, `res_out`, the row counts that are the class
//      rule, the boundary firing, and the router — its logits peak-relative and
//      its selection *rule* asserted against the device's own logits.
//   C. THE LOOP'S STATE, AND ITS NON-VACUITY — the whole accumulated state at the
//      end (every local ring slot, every committed compressed row, every position)
//      against the oracle's; closed forms for the ring positions that need no
//      oracle at all; the entry materialized at the HCA boundary shown to be
//      still there, bit-identical, hundreds of steps later and load-bearing in the
//      final attention row-set; the drift between the device's residual and the
//      oracle's measured to be non-zero — a positive measurement that the
//      device's chain is its own and not the item-17 shortcut; and the number of
//      router steps where the two selections disagreed, which is the measurement
//      behind the finding below.
//
// Deliberately NOT covered here, and named so it is not mistaken for coverage:
//   * the routed-expert arithmetic — Tier-1 gates 13/15 and item 16 own it, so the
//     experts here are six synthetic payloads encoded with the oracle's own
//     encoder (as in item 17). The 408-step run would otherwise be a multi-minute
//     page-in over a 145 GB container for a path already certified;
//   * the 128-token sliding window and the real `index_topk = 512` — the window is
//     shrunk to 4 and the top-k to 3 so the ring wraps and CSA's selection is
//     non-degenerate inside this run (items 16/17 do the same, and both are
//     launch parameters);
//   * any tiering: the experts are supplied directly.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "support/v4_layer_body_gate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

using aeon::reference::LayerBodyResult;
using aeon::reference::LayerBodyShape;
using aeon::reference::LayerBodyWeights;
using aeon::reference::LayerKvState;
using aeon::reference::RopeTableRef;

using aeon::testgate::check;
using aeon::testgate::committed_entries;
using aeon::testgate::GateExpertExecutor;
using aeon::testgate::hash_row_for_token;
using aeon::testgate::kHeadDim;
using aeon::testgate::kHcDim;
using aeon::testgate::kHidden;
using aeon::testgate::kRoutedExperts;
using aeon::testgate::load_layer_weights;
using aeon::testgate::num;
using aeon::testgate::read_float;
using aeon::testgate::report;
using aeon::testgate::to_half;
using aeon::testgate::upload_and_read;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kMaxSeq = 512;

// Shrunk exactly as in items 16 and 17: the local ring must wrap within a few
// tokens, and CSA must have more candidates than `index_topk` slots for the
// selection to be a real selection. Both are launch parameters, not model facts.
constexpr uint32_t kLocalWindow = 4;
constexpr uint32_t kIndexTopk = 3;

// HCA (ratio 128) commits its first entry at position 127, so 129 tokens is the
// hard floor for crossing a boundary; 136 leaves eight tokens on the far side.
constexpr uint32_t kTokens = 136;

constexpr uint32_t kStackSize = 3;

// Position of a token's decayed slot in a `capacity`-wide ring after `tokens`
// writes: the largest written position `p < tokens` with `p % capacity == slot`,
// or -1 if the ring never reached that slot.
int64_t ring_position_for_slot(uint32_t slot, uint32_t capacity, uint32_t tokens) {
    for (int64_t p = static_cast<int64_t>(tokens) - 1; p >= 0; --p) {
        if (p % static_cast<int64_t>(capacity) == static_cast<int64_t>(slot)) return p;
    }
    return -1;
}

// Token ids cycled through the run. Layers 0 and 2 are hash layers and layer 3 is
// biased, so both router branches are exercised by the same list.
const std::array<uint32_t, 10> kTokenIds =
    {1000, 42, 7777, 1780, 90125, 130, 55, 4096, 22222, 396};

// One layer of the serial stack: the real device layer, plus the oracle's weights
// and state for the same layer.
struct StackLayer {
    const char* label{nullptr};
    uint32_t layer_id{0};
    aeon::core::V4Layer device;
    LayerBodyShape shape{};
    RopeTableRef rope{};
    LayerBodyWeights weights{};
    LayerKvState state{};
};

// The comparison basis is the shared peak-relative `report`. The chain rounds
// every stage to fp16, so `max_abs / peak` is the honest instrument; 4e-3 is
// about four times the measured floor for the tightest checkpoint.
constexpr double kTol = 4e-3;

} // namespace

int main() {
    std::cout << "[Gate] Tier-2: serial multi-token decode across compressor boundaries\n";
    aeon::core::select_compute_device(true);
    bool ok = true;

    aeon::core::AeonModelLoader loader;
    loader.open_model(kModelDir);
    const aeon::core::DeepSeekV4Config cfg =
        aeon::core::DeepSeekV4Config::load_from_json(std::string(kModelDir) + "/config.json");
    const std::vector<aeon::core::V4LayerSpec> specs =
        aeon::core::V4ModelSpec::resolve_layers(cfg);

    // -------------------------------------------------------------------
    // A. Oracle self-checks (closed forms, before any device work)
    // -------------------------------------------------------------------
    std::cout << "\n--- A. oracle self-checks ---\n";
    {
        // The residual hand-back has a closed form: with the sublayer's post-mix
        // at zero and an identity comb, the layer's output *is* the input. This is
        // the property the serial loop depends on and the one a wrong comb
        // orientation would break, so it is pinned before the loop relies on it.
        std::vector<double> res(kHcDim, 0.0);
        for (size_t i = 0; i < res.size(); ++i) res[i] = 0.01 * std::sin(0.001 * static_cast<double>(i));
        std::vector<double> silent(kHidden, 0.0);
        std::vector<double> post(4, 0.0);
        std::vector<double> comb(16, 0.0);
        for (uint32_t j = 0; j < 4; ++j) comb[j * 4 + j] = 1.0;
        const std::vector<double> passed =
            aeon::reference::hc_post(silent, res, post, comb, kHidden);
        double delta = 0.0;
        for (size_t i = 0; i < passed.size(); ++i) delta = std::fmax(delta, std::fabs(passed[i] - res[i]));
        ok &= check("A: the residual passes through an identity comb untouched",
                    delta == 0.0, "max|delta|=" + num(delta, 12));

        // The compressor ring is position-addressed: `p` and `p + capacity` land
        // in the same slot, and the slot records the position, not the write
        // count. The loop's state contract depends on this and nothing else, so
        // it is checked without any weights.
        const uint32_t capacity = 8;
        const uint32_t width = 4;
        aeon::reference::CompressorRing ring;
        ring.reset(width, capacity);
        std::vector<double> kv_row(width, 0.0), score_row(width, 0.0), ape(width, 0.0);
        ring.store(3, 4, kv_row, score_row, ape);
        ring.store(3 + static_cast<int64_t>(capacity), 4, kv_row, score_row, ape);
        ok &= check("A: the ring slot is position modulo capacity",
                    ring.positions[3] == 3 + static_cast<int64_t>(capacity),
                    "slot 3 holds position " + std::to_string(ring.positions[3]));
    }

    // -------------------------------------------------------------------
    // The stack: one real device layer per attention class, in decode order
    // -------------------------------------------------------------------
    struct LayerPlan {
        const char* label;
        uint32_t layer_id;
    };
    const LayerPlan plans[kStackSize] = {
        {"Sliding", 0},
        {"CSA", 2},
        {"HCA", 3},
    };

    std::vector<StackLayer> stack;
    stack.reserve(kStackSize);
    for (const LayerPlan& plan : plans) {
        stack.emplace_back();
        StackLayer& s = stack.back();
        s.label = plan.label;
        s.layer_id = plan.layer_id;

        aeon::core::V4LayerSpec spec = specs.at(plan.layer_id);
        spec.sliding_window = static_cast<int32_t>(kLocalWindow);
        if (spec.attention_kind == aeon::core::V4AttentionKind::CSA) {
            spec.index_topk = static_cast<int32_t>(kIndexTopk);
        }
        s.device.init_with_loader(spec, loader, kMaxSeq);

        s.shape.local_capacity = s.device.local_cache_capacity();
        s.shape.compress_ratio = spec.compression_ratio;
        s.shape.index_n_heads = static_cast<uint32_t>(spec.index_n_heads);
        s.shape.index_head_dim = static_cast<uint32_t>(spec.index_head_dim);
        s.shape.index_topk = static_cast<uint32_t>(spec.index_topk);
        s.rope = aeon::reference::rope_table(
            aeon::reference::rope_spec_for(
                aeon::reference::rope_class_for_ratio(spec.compression_ratio)),
            kTokens);
        s.weights = load_layer_weights(loader, plan.layer_id);
        s.state.reset(s.shape, s.device.local_cache_capacity());
    }

    // -------------------------------------------------------------------
    // Device fixtures
    // -------------------------------------------------------------------
    aeon::core::PipelineScratchBuffers scratch;
    scratch.allocate();

    // The device's RoPE tables are built from the oracle's, so this gate measures
    // the layer body rather than a table discrepancy (table precision has gate 6).
    const size_t table_len = static_cast<size_t>(kTokens) * 32;
    float* d_cos = nullptr;
    float* d_sin = nullptr;
    float* d_cos_c = nullptr;
    float* d_sin_c = nullptr;
    CHECK_HIP(hipMalloc(&d_cos, table_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_sin, table_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_cos_c, table_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_sin_c, table_len * sizeof(float)));
    {
        const auto upload = [&](float* device, const std::vector<double>& values) {
            std::vector<float> host(values.size());
            for (size_t i = 0; i < values.size(); ++i) host[i] = static_cast<float>(values[i]);
            CHECK_HIP(hipMemcpy(device, host.data(), host.size() * sizeof(float),
                                hipMemcpyHostToDevice));
        };
        const RopeTableRef sliding = aeon::reference::rope_table(
            aeon::reference::rope_spec_for(aeon::reference::RopeClass::Sliding), kTokens);
        const RopeTableRef compressed = aeon::reference::rope_table(
            aeon::reference::rope_spec_for(aeon::reference::RopeClass::Compressed), kTokens);
        upload(d_cos, sliding.cos);
        upload(d_sin, sliding.sin);
        upload(d_cos_c, compressed.cos);
        upload(d_sin_c, compressed.sin);
    }
    aeon::core::V4LayerBodyTables tables{d_cos, d_sin, d_cos_c, d_sin_c};
    aeon::core::V4NullLayerBodyObserver observer;

    // Six synthetic experts, encoded once and shared by every layer and step. The
    // routed-expert arithmetic is certified elsewhere; re-decoding real payloads
    // for 408 steps would be the whole cost of this gate and would certify nothing
    // new. The payloads are valid W4A16 bytes in the artifact's own layout, so the
    // kernel reads them with no special case.
    std::array<std::vector<uint8_t>, kRoutedExperts> payloads;
    std::vector<aeon::reference::DecodedExpertWeights> decoded(kRoutedExperts);
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        payloads[k] = aeon::testgate::make_synthetic_payload(k + 1);
        decoded[k] = aeon::reference::decode_expert_weights(payloads[k].data());
    }
    for (StackLayer& s : stack) {
        for (uint32_t k = 0; k < kRoutedExperts; ++k) {
            s.weights.routed_decoded[k] = &decoded[k];
        }
    }

    GateExpertExecutor executor;
    executor.loader = nullptr;   // synthetic payloads
    executor.scratch = &scratch;
    executor.stream = 0;
    // **Which accumulation this gate requires, stated explicitly (trap 38).** The
    // finding this gate produced is that the *default* routed-expert path is not
    // bit-reproducible: `atomicAdd`'s order across the six experts is the
    // scheduler's, so `moe_out` differs between runs of the same binary and a
    // router near-tie eventually flips. The measurements are below and in the
    // header (`0–2` such steps per run, `moe_out` up to `7.0e-3` of peak).
    //
    // That nondeterminism cannot be *both* the finding and an assertion this gate
    // makes. Driving the atomic path here made the gate fail roughly one run in ten
    // — not on the near-tie step itself, but later: once the device's trajectory
    // and the oracle's diverge, the oracle's *state* (the ring keys written in past
    // steps) no longer belongs to the device, and the next step's `attn_proj`
    // disagreement is a consequence of the divergence, not of a defect. A gate that
    // is red one run in ten is not a usable signal. So the gate now **requires the
    // deterministic accumulation** — the same decision item 19 made, and the one
    // trap 38 says every loop gate must take — and the nondeterminism is reported
    // as a measurement (`near_tie`, `selection_mismatch`, `drift`) rather than
    // inferred from a flaky red line.
    executor.deterministic = true;
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        executor.synthetic[k] = payloads[k].data();
        CHECK_HIP(hipMalloc(&executor.d_payload[k], aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
    }

    // Seed both sides from the same fp16-rounded embedding row, broadcast to the
    // four HC streams (plan Step 1). This is the only value the loop is ever
    // handed from outside; everything after it is the device's own.
    {
        std::vector<double> seed(kHcDim, 0.0);
        const __half* embed = loader.get_data_ptr<__half>("embed.weight");
        const __half* row = embed + static_cast<size_t>(kTokenIds[0]) * kHidden;
        for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t h = 0; h < kHidden; ++h) {
                seed[j * kHidden + h] = static_cast<double>(__half2float(row[h]));
            }
        }
        const std::vector<__half> seed_half = to_half(seed);
        std::vector<float> seed_float(kHcDim);
        for (size_t i = 0; i < seed_float.size(); ++i) {
            seed_float[i] = __half2float(seed_half[i]);
        }
        CHECK_HIP(hipMemcpy(scratch.d_res_in_half, seed_half.data(),
                            kHcDim * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(scratch.d_res_in, seed_float.data(),
                            kHcDim * sizeof(float), hipMemcpyHostToDevice));
    }

    // -------------------------------------------------------------------
    // B. The serial loop — three classes in decode order, device-driven
    // -------------------------------------------------------------------
    std::cout << "\n--- B. serial decode, three classes in one stack ---\n";

    std::vector<double> previous_oracle_res;   // the oracle's own last output
    double drift_max = 0.0;                    // how far the device's chain is from it
    std::vector<double> boundary_snapshot;     // HCA's committed entry, as written
    uint32_t boundary_index = 0;
    bool have_boundary = false;
    LayerBodyResult final_hca;
    uint32_t near_tie_steps = 0;               // the rule needed the tie allowance
    uint32_t selection_mismatch_steps = 0;     // oracle and device picked differently
    double worst_selection_gap = 0.0;

    for (uint32_t t = 0; t < kTokens; ++t) {
        const uint32_t pos = t;
        const uint32_t token = kTokenIds[t % kTokenIds.size()];
        if (t % 16 == 0 || t + 1 == kTokens) {
            std::printf("\n  -- token %u/%u (position %u, id %u) --\n", t + 1, kTokens, pos, token);
        }

        for (StackLayer& s : stack) {
            // The residual this layer is about to consume is read out of the
            // device *before* the device runs. That value is both the oracle's
            // input and, at the start of the next layer, the check that the
            // device's chain is its own.
            const std::vector<double> input_res = read_float(0, scratch.d_res_in, kHcDim);

            if (previous_oracle_res.size() == input_res.size()) {
                double drift = 0.0;
                for (size_t i = 0; i < input_res.size(); ++i) {
                    const double half_previous = static_cast<double>(__half2float(
                        __float2half(static_cast<float>(previous_oracle_res[i]))));
                    drift = std::fmax(drift, std::fabs(input_res[i] - half_previous));
                }
                drift_max = std::fmax(drift_max, drift);
            }

            s.weights.tid2eid_row = hash_row_for_token(loader, s.layer_id, token);

            // The device runs first. Its *discrete* selection is one of its
            // outputs, and the reference is driven by it (`routed_ids_override`),
            // so the combine below is an arithmetic comparison rather than a
            // coincidence hunt across two independent selections. The rule that
            // produced the selection is certified on its own, two blocks down.
            // Nothing here writes `d_res_in`: the residual the device consumed was
            // read above and the one it produced is read next.
            aeon::core::run_layer_body_decoding(s.device, scratch, tables, token, pos,
                                                0, executor, observer);
            CHECK_HIP(hipStreamSynchronize(0));
            const std::vector<double> device_res_out =
                read_float(0, scratch.d_res_in, kHcDim);
            const std::vector<double> device_weights =
                read_float(0, scratch.d_topk_weights, kRoutedExperts);
            const std::vector<float> device_weights_float(
                device_weights.begin(), device_weights.end());

            s.weights.routed_ids_override = executor.last_ids.data();
            s.weights.routed_weights_override = device_weights_float.data();

            const LayerBodyResult want = aeon::reference::layer_body(
                s.shape, s.weights, s.rope, pos, input_res, s.state);
            previous_oracle_res = want.res_out;

            const uint32_t local_capacity = s.device.local_cache_capacity();
            const uint32_t slot = pos % local_capacity;

            ok &= report("      x_norm    (attention RMSNorm)", want.x_norm,
                         upload_and_read(0, scratch.d_x_norm, kHidden), kTol);
            ok &= report("      kv_rot    (local ring row, slot pos%cap)", want.kv_rot,
                         upload_and_read(0,
                             s.device.d_local_key_cache + static_cast<size_t>(slot) * kHeadDim,
                             kHeadDim), kTol);
            ok &= report("      attn_proj (attention + inverse RoPE + wo)", want.attn_proj,
                         upload_and_read(0, scratch.d_attn_proj, kHidden), kTol);

            // The router: the logits on the peak-relative basis, and the *rule*
            // checked against the device's own logits (trap 37). The rule is the
            // pass criterion for the selection — not an elementwise comparison of
            // the two selections — because the device's serial loop is not
            // bit-reproducible: the default MoE path accumulates with `atomicAdd`,
            // whose order across the six experts is undefined, so the device's
            // trajectory drifts from any fixed reference and a router step whose
            // 6th and 7th candidates sit inside that drift lands on either side.
            // The rule check cannot be moved by that; an elementwise one can, and
            // did, once in the runs that established this gate.
            ok &= report("      router logits", want.router_logits,
                         read_float(0, scratch.d_router_logits, 256), kTol);
            {
                const std::vector<double> device_logits =
                    read_float(0, scratch.d_router_logits, 256);

                aeon::reference::RouterSelection rule;
                std::vector<double> choice(s.shape.num_experts, 0.0);
                if (s.layer_id < 3) {
                    const int64_t* row = hash_row_for_token(loader, s.layer_id, token);
                    rule = aeon::reference::router_hash(
                        device_logits,
                        std::vector<int64_t>(row, row + kRoutedExperts),
                        s.shape.routed_scaling);
                    for (uint32_t e = 0; e < s.shape.num_experts; ++e) {
                        choice[e] = aeon::reference::router_score(device_logits[e]);
                    }
                } else {
                    rule = aeon::reference::router_topk(
                        device_logits,
                        std::vector<double>(s.weights.gate_bias,
                                            s.weights.gate_bias + s.shape.num_experts),
                        kRoutedExperts, s.shape.routed_scaling);
                    for (uint32_t e = 0; e < s.shape.num_experts; ++e) {
                        choice[e] = aeon::reference::router_score(device_logits[e]) +
                                    s.weights.gate_bias[e];
                    }
                }

                bool ids_ok = (executor.last_ids.size() == rule.ids.size());
                for (size_t i = 0; ids_ok && i < rule.ids.size(); ++i) {
                    ids_ok = (executor.last_ids[i] == rule.ids[i]);
                }

                double weight_delta = 0.0;
                double worst_gap = 0.0;
                if (ids_ok) {
                    // The same ids, so the weights are comparable elementwise.
                    // The device's fp32 `softplus_sqrt` against this file's fp64
                    // one is the only difference that can move a weight.
                    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
                        weight_delta = std::fmax(
                            weight_delta, std::fabs(device_weights[k] - rule.weights[k]));
                    }
                    ids_ok = (weight_delta <= 1e-5);
                } else {
                    // A different pick is only legitimate when it is a tie: the
                    // swapped experts have to be within 1e-6 of each other in
                    // selection value. A rule error cannot hide behind that.
                    double cutoff = std::numeric_limits<double>::infinity();
                    for (int32_t id : rule.ids) cutoff = std::fmin(cutoff, choice[id]);
                    for (int32_t id : executor.last_ids) {
                        worst_gap = std::fmax(worst_gap, cutoff - choice[id]);
                    }
                    ids_ok = (worst_gap <= 1e-6);
                    if (ids_ok) ++near_tie_steps;
                }
                ok &= check("      routed ids and weights follow the rule",
                            ids_ok,
                            ids_ok ? ("exact, max|dweight|=" + num(weight_delta, 9))
                                   : ("gap=" + num(worst_gap, 9)));

                // Measurement only, not a pass criterion: how often the oracle's
                // own fp64 selection and the device's disagreement, and by how much
                // in selection value. This is the channel the drift above travels
                // through, so it is counted and reported rather than left to be
                // rediscovered. (The reference's *combine* does not depend on it —
                // it is driven by the device's selection.)
                aeon::reference::RouterSelection oracle_selection;
                if (s.layer_id < 3) {
                    const int64_t* row = hash_row_for_token(loader, s.layer_id, token);
                    oracle_selection = aeon::reference::router_hash(
                        want.router_logits,
                        std::vector<int64_t>(row, row + kRoutedExperts),
                        s.shape.routed_scaling);
                } else {
                    oracle_selection = aeon::reference::router_topk(
                        want.router_logits,
                        std::vector<double>(s.weights.gate_bias,
                                            s.weights.gate_bias + s.shape.num_experts),
                        kRoutedExperts, s.shape.routed_scaling);
                }
                bool same_ids = (oracle_selection.ids.size() == executor.last_ids.size());
                for (size_t i = 0; same_ids && i < oracle_selection.ids.size(); ++i) {
                    same_ids = (oracle_selection.ids[i] == executor.last_ids[i]);
                }
                if (!same_ids) {
                    ++selection_mismatch_steps;
                    std::vector<double> selection(s.shape.num_experts, 0.0);
                    for (uint32_t e = 0; e < s.shape.num_experts; ++e) {
                        const double bias = (s.layer_id < 3) ? 0.0 : s.weights.gate_bias[e];
                        selection[e] =
                            aeon::reference::router_score(want.router_logits[e]) + bias;
                    }
                    double cutoff = std::numeric_limits<double>::infinity();
                    for (int32_t id : oracle_selection.ids) {
                        cutoff = std::fmin(cutoff, selection[id]);
                    }
                    for (int32_t id : executor.last_ids) {
                        worst_selection_gap = std::fmax(worst_selection_gap,
                                                        cutoff - selection[id]);
                    }
                }
            }

            ok &= report("      moe_out   (routed + shared combine)", want.moe_out,
                         upload_and_read(0, scratch.d_moe_accum, kHidden), kTol);

            ok &= report("      res_out   (carried to the next layer)", want.res_out,
                         device_res_out, kTol);

            ok &= check("      local rows read = min(pos+1, window)",
                        want.local_keys_read == std::min<uint32_t>(pos + 1, local_capacity),
                        "read=" + std::to_string(want.local_keys_read));

            if (s.shape.is_compressed()) {
                const int64_t ratio = s.shape.compress_ratio;
                const bool is_boundary = (static_cast<int64_t>(pos) + 1) % ratio == 0;
                const uint32_t committed = committed_entries(s.device, pos, ratio);

                ok &= check("      boundary fires exactly on the ratio",
                            want.emitted_compressed == is_boundary &&
                                s.device.compressed_entry_count_ == committed,
                            std::string(is_boundary ? "boundary" : "interior") +
                                " committed=" +
                                std::to_string(s.device.compressed_entry_count_));

                // The APE-adjusted partial row, straight out of the device ring.
                // Over 136 tokens the ring wraps 17 times on CSA, so this is also
                // the loop's ring bookkeeping under wrap.
                const uint32_t partial_capacity = static_cast<uint32_t>(
                    s.device.state_layout().compressor_partial_capacity);
                const uint32_t partial_slot = pos % partial_capacity;
                const size_t at = static_cast<size_t>(partial_slot) * s.shape.compressor_width();
                const std::vector<double> oracle_score(
                    s.state.compressor.score.begin() + at,
                    s.state.compressor.score.begin() + at + s.shape.compressor_width());
                ok &= report("      compressor partial row (APE-adjusted)", oracle_score,
                             read_float(0, s.device.d_compressor_partial_score + at,
                                        s.shape.compressor_width()), kTol);

                const uint32_t expected_rows = s.shape.uses_indexer()
                    ? std::min<uint32_t>(committed, s.shape.index_topk)
                    : committed;
                ok &= check("      compressed rows follow the class rule",
                            want.compressed_keys_read == expected_rows,
                            "read=" + std::to_string(want.compressed_keys_read) +
                                " expected=" + std::to_string(expected_rows));
            }

            if (s.layer_id == 3 && want.emitted_compressed) {
                boundary_index = want.compressed_index;
                boundary_snapshot = upload_and_read(
                    0, s.device.d_compressed_key_cache +
                           static_cast<size_t>(want.compressed_index) * kHeadDim,
                    kHeadDim);
                have_boundary = true;
            }
            if (s.layer_id == 3) final_hca = want;
        }
    }

    // -------------------------------------------------------------------
    // C. The loop's accumulated state
    // -------------------------------------------------------------------
    std::cout << "\n--- C. the loop's accumulated state ---\n";

    for (StackLayer& s : stack) {
        const uint32_t local_capacity = s.device.local_cache_capacity();
        std::printf("  [%s] layer %u, %u local slots\n", s.label, s.layer_id, local_capacity);

        // The whole local ring, every slot — not just the one the last token wrote.
        ok &= report("    local ring (all slots)", s.state.local.keys,
                     upload_and_read(0, s.device.d_local_key_cache,
                                     static_cast<size_t>(local_capacity) * kHeadDim), kTol);

        // Every position the ring records, against a closed form that needs no
        // oracle: slot `s` holds the last position written to it, which after this
        // many tokens is the largest `p < kTokens` congruent to `s` modulo the
        // capacity.
        {
            const std::vector<double> got =
                upload_and_read(0, s.device.d_local_positions, local_capacity);
            bool same = true;
            int64_t first_bad = 0;
            for (uint32_t slot = 0; slot < local_capacity; ++slot) {
                const int64_t expect =
                    ring_position_for_slot(slot, local_capacity, kTokens);
                if (got[slot] != static_cast<double>(expect)) {
                    if (same) first_bad = static_cast<int64_t>(slot);
                    same = false;
                }
            }
            ok &= check("    local ring positions are the last write per slot", same,
                        same ? "closed form"
                             : ("slot " + std::to_string(first_bad) +
                                " = " + num(got[static_cast<size_t>(first_bad)])));
        }

        if (!s.shape.is_compressed()) continue;

        const int64_t ratio = s.shape.compress_ratio;
        const uint32_t committed = s.device.compressed_entry_count_;
        ok &= check("    committed entry count matches the oracle's state",
                    s.state.compressed_keys.size() == committed,
                    "device=" + std::to_string(committed) +
                        " oracle=" + std::to_string(s.state.compressed_keys.size()));

        // Every committed compressed entry, which is the state a boundary wrote and
        // every later token read.
        {
            std::vector<double> want;
            for (uint32_t i = 0; i < committed; ++i) {
                want.insert(want.end(), s.state.compressed_keys[i].begin(),
                            s.state.compressed_keys[i].end());
            }
            ok &= report("    committed compressed entries (all of them)", want,
                         upload_and_read(0, s.device.d_compressed_key_cache,
                                         static_cast<size_t>(committed) * kHeadDim), kTol);
        }

        // Their positions: entry `i` was committed at the window's last position,
        // `(i + 1) * ratio - 1`.
        {
            const std::vector<double> got = upload_and_read(
                0, s.device.d_compressed_positions, committed);
            std::vector<double> want(committed);
            bool same = true;
            for (uint32_t i = 0; i < committed; ++i) {
                want[i] = static_cast<double>(
                    (static_cast<int64_t>(i) + 1) * ratio - 1);
                if (got[i] != want[i]) same = false;
            }
            ok &= check("    compressed entry positions are (i+1)*ratio - 1", same,
                        same ? "closed form" : ("device[0]=" + num(got.empty() ? 0.0 : got[0])));
        }

        // The compressor partial ring: every slot's recorded position.
        {
            const uint32_t capacity = static_cast<uint32_t>(
                s.device.state_layout().compressor_partial_capacity);
            const std::vector<double> got = upload_and_read(
                0, s.device.d_compressor_partial_positions, capacity);
            bool same = true;
            int64_t first_bad = 0;
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                const int64_t expect = ring_position_for_slot(slot, capacity, kTokens);
                if (got[slot] != static_cast<double>(expect)) {
                    if (same) first_bad = static_cast<int64_t>(slot);
                    same = false;
                }
            }
            ok &= check("    compressor partial ring positions track the writes", same,
                        same ? "closed form"
                             : ("slot " + std::to_string(first_bad) +
                                " = " + num(got[static_cast<size_t>(first_bad)])));
        }
    }

    // -------------------------------------------------------------------
    // C. Non-vacuity
    // -------------------------------------------------------------------
    {
        const StackLayer& hca = stack.back();

        // (i) The entry the HCA boundary materialized, hundreds of steps back, is
        //     still there and unchanged. This is the loop's state contract in one
        //     line: a state that is recomputed or cleared each step would pass
        //     every per-step comparison and fail here.
        if (have_boundary && boundary_index < static_cast<uint32_t>(kHeadDim)) {
            const std::vector<double> now = upload_and_read(
                0, hca.device.d_compressed_key_cache +
                       static_cast<size_t>(boundary_index) * kHeadDim,
                kHeadDim);
            double delta = 0.0;
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                delta = std::fmax(delta, std::fabs(now[d] - boundary_snapshot[d]));
            }
            ok &= check("C: the boundary's entry survives the whole loop",
                        delta == 0.0, "max|delta|=" + num(delta, 12));
        } else {
            ok &= check("C: the boundary's entry survives the whole loop", false,
                        "no HCA boundary was reached");
        }

        // (ii) That entry is load-bearing in the final attention row-set: dropping
        //      the compressed rows moves attention, so "the loop read its own
        //      state" is a measured property rather than a claim.
        const int64_t last = static_cast<int64_t>(kTokens) - 1;
        const int64_t ratio = hca.shape.compress_ratio;
        const int64_t committed = (last + 1) / ratio;
        const std::vector<double> sink(
            hca.weights.attn_sink, hca.weights.attn_sink + hca.shape.num_heads);
        const auto attention_with = [&](bool include_compressed) {
            const std::vector<size_t> slots = hca.state.local.gather(last);
            std::vector<double> keys;
            for (size_t slot : slots) {
                keys.insert(keys.end(),
                            hca.state.local.keys.begin() + slot * kHeadDim,
                            hca.state.local.keys.begin() + (slot + 1) * kHeadDim);
            }
            if (include_compressed) {
                for (int64_t i = 0; i < committed; ++i) {
                    const std::vector<double>& row =
                        hca.state.compressed_keys[static_cast<size_t>(i)];
                    keys.insert(keys.end(), row.begin(), row.end());
                }
            }
            return aeon::reference::attention_scores_sink(
                final_hca.q_rot, hca.shape.num_heads, kHeadDim, keys,
                keys.size() / kHeadDim, sink, hca.shape.attn_scale());
        };

        const double peak = aeon::reference::peak_abs(final_hca.attn_out);
        const std::vector<double> local_only = attention_with(false);
        double moved = 0.0;
        for (size_t i = 0; i < local_only.size(); ++i) {
            moved = std::fmax(moved, std::fabs(local_only[i] - final_hca.attn_out[i]));
        }
        const double moved_fraction = (peak > 0.0) ? moved / peak : moved;
        ok &= check("C: the committed compressed row is in the final row-set",
                    moved_fraction > 0.05,
                    "committed=" + std::to_string(committed) +
                        " delta/peak=" + num(moved_fraction));

        // (iii) The device's residual chain is its own. In items 16 and 17 the
        //       harness writes `to_half(oracle.res_out)` into `d_res_in` after
        //       every step, so this quantity is exactly zero by construction. Here
        //       it is the measured distance between the device's trajectory and
        //       the oracle's, and a non-zero value is the positive evidence that
        //       the shortcut is gone.
        ok &= check("C: the device's residual is its own (never re-seeded)",
                    drift_max > 0.0, "max|device - half(oracle)|=" + num(drift_max, 9));

        // (iv) How often the oracle's own selection and the device's disagree.
        //      With the deterministic accumulation this is no longer a drift
        //      channel: the residue is **exact ties reordered**, which the
        //      positional id comparison counts as a mismatch while the selection
        //      *values* are identical — hence `worst gap = 0.0`. That is benign by
        //      construction (the combine is driven by the device's own ids), and
        //      the counter is kept as a measurement: a non-zero gap here would mean
        //      a rule disagreement, and a gap that *grows* would mean drift.
        std::printf("  %-52s near_tie=%u\n",
                    "C: near-tie allowance in the router rule check", near_tie_steps);
        std::printf("  %-52s %u of %u, worst gap %.3e\n",
                    "C: oracle vs device selection", selection_mismatch_steps,
                    kTokens * kStackSize, worst_selection_gap);
    }

    for (uint32_t k = 0; k < kRoutedExperts; ++k) CHECK_HIP(hipFree(executor.d_payload[k]));
    CHECK_HIP(hipFree(d_cos));
    CHECK_HIP(hipFree(d_sin));
    CHECK_HIP(hipFree(d_cos_c));
    CHECK_HIP(hipFree(d_sin_c));

    std::cout << "\n[Tier-2 serial decode] " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
