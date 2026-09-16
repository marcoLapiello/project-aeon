// -----------------------------------------------------------------------------
// Tier-2 gate, item 17 — the compressed layer classes (CSA ratio 4, then HCA
// ratio 128), versus the same independent fp64 composition.
//
// Item 16 certified the Sliding class: `core/v4_layer_body.hpp` with no
// compressor and no indexer. This gate certifies the *other* two branches of that
// same body, which is where the plan's trap 33 lives:
//
//   ratio  4 — CSA: local rows + the indexer-selected `index_topk` compressed rows
//   ratio 128 — HCA: local rows + EVERY committed compressed row, and no indexer
//
// What is asserted:
//
//   A. ORACLE SELF-CHECK — the APE rule and the row index in closed form, so a
//      wrong oracle cannot certify a wrong kernel.
//   B. COMPOSITION       — device checkpoints against the oracle on the artifact's
//      real layer-2 (CSA) and layer-3 (HCA) weights, across a sequence that
//      crosses ratio boundaries: `x_norm`, `q_rot`, `kv_rot`, the compressor
//      projections and its APE-adjusted partial ring row, the materialized
//      compressed entry, the indexer scores and top-k, `attn_proj`, `ffn_norm`,
//      `moe_out` and `res_out`.
//   C. NON-VACUITY / CLASS DISTINCTION — the row-set rules are shown
//      load-bearing: dropping the compressed rows moves attention, HCA's "every
//      row" differs from dropping the oldest, and CSA's selection differs from
//      "the newest k". A green suite that could not see these would not be
//      evidence for the one property that separates the classes.
//
// Deliberately NOT covered here, and named so it is not mistaken for coverage:
//   * serial multi-token state evolution *without* re-seeding — the loop locks
//     the residual to the oracle between steps so this measures one layer's
//     composition rather than the drift of a long loop. That is item 18;
//   * the SwiGLU/quantization arithmetic of the routed experts — Tier-1 gates
//     13/15 and item 16 own it, so here the experts are synthetic payloads
//     encoded with the oracle's own encoder. The 257-token HCA run would
//     otherwise spend minutes paging a 145 GB container for a path already
//     certified.
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

using aeon::reference::CompressorRing;
using aeon::reference::LayerBodyResult;
using aeon::reference::LayerBodyShape;
using aeon::reference::LayerBodyWeights;
using aeon::reference::LayerKvState;
using aeon::reference::RopeClass;
using aeon::reference::RopeTableRef;

using aeon::testgate::check;
using aeon::testgate::GateExpertExecutor;
using aeon::testgate::kHeadDim;
using aeon::testgate::kHcDim;
using aeon::testgate::kHidden;
using aeon::testgate::kRoutedExperts;
using aeon::testgate::kTotalQ;
using aeon::testgate::num;
using aeon::testgate::read_float;
using aeon::testgate::report;
using aeon::testgate::to_half;
using aeon::testgate::upload_and_read;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kMaxSeq = 512;

// Token ids cycled through the run. Layer 2 is a hash layer and layer 3 is a
// biased one, so both router branches are exercised by the same list.
const std::array<uint32_t, 10> kTokenIds = {1000, 42, 7777, 1780, 90125, 130, 55, 4096, 22222, 396};

// One class's test parameters. The local window is shrunk so the ring wraps
// within a handful of tokens (as in item 16), and for CSA `index_topk` is shrunk
// so a *real* selection — more candidates than slots — happens inside a short
// run.
struct ClassRun {
    const char* label;
    uint32_t layer_id;
    int32_t compress_ratio;
    uint32_t local_window;
    uint32_t index_topk;
    uint32_t tokens;
};

// Reads one layer's weights out of the artifact. The compressor pointers stay
// null on a Sliding layer and the indexer pointers stay null on HCA, which is
// what makes "a Sliding layer never reads them" and "HCA has no indexer"
// structural rather than asserted.
LayerBodyWeights load_weights(const aeon::core::AeonModelLoader& loader,
                              uint32_t layer_id) {
    const std::string p = "layers." + std::to_string(layer_id) + ".";
    const auto f16 = [&](const std::string& name) {
        return reinterpret_cast<const uint16_t*>(loader.get_tensor(p + name).data);
    };

    LayerBodyWeights w{};
    w.hc_attn_fn = loader.get_data_ptr<float>(p + "hc_attn_fn");
    w.hc_attn_base = loader.get_data_ptr<float>(p + "hc_attn_base");
    w.hc_attn_scale = loader.get_data_ptr<float>(p + "hc_attn_scale");
    w.hc_ffn_fn = loader.get_data_ptr<float>(p + "hc_ffn_fn");
    w.hc_ffn_base = loader.get_data_ptr<float>(p + "hc_ffn_base");
    w.hc_ffn_scale = loader.get_data_ptr<float>(p + "hc_ffn_scale");
    w.attn_norm = f16("attn_norm.weight");
    w.wq_a = f16("attn.wq_a.weight");
    w.q_norm = f16("attn.q_norm.weight");
    w.wq_b = f16("attn.wq_b.weight");
    w.wkv = f16("attn.wkv.weight");
    w.kv_norm = f16("attn.kv_norm.weight");
    w.attn_sink = loader.get_data_ptr<float>(p + "attn.attn_sink");
    w.wo_a = f16("attn.wo_a.weight");
    w.wo_b = f16("attn.wo_b.weight");
    w.ffn_norm = f16("ffn_norm.weight");
    w.gate_weight = f16("ffn.gate.weight");
    w.shared_w1 = f16("ffn.shared_experts.w1.weight");
    w.shared_w3 = f16("ffn.shared_experts.w3.weight");
    w.shared_w2 = f16("ffn.shared_experts.w2.weight");

    if (layer_id < 3) {
        w.gate_bias = nullptr;
        // The token's *row* is selected per step; storing the table base here
        // would silently route every token with token 0's expert set.
        w.tid2eid_row = nullptr;
    } else {
        w.gate_bias = loader.get_data_ptr<float>(p + "ffn.gate.bias");
        w.tid2eid_row = nullptr;
    }

    if (loader.has_tensor(p + "attn.compressor.wkv.weight")) {
        w.compressor_wkv = f16("attn.compressor.wkv.weight");
        w.compressor_wgate = f16("attn.compressor.wgate.weight");
        w.compressor_norm = f16("attn.compressor.norm.weight");
        w.compressor_ape = loader.get_data_ptr<float>(p + "attn.compressor.ape");
    }
    if (loader.has_tensor(p + "attn.indexer.wq_b.weight")) {
        w.indexer_wq_b = f16("attn.indexer.wq_b.weight");
        w.indexer_weights_proj = f16("attn.indexer.weights_proj.weight");
        w.indexer_compressor_wkv = f16("attn.indexer.compressor.wkv.weight");
        w.indexer_compressor_wgate = f16("attn.indexer.compressor.wgate.weight");
        w.indexer_compressor_norm = f16("attn.indexer.compressor.norm.weight");
        w.indexer_compressor_ape = loader.get_data_ptr<float>(p + "attn.indexer.compressor.ape");
    }
    return w;
}

// The committed-entry count the device reports for a position: `(pos+1)/ratio`,
// capped by the compressed capacity.
uint32_t committed_entries(const aeon::core::V4Layer& layer, uint32_t pos,
                           int64_t ratio) {
    return static_cast<uint32_t>(std::min<int64_t>(
        static_cast<int64_t>(layer.state_layout().compressed_capacity),
        (static_cast<int64_t>(pos) + 1) / ratio));
}

// -----------------------------------------------------------------------------
// One class's run. Returns true if every check passed.
// -----------------------------------------------------------------------------
bool run_class(const ClassRun& run, aeon::core::AeonModelLoader& loader,
               const std::vector<aeon::core::V4LayerSpec>& specs,
               const std::array<std::vector<uint8_t>, kRoutedExperts>& payloads) {
    std::printf("\n=== %s (layer %u, ratio %d) ===\n", run.label, run.layer_id,
                run.compress_ratio);
    bool ok = true;

    aeon::core::V4LayerSpec spec = specs.at(run.layer_id);
    spec.sliding_window = static_cast<int32_t>(run.local_window);
    if (run.compress_ratio == 4) spec.index_topk = static_cast<int32_t>(run.index_topk);

    aeon::core::V4Layer layer;
    layer.init_with_loader(spec, loader, kMaxSeq);
    aeon::core::PipelineScratchBuffers scratch;
    scratch.allocate();

    LayerBodyShape shape;
    shape.local_capacity = layer.local_cache_capacity();
    shape.compress_ratio = spec.compression_ratio;
    shape.index_n_heads = static_cast<uint32_t>(spec.index_n_heads);
    shape.index_head_dim = static_cast<uint32_t>(spec.index_head_dim);
    shape.index_topk = static_cast<uint32_t>(spec.index_topk);

    // The class's RoPE base: plain for ratio 0, YaRN-on-compressed otherwise.
    const RopeClass rope_class = aeon::reference::rope_class_for_ratio(spec.compression_ratio);
    const RopeTableRef rope_ref = aeon::reference::rope_table(
        aeon::reference::rope_spec_for(rope_class), run.tokens + 2);
    const RopeTableRef sliding_ref = aeon::reference::rope_table(
        aeon::reference::rope_spec_for(RopeClass::Sliding), run.tokens + 2);

    float* d_cos = nullptr;
    float* d_sin = nullptr;
    float* d_cos_c = nullptr;
    float* d_sin_c = nullptr;
    const size_t table_len = (run.tokens + 2) * 32;
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
        // Both slots must be correct, because the gate chooses which one the body
        // reads through the layer's class. (Table precision has its own gate.)
        upload(d_cos, sliding_ref.cos);
        upload(d_sin, sliding_ref.sin);
        upload(d_cos_c, rope_ref.cos);
        upload(d_sin_c, rope_ref.sin);
    }
    aeon::core::V4LayerBodyTables tables{d_cos, d_sin, d_cos_c, d_sin_c};
    aeon::core::V4NullLayerBodyObserver observer;

    GateExpertExecutor executor;
    executor.loader = nullptr;   // synthetic payloads
    executor.scratch = &scratch;
    executor.stream = 0;
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        executor.synthetic[k] = payloads[k].data();
        CHECK_HIP(hipMalloc(&executor.d_payload[k], aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
    }

    LayerBodyWeights w = load_weights(loader, run.layer_id);
    for (uint32_t k = 0; k < kRoutedExperts; ++k) w.routed_payloads[k] = payloads[k].data();

    // Decode the six fixed payloads once. The routed-expert arithmetic is
    // certified by Tier-1 gates 13/15 and by item 16 on the artifact's real
    // experts; decoding them again for every one of 257 tokens would be the whole
    // cost of this gate and would certify nothing new.
    std::vector<aeon::reference::DecodedExpertWeights> decoded(kRoutedExperts);
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        decoded[k] = aeon::reference::decode_expert_weights(payloads[k].data());
        w.routed_decoded[k] = &decoded[k];
    }

    // Hash layers route through `tid2eid[token]`, so the oracle needs the token's
    // row, not the table base. It is selected inside the loop below.
    const int64_t* tid2eid_table = (run.layer_id < 3)
        ? loader.get_data_ptr<int64_t>(
              "layers." + std::to_string(run.layer_id) + ".ffn.gate.tid2eid")
        : nullptr;

    LayerKvState oracle_state;
    oracle_state.reset(shape, layer.local_cache_capacity());

    // Seed both sides from the same fp16-rounded embedding row, broadcast to the
    // four HC streams (plan Step 1).
    std::vector<double> residual(kHcDim, 0.0);
    {
        const __half* embed = loader.get_data_ptr<__half>("embed.weight");
        const __half* row = embed + static_cast<size_t>(kTokenIds[0]) * kHidden;
        for (uint32_t j = 0; j < 4; ++j)
            for (uint32_t h = 0; h < kHidden; ++h)
                residual[j * kHidden + h] = static_cast<double>(__half2float(row[h]));
    }
    {
        const std::vector<__half> seed_half = to_half(residual);
        std::vector<float> seed_float(kHcDim);
        for (size_t i = 0; i < seed_float.size(); ++i)
            seed_float[i] = __half2float(seed_half[i]);
        CHECK_HIP(hipMemcpy(scratch.d_res_in_half, seed_half.data(),
                            kHcDim * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(scratch.d_res_in, seed_float.data(),
                            kHcDim * sizeof(float), hipMemcpyHostToDevice));
    }

    const uint32_t width = shape.compressor_width();
    const int64_t ratio = run.compress_ratio;

    // -------------------------------------------------------------------
    // A. Oracle self-checks for the compressed rules
    // -------------------------------------------------------------------
    std::printf("\n--- A. oracle self-checks (%s) ---\n", run.label);
    {
        // The fp16 decode is on the hot path of every matvec in the graph, so
        // the fast bit-assembly path is checked against the `ldexp` definition it
        // replaced, over a wide sample of bit patterns (all exponents, all
        // signs), rather than trusted.
        {
            double worst = 0.0;
            for (uint32_t bits = 0; bits < 65536u; bits += 7u) {
                const uint16_t pattern = static_cast<uint16_t>(bits);
                const int sign = (pattern >> 15) & 1;
                const int exponent = (pattern >> 10) & 0x1F;
                const int mantissa = pattern & 0x3FF;
                if (exponent == 0x1F) continue;   // NaN: not comparable by ==
                const double reference = sign != 0
                    ? -std::ldexp(exponent == 0 ? static_cast<double>(mantissa)
                                                : 1.0 + static_cast<double>(mantissa) / 1024.0,
                                  exponent == 0 ? -24 : exponent - 15)
                    : std::ldexp(exponent == 0 ? static_cast<double>(mantissa)
                                               : 1.0 + static_cast<double>(mantissa) / 1024.0,
                                 exponent == 0 ? -24 : exponent - 15);
                const double got = aeon::reference::half_bits_to_double(pattern);
                if (reference != got) worst = std::fmax(worst, std::fabs(reference - got));
            }
            ok &= check("A: fp16 decode is bit-exact vs ldexp",
                        worst == 0.0, "worst=" + num(worst, 12));
        }

        CompressorRing ring;
        ring.reset(width, shape.compressor_capacity());
        std::vector<double> kv_row(width), score_row(width);
        std::vector<double> ape(static_cast<size_t>(ratio) * width);
        for (uint32_t d = 0; d < width; ++d) {
            kv_row[d] = 1.0 + 0.001 * d;
            score_row[d] = 2.0 - 0.001 * d;
        }
        for (size_t i = 0; i < ape.size(); ++i) ape[i] = 0.5 + 0.001 * static_cast<double>(i);
        const int64_t position = ratio + 3;
        ring.store(position, ratio, kv_row, score_row, ape);

        const size_t slot = static_cast<size_t>(position % ring.capacity);
        const size_t ape_row = static_cast<size_t>(position % ratio) * width;
        double kv_delta = 0.0;
        double score_delta = 0.0;
        for (uint32_t d = 0; d < width; ++d) {
            kv_delta = std::fmax(kv_delta,
                std::fabs(ring.kv[slot * width + d] - kv_row[d]));
            score_delta = std::fmax(score_delta,
                std::fabs(ring.score[slot * width + d] - score_row[d] - ape[ape_row + d]));
        }
        ok &= check("A: compressor ring adds APE to score only",
                    kv_delta == 0.0 && score_delta < 1e-9,
                    "kv=" + num(kv_delta, 12) + " score=" + num(score_delta, 12));

        // The APE row index is a genuine modulo of the position: `p` and `p+ratio`
        // use the same row, `p` and `p+1` do not. The marker goes on the row the
        // position actually selects, or the test would compare zero against zero.
        std::vector<double> marker(static_cast<size_t>(ratio) * width, 0.0);
        marker[static_cast<size_t>(position % ratio) * width] = 1.0;
        const std::vector<double> at_p = aeon::reference::compressor_ape_apply(
            score_row, marker, position, ratio, width);
        const std::vector<double> at_p_plus_ratio = aeon::reference::compressor_ape_apply(
            score_row, marker, position + ratio, ratio, width);
        const std::vector<double> at_p_plus_one = aeon::reference::compressor_ape_apply(
            score_row, marker, position + 1, ratio, width);
        double same = 0.0;
        double differs = 0.0;
        for (uint32_t d = 0; d < width; ++d) {
            same = std::fmax(same, std::fabs(at_p[d] - at_p_plus_ratio[d]));
            differs = std::fmax(differs, std::fabs(at_p[d] - at_p_plus_one[d]));
        }
        ok &= check("A: APE row index is position % ratio",
                    same == 0.0 && differs > 0.5,
                    "same=" + num(same, 12) + " diff=" + num(differs));

        // The materializer reads a full window and produces a finite, normalized
        // row. (The window reduction itself, including the two-segment overlap
        // layout, is gated at Tier 1, item 10; this line keeps the composition's
        // hand-off honest.)
        CompressorRing filled;
        filled.reset(width, shape.compressor_capacity());
        for (uint32_t p = 0; p < shape.compressor_capacity(); ++p) {
            std::vector<double> k(width), s(width);
            for (uint32_t d = 0; d < width; ++d) {
                k[d] = 0.3 * std::sin(0.01 * (d + 7 * p));
                s[d] = 0.2 * std::cos(0.013 * (d + 3 * p));
            }
            std::vector<double> zero(static_cast<size_t>(ratio) * width, 0.0);
            filled.store(position - static_cast<int64_t>(shape.compressor_capacity()) + 1
                             + static_cast<int64_t>(p),
                         ratio, k, s, zero);
        }
        const std::vector<double> roped = aeon::reference::compressor_materialize(
            filled, shape.head_dim, static_cast<uint32_t>(ratio), position,
            aeon::reference::half_bits_to_doubles(w.compressor_norm, shape.head_dim),
            rope_ref, shape.eps);
        double peak = 0.0;
        for (double v : roped) peak = std::fmax(peak, std::fabs(v));
        ok &= check("A: materialize produces a finite non-trivial row",
                    std::isfinite(peak) && peak > 0.0, "peak=" + num(peak));
    }

    // -------------------------------------------------------------------
    // B. The layer body, device versus oracle
    // -------------------------------------------------------------------
    std::printf("\n--- B. layer body, real weights (%s) ---\n", run.label);

    LayerBodyResult final_result;
    for (uint32_t t = 0; t < run.tokens; ++t) {
        const uint32_t pos = t;
        const uint32_t token = kTokenIds[t % kTokenIds.size()];

        aeon::core::run_layer_body_decoding(layer, scratch, tables, token, pos,
                                            0, executor, observer);
        CHECK_HIP(hipStreamSynchronize(0));

        LayerKvState next_state = oracle_state;
        if (tid2eid_table != nullptr) {
            w.tid2eid_row = tid2eid_table + static_cast<size_t>(token) * kRoutedExperts;
        }
        const LayerBodyResult want = aeon::reference::layer_body(
            shape, w, rope_ref, pos, residual, next_state);
        final_result = want;

        const bool is_boundary = (static_cast<int64_t>(pos) + 1) % ratio == 0;
        ok &= check("    boundary fires exactly on the ratio",
                    want.emitted_compressed == is_boundary &&
                        layer.compressed_entry_count_ == committed_entries(layer, pos, ratio),
                    std::string(is_boundary ? "boundary" : "interior") +
                        " committed=" + std::to_string(layer.compressed_entry_count_));

        // ---- projections and the local KV row ----
        ok &= report("    x_norm  (attention RMSNorm)",
                     want.x_norm, upload_and_read(0, scratch.d_x_norm, kHidden), 3e-3);
        ok &= report("    q_rot   (MLA + per-head norm + RoPE)",
                     want.q_rot, upload_and_read(0, scratch.d_q, kTotalQ), 3e-3);
        {
            const uint32_t slot = pos % layer.local_cache_capacity();
            ok &= report("    kv_rot  (single shared K=V row)", want.kv_rot,
                         upload_and_read(0,
                             layer.d_local_key_cache + static_cast<size_t>(slot) * kHeadDim,
                             kHeadDim), 3e-3);
        }

        // ---- the compressor projections, before the APE ----
        ok &= report("    compressor_kv    (raw wkv)", want.compressor_kv,
                     upload_and_read(0, scratch.d_compressor_kv, width), 3e-3);
        ok &= report("    compressor_score (raw wgate)", want.compressor_score,
                     upload_and_read(0, scratch.d_compressor_score, width), 3e-3);

        // ---- the APE-adjusted partial row, out of the device ring ----
        {
            const uint32_t capacity =
                static_cast<uint32_t>(layer.state_layout().compressor_partial_capacity);
            const uint32_t slot = pos % capacity;
            const std::vector<double> device_score = read_float(
                0, layer.d_compressor_partial_score + static_cast<size_t>(slot) * width, width);
            const std::vector<double> device_kv = read_float(
                0, layer.d_compressor_partial_kv + static_cast<size_t>(slot) * width, width);
            const size_t at = static_cast<size_t>(slot) * width;
            const std::vector<double> oracle_score(
                next_state.compressor.score.begin() + at,
                next_state.compressor.score.begin() + at + width);
            const std::vector<double> oracle_kv(
                next_state.compressor.kv.begin() + at,
                next_state.compressor.kv.begin() + at + width);
            ok &= report("    partial_score (APE-adjusted ring row)",
                         oracle_score, device_score, 3e-3);
            ok &= report("    partial_kv    (ring row, no APE)",
                         oracle_kv, device_kv, 3e-3);

            double ape_magnitude = 0.0;
            for (uint32_t d = 0; d < width; ++d) {
                ape_magnitude = std::fmax(ape_magnitude,
                    std::fabs(oracle_score[d] - oracle_kv[d]));
            }
            ok &= check("    the APE is non-zero for this layer",
                        ape_magnitude > 1e-3, "max|ape|=" + num(ape_magnitude));
        }

        // ---- the materialized compressed entry ----
        if (want.emitted_compressed) {
            const std::vector<double> got_entry = upload_and_read(
                0, layer.d_compressed_key_cache +
                       static_cast<size_t>(want.compressed_index) * kHeadDim,
                kHeadDim);
            ok &= report("    compressed entry (materialized row)",
                         want.compressed_entry, got_entry, 3e-3);
            const std::vector<double> got_value = upload_and_read(
                0, layer.d_compressed_value_cache +
                       static_cast<size_t>(want.compressed_index) * kHeadDim,
                kHeadDim);
            bool identical = true;
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                identical = identical && (got_entry[d] == got_value[d]);
            }
            ok &= check("    compressed value row equals the key row (trap 6)",
                        identical, "bit-identical");
        }

        // ---- the indexer (CSA only) ----
        if (run.compress_ratio == 4) {
            const uint32_t candidates = committed_entries(layer, pos, ratio);
            if (candidates != 0) {
                // The indexer score is a small, sign-indefinite quantity that can
                // sit near zero (it is a rectified weighted sum), so a
                // peak-relative bound alone would reject a correct
                // implementation whenever the peak is at the fp16 noise floor.
                ok &= report("    indexer scores", want.indexer_scores,
                             read_float(0, layer.d_indexer_scores, candidates), 3e-2, 5e-4);

                const uint32_t take = std::min<uint32_t>(candidates, run.index_topk);
                std::vector<int32_t> host(run.index_topk);
                CHECK_HIP(hipMemcpyAsync(host.data(), layer.d_indexer_topk_indices,
                                         run.index_topk * sizeof(int32_t),
                                         hipMemcpyDeviceToHost, 0));
                CHECK_HIP(hipStreamSynchronize(0));
                host.resize(take);
                bool topk_ok = (host.size() == want.indexer_topk.size());
                for (size_t i = 0; topk_ok && i < host.size(); ++i) {
                    topk_ok = (host[i] == want.indexer_topk[i]);
                }
                ok &= check("    indexer top-k selection matches exactly",
                            topk_ok,
                            "candidates=" + std::to_string(candidates) +
                                " take=" + std::to_string(take));
            }
        }

        // ---- row-set counts, which are the class rule itself ----
        ok &= check("    local row count is min(pos+1, window)",
                    want.local_keys_read ==
                        std::min<uint32_t>(pos + 1, layer.local_cache_capacity()),
                    "read=" + std::to_string(want.local_keys_read));
        {
            const uint32_t committed = committed_entries(layer, pos, ratio);
            const uint32_t expected = (run.compress_ratio == 4)
                ? std::min<uint32_t>(committed,
                                     static_cast<uint32_t>(want.indexer_topk.size()))
                : committed;
            ok &= check("    compressed row count follows the class rule",
                        want.compressed_keys_read == expected,
                        "read=" + std::to_string(want.compressed_keys_read) +
                            " expected=" + std::to_string(expected) +
                            " committed=" + std::to_string(committed));
        }

        // ---- the rest of the layer ----
        ok &= report("    attn_proj (attention + inverse RoPE + wo)", want.attn_proj,
                     upload_and_read(0, scratch.d_attn_proj, kHidden), 4e-3);
        ok &= report("    ffn_norm  (HC FFN pre-mix + RMSNorm)", want.ffn_norm,
                     upload_and_read(0, scratch.d_ffn_norm_act, kHidden), 4e-3);
        ok &= report("    routed weights", want.routed_weights,
                     upload_and_read(0, scratch.d_topk_weights, kRoutedExperts), 1e-2);
        ok &= report("    router logits", want.router_logits,
                     read_float(0, scratch.d_router_logits, 256), 3e-3);
        {
            // The selection rule is checked against the **device's own logits**,
            // not the oracle's. The oracle's fp64 GEMV and the device's fp16 one
            // disagree at the ~1e-4 level on the logits, which is enough to
            // resolve a near-tie either way — so re-deriving the top-6 from the
            // device's logits is what makes this an exact test of the *rule*
            // (bias after softplus, flat top-6, ties to the lower index) instead
            // of a coincidence. The logits themselves are compared on the line
            // above, which is where their precision belongs.
            const std::vector<double> device_logits =
                read_float(0, scratch.d_router_logits, 256);

            std::vector<int32_t> expected_ids;
            if (run.layer_id < 3) {
                const int64_t* row = tid2eid_table + static_cast<size_t>(token) * kRoutedExperts;
                for (uint32_t k = 0; k < kRoutedExperts; ++k) {
                    expected_ids.push_back(static_cast<int32_t>(row[k]));
                }
            } else {
                std::vector<double> selection(shape.num_experts, 0.0);
                for (uint32_t e = 0; e < shape.num_experts; ++e) {
                    const double bias = (w.gate_bias != nullptr) ? w.gate_bias[e] : 0.0;
                    selection[e] = aeon::reference::router_score(device_logits[e]) + bias;
                }
                expected_ids = aeon::reference::topk_indices(selection, kRoutedExperts);
            }

            bool ids_ok = (executor.last_ids.size() == expected_ids.size());
            for (size_t i = 0; ids_ok && i < expected_ids.size(); ++i) {
                ids_ok = (executor.last_ids[i] == expected_ids[i]);
            }
            // The fp32 `softplus_sqrt` in the kernel against the fp64 score here
            // is the only remaining source of a different pick, and it is bounded
            // by this: the swapped experts must be within 1e-6 of each other in
            // selection value. A rule error cannot hide behind that.
            double worst_gap = 0.0;
            if (!ids_ok) {
                std::vector<double> selection(shape.num_experts, 0.0);
                for (uint32_t e = 0; e < shape.num_experts; ++e) {
                    const double bias = (w.gate_bias != nullptr) ? w.gate_bias[e] : 0.0;
                    selection[e] = aeon::reference::router_score(device_logits[e]) + bias;
                }
                double cutoff = std::numeric_limits<double>::infinity();
                for (int32_t id : expected_ids) cutoff = std::fmin(cutoff, selection[id]);
                for (int32_t id : executor.last_ids) {
                    worst_gap = std::fmax(worst_gap, cutoff - selection[id]);
                }
                ids_ok = (worst_gap <= 1e-6);
            }
            ok &= check("    routed ids match the rule on the device's logits",
                        ids_ok, ids_ok ? "exact" : ("gap=" + num(worst_gap, 9)));
        }
        ok &= report("    moe_out   (routed + shared combine)", want.moe_out,
                     upload_and_read(0, scratch.d_moe_accum, kHidden), 4e-3);
        ok &= report("    res_out   (the layer's output)", want.res_out,
                     read_float(0, scratch.d_res_in, kHcDim), 4e-3);

        oracle_state = next_state;

        // Lock the next step to the oracle's residual: this measures the layer's
        // composition, not the drift of a long loop (that is item 18's gate).
        const std::vector<__half> next_half = to_half(want.res_out);
        std::vector<float> next_float(kHcDim);
        for (size_t i = 0; i < next_float.size(); ++i) next_float[i] = __half2float(next_half[i]);
        CHECK_HIP(hipMemcpy(scratch.d_res_in_half, next_half.data(),
                            kHcDim * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(scratch.d_res_in, next_float.data(),
                            kHcDim * sizeof(float), hipMemcpyHostToDevice));
        for (size_t i = 0; i < residual.size(); ++i)
            residual[i] = static_cast<double>(next_float[i]);
    }

    // -------------------------------------------------------------------
    // C. Non-vacuity: the class's row-set rule is load-bearing
    //
    // The oracle state left by section B is reused, so this measures exactly the
    // entries the run produced.
    // -------------------------------------------------------------------
    std::printf("\n--- C. the class rule is load-bearing (%s) ---\n", run.label);
    {
        const uint32_t last = run.tokens - 1;
        const int64_t committed = (static_cast<int64_t>(last) + 1) / ratio;
        const std::vector<double> sink(w.attn_sink, w.attn_sink + shape.num_heads);
        const auto attention_with = [&](const std::vector<int32_t>& selection,
                                        bool all_compressed) {
            const std::vector<size_t> slots =
                oracle_state.local.gather(static_cast<int64_t>(last));
            std::vector<double> keys;
            for (size_t slot : slots) {
                keys.insert(keys.end(), oracle_state.local.keys.begin() + slot * kHeadDim,
                            oracle_state.local.keys.begin() + (slot + 1) * kHeadDim);
            }
            const auto append = [&](int32_t index) {
                if (index < 0 || index >= committed) return;
                const std::vector<double>& row =
                    oracle_state.compressed_keys[static_cast<size_t>(index)];
                keys.insert(keys.end(), row.begin(), row.end());
            };
            if (all_compressed) {
                for (int64_t i = 0; i < committed; ++i) append(static_cast<int32_t>(i));
            } else {
                for (int32_t index : selection) append(index);
            }
            return aeon::reference::attention_scores_sink(
                final_result.q_rot, shape.num_heads, kHeadDim, keys,
                keys.size() / kHeadDim, sink, shape.attn_scale());
        };

        const double peak = aeon::reference::peak_abs(final_result.attn_out);
        const auto moves = [&](const std::vector<double>& got) {
            double delta = 0.0;
            for (size_t i = 0; i < got.size(); ++i)
                delta = std::fmax(delta, std::fabs(got[i] - final_result.attn_out[i]));
            return (peak > 0.0) ? delta / peak : delta;
        };

        // (i) The compressed rows are genuinely in the row-set.
        const double local_only = moves(attention_with({}, false));
        ok &= check("C: compressed rows are in the attention row-set",
                    local_only > 0.05, "delta/peak=" + num(local_only));

        if (run.compress_ratio == 4) {
            // (ii) CSA's selected row-set is load-bearing: replacing one selected
            //      entry with a candidate the indexer rejected must move
            //      attention. (Formulated as a swap rather than as "the newest k"
            //      so it cannot fail a correct implementation that happens to
            //      pick the newest rows.)
            std::vector<int32_t> altered = final_result.indexer_topk;
            bool swapped = false;
            for (int64_t i = 0; i < committed && !swapped; ++i) {
                if (std::find(altered.begin(), altered.end(), static_cast<int32_t>(i)) ==
                    altered.end()) {
                    altered.back() = static_cast<int32_t>(i);
                    swapped = true;
                }
            }
            const double selection_matters = moves(attention_with(altered, false));
            ok &= check("C: CSA's chosen row-set is load-bearing",
                        swapped && selection_matters > 0.01,
                        "candidates=" + std::to_string(committed) +
                            " delta/peak=" + num(selection_matters));
        } else {
            // (iii) HCA reads **every** committed row: dropping the oldest — a
            //       row a top-k would be free to skip — must move attention.
            ok &= check("C: HCA has more than one committed row",
                        committed >= 2, "committed=" + std::to_string(committed));
            std::vector<int32_t> all_but_oldest;
            for (int64_t i = 1; i < committed; ++i) {
                all_but_oldest.push_back(static_cast<int32_t>(i));
            }
            const double oldest_matters = moves(attention_with(all_but_oldest, false));
            ok &= check("C: HCA reads the oldest compressed row too",
                        oldest_matters > 0.01, "delta/peak=" + num(oldest_matters));
        }
    }

    for (uint32_t k = 0; k < kRoutedExperts; ++k) CHECK_HIP(hipFree(executor.d_payload[k]));
    CHECK_HIP(hipFree(d_cos));
    CHECK_HIP(hipFree(d_sin));
    CHECK_HIP(hipFree(d_cos_c));
    CHECK_HIP(hipFree(d_sin_c));
    return ok;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-2: compressed layer classes (CSA, HCA)\n";
    aeon::core::select_compute_device(true);

    aeon::core::AeonModelLoader loader;
    loader.open_model(kModelDir);
    const aeon::core::DeepSeekV4Config cfg =
        aeon::core::DeepSeekV4Config::load_from_json(std::string(kModelDir) + "/config.json");
    const std::vector<aeon::core::V4LayerSpec> specs =
        aeon::core::V4ModelSpec::resolve_layers(cfg);

    // Six distinct synthetic experts, encoded once and shared by both classes.
    std::array<std::vector<uint8_t>, kRoutedExperts> payloads;
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        payloads[k] = aeon::testgate::make_synthetic_payload(k + 1);
    }

    // The local window is shrunk from 128 so the ring wraps in a few tokens; for
    // CSA `index_topk` is shrunk from 512 so the selection is non-degenerate
    // (more candidates than slots) inside a short run. HCA needs 257 tokens to
    // commit two compressed entries at ratio 128.
    const ClassRun runs[] = {
        {"CSA", 2, 4, 4, 3, 20},
        {"HCA", 3, 128, 4, 512, 257},
    };

    bool ok = true;
    for (const ClassRun& run : runs) {
        ok &= run_class(run, loader, specs, payloads);
    }

    std::cout << "\n[Tier-2 compressed classes] " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
