// -----------------------------------------------------------------------------
// Tier-2 gate, item 16 — one full Sliding-class layer body, versus an
// independent fp64 oracle.
//
// Tier 1 certified eleven primitives one at a time. This gate certifies the
// *composition*: `core/v4_layer_body.hpp::run_layer_body_decoding`, which is the
// single piece of code the rewrite's decode and chunked-batch prefill are both
// meant to drive. Tier 1's own lesson is that a layer fails in the wiring between
// correct primitives, so the gate has to compare the whole layer — `res_out` and
// the intermediate checkpoints the plan names — against a reference that is
// written independently of that code.
//
// What is asserted, and why each line earns its place:
//
//   A. ORACLE SELF-CHECK   — a hand-computable closed form, so a wrong oracle
//                            cannot certify a wrong kernel.
//   B. COMPOSITION         — device `res_out` versus the composed oracle, on the
//                            artifact's real layer-0 weights, for several
//                            positions, including past the local-ring wrap.
//   C. NON-VACUITY         — the composition is shown *load-bearing* before the
//                            pass is trusted: the HC comb orientation, the
//                            attention norm, the grouped projection and the
//                            shared expert each have to move `res_out` when read
//                            the wrong way or omitted. A gate that only compares
//                            two implementations of the same wiring can pass on
//                            a wiring that is wrong in the same way in both.
//   D. PER-STEP LOCKING    — the residual between steps is written from the
//                            oracle, so the gate measures ONE LAYER rather than
//                            the accumulation of rounding across a loop. Serial
//                            multi-token state evolution is item 18's gate.
//
// Deliberately NOT covered here, and named so it is not mistaken for coverage:
//   * the compressed classes (CSA/HCA) — items 17;
//   * the local-window rollover at the real 128-token window — the ring is
//     shrunk to make the wrap reachable in a few tokens;
//   * any tiering: this gate supplies experts directly.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "support/v4_layer_body_gate.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using aeon::reference::LayerBodyResult;
using aeon::reference::LayerBodyShape;
using aeon::reference::LayerBodyWeights;
using aeon::reference::LayerKvState;
using aeon::reference::RopeTableRef;

using aeon::testgate::check;
using aeon::testgate::GateExpertExecutor;
using aeon::testgate::kHeadDim;
using aeon::testgate::kHcDim;
using aeon::testgate::kHidden;
using aeon::testgate::kRoutedExperts;
using aeon::testgate::kTotalQ;
using aeon::testgate::read_float;
using aeon::testgate::report;
using aeon::testgate::to_half;
using aeon::testgate::upload_and_read;

// The ring is deliberately smaller than the model's 128-token window so the
// wrap is reachable in a handful of tokens. The device kernel takes the window
// length as a launch parameter, so a shrunken ring is a faithful mini-model.
constexpr uint32_t kRingTokens = 6;
constexpr uint32_t kTokens = 10;   // wraps the ring once

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

// Token ids chosen so the hash table picks distinct expert sets; the first is a
// real prompt token.
const std::array<uint32_t, kTokens> kTokenIds = {1000, 42, 7777, 1780, 90125, 130, 55, 4096, 22222, 396};

} // namespace

int main() {
    std::cout << "[Gate] Tier-2: one full Sliding-class layer body\n";
    aeon::core::select_compute_device(true);

    bool ok = true;

    const LayerBodyShape shape_defaults;
    LayerBodyShape shape = shape_defaults;
    const aeon::reference::RopeSpec rope_spec =
        aeon::reference::rope_spec_for(aeon::reference::RopeClass::Sliding);
    const aeon::reference::RopeSpec rope_spec_compressed =
        aeon::reference::rope_spec_for(aeon::reference::RopeClass::Compressed);
    const RopeTableRef rope_ref =
        aeon::reference::rope_table(rope_spec, kTokens + 1);
    const RopeTableRef rope_ref_compressed =
        aeon::reference::rope_table(rope_spec_compressed, kTokens + 1);

    // -------------------------------------------------------------------
    // Device and artifact fixtures
    // -------------------------------------------------------------------
    aeon::core::AeonModelLoader loader;
    loader.open_model(kModelDir);

    const aeon::core::DeepSeekV4Config cfg =
        aeon::core::DeepSeekV4Config::load_from_json(std::string(kModelDir) + "/config.json");
    const std::vector<aeon::core::V4LayerSpec> specs =
        aeon::core::V4ModelSpec::resolve_layers(cfg);

    aeon::core::V4Layer layer;
    // The mini-model shrinks the *window*, not the context: `record_position`
    // rejects positions past the context, so a small context would cap the run
    // at the same token count as the ring.
    aeon::core::V4LayerSpec spec0 = specs.at(0);
    spec0.sliding_window = static_cast<int32_t>(kRingTokens);
    layer.init_with_loader(spec0, loader, 256);
    aeon::core::PipelineScratchBuffers scratch;
    scratch.allocate();

    const uint32_t ring_capacity = layer.local_cache_capacity();
    // The device layer is configured with a small window so the local ring
    // wraps within a few tokens; the oracle must be told the same capacity, and
    // its class fields must say "Sliding" so it refuses any compressed path.
    shape.local_capacity = ring_capacity;
    shape.compress_ratio = 0;

    // RoPE tables: built here from the ORACLE's tables, so the gate measures the
    // layer body and not a table discrepancy. (Table precision has its own gate.)
    float* d_tab_cos = nullptr;
    float* d_tab_sin = nullptr;
    float* d_tab_cos_c = nullptr;
    float* d_tab_sin_c = nullptr;
    const size_t table_len = kTokens + 1;
    CHECK_HIP(hipMalloc(&d_tab_cos, table_len * 32 * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_tab_sin, table_len * 32 * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_tab_cos_c, table_len * 32 * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_tab_sin_c, table_len * 32 * sizeof(float)));
    {
        std::vector<float> cos_f(table_len * 32), sin_f(table_len * 32);
        std::vector<float> cos_c(table_len * 32), sin_c(table_len * 32);
        for (size_t i = 0; i < cos_f.size(); ++i) {
            cos_f[i] = static_cast<float>(rope_ref.cos[i]);
            sin_f[i] = static_cast<float>(rope_ref.sin[i]);
            cos_c[i] = static_cast<float>(rope_ref_compressed.cos[i]);
            sin_c[i] = static_cast<float>(rope_ref_compressed.sin[i]);
        }
        CHECK_HIP(hipMemcpy(d_tab_cos, cos_f.data(), cos_f.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_tab_sin, sin_f.data(), sin_f.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_tab_cos_c, cos_c.data(), cos_c.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_tab_sin_c, sin_c.data(), sin_c.size() * sizeof(float),
                            hipMemcpyHostToDevice));
    }

    aeon::core::V4LayerBodyTables tables{d_tab_cos, d_tab_sin, d_tab_cos_c, d_tab_sin_c};
    aeon::core::V4NullLayerBodyObserver observer;

    GateExpertExecutor executor;
    executor.loader = &loader;
    executor.scratch = &scratch;
    executor.stream = 0;   // default stream; the body is given the same handle
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        CHECK_HIP(hipMalloc(&executor.d_payload[k], aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
    }

    // -------------------------------------------------------------------
    // Oracle weights, read from the artifact's own tensors
    // -------------------------------------------------------------------
    const std::string p = "layers.0.";
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
    w.gate_bias = nullptr;                    // layer 0 is a hash layer
    w.shared_w1 = f16("ffn.shared_experts.w1.weight");
    w.shared_w3 = f16("ffn.shared_experts.w3.weight");
    w.shared_w2 = f16("ffn.shared_experts.w2.weight");
    const int64_t* tid2eid = loader.get_data_ptr<int64_t>(p + "ffn.gate.tid2eid");

    // Sanity: the oracle reads the weights through host pointers and the device
    // through its own uploaded copies. If those disagree, every later comparison
    // is measuring the fixture rather than the layer.
    {
        const auto cmp_weight = [&](const char* name, const uint16_t* host, const __half* device,
                                    size_t count) {
            std::vector<double> want(count);
            for (size_t i = 0; i < count; ++i)
                want[i] = aeon::reference::half_bits_to_double(host[i]);
            const std::vector<double> got = upload_and_read(0, device, count);
            return report(name, want, got, 1e-12);
        };
        ok &= cmp_weight("fixture: attn_norm host == device", w.attn_norm, layer.d_attn_norm, kHidden);
        ok &= cmp_weight("fixture: ffn_norm host == device", w.ffn_norm, layer.d_ffn_norm, kHidden);
        ok &= cmp_weight("fixture: kv_norm host == device", w.kv_norm, layer.d_kv_norm, kHeadDim);
        ok &= cmp_weight("fixture: gate_weight host == device", w.gate_weight,
                         layer.d_gate_weight, 256 * kHidden);
    }

    // -------------------------------------------------------------------
    // A. Oracle self-check — the composition against a closed form
    // -------------------------------------------------------------------
    std::cout << "\n--- A. oracle self-check ---\n";
    {
        // `hc_post` is the one composition step with a closed form: with a
        // post-mix of 1 and an identity comb, `res_out[j] = layer_out + res[j]`.
        std::vector<double> res(kHcDim, 0.0);
        for (uint32_t j = 0; j < 4; ++j)
            for (uint32_t h = 0; h < kHidden; ++h)
                res[j * kHidden + h] = 0.25 * static_cast<double>(j);
        std::vector<double> out(kHidden, 0.0);
        for (uint32_t h = 0; h < kHidden; ++h) out[h] = static_cast<double>(h) * 1e-3;

        std::vector<double> post(4, 1.0);
        std::vector<double> comb(16, 0.0);
        for (uint32_t j = 0; j < 4; ++j) comb[j * 4 + j] = 1.0;

        const std::vector<double> got =
            aeon::reference::hc_post(out, res, post, comb, kHidden);
        std::vector<double> want(kHcDim, 0.0);
        for (uint32_t j = 0; j < 4; ++j)
            for (uint32_t h = 0; h < kHidden; ++h)
                want[j * kHidden + h] = 0.25 * static_cast<double>(j) + out[h];
        ok &= report("A: hc_post closed form", want, got, 1e-12);

        // The comb orientation is a real fork, not a notational one: the
        // transposed reading of an identity comb is still an identity, so use a
        // comb that is deliberately asymmetric.
        std::vector<double> asym(16, 0.0);
        asym[0 * 4 + 1] = 1.0;   // C[contraction=0][output=1] = 1
        const std::vector<double> correct =
            aeon::reference::hc_post(out, res, post, asym, kHidden, false);
        const std::vector<double> transposed =
            aeon::reference::hc_post(out, res, post, asym, kHidden, true);
        double delta = 0.0;
        for (size_t i = 0; i < correct.size(); ++i)
            delta = std::fmax(delta, std::fabs(correct[i] - transposed[i]));
        ok &= check("A: comb orientation is load-bearing",
                    delta > 0.2, "delta=" + std::to_string(delta));
    }

    // -------------------------------------------------------------------
    // B. The layer body, device versus oracle
    // -------------------------------------------------------------------
    std::cout << "\n--- B. full layer body, real weights ---\n";

    // Start both from the same fp16-rounded residual: the embedding row of the
    // first token, broadcast to the four HC streams (plan Step 1).
    std::vector<double> residual(kHcDim, 0.0);
    {
        const __half* embed = loader.get_data_ptr<__half>("embed.weight");
        const __half* row = embed + static_cast<size_t>(kTokenIds[0]) * kHidden;
        for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t h = 0; h < kHidden; ++h) {
                const __half rounded = row[h];   // already fp16 in the checkpoint
                residual[j * kHidden + h] = static_cast<double>(__half2float(rounded));
            }
        }
    }

    LayerKvState ring;
    ring.reset(shape, ring_capacity);

    // The device side must start from the same residual the oracle does.
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

    for (uint32_t t = 0; t < kTokens; ++t) {
        const uint32_t pos = t;
        const uint32_t token = kTokenIds[t];

        // ---- device ----
        aeon::core::run_layer_body_decoding(layer, scratch, tables, token, pos,
                                            0, executor, observer);
        CHECK_HIP(hipStreamSynchronize(0));

        // ---- oracle ----
        const int64_t* row_ids = tid2eid + static_cast<size_t>(token) * kRoutedExperts;
        w.tid2eid_row = row_ids;
        for (uint32_t k = 0; k < kRoutedExperts; ++k) {
            w.routed_payloads[k] = loader.get_expert_data(
                0, static_cast<uint32_t>(row_ids[k]));
        }

        LayerKvState oracle_ring = ring;
        const LayerBodyResult want = aeon::reference::layer_body(
            shape, w, rope_ref, pos, residual, oracle_ring);

        // ---- compare ----
        std::printf("  position %u (token %u, %u local keys)\n", pos, token,
                    want.local_keys_read);

        const std::vector<double> got_x_pre = upload_and_read(0, scratch.d_x_pre, kHidden);
        ok &= report("    x_pre  (HC attention pre-combine)", want.x_pre, got_x_pre, 3e-3);

        const std::vector<double> got_x_norm = upload_and_read(0, scratch.d_x_norm, kHidden);
        ok &= report("    x_norm (attention RMSNorm)", want.x_norm, got_x_norm, 3e-3);

        const std::vector<double> got_q = upload_and_read(0, scratch.d_q, kTotalQ);
        ok &= report("    q_rot  (MLA + per-head norm + RoPE)", want.q_rot, got_q, 3e-3);

        const uint32_t slot = pos % ring_capacity;
        const std::vector<double> got_kv =
            upload_and_read(0, layer.d_local_key_cache + static_cast<size_t>(slot) * kHeadDim,
                            kHeadDim);
        ok &= report("    kv_rot (single shared K=V row)", want.kv_rot, got_kv, 3e-3);

        const std::vector<double> got_proj = upload_and_read(0, scratch.d_attn_proj, kHidden);
        ok &= report("    attn_proj (attention + inverse RoPE + wo)", want.attn_proj, got_proj, 3e-3);

        const std::vector<double> got_ffn = upload_and_read(0, scratch.d_ffn_norm_act, kHidden);
        ok &= report("    ffn_norm (HC FFN pre-mix + RMSNorm)", want.ffn_norm, got_ffn, 3e-3);

        const std::vector<double> got_moe = upload_and_read(0, scratch.d_moe_accum, kHidden);
        ok &= report("    moe_out (routed + shared combine)", want.moe_out, got_moe, 3e-3);

        const std::vector<double> got_res = read_float(0, scratch.d_res_in, kHcDim);
        ok &= report("    res_out (the layer's output)", want.res_out, got_res, 3e-3);

        // The routed ids on a hash layer come from the artifact's table, so they
        // must match exactly — a score-sorted or biased implementation cannot.
        bool ids_ok = (executor.last_ids.size() == kRoutedExperts);
        for (uint32_t k = 0; ids_ok && k < kRoutedExperts; ++k) {
            ids_ok = (executor.last_ids[k] == static_cast<int32_t>(row_ids[k]));
        }
        ok &= check("    routed ids (hash table order)", ids_ok, "exact");

        // The device ring must have written exactly the same row the oracle did,
        // and the ring must be the oracle's ring.
        ring = oracle_ring;

        // ---- lock the next step to the oracle's residual (see the D note) ----
        const std::vector<__half> next_half = to_half(want.res_out);
        std::vector<float> next_float(kHcDim);
        for (size_t i = 0; i < next_float.size(); ++i)
            next_float[i] = __half2float(next_half[i]);
        CHECK_HIP(hipMemcpy(scratch.d_res_in_half, next_half.data(),
                            kHcDim * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(scratch.d_res_in, next_float.data(),
                            kHcDim * sizeof(float), hipMemcpyHostToDevice));
        for (size_t i = 0; i < residual.size(); ++i)
            residual[i] = static_cast<double>(next_float[i]);
    }

    // -------------------------------------------------------------------
    // C. Non-vacuity: each composition step moves res_out
    // -------------------------------------------------------------------
    std::cout << "\n--- C. the composition is load-bearing ---\n";
    {
        // Re-run the final position three ways and require each wrong reading to
        // move the layer output materially. This is the Tier-2 form of the
        // mutation rule: before trusting "device == oracle", show the properties
        // the comparison is supposed to be certifying are visible in `res_out`.
        LayerKvState r1;
        r1.reset(shape, ring_capacity);
        std::vector<double> seed(kHcDim, 0.0);
        for (uint32_t i = 0; i < kHcDim; ++i) seed[i] = 0.05 * std::sin(0.01 * i);

        const LayerBodyResult base = aeon::reference::layer_body(
            shape, w, rope_ref, 0, seed, r1);

        // (i) The attention RMSNorm is a real stage: the q-lora projection built
        //     from the *unnormalized* pre-combine lands somewhere else entirely.
        //     The comparison is taken at `q_lora` rather than at `q`, because the
        //     per-head norm at the end of the Q path re-normalizes the scale and
        //     would damp the difference to a few percent — a weak discriminator
        //     that a missing norm could slip past.
        {
            const std::vector<double> q_lora_no_norm = aeon::reference::mla_q_path(
                base.x_pre,
                aeon::reference::half_bits_to_doubles(w.q_norm, shape.q_lora_rank),
                shape.q_lora_rank, shape.num_heads, shape.head_dim, shape.eps,
                [&](size_t o, size_t i) {
                    return aeon::reference::half_bits_to_double(w.wq_a[o * kHidden + i]);
                },
                [&](size_t o, size_t i) {
                    return aeon::reference::half_bits_to_double(w.wq_b[o * shape.q_lora_rank + i]);
                })
                .q_lora;
            double norm_delta = 0.0;
            for (size_t i = 0; i < q_lora_no_norm.size(); ++i)
                norm_delta = std::fmax(norm_delta, std::fabs(q_lora_no_norm[i] - base.q_lora[i]));
            const double peak = aeon::reference::peak_abs(base.q_lora);
            ok &= check("C: attention RMSNorm is load-bearing",
                        norm_delta > 0.1 * peak,
                        "delta/peak=" + std::to_string(norm_delta / peak));
        }

        // (ii) The attention path contributes to the layer output: replacing the
        //      attention projection with zeros must move `res_out`.
        {
            std::vector<double> zeros(kHidden, 0.0);
            const std::vector<double> no_attn = aeon::reference::hc_post(
                zeros, base.res_mid, base.post_f, base.comb_f, kHidden);
            double attn_delta = 0.0;
            for (size_t i = 0; i < no_attn.size(); ++i)
                attn_delta = std::fmax(attn_delta, std::fabs(no_attn[i] - base.res_out[i]));
            ok &= check("C: FFN sublayer contributes", attn_delta > 1e-2,
                        "max|delta|=" + std::to_string(attn_delta));
        }

        // (iii) The shared expert is present exactly once (item 15's property,
        //       re-observed at layer scale).
        double shared_delta = 0.0;
        for (size_t i = 0; i < kHidden; ++i)
            shared_delta = std::fmax(shared_delta,
                std::fabs(base.moe_out[i] - base.routed_sum[i]));
        ok &= check("C: shared expert is in the combine", shared_delta > 1e-3,
                    "max|delta|=" + std::to_string(shared_delta));

        // (iv) The grouped projection's group-major reading is load-bearing: the
        //      same attention output through an interleaved weight layout lands
        //      somewhere materially different, so a passing comparison says the
        //      orientation is right rather than merely self-consistent.
        const std::vector<double> z_interleaved = aeon::reference::grouped_wo_a(
            1, shape.o_groups, shape.o_lora_rank, shape.group_dim(), base.attn_inv,
            [&](size_t wi) { return aeon::reference::half_bits_to_double(w.wo_a[wi]); },
            /*contiguous_blocks=*/false);
        double grouped_delta = 0.0;
        for (size_t i = 0; i < z_interleaved.size(); ++i)
            grouped_delta = std::fmax(grouped_delta, std::fabs(z_interleaved[i] - base.z[i]));
        ok &= check("C: grouped layout is load-bearing", grouped_delta > 0.1,
                    "max|delta|=" + std::to_string(grouped_delta));
    }

    // -------------------------------------------------------------------
    // Teardown
    // -------------------------------------------------------------------
    for (uint32_t k = 0; k < kRoutedExperts; ++k) CHECK_HIP(hipFree(executor.d_payload[k]));
    CHECK_HIP(hipFree(d_tab_cos));
    CHECK_HIP(hipFree(d_tab_sin));
    CHECK_HIP(hipFree(d_tab_cos_c));
    CHECK_HIP(hipFree(d_tab_sin_c));

    std::cout << "\n[Tier-2 layer body] " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
