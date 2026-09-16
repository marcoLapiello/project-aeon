#pragma once

// -----------------------------------------------------------------------------
// Shared fixture for the Tier-2 layer-body gates (items 16, 17 and 18).
//
// All three gates drive `core/v4_layer_body.hpp` on real device state and compare
// every checkpoint against `reference/dsv4_oracle.hpp`. What they share is the
// scaffolding, not the assertions: the comparison basis, the checkpoint read
// helpers, the artifact reads, and the routed-expert seam. Duplicating that
// scaffolding between gates is how the copies drift, so it lives here.
//
// The routed-expert seam has two modes on purpose:
//
//   * ARTIFACT — payloads are read from the 145 GB expert container, which is
//     what item 16 does; that is the real thing and it costs ~3 s per token in
//     page-in.
//   * SYNTHETIC — payloads are encoded locally with the oracle's own
//     `swizzled_encode` and uploaded once. Items 17 and 18 need runs long enough
//     to cross compressor boundaries (128 tokens for one HCA entry; 136 for a
//     readable one), which the artifact store would make a multi-minute test;
//     the routed-expert arithmetic itself is already certified by Tier-1 gates
//     13/15 and by item 16, so those gates spend their budget on the serial
//     state and the compressed path.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "infrastructure/core/aeon_loader.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)
#endif

namespace aeon::testgate {

using aeon::reference::ErrorStats;

constexpr uint32_t kHidden = 4096;
constexpr uint32_t kHcDim = 4 * kHidden;
constexpr uint32_t kHeadDim = 512;
constexpr uint32_t kTotalQ = 64 * kHeadDim;
constexpr uint32_t kRoutedExperts = 6;

inline std::vector<__half> to_half(const std::vector<double>& v) {
    std::vector<__half> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = __float2half(static_cast<float>(v[i]));
    return out;
}

// The comparison basis is `max_abs` against the oracle's own peak. The chain
// stores every stage in fp16, so what matters is "how far off, as a fraction of
// what this tensor's scale is". `max_rel` is not the pass criterion: its
// denominator is floored, so on a checkpoint whose values span three decades it
// is dominated by elements near the floor.
//
// `abs_floor` is for a quantity that can *collapse* toward zero: when a tensor's
// own peak is at the noise floor, a peak-relative bound is the wrong instrument
// and rejects a correct implementation. A difference below `abs_floor` passes
// regardless of the peak. This is the same lesson Tier 1 recorded twice (the
// RoPE `cos` zero crossing and the dominated sink), now factored into the
// comparison itself.
inline bool report(const char* label, const std::vector<double>& want,
                   const std::vector<double>& got, double tol_frac,
                   double abs_floor = 0.0) {
    const ErrorStats s = aeon::reference::compare(want, got, 0.1);
    if (s.size_mismatch) {
        std::printf("  %-52s SIZE MISMATCH                FAIL\n", label);
        return false;
    }
    const double peak = aeon::reference::peak_abs(want);
    const double frac = (peak > 0.0) ? s.max_abs / peak : s.max_abs;
    const bool pass = std::isfinite(frac) && (frac <= tol_frac || s.max_abs <= abs_floor);
    std::printf("  %-52s max_abs=%.3e  =%.2e*peak  (peak=%.3e)  %s\n",
                label, s.max_abs, frac, peak, pass ? "PASS" : "FAIL");
    return pass;
}

inline bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-52s %-22s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

inline std::string num(double value, int precision = 6) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    return std::string(buffer);
}

template <typename T>
std::vector<double> upload_and_read(hipStream_t stream, const T* device, size_t count) {
    std::vector<T> host(count);
    CHECK_HIP(hipMemcpyAsync(host.data(), device, count * sizeof(T),
                             hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    std::vector<double> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = static_cast<double>(host[i]);
    return out;
}

inline std::vector<double> read_float(hipStream_t stream, const float* device, size_t count) {
    std::vector<float> host(count);
    CHECK_HIP(hipMemcpyAsync(host.data(), device, count * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    return std::vector<double>(host.begin(), host.end());
}

// -----------------------------------------------------------------------------
// Fixture reads shared by the Tier-2 gates.
// -----------------------------------------------------------------------------

// One layer's weights out of the artifact, in the oracle's own struct. The
// compressor pointers are left null on a Sliding layer and the indexer pointers
// on HCA, which is what makes "a Sliding layer never reads them" and "HCA has no
// indexer" structural rather than asserted.
//
// `tid2eid_row` is deliberately left null even on a hash layer: the *token's* row
// has to be selected per step, and storing the table base here would silently
// route every token with token 0's expert set.
inline aeon::reference::LayerBodyWeights load_layer_weights(
    const aeon::core::AeonModelLoader& loader, uint32_t layer_id) {
    const std::string p = "layers." + std::to_string(layer_id) + ".";
    const auto f16 = [&](const std::string& name) {
        return reinterpret_cast<const uint16_t*>(loader.get_tensor(p + name).data);
    };

    aeon::reference::LayerBodyWeights w{};
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
    } else {
        w.gate_bias = loader.get_data_ptr<float>(p + "ffn.gate.bias");
    }
    w.tid2eid_row = nullptr;

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
        w.indexer_compressor_ape =
            loader.get_data_ptr<float>(p + "attn.indexer.compressor.ape");
    }
    return w;
}

// The hash table's row for one token, or null on a biased layer (layers >= 3).
inline const int64_t* hash_row_for_token(const aeon::core::AeonModelLoader& loader,
                                         uint32_t layer_id, uint32_t token) {
    if (layer_id >= 3) return nullptr;
    const int64_t* table = loader.get_data_ptr<int64_t>(
        "layers." + std::to_string(layer_id) + ".ffn.gate.tid2eid");
    return table + static_cast<size_t>(token) * kRoutedExperts;
}

// The committed-entry count the device reports for a position: `(pos+1)/ratio`,
// capped by the compressed capacity.
inline uint32_t committed_entries(const aeon::core::V4Layer& layer, uint32_t pos,
                                  int64_t ratio) {
    return static_cast<uint32_t>(std::min<int64_t>(
        static_cast<int64_t>(layer.state_layout().compressed_capacity),
        (static_cast<int64_t>(pos) + 1) / ratio));
}

// -----------------------------------------------------------------------------
// The routed-expert seam.
//
// The pipeline implements this against the tiered supply system (Hot/Warm/Cold
// promotion, prefetch, leases). Here it runs the same fused kernels over payloads
// supplied directly, so the layer body's arithmetic is exercised identically
// without any storage tier involved — and without any of the routing *plumbing*
// being replaced, because the ids and the weights still come from the device.
// -----------------------------------------------------------------------------
class GateExpertExecutor final : public aeon::core::V4RoutedExpertExecutor {
public:
    // ARTIFACT mode: resolve each selected expert out of the model artifact.
    aeon::core::AeonModelLoader* loader{nullptr};
    // SYNTHETIC mode: when `synthetic[k]` is non-null it is used for slot k and
    // the loader is ignored. The payloads are uploaded per call by this class.
    const uint8_t* synthetic[kRoutedExperts]{};

    // Genuinely per-call scratch (the routed accumulate is serialized per token
    // in both the decode path and the chunk path, so one set of buffers serves
    // either).
    aeon::core::PipelineScratchBuffers* scratch{nullptr};
    hipStream_t stream{0};
    uint8_t* d_payload[kRoutedExperts]{};
    std::vector<int32_t> last_ids;
    // The fused W2 kernel reads the per-expert routing weight *on the device*, so
    // the deterministic path cannot hand it a host scalar. Allocated on first use.
    float* d_unit_weight{nullptr};

    // When true, the six experts are summed in a **fixed** order — one dispatch
    // per expert, accumulated in slot order — instead of through the fused
    // `atomicAdd` path. The fused path's order across experts is the scheduler's,
    // so `moe_out` differs between two runs of the same binary by ~1e-7; that is
    // the property trap 38 records. Item 18 deliberately drives the atomic path to
    // surface it, and item 19's `chunk ≡ serial` gate needs the deterministic one
    // or a byte-exact comparison between two runs of the *same* schedule would be
    // meaningless. **Which accumulation a gate requires is a decision, not a
    // default** — this flag is how a gate states it.
    bool deterministic{false};

    void on_routing_ready(uint32_t layer_id, uint32_t position,
                          const std::vector<int32_t>& ids,
                          const std::vector<float>& weights) override {
        (void)layer_id;
        (void)position;
        (void)weights;
        last_ids = ids;
    }

    void accumulate_routed(uint32_t layer_id, uint32_t position,
                           const half* expert_input, const float* expert_weights,
                           half* moe_accum) override {
        (void)position;

        aeon::kernel::SwizzledW13ExpertPtrs w13{};
        aeon::kernel::SwizzledW2ExpertPtrs w2{};
        for (uint32_t k = 0; k < kRoutedExperts; ++k) {
            const uint8_t* payload = (synthetic[k] != nullptr)
                ? synthetic[k]
                : loader->get_expert_data(layer_id, static_cast<uint32_t>(last_ids[k]));
            CHECK_HIP(hipMemcpyAsync(d_payload[k], payload,
                                     aeon::core::AEON_SWIZZLED_EXPERT_BYTES,
                                     hipMemcpyHostToDevice, stream));
            const uint8_t* base = d_payload[k];
            w13.w1[k] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W1_PACKED_OFFSET);
            w13.s1[k] = reinterpret_cast<const __half*>(base + aeon::core::AEON_W1_SCALE_OFFSET);
            w13.w3[k] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W3_PACKED_OFFSET);
            w13.s3[k] = reinterpret_cast<const __half*>(base + aeon::core::AEON_W3_SCALE_OFFSET);
            w2.w2[k] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W2_PACKED_OFFSET);
            w2.s2[k] = reinterpret_cast<const __half*>(base + aeon::core::AEON_W2_SCALE_OFFSET);
        }

        if (!deterministic) {
            // Mirrors the pipeline's default (atomic) accumulation exactly. One
            // dispatch for all six experts; the shared expert is folded in as
            // `initial_output`, which is the fused form the pipeline uses.
            aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<8, 4, 8, 16>(
                expert_input, w13, scratch->d_swizzled_expert_hidden,
                scratch->d_swizzled_moe_accum_f32, kHidden, kRoutedExperts,
                2048, kHidden, 10.0f, stream);
            CHECK_HIP(hipMemsetAsync(scratch->d_swizzled_counters, 0,
                                     64 * sizeof(int32_t), stream));
            aeon::kernel::dispatch_aeon_moe_fused_w2_accum<8, 8, 4, 16>(
                scratch->d_swizzled_expert_hidden, w2, expert_weights,
                moe_accum, scratch->d_swizzled_moe_accum_f32, moe_accum,
                scratch->d_swizzled_counters, kRoutedExperts, kHidden, 2048, stream);
            return;
        }

        // Deterministic: one expert at a time, in slot order, so the sum is
        // `Σ_k w_k · W2_k · swiglu(W1_k·x)` with a fixed reduction order. This is
        // the same shape as the pipeline's own `deterministic_expert_accumulation_`
        // path (per-expert GEMV plus `v4_pipeline_accumulate_expert_kernel`), and
        // like that path its accumulator is fp16 — which is the point: it is a
        // *reproducible* order, not a more accurate one.
        half* expert_hidden = scratch->d_swizzled_expert_hidden;
        half* expert_down = scratch->d_expert_down;   // [M_PAD * hidden]
        float* expert_down_f32 = scratch->d_swizzled_moe_accum_f32;
        if (d_unit_weight == nullptr) {
            const float one = 1.0f;
            CHECK_HIP(hipMalloc(&d_unit_weight, sizeof(float)));
            CHECK_HIP(hipMemcpy(d_unit_weight, &one, sizeof(float), hipMemcpyHostToDevice));
        }

        for (uint32_t k = 0; k < kRoutedExperts; ++k) {
            // The two dispatches index their weight structs by expert id, so each
            // call is given a one-slot view of this expert.
            aeon::kernel::SwizzledW13ExpertPtrs single13{};
            aeon::kernel::SwizzledW2ExpertPtrs single2{};
            single13.w1[0] = w13.w1[k];
            single13.s1[0] = w13.s1[k];
            single13.w3[0] = w13.w3[k];
            single13.s3[0] = w13.s3[k];
            single2.w2[0] = w2.w2[k];
            single2.s2[0] = w2.s2[k];

            aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<8, 4, 8, 16>(
                expert_input, single13, expert_hidden, expert_down_f32,
                kHidden, 1, 2048, kHidden, 10.0f, stream);

            const int32_t zero_counter = 0;
            CHECK_HIP(hipMemcpyAsync(scratch->d_swizzled_counters, &zero_counter,
                                     sizeof(int32_t), hipMemcpyHostToDevice, stream));
            // `d_unit_weight` is a *device* 1.0: the kernel reads the routing
            // weight from device memory, so passing a host scalar's address here
            // is an illegal access, not a wrong number.
            aeon::kernel::dispatch_aeon_moe_fused_w2_accum<8, 8, 4, 16>(
                expert_hidden, single2, d_unit_weight, nullptr, expert_down_f32,
                expert_down, scratch->d_swizzled_counters, 1, kHidden, 2048, stream);

            const int blocks = (kHidden + 255) / 256;
            aeon::kernel::v4_pipeline_accumulate_expert_kernel
                <<<blocks, 256, 0, stream>>>(moe_accum, expert_down,
                                             expert_weights[k], kHidden);
        }
    }
};

// -----------------------------------------------------------------------------
// Synthetic expert payloads.
//
// Encoded with the oracle's own `swizzled_encode` so the bytes are a *valid*
// W4A16 expert in the artifact's layout — the kernel reads them with no special
// case — and the oracle decodes the same bytes with `expert_ffn`. The weights are
// kept small (a uniform ±0.04) so the fp16 intermediates stay in range: a
// 4096-wide dot of unit-scale activations against weights of that size lands in
// the low single digits, which is where the real experts land too.
// -----------------------------------------------------------------------------
inline std::vector<uint8_t> make_synthetic_payload(uint32_t seed) {
    std::vector<uint8_t> payload(aeon::core::AEON_SWIZZLED_EXPERT_BYTES, 0);
    std::vector<double> weights;

    uint32_t state = seed * 2654435761u + 1u;
    const auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return (static_cast<double>(state >> 8) / 16777216.0) * 2.0 - 1.0;  // [-1, 1)
    };

    const auto encode = [&](aeon::reference::SwizzledKind kind, size_t rows, size_t columns) {
        weights.resize(rows * columns);
        for (double& value : weights) value = 0.04 * next();
        aeon::reference::swizzled_encode(payload.data(), kind, weights);
    };

    encode(aeon::reference::SwizzledKind::W1, 2048, 4096);
    encode(aeon::reference::SwizzledKind::W3, 2048, 4096);
    encode(aeon::reference::SwizzledKind::W2, 4096, 2048);
    return payload;
}

} // namespace aeon::testgate
