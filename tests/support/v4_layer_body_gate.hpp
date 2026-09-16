#pragma once

// -----------------------------------------------------------------------------
// Shared fixture for the Tier-2 layer-body gates (items 16 and 17).
//
// Both gates drive `core/v4_layer_body.hpp` on real device state and compare
// every checkpoint against `reference/dsv4_oracle.hpp`. What they share is the
// scaffolding, not the assertions: the comparison basis, the checkpoint read
// helpers, and the routed-expert seam. Duplicating that scaffolding between two
// gates is how the two copies drift, so it lives here.
//
// The routed-expert seam has two modes on purpose:
//
//   * ARTIFACT — payloads are read from the 145 GB expert container, which is
//     what item 16 does; that is the real thing and it costs ~3 s per token in
//     page-in.
//   * SYNTHETIC — payloads are encoded locally with the oracle's own
//     `swizzled_encode` and uploaded once. Item 17 needs a 128-token HCA run,
//     which the artifact store would make a multi-minute test; the routed-expert
//     arithmetic itself is already certified by Tier-1 gates 13/15 and by item
//     16, so the compressed gate spends its budget on the compressed path.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "infrastructure/core/aeon_loader.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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
    // the loader is ignored.
    const uint8_t* synthetic[kRoutedExperts]{};

    aeon::core::PipelineScratchBuffers* scratch{nullptr};
    hipStream_t stream{0};
    uint8_t* d_payload[kRoutedExperts]{};
    std::vector<int32_t> last_ids;

    void accumulate_routed(uint32_t layer_id, uint32_t position,
                           __half* moe_accum) override {
        (void)position;

        last_ids.assign(kRoutedExperts, 0);
        CHECK_HIP(hipMemcpyAsync(last_ids.data(), scratch->d_topk_indices,
                                 kRoutedExperts * sizeof(int32_t),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

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

        // Mirrors the pipeline's default (atomic) accumulation exactly.
        aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<8, 4, 8, 16>(
            scratch->d_ffn_norm_act, w13, scratch->d_swizzled_expert_hidden,
            scratch->d_swizzled_moe_accum_f32, kHidden, kRoutedExperts,
            2048, kHidden, 10.0f, stream);
        CHECK_HIP(hipMemsetAsync(scratch->d_swizzled_counters, 0,
                                 64 * sizeof(int32_t), stream));
        aeon::kernel::dispatch_aeon_moe_fused_w2_accum<8, 8, 4, 16>(
            scratch->d_swizzled_expert_hidden, w2, scratch->d_topk_weights,
            moe_accum, scratch->d_swizzled_moe_accum_f32, moe_accum,
            scratch->d_swizzled_counters, kRoutedExperts, kHidden, 2048, stream);
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
