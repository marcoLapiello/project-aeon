// -----------------------------------------------------------------------------
// Gate: the G2 tiled causal attention primitive, versus an independent fp64
// reference.
//
// The kernel under test is `causal_attention_fp16_wave32_kernel`: one launch for a
// whole tile of queries, each attending a shared key union masked to its own causal
// window. It is compared against `aeon::reference::attention_scores_sink`, the
// certified fp64 reference that `test_v4_attention_sink_oracle` already pins the
// scalar kernel against, so this is not a second copy of the algorithm.
//
// What this proves beyond closeness:
//
//   * THE WINDOW IS EXACTLY THE MASK. For a query at `p`, the union rows with
//     `p - W + 1 <= key_position <= p` contribute, and no others. A query whose
//     window is a strict sub-range of the union must match the reference over that
//     sub-range alone — the property the tiled formulation turns on.
//   * A NULL BIAS IS ORDINARY CAUSAL ATTENTION. With no bias the sink term is
//     absent (its `exp(-inf) = 0`), and the output matches the reference called with
//     a sink low enough to be inert.
//   * FULL CAUSAL (`window <= 0`) REACHES BACK TO POSITION 0.
//
// The comparison is a tolerance, not bit-exact: the reference is fp64 and the kernel
// fp16 output, exactly as the scalar gate judges. The tile's own reorder — should a
// later body sum across warps — is judged separately at the model's greedy bar.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "platform/tiled_causal_attention.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_HIP(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

namespace {

using aeon::reference::ErrorStats;

constexpr size_t  kHeads   = 8;      // the real DSV4 head count is 64; 8 exercises >1 block per query
constexpr size_t  kHeadDim = 512;    // the real width, because the scale is 1/sqrt(head_dim)
constexpr size_t  kKeys    = 40;     // the shared key union
constexpr size_t  kQueries = 12;     // queries, at the tail of the union
constexpr int64_t kWindow  = 8;      // small, so a query's window is a strict sub-range
constexpr double  kScale   = 0.04419417382415922; // 1 / sqrt(512)
constexpr double  kTol     = 2e-3;   // fraction of peak; output is fp16

std::vector<double> widen(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(__half2float(v[i]));
    return out;
}

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got) {
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-48s SIZE MISMATCH                       FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(s.max_rel) && s.max_rel <= kTol;
    std::printf("  %-48s max_abs=%.3e  max_rel=%.3e  %s\n",
                label, s.max_abs, s.max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

int64_t window_first(int64_t position, int64_t window) {
    if (window <= 0) return 0;
    return position >= window - 1 ? position - (window - 1) : 0;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: tiled causal attention vs fp64 reference\n";
    aeon::core::select_compute_device(true);

    bool ok = true;
    aeon::reference::Rng gen(0x7A1EDCA7ull);

    // --- inputs --------------------------------------------------------------
    std::vector<__half> h_keys(kKeys * kHeadDim);
    for (size_t i = 0; i < h_keys.size(); ++i) {
        h_keys[i] = __float2half(static_cast<float>(gen.symmetric(1.0)));
    }

    // Queries at the tail of the union so every query's window is inside it.
    std::vector<int64_t> h_query_positions(kQueries);
    for (size_t r = 0; r < kQueries; ++r) {
        h_query_positions[r] = static_cast<int64_t>(kKeys - kQueries + r);
    }
    std::vector<int64_t> h_key_positions(kKeys);
    for (size_t j = 0; j < kKeys; ++j) h_key_positions[j] = static_cast<int64_t>(j);

    std::vector<__half> h_q(kQueries * kHeads * kHeadDim);
    for (size_t i = 0; i < h_q.size(); ++i) {
        h_q[i] = __float2half(static_cast<float>(gen.symmetric(1.0)));
    }

    std::vector<float> h_sink(kHeads);
    for (size_t h = 0; h < kHeads; ++h) {
        h_sink[h] = static_cast<float>((static_cast<int>(h % 7) - 3) * 0.9 + 0.25);
    }

    // --- device buffers ------------------------------------------------------
    __half *d_q = nullptr, *d_k = nullptr, *d_out = nullptr;
    int64_t *d_kpos = nullptr, *d_qpos = nullptr;
    float* d_sink = nullptr;
    CHECK_HIP(hipMalloc(&d_q, h_q.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_k, h_keys.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_out, h_q.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_kpos, kKeys * sizeof(int64_t)));
    CHECK_HIP(hipMalloc(&d_qpos, kQueries * sizeof(int64_t)));
    CHECK_HIP(hipMalloc(&d_sink, kHeads * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_k, h_keys.data(), h_keys.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_kpos, h_key_positions.data(), kKeys * sizeof(int64_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_qpos, h_query_positions.data(), kQueries * sizeof(int64_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sink, h_sink.data(), kHeads * sizeof(float), hipMemcpyHostToDevice));

    const int q_stride = static_cast<int>(kHeads * kHeadDim);
    const int out_stride = q_stride;

    auto run = [&](int window, const float* bias) {
        CHECK_HIP(hipMemset(d_out, 0, h_q.size() * sizeof(__half)));
        aeon::dispatch_causal_attention_fp16(
            d_q, q_stride, d_k, static_cast<int>(kHeadDim), d_k, static_cast<int>(kHeadDim),
            d_kpos, d_qpos, d_out, out_stride,
            static_cast<int>(kQueries), static_cast<int>(kHeads),
            static_cast<int>(kHeadDim), static_cast<int>(kKeys), window,
            bias, static_cast<float>(kScale), 0);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<__half> out(h_q.size());
        CHECK_HIP(hipMemcpy(out.data(), d_out, out.size() * sizeof(__half), hipMemcpyDeviceToHost));
        return widen(out);
    };

    // Reference for one (query, head): the window's key rows under the same softmax.
    const std::vector<double> q_d = widen(h_q);
    const std::vector<double> k_d = widen(h_keys);
    auto reference = [&](int window, const std::vector<double>& sink, size_t query, size_t head) {
        const int64_t position = h_query_positions[query];
        const size_t first = static_cast<size_t>(window_first(position, window));
        const size_t count = static_cast<size_t>(position) - first + 1;
        const std::vector<double> query_rows(q_d.begin() + (query * kHeads + head) * kHeadDim,
                                             q_d.begin() + (query * kHeads + head + 1) * kHeadDim);
        const std::vector<double> keys_window(k_d.begin() + first * kHeadDim,
                                              k_d.begin() + (first + count) * kHeadDim);
        const std::vector<double> one_sink{sink[head]};
        return aeon::reference::attention_scores_sink(
            query_rows, 1, kHeadDim, keys_window, count, one_sink, kScale);
    };

    // Compares every (query, head) of a run against the reference. The reference
    // emits [1, head_dim] per call, so the kernel's per-head slice is collected.
    auto compare_all = [&](const char* prefix, const std::vector<double>& got, int window,
                           const std::vector<double>& sink) {
        bool all = true;
        for (size_t r = 0; r < kQueries; ++r) {
            for (size_t h = 0; h < kHeads; ++h) {
                const std::vector<double> want = reference(window, sink, r, h);
                const std::vector<double> slice(got.begin() + (r * kHeads + h) * kHeadDim,
                                                got.begin() + (r * kHeads + h + 1) * kHeadDim);
                char label[96];
                std::snprintf(label, sizeof(label), "%s q%zu h%zu", prefix, r, h);
                if (!report(label, want, slice)) all = false;
            }
        }
        return all;
    };

    // --- A. sliding window with a sink ---------------------------------------
    std::printf("--- A. sliding window W=%lld with per-head sink ---\n", (long long)kWindow);
    {
        const std::vector<double> sink_d(h_sink.begin(), h_sink.end());
        ok &= compare_all("A", run(static_cast<int>(kWindow), d_sink), static_cast<int>(kWindow), sink_d);
    }

    // --- B. full causal (window <= 0) ----------------------------------------
    std::printf("--- B. full causal (window 0) with sink ---\n");
    {
        const std::vector<double> sink_d(h_sink.begin(), h_sink.end());
        ok &= compare_all("B", run(0, d_sink), 0, sink_d);
    }

    // --- C. null bias is ordinary causal attention ---------------------------
    std::printf("--- C. null bias is inert (sink far below every score) ---\n");
    {
        // A null bias must equal a sink low enough that its mass is negligible.
        std::vector<double> inert(kHeads, -1000.0);
        ok &= compare_all("C", run(static_cast<int>(kWindow), nullptr),
                          static_cast<int>(kWindow), inert);
    }

    std::cout << (ok ? "[Tier-1 tiled causal attention] PASS\n"
                     : "[Tier-1 tiled causal attention] FAIL\n");
    return ok ? 0 : 1;
}
