// -----------------------------------------------------------------------------
// Gate: the G2 tiled causal attention primitive, versus an independent fp64
// reference.
//
// The kernel under test is `causal_attention_fp16_wave32_kernel`: one launch for a
// whole tile of queries, each attending one or two shared key blocks masked to its own
// causal (and window) past. It is compared against
// `aeon::reference::attention_scores_sink`, the certified fp64 reference that
// `test_v4_attention_sink_oracle` already pins the scalar kernel against, so this is
// not a second copy of the algorithm.
//
// What this proves beyond closeness:
//
//   * A SHARED UNION AND A MASK ARE EXACT. For a query at `p`, the union rows with
//     `p - W + 1 <= key_position <= p` contribute, and no others — the property the
//     tiled formulation turns on.
//   * TWO BLOCKS WITH DIFFERENT WINDOWS SHARE ONE SOFTMAX. The window block and the
//     causal-only block are the same arithmetic as one concatenated key set, which is
//     what a windowed + compressed layer needs.
//   * A NULL BIAS IS ORDINARY CAUSAL ATTENTION, AND FULL CAUSAL REACHES POSITION 0.
//
// The comparison is a tolerance, not bit-exact: the reference is fp64 and the kernel
// fp16 output, exactly as the scalar gate judges.
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

using aeon::CausalAttentionBlock;
using aeon::reference::ErrorStats;

constexpr size_t  kHeads   = 8;      // the real DSV4 head count is 64; 8 exercises >1 block per query
constexpr size_t  kHeadDim = 512;    // the real width, because the scale is 1/sqrt(head_dim)
constexpr size_t  kKeys    = 40;     // the shared key sequence
constexpr size_t  kQueries = 12;     // queries, at the tail of the sequence (single-block cases)
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

// A block as the gate holds it on the host: where its rows live in the shared key
// sequence, and its window.
struct HostBlock {
    size_t   first  = 0;   // first key row in the sequence
    int      rows   = 0;
    int64_t  window = 0;   // 0 => causal only
};

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

    std::vector<int64_t> h_key_positions(kKeys);
    for (size_t j = 0; j < kKeys; ++j) h_key_positions[j] = static_cast<int64_t>(j);

    std::vector<int64_t> h_query_positions(kQueries);
    for (size_t r = 0; r < kQueries; ++r) {
        h_query_positions[r] = static_cast<int64_t>(kKeys - kQueries + r);
    }

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
    const int key_stride = static_cast<int>(kHeadDim);

    // Runs the primitive for a tile of queries and returns the widened output.
    // `host0`/`host1` describe the two blocks; block 1 may be empty.
    auto run = [&](const HostBlock& host0, const HostBlock& host1,
                   const int64_t* d_query_positions, int count,
                   const float* bias) {
        CausalAttentionBlock block0;
        block0.keys = d_k + host0.first * kHeadDim;
        block0.values = block0.keys;
        block0.positions = d_kpos + host0.first;
        block0.rows = host0.rows;
        block0.key_stride = key_stride;
        block0.value_stride = key_stride;
        block0.window = static_cast<int>(host0.window);

        CausalAttentionBlock block1;
        if (host1.rows > 0) {
            block1.keys = d_k + host1.first * kHeadDim;
            block1.values = block1.keys;
            block1.positions = d_kpos + host1.first;
            block1.rows = host1.rows;
            block1.key_stride = key_stride;
            block1.value_stride = key_stride;
            block1.window = static_cast<int>(host1.window);
        }

        CHECK_HIP(hipMemset(d_out, 0, h_q.size() * sizeof(__half)));
        aeon::dispatch_causal_attention_fp16(
            d_q, q_stride, block0, block1, d_query_positions, d_out, out_stride,
            count, static_cast<int>(kHeads), static_cast<int>(kHeadDim),
            bias, static_cast<float>(kScale), 0);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<__half> out(h_q.size());
        CHECK_HIP(hipMemcpy(out.data(), d_out, out.size() * sizeof(__half), hipMemcpyDeviceToHost));
        return widen(out);
    };

    // The fp64 reference for one (query, head): concatenate the two blocks' valid rows
    // in block order and run the certified softmax over them. `q_source` is the query
    // values on the host.
    const std::vector<double> q_d = widen(h_q);
    const std::vector<double> k_d = widen(h_keys);
    auto reference = [&](const HostBlock& host0, const HostBlock& host1,
                         int64_t query_position, size_t query, size_t head,
                         const std::vector<double>& sink, const std::vector<double>& q_source) {
        std::vector<double> rows;
        for (const HostBlock& block : {host0, host1}) {
            if (block.rows <= 0) continue;
            const int64_t first = window_first(query_position, block.window);
            for (int r = 0; r < block.rows; ++r) {
                const int64_t position = h_key_positions[block.first + static_cast<size_t>(r)];
                if (position >= first && position <= query_position) {
                    const size_t base = (block.first + static_cast<size_t>(r)) * kHeadDim;
                    rows.insert(rows.end(), k_d.begin() + base, k_d.begin() + base + kHeadDim);
                }
            }
        }
        const size_t count = rows.size() / kHeadDim;
        if (count == 0) return std::vector<double>(kHeadDim, 0.0);
        const std::vector<double> query_rows(
            q_source.begin() + (query * kHeads + head) * kHeadDim,
            q_source.begin() + (query * kHeads + head + 1) * kHeadDim);
        const std::vector<double> one_sink{sink[head]};
        return aeon::reference::attention_scores_sink(
            query_rows, 1, kHeadDim, rows, count, one_sink, kScale);
    };

    auto compare_all = [&](const char* prefix, const std::vector<double>& got,
                           const HostBlock& host0, const HostBlock& host1,
                           const std::vector<int64_t>& query_positions,
                           const std::vector<double>& sink,
                           const std::vector<double>& q_source) {
        bool all = true;
        for (size_t r = 0; r < query_positions.size(); ++r) {
            for (size_t h = 0; h < kHeads; ++h) {
                const std::vector<double> want =
                    reference(host0, host1, query_positions[r], r, h, sink, q_source);
                const std::vector<double> slice(got.begin() + (r * kHeads + h) * kHeadDim,
                                                got.begin() + (r * kHeads + h + 1) * kHeadDim);
                char label[96];
                std::snprintf(label, sizeof(label), "%s q%zu h%zu", prefix, r, h);
                if (!report(label, want, slice)) all = false;
            }
        }
        return all;
    };

    const std::vector<double> sink_d(h_sink.begin(), h_sink.end());

    // --- A. sliding window with a sink ---------------------------------------
    std::printf("--- A. sliding window W=%lld with per-head sink ---\n", (long long)kWindow);
    {
        const HostBlock block0{0, static_cast<int>(kKeys), kWindow};
        ok &= compare_all("A", run(block0, {}, d_qpos, static_cast<int>(kQueries), d_sink),
                          block0, {}, h_query_positions, sink_d, q_d);
    }

    // --- B. full causal (window 0) -------------------------------------------
    std::printf("--- B. full causal (window 0) with sink ---\n");
    {
        const HostBlock block0{0, static_cast<int>(kKeys), 0};
        ok &= compare_all("B", run(block0, {}, d_qpos, static_cast<int>(kQueries), d_sink),
                          block0, {}, h_query_positions, sink_d, q_d);
    }

    // --- C. null bias is ordinary causal attention ---------------------------
    std::printf("--- C. null bias is inert (sink far below every score) ---\n");
    {
        const HostBlock block0{0, static_cast<int>(kKeys), kWindow};
        std::vector<double> inert(kHeads, -1000.0);
        ok &= compare_all("C", run(block0, {}, d_qpos, static_cast<int>(kQueries), nullptr),
                          block0, {}, h_query_positions, inert, q_d);
    }

    // --- D. two blocks, different windows, one softmax -----------------------
    // Block 0 is a recent window (rows 0..5, window 3); block 1 is an older causal set
    // (rows 6..11, no window). Queries at 6..11 sit at the junction, so both blocks
    // contribute and block 1's causal mask genuinely excludes the rows ahead.
    std::printf("--- D. window block + causal-only block, one softmax ---\n");
    {
        const HostBlock block0{0, 6, 3};
        const HostBlock block1{6, 6, 0};
        std::vector<int64_t> query_positions;
        for (int64_t p = 6; p < 12; ++p) query_positions.push_back(p);

        const int count = static_cast<int>(query_positions.size());
        int64_t* d_qpos_d = nullptr;
        CHECK_HIP(hipMalloc(&d_qpos_d, query_positions.size() * sizeof(int64_t)));
        CHECK_HIP(hipMemcpy(d_qpos_d, query_positions.data(),
                            query_positions.size() * sizeof(int64_t), hipMemcpyHostToDevice));
        std::vector<__half> h_qd(static_cast<size_t>(count) * kHeads * kHeadDim);
        for (size_t i = 0; i < h_qd.size(); ++i) {
            h_qd[i] = __float2half(static_cast<float>(gen.symmetric(1.0)));
        }
        // d_q is large enough for `count` rows and the kernel indexes by `query`
        // against `query_positions`, so overwriting its prefix is a valid tile.
        CHECK_HIP(hipMemcpy(d_q, h_qd.data(), h_qd.size() * sizeof(__half), hipMemcpyHostToDevice));
        const std::vector<double> got = run(block0, block1, d_qpos_d, count, d_sink);
        CHECK_HIP(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipFree(d_qpos_d));

        ok &= compare_all("D", got, block0, block1, query_positions, sink_d, widen(h_qd));
    }

    std::cout << (ok ? "[Tier-1 tiled causal attention] PASS\n"
                     : "[Tier-1 tiled causal attention] FAIL\n");
    return ok ? 0 : 1;
}
