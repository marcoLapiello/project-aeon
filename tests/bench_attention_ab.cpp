// -----------------------------------------------------------------------------
// Tiled-causal-attention A/B: the per-query split kernel versus the query-tiled
// WMMA QKᵀ kernel, at production shapes.
//
// ## What question this answers
//
// `KERNELS_IMPROVEMENT` Area 6 says the attention kernel is now the largest single
// term (about `48%` of prefill GPU) and that the one-warp WMMA QKᵀ attempt was
// **neutral** — "matrix-core throughput and occupancy trade exactly". That record
// leaves two things unmeasured, and they are the two levers this file separates:
//
//   1. **Warp count, at a fixed kernel.** The split kernel's `kCausalAttentionWarps`
//      is a compile-time constant (4) that has never been swept. If the key scan is
//      the cost, more warps scan fewer keys each; if occupancy is the cost, more warps
//      lower it. The sweep says which.
//
//   2. **Query tiling.** The split kernel is `(head, query)` — it re-reads the whole
//      key union **per query**. The WMMA kernel is `(head, query-tile of 16)` and reads
//      the union once for 16 queries, i.e. `16x` less key traffic. If the kernel is
//      key-bandwidth-bound, tiling is the win regardless of the matrix cores; the
//      single-warp build lost only because one warp per block underfilled the GPU.
//
// Neither is answerable from the source; both are budgets that meet only on silicon.
//
// ## What is measured and what is not
//
// Only the **attention kernel** is timed. The layer-body tail (HC, norms) is separate
// and measured by the phase profiler. Keys and values are DRAM-resident and larger than
// the `96 MiB` Infinity Cache in the long-row cases, so the byte figures are a lower
// bound on what the kernel pulls through the hierarchy.
//
// **Correctness is checked, not assumed.** Every arm writes the same `[count, heads,
// head_dim]` output, and the arms are compared elementwise; without that this is a race
// between programs that might not compute the same function. The reference for the
// formulation is `tests/test_tiled_causal_attention.cpp`; this file only shows the arms
// agree.
//
// It does not assert. The numbers are the deliverable.
//
// Usage: bench_attention_ab [queries...] [--rows <total_rows>]
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "platform/tiled_causal_attention.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
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

constexpr int kHeads = 64;              // the real DSV4 head count
constexpr int kHeadDim = 512;           // the real width
constexpr int64_t kWindow = 128;        // the real sliding-window length
constexpr float kScale = 0.04419417382415922f;  // 1 / sqrt(512)
constexpr int kRepetitions = 5;

struct Buffers {
    __half* q{nullptr};       // [count, heads, head_dim]
    __half* keys{nullptr};    // [rows, head_dim]
    __half* out{nullptr};     // [count, heads, head_dim]
    int64_t* positions{nullptr};  // [rows]
};

__half* alloc_half(size_t n) {
    __half* p = nullptr;
    CHECK_HIP(hipMalloc(&p, n * sizeof(__half)));
    return p;
}

Buffers make_buffers(int count, int rows) {
    Buffers b;
    b.q = alloc_half(static_cast<size_t>(count) * kHeads * kHeadDim);
    b.keys = alloc_half(static_cast<size_t>(rows) * kHeadDim);
    b.out = alloc_half(static_cast<size_t>(count) * kHeads * kHeadDim);
    CHECK_HIP(hipMalloc(&b.positions, static_cast<size_t>(rows) * sizeof(int64_t)));

    std::vector<__half> q(static_cast<size_t>(count) * kHeads * kHeadDim);
    std::vector<__half> k(static_cast<size_t>(rows) * kHeadDim);
    std::vector<int64_t> p(static_cast<size_t>(rows));
    // A deterministic ramp keeps the outputs comparable and non-degenerate without
    // any distribution claim; the timing does not depend on the values.
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = __float2half(0.001f * static_cast<float>((i * 2654435761u) % 1000) - 0.5f);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = __float2half(0.001f * static_cast<float>((i * 40503u) % 1000) - 0.5f);
    }
    for (int i = 0; i < rows; ++i) p[static_cast<size_t>(i)] = i;
    CHECK_HIP(hipMemcpy(b.q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(b.keys, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(b.positions, p.data(), p.size() * sizeof(int64_t),
                        hipMemcpyHostToDevice));
    return b;
}

void free_buffers(Buffers& b) {
    CHECK_HIP(hipFree(b.q));
    CHECK_HIP(hipFree(b.keys));
    CHECK_HIP(hipFree(b.out));
    CHECK_HIP(hipFree(b.positions));
    b = Buffers{};
}

// Queries sit at the tail of the key sequence, so every window is a real sub-range of
// the union the caller supplies.
CausalAttentionBlock make_block(const Buffers& b, int rows, int window) {
    CausalAttentionBlock block;
    block.keys = b.keys;
    block.values = b.keys;  // MLA: V = K
    block.positions = b.positions;
    block.rows = rows;
    block.key_stride = kHeadDim;
    block.value_stride = kHeadDim;
    block.window = window;
    return block;
}

double time_ms(const std::function<void()>& launch, hipStream_t stream) {
    // Warm up once so a first-call module load is not measured.
    launch();
    CHECK_HIP(hipStreamSynchronize(stream));
    hipEvent_t a = nullptr, c = nullptr;
    CHECK_HIP(hipEventCreate(&a));
    CHECK_HIP(hipEventCreate(&c));
    double best = 1e30;
    for (int r = 0; r < kRepetitions; ++r) {
        CHECK_HIP(hipEventRecord(a, stream));
        launch();
        CHECK_HIP(hipEventRecord(c, stream));
        CHECK_HIP(hipEventSynchronize(c));
        float ms = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&ms, a, c));
        best = std::min(best, static_cast<double>(ms));
    }
    CHECK_HIP(hipEventDestroy(a));
    CHECK_HIP(hipEventDestroy(c));
    return best;
}

bool agree(const std::vector<__half>& want, const std::vector<__half>& got) {
    double peak = 1e-9;
    for (__half v : want) peak = std::max(peak, std::fabs(static_cast<double>(__half2float(v))));
    for (__half v : got) peak = std::max(peak, std::fabs(static_cast<double>(__half2float(v))));
    double worst = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(__half2float(want[i])) -
                                          static_cast<double>(__half2float(got[i]))));
    }
    return worst / peak < 5e-3;
}

std::vector<__half> read_out(const Buffers& b, int count) {
    std::vector<__half> v(static_cast<size_t>(count) * kHeads * kHeadDim);
    CHECK_HIP(hipMemcpy(v.data(), b.out, v.size() * sizeof(__half),
                        hipMemcpyDeviceToHost));
    return v;
}

} // namespace

int main(int argc, char** argv) {
    aeon::core::select_compute_device(true);
    hipStream_t stream = nullptr;
    CHECK_HIP(hipStreamCreate(&stream));

    std::vector<int> query_counts = {64, 256};
    int rows = 297;  // ~ the swept-prefill union: 128 local + 169 compressed
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--rows" && i + 1 < argc) {
            rows = std::atoi(argv[++i]);
        } else {
            query_counts.push_back(std::atoi(argv[i]));
        }
    }

    std::printf("\nTiled causal attention A/B — heads=%d head_dim=%d window=%lld rows=%d\n",
                kHeads, kHeadDim, static_cast<long long>(kWindow), rows);
    std::printf("  key bytes read to DRAM floor: rows*head_dim*2 = %.2f MiB (re-read per query)\n\n",
                static_cast<double>(rows) * kHeadDim * 2 / (1024 * 1024));

    for (int count : query_counts) {
        if (count <= 0) continue;
        Buffers b = make_buffers(count, rows);
        const CausalAttentionBlock block = make_block(b, rows, static_cast<int>(kWindow));
        const CausalAttentionBlock empty;  // no block 1

        // Reference arm: the production split kernel at its default warp count.
        auto run_split4 = [&] {
            aeon::dispatch_causal_attention_split_fp16(
                b.q, kHeads * kHeadDim, block, empty, nullptr, 0,
                static_cast<int64_t>(rows - count), 1, b.out, kHeads * kHeadDim,
                count, kHeads, kHeadDim, nullptr, kScale, stream);
        };
        auto run_split2 = [&] {
            aeon::dispatch_causal_attention_split_fp16<2>(
                b.q, kHeads * kHeadDim, block, empty, nullptr, 0,
                static_cast<int64_t>(rows - count), 1, b.out, kHeads * kHeadDim,
                count, kHeads, kHeadDim, nullptr, kScale, stream);
        };
        auto run_split8 = [&] {
            aeon::dispatch_causal_attention_split_fp16<8>(
                b.q, kHeads * kHeadDim, block, empty, nullptr, 0,
                static_cast<int64_t>(rows - count), 1, b.out, kHeads * kHeadDim,
                count, kHeads, kHeadDim, nullptr, kScale, stream);
        };
        auto run_single = [&] {
            aeon::dispatch_causal_attention_fp16(
                b.q, kHeads * kHeadDim, block, empty,
                static_cast<int64_t>(rows - count), 1, b.out, kHeads * kHeadDim,
                count, kHeads, kHeadDim, nullptr, kScale, stream);
        };
        auto run_wmma = [&] {
            aeon::dispatch_causal_attention_wmma_qk_fp16(
                b.q, kHeads * kHeadDim, block, empty,
                static_cast<int64_t>(rows - count), 1, b.out, kHeads * kHeadDim,
                count, kHeads, kHeadDim, nullptr, kScale, stream);
        };

        run_split4();
        CHECK_HIP(hipStreamSynchronize(stream));
        const std::vector<__half> reference = read_out(b, count);

        struct Arm { const char* name; std::function<void()> run; };
        const Arm arms[] = {
            {"split warps=2", run_split2},
            {"split warps=4", run_split4},
            {"split warps=8", run_split8},
            {"single warp  ", run_single},
            {"wmma qk 1warp", run_wmma},
        };

        std::printf("  count=%d\n", count);
        for (const Arm& arm : arms) {
            const double ms = time_ms(arm.run, stream);
            CHECK_HIP(hipStreamSynchronize(stream));
            const std::vector<__half> got = read_out(b, count);
            const bool ok = agree(reference, got);
            std::printf("    %-14s %8.3f ms   %s\n", arm.name, ms, ok ? "agree" : "DISAGREE");
        }
        std::printf("\n");
        free_buffers(b);
    }

    CHECK_HIP(hipStreamDestroy(stream));
    return 0;
}
