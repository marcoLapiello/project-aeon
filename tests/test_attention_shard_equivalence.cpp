// -----------------------------------------------------------------------------
// Feasibility gate: attention and KV placement under tensor partitioning (R8).
//
// `MULTI_GPU_REQUIREMENTS.md` R8 states that attention state follows the attention
// class: a state that partitions by head is sharded across ranks, while a shared
// latent that cannot be split by head — DeepSeek-V4's compressed MQA, whose 512-dim
// key/value latent is common to all 64 heads — is **replicated**. This gate proves
// both halves through the real attention kernels.
//
// The structure that makes it work, and the two properties that make it exact:
//
//   * `v4_sliding_window_attn_wave32_kernel` is indexed by head: head `h` reads its
//     own `q[token, h, :]` and the **shared** `k[j, :]`. There is no cross-head term,
//     so a head block is a pure slice of the whole output — and every rank reads the
//     same `k`, which is what "replicated latent" means operationally.
//
//   * `v4_grouped_wo_a_wave32_kernel` couples heads only *within* a group of 8
//     (`DSV4_HEADS_PER_GROUP`). A rank block must therefore be group-aligned; the
//     gate checks the degrees that are (`{2,4,8}` → 4/2/1 groups per rank) and shows
//     the group's `z` columns are exactly the whole's slice.
//
// The final `wo_b` projection mixes all groups and is the row-parallel reduction
// point: the gate confirms the per-rank partials sum to the whole within fp tolerance
// (R9's cross-rank shape), computed independently in double.
// -----------------------------------------------------------------------------

#include "platform/device.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention_kernels.hpp"
#include "architecture/deepseek_v4/kernels/v4_grouped_wo.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_HIP(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

namespace {

namespace kernel = aeon::kernel;

constexpr int kTokens = 16;                                   // a full sliding window
constexpr int kHeads = static_cast<int>(kernel::DSV4_NUM_HEADS);          // 64
constexpr int kHeadDim = static_cast<int>(kernel::DSV4_HEAD_DIM);         // 512
constexpr int kGroups = static_cast<int>(kernel::DSV4_O_GROUPS);          // 8
constexpr int kGroupDim = static_cast<int>(kernel::DSV4_GROUP_HEADS_DIM); // 4096
constexpr int kOLora = static_cast<int>(kernel::DSV4_O_LORA_RANK);        // 1024
constexpr int kTotalLora = static_cast<int>(kernel::DSV4_TOTAL_O_LORA_DIM); // 8192
constexpr int kHidden = 4096;

uint32_t rng_state = 0x1234ABCDu;
float next_float() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (static_cast<float>((rng_state >> 8) & 0xFFFFu) / 32768.0f) - 1.0f;
}

template <typename T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    CHECK_HIP(hipMalloc(&device, host.size() * sizeof(T)));
    CHECK_HIP(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice));
    return device;
}

bool report(const char* label, double max_abs, double tolerance) {
    const bool pass = std::isfinite(max_abs) && max_abs <= tolerance;
    std::printf("  %-52s max_abs=%.3e  %s\n", label, max_abs, pass ? "PASS" : "FAIL");
    return pass;
}

} // namespace

int main() {
    std::cout << "[Gate] Multi-GPU feasibility: attention / KV placement under sharding\n";
    aeon::core::select_compute_device(true);

    std::vector<half> q(static_cast<std::size_t>(kTokens) * kHeads * kHeadDim);
    std::vector<half> k(static_cast<std::size_t>(kTokens) * kHeadDim);
    std::vector<float> sink(kHeads);
    for (auto& v : q) v = __float2half(next_float() * 0.3f);
    for (auto& v : k) v = __float2half(next_float() * 0.3f);
    for (auto& v : sink) v = next_float() * 0.5f;

    std::vector<half> wo_a(static_cast<std::size_t>(kGroups) * kOLora * kGroupDim);
    for (auto& v : wo_a) v = __float2half(next_float() * 0.05f);

    half* d_q = upload(q);
    half* d_k = upload(k);
    float* d_sink = upload(sink);
    half* d_wo_a = upload(wo_a);

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    // --- whole: attention over all heads, then wo_a over all groups ---
    half* d_attn_whole = nullptr;
    CHECK_HIP(hipMalloc(&d_attn_whole, q.size() * sizeof(half)));
    CHECK_HIP(hipMemset(d_attn_whole, 0, q.size() * sizeof(half)));
    {
        const dim3 grid(kHeads, kTokens);
        const dim3 block(32);
        kernel::v4_sliding_window_attn_wave32_kernel<<<grid, block>>>(
            d_q, d_k, d_sink, d_attn_whole, kTokens, kTokens, scale);
    }

    half* d_z_whole = nullptr;
    CHECK_HIP(hipMalloc(&d_z_whole, static_cast<std::size_t>(kTokens) * kTotalLora * sizeof(half)));
    CHECK_HIP(hipMemset(d_z_whole, 0, static_cast<std::size_t>(kTokens) * kTotalLora * sizeof(half)));
    {
        const dim3 grid(kOLora, kGroups, kTokens);
        const dim3 block(32);
        kernel::v4_grouped_wo_a_wave32_kernel<<<grid, block>>>(
            d_attn_whole, d_wo_a, d_z_whole, kTokens);
    }
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<half> attn_whole(q.size());
    std::vector<half> z_whole(static_cast<std::size_t>(kTokens) * kTotalLora);
    CHECK_HIP(hipMemcpy(attn_whole.data(), d_attn_whole, attn_whole.size() * sizeof(half),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(z_whole.data(), d_z_whole, z_whole.size() * sizeof(half),
                        hipMemcpyDeviceToHost));

    bool ok = true;

    // ------------------------------------------------------------------
    // A. Attention partitions by head: a rank's head block is exactly the
    //    whole's slice, with every rank reading the same (replicated) latent.
    // ------------------------------------------------------------------
    for (int degree : {2, 4, 8}) {
        const int heads_per_rank = kHeads / degree;
        double worst_attn = 0.0;
        for (int rank = 0; rank < degree; ++rank) {
            const int head_base = rank * heads_per_rank;
            half* d_rank_attn = nullptr;
            CHECK_HIP(hipMalloc(&d_rank_attn, q.size() * sizeof(half)));
            CHECK_HIP(hipMemset(d_rank_attn, 0, q.size() * sizeof(half)));
            const dim3 grid(heads_per_rank, kTokens);
            const dim3 block(32);
            // Offset the base pointers by the head range; the token stride is the
            // full-layout stride, so `head` indexes the rank's own slice. The
            // **per-head attention sink shards with the head** too — a rank owns its
            // heads' sink entries, so the sink pointer advances with the block.
            kernel::v4_sliding_window_attn_wave32_kernel<<<grid, block>>>(
                d_q + static_cast<std::size_t>(head_base) * kHeadDim, d_k,
                d_sink + head_base,
                d_rank_attn + static_cast<std::size_t>(head_base) * kHeadDim,
                kTokens, kTokens, scale);
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipDeviceSynchronize());
            std::vector<half> rank_attn(q.size());
            CHECK_HIP(hipMemcpy(rank_attn.data(), d_rank_attn, rank_attn.size() * sizeof(half),
                                hipMemcpyDeviceToHost));
            for (int t = 0; t < kTokens; ++t) {
                for (int h = 0; h < heads_per_rank; ++h) {
                    for (int d = 0; d < kHeadDim; ++d) {
                        const std::size_t rank_index =
                            (static_cast<std::size_t>(t) * kHeads + head_base + h) * kHeadDim + d;
                        worst_attn = std::max(worst_attn, static_cast<double>(std::abs(
                            __half2float(rank_attn[rank_index]) -
                            __half2float(attn_whole[rank_index]))));
                    }
                }
            }
            CHECK_HIP(hipFree(d_rank_attn));
        }
        char label[72];
        std::snprintf(label, sizeof(label),
                      "TP%-2d attention head block == whole slice", degree);
        ok &= report(label, worst_attn, 0.0);
    }

    // ------------------------------------------------------------------
    // C. Grouped W_o_a partitions by group: a rank's z columns are the whole's
    //    slice. The rank block owns the groups covered by its head block.
    // ------------------------------------------------------------------
    for (int degree : {2, 4, 8}) {
        const int groups_per_rank = kGroups / degree;
        double worst_z = 0.0;
        for (int rank = 0; rank < degree; ++rank) {
            const int group_base = rank * groups_per_rank;
            half* d_rank_z = nullptr;
            CHECK_HIP(hipMalloc(&d_rank_z,
                                static_cast<std::size_t>(kTokens) * kTotalLora * sizeof(half)));
            CHECK_HIP(hipMemset(d_rank_z, 0,
                                static_cast<std::size_t>(kTokens) * kTotalLora * sizeof(half)));
            const dim3 grid(kOLora, groups_per_rank, kTokens);
            const dim3 block(32);
            kernel::v4_grouped_wo_a_wave32_kernel<<<grid, block>>>(
                d_attn_whole + static_cast<std::size_t>(group_base) * kGroupDim,
                d_wo_a + static_cast<std::size_t>(group_base) * kOLora * kGroupDim,
                d_rank_z + static_cast<std::size_t>(group_base) * kOLora,
                kTokens);
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipDeviceSynchronize());
            std::vector<half> rank_z(static_cast<std::size_t>(kTokens) * kTotalLora);
            CHECK_HIP(hipMemcpy(rank_z.data(), d_rank_z, rank_z.size() * sizeof(half),
                                hipMemcpyDeviceToHost));
            for (int t = 0; t < kTokens; ++t) {
                for (int r = 0; r < groups_per_rank * kOLora; ++r) {
                    const std::size_t index =
                        static_cast<std::size_t>(t) * kTotalLora + group_base * kOLora + r;
                    worst_z = std::max(worst_z, static_cast<double>(std::abs(
                        __half2float(rank_z[index]) - __half2float(z_whole[index]))));
                }
            }
            CHECK_HIP(hipFree(d_rank_z));
        }
        char label[72];
        std::snprintf(label, sizeof(label),
                      "TP%-2d grouped W_o_a z block == whole slice", degree);
        ok &= report(label, worst_z, 0.0);
    }

    // ------------------------------------------------------------------
    // D. The group-mixing W_o_b is the reduction point: per-rank partials sum to
    //    the whole. Computed independently in double from the fp16 z, so the only
    //    difference is fp summation, not the kernel.
    // ------------------------------------------------------------------
    std::vector<half> wo_b(static_cast<std::size_t>(kHidden) * kTotalLora);
    for (auto& v : wo_b) v = __float2half(next_float() * 0.05f);

    for (int degree : {2, 4, 8}) {
        const int lora_per_rank = kTotalLora / degree;
        double worst_b = 0.0;
        for (int t = 0; t < kTokens; ++t) {
            for (int o = 0; o < kHidden; ++o) {
                double whole = 0.0;
                for (int r = 0; r < kTotalLora; ++r) {
                    whole += static_cast<double>(__half2float(wo_b[static_cast<std::size_t>(o) * kTotalLora + r])) *
                             static_cast<double>(__half2float(z_whole[static_cast<std::size_t>(t) * kTotalLora + r]));
                }
                double partial = 0.0;
                for (int rank = 0; rank < degree; ++rank) {
                    for (int r = 0; r < lora_per_rank; ++r) {
                        const std::size_t column = static_cast<std::size_t>(rank) * lora_per_rank + r;
                        partial += static_cast<double>(__half2float(wo_b[static_cast<std::size_t>(o) * kTotalLora + column])) *
                                   static_cast<double>(__half2float(z_whole[static_cast<std::size_t>(t) * kTotalLora + column]));
                    }
                }
                worst_b = std::max(worst_b, std::abs(whole - partial));
            }
        }
        char label[72];
        std::snprintf(label, sizeof(label),
                      "TP%-2d W_o_b partials == whole (fp)", degree);
        ok &= report(label, worst_b, 1e-2);
    }

    if (!ok) {
        std::cout << "[FAIL] attention / KV placement is not shard-invariant" << std::endl;
        return 1;
    }
    std::cout << "[PASS] Attention shards by head, the KV latent replicates, and the "
                 "group mixing reduces exactly" << std::endl;
    return 0;
}
