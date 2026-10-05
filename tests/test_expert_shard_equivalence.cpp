// -----------------------------------------------------------------------------
// Feasibility gate: the full routed expert pair (W1/W3 -> clamped SwiGLU -> W2)
// computed under the proposed multi-GPU partition equals the whole-expert
// computation, within fp16 tolerance.
//
// This is the end-to-end companion to `test_w2_shard_equivalence`, which proved
// the down-projection K-shard in isolation. Here the whole routed body runs
// through the real grouped-WMMA kernels (`dispatch_aeon_moe_grouped_w13_swiglu_wmma`
// and `dispatch_aeon_moe_grouped_w2_wmma`) in two arms:
//
//   * **whole**  — one expert, W1/W3 `[2048, 4096]`, W2 `[4096, 2048]`.
//   * **shard**  — the same expert split into `SHARDS` pieces along the
//     intermediate dimension: rank `r` owns intermediate rows
//     `[r*256, (r+1)*256)` of W1/W3 (an **N-shard**) and intermediate columns
//     `[r*256, (r+1)*256)` of W2 (a **K-shard**). Each rank produces its own
//     256-wide activation and its own partial 4096-wide output; the partials sum
//     to the whole. This is the exact seam the format proposal creates: a rank's
//     W13 shard output *is* its W2 shard input, so the two shards must agree on
//     which intermediate indices they own.
//
// The subtlety this gate exists to catch is that the W1/W3 N-shard and the W2
// K-shard are cut along *different tensor axes* (rows vs columns) but must name
// the *same* intermediate range. A boundary off by one row would still produce a
// plausible-looking number — the down projection would just be pairing a rank's
// activation with another rank's weights — which is why the arms are compared to
// each other and to an independent source-layout reference.
//
// The reference decodes the pack-quantized source words directly (zero point 8,
// one fp16 scale per 32-wide group), sharing no code with the swizzle, the feed,
// or either kernel.
// -----------------------------------------------------------------------------

#include "platform/device.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "architecture/deepseek_v4/kernels/moe_grouped_dispatch.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_w4a16_swizzle.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
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
namespace reference = aeon::reference;

constexpr int kHidden = 4096;                          // W1/W3 columns, W2 rows
constexpr int kInter = 2048;                           // W1/W3 rows, W2 columns
constexpr int kShards = 8;                             // declared max decomposition
constexpr int kInterShard = kInter / kShards;          // 256
constexpr int kTokens = 20;                            // a partial M tile
constexpr double kLimit = 10.0;

constexpr double kFfnTol = 5e-3;                       // vs the independent reference
constexpr double kTightTol = 1e-3;                     // shard arm vs whole arm

uint32_t make_source_word(std::size_t index, uint32_t seed) {
    uint32_t word = 0;
    for (int nibble = 0; nibble < 8; ++nibble) {
        const uint32_t value = static_cast<uint32_t>(
            (index * 3 + nibble * 5 + seed) % 16);
        word |= value << (4 * nibble);
    }
    return word;
}

void fill_source(std::vector<uint32_t>& packed, std::vector<half>& scale, uint32_t seed) {
    for (std::size_t index = 0; index < packed.size(); ++index) {
        packed[index] = make_source_word(index, seed);
    }
    for (std::size_t index = 0; index < scale.size(); ++index) {
        scale[index] = __float2half(
            0.00390625f * static_cast<float>(1 + (index + seed) % 29));
    }
}

// Independent dequantization of one source-layout element.
inline double source_weight(const std::vector<uint32_t>& packed,
                            const std::vector<half>& scale,
                            int columns, int row, int column) {
    const uint32_t word = packed[static_cast<std::size_t>(row) * (columns / 8) + column / 8];
    const int nibble = static_cast<int>((word >> ((column % 8) * 4)) & 0xFu);
    const double s = __half2float(
        scale[static_cast<std::size_t>(row) * (columns / 32) + column / 32]);
    return static_cast<double>(nibble - 8) * s;
}

// W1/W3 N-shard: rows [shard*kInterShard, (shard+1)*kInterShard) of an
// `[kInter, kHidden]` matrix, all columns. A contiguous row range in the source.
void extract_w13_shard(const std::vector<uint32_t>& packed,
                       const std::vector<half>& scale,
                       int shard,
                       std::vector<uint32_t>& shard_packed,
                       std::vector<half>& shard_scale) {
    const std::size_t words_per_row = kHidden / 8;
    const std::size_t scales_per_row = kHidden / 32;
    const int row_base = shard * kInterShard;
    for (int row = 0; row < kInterShard; ++row) {
        const int source_row = row_base + row;
        for (std::size_t w = 0; w < words_per_row; ++w) {
            shard_packed[static_cast<std::size_t>(row) * words_per_row + w] =
                packed[static_cast<std::size_t>(source_row) * words_per_row + w];
        }
        for (std::size_t s = 0; s < scales_per_row; ++s) {
            shard_scale[static_cast<std::size_t>(row) * scales_per_row + s] =
                scale[static_cast<std::size_t>(source_row) * scales_per_row + s];
        }
    }
}

// W2 K-shard: columns [shard*kInterShard, (shard+1)*kInterShard) of an
// `[kHidden, kInter]` matrix, all rows. Within a row it is a contiguous run of
// 32-word / 8-scale groups.
void extract_w2_shard(const std::vector<uint32_t>& packed,
                      const std::vector<half>& scale,
                      int shard,
                      std::vector<uint32_t>& shard_packed,
                      std::vector<half>& shard_scale) {
    const std::size_t words_per_row = kInter / 8;              // 256
    const std::size_t scales_per_row = kInter / 32;            // 64
    const std::size_t shard_words_per_row = kInterShard / 8;   // 32
    const std::size_t shard_scales_per_row = kInterShard / 32; // 8
    const std::size_t word_base = shard * shard_words_per_row;
    const std::size_t scale_base = shard * shard_scales_per_row;
    for (int row = 0; row < kHidden; ++row) {
        for (std::size_t w = 0; w < shard_words_per_row; ++w) {
            shard_packed[static_cast<std::size_t>(row) * shard_words_per_row + w] =
                packed[static_cast<std::size_t>(row) * words_per_row + word_base + w];
        }
        for (std::size_t s = 0; s < shard_scales_per_row; ++s) {
            shard_scale[static_cast<std::size_t>(row) * shard_scales_per_row + s] =
                scale[static_cast<std::size_t>(row) * scales_per_row + scale_base + s];
        }
    }
}

struct Error {
    double max_abs{0.0};
    double max_rel{0.0};
};

Error compare(const std::vector<double>& want, const std::vector<double>& got) {
    Error error;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double difference = std::abs(want[i] - got[i]);
        const double denominator = std::max(std::abs(want[i]), 1.0);
        error.max_abs = std::max(error.max_abs, difference);
        error.max_rel = std::max(error.max_rel, difference / denominator);
    }
    return error;
}

bool report(const char* label, const Error& error, double tolerance) {
    const bool pass = std::isfinite(error.max_rel) && error.max_rel <= tolerance;
    std::printf("  %-46s max_abs=%.3e max_rel=%.3e  %s\n", label, error.max_abs,
                error.max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

// Multi-expert, permuted, topology-parameterized companion to the single-expert
// arm in `main`. It answers two questions that arm cannot:
//
//   * the grouped path's token permutation and lumpy per-expert token counts still
//     partition cleanly — the W13 N-shard of expert `e` must feed exactly *its own*
//     W2 K-shard, for the right tokens and in the right order; and
//   * a rank's block may be an **arbitrary contiguous group of shards**, not only
//     the finest one. At `TP = 2` a rank owns intermediate `[0, 1024)` as a single
//     swizzled matrix, so every supported degree `{1, 2, 4, 8}` is checked, not
//     just the 8-way case.
bool verify_multi_expert_topology() {
    // The two experts below span tokens 0..24, so this arm's token budget is 25 —
    // deliberately different from the single-expert arm's `kTokens`.
    constexpr int kTokens = 25;
    struct Matrix {
        std::vector<uint32_t> packed;
        std::vector<half> scale;
    };

    const std::vector<std::vector<int>> expert_tokens = {
        {17, 3, 9, 0, 14, 6, 19, 1, 11, 8, 16, 2, 13, 5, 18, 4, 10, 7, 15, 12},
        {24, 20, 22, 21, 23},
    };
    constexpr int kExperts = 2;

    const std::size_t w13_words = static_cast<std::size_t>(kInter) * kHidden / 8;
    const std::size_t w13_scales = static_cast<std::size_t>(kInter) * (kHidden / 32);
    const std::size_t w2_words = static_cast<std::size_t>(kHidden) * kInter / 8;
    const std::size_t w2_scales = static_cast<std::size_t>(kHidden) * (kInter / 32);

    std::vector<Matrix> source_w1(kExperts), source_w3(kExperts), source_w2(kExperts);
    std::vector<Matrix> whole_w1(kExperts), whole_w3(kExperts), whole_w2(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        source_w1[e] = {std::vector<uint32_t>(w13_words), std::vector<half>(w13_scales)};
        source_w3[e] = {std::vector<uint32_t>(w13_words), std::vector<half>(w13_scales)};
        source_w2[e] = {std::vector<uint32_t>(w2_words), std::vector<half>(w2_scales)};
        fill_source(source_w1[e].packed, source_w1[e].scale, static_cast<uint32_t>(10 + e * 7));
        fill_source(source_w3[e].packed, source_w3[e].scale, static_cast<uint32_t>(20 + e * 7));
        fill_source(source_w2[e].packed, source_w2[e].scale, static_cast<uint32_t>(30 + e * 7));
        whole_w1[e] = {std::vector<uint32_t>(w13_words), std::vector<half>(w13_scales)};
        whole_w3[e] = {std::vector<uint32_t>(w13_words), std::vector<half>(w13_scales)};
        whole_w2[e] = {std::vector<uint32_t>(w2_words), std::vector<half>(w2_scales)};
        aeon::swizzle_w4a16(source_w1[e].packed.data(), source_w1[e].scale.data(),
                            whole_w1[e].packed.data(), whole_w1[e].scale.data(),
                            kInter, kHidden, aeon::kCfgW13);
        aeon::swizzle_w4a16(source_w3[e].packed.data(), source_w3[e].scale.data(),
                            whole_w3[e].packed.data(), whole_w3[e].scale.data(),
                            kInter, kHidden, aeon::kCfgW13);
        aeon::swizzle_w4a16(source_w2[e].packed.data(), source_w2[e].scale.data(),
                            whole_w2[e].packed.data(), whole_w2[e].scale.data(),
                            kHidden, kInter, aeon::kCfgW2);
    }

    reference::Rng rng(0xF00DBEEFull);
    std::vector<half> activation(static_cast<std::size_t>(kTokens) * kHidden);
    for (int t = 0; t < kTokens; ++t) {
        for (int k = 0; k < kHidden; ++k) {
            activation[static_cast<std::size_t>(t) * kHidden + k] =
                __float2half(static_cast<float>(rng.symmetric(0.5)));
        }
    }
    // Uniform routing weight. The down-projection epilogue scales by
    // `draw_weights[position]` — the permutation *slot*, not the token's own weight —
    // so a non-uniform array would need a per-expert upload to stay consistent with
    // the token-indexed reference. The scaling path is exercised by the identity-order
    // single-expert arm; this arm isolates shard invariance.
    std::vector<float> draw_weights(kTokens, 1.0f);

    // Independent reference from the source words, in double.
    const std::size_t contrib_elems = static_cast<std::size_t>(kTokens) * kHidden;
    std::vector<double> reference_out(contrib_elems, 0.0);
    long clamped = 0;
    for (int e = 0; e < kExperts; ++e) {
        for (int token : expert_tokens[e]) {
            std::vector<double> x(kHidden);
            for (int k = 0; k < kHidden; ++k) {
                x[k] = static_cast<double>(
                    __half2float(activation[static_cast<std::size_t>(token) * kHidden + k]));
            }
            std::vector<double> hidden(kInter);
            for (int n = 0; n < kInter; ++n) {
                double gate = 0.0, up = 0.0;
                for (int k = 0; k < kHidden; ++k) {
                    gate += source_weight(source_w1[e].packed, source_w1[e].scale, kHidden, n, k) * x[k];
                    up += source_weight(source_w3[e].packed, source_w3[e].scale, kHidden, n, k) * x[k];
                }
                if (std::fabs(gate) > kLimit || std::fabs(up) > kLimit) ++clamped;
                const double act = reference::clamped_swiglu(gate, up, kLimit);
                hidden[n] = static_cast<double>(__half2float(__float2half(
                    static_cast<float>(act))));
            }
            for (int o = 0; o < kHidden; ++o) {
                double out = 0.0;
                for (int n = 0; n < kInter; ++n) {
                    out += source_weight(source_w2[e].packed, source_w2[e].scale, kInter, o, n) * hidden[n];
                }
                reference_out[static_cast<std::size_t>(token) * kHidden + o] =
                    static_cast<double>(draw_weights[token]) * out;
            }
        }
    }

    auto upload = [](const void* host, std::size_t bytes) {
        void* device = nullptr;
        CHECK_HIP(hipMalloc(&device, bytes));
        CHECK_HIP(hipMemcpy(device, host, bytes, hipMemcpyHostToDevice));
        return device;
    };
    auto extract_w13_block = [](const Matrix& whole, int row_base, int block_rows) {
        Matrix block;
        const int words_per_row = kHidden / 8;
        const int scales_per_row = kHidden / 32;
        block.packed.resize(static_cast<std::size_t>(block_rows) * words_per_row);
        block.scale.resize(static_cast<std::size_t>(block_rows) * scales_per_row);
        for (int r = 0; r < block_rows; ++r) {
            for (int w = 0; w < words_per_row; ++w) {
                block.packed[static_cast<std::size_t>(r) * words_per_row + w] =
                    whole.packed[static_cast<std::size_t>(row_base + r) * words_per_row + w];
            }
            for (int s = 0; s < scales_per_row; ++s) {
                block.scale[static_cast<std::size_t>(r) * scales_per_row + s] =
                    whole.scale[static_cast<std::size_t>(row_base + r) * scales_per_row + s];
            }
        }
        return block;
    };
    auto extract_w2_block = [](const Matrix& whole, int col_base, int block_cols) {
        Matrix block;
        const int words_per_row = kInter / 8;
        const int scales_per_row = kInter / 32;
        const int block_words = block_cols / 8;
        const int block_scales = block_cols / 32;
        block.packed.resize(static_cast<std::size_t>(kHidden) * block_words);
        block.scale.resize(static_cast<std::size_t>(kHidden) * block_scales);
        for (int r = 0; r < kHidden; ++r) {
            for (int w = 0; w < block_words; ++w) {
                block.packed[static_cast<std::size_t>(r) * block_words + w] =
                    whole.packed[static_cast<std::size_t>(r) * words_per_row + col_base / 8 + w];
            }
            for (int s = 0; s < block_scales; ++s) {
                block.scale[static_cast<std::size_t>(r) * block_scales + s] =
                    whole.scale[static_cast<std::size_t>(r) * scales_per_row + col_base / 32 + s];
            }
        }
        return block;
    };
    auto swizzle_13 = [](const Matrix& source, int rows, int columns) {
        Matrix swizzled{std::vector<uint32_t>(static_cast<std::size_t>(rows) * columns / 8),
                        std::vector<half>(static_cast<std::size_t>(rows) * (columns / 32))};
        aeon::swizzle_w4a16(source.packed.data(), source.scale.data(),
                            swizzled.packed.data(), swizzled.scale.data(), rows, columns,
                            aeon::kCfgW13);
        return swizzled;
    };
    auto swizzle_2 = [](const Matrix& source, int rows, int columns) {
        Matrix swizzled{std::vector<uint32_t>(static_cast<std::size_t>(rows) * columns / 8),
                        std::vector<half>(static_cast<std::size_t>(rows) * (columns / 32))};
        aeon::swizzle_w4a16(source.packed.data(), source.scale.data(),
                            swizzled.packed.data(), swizzled.scale.data(), rows, columns,
                            aeon::kCfgW2);
        return swizzled;
    };

    half* d_activation = static_cast<half*>(
        upload(activation.data(), activation.size() * sizeof(half)));
    float* d_draw_weights = static_cast<float*>(
        upload(draw_weights.data(), draw_weights.size() * sizeof(float)));

    // --- whole arm, per expert ---
    std::vector<double> whole_out(contrib_elems, 0.0);
    std::vector<std::vector<half>> whole_hidden(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        const int count = static_cast<int>(expert_tokens[e].size());
        uint32_t* d_w1p = static_cast<uint32_t*>(upload(whole_w1[e].packed.data(), whole_w1[e].packed.size() * sizeof(uint32_t)));
        half* d_w1s = static_cast<half*>(upload(whole_w1[e].scale.data(), whole_w1[e].scale.size() * sizeof(half)));
        uint32_t* d_w3p = static_cast<uint32_t*>(upload(whole_w3[e].packed.data(), whole_w3[e].packed.size() * sizeof(uint32_t)));
        half* d_w3s = static_cast<half*>(upload(whole_w3[e].scale.data(), whole_w3[e].scale.size() * sizeof(half)));
        uint32_t* d_w2p = static_cast<uint32_t*>(upload(whole_w2[e].packed.data(), whole_w2[e].packed.size() * sizeof(uint32_t)));
        half* d_w2s = static_cast<half*>(upload(whole_w2[e].scale.data(), whole_w2[e].scale.size() * sizeof(half)));
        int* d_offsets = static_cast<int*>(upload(std::array<int, 2>{0, count}.data(), 2 * sizeof(int)));
        int* d_tokens = static_cast<int*>(upload(expert_tokens[e].data(), count * sizeof(int)));

        kernel::SwizzledW13ExpertPtrs w13{};
        w13.w1[0] = reinterpret_cast<const uint4*>(d_w1p); w13.s1[0] = d_w1s;
        w13.w3[0] = reinterpret_cast<const uint4*>(d_w3p); w13.s3[0] = d_w3s;
        kernel::SwizzledW2ExpertPtrs w2{};
        w2.w2[0] = reinterpret_cast<const uint4*>(d_w2p); w2.s2[0] = d_w2s;

        half* d_hidden = nullptr;
        float* d_contrib = nullptr;
        CHECK_HIP(hipMalloc(&d_hidden, static_cast<std::size_t>(count) * kInter * sizeof(half)));
        CHECK_HIP(hipMemset(d_hidden, 0, static_cast<std::size_t>(count) * kInter * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_contrib, contrib_elems * sizeof(float)));
        CHECK_HIP(hipMemset(d_contrib, 0, contrib_elems * sizeof(float)));

        kernel::dispatch_aeon_moe_grouped_w13_swiglu_wmma<4, 4, 8, kernel::kMoeGroupedMTiles>(
            d_activation, d_offsets, d_tokens, w13, d_hidden, /*expert_count=*/1, count,
            kInter, kHidden, static_cast<float>(kLimit));
        kernel::dispatch_aeon_moe_grouped_w2_wmma<4, 8, 4, kernel::kMoeGroupedMTiles>(
            d_hidden, d_offsets, d_tokens, d_draw_weights, w2, d_contrib,
            /*expert_count=*/1, count, kHidden, kInter);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<float> host_contrib(contrib_elems);
        CHECK_HIP(hipMemcpy(host_contrib.data(), d_contrib, contrib_elems * sizeof(float),
                            hipMemcpyDeviceToHost));
        for (std::size_t i = 0; i < contrib_elems; ++i) whole_out[i] += host_contrib[i];

        whole_hidden[e].resize(static_cast<std::size_t>(count) * kInter);
        CHECK_HIP(hipMemcpy(whole_hidden[e].data(), d_hidden,
                            whole_hidden[e].size() * sizeof(half), hipMemcpyDeviceToHost));

        CHECK_HIP(hipFree(d_hidden)); CHECK_HIP(hipFree(d_contrib));
        CHECK_HIP(hipFree(d_offsets)); CHECK_HIP(hipFree(d_tokens));
        CHECK_HIP(hipFree(d_w1p)); CHECK_HIP(hipFree(d_w1s));
        CHECK_HIP(hipFree(d_w3p)); CHECK_HIP(hipFree(d_w3s));
        CHECK_HIP(hipFree(d_w2p)); CHECK_HIP(hipFree(d_w2s));
    }

    bool ok = true;
    ok &= report("multi-expert whole vs independent reference",
                 compare(reference_out, whole_out), kFfnTol);

    // --- shard arm, per topology degree ---
    const int degrees[4] = {1, 2, 4, 8};
    double worst_hidden_overall = 0.0;
    for (int degree : degrees) {
        const int block = kInter / degree;
        std::vector<double> sharded_out(contrib_elems, 0.0);
        for (int e = 0; e < kExperts; ++e) {
            const int count = static_cast<int>(expert_tokens[e].size());

            // Extract the rank block from the **source** layout, then swizzle it as
            // its own matrix — exactly what a partition-aware converter would emit.
            // (Slicing the already-swizzled whole would be a shortcut that never
            // exercises the format proposal.)
            std::vector<Matrix> rank_w1(degree), rank_w3(degree), rank_w2(degree);
            for (int r = 0; r < degree; ++r) {
                rank_w1[r] = swizzle_13(extract_w13_block(source_w1[e], r * block, block), block, kHidden);
                rank_w3[r] = swizzle_13(extract_w13_block(source_w3[e], r * block, block), block, kHidden);
                rank_w2[r] = swizzle_2(extract_w2_block(source_w2[e], r * block, block), kHidden, block);
            }

            std::vector<void*> allocations;
            kernel::SwizzledW13ExpertPtrs w13{};
            kernel::SwizzledW2ExpertPtrs w2{};
            for (int r = 0; r < degree; ++r) {
                uint32_t* p1 = static_cast<uint32_t*>(upload(rank_w1[r].packed.data(), rank_w1[r].packed.size() * sizeof(uint32_t)));
                half* s1 = static_cast<half*>(upload(rank_w1[r].scale.data(), rank_w1[r].scale.size() * sizeof(half)));
                uint32_t* p3 = static_cast<uint32_t*>(upload(rank_w3[r].packed.data(), rank_w3[r].packed.size() * sizeof(uint32_t)));
                half* s3 = static_cast<half*>(upload(rank_w3[r].scale.data(), rank_w3[r].scale.size() * sizeof(half)));
                uint32_t* p2 = static_cast<uint32_t*>(upload(rank_w2[r].packed.data(), rank_w2[r].packed.size() * sizeof(uint32_t)));
                half* s2 = static_cast<half*>(upload(rank_w2[r].scale.data(), rank_w2[r].scale.size() * sizeof(half)));
                allocations.insert(allocations.end(), {p1, s1, p3, s3, p2, s2});
                w13.w1[r] = reinterpret_cast<const uint4*>(p1); w13.s1[r] = s1;
                w13.w3[r] = reinterpret_cast<const uint4*>(p3); w13.s3[r] = s3;
                w2.w2[r] = reinterpret_cast<const uint4*>(p2); w2.s2[r] = s2;
            }

            std::vector<int> offsets(degree + 1);
            std::vector<int> tokens(static_cast<std::size_t>(degree) * count);
            for (int r = 0; r < degree; ++r) {
                offsets[r] = r * count;
                for (int t = 0; t < count; ++t) tokens[r * count + t] = expert_tokens[e][t];
            }
            offsets[degree] = degree * count;
            int* d_offsets = static_cast<int*>(upload(offsets.data(), offsets.size() * sizeof(int)));
            int* d_tokens = static_cast<int*>(upload(tokens.data(), tokens.size() * sizeof(int)));
            allocations.insert(allocations.end(), {d_offsets, d_tokens});

            const std::size_t hidden_elems = static_cast<std::size_t>(degree) * count * block;
            half* d_hidden = nullptr;
            CHECK_HIP(hipMalloc(&d_hidden, hidden_elems * sizeof(half)));
            CHECK_HIP(hipMemset(d_hidden, 0, hidden_elems * sizeof(half)));
            allocations.push_back(d_hidden);

            kernel::dispatch_aeon_moe_grouped_w13_swiglu_wmma<4, 4, 8, kernel::kMoeGroupedMTiles>(
                d_activation, d_offsets, d_tokens, w13, d_hidden, /*expert_count=*/degree,
                count, block, kHidden, static_cast<float>(kLimit));
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipDeviceSynchronize());

            // Diagnostic: the sharded W13 intermediate must equal the whole's, with
            // rank `r`, token `m`, local column `k` mapped to whole column `r*block+k`.
            if (degree > 1) {
                std::vector<half> shard_hidden(static_cast<std::size_t>(degree) * count * block);
                CHECK_HIP(hipMemcpy(shard_hidden.data(), d_hidden,
                                    shard_hidden.size() * sizeof(half), hipMemcpyDeviceToHost));
                double worst = 0.0;
                int worst_r = -1;
                for (int r = 0; r < degree; ++r) {
                    for (int m = 0; m < count; ++m) {
                        for (int k = 0; k < block; ++k) {
                            const double got = __half2float(shard_hidden[(static_cast<std::size_t>(r) * count + m) * block + k]);
                            const double want = __half2float(whole_hidden[e][static_cast<std::size_t>(m) * kInter + r * block + k]);
                            const double diff = std::abs(got - want);
                            if (diff > worst) { worst = diff; worst_r = r; }
                        }
                    }
                }
                worst_hidden_overall = std::max(worst_hidden_overall, worst);
                (void)worst_r;
            }

            for (int r = 0; r < degree; ++r) {
                int* d_one_offsets = static_cast<int*>(upload(std::array<int, 2>{0, count}.data(), 2 * sizeof(int)));
                int* d_one_tokens = static_cast<int*>(upload(expert_tokens[e].data(), count * sizeof(int)));
                float* d_partial = nullptr;
                CHECK_HIP(hipMalloc(&d_partial, contrib_elems * sizeof(float)));
                CHECK_HIP(hipMemset(d_partial, 0, contrib_elems * sizeof(float)));
                // The down projection dispatches one rank at a time, so the pointer
                // table must carry *that rank's* weights at index 0 — `expert_count=1`
                // reads `weights.w2[0]`.
                kernel::SwizzledW2ExpertPtrs one{};
                one.w2[0] = w2.w2[r];
                one.s2[0] = w2.s2[r];
                const half* rank_input = d_hidden + static_cast<std::size_t>(r) * count * block;
                kernel::dispatch_aeon_moe_grouped_w2_wmma<4, 8, 4, kernel::kMoeGroupedMTiles>(
                    rank_input, d_one_offsets, d_one_tokens, d_draw_weights, one, d_partial,
                    /*expert_count=*/1, count, kHidden, block);
                CHECK_HIP(hipGetLastError());
                CHECK_HIP(hipDeviceSynchronize());
                std::vector<float> host_partial(contrib_elems);
                CHECK_HIP(hipMemcpy(host_partial.data(), d_partial, contrib_elems * sizeof(float),
                                    hipMemcpyDeviceToHost));
                for (std::size_t i = 0; i < contrib_elems; ++i) sharded_out[i] += host_partial[i];
                CHECK_HIP(hipFree(d_one_offsets)); CHECK_HIP(hipFree(d_one_tokens));
                CHECK_HIP(hipFree(d_partial));
            }

            for (void* pointer : allocations) CHECK_HIP(hipFree(pointer));
        }

        char label[64];
        std::snprintf(label, sizeof(label), "TP%-2d sharded vs independent reference", degree);
        ok &= report(label, compare(reference_out, sharded_out), kFfnTol);
        std::snprintf(label, sizeof(label), "TP%-2d sharded vs whole", degree);
        ok &= report(label, compare(whole_out, sharded_out), kTightTol);
    }

    std::printf("  %-46s max_abs=%.3e  %s\n", "W13 intermediate bit-exact under sharding",
                worst_hidden_overall, worst_hidden_overall == 0.0 ? "PASS" : "FAIL");
    ok &= (worst_hidden_overall == 0.0);

    std::printf("  %-46s %s\n", "multi-expert clamp fired",
                clamped > 0 ? "PASS" : "FAIL (fixture too tame)");
    ok &= clamped > 0;

    CHECK_HIP(hipFree(d_activation));
    CHECK_HIP(hipFree(d_draw_weights));
    return ok;
}

} // namespace

int main() {
    std::cout << "[Gate] Multi-GPU feasibility: routed expert pair under sharding\n";
    aeon::core::select_compute_device(true);

    reference::Rng rng(0x5EED1234ull);

    // ------------------------------------------------------------------
    // Source weights: pack-quantized, row-major, zero point 8.
    // ------------------------------------------------------------------
    std::vector<uint32_t> w1_packed(static_cast<std::size_t>(kInter) * kHidden / 8);
    std::vector<half> w1_scale(static_cast<std::size_t>(kInter) * (kHidden / 32));
    std::vector<uint32_t> w3_packed(w1_packed.size());
    std::vector<half> w3_scale(w1_scale.size());
    std::vector<uint32_t> w2_packed(static_cast<std::size_t>(kHidden) * kInter / 8);
    std::vector<half> w2_scale(static_cast<std::size_t>(kHidden) * (kInter / 32));
    fill_source(w1_packed, w1_scale, 1);
    fill_source(w3_packed, w3_scale, 2);
    fill_source(w2_packed, w2_scale, 3);

    std::vector<half> activation(static_cast<std::size_t>(kTokens) * kHidden);
    for (int t = 0; t < kTokens; ++t) {
        for (int k = 0; k < kHidden; ++k) {
            activation[static_cast<std::size_t>(t) * kHidden + k] =
                __float2half(static_cast<float>(rng.symmetric(0.5)));
        }
    }
    std::vector<float> draw_weights(kTokens);
    for (int t = 0; t < kTokens; ++t) {
        draw_weights[t] = 0.3f + 0.05f * static_cast<float>(t % 4);
    }

    // ------------------------------------------------------------------
    // Independent reference, in double, from the source words.
    // ------------------------------------------------------------------
    std::vector<double> reference_output(static_cast<std::size_t>(kTokens) * kHidden, 0.0);
    long clamped = 0;
    for (int t = 0; t < kTokens; ++t) {
        std::vector<double> x(kHidden);
        for (int k = 0; k < kHidden; ++k) {
            x[k] = static_cast<double>(
                __half2float(activation[static_cast<std::size_t>(t) * kHidden + k]));
        }
        std::vector<double> hidden_fp16(kInter);
        for (int n = 0; n < kInter; ++n) {
            double gate = 0.0, up = 0.0;
            for (int k = 0; k < kHidden; ++k) {
                gate += source_weight(w1_packed, w1_scale, kHidden, n, k) * x[k];
                up += source_weight(w3_packed, w3_scale, kHidden, n, k) * x[k];
            }
            if (std::fabs(gate) > kLimit || std::fabs(up) > kLimit) ++clamped;
            // The device stores the intermediate as fp16, so the reference rounds
            // it too — otherwise the single rounding would be charged as error.
            const double act = reference::clamped_swiglu(gate, up, kLimit);
            hidden_fp16[n] = static_cast<double>(__half2float(__float2half(
                static_cast<float>(act))));
        }
        for (int o = 0; o < kHidden; ++o) {
            double out = 0.0;
            for (int n = 0; n < kInter; ++n) {
                out += source_weight(w2_packed, w2_scale, kInter, o, n) * hidden_fp16[n];
            }
            reference_output[static_cast<std::size_t>(t) * kHidden + o] =
                static_cast<double>(draw_weights[t]) * out;
        }
    }

    // ------------------------------------------------------------------
    // Build the swizzled device payloads: whole and per-shard.
    // ------------------------------------------------------------------
    std::vector<uint32_t> w1_full(w1_packed.size());
    std::vector<half> w1s_full(w1_scale.size());
    std::vector<uint32_t> w3_full(w3_packed.size());
    std::vector<half> w3s_full(w3_scale.size());
    std::vector<uint32_t> w2_full(w2_packed.size());
    std::vector<half> w2s_full(w2_scale.size());
    aeon::swizzle_w4a16(w1_packed.data(), w1_scale.data(), w1_full.data(), w1s_full.data(),
                        kInter, kHidden, aeon::kCfgW13);
    aeon::swizzle_w4a16(w3_packed.data(), w3_scale.data(), w3_full.data(), w3s_full.data(),
                        kInter, kHidden, aeon::kCfgW13);
    aeon::swizzle_w4a16(w2_packed.data(), w2_scale.data(), w2_full.data(), w2s_full.data(),
                        kHidden, kInter, aeon::kCfgW2);

    const std::size_t w13_shard_words = static_cast<std::size_t>(kInterShard) * kHidden / 8;
    const std::size_t w13_shard_scales = static_cast<std::size_t>(kInterShard) * (kHidden / 32);
    const std::size_t w2_shard_words = static_cast<std::size_t>(kHidden) * kInterShard / 8;
    const std::size_t w2_shard_scales = static_cast<std::size_t>(kHidden) * (kInterShard / 32);

    std::vector<std::vector<uint32_t>> w1_shard(kShards, std::vector<uint32_t>(w13_shard_words));
    std::vector<std::vector<half>> w1s_shard(kShards, std::vector<half>(w13_shard_scales));
    std::vector<std::vector<uint32_t>> w3_shard(kShards, std::vector<uint32_t>(w13_shard_words));
    std::vector<std::vector<half>> w3s_shard(kShards, std::vector<half>(w13_shard_scales));
    std::vector<std::vector<uint32_t>> w2_shard(kShards, std::vector<uint32_t>(w2_shard_words));
    std::vector<std::vector<half>> w2s_shard(kShards, std::vector<half>(w2_shard_scales));

    for (int shard = 0; shard < kShards; ++shard) {
        std::vector<uint32_t> w13_source_words(w13_shard_words);
        std::vector<half> w13_source_scales(w13_shard_scales);
        extract_w13_shard(w1_packed, w1_scale, shard, w13_source_words, w13_source_scales);
        aeon::swizzle_w4a16(w13_source_words.data(), w13_source_scales.data(),
                            w1_shard[shard].data(), w1s_shard[shard].data(),
                            kInterShard, kHidden, aeon::kCfgW13);
        extract_w13_shard(w3_packed, w3_scale, shard, w13_source_words, w13_source_scales);
        aeon::swizzle_w4a16(w13_source_words.data(), w13_source_scales.data(),
                            w3_shard[shard].data(), w3s_shard[shard].data(),
                            kInterShard, kHidden, aeon::kCfgW13);

        std::vector<uint32_t> w2_source_words(w2_shard_words);
        std::vector<half> w2_source_scales(w2_shard_scales);
        extract_w2_shard(w2_packed, w2_scale, shard, w2_source_words, w2_source_scales);
        aeon::swizzle_w4a16(w2_source_words.data(), w2_source_scales.data(),
                            w2_shard[shard].data(), w2s_shard[shard].data(),
                            kHidden, kInterShard, aeon::kCfgW2);
    }

    // ------------------------------------------------------------------
    // Device uploads.
    // ------------------------------------------------------------------
    auto upload = [](const void* host, std::size_t bytes) {
        void* device = nullptr;
        CHECK_HIP(hipMalloc(&device, bytes));
        CHECK_HIP(hipMemcpy(device, host, bytes, hipMemcpyHostToDevice));
        return device;
    };

    half* d_activation = static_cast<half*>(upload(activation.data(), activation.size() * sizeof(half)));
    int* d_expert_offsets = static_cast<int*>(upload(std::array<int, 2>{0, kTokens}.data(), 2 * sizeof(int)));
    std::vector<int> tokens(kTokens);
    for (int t = 0; t < kTokens; ++t) tokens[t] = t;
    int* d_token_indices = static_cast<int*>(upload(tokens.data(), tokens.size() * sizeof(int)));
    int* d_draw_indices = static_cast<int*>(upload(tokens.data(), tokens.size() * sizeof(int)));
    float* d_draw_weights = static_cast<float*>(upload(draw_weights.data(), draw_weights.size() * sizeof(float)));

    // Whole arm.
    uint32_t* d_w1 = static_cast<uint32_t*>(upload(w1_full.data(), w1_full.size() * sizeof(uint32_t)));
    half* d_w1s = static_cast<half*>(upload(w1s_full.data(), w1s_full.size() * sizeof(half)));
    uint32_t* d_w3 = static_cast<uint32_t*>(upload(w3_full.data(), w3_full.size() * sizeof(uint32_t)));
    half* d_w3s = static_cast<half*>(upload(w3s_full.data(), w3s_full.size() * sizeof(half)));
    uint32_t* d_w2 = static_cast<uint32_t*>(upload(w2_full.data(), w2_full.size() * sizeof(uint32_t)));
    half* d_w2s = static_cast<half*>(upload(w2s_full.data(), w2s_full.size() * sizeof(half)));

    kernel::SwizzledW13ExpertPtrs whole_w13{};
    whole_w13.w1[0] = reinterpret_cast<const uint4*>(d_w1);
    whole_w13.s1[0] = d_w1s;
    whole_w13.w3[0] = reinterpret_cast<const uint4*>(d_w3);
    whole_w13.s3[0] = d_w3s;
    kernel::SwizzledW2ExpertPtrs whole_w2{};
    whole_w2.w2[0] = reinterpret_cast<const uint4*>(d_w2);
    whole_w2.s2[0] = d_w2s;

    const std::size_t hidden_whole_elems = static_cast<std::size_t>(kTokens) * kInter;
    const std::size_t contrib_whole_elems = static_cast<std::size_t>(kTokens) * kHidden;
    half* d_hidden_whole = nullptr;
    float* d_contrib_whole = nullptr;
    CHECK_HIP(hipMalloc(&d_hidden_whole, hidden_whole_elems * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_contrib_whole, contrib_whole_elems * sizeof(float)));
    CHECK_HIP(hipMemset(d_hidden_whole, 0, hidden_whole_elems * sizeof(half)));
    CHECK_HIP(hipMemset(d_contrib_whole, 0, contrib_whole_elems * sizeof(float)));

    kernel::dispatch_aeon_moe_grouped_w13_swiglu_wmma<4, 4, 8, kernel::kMoeGroupedMTiles>(
        d_activation, d_expert_offsets, d_token_indices, whole_w13, d_hidden_whole,
        /*expert_count=*/1, kTokens, kInter, kHidden, static_cast<float>(kLimit));
    kernel::dispatch_aeon_moe_grouped_w2_wmma<4, 8, 4, kernel::kMoeGroupedMTiles>(
        d_hidden_whole, d_expert_offsets, d_draw_indices, d_draw_weights, whole_w2,
        d_contrib_whole, /*expert_count=*/1, kTokens, kHidden, kInter);
    CHECK_HIP(hipGetLastError());

    // ------------------------------------------------------------------
    // Shard arm. W13 runs once with `kShards` "experts" (one per shard), so
    // each rank writes its own `kInterShard`-wide activation; W2 then runs once
    // per shard into its own contribution buffer, because the down projection's
    // output has exactly one writer per element.
    // ------------------------------------------------------------------
    std::vector<int> shard_offsets(kShards + 1);
    std::vector<int> shard_tokens(kShards * kTokens);
    for (int shard = 0; shard < kShards; ++shard) {
        shard_offsets[shard] = shard * kTokens;
        for (int t = 0; t < kTokens; ++t) {
            shard_tokens[shard * kTokens + t] = t;
        }
    }
    shard_offsets[kShards] = kShards * kTokens;
    int* d_shard_offsets = static_cast<int*>(upload(shard_offsets.data(), shard_offsets.size() * sizeof(int)));
    int* d_shard_tokens = static_cast<int*>(upload(shard_tokens.data(), shard_tokens.size() * sizeof(int)));

    std::vector<uint32_t*> d_w1_shard(kShards), d_w3_shard(kShards), d_w2_shard(kShards);
    std::vector<half*> d_w1s_shard(kShards), d_w3s_shard(kShards), d_w2s_shard(kShards);
    kernel::SwizzledW13ExpertPtrs shard_w13{};
    kernel::SwizzledW2ExpertPtrs shard_w2{};
    for (int shard = 0; shard < kShards; ++shard) {
        d_w1_shard[shard] = static_cast<uint32_t*>(upload(w1_shard[shard].data(), w1_shard[shard].size() * sizeof(uint32_t)));
        d_w1s_shard[shard] = static_cast<half*>(upload(w1s_shard[shard].data(), w1s_shard[shard].size() * sizeof(half)));
        d_w3_shard[shard] = static_cast<uint32_t*>(upload(w3_shard[shard].data(), w3_shard[shard].size() * sizeof(uint32_t)));
        d_w3s_shard[shard] = static_cast<half*>(upload(w3s_shard[shard].data(), w3s_shard[shard].size() * sizeof(half)));
        d_w2_shard[shard] = static_cast<uint32_t*>(upload(w2_shard[shard].data(), w2_shard[shard].size() * sizeof(uint32_t)));
        d_w2s_shard[shard] = static_cast<half*>(upload(w2s_shard[shard].data(), w2s_shard[shard].size() * sizeof(half)));
        shard_w13.w1[shard] = reinterpret_cast<const uint4*>(d_w1_shard[shard]);
        shard_w13.s1[shard] = d_w1s_shard[shard];
        shard_w13.w3[shard] = reinterpret_cast<const uint4*>(d_w3_shard[shard]);
        shard_w13.s3[shard] = d_w3s_shard[shard];
        shard_w2.w2[shard] = reinterpret_cast<const uint4*>(d_w2_shard[shard]);
        shard_w2.s2[shard] = d_w2s_shard[shard];
    }

    const std::size_t shard_hidden_elems = static_cast<std::size_t>(kShards) * kTokens * kInterShard;
    half* d_hidden_shard = nullptr;
    CHECK_HIP(hipMalloc(&d_hidden_shard, shard_hidden_elems * sizeof(half)));
    CHECK_HIP(hipMemset(d_hidden_shard, 0, shard_hidden_elems * sizeof(half)));

    kernel::dispatch_aeon_moe_grouped_w13_swiglu_wmma<4, 4, 8, kernel::kMoeGroupedMTiles>(
        d_activation, d_shard_offsets, d_shard_tokens, shard_w13, d_hidden_shard,
        /*expert_count=*/kShards, kTokens, kInterShard, kHidden, static_cast<float>(kLimit));
    CHECK_HIP(hipGetLastError());

    std::vector<float> shard_contrib(static_cast<std::size_t>(kShards) * kTokens * kHidden, 0.0f);
    for (int shard = 0; shard < kShards; ++shard) {
        float* d_shard_contrib = nullptr;
        CHECK_HIP(hipMalloc(&d_shard_contrib, contrib_whole_elems * sizeof(float)));
        CHECK_HIP(hipMemset(d_shard_contrib, 0, contrib_whole_elems * sizeof(float)));
        const half* shard_input = d_hidden_shard + static_cast<std::size_t>(shard) * kTokens * kInterShard;
        kernel::SwizzledW2ExpertPtrs one{};
        one.w2[0] = shard_w2.w2[shard];
        one.s2[0] = shard_w2.s2[shard];
        kernel::dispatch_aeon_moe_grouped_w2_wmma<4, 8, 4, kernel::kMoeGroupedMTiles>(
            shard_input, d_expert_offsets, d_draw_indices, d_draw_weights, one,
            d_shard_contrib, /*expert_count=*/1, kTokens, kHidden, kInterShard);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(shard_contrib.data() + static_cast<std::size_t>(shard) * kTokens * kHidden,
                            d_shard_contrib, contrib_whole_elems * sizeof(float),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipFree(d_shard_contrib));
    }

    std::vector<float> whole_contrib(contrib_whole_elems);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(whole_contrib.data(), d_contrib_whole,
                        contrib_whole_elems * sizeof(float), hipMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // Compare.
    // ------------------------------------------------------------------
    std::vector<double> want(reference_output.begin(), reference_output.end());
    std::vector<double> got_whole(whole_contrib.begin(), whole_contrib.end());
    std::vector<double> got_shard(contrib_whole_elems, 0.0);
    for (std::size_t i = 0; i < contrib_whole_elems; ++i) {
        double sum = 0.0;
        for (int shard = 0; shard < kShards; ++shard) {
            sum += static_cast<double>(shard_contrib[static_cast<std::size_t>(shard) * contrib_whole_elems + i]);
        }
        got_shard[i] = sum;
    }

    bool ok = true;
    ok &= report("whole expert vs independent reference", compare(want, got_whole), kFfnTol);
    ok &= report("sharded expert vs independent reference", compare(want, got_shard), kFfnTol);
    ok &= report("sharded expert vs whole expert", compare(got_whole, got_shard), kTightTol);

    std::printf("  %-46s %s\n", "clamp fired in this fixture",
                clamped > 0 ? "PASS" : "FAIL (fixture too tame)");
    ok &= clamped > 0;

    ok &= verify_multi_expert_topology();

    if (!ok) {
        std::cout << "[FAIL] routed expert pair is not shard-invariant" << std::endl;
        return 1;
    }
    std::cout << "[PASS] The routed expert pair is identical under the "
              << kShards << "-way intermediate partition" << std::endl;
    return 0;
}
