// -----------------------------------------------------------------------------
// Feasibility gate: the routed W2 down-projection consumed as independently
// swizzled K-shards equals the whole-expert consumption, within fp16 tolerance,
// and the grouped-WMMA feed reads a shard exactly as it reads the whole.
//
// This is the verification behind the multi-GPU requirement that a tensor's
// partition structure may live in the stored artifact (see
// `plans-and-docs/specs-and-requirements/multi-gpu/MULTI_GPU_REQUIREMENTS.md`
// R3/R4/R5/R9). `W1`/`W3` are N-sharded, so a shard is a contiguous range of the
// whole swizzle's row-blocks and the property is immediate. `W2` is the only
// K-sharded expert tensor, and a K-shard changes the swizzle's *iteration stride*
// rather than only its block count, so it is the case that must be shown, not
// assumed.
//
// Two independent things are checked, against two different consumers:
//
//   A. The fused W2 contribution kernel (`dispatch_aeon_moe_fused_w2_contrib`)
//      run once per rank over its K-shard, with the per-shard fp32 contributions
//      summed, reproduces the whole-expert contribution. Graded against an
//      independent double-precision source-layout reference, so a shared helper
//      cannot make a wrong kernel agree with itself.
//
//   B. The grouped-WMMA feed (`grouped_dequant_w4a16_slab`) produces a
//      bit-identical `[K][N]` LDS slab whether it is given the whole matrix or the
//      corresponding shard. Bit-exact because the dequantization is per-weight and
//      deterministic: the feed must be *unperturbed* by the split, not merely
//      close.
//
// The shard extraction reads the *source* (unswizzled) layout and swizzles each
// shard from scratch — exactly what a converter would emit — rather than slicing
// an already-swizzled payload, so the test exercises the format proposal rather
// than a shortcut.
// -----------------------------------------------------------------------------

#include "platform/device.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_w4a16_swizzle.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "backend/swizzled_w4a16/kernels/swizzled_w4a16_feed.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

// The routed W2 geometry this checkpoint uses: [hidden=4096, intermediate=2048],
// swizzled with the W2 geometry (`kCfgW2`). `SHARDS` is the declared maximum
// decomposition — the multi-GPU format proposal fixes it at conversion time.
constexpr int N = 4096;
constexpr int K = 2048;
constexpr int GROUPS = K / 32;                       // 64
constexpr int WAVES = 8;
constexpr int RPW = 8;
constexpr int LPR = 4;
constexpr int ITERS_WHOLE = K / (LPR * 32);          // 16
constexpr int SHARDS = 8;
constexpr int K_SHARD = K / SHARDS;                  // 256
constexpr int GROUPS_SHARD = K_SHARD / 32;           // 8
constexpr int ITERS_SHARD = K_SHARD / (LPR * 32);    // 2

constexpr double kTolerance = 1e-3;                  // the project's fp16 ε

uint32_t make_source_word(std::size_t index) {
    uint32_t word = 0;
    for (int nibble = 0; nibble < 8; ++nibble) {
        const uint32_t value = static_cast<uint32_t>((index * 3 + nibble * 5 + 1) % 16);
        word |= value << (4 * nibble);
    }
    return word;
}

// One K-shard of a source-layout matrix, in the source layout. Shard `s` owns the
// global 32-wide groups `[s*GROUPS_SHARD, (s+1)*GROUPS_SHARD)` of every row; a
// group is four packed words and one scale, so the copy is word- and scale-granular.
void extract_shard_source(const std::vector<uint32_t>& whole_packed,
                          const std::vector<half>& whole_scale,
                          int shard,
                          std::vector<uint32_t>& shard_packed,
                          std::vector<half>& shard_scale) {
    const int words_per_row_whole = K / 8;
    const int words_per_row_shard = K_SHARD / 8;
    for (int row = 0; row < N; ++row) {
        for (int group = 0; group < GROUPS_SHARD; ++group) {
            const int global_group = shard * GROUPS_SHARD + group;
            for (int word = 0; word < 4; ++word) {
                shard_packed[static_cast<std::size_t>(row) * words_per_row_shard +
                             group * 4 + word] =
                    whole_packed[static_cast<std::size_t>(row) * words_per_row_whole +
                                 global_group * 4 + word];
            }
            shard_scale[static_cast<std::size_t>(row) * GROUPS_SHARD + group] =
                whole_scale[static_cast<std::size_t>(row) * GROUPS + global_group];
        }
    }
}

// Independent fp32/fp64 source-layout reference: dequantize straight from the
// pack-quantized words (zero point 8, per-32 scale) and dot with the activation.
// It shares no code with the swizzle or either kernel.
std::vector<double> reference_dot(const std::vector<uint32_t>& source_packed,
                                  const std::vector<half>& source_scale,
                                  const std::vector<half>& activation) {
    std::vector<double> out(N, 0.0);
    for (int row = 0; row < N; ++row) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            const uint32_t word =
                source_packed[static_cast<std::size_t>(row) * (K / 8) + k / 8];
            const int nibble = static_cast<int>((word >> ((k % 8) * 4)) & 0xFu);
            const double scale = __half2float(
                source_scale[static_cast<std::size_t>(row) * GROUPS + k / 32]);
            sum += static_cast<double>(__half2float(activation[k])) *
                   static_cast<double>(nibble - 8) * scale;
        }
        out[row] = sum;
    }
    return out;
}

// -----------------------------------------------------------------------------
// A. Whole-expert W2 contribution versus the sum of per-shard contributions.
// -----------------------------------------------------------------------------
bool verify_contribution_equivalence() {
    const std::size_t packed_words = static_cast<std::size_t>(N) * K / 8;
    const std::size_t scale_count = static_cast<std::size_t>(N) * GROUPS;
    const std::size_t shard_packed_words = static_cast<std::size_t>(N) * K_SHARD / 8;
    const std::size_t shard_scale_count = static_cast<std::size_t>(N) * GROUPS_SHARD;

    std::vector<half> activation(K);
    for (int k = 0; k < K; ++k) {
        activation[k] = __float2half(static_cast<float>(((k * 7) % 23) - 11) * 0.05f);
    }
    std::vector<uint32_t> source_packed(packed_words);
    std::vector<half> source_scale(scale_count);
    for (std::size_t index = 0; index < source_packed.size(); ++index) {
        source_packed[index] = make_source_word(index);
    }
    for (std::size_t index = 0; index < source_scale.size(); ++index) {
        source_scale[index] = __float2half(0.00390625f * static_cast<float>(1 + index % 29));
    }

    const std::vector<double> reference = reference_dot(source_packed, source_scale, activation);

    // Whole matrix: one swizzle over the full K.
    std::vector<uint32_t> whole_packed(packed_words);
    std::vector<half> whole_scale(scale_count);
    aeon::swizzle_w4a16(source_packed.data(), source_scale.data(),
                        whole_packed.data(), whole_scale.data(), N, K, aeon::kCfgW2);

    // Shards: extracted from the source layout and swizzled independently.
    std::vector<std::vector<uint32_t>> shard_packed(
        SHARDS, std::vector<uint32_t>(shard_packed_words));
    std::vector<std::vector<half>> shard_scale(SHARDS, std::vector<half>(shard_scale_count));
    for (int shard = 0; shard < SHARDS; ++shard) {
        std::vector<uint32_t> shard_source_packed(shard_packed_words);
        std::vector<half> shard_source_scale(shard_scale_count);
        extract_shard_source(source_packed, source_scale, shard,
                             shard_source_packed, shard_source_scale);
        aeon::swizzle_w4a16(shard_source_packed.data(), shard_source_scale.data(),
                            shard_packed[shard].data(), shard_scale[shard].data(),
                            N, K_SHARD, aeon::kCfgW2);
    }

    // --- device ---
    half* device_activation = nullptr;
    uint32_t* device_whole_packed = nullptr;
    half* device_whole_scale = nullptr;
    std::vector<uint32_t*> device_shard_packed(SHARDS, nullptr);
    std::vector<half*> device_shard_scale(SHARDS, nullptr);
    float* device_whole_contrib = nullptr;
    float* device_shard_contrib = nullptr;
    float* device_whole_topk = nullptr;
    float* device_shard_topk = nullptr;

    CHECK_HIP(hipMalloc(&device_activation, activation.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_whole_packed, whole_packed.size() * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_whole_scale, whole_scale.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_whole_contrib, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_shard_contrib, SHARDS * N * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_whole_topk, sizeof(float)));
    CHECK_HIP(hipMalloc(&device_shard_topk, SHARDS * sizeof(float)));

    CHECK_HIP(hipMemcpy(device_activation, activation.data(), activation.size() * sizeof(half),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_whole_packed, whole_packed.data(),
                        whole_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_whole_scale, whole_scale.data(), whole_scale.size() * sizeof(half),
                        hipMemcpyHostToDevice));

    const float one = 1.0f;
    CHECK_HIP(hipMemcpy(device_whole_topk, &one, sizeof(float), hipMemcpyHostToDevice));

    // The whole path: one expert, full K, weight 1.
    aeon::kernel::SwizzledW2ExpertPtrs whole_weights{};
    whole_weights.w2[0] = reinterpret_cast<const uint4*>(device_whole_packed);
    whole_weights.s2[0] = device_whole_scale;
    aeon::kernel::dispatch_aeon_moe_fused_w2_contrib<WAVES, RPW, LPR, ITERS_WHOLE>(
        device_activation, whole_weights, device_whole_topk, device_whole_contrib,
        /*expert_count=*/1, N, K);
    CHECK_HIP(hipGetLastError());

    // The shard path: one "expert" per shard, each owning its K-slice of the
    // activation. The per-expert hidden stride is `LPR*ITERS_SHARD*32 = K_SHARD`,
    // so shard `s` reads `activation[s*K_SHARD .. ]` — contiguous, as the proposal
    // requires. The kernel writes one fp32 slice per shard.
    aeon::kernel::SwizzledW2ExpertPtrs shard_weights{};
    std::vector<float> shard_topk(SHARDS, 1.0f);
    for (int shard = 0; shard < SHARDS; ++shard) {
        CHECK_HIP(hipMalloc(&device_shard_packed[shard],
                            shard_packed[shard].size() * sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&device_shard_scale[shard],
                            shard_scale[shard].size() * sizeof(half)));
        CHECK_HIP(hipMemcpy(device_shard_packed[shard], shard_packed[shard].data(),
                            shard_packed[shard].size() * sizeof(uint32_t),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_shard_scale[shard], shard_scale[shard].data(),
                            shard_scale[shard].size() * sizeof(half), hipMemcpyHostToDevice));
        shard_weights.w2[shard] = reinterpret_cast<const uint4*>(device_shard_packed[shard]);
        shard_weights.s2[shard] = device_shard_scale[shard];
    }
    CHECK_HIP(hipMemcpy(device_shard_topk, shard_topk.data(), SHARDS * sizeof(float),
                        hipMemcpyHostToDevice));
    aeon::kernel::dispatch_aeon_moe_fused_w2_contrib<WAVES, RPW, LPR, ITERS_SHARD>(
        device_activation, shard_weights, device_shard_topk, device_shard_contrib,
        /*expert_count=*/SHARDS, N, K_SHARD);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<float> whole_contrib(N);
    std::vector<float> shard_contrib(SHARDS * N);
    CHECK_HIP(hipMemcpy(whole_contrib.data(), device_whole_contrib, N * sizeof(float),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(shard_contrib.data(), device_shard_contrib, SHARDS * N * sizeof(float),
                        hipMemcpyDeviceToHost));

    double max_abs_whole_ref = 0.0, max_rel_whole_ref = 0.0;
    double max_abs_shard_ref = 0.0, max_rel_shard_ref = 0.0;
    double max_abs_shard_whole = 0.0, max_rel_shard_whole = 0.0;
    for (int row = 0; row < N; ++row) {
        double shard_sum = 0.0;
        for (int shard = 0; shard < SHARDS; ++shard) {
            shard_sum += static_cast<double>(shard_contrib[shard * N + row]);
        }
        const double whole = static_cast<double>(whole_contrib[row]);
        const double ref = reference[row];
        const double denominator = std::max(std::abs(ref), 1.0);

        max_abs_whole_ref = std::max(max_abs_whole_ref, std::abs(whole - ref));
        max_rel_whole_ref = std::max(max_rel_whole_ref, std::abs(whole - ref) / denominator);
        max_abs_shard_ref = std::max(max_abs_shard_ref, std::abs(shard_sum - ref));
        max_rel_shard_ref = std::max(max_rel_shard_ref, std::abs(shard_sum - ref) / denominator);
        max_abs_shard_whole = std::max(max_abs_shard_whole, std::abs(shard_sum - whole));
        max_rel_shard_whole = std::max(max_rel_shard_whole, std::abs(shard_sum - whole) / denominator);
    }

    std::cout << "[A] W2 whole vs independent reference:   max_abs=" << max_abs_whole_ref
              << " max_rel=" << max_rel_whole_ref << std::endl;
    std::cout << "[A] W2 shard-sum vs independent reference: max_abs=" << max_abs_shard_ref
              << " max_rel=" << max_rel_shard_ref << std::endl;
    std::cout << "[A] W2 shard-sum vs whole:                max_abs=" << max_abs_shard_whole
              << " max_rel=" << max_rel_shard_whole << std::endl;

    const bool pass = max_rel_whole_ref <= kTolerance &&
                      max_rel_shard_ref <= kTolerance &&
                      max_rel_shard_whole <= kTolerance;

    CHECK_HIP(hipFree(device_activation));
    CHECK_HIP(hipFree(device_whole_packed));
    CHECK_HIP(hipFree(device_whole_scale));
    CHECK_HIP(hipFree(device_whole_contrib));
    CHECK_HIP(hipFree(device_shard_contrib));
    CHECK_HIP(hipFree(device_whole_topk));
    CHECK_HIP(hipFree(device_shard_topk));
    for (int shard = 0; shard < SHARDS; ++shard) {
        CHECK_HIP(hipFree(device_shard_packed[shard]));
        CHECK_HIP(hipFree(device_shard_scale[shard]));
    }
    return pass;
}

// -----------------------------------------------------------------------------
// B. Grouped-WMMA feed: a shard's slab is bit-identical to the whole's slab.
// -----------------------------------------------------------------------------
template <int RPW_, int LPR_>
__global__ void feed_slab_kernel(
    const uint4* __restrict__ packed,
    const half* __restrict__ scale,
    half* __restrict__ tile,
    int tile_n_columns,
    int n_base,
    int k_base,
    int iterations
) {
    aeon::kernel::grouped_dequant_w4a16_slab<RPW_, LPR_>(
        packed, scale, tile, tile_n_columns, n_base, k_base, iterations);
}

bool verify_feed_unperturbed() {
    constexpr int TILE_N = 64;                 // columns of the N tile
    constexpr int SLAB_HALVES = 64 * TILE_N;   // [K=64][N=TILE_N]
    const std::size_t packed_words = static_cast<std::size_t>(N) * K / 8;
    const std::size_t scale_count = static_cast<std::size_t>(N) * GROUPS;
    const std::size_t shard_packed_words = static_cast<std::size_t>(N) * K_SHARD / 8;
    const std::size_t shard_scale_count = static_cast<std::size_t>(N) * GROUPS_SHARD;

    std::vector<uint32_t> source_packed(packed_words);
    std::vector<half> source_scale(scale_count);
    for (std::size_t index = 0; index < source_packed.size(); ++index) {
        source_packed[index] = make_source_word(index);
    }
    for (std::size_t index = 0; index < source_scale.size(); ++index) {
        source_scale[index] = __float2half(0.00390625f * static_cast<float>(1 + index % 29));
    }

    std::vector<uint32_t> whole_packed(packed_words);
    std::vector<half> whole_scale(scale_count);
    aeon::swizzle_w4a16(source_packed.data(), source_scale.data(),
                        whole_packed.data(), whole_scale.data(), N, K, aeon::kCfgW2);

    std::vector<std::vector<uint32_t>> shard_packed(
        SHARDS, std::vector<uint32_t>(shard_packed_words));
    std::vector<std::vector<half>> shard_scale(SHARDS, std::vector<half>(shard_scale_count));
    for (int shard = 0; shard < SHARDS; ++shard) {
        std::vector<uint32_t> shard_source_packed(shard_packed_words);
        std::vector<half> shard_source_scale(shard_scale_count);
        extract_shard_source(source_packed, source_scale, shard,
                             shard_source_packed, shard_source_scale);
        aeon::swizzle_w4a16(shard_source_packed.data(), shard_source_scale.data(),
                            shard_packed[shard].data(), shard_scale[shard].data(),
                            N, K_SHARD, aeon::kCfgW2);
    }

    uint32_t* device_whole_packed = nullptr;
    half* device_whole_scale = nullptr;
    std::vector<uint32_t*> device_shard_packed(SHARDS, nullptr);
    std::vector<half*> device_shard_scale(SHARDS, nullptr);
    half* device_tile_whole = nullptr;
    half* device_tile_shard = nullptr;

    CHECK_HIP(hipMalloc(&device_whole_packed, whole_packed.size() * sizeof(uint32_t)));
    CHECK_HIP(hipMalloc(&device_whole_scale, whole_scale.size() * sizeof(half)));
    CHECK_HIP(hipMemcpy(device_whole_packed, whole_packed.data(),
                        whole_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_whole_scale, whole_scale.data(),
                        whole_scale.size() * sizeof(half), hipMemcpyHostToDevice));
    for (int shard = 0; shard < SHARDS; ++shard) {
        CHECK_HIP(hipMalloc(&device_shard_packed[shard],
                            shard_packed[shard].size() * sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&device_shard_scale[shard],
                            shard_scale[shard].size() * sizeof(half)));
        CHECK_HIP(hipMemcpy(device_shard_packed[shard], shard_packed[shard].data(),
                            shard_packed[shard].size() * sizeof(uint32_t),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_shard_scale[shard], shard_scale[shard].data(),
                            shard_scale[shard].size() * sizeof(half), hipMemcpyHostToDevice));
    }
    CHECK_HIP(hipMalloc(&device_tile_whole, SLAB_HALVES * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_tile_shard, SLAB_HALVES * sizeof(half)));

    const dim3 block(2 * TILE_N);
    const dim3 grid(1);
    int mismatches = 0;
    std::vector<half> tile_whole(SLAB_HALVES);
    std::vector<half> tile_shard(SLAB_HALVES);

    // Every shard, every even local group pair (the feed consumes two 32-wide
    // groups at a time, so the local k_base must be even in groups), and two row
    // tiles to cover more than one row block.
    for (int n_base : {0, 192}) {
        for (int shard = 0; shard < SHARDS; ++shard) {
            for (int local_group = 0; local_group < GROUPS_SHARD; local_group += 2) {
                const int global_group = shard * GROUPS_SHARD + local_group;
                const int global_k_base = global_group * 32;
                const int local_k_base = local_group * 32;

                feed_slab_kernel<RPW, LPR><<<grid, block>>>(
                    reinterpret_cast<const uint4*>(device_whole_packed),
                    device_whole_scale, device_tile_whole, TILE_N, n_base,
                    global_k_base, ITERS_WHOLE);
                feed_slab_kernel<RPW, LPR><<<grid, block>>>(
                    reinterpret_cast<const uint4*>(device_shard_packed[shard]),
                    device_shard_scale[shard], device_tile_shard, TILE_N, n_base,
                    local_k_base, ITERS_SHARD);
                CHECK_HIP(hipGetLastError());
                CHECK_HIP(hipDeviceSynchronize());

                CHECK_HIP(hipMemcpy(tile_whole.data(), device_tile_whole,
                                    SLAB_HALVES * sizeof(half), hipMemcpyDeviceToHost));
                CHECK_HIP(hipMemcpy(tile_shard.data(), device_tile_shard,
                                    SLAB_HALVES * sizeof(half), hipMemcpyDeviceToHost));

                for (int index = 0; index < SLAB_HALVES; ++index) {
                    if (tile_whole[index] != tile_shard[index]) {
                        ++mismatches;
                    }
                }
            }
        }
    }

    if (mismatches == 0) {
        std::cout << "[B] Grouped feed slabs bit-identical across "
                  << SHARDS << " shards x " << (GROUPS_SHARD / 2) << " group pairs x 2 row tiles"
                  << std::endl;
    } else {
        std::cout << "[B] Grouped feed slab MISMATCHES: " << mismatches << std::endl;
    }

    CHECK_HIP(hipFree(device_whole_packed));
    CHECK_HIP(hipFree(device_whole_scale));
    CHECK_HIP(hipFree(device_tile_whole));
    CHECK_HIP(hipFree(device_tile_shard));
    for (int shard = 0; shard < SHARDS; ++shard) {
        CHECK_HIP(hipFree(device_shard_packed[shard]));
        CHECK_HIP(hipFree(device_shard_scale[shard]));
    }
    return mismatches == 0;
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);
    std::cout << "[Gate] Multi-GPU feasibility: routed W2 K-shard equivalence" << std::endl;

    const bool contribution_ok = verify_contribution_equivalence();
    const bool feed_ok = verify_feed_unperturbed();

    if (!contribution_ok || !feed_ok) {
        std::cout << "[FAIL] shard equivalence: contribution="
                  << (contribution_ok ? "pass" : "fail")
                  << " feed=" << (feed_ok ? "pass" : "fail") << std::endl;
        return 1;
    }
    std::cout << "[PASS] W2 K-shard consumption equals whole-expert consumption (fp16 ε), "
                 "and the grouped-WMMA feed is bit-unperturbed" << std::endl;
    return 0;
}
