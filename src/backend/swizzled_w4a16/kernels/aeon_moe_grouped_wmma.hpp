#pragma once

// -----------------------------------------------------------------------------
// Grouped W4A16 expert GEMM on the Wave32 WMMA unit.
//
// The single-token expert kernels (`aeon_moe_fused_w13_swiglu`, ...) are GEMVs:
// one token per launch, so a weight byte fetched for one token cannot be reused
// by the next. Over a chunk of T tokens every expert weight is therefore read
// once per token that routes to it. This kernel instead sorts the chunk's
// (token, slot) pairs by expert — a device-side permutation supplied by the
// caller — and runs one GEMM per expert with the tokens of that expert as the M
// dimension, so each weight byte is read once no matter how many tokens want it.
//
// Shape and why:
//
//   * M tile = 16 tokens, one WMMA tile, matching the instruction exactly.
//   * N tile = 16 output rows per wave. The accumulator's lane encodes N and its
//     slot encodes M, so one wave produces one complete 16x16 output tile and
//     needs no cross-wave reduction.
//   * K moves in 64-wide blocks: two 32-wide quantization groups, which is what
//     lets one workgroup fill its LDS with exactly one int4 load per thread.
//
// The weights stay in the existing swizzled layout. That layout puts one 32-wide
// K group of one row in one lane's `uint4`, which is neither the A nor the B
// fragment order, so it cannot be fed to WMMA directly — but it is exactly the
// right shape to *dequantize* into LDS, which is the transposition this kernel
// does. No converter change is needed, and the artifact format is untouched.
//
// Partially filled M tiles (an expert whose token count is not a multiple of 16)
// are padded by repeating a resident token's row; the rows past the count are
// masked at the store. Giving them a separate GEMV path would need a second set
// of launch shapes for a case the permutation can absorb.
// -----------------------------------------------------------------------------

#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "platform/rdna3/wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace aeon::kernel {

// K per LDS fill: two quantization groups of 32, so `WAVES * 32` threads load
// exactly `WAVES * 16 * 2` int4 words — one per thread, no loop and no divisor.
inline constexpr int kGroupedWmmaKBlock = 64;

// Dequantizes one `[kBlock, nTile]` slab of a swizzled matrix into LDS in the
// `[K][N]` order the WMMA B fragment reads.
//
// Each thread owns one (row, group) pair, which is exactly one swizzled storage
// slot: a `uint4` of 32 nibbles and the fp16 scale that covers them. The nibbles
// are unpacked to their signed values (the format's zero point is 8) and scaled
// in fp16 — one rounding per weight, the cheaper of the two orderings available
// before the matrix multiply.
//
// Within a row block the eight groups of a row are contiguous, but consecutive
// threads here are consecutive *rows* of the same group, so the loads stride by
// 32 slots (512 bytes). That is the access pattern the swizzled layout dictates;
// the slab is small and re-read every K step, so the cost is accepted rather than
// worked around with a second on-device transpose.
template <int RPW, int LPR>
__device__ __forceinline__ void grouped_dequant_w13_slab(
    const uint4* __restrict__ packed,
    const half* __restrict__ scale,
    half* __restrict__ tile,
    int tile_n_columns,
    int n_base,
    int k_base,
    int iterations
) {
    const int nn = threadIdx.x >> 1;         // output row within the N tile
    const int group_offset = threadIdx.x & 1; // which of the two 32-wide groups
    const int n = n_base + nn;
    const int group = k_base / 32 + group_offset;

    const int block = n / RPW;
    const int row_in_block = n % RPW;
    const int slice = group % LPR;
    const int iteration = group / LPR;
    const int lane = row_in_block * LPR + slice;
    const size_t storage = static_cast<size_t>(block * iterations + iteration) * 32u +
                           static_cast<size_t>(lane);

    const uint4 words = packed[storage];
    const half2 scale_pair = __half2half2(scale[storage]);
    const uint32_t raw[4] = {words.x, words.y, words.z, words.w};

    half* destination = tile + static_cast<size_t>(group_offset * 32) * tile_n_columns + nn;
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            // The packer interleaves the eight nibbles of a word as the four
            // `half2` pairs of a 32-wide group read left to right: pair `p` of
            // word `w` covers k = 2*(4w + p) and 2*(4w + p) + 1.
            const half2 values = __hmul2(unpack2(raw[word], pair), scale_pair);
            const int k = (word * 4 + pair) * 2;
            destination[static_cast<size_t>(k) * tile_n_columns] = __low2half(values);
            destination[static_cast<size_t>(k + 1) * tile_n_columns] = __high2half(values);
        }
    }
}

// Gate/up projections for one expert group with the clamped SwiGLU fused into the
// epilogue, so the 2048-wide intermediate never round-trips through memory between
// the two projections and their activation.
//
// `expert_offsets` and `token_indices` are the permutation: expert e owns
// `expert_offsets[e] .. expert_offsets[e+1]` of `token_indices`, and each entry is
// a row of `activation`.
template <int WAVES, int RPW, int LPR>
__global__ __launch_bounds__(WAVES * 32)
void aeon_moe_grouped_w13_swiglu_wmma_kernel(
    const half* __restrict__ activation,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ token_indices,
    SwizzledW13ExpertPtrs weights,
    half* __restrict__ expert_hidden,
    int expert_count,
    int expert_hidden_tokens,
    int N,
    int K,
    float swiglu_limit
) {
    constexpr int kNTile = WAVES * 16;
    constexpr int kM = 16;

    const int expert = blockIdx.y;
    if (expert >= expert_count) {
        return;
    }

    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int lane_axis = aeon::rdna3::wmma_lane_axis(lane);
    const int lane_parity = aeon::rdna3::wmma_lane_parity(lane);
    // The LDS slab holds the whole workgroup's `kNTile` output rows and is filled
    // cooperatively; each wave then reads its own 16-row slice out of it.
    const int n_base = blockIdx.x * kNTile;
    const int n_wave = wave * kM;

    __shared__ half w1_slab[kGroupedWmmaKBlock * kNTile];
    __shared__ half w3_slab[kGroupedWmmaKBlock * kNTile];

    const int first = expert_offsets[expert];
    const int count = expert_offsets[expert + 1] - first;
    const int pad_row = token_indices[first];

    const int iterations = (K / 32) / LPR;

    for (int m_base = 0; m_base < count; m_base += kM) {
        // A padded row repeats `pad_row`, which is always in range, so the A
        // fragment load never touches memory past the group's token list.
        const int row_in_group = m_base + lane_axis;
        const int activation_row = row_in_group < count
            ? token_indices[first + row_in_group]
            : pad_row;
        const half* activation_row_ptr = activation + static_cast<size_t>(activation_row) * K;

        aeon::rdna3::f32_vec8 gate_accumulator = aeon::rdna3::wmma_zero_accumulator();
        aeon::rdna3::f32_vec8 up_accumulator = aeon::rdna3::wmma_zero_accumulator();

        for (int k_base = 0; k_base < K; k_base += kGroupedWmmaKBlock) {
            grouped_dequant_w13_slab<RPW, LPR>(
                weights.w1[expert], weights.s1[expert], w1_slab, kNTile, n_base, k_base,
                iterations);
            grouped_dequant_w13_slab<RPW, LPR>(
                weights.w3[expert], weights.s3[expert], w3_slab, kNTile, n_base, k_base,
                iterations);
            __syncthreads();

            #pragma unroll
            for (int k_sub = 0; k_sub < kGroupedWmmaKBlock / aeon::rdna3::kWmmaTileK;
                 ++k_sub) {
                const aeon::rdna3::f16_vec16 a_fragment = aeon::rdna3::wmma_load_a_row(
                    activation_row_ptr + k_base + k_sub * aeon::rdna3::kWmmaTileK);
                const aeon::rdna3::f16_vec16 gate_fragment =
                    aeon::rdna3::wmma_load_b_from_lds(
                        w1_slab + k_sub * aeon::rdna3::kWmmaTileK * kNTile, kNTile,
                        n_wave + lane_axis);
                const aeon::rdna3::f16_vec16 up_fragment =
                    aeon::rdna3::wmma_load_b_from_lds(
                        w3_slab + k_sub * aeon::rdna3::kWmmaTileK * kNTile, kNTile,
                        n_wave + lane_axis);
                gate_accumulator =
                    aeon::rdna3::wmma_mma(a_fragment, gate_fragment, gate_accumulator);
                up_accumulator =
                    aeon::rdna3::wmma_mma(a_fragment, up_fragment, up_accumulator);
            }
            __syncthreads();
        }

        const int n = n_base + n_wave + lane_axis;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int m = m_base + 2 * slot + lane_parity;
            if (m < count) {
                expert_hidden[(static_cast<size_t>(expert) * expert_hidden_tokens + m) * N +
                              n] =
                    __float2half(aeon_swiglu_clamped(gate_accumulator[slot],
                                                     up_accumulator[slot], swiglu_limit));
            }
        }
    }
}

template <int WAVES, int RPW, int LPR>
inline void dispatch_aeon_moe_grouped_w13_swiglu_wmma(
    const half* activation,
    const int* expert_offsets,
    const int* token_indices,
    const SwizzledW13ExpertPtrs& weights,
    half* expert_hidden,
    int expert_count,
    int expert_hidden_tokens,
    int N,
    int K,
    float swiglu_limit,
    hipStream_t stream = 0
) {
    static_assert(RPW * LPR == 32, "RPW and LPR must describe one Wave32");
    if (expert_count <= 0 || expert_count > kAeonSwizzledMaxExperts) {
        throw std::invalid_argument(
            "dispatch_aeon_moe_grouped_w13_swiglu_wmma: incompatible expert count");
    }
    if (N <= 0 || N % (WAVES * 16) != 0 || K <= 0 || K % kGroupedWmmaKBlock != 0 ||
        (K / 32) % LPR != 0) {
        throw std::invalid_argument(
            "dispatch_aeon_moe_grouped_w13_swiglu_wmma: incompatible N/K shape");
    }

    const dim3 block(WAVES * 32);
    const dim3 grid(N / (WAVES * 16), expert_count);
    aeon_moe_grouped_w13_swiglu_wmma_kernel<WAVES, RPW, LPR>
        <<<grid, block, 0, stream>>>(activation, expert_offsets, token_indices, weights,
                                     expert_hidden, expert_count, expert_hidden_tokens, N,
                                     K, swiglu_limit);
}

} // namespace aeon::kernel
