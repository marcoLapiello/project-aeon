#pragma once

// -----------------------------------------------------------------------------
// G3 (weight format): the swizzled W4A16 weight feed for the grouped WMMA kernels.
//
// `platform/rdna3/moe_grouped_ffn.hpp` owns the MMA loop but knows nothing about how
// weights are stored. This file is the other half: given one expert's packed payload,
// it dequantizes a K-tile of it into the shared-memory slab the WMMA B fragment reads.
// It is the seam between the format and the architecture, and it is the only place the
// swizzled layout meets the WMMA unit.
//
// The layout contract it satisfies, and why the transposition happens here:
//
// The swizzled format puts one 32-wide K group of one output row in one lane's
// `uint4`, which is neither the A nor the B fragment order — so it cannot be fed to
// WMMA directly. But it is exactly the right shape to *dequantize* into an LDS tile
// laid out `[K][N]` row-major (K the slow axis, N the fast axis), which is the order
// `wmma_load_b_from_lds` reads. That transposition is this file's whole job, and it is
// why no converter change is needed and the artifact format is untouched.
//
// The class is templated on the swizzle geometry (`RPW` rows per block, `LPR` slices
// per row) rather than hardcoding it, so a different tile geometry is a different
// instantiation, and the kernel above carries none of it.
// -----------------------------------------------------------------------------

#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace aeon::kernel {

// Dequantizes one `[kBlock, nTile]` slab of a swizzled matrix into LDS in the
// `[K][N]` order the WMMA B fragment reads.
//
// Each thread owns one (row, group) pair, which is exactly one swizzled storage slot:
// a `uint4` of 32 nibbles and the fp16 scale that covers them. The nibbles are
// unpacked to their signed values (the format's zero point is 8) and scaled in fp16 —
// one rounding per weight, the cheaper of the two orderings available before the
// matrix multiply.
//
// Within a row block the eight groups of a row are contiguous, but consecutive
// threads here are consecutive *rows* of the same group, so the loads stride by 32
// slots (512 bytes). That is the access pattern the swizzled layout dictates; the slab
// is small and re-read every K step, so the cost is accepted rather than worked around
// with a second on-device transpose.
template <int RPW, int LPR>
__device__ __forceinline__ void grouped_dequant_w4a16_slab(
    const uint4* __restrict__ packed,
    const half* __restrict__ scale,
    half* __restrict__ tile,
    int tile_n_columns,
    int n_base,
    int k_base,
    int iterations
) {
    const int nn = threadIdx.x >> 1;          // output row within the N tile
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

// The feed policy the grouped kernels consume. `kMaxExperts` is how many experts one
// launch may address: the weight set is passed as a by-value pointer table
// (`SwizzledW13ExpertPtrs` / `SwizzledW2ExpertPtrs`), and that table's length is what
// bounds a dispatch. The executor batches a chunk's larger union into ranges of this
// size; it is an interface capacity, not an architecture or a format limit.
template <int RPW, int LPR>
struct SwizzledW4A16Feed {
    static_assert(RPW * LPR == 32, "RPW and LPR must describe one Wave32");

    static constexpr int kMaxExperts = kAeonSwizzledMaxExperts;
    static constexpr int kLpr = LPR;

    using GateUpWeights = SwizzledW13ExpertPtrs;
    using DownWeights = SwizzledW2ExpertPtrs;

    __device__ static void fill_gate_up_slab(
        const GateUpWeights& weights,
        int expert,
        half* gate_slab,
        half* up_slab,
        int n_tile,
        int n_base,
        int k_base,
        int iterations
    ) {
        grouped_dequant_w4a16_slab<RPW, LPR>(
            weights.w1[expert], weights.s1[expert], gate_slab, n_tile, n_base, k_base,
            iterations);
        grouped_dequant_w4a16_slab<RPW, LPR>(
            weights.w3[expert], weights.s3[expert], up_slab, n_tile, n_base, k_base,
            iterations);
    }

    __device__ static void fill_down_slab(
        const DownWeights& weights,
        int expert,
        half* down_slab,
        int n_tile,
        int n_base,
        int k_base,
        int iterations
    ) {
        grouped_dequant_w4a16_slab<RPW, LPR>(
            weights.w2[expert], weights.s2[expert], down_slab, n_tile, n_base, k_base,
            iterations);
    }
};

} // namespace aeon::kernel
