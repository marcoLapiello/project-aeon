#pragma once

// -----------------------------------------------------------------------------
// RDNA3 (gfx1100) Wave32 FP16 WMMA primitive.
//
// One instruction, `v_wmma_f32_16x16x16_f16_w32`, computes a 16x16x16
// `D = A x B + C` with fp16 operands and fp32 accumulate. It is the only
// matrix primitive in this file: everything format- or model-specific about
// *what* is fed to it lives elsewhere (G3 for weight decoding, G4 for the op).
//
// ## Fragment lane map
//
// The operand packing is not documented in one place, and a wrong reading of it
// produces a kernel that still runs and still returns numbers — it just computes
// `A x B^T`, or a transposed output, and only a comparison against a reference
// written from the definition of a matrix multiply catches it. The four facts,
// verified on silicon by `tests/test_rdna3_wmma_oracle.cpp`:
//
//   * A is row-major in M, K in the slot: lane t, slot i  ->  A[t & 15][i]
//   * B has K in the slot, N on the lane:  lane t, slot i  ->  B[i][t & 15]
//   * C has N on the lane, M split by the lane's high bit:
//         lane t, slot i  ->  C[2*i + (t >> 4)][t & 15]
//     so lanes 0..15 hold the even output rows and 16..31 the odd ones, each
//     lane holding all 8 of its column's rows.
//   * the two halves of the wave must carry **identical** input fragments (AMD's
//     "doubled" wave32 layout): row m and column n are read from lane
//     `(index) + 16*(m & 1)`, so a fragment built from `lane & 15` satisfies this
//     by construction, while feeding the halves different data makes the two row
//     parities disagree. Only half the operand is ever read for a given output
//     row; the other half covers the other parity.
//   * nothing about the K axis is packed: K is the fastest-varying index of both
//     operands, so a fragment is built by reading 16 consecutive halves of a row
//     (A) or of an LDS column (B).
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>

namespace aeon::rdna3 {

// Element counts are fixed by the instruction and differ between the two
// RDNA generations that ship it: gfx11 packs 16 halves per input fragment and
// 8 fp32 per accumulator, gfx12 packs 8 and 8. Only gfx11 is targeted here.
constexpr int kWmmaLaneCount = 32;
constexpr int kWmmaTileM = 16;
constexpr int kWmmaTileN = 16;
constexpr int kWmmaTileK = 16;

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    !defined(__HIP_DEVICE_COMPILE__)

using f16_vec16 = _Float16 __attribute__((ext_vector_type(16)));
using f32_vec8 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ f32_vec8 wmma_mma(f16_vec16 a, f16_vec16 b, f32_vec8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ f32_vec8 wmma_zero_accumulator() {
    return f32_vec8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}

// Lane indices named once so a caller cannot silently pick the other one. The
// low half indexes the M row of A, the N column of B, and the N column of C;
// the high bit only ever selects C's even/odd row parity.
__device__ __forceinline__ int wmma_lane_axis(int lane) { return lane & 15; }
__device__ __forceinline__ int wmma_lane_parity(int lane) { return lane >> 4; }

// A's fragment for one 16x16 tile: `m` selects the activation row, and the K
// axis is the slot. Both wave halves read the same `m`, which is what makes the
// duplicated layout a load-once-and-broadcast rather than a second read.
//
// `row` must point at 16 halves that are 16-byte aligned; the compiler lowers
// the copy to two `global_load_b128`. It is deliberately not a per-element loop:
// sixteen 2-byte loads serialise on the same address chain.
__device__ __forceinline__ f16_vec16 wmma_load_a_row(const __half* __restrict__ row) {
    f16_vec16 fragment;
    __builtin_memcpy(&fragment, row, sizeof(fragment));
    return fragment;
}

// B's fragment for one 16x16 tile, read out of an LDS tile laid out `[K][N]`
// row-major so that `tile[i][n]` is the (K=i, N=n) weight. Slot i is the K
// index and the lane axis is N, so every lane walks a different LDS row.
//
// The gather goes through a local `uint16_t` array and one 32-byte copy rather
// than element assignment: the element addresses of an `ext_vector_type` are not
// reliably usable as C pointers across compiler versions, so only the whole
// vector is memcpy'd. The intermediate stays in registers.
//
// `fp16_bits` exists because `uint16_t x = some_half` is a **numeric** conversion
// (`__half`'s integral `operator T()` yields the value, so 0.5 becomes 0), not a
// bit reinterpretation. Reading an fp16 tensor through the numeric path silently
// zeroes most of the fragment, and a kernel whose weights are all zero still runs
// and still returns finite numbers.
__device__ __forceinline__ uint16_t fp16_bits(__half value) {
    uint16_t bits = 0;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__device__ __forceinline__ f16_vec16 wmma_load_b_from_lds(
    const __half* __restrict__ tile, int n_columns_in_tile, int n_index) {
    uint16_t bits[kWmmaTileK];
    #pragma unroll
    for (int i = 0; i < kWmmaTileK; ++i) {
        bits[i] = fp16_bits(tile[static_cast<size_t>(i) * n_columns_in_tile + n_index]);
    }
    f16_vec16 fragment;
    __builtin_memcpy(&fragment, bits, sizeof(fragment));
    return fragment;
}

#endif  // gfx11 device pass, or any host pass

} // namespace aeon::rdna3
