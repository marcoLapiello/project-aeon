#pragma once

// -----------------------------------------------------------------------------
// G2 (RDNA3): dense fp16 GEMM, `Y[T, N] = X[T, K] . W[N, K]^T`, on the Wave32 WMMA.
//
// `W` is the checkpoint's `[out, in]` row-major layout, so the B fragment of the
// instruction (K in the slot, N on the lane) is 16 consecutive halves of one weight
// row: both operands are read straight from global memory, with no LDS staging and no
// transpose. A wave owns 16 output columns and `MTILES` token tiles, so each weight
// fragment is loaded once and multiplied against every token tile it serves.
//
// Ragged `T` and `N` are handled by clamping the load row and masking the store; `K`
// must be a multiple of 16 and the row pitches a multiple of 8 halves (16-byte loads).
// Accumulation is fp32 and the result is rounded to fp16 once, like the GEMV it
// replaces, so the two differ only in summation order.
// -----------------------------------------------------------------------------

#include "platform/rdna3/wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace aeon::rdna3 {

constexpr int kDenseGemmWaves = 4;

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    !defined(__HIP_DEVICE_COMPILE__)

template <int MTILES>
__global__ void __launch_bounds__(kDenseGemmWaves * kWmmaLaneCount)
dense_gemm_fp16_wmma_kernel(
    const __half* __restrict__ x,   // [tokens, x_stride]
    const __half* __restrict__ w,   // [out_dim, in_dim]
    __half* __restrict__ y,         // [tokens, y_stride]
    int tokens, int in_dim, int out_dim, int x_stride, int y_stride) {
    const int lane = threadIdx.x & (kWmmaLaneCount - 1);
    const int wave = threadIdx.x / kWmmaLaneCount;
    const int axis = wmma_lane_axis(lane);
    const int n_base = (blockIdx.x * kDenseGemmWaves + wave) * kWmmaTileN;
    if (n_base >= out_dim) return;

    const int m_base = blockIdx.y * MTILES * kWmmaTileM;
    const __half* w_row = w + static_cast<size_t>(min(n_base + axis, out_dim - 1)) * in_dim;
    const __half* x_rows[MTILES];
    f32_vec8 acc[MTILES];
    #pragma unroll
    for (int t = 0; t < MTILES; ++t) {
        const int row = min(m_base + t * kWmmaTileM + axis, tokens - 1);
        x_rows[t] = x + static_cast<size_t>(row) * x_stride;
        acc[t] = wmma_zero_accumulator();
    }

    for (int k = 0; k < in_dim; k += kWmmaTileK) {
        const f16_vec16 b = wmma_load_a_row(w_row + k);
        #pragma unroll
        for (int t = 0; t < MTILES; ++t) {
            acc[t] = wmma_mma(wmma_load_a_row(x_rows[t] + k), b, acc[t]);
        }
    }

    const int col = n_base + axis;
    const int parity = wmma_lane_parity(lane);
    if (col >= out_dim) return;
    #pragma unroll
    for (int t = 0; t < MTILES; ++t) {
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int row = m_base + t * kWmmaTileM + 2 * slot + parity;
            if (row < tokens) {
                y[static_cast<size_t>(row) * y_stride + col] = __float2half(acc[t][slot]);
            }
        }
    }
}

#endif  // gfx11 device pass, or any host pass

// Token tiles per wave: enough to amortize each weight fragment across the chunk
// without leaving the grid short of waves for a small `tokens`.
inline int dense_gemm_token_tiles(int tokens) {
    return tokens <= 16 ? 1 : (tokens <= 32 ? 2 : 4);
}

inline void dispatch_dense_gemm_fp16(
    const __half* x, const __half* w, __half* y,
    int tokens, int in_dim, int out_dim, int x_stride, int y_stride, hipStream_t stream) {
    if (tokens <= 0) return;
    if (in_dim % kWmmaTileK != 0 || x_stride % 8 != 0 || x_stride < in_dim ||
        y_stride < out_dim) {
        throw std::invalid_argument("dispatch_dense_gemm_fp16: incompatible shape or pitch");
    }
    const int mtiles = dense_gemm_token_tiles(tokens);
    const int cols_per_block = kDenseGemmWaves * kWmmaTileN;
    const dim3 grid((out_dim + cols_per_block - 1) / cols_per_block,
                    (tokens + mtiles * kWmmaTileM - 1) / (mtiles * kWmmaTileM));
    const dim3 block(kDenseGemmWaves * kWmmaLaneCount);
    switch (mtiles) {
        case 1:
            dense_gemm_fp16_wmma_kernel<1><<<grid, block, 0, stream>>>(
                x, w, y, tokens, in_dim, out_dim, x_stride, y_stride);
            break;
        case 2:
            dense_gemm_fp16_wmma_kernel<2><<<grid, block, 0, stream>>>(
                x, w, y, tokens, in_dim, out_dim, x_stride, y_stride);
            break;
        default:
            dense_gemm_fp16_wmma_kernel<4><<<grid, block, 0, stream>>>(
                x, w, y, tokens, in_dim, out_dim, x_stride, y_stride);
            break;
    }
}

} // namespace aeon::rdna3
