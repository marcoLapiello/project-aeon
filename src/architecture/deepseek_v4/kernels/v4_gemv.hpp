#pragma once

// -----------------------------------------------------------------------------
// DeepSeek-V4 kept primitive: FP16 GEMV (dense projections).
//
// The dense backbone's projections are all `y = W @ x` with fp16 weights and a
// single token: `wq_a`, `wq_b`, `wkv`, the compressor gates, the indexer
// projections, the router gate, the shared expert, and the LM head. Extracted
// from the pre-rewrite attention header so the dense path can be composed and
// gated outside the legacy graph, as with `v4_norm.hpp` and `v4_rope.hpp`.
//
// Two forms:
//   `v4_gemv_fp16_kernel`      — one half per lane per step. Used for the MLA
//                                projections and the compressor/indexer gates.
//   `v4_gemv_fp16_vec8_kernel` — eight halves per lane per step via `uint4`.
//                                Used for the shared expert and LM head.
//                                Requires `in_dim % 8 == 0`.
//
// Both accumulate in fp32 and write fp16, and both use `gridDim.x` as the output
// stride, so the launch must be `dim3(out_dim, num_tokens)` with 32 threads.
//
// NOTE on weight orientation: `W` is `[out_dim, in_dim]` row-major — the same
// layout the checkpoint stores (`nn.Linear.weight`) — and the kernel computes
// `y[o] = Σ_i W[o, i] · x[i]`. There is no transpose anywhere in the path.
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace aeon::kernel {

// One lane per output column, striding the input dimension by the wave width.
// `y[token * gridDim.x + out_col] = dot(x_row, w_row)`.
__global__ void v4_gemv_fp16_kernel(
    const __half* __restrict__ x,       // [T, in_dim]
    const __half* __restrict__ w,       // [out_dim, in_dim]
    __half*       __restrict__ y,       // [T, out_dim]
    int in_dim
) {
    int out_col = blockIdx.x;           // row of W, output feature index
    int token   = blockIdx.y;           // token index
    int lane    = threadIdx.x;          // 0..31

    const __half* x_row = x + token * in_dim;
    const __half* w_row = w + out_col * in_dim;

    float dot = 0.0f;
    for (int i = lane; i < in_dim; i += 32) {
        dot += __half2float(x_row[i]) * __half2float(w_row[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_xor(dot, offset, 32);
    }

    if (lane == 0) {
        y[token * gridDim.x + out_col] = __float2half(dot);
    }
}

// Vectorized FP16 GEMV: each lane streams 8 halves per iteration via uint4 — 8x
// fewer global transactions and FP32 FMA accumulation.
__global__ void __launch_bounds__(32) v4_gemv_fp16_vec8_kernel(
    const __half* __restrict__ x,       // [T, in_dim]
    const __half* __restrict__ w,       // [out_dim, in_dim]
    __half*       __restrict__ y,       // [T, out_dim]
    int in_dim
) {
    int out_col = blockIdx.x;
    int token   = blockIdx.y;
    int lane    = threadIdx.x;

    const uint4* x_row = reinterpret_cast<const uint4*>(x + token * in_dim);
    const uint4* w_row = reinterpret_cast<const uint4*>(w + out_col * in_dim);
    const int vec_dim = in_dim >> 3; // 8 halves per uint4

    float dot = 0.0f;
    for (int i = lane; i < vec_dim; i += 32) {
        const uint4 xv = __ldg(x_row + i);
        const uint4 wv = __ldg(w_row + i);
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        const __half2* wh = reinterpret_cast<const __half2*>(&wv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 xf = __half22float2(xh[j]);
            const float2 wf = __half22float2(wh[j]);
            dot = fmaf(xf.x, wf.x, dot);
            dot = fmaf(xf.y, wf.y, dot);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_xor(dot, offset, 32);
    }

    if (lane == 0) {
        y[token * gridDim.x + out_col] = __float2half(dot);
    }
}

} // namespace aeon::kernel
