#pragma once

// The grouped W_o_a attention-output projection (Wave32) and the half->float
// elementwise conversion that shares this module. Split out of
// `v4_attention.hpp`.

#include "architecture/deepseek_v4/kernels/v4_attention_config.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace aeon::kernel {

// Grouped W_o_a projection (Wave32):
// For each group g in 0..7: input is 8 heads x 512 = 4096 half elements.
// Projected by W_o_a[g]: [1024, 4096] -> Z[g]: [1024]
// Total output is [T, 8192]
__global__ void v4_grouped_wo_a_wave32_kernel(
    const __half* __restrict__ attn_out, // [T, 64, 512] -> viewed as [T, 8, 4096]
    const __half* __restrict__ wo_a_weight,// [8, 1024, 4096]
    __half*       __restrict__ z_out,    // [T, 8, 1024] = [T, 8192]
    int total_tokens
) {
    int out_col = blockIdx.x; // 0..1023
    int group   = blockIdx.y; // 0..7
    int token   = blockIdx.z; // 0..total_tokens-1
    int lane    = threadIdx.x;// 0..31

    const __half* in_vec = attn_out + token * (DSV4_O_GROUPS * DSV4_GROUP_HEADS_DIM) + group * DSV4_GROUP_HEADS_DIM;
    const __half* w_row  = wo_a_weight + group * (DSV4_O_LORA_RANK * DSV4_GROUP_HEADS_DIM) + out_col * DSV4_GROUP_HEADS_DIM;

    float dot = 0.0f;
    // Each thread processes 4096 / 32 = 128 elements
    #pragma unroll 4
    for (int i = lane; i < (int)DSV4_GROUP_HEADS_DIM; i += 32) {
        dot += __half2float(in_vec[i]) * __half2float(w_row[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_xor(dot, offset, 32);
    }

    if (lane == 0) {
        z_out[token * DSV4_TOTAL_O_LORA_DIM + group * DSV4_O_LORA_RANK + out_col] = __float2half(dot);
    }
}

// Half -> float elementwise conversion (replaces the per-layer D2H/sync/CPU/
// H2D router-logits round-trip with a single device kernel).
__global__ void v4_half_to_float_n_kernel(
    const __half* __restrict__ in,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

} // namespace aeon::kernel
