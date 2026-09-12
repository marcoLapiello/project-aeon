#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>

namespace aeon::kernel {

// Clamped SwiGLU Kernel
__global__ void v4_pipeline_swiglu_clamp_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int total_elements,
    float limit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
        float silu_g = g / (1.0f + expf(-g));
        out[idx] = __float2half(silu_g * u);
    }
}

// Accumulate weighted expert output into token hidden state
__global__ void v4_pipeline_accumulate_expert_kernel(
    half* __restrict__ accum_out,
    const half* __restrict__ expert_out,
    float weight,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        float acc = __half2float(accum_out[idx]);
        float exp = __half2float(expert_out[idx]);
        accum_out[idx] = __float2half(acc + weight * exp);
    }
}

// Convert FP16 array to float array
__global__ void v4_half_to_float_kernel(
    const half* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

// Convert float array to FP16 array
__global__ void v4_float_to_half_kernel(
    const float* __restrict__ src,
    half* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

} // namespace aeon::kernel
