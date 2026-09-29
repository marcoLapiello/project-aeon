#pragma once

// The Hyper-Connections head reduction kernel (Wave32). Split out of
// `v4_attention.hpp`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace aeon::kernel {

// Hyper-Connections head reduction kernel (Wave32).
__global__ void __launch_bounds__(32) hc_head_wave32_kernel(
    const float* __restrict__ residual_in, // [4, 4096] = 16384 floats
    const float* __restrict__ hc_head_fn,  // [4, 16384] floats
    const float* __restrict__ hc_head_base,// [4] floats
    const float* __restrict__ hc_head_scale,// [1] float
    __half*      __restrict__ out,         // [4096] half
    int hidden_dim,                        // 4096
    int hc_mult,                           // 4
    float rms_eps,                         // 1e-6f
    float hc_eps                           // 1e-6f
) {
    int total_hc_dim = hc_mult * hidden_dim; // 16384
    int lane = threadIdx.x; // 0..31

    // Mean square over total_hc_dim
    float sum_sq = 0.0f;
    for (int i = lane; i < total_hc_dim; i += 32) {
        float v = residual_in[i];
        sum_sq += v * v;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor(sum_sq, offset, 32);
    }
    float rsqrt = rsqrtf((sum_sq / (float)total_hc_dim) + rms_eps);

    // Linear projection for each of the 4 streams
    __shared__ float s_pre[4];
    for (int s = 0; s < hc_mult; ++s) {
        float dot = 0.0f;
        const float* fn_row = hc_head_fn + s * total_hc_dim;
        for (int i = lane; i < total_hc_dim; i += 32) {
            dot += residual_in[i] * fn_row[i];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }
        if (lane == 0) {
            float mix = dot * rsqrt;
            float val = mix * hc_head_scale[0] + hc_head_base[s];
            float pre = (1.0f / (1.0f + expf(-val))) + hc_eps;
            s_pre[s] = pre;
        }
    }
    __syncthreads();

    // Combine streams into output [4096]
    for (int h = lane; h < hidden_dim; h += 32) {
        float acc = 0.0f;
        for (int s = 0; s < hc_mult; ++s) {
            acc += s_pre[s] * residual_in[s * hidden_dim + h];
        }
        out[h] = __float2half(acc);
    }
}

} // namespace aeon::kernel
