#pragma once

// -----------------------------------------------------------------------------
// The model's clamped SwiGLU: `silu(min(gate, limit)) * clamp(up, ±limit)`.
//
// This is a DeepSeek-V4 choice — the `swiglu_limit` is a config knob — so the
// kernel stays in the model tree. The reusable pieces that used to share this
// file now live in `platform/ops/cast.hpp` and `platform/ops/moe_accumulate.hpp`.
// -----------------------------------------------------------------------------

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>

namespace aeon::kernel {

// Clamped SwiGLU Kernel
__global__ void v4_swiglu_clamp_kernel(
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

} // namespace aeon::kernel
