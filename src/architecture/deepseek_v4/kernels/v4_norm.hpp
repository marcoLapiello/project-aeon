#pragma once

// -----------------------------------------------------------------------------
// DeepSeek-V4 kept primitive: Wave32 RMSNorm.
//
// This kernel is part of the layer the rewrite *keeps* — only the graph that
// composes it is rebuilt. It was extracted from the pre-rewrite attention header
// so that a single primitive can be compiled, gated, and certified on its own,
// without dragging the legacy graph into the target. The legacy header now
// includes this one, so there is exactly one definition.
//
// Shape contract: one warp (32 lanes) per token row; lane `l` handles elements
// `l, l+32, l+64, …` and the sum-of-squares is reduced across the wave with
// `__shfl_xor`. Both forms accumulate in fp32 and write fp16.
//
// The two forms exist because the graph uses both:
//   - weighted (`v4_rmsnorm_wave32_kernel`)      — Step 2.1 attention norm.
//   - unit     (`v4_rmsnorm_unit_wave32_kernel`) — weightless norm sites.
// Which site uses which is a decision recorded in the plan (Step 2.1/2.2), not
// something this header decides.
//
// Verification: the gate is `tests/test_v4_norm_oracle.cpp`, which compares these
// kernels against `reference/dsv4_oracle.hpp` — an fp64 reference that shares no
// code with this file.
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace aeon::kernel {

// Weighted RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight.
__global__ void __launch_bounds__(32) v4_rmsnorm_wave32_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int dim,
    float eps
) {
    const int lane = threadIdx.x; // 0..31
    const int row = blockIdx.x;

    const __half* in_row = input + row * dim;
    __half* out_row = output + row * dim;

    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        const float v = __half2float(in_row[i]);
        sum_sq += v * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor(sum_sq, offset, 32);
    }

    const float inv_rms = rsqrtf((sum_sq / static_cast<float>(dim)) + eps);

    for (int i = lane; i < dim; i += 32) {
        const float v = __half2float(in_row[i]);
        const float w = __half2float(weight[i]);
        out_row[i] = __float2half(v * inv_rms * w);
    }
}

// Weightless RMSNorm: out = x * rsqrt(mean(x^2) + eps).
__global__ void __launch_bounds__(32) v4_rmsnorm_unit_wave32_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int dim,
    float eps
) {
    const int lane = threadIdx.x;
    const int row = blockIdx.x;

    const __half* in_row = input + row * dim;
    __half* out_row = output + row * dim;

    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        const float v = __half2float(in_row[i]);
        sum_sq += v * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor(sum_sq, offset, 32);
    }

    const float inv_rms = rsqrtf((sum_sq / static_cast<float>(dim)) + eps);

    for (int i = lane; i < dim; i += 32) {
        out_row[i] = __float2half(__half2float(in_row[i]) * inv_rms);
    }
}

} // namespace aeon::kernel
