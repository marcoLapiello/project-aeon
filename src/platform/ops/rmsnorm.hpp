#pragma once

// -----------------------------------------------------------------------------
// Model-agnostic primitive: Wave32 RMSNorm.
//
// It lives in its own header so the primitive can be compiled, gated, and
// certified on its own, without pulling in the composition that calls it;
// `v4_attention.hpp` includes this one, so there is exactly one definition.
//
// Shape contract: one warp (32 lanes) per token row; lane `l` handles elements
// `l, l+32, l+64, …` and the sum-of-squares is reduced across the wave with
// `__shfl_xor`. Both forms accumulate in fp32 and write fp16.
//
// The two forms exist because the graph uses both:
//   - weighted (`rmsnorm_wave32_kernel`)      — the attention norm site.
//   - unit     (`rmsnorm_unit_wave32_kernel`) — the weightless norm sites.
// Which site uses which is a per-site decision, not something this header decides.
//
// Verification: the gate is `tests/test_v4_norm_oracle.cpp`, which compares these
// kernels against `reference/dsv4_oracle.hpp` — an fp64 reference that shares no
// code with this file.
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace aeon::kernel {

// Weighted RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight.
//
// `out_stride` is the row pitch of `output`, defaulting to `dim` (the input pitch).
// A caller whose rows are wider than the data they hold — the FFN norm writes row 0 of
// a 16-row padded tile — passes its own pitch so a batched launch does not have each
// row overwrite the next tile's row 0.
__global__ void __launch_bounds__(32) rmsnorm_wave32_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int dim,
    float eps,
    int out_stride = 0
) {
    const int lane = threadIdx.x; // 0..31
    const int row = blockIdx.x;

    const __half* in_row = input + row * dim;
    __half* out_row = output + static_cast<size_t>(row) * (out_stride > 0 ? out_stride : dim);

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
__global__ void __launch_bounds__(32) rmsnorm_unit_wave32_kernel(
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
