#pragma once

// -----------------------------------------------------------------------------
// Model-agnostic primitive: FP16 <-> FP32 elementwise casts.
//
// Trivial device copies used wherever a path needs the other precision — the
// graph's checkpoint reads and the attention HC stages, for instance. Split out
// of the former `v4_pipeline_ops.hpp`, which mixed them with the model's clamped
// SwiGLU; they take only a pointer and a length and know nothing of any model.
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace aeon::kernel {

// Convert FP16 array to float array
__global__ void half_to_float_kernel(
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
__global__ void float_to_half_kernel(
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
