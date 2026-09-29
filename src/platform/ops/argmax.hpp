#pragma once

// The GPU argmax over the logit head: a per-block partial scan and a final
// cross-block reduce. Two launches, because HIP has no implicit grid-wide
// barrier between blocks in one ordinary kernel launch. Split out of
// `v4_attention.hpp`, where it was misfiled — this is the LM-head argmax, not
// an attention kernel.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <climits>
#include <cstdint>

namespace aeon::kernel {

// GPU argmax over the [129280] FP16 logit head.
// Phase 1 and Phase 2 are separate launches because HIP has no implicit
// grid-wide barrier between blocks in one ordinary kernel launch.
__global__ void __launch_bounds__(256) argmax_fp16_partial_kernel(
    const __half* __restrict__ logits,
    int n,
    float* __restrict__ partial_vals,   // [argmax_blocks]
    int32_t* __restrict__ partial_idx   // [argmax_blocks]
) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];

    const int tid = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;

    float best = -INFINITY;
    int best_i = n; // sentinel: larger than any real index so real winners beat it
    for (int i = gtid; i < n; i += stride) {
        float v = __half2float(logits[i]);
        if (v > best) { best = v; best_i = i; }
    }

    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();

    // Block reduction: on ties keep the smaller index (first-max-wins)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = s_val[tid + s];
            int   oi = s_idx[tid + s];
            if (ov > s_val[tid] || (ov == s_val[tid] && oi < s_idx[tid])) {
                s_val[tid] = ov;
                s_idx[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_vals[blockIdx.x] = s_val[0];
        partial_idx[blockIdx.x]  = s_idx[0];
    }
}

__global__ void __launch_bounds__(256) argmax_partial_reduce_kernel(
    const float* __restrict__ partial_vals,
    const int32_t* __restrict__ partial_idx,
    int partial_count,
    int32_t* __restrict__ out_idx
) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];

    const int tid = threadIdx.x;
    float best = -INFINITY;
    int best_i = INT_MAX;
    for (int i = tid; i < partial_count; i += blockDim.x) {
        const float value = partial_vals[i];
        const int index = partial_idx[i];
        if (value > best || (value == best && index < best_i)) {
            best = value;
            best_i = index;
        }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_value = s_val[tid + stride];
            const int other_index = s_idx[tid + stride];
            if (other_value > s_val[tid] ||
                (other_value == s_val[tid] && other_index < s_idx[tid])) {
                s_val[tid] = other_value;
                s_idx[tid] = other_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_idx[0] = s_idx[0];
    }
}

} // namespace aeon::kernel
