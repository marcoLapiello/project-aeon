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

__global__ void v4_pipeline_swiglu_clamp_batched_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int total_elements,
    float limit
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        float gate_value = __half2float(gate[index]);
        float up_value = __half2float(up[index]);
        gate_value = fminf(gate_value, limit);
        up_value = fminf(fmaxf(up_value, -limit), limit);
        out[index] = __float2half(
            (gate_value / (1.0f + expf(-gate_value))) * up_value);
    }
}

// Accumulate weighted expert output into token hidden state.
//
// **SUPERSEDED — do not select this in any new path.** Its accumulator is stored in
// fp16 and re-rounded on every one of the six steps, which violates plan §2.10.3
// ("accumulate in fp32"); measured against the fixed-order pair below it is 3.4x
// less accurate on the model's own routing-weight shape. It is retained only
// because the pre-rewrite graph behind `AEON_ENABLE_LEGACY_V4_GRAPH` still calls
// it. The replacement is `aeon_moe_fused_w2_contrib_kernel` +
// `v4_moe_accumulate_fixed_order_kernel`, which is fp32 *and* has a fixed order.
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

// Fixed-order fp32 accumulation of the routed experts — stage 2 of 2.
//
// `contrib` holds one weighted contribution per expert per output element, each
// written by exactly one thread by `aeon_moe_fused_w2_contrib_kernel`, so this
// kernel is a plain read: the sum runs in **slot order** `0 … expert_count-1` in
// fp32, and the single `__float2half` at the end is the only rounding in the whole
// routed path.
//
// It is the path that satisfies both requirements at once, which neither previous
// path did:
//
//   * plan §2.10.3 requires **fp32 accumulation**. The fp16 read-modify-write
//     above (`v4_pipeline_accumulate_expert_kernel`, still used by the pre-rewrite
//     graph) violates that: it stores its accumulator in fp16 and re-rounds on
//     each of the six steps.
//   * trap 38 requires a **fixed order**. `atomicAdd` cannot provide one, because
//     the order in which the six expert blocks reach a given element is the
//     scheduler's.
//
// Together those two constraints had left no correct option: gates that needed
// reproducibility had to accept the less accurate path. Splitting the accumulation
// into per-expert slices plus a fixed-order reduce removes the trade entirely —
// and it is faster than the fp16 path, which needs twelve launches where this pair
// needs two.
//
// `shared_output` is folded in as the **initial value of the fp32 accumulator**,
// which is the reference's own fused shape: the shared expert's contribution enters
// as an addend of the accumulation, not as a post-hoc `+=` on a completed sum.
// Pass `nullptr` when there is no shared expert (the sequential gates do).
__global__ void v4_moe_accumulate_fixed_order_kernel(
    const float* __restrict__ contrib,       // [expert_count, hidden_dim]
    int expert_count,
    const half* __restrict__ shared_output,  // [hidden_dim], nullable
    half* __restrict__ out,                  // [hidden_dim]
    int hidden_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) {
        return;
    }

    float accumulator = (shared_output != nullptr)
        ? __half2float(shared_output[idx])
        : 0.0f;
    for (int expert = 0; expert < expert_count; ++expert) {
        accumulator += contrib[static_cast<size_t>(expert) * hidden_dim + idx];
    }
    out[idx] = __float2half(accumulator);
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
