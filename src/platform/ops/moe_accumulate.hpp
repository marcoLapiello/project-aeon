#pragma once

// -----------------------------------------------------------------------------
// Model-agnostic primitive: reduce per-expert MoE contributions into a token.
//
// Split out of the former `v4_pipeline_ops.hpp`, which mixed them with the
// model's clamped SwiGLU. These kernels are parameterised (expert count, hidden
// dim, nullable shared output) and carry no model choice, so any MoE
// architecture can use them.
//
// Two variants of the same reduction, and the choice between them matters:
// `moe_accumulate_expert_kernel` keeps its accumulator in fp16 (superseded, kept
// only as a gate control), while `moe_accumulate_fixed_order_kernel` accumulates
// in fp32 in slot order with a single rounding.
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace aeon::kernel {

// Accumulate weighted expert output into token hidden state.
//
// **SUPERSEDED — do not select this in any new path.** Its accumulator is stored in
// fp16 and re-rounded on every one of the six steps, which violates the fp32
// accumulation rule the fixed-order pair below satisfies; measured against that
// pair it is 3.4x less accurate on the model's own routing-weight shape. It is
// retained **only as a gate control**: the deterministic fixtures use it to check
// that a run reproducing the fp16 order still matches, so its reader is
// `tests/support/`. The replacement is `aeon_moe_fused_w2_contrib_kernel` +
// `moe_accumulate_fixed_order_kernel`, which is fp32 *and* has a fixed order.
__global__ void moe_accumulate_expert_kernel(
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
//   * **fp32 accumulation**. The fp16 read-modify-write above
//     (`moe_accumulate_expert_kernel`) violates that: it stores its accumulator in
//     fp16 and re-rounds on each of the six steps.
//   * **a fixed order**. `atomicAdd` cannot provide one, because the order in which
//     the six expert blocks reach a given element is the scheduler's.
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
__global__ void moe_accumulate_fixed_order_kernel(
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

} // namespace aeon::kernel
