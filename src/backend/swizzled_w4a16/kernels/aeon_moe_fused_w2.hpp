#pragma once

#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace aeon::kernel {

struct SwizzledW2ExpertPtrs {
    const uint4* w2[kAeonSwizzledMaxExperts];
    const half* s2[kAeonSwizzledMaxExperts];
};

// One row of one expert's W2 projection, reduced across the lanes of a Wave32.
//
// Extracted so the two accumulation paths below cannot drift: the atomic path and
// the fixed-order path must compute *exactly* the same per-expert number, and the
// only difference between them must be how those numbers are combined. Two copies
// of this loop would let a fix land in one and not the other, which is the failure
// mode the pair exists to rule out.
//
// The shuffles run for **every** lane in the wave, including lanes whose
// `(expert, row)` is out of range, because `__shfl_xor` is a wave-synchronous
// operation: an early return would deadlock the lanes that still need to
// participate.

template <int RPW, int LPR, int ITERS>
__device__ __forceinline__ float swizzled_w2_row_dot(
    const uint4* __restrict__ w2,
    const half* __restrict__ s2,
    const half* __restrict__ activation,
    int lane,
    int slice,
    size_t block_offset,
    bool valid
) {
    static_assert(RPW > 0 && LPR > 0 && RPW * LPR == 32,
                  "RPW and LPR must describe one Wave32");

    float accumulator = 0.0f;
    if (valid) {
        uint4 current_words = w2[block_offset + lane];
        float current_scale = __half2float(s2[block_offset + lane]);

        for (int iteration = 0; iteration < ITERS; ++iteration) {
            const int group = iteration * LPR + slice;
            uint4 next_words = current_words;
            float next_scale = current_scale;
            if (iteration + 1 < ITERS) {
                next_words = w2[block_offset + (iteration + 1) * 32 + lane];
                next_scale = __half2float(s2[block_offset + (iteration + 1) * 32 + lane]);
            }

            const half2* activation_pairs =
                reinterpret_cast<const half2*>(activation + group * 32);
            accumulator += current_scale *
                           swizzled_group_dot(current_words, activation_pairs);
            current_words = next_words;
            current_scale = next_scale;
        }
    }

    #pragma unroll
    for (int offset = LPR / 2; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor(accumulator, offset, 32);
    }
    return accumulator;
}

template <int WAVES, int RPW, int LPR, int ITERS>
__global__ __launch_bounds__(WAVES * 32)
void aeon_moe_fused_w2_accum_kernel(    const half* __restrict__ expert_hidden,
    SwizzledW2ExpertPtrs weights,
    const float* __restrict__ topk_weights,
    const half* __restrict__ initial_output,
    float* __restrict__ output_f32,
    half* __restrict__ output_f16,
    int* __restrict__ counters,
    int expert_count,
    int N
) {
    static_assert(WAVES > 0, "WAVES must be positive");
    static_assert(RPW > 0 && LPR > 0 && RPW * LPR == 32,
                  "RPW and LPR must describe one Wave32");

    const int expert = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int row_block = blockIdx.x * WAVES + wave;
    const int row = row_block * RPW + lane / LPR;
    const int slice = lane % LPR;
    const size_t block_offset = static_cast<size_t>(row_block) * ITERS * 32;

    float accumulator = 0.0f;
    if (expert < expert_count && row < N) {
        // `swizzled_w2_row_dot` already reduces across the `LPR` slices of this row
        // (its closing `__shfl_xor` loop), so the caller must **not** reduce again.
        // It did, and the two reductions composed: the second one summed each slice's
        // already-complete row total, multiplying the routed contribution by `LPR`
        // (a silent 4x on the committed path). The kernel returned a plausible number
        // at the wrong scale, which is why only an oracle comparison saw it — items
        // 16/17/18 at `moe_out`, while any run-vs-run check agreed with itself.
        accumulator = swizzled_w2_row_dot<RPW, LPR, ITERS>(
            weights.w2[expert],
            weights.s2[expert],
            expert_hidden + static_cast<size_t>(expert) * (LPR * ITERS * 32),
            lane, slice, block_offset,
            /*valid=*/true);
    }

    if (expert < expert_count && row < N && slice == 0) {
        float contribution = topk_weights[expert] * accumulator;
        if (expert == 0 && initial_output != nullptr) {
            contribution += __half2float(initial_output[row]);
        }
        atomicAdd(&output_f32[row], contribution);
    }

    __syncthreads();
    __threadfence();
    __shared__ int last_block;
    if (threadIdx.x == 0) {
        last_block = (atomicAdd(&counters[blockIdx.x], 1) == expert_count - 1);
    }
    __syncthreads();

    if (last_block) {
        const int base = blockIdx.x * WAVES * RPW;
        for (int index = threadIdx.x; index < WAVES * RPW; index += WAVES * 32) {
            if (base + index < N) {
                output_f16[base + index] = __float2half(output_f32[base + index]);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            counters[blockIdx.x] = 0;
        }
    }
}

template <int WAVES, int RPW, int LPR, int ITERS>
inline void dispatch_aeon_moe_fused_w2_accum(
    const half* expert_hidden,
    const SwizzledW2ExpertPtrs& weights,
    const float* topk_weights,
    const half* initial_output,
    float* output_f32,
    half* output_f16,
    int* counters,
    int expert_count,
    int N,
    int K,
    hipStream_t stream = 0
) {
    static_assert(RPW * LPR == 32, "RPW and LPR must describe one Wave32");
    const int expected_k = ITERS * LPR * 32;
    if (expert_count <= 0 || expert_count > kAeonSwizzledMaxExperts ||
        N <= 0 || K != expected_k || N % RPW != 0) {
        throw std::invalid_argument(
            "dispatch_aeon_moe_fused_w2_accum: incompatible expert or N/K shape");
    }

    constexpr int threads_per_block = WAVES * 32;
    const dim3 block(threads_per_block);
    const dim3 grid((N / RPW + WAVES - 1) / WAVES, expert_count);
    aeon_moe_fused_w2_accum_kernel<WAVES, RPW, LPR, ITERS>
        <<<grid, block, 0, stream>>>(expert_hidden, weights, topk_weights,
                          initial_output,
                                      output_f32, output_f16, counters,
                                      expert_count, N);
}

// ---------------------------------------------------------------------------
// The fixed-order routed accumulation, stage 1 of 2.
//
// Each expert writes its own weighted contribution into its own fp32 slice instead
// of adding into a shared row. `contrib[expert * N + row]` therefore has exactly
// one writer, which makes the result **deterministic by construction** rather than
// by observation — there is no cross-expert communication left to order.
//
// This is the piece that removes the trade the previous two paths forced:
//
//   * `aeon_moe_fused_w2_accum_kernel` accumulates with `atomicAdd`, whose order
//     across experts is the scheduler's and so is not reproducible (trap 38);
//   * the `v4_pipeline_accumulate_expert_kernel` path is reproducible but keeps its
//     accumulator in **fp16** and re-rounds on each of the six steps, which is what
//     plan §2.10.3 forbids ("accumulate in fp32").
//
// Neither had both properties, so every gate that needed reproducibility had to
// accept the less accurate one. Stage 2 (`v4_moe_accumulate_fixed_order_kernel`)
// sums these slices in slot order in fp32 and rounds once, which is both.
//
// The per-expert arithmetic is `swizzled_w2_row_dot`, shared with the atomic path,
// so the two can differ only in how the numbers are combined.
template <int WAVES, int RPW, int LPR, int ITERS>
__global__ __launch_bounds__(WAVES * 32)
void aeon_moe_fused_w2_contrib_kernel(
    const half* __restrict__ expert_hidden,
    SwizzledW2ExpertPtrs weights,
    const float* __restrict__ topk_weights,
    float* __restrict__ contrib,
    int expert_count,
    int N
) {
    static_assert(WAVES > 0, "WAVES must be positive");
    static_assert(RPW > 0 && LPR > 0 && RPW * LPR == 32,
                  "RPW and LPR must describe one Wave32");

    const int expert = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int row_block = blockIdx.x * WAVES + wave;
    const int row = row_block * RPW + lane / LPR;
    const int slice = lane % LPR;
    const size_t block_offset = static_cast<size_t>(row_block) * ITERS * 32;

    const bool valid = (expert < expert_count) && (row < N);
    const float accumulator = swizzled_w2_row_dot<RPW, LPR, ITERS>(
        weights.w2[expert],
        weights.s2[expert],
        expert_hidden + static_cast<size_t>(expert) * (LPR * ITERS * 32),
        lane, slice, block_offset, valid);

    if (valid && slice == 0) {
        contrib[static_cast<size_t>(expert) * N + row] = topk_weights[expert] * accumulator;
    }
}

template <int WAVES, int RPW, int LPR, int ITERS>
inline void dispatch_aeon_moe_fused_w2_contrib(
    const half* expert_hidden,
    const SwizzledW2ExpertPtrs& weights,
    const float* topk_weights,
    float* contrib,
    int expert_count,
    int N,
    int K,
    hipStream_t stream = 0
) {
    static_assert(RPW * LPR == 32, "RPW and LPR must describe one Wave32");
    const int expected_k = ITERS * LPR * 32;
    if (expert_count <= 0 || expert_count > kAeonSwizzledMaxExperts ||
        N <= 0 || K != expected_k || N % RPW != 0 || contrib == nullptr) {
        throw std::invalid_argument(
            "dispatch_aeon_moe_fused_w2_contrib: incompatible expert or N/K shape");
    }

    constexpr int threads_per_block = WAVES * 32;
    const dim3 block(threads_per_block);
    const dim3 grid((N / RPW + WAVES - 1) / WAVES, expert_count);
    aeon_moe_fused_w2_contrib_kernel<WAVES, RPW, LPR, ITERS>
        <<<grid, block, 0, stream>>>(expert_hidden, weights, topk_weights,
                                      contrib, expert_count, N);
}

} // namespace aeon::kernel