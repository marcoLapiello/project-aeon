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

template <int WAVES, int RPW, int LPR, int ITERS>
__global__ __launch_bounds__(WAVES * 32)
void aeon_moe_fused_w2_accum_kernel(
    const half* __restrict__ expert_hidden,
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
        const uint4* w2 = weights.w2[expert];
        const half* s2 = weights.s2[expert];
        const half* activation = expert_hidden + static_cast<size_t>(expert) * (LPR * ITERS * 32);
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
    if (expert_count <= 0 || expert_count > kAeonSwizzledMaxExperts ||
        K != ITERS * LPR * 32 || N % RPW != 0) {
        return;
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

} // namespace aeon::kernel