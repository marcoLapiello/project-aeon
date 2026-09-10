#pragma once

#include "kernel/aeon_w4a16_swizzled_gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace aeon::kernel {

constexpr int kAeonSwizzledMaxExperts = 8;

struct SwizzledW13ExpertPtrs {
    const uint4* w1[kAeonSwizzledMaxExperts];
    const half* s1[kAeonSwizzledMaxExperts];
    const uint4* w3[kAeonSwizzledMaxExperts];
    const half* s3[kAeonSwizzledMaxExperts];
};

__device__ __forceinline__ float aeon_swiglu_clamped(float gate, float up, float limit) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
    return (gate / (1.0f + expf(-gate))) * up;
}

template <int WAVES, int RPW, int LPR, int ITERS, bool STAGE_ACTIVATION = false>
__global__ __launch_bounds__(WAVES * 32)
void aeon_moe_fused_w13_swiglu_kernel(
    const half* __restrict__ activation,
    SwizzledW13ExpertPtrs weights,
    half* __restrict__ expert_hidden,
    float* __restrict__ output_f32,
    int output_dim,
    int expert_count,
    int N,
    float swiglu_limit
) {
    static_assert(WAVES > 0, "WAVES must be positive");
    static_assert(RPW > 0 && LPR > 0 && RPW * LPR == 32,
                  "RPW and LPR must describe one Wave32");

    const int expert = blockIdx.y;
    if (expert >= expert_count) {
        return;
    }

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int row_block = blockIdx.x * WAVES + wave;
    const int row = row_block * RPW + lane / LPR;
    const int slice = lane % LPR;
    const size_t block_offset = static_cast<size_t>(row_block) * ITERS * 32;

    extern __shared__ uint4 shared_activation_words[];
    const half* activation_source = activation;
    if constexpr (STAGE_ACTIVATION) {
        uint4* staged_activation = shared_activation_words;
        constexpr int activation_words = ITERS * LPR * 32 / 8;
        const uint4* source_activation = reinterpret_cast<const uint4*>(activation);
        for (int index = threadIdx.x; index < activation_words; index += blockDim.x) {
            staged_activation[index] = source_activation[index];
        }
        __syncthreads();
        activation_source = reinterpret_cast<const half*>(staged_activation);
    }

    if (blockIdx.y == 0 && output_f32 != nullptr) {
        for (int index = blockIdx.x * blockDim.x + threadIdx.x;
             index < output_dim;
             index += gridDim.x * blockDim.x) {
            output_f32[index] = 0.0f;
        }
    }

    float gate_accumulator = 0.0f;
    float up_accumulator = 0.0f;
    if (row < N) {
        const uint4* w1 = weights.w1[expert];
        const half* s1 = weights.s1[expert];
        const uint4* w3 = weights.w3[expert];
        const half* s3 = weights.s3[expert];
        uint4 current_w1 = w1[block_offset + lane];
        uint4 current_w3 = w3[block_offset + lane];
        float current_s1 = __half2float(s1[block_offset + lane]);
        float current_s3 = __half2float(s3[block_offset + lane]);

        for (int iteration = 0; iteration < ITERS; ++iteration) {
            const int group = iteration * LPR + slice;
            uint4 next_w1 = current_w1;
            uint4 next_w3 = current_w3;
            float next_s1 = current_s1;
            float next_s3 = current_s3;
            if (iteration + 1 < ITERS) {
                next_w1 = w1[block_offset + (iteration + 1) * 32 + lane];
                next_w3 = w3[block_offset + (iteration + 1) * 32 + lane];
                next_s1 = __half2float(s1[block_offset + (iteration + 1) * 32 + lane]);
                next_s3 = __half2float(s3[block_offset + (iteration + 1) * 32 + lane]);
            }

            const half2* activation_pairs =
                reinterpret_cast<const half2*>(activation_source + group * 32);
            gate_accumulator += current_s1 *
                                swizzled_group_dot(current_w1, activation_pairs);
            up_accumulator += current_s3 *
                              swizzled_group_dot(current_w3, activation_pairs);
            current_w1 = next_w1;
            current_w3 = next_w3;
            current_s1 = next_s1;
            current_s3 = next_s3;
        }
    }

    #pragma unroll
    for (int offset = LPR / 2; offset > 0; offset >>= 1) {
        gate_accumulator += __shfl_xor(gate_accumulator, offset, 32);
        up_accumulator += __shfl_xor(up_accumulator, offset, 32);
    }

    if (row < N && slice == 0) {
        expert_hidden[static_cast<size_t>(expert) * N + row] =
            __float2half(aeon_swiglu_clamped(gate_accumulator, up_accumulator, swiglu_limit));
    }
}

template <int WAVES, int RPW, int LPR, int ITERS, bool STAGE_ACTIVATION = false>
inline void dispatch_aeon_moe_fused_w13_swiglu(
    const half* activation,
    const SwizzledW13ExpertPtrs& weights,
    half* expert_hidden,
    float* output_f32,
    int output_dim,
    int expert_count,
    int N,
    int K,
    float swiglu_limit,
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
    constexpr size_t shared_bytes = STAGE_ACTIVATION
        ? static_cast<size_t>(ITERS * LPR * 32) * sizeof(half)
        : 0;
    aeon_moe_fused_w13_swiglu_kernel<WAVES, RPW, LPR, ITERS, STAGE_ACTIVATION>
        <<<grid, block, shared_bytes, stream>>>(activation, weights, expert_hidden,
                                                output_f32, output_dim,
                                                expert_count, N, swiglu_limit);
}

} // namespace aeon::kernel