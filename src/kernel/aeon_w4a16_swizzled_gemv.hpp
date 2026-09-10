#pragma once

#include "kernel/aeon_w4a16_swizzle.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace aeon::kernel {

using half2v = _Float16 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float fdot2(half2 a, half2 b, float accumulator) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2v, a),
                                  __builtin_bit_cast(half2v, b),
                                  accumulator, false);
#else
    return accumulator + __half2float(a.x) * __half2float(b.x) +
           __half2float(a.y) * __half2float(b.y);
#endif
}

__device__ __forceinline__ half2 unpack2(uint32_t word, int pair) {
    const uint32_t bits = ((word >> (4 * pair)) & 0x000F000Fu) | 0x64006400u;
    const half2 encoded = *reinterpret_cast<const half2*>(&bits);
    return __hsub2(encoded, __float2half2_rn(1032.0f));
}

__device__ __forceinline__ float swizzled_group_dot(
    uint4 words,
    const half2* activation_pairs
) {
    const uint32_t packed_words[4] = {words.x, words.y, words.z, words.w};
    float dot = 0.0f;

    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        const half2* word_activation = activation_pairs + word * 4;
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            dot = fdot2(unpack2(packed_words[word], pair),
                        word_activation[pair], dot);
        }
    }
    return dot;
}

template <int WAVES, int RPW, int LPR, int ITERS>
__global__ __launch_bounds__(WAVES * 32)
void aeon_w4a16_swizzled_gemv_kernel(
    const half* __restrict__ activation,
    const uint4* __restrict__ packed,
    const half* __restrict__ scale,
    half* __restrict__ output,
    int N
) {
    static_assert(WAVES > 0, "WAVES must be positive");
    static_assert(RPW > 0 && LPR > 0 && RPW * LPR == 32,
                  "RPW and LPR must describe one Wave32");

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int row_block = blockIdx.x * WAVES + wave;
    const int row = row_block * RPW + lane / LPR;
    const int slice = lane % LPR;
    const size_t block_offset = static_cast<size_t>(row_block) * ITERS * 32;

    float accumulator = 0.0f;
    if (row < N) {
        uint4 current_words = packed[block_offset + lane];
        float current_scale = __half2float(scale[block_offset + lane]);
        for (int iteration = 0; iteration < ITERS; ++iteration) {
            const int group = iteration * LPR + slice;
            uint4 next_words = current_words;
            float next_scale = current_scale;
            if (iteration + 1 < ITERS) {
                next_words = packed[block_offset + (iteration + 1) * 32 + lane];
                next_scale = __half2float(
                    scale[block_offset + (iteration + 1) * 32 + lane]);
            }

            const half2* activation_pairs =
                reinterpret_cast<const half2*>(activation + group * 32);
            accumulator += current_scale * swizzled_group_dot(current_words, activation_pairs);
            current_words = next_words;
            current_scale = next_scale;
        }
    }

    #pragma unroll
    for (int offset = LPR / 2; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor(accumulator, offset, 32);
    }

    if (row < N && slice == 0) {
        output[row] = __float2half(accumulator);
    }
}

template <int WAVES, int RPW, int LPR, int ITERS>
__global__ __launch_bounds__(WAVES * 32)
void aeon_w4a16_swizzled_dual_gemv_kernel(
    const half* __restrict__ activation,
    const uint4* __restrict__ packed_a,
    const half* __restrict__ scale_a,
    const uint4* __restrict__ packed_b,
    const half* __restrict__ scale_b,
    half* __restrict__ output_a,
    half* __restrict__ output_b,
    int N
) {
    static_assert(WAVES > 0, "WAVES must be positive");
    static_assert(RPW > 0 && LPR > 0 && RPW * LPR == 32,
                  "RPW and LPR must describe one Wave32");

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int row_block = blockIdx.x * WAVES + wave;
    const int row = row_block * RPW + lane / LPR;
    const int slice = lane % LPR;
    const size_t block_offset = static_cast<size_t>(row_block) * ITERS * 32;

    float accumulator_a = 0.0f;
    float accumulator_b = 0.0f;
    if (row < N) {
        uint4 current_words_a = packed_a[block_offset + lane];
        uint4 current_words_b = packed_b[block_offset + lane];
        float current_scale_a = __half2float(scale_a[block_offset + lane]);
        float current_scale_b = __half2float(scale_b[block_offset + lane]);

        for (int iteration = 0; iteration < ITERS; ++iteration) {
            const int group = iteration * LPR + slice;
            uint4 next_words_a = current_words_a;
            uint4 next_words_b = current_words_b;
            float next_scale_a = current_scale_a;
            float next_scale_b = current_scale_b;
            if (iteration + 1 < ITERS) {
                next_words_a = packed_a[block_offset + (iteration + 1) * 32 + lane];
                next_words_b = packed_b[block_offset + (iteration + 1) * 32 + lane];
                next_scale_a = __half2float(
                    scale_a[block_offset + (iteration + 1) * 32 + lane]);
                next_scale_b = __half2float(
                    scale_b[block_offset + (iteration + 1) * 32 + lane]);
            }

            const half2* activation_pairs =
                reinterpret_cast<const half2*>(activation + group * 32);
            accumulator_a += current_scale_a *
                             swizzled_group_dot(current_words_a, activation_pairs);
            accumulator_b += current_scale_b *
                             swizzled_group_dot(current_words_b, activation_pairs);
            current_words_a = next_words_a;
            current_words_b = next_words_b;
            current_scale_a = next_scale_a;
            current_scale_b = next_scale_b;
        }
    }

    #pragma unroll
    for (int offset = LPR / 2; offset > 0; offset >>= 1) {
        accumulator_a += __shfl_xor(accumulator_a, offset, 32);
        accumulator_b += __shfl_xor(accumulator_b, offset, 32);
    }

    if (row < N && slice == 0) {
        output_a[row] = __float2half(accumulator_a);
        output_b[row] = __float2half(accumulator_b);
    }
}

template <int WAVES, int RPW, int LPR, int ITERS>
inline void dispatch_aeon_w4a16_swizzled_gemv(
    const half* activation,
    const uint32_t* packed,
    const half* scale,
    half* output,
    int N,
    int K,
    hipStream_t stream = 0
) {
    static_assert(RPW * LPR == 32, "RPW and LPR must describe one Wave32");
    if (K != ITERS * LPR * 32 || N % RPW != 0) {
        return;
    }

    constexpr int threads_per_block = WAVES * 32;
    const dim3 block(threads_per_block);
    const dim3 grid((N / RPW + WAVES - 1) / WAVES);
    aeon_w4a16_swizzled_gemv_kernel<WAVES, RPW, LPR, ITERS>
        <<<grid, block, 0, stream>>>(activation,
                                      reinterpret_cast<const uint4*>(packed),
                                      scale, output, N);
}

                            template <int WAVES, int RPW, int LPR, int ITERS>
                            inline void dispatch_aeon_w4a16_swizzled_dual_gemv(
                                const half* activation,
                                const uint32_t* packed_a,
                                const half* scale_a,
                                const uint32_t* packed_b,
                                const half* scale_b,
                                half* output_a,
                                half* output_b,
                                int N,
                                int K,
                                hipStream_t stream = 0
                            ) {
                                static_assert(RPW * LPR == 32, "RPW and LPR must describe one Wave32");
                                if (K != ITERS * LPR * 32 || N % RPW != 0) {
                                    return;
                                }

                                constexpr int threads_per_block = WAVES * 32;
                                const dim3 block(threads_per_block);
                                const dim3 grid((N / RPW + WAVES - 1) / WAVES);
                                aeon_w4a16_swizzled_dual_gemv_kernel<WAVES, RPW, LPR, ITERS>
                                    <<<grid, block, 0, stream>>>(activation,
                                                                  reinterpret_cast<const uint4*>(packed_a),
                                                                  scale_a,
                                                                  reinterpret_cast<const uint4*>(packed_b),
                                                                  scale_b,
                                                                  output_a, output_b, N);
                            }

} // namespace aeon::kernel