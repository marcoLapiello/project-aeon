#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdint>

namespace aeon::kernel {

using namespace rocwmma;

constexpr uint32_t WMMA_M = 16;
constexpr uint32_t WMMA_N = 16;
constexpr uint32_t WMMA_K = 16;

constexpr uint32_t GEMM_BLOCK_M = 16;
constexpr uint32_t GEMM_BLOCK_N = 64;
constexpr uint32_t GEMM_THREADS = 128; // 4 Wave32 wavefronts

// Fused INT4 (W4A16) Dequantization + WMMA GEMM Kernel:
// Output D = A @ W^T
// A: (M, K) FP16
// W_packed: (N, K/8) uint32_t (GPTQ / compressed-tensors sequential packing)
// W_scale: (N, K/32) FP16
// D: (M, N) FP16
__global__ void __launch_bounds__(GEMM_THREADS) wmma_fused_int4_gemm_kernel(
    const half* __restrict__ a,
    const uint32_t* __restrict__ w_packed,
    const half* __restrict__ w_scale,
    half* __restrict__ d,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    uint32_t block_m = blockIdx.y * GEMM_BLOCK_M;
    uint32_t block_n = blockIdx.x * GEMM_BLOCK_N;

    uint32_t warp_id = threadIdx.x / 32; // 0..3
    uint32_t warp_m = 0;                 // 1 warp along M (M=16)
    uint32_t warp_n = warp_id * WMMA_N;  // 16 cols per warp -> 4 * 16 = 64

    uint32_t global_m = block_m + warp_m;

    // LDS for staging A (16 x 16) and B (16 x 64)
    __shared__ half lds_a[16][16];
    __shared__ half lds_b[16][GEMM_BLOCK_N]; // (k, n)

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> accum;
    fill_fragment(accum, 0.0f);

    uint32_t tid = threadIdx.x;

    // Loop over K in steps of WMMA_K = 16
    for (uint32_t k_step = 0; k_step < K; k_step += WMMA_K) {
        // Cooperative load A into LDS: 16x16 = 256 elements
        if (tid < 128) {
            uint32_t row = tid / 16;
            uint32_t col = tid % 16;
            if (global_m + row < M && (k_step + col) < K) {
                lds_a[row][col] = a[(global_m + row) * K + (k_step + col)];
            } else {
                lds_a[row][col] = __float2half(0.0f);
            }
        }

        // Cooperative unpack W into LDS B: 1024 elements = 128 uint32_t words
        uint32_t col_in_block = tid / 2; // 0..63
        uint32_t word_in_k = tid % 2;    // 0..1 (0: k=0..7, 1: k=8..15)

        uint32_t abs_n = block_n + col_in_block;

        if (abs_n < N) {
            uint32_t packed = w_packed[abs_n * (K / 8) + (k_step / 8) + word_in_k];
            half sc = w_scale[abs_n * (K / 32) + (k_step / 32)];
            float sc_f = __half2float(sc);

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int nib = (packed >> (i * 4)) & 0xF;
                float val = (float)(nib - 8) * sc_f;
                lds_b[word_in_k * 8 + i][col_in_block] = __float2half(val);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                lds_b[word_in_k * 8 + i][col_in_block] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // WMMA execution for this warp
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_b;

        load_matrix_sync(frag_a, &lds_a[0][0], 16);
        load_matrix_sync(frag_b, &lds_b[0][warp_n], GEMM_BLOCK_N);

        mma_sync(accum, frag_a, frag_b, accum);
        __syncthreads();
    }

    // Store accumulator back to LDS and write to output
    __shared__ float lds_out[GEMM_BLOCK_M][GEMM_BLOCK_N];
    store_matrix_sync(&lds_out[warp_m][warp_n], accum, GEMM_BLOCK_N, mem_row_major);
    __syncthreads();

    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        uint32_t elem = tid * 8 + i;
        uint32_t r = elem / GEMM_BLOCK_N;
        uint32_t c = elem % GEMM_BLOCK_N;
        if ((block_m + r) < M && (block_n + c) < N) {
            d[(block_m + r) * N + (block_n + c)] = __float2half(lds_out[r][c]);
        }
    }
}

inline void dispatch_w4a16_gemm(
    const half* d_a,
    const uint32_t* d_w_packed,
    const half* d_w_scale,
    half* d_out,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    hipStream_t stream = 0
) {
    dim3 block(GEMM_THREADS);
    dim3 grid((N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N, (M + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M);
    wmma_fused_int4_gemm_kernel<<<grid, block, 0, stream>>>(d_a, d_w_packed, d_w_scale, d_out, M, N, K);
}

} // namespace aeon::kernel
