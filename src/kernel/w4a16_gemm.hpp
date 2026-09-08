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

// ---------------------------------------------------------------------------
// Decode-optimized fused INT4 (W4A16) GEMV (single-token path, M == 1).
//
//   out[n] = sum_k a[k] * dequant(W[n,k])
//
// a        : (K) FP16 activations (row 0 of A)
// W_packed : (N, K/8) uint32 (GPTQ / compressed-tensors sequential packing)
// W_scale  : (N, K/32) FP16 (one scale per 32 weights)
// out      : (N) FP16 — only row 0 of the (M_PAD, N) output is written
//
// Rationale: production decode runs one token at a time, so the WMMA tile
// kernel above wastes 15/16 of its lanes on padding and serializes ~256
// dependent global->LDS->WMMA rounds behind two __syncthreads per K step
// (~140 us for a job whose memory floor is ~4.4 us at 960 GB/s). This kernel
// assigns one Wave32 warp per output row: lanes stream the row's packed
// weights as perfectly coalesced uint4 loads (one uint4 == 32 nibbles ==
// exactly one 32-wide scale group), dequantize inline with FP32 FMA against
// vectorized activation loads, and finish with a warp shuffle reduction.
// No shared memory, no __syncthreads, and 2048-4096 warps of parallel work
// instead of 32-64 thread blocks.
// ---------------------------------------------------------------------------
constexpr uint32_t GEMV_WARPS_PER_BLOCK = 8; // 256 threads

__global__ void __launch_bounds__(GEMV_WARPS_PER_BLOCK * 32) w4a16_gemv_kernel(
    const half* __restrict__ a,
    const uint32_t* __restrict__ w_packed,
    const half* __restrict__ w_scale,
    half* __restrict__ out,
    uint32_t N,
    uint32_t K
) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;

    const uint32_t words_per_row = K >> 3;  // K/8 packed words
    const uint32_t groups_per_row = K >> 5; // K/32 scale groups
    const uint4* __restrict__ w_row =
        reinterpret_cast<const uint4*>(w_packed + static_cast<size_t>(row) * words_per_row);
    const half* __restrict__ s_row = w_scale + static_cast<size_t>(row) * groups_per_row;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    // groups_per_row/32 is 4 (K=4096) or 2 (K=2048): unroll so the independent
    // uint4 weight loads of every group are in flight together.
    #pragma unroll 4
    for (uint32_t g = lane; g < groups_per_row; g += 32u) {
        // One uint4 = 4 words = 32 nibbles = exactly one 32-wide scale group.
        const uint4 pw = __ldg(w_row + g);
        const float sc = __half2float(__ldg(s_row + g));

        // Matching 32 activations: 4 x uint4 (8 halves each), 64B-aligned.
        const uint4* __restrict__ a_grp = reinterpret_cast<const uint4*>(a + g * 32);
        const uint4 avals[4] = {
            __ldg(a_grp + 0), __ldg(a_grp + 1), __ldg(a_grp + 2), __ldg(a_grp + 3)
        };
        const uint32_t wwords[4] = {pw.x, pw.y, pw.z, pw.w};

        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const __half2* ah = reinterpret_cast<const __half2*>(&avals[wi]);
            const uint32_t w = wwords[wi];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { // 4 half2 = 8 halves = 8 nibbles
                const float2 af = __half22float2(ah[i]);
                const float wlo = static_cast<float>((w >> (i * 8)) & 0xF) - 8.0f;
                const float whi = static_cast<float>((w >> (i * 8 + 4)) & 0xF) - 8.0f;
                acc0 = fmaf(af.x, wlo * sc, acc0);
                acc1 = fmaf(af.y, whi * sc, acc1);
            }
        }
    }
    float acc = acc0 + acc1;

    // Wave32 shuffle reduction (codebase-standard xor form)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset, 32);
    }
    if (lane == 0) {
        out[row] = __float2half(acc);
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
    // Single-token decode path: warp-per-row GEMV. Writes only output row 0.
    if (M == 1 && (K % 32u) == 0u) {
        dim3 block(GEMV_WARPS_PER_BLOCK * 32);
        dim3 grid((N + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK);
        w4a16_gemv_kernel<<<grid, block, 0, stream>>>(d_a, d_w_packed, d_w_scale, d_out, N, K);
        return;
    }

    dim3 block(GEMM_THREADS);
    dim3 grid((N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N, (M + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M);
    wmma_fused_int4_gemm_kernel<<<grid, block, 0, stream>>>(d_a, d_w_packed, d_w_scale, d_out, M, N, K);
}

} // namespace aeon::kernel
