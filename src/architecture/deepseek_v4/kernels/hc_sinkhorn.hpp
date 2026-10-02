#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace aeon::kernel {

// Hyper-Connections Parameters for DeepSeek-V4
// hc_mult = 4
// mix_hc = 4 * (2 + 4) = 24
// hidden_size = 4096
// hc_hidden_size = 4 * 4096 = 16384

// Parallelized Wave32 Kernel for HC Projection:
// Launches with gridDim = 24 (1 block per output mix).
// BlockDim = 256 (8 Wave32 warps per block).
// Each block computes the RMS of residual_in in parallel across its 8 warps using float4 vectorized loads,
// and computes the dot product of residual_in with its assigned row of fn (fn[mix_idx, :]).
__global__ void __launch_bounds__(256) hc_project_kernel(
    const float* __restrict__ residual_in,
    const float* __restrict__ fn,
    float* __restrict__ mixes,
    int hidden_size,
    int hc_mult,
    float rms_eps
) {
    int mix_idx = blockIdx.x;
    // The token is `blockIdx.y`: a chunk runs one block column per (mix, token), a
    // single-token launch leaves it at 0. The residual is `[num_tokens, hc_mult,
    // hidden]` and `mixes` `[num_tokens, 24]`, both contiguous per token, so the
    // offsets need no pitch argument.
    int token_idx = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5; // 0..7

    int hc_hidden_size = hc_mult * hidden_size; // 16384

    __shared__ float s_warp_sqr[8];
    __shared__ float s_warp_dot[8];

    const float* token_residual = residual_in + static_cast<size_t>(token_idx) * hc_hidden_size;
    float* token_mixes = mixes + static_cast<size_t>(token_idx) * 24;
    const float4* res4 = reinterpret_cast<const float4*>(token_residual);
    const float4* fn4 = reinterpret_cast<const float4*>(fn + mix_idx * hc_hidden_size);
    int num_f4 = hc_hidden_size / 4; // 4096

    float local_sqr = 0.0f;
    float local_dot = 0.0f;

    for (int i = tid; i < num_f4; i += 256) {
        float4 r = res4[i];
        float4 f = fn4[i];

        local_sqr += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        local_dot += r.x * f.x + r.y * f.y + r.z * f.z + r.w * f.w;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sqr += __shfl_xor(local_sqr, offset, 32);
        local_dot += __shfl_xor(local_dot, offset, 32);
    }

    if (lane == 0) {
        s_warp_sqr[wid] = local_sqr;
        s_warp_dot[wid] = local_dot;
    }
    __syncthreads();

    if (wid == 0) {
        float block_sqr = (lane < 8) ? s_warp_sqr[lane] : 0.0f;
        float block_dot = (lane < 8) ? s_warp_dot[lane] : 0.0f;

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            block_sqr += __shfl_xor(block_sqr, offset, 32);
            block_dot += __shfl_xor(block_dot, offset, 32);
        }

        if (lane == 0) {
            float rms = rsqrtf((block_sqr / static_cast<float>(hc_hidden_size)) + rms_eps);
            token_mixes[mix_idx] = block_dot * rms;
        }
    }
}

// Vectorized Pre-Combine Kernel:
// Pre-combines 4 residual streams into layer_input [4096] half elements using float4 and half2 vectorization.
// Each thread processes 4 output elements (1 float4 load per stream, producing 2 half2 elements).
__global__ void __launch_bounds__(256) hc_pre_combine_kernel(
    const float* __restrict__ residual_in,
    const float* __restrict__ pre_mix,
    __half* __restrict__ layer_input,
    int hidden_size,
    int hc_mult
) {
    // Token on `blockIdx.y`, as in `hc_project_kernel`; the block's `x` tiles the
    // hidden axis. `residual_in` is `[num_tokens, hc_mult, hidden]`, `pre_mix`
    // `[num_tokens, 4]` and `layer_input` `[num_tokens, hidden]`, all contiguous.
    int token_idx = blockIdx.y;
    const float* token_residual =
        residual_in + static_cast<size_t>(token_idx) * hc_mult * hidden_size;
    const float* token_pre = pre_mix + static_cast<size_t>(token_idx) * 4;
    __half* token_out = layer_input + static_cast<size_t>(token_idx) * hidden_size;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;
    const bool valid = idx < hidden_size;

    __shared__ float s_pre[4];
    if (threadIdx.x < 4) {
        s_pre[threadIdx.x] = token_pre[threadIdx.x];
    }
    __syncthreads();
    if (!valid) return;

    float p0 = s_pre[0];
    float p1 = s_pre[1];
    float p2 = s_pre[2];
    float p3 = s_pre[3];

    const float4* r0 = reinterpret_cast<const float4*>(token_residual + 0 * hidden_size);
    const float4* r1 = reinterpret_cast<const float4*>(token_residual + 1 * hidden_size);
    const float4* r2 = reinterpret_cast<const float4*>(token_residual + 2 * hidden_size);
    const float4* r3 = reinterpret_cast<const float4*>(token_residual + 3 * hidden_size);

    float4 v0 = r0[tid];
    float4 v1 = r1[tid];
    float4 v2 = r2[tid];
    float4 v3 = r3[tid];

    float out0 = p0 * v0.x + p1 * v1.x + p2 * v2.x + p3 * v3.x;
    float out1 = p0 * v0.y + p1 * v1.y + p2 * v2.y + p3 * v3.y;
    float out2 = p0 * v0.z + p1 * v1.z + p2 * v2.z + p3 * v3.z;
    float out3 = p0 * v0.w + p1 * v1.w + p2 * v2.w + p3 * v3.w;

    half2 h01 = __floats2half2_rn(out0, out1);
    half2 h23 = __floats2half2_rn(out2, out3);

    half2* out_ptr = reinterpret_cast<half2*>(token_out + idx);
    out_ptr[0] = h01;
    out_ptr[1] = h23;
}

// GPU Wave32 Kernel for HC Sinkhorn and Pre/Post Mix calculation
// 1 block of 32 threads computes the Sinkhorn normalization of the 4x4 matrix
__global__ void __launch_bounds__(32) hc_sinkhorn_normalize_kernel(
    const float* __restrict__ mixes,      // [num_tokens, 24]
    const float* __restrict__ hc_scale,   // [3]
    const float* __restrict__ hc_base,    // [24]
    float* __restrict__ pre_mix,          // [num_tokens, 4]
    float* __restrict__ post_mix,         // [num_tokens, 4]
    float* __restrict__ comb_mix,         // [num_tokens, 16]
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_alpha,
    int sinkhorn_iters
) {
    int token_idx = blockIdx.x;
    int lane = threadIdx.x;

    const float* token_mixes = mixes + token_idx * 24;
    float* token_pre = pre_mix + token_idx * 4;
    float* token_post = post_mix + token_idx * 4;
    float* token_comb = comb_mix + token_idx * 16;

    __shared__ float s_cm[16];

    // Thread 0..3 compute pre_mix and post_mix
    if (lane < 4) {
        // Pre-mix
        float pre_val = token_mixes[lane] * hc_scale[0] + hc_base[lane];
        token_pre[lane] = (1.0f / (1.0f + expf(-pre_val))) + hc_pre_eps;

        // Post-mix
        float post_val = token_mixes[lane + 4] * hc_scale[1] + hc_base[lane + 4];
        token_post[lane] = (1.0f / (1.0f + expf(-post_val))) * hc_post_alpha;
    }

    // Thread 0..15 compute initial comb_mix logits
    if (lane < 16) {
        float cm_val = token_mixes[lane + 8] * hc_scale[2] + hc_base[lane + 8];
        s_cm[lane] = cm_val;
    }
    __syncthreads();

    // 4 threads (lane 0..3) handle the 4 rows of softmax
    if (lane < 4) {
        int row = lane;
        float r0 = s_cm[row * 4 + 0];
        float r1 = s_cm[row * 4 + 1];
        float r2 = s_cm[row * 4 + 2];
        float r3 = s_cm[row * 4 + 3];

        float max_v = fmaxf(fmaxf(r0, r1), fmaxf(r2, r3));
        float e0 = expf(r0 - max_v);
        float e1 = expf(r1 - max_v);
        float e2 = expf(r2 - max_v);
        float e3 = expf(r3 - max_v);
        float sum_e = e0 + e1 + e2 + e3;

        s_cm[row * 4 + 0] = (e0 / sum_e) + hc_sinkhorn_eps;
        s_cm[row * 4 + 1] = (e1 / sum_e) + hc_sinkhorn_eps;
        s_cm[row * 4 + 2] = (e2 / sum_e) + hc_sinkhorn_eps;
        s_cm[row * 4 + 3] = (e3 / sum_e) + hc_sinkhorn_eps;
    }
    __syncthreads();

    // Column normalization by 4 threads (lane 0..3)
    if (lane < 4) {
        int col = lane;
        float col_sum = s_cm[0 * 4 + col] + s_cm[1 * 4 + col] + s_cm[2 * 4 + col] + s_cm[3 * 4 + col];
        float inv_col = 1.0f / (col_sum + hc_sinkhorn_eps);
        s_cm[0 * 4 + col] *= inv_col;
        s_cm[1 * 4 + col] *= inv_col;
        s_cm[2 * 4 + col] *= inv_col;
        s_cm[3 * 4 + col] *= inv_col;
    }
    __syncthreads();

    // Sinkhorn iterations
    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
        // Row normalize
        if (lane < 4) {
            int row = lane;
            float row_sum = s_cm[row * 4 + 0] + s_cm[row * 4 + 1] + s_cm[row * 4 + 2] + s_cm[row * 4 + 3];
            float inv_row = 1.0f / (row_sum + hc_sinkhorn_eps);
            s_cm[row * 4 + 0] *= inv_row;
            s_cm[row * 4 + 1] *= inv_row;
            s_cm[row * 4 + 2] *= inv_row;
            s_cm[row * 4 + 3] *= inv_row;
        }
        __syncthreads();

        // Col normalize
        if (lane < 4) {
            int col = lane;
            float col_sum = s_cm[0 * 4 + col] + s_cm[1 * 4 + col] + s_cm[2 * 4 + col] + s_cm[3 * 4 + col];
            float inv_col = 1.0f / (col_sum + hc_sinkhorn_eps);
            s_cm[0 * 4 + col] *= inv_col;
            s_cm[1 * 4 + col] *= inv_col;
            s_cm[2 * 4 + col] *= inv_col;
            s_cm[3 * 4 + col] *= inv_col;
        }
        __syncthreads();
    }

    if (lane < 16) {
        token_comb[lane] = s_cm[lane];
    }
}

// Wave32 Kernel for HC Post expansion:
// residual_out[hco, h] = post_mix[hco] * layer_out[h] + sum_hci(comb_mix[hci, hco] * residual_in[hci, h])
__global__ void hc_post_kernel(
    const __half* __restrict__ layer_out,       // [num_tokens, hidden_size]
    const __half* __restrict__ residual_in,     // [num_tokens, 4, hidden_size]
    const float* __restrict__ post_mix,         // [num_tokens, 4]
    const float* __restrict__ comb_mix,         // [num_tokens, 16]
    __half* __restrict__ residual_out,          // [num_tokens, 4, hidden_size]
    int hidden_size
) {
    int token_idx = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (h < hidden_size) {
        float l_out = __half2float(layer_out[token_idx * hidden_size + h]);
        float r0 = __half2float(residual_in[token_idx * 4 * hidden_size + 0 * hidden_size + h]);
        float r1 = __half2float(residual_in[token_idx * 4 * hidden_size + 1 * hidden_size + h]);
        float r2 = __half2float(residual_in[token_idx * 4 * hidden_size + 2 * hidden_size + h]);
        float r3 = __half2float(residual_in[token_idx * 4 * hidden_size + 3 * hidden_size + h]);

        const float* pm = post_mix + token_idx * 4;
        const float* cm = comb_mix + token_idx * 16;

        #pragma unroll
        for (int hco = 0; hco < 4; ++hco) {
            float val = pm[hco] * l_out;
            val += cm[0 * 4 + hco] * r0;
            val += cm[1 * 4 + hco] * r1;
            val += cm[2 * 4 + hco] * r2;
            val += cm[3 * 4 + hco] * r3;
            residual_out[token_idx * 4 * hidden_size + hco * hidden_size + h] = __float2half(val);
        }
    }
}

} // namespace aeon::kernel
