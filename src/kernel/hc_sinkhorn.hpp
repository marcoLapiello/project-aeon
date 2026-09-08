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

// CPU Reference for HC Sinkhorn and Pre/Post projection
inline void cpu_sinkhorn_and_mix(
    const float* residual_in,       // [hc_mult, hidden_size]
    const float* fn,                // [24, 16384]
    const float* base,              // [24]
    const float* scale,             // [3]
    float* layer_input,             // [hidden_size]
    float* post_mix,                // [hc_mult]
    float* comb_mix,                // [hc_mult, hc_mult]
    int hidden_size = 4096,
    int hc_mult = 4,
    float rms_eps = 1e-6f,
    float hc_pre_eps = 1e-6f,
    float hc_sinkhorn_eps = 1e-6f,
    float hc_post_alpha = 2.0f,
    int sinkhorn_iters = 20
) {
    int hc_hidden_size = hc_mult * hidden_size;
    int hc_mult3 = hc_mult * (2 + hc_mult); // 24

    // 1. RMS of flattened residual
    float sqrsum = 0.0f;
    for (int i = 0; i < hc_hidden_size; ++i) {
        sqrsum += residual_in[i] * residual_in[i];
    }
    float rms = 1.0f / std::sqrt((sqrsum / (float)hc_hidden_size) + rms_eps);

    // 2. Linear projection: mixes = (residual_flat @ fn.T) * rms
    std::vector<float> mixes(hc_mult3, 0.0f);
    for (int j = 0; j < hc_mult3; ++j) {
        float dot = 0.0f;
        for (int k = 0; k < hc_hidden_size; ++k) {
            dot += residual_in[k] * fn[j * hc_hidden_size + k];
        }
        mixes[j] = dot * rms;
    }

    // 3. Pre mix: sigmoid(mixes[0..3] * scale[0] + base[0..3]) + pre_eps
    std::vector<float> pre_mix(hc_mult);
    for (int j = 0; j < hc_mult; ++j) {
        float val = mixes[j] * scale[0] + base[j];
        pre_mix[j] = (1.0f / (1.0f + std::exp(-val))) + hc_pre_eps;
    }

    // 4. Layer input = sum_j (pre_mix[j] * residual[j, :])
    for (int h = 0; h < hidden_size; ++h) {
        float acc = 0.0f;
        for (int j = 0; j < hc_mult; ++j) {
            acc += pre_mix[j] * residual_in[j * hidden_size + h];
        }
        layer_input[h] = acc;
    }

    // 5. Post mix: sigmoid(mixes[4..7] * scale[1] + base[4..7]) * alpha
    for (int j = 0; j < hc_mult; ++j) {
        float val = mixes[j + hc_mult] * scale[1] + base[j + hc_mult];
        post_mix[j] = (1.0f / (1.0f + std::exp(-val))) * hc_post_alpha;
    }

    // 6. Comb mix: initial logits from mixes[8..23] * scale[2] + base[8..23]
    std::vector<float> cm(hc_mult * hc_mult);
    for (int j = 0; j < hc_mult; ++j) {
        for (int k = 0; k < hc_mult; ++k) {
            int idx = j * hc_mult + k + hc_mult * 2;
            cm[j * hc_mult + k] = mixes[idx] * scale[2] + base[idx];
        }
    }

    // 7. Sinkhorn normalization:
    // Softmax along rows + sinkhorn_eps
    for (int j = 0; j < hc_mult; ++j) {
        float row_max = cm[j * hc_mult + 0];
        for (int k = 1; k < hc_mult; ++k) {
            if (cm[j * hc_mult + k] > row_max) row_max = cm[j * hc_mult + k];
        }
        float sum_exp = 0.0f;
        for (int k = 0; k < hc_mult; ++k) {
            cm[j * hc_mult + k] = std::exp(cm[j * hc_mult + k] - row_max);
            sum_exp += cm[j * hc_mult + k];
        }
        for (int k = 0; k < hc_mult; ++k) {
            cm[j * hc_mult + k] = (cm[j * hc_mult + k] / sum_exp) + hc_sinkhorn_eps;
        }
    }

    // Normalize columns
    for (int k = 0; k < hc_mult; ++k) {
        float col_sum = 0.0f;
        for (int j = 0; j < hc_mult; ++j) {
            col_sum += cm[j * hc_mult + k];
        }
        for (int j = 0; j < hc_mult; ++j) {
            cm[j * hc_mult + k] /= (col_sum + hc_sinkhorn_eps);
        }
    }

    // Iterations (sinkhorn_iters - 1)
    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
        // Row normalize
        for (int j = 0; j < hc_mult; ++j) {
            float row_sum = 0.0f;
            for (int k = 0; k < hc_mult; ++k) row_sum += cm[j * hc_mult + k];
            for (int k = 0; k < hc_mult; ++k) cm[j * hc_mult + k] /= (row_sum + hc_sinkhorn_eps);
        }
        // Column normalize
        for (int k = 0; k < hc_mult; ++k) {
            float col_sum = 0.0f;
            for (int j = 0; j < hc_mult; ++j) col_sum += cm[j * hc_mult + k];
            for (int j = 0; j < hc_mult; ++j) cm[j * hc_mult + k] /= (col_sum + hc_sinkhorn_eps);
        }
    }

    for (int i = 0; i < hc_mult * hc_mult; ++i) {
        comb_mix[i] = cm[i];
    }
}

// CPU Reference for HC Post
inline void cpu_hc_post(
    const float* layer_out,       // [hidden_size]
    const float* residual_in,     // [hc_mult, hidden_size]
    const float* post_mix,        // [hc_mult]
    const float* comb_mix,        // [hc_mult, hc_mult]
    float* residual_out,          // [hc_mult, hidden_size]
    int hidden_size = 4096,
    int hc_mult = 4
) {
    // x_local[i_hco, i1_h] = post_mix[i_hco] * layer_out[i1_h] + sum_hci(comb_mix[i_hci, i_hco] * residual[i_hci, i1_h])
    for (int hco = 0; hco < hc_mult; ++hco) {
        for (int h = 0; h < hidden_size; ++h) {
            float val = post_mix[hco] * layer_out[h];
            for (int hci = 0; hci < hc_mult; ++hci) {
                val += comb_mix[hci * hc_mult + hco] * residual_in[hci * hidden_size + h];
            }
            residual_out[hco * hidden_size + h] = val;
        }
    }
}

__global__ void __launch_bounds__(32) hc_project_kernel(
    const float* __restrict__ residual_in,
    const float* __restrict__ fn,
    float* __restrict__ mixes,
    int hidden_size,
    int hc_mult,
    float rms_eps
) {
    int lane = threadIdx.x;
    int hc_hidden_size = hc_mult * hidden_size;
    int mix_count = hc_mult * (2 + hc_mult);

    float sqrsum = 0.0f;
    for (int i = lane; i < hc_hidden_size; i += 32) {
        float value = residual_in[i];
        sqrsum += value * value;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sqrsum += __shfl_xor(sqrsum, offset, 32);
    }

    float rms = rsqrtf((sqrsum / static_cast<float>(hc_hidden_size)) + rms_eps);
    for (int mix_idx = 0; mix_idx < mix_count; ++mix_idx) {
        float dot = 0.0f;
        const float* fn_row = fn + mix_idx * hc_hidden_size;
        for (int i = lane; i < hc_hidden_size; i += 32) {
            dot += residual_in[i] * fn_row[i];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }

        if (lane == 0) {
            mixes[mix_idx] = dot * rms;
        }
    }
}

__global__ void hc_pre_combine_kernel(
    const float* __restrict__ residual_in,
    const float* __restrict__ pre_mix,
    __half* __restrict__ layer_input,
    int hidden_size,
    int hc_mult
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden_size) return;

    float value = 0.0f;
    for (int stream = 0; stream < hc_mult; ++stream) {
        value += pre_mix[stream] * residual_in[stream * hidden_size + h];
    }
    layer_input[h] = __float2half(value);
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
