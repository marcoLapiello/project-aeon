#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <cassert>

namespace aeon::kernel {

// Hyperparameters for DeepSeek-V4 Attention
constexpr uint32_t DSV4_HIDDEN_SIZE = 4096;
constexpr uint32_t DSV4_NUM_HEADS = 64;
constexpr uint32_t DSV4_HEAD_DIM = 512;
constexpr uint32_t DSV4_ROPE_DIM = 64;
constexpr uint32_t DSV4_NOPE_DIM = DSV4_HEAD_DIM - DSV4_ROPE_DIM; // 448
constexpr uint32_t DSV4_Q_LORA_RANK = 1024;
constexpr uint32_t DSV4_O_GROUPS = 8;
constexpr uint32_t DSV4_O_LORA_RANK = 1024;
constexpr uint32_t DSV4_HEADS_PER_GROUP = DSV4_NUM_HEADS / DSV4_O_GROUPS; // 8
constexpr uint32_t DSV4_GROUP_HEADS_DIM = DSV4_HEADS_PER_GROUP * DSV4_HEAD_DIM; // 4096
constexpr uint32_t DSV4_TOTAL_O_LORA_DIM = DSV4_O_GROUPS * DSV4_O_LORA_RANK; // 8192
constexpr uint32_t DSV4_SLIDING_WINDOW = 128;
constexpr float DSV4_ROPE_THETA = 10000.0f;
constexpr float DSV4_ATTN_SCALE = 0.04419417382415922f; // 1.0f / sqrtf(512.0f)

// ---------------------------------------------------------------------------
// RoPE and YaRN configuration
// ---------------------------------------------------------------------------
struct RopeTable {
    uint32_t max_seq_len{4096};
    uint32_t rope_dim{DSV4_ROPE_DIM};
    uint32_t half_rope{DSV4_ROPE_DIM / 2}; // 32
    std::vector<float> cos_cache; // [max_seq_len, half_rope]
    std::vector<float> sin_cache; // [max_seq_len, half_rope]

    void init(
        uint32_t seq_len = 4096,
        float theta = DSV4_ROPE_THETA,
        float factor = 1.0f,
        float beta_fast = 32.0f,
        float beta_slow = 1.0f,
        uint32_t orig_max_pos = 65536
    ) {
        max_seq_len = seq_len;
        cos_cache.resize(max_seq_len * half_rope);
        sin_cache.resize(max_seq_len * half_rope);

        for (uint32_t k = 0; k < half_rope; ++k) {
            float freq = 1.0f / std::pow(theta, (2.0f * k) / (float)rope_dim);
            if (factor > 1.0f) {
                // YaRN frequency correction
                float low = std::floor(orig_max_pos / (2.0f * M_PI * std::pow(theta, (2.0f * (half_rope - 1)) / (float)rope_dim)));
                float high = std::ceil(orig_max_pos / (2.0f * M_PI * std::pow(theta, 0.0f)));
                float w = 0.0f;
                if (k < low) w = 0.0f;
                else if (k > high) w = 1.0f;
                else w = (k - low) / (high - low + 1e-5f);
                freq = (1.0f - w) * (freq / factor) + w * freq;
            }

            for (uint32_t pos = 0; pos < max_seq_len; ++pos) {
                float angle = pos * freq;
                cos_cache[pos * half_rope + k] = std::cos(angle);
                sin_cache[pos * half_rope + k] = std::sin(angle);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Device Kernels (RDNA3 Wave32 Optimized)
// ---------------------------------------------------------------------------

// 1. Wave32 RMSNorm kernel: 1 warp (32 threads) per token row
__global__ void __launch_bounds__(32) v4_rmsnorm_wave32_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int dim,
    float eps
) {
    int lane = threadIdx.x; // 0..31
    int row = blockIdx.x;

    const __half* in_row = input + row * dim;
    __half* out_row = output + row * dim;

    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float v = __half2float(in_row[i]);
        sum_sq += v * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor(sum_sq, offset, 32);
    }

    float inv_rms = rsqrtf((sum_sq / (float)dim) + eps);

    for (int i = lane; i < dim; i += 32) {
        float v = __half2float(in_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(v * inv_rms * w);
    }
}

__global__ void __launch_bounds__(32) v4_rmsnorm_unit_wave32_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int dim,
    float eps
) {
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const __half* in_row = input + row * dim;
    __half* out_row = output + row * dim;

    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        const float value = __half2float(in_row[i]);
        sum_sq += value * value;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor(sum_sq, offset, 32);
    }

    const float inv_rms = rsqrtf((sum_sq / static_cast<float>(dim)) + eps);
    for (int i = lane; i < dim; i += 32) {
        out_row[i] = __float2half(__half2float(in_row[i]) * inv_rms);
    }
}

// 2. Wave32 Forward GPT-J RoPE on trailing 64 elements of [T, num_heads, head_dim] or [T, 1, head_dim]
// Interleaved layout: for k in 0..31: out[2k] = x[2k]*cos - x[2k+1]*sin, out[2k+1] = x[2k]*sin + x[2k+1]*cos
__global__ void __launch_bounds__(32) v4_forward_rope_wave32_kernel(
    __half* __restrict__ vec,           // [num_tokens, num_heads, head_dim]
    const float* __restrict__ cos_cache,// [max_seq, 32]
    const float* __restrict__ sin_cache,// [max_seq, 32]
    int num_heads,
    int head_dim,                       // 512
    int nope_dim,                       // 448
    int half_rope                       // 32
) {
    int token_idx = blockIdx.y;
    int head_idx  = blockIdx.x;
    int k         = threadIdx.x; // 0..31 (one thread per frequency pair)

    if (k < half_rope) {
        int base_idx = token_idx * (num_heads * head_dim) + head_idx * head_dim + nope_dim + 2 * k;

        float c = cos_cache[token_idx * half_rope + k];
        float s = sin_cache[token_idx * half_rope + k];

        float x0 = __half2float(vec[base_idx + 0]);
        float x1 = __half2float(vec[base_idx + 1]);

        float rot0 = x0 * c - x1 * s;
        float rot1 = x0 * s + x1 * c;

        vec[base_idx + 0] = __float2half(rot0);
        vec[base_idx + 1] = __float2half(rot1);
    }
}

// 3. Wave32 Inverse GPT-J RoPE on trailing 64 elements of attention output [T, 64, 512]
// Inverse applies negative sin: out[2k] = x[2k]*cos + x[2k+1]*sin, out[2k+1] = x[2k+1]*cos - x[2k]*sin
__global__ void __launch_bounds__(32) v4_inverse_rope_wave32_kernel(
    __half* __restrict__ vec,           // [num_tokens, 64, 512]
    const float* __restrict__ cos_cache,// [max_seq, 32]
    const float* __restrict__ sin_cache,// [max_seq, 32]
    int num_heads,                      // 64
    int head_dim,                       // 512
    int nope_dim,                       // 448
    int half_rope                       // 32
) {
    int token_idx = blockIdx.y;
    int head_idx  = blockIdx.x;
    int k         = threadIdx.x; // 0..31

    if (k < half_rope) {
        int base_idx = token_idx * (num_heads * head_dim) + head_idx * head_dim + nope_dim + 2 * k;

        float c = cos_cache[token_idx * half_rope + k];
        float s = sin_cache[token_idx * half_rope + k];

        float x0 = __half2float(vec[base_idx + 0]);
        float x1 = __half2float(vec[base_idx + 1]);

        float inv0 = x0 * c + x1 * s;
        float inv1 = x1 * c - x0 * s;

        vec[base_idx + 0] = __float2half(inv0);
        vec[base_idx + 1] = __float2half(inv1);
    }
}

// 4. Causal Sliding-Window Attention Kernel with Attention Sink (Wave32)
// Q: [num_tokens, 64, 512]
// K: [num_tokens, 512] (Single KV head shared across all 64 Q heads)
// Out: [num_tokens, 64, 512]
// sink: [64] (float per head)
// Window size: W = 128
__global__ void __launch_bounds__(32) v4_sliding_window_attn_wave32_kernel(
    const __half* __restrict__ q,       // [T, 64, 512]
    const __half* __restrict__ k,       // [T, 512]
    const float*  __restrict__ attn_sink,// [64]
    __half*       __restrict__ out,     // [T, 64, 512]
    int total_tokens,
    int window_size,                    // 128
    float scale                         // 1.0f / sqrt(512)
) {
    int head = blockIdx.x;              // 0..63
    int token = blockIdx.y;             // 0..total_tokens-1
    int lane = threadIdx.x;             // 0..31

    // LDS storage for up to 128 attention scores in the sliding window
    __shared__ float lds_scores[DSV4_SLIDING_WINDOW];

    int j_start = max(0, token - window_size + 1);
    int num_keys = token - j_start + 1;

    const __half* q_ptr = q + token * (DSV4_NUM_HEADS * DSV4_HEAD_DIM) + head * DSV4_HEAD_DIM;

    // Phase 1: Compute scaled dot products Q_i * K_j for each key j in sliding window
    for (int step = 0; step < num_keys; ++step) {
        int j = j_start + step;
        const __half* k_ptr = k + j * DSV4_HEAD_DIM;

        float dot = 0.0f;
        // Each of the 32 threads computes 512 / 32 = 16 elements
        #pragma unroll 4
        for (int d = lane * 16; d < (lane + 1) * 16; ++d) {
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }

        if (lane == 0) {
            lds_scores[step] = dot * scale;
        }
    }
    __syncthreads();

    // Phase 2: Softmax with Attention Sink
    // Find maximum among all key scores and the head attention sink
    float max_score = attn_sink[head];
    for (int step = 0; step < num_keys; ++step) {
        max_score = fmaxf(max_score, lds_scores[step]);
    }

    // Compute denominator: sum of exp(score - max) + exp(sink - max)
    float sink_weight = expf(attn_sink[head] - max_score);
    float sum_exp = sink_weight;

    for (int step = 0; step < num_keys; ++step) {
        float p = expf(lds_scores[step] - max_score);
        lds_scores[step] = p; // Store unnormalized exp
        sum_exp += p;
    }

    float inv_sum = 1.0f / fmaxf(sum_exp, 1e-30f);
    __syncthreads();

    // Phase 3: Weighted sum of Value vectors (V = K in DeepSeek MLA)
    // Note: The attention sink contributes only to the denominator, absorbing probability mass!
    __half* out_ptr = out + token * (DSV4_NUM_HEADS * DSV4_HEAD_DIM) + head * DSV4_HEAD_DIM;

    #pragma unroll 4
    for (int d = lane * 16; d < (lane + 1) * 16; ++d) {
        float acc = 0.0f;
        for (int step = 0; step < num_keys; ++step) {
            int j = j_start + step;
            float weight = lds_scores[step] * inv_sum;
            acc += weight * __half2float(k[j * DSV4_HEAD_DIM + d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}

// 5. Grouped W_o_a Projection Kernel:
// For each group g in 0..7: input is 8 heads x 512 = 4096 half elements.
// Projected by W_o_a[g]: [1024, 4096] -> Z[g]: [1024]
// Total output is [T, 8192]
__global__ void v4_grouped_wo_a_wave32_kernel(
    const __half* __restrict__ attn_out, // [T, 64, 512] -> viewed as [T, 8, 4096]
    const __half* __restrict__ wo_a_weight,// [8, 1024, 4096]
    __half*       __restrict__ z_out,    // [T, 8, 1024] = [T, 8192]
    int total_tokens
) {
    int out_col = blockIdx.x; // 0..1023
    int group   = blockIdx.y; // 0..7
    int token   = blockIdx.z; // 0..total_tokens-1
    int lane    = threadIdx.x;// 0..31

    const __half* in_vec = attn_out + token * (DSV4_O_GROUPS * DSV4_GROUP_HEADS_DIM) + group * DSV4_GROUP_HEADS_DIM;
    const __half* w_row  = wo_a_weight + group * (DSV4_O_LORA_RANK * DSV4_GROUP_HEADS_DIM) + out_col * DSV4_GROUP_HEADS_DIM;

    float dot = 0.0f;
    // Each thread processes 4096 / 32 = 128 elements
    #pragma unroll 4
    for (int i = lane; i < (int)DSV4_GROUP_HEADS_DIM; i += 32) {
        dot += __half2float(in_vec[i]) * __half2float(w_row[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_xor(dot, offset, 32);
    }

    if (lane == 0) {
        z_out[token * DSV4_TOTAL_O_LORA_DIM + group * DSV4_O_LORA_RANK + out_col] = __float2half(dot);
    }
}

// 6. Fast GEMV kernel for FP16 Linear Projections (Dense 1D matrix-vector product per token)
// Computes Y = X @ W.T where W is [out_dim, in_dim]
__global__ void v4_gemv_fp16_kernel(
    const __half* __restrict__ x,       // [T, in_dim]
    const __half* __restrict__ w,       // [out_dim, in_dim]
    __half*       __restrict__ y,       // [T, out_dim]
    int in_dim
) {
    int out_col = blockIdx.x;           // row of W, output feature index
    int token   = blockIdx.y;           // token index
    int lane    = threadIdx.x;          // 0..31

    const __half* x_row = x + token * in_dim;
    const __half* w_row = w + out_col * in_dim;

    float dot = 0.0f;
    for (int i = lane; i < in_dim; i += 32) {
        dot += __half2float(x_row[i]) * __half2float(w_row[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_xor(dot, offset, 32);
    }

    if (lane == 0) {
        y[token * gridDim.x + out_col] = __float2half(dot);
    }
}

// Vectorized FP16 GEMV (Expert Review Step 5): each lane streams 8 halves per
// iteration via uint4 (vs 1 half in v4_gemv_fp16_kernel) — 8x fewer global
// transactions and FP32 FMA accumulation. Used for the shared-expert and LM
// head projections; requires in_dim % 8 == 0 (satisfied by H=4096/2048).
__global__ void __launch_bounds__(32) v4_gemv_fp16_vec8_kernel(
    const __half* __restrict__ x,       // [T, in_dim]
    const __half* __restrict__ w,       // [out_dim, in_dim]
    __half*       __restrict__ y,       // [T, out_dim]
    int in_dim
) {
    int out_col = blockIdx.x;
    int token   = blockIdx.y;
    int lane    = threadIdx.x;

    const uint4* x_row = reinterpret_cast<const uint4*>(x + token * in_dim);
    const uint4* w_row = reinterpret_cast<const uint4*>(w + out_col * in_dim);
    const int vec_dim = in_dim >> 3; // 8 halves per uint4

    float dot = 0.0f;
    for (int i = lane; i < vec_dim; i += 32) {
        const uint4 xv = __ldg(x_row + i);
        const uint4 wv = __ldg(w_row + i);
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        const __half2* wh = reinterpret_cast<const __half2*>(&wv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 xf = __half22float2(xh[j]);
            const float2 wf = __half22float2(wh[j]);
            dot = fmaf(xf.x, wf.x, dot);
            dot = fmaf(xf.y, wf.y, dot);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_xor(dot, offset, 32);
    }

    if (lane == 0) {
        y[token * gridDim.x + out_col] = __float2half(dot);
    }
}

// Half -> float elementwise conversion (replaces the per-layer D2H/sync/CPU/
// H2D router-logits round-trip with a single device kernel).
__global__ void v4_half_to_float_n_kernel(
    const __half* __restrict__ in,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// GPU argmax over the [129280] FP16 logit head (Expert Review Step 5).
// Phase 1 and Phase 2 are separate launches because HIP has no implicit
// grid-wide barrier between blocks in one ordinary kernel launch.
constexpr int V4_ARGMAX_BLOCKS = 505;

__global__ void __launch_bounds__(256) v4_argmax_fp16_partial_kernel(
    const __half* __restrict__ logits,
    int n,
    float* __restrict__ partial_vals,   // [V4_ARGMAX_BLOCKS]
    int32_t* __restrict__ partial_idx   // [V4_ARGMAX_BLOCKS]
) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];

    const int tid = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;

    float best = -INFINITY;
    int best_i = n; // sentinel: larger than any real index so real winners beat it
    for (int i = gtid; i < n; i += stride) {
        float v = __half2float(logits[i]);
        if (v > best) { best = v; best_i = i; }
    }

    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();

    // Block reduction: on ties keep the smaller index (first-max-wins)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = s_val[tid + s];
            int   oi = s_idx[tid + s];
            if (ov > s_val[tid] || (ov == s_val[tid] && oi < s_idx[tid])) {
                s_val[tid] = ov;
                s_idx[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_vals[blockIdx.x] = s_val[0];
        partial_idx[blockIdx.x]  = s_idx[0];
    }
}

__global__ void __launch_bounds__(256) v4_argmax_partial_reduce_kernel(
    const float* __restrict__ partial_vals,
    const int32_t* __restrict__ partial_idx,
    int partial_count,
    int32_t* __restrict__ out_idx
) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];

    const int tid = threadIdx.x;
    float best = -INFINITY;
    int best_i = INT_MAX;
    for (int i = tid; i < partial_count; i += blockDim.x) {
        const float value = partial_vals[i];
        const int index = partial_idx[i];
        if (value > best || (value == best && index < best_i)) {
            best = value;
            best_i = index;
        }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_value = s_val[tid + stride];
            const int other_index = s_idx[tid + stride];
            if (other_value > s_val[tid] ||
                (other_value == s_val[tid] && other_index < s_idx[tid])) {
                s_val[tid] = other_value;
                s_idx[tid] = other_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_idx[0] = s_idx[0];
    }
}
__global__ void __launch_bounds__(32) v4_forward_rope_at_pos_wave32_kernel(
    __half* __restrict__ vec,           // [num_heads, head_dim]
    const float* __restrict__ cos_cache,// [max_seq, 32]
    const float* __restrict__ sin_cache,// [max_seq, 32]
    int pos,
    int num_heads,
    int head_dim,                       // 512
    int nope_dim,                       // 448
    int half_rope                       // 32
) {
    int head_idx = blockIdx.x;
    int k        = threadIdx.x; // 0..31

    if (k < half_rope) {
        int base_idx = head_idx * head_dim + nope_dim + 2 * k;

        float c = cos_cache[pos * half_rope + k];
        float s = sin_cache[pos * half_rope + k];

        float x0 = __half2float(vec[base_idx + 0]);
        float x1 = __half2float(vec[base_idx + 1]);

        float rot0 = x0 * c - x1 * s;
        float rot1 = x0 * s + x1 * c;

        vec[base_idx + 0] = __float2half(rot0);
        vec[base_idx + 1] = __float2half(rot1);
    }
}

// 8. Single-position Inverse RoPE for autoregressive generation
__global__ void __launch_bounds__(32) v4_inverse_rope_at_pos_wave32_kernel(
    __half* __restrict__ vec,           // [num_heads, head_dim]
    const float* __restrict__ cos_cache,// [max_seq, 32]
    const float* __restrict__ sin_cache,// [max_seq, 32]
    int pos,
    int num_heads,
    int head_dim,                       // 512
    int nope_dim,                       // 448
    int half_rope                       // 32
) {
    int head_idx = blockIdx.x;
    int k        = threadIdx.x; // 0..31

    if (k < half_rope) {
        int base_idx = head_idx * head_dim + nope_dim + 2 * k;

        float c = cos_cache[pos * half_rope + k];
        float s = sin_cache[pos * half_rope + k];

        float x0 = __half2float(vec[base_idx + 0]);
        float x1 = __half2float(vec[base_idx + 1]);

        float inv0 = x0 * c + x1 * s;
        float inv1 = x1 * c - x0 * s;

        vec[base_idx + 0] = __float2half(inv0);
        vec[base_idx + 1] = __float2half(inv1);
    }
}

// 9. Autoregressive Sliding-Window Attention with persistent KV Cache
__global__ void __launch_bounds__(32) v4_cached_sliding_window_attn_wave32_kernel(
    const __half* __restrict__ q,         // [64, 512]
    const __half* __restrict__ kv_cache,  // [max_seq_len, 512]
    const float*  __restrict__ attn_sink, // [64]
    __half*       __restrict__ out,       // [64, 512]
    int current_pos,                      // sequence index (0, 1, 2, ...)
    int window_size,                      // 128
    float scale                           // 1.0f / sqrt(512)
) {
    int head = blockIdx.x;                // 0..63
    int lane = threadIdx.x;               // 0..31

    __shared__ float lds_scores[DSV4_SLIDING_WINDOW];

    int j_start = max(0, current_pos - window_size + 1);
    int num_keys = current_pos - j_start + 1;

    const __half* q_ptr = q + head * DSV4_HEAD_DIM;

    // Phase 1: Dot products with cached keys
    for (int step = 0; step < num_keys; ++step) {
        int j = j_start + step;
        const __half* k_ptr = kv_cache + j * DSV4_HEAD_DIM;

        float dot = 0.0f;
        #pragma unroll 4
        for (int d = lane * 16; d < (lane + 1) * 16; ++d) {
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }

        if (lane == 0) {
            lds_scores[step] = dot * scale;
        }
    }
    __syncthreads();

    // Phase 2: Softmax with attention sink
    float max_score = attn_sink[head];
    for (int step = 0; step < num_keys; ++step) {
        max_score = fmaxf(max_score, lds_scores[step]);
    }

    float sink_weight = expf(attn_sink[head] - max_score);
    float sum_exp = sink_weight;

    for (int step = 0; step < num_keys; ++step) {
        float p = expf(lds_scores[step] - max_score);
        lds_scores[step] = p;
        sum_exp += p;
    }

    float inv_sum = 1.0f / fmaxf(sum_exp, 1e-30f);
    __syncthreads();

    // Phase 3: Weighted sum of Value vectors (V = K)
    __half* out_ptr = out + head * DSV4_HEAD_DIM;

    #pragma unroll 4
    for (int d = lane * 16; d < (lane + 1) * 16; ++d) {
        float acc = 0.0f;
        for (int step = 0; step < num_keys; ++step) {
            int j = j_start + step;
            float weight = lds_scores[step] * inv_sum;
            acc += weight * __half2float(kv_cache[j * DSV4_HEAD_DIM + d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}

// 10. Hyper-Connections Head Reduction Kernel
__global__ void __launch_bounds__(32) hc_head_wave32_kernel(
    const float* __restrict__ residual_in, // [4, 4096] = 16384 floats
    const float* __restrict__ hc_head_fn,  // [4, 16384] floats
    const float* __restrict__ hc_head_base,// [4] floats
    const float* __restrict__ hc_head_scale,// [1] float
    __half*      __restrict__ out,         // [4096] half
    int hidden_dim,                        // 4096
    int hc_mult,                           // 4
    float rms_eps,                         // 1e-6f
    float hc_eps                           // 1e-6f
) {
    int total_hc_dim = hc_mult * hidden_dim; // 16384
    int lane = threadIdx.x; // 0..31

    // Step 1: Mean square over total_hc_dim
    float sum_sq = 0.0f;
    for (int i = lane; i < total_hc_dim; i += 32) {
        float v = residual_in[i];
        sum_sq += v * v;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor(sum_sq, offset, 32);
    }
    float rsqrt = rsqrtf((sum_sq / (float)total_hc_dim) + rms_eps);

    // Step 2: Linear projection for each of the 4 streams
    __shared__ float s_pre[4];
    for (int s = 0; s < hc_mult; ++s) {
        float dot = 0.0f;
        const float* fn_row = hc_head_fn + s * total_hc_dim;
        for (int i = lane; i < total_hc_dim; i += 32) {
            dot += residual_in[i] * fn_row[i];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }
        if (lane == 0) {
            float mix = dot * rsqrt;
            float val = mix * hc_head_scale[0] + hc_head_base[s];
            float pre = (1.0f / (1.0f + expf(-val))) + hc_eps;
            s_pre[s] = pre;
        }
    }
    __syncthreads();

    // Step 3: Combine streams into output [4096]
    for (int h = lane; h < hidden_dim; h += 32) {
        float acc = 0.0f;
        for (int s = 0; s < hc_mult; ++s) {
            acc += s_pre[s] * residual_in[s * hidden_dim + h];
        }
        out[h] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementations for Precision Verification
// ---------------------------------------------------------------------------
inline void cpu_rmsnorm(
    const float* input,
    const float* weight,
    float* output,
    int dim,
    float eps = 1e-6f
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum_sq += input[i] * input[i];
    }
    float inv_rms = 1.0f / std::sqrt((sum_sq / (float)dim) + eps);
    for (int i = 0; i < dim; ++i) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

inline void cpu_forward_rope(
    float* vec,
    int token_idx,
    const RopeTable& rope,
    int num_heads = DSV4_NUM_HEADS,
    int head_dim = DSV4_HEAD_DIM,
    int nope_dim = DSV4_NOPE_DIM,
    int half_rope = DSV4_ROPE_DIM / 2
) {
    for (int h = 0; h < num_heads; ++h) {
        for (int k = 0; k < half_rope; ++k) {
            int base_idx = h * head_dim + nope_dim + 2 * k;
            float c = rope.cos_cache[token_idx * half_rope + k];
            float s = rope.sin_cache[token_idx * half_rope + k];

            float x0 = vec[base_idx + 0];
            float x1 = vec[base_idx + 1];

            vec[base_idx + 0] = x0 * c - x1 * s;
            vec[base_idx + 1] = x0 * s + x1 * c;
        }
    }
}

inline void cpu_inverse_rope(
    float* vec,
    int token_idx,
    const RopeTable& rope,
    int num_heads = DSV4_NUM_HEADS,
    int head_dim = DSV4_HEAD_DIM,
    int nope_dim = DSV4_NOPE_DIM,
    int half_rope = DSV4_ROPE_DIM / 2
) {
    for (int h = 0; h < num_heads; ++h) {
        for (int k = 0; k < half_rope; ++k) {
            int base_idx = h * head_dim + nope_dim + 2 * k;
            float c = rope.cos_cache[token_idx * half_rope + k];
            float s = rope.sin_cache[token_idx * half_rope + k];

            float x0 = vec[base_idx + 0];
            float x1 = vec[base_idx + 1];

            vec[base_idx + 0] = x0 * c + x1 * s;
            vec[base_idx + 1] = x1 * c - x0 * s;
        }
    }
}

inline void cpu_sliding_window_attention(
    const std::vector<float>& all_q,    // [T, 64, 512]
    const std::vector<float>& all_k,    // [T, 512]
    const std::vector<float>& attn_sink,// [64]
    std::vector<float>& all_out,        // [T, 64, 512]
    int total_tokens,
    int window_size = DSV4_SLIDING_WINDOW,
    float scale = DSV4_ATTN_SCALE
) {
    all_out.assign(total_tokens * DSV4_NUM_HEADS * DSV4_HEAD_DIM, 0.0f);

    for (int t = 0; t < total_tokens; ++t) {
        int j_start = std::max(0, t - window_size + 1);
        int num_keys = t - j_start + 1;

        for (int h = 0; h < (int)DSV4_NUM_HEADS; ++h) {
            const float* q_ptr = all_q.data() + t * (DSV4_NUM_HEADS * DSV4_HEAD_DIM) + h * DSV4_HEAD_DIM;

            std::vector<float> scores(num_keys);
            float max_score = attn_sink[h];

            for (int step = 0; step < num_keys; ++step) {
                int j = j_start + step;
                const float* k_ptr = all_k.data() + j * DSV4_HEAD_DIM;

                float dot = 0.0f;
                for (int d = 0; d < (int)DSV4_HEAD_DIM; ++d) {
                    dot += q_ptr[d] * k_ptr[d];
                }
                scores[step] = dot * scale;
                max_score = std::max(max_score, scores[step]);
            }

            float sum_exp = std::exp(attn_sink[h] - max_score);
            for (int step = 0; step < num_keys; ++step) {
                scores[step] = std::exp(scores[step] - max_score);
                sum_exp += scores[step];
            }

            float inv_sum = 1.0f / std::max(sum_exp, 1e-30f);

            float* out_ptr = all_out.data() + t * (DSV4_NUM_HEADS * DSV4_HEAD_DIM) + h * DSV4_HEAD_DIM;
            for (int d = 0; d < (int)DSV4_HEAD_DIM; ++d) {
                float acc = 0.0f;
                for (int step = 0; step < num_keys; ++step) {
                    int j = j_start + step;
                    acc += (scores[step] * inv_sum) * all_k[j * DSV4_HEAD_DIM + d];
                }
                out_ptr[d] = acc;
            }
        }
    }
}

} // namespace aeon::kernel
