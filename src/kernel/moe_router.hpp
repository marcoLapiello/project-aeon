#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace aeon::kernel {

// DeepSeek-V4 MoE Router Parameters:
// n_routed_experts = 256
// top_k = 6
// routed_scaling_factor = 1.5f

__device__ inline float softplus_sqrt(float x) {
    // Numerically stable sqrt(softplus(x))
    // softplus(x) = log(1 + exp(x))
    if (x > 20.0f) {
        return sqrtf(x);
    } else if (x < -20.0f) {
        return sqrtf(expf(x));
    } else {
        return sqrtf(log1pf(expf(x)));
    }
}

// CPU Reference Router Implementation
inline void cpu_moe_router(
    const float* router_logits,       // [num_tokens, n_routed_experts]
    const float* bias,                // [n_routed_experts] (optional, for layers >= num_hash_layers)
    const int64_t* hash_table,        // [vocab_size, top_k] (optional, for layers < num_hash_layers)
    const int32_t* token_ids,         // [num_tokens] (used if hash_table != nullptr)
    float* topk_weights,              // [num_tokens, top_k]
    int32_t* topk_indices,            // [num_tokens, top_k]
    int num_tokens,
    int n_routed_experts = 256,
    int top_k = 6,
    float routed_scaling_factor = 1.5f,
    bool renormalize = true
) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* logits = router_logits + t * n_routed_experts;
        float* out_weights = topk_weights + t * top_k;
        int32_t* out_indices = topk_indices + t * top_k;

        std::vector<float> scores(n_routed_experts);
        std::vector<float> scores_for_choice(n_routed_experts);

        for (int e = 0; e < n_routed_experts; ++e) {
            float x = logits[e];
            float sp;
            if (x > 20.0f) sp = x;
            else if (x < -20.0f) sp = std::exp(x);
            else sp = std::log1p(std::exp(x));
            scores[e] = std::sqrt(sp);

            float b = (bias != nullptr) ? bias[e] : 0.0f;
            scores_for_choice[e] = scores[e] + b;
        }

        if (hash_table != nullptr && token_ids != nullptr) {
            // Hash routing: indices are pre-determined by token ID
            int32_t tid = token_ids[t];
            const int64_t* entry = hash_table + tid * top_k;
            for (int k = 0; k < top_k; ++k) {
                out_indices[k] = static_cast<int32_t>(entry[k]);
                out_weights[k] = scores[out_indices[k]];
            }
        } else {
            // Top-k based on scores_for_choice
            std::vector<int> idx(n_routed_experts);
            for (int i = 0; i < n_routed_experts; ++i) idx[i] = i;

            std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                [&](int a, int b) {
                    return scores_for_choice[a] > scores_for_choice[b];
                });

            for (int k = 0; k < top_k; ++k) {
                out_indices[k] = idx[k];
                out_weights[k] = scores[idx[k]];
            }
        }

        // Renormalization
        if (renormalize) {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k) sum += out_weights[k];
            float inv_sum = 1.0f / (sum + 1e-20f);
            for (int k = 0; k < top_k; ++k) out_weights[k] *= inv_sum;
        }

        // Routed scaling factor
        for (int k = 0; k < top_k; ++k) {
            out_weights[k] *= routed_scaling_factor;
        }
    }
}

// Wave32 MoE Router Kernel on Silicon
// 1 block of 64 threads per token (2 Wave32 warps process 256 experts: 4 experts per thread)
__global__ void __launch_bounds__(64) moe_router_kernel(
    const float* __restrict__ router_logits,
    const float* __restrict__ bias,
    const int64_t* __restrict__ hash_table,
    const int32_t* __restrict__ token_ids,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_indices,
    int n_routed_experts,
    int top_k,
    float routed_scaling_factor,
    bool renormalize
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x; // 0..63

    const float* logits = router_logits + token_idx * n_routed_experts;
    float* out_weights = topk_weights + token_idx * top_k;
    int32_t* out_indices = topk_indices + token_idx * top_k;

    __shared__ float s_scores[256];
    __shared__ float s_choice[256];
    __shared__ int s_topk_idx[6];
    __shared__ float s_topk_val[6];

    // Compute sqrtsoftplus scores in parallel: 256 / 64 = 4 items per thread
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int e = tid + i * 64;
        if (e < n_routed_experts) {
            float x = logits[e];
            float sc = softplus_sqrt(x);
            s_scores[e] = sc;
            float b = (bias != nullptr) ? bias[e] : 0.0f;
            s_choice[e] = sc + b;
        }
    }
    __syncthreads();

    // Expert Selection:
    if (hash_table != nullptr && token_ids != nullptr) {
        // Hash mode (layers 0..2)
        if (tid < top_k) {
            int32_t token_id = token_ids[token_idx];
            int32_t selected_expert = static_cast<int32_t>(hash_table[token_id * top_k + tid]);
            s_topk_idx[tid] = selected_expert;
            s_topk_val[tid] = s_scores[selected_expert];
        }
    } else {
        // Softplus top-6 selection by single warp
        if (tid == 0) {
            // Find top-6 largest in s_choice
            for (int k = 0; k < top_k; ++k) {
                float best_val = -1e30f;
                int best_idx = -1;
                for (int e = 0; e < n_routed_experts; ++e) {
                    if (s_choice[e] > best_val) {
                        best_val = s_choice[e];
                        best_idx = e;
                    }
                }
                s_topk_idx[k] = best_idx;
                s_topk_val[k] = s_scores[best_idx];
                s_choice[best_idx] = -1e30f; // mask out for next top-k pick
            }
        }
    }
    __syncthreads();

    if (tid < top_k) {
        float val = s_topk_val[tid];
        out_indices[tid] = s_topk_idx[tid];

        // Thread 0 computes sum for normalization
        __shared__ float s_sum;
        if (tid == 0) {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k) sum += s_topk_val[k];
            s_sum = sum;
        }
        __syncthreads();

        if (renormalize) {
            val = val / (s_sum + 1e-20f);
        }
        val *= routed_scaling_factor;
        out_weights[tid] = val;
    }
}

} // namespace aeon::kernel
