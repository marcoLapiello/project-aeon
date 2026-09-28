#pragma once

// CPU reference implementations used to verify the attention kernels' precision.
//
// NOTE (anti-circularity, Monolith Module Split Tier C3). These two functions are
// **unused** — no kernel, test or tool calls them (grep over src/ and tests/ finds
// only these definitions). The gate that certifies the sliding-window attention
// kernel, `tests/test_v4_attention_sink_oracle.cpp`, compares against the
// independent fp64 reference `aeon::reference::attention_scores_sink` in
// `reference/dsv4_oracle.hpp`, so there is no shared-helper circularity today.
// They are relocated here purely as a file-location change, out of the kernel
// header the split plan flagged; deleting the dead code is left as a separate
// task so this split stays behaviour-neutral.

#include "architecture/deepseek_v4/kernels/v4_attention_config.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace aeon::kernel {

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
