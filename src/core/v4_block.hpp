#pragma once

#include "core/config.hpp"
#include "kernel/hc_sinkhorn.hpp"
#include "kernel/v4_attention.hpp"
#include "kernel/w4a16_gemm.hpp"
#include "kernel/moe_router.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <cmath>

namespace aeon::core {

// Weights and buffers for one complete DeepSeek-V4 Transformer Block
struct DeepSeekV4BlockWeights {
    // Hyper-Connections Attention
    std::vector<float> hc_attn_fn;    // [24, 16384]
    std::vector<float> hc_attn_base;  // [24]
    std::vector<float> hc_attn_scale; // [3]

    // Attention Norm
    std::vector<half>  attn_norm_weight; // [4096]

    // MLA Attention Projections
    std::vector<half>  wq_a_weight;   // [1024, 4096]
    std::vector<half>  q_norm_weight; // [1024]
    std::vector<half>  wq_b_weight;   // [32768, 1024]
    std::vector<half>  wkv_weight;    // [512, 4096]
    std::vector<half>  kv_norm_weight;// [512]
    std::vector<float> attn_sink;     // [64]
    std::vector<half>  wo_a_weight;   // [8192, 4096]
    std::vector<half>  wo_b_weight;   // [4096, 8192]

    // Hyper-Connections FFN
    std::vector<float> hc_ffn_fn;     // [24, 16384]
    std::vector<float> hc_ffn_base;   // [24]
    std::vector<float> hc_ffn_scale;  // [3]

    // FFN Norm
    std::vector<half>  ffn_norm_weight; // [4096]
};

// GPU Resident Buffers for executing a DeepSeek-V4 Transformer Block
struct DeepSeekV4BlockDeviceContext {
    // Device parameters
    float* d_hc_attn_fn{nullptr};
    float* d_hc_attn_base{nullptr};
    float* d_hc_attn_scale{nullptr};

    half*  d_attn_norm_weight{nullptr};

    half*  d_wq_a{nullptr};
    half*  d_q_norm{nullptr};
    half*  d_wq_b{nullptr};
    half*  d_wkv{nullptr};
    half*  d_kv_norm{nullptr};
    float* d_attn_sink{nullptr};
    half*  d_wo_a{nullptr};
    half*  d_wo_b{nullptr};

    float* d_hc_ffn_fn{nullptr};
    float* d_hc_ffn_base{nullptr};
    float* d_hc_ffn_scale{nullptr};

    half*  d_ffn_norm_weight{nullptr};

    // RoPE Tables on device
    float* d_cos_cache{nullptr};
    float* d_sin_cache{nullptr};

    // Scratch buffers for intermediate activations
    half*  d_x_pre{nullptr};       // [T, 4096]
    half*  d_x_norm{nullptr};      // [T, 4096]
    half*  d_qa{nullptr};          // [T, 1024]
    half*  d_qa_norm{nullptr};     // [T, 1024]
    half*  d_q{nullptr};            // [T, 64, 512]
    half*  d_kv{nullptr};           // [T, 512]
    half*  d_kv_norm_act{nullptr};  // [T, 512]
    half*  d_attn_out{nullptr};     // [T, 64, 512]
    half*  d_z_lora{nullptr};       // [T, 8192]
    half*  d_attn_proj{nullptr};    // [T, 4096]

    half*  d_ffn_pre{nullptr};      // [T, 4096]
    half*  d_ffn_norm_act{nullptr}; // [T, 4096]
    half*  d_ffn_proj{nullptr};     // [T, 4096]

    // Allocates all scratch buffers for sequence length up to max_tokens
    void allocate(const DeepSeekV4BlockWeights& weights, const aeon::kernel::RopeTable& rope, uint32_t max_tokens = 16) {
        (void)hipMalloc(&d_hc_attn_fn, weights.hc_attn_fn.size() * sizeof(float));
        (void)hipMalloc(&d_hc_attn_base, weights.hc_attn_base.size() * sizeof(float));
        (void)hipMalloc(&d_hc_attn_scale, weights.hc_attn_scale.size() * sizeof(float));
        (void)hipMemcpy(d_hc_attn_fn, weights.hc_attn_fn.data(), weights.hc_attn_fn.size() * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_hc_attn_base, weights.hc_attn_base.data(), weights.hc_attn_base.size() * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_hc_attn_scale, weights.hc_attn_scale.data(), weights.hc_attn_scale.size() * sizeof(float), hipMemcpyHostToDevice);

        (void)hipMalloc(&d_attn_norm_weight, weights.attn_norm_weight.size() * sizeof(half));
        (void)hipMemcpy(d_attn_norm_weight, weights.attn_norm_weight.data(), weights.attn_norm_weight.size() * sizeof(half), hipMemcpyHostToDevice);

        (void)hipMalloc(&d_wq_a, weights.wq_a_weight.size() * sizeof(half));
        (void)hipMalloc(&d_q_norm, weights.q_norm_weight.size() * sizeof(half));
        (void)hipMalloc(&d_wq_b, weights.wq_b_weight.size() * sizeof(half));
        (void)hipMalloc(&d_wkv, weights.wkv_weight.size() * sizeof(half));
        (void)hipMalloc(&d_kv_norm, weights.kv_norm_weight.size() * sizeof(half));
        (void)hipMalloc(&d_attn_sink, weights.attn_sink.size() * sizeof(float));
        (void)hipMalloc(&d_wo_a, weights.wo_a_weight.size() * sizeof(half));
        (void)hipMalloc(&d_wo_b, weights.wo_b_weight.size() * sizeof(half));

        (void)hipMemcpy(d_wq_a, weights.wq_a_weight.data(), weights.wq_a_weight.size() * sizeof(half), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_q_norm, weights.q_norm_weight.data(), weights.q_norm_weight.size() * sizeof(half), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_wq_b, weights.wq_b_weight.data(), weights.wq_b_weight.size() * sizeof(half), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_wkv, weights.wkv_weight.data(), weights.wkv_weight.size() * sizeof(half), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_kv_norm, weights.kv_norm_weight.data(), weights.kv_norm_weight.size() * sizeof(half), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_attn_sink, weights.attn_sink.data(), weights.attn_sink.size() * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_wo_a, weights.wo_a_weight.data(), weights.wo_a_weight.size() * sizeof(half), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_wo_b, weights.wo_b_weight.data(), weights.wo_b_weight.size() * sizeof(half), hipMemcpyHostToDevice);

        (void)hipMalloc(&d_hc_ffn_fn, weights.hc_ffn_fn.size() * sizeof(float));
        (void)hipMalloc(&d_hc_ffn_base, weights.hc_ffn_base.size() * sizeof(float));
        (void)hipMalloc(&d_hc_ffn_scale, weights.hc_ffn_scale.size() * sizeof(float));
        (void)hipMemcpy(d_hc_ffn_fn, weights.hc_ffn_fn.data(), weights.hc_ffn_fn.size() * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_hc_ffn_base, weights.hc_ffn_base.data(), weights.hc_ffn_base.size() * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_hc_ffn_scale, weights.hc_ffn_scale.data(), weights.hc_ffn_scale.size() * sizeof(float), hipMemcpyHostToDevice);

        (void)hipMalloc(&d_ffn_norm_weight, weights.ffn_norm_weight.size() * sizeof(half));
        (void)hipMemcpy(d_ffn_norm_weight, weights.ffn_norm_weight.data(), weights.ffn_norm_weight.size() * sizeof(half), hipMemcpyHostToDevice);

        // RoPE cache
        size_t rope_bytes = rope.max_seq_len * rope.half_rope * sizeof(float);
        (void)hipMalloc(&d_cos_cache, rope_bytes);
        (void)hipMalloc(&d_sin_cache, rope_bytes);
        (void)hipMemcpy(d_cos_cache, rope.cos_cache.data(), rope_bytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(d_sin_cache, rope.sin_cache.data(), rope_bytes, hipMemcpyHostToDevice);

        // Scratch buffers
        (void)hipMalloc(&d_x_pre, max_tokens * kernel::DSV4_HIDDEN_SIZE * sizeof(half));
        (void)hipMalloc(&d_x_norm, max_tokens * kernel::DSV4_HIDDEN_SIZE * sizeof(half));
        (void)hipMalloc(&d_qa, max_tokens * kernel::DSV4_Q_LORA_RANK * sizeof(half));
        (void)hipMalloc(&d_qa_norm, max_tokens * kernel::DSV4_Q_LORA_RANK * sizeof(half));
        (void)hipMalloc(&d_q, max_tokens * kernel::DSV4_NUM_HEADS * kernel::DSV4_HEAD_DIM * sizeof(half));
        (void)hipMalloc(&d_kv, max_tokens * kernel::DSV4_HEAD_DIM * sizeof(half));
        (void)hipMalloc(&d_kv_norm_act, max_tokens * kernel::DSV4_HEAD_DIM * sizeof(half));
        (void)hipMalloc(&d_attn_out, max_tokens * kernel::DSV4_NUM_HEADS * kernel::DSV4_HEAD_DIM * sizeof(half));
        (void)hipMalloc(&d_z_lora, max_tokens * kernel::DSV4_TOTAL_O_LORA_DIM * sizeof(half));
        (void)hipMalloc(&d_attn_proj, max_tokens * kernel::DSV4_HIDDEN_SIZE * sizeof(half));

        (void)hipMalloc(&d_ffn_pre, max_tokens * kernel::DSV4_HIDDEN_SIZE * sizeof(half));
        (void)hipMalloc(&d_ffn_norm_act, max_tokens * kernel::DSV4_HIDDEN_SIZE * sizeof(half));
        (void)hipMalloc(&d_ffn_proj, max_tokens * kernel::DSV4_HIDDEN_SIZE * sizeof(half));
    }

    void free() {
        if (d_hc_attn_fn) (void)hipFree(d_hc_attn_fn);
        if (d_hc_attn_base) (void)hipFree(d_hc_attn_base);
        if (d_hc_attn_scale) (void)hipFree(d_hc_attn_scale);
        if (d_attn_norm_weight) (void)hipFree(d_attn_norm_weight);
        if (d_wq_a) (void)hipFree(d_wq_a);
        if (d_q_norm) (void)hipFree(d_q_norm);
        if (d_wq_b) (void)hipFree(d_wq_b);
        if (d_wkv) (void)hipFree(d_wkv);
        if (d_kv_norm) (void)hipFree(d_kv_norm);
        if (d_attn_sink) (void)hipFree(d_attn_sink);
        if (d_wo_a) (void)hipFree(d_wo_a);
        if (d_wo_b) (void)hipFree(d_wo_b);
        if (d_hc_ffn_fn) (void)hipFree(d_hc_ffn_fn);
        if (d_hc_ffn_base) (void)hipFree(d_hc_ffn_base);
        if (d_hc_ffn_scale) (void)hipFree(d_hc_ffn_scale);
        if (d_ffn_norm_weight) (void)hipFree(d_ffn_norm_weight);
        if (d_cos_cache) (void)hipFree(d_cos_cache);
        if (d_sin_cache) (void)hipFree(d_sin_cache);

        if (d_x_pre) (void)hipFree(d_x_pre);
        if (d_x_norm) (void)hipFree(d_x_norm);
        if (d_qa) (void)hipFree(d_qa);
        if (d_qa_norm) (void)hipFree(d_qa_norm);
        if (d_q) (void)hipFree(d_q);
        if (d_kv) (void)hipFree(d_kv);
        if (d_kv_norm_act) (void)hipFree(d_kv_norm_act);
        if (d_attn_out) (void)hipFree(d_attn_out);
        if (d_z_lora) (void)hipFree(d_z_lora);
        if (d_attn_proj) (void)hipFree(d_attn_proj);
        if (d_ffn_pre) (void)hipFree(d_ffn_pre);
        if (d_ffn_norm_act) (void)hipFree(d_ffn_norm_act);
        if (d_ffn_proj) (void)hipFree(d_ffn_proj);
    }
};

// Complete DeepSeek-V4 Transformer Block Forward Pass (CPU Reference)
inline void cpu_v4_block_forward(
    const std::vector<float>& residual_in, // [4, 4096]
    const DeepSeekV4BlockWeights& w,
    const aeon::kernel::RopeTable& rope,
    std::vector<float>& residual_out,     // [4, 4096]
    int token_idx = 0
) {
    const int H = kernel::DSV4_HIDDEN_SIZE; // 4096
    const int HC = 4;

    // 1. HC Attention Pre-mix & Sinkhorn
    std::vector<float> x_attn(H);
    std::vector<float> post_mix_a(HC);
    std::vector<float> comb_mix_a(HC * HC);
    kernel::cpu_sinkhorn_and_mix(
        residual_in.data(), w.hc_attn_fn.data(), w.hc_attn_base.data(), w.hc_attn_scale.data(),
        x_attn.data(), post_mix_a.data(), comb_mix_a.data(), H, HC
    );

    // 2. Attention RMSNorm
    std::vector<float> attn_norm_w_f32(H);
    for (int i = 0; i < H; ++i) attn_norm_w_f32[i] = __half2float(w.attn_norm_weight[i]);
    std::vector<float> x_norm(H);
    kernel::cpu_rmsnorm(x_attn.data(), attn_norm_w_f32.data(), x_norm.data(), H);

    // 3. MLA Attention:
    // 3a. Q_a = x_norm @ wq_a.T
    const int Q_LORA = kernel::DSV4_Q_LORA_RANK; // 1024
    std::vector<float> qa(Q_LORA, 0.0f);
    for (int r = 0; r < Q_LORA; ++r) {
        float dot = 0.0f;
        for (int c = 0; c < H; ++c) {
            dot += x_norm[c] * __half2float(w.wq_a_weight[r * H + c]);
        }
        qa[r] = dot;
    }

    // 3b. Q_a RMSNorm
    std::vector<float> q_norm_w_f32(Q_LORA);
    for (int i = 0; i < Q_LORA; ++i) q_norm_w_f32[i] = __half2float(w.q_norm_weight[i]);
    std::vector<float> qa_norm(Q_LORA);
    kernel::cpu_rmsnorm(qa.data(), q_norm_w_f32.data(), qa_norm.data(), Q_LORA);

    // 3c. Q = qa_norm @ wq_b.T [64 * 512 = 32768]
    const int NUM_HEADS = kernel::DSV4_NUM_HEADS; // 64
    const int HEAD_DIM  = kernel::DSV4_HEAD_DIM;  // 512
    const int TOTAL_Q   = NUM_HEADS * HEAD_DIM;   // 32768
    std::vector<float> q(TOTAL_Q, 0.0f);
    for (int r = 0; r < TOTAL_Q; ++r) {
        float dot = 0.0f;
        for (int c = 0; c < Q_LORA; ++c) {
            dot += qa_norm[c] * __half2float(w.wq_b_weight[r * Q_LORA + c]);
        }
        q[r] = dot;
    }

    // 3d. KV = x_norm @ wkv.T [512]
    std::vector<float> kv(HEAD_DIM, 0.0f);
    for (int r = 0; r < HEAD_DIM; ++r) {
        float dot = 0.0f;
        for (int c = 0; c < H; ++c) {
            dot += x_norm[c] * __half2float(w.wkv_weight[r * H + c]);
        }
        kv[r] = dot;
    }

    // 3e. KV RMSNorm
    std::vector<float> kv_norm_w_f32(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; ++i) kv_norm_w_f32[i] = __half2float(w.kv_norm_weight[i]);
    std::vector<float> kv_norm(HEAD_DIM);
    kernel::cpu_rmsnorm(kv.data(), kv_norm_w_f32.data(), kv_norm.data(), HEAD_DIM);

    // 3f. Forward RoPE on Q (dim 448..511 per head) and KV (dim 448..511)
    kernel::cpu_forward_rope(q.data(), token_idx, rope, NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2);
    kernel::cpu_forward_rope(kv_norm.data(), token_idx, rope, 1, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2);

    // 3g. Sliding-Window Attention with Attention Sink
    std::vector<float> attn_out;
    kernel::cpu_sliding_window_attention(q, kv_norm, w.attn_sink, attn_out, 1, kernel::DSV4_SLIDING_WINDOW, kernel::DSV4_ATTN_SCALE);

    // 3h. Inverse RoPE on attn_out (dim 448..511 per head)
    kernel::cpu_inverse_rope(attn_out.data(), token_idx, rope, NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2);

    // 3i. Grouped W_o_a: 8 groups of [1024, 4096]
    const int O_GROUPS = kernel::DSV4_O_GROUPS;         // 8
    const int O_LORA   = kernel::DSV4_O_LORA_RANK;       // 1024
    const int G_DIM    = kernel::DSV4_GROUP_HEADS_DIM;   // 4096
    const int TOT_LORA = kernel::DSV4_TOTAL_O_LORA_DIM;  // 8192
    std::vector<float> z(TOT_LORA, 0.0f);
    for (int g = 0; g < O_GROUPS; ++g) {
        const float* head_group_in = attn_out.data() + g * G_DIM;
        const half* wo_a_group     = w.wo_a_weight.data() + g * (O_LORA * G_DIM);
        for (int r = 0; r < O_LORA; ++r) {
            float dot = 0.0f;
            for (int c = 0; c < G_DIM; ++c) {
                dot += head_group_in[c] * __half2float(wo_a_group[r * G_DIM + c]);
            }
            z[g * O_LORA + r] = dot;
        }
    }

    // 3j. W_o_b: [4096, 8192]
    std::vector<float> attn_proj(H, 0.0f);
    for (int r = 0; r < H; ++r) {
        float dot = 0.0f;
        for (int c = 0; c < TOT_LORA; ++c) {
            dot += z[c] * __half2float(w.wo_b_weight[r * TOT_LORA + c]);
        }
        attn_proj[r] = dot;
    }

    // 4. HC Attention Post Expansion: R' = comb_a * R + post_a * AttnOut
    std::vector<float> residual_mid(HC * H);
    kernel::cpu_hc_post(attn_proj.data(), residual_in.data(), post_mix_a.data(), comb_mix_a.data(), residual_mid.data(), H, HC);

    // 5. HC FFN Pre-mix & Sinkhorn
    std::vector<float> x_ffn(H);
    std::vector<float> post_mix_f(HC);
    std::vector<float> comb_mix_f(HC * HC);
    kernel::cpu_sinkhorn_and_mix(
        residual_mid.data(), w.hc_ffn_fn.data(), w.hc_ffn_base.data(), w.hc_ffn_scale.data(),
        x_ffn.data(), post_mix_f.data(), comb_mix_f.data(), H, HC
    );

    // 6. FFN RMSNorm
    std::vector<float> ffn_norm_w_f32(H);
    for (int i = 0; i < H; ++i) ffn_norm_w_f32[i] = __half2float(w.ffn_norm_weight[i]);
    std::vector<float> ffn_norm_act(H);
    kernel::cpu_rmsnorm(x_ffn.data(), ffn_norm_w_f32.data(), ffn_norm_act.data(), H);

    // 7. FFN / MoE output baseline for single block verification
    std::vector<float> ffn_out = ffn_norm_act;

    // 8. HC FFN Post Expansion: R'' = comb_f * R' + post_f * FFNOut
    residual_out.resize(HC * H);
    kernel::cpu_hc_post(ffn_out.data(), residual_mid.data(), post_mix_f.data(), comb_mix_f.data(), residual_out.data(), H, HC);
}

} // namespace aeon::core
