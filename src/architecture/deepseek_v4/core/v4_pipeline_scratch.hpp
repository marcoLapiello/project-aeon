#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)
#endif

namespace aeon::core {

// Scratch Device Buffers Reusable Across All Layers
struct PipelineScratchBuffers {
    // Residuals [4, 4096] in float and half
    float* d_res_in{nullptr};
    float* d_res_mid{nullptr};
    float* d_res_out{nullptr};
    half*  d_res_in_half{nullptr};
    half*  d_res_mid_half{nullptr};
    half*  d_res_out_half{nullptr};

    // HC Attention Sinkhorn
    float* d_mixes_a{nullptr};
    float* d_pre_a{nullptr};
    float* d_post_a{nullptr};
    float* d_comb_a{nullptr};

    // HC FFN Sinkhorn
    float* d_mixes_f{nullptr};
    float* d_pre_f{nullptr};
    float* d_post_f{nullptr};
    float* d_comb_f{nullptr};

    // Attention Activations (padded to 16 rows for WMMA compatibility)
    half*  d_x_pre{nullptr};       // [16, 4096]
    half*  d_x_norm{nullptr};      // [16, 4096]
    half*  d_qa{nullptr};          // [16, 1024]
    half*  d_qa_norm{nullptr};     // [16, 1024]
    half*  d_q{nullptr};           // [16, 64, 512]
    half*  d_kv{nullptr};          // [16, 512]
    half*  d_kv_norm_act{nullptr}; // [16, 512]
    half*  d_compressor_kv{nullptr}; // [16, 1024]
    half*  d_compressor_score{nullptr}; // [16, 1024]
    half*  d_indexer_query{nullptr}; // [16, 8192]
    half*  d_indexer_weights{nullptr}; // [16, 64]
    half*  d_indexer_compressor_kv{nullptr}; // [16, 256]
    half*  d_indexer_compressor_score{nullptr}; // [16, 256]
    int32_t* d_indexer_topk_indices{nullptr}; // [16, 512]
    int32_t* d_indexer_candidate_count{nullptr}; // [16]
    half*  d_attn_out{nullptr};    // [16, 64, 512]
    half*  d_z_lora{nullptr};      // [16, 8192]
    half*  d_attn_proj{nullptr};   // [16, 4096]

    // MoE Activations
    half*  d_ffn_pre{nullptr};      // [16, 4096]
    half*  d_ffn_norm_act{nullptr}; // [16, 4096]
    half*  d_router_logits_half{nullptr}; // [256] GEMV output
    float* d_router_logits{nullptr};       // [256] converted router logits
    float* d_topk_weights{nullptr}; // [6]
    int32_t* d_topk_indices{nullptr};// [6]
    int32_t* d_token_id{nullptr};   // [1]

    half*  d_shared_gate{nullptr};  // [16, 2048]
    half*  d_shared_up{nullptr};    // [16, 2048]
    half*  d_shared_swiglu{nullptr};// [16, 2048]
    half*  d_shared_down{nullptr};  // [16, 4096]

    half*  d_moe_accum{nullptr};    // [16, 4096]
    half*  d_expert_gate{nullptr};  // [16, 2048]
    half*  d_expert_up{nullptr};    // [16, 2048]
    half*  d_expert_swiglu{nullptr};// [16, 2048]
    half*  d_expert_down{nullptr};  // [16, 4096]

    // Parallel swizzled routed-expert path: [8, 2048], [4096], and [64]
    half* d_swizzled_expert_hidden{nullptr};
    float* d_swizzled_moe_accum_f32{nullptr};
    int32_t* d_swizzled_counters{nullptr};

    // Head Activations
    half*  d_hc_head_out{nullptr};  // [4096]
    half*  d_head_norm{nullptr};    // [4096]
    half*  d_logits{nullptr};       // [129280]
    float*   d_argmax_partial_vals{nullptr}; // [505] GPU argmax block partials
    int32_t* d_argmax_partial_idx{nullptr};  // [505]
    int32_t* d_argmax_result{nullptr}; // [1] GPU argmax output token id

    PipelineScratchBuffers() = default;

    ~PipelineScratchBuffers() {
        free();
    }

    // Move-only semantics to prevent accidental double-free
    PipelineScratchBuffers(const PipelineScratchBuffers&) = delete;
    PipelineScratchBuffers& operator=(const PipelineScratchBuffers&) = delete;

    PipelineScratchBuffers(PipelineScratchBuffers&& other) noexcept {
        move_from(std::move(other));
    }

    PipelineScratchBuffers& operator=(PipelineScratchBuffers&& other) noexcept {
        if (this != &other) {
            free();
            move_from(std::move(other));
        }
        return *this;
    }

    void allocate() {
        constexpr int H = 4096;
        constexpr int HC = 4;
        constexpr int HC_DIM = HC * H;
        constexpr int HC_MULT3 = 24;
        constexpr int M = 16; // Padded row size for WMMA

        CHECK_HIP(hipMalloc(&d_res_in, HC_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_res_mid, HC_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_res_out, HC_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_res_in_half, HC_DIM * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_res_mid_half, HC_DIM * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_res_out_half, HC_DIM * sizeof(half)));

        CHECK_HIP(hipMalloc(&d_mixes_a, HC_MULT3 * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_pre_a, HC * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_post_a, HC * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_comb_a, HC * HC * sizeof(float)));

        CHECK_HIP(hipMalloc(&d_mixes_f, HC_MULT3 * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_pre_f, HC * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_post_f, HC * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_comb_f, HC * HC * sizeof(float)));

        CHECK_HIP(hipMalloc(&d_x_pre, M * H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_x_norm, M * H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_qa, M * 1024 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_qa_norm, M * 1024 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_q, M * 64 * 512 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_kv, M * 512 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_kv_norm_act, M * 512 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_compressor_kv, M * 1024 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_compressor_score, M * 1024 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_indexer_query, M * 8192 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_indexer_weights, M * 64 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_indexer_compressor_kv, M * 256 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_indexer_compressor_score, M * 256 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_indexer_topk_indices, M * 512 * sizeof(int32_t)));
        CHECK_HIP(hipMalloc(&d_indexer_candidate_count, M * sizeof(int32_t)));
        CHECK_HIP(hipMalloc(&d_attn_out, M * 64 * 512 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_z_lora, M * 8192 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_attn_proj, M * H * sizeof(half)));

        CHECK_HIP(hipMalloc(&d_ffn_pre, M * H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_ffn_norm_act, M * H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_router_logits_half, 256 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_router_logits, 256 * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_topk_weights, 6 * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_topk_indices, 6 * sizeof(int32_t)));
        CHECK_HIP(hipMalloc(&d_token_id, 1 * sizeof(int32_t)));

        CHECK_HIP(hipMalloc(&d_shared_gate, M * 2048 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_shared_up, M * 2048 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_shared_swiglu, M * 2048 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_shared_down, M * H * sizeof(half)));

        CHECK_HIP(hipMalloc(&d_moe_accum, M * H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_expert_gate, M * 2048 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_expert_up, M * 2048 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_expert_swiglu, M * 2048 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_expert_down, M * H * sizeof(half)));

        CHECK_HIP(hipMalloc(&d_swizzled_expert_hidden, 8 * 2048 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_swizzled_moe_accum_f32, H * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_swizzled_counters, 64 * sizeof(int32_t)));
        CHECK_HIP(hipMemset(d_swizzled_counters, 0, 64 * sizeof(int32_t)));

        CHECK_HIP(hipMalloc(&d_hc_head_out, H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_head_norm, H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_logits, 129280 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_argmax_partial_vals, 505 * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_argmax_partial_idx, 505 * sizeof(int32_t)));
        CHECK_HIP(hipMalloc(&d_argmax_result, sizeof(int32_t)));

        // Clear initial padded memory
        CHECK_HIP(hipMemset(d_x_pre, 0, M * H * sizeof(half)));
        CHECK_HIP(hipMemset(d_x_norm, 0, M * H * sizeof(half)));
    }

    void free() {
        if (d_res_in) { (void)hipFree(d_res_in); d_res_in = nullptr; }
        if (d_res_mid) { (void)hipFree(d_res_mid); d_res_mid = nullptr; }
        if (d_res_out) { (void)hipFree(d_res_out); d_res_out = nullptr; }
        if (d_res_in_half) { (void)hipFree(d_res_in_half); d_res_in_half = nullptr; }
        if (d_res_mid_half) { (void)hipFree(d_res_mid_half); d_res_mid_half = nullptr; }
        if (d_res_out_half) { (void)hipFree(d_res_out_half); d_res_out_half = nullptr; }

        if (d_mixes_a) { (void)hipFree(d_mixes_a); d_mixes_a = nullptr; }
        if (d_pre_a) { (void)hipFree(d_pre_a); d_pre_a = nullptr; }
        if (d_post_a) { (void)hipFree(d_post_a); d_post_a = nullptr; }
        if (d_comb_a) { (void)hipFree(d_comb_a); d_comb_a = nullptr; }

        if (d_mixes_f) { (void)hipFree(d_mixes_f); d_mixes_f = nullptr; }
        if (d_pre_f) { (void)hipFree(d_pre_f); d_pre_f = nullptr; }
        if (d_post_f) { (void)hipFree(d_post_f); d_post_f = nullptr; }
        if (d_comb_f) { (void)hipFree(d_comb_f); d_comb_f = nullptr; }

        if (d_x_pre) { (void)hipFree(d_x_pre); d_x_pre = nullptr; }
        if (d_x_norm) { (void)hipFree(d_x_norm); d_x_norm = nullptr; }
        if (d_qa) { (void)hipFree(d_qa); d_qa = nullptr; }
        if (d_qa_norm) { (void)hipFree(d_qa_norm); d_qa_norm = nullptr; }
        if (d_q) { (void)hipFree(d_q); d_q = nullptr; }
        if (d_kv) { (void)hipFree(d_kv); d_kv = nullptr; }
        if (d_kv_norm_act) { (void)hipFree(d_kv_norm_act); d_kv_norm_act = nullptr; }
        if (d_compressor_kv) { (void)hipFree(d_compressor_kv); d_compressor_kv = nullptr; }
        if (d_compressor_score) { (void)hipFree(d_compressor_score); d_compressor_score = nullptr; }
        if (d_indexer_query) { (void)hipFree(d_indexer_query); d_indexer_query = nullptr; }
        if (d_indexer_weights) { (void)hipFree(d_indexer_weights); d_indexer_weights = nullptr; }
        if (d_indexer_compressor_kv) { (void)hipFree(d_indexer_compressor_kv); d_indexer_compressor_kv = nullptr; }
        if (d_indexer_compressor_score) { (void)hipFree(d_indexer_compressor_score); d_indexer_compressor_score = nullptr; }
        if (d_indexer_topk_indices) { (void)hipFree(d_indexer_topk_indices); d_indexer_topk_indices = nullptr; }
        if (d_indexer_candidate_count) { (void)hipFree(d_indexer_candidate_count); d_indexer_candidate_count = nullptr; }
        if (d_attn_out) { (void)hipFree(d_attn_out); d_attn_out = nullptr; }
        if (d_z_lora) { (void)hipFree(d_z_lora); d_z_lora = nullptr; }
        if (d_attn_proj) { (void)hipFree(d_attn_proj); d_attn_proj = nullptr; }

        if (d_ffn_pre) { (void)hipFree(d_ffn_pre); d_ffn_pre = nullptr; }
        if (d_ffn_norm_act) { (void)hipFree(d_ffn_norm_act); d_ffn_norm_act = nullptr; }
        if (d_router_logits_half) { (void)hipFree(d_router_logits_half); d_router_logits_half = nullptr; }
        if (d_router_logits) { (void)hipFree(d_router_logits); d_router_logits = nullptr; }
        if (d_topk_weights) { (void)hipFree(d_topk_weights); d_topk_weights = nullptr; }
        if (d_topk_indices) { (void)hipFree(d_topk_indices); d_topk_indices = nullptr; }
        if (d_token_id) { (void)hipFree(d_token_id); d_token_id = nullptr; }

        if (d_shared_gate) { (void)hipFree(d_shared_gate); d_shared_gate = nullptr; }
        if (d_shared_up) { (void)hipFree(d_shared_up); d_shared_up = nullptr; }
        if (d_shared_swiglu) { (void)hipFree(d_shared_swiglu); d_shared_swiglu = nullptr; }
        if (d_shared_down) { (void)hipFree(d_shared_down); d_shared_down = nullptr; }

        if (d_moe_accum) { (void)hipFree(d_moe_accum); d_moe_accum = nullptr; }
        if (d_expert_gate) { (void)hipFree(d_expert_gate); d_expert_gate = nullptr; }
        if (d_expert_up) { (void)hipFree(d_expert_up); d_expert_up = nullptr; }
        if (d_expert_swiglu) { (void)hipFree(d_expert_swiglu); d_expert_swiglu = nullptr; }
        if (d_expert_down) { (void)hipFree(d_expert_down); d_expert_down = nullptr; }
        if (d_swizzled_expert_hidden) { (void)hipFree(d_swizzled_expert_hidden); d_swizzled_expert_hidden = nullptr; }
        if (d_swizzled_moe_accum_f32) { (void)hipFree(d_swizzled_moe_accum_f32); d_swizzled_moe_accum_f32 = nullptr; }
        if (d_swizzled_counters) { (void)hipFree(d_swizzled_counters); d_swizzled_counters = nullptr; }

        if (d_hc_head_out) { (void)hipFree(d_hc_head_out); d_hc_head_out = nullptr; }
        if (d_head_norm) { (void)hipFree(d_head_norm); d_head_norm = nullptr; }
        if (d_logits) { (void)hipFree(d_logits); d_logits = nullptr; }
        if (d_argmax_partial_vals) { (void)hipFree(d_argmax_partial_vals); d_argmax_partial_vals = nullptr; }
        if (d_argmax_partial_idx) { (void)hipFree(d_argmax_partial_idx); d_argmax_partial_idx = nullptr; }
        if (d_argmax_result) { (void)hipFree(d_argmax_result); d_argmax_result = nullptr; }
    }

private:
    void move_from(PipelineScratchBuffers&& o) noexcept {
        d_res_in = o.d_res_in; o.d_res_in = nullptr;
        d_res_mid = o.d_res_mid; o.d_res_mid = nullptr;
        d_res_out = o.d_res_out; o.d_res_out = nullptr;
        d_res_in_half = o.d_res_in_half; o.d_res_in_half = nullptr;
        d_res_mid_half = o.d_res_mid_half; o.d_res_mid_half = nullptr;
        d_res_out_half = o.d_res_out_half; o.d_res_out_half = nullptr;

        d_mixes_a = o.d_mixes_a; o.d_mixes_a = nullptr;
        d_pre_a = o.d_pre_a; o.d_pre_a = nullptr;
        d_post_a = o.d_post_a; o.d_post_a = nullptr;
        d_comb_a = o.d_comb_a; o.d_comb_a = nullptr;

        d_mixes_f = o.d_mixes_f; o.d_mixes_f = nullptr;
        d_pre_f = o.d_pre_f; o.d_pre_f = nullptr;
        d_post_f = o.d_post_f; o.d_post_f = nullptr;
        d_comb_f = o.d_comb_f; o.d_comb_f = nullptr;

        d_x_pre = o.d_x_pre; o.d_x_pre = nullptr;
        d_x_norm = o.d_x_norm; o.d_x_norm = nullptr;
        d_qa = o.d_qa; o.d_qa = nullptr;
        d_qa_norm = o.d_qa_norm; o.d_qa_norm = nullptr;
        d_q = o.d_q; o.d_q = nullptr;
        d_kv = o.d_kv; o.d_kv = nullptr;
        d_kv_norm_act = o.d_kv_norm_act; o.d_kv_norm_act = nullptr;
        d_compressor_kv = o.d_compressor_kv; o.d_compressor_kv = nullptr;
        d_compressor_score = o.d_compressor_score; o.d_compressor_score = nullptr;
        d_indexer_query = o.d_indexer_query; o.d_indexer_query = nullptr;
        d_indexer_weights = o.d_indexer_weights; o.d_indexer_weights = nullptr;
        d_indexer_compressor_kv = o.d_indexer_compressor_kv; o.d_indexer_compressor_kv = nullptr;
        d_indexer_compressor_score = o.d_indexer_compressor_score; o.d_indexer_compressor_score = nullptr;
        d_indexer_topk_indices = o.d_indexer_topk_indices; o.d_indexer_topk_indices = nullptr;
        d_indexer_candidate_count = o.d_indexer_candidate_count; o.d_indexer_candidate_count = nullptr;
        d_attn_out = o.d_attn_out; o.d_attn_out = nullptr;
        d_z_lora = o.d_z_lora; o.d_z_lora = nullptr;
        d_attn_proj = o.d_attn_proj; o.d_attn_proj = nullptr;

        d_ffn_pre = o.d_ffn_pre; o.d_ffn_pre = nullptr;
        d_ffn_norm_act = o.d_ffn_norm_act; o.d_ffn_norm_act = nullptr;
        d_router_logits_half = o.d_router_logits_half; o.d_router_logits_half = nullptr;
        d_router_logits = o.d_router_logits; o.d_router_logits = nullptr;
        d_topk_weights = o.d_topk_weights; o.d_topk_weights = nullptr;
        d_topk_indices = o.d_topk_indices; o.d_topk_indices = nullptr;
        d_token_id = o.d_token_id; o.d_token_id = nullptr;

        d_shared_gate = o.d_shared_gate; o.d_shared_gate = nullptr;
        d_shared_up = o.d_shared_up; o.d_shared_up = nullptr;
        d_shared_swiglu = o.d_shared_swiglu; o.d_shared_swiglu = nullptr;
        d_shared_down = o.d_shared_down; o.d_shared_down = nullptr;

        d_moe_accum = o.d_moe_accum; o.d_moe_accum = nullptr;
        d_expert_gate = o.d_expert_gate; o.d_expert_gate = nullptr;
        d_expert_up = o.d_expert_up; o.d_expert_up = nullptr;
        d_expert_swiglu = o.d_expert_swiglu; o.d_expert_swiglu = nullptr;
        d_expert_down = o.d_expert_down; o.d_expert_down = nullptr;
        d_swizzled_expert_hidden = o.d_swizzled_expert_hidden; o.d_swizzled_expert_hidden = nullptr;
        d_swizzled_moe_accum_f32 = o.d_swizzled_moe_accum_f32; o.d_swizzled_moe_accum_f32 = nullptr;
        d_swizzled_counters = o.d_swizzled_counters; o.d_swizzled_counters = nullptr;

        d_hc_head_out = o.d_hc_head_out; o.d_hc_head_out = nullptr;
        d_head_norm = o.d_head_norm; o.d_head_norm = nullptr;
        d_logits = o.d_logits; o.d_logits = nullptr;
        d_argmax_partial_vals = o.d_argmax_partial_vals; o.d_argmax_partial_vals = nullptr;
        d_argmax_partial_idx = o.d_argmax_partial_idx; o.d_argmax_partial_idx = nullptr;
        d_argmax_result = o.d_argmax_result; o.d_argmax_result = nullptr;
    }
};

struct PipelineBatchScratchBuffers {
    static constexpr int kMaxTokens = 16;

    float* d_res_in{nullptr};
    float* d_res_mid{nullptr};
    float* d_res_out{nullptr};
    half* d_res_in_half{nullptr};
    half* d_res_mid_half{nullptr};
    half* d_res_out_half{nullptr};

    float* d_mixes_a{nullptr};
    float* d_pre_a{nullptr};
    float* d_post_a{nullptr};
    float* d_comb_a{nullptr};
    float* d_mixes_f{nullptr};
    float* d_pre_f{nullptr};
    float* d_post_f{nullptr};
    float* d_comb_f{nullptr};

    half* d_x_pre{nullptr};
    half* d_x_norm{nullptr};
    half* d_qa{nullptr};
    half* d_qa_norm{nullptr};
    half* d_q{nullptr};
    half* d_kv{nullptr};
    half* d_kv_norm_act{nullptr};
    half* d_kv_rotated{nullptr};
    half* d_compressor_kv{nullptr};
    half* d_compressor_score{nullptr};
    half* d_indexer_query{nullptr};
    half* d_indexer_weights{nullptr};
    half* d_indexer_compressor_kv{nullptr};
    half* d_indexer_compressor_score{nullptr};
    half* d_attn_out{nullptr};
    half* d_z_lora{nullptr};
    half* d_attn_proj{nullptr};

    half* d_ffn_pre{nullptr};
    half* d_ffn_norm_act{nullptr};
    half* d_router_logits_half{nullptr};
    float* d_router_logits{nullptr};
    float* d_topk_weights{nullptr};
    int32_t* d_topk_indices{nullptr};
    int32_t* d_token_ids{nullptr};
    half* d_shared_gate{nullptr};
    half* d_shared_up{nullptr};
    half* d_shared_swiglu{nullptr};
    half* d_moe_accum{nullptr};

    ~PipelineBatchScratchBuffers() {
        free();
    }

    void allocate() {
        constexpr size_t B = kMaxTokens;
        constexpr size_t H = 4096;
        constexpr size_t HC_DIM = 4 * H;
        constexpr size_t Q_LORA = 1024;
        constexpr size_t TOTAL_Q = 64 * 512;
        constexpr size_t COMPRESSOR_WIDTH = 2 * 512;
        constexpr size_t INDEXER_Q = 64 * 128;
        constexpr size_t INDEXER_WIDTH = 2 * 128;
        constexpr size_t O_LORA = 8 * 1024;
        constexpr size_t INTER_DIM = 2048;
        constexpr size_t ROUTED_EXPERTS = 6;
        constexpr size_t ROUTER_EXPERTS = 256;

        allocate_array(d_res_in, B * HC_DIM);
        allocate_array(d_res_mid, B * HC_DIM);
        allocate_array(d_res_out, B * HC_DIM);
        allocate_array(d_res_in_half, B * HC_DIM);
        allocate_array(d_res_mid_half, B * HC_DIM);
        allocate_array(d_res_out_half, B * HC_DIM);

        allocate_array(d_mixes_a, B * 24);
        allocate_array(d_pre_a, B * 4);
        allocate_array(d_post_a, B * 4);
        allocate_array(d_comb_a, B * 16);
        allocate_array(d_mixes_f, B * 24);
        allocate_array(d_pre_f, B * 4);
        allocate_array(d_post_f, B * 4);
        allocate_array(d_comb_f, B * 16);

        allocate_array(d_x_pre, B * H);
        allocate_array(d_x_norm, B * H);
        allocate_array(d_qa, B * Q_LORA);
        allocate_array(d_qa_norm, B * Q_LORA);
        allocate_array(d_q, B * TOTAL_Q);
        allocate_array(d_kv, B * 512);
        allocate_array(d_kv_norm_act, B * 512);
        allocate_array(d_kv_rotated, B * 512);
        allocate_array(d_compressor_kv, B * COMPRESSOR_WIDTH);
        allocate_array(d_compressor_score, B * COMPRESSOR_WIDTH);
        allocate_array(d_indexer_query, B * INDEXER_Q);
        allocate_array(d_indexer_weights, B * 64);
        allocate_array(d_indexer_compressor_kv, B * INDEXER_WIDTH);
        allocate_array(d_indexer_compressor_score, B * INDEXER_WIDTH);
        allocate_array(d_attn_out, B * TOTAL_Q);
        allocate_array(d_z_lora, B * O_LORA);
        allocate_array(d_attn_proj, B * H);

        allocate_array(d_ffn_pre, B * H);
        allocate_array(d_ffn_norm_act, B * H);
        allocate_array(d_router_logits_half, B * ROUTER_EXPERTS);
        allocate_array(d_router_logits, B * ROUTER_EXPERTS);
        allocate_array(d_topk_weights, B * ROUTED_EXPERTS);
        allocate_array(d_topk_indices, B * ROUTED_EXPERTS);
        allocate_array(d_token_ids, B);
        allocate_array(d_shared_gate, B * INTER_DIM);
        allocate_array(d_shared_up, B * INTER_DIM);
        allocate_array(d_shared_swiglu, B * INTER_DIM);
        allocate_array(d_moe_accum, B * H);
    }

    void free() noexcept {
        free_array(d_res_in);
        free_array(d_res_mid);
        free_array(d_res_out);
        free_array(d_res_in_half);
        free_array(d_res_mid_half);
        free_array(d_res_out_half);
        free_array(d_mixes_a);
        free_array(d_pre_a);
        free_array(d_post_a);
        free_array(d_comb_a);
        free_array(d_mixes_f);
        free_array(d_pre_f);
        free_array(d_post_f);
        free_array(d_comb_f);
        free_array(d_x_pre);
        free_array(d_x_norm);
        free_array(d_qa);
        free_array(d_qa_norm);
        free_array(d_q);
        free_array(d_kv);
        free_array(d_kv_norm_act);
        free_array(d_kv_rotated);
        free_array(d_compressor_kv);
        free_array(d_compressor_score);
        free_array(d_indexer_query);
        free_array(d_indexer_weights);
        free_array(d_indexer_compressor_kv);
        free_array(d_indexer_compressor_score);
        free_array(d_attn_out);
        free_array(d_z_lora);
        free_array(d_attn_proj);
        free_array(d_ffn_pre);
        free_array(d_ffn_norm_act);
        free_array(d_router_logits_half);
        free_array(d_router_logits);
        free_array(d_topk_weights);
        free_array(d_topk_indices);
        free_array(d_token_ids);
        free_array(d_shared_gate);
        free_array(d_shared_up);
        free_array(d_shared_swiglu);
        free_array(d_moe_accum);
    }

private:
    template<typename T>
    static void allocate_array(T*& pointer, size_t count) {
        CHECK_HIP(hipMalloc(&pointer, count * sizeof(T)));
        CHECK_HIP(hipMemset(pointer, 0, count * sizeof(T)));
    }

    template<typename T>
    static void free_array(T*& pointer) noexcept {
        if (pointer != nullptr) {
            (void)hipFree(pointer);
            pointer = nullptr;
        }
    }
};

} // namespace aeon::core
