#pragma once

#include "core/config.hpp"
#include "core/device.hpp"
#include "core/safetensors_loader.hpp"
#include "kernel/hc_sinkhorn.hpp"
#include "kernel/moe_router.hpp"
#include "kernel/v4_attention.hpp"
#include "kernel/w4a16_gemm.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

namespace aeon::core {

// Clamped SwiGLU Kernel
__global__ void v4_pipeline_swiglu_clamp_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int total_elements,
    float limit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
        float silu_g = g / (1.0f + expf(-g));
        out[idx] = __float2half(silu_g * u);
    }
}

// Accumulate weighted expert output into token hidden state
__global__ void v4_pipeline_accumulate_expert_kernel(
    half* __restrict__ accum_out,
    const half* __restrict__ expert_out,
    float weight,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        float acc = __half2float(accum_out[idx]);
        float exp = __half2float(expert_out[idx]);
        accum_out[idx] = __float2half(acc + weight * exp);
    }
}

// Convert FP16 array to float array
__global__ void v4_half_to_float_kernel(
    const half* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

// Convert float array to FP16 array
__global__ void v4_float_to_half_kernel(
    const float* __restrict__ src,
    half* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Tier 1 VRAM Slot for an INT4-W4A16 Routed Expert
struct VRAMExpertSlot {
    uint32_t* d_w1_packed{nullptr};
    half*     d_w1_scale{nullptr};
    uint32_t* d_w2_packed{nullptr};
    half*     d_w2_scale{nullptr};
    uint32_t* d_w3_packed{nullptr};
    half*     d_w3_scale{nullptr};
    int32_t   resident_expert_id{-1};
};

// Pointers to Host / Mmap memory for one routed expert
struct HostExpertSource {
    const uint32_t* w1_packed{nullptr};
    const half*     w1_scale{nullptr};
    const uint32_t* w2_packed{nullptr};
    const half*     w2_scale{nullptr};
    const uint32_t* w3_packed{nullptr};
    const half*     w3_scale{nullptr};
};

// Device Context and Weights for One Complete Transformer Block
class V4Layer {
public:
    int layer_id{0};
    bool is_hash_layer{true};

    // Attention Weights on Device
    half*  d_attn_norm{nullptr};  // [4096]
    half*  d_wq_a{nullptr};       // [1024, 4096]
    half*  d_q_norm{nullptr};     // [1024]
    half*  d_wq_b{nullptr};       // [32768, 1024]
    half*  d_wkv{nullptr};        // [512, 4096]
    half*  d_kv_norm{nullptr};    // [512]
    float* d_attn_sink{nullptr};  // [64]
    half*  d_wo_a{nullptr};       // [8192, 4096]
    half*  d_wo_b{nullptr};       // [4096, 8192]

    // Hyper-Connections Attention
    float* d_hc_attn_fn{nullptr};    // [24, 16384]
    float* d_hc_attn_base{nullptr};  // [24]
    float* d_hc_attn_scale{nullptr}; // [3]

    // FFN Weights on Device
    half*  d_ffn_norm{nullptr};   // [4096]
    float* d_hc_ffn_fn{nullptr};    // [24, 16384]
    float* d_hc_ffn_base{nullptr};  // [24]
    float* d_hc_ffn_scale{nullptr}; // [3]

    // Shared Expert (FP16 unquantized)
    half*  d_shared_w1{nullptr};  // [2048, 4096]
    half*  d_shared_w2{nullptr};  // [4096, 2048]
    half*  d_shared_w3{nullptr};  // [2048, 4096]

    // MoE Router Weights
    int64_t* d_tid2eid{nullptr};     // [129280, 6] (for hash layers)
    half*    d_gate_weight{nullptr}; // [256, 4096]

    // Persistent Sliding-Window KV Cache on Device: [max_seq_len, 512]
    half*  d_kv_cache{nullptr};
    uint32_t max_seq_len_{4096};

    // Tier 1 VRAM LRU Cache for Routed Experts
    uint32_t vram_capacity_{8};
    std::vector<VRAMExpertSlot> vram_slots_;
    std::vector<uint32_t> free_slots_;
    std::list<uint32_t> lru_list_;
    std::unordered_map<uint32_t, std::pair<uint32_t, std::list<uint32_t>::iterator>> lru_map_;

    // Tier 2 Host / Mmap Sources for 256 Routed Experts
    std::vector<HostExpertSource> host_experts_;

    // Cache Stats
    uint64_t cache_hits{0};
    uint64_t cache_misses{0};

    void init(int id, const SafetensorsLoader& loader, uint32_t max_seq = 4096, uint32_t vram_slots = 8) {
        layer_id = id;
        is_hash_layer = (id < 3);
        max_seq_len_ = max_seq;
        vram_capacity_ = vram_slots;

        std::string pfx = "layers." + std::to_string(layer_id) + ".";

        // 1. Attention Weights
        upload_tensor(loader, pfx + "attn_norm.weight", &d_attn_norm);
        upload_tensor(loader, pfx + "attn.wq_a.weight", &d_wq_a);
        upload_tensor(loader, pfx + "attn.q_norm.weight", &d_q_norm);
        upload_tensor(loader, pfx + "attn.wq_b.weight", &d_wq_b);
        upload_tensor(loader, pfx + "attn.wkv.weight", &d_wkv);
        upload_tensor(loader, pfx + "attn.kv_norm.weight", &d_kv_norm);
        upload_tensor(loader, pfx + "attn.attn_sink", &d_attn_sink);
        upload_tensor(loader, pfx + "attn.wo_a.weight", &d_wo_a);
        upload_tensor(loader, pfx + "attn.wo_b.weight", &d_wo_b);

        upload_tensor(loader, pfx + "hc_attn_fn", &d_hc_attn_fn);
        upload_tensor(loader, pfx + "hc_attn_base", &d_hc_attn_base);
        upload_tensor(loader, pfx + "hc_attn_scale", &d_hc_attn_scale);

        // 2. FFN Weights
        upload_tensor(loader, pfx + "ffn_norm.weight", &d_ffn_norm);
        upload_tensor(loader, pfx + "hc_ffn_fn", &d_hc_ffn_fn);
        upload_tensor(loader, pfx + "hc_ffn_base", &d_hc_ffn_base);
        upload_tensor(loader, pfx + "hc_ffn_scale", &d_hc_ffn_scale);

        // 3. Shared Expert
        upload_tensor(loader, pfx + "ffn.shared_experts.w1.weight", &d_shared_w1);
        upload_tensor(loader, pfx + "ffn.shared_experts.w2.weight", &d_shared_w2);
        upload_tensor(loader, pfx + "ffn.shared_experts.w3.weight", &d_shared_w3);

        // 4. Router
        if (is_hash_layer) {
            upload_tensor(loader, pfx + "ffn.gate.tid2eid", &d_tid2eid);
        }
        upload_tensor(loader, pfx + "ffn.gate.weight", &d_gate_weight);

        // 5. Allocate Persistent KV Cache on Device
        CHECK_HIP(hipMalloc(&d_kv_cache, max_seq_len_ * kernel::DSV4_HEAD_DIM * sizeof(half)));
        CHECK_HIP(hipMemset(d_kv_cache, 0, max_seq_len_ * kernel::DSV4_HEAD_DIM * sizeof(half)));

        // 6. Index Host / Mmap Sources for 256 Routed Experts
        host_experts_.resize(256);
        for (int e = 0; e < 256; ++e) {
            std::string exp_pfx = pfx + "ffn.experts." + std::to_string(e) + ".";
            host_experts_[e].w1_packed = loader.get_data_ptr<uint32_t>(exp_pfx + "w1.weight_packed");
            host_experts_[e].w1_scale  = loader.get_data_ptr<half>(exp_pfx + "w1.weight_scale");
            host_experts_[e].w2_packed = loader.get_data_ptr<uint32_t>(exp_pfx + "w2.weight_packed");
            host_experts_[e].w2_scale  = loader.get_data_ptr<half>(exp_pfx + "w2.weight_scale");
            host_experts_[e].w3_packed = loader.get_data_ptr<uint32_t>(exp_pfx + "w3.weight_packed");
            host_experts_[e].w3_scale  = loader.get_data_ptr<half>(exp_pfx + "w3.weight_scale");
        }

        // 7. Allocate Tier 1 VRAM LRU Cache Slots
        vram_slots_.resize(vram_capacity_);
        free_slots_.clear();
        lru_list_.clear();
        lru_map_.clear();

        constexpr size_t W1_BYTES = 2048 * 512 * sizeof(uint32_t); // 4MB
        constexpr size_t W1_SC_BYTES = 2048 * 128 * sizeof(half);  // 512KB
        constexpr size_t W2_BYTES = 4096 * 256 * sizeof(uint32_t); // 4MB
        constexpr size_t W2_SC_BYTES = 4096 * 64 * sizeof(half);   // 512KB
        constexpr size_t W3_BYTES = 2048 * 512 * sizeof(uint32_t); // 4MB
        constexpr size_t W3_SC_BYTES = 2048 * 128 * sizeof(half);  // 512KB

        for (uint32_t s = 0; s < vram_capacity_; ++s) {
            CHECK_HIP(hipMalloc(&vram_slots_[s].d_w1_packed, W1_BYTES));
            CHECK_HIP(hipMalloc(&vram_slots_[s].d_w1_scale, W1_SC_BYTES));
            CHECK_HIP(hipMalloc(&vram_slots_[s].d_w2_packed, W2_BYTES));
            CHECK_HIP(hipMalloc(&vram_slots_[s].d_w2_scale, W2_SC_BYTES));
            CHECK_HIP(hipMalloc(&vram_slots_[s].d_w3_packed, W3_BYTES));
            CHECK_HIP(hipMalloc(&vram_slots_[s].d_w3_scale, W3_SC_BYTES));
            vram_slots_[s].resident_expert_id = -1;
            free_slots_.push_back(s);
        }
    }

    // Access or Stream an Expert into Tier 1 VRAM Slot
    uint32_t acquire_expert_slot(uint32_t expert_id, hipStream_t stream = 0) {
        auto it = lru_map_.find(expert_id);
        if (it != lru_map_.end()) {
            cache_hits++;
            uint32_t slot_idx = it->second.first;
            lru_list_.erase(it->second.second);
            lru_list_.push_front(expert_id);
            it->second.second = lru_list_.begin();
            return slot_idx;
        }

        cache_misses++;
        uint32_t slot_idx = 0;
        if (!free_slots_.empty()) {
            slot_idx = free_slots_.back();
            free_slots_.pop_back();
        } else {
            // Evict least recently used expert
            uint32_t evict_eid = lru_list_.back();
            lru_list_.pop_back();
            slot_idx = lru_map_[evict_eid].first;
            lru_map_.erase(evict_eid);
        }

        // Stream expert from Tier 2 Host / Mmap into VRAM slot
        const auto& src = host_experts_[expert_id];
        constexpr size_t W1_BYTES = 2048 * 512 * sizeof(uint32_t);
        constexpr size_t W1_SC_BYTES = 2048 * 128 * sizeof(half);
        constexpr size_t W2_BYTES = 4096 * 256 * sizeof(uint32_t);
        constexpr size_t W2_SC_BYTES = 4096 * 64 * sizeof(half);
        constexpr size_t W3_BYTES = 2048 * 512 * sizeof(uint32_t);
        constexpr size_t W3_SC_BYTES = 2048 * 128 * sizeof(half);

        CHECK_HIP(hipMemcpyAsync(vram_slots_[slot_idx].d_w1_packed, src.w1_packed, W1_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(vram_slots_[slot_idx].d_w1_scale, src.w1_scale, W1_SC_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(vram_slots_[slot_idx].d_w2_packed, src.w2_packed, W2_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(vram_slots_[slot_idx].d_w2_scale, src.w2_scale, W2_SC_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(vram_slots_[slot_idx].d_w3_packed, src.w3_packed, W3_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(vram_slots_[slot_idx].d_w3_scale, src.w3_scale, W3_SC_BYTES, hipMemcpyHostToDevice, stream));

        vram_slots_[slot_idx].resident_expert_id = expert_id;
        lru_list_.push_front(expert_id);
        lru_map_[expert_id] = {slot_idx, lru_list_.begin()};

        return slot_idx;
    }

    void free() {
        if (d_attn_norm) (void)hipFree(d_attn_norm);
        if (d_wq_a) (void)hipFree(d_wq_a);
        if (d_q_norm) (void)hipFree(d_q_norm);
        if (d_wq_b) (void)hipFree(d_wq_b);
        if (d_wkv) (void)hipFree(d_wkv);
        if (d_kv_norm) (void)hipFree(d_kv_norm);
        if (d_attn_sink) (void)hipFree(d_attn_sink);
        if (d_wo_a) (void)hipFree(d_wo_a);
        if (d_wo_b) (void)hipFree(d_wo_b);
        if (d_hc_attn_fn) (void)hipFree(d_hc_attn_fn);
        if (d_hc_attn_base) (void)hipFree(d_hc_attn_base);
        if (d_hc_attn_scale) (void)hipFree(d_hc_attn_scale);

        if (d_ffn_norm) (void)hipFree(d_ffn_norm);
        if (d_hc_ffn_fn) (void)hipFree(d_hc_ffn_fn);
        if (d_hc_ffn_base) (void)hipFree(d_hc_ffn_base);
        if (d_hc_ffn_scale) (void)hipFree(d_hc_ffn_scale);

        if (d_shared_w1) (void)hipFree(d_shared_w1);
        if (d_shared_w2) (void)hipFree(d_shared_w2);
        if (d_shared_w3) (void)hipFree(d_shared_w3);

        if (d_tid2eid) (void)hipFree(d_tid2eid);
        if (d_gate_weight) (void)hipFree(d_gate_weight);
        if (d_kv_cache) (void)hipFree(d_kv_cache);

        for (auto& slot : vram_slots_) {
            if (slot.d_w1_packed) (void)hipFree(slot.d_w1_packed);
            if (slot.d_w1_scale) (void)hipFree(slot.d_w1_scale);
            if (slot.d_w2_packed) (void)hipFree(slot.d_w2_packed);
            if (slot.d_w2_scale) (void)hipFree(slot.d_w2_scale);
            if (slot.d_w3_packed) (void)hipFree(slot.d_w3_packed);
            if (slot.d_w3_scale) (void)hipFree(slot.d_w3_scale);
        }
        vram_slots_.clear();
    }

private:
    template<typename T>
    void upload_tensor(const SafetensorsLoader& loader, const std::string& name, T** d_ptr) {
        if (!loader.has_tensor(name)) {
            *d_ptr = nullptr;
            return;
        }
        const auto& t = loader.get_tensor(name);
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(d_ptr), t.byte_size));
        CHECK_HIP(hipMemcpy(*d_ptr, t.data, t.byte_size, hipMemcpyHostToDevice));
    }
};

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
    half*  d_attn_out{nullptr};    // [16, 64, 512]
    half*  d_z_lora{nullptr};      // [16, 8192]
    half*  d_attn_proj{nullptr};   // [16, 4096]

    // MoE Activations
    half*  d_ffn_pre{nullptr};      // [16, 4096]
    half*  d_ffn_norm_act{nullptr}; // [16, 4096]
    float* d_router_logits{nullptr};// [256]
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

    // Head Activations
    half*  d_hc_head_out{nullptr};  // [4096]
    half*  d_head_norm{nullptr};    // [4096]
    half*  d_logits{nullptr};       // [129280]

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
        CHECK_HIP(hipMalloc(&d_attn_out, M * 64 * 512 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_z_lora, M * 8192 * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_attn_proj, M * H * sizeof(half)));

        CHECK_HIP(hipMalloc(&d_ffn_pre, M * H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_ffn_norm_act, M * H * sizeof(half)));
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

        CHECK_HIP(hipMalloc(&d_hc_head_out, H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_head_norm, H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_logits, 129280 * sizeof(half)));

        // Clear initial padded memory
        CHECK_HIP(hipMemset(d_x_pre, 0, M * H * sizeof(half)));
        CHECK_HIP(hipMemset(d_x_norm, 0, M * H * sizeof(half)));
    }

    void free() {
        if (d_res_in) (void)hipFree(d_res_in);
        if (d_res_mid) (void)hipFree(d_res_mid);
        if (d_res_out) (void)hipFree(d_res_out);
        if (d_res_in_half) (void)hipFree(d_res_in_half);
        if (d_res_mid_half) (void)hipFree(d_res_mid_half);
        if (d_res_out_half) (void)hipFree(d_res_out_half);

        if (d_mixes_a) (void)hipFree(d_mixes_a);
        if (d_pre_a) (void)hipFree(d_pre_a);
        if (d_post_a) (void)hipFree(d_post_a);
        if (d_comb_a) (void)hipFree(d_comb_a);

        if (d_mixes_f) (void)hipFree(d_mixes_f);
        if (d_pre_f) (void)hipFree(d_pre_f);
        if (d_post_f) (void)hipFree(d_post_f);
        if (d_comb_f) (void)hipFree(d_comb_f);

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
        if (d_router_logits) (void)hipFree(d_router_logits);
        if (d_topk_weights) (void)hipFree(d_topk_weights);
        if (d_topk_indices) (void)hipFree(d_topk_indices);
        if (d_token_id) (void)hipFree(d_token_id);

        if (d_shared_gate) (void)hipFree(d_shared_gate);
        if (d_shared_up) (void)hipFree(d_shared_up);
        if (d_shared_swiglu) (void)hipFree(d_shared_swiglu);
        if (d_shared_down) (void)hipFree(d_shared_down);

        if (d_moe_accum) (void)hipFree(d_moe_accum);
        if (d_expert_gate) (void)hipFree(d_expert_gate);
        if (d_expert_up) (void)hipFree(d_expert_up);
        if (d_expert_swiglu) (void)hipFree(d_expert_swiglu);
        if (d_expert_down) (void)hipFree(d_expert_down);

        if (d_hc_head_out) (void)hipFree(d_hc_head_out);
        if (d_head_norm) (void)hipFree(d_head_norm);
        if (d_logits) (void)hipFree(d_logits);
    }
};

// Complete DeepSeek-V4 Autoregressive Multi-Layer Pipeline Engine
class V4Pipeline {
public:
    SafetensorsLoader loader;
    std::vector<std::unique_ptr<V4Layer>> layers;
    PipelineScratchBuffers scratch;
    kernel::RopeTable rope_table;

    // Model Level Resident Weights
    const half* host_embed_table{nullptr}; // [129280, 4096] in host / mmap
    float* d_hc_head_fn{nullptr};          // [4, 16384]
    float* d_hc_head_base{nullptr};        // [4]
    float* d_hc_head_scale{nullptr};       // [1]
    half*  d_lm_head{nullptr};             // [129280, 4096] on device
    half*  d_final_norm{nullptr};          // [4096] on device

    hipStream_t compute_stream{0};
    hipStream_t sdma_stream{0};

    uint32_t num_layers_{0};
    uint32_t current_seq_len_{0};

    V4Pipeline() = default;

    ~V4Pipeline() {
        free_all();
    }

    void init(
        const std::string& snapshot_dir,
        uint32_t num_layers = 2,
        uint32_t vram_slots_per_layer = 8,
        uint32_t max_seq_len = 4096
    ) {
        std::cout << "================================================================================" << std::endl;
        std::cout << "        Project Aeon — DeepSeek-V4 Multi-Layer Execution Pipeline               " << std::endl;
        std::cout << "================================================================================" << std::endl;

        num_layers_ = num_layers;
        current_seq_len_ = 0;

        // 1. Initialize streams
        CHECK_HIP(hipStreamCreate(&compute_stream));
        CHECK_HIP(hipStreamCreate(&sdma_stream));

        // 2. Open Safetensors Shards via Zero-Copy Mmap
        std::cout << "[Pipeline] Opening Safetensors shards from " << snapshot_dir << "..." << std::endl;
        loader.open_shard(snapshot_dir + "/model-00001.safetensors");
        loader.open_shard(snapshot_dir + "/model-00002.safetensors");
        std::cout << "  > Total tensors indexed: " << loader.total_tensors() << std::endl;

        // 3. Initialize RoPE Tables
        std::cout << "[Pipeline] Initializing RoPE tables (max_seq=" << max_seq_len << ")..." << std::endl;
        rope_table.init(max_seq_len, kernel::DSV4_ROPE_THETA, 1.0f);

        // Upload RoPE caches to GPU
        size_t rope_bytes = rope_table.max_seq_len * rope_table.half_rope * sizeof(float);
        CHECK_HIP(hipMalloc(&d_cos_cache_, rope_bytes));
        CHECK_HIP(hipMalloc(&d_sin_cache_, rope_bytes));
        CHECK_HIP(hipMemcpy(d_cos_cache_, rope_table.cos_cache.data(), rope_bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_sin_cache_, rope_table.sin_cache.data(), rope_bytes, hipMemcpyHostToDevice));

        // 4. Model-level Weights
        std::cout << "[Pipeline] Binding model-level embeddings and LM head..." << std::endl;
        host_embed_table = loader.get_data_ptr<half>("embed.weight");

        const auto& head_t = loader.get_tensor("head.weight");
        std::cout << "  > Uploading LM Head [129280, 4096] (" << (head_t.byte_size / (1024*1024)) << " MB) to VRAM..." << std::endl;
        CHECK_HIP(hipMalloc(&d_lm_head, head_t.byte_size));
        CHECK_HIP(hipMemcpy(d_lm_head, head_t.data, head_t.byte_size, hipMemcpyHostToDevice));

        // HC Head
        const auto& fn_t = loader.get_tensor("hc_head_fn");
        const auto& base_t = loader.get_tensor("hc_head_base");
        const auto& sc_t = loader.get_tensor("hc_head_scale");
        CHECK_HIP(hipMalloc(&d_hc_head_fn, fn_t.byte_size));
        CHECK_HIP(hipMalloc(&d_hc_head_base, base_t.byte_size));
        CHECK_HIP(hipMalloc(&d_hc_head_scale, sc_t.byte_size));
        CHECK_HIP(hipMemcpy(d_hc_head_fn, fn_t.data, fn_t.byte_size, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_hc_head_base, base_t.data, base_t.byte_size, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_hc_head_scale, sc_t.data, sc_t.byte_size, hipMemcpyHostToDevice));

        // Final norm: use layers.0.ffn_norm.weight as fallback if model-level norm.weight is in an un-downloaded shard
        if (loader.has_tensor("norm.weight")) {
            const auto& norm_t = loader.get_tensor("norm.weight");
            CHECK_HIP(hipMalloc(&d_final_norm, norm_t.byte_size));
            CHECK_HIP(hipMemcpy(d_final_norm, norm_t.data, norm_t.byte_size, hipMemcpyHostToDevice));
        } else {
            const auto& norm_t = loader.get_tensor("layers.0.ffn_norm.weight");
            CHECK_HIP(hipMalloc(&d_final_norm, norm_t.byte_size));
            CHECK_HIP(hipMemcpy(d_final_norm, norm_t.data, norm_t.byte_size, hipMemcpyHostToDevice));
        }

        // 5. Allocate Reusable Pipeline Scratch Buffers
        std::cout << "[Pipeline] Allocating intermediate GPU scratch buffers..." << std::endl;
        scratch.allocate();

        // 6. Initialize Consecutive Transformer Layers
        layers.resize(num_layers_);
        for (uint32_t l = 0; l < num_layers_; ++l) {
            std::cout << "[Pipeline] Initializing DeepSeek-V4 Layer " << l << " (Tier 1 slots=" << vram_slots_per_layer << ")..." << std::endl;
            layers[l] = std::make_unique<V4Layer>();
            layers[l]->init(l, loader, max_seq_len, vram_slots_per_layer);
        }

        std::cout << "[Pipeline] Engine ready! Configured for " << num_layers_ << " chained layers." << std::endl;
    }

    // Run Single Autoregressive Step for token_id at sequence position `pos`
    // Returns next token ID via greedy argmax
    uint32_t step(uint32_t token_id, uint32_t pos) {
        constexpr int H = kernel::DSV4_HIDDEN_SIZE; // 4096
        constexpr int HC = 4;
        constexpr int HC_DIM = HC * H;             // 16384
        constexpr int M_PAD = 16;
        constexpr int Q_LORA = kernel::DSV4_Q_LORA_RANK;
        constexpr int HEAD_DIM = kernel::DSV4_HEAD_DIM;
        constexpr int NUM_HEADS = kernel::DSV4_NUM_HEADS;
        constexpr int TOTAL_Q = NUM_HEADS * HEAD_DIM;
        constexpr int O_LORA = kernel::DSV4_O_LORA_RANK;
        constexpr int O_GROUPS = kernel::DSV4_O_GROUPS;
        constexpr int TOT_LORA = kernel::DSV4_TOTAL_O_LORA_DIM;
        constexpr int INTER_DIM = 2048;

        // 1. Embed Token & Replicate to 4 HC streams
        const half* token_emb = host_embed_table + token_id * H;

        // Replicate embedding across 4 streams into d_res_in_half on GPU
        // Shape [4, 4096]
        for (int s = 0; s < HC; ++s) {
            CHECK_HIP(hipMemcpyAsync(scratch.d_res_in_half + s * H, token_emb, H * sizeof(half), hipMemcpyHostToDevice, compute_stream));
        }
        // Convert to float for HC pre-mix
        v4_half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, compute_stream>>>(scratch.d_res_in_half, scratch.d_res_in, HC_DIM);

        // Upload single token ID for hash routing
        int32_t h_token = static_cast<int32_t>(token_id);
        CHECK_HIP(hipMemcpyAsync(scratch.d_token_id, &h_token, sizeof(int32_t), hipMemcpyHostToDevice, compute_stream));

        // 2. Execute Consecutive Transformer Layers
        for (uint32_t l = 0; l < num_layers_; ++l) {
            auto& layer = *layers[l];

            // -----------------------------------------------------------------
            // A. Hyper-Connections Attention Pre-Mix & Sinkhorn
            // -----------------------------------------------------------------
            // Host mixes projection or device mixes projection
            // mixes = (res @ fn.T) * rms
            // Mixes projection (24 outputs from 16384 float inputs)
            // We launch Wave32 GEMV for mixes [24, 16384]
            // We use cpu/device helper or kernel
            // Since HC_DIM = 16384, we do dot products on compute_stream:
            // First compute mean square
            std::vector<float> h_res(HC_DIM);
            CHECK_HIP(hipMemcpyAsync(h_res.data(), scratch.d_res_in, HC_DIM * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            float sqrsum = 0.0f;
            for (int i = 0; i < HC_DIM; ++i) sqrsum += h_res[i] * h_res[i];
            float rms = 1.0f / std::sqrt((sqrsum / (float)HC_DIM) + 1e-6f);

            // Fetch fn from loader for layer l
            std::string pfx = "layers." + std::to_string(l) + ".";
            const float* hc_fn_ptr = loader.get_data_ptr<float>(pfx + "hc_attn_fn");
            std::vector<float> h_mixes_a(24);
            for (int j = 0; j < 24; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < HC_DIM; ++k) dot += h_res[k] * hc_fn_ptr[j * HC_DIM + k];
                h_mixes_a[j] = dot * rms;
            }
            CHECK_HIP(hipMemcpyAsync(scratch.d_mixes_a, h_mixes_a.data(), 24 * sizeof(float), hipMemcpyHostToDevice, compute_stream));

            // Sinkhorn Normalize Kernel
            hipLaunchKernelGGL(
                kernel::hc_sinkhorn_normalize_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_mixes_a, layer.d_hc_attn_scale, layer.d_hc_attn_base,
                scratch.d_pre_a, scratch.d_post_a, scratch.d_comb_a,
                1e-6f, 1e-6f, 2.0f, 20
            );

            // Pre-combine streams: x_pre = sum_{s} pre_a[s] * res_in[s]
            std::vector<float> h_pre_a(HC);
            CHECK_HIP(hipMemcpyAsync(h_pre_a.data(), scratch.d_pre_a, HC * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            std::vector<half> h_x_pre(H);
            for (int h = 0; h < H; ++h) {
                float acc = 0.0f;
                for (int s = 0; s < HC; ++s) acc += h_pre_a[s] * h_res[s * H + h];
                h_x_pre[h] = __float2half(acc);
            }
            // Copy row 0 into d_x_pre (M_PAD rows)
            CHECK_HIP(hipMemcpyAsync(scratch.d_x_pre, h_x_pre.data(), H * sizeof(half), hipMemcpyHostToDevice, compute_stream));

            // -----------------------------------------------------------------
            // B. Attention RMSNorm
            // -----------------------------------------------------------------
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_x_pre, layer.d_attn_norm, scratch.d_x_norm, H, 1e-6f
            );

            // -----------------------------------------------------------------
            // C. MLA Projections
            // -----------------------------------------------------------------
            // Q_a = x_norm @ wq_a.T [1024]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(Q_LORA, 1), dim3(32), 0, compute_stream,
                scratch.d_x_norm, layer.d_wq_a, scratch.d_qa, H
            );

            // Q_a RMSNorm
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_qa, layer.d_q_norm, scratch.d_qa_norm, Q_LORA, 1e-6f
            );

            // Q = qa_norm @ wq_b.T [64 * 512 = 32768]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(TOTAL_Q, 1), dim3(32), 0, compute_stream,
                scratch.d_qa_norm, layer.d_wq_b, scratch.d_q, Q_LORA
            );

            // KV = x_norm @ wkv.T [512]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(HEAD_DIM, 1), dim3(32), 0, compute_stream,
                scratch.d_x_norm, layer.d_wkv, scratch.d_kv, H
            );

            // KV RMSNorm
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_kv, layer.d_kv_norm, scratch.d_kv_norm_act, HEAD_DIM, 1e-6f
            );

            // -----------------------------------------------------------------
            // D. RoPE & KV Cache Persistence
            // -----------------------------------------------------------------
            hipLaunchKernelGGL(
                kernel::v4_forward_rope_at_pos_wave32_kernel,
                dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                scratch.d_q, d_cos_cache_, d_sin_cache_, pos,
                NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2
            );

            hipLaunchKernelGGL(
                kernel::v4_forward_rope_at_pos_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_kv_norm_act, d_cos_cache_, d_sin_cache_, pos,
                1, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2
            );

            // Insert new KV vector into layer's persistent KV Cache at `pos`
            CHECK_HIP(hipMemcpyAsync(
                layer.d_kv_cache + pos * HEAD_DIM,
                scratch.d_kv_norm_act,
                HEAD_DIM * sizeof(half),
                hipMemcpyDeviceToDevice,
                compute_stream
            ));

            // -----------------------------------------------------------------
            // E. Autoregressive Sliding-Window Attention over Cached States
            // -----------------------------------------------------------------
            hipLaunchKernelGGL(
                kernel::v4_cached_sliding_window_attn_wave32_kernel,
                dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                scratch.d_q, layer.d_kv_cache, layer.d_attn_sink, scratch.d_attn_out,
                pos, kernel::DSV4_SLIDING_WINDOW, kernel::DSV4_ATTN_SCALE
            );

            // Inverse RoPE on attention output
            hipLaunchKernelGGL(
                kernel::v4_inverse_rope_at_pos_wave32_kernel,
                dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                scratch.d_attn_out, d_cos_cache_, d_sin_cache_, pos,
                NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2
            );

            // Grouped W_o_a: 8 groups x [1024, 4096] -> [8192]
            hipLaunchKernelGGL(
                kernel::v4_grouped_wo_a_wave32_kernel,
                dim3(O_LORA, O_GROUPS, 1), dim3(32), 0, compute_stream,
                scratch.d_attn_out, layer.d_wo_a, scratch.d_z_lora, 1
            );

            // W_o_b: [4096, 8192] -> [4096]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(H, 1), dim3(32), 0, compute_stream,
                scratch.d_z_lora, layer.d_wo_b, scratch.d_attn_proj, TOT_LORA
            );

            // -----------------------------------------------------------------
            // F. HC Attention Post Expansion: res_mid = comb_a * res_in + post_a * attn_proj
            // -----------------------------------------------------------------
            v4_float_to_half_kernel<<<(HC_DIM + 255) / 256, 256, 0, compute_stream>>>(scratch.d_res_in, scratch.d_res_in_half, HC_DIM);

            hipLaunchKernelGGL(
                kernel::hc_post_kernel,
                dim3((H + 255) / 256, 1), dim3(256), 0, compute_stream,
                scratch.d_attn_proj, scratch.d_res_in_half, scratch.d_post_a, scratch.d_comb_a, scratch.d_res_mid_half, H
            );

            v4_half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, compute_stream>>>(scratch.d_res_mid_half, scratch.d_res_mid, HC_DIM);

            // -----------------------------------------------------------------
            // G. HC FFN Pre-Mix & Sinkhorn
            // -----------------------------------------------------------------
            std::vector<float> h_res_mid(HC_DIM);
            CHECK_HIP(hipMemcpyAsync(h_res_mid.data(), scratch.d_res_mid, HC_DIM * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            float sqrsum_f = 0.0f;
            for (int i = 0; i < HC_DIM; ++i) sqrsum_f += h_res_mid[i] * h_res_mid[i];
            float rms_f = 1.0f / std::sqrt((sqrsum_f / (float)HC_DIM) + 1e-6f);

            const float* hc_ffn_fn_ptr = loader.get_data_ptr<float>(pfx + "hc_ffn_fn");
            std::vector<float> h_mixes_f(24);
            for (int j = 0; j < 24; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < HC_DIM; ++k) dot += h_res_mid[k] * hc_ffn_fn_ptr[j * HC_DIM + k];
                h_mixes_f[j] = dot * rms_f;
            }
            CHECK_HIP(hipMemcpyAsync(scratch.d_mixes_f, h_mixes_f.data(), 24 * sizeof(float), hipMemcpyHostToDevice, compute_stream));

            hipLaunchKernelGGL(
                kernel::hc_sinkhorn_normalize_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_mixes_f, layer.d_hc_ffn_scale, layer.d_hc_ffn_base,
                scratch.d_pre_f, scratch.d_post_f, scratch.d_comb_f,
                1e-6f, 1e-6f, 2.0f, 20
            );

            std::vector<float> h_pre_f(HC);
            CHECK_HIP(hipMemcpyAsync(h_pre_f.data(), scratch.d_pre_f, HC * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            std::vector<half> h_ffn_pre(H);
            for (int h = 0; h < H; ++h) {
                float acc = 0.0f;
                for (int s = 0; s < HC; ++s) acc += h_pre_f[s] * h_res_mid[s * H + h];
                h_ffn_pre[h] = __float2half(acc);
            }
            CHECK_HIP(hipMemcpyAsync(scratch.d_ffn_pre, h_ffn_pre.data(), H * sizeof(half), hipMemcpyHostToDevice, compute_stream));

            // FFN RMSNorm
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_ffn_pre, layer.d_ffn_norm, scratch.d_ffn_norm_act, H, 1e-6f
            );

            // Replicate row 0 to M_PAD rows of d_ffn_norm_act for WMMA compatibility
            for (int r = 1; r < M_PAD; ++r) {
                CHECK_HIP(hipMemcpyAsync(scratch.d_ffn_norm_act + r * H, scratch.d_ffn_norm_act, H * sizeof(half), hipMemcpyDeviceToDevice, compute_stream));
            }

            // -----------------------------------------------------------------
            // H. MoE Routing & Expert Execution
            // -----------------------------------------------------------------
            // 1. Router Logits: gate_weight @ ffn_norm_act [256]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(256, 1), dim3(32), 0, compute_stream,
                scratch.d_ffn_norm_act, layer.d_gate_weight, reinterpret_cast<half*>(scratch.d_router_logits), H
            );
            // Convert logits from half to float in-place
            // Or router_logits directly
            std::vector<half> h_rlogits_half(256);
            CHECK_HIP(hipMemcpyAsync(h_rlogits_half.data(), scratch.d_router_logits, 256 * sizeof(half), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));
            std::vector<float> h_rlogits(256);
            for (int i = 0; i < 256; ++i) h_rlogits[i] = __half2float(h_rlogits_half[i]);
            CHECK_HIP(hipMemcpyAsync(scratch.d_router_logits, h_rlogits.data(), 256 * sizeof(float), hipMemcpyHostToDevice, compute_stream));

            // Launch Router Kernel (Hash mode for layers 0..2)
            hipLaunchKernelGGL(
                kernel::moe_router_kernel,
                dim3(1), dim3(64), 0, compute_stream,
                scratch.d_router_logits, nullptr, layer.d_tid2eid, scratch.d_token_id,
                scratch.d_topk_weights, scratch.d_topk_indices,
                256, 6, 1.5f, true
            );

            std::vector<float> h_topk_weights(6);
            std::vector<int32_t> h_topk_indices(6);
            CHECK_HIP(hipMemcpyAsync(h_topk_weights.data(), scratch.d_topk_weights, 6 * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipMemcpyAsync(h_topk_indices.data(), scratch.d_topk_indices, 6 * sizeof(int32_t), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            // Clear MoE accumulation buffer
            CHECK_HIP(hipMemsetAsync(scratch.d_moe_accum, 0, M_PAD * H * sizeof(half), compute_stream));

            // 2. Shared Expert (FP16 unquantized)
            // w1 [2048, 4096] & w3 [2048, 4096]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(INTER_DIM, 1), dim3(32), 0, compute_stream,
                scratch.d_ffn_norm_act, layer.d_shared_w1, scratch.d_shared_gate, H
            );
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(INTER_DIM, 1), dim3(32), 0, compute_stream,
                scratch.d_ffn_norm_act, layer.d_shared_w3, scratch.d_shared_up, H
            );

            // Clamped SwiGLU
            int swiglu_threads = 256;
            int swiglu_blocks = (INTER_DIM + swiglu_threads - 1) / swiglu_threads;
            hipLaunchKernelGGL(
                v4_pipeline_swiglu_clamp_kernel,
                dim3(swiglu_blocks), dim3(swiglu_threads), 0, compute_stream,
                scratch.d_shared_gate, scratch.d_shared_up, scratch.d_shared_swiglu, INTER_DIM, 10.0f
            );

            // w2 [4096, 2048]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(H, 1), dim3(32), 0, compute_stream,
                scratch.d_shared_swiglu, layer.d_shared_w2, scratch.d_moe_accum, INTER_DIM
            );

            // 3. 6 Routed Experts (INT4-W4A16 WMMA GEMM)
            for (int k = 0; k < 6; ++k) {
                uint32_t expert_id = h_topk_indices[k];
                float expert_weight = h_topk_weights[k];

                uint32_t slot = layer.acquire_expert_slot(expert_id, compute_stream);
                const auto& eslot = layer.vram_slots_[slot];

                // w1 & w3 via fused W4A16 WMMA
                kernel::dispatch_w4a16_gemm(
                    scratch.d_ffn_norm_act, eslot.d_w1_packed, eslot.d_w1_scale,
                    scratch.d_expert_gate, M_PAD, INTER_DIM, H, compute_stream
                );
                kernel::dispatch_w4a16_gemm(
                    scratch.d_ffn_norm_act, eslot.d_w3_packed, eslot.d_w3_scale,
                    scratch.d_expert_up, M_PAD, INTER_DIM, H, compute_stream
                );

                // SwiGLU clamp
                int m_swiglu_blocks = (M_PAD * INTER_DIM + swiglu_threads - 1) / swiglu_threads;
                hipLaunchKernelGGL(
                    v4_pipeline_swiglu_clamp_kernel,
                    dim3(m_swiglu_blocks), dim3(swiglu_threads), 0, compute_stream,
                    scratch.d_expert_gate, scratch.d_expert_up, scratch.d_expert_swiglu, M_PAD * INTER_DIM, 10.0f
                );

                // w2
                kernel::dispatch_w4a16_gemm(
                    scratch.d_expert_swiglu, eslot.d_w2_packed, eslot.d_w2_scale,
                    scratch.d_expert_down, M_PAD, H, INTER_DIM, compute_stream
                );

                // Accumulate row 0
                hipLaunchKernelGGL(
                    v4_pipeline_accumulate_expert_kernel,
                    dim3((H + 255) / 256), dim3(256), 0, compute_stream,
                    scratch.d_moe_accum, scratch.d_expert_down, expert_weight, H
                );
            }

            // -----------------------------------------------------------------
            // I. HC FFN Post Expansion: res_out = comb_f * res_mid + post_f * moe_accum
            // -----------------------------------------------------------------
            hipLaunchKernelGGL(
                kernel::hc_post_kernel,
                dim3((H + 255) / 256, 1), dim3(256), 0, compute_stream,
                scratch.d_moe_accum, scratch.d_res_mid_half, scratch.d_post_f, scratch.d_comb_f, scratch.d_res_out_half, H
            );

            // Copy res_out into res_in for next layer
            CHECK_HIP(hipMemcpyAsync(
                scratch.d_res_in_half, scratch.d_res_out_half, HC_DIM * sizeof(half),
                hipMemcpyDeviceToDevice, compute_stream
            ));
            v4_half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, compute_stream>>>(scratch.d_res_in_half, scratch.d_res_in, HC_DIM);
        }

        // 3. HC Head Reduction on Final Residual
        hipLaunchKernelGGL(
            kernel::hc_head_wave32_kernel,
            dim3(1), dim3(32), 0, compute_stream,
            scratch.d_res_in, d_hc_head_fn, d_hc_head_base, d_hc_head_scale,
            scratch.d_hc_head_out, H, HC, 1e-6f, 1e-6f
        );

        // 4. Final RMSNorm
        hipLaunchKernelGGL(
            kernel::v4_rmsnorm_wave32_kernel,
            dim3(1), dim3(32), 0, compute_stream,
            scratch.d_hc_head_out, d_final_norm, scratch.d_head_norm, H, 1e-6f
        );

        // 5. LM Head Projection: logits = head_norm @ lm_head.T [129280]
        hipLaunchKernelGGL(
            kernel::v4_gemv_fp16_kernel,
            dim3(129280, 1), dim3(32), 0, compute_stream,
            scratch.d_head_norm, d_lm_head, scratch.d_logits, H
        );

        // 6. Argmax Sampling
        std::vector<half> h_logits(129280);
        CHECK_HIP(hipMemcpyAsync(h_logits.data(), scratch.d_logits, 129280 * sizeof(half), hipMemcpyDeviceToHost, compute_stream));
        CHECK_HIP(hipStreamSynchronize(compute_stream));

        uint32_t best_tok = 0;
        float best_val = -1e30f;
        for (uint32_t v = 0; v < 129280; ++v) {
            float val = __half2float(h_logits[v]);
            if (val > best_val) {
                best_val = val;
                best_tok = v;
            }
        }

        current_seq_len_ = pos + 1;
        return best_tok;
    }

    // Prefill Prompt and Generate Next Tokens
    std::vector<uint32_t> generate(
        const std::vector<uint32_t>& prompt,
        uint32_t max_new_tokens = 16,
        double* out_ttft_ms = nullptr,
        double* out_tok_per_sec = nullptr
    ) {
        if (prompt.empty()) return {};

        std::vector<uint32_t> generated;
        current_seq_len_ = 0;

        // Reset layer KV caches
        for (auto& l : layers) {
            CHECK_HIP(hipMemset(l->d_kv_cache, 0, l->max_seq_len_ * kernel::DSV4_HEAD_DIM * sizeof(half)));
        }

        // Prefill Phase
        auto t_prefill_start = std::chrono::high_resolution_clock::now();
        uint32_t next_tok = 0;
        for (size_t i = 0; i < prompt.size(); ++i) {
            next_tok = step(prompt[i], i);
        }
        auto t_prefill_end = std::chrono::high_resolution_clock::now();

        double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
        if (out_ttft_ms) *out_ttft_ms = ttft_ms;

        generated.push_back(next_tok);

        // Autoregressive Decoding Phase
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        for (uint32_t step_idx = 1; step_idx < max_new_tokens; ++step_idx) {
            uint32_t pos = prompt.size() + step_idx - 1;
            next_tok = step(next_tok, pos);
            generated.push_back(next_tok);
        }
        auto t_decode_end = std::chrono::high_resolution_clock::now();

        double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
        double tok_sec = (max_new_tokens > 1) ? ((max_new_tokens - 1) / (decode_ms / 1000.0)) : 0.0;
        if (out_tok_per_sec) *out_tok_per_sec = tok_sec;

        return generated;
    }

    void free_all() {
        if (compute_stream) { (void)hipStreamDestroy(compute_stream); compute_stream = 0; }
        if (sdma_stream) { (void)hipStreamDestroy(sdma_stream); sdma_stream = 0; }
        if (d_cos_cache_) { (void)hipFree(d_cos_cache_); d_cos_cache_ = nullptr; }
        if (d_sin_cache_) { (void)hipFree(d_sin_cache_); d_sin_cache_ = nullptr; }
        if (d_lm_head) { (void)hipFree(d_lm_head); d_lm_head = nullptr; }
        if (d_hc_head_fn) { (void)hipFree(d_hc_head_fn); d_hc_head_fn = nullptr; }
        if (d_hc_head_base) { (void)hipFree(d_hc_head_base); d_hc_head_base = nullptr; }
        if (d_hc_head_scale) { (void)hipFree(d_hc_head_scale); d_hc_head_scale = nullptr; }
        if (d_final_norm) { (void)hipFree(d_final_norm); d_final_norm = nullptr; }

        scratch.free();
        for (auto& l : layers) {
            if (l) l->free();
        }
        layers.clear();
        loader.close_all();
    }

private:
    float* d_cos_cache_{nullptr};
    float* d_sin_cache_{nullptr};
};

} // namespace aeon::core
