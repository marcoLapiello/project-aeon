#pragma once

#include "core/aeon_loader.hpp"
#include "kernel/v4_attention.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
    const int64_t* host_tid2eid{nullptr}; // Host pointer for lookahead routing
    half*    d_gate_weight{nullptr}; // [256, 4096]
    float*   d_gate_bias{nullptr};   // [256] correction bias for non-hash layers

    // Persistent Sliding-Window KV Cache on Device: [max_seq_len, 512]
    half*  d_kv_cache{nullptr};
    uint32_t max_seq_len_{4096};

    // Cache Stats (layer-level hits/misses reported via unified registry)
    uint64_t cache_hits{0};
    uint64_t cache_misses{0};

    V4Layer() = default;

    ~V4Layer() {
        free();
    }

    // Move-only semantics to prevent double-free
    V4Layer(const V4Layer&) = delete;
    V4Layer& operator=(const V4Layer&) = delete;

    V4Layer(V4Layer&& other) noexcept {
        move_from(std::move(other));
    }

    V4Layer& operator=(V4Layer&& other) noexcept {
        if (this != &other) {
            free();
            move_from(std::move(other));
        }
        return *this;
    }

    template<typename LoaderT>
    void init_with_loader(int id, const LoaderT& loader, uint32_t max_seq = 4096) {
        layer_id = id;
        is_hash_layer = (id < 3);
        max_seq_len_ = max_seq;

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
            if (loader.has_tensor(pfx + "ffn.gate.tid2eid")) {
                host_tid2eid = reinterpret_cast<const int64_t*>(loader.get_tensor(pfx + "ffn.gate.tid2eid").data);
            }
        }
        upload_tensor(loader, pfx + "ffn.gate.weight", &d_gate_weight);
        if (!is_hash_layer) {
            upload_tensor(loader, pfx + "ffn.gate.bias", &d_gate_bias);
        }

        // 5. Allocate Persistent KV Cache on Device
        CHECK_HIP(hipMalloc(&d_kv_cache, max_seq_len_ * kernel::DSV4_HEAD_DIM * sizeof(half)));
        CHECK_HIP(hipMemset(d_kv_cache, 0, max_seq_len_ * kernel::DSV4_HEAD_DIM * sizeof(half)));
    }

    void init(int id, const AeonModelLoader& loader, uint32_t max_seq = 4096) {
        init_with_loader(id, loader, max_seq);
    }

    // Initialize layer with unified expert pool (dense weights + KV cache)
    template<typename LoaderT>
    void init_global(int id, const LoaderT& loader, uint32_t max_seq = 4096) {
        init_with_loader(id, loader, max_seq);
    }

    void free() {
        if (d_attn_norm) { (void)hipFree(d_attn_norm); d_attn_norm = nullptr; }
        if (d_wq_a) { (void)hipFree(d_wq_a); d_wq_a = nullptr; }
        if (d_q_norm) { (void)hipFree(d_q_norm); d_q_norm = nullptr; }
        if (d_wq_b) { (void)hipFree(d_wq_b); d_wq_b = nullptr; }
        if (d_wkv) { (void)hipFree(d_wkv); d_wkv = nullptr; }
        if (d_kv_norm) { (void)hipFree(d_kv_norm); d_kv_norm = nullptr; }
        if (d_attn_sink) { (void)hipFree(d_attn_sink); d_attn_sink = nullptr; }
        if (d_wo_a) { (void)hipFree(d_wo_a); d_wo_a = nullptr; }
        if (d_wo_b) { (void)hipFree(d_wo_b); d_wo_b = nullptr; }
        if (d_hc_attn_fn) { (void)hipFree(d_hc_attn_fn); d_hc_attn_fn = nullptr; }
        if (d_hc_attn_base) { (void)hipFree(d_hc_attn_base); d_hc_attn_base = nullptr; }
        if (d_hc_attn_scale) { (void)hipFree(d_hc_attn_scale); d_hc_attn_scale = nullptr; }

        if (d_ffn_norm) { (void)hipFree(d_ffn_norm); d_ffn_norm = nullptr; }
        if (d_hc_ffn_fn) { (void)hipFree(d_hc_ffn_fn); d_hc_ffn_fn = nullptr; }
        if (d_hc_ffn_base) { (void)hipFree(d_hc_ffn_base); d_hc_ffn_base = nullptr; }
        if (d_hc_ffn_scale) { (void)hipFree(d_hc_ffn_scale); d_hc_ffn_scale = nullptr; }

        if (d_shared_w1) { (void)hipFree(d_shared_w1); d_shared_w1 = nullptr; }
        if (d_shared_w2) { (void)hipFree(d_shared_w2); d_shared_w2 = nullptr; }
        if (d_shared_w3) { (void)hipFree(d_shared_w3); d_shared_w3 = nullptr; }

        if (d_tid2eid) { (void)hipFree(d_tid2eid); d_tid2eid = nullptr; }
        if (d_gate_weight) { (void)hipFree(d_gate_weight); d_gate_weight = nullptr; }
        if (d_gate_bias) { (void)hipFree(d_gate_bias); d_gate_bias = nullptr; }
        if (d_kv_cache) { (void)hipFree(d_kv_cache); d_kv_cache = nullptr; }
    }

private:
    void move_from(V4Layer&& o) noexcept {
        layer_id = o.layer_id;
        is_hash_layer = o.is_hash_layer;

        d_attn_norm = o.d_attn_norm; o.d_attn_norm = nullptr;
        d_wq_a = o.d_wq_a; o.d_wq_a = nullptr;
        d_q_norm = o.d_q_norm; o.d_q_norm = nullptr;
        d_wq_b = o.d_wq_b; o.d_wq_b = nullptr;
        d_wkv = o.d_wkv; o.d_wkv = nullptr;
        d_kv_norm = o.d_kv_norm; o.d_kv_norm = nullptr;
        d_attn_sink = o.d_attn_sink; o.d_attn_sink = nullptr;
        d_wo_a = o.d_wo_a; o.d_wo_a = nullptr;
        d_wo_b = o.d_wo_b; o.d_wo_b = nullptr;

        d_hc_attn_fn = o.d_hc_attn_fn; o.d_hc_attn_fn = nullptr;
        d_hc_attn_base = o.d_hc_attn_base; o.d_hc_attn_base = nullptr;
        d_hc_attn_scale = o.d_hc_attn_scale; o.d_hc_attn_scale = nullptr;

        d_ffn_norm = o.d_ffn_norm; o.d_ffn_norm = nullptr;
        d_hc_ffn_fn = o.d_hc_ffn_fn; o.d_hc_ffn_fn = nullptr;
        d_hc_ffn_base = o.d_hc_ffn_base; o.d_hc_ffn_base = nullptr;
        d_hc_ffn_scale = o.d_hc_ffn_scale; o.d_hc_ffn_scale = nullptr;

        d_shared_w1 = o.d_shared_w1; o.d_shared_w1 = nullptr;
        d_shared_w2 = o.d_shared_w2; o.d_shared_w2 = nullptr;
        d_shared_w3 = o.d_shared_w3; o.d_shared_w3 = nullptr;

        d_tid2eid = o.d_tid2eid; o.d_tid2eid = nullptr;
        host_tid2eid = o.host_tid2eid; o.host_tid2eid = nullptr;
        d_gate_weight = o.d_gate_weight; o.d_gate_weight = nullptr;
        d_gate_bias = o.d_gate_bias; o.d_gate_bias = nullptr;

        d_kv_cache = o.d_kv_cache; o.d_kv_cache = nullptr;
        max_seq_len_ = o.max_seq_len_;

        cache_hits = o.cache_hits;
        cache_misses = o.cache_misses;
    }

    template<typename LoaderT, typename T>
    void upload_tensor(const LoaderT& loader, const std::string& name, T** d_ptr) {
        if (!loader.has_tensor(name)) {
            *d_ptr = nullptr;
            return;
        }
        const auto& t = loader.get_tensor(name);
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(d_ptr), t.byte_size));
        CHECK_HIP(hipMemcpy(*d_ptr, t.data, t.byte_size, hipMemcpyHostToDevice));
    }
};

} // namespace aeon::core
