#pragma once

#include "core/aeon_loader.hpp"
#include "core/safetensors_loader.hpp"
#include "kernel/v4_attention.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)
#endif

namespace aeon::core {

// Tier 1 VRAM Slot for an INT4-W4A16 Routed Expert (Local Mode Fallback)
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
    const float* h_hc_attn_fn{nullptr};
    float* d_hc_attn_base{nullptr};  // [24]
    float* d_hc_attn_scale{nullptr}; // [3]

    // FFN Weights on Device
    half*  d_ffn_norm{nullptr};   // [4096]
    float* d_hc_ffn_fn{nullptr};    // [24, 16384]
    const float* h_hc_ffn_fn{nullptr};
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
    void init_with_loader(int id, const LoaderT& loader, uint32_t max_seq = 4096, uint32_t vram_slots = 8) {
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
        if (loader.has_tensor(pfx + "hc_attn_fn")) {
            h_hc_attn_fn = loader.template get_data_ptr<float>(pfx + "hc_attn_fn");
        }
        upload_tensor(loader, pfx + "hc_attn_base", &d_hc_attn_base);
        upload_tensor(loader, pfx + "hc_attn_scale", &d_hc_attn_scale);

        // 2. FFN Weights
        upload_tensor(loader, pfx + "ffn_norm.weight", &d_ffn_norm);
        upload_tensor(loader, pfx + "hc_ffn_fn", &d_hc_ffn_fn);
        if (loader.has_tensor(pfx + "hc_ffn_fn")) {
            h_hc_ffn_fn = loader.template get_data_ptr<float>(pfx + "hc_ffn_fn");
        }
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
        bind_host_experts(loader, pfx);

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

    void init(int id, const SafetensorsLoader& loader, uint32_t max_seq = 4096, uint32_t vram_slots = 8) {
        init_with_loader(id, loader, max_seq, vram_slots);
    }

    void init(int id, const AeonModelLoader& loader, uint32_t max_seq = 4096, uint32_t vram_slots = 8) {
        init_with_loader(id, loader, max_seq, vram_slots);
    }

    // Initialize layer in Global Pool mode (does not allocate private VRAM slots)
    template<typename LoaderT>
    void init_global(int id, const LoaderT& loader, uint32_t max_seq = 4096) {
        layer_id = id;
        is_hash_layer = (id < 3);
        max_seq_len_ = max_seq;
        vram_capacity_ = 0; // Managed by GlobalVRAMExpertPool

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
        if (loader.has_tensor(pfx + "hc_attn_fn")) {
            h_hc_attn_fn = loader.template get_data_ptr<float>(pfx + "hc_attn_fn");
        }
        upload_tensor(loader, pfx + "hc_attn_base", &d_hc_attn_base);
        upload_tensor(loader, pfx + "hc_attn_scale", &d_hc_attn_scale);

        // 2. FFN Weights
        upload_tensor(loader, pfx + "ffn_norm.weight", &d_ffn_norm);
        upload_tensor(loader, pfx + "hc_ffn_fn", &d_hc_ffn_fn);
        if (loader.has_tensor(pfx + "hc_ffn_fn")) {
            h_hc_ffn_fn = loader.template get_data_ptr<float>(pfx + "hc_ffn_fn");
        }
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
        bind_host_experts(loader, pfx);
    }

    // Access or Stream an Expert into Tier 1 VRAM Slot (Local mode fallback)
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
        h_hc_attn_fn = nullptr;

        if (d_ffn_norm) { (void)hipFree(d_ffn_norm); d_ffn_norm = nullptr; }
        if (d_hc_ffn_fn) { (void)hipFree(d_hc_ffn_fn); d_hc_ffn_fn = nullptr; }
        if (d_hc_ffn_base) { (void)hipFree(d_hc_ffn_base); d_hc_ffn_base = nullptr; }
        if (d_hc_ffn_scale) { (void)hipFree(d_hc_ffn_scale); d_hc_ffn_scale = nullptr; }
        h_hc_ffn_fn = nullptr;

        if (d_shared_w1) { (void)hipFree(d_shared_w1); d_shared_w1 = nullptr; }
        if (d_shared_w2) { (void)hipFree(d_shared_w2); d_shared_w2 = nullptr; }
        if (d_shared_w3) { (void)hipFree(d_shared_w3); d_shared_w3 = nullptr; }

        if (d_tid2eid) { (void)hipFree(d_tid2eid); d_tid2eid = nullptr; }
        if (d_gate_weight) { (void)hipFree(d_gate_weight); d_gate_weight = nullptr; }
        if (d_kv_cache) { (void)hipFree(d_kv_cache); d_kv_cache = nullptr; }

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
        h_hc_attn_fn = o.h_hc_attn_fn; o.h_hc_attn_fn = nullptr;
        d_hc_attn_base = o.d_hc_attn_base; o.d_hc_attn_base = nullptr;
        d_hc_attn_scale = o.d_hc_attn_scale; o.d_hc_attn_scale = nullptr;

        d_ffn_norm = o.d_ffn_norm; o.d_ffn_norm = nullptr;
        d_hc_ffn_fn = o.d_hc_ffn_fn; o.d_hc_ffn_fn = nullptr;
        h_hc_ffn_fn = o.h_hc_ffn_fn; o.h_hc_ffn_fn = nullptr;
        d_hc_ffn_base = o.d_hc_ffn_base; o.d_hc_ffn_base = nullptr;
        d_hc_ffn_scale = o.d_hc_ffn_scale; o.d_hc_ffn_scale = nullptr;

        d_shared_w1 = o.d_shared_w1; o.d_shared_w1 = nullptr;
        d_shared_w2 = o.d_shared_w2; o.d_shared_w2 = nullptr;
        d_shared_w3 = o.d_shared_w3; o.d_shared_w3 = nullptr;

        d_tid2eid = o.d_tid2eid; o.d_tid2eid = nullptr;
        d_gate_weight = o.d_gate_weight; o.d_gate_weight = nullptr;

        d_kv_cache = o.d_kv_cache; o.d_kv_cache = nullptr;
        max_seq_len_ = o.max_seq_len_;

        vram_capacity_ = o.vram_capacity_;
        vram_slots_ = std::move(o.vram_slots_);
        free_slots_ = std::move(o.free_slots_);
        lru_list_ = std::move(o.lru_list_);
        lru_map_ = std::move(o.lru_map_);

        host_experts_ = std::move(o.host_experts_);

        cache_hits = o.cache_hits;
        cache_misses = o.cache_misses;
    }

    void bind_host_experts(const SafetensorsLoader& loader, const std::string& pfx) {
        for (int e = 0; e < 256; ++e) {
            std::string exp_pfx = pfx + "ffn.experts." + std::to_string(e) + ".";
            host_experts_[e].w1_packed = loader.get_data_ptr<uint32_t>(exp_pfx + "w1.weight_packed");
            host_experts_[e].w1_scale  = loader.get_data_ptr<half>(exp_pfx + "w1.weight_scale");
            host_experts_[e].w2_packed = loader.get_data_ptr<uint32_t>(exp_pfx + "w2.weight_packed");
            host_experts_[e].w2_scale  = loader.get_data_ptr<half>(exp_pfx + "w2.weight_scale");
            host_experts_[e].w3_packed = loader.get_data_ptr<uint32_t>(exp_pfx + "w3.weight_packed");
            host_experts_[e].w3_scale  = loader.get_data_ptr<half>(exp_pfx + "w3.weight_scale");
        }
    }

    void bind_host_experts(const AeonModelLoader& loader, const std::string& /*pfx*/) {
        for (int e = 0; e < 256; ++e) {
            const uint8_t* raw = loader.get_expert_data(layer_id, e);
            host_experts_[e].w1_packed = reinterpret_cast<const uint32_t*>(raw + AEON_W1_PACKED_OFFSET);
            host_experts_[e].w1_scale  = reinterpret_cast<const half*>(raw + AEON_W1_SCALE_OFFSET);
            host_experts_[e].w2_packed = reinterpret_cast<const uint32_t*>(raw + AEON_W2_PACKED_OFFSET);
            host_experts_[e].w2_scale  = reinterpret_cast<const half*>(raw + AEON_W2_SCALE_OFFSET);
            host_experts_[e].w3_packed = reinterpret_cast<const uint32_t*>(raw + AEON_W3_PACKED_OFFSET);
            host_experts_[e].w3_scale  = reinterpret_cast<const half*>(raw + AEON_W3_SCALE_OFFSET);
        }
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
