#pragma once

#include "core/v4_dense_weight_binding.hpp"
#include "kernel/v4_attention.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <list>
#include <utility>

namespace aeon::core {

// Device Context and Weights for One Complete Transformer Block
class V4Layer : public V4DenseWeightBinding {
public:
    int layer_id{0};
    bool is_hash_layer{true};

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

    V4Layer(V4Layer&& other) noexcept
        : V4DenseWeightBinding(std::move(other)) {
        move_from(std::move(other));
    }

    V4Layer& operator=(V4Layer&& other) noexcept {
        if (this != &other) {
            free();
            V4DenseWeightBinding::operator=(std::move(other));
            move_from(std::move(other));
        }
        return *this;
    }

    template<typename LoaderT>
    void init_with_loader(int id, const LoaderT& loader, uint32_t max_seq = 4096) {
        layer_id = id;
        is_hash_layer = (id < 3);
        max_seq_len_ = max_seq;
        bind_dense_weights(layer_id, is_hash_layer, loader);

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
        free_dense_weights();
        if (d_kv_cache) { (void)hipFree(d_kv_cache); d_kv_cache = nullptr; }
    }

private:
    void move_from(V4Layer&& o) noexcept {
        layer_id = o.layer_id;
        is_hash_layer = o.is_hash_layer;
        d_kv_cache = o.d_kv_cache; o.d_kv_cache = nullptr;
        max_seq_len_ = o.max_seq_len_;

        cache_hits = o.cache_hits;
        cache_misses = o.cache_misses;
    }

};

} // namespace aeon::core
