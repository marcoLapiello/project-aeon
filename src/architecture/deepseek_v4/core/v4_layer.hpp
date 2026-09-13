#pragma once

#include "architecture/deepseek_v4/core/v4_dense_weight_binding.hpp"
#include "architecture/deepseek_v4/core/v4_layer_state.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace aeon::core {

// Device Context and Weights for One Complete Transformer Block
class V4Layer : public V4DenseWeightBinding {
public:
    int layer_id{0};
    bool is_hash_layer{true};
    half* d_local_key_cache{nullptr};
    half* d_local_value_cache{nullptr};
    int64_t* d_local_positions{nullptr};
    half* d_kv_cache{nullptr};

    half* d_compressed_key_cache{nullptr};
    half* d_compressed_value_cache{nullptr};
    int64_t* d_compressed_positions{nullptr};

    float* d_compressor_partial_kv{nullptr};
    float* d_compressor_partial_score{nullptr};
    int64_t* d_compressor_partial_positions{nullptr};

    half* d_indexer_key_cache{nullptr};
    int64_t* d_indexer_positions{nullptr};
    float* d_indexer_partial_kv{nullptr};
    float* d_indexer_partial_score{nullptr};
    int64_t* d_indexer_partial_positions{nullptr};
    half* d_indexer_query{nullptr};
    float* d_indexer_weights{nullptr};
    float* d_indexer_scores{nullptr};
    int32_t* d_indexer_topk_indices{nullptr};

    uint32_t max_seq_len_{4096};
    uint32_t local_valid_count_{0};
    uint32_t compressor_partial_count_{0};
    uint32_t compressed_entry_count_{0};
    uint32_t indexer_candidate_count_{0};

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
    void init_with_loader(const V4LayerSpec& spec, const LoaderT& loader, uint32_t max_seq = 4096) {
        free();
        model_spec_ = spec;
        layer_id = static_cast<int>(spec.layer_id);
        is_hash_layer = (spec.layer_id < 3);
        max_seq_len_ = max_seq;
        state_layout_ = V4LayerStateLayout::from_spec(model_spec_, max_seq_len_);
        bind_dense_weights(model_spec_, loader);
        try {
            allocate_state();
        } catch (...) {
            free();
            throw;
        }
    }

    void initialize_state(const V4LayerSpec& spec, uint32_t max_seq = 4096) {
        free();
        model_spec_ = spec;
        layer_id = static_cast<int>(spec.layer_id);
        is_hash_layer = (spec.layer_id < 3);
        max_seq_len_ = max_seq;
        state_layout_ = V4LayerStateLayout::from_spec(model_spec_, max_seq_len_);
        try {
            allocate_state();
        } catch (...) {
            free();
            throw;
        }
    }

    const V4LayerSpec& spec() const noexcept {
        return model_spec_;
    }

    const V4LayerStateLayout& state_layout() const noexcept {
        return state_layout_;
    }

    uint32_t local_cache_capacity() const noexcept {
        return state_layout_.local_capacity;
    }

    void record_position(uint64_t position) {
        if (position >= max_seq_len_) {
            throw std::out_of_range("V4Layer: position exceeds configured context capacity");
        }
        local_valid_count_ = std::min<uint32_t>(
            state_layout_.local_capacity,
            static_cast<uint32_t>(position + 1));
        if (!state_layout_.is_compressed()) return;

        compressor_partial_count_ = std::min<uint32_t>(
            state_layout_.compressor_partial_capacity,
            static_cast<uint32_t>(position + 1));
        compressed_entry_count_ = std::min<uint32_t>(
            state_layout_.compressed_capacity,
            static_cast<uint32_t>((position + 1) / static_cast<uint64_t>(state_layout_.compression_ratio)));
        indexer_candidate_count_ = state_layout_.uses_indexer() ? compressed_entry_count_ : 0;
    }

    void reset_generation_state() {
        clear_state_buffer(d_local_key_cache, state_layout_.local_vector_bytes());
        clear_state_buffer(d_local_value_cache, state_layout_.local_vector_bytes());
        clear_state_buffer(d_local_positions, state_layout_.local_metadata_bytes(), 0xFF);
        clear_state_buffer(d_compressed_key_cache, state_layout_.compressed_vector_bytes());
        clear_state_buffer(d_compressed_value_cache, state_layout_.compressed_vector_bytes());
        clear_state_buffer(d_compressed_positions, state_layout_.compressed_metadata_bytes(), 0xFF);
        clear_state_buffer(d_compressor_partial_kv, state_layout_.compressor_partial_vector_bytes());
        clear_state_buffer(d_compressor_partial_score, state_layout_.compressor_partial_vector_bytes());
        clear_state_buffer(d_compressor_partial_positions, state_layout_.compressor_partial_metadata_bytes(), 0xFF);
        clear_state_buffer(d_indexer_key_cache, state_layout_.indexer_key_bytes());
        clear_state_buffer(d_indexer_positions, state_layout_.indexer_metadata_bytes(), 0xFF);
        clear_state_buffer(d_indexer_partial_kv, state_layout_.indexer_partial_vector_bytes());
        clear_state_buffer(d_indexer_partial_score, state_layout_.indexer_partial_vector_bytes());
        clear_state_buffer(d_indexer_partial_positions, state_layout_.indexer_partial_metadata_bytes(), 0xFF);
        clear_state_buffer(d_indexer_query, state_layout_.indexer_query_bytes());
        clear_state_buffer(d_indexer_weights, state_layout_.indexer_weights_bytes());
        clear_state_buffer(d_indexer_scores, state_layout_.indexer_scores_bytes());
        clear_state_buffer(d_indexer_topk_indices, state_layout_.indexer_topk_bytes(), 0xFF);
        local_valid_count_ = 0;
        compressor_partial_count_ = 0;
        compressed_entry_count_ = 0;
        indexer_candidate_count_ = 0;
    }

    void free() {
        free_dense_weights();
        free_state();
    }

private:
    V4LayerSpec model_spec_{};
    V4LayerStateLayout state_layout_{};

    template<typename T>
    static void allocate_state_buffer(T*& pointer, size_t bytes) {
        if (bytes == 0) return;
        CHECK_HIP(hipMalloc(&pointer, bytes));
        CHECK_HIP(hipMemset(pointer, 0, bytes));
    }

    template<typename T>
    static void clear_state_buffer(T* pointer, size_t bytes, int value = 0) {
        if (pointer != nullptr && bytes != 0) CHECK_HIP(hipMemset(pointer, value, bytes));
    }

    void allocate_state() {
        allocate_state_buffer(d_local_key_cache, state_layout_.local_vector_bytes());
        allocate_state_buffer(d_local_value_cache, state_layout_.local_vector_bytes());
        allocate_state_buffer(d_local_positions, state_layout_.local_metadata_bytes());
        d_kv_cache = d_local_key_cache;

        if (!state_layout_.is_compressed()) {
            reset_generation_state();
            return;
        }

        allocate_state_buffer(d_compressed_key_cache, state_layout_.compressed_vector_bytes());
        allocate_state_buffer(d_compressed_value_cache, state_layout_.compressed_vector_bytes());
        allocate_state_buffer(d_compressed_positions, state_layout_.compressed_metadata_bytes());
        allocate_state_buffer(d_compressor_partial_kv, state_layout_.compressor_partial_vector_bytes());
        allocate_state_buffer(d_compressor_partial_score, state_layout_.compressor_partial_vector_bytes());
        allocate_state_buffer(d_compressor_partial_positions, state_layout_.compressor_partial_metadata_bytes());

        if (!state_layout_.uses_indexer()) {
            reset_generation_state();
            return;
        }

        allocate_state_buffer(d_indexer_key_cache, state_layout_.indexer_key_bytes());
        allocate_state_buffer(d_indexer_positions, state_layout_.indexer_metadata_bytes());
        allocate_state_buffer(d_indexer_partial_kv, state_layout_.indexer_partial_vector_bytes());
        allocate_state_buffer(d_indexer_partial_score, state_layout_.indexer_partial_vector_bytes());
        allocate_state_buffer(d_indexer_partial_positions, state_layout_.indexer_partial_metadata_bytes());
        allocate_state_buffer(d_indexer_query, state_layout_.indexer_query_bytes());
        allocate_state_buffer(d_indexer_weights, state_layout_.indexer_weights_bytes());
        allocate_state_buffer(d_indexer_scores, state_layout_.indexer_scores_bytes());
        allocate_state_buffer(d_indexer_topk_indices, state_layout_.indexer_topk_bytes());
        reset_generation_state();
    }

    void free_state() noexcept {
        if (d_local_key_cache) { (void)hipFree(d_local_key_cache); d_local_key_cache = nullptr; }
        if (d_local_value_cache) { (void)hipFree(d_local_value_cache); d_local_value_cache = nullptr; }
        if (d_local_positions) { (void)hipFree(d_local_positions); d_local_positions = nullptr; }
        d_kv_cache = nullptr;

        if (d_compressed_key_cache) { (void)hipFree(d_compressed_key_cache); d_compressed_key_cache = nullptr; }
        if (d_compressed_value_cache) { (void)hipFree(d_compressed_value_cache); d_compressed_value_cache = nullptr; }
        if (d_compressed_positions) { (void)hipFree(d_compressed_positions); d_compressed_positions = nullptr; }
        if (d_compressor_partial_kv) { (void)hipFree(d_compressor_partial_kv); d_compressor_partial_kv = nullptr; }
        if (d_compressor_partial_score) { (void)hipFree(d_compressor_partial_score); d_compressor_partial_score = nullptr; }
        if (d_compressor_partial_positions) { (void)hipFree(d_compressor_partial_positions); d_compressor_partial_positions = nullptr; }

        if (d_indexer_key_cache) { (void)hipFree(d_indexer_key_cache); d_indexer_key_cache = nullptr; }
        if (d_indexer_positions) { (void)hipFree(d_indexer_positions); d_indexer_positions = nullptr; }
        if (d_indexer_partial_kv) { (void)hipFree(d_indexer_partial_kv); d_indexer_partial_kv = nullptr; }
        if (d_indexer_partial_score) { (void)hipFree(d_indexer_partial_score); d_indexer_partial_score = nullptr; }
        if (d_indexer_partial_positions) { (void)hipFree(d_indexer_partial_positions); d_indexer_partial_positions = nullptr; }
        if (d_indexer_query) { (void)hipFree(d_indexer_query); d_indexer_query = nullptr; }
        if (d_indexer_weights) { (void)hipFree(d_indexer_weights); d_indexer_weights = nullptr; }
        if (d_indexer_scores) { (void)hipFree(d_indexer_scores); d_indexer_scores = nullptr; }
        if (d_indexer_topk_indices) { (void)hipFree(d_indexer_topk_indices); d_indexer_topk_indices = nullptr; }

        max_seq_len_ = 4096;
        local_valid_count_ = 0;
        compressor_partial_count_ = 0;
        compressed_entry_count_ = 0;
        indexer_candidate_count_ = 0;
        state_layout_ = {};
    }

    void move_from(V4Layer&& o) noexcept {
        layer_id = o.layer_id;
        is_hash_layer = o.is_hash_layer;
        model_spec_ = o.model_spec_;
        state_layout_ = o.state_layout_;
        d_local_key_cache = o.d_local_key_cache; o.d_local_key_cache = nullptr;
        d_local_value_cache = o.d_local_value_cache; o.d_local_value_cache = nullptr;
        d_local_positions = o.d_local_positions; o.d_local_positions = nullptr;
        d_kv_cache = d_local_key_cache;
        d_compressed_key_cache = o.d_compressed_key_cache; o.d_compressed_key_cache = nullptr;
        d_compressed_value_cache = o.d_compressed_value_cache; o.d_compressed_value_cache = nullptr;
        d_compressed_positions = o.d_compressed_positions; o.d_compressed_positions = nullptr;
        d_compressor_partial_kv = o.d_compressor_partial_kv; o.d_compressor_partial_kv = nullptr;
        d_compressor_partial_score = o.d_compressor_partial_score; o.d_compressor_partial_score = nullptr;
        d_compressor_partial_positions = o.d_compressor_partial_positions; o.d_compressor_partial_positions = nullptr;
        d_indexer_key_cache = o.d_indexer_key_cache; o.d_indexer_key_cache = nullptr;
        d_indexer_positions = o.d_indexer_positions; o.d_indexer_positions = nullptr;
        d_indexer_partial_kv = o.d_indexer_partial_kv; o.d_indexer_partial_kv = nullptr;
        d_indexer_partial_score = o.d_indexer_partial_score; o.d_indexer_partial_score = nullptr;
        d_indexer_partial_positions = o.d_indexer_partial_positions; o.d_indexer_partial_positions = nullptr;
        d_indexer_query = o.d_indexer_query; o.d_indexer_query = nullptr;
        d_indexer_weights = o.d_indexer_weights; o.d_indexer_weights = nullptr;
        d_indexer_scores = o.d_indexer_scores; o.d_indexer_scores = nullptr;
        d_indexer_topk_indices = o.d_indexer_topk_indices; o.d_indexer_topk_indices = nullptr;
        max_seq_len_ = o.max_seq_len_;
        local_valid_count_ = o.local_valid_count_;
        compressor_partial_count_ = o.compressor_partial_count_;
        compressed_entry_count_ = o.compressed_entry_count_;
        indexer_candidate_count_ = o.indexer_candidate_count_;

        cache_hits = o.cache_hits;
        cache_misses = o.cache_misses;
        o.d_kv_cache = nullptr;
    }

};

} // namespace aeon::core
