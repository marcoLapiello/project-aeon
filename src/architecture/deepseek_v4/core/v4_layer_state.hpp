#pragma once

#include "architecture/deepseek_v4/core/v4_model_spec.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace aeon::core {

struct V4LayerStateLayout {
    uint32_t max_seq_len{0};
    uint32_t local_capacity{0};
    uint32_t head_dim{0};
    uint32_t index_n_heads{0};
    uint32_t index_head_dim{0};
    uint32_t index_topk{0};
    int32_t compression_ratio{0};
    uint32_t compressed_capacity{0};
    uint32_t compressor_partial_capacity{0};
    uint32_t compressor_width{0};
    uint32_t indexer_partial_capacity{0};
    uint32_t indexer_width{0};

    static V4LayerStateLayout from_spec(const V4LayerSpec& spec, uint32_t requested_seq_len) {
        if (requested_seq_len == 0 || spec.head_dim <= 0 || spec.sliding_window <= 0) {
            throw std::invalid_argument("V4LayerStateLayout: invalid layer dimensions");
        }
        if (spec.attention_kind == V4AttentionKind::Sliding && spec.compression_ratio != 0) {
            throw std::invalid_argument("V4LayerStateLayout: Sliding layer must have ratio 0");
        }
        if (spec.attention_kind == V4AttentionKind::CSA && spec.compression_ratio != 4) {
            throw std::invalid_argument("V4LayerStateLayout: CSA layer must have ratio 4");
        }
        if (spec.attention_kind == V4AttentionKind::HCA && spec.compression_ratio != 128) {
            throw std::invalid_argument("V4LayerStateLayout: HCA layer must have ratio 128");
        }

        V4LayerStateLayout layout;
        layout.max_seq_len = requested_seq_len;
        layout.local_capacity = std::min<uint32_t>(requested_seq_len, static_cast<uint32_t>(spec.sliding_window));
        layout.head_dim = static_cast<uint32_t>(spec.head_dim);
        layout.index_n_heads = static_cast<uint32_t>(spec.index_n_heads);
        layout.index_head_dim = static_cast<uint32_t>(spec.index_head_dim);
        layout.index_topk = static_cast<uint32_t>(spec.index_topk);
        layout.compression_ratio = spec.compression_ratio;
        if (spec.compression_ratio == 0) return layout;

        const uint32_t coefficient = spec.compression_ratio == 4 ? 2u : 1u;
        layout.compressed_capacity = (requested_seq_len + static_cast<uint32_t>(spec.compression_ratio) - 1u) /
                                     static_cast<uint32_t>(spec.compression_ratio);
        layout.compressor_partial_capacity = coefficient * static_cast<uint32_t>(spec.compression_ratio);
        layout.compressor_width = coefficient * static_cast<uint32_t>(spec.head_dim);
        if (spec.attention_kind == V4AttentionKind::CSA) {
            if (spec.index_n_heads <= 0 || spec.index_head_dim <= 0 || spec.index_topk <= 0) {
                throw std::invalid_argument("V4LayerStateLayout: invalid CSA indexer dimensions");
            }
            layout.indexer_partial_capacity = layout.compressor_partial_capacity;
            layout.indexer_width = coefficient * static_cast<uint32_t>(spec.index_head_dim);
        }
        return layout;
    }

    bool is_compressed() const noexcept {
        return compression_ratio != 0;
    }

    bool uses_indexer() const noexcept {
        return compression_ratio == 4;
    }

    size_t local_vector_bytes() const {
        return checked_product(local_capacity, head_dim, sizeof(uint16_t));
    }

    size_t local_cache_bytes() const {
        return local_vector_bytes() * 2;
    }

    size_t local_metadata_bytes() const {
        return checked_product(local_capacity, sizeof(int64_t));
    }

    size_t compressed_cache_bytes() const {
        return compressed_vector_bytes() * 2;
    }

    size_t compressed_vector_bytes() const {
        return checked_product(compressed_capacity, head_dim, sizeof(uint16_t));
    }

    size_t compressed_metadata_bytes() const {
        return checked_product(compressed_capacity, sizeof(int64_t));
    }

    size_t compressor_state_bytes() const {
        return compressor_partial_vector_bytes() * 2 + compressor_partial_metadata_bytes();
    }

    size_t compressor_partial_vector_bytes() const {
        return checked_product(compressor_partial_capacity, compressor_width, sizeof(float));
    }

    size_t compressor_partial_metadata_bytes() const {
        return checked_product(compressor_partial_capacity, sizeof(int64_t));
    }

    size_t indexer_cache_bytes() const {
        if (!uses_indexer()) return 0;
        return indexer_key_bytes() + indexer_metadata_bytes();
    }

    size_t indexer_key_bytes() const {
        return checked_product(compressed_capacity, index_head_dim, sizeof(uint16_t));
    }

    size_t indexer_metadata_bytes() const {
        return checked_product(compressed_capacity, sizeof(int64_t));
    }

    size_t indexer_workspace_bytes() const {
        if (!uses_indexer()) return 0;
        return indexer_partial_vector_bytes() * 2 + indexer_partial_metadata_bytes() +
               indexer_query_bytes() + indexer_weights_bytes() +
               indexer_scores_bytes() + indexer_topk_bytes();
    }

    size_t indexer_partial_vector_bytes() const {
        return checked_product(indexer_partial_capacity, indexer_width, sizeof(float));
    }

    size_t indexer_partial_metadata_bytes() const {
        return checked_product(indexer_partial_capacity, sizeof(int64_t));
    }

    size_t indexer_query_bytes() const {
        return checked_product(index_n_heads, index_head_dim, sizeof(uint16_t));
    }

    size_t indexer_weights_bytes() const {
        return checked_product(index_n_heads, sizeof(float));
    }

    size_t indexer_scores_bytes() const {
        return checked_product(compressed_capacity, sizeof(float));
    }

    size_t indexer_topk_bytes() const {
        return checked_product(index_topk, sizeof(int32_t));
    }

    size_t local_state_bytes() const {
        return local_cache_bytes() + local_metadata_bytes();
    }

    size_t compressed_state_bytes() const {
        return compressed_cache_bytes() + compressed_metadata_bytes();
    }

    size_t total_device_bytes() const {
        return local_state_bytes() + compressed_state_bytes() +
               compressor_state_bytes() + indexer_cache_bytes() + indexer_workspace_bytes();
    }

private:
    static size_t checked_product(size_t first) {
        return first;
    }

    template<typename... Values>
    static size_t checked_product(size_t first, Values... values) {
        if (first != 0 && first > std::numeric_limits<size_t>::max() / checked_product(values...)) {
            throw std::overflow_error("V4LayerStateLayout: state byte count overflow");
        }
        return first * checked_product(values...);
    }
};

} // namespace aeon::core