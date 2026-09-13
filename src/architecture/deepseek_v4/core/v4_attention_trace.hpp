#pragma once

#include "architecture/deepseek_v4/core/v4_model_spec.hpp"

#include <hip/hip_fp16.h>

#include <cstdint>
#include <vector>

namespace aeon::core {

struct V4AttentionTraceRecord {
    uint32_t position{0};
    V4AttentionKind attention_kind{V4AttentionKind::Sliding};
    uint32_t local_valid_count{0};
    uint32_t compressor_partial_count{0};
    uint32_t compressed_entry_count{0};
    uint32_t indexer_candidate_count{0};

    // Projection inputs before class-specific RoPE and cache mutation.
    std::vector<half> query;
    std::vector<half> local_key;
    std::vector<half> local_value;
    std::vector<half> compressor_kv;
    std::vector<half> compressor_score;
    std::vector<half> indexer_query;
    std::vector<half> indexer_weights;
    std::vector<half> indexer_compressor_kv;
    std::vector<half> indexer_compressor_score;

    // Attention state and outputs after the corresponding production steps.
    std::vector<half> rotated_query;
    std::vector<half> rotated_local_key;
    std::vector<half> attention_output;
    std::vector<half> inverse_rope_output;
    std::vector<half> grouped_output;

    std::vector<half> local_key_cache;
    std::vector<half> local_value_cache;
    std::vector<int64_t> local_positions;
    std::vector<float> compressor_partial_kv;
    std::vector<float> compressor_partial_score;
    std::vector<int64_t> compressor_partial_positions;
    std::vector<float> indexer_partial_kv;
    std::vector<float> indexer_partial_score;
    std::vector<int64_t> indexer_partial_positions;
    std::vector<half> compressed_key_cache;
    std::vector<half> compressed_value_cache;
    std::vector<int64_t> compressed_positions;
    std::vector<half> indexer_key_cache;
    std::vector<int64_t> indexer_positions;
    std::vector<float> indexer_scores;
    std::vector<int32_t> indexer_topk_indices;
};

} // namespace aeon::core