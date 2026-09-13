#include "architecture/deepseek_v4/reference/v4_attention_oracle.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using aeon::core::V4AttentionKind;
using aeon::reference::V4AttentionOracle;
using aeon::reference::V4OracleConfig;
using aeon::reference::V4OracleStateSnapshot;
using aeon::reference::V4OracleTokenInput;

V4OracleConfig make_config(V4AttentionKind kind, int32_t ratio, int32_t topk = 3) {
    V4OracleConfig config;
    config.layer_spec.layer_id = kind == V4AttentionKind::Sliding ? 0u :
        (kind == V4AttentionKind::CSA ? 2u : 3u);
    config.layer_spec.attention_kind = kind;
    config.layer_spec.compression_ratio = ratio;
    config.num_heads = 2;
    config.head_dim = 8;
    config.index_n_heads = 3;
    config.index_head_dim = 4;
    config.index_topk = topk;
    config.sliding_window = 128;
    config.rope_dim = 4;
    config.main_rope.theta = 10000.0f;
    config.compressed_rope.theta = 160000.0f;
    config.compressed_rope.factor = 16.0f;
    config.indexer_rope.theta = 160000.0f;
    config.indexer_rope.factor = 16.0f;
    config.attention_sink = {0.1f, -0.2f};
    if (ratio != 0) {
        const int32_t coefficient = ratio == 4 ? 2 : 1;
        config.compressor_ape.assign(static_cast<size_t>(ratio * coefficient * config.head_dim), 0.0f);
        config.compressor_ape[0] = 0.25f;
        config.compressor_norm.assign(static_cast<size_t>(config.head_dim), 1.0f);
        if (kind == V4AttentionKind::CSA) {
            config.indexer_compressor_ape.assign(static_cast<size_t>(ratio * coefficient * config.index_head_dim), 0.0f);
            config.indexer_compressor_ape[0] = -0.125f;
            config.indexer_compressor_norm.assign(static_cast<size_t>(config.index_head_dim), 1.0f);
        }
    }
    return config;
}

V4OracleTokenInput make_input(const V4OracleConfig& config, int64_t position) {
    V4OracleTokenInput input;
    input.position = position;
    input.query.resize(static_cast<size_t>(config.num_heads * config.head_dim));
    input.local_key.resize(static_cast<size_t>(config.head_dim));
    input.local_value.resize(static_cast<size_t>(config.head_dim));
    for (size_t index = 0; index < input.query.size(); ++index) {
        input.query[index] = 0.01f * static_cast<float>((index % 7) + 1) + 0.0001f * static_cast<float>(position);
    }
    for (size_t index = 0; index < input.local_key.size(); ++index) {
        input.local_key[index] = 0.02f * static_cast<float>((index % 5) + 1) + 0.0002f * static_cast<float>(position);
        input.local_value[index] = 0.03f * static_cast<float>((index % 3) + 1) - 0.0003f * static_cast<float>(position);
    }
    if (config.layer_spec.compression_ratio == 0) return input;

    const int32_t coefficient = config.layer_spec.compression_ratio == 4 ? 2 : 1;
    input.compressor_kv.resize(static_cast<size_t>(coefficient * config.head_dim));
    input.compressor_score.resize(input.compressor_kv.size(), 0.0f);
    for (size_t index = 0; index < input.compressor_kv.size(); ++index) {
        input.compressor_kv[index] = 0.04f * static_cast<float>((index % 4) + 1) + 0.0005f * static_cast<float>(position);
        input.compressor_score[index] = 0.001f * static_cast<float>(static_cast<int32_t>(index % 3) - 1);
    }
    if (config.layer_spec.attention_kind != V4AttentionKind::CSA) return input;

    input.indexer_query.assign(static_cast<size_t>(config.index_n_heads * config.index_head_dim), 0.0f);
    input.indexer_weights.assign(static_cast<size_t>(config.index_n_heads), 1.0f);
    input.indexer_compressor_kv.resize(static_cast<size_t>(coefficient * config.index_head_dim));
    input.indexer_compressor_score.assign(input.indexer_compressor_kv.size(), 0.0f);
    for (size_t index = 0; index < input.indexer_compressor_kv.size(); ++index) {
        input.indexer_compressor_kv[index] = 0.05f * static_cast<float>((index % 3) + 1) + 0.0007f * static_cast<float>(position);
    }
    return input;
}

void assert_snapshot_equal(const V4OracleStateSnapshot& left, const V4OracleStateSnapshot& right) {
    assert(left.next_position == right.next_position);
    assert(left.local_cache.size() == right.local_cache.size());
    for (size_t index = 0; index < left.local_cache.size(); ++index) {
        assert(left.local_cache[index].position == right.local_cache[index].position);
        assert(left.local_cache[index].key == right.local_cache[index].key);
        assert(left.local_cache[index].value == right.local_cache[index].value);
    }
    assert(left.compressor_partial.size() == right.compressor_partial.size());
    for (size_t index = 0; index < left.compressor_partial.size(); ++index) {
        assert(left.compressor_partial[index].position == right.compressor_partial[index].position);
        assert(left.compressor_partial[index].kv == right.compressor_partial[index].kv);
        assert(left.compressor_partial[index].score == right.compressor_partial[index].score);
    }
    assert(left.indexer_partial.size() == right.indexer_partial.size());
    for (size_t index = 0; index < left.indexer_partial.size(); ++index) {
        assert(left.indexer_partial[index].position == right.indexer_partial[index].position);
        assert(left.indexer_partial[index].kv == right.indexer_partial[index].kv);
        assert(left.indexer_partial[index].score == right.indexer_partial[index].score);
    }
    assert(left.compressed_entries.size() == right.compressed_entries.size());
    for (size_t index = 0; index < left.compressed_entries.size(); ++index) {
        const auto& lhs = left.compressed_entries[index];
        const auto& rhs = right.compressed_entries[index];
        assert(lhs.boundary_position == rhs.boundary_position);
        assert(lhs.rope_position == rhs.rope_position);
        assert(lhs.rope_kind == rhs.rope_kind);
        assert(lhs.key == rhs.key);
        assert(lhs.value == rhs.value);
    }
    assert(left.indexer_entries.size() == right.indexer_entries.size());
    for (size_t index = 0; index < left.indexer_entries.size(); ++index) {
        const auto& lhs = left.indexer_entries[index];
        const auto& rhs = right.indexer_entries[index];
        assert(lhs.boundary_position == rhs.boundary_position);
        assert(lhs.rope_position == rhs.rope_position);
        assert(lhs.rope_kind == rhs.rope_kind);
        assert(lhs.key == rhs.key);
    }
    assert(left.last_indexer_query == right.last_indexer_query);
    assert(left.last_indexer_weights == right.last_indexer_weights);
    assert(left.last_step.position == right.last_step.position);
    assert(left.last_step.compressed_entry_created == right.last_step.compressed_entry_created);
    assert(left.last_step.compressed_entry_index == right.last_step.compressed_entry_index);
    assert(left.last_step.compressed_rope_position == right.last_step.compressed_rope_position);
    assert(left.last_step.local_valid_count == right.last_step.local_valid_count);
    assert(left.last_step.compressed_entry_count == right.last_step.compressed_entry_count);
    assert(left.last_step.selected_compressed_count == right.last_step.selected_compressed_count);
    assert(left.last_step.indexer_candidate_count == right.last_step.indexer_candidate_count);
    assert(left.last_step.local_positions == right.last_step.local_positions);
    assert(left.last_step.selected_compressed_positions == right.last_step.selected_compressed_positions);
    assert(left.last_step.rotated_query == right.last_step.rotated_query);
    assert(left.last_step.attention_output == right.last_step.attention_output);
    assert(left.last_step.inverse_rope_output == right.last_step.inverse_rope_output);
    assert(left.last_step.indexer_scores == right.last_step.indexer_scores);
    assert(left.last_step.topk_indices == right.last_step.topk_indices);
}

void test_sliding_boundaries() {
    const auto config = make_config(V4AttentionKind::Sliding, 0);
    V4AttentionOracle oracle(config);
    for (int64_t position = 0; position <= 131; ++position) {
        const auto result = oracle.append(make_input(config, position));
        assert(!result.compressed_entry_created);
        assert(result.compressed_entry_count == 0);
        assert(result.local_valid_count == static_cast<size_t>(std::min<int64_t>(position + 1, 128)));
        if (position == 131) {
            assert(result.local_positions.front() == 4);
            assert(result.local_positions.back() == 131);
        }
    }
    assert(oracle.compact_trace().find("layer=Sliding") != std::string::npos);
}

void test_csa_boundaries_and_topk() {
    const auto config = make_config(V4AttentionKind::CSA, 4);
    V4AttentionOracle oracle(config);
    for (int64_t position = 0; position <= 131; ++position) {
        const auto result = oracle.append(make_input(config, position));
        const bool boundary = ((position + 1) % 4) == 0;
        assert(result.compressed_entry_created == boundary);
        assert(result.compressed_entry_count == static_cast<size_t>((position + 1) / 4));
        assert(result.indexer_candidate_count == result.compressed_entry_count);
        assert(result.topk_indices.size() == 3);
        if (position == 7) {
            assert(result.compressed_rope_position == 4);
            assert(result.topk_indices == std::vector<int32_t>({0, 1, -1}));
        }
        if (position == 19) {
            assert(result.indexer_candidate_count == 5);
            assert(result.topk_indices == std::vector<int32_t>({0, 1, 2}));
        }
        if (position == 127) {
            assert(result.compressed_entry_count == 32);
            assert(result.compressed_rope_position == 124);
        }
        if (position == 128) assert(!result.compressed_entry_created);
        if (position == 131) {
            assert(result.compressed_entry_count == 33);
            assert(result.compressed_rope_position == 128);
        }
    }
    assert(oracle.last_step().inverse_rope_output.size() == 16);
    assert(oracle.compact_trace().find("layer=CSA") != std::string::npos);
    assert(oracle.compact_trace().find("ropes=main@compressed") != std::string::npos);
}

void test_hca_boundaries() {
    const auto config = make_config(V4AttentionKind::HCA, 128);
    V4AttentionOracle oracle(config);
    for (int64_t position = 0; position <= 131; ++position) {
        const auto result = oracle.append(make_input(config, position));
        assert(result.compressed_entry_created == (position == 127));
        assert(result.compressed_entry_count == (position >= 127 ? 1u : 0u));
        if (position == 127) assert(result.compressed_rope_position == 0);
        if (position == 131) assert(result.selected_compressed_count == 1);
    }
}

void test_csa_real_topk_width() {
    const auto config = make_config(V4AttentionKind::CSA, 4, 512);
    V4AttentionOracle oracle(config);
    V4OracleTokenInput input = make_input(config, 0);
    std::fill(input.query.begin(), input.query.end(), 0.0f);
    std::fill(input.indexer_query.begin(), input.indexer_query.end(), 0.0f);
    for (int64_t position = 0; position <= 2051; ++position) {
        input.position = position;
        const auto result = oracle.append(input);
        if (position == 2047) {
            assert(result.indexer_candidate_count == 512);
            assert(result.topk_indices.front() == 0);
            assert(result.topk_indices.back() == 511);
        }
        if (position == 2051) {
            assert(result.indexer_candidate_count == 513);
            assert(result.topk_indices.front() == 0);
            assert(result.topk_indices.back() == 511);
            assert(std::find(result.topk_indices.begin(), result.topk_indices.end(), 512) == result.topk_indices.end());
        }
    }
}

void test_chunk_and_serialized_equivalence() {
    const auto config = make_config(V4AttentionKind::CSA, 4);
    std::vector<V4OracleTokenInput> inputs;
    inputs.reserve(132);
    for (int64_t position = 0; position <= 131; ++position) inputs.push_back(make_input(config, position));

    V4AttentionOracle one_shot(config);
    one_shot.append_batch(inputs);

    V4AttentionOracle chunked(config);
    size_t start = 0;
    for (const size_t boundary : {size_t{1}, size_t{3}, size_t{4}, size_t{7}, size_t{16}, size_t{127}, size_t{128}, size_t{132}}) {
        const size_t end = std::min(boundary, inputs.size());
        if (end > start) chunked.append_batch(std::span<const V4OracleTokenInput>(inputs.data() + start, end - start));
        start = end;
    }
    assert_snapshot_equal(one_shot.snapshot(), chunked.snapshot());

    V4AttentionOracle serialized_prefix(config);
    serialized_prefix.append_batch(std::span<const V4OracleTokenInput>(inputs.data(), 73));
    const auto bytes = serialized_prefix.serialize_state();
    V4AttentionOracle serialized_suffix(config);
    serialized_suffix.restore_serialized(bytes);
    serialized_suffix.append_batch(std::span<const V4OracleTokenInput>(inputs.data() + 73, inputs.size() - 73));
    assert_snapshot_equal(one_shot.snapshot(), serialized_suffix.snapshot());

    V4AttentionOracle reset_oracle(config);
    reset_oracle.append_batch(std::span<const V4OracleTokenInput>(inputs.data(), 9));
    reset_oracle.reset();
    reset_oracle.append_batch(std::span<const V4OracleTokenInput>(inputs.data(), 9));
    V4AttentionOracle fresh_prefix(config);
    fresh_prefix.append_batch(std::span<const V4OracleTokenInput>(inputs.data(), 9));
    assert_snapshot_equal(reset_oracle.snapshot(), fresh_prefix.snapshot());
}

} // namespace

int main() {
    test_sliding_boundaries();
    test_csa_boundaries_and_topk();
    test_hca_boundaries();
    test_csa_real_topk_width();
    test_chunk_and_serialized_equivalence();
    std::cout << "V4 attention oracle passed: Sliding/C4A/C128A boundaries, local ring positions, deterministic top-k, and serialized state equivalence" << std::endl;
    return 0;
}