#pragma once

#include "architecture/deepseek_v4/core/v4_model_spec.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aeon::reference {

enum class V4OracleRopeKind : uint8_t {
    Main,
    Compressed,
    Indexer
};

struct V4OracleRopeTable {
    std::string identity{"main"};
    int32_t head_dim{0};
    int32_t rope_dim{0};
    float theta{10000.0f};
    float factor{1.0f};
    float beta_fast{32.0f};
    float beta_slow{1.0f};
    int32_t original_max_position{65536};

    void validate() const {
        if (head_dim <= 0 || rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) {
            throw std::invalid_argument("V4AttentionOracle: invalid RoPE dimensions for " + identity);
        }
        if (!(theta > 0.0f) || !(factor > 0.0f) || beta_fast < beta_slow || original_max_position <= 0) {
            throw std::invalid_argument("V4AttentionOracle: invalid RoPE parameters for " + identity);
        }
    }

    float inverse_frequency(int32_t pair) const {
        const float exponent = (2.0f * static_cast<float>(pair)) / static_cast<float>(rope_dim);
        float frequency = 1.0f / std::pow(theta, exponent);
        if (factor <= 1.0f) return frequency;

        constexpr float pi = 3.14159265358979323846f;
        const auto correction_dim = [this](float rotations) {
            return static_cast<float>(rope_dim) *
                std::log(static_cast<float>(original_max_position) / (rotations * 2.0f * pi)) /
                (2.0f * std::log(theta));
        };
        const float low = std::max(0.0f, std::floor(correction_dim(beta_fast)));
        const float high = std::min(
            static_cast<float>(rope_dim / 2 - 1),
            std::ceil(correction_dim(beta_slow)));
        if (low >= high) {
            return static_cast<float>(pair) < low ? frequency : frequency / factor;
        }
        const float ramp = std::clamp(
            (static_cast<float>(pair) - low) / (high - low), 0.0f, 1.0f);
        return ramp * (frequency / factor) + (1.0f - ramp) * frequency;
    }

    void apply(std::vector<float>& values, int32_t heads, int64_t position) const {
        apply_impl(values, heads, position, false);
    }

    void apply_inverse(std::vector<float>& values, int32_t heads, int64_t position) const {
        apply_impl(values, heads, position, true);
    }

private:
    void apply_impl(std::vector<float>& values, int32_t heads, int64_t position, bool inverse) const {
        validate();
        if (heads <= 0 || position < 0 || values.size() != static_cast<size_t>(heads * head_dim)) {
            throw std::invalid_argument("V4AttentionOracle: invalid vector for RoPE " + identity);
        }
        const int32_t half_rope = rope_dim / 2;
        const int32_t nope_dim = head_dim - rope_dim;
        for (int32_t head = 0; head < heads; ++head) {
            const size_t head_offset = static_cast<size_t>(head * head_dim + nope_dim);
            for (int32_t pair = 0; pair < half_rope; ++pair) {
                const float angle = static_cast<float>(position) * inverse_frequency(pair);
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const size_t offset = head_offset + static_cast<size_t>(2 * pair);
                const float even = values[offset];
                const float odd = values[offset + 1];
                if (!inverse) {
                    values[offset] = even * cosine - odd * sine;
                    values[offset + 1] = even * sine + odd * cosine;
                } else {
                    values[offset] = even * cosine + odd * sine;
                    values[offset + 1] = odd * cosine - even * sine;
                }
            }
        }
    }
};

struct V4OracleConfig {
    core::V4LayerSpec layer_spec{};
    int32_t num_heads{64};
    int32_t head_dim{512};
    int32_t index_n_heads{64};
    int32_t index_head_dim{128};
    int32_t index_topk{512};
    int32_t sliding_window{128};
    int32_t rope_dim{64};
    float attention_scale{0.0f};
    float indexer_softmax_scale{0.0f};
    float indexer_head_scale{0.0f};
    float rms_norm_eps{1e-6f};
    std::vector<float> attention_sink;
    std::vector<float> compressor_ape;
    std::vector<float> compressor_norm;
    std::vector<float> indexer_compressor_ape;
    std::vector<float> indexer_compressor_norm;
    V4OracleRopeTable main_rope{"main"};
    V4OracleRopeTable compressed_rope{"compressed"};
    V4OracleRopeTable indexer_rope{"indexer"};
};

struct V4OracleTokenInput {
    int64_t position{0};
    std::vector<float> query;
    std::vector<float> local_key;
    std::vector<float> local_value;
    std::vector<float> compressor_kv;
    std::vector<float> compressor_score;
    std::vector<float> indexer_query;
    std::vector<float> indexer_weights;
    std::vector<float> indexer_compressor_kv;
    std::vector<float> indexer_compressor_score;
};

struct V4OracleLocalEntry {
    int64_t position{-1};
    std::vector<float> key;
    std::vector<float> value;
};

struct V4OraclePartialRow {
    int64_t position{-1};
    std::vector<float> kv;
    std::vector<float> score;
};

struct V4OracleCompressedEntry {
    int64_t boundary_position{-1};
    int64_t rope_position{-1};
    V4OracleRopeKind rope_kind{V4OracleRopeKind::Compressed};
    std::vector<float> key;
    std::vector<float> value;
};

struct V4OracleIndexerEntry {
    int64_t boundary_position{-1};
    int64_t rope_position{-1};
    V4OracleRopeKind rope_kind{V4OracleRopeKind::Indexer};
    std::vector<float> key;
};

struct V4OracleStepResult {
    int64_t position{-1};
    bool compressed_entry_created{false};
    size_t compressed_entry_index{0};
    int64_t compressed_rope_position{-1};
    size_t local_valid_count{0};
    size_t compressed_entry_count{0};
    size_t selected_compressed_count{0};
    size_t indexer_candidate_count{0};
    std::vector<int64_t> local_positions;
    std::vector<int64_t> selected_compressed_positions;
    std::vector<float> rotated_query;
    std::vector<float> attention_output;
    std::vector<float> inverse_rope_output;
    std::vector<float> indexer_scores;
    std::vector<int32_t> topk_indices;
};

struct V4OracleStateSnapshot {
    int64_t next_position{0};
    std::vector<V4OracleLocalEntry> local_cache;
    std::vector<V4OraclePartialRow> compressor_partial;
    std::vector<V4OraclePartialRow> indexer_partial;
    std::vector<V4OracleCompressedEntry> compressed_entries;
    std::vector<V4OracleIndexerEntry> indexer_entries;
    std::vector<float> last_indexer_query;
    std::vector<float> last_indexer_weights;
    V4OracleStepResult last_step;
};

class V4AttentionOracle {
public:
    explicit V4AttentionOracle(V4OracleConfig config)
        : config_(prepare_config(std::move(config))) {
        reset();
    }

    const V4OracleConfig& config() const noexcept {
        return config_;
    }

    void reset(int64_t start_position = 0) {
        if (start_position < 0) {
            throw std::invalid_argument("V4AttentionOracle: reset position must be non-negative");
        }
        next_position_ = start_position;
        local_cache_.assign(static_cast<size_t>(config_.sliding_window), {});
        for (auto& entry : local_cache_) {
            entry.key.assign(static_cast<size_t>(config_.head_dim), 0.0f);
            entry.value.assign(static_cast<size_t>(config_.head_dim), 0.0f);
        }
        compressor_partial_.clear();
        indexer_partial_.clear();
        compressed_entries_.clear();
        indexer_entries_.clear();
        last_indexer_query_.clear();
        last_indexer_weights_.clear();
        last_step_ = {};
    }

    V4OracleStepResult append(const V4OracleTokenInput& input) {
        validate_input(input);
        if (input.position != next_position_) {
            throw std::invalid_argument(
                "V4AttentionOracle: expected absolute position " + std::to_string(next_position_) +
                ", received " + std::to_string(input.position));
        }

        const auto& attention_rope = is_compressed() ? config_.compressed_rope : config_.main_rope;
        std::vector<float> rotated_query = input.query;
        attention_rope.apply(rotated_query, config_.num_heads, input.position);

        std::vector<float> rotated_local_key = input.local_key;
        attention_rope.apply(rotated_local_key, 1, input.position);
        const size_t local_slot = static_cast<size_t>(input.position % config_.sliding_window);
        local_cache_[local_slot].position = input.position;
        local_cache_[local_slot].key = std::move(rotated_local_key);
        local_cache_[local_slot].value = input.local_value;

        bool compressed_entry_created = false;
        size_t compressed_entry_index = 0;
        int64_t compressed_rope_position = -1;
        if (is_compressed()) {
            append_partial(
                compressor_partial_,
                input.position,
                input.compressor_kv,
                input.compressor_score,
                config_.compressor_ape,
                config_.layer_spec.compression_ratio,
                config_.head_dim);

            if (is_csa()) {
                append_partial(
                    indexer_partial_,
                    input.position,
                    input.indexer_compressor_kv,
                    input.indexer_compressor_score,
                    config_.indexer_compressor_ape,
                    config_.layer_spec.compression_ratio,
                    config_.index_head_dim);
            }

            if ((input.position + 1) % config_.layer_spec.compression_ratio == 0) {
                const V4OracleCompressedEntry entry = create_compressed_entry(
                    compressor_partial_,
                    input.position,
                    config_.layer_spec.compression_ratio,
                    config_.head_dim,
                    config_.compressor_norm,
                    config_.compressed_rope,
                    V4OracleRopeKind::Compressed,
                    config_.rms_norm_eps);
                compressed_entries_.push_back(entry);
                compressed_entry_created = true;
                compressed_entry_index = compressed_entries_.size() - 1;
                compressed_rope_position = entry.rope_position;

                if (is_csa()) {
                    const V4OracleCompressedEntry indexer_entry = create_compressed_entry(
                        indexer_partial_,
                        input.position,
                        config_.layer_spec.compression_ratio,
                        config_.index_head_dim,
                        config_.indexer_compressor_norm,
                        config_.indexer_rope,
                        V4OracleRopeKind::Indexer,
                        config_.rms_norm_eps);
                    indexer_entries_.push_back({
                        indexer_entry.boundary_position,
                        indexer_entry.rope_position,
                        indexer_entry.rope_kind,
                        indexer_entry.key
                    });
                }
            }
        }

        V4OracleStepResult result;
        result.position = input.position;
        result.compressed_entry_created = compressed_entry_created;
        result.compressed_entry_index = compressed_entry_index;
        result.compressed_rope_position = compressed_rope_position;
        result.local_valid_count = collect_local_positions(input.position, result.local_positions);
        result.compressed_entry_count = compressed_entries_.size();
        result.rotated_query = rotated_query;

        if (is_csa()) {
            std::vector<float> rotated_indexer_query = input.indexer_query;
            config_.indexer_rope.apply(rotated_indexer_query, config_.index_n_heads, input.position);
            last_indexer_query_ = rotated_indexer_query;
            last_indexer_weights_ = input.indexer_weights;
            result.indexer_scores = compute_indexer_scores();
            result.indexer_candidate_count = indexer_entries_.size();
            result.topk_indices = select_topk(result.indexer_scores);
            for (const int32_t index : result.topk_indices) {
                if (index >= 0) {
                    result.selected_compressed_positions.push_back(
                        compressed_entries_.at(static_cast<size_t>(index)).rope_position);
                }
            }
        } else if (config_.layer_spec.attention_kind == core::V4AttentionKind::HCA) {
            for (const auto& entry : compressed_entries_) {
                result.selected_compressed_positions.push_back(entry.rope_position);
            }
        }

        result.selected_compressed_count = result.selected_compressed_positions.size();
        result.attention_output = compute_attention(result, rotated_query);
        result.inverse_rope_output = result.attention_output;
        attention_rope.apply_inverse(result.inverse_rope_output, config_.num_heads, input.position);

        last_step_ = result;
        if (input.position == std::numeric_limits<int64_t>::max()) {
            throw std::overflow_error("V4AttentionOracle: absolute position overflow");
        }
        next_position_ = input.position + 1;
        return result;
    }

    std::vector<V4OracleStepResult> append_batch(std::span<const V4OracleTokenInput> inputs) {
        std::vector<V4OracleStepResult> results;
        results.reserve(inputs.size());
        for (const auto& input : inputs) results.push_back(append(input));
        return results;
    }

    V4OracleStateSnapshot snapshot() const {
        return {
            next_position_,
            local_cache_,
            compressor_partial_,
            indexer_partial_,
            compressed_entries_,
            indexer_entries_,
            last_indexer_query_,
            last_indexer_weights_,
            last_step_
        };
    }

    void restore(const V4OracleStateSnapshot& state) {
        validate_snapshot(state);
        next_position_ = state.next_position;
        local_cache_ = state.local_cache;
        compressor_partial_ = state.compressor_partial;
        indexer_partial_ = state.indexer_partial;
        compressed_entries_ = state.compressed_entries;
        indexer_entries_ = state.indexer_entries;
        last_indexer_query_ = state.last_indexer_query;
        last_indexer_weights_ = state.last_indexer_weights;
        last_step_ = state.last_step;
    }

    std::vector<uint8_t> serialize_state() const {
        std::vector<uint8_t> bytes;
        write_scalar<uint32_t>(bytes, 0x5634414Fu);
        write_scalar<uint32_t>(bytes, 1u);
        write_scalar<uint8_t>(bytes, static_cast<uint8_t>(config_.layer_spec.attention_kind));
        write_scalar<int32_t>(bytes, config_.layer_spec.compression_ratio);
        write_scalar<int32_t>(bytes, config_.num_heads);
        write_scalar<int32_t>(bytes, config_.head_dim);
        write_scalar<int32_t>(bytes, config_.index_n_heads);
        write_scalar<int32_t>(bytes, config_.index_head_dim);
        write_scalar<int32_t>(bytes, config_.index_topk);
        write_scalar<int32_t>(bytes, config_.sliding_window);

        const V4OracleStateSnapshot state = snapshot();
        write_scalar<int64_t>(bytes, state.next_position);
        write_local_entries(bytes, state.local_cache);
        write_partial_rows(bytes, state.compressor_partial);
        write_partial_rows(bytes, state.indexer_partial);
        write_compressed_entries(bytes, state.compressed_entries);
        write_indexer_entries(bytes, state.indexer_entries);
        write_vector(bytes, state.last_indexer_query);
        write_vector(bytes, state.last_indexer_weights);
        write_step(bytes, state.last_step);
        return bytes;
    }

    void restore_serialized(std::span<const uint8_t> bytes) {
        ByteReader reader(bytes);
        if (reader.read<uint32_t>() != 0x5634414Fu || reader.read<uint32_t>() != 1u) {
            throw std::invalid_argument("V4AttentionOracle: unsupported serialized state");
        }
        if (reader.read<uint8_t>() != static_cast<uint8_t>(config_.layer_spec.attention_kind) ||
            reader.read<int32_t>() != config_.layer_spec.compression_ratio ||
            reader.read<int32_t>() != config_.num_heads ||
            reader.read<int32_t>() != config_.head_dim ||
            reader.read<int32_t>() != config_.index_n_heads ||
            reader.read<int32_t>() != config_.index_head_dim ||
            reader.read<int32_t>() != config_.index_topk ||
            reader.read<int32_t>() != config_.sliding_window) {
            throw std::invalid_argument("V4AttentionOracle: serialized state does not match config");
        }

        V4OracleStateSnapshot state;
        state.next_position = reader.read<int64_t>();
        state.local_cache = read_local_entries(reader);
        state.compressor_partial = read_partial_rows(reader);
        state.indexer_partial = read_partial_rows(reader);
        state.compressed_entries = read_compressed_entries(reader);
        state.indexer_entries = read_indexer_entries(reader);
        state.last_indexer_query = reader.read_vector<float>();
        state.last_indexer_weights = reader.read_vector<float>();
        state.last_step = read_step(reader);
        reader.require_end();
        restore(state);
    }

    const V4OracleStepResult& last_step() const noexcept {
        return last_step_;
    }

    std::string compact_trace() const {
        std::ostringstream output;
        output << "layer=" << core::v4_attention_kind_name(config_.layer_spec.attention_kind)
               << " ratio=" << config_.layer_spec.compression_ratio
               << " next_position=" << next_position_
               << " local_valid=" << last_step_.local_valid_count
               << " compressed_entries=" << compressed_entries_.size();
        if (is_csa()) {
            output << " indexer_candidates=" << last_step_.indexer_candidate_count << " topk=[";
            const size_t trace_count = std::min<size_t>(last_step_.topk_indices.size(), 16);
            for (size_t index = 0; index < trace_count; ++index) {
                if (index != 0) output << ',';
                output << last_step_.topk_indices[index];
            }
            if (trace_count < last_step_.topk_indices.size()) output << ",...";
            output << ']';
        }
        output << " ropes=" << config_.main_rope.identity << '@' << config_.compressed_rope.identity
               << " last_position=" << last_step_.position;
        return output.str();
    }

private:
    static V4OracleConfig prepare_config(V4OracleConfig config) {
        if (config.layer_spec.attention_kind == core::V4AttentionKind::Sliding) {
            if (config.layer_spec.compression_ratio != 0) {
                throw std::invalid_argument("V4AttentionOracle: Sliding layer must have ratio 0");
            }
        } else if (config.layer_spec.attention_kind == core::V4AttentionKind::CSA) {
            if (config.layer_spec.compression_ratio != 4) {
                throw std::invalid_argument("V4AttentionOracle: CSA layer must have ratio 4");
            }
        } else if (config.layer_spec.attention_kind == core::V4AttentionKind::HCA) {
            if (config.layer_spec.compression_ratio != 128) {
                throw std::invalid_argument("V4AttentionOracle: HCA layer must have ratio 128");
            }
        } else {
            throw std::invalid_argument("V4AttentionOracle: unsupported layer class");
        }
        if (config.num_heads <= 0 || config.head_dim <= 0 || config.index_n_heads <= 0 ||
            config.index_head_dim <= 0 || config.index_topk <= 0 || config.sliding_window <= 0 ||
            config.rope_dim <= 0 || (config.rope_dim % 2) != 0 || config.rope_dim > config.head_dim) {
            throw std::invalid_argument("V4AttentionOracle: invalid dimensions");
        }
        if (!(config.rms_norm_eps > 0.0f)) {
            throw std::invalid_argument("V4AttentionOracle: rms_norm_eps must be positive");
        }
        config.attention_scale = config.attention_scale > 0.0f
            ? config.attention_scale
            : 1.0f / std::sqrt(static_cast<float>(config.head_dim));
        config.indexer_softmax_scale = config.indexer_softmax_scale > 0.0f
            ? config.indexer_softmax_scale
            : 1.0f / std::sqrt(static_cast<float>(config.index_head_dim));
        config.indexer_head_scale = config.indexer_head_scale > 0.0f
            ? config.indexer_head_scale
            : 1.0f / std::sqrt(static_cast<float>(config.index_n_heads));

        const int32_t ratio = config.layer_spec.compression_ratio;
        const int32_t coefficient = ratio == 4 ? 2 : 1;
        const size_t compressor_width = static_cast<size_t>(std::max(ratio, 1) * coefficient * config.head_dim);
        const size_t indexer_width = static_cast<size_t>(std::max(ratio, 1) * coefficient * config.index_head_dim);
        fill_or_validate(config.attention_sink, static_cast<size_t>(config.num_heads), 0.0f, "attention_sink");
        if (ratio == 0) {
            config.compressor_ape.clear();
            config.compressor_norm.clear();
            config.indexer_compressor_ape.clear();
            config.indexer_compressor_norm.clear();
        } else {
            fill_or_validate(config.compressor_ape, compressor_width, 0.0f, "compressor_ape");
            fill_or_validate(config.compressor_norm, static_cast<size_t>(config.head_dim), 1.0f, "compressor_norm");
            if (config.layer_spec.attention_kind == core::V4AttentionKind::CSA) {
                fill_or_validate(config.indexer_compressor_ape, indexer_width, 0.0f, "indexer_compressor_ape");
                fill_or_validate(config.indexer_compressor_norm, static_cast<size_t>(config.index_head_dim), 1.0f, "indexer_compressor_norm");
            } else {
                config.indexer_compressor_ape.clear();
                config.indexer_compressor_norm.clear();
            }
        }

        prepare_rope(config.main_rope, config.head_dim, config.rope_dim, "main");
        prepare_rope(config.compressed_rope, config.head_dim, config.rope_dim, "compressed");
        prepare_rope(config.indexer_rope, config.index_head_dim, config.rope_dim, "indexer");
        return config;
    }

    static void prepare_rope(
        V4OracleRopeTable& table,
        int32_t head_dim,
        int32_t default_rope_dim,
        const char* default_identity
    ) {
        if (table.identity.empty()) table.identity = default_identity;
        table.head_dim = head_dim;
        if (table.rope_dim == 0) table.rope_dim = default_rope_dim;
        table.validate();
    }

    static void fill_or_validate(
        std::vector<float>& values,
        size_t expected,
        float fill,
        const char* name
    ) {
        if (values.empty()) {
            values.assign(expected, fill);
        } else if (values.size() != expected) {
            throw std::invalid_argument(
                std::string("V4AttentionOracle: invalid ") + name + " size");
        }
    }

    bool is_compressed() const noexcept {
        return config_.layer_spec.attention_kind != core::V4AttentionKind::Sliding;
    }

    bool is_csa() const noexcept {
        return config_.layer_spec.attention_kind == core::V4AttentionKind::CSA;
    }

    int32_t compressor_coefficient() const noexcept {
        return config_.layer_spec.compression_ratio == 4 ? 2 : 1;
    }

    int32_t compressor_window() const noexcept {
        return compressor_coefficient() * config_.layer_spec.compression_ratio;
    }

    void validate_input(const V4OracleTokenInput& input) const {
        if (input.position < 0) {
            throw std::invalid_argument("V4AttentionOracle: token position must be non-negative");
        }
        expect_size(input.query, static_cast<size_t>(config_.num_heads * config_.head_dim), "query");
        expect_size(input.local_key, static_cast<size_t>(config_.head_dim), "local_key");
        expect_size(input.local_value, static_cast<size_t>(config_.head_dim), "local_value");
        if (!is_compressed()) return;

        const size_t compressor_width = static_cast<size_t>(compressor_coefficient() * config_.head_dim);
        expect_size(input.compressor_kv, compressor_width, "compressor_kv");
        expect_size(input.compressor_score, compressor_width, "compressor_score");
        if (!is_csa()) return;

        expect_size(input.indexer_query, static_cast<size_t>(config_.index_n_heads * config_.index_head_dim), "indexer_query");
        expect_size(input.indexer_weights, static_cast<size_t>(config_.index_n_heads), "indexer_weights");
        const size_t indexer_width = static_cast<size_t>(compressor_coefficient() * config_.index_head_dim);
        expect_size(input.indexer_compressor_kv, indexer_width, "indexer_compressor_kv");
        expect_size(input.indexer_compressor_score, indexer_width, "indexer_compressor_score");
    }

    static void expect_size(const std::vector<float>& values, size_t expected, const char* name) {
        if (values.size() != expected) {
            throw std::invalid_argument(
                std::string("V4AttentionOracle: invalid ") + name + " size");
        }
    }

    static void append_partial(
        std::vector<V4OraclePartialRow>& rows,
        int64_t position,
        const std::vector<float>& kv,
        const std::vector<float>& score,
        const std::vector<float>& ape,
        int32_t ratio,
        int32_t head_dim
    ) {
        const size_t width = static_cast<size_t>((ratio == 4 ? 2 : 1) * head_dim);
        V4OraclePartialRow row;
        row.position = position;
        row.kv = kv;
        row.score = score;
        const size_t ape_offset = static_cast<size_t>(position % ratio) * width;
        for (size_t index = 0; index < width; ++index) {
            row.score[index] += ape[ape_offset + index];
        }
        rows.push_back(std::move(row));
        const size_t max_rows = static_cast<size_t>((ratio == 4 ? 2 : 1) * ratio);
        if (rows.size() > max_rows) {
            rows.erase(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(rows.size() - max_rows));
        }
    }

    static std::vector<float> compress_rows(
        const std::vector<V4OraclePartialRow>& rows,
        int64_t boundary_position,
        int32_t ratio,
        int32_t head_dim,
        const std::vector<float>& norm,
        float eps
    ) {
        const int32_t coefficient = ratio == 4 ? 2 : 1;
        const int32_t window = coefficient * ratio;
        const int64_t start = boundary_position - window + 1;
        std::vector<float> compressed(static_cast<size_t>(head_dim), 0.0f);
        for (int32_t dimension = 0; dimension < head_dim; ++dimension) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (int32_t offset = 0; offset < window; ++offset) {
                const int64_t position = start + offset;
                if (position < 0) continue;
                const auto row = std::find_if(rows.begin(), rows.end(), [position](const auto& candidate) {
                    return candidate.position == position;
                });
                if (row == rows.end()) continue;
                const size_t segment = static_cast<size_t>(offset / ratio) * static_cast<size_t>(head_dim);
                maximum = std::max(maximum, row->score[segment + static_cast<size_t>(dimension)]);
            }
            float denominator = 0.0f;
            for (int32_t offset = 0; offset < window; ++offset) {
                const int64_t position = start + offset;
                if (position < 0) continue;
                const auto row = std::find_if(rows.begin(), rows.end(), [position](const auto& candidate) {
                    return candidate.position == position;
                });
                if (row == rows.end()) continue;
                const size_t segment = static_cast<size_t>(offset / ratio) * static_cast<size_t>(head_dim);
                denominator += std::exp(row->score[segment + static_cast<size_t>(dimension)] - maximum);
            }
            if (denominator <= 0.0f) continue;
            for (int32_t offset = 0; offset < window; ++offset) {
                const int64_t position = start + offset;
                if (position < 0) continue;
                const auto row = std::find_if(rows.begin(), rows.end(), [position](const auto& candidate) {
                    return candidate.position == position;
                });
                if (row == rows.end()) continue;
                const size_t segment = static_cast<size_t>(offset / ratio) * static_cast<size_t>(head_dim);
                const float weight = std::exp(row->score[segment + static_cast<size_t>(dimension)] - maximum) / denominator;
                compressed[static_cast<size_t>(dimension)] +=
                    weight * row->kv[segment + static_cast<size_t>(dimension)];
            }
        }

        float sum_sq = 0.0f;
        for (const float value : compressed) sum_sq += value * value;
        const float inverse_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(head_dim) + eps);
        for (int32_t dimension = 0; dimension < head_dim; ++dimension) {
            compressed[static_cast<size_t>(dimension)] *= inverse_rms * norm[static_cast<size_t>(dimension)];
        }
        return compressed;
    }

    static V4OracleCompressedEntry create_compressed_entry(
        const std::vector<V4OraclePartialRow>& rows,
        int64_t boundary_position,
        int32_t ratio,
        int32_t head_dim,
        const std::vector<float>& norm,
        const V4OracleRopeTable& rope,
        V4OracleRopeKind rope_kind,
        float rms_norm_eps
    ) {
        std::vector<float> value = compress_rows(rows, boundary_position, ratio, head_dim, norm, rms_norm_eps);
        const int64_t rope_position = (boundary_position / ratio) * ratio;
        rope.apply(value, 1, rope_position);
        return {boundary_position, rope_position, rope_kind, value, value};
    }

    size_t collect_local_positions(int64_t current_position, std::vector<int64_t>& positions) const {
        positions.clear();
        for (const auto& entry : local_cache_) {
            if (entry.position >= 0 && entry.position <= current_position) positions.push_back(entry.position);
        }
        std::sort(positions.begin(), positions.end());
        return positions.size();
    }

    std::vector<float> compute_indexer_scores() const {
        std::vector<float> scores(indexer_entries_.size(), 0.0f);
        for (size_t candidate = 0; candidate < indexer_entries_.size(); ++candidate) {
            const auto& key = indexer_entries_[candidate].key;
            float score = 0.0f;
            for (int32_t head = 0; head < config_.index_n_heads; ++head) {
                float dot = 0.0f;
                const size_t offset = static_cast<size_t>(head * config_.index_head_dim);
                for (int32_t dimension = 0; dimension < config_.index_head_dim; ++dimension) {
                    dot += last_indexer_query_[offset + static_cast<size_t>(dimension)] * key[static_cast<size_t>(dimension)];
                }
                score += dot * last_indexer_weights_[static_cast<size_t>(head)] *
                    config_.indexer_softmax_scale * config_.indexer_head_scale;
            }
            scores[candidate] = score;
        }
        return scores;
    }

    std::vector<int32_t> select_topk(const std::vector<float>& scores) const {
        std::vector<int32_t> selected(static_cast<size_t>(config_.index_topk), -1);
        if (scores.size() <= static_cast<size_t>(config_.index_topk)) {
            for (size_t index = 0; index < scores.size(); ++index) selected[index] = static_cast<int32_t>(index);
            return selected;
        }
        std::vector<int32_t> order(scores.size());
        for (size_t index = 0; index < scores.size(); ++index) order[index] = static_cast<int32_t>(index);
        std::stable_sort(order.begin(), order.end(), [&scores](int32_t left, int32_t right) {
            if (scores[static_cast<size_t>(left)] != scores[static_cast<size_t>(right)]) {
                return scores[static_cast<size_t>(left)] > scores[static_cast<size_t>(right)];
            }
            return left < right;
        });
        std::copy_n(order.begin(), static_cast<size_t>(config_.index_topk), selected.begin());
        return selected;
    }

    std::vector<float> compute_attention(
        const V4OracleStepResult& result,
        const std::vector<float>& query
    ) const {
        std::vector<const V4OracleLocalEntry*> local_entries;
        for (const auto& entry : local_cache_) {
            if (entry.position >= 0 && entry.position <= result.position) local_entries.push_back(&entry);
        }
        std::sort(local_entries.begin(), local_entries.end(), [](const auto* left, const auto* right) {
            return left->position < right->position;
        });

        std::vector<const V4OracleCompressedEntry*> compressed_entries;
        if (is_csa()) {
            for (const int32_t index : result.topk_indices) {
                if (index >= 0) compressed_entries.push_back(&compressed_entries_[static_cast<size_t>(index)]);
            }
        } else if (config_.layer_spec.attention_kind == core::V4AttentionKind::HCA) {
            for (const auto& entry : compressed_entries_) compressed_entries.push_back(&entry);
        }

        std::vector<float> output(static_cast<size_t>(config_.num_heads * config_.head_dim), 0.0f);
        for (int32_t head = 0; head < config_.num_heads; ++head) {
            const size_t query_offset = static_cast<size_t>(head * config_.head_dim);
            std::vector<const std::vector<float>*> attention_keys;
            std::vector<const std::vector<float>*> attention_values;
            attention_keys.reserve(local_entries.size() + compressed_entries.size());
            attention_values.reserve(local_entries.size() + compressed_entries.size());
            for (const auto* entry : local_entries) {
                attention_keys.push_back(&entry->key);
                attention_values.push_back(&entry->value);
            }
            for (const auto* entry : compressed_entries) {
                attention_keys.push_back(&entry->key);
                attention_values.push_back(&entry->value);
            }

            std::vector<float> scores;
            scores.reserve(attention_keys.size());
            float maximum = config_.attention_sink[static_cast<size_t>(head)];
            for (const auto* key : attention_keys) {
                const float score = dot(query, query_offset, *key, config_.attention_scale);
                scores.push_back(score);
                maximum = std::max(maximum, score);
            }

            float denominator = std::exp(config_.attention_sink[static_cast<size_t>(head)] - maximum);
            for (const float score : scores) denominator += std::exp(score - maximum);
            const float inverse_denominator = 1.0f / std::max(denominator, 1e-30f);
            for (float& score : scores) score = std::exp(score - maximum) * inverse_denominator;
            for (int32_t dimension = 0; dimension < config_.head_dim; ++dimension) {
                float value = 0.0f;
                for (size_t entry_index = 0; entry_index < attention_values.size(); ++entry_index) {
                    value += scores[entry_index] *
                        (*attention_values[entry_index])[static_cast<size_t>(dimension)];
                }
                output[query_offset + static_cast<size_t>(dimension)] = value;
            }
        }
        return output;
    }

    static float dot(
        const std::vector<float>& left,
        size_t left_offset,
        const std::vector<float>& right,
        float scale
    ) {
        float value = 0.0f;
        for (size_t index = 0; index < right.size(); ++index) value += left[left_offset + index] * right[index];
        return value * scale;
    }

    void validate_snapshot(const V4OracleStateSnapshot& state) const {
        if (state.next_position < 0 || state.local_cache.size() != static_cast<size_t>(config_.sliding_window)) {
            throw std::invalid_argument("V4AttentionOracle: invalid state cache");
        }
        for (const auto& entry : state.local_cache) {
            if (entry.key.size() != static_cast<size_t>(config_.head_dim) ||
                entry.value.size() != static_cast<size_t>(config_.head_dim)) {
                throw std::invalid_argument("V4AttentionOracle: invalid local state vector");
            }
        }
        validate_partial_rows(state.compressor_partial, config_.head_dim, is_compressed());
        validate_partial_rows(state.indexer_partial, config_.index_head_dim, is_csa());
        for (const auto& entry : state.compressed_entries) {
            if (entry.key.size() != static_cast<size_t>(config_.head_dim) ||
                entry.value.size() != static_cast<size_t>(config_.head_dim)) {
                throw std::invalid_argument("V4AttentionOracle: invalid compressed state vector");
            }
        }
        for (const auto& entry : state.indexer_entries) {
            if (entry.key.size() != static_cast<size_t>(config_.index_head_dim)) {
                throw std::invalid_argument("V4AttentionOracle: invalid indexer state vector");
            }
        }
        if (state.last_indexer_query.size() != (is_csa() ? static_cast<size_t>(config_.index_n_heads * config_.index_head_dim) : 0) ||
            state.last_indexer_weights.size() != (is_csa() ? static_cast<size_t>(config_.index_n_heads) : 0)) {
            throw std::invalid_argument("V4AttentionOracle: invalid last indexer state");
        }
        validate_step(state.last_step);
    }

    void validate_partial_rows(
        const std::vector<V4OraclePartialRow>& rows,
        int32_t head_dim,
        bool required
    ) const {
        if (!required && !rows.empty()) throw std::invalid_argument("V4AttentionOracle: unexpected partial state");
        const size_t width = required ? static_cast<size_t>(compressor_coefficient() * head_dim) : 0;
        for (const auto& row : rows) {
            if (row.position < 0 || row.kv.size() != width || row.score.size() != width) {
                throw std::invalid_argument("V4AttentionOracle: invalid partial state row");
            }
        }
    }

    void validate_step(const V4OracleStepResult& step) const {
        if (step.position < -1) throw std::invalid_argument("V4AttentionOracle: invalid last step");
        if (step.rotated_query.size() != (step.position < 0 ? 0 : static_cast<size_t>(config_.num_heads * config_.head_dim)) ||
            step.attention_output.size() != (step.position < 0 ? 0 : static_cast<size_t>(config_.num_heads * config_.head_dim)) ||
            step.inverse_rope_output.size() != (step.position < 0 ? 0 : static_cast<size_t>(config_.num_heads * config_.head_dim))) {
            throw std::invalid_argument("V4AttentionOracle: invalid last attention state");
        }
        if (step.topk_indices.size() != (is_csa() && step.position >= 0 ? static_cast<size_t>(config_.index_topk) : 0)) {
            throw std::invalid_argument("V4AttentionOracle: invalid last top-k state");
        }
    }

    static void append_bytes(std::vector<uint8_t>& output, const void* data, size_t size) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        output.insert(output.end(), bytes, bytes + size);
    }

    template<typename T>
    static void write_scalar(std::vector<uint8_t>& output, T value) {
        append_bytes(output, &value, sizeof(value));
    }

    template<typename T>
    static void write_vector(std::vector<uint8_t>& output, const std::vector<T>& values) {
        write_scalar<uint64_t>(output, static_cast<uint64_t>(values.size()));
        if (!values.empty()) append_bytes(output, values.data(), values.size() * sizeof(T));
    }

    static void write_local_entries(std::vector<uint8_t>& output, const std::vector<V4OracleLocalEntry>& entries) {
        write_scalar<uint64_t>(output, static_cast<uint64_t>(entries.size()));
        for (const auto& entry : entries) {
            write_scalar<int64_t>(output, entry.position);
            write_vector(output, entry.key);
            write_vector(output, entry.value);
        }
    }

    static void write_partial_rows(std::vector<uint8_t>& output, const std::vector<V4OraclePartialRow>& rows) {
        write_scalar<uint64_t>(output, static_cast<uint64_t>(rows.size()));
        for (const auto& row : rows) {
            write_scalar<int64_t>(output, row.position);
            write_vector(output, row.kv);
            write_vector(output, row.score);
        }
    }

    static void write_compressed_entries(std::vector<uint8_t>& output, const std::vector<V4OracleCompressedEntry>& entries) {
        write_scalar<uint64_t>(output, static_cast<uint64_t>(entries.size()));
        for (const auto& entry : entries) {
            write_scalar<int64_t>(output, entry.boundary_position);
            write_scalar<int64_t>(output, entry.rope_position);
            write_scalar<uint8_t>(output, static_cast<uint8_t>(entry.rope_kind));
            write_vector(output, entry.key);
            write_vector(output, entry.value);
        }
    }

    static void write_indexer_entries(std::vector<uint8_t>& output, const std::vector<V4OracleIndexerEntry>& entries) {
        write_scalar<uint64_t>(output, static_cast<uint64_t>(entries.size()));
        for (const auto& entry : entries) {
            write_scalar<int64_t>(output, entry.boundary_position);
            write_scalar<int64_t>(output, entry.rope_position);
            write_scalar<uint8_t>(output, static_cast<uint8_t>(entry.rope_kind));
            write_vector(output, entry.key);
        }
    }

    static void write_step(std::vector<uint8_t>& output, const V4OracleStepResult& step) {
        write_scalar<int64_t>(output, step.position);
        write_scalar<uint8_t>(output, step.compressed_entry_created ? 1u : 0u);
        write_scalar<uint64_t>(output, static_cast<uint64_t>(step.compressed_entry_index));
        write_scalar<int64_t>(output, step.compressed_rope_position);
        write_scalar<uint64_t>(output, static_cast<uint64_t>(step.local_valid_count));
        write_scalar<uint64_t>(output, static_cast<uint64_t>(step.compressed_entry_count));
        write_scalar<uint64_t>(output, static_cast<uint64_t>(step.selected_compressed_count));
        write_scalar<uint64_t>(output, static_cast<uint64_t>(step.indexer_candidate_count));
        write_vector(output, step.local_positions);
        write_vector(output, step.selected_compressed_positions);
        write_vector(output, step.rotated_query);
        write_vector(output, step.attention_output);
        write_vector(output, step.inverse_rope_output);
        write_vector(output, step.indexer_scores);
        write_vector(output, step.topk_indices);
    }

    class ByteReader {
    public:
        explicit ByteReader(std::span<const uint8_t> bytes)
            : bytes_(bytes) {}

        template<typename T>
        T read() {
            if (offset_ > bytes_.size() || sizeof(T) > bytes_.size() - offset_) {
                throw std::invalid_argument("V4AttentionOracle: truncated serialized state");
            }
            T value;
            std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
            offset_ += sizeof(T);
            return value;
        }

        template<typename T>
        std::vector<T> read_vector() {
            const uint64_t count = read<uint64_t>();
            if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
                count > (bytes_.size() - offset_) / sizeof(T)) {
                throw std::invalid_argument("V4AttentionOracle: invalid serialized vector");
            }
            std::vector<T> values(static_cast<size_t>(count));
            if (!values.empty()) {
                std::memcpy(values.data(), bytes_.data() + offset_, values.size() * sizeof(T));
                offset_ += values.size() * sizeof(T);
            }
            return values;
        }

        void require_end() const {
            if (offset_ != bytes_.size()) throw std::invalid_argument("V4AttentionOracle: trailing serialized state bytes");
        }

    private:
        std::span<const uint8_t> bytes_;
        size_t offset_{0};
    };

    static std::vector<V4OracleLocalEntry> read_local_entries(ByteReader& reader) {
        const uint64_t count = reader.read<uint64_t>();
        std::vector<V4OracleLocalEntry> entries;
        entries.reserve(static_cast<size_t>(count));
        for (uint64_t index = 0; index < count; ++index) {
            entries.push_back({reader.read<int64_t>(), reader.read_vector<float>(), reader.read_vector<float>()});
        }
        return entries;
    }

    static std::vector<V4OraclePartialRow> read_partial_rows(ByteReader& reader) {
        const uint64_t count = reader.read<uint64_t>();
        std::vector<V4OraclePartialRow> rows;
        rows.reserve(static_cast<size_t>(count));
        for (uint64_t index = 0; index < count; ++index) {
            rows.push_back({reader.read<int64_t>(), reader.read_vector<float>(), reader.read_vector<float>()});
        }
        return rows;
    }

    static std::vector<V4OracleCompressedEntry> read_compressed_entries(ByteReader& reader) {
        const uint64_t count = reader.read<uint64_t>();
        std::vector<V4OracleCompressedEntry> entries;
        entries.reserve(static_cast<size_t>(count));
        for (uint64_t index = 0; index < count; ++index) {
            entries.push_back({
                reader.read<int64_t>(),
                reader.read<int64_t>(),
                static_cast<V4OracleRopeKind>(reader.read<uint8_t>()),
                reader.read_vector<float>(),
                reader.read_vector<float>()
            });
        }
        return entries;
    }

    static std::vector<V4OracleIndexerEntry> read_indexer_entries(ByteReader& reader) {
        const uint64_t count = reader.read<uint64_t>();
        std::vector<V4OracleIndexerEntry> entries;
        entries.reserve(static_cast<size_t>(count));
        for (uint64_t index = 0; index < count; ++index) {
            entries.push_back({
                reader.read<int64_t>(),
                reader.read<int64_t>(),
                static_cast<V4OracleRopeKind>(reader.read<uint8_t>()),
                reader.read_vector<float>()
            });
        }
        return entries;
    }

    static V4OracleStepResult read_step(ByteReader& reader) {
        V4OracleStepResult step;
        step.position = reader.read<int64_t>();
        step.compressed_entry_created = reader.read<uint8_t>() != 0;
        step.compressed_entry_index = static_cast<size_t>(reader.read<uint64_t>());
        step.compressed_rope_position = reader.read<int64_t>();
        step.local_valid_count = static_cast<size_t>(reader.read<uint64_t>());
        step.compressed_entry_count = static_cast<size_t>(reader.read<uint64_t>());
        step.selected_compressed_count = static_cast<size_t>(reader.read<uint64_t>());
        step.indexer_candidate_count = static_cast<size_t>(reader.read<uint64_t>());
        step.local_positions = reader.read_vector<int64_t>();
        step.selected_compressed_positions = reader.read_vector<int64_t>();
        step.rotated_query = reader.read_vector<float>();
        step.attention_output = reader.read_vector<float>();
        step.inverse_rope_output = reader.read_vector<float>();
        step.indexer_scores = reader.read_vector<float>();
        step.topk_indices = reader.read_vector<int32_t>();
        return step;
    }

    V4OracleConfig config_;
    int64_t next_position_{0};
    std::vector<V4OracleLocalEntry> local_cache_;
    std::vector<V4OraclePartialRow> compressor_partial_;
    std::vector<V4OraclePartialRow> indexer_partial_;
    std::vector<V4OracleCompressedEntry> compressed_entries_;
    std::vector<V4OracleIndexerEntry> indexer_entries_;
    std::vector<float> last_indexer_query_;
    std::vector<float> last_indexer_weights_;
    V4OracleStepResult last_step_;
};

} // namespace aeon::reference