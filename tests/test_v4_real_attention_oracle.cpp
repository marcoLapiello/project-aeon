#include "infrastructure/core/aeon_loader.hpp"
#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_model_contract.hpp"
#include "architecture/deepseek_v4/reference/v4_attention_oracle.hpp"
#include "architecture/deepseek_v4/reference/v4_int4_reference.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using aeon::core::DeepSeekV4Config;
using aeon::core::LoadedTensor;
using aeon::core::V4AttentionKind;
using aeon::core::V4LayerSpec;
using aeon::reference::V4AttentionOracle;
using aeon::reference::V4OracleConfig;
using aeon::reference::V4OracleStateSnapshot;
using aeon::reference::V4OracleTokenInput;

constexpr int32_t kTraceTokens = 132;
constexpr int32_t kProjectionSeeds = 4;
constexpr float kTraceTolerance = 2e-5f;

const LoadedTensor& require_tensor(
    const aeon::core::AeonModelLoader& loader,
    const std::string& name,
    const std::vector<int64_t>& shape,
    const std::string& dtype
) {
    if (!loader.has_tensor(name)) {
        throw std::runtime_error("Missing real attention trace tensor " + name);
    }
    const auto& tensor = loader.get_tensor(name);
    if (tensor.dtype != dtype || tensor.shape != shape) {
        throw std::runtime_error("Unexpected real attention trace tensor contract for " + name);
    }
    return tensor;
}

std::vector<uint16_t> load_half_bits(
    const aeon::core::AeonModelLoader& loader,
    const std::string& name,
    const std::vector<int64_t>& shape
) {
    const auto& tensor = require_tensor(loader, name, shape, "F16");
    size_t count = 1;
    for (const int64_t dimension : shape) count *= static_cast<size_t>(dimension);
    std::vector<uint16_t> bits(count);
    for (size_t index = 0; index < count; ++index) {
        bits[index] = aeon::reference::load_u16(tensor.data + index * sizeof(uint16_t));
    }
    return bits;
}

std::vector<float> load_float_values(
    const aeon::core::AeonModelLoader& loader,
    const std::string& name,
    const std::vector<int64_t>& shape
) {
    const auto& tensor = require_tensor(loader, name, shape, "F32");
    size_t count = 1;
    for (const int64_t dimension : shape) count *= static_cast<size_t>(dimension);
    std::vector<float> values(count);
    std::memcpy(values.data(), tensor.data, values.size() * sizeof(float));
    return values;
}

std::vector<float> bits_to_float(const std::vector<uint16_t>& bits) {
    std::vector<float> values(bits.size());
    for (size_t index = 0; index < bits.size(); ++index) {
        values[index] = aeon::reference::fp16_to_float(bits[index]);
    }
    return values;
}

std::vector<uint16_t> float_to_half_bits(const std::vector<float>& values) {
    std::vector<uint16_t> bits(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        const half value = __float2half(values[index]);
        std::memcpy(&bits[index], &value, sizeof(uint16_t));
    }
    return bits;
}

std::vector<uint16_t> make_hidden(int32_t layer_id, int32_t seed) {
    std::vector<uint16_t> bits(4096);
    for (size_t index = 0; index < bits.size(); ++index) {
        const int32_t pattern = static_cast<int32_t>(
            (index * static_cast<size_t>(seed + layer_id + 11) +
             static_cast<size_t>(layer_id * 7 + seed * 13)) % 41) - 20;
        const half value = __float2half(static_cast<float>(pattern) * 0.00390625f);
        std::memcpy(&bits[index], &value, sizeof(uint16_t));
    }
    return bits;
}

std::vector<float> gemv_float(
    const aeon::core::AeonModelLoader& loader,
    const std::string& name,
    const std::vector<uint16_t>& activation,
    int32_t rows,
    int32_t columns
) {
    const auto& tensor = require_tensor(
        loader, name, {rows, columns}, "F16");
    std::vector<float> output(static_cast<size_t>(rows));
    aeon::reference::decode_fp16_gemv(
        tensor.data, activation.data(), rows, columns, output.data());
    return output;
}

std::vector<uint16_t> gemv_half(
    const aeon::core::AeonModelLoader& loader,
    const std::string& name,
    const std::vector<uint16_t>& activation,
    int32_t rows,
    int32_t columns
) {
    return float_to_half_bits(gemv_float(loader, name, activation, rows, columns));
}

std::vector<uint16_t> rmsnorm_half(
    const std::vector<uint16_t>& input,
    const std::vector<uint16_t>& weight,
    float epsilon
) {
    assert(input.size() == weight.size());
    float sum_squared = 0.0f;
    for (size_t index = 0; index < input.size(); ++index) {
        const float value = aeon::reference::fp16_to_float(input[index]);
        sum_squared += value * value;
    }
    const float inverse_rms = 1.0f / std::sqrt(
        sum_squared / static_cast<float>(input.size()) + epsilon);
    std::vector<float> output(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        output[index] = aeon::reference::fp16_to_float(input[index]) * inverse_rms *
            aeon::reference::fp16_to_float(weight[index]);
    }
    return float_to_half_bits(output);
}

std::vector<uint16_t> unit_rmsnorm_half(
    const std::vector<uint16_t>& input,
    int32_t width,
    float epsilon
) {
    assert(input.size() % static_cast<size_t>(width) == 0);
    std::vector<float> output(input.size());
    for (size_t start = 0; start < input.size(); start += static_cast<size_t>(width)) {
        float sum_squared = 0.0f;
        for (int32_t index = 0; index < width; ++index) {
            const float value = aeon::reference::fp16_to_float(input[start + static_cast<size_t>(index)]);
            sum_squared += value * value;
        }
        const float inverse_rms = 1.0f / std::sqrt(
            sum_squared / static_cast<float>(width) + epsilon);
        for (int32_t index = 0; index < width; ++index) {
            output[start + static_cast<size_t>(index)] =
                aeon::reference::fp16_to_float(input[start + static_cast<size_t>(index)]) * inverse_rms;
        }
    }
    return float_to_half_bits(output);
}

V4OracleConfig make_oracle_config(
    const DeepSeekV4Config& model_config,
    const V4LayerSpec& layer_spec,
    const aeon::core::AeonModelLoader& loader
) {
    const std::string prefix = "layers." + std::to_string(layer_spec.layer_id) + ".";
    V4OracleConfig config;
    config.layer_spec = layer_spec;
    config.num_heads = model_config.num_attention_heads;
    config.head_dim = model_config.head_dim;
    config.index_n_heads = model_config.index_n_heads;
    config.index_head_dim = model_config.index_head_dim;
    config.index_topk = model_config.index_topk;
    config.sliding_window = model_config.sliding_window;
    config.rope_dim = model_config.qk_rope_head_dim;
    config.rms_norm_eps = model_config.rms_norm_eps;
    config.attention_sink = load_float_values(
        loader, prefix + "attn.attn_sink", {model_config.num_attention_heads});

    config.main_rope.identity = "main";
    config.main_rope.head_dim = model_config.head_dim;
    config.main_rope.rope_dim = model_config.qk_rope_head_dim;
    config.main_rope.theta = model_config.rope_theta;
    config.main_rope.factor = 1.0f;
    config.main_rope.beta_fast = model_config.rope_scaling.beta_fast;
    config.main_rope.beta_slow = model_config.rope_scaling.beta_slow;
    config.main_rope.original_max_position = model_config.original_max_position_embeddings;

    for (auto* table : {&config.compressed_rope, &config.indexer_rope}) {
        table->theta = model_config.compress_rope_theta;
        table->factor = model_config.rope_scaling.factor;
        table->beta_fast = model_config.rope_scaling.beta_fast;
        table->beta_slow = model_config.rope_scaling.beta_slow;
        table->original_max_position = model_config.rope_scaling.original_max_position_embeddings;
    }
    config.compressed_rope.identity = "compressed";
    config.compressed_rope.head_dim = model_config.head_dim;
    config.compressed_rope.rope_dim = model_config.qk_rope_head_dim;
    config.indexer_rope.identity = "indexer";
    config.indexer_rope.head_dim = model_config.index_head_dim;
    config.indexer_rope.rope_dim = model_config.qk_rope_head_dim;

    if (layer_spec.attention_kind == V4AttentionKind::Sliding) return config;

    const int32_t coefficient = layer_spec.compression_ratio == 4 ? 2 : 1;
    const int32_t compressor_width = coefficient * model_config.head_dim;
    config.compressor_ape = load_float_values(
        loader, prefix + "attn.compressor.ape",
        {layer_spec.compression_ratio, compressor_width});
    config.compressor_norm = bits_to_float(load_half_bits(
        loader, prefix + "attn.compressor.norm.weight", {model_config.head_dim}));

    if (layer_spec.attention_kind == V4AttentionKind::CSA) {
        const int32_t indexer_width = coefficient * model_config.index_head_dim;
        config.indexer_compressor_ape = load_float_values(
            loader, prefix + "attn.indexer.compressor.ape",
            {layer_spec.compression_ratio, indexer_width});
        config.indexer_compressor_norm = bits_to_float(load_half_bits(
            loader, prefix + "attn.indexer.compressor.norm.weight",
            {model_config.index_head_dim}));
    }
    return config;
}

V4OracleTokenInput make_projection_seed(
    const aeon::core::AeonModelLoader& loader,
    const DeepSeekV4Config& model_config,
    const V4LayerSpec& layer_spec,
    const V4OracleConfig& oracle_config,
    int32_t seed
) {
    const std::string prefix = "layers." + std::to_string(layer_spec.layer_id) + ".";
    const std::vector<uint16_t> hidden = make_hidden(
        static_cast<int32_t>(layer_spec.layer_id), seed);
    const std::vector<uint16_t> attn_norm = load_half_bits(
        loader, prefix + "attn_norm.weight", {model_config.hidden_size});
    const std::vector<uint16_t> q_norm = load_half_bits(
        loader, prefix + "attn.q_norm.weight", {model_config.q_lora_rank});
    const std::vector<uint16_t> kv_norm = load_half_bits(
        loader, prefix + "attn.kv_norm.weight", {model_config.head_dim});

    const std::vector<uint16_t> normalized = rmsnorm_half(
        hidden, attn_norm, model_config.rms_norm_eps);
    const std::vector<uint16_t> q_lora = gemv_half(
        loader, prefix + "attn.wq_a.weight", normalized,
        model_config.q_lora_rank, model_config.hidden_size);
    const std::vector<uint16_t> q_lora_norm = rmsnorm_half(
        q_lora, q_norm, model_config.rms_norm_eps);
    const std::vector<uint16_t> query_projected = gemv_half(
        loader, prefix + "attn.wq_b.weight", q_lora_norm,
        model_config.num_attention_heads * model_config.head_dim,
        model_config.q_lora_rank);
    const std::vector<uint16_t> query = unit_rmsnorm_half(
        query_projected, model_config.head_dim, model_config.rms_norm_eps);
    const std::vector<uint16_t> kv_projected = gemv_half(
        loader, prefix + "attn.wkv.weight", normalized,
        model_config.head_dim, model_config.hidden_size);
    const std::vector<uint16_t> kv = rmsnorm_half(
        kv_projected, kv_norm, model_config.rms_norm_eps);

    V4OracleTokenInput input;
    input.query = bits_to_float(query);
    input.local_key = bits_to_float(kv);
    input.local_value = input.local_key;
    if (layer_spec.attention_kind == V4AttentionKind::Sliding) return input;

    input.compressor_kv = gemv_float(
        loader, prefix + "attn.compressor.wkv.weight", normalized,
        oracle_config.compressor_ape.empty()
            ? model_config.head_dim
            : static_cast<int32_t>(oracle_config.compressor_ape.size() /
                                   static_cast<size_t>(layer_spec.compression_ratio)),
        model_config.hidden_size);
    input.compressor_score = gemv_float(
        loader, prefix + "attn.compressor.wgate.weight", normalized,
        static_cast<int32_t>(input.compressor_kv.size()), model_config.hidden_size);
    if (layer_spec.attention_kind != V4AttentionKind::CSA) return input;

    input.indexer_query = gemv_float(
        loader, prefix + "attn.indexer.wq_b.weight", q_lora_norm,
        model_config.index_n_heads * model_config.index_head_dim,
        model_config.q_lora_rank);
    input.indexer_weights = gemv_float(
        loader, prefix + "attn.indexer.weights_proj.weight", normalized,
        model_config.index_n_heads, model_config.hidden_size);
    input.indexer_compressor_kv = gemv_float(
        loader, prefix + "attn.indexer.compressor.wkv.weight", normalized,
        static_cast<int32_t>(oracle_config.indexer_compressor_ape.size() /
                             static_cast<size_t>(layer_spec.compression_ratio)),
        model_config.hidden_size);
    input.indexer_compressor_score = gemv_float(
        loader, prefix + "attn.indexer.compressor.wgate.weight", normalized,
        static_cast<int32_t>(input.indexer_compressor_kv.size()), model_config.hidden_size);
    return input;
}

std::vector<V4OracleTokenInput> make_real_inputs(
    const aeon::core::AeonModelLoader& loader,
    const DeepSeekV4Config& model_config,
    const V4LayerSpec& layer_spec,
    const V4OracleConfig& oracle_config
) {
    std::vector<V4OracleTokenInput> seeds;
    const int32_t seed_count = layer_spec.attention_kind == V4AttentionKind::Sliding
        ? 1 : kProjectionSeeds;
    seeds.reserve(static_cast<size_t>(seed_count));
    for (int32_t seed = 0; seed < seed_count; ++seed) {
        seeds.push_back(make_projection_seed(
            loader, model_config, layer_spec, oracle_config, seed));
    }

    std::vector<V4OracleTokenInput> inputs;
    inputs.reserve(kTraceTokens);
    for (int32_t position = 0; position < kTraceTokens; ++position) {
        V4OracleTokenInput input = seeds[static_cast<size_t>(position % seed_count)];
        input.position = position;
        inputs.push_back(std::move(input));
    }
    return inputs;
}

std::vector<float> independent_compressed_entry(
    const V4OracleConfig& config,
    std::span<const V4OracleTokenInput> inputs,
    int32_t boundary,
    bool indexer
) {
    const int32_t ratio = config.layer_spec.compression_ratio;
    const int32_t coefficient = ratio == 4 ? 2 : 1;
    const int32_t dimension = indexer ? config.index_head_dim : config.head_dim;
    const int32_t width = coefficient * dimension;
    const auto& ape = indexer ? config.indexer_compressor_ape : config.compressor_ape;
    const auto& norm = indexer ? config.indexer_compressor_norm : config.compressor_norm;
    const auto& rope = indexer ? config.indexer_rope : config.compressed_rope;
    const int32_t window = coefficient * ratio;
    const int32_t start = boundary - window + 1;

    std::vector<float> compressed(static_cast<size_t>(dimension), 0.0f);
    for (int32_t column = 0; column < dimension; ++column) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int32_t offset = 0; offset < window; ++offset) {
            const int32_t position = start + offset;
            const auto& input = inputs[static_cast<size_t>(position)];
            const auto& score = indexer ? input.indexer_compressor_score : input.compressor_score;
            const size_t segment = static_cast<size_t>(offset / ratio * dimension + column);
            const size_t ape_index = static_cast<size_t>(position % ratio * width) + segment;
            maximum = std::max(maximum, score[segment] + ape[ape_index]);
        }

        float denominator = 0.0f;
        for (int32_t offset = 0; offset < window; ++offset) {
            const int32_t position = start + offset;
            const auto& input = inputs[static_cast<size_t>(position)];
            const auto& score = indexer ? input.indexer_compressor_score : input.compressor_score;
            const size_t segment = static_cast<size_t>(offset / ratio * dimension + column);
            const size_t ape_index = static_cast<size_t>(position % ratio * width) + segment;
            denominator += std::exp(score[segment] + ape[ape_index] - maximum);
        }

        for (int32_t offset = 0; offset < window; ++offset) {
            const int32_t position = start + offset;
            const auto& input = inputs[static_cast<size_t>(position)];
            const auto& values = indexer ? input.indexer_compressor_kv : input.compressor_kv;
            const auto& score = indexer ? input.indexer_compressor_score : input.compressor_score;
            const size_t segment = static_cast<size_t>(offset / ratio * dimension + column);
            const size_t ape_index = static_cast<size_t>(position % ratio * width) + segment;
            const float weight = std::exp(score[segment] + ape[ape_index] - maximum) / denominator;
            compressed[static_cast<size_t>(column)] += weight * values[segment];
        }
    }

    float sum_squared = 0.0f;
    for (const float value : compressed) sum_squared += value * value;
    const float inverse_rms = 1.0f / std::sqrt(
        sum_squared / static_cast<float>(dimension) + config.rms_norm_eps);
    for (int32_t column = 0; column < dimension; ++column) {
        compressed[static_cast<size_t>(column)] *= inverse_rms * norm[static_cast<size_t>(column)];
    }
    const int64_t rope_position = (boundary / ratio) * ratio;
    rope.apply(compressed, 1, rope_position);
    return compressed;
}

float max_abs_difference(const std::vector<float>& left, const std::vector<float>& right) {
    assert(left.size() == right.size());
    float maximum = 0.0f;
    for (size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

void assert_state_bytes_equal(
    const V4AttentionOracle& left,
    const V4AttentionOracle& right,
    const std::string& label
) {
    const auto left_bytes = left.serialize_state();
    const auto right_bytes = right.serialize_state();
    if (left_bytes != right_bytes) {
        throw std::runtime_error("Real Stage 2 state mismatch: " + label);
    }
}

void check_rope_tables(const V4OracleConfig& config) {
    if (config.layer_spec.attention_kind == V4AttentionKind::Sliding) return;
    std::vector<float> main(config.head_dim, 0.0f);
    std::vector<float> compressed(config.head_dim, 0.0f);
    const int32_t base = config.head_dim - config.rope_dim + 2;
    main[static_cast<size_t>(base)] = 1.0f;
    compressed[static_cast<size_t>(base)] = 1.0f;
    config.main_rope.apply(main, 1, 65536);
    config.compressed_rope.apply(compressed, 1, 65536);
    if (max_abs_difference(main, compressed) <= 1e-4f) {
        throw std::runtime_error("Real Stage 2 RoPE tables did not diverge at position 65536");
    }
}

void check_real_layer(
    const aeon::core::AeonModelLoader& loader,
    const DeepSeekV4Config& model_config,
    const V4LayerSpec& layer_spec
) {
    const V4OracleConfig oracle_config = make_oracle_config(model_config, layer_spec, loader);
    check_rope_tables(oracle_config);
    const auto inputs = make_real_inputs(loader, model_config, layer_spec, oracle_config);

    V4AttentionOracle one_shot(oracle_config);
    const auto one_shot_results = one_shot.append_batch(inputs);
    V4AttentionOracle chunked(oracle_config);
    size_t start = 0;
    for (const size_t boundary : {size_t{1}, size_t{3}, size_t{4}, size_t{7}, size_t{16}, size_t{127}, size_t{128}, size_t{132}}) {
        const size_t end = std::min(boundary, inputs.size());
        if (end > start) {
            chunked.append_batch(std::span<const V4OracleTokenInput>(inputs.data() + start, end - start));
        }
        start = end;
    }
    assert_state_bytes_equal(one_shot, chunked, "chunked layer " + std::to_string(layer_spec.layer_id));

    V4AttentionOracle serialized_prefix(oracle_config);
    serialized_prefix.append_batch(std::span<const V4OracleTokenInput>(inputs.data(), 73));
    const auto serialized = serialized_prefix.serialize_state();
    V4AttentionOracle serialized_suffix(oracle_config);
    serialized_suffix.restore_serialized(serialized);
    serialized_suffix.append_batch(std::span<const V4OracleTokenInput>(inputs.data() + 73, inputs.size() - 73));
    assert_state_bytes_equal(one_shot, serialized_suffix, "serialized layer " + std::to_string(layer_spec.layer_id));

    if (layer_spec.attention_kind == V4AttentionKind::Sliding) {
        assert(one_shot_results[0].local_valid_count == 1);
        assert(one_shot_results[127].local_valid_count == 128);
        assert(one_shot_results[131].local_positions.front() == 4);
        assert(one_shot_results[131].local_positions.back() == 131);
    } else {
        const int32_t ratio = layer_spec.compression_ratio;
        const int32_t first_boundary = ratio - 1;
        assert(one_shot_results[static_cast<size_t>(first_boundary)].compressed_entry_created);
        assert(one_shot_results[static_cast<size_t>(first_boundary)].compressed_entry_count == 1);
        assert(!one_shot_results[static_cast<size_t>(ratio)].compressed_entry_created);
        const int32_t boundary = layer_spec.attention_kind == V4AttentionKind::CSA ? 127 : 127;
        const size_t entry_index = static_cast<size_t>(boundary / ratio);
        const auto snapshot = one_shot.snapshot();
        const auto expected = independent_compressed_entry(
            oracle_config, inputs, boundary, false);
        const float value_error = max_abs_difference(
            snapshot.compressed_entries.at(entry_index).value, expected);
        if (value_error > kTraceTolerance) {
            throw std::runtime_error(
                "Real Stage 2 compressed value mismatch at layer " +
                std::to_string(layer_spec.layer_id) + ": " + std::to_string(value_error));
        }
        assert(one_shot_results[127].compressed_entry_count == 1 ||
               one_shot_results[127].compressed_entry_count == 32);
        assert(one_shot_results[131].compressed_entry_count == 1 ||
               one_shot_results[131].compressed_entry_count == 33);

        if (layer_spec.attention_kind == V4AttentionKind::CSA) {
            const auto expected_indexer = independent_compressed_entry(
                oracle_config, inputs, boundary, true);
            const float indexer_error = max_abs_difference(
                snapshot.indexer_entries.at(entry_index).key, expected_indexer);
            if (indexer_error > kTraceTolerance) {
                throw std::runtime_error(
                    "Real Stage 2 indexer key mismatch: " + std::to_string(indexer_error));
            }
            assert(one_shot_results[131].indexer_candidate_count == 33);
            assert(one_shot_results[131].topk_indices.size() == 512);
            assert(one_shot_results[131].topk_indices[33] == -1);
        } else {
            assert(one_shot_results[127].compressed_entry_created);
            assert(one_shot_results[131].selected_compressed_count == 1);
        }
    }

    std::cout << "[PASS] real Stage 2 layer " << layer_spec.layer_id << " "
              << aeon::core::v4_attention_kind_name(layer_spec.attention_kind)
              << " trace; " << one_shot.compact_trace() << std::endl;
}

} // namespace

int main() {
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    std::cout << "[Stage 2] Loading real dense tensors and model contract..." << std::endl;
    const auto model_config = DeepSeekV4Config::load_from_json(model_dir + "/config.json");
    const auto layer_specs = aeon::core::V4ModelSpec::resolve_layers(model_config);
    aeon::core::AeonModelLoader loader;
    loader.open_model(model_dir);
    aeon::core::V4ModelContract::validate(model_config, loader);

    for (const uint32_t layer_id : {0u, 2u, 3u}) {
        std::cout << "[Stage 2] Projecting real inputs for layer " << layer_id << "..." << std::endl;
        check_real_layer(loader, model_config, layer_specs.at(layer_id));
    }

    std::cout << "Real Stage 2 attention oracle passed: layers 0/2/3, real dense tensors, "
              << "boundary traces, chunk equivalence, serialization, and class-aware RoPE" << std::endl;
    return 0;
}