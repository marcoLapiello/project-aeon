#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_model_contract.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline.hpp"
#include "architecture/deepseek_v4/reference/v4_attention_oracle.hpp"
#include "architecture/deepseek_v4/reference/v4_int4_reference.hpp"
#include "platform/rdna3/device.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using aeon::core::AeonModelLoader;
using aeon::core::DeepSeekV4Config;
using aeon::core::LoadedTensor;
using aeon::core::V4AttentionKind;
using aeon::core::V4AttentionTraceRecord;
using aeon::core::V4LayerSpec;
using aeon::reference::V4AttentionOracle;
using aeon::reference::V4OracleConfig;
using aeon::reference::V4OracleStateSnapshot;
using aeon::reference::V4OracleTokenInput;

constexpr uint32_t kTraceTokens = 132;
constexpr float kAttentionTolerance = 2.0e-2f;
constexpr float kStateTolerance = 2.0e-3f;
constexpr float kIndexerScoreTolerance = 1.0e-1f;
constexpr float kGroupedProjectionTolerance = 2.0e-2f;

const LoadedTensor& require_tensor(
    const AeonModelLoader& loader,
    const std::string& name,
    const std::vector<int64_t>& shape,
    const std::string& dtype
) {
    if (!loader.has_tensor(name)) {
        throw std::runtime_error("Missing Stage 4 trace tensor " + name);
    }
    const auto& tensor = loader.get_tensor(name);
    if (tensor.dtype != dtype || tensor.shape != shape) {
        throw std::runtime_error("Unexpected Stage 4 trace tensor contract for " + name);
    }
    return tensor;
}

float load_half_value(const LoadedTensor& tensor, size_t index) {
    const uint16_t bits = aeon::reference::load_u16(
        tensor.data + index * sizeof(uint16_t));
    return aeon::reference::fp16_to_float(bits);
}

std::vector<float> load_float_values(
    const AeonModelLoader& loader,
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

std::vector<float> load_half_values(
    const AeonModelLoader& loader,
    const std::string& name,
    const std::vector<int64_t>& shape
) {
    const auto& tensor = require_tensor(loader, name, shape, "F16");
    size_t count = 1;
    for (const int64_t dimension : shape) count *= static_cast<size_t>(dimension);
    std::vector<float> values(count);
    for (size_t index = 0; index < count; ++index) {
        values[index] = load_half_value(tensor, index);
    }
    return values;
}

std::vector<float> half_to_float(std::span<const half> values) {
    std::vector<float> result(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        result[index] = __half2float(values[index]);
    }
    return result;
}

float max_abs_difference(std::span<const half> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Stage 4 trace vector size mismatch");
    }
    float maximum = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(__half2float(actual[index]) - expected[index]));
    }
    return maximum;
}

float max_abs_difference(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Stage 4 trace vector size mismatch");
    }
    float maximum = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    return maximum;
}

void require_close(
    float error,
    float tolerance,
    const std::string& label,
    uint32_t position,
    uint32_t layer_id
) {
    if (error > tolerance) {
        throw std::runtime_error(
            label + " mismatch at layer " + std::to_string(layer_id) +
            ", position " + std::to_string(position) +
            ": max_abs=" + std::to_string(error) +
            ", tolerance=" + std::to_string(tolerance));
    }
}

V4OracleConfig make_oracle_config(
    const DeepSeekV4Config& model_config,
    const V4LayerSpec& layer_spec,
    const AeonModelLoader& loader
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
    config.compressor_norm = load_half_values(
        loader, prefix + "attn.compressor.norm.weight", {model_config.head_dim});

    if (layer_spec.attention_kind == V4AttentionKind::CSA) {
        const int32_t indexer_width = coefficient * model_config.index_head_dim;
        config.indexer_compressor_ape = load_float_values(
            loader, prefix + "attn.indexer.compressor.ape",
            {layer_spec.compression_ratio, indexer_width});
        config.indexer_compressor_norm = load_half_values(
            loader, prefix + "attn.indexer.compressor.norm.weight",
            {model_config.index_head_dim});
    }
    return config;
}

V4OracleTokenInput make_oracle_input(const V4AttentionTraceRecord& trace) {
    V4OracleTokenInput input;
    input.position = trace.position;
    input.query = half_to_float(trace.query);
    input.local_key = half_to_float(trace.local_key);
    input.local_value = half_to_float(trace.local_value);
    input.compressor_kv = half_to_float(trace.compressor_kv);
    input.compressor_score = half_to_float(trace.compressor_score);
    input.indexer_query = half_to_float(trace.indexer_query);
    input.indexer_weights = half_to_float(trace.indexer_weights);
    input.indexer_compressor_kv = half_to_float(trace.indexer_compressor_kv);
    input.indexer_compressor_score = half_to_float(trace.indexer_compressor_score);
    return input;
}

void compare_local_state(
    const V4AttentionTraceRecord& trace,
    const V4OracleStateSnapshot& expected,
    uint32_t layer_id
) {
    if (trace.local_positions.size() != expected.local_cache.size()) {
        throw std::runtime_error("Stage 4 local cache capacity mismatch");
    }
    for (size_t slot = 0; slot < expected.local_cache.size(); ++slot) {
        const auto& expected_entry = expected.local_cache[slot];
        if (trace.local_positions[slot] != expected_entry.position) {
            throw std::runtime_error("Stage 4 local cache position mismatch at layer " +
                                     std::to_string(layer_id));
        }
        if (expected_entry.position < 0) continue;
        const size_t offset = slot * expected_entry.key.size();
        require_close(
            max_abs_difference(
                std::span<const half>(trace.local_key_cache.data() + offset, expected_entry.key.size()),
                expected_entry.key),
            kAttentionTolerance, "local key", trace.position, layer_id);
        require_close(
            max_abs_difference(
                std::span<const half>(trace.local_value_cache.data() + offset, expected_entry.value.size()),
                expected_entry.value),
            kAttentionTolerance, "local value", trace.position, layer_id);
    }
}

void compare_partial_state(
    std::span<const float> actual_kv,
    std::span<const float> actual_score,
    std::span<const int64_t> actual_positions,
    const std::vector<aeon::reference::V4OraclePartialRow>& expected,
    size_t width,
    float tolerance,
    const std::string& label,
    uint32_t position,
    uint32_t layer_id
) {
    for (const auto& row : expected) {
        const size_t slot = static_cast<size_t>(row.position % static_cast<int64_t>(actual_positions.size()));
        if (actual_positions[slot] != row.position) {
            throw std::runtime_error(label + " position mismatch at layer " +
                                     std::to_string(layer_id));
        }
        const size_t offset = slot * width;
        require_close(
            max_abs_difference(
                actual_kv.subspan(offset, width), row.kv),
            tolerance, label + " kv", position, layer_id);
        require_close(
            max_abs_difference(
                actual_score.subspan(offset, width), row.score),
            tolerance, label + " score", position, layer_id);
    }
}

void compare_compressed_state(
    const V4AttentionTraceRecord& trace,
    const V4OracleStateSnapshot& expected,
    uint32_t layer_id
) {
    for (size_t index = 0; index < expected.compressed_entries.size(); ++index) {
        const auto& expected_entry = expected.compressed_entries[index];
        if (trace.compressed_positions[index] != expected_entry.boundary_position) {
            throw std::runtime_error("Stage 4 compressed position mismatch at layer " +
                                     std::to_string(layer_id));
        }
        const size_t offset = index * expected_entry.key.size();
        require_close(
            max_abs_difference(
                std::span<const half>(trace.compressed_key_cache.data() + offset, expected_entry.key.size()),
                expected_entry.key),
            kAttentionTolerance, "compressed key", trace.position, layer_id);
        require_close(
            max_abs_difference(
                std::span<const half>(trace.compressed_value_cache.data() + offset, expected_entry.value.size()),
                expected_entry.value),
            kAttentionTolerance, "compressed value", trace.position, layer_id);
    }

    for (size_t index = 0; index < expected.indexer_entries.size(); ++index) {
        const auto& expected_entry = expected.indexer_entries[index];
        if (trace.indexer_positions[index] != expected_entry.boundary_position) {
            throw std::runtime_error("Stage 4 indexer position mismatch at layer " +
                                     std::to_string(layer_id));
        }
        const size_t offset = index * expected_entry.key.size();
        require_close(
            max_abs_difference(
                std::span<const half>(trace.indexer_key_cache.data() + offset, expected_entry.key.size()),
                expected_entry.key),
            kAttentionTolerance, "indexer key", trace.position, layer_id);
    }
}

void compare_grouped_projection(
    const AeonModelLoader& loader,
    const V4LayerSpec& layer_spec,
    const V4AttentionTraceRecord& trace
) {
    const std::string prefix = "layers." + std::to_string(layer_spec.layer_id) + ".";
    const auto& wo_a = require_tensor(
        loader, prefix + "attn.wo_a.weight", {8192, 4096}, "F16");
    const auto& wo_b = require_tensor(
        loader, prefix + "attn.wo_b.weight", {4096, 8192}, "F16");

    std::vector<half> z(8192, __float2half(0.0f));
    for (size_t group = 0; group < 8; ++group) {
        const size_t input_offset = group * 4096;
        const size_t weight_group_offset = group * 1024 * 4096;
        for (size_t output = 0; output < 1024; ++output) {
            float dot = 0.0f;
            const size_t weight_offset = weight_group_offset + output * 4096;
            for (size_t dimension = 0; dimension < 4096; ++dimension) {
                dot += __half2float(trace.inverse_rope_output[input_offset + dimension]) *
                    load_half_value(wo_a, weight_offset + dimension);
            }
            z[group * 1024 + output] = __float2half(dot);
        }
    }

    std::vector<half> expected(4096, __float2half(0.0f));
    for (size_t output = 0; output < 4096; ++output) {
        float dot = 0.0f;
        const size_t weight_offset = output * 8192;
        for (size_t dimension = 0; dimension < 8192; ++dimension) {
            dot += __half2float(z[dimension]) * load_half_value(wo_b, weight_offset + dimension);
        }
        expected[output] = __float2half(dot);
    }

    require_close(
        max_abs_difference(
            trace.grouped_output,
            half_to_float(expected)),
        kGroupedProjectionTolerance, "grouped output", trace.position, layer_spec.layer_id);
}

void compare_layer_trace(
    const AeonModelLoader& loader,
    const DeepSeekV4Config& model_config,
    const V4LayerSpec& layer_spec,
    const std::vector<V4AttentionTraceRecord>& traces
) {
    if (traces.size() != kTraceTokens) {
        throw std::runtime_error("Stage 4 trace did not capture all requested positions");
    }
    const V4OracleConfig oracle_config = make_oracle_config(model_config, layer_spec, loader);
    V4AttentionOracle oracle(oracle_config);
    const auto& first_layout = traces.front();

    for (const auto& trace : traces) {
        if (trace.position != static_cast<uint32_t>(&trace - traces.data())) {
            throw std::runtime_error("Stage 4 trace positions are not contiguous");
        }
        const V4OracleTokenInput input = make_oracle_input(trace);
        const auto result = oracle.append(input);
        const auto state = oracle.snapshot();

        if (trace.local_valid_count != result.local_valid_count ||
            trace.compressed_entry_count != result.compressed_entry_count ||
            trace.indexer_candidate_count != result.indexer_candidate_count) {
            throw std::runtime_error("Stage 4 trace state counts differ at layer " +
                                     std::to_string(layer_spec.layer_id));
        }

        require_close(
            max_abs_difference(trace.rotated_query, result.rotated_query),
            kAttentionTolerance, "rotated query", trace.position, layer_spec.layer_id);
        require_close(
            max_abs_difference(trace.attention_output, result.attention_output),
            kAttentionTolerance, "attention output", trace.position, layer_spec.layer_id);
        require_close(
            max_abs_difference(trace.inverse_rope_output, result.inverse_rope_output),
            kAttentionTolerance, "inverse RoPE output", trace.position, layer_spec.layer_id);

        compare_local_state(trace, state, layer_spec.layer_id);
        if (layer_spec.attention_kind != V4AttentionKind::Sliding) {
            compare_partial_state(
                trace.compressor_partial_kv,
                trace.compressor_partial_score,
                trace.compressor_partial_positions,
                state.compressor_partial,
                static_cast<size_t>(layer_spec.compression_ratio == 4 ? 2 : 1) *
                    static_cast<size_t>(model_config.head_dim),
                kStateTolerance,
                "compressor partial",
                trace.position,
                layer_spec.layer_id);
            compare_compressed_state(trace, state, layer_spec.layer_id);

            if (layer_spec.attention_kind == V4AttentionKind::CSA) {
                compare_partial_state(
                    trace.indexer_partial_kv,
                    trace.indexer_partial_score,
                    trace.indexer_partial_positions,
                    state.indexer_partial,
                    static_cast<size_t>(2 * model_config.index_head_dim),
                    kStateTolerance,
                    "indexer partial",
                    trace.position,
                    layer_spec.layer_id);
                require_close(
                    max_abs_difference(trace.indexer_scores, result.indexer_scores),
                    kIndexerScoreTolerance,
                    "indexer scores",
                    trace.position,
                    layer_spec.layer_id);
                if (trace.indexer_topk_indices != result.topk_indices) {
                    throw std::runtime_error("Stage 4 indexer top-k mismatch at layer " +
                                             std::to_string(layer_spec.layer_id) +
                                             ", position " + std::to_string(trace.position));
                }
            }
        }

        if (trace.position == kTraceTokens - 1) {
            compare_grouped_projection(loader, layer_spec, trace);
        }
    }

    (void)first_layout;
    std::cout << "[PASS] Stage 4 HIP/oracle trace layer " << layer_spec.layer_id << " "
              << aeon::core::v4_attention_kind_name(layer_spec.attention_kind)
              << " through position " << (kTraceTokens - 1) << std::endl;
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);

    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    const auto model_config = DeepSeekV4Config::load_from_json(model_dir + "/config.json");
    const auto layer_specs = aeon::core::V4ModelSpec::resolve_layers(model_config);

    aeon::core::AeonRuntimeConfig runtime_config;
    runtime_config.context_size = kTraceTokens;
    runtime_config.warm_host_bytes = 0;

    aeon::core::V4Pipeline pipeline;
    pipeline.initialize(model_dir, runtime_config);
    aeon::core::RoutingPhase phase = aeon::core::RoutingPhase::Prefill;

    for (const uint32_t layer_id : {0u, 2u, 3u}) {
        pipeline.enable_attention_trace(layer_id, kTraceTokens);
        pipeline.reset_generation_state();
        for (uint32_t position = 0; position < kTraceTokens; ++position) {
            const uint32_t next_token = pipeline.step(1u + position, position, phase);
            assert(next_token < 129280u);
        }
        compare_layer_trace(
            pipeline.aeon_loader,
            model_config,
            layer_specs.at(layer_id),
            pipeline.attention_trace());
    }

    std::cout << "V4 Stage 4 HIP/oracle traces passed: layers 0/2/3, cache state, "
              << "C4/C128 entries, indexer top-k, attention outputs, and grouped projection" << std::endl;
    return 0;
}