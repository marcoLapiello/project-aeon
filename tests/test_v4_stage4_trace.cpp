#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_model_contract.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline.hpp"
#include "architecture/deepseek_v4/kernels/hc_sinkhorn.hpp"
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
constexpr float kFp16RelativeTolerance = 2.0e-3f;

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

std::vector<uint16_t> half_to_bits(std::span<const half> values) {
    std::vector<uint16_t> bits(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        std::memcpy(&bits[index], &values[index], sizeof(uint16_t));
    }
    return bits;
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

float fp16_tensor_tolerance(std::span<const float> expected) {
    float scale = 1.0f;
    for (const float value : expected) scale = std::max(scale, std::abs(value));
    return kAttentionTolerance + kFp16RelativeTolerance * scale;
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
            fp16_tensor_tolerance(expected_entry.key), "local key", trace.position, layer_id);
        require_close(
            max_abs_difference(
                std::span<const half>(trace.local_value_cache.data() + offset, expected_entry.value.size()),
                expected_entry.value),
            fp16_tensor_tolerance(expected_entry.value), "local value", trace.position, layer_id);
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
            fp16_tensor_tolerance(expected_entry.key), "compressed key", trace.position, layer_id);
        require_close(
            max_abs_difference(
                std::span<const half>(trace.compressed_value_cache.data() + offset, expected_entry.value.size()),
                expected_entry.value),
            fp16_tensor_tolerance(expected_entry.value), "compressed value", trace.position, layer_id);
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
            fp16_tensor_tolerance(expected_entry.key), "indexer key", trace.position, layer_id);
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
        fp16_tensor_tolerance(half_to_float(expected)),
        "grouped output", trace.position, layer_spec.layer_id);
}

void compare_layer_zero_attention_input(
    const AeonModelLoader& loader,
    const DeepSeekV4Config& model_config,
    const V4AttentionTraceRecord& trace
) {
    if (trace.position != 0 || trace.attention_normalized_input.empty()) return;

    constexpr int kHcStreams = 4;
    constexpr int kHiddenSize = 4096;
    const auto& embedding = require_tensor(
        loader, "embed.weight",
        {model_config.vocab_size, kHiddenSize}, "F16");
    const auto& hc_fn = require_tensor(
        loader, "layers.0.hc_attn_fn",
        {kHcStreams * (2 + kHcStreams), kHcStreams * kHiddenSize}, "F32");
    const auto& hc_base = require_tensor(
        loader, "layers.0.hc_attn_base",
        {kHcStreams * (2 + kHcStreams)}, "F32");
    const auto& hc_scale = require_tensor(
        loader, "layers.0.hc_attn_scale", {3}, "F32");
    const auto& attention_norm = require_tensor(
        loader, "layers.0.attn_norm.weight", {kHiddenSize}, "F16");

    std::vector<float> residual(kHcStreams * kHiddenSize);
    for (int stream = 0; stream < kHcStreams; ++stream) {
        for (int hidden = 0; hidden < kHiddenSize; ++hidden) {
            const size_t embedding_index =
                static_cast<size_t>(trace.token_id) * kHiddenSize + static_cast<size_t>(hidden);
            residual[static_cast<size_t>(stream) * kHiddenSize + hidden] =
                load_half_value(embedding, embedding_index);
        }
    }

    std::vector<float> expected_mixes(kHcStreams * (2 + kHcStreams));
    if (trace.token_id >= static_cast<uint32_t>(model_config.vocab_size)) {
        throw std::runtime_error("Stage 4 trace token ID exceeds model vocabulary");
    }
    const float* fn_values = reinterpret_cast<const float*>(hc_fn.data);
    const float* base_values = reinterpret_cast<const float*>(hc_base.data);
    const float* scale_values = reinterpret_cast<const float*>(hc_scale.data);
    float squared_residual = 0.0f;
    for (float value : residual) squared_residual += value * value;
    const float residual_inverse_rms = 1.0f / std::sqrt(
        squared_residual / static_cast<float>(residual.size()) + 1e-6f);
    for (size_t mix = 0; mix < expected_mixes.size(); ++mix) {
        float dot = 0.0f;
        for (size_t index = 0; index < residual.size(); ++index) {
            dot += residual[index] * fn_values[mix * residual.size() + index];
        }
        expected_mixes[mix] = dot * residual_inverse_rms;
    }
    std::vector<float> expected_pre_mix(kHcStreams);
    for (int stream = 0; stream < kHcStreams; ++stream) {
        expected_pre_mix[static_cast<size_t>(stream)] = 1.0f / (1.0f + std::exp(-(
            expected_mixes[static_cast<size_t>(stream)] * scale_values[0] +
            base_values[stream]))) + 1e-6f;
    }

    float squared_sum = 0.0f;
    std::vector<float> expected(kHiddenSize);
    std::vector<float> expected_precombined(kHiddenSize);
    for (int hidden = 0; hidden < kHiddenSize; ++hidden) {
        float value = 0.0f;
        for (int stream = 0; stream < kHcStreams; ++stream) {
            value += expected_pre_mix[static_cast<size_t>(stream)] *
                residual[static_cast<size_t>(stream) * kHiddenSize + hidden];
        }
        expected_precombined[static_cast<size_t>(hidden)] = __half2float(__float2half(value));
        squared_sum += expected_precombined[static_cast<size_t>(hidden)] *
            expected_precombined[static_cast<size_t>(hidden)];
    }
    const float inverse_rms = 1.0f / std::sqrt(
        squared_sum / static_cast<float>(kHiddenSize) + 1e-6f);
    for (int hidden = 0; hidden < kHiddenSize; ++hidden) {
        expected[hidden] = expected_precombined[static_cast<size_t>(hidden)] *
            inverse_rms * load_half_value(
            attention_norm, static_cast<size_t>(hidden));
    }

    require_close(
        max_abs_difference(trace.attention_hc_mixes, expected_mixes),
        kStateTolerance,
        "layer 0 HC mixes",
        trace.position,
        0);
    require_close(
        max_abs_difference(trace.attention_hc_pre_mix, expected_pre_mix),
        kStateTolerance,
        "layer 0 HC pre mix",
        trace.position,
        0);
    require_close(
        max_abs_difference(trace.attention_precombined_input, expected_precombined),
        fp16_tensor_tolerance(expected_precombined),
        "layer 0 HC precombined input",
        trace.position,
        0);
    require_close(
        max_abs_difference(trace.attention_normalized_input, expected),
        fp16_tensor_tolerance(expected),
        "layer 0 HC attention input",
        trace.position,
        0);
}

void compare_layer_attention_input_from_trace(
    const AeonModelLoader& loader,
    const V4LayerSpec& layer_spec,
    const V4AttentionTraceRecord& trace
) {
    if (trace.position != 0 || trace.block_residual_input.empty()) return;

    constexpr int kHcStreams = 4;
    constexpr int kHiddenSize = 4096;
    const std::string prefix = "layers." + std::to_string(layer_spec.layer_id) + ".";
    const auto& hc_fn = require_tensor(
        loader, prefix + "hc_attn_fn",
        {kHcStreams * (2 + kHcStreams), kHcStreams * kHiddenSize}, "F32");
    const auto& hc_base = require_tensor(
        loader, prefix + "hc_attn_base",
        {kHcStreams * (2 + kHcStreams)}, "F32");
    const auto& hc_scale = require_tensor(
        loader, prefix + "hc_attn_scale", {3}, "F32");
    const auto& attention_norm = require_tensor(
        loader, prefix + "attn_norm.weight", {kHiddenSize}, "F16");

    const std::vector<float>& residual = trace.block_residual_input;
    const float* fn_values = reinterpret_cast<const float*>(hc_fn.data);
    const float* base_values = reinterpret_cast<const float*>(hc_base.data);
    const float* scale_values = reinterpret_cast<const float*>(hc_scale.data);
    std::vector<float> expected_mixes(kHcStreams * (2 + kHcStreams));
    float squared_residual = 0.0f;
    for (float value : residual) squared_residual += value * value;
    const float residual_inverse_rms = 1.0f / std::sqrt(
        squared_residual / static_cast<float>(residual.size()) + 1e-6f);
    for (size_t mix = 0; mix < expected_mixes.size(); ++mix) {
        float dot = 0.0f;
        for (size_t index = 0; index < residual.size(); ++index) {
            dot += residual[index] * fn_values[mix * residual.size() + index];
        }
        expected_mixes[mix] = dot * residual_inverse_rms;
    }

    std::vector<float> expected_pre_mix(kHcStreams);
    for (int stream = 0; stream < kHcStreams; ++stream) {
        expected_pre_mix[static_cast<size_t>(stream)] = 1.0f / (1.0f + std::exp(-(
            expected_mixes[static_cast<size_t>(stream)] * scale_values[0] +
            base_values[stream]))) + 1e-6f;
    }

    std::vector<float> expected_precombined(kHiddenSize);
    for (int hidden = 0; hidden < kHiddenSize; ++hidden) {
        float value = 0.0f;
        for (int stream = 0; stream < kHcStreams; ++stream) {
            value += trace.attention_hc_pre_mix[static_cast<size_t>(stream)] *
                residual[static_cast<size_t>(stream) * kHiddenSize + hidden];
        }
        expected_precombined[hidden] = __half2float(__float2half(value));
    }
    float squared_sum = 0.0f;
    for (float value : expected_precombined) squared_sum += value * value;
    const float inverse_rms = 1.0f / std::sqrt(
        squared_sum / static_cast<float>(kHiddenSize) + 1e-6f);
    std::vector<float> expected_norm(kHiddenSize);
    for (int hidden = 0; hidden < kHiddenSize; ++hidden) {
        expected_norm[hidden] = expected_precombined[hidden] * inverse_rms *
            load_half_value(attention_norm, static_cast<size_t>(hidden));
    }

    const float expected_pre_mix_error = [&]() {
        float maximum = 0.0f;
        for (int stream = 0; stream < kHcStreams; ++stream) {
            const float expected_pre_mix = 1.0f / (1.0f + std::exp(-(
                expected_mixes[stream] * scale_values[0] + base_values[stream]))) + 1e-6f;
            maximum = std::max(
                maximum,
                std::abs(trace.attention_hc_pre_mix[static_cast<size_t>(stream)] -
                         expected_pre_mix));
        }
        return maximum;
    }();
    if (max_abs_difference(trace.attention_precombined_input, expected_precombined) >
            kAttentionTolerance) {
        std::vector<float> production_precombined(kHiddenSize);
        for (int hidden = 0; hidden < kHiddenSize; ++hidden) {
            float value = 0.0f;
            for (int stream = 0; stream < kHcStreams; ++stream) {
                value += trace.attention_hc_pre_mix[static_cast<size_t>(stream)] *
                    residual[static_cast<size_t>(stream) * kHiddenSize + hidden];
            }
            production_precombined[static_cast<size_t>(hidden)] =
                __half2float(__float2half(value));
        }
        std::cerr << "HC diagnostic layer " << layer_spec.layer_id
                  << " position " << trace.position
                  << " mix_error=" << max_abs_difference(
                      trace.attention_hc_mixes, expected_mixes)
                  << " pre_mix_error=" << expected_pre_mix_error
                  << " precombined_with_production_pre_mix_error="
                  << max_abs_difference(
                      trace.attention_precombined_input, production_precombined)
                  << std::endl;
        for (int stream = 0; stream < kHcStreams; ++stream) {
            const float expected_pre_mix = 1.0f / (1.0f + std::exp(-(
                expected_mixes[stream] * scale_values[0] + base_values[stream]))) + 1e-6f;
            std::cerr << "  stream " << stream
                      << " production_pre=" << trace.attention_hc_pre_mix[static_cast<size_t>(stream)]
                      << " expected_pre=" << expected_pre_mix << std::endl;
        }
        size_t maximum_index = 0;
        float maximum_error = 0.0f;
        for (size_t index = 0; index < production_precombined.size(); ++index) {
            const float error = std::abs(
                __half2float(trace.attention_precombined_input[index]) -
                production_precombined[index]);
            if (error > maximum_error) {
                maximum_error = error;
                maximum_index = index;
            }
        }
        std::cerr << "  max_precombined_index=" << maximum_index
                  << " actual=" << __half2float(trace.attention_precombined_input[maximum_index])
                  << " expected=" << production_precombined[maximum_index] << std::endl;
        for (int stream = 0; stream < kHcStreams; ++stream) {
            std::cerr << "  residual[" << stream << "]="
                      << residual[static_cast<size_t>(stream) * kHiddenSize + maximum_index]
                      << std::endl;
        }
    }

    require_close(
        max_abs_difference(trace.attention_hc_mixes, expected_mixes),
        kStateTolerance,
        "HC attention mixes",
        trace.position,
        layer_spec.layer_id);
    require_close(
        max_abs_difference(trace.attention_hc_pre_mix, expected_pre_mix),
        kStateTolerance,
        "HC attention pre mix",
        trace.position,
        layer_spec.layer_id);
    require_close(
        max_abs_difference(trace.attention_precombined_input, expected_precombined),
        fp16_tensor_tolerance(expected_precombined),
        "HC attention precombined input",
        trace.position,
        layer_spec.layer_id);
    require_close(
        max_abs_difference(trace.attention_normalized_input, expected_norm),
        fp16_tensor_tolerance(expected_norm),
        "HC attention normalized input",
        trace.position,
        layer_spec.layer_id);
}

void compare_layer_post_attention(
    const AeonModelLoader& loader,
    const DeepSeekV4Config& model_config,
    const V4LayerSpec& layer_spec,
    const V4AttentionTraceRecord& trace
) {
    if (trace.position != 0 || trace.attention_post_residual.empty()) return;

    constexpr int kHcStreams = 4;
    constexpr int kHiddenSize = 4096;
    const std::string prefix = "layers." + std::to_string(layer_spec.layer_id) + ".";
    const auto& ffn_fn = require_tensor(
        loader, prefix + "hc_ffn_fn",
        {kHcStreams * (2 + kHcStreams), kHcStreams * kHiddenSize}, "F32");
    const auto& ffn_base = require_tensor(
        loader, prefix + "hc_ffn_base",
        {kHcStreams * (2 + kHcStreams)}, "F32");
    const auto& ffn_scale = require_tensor(
        loader, prefix + "hc_ffn_scale", {3}, "F32");
    const auto& ffn_norm = require_tensor(
        loader, prefix + "ffn_norm.weight", {kHiddenSize}, "F16");

    std::vector<float> residual(trace.block_residual_input.size());
    for (size_t index = 0; index < residual.size(); ++index) {
        residual[index] = __half2float(__float2half(trace.block_residual_input[index]));
    }

    std::vector<float> attention_output = half_to_float(trace.grouped_output);
    std::vector<float> expected_residual(kHcStreams * kHiddenSize);
    aeon::kernel::cpu_hc_post(
        attention_output.data(),
        residual.data(),
        trace.attention_hc_post_mix.data(),
        trace.attention_hc_comb_mix.data(),
        expected_residual.data(),
        kHiddenSize,
        kHcStreams);
    std::vector<float> expected_residual_half(expected_residual.size());
    for (size_t index = 0; index < expected_residual.size(); ++index) {
        expected_residual_half[index] = __half2float(__float2half(expected_residual[index]));
    }
    require_close(
        max_abs_difference(trace.attention_post_residual, expected_residual_half),
        fp16_tensor_tolerance(expected_residual_half),
        "HC attention post residual",
        trace.position,
        layer_spec.layer_id);

    const std::vector<float> actual_residual = half_to_float(trace.attention_post_residual);
    std::vector<float> expected_ffn_input(kHiddenSize);
    std::vector<float> unused_post_mix(kHcStreams);
    std::vector<float> unused_comb_mix(kHcStreams * kHcStreams);
    aeon::kernel::cpu_sinkhorn_and_mix(
        actual_residual.data(),
        reinterpret_cast<const float*>(ffn_fn.data),
        reinterpret_cast<const float*>(ffn_base.data),
        reinterpret_cast<const float*>(ffn_scale.data),
        expected_ffn_input.data(),
        unused_post_mix.data(),
        unused_comb_mix.data(),
        kHiddenSize,
        kHcStreams,
        1e-6f,
        1e-6f,
        1e-6f,
        2.0f,
        20);
    std::vector<float> expected_ffn_input_half(kHiddenSize);
    for (size_t index = 0; index < expected_ffn_input_half.size(); ++index) {
        expected_ffn_input_half[index] = __half2float(__float2half(expected_ffn_input[index]));
    }
    require_close(
        max_abs_difference(trace.ffn_precombined_input, expected_ffn_input_half),
        fp16_tensor_tolerance(expected_ffn_input_half),
        "HC FFN input",
        trace.position,
        layer_spec.layer_id);

    float squared_sum = 0.0f;
    for (float value : expected_ffn_input_half) squared_sum += value * value;
    const float inverse_rms = 1.0f / std::sqrt(
        squared_sum / static_cast<float>(kHiddenSize) + 1e-6f);
    std::vector<float> expected_ffn_norm(kHiddenSize);
    for (int hidden = 0; hidden < kHiddenSize; ++hidden) {
        expected_ffn_norm[hidden] = expected_ffn_input_half[hidden] * inverse_rms *
            load_half_value(ffn_norm, static_cast<size_t>(hidden));
    }
    require_close(
        max_abs_difference(trace.ffn_normalized_input, expected_ffn_norm),
        fp16_tensor_tolerance(expected_ffn_norm),
        "FFN normalized input",
        trace.position,
        layer_spec.layer_id);
}

void compare_layer_moe(
    const AeonModelLoader& loader,
    const V4LayerSpec& layer_spec,
    const V4AttentionTraceRecord& trace
) {
    if (trace.position != 0 || trace.moe_output.empty()) return;

    constexpr int kHiddenSize = 4096;
    constexpr int kIntermediateSize = 2048;
    constexpr int kRouterExperts = 256;
    const std::string prefix = "layers." + std::to_string(layer_spec.layer_id) + ".";
    const std::vector<uint16_t> input_bits = half_to_bits(trace.ffn_normalized_input);
    const std::vector<float> input_values = half_to_float(trace.ffn_normalized_input);

    const auto& router = require_tensor(
        loader, prefix + "ffn.gate.weight", {kRouterExperts, kHiddenSize}, "F16");
    std::vector<float> expected_router(kRouterExperts);
    aeon::reference::decode_fp16_gemv(
        router.data, input_bits.data(), kRouterExperts, kHiddenSize, expected_router.data());
    require_close(
        max_abs_difference(trace.router_logits, expected_router),
        0.1f,
        "router logits",
        trace.position,
        layer_spec.layer_id);

    const auto& shared_w1 = require_tensor(
        loader, prefix + "ffn.shared_experts.w1.weight",
        {kIntermediateSize, kHiddenSize}, "F16");
    const auto& shared_w3 = require_tensor(
        loader, prefix + "ffn.shared_experts.w3.weight",
        {kIntermediateSize, kHiddenSize}, "F16");
    const auto& shared_w2 = require_tensor(
        loader, prefix + "ffn.shared_experts.w2.weight",
        {kHiddenSize, kIntermediateSize}, "F16");
    std::vector<float> shared_gate(kIntermediateSize);
    std::vector<float> shared_up(kIntermediateSize);
    aeon::reference::decode_fp16_gemv(
        shared_w1.data, input_bits.data(), kIntermediateSize, kHiddenSize, shared_gate.data());
    aeon::reference::decode_fp16_gemv(
        shared_w3.data, input_bits.data(), kIntermediateSize, kHiddenSize, shared_up.data());
    std::vector<uint16_t> shared_hidden_bits(kIntermediateSize);
    for (int index = 0; index < kIntermediateSize; ++index) {
        const float gate = std::min(shared_gate[index], 10.0f);
        const float up = std::min(std::max(shared_up[index], -10.0f), 10.0f);
        const half hidden = __float2half((gate / (1.0f + std::exp(-gate))) * up);
        std::memcpy(&shared_hidden_bits[index], &hidden, sizeof(uint16_t));
    }
    std::vector<float> expected_shared(kHiddenSize);
    aeon::reference::decode_fp16_gemv(
        shared_w2.data, shared_hidden_bits.data(), kHiddenSize, kIntermediateSize,
        expected_shared.data());
    require_close(
        max_abs_difference(trace.shared_expert_output, expected_shared),
        fp16_tensor_tolerance(expected_shared),
        "shared expert output",
        trace.position,
        layer_spec.layer_id);

    std::vector<float> expected_moe = expected_shared;
    for (size_t expert_index = 0; expert_index < trace.routed_expert_indices.size(); ++expert_index) {
        const uint8_t* payload = loader.get_expert_data(
            static_cast<uint32_t>(layer_spec.layer_id),
            static_cast<uint32_t>(trace.routed_expert_indices[expert_index]));
        std::vector<float> expert_output(kHiddenSize);
        aeon::reference::decode_routed_ffn(
            payload, input_values.data(), expert_output.data(), 10.0f);
        for (int hidden = 0; hidden < kHiddenSize; ++hidden) {
            expected_moe[hidden] += trace.routed_expert_weights[expert_index] * expert_output[hidden];
        }
    }
    require_close(
        max_abs_difference(trace.moe_output, expected_moe),
        fp16_tensor_tolerance(expected_moe),
        "combined MoE output",
        trace.position,
        layer_spec.layer_id);

    const std::vector<float> residual_mid = half_to_float(trace.attention_post_residual);
    const std::vector<float> actual_moe = half_to_float(trace.moe_output);
    std::vector<float> expected_residual(residual_mid.size());
    aeon::kernel::cpu_hc_post(
        actual_moe.data(),
        residual_mid.data(),
        trace.ffn_hc_post_mix.data(),
        trace.ffn_hc_comb_mix.data(),
        expected_residual.data(),
        kHiddenSize,
        4);
    std::vector<float> expected_residual_half(expected_residual.size());
    for (size_t index = 0; index < expected_residual.size(); ++index) {
        expected_residual_half[index] = __half2float(__float2half(expected_residual[index]));
    }
    require_close(
        max_abs_difference(trace.post_ffn_residual, expected_residual_half),
        fp16_tensor_tolerance(expected_residual_half),
        "HC FFN post residual",
        trace.position,
        layer_spec.layer_id);
}

void compare_layer_trace(
    const AeonModelLoader& loader,
    const DeepSeekV4Config& model_config,
    const V4LayerSpec& layer_spec,
    const std::vector<V4AttentionTraceRecord>& traces,
    size_t expected_trace_tokens = kTraceTokens
) {
    if (traces.size() != expected_trace_tokens) {
        throw std::runtime_error("Stage 4 trace did not capture all requested positions");
    }
    const V4OracleConfig oracle_config = make_oracle_config(model_config, layer_spec, loader);
    V4AttentionOracle oracle(oracle_config);

    for (size_t trace_index = 0; trace_index < traces.size(); ++trace_index) {
        const auto& trace = traces[trace_index];
        if (trace.position != trace_index) {
            throw std::runtime_error("Stage 4 trace positions are not contiguous");
        }
        const V4OracleTokenInput input = make_oracle_input(trace);
        const auto result = oracle.append(input);
        const auto state = oracle.snapshot();

        if (layer_spec.layer_id == 0) {
            compare_layer_zero_attention_input(loader, model_config, trace);
        } else {
            compare_layer_attention_input_from_trace(loader, layer_spec, trace);
        }
        compare_layer_post_attention(loader, model_config, layer_spec, trace);
        compare_layer_moe(loader, layer_spec, trace);

        if (trace.local_valid_count != result.local_valid_count ||
            trace.compressed_entry_count != result.compressed_entry_count ||
            trace.indexer_candidate_count != result.indexer_candidate_count) {
            throw std::runtime_error("Stage 4 trace state counts differ at layer " +
                                     std::to_string(layer_spec.layer_id));
        }

        require_close(
            max_abs_difference(trace.rotated_query, result.rotated_query),
            fp16_tensor_tolerance(result.rotated_query),
            "rotated query", trace.position, layer_spec.layer_id);
        require_close(
            max_abs_difference(trace.attention_output, result.attention_output),
            fp16_tensor_tolerance(result.attention_output),
            "attention output", trace.position, layer_spec.layer_id);
        require_close(
            max_abs_difference(trace.inverse_rope_output, result.inverse_rope_output),
            fp16_tensor_tolerance(result.inverse_rope_output),
            "inverse RoPE output", trace.position, layer_spec.layer_id);

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

        if (trace.position == expected_trace_tokens - 1) {
            compare_grouped_projection(loader, layer_spec, trace);
        }
    }

    std::cout << "[PASS] Stage 4 HIP/oracle trace layer " << layer_spec.layer_id << " "
              << aeon::core::v4_attention_kind_name(layer_spec.attention_kind)
              << " through position " << (expected_trace_tokens - 1) << std::endl;
}

void compare_all_layer_boundary_state(const V4LayerSpec& layer_spec,
                                      const V4AttentionTraceRecord& trace) {
    constexpr uint32_t kLocalCapacity = 128;
    constexpr uint32_t kIndexerTopK = 512;
    const uint32_t position = trace.position;
    const uint32_t expected_local_count = std::min(position + 1u, kLocalCapacity);
    if (trace.local_valid_count != expected_local_count) {
        throw std::runtime_error(
            "All-layer boundary local count mismatch at layer " +
            std::to_string(layer_spec.layer_id) + ", position " +
            std::to_string(position));
    }
    if (trace.local_positions.size() != kLocalCapacity) {
        throw std::runtime_error("All-layer boundary local cache capacity mismatch");
    }
    for (uint32_t slot = 0; slot < kLocalCapacity; ++slot) {
        int64_t expected_position = -1;
        if (slot <= position || position >= kLocalCapacity) {
            const uint32_t distance = (position % kLocalCapacity + kLocalCapacity - slot) %
                kLocalCapacity;
            if (distance <= position) {
                expected_position = static_cast<int64_t>(position - distance);
            }
        }
        if (trace.local_positions[slot] != expected_position) {
            throw std::runtime_error(
                "All-layer boundary local ring mismatch at layer " +
                std::to_string(layer_spec.layer_id) + ", position " +
                std::to_string(position) + ", slot " + std::to_string(slot));
        }
    }

    const uint32_t ratio = static_cast<uint32_t>(layer_spec.compression_ratio);
    const uint32_t expected_compressed_count = layer_spec.attention_kind == V4AttentionKind::Sliding
        ? 0u
        : (position + 1u) / ratio;
    if (trace.compressed_entry_count != expected_compressed_count) {
        throw std::runtime_error(
            "All-layer boundary compressed count mismatch at layer " +
            std::to_string(layer_spec.layer_id) + ", position " +
            std::to_string(position));
    }
    for (uint32_t index = 0; index < expected_compressed_count; ++index) {
        const int64_t expected_boundary = static_cast<int64_t>((index + 1u) * ratio - 1u);
        if (trace.compressed_positions[index] != expected_boundary) {
            throw std::runtime_error(
                "All-layer boundary compressed position mismatch at layer " +
                std::to_string(layer_spec.layer_id) + ", position " +
                std::to_string(position));
        }
    }

    if (layer_spec.attention_kind == V4AttentionKind::CSA) {
        if (trace.indexer_candidate_count != expected_compressed_count ||
            trace.indexer_topk_indices.size() != kIndexerTopK) {
            throw std::runtime_error(
                "All-layer boundary CSA candidate contract mismatch at layer " +
                std::to_string(layer_spec.layer_id) + ", position " +
                std::to_string(position));
        }
        for (uint32_t index = 0; index < kIndexerTopK; ++index) {
            const int32_t expected_index = index < expected_compressed_count
                ? static_cast<int32_t>(index)
                : -1;
            if (trace.indexer_topk_indices[index] != expected_index) {
                throw std::runtime_error(
                    "All-layer boundary CSA top-k mismatch at layer " +
                    std::to_string(layer_spec.layer_id) + ", position " +
                    std::to_string(position));
            }
        }
    } else if (trace.indexer_candidate_count != 0) {
        throw std::runtime_error(
            "All-layer boundary non-CSA indexer state is non-empty at layer " +
            std::to_string(layer_spec.layer_id) + ", position " +
            std::to_string(position));
    }

    const auto require_finite = [](std::span<const half> values, const char* label,
                                   uint32_t layer_id, uint32_t position) {
        for (const half value : values) {
            if (!std::isfinite(__half2float(value))) {
                throw std::runtime_error(
                    std::string("All-layer boundary ") + label +
                    " contains a non-finite value at layer " + std::to_string(layer_id) +
                    ", position " + std::to_string(position));
            }
        }
    };
    require_finite(trace.attention_output, "attention output", layer_spec.layer_id, position);
    require_finite(trace.inverse_rope_output, "inverse RoPE output", layer_spec.layer_id, position);
}

void compare_full_block_residual_continuity(
    const std::vector<V4AttentionTraceRecord>& traces
) {
    for (size_t index = 1; index < traces.size(); ++index) {
        const auto& previous = traces[index - 1];
        const auto& current = traces[index];
        if (current.layer_id != previous.layer_id + 1 ||
            current.position != previous.position) {
            throw std::runtime_error("Full-block trace layers are not contiguous");
        }
        if (previous.post_ffn_residual.size() != current.block_residual_input.size()) {
            throw std::runtime_error(
                "Full-block residual continuity vector size mismatch between layers " +
                std::to_string(previous.layer_id) + " and " +
                std::to_string(current.layer_id));
        }
        std::vector<float> expected_input(previous.post_ffn_residual.size());
        for (size_t element = 0; element < expected_input.size(); ++element) {
            expected_input[element] = __half2float(previous.post_ffn_residual[element]);
        }
        require_close(
            max_abs_difference(
                std::span<const float>(current.block_residual_input),
                std::span<const float>(expected_input)),
            fp16_tensor_tolerance(expected_input),
            "full-block residual continuity",
            current.position,
            current.layer_id);
    }
}

} // namespace

int main(int argc, char** argv) {
    aeon::core::select_compute_device(true);

    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    const auto model_config = DeepSeekV4Config::load_from_json(model_dir + "/config.json");
    const auto layer_specs = aeon::core::V4ModelSpec::resolve_layers(model_config);

    aeon::core::AeonRuntimeConfig runtime_config;
    runtime_config.context_size = kTraceTokens;
    runtime_config.warm_host_bytes = 0;
    runtime_config.deterministic_expert_accumulation = true;

    aeon::core::V4Pipeline pipeline;
    pipeline.initialize(model_dir, runtime_config);
    aeon::core::RoutingPhase phase = aeon::core::RoutingPhase::Prefill;

    if (argc > 1) {
        const std::string mode = argv[1];
        std::vector<uint32_t> prelude_layers;
        const bool no_trace_prelude = mode == "no-trace-layer0-layer2-layer3";
        const bool synchronize_device = mode == "device-sync-layer0-layer2-layer3";
        const bool compare_all_layers = argc > 2 && std::string(argv[2]) == "compare-all";
        if (mode == "real-only") {
            const std::vector<uint32_t> real_prompt_ids = {
                0u, 128803u, 3085u, 344u, 223u, 20u, 223u, 13u, 223u, 20u,
                33u, 9361u, 418u, 1438u, 270u, 1167u, 16u, 128804u, 128822u};
            std::vector<uint32_t> real_prompt_positions;
            real_prompt_positions.reserve(real_prompt_ids.size());
            for (uint32_t position = 0; position < real_prompt_ids.size(); ++position) {
                real_prompt_positions.push_back(position);
            }
            pipeline.enable_attention_trace_all_layers(real_prompt_positions);
            pipeline.reset_generation_state();
            for (uint32_t position = 0; position < real_prompt_ids.size(); ++position) {
                const uint32_t next_token = pipeline.step(
                    real_prompt_ids[position], position, phase);
                assert(next_token < 129280u);
            }
            const auto& real_traces = pipeline.attention_trace();
            for (const auto& layer_spec : layer_specs) {
                std::vector<V4AttentionTraceRecord> layer_traces;
                layer_traces.reserve(real_prompt_ids.size());
                for (const auto& trace : real_traces) {
                    if (trace.layer_id == layer_spec.layer_id) layer_traces.push_back(trace);
                }
                compare_layer_trace(
                    pipeline.aeon_loader,
                    model_config,
                    layer_spec,
                    layer_traces,
                    real_prompt_ids.size());
            }
            std::cout << "[PASS] Stage 4 real-prompt isolation" << std::endl;
            return 0;
        }
        if (mode == "layer0") {
            prelude_layers = {0u};
        } else if (mode == "layer2") {
            prelude_layers = {2u};
        } else if (mode == "layer3") {
            prelude_layers = {3u};
        } else if (mode == "layer0-layer2") {
            prelude_layers = {0u, 2u};
        } else if (mode == "layer0-layer3") {
            prelude_layers = {0u, 3u};
        } else if (mode == "layer2-layer3") {
            prelude_layers = {2u, 3u};
        } else if (mode == "layer0-layer2-layer3") {
            prelude_layers = {0u, 2u, 3u};
        } else if (synchronize_device) {
            prelude_layers = {0u, 2u, 3u};
        } else if (!no_trace_prelude && mode != "fresh") {
            throw std::invalid_argument("Unknown Stage 4 isolation mode: " + mode);
        }

        for (const uint32_t layer_id : prelude_layers) {
            if (no_trace_prelude) pipeline.disable_attention_trace();
            else pipeline.enable_attention_trace(layer_id, kTraceTokens);
            pipeline.reset_generation_state();
            for (uint32_t position = 0; position < kTraceTokens; ++position) {
                const uint32_t next_token = pipeline.step(1u + position, position, phase);
                assert(next_token < 129280u);
                if (synchronize_device && hipDeviceSynchronize() != hipSuccess) {
                    throw std::runtime_error("Stage 4 isolation device synchronization failed");
                }
            }
            if (argc > 2 &&
                (std::string(argv[2]) == "compare" || compare_all_layers)) {
                compare_layer_trace(
                    pipeline.aeon_loader,
                    model_config,
                    layer_specs.at(layer_id),
                    pipeline.attention_trace());
            }
        }

        pipeline.enable_attention_trace_all_layers(0);
        pipeline.reset_generation_state();
        const uint32_t first_token = pipeline.step(1u, 0u, phase);
        assert(first_token < 129280u);
        if (synchronize_device && hipDeviceSynchronize() != hipSuccess) {
            throw std::runtime_error("Stage 4 isolation device synchronization failed");
        }
        const auto& isolation_traces = pipeline.attention_trace();
        const auto layer_zero = std::find_if(
            isolation_traces.begin(), isolation_traces.end(),
            [](const V4AttentionTraceRecord& trace) { return trace.layer_id == 0; });
        if (layer_zero == isolation_traces.end()) {
            throw std::runtime_error("Stage 4 isolation did not capture layer 0");
        }
        compare_layer_zero_attention_input(
            pipeline.aeon_loader, model_config, *layer_zero);
        if (compare_all_layers) {
            for (const auto& trace : isolation_traces) {
                const auto& layer_spec = layer_specs.at(trace.layer_id);
                if (trace.layer_id == 0) {
                    compare_layer_zero_attention_input(
                        pipeline.aeon_loader, model_config, trace);
                } else {
                    compare_layer_attention_input_from_trace(
                        pipeline.aeon_loader, layer_spec, trace);
                }
                compare_layer_post_attention(
                    pipeline.aeon_loader, model_config, layer_spec, trace);
                compare_layer_moe(pipeline.aeon_loader, layer_spec, trace);
            }
        }
        std::cout << "[PASS] Stage 4 reset isolation mode " << mode << std::endl;
        return 0;
    }

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

    pipeline.enable_attention_trace_all_layers(0);
    pipeline.reset_generation_state();
    const uint32_t first_token = pipeline.step(1u, 0u, phase);
    assert(first_token < 129280u);
    const auto& all_layer_traces = pipeline.attention_trace();
    if (all_layer_traces.size() != layer_specs.size()) {
        throw std::runtime_error(
            "All-layer block trace captured " + std::to_string(all_layer_traces.size()) +
            " records for " + std::to_string(layer_specs.size()) + " layers");
    }
    for (const auto& trace : all_layer_traces) {
        const auto& layer_spec = layer_specs.at(trace.layer_id);
        if (trace.position != 0 || trace.attention_kind != layer_spec.attention_kind) {
            throw std::runtime_error(
                "All-layer block trace contract mismatch at layer " +
                std::to_string(trace.layer_id));
        }
        if (trace.block_residual_input.empty() || trace.ffn_normalized_input.empty() ||
            trace.router_logits.empty() || trace.shared_expert_output.empty() ||
            trace.moe_output.empty() || trace.post_ffn_residual.empty()) {
            throw std::runtime_error(
                "All-layer block trace missing a full-block checkpoint at layer " +
                std::to_string(trace.layer_id));
        }
        if (trace.layer_id == 0) {
            compare_layer_zero_attention_input(pipeline.aeon_loader, model_config, trace);
        } else {
            compare_layer_attention_input_from_trace(
                pipeline.aeon_loader, layer_spec, trace);
        }
        compare_layer_post_attention(
            pipeline.aeon_loader, model_config, layer_spec, trace);
        compare_layer_moe(pipeline.aeon_loader, layer_spec, trace);
    }
    compare_full_block_residual_continuity(all_layer_traces);

    const std::vector<uint32_t> boundary_positions = {
        0u, 3u, 4u, 7u, 8u, 123u, 124u, 126u, 127u, 128u, 131u};
    pipeline.enable_attention_trace_all_layers(boundary_positions);
    pipeline.reset_generation_state();
    for (uint32_t position = 0; position < kTraceTokens; ++position) {
        const uint32_t next_token = pipeline.step(1u + position, position, phase);
        assert(next_token < 129280u);
    }
    const auto& boundary_traces = pipeline.attention_trace();
    if (boundary_traces.size() != layer_specs.size() * boundary_positions.size()) {
        throw std::runtime_error(
            "All-layer boundary trace captured " + std::to_string(boundary_traces.size()) +
            " records instead of " +
            std::to_string(layer_specs.size() * boundary_positions.size()));
    }
    size_t boundary_index = 0;
    for (const uint32_t position : boundary_positions) {
        for (const auto& layer_spec : layer_specs) {
            const auto& trace = boundary_traces[boundary_index++];
            if (trace.layer_id != layer_spec.layer_id || trace.position != position ||
                trace.attention_kind != layer_spec.attention_kind) {
                throw std::runtime_error("All-layer boundary trace ordering mismatch");
            }
            compare_all_layer_boundary_state(layer_spec, trace);
        }
    }

    const std::vector<uint32_t> real_prompt_ids = {
        0u, 128803u, 3085u, 344u, 223u, 20u, 223u, 13u, 223u, 20u,
        33u, 9361u, 418u, 1438u, 270u, 1167u, 16u, 128804u, 128822u};
    std::vector<uint32_t> real_prompt_positions;
    real_prompt_positions.reserve(real_prompt_ids.size());
    for (uint32_t position = 0; position < real_prompt_ids.size(); ++position) {
        real_prompt_positions.push_back(position);
    }
    pipeline.enable_attention_trace_all_layers(real_prompt_positions);
    pipeline.reset_generation_state();
    for (uint32_t position = 0; position < real_prompt_ids.size(); ++position) {
        const uint32_t next_token = pipeline.step(real_prompt_ids[position], position, phase);
        assert(next_token < 129280u);
    }
    const auto& real_prompt_traces = pipeline.attention_trace();
    if (real_prompt_traces.size() != layer_specs.size() * real_prompt_ids.size()) {
        throw std::runtime_error("Real prompt all-layer trace count mismatch");
    }
    for (const auto& layer_spec : layer_specs) {
        std::vector<V4AttentionTraceRecord> layer_traces;
        layer_traces.reserve(real_prompt_ids.size());
        for (const auto& trace : real_prompt_traces) {
            if (trace.layer_id == layer_spec.layer_id) layer_traces.push_back(trace);
        }
        compare_layer_trace(
            pipeline.aeon_loader,
            model_config,
            layer_spec,
            layer_traces,
            real_prompt_ids.size());
    }

    std::cout << "V4 Stage 4 HIP/oracle traces passed: layers 0/2/3, cache state, "
              << "C4/C128 entries, indexer top-k, attention outputs, grouped projection, "
              << "all 43 full-block compositions, all-layer attention boundaries, "
              << "and the real formatted prompt" << std::endl;
    return 0;
}