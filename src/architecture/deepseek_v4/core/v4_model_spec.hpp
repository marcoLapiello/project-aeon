#pragma once

#include "architecture/deepseek_v4/core/config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::core {

enum class V4AttentionKind : uint8_t {
    Sliding,
    CSA,
    HCA
};

inline const char* v4_attention_kind_name(V4AttentionKind kind) noexcept {
    switch (kind) {
        case V4AttentionKind::Sliding: return "Sliding";
        case V4AttentionKind::CSA: return "CSA";
        case V4AttentionKind::HCA: return "HCA";
    }
    return "Unknown";
}

struct V4LayerSpec {
    uint32_t layer_id{0};
    V4AttentionKind attention_kind{V4AttentionKind::Sliding};
    int32_t compression_ratio{0};

    bool uses_indexer() const noexcept {
        return attention_kind == V4AttentionKind::CSA;
    }
};

class V4ModelSpec {
public:
    static void validate_config(const DeepSeekV4Config& config) {
        const auto require_field = [&](const std::string& field) {
            if (!config.has_field(field)) {
                throw std::invalid_argument("V4ModelSpec: missing required config field " + field);
            }
        };
        require_field("compress_ratios");
        require_field("index_head_dim");
        require_field("index_n_heads");
        require_field("index_topk");
        require_field("compress_rope_theta");
        require_field("o_groups");
        require_field("rope_scaling.type");
        require_field("rope_scaling.factor");
        require_field("rope_scaling.beta_fast");
        require_field("rope_scaling.beta_slow");
        require_field("rope_scaling.original_max_position_embeddings");
        require_field("quantization_config");
        require_field("quantization_config.config_groups.group_0.weights");

        const auto expect = [](bool condition, const std::string& message) {
            if (!condition) throw std::invalid_argument("V4ModelSpec: " + message);
        };
        const auto close = [](float left, float right) {
            return std::fabs(left - right) <= 1e-6f * std::max(1.0f, std::fabs(right));
        };

        expect(config.model_type == "deepseek_v4", "model_type is unsupported");
        expect(config.architectures.size() == 1 && config.architectures.front() == "DeepseekV4ForCausalLM",
               "architectures do not describe the selected base decoder");
        expect(config.vocab_size == 129280, "vocab_size must be 129280");
        expect(config.hidden_size == 4096, "hidden_size must be 4096");
        expect(config.moe_intermediate_size == 2048, "moe_intermediate_size must be 2048");
        expect(config.num_hidden_layers == 43, "num_hidden_layers must be 43");
        expect(config.num_attention_heads == 64, "num_attention_heads must be 64");
        expect(config.num_key_value_heads == 1, "num_key_value_heads must be 1");
        expect(config.head_dim == 512, "head_dim must be 512");
        expect(config.q_lora_rank == 1024, "q_lora_rank must be 1024");
        expect(config.o_lora_rank == 1024, "o_lora_rank must be 1024");
        expect(config.qk_rope_head_dim == 64, "qk_rope_head_dim must be 64");
        expect(config.sliding_window == 128, "sliding_window must be 128");
        expect(config.index_head_dim == 128, "index_head_dim must be 128");
        expect(config.index_n_heads == 64, "index_n_heads must be 64");
        expect(config.index_topk == 512, "index_topk must be 512");
        expect(config.o_groups == 8, "o_groups must be 8");
        expect(close(config.rope_theta, 10000.0f), "rope_theta must be 10000");
        expect(close(config.compress_rope_theta, 160000.0f), "compress_rope_theta must be 160000");
        expect(config.rope_scaling.type == "yarn", "rope_scaling.type must be yarn");
        expect(close(config.rope_scaling.factor, 16.0f), "rope_scaling.factor must be 16");
        expect(close(config.rope_scaling.beta_fast, 32.0f), "rope_scaling.beta_fast must be 32");
        expect(close(config.rope_scaling.beta_slow, 1.0f), "rope_scaling.beta_slow must be 1");
        expect(config.rope_scaling.original_max_position_embeddings == 65536,
               "rope_scaling.original_max_position_embeddings must be 65536");
        expect(config.max_position_embeddings == 1048576, "max_position_embeddings must be 1048576");
        expect(config.original_max_position_embeddings == 65536,
               "original_max_position_embeddings must be 65536");
        expect(config.rope_factor == 16, "rope_factor must be 16");
        expect(close(config.rope_beta_fast, 32.0f), "rope_beta_fast must be 32");
        expect(close(config.rope_beta_slow, 1.0f), "rope_beta_slow must be 1");
        expect(config.n_routed_experts == 256, "n_routed_experts must be 256");
        expect(config.n_shared_experts == 1, "n_shared_experts must be 1");
        expect(config.num_experts_per_tok == 6, "num_experts_per_tok must be 6");
        expect(config.num_hash_layers == 3, "num_hash_layers must be 3");
        expect(close(config.routed_scaling_factor, 1.5f), "routed_scaling_factor must be 1.5");
        expect(config.scoring_func == "sqrtsoftplus", "scoring_func must be sqrtsoftplus");
        expect(config.topk_method == "noaux_tc", "topk_method must be noaux_tc");
        expect(config.norm_topk_prob, "norm_topk_prob must be true");
        expect(close(config.swiglu_limit, 10.0f), "swiglu_limit must be 10");
        expect(config.hc_mult == 4, "hc_mult must be 4");
        expect(config.hc_sinkhorn_iters == 20, "hc_sinkhorn_iters must be 20");
        expect(close(config.hc_eps, 1e-6f), "hc_eps must be 1e-6");
        expect(close(config.rms_norm_eps, 1e-6f), "rms_norm_eps must be 1e-6");
        expect(config.expert_dtype == "fp4", "expert_dtype must be fp4 in the source config");
        expect(config.num_nextn_predict_layers == 1, "num_nextn_predict_layers must be 1");
        expect(config.dspark_block_size == 5, "dspark_block_size must be 5");
        expect(config.dspark_noise_token_id == 128799, "dspark_noise_token_id must be 128799");
        expect(config.dspark_target_layer_ids == std::vector<int32_t>({40, 41, 42}),
               "dspark_target_layer_ids must be [40, 41, 42]");
        expect(config.dspark_markov_rank == 256, "dspark_markov_rank must be 256");
        expect(config.quant.quant_method == "compressed-tensors", "quant_method is unsupported");
        expect(config.quant.format == "pack-quantized", "quantization format is unsupported");
        expect(config.quant.type == "int", "quantization type is unsupported");
        expect(config.quant.num_bits == 4, "quantization num_bits must be 4");
        expect(config.quant.symmetric, "quantization must be symmetric");
        expect(config.quant.strategy == "group", "quantization strategy is unsupported");
        expect(config.quant.group_size == 32, "quantization group_size must be 32");

        const size_t expected_schedule_size = static_cast<size_t>(config.num_hidden_layers) + 3;
        expect(config.compress_ratios.size() == expected_schedule_size,
               "compress_ratios must contain 43 base entries and 3 auxiliary entries");
        expect(config.compress_ratios[0] == 0 && config.compress_ratios[1] == 0,
               "layers 0-1 must be sliding attention");
        for (int32_t layer = 2; layer < config.num_hidden_layers; ++layer) {
            const int32_t expected_ratio = (layer % 2 == 0) ? 4 : 128;
            expect(config.compress_ratios[static_cast<size_t>(layer)] == expected_ratio,
                   "base decoder compression schedule is incompatible at layer " + std::to_string(layer));
        }
        for (size_t index = static_cast<size_t>(config.num_hidden_layers); index < config.compress_ratios.size(); ++index) {
            expect(config.compress_ratios[index] == 0,
                   "auxiliary compression schedule entries must be zero at index " + std::to_string(index));
        }
    }

    static std::vector<V4LayerSpec> resolve_layers(const DeepSeekV4Config& config) {
        validate_config(config);
        std::vector<V4LayerSpec> layers;
        layers.reserve(static_cast<size_t>(config.num_hidden_layers));
        for (int32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const int32_t ratio = config.compress_ratios[static_cast<size_t>(layer)];
            V4LayerSpec spec;
            spec.layer_id = static_cast<uint32_t>(layer);
            spec.compression_ratio = ratio;
            spec.attention_kind = ratio == 0
                ? V4AttentionKind::Sliding
                : (ratio == 4 ? V4AttentionKind::CSA : V4AttentionKind::HCA);
            layers.push_back(spec);
        }
        return layers;
    }
};

} // namespace aeon::core