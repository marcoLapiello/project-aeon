#pragma once

#include "infrastructure/core/json.hpp"

#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_set>

namespace aeon::core {

struct RopeScalingConfig {
    std::string type{"yarn"};
    float factor{1.0f};
    float beta_fast{32.0f};
    float beta_slow{1.0f};
    int32_t original_max_position_embeddings{65536};
};

struct QuantConfig {
    std::string quant_method{"compressed-tensors"};
    std::string format{"pack-quantized"};
    std::string type{"int"};
    int32_t num_bits{4};
    bool symmetric{true};
    std::string strategy{"group"};
    int32_t group_size{32};
};

struct DeepSeekV4Config {
    std::string model_type{"deepseek_v4"};
    std::vector<std::string> architectures{"DeepseekV4ForCausalLM"};

    int32_t vocab_size{129280};
    int32_t hidden_size{4096};
    int32_t moe_intermediate_size{2048};
    int32_t num_hidden_layers{43};
    int32_t num_attention_heads{64};
    int32_t num_key_value_heads{1};
    int32_t head_dim{512};
    int32_t q_lora_rank{1024};
    int32_t o_lora_rank{1024};
    int32_t qk_rope_head_dim{64};
    int32_t sliding_window{128};
    int32_t index_head_dim{128};
    int32_t index_n_heads{64};
    int32_t index_topk{512};
    int32_t o_groups{8};
    float compress_rope_theta{160000.0f};
    std::vector<int32_t> compress_ratios;

    int32_t n_routed_experts{256};
    int32_t n_shared_experts{1};
    int32_t num_experts_per_tok{6};
    int32_t num_hash_layers{3};
    float routed_scaling_factor{1.5f};
    std::string scoring_func{"sqrtsoftplus"};
    std::string topk_method{"noaux_tc"};
    bool norm_topk_prob{true};
    float swiglu_limit{10.0f};

    // Hyper-Connections (HC) configuration
    int32_t hc_mult{4};
    int32_t hc_sinkhorn_iters{20};
    float hc_eps{1e-06f};

    // Normalization & embeddings
    float rms_norm_eps{1e-06f};
    int32_t max_position_embeddings{1048576};
    int32_t original_max_position_embeddings{65536};
    float rope_theta{10000.0f};
    int32_t rope_factor{16};
    float rope_beta_fast{32.0f};
    float rope_beta_slow{1.0f};
    RopeScalingConfig rope_scaling;

    std::string expert_dtype{"fp4"};
    int32_t num_nextn_predict_layers{1};
    int32_t dspark_block_size{5};
    int32_t dspark_noise_token_id{128799};
    std::vector<int32_t> dspark_target_layer_ids;
    int32_t dspark_markov_rank{256};

    QuantConfig quant;
    std::unordered_set<std::string> present_fields;

    bool has_field(const std::string& path) const {
        return present_fields.find(path) != present_fields.end();
    }

    static DeepSeekV4Config load_from_json(const std::string& json_path) {
        const JsonValue document = JsonValue::parse_file(json_path);
        if (!document.is_object()) {
            throw std::runtime_error("DeepSeekV4Config: root value must be an object");
        }
        DeepSeekV4Config cfg;

        auto field = [&](const JsonValue& object, const std::string& key, const std::string& path) -> const JsonValue* {
            const JsonValue* value = object.find(key);
            if (value != nullptr) cfg.present_fields.insert(path);
            return value;
        };
        auto read_int = [&](const JsonValue& object, const std::string& key, int32_t fallback, const std::string& path) {
            const JsonValue* value = field(object, key, path);
            if (value == nullptr) return fallback;
            const int64_t parsed = value->as_int64();
            if (parsed < std::numeric_limits<int32_t>::min() || parsed > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("DeepSeekV4Config: integer field out of range: " + path);
            }
            return static_cast<int32_t>(parsed);
        };
        auto read_float = [&](const JsonValue& object, const std::string& key, float fallback, const std::string& path) {
            const JsonValue* value = field(object, key, path);
            return value == nullptr ? fallback : static_cast<float>(value->as_number());
        };
        auto read_string = [&](const JsonValue& object, const std::string& key, const std::string& fallback, const std::string& path) {
            const JsonValue* value = field(object, key, path);
            return value == nullptr ? fallback : value->as_string();
        };
        auto read_bool = [&](const JsonValue& object, const std::string& key, bool fallback, const std::string& path) {
            const JsonValue* value = field(object, key, path);
            return value == nullptr ? fallback : value->as_bool();
        };

        cfg.model_type = read_string(document, "model_type", cfg.model_type, "model_type");
        if (const JsonValue* value = field(document, "architectures", "architectures")) {
            cfg.architectures.clear();
            for (const auto& architecture : value->as_array()) cfg.architectures.push_back(architecture.as_string());
        }
        cfg.vocab_size = read_int(document, "vocab_size", cfg.vocab_size, "vocab_size");
        cfg.hidden_size = read_int(document, "hidden_size", cfg.hidden_size, "hidden_size");
        cfg.moe_intermediate_size = read_int(document, "moe_intermediate_size", cfg.moe_intermediate_size, "moe_intermediate_size");
        cfg.num_hidden_layers = read_int(document, "num_hidden_layers", cfg.num_hidden_layers, "num_hidden_layers");
        cfg.num_attention_heads = read_int(document, "num_attention_heads", cfg.num_attention_heads, "num_attention_heads");
        cfg.num_key_value_heads = read_int(document, "num_key_value_heads", cfg.num_key_value_heads, "num_key_value_heads");
        cfg.head_dim = read_int(document, "head_dim", cfg.head_dim, "head_dim");
        cfg.q_lora_rank = read_int(document, "q_lora_rank", cfg.q_lora_rank, "q_lora_rank");
        cfg.o_lora_rank = read_int(document, "o_lora_rank", cfg.o_lora_rank, "o_lora_rank");
        cfg.qk_rope_head_dim = read_int(document, "qk_rope_head_dim", cfg.qk_rope_head_dim, "qk_rope_head_dim");
        cfg.sliding_window = read_int(document, "sliding_window", cfg.sliding_window, "sliding_window");
        cfg.index_head_dim = read_int(document, "index_head_dim", cfg.index_head_dim, "index_head_dim");
        cfg.index_n_heads = read_int(document, "index_n_heads", cfg.index_n_heads, "index_n_heads");
        cfg.index_topk = read_int(document, "index_topk", cfg.index_topk, "index_topk");
        cfg.o_groups = read_int(document, "o_groups", cfg.o_groups, "o_groups");
        cfg.compress_rope_theta = read_float(document, "compress_rope_theta", cfg.compress_rope_theta, "compress_rope_theta");

        if (const JsonValue* value = field(document, "compress_ratios", "compress_ratios")) {
            cfg.compress_ratios.clear();
            for (const auto& ratio : value->as_array()) {
                const int64_t parsed = ratio.as_int64();
                if (parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
                    throw std::runtime_error("DeepSeekV4Config: compression ratio out of range");
                }
                cfg.compress_ratios.push_back(static_cast<int32_t>(parsed));
            }
        }

        cfg.n_routed_experts = read_int(document, "n_routed_experts", cfg.n_routed_experts, "n_routed_experts");
        cfg.n_shared_experts = read_int(document, "n_shared_experts", cfg.n_shared_experts, "n_shared_experts");
        cfg.num_experts_per_tok = read_int(document, "num_experts_per_tok", cfg.num_experts_per_tok, "num_experts_per_tok");
        cfg.num_hash_layers = read_int(document, "num_hash_layers", cfg.num_hash_layers, "num_hash_layers");
        cfg.routed_scaling_factor = read_float(document, "routed_scaling_factor", cfg.routed_scaling_factor, "routed_scaling_factor");
        cfg.scoring_func = read_string(document, "scoring_func", cfg.scoring_func, "scoring_func");
        cfg.topk_method = read_string(document, "topk_method", cfg.topk_method, "topk_method");
        cfg.norm_topk_prob = read_bool(document, "norm_topk_prob", cfg.norm_topk_prob, "norm_topk_prob");
        cfg.swiglu_limit = read_float(document, "swiglu_limit", cfg.swiglu_limit, "swiglu_limit");

        cfg.hc_mult = read_int(document, "hc_mult", cfg.hc_mult, "hc_mult");
        cfg.hc_sinkhorn_iters = read_int(document, "hc_sinkhorn_iters", cfg.hc_sinkhorn_iters, "hc_sinkhorn_iters");
        cfg.hc_eps = read_float(document, "hc_eps", cfg.hc_eps, "hc_eps");

        cfg.rms_norm_eps = read_float(document, "rms_norm_eps", cfg.rms_norm_eps, "rms_norm_eps");
        cfg.max_position_embeddings = read_int(document, "max_position_embeddings", cfg.max_position_embeddings, "max_position_embeddings");
        cfg.rope_theta = read_float(document, "rope_theta", cfg.rope_theta, "rope_theta");
        cfg.expert_dtype = read_string(document, "expert_dtype", cfg.expert_dtype, "expert_dtype");
        cfg.num_nextn_predict_layers = read_int(document, "num_nextn_predict_layers", cfg.num_nextn_predict_layers, "num_nextn_predict_layers");
        cfg.dspark_block_size = read_int(document, "dspark_block_size", cfg.dspark_block_size, "dspark_block_size");
        cfg.dspark_noise_token_id = read_int(document, "dspark_noise_token_id", cfg.dspark_noise_token_id, "dspark_noise_token_id");
        cfg.dspark_markov_rank = read_int(document, "dspark_markov_rank", cfg.dspark_markov_rank, "dspark_markov_rank");
        if (const JsonValue* value = field(document, "dspark_target_layer_ids", "dspark_target_layer_ids")) {
            cfg.dspark_target_layer_ids.clear();
            for (const auto& layer_id : value->as_array()) {
                const int64_t parsed = layer_id.as_int64();
                if (parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
                    throw std::runtime_error("DeepSeekV4Config: DSpark layer ID out of range");
                }
                cfg.dspark_target_layer_ids.push_back(static_cast<int32_t>(parsed));
            }
        }

        if (const JsonValue* value = field(document, "rope_scaling", "rope_scaling")) {
            const JsonValue& rope_scaling = *value;
            cfg.rope_scaling.type = read_string(rope_scaling, "type", cfg.rope_scaling.type, "rope_scaling.type");
            cfg.rope_scaling.factor = read_float(rope_scaling, "factor", cfg.rope_scaling.factor, "rope_scaling.factor");
            cfg.rope_scaling.beta_fast = read_float(rope_scaling, "beta_fast", cfg.rope_scaling.beta_fast, "rope_scaling.beta_fast");
            cfg.rope_scaling.beta_slow = read_float(rope_scaling, "beta_slow", cfg.rope_scaling.beta_slow, "rope_scaling.beta_slow");
            cfg.rope_scaling.original_max_position_embeddings = read_int(
                rope_scaling,
                "original_max_position_embeddings",
                cfg.rope_scaling.original_max_position_embeddings,
                "rope_scaling.original_max_position_embeddings");
            cfg.rope_factor = static_cast<int32_t>(cfg.rope_scaling.factor);
            cfg.rope_beta_fast = cfg.rope_scaling.beta_fast;
            cfg.rope_beta_slow = cfg.rope_scaling.beta_slow;
            cfg.original_max_position_embeddings = cfg.rope_scaling.original_max_position_embeddings;
        }

        if (const JsonValue* value = field(document, "quantization_config", "quantization_config")) {
            const JsonValue& quantization = *value;
            cfg.quant.quant_method = read_string(quantization, "quant_method", cfg.quant.quant_method, "quantization_config.quant_method");
            if (const JsonValue* groups = field(quantization, "config_groups", "quantization_config.config_groups")) {
                const JsonValue& group_zero = groups->at("group_0");
                cfg.quant.format = read_string(group_zero, "format", cfg.quant.format, "quantization_config.config_groups.group_0.format");
                if (const JsonValue* weights = field(group_zero, "weights", "quantization_config.config_groups.group_0.weights")) {
                    cfg.quant.num_bits = read_int(*weights, "num_bits", cfg.quant.num_bits, "quantization_config.config_groups.group_0.weights.num_bits");
                    cfg.quant.type = read_string(*weights, "type", cfg.quant.type, "quantization_config.config_groups.group_0.weights.type");
                    cfg.quant.symmetric = read_bool(*weights, "symmetric", cfg.quant.symmetric, "quantization_config.config_groups.group_0.weights.symmetric");
                    cfg.quant.strategy = read_string(*weights, "strategy", cfg.quant.strategy, "quantization_config.config_groups.group_0.weights.strategy");
                    cfg.quant.group_size = read_int(*weights, "group_size", cfg.quant.group_size, "quantization_config.config_groups.group_0.weights.group_size");
                }
            }
        }

        return cfg;
    }
};

} // namespace aeon::core
