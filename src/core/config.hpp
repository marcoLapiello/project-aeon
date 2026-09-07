#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace aeon::core {

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

    // MoE configuration
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

    // Quantization configuration
    QuantConfig quant;

    static DeepSeekV4Config load_from_json(const std::string& json_path) {
        std::ifstream file(json_path);
        if (!file.is_open()) {
            throw std::runtime_error("DeepSeekV4Config: Failed to open config file: " + json_path);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string text = buffer.str();

        DeepSeekV4Config cfg;

        auto extract_int = [&](const std::string& key, int32_t default_val) -> int32_t {
            std::string pattern = "\"" + key + "\"";
            size_t pos = text.find(pattern);
            if (pos == std::string::npos) return default_val;
            size_t colon = text.find(':', pos);
            if (colon == std::string::npos) return default_val;
            size_t start = text.find_first_of("-0123456789", colon);
            if (start == std::string::npos) return default_val;
            size_t end = text.find_first_not_of("-0123456789", start);
            return std::stoi(text.substr(start, end - start));
        };

        auto extract_float = [&](const std::string& key, float default_val) -> float {
            std::string pattern = "\"" + key + "\"";
            size_t pos = text.find(pattern);
            if (pos == std::string::npos) return default_val;
            size_t colon = text.find(':', pos);
            if (colon == std::string::npos) return default_val;
            size_t start = text.find_first_of("-0123456789.", colon);
            if (start == std::string::npos) return default_val;
            size_t end = text.find_first_not_of("-0123456789.eE", start);
            return std::stof(text.substr(start, end - start));
        };

        auto extract_string = [&](const std::string& key, const std::string& default_val) -> std::string {
            std::string pattern = "\"" + key + "\"";
            size_t pos = text.find(pattern);
            if (pos == std::string::npos) return default_val;
            size_t colon = text.find(':', pos);
            if (colon == std::string::npos) return default_val;
            size_t start = text.find('"', colon);
            if (start == std::string::npos) return default_val;
            size_t end = text.find('"', start + 1);
            if (end == std::string::npos) return default_val;
            return text.substr(start + 1, end - start - 1);
        };

        auto extract_bool = [&](const std::string& key, bool default_val) -> bool {
            std::string pattern = "\"" + key + "\"";
            size_t pos = text.find(pattern);
            if (pos == std::string::npos) return default_val;
            size_t colon = text.find(':', pos);
            if (colon == std::string::npos) return default_val;
            size_t true_pos = text.find("true", colon);
            size_t false_pos = text.find("false", colon);
            if (true_pos != std::string::npos && (false_pos == std::string::npos || true_pos < false_pos)) {
                return true;
            }
            if (false_pos != std::string::npos) {
                return false;
            }
            return default_val;
        };

        cfg.vocab_size = extract_int("vocab_size", cfg.vocab_size);
        cfg.hidden_size = extract_int("hidden_size", cfg.hidden_size);
        cfg.moe_intermediate_size = extract_int("moe_intermediate_size", cfg.moe_intermediate_size);
        cfg.num_hidden_layers = extract_int("num_hidden_layers", cfg.num_hidden_layers);
        cfg.num_attention_heads = extract_int("num_attention_heads", cfg.num_attention_heads);
        cfg.num_key_value_heads = extract_int("num_key_value_heads", cfg.num_key_value_heads);
        cfg.head_dim = extract_int("head_dim", cfg.head_dim);
        cfg.q_lora_rank = extract_int("q_lora_rank", cfg.q_lora_rank);
        cfg.o_lora_rank = extract_int("o_lora_rank", cfg.o_lora_rank);
        cfg.qk_rope_head_dim = extract_int("qk_rope_head_dim", cfg.qk_rope_head_dim);
        cfg.sliding_window = extract_int("sliding_window", cfg.sliding_window);

        cfg.n_routed_experts = extract_int("n_routed_experts", cfg.n_routed_experts);
        cfg.n_shared_experts = extract_int("n_shared_experts", cfg.n_shared_experts);
        cfg.num_experts_per_tok = extract_int("num_experts_per_tok", cfg.num_experts_per_tok);
        cfg.num_hash_layers = extract_int("num_hash_layers", cfg.num_hash_layers);
        cfg.routed_scaling_factor = extract_float("routed_scaling_factor", cfg.routed_scaling_factor);
        cfg.scoring_func = extract_string("scoring_func", cfg.scoring_func);
        cfg.topk_method = extract_string("topk_method", cfg.topk_method);
        cfg.norm_topk_prob = extract_bool("norm_topk_prob", cfg.norm_topk_prob);
        cfg.swiglu_limit = extract_float("swiglu_limit", cfg.swiglu_limit);

        cfg.hc_mult = extract_int("hc_mult", cfg.hc_mult);
        cfg.hc_sinkhorn_iters = extract_int("hc_sinkhorn_iters", cfg.hc_sinkhorn_iters);
        cfg.hc_eps = extract_float("hc_eps", cfg.hc_eps);

        cfg.rms_norm_eps = extract_float("rms_norm_eps", cfg.rms_norm_eps);
        cfg.max_position_embeddings = extract_int("max_position_embeddings", cfg.max_position_embeddings);
        cfg.rope_theta = extract_float("rope_theta", cfg.rope_theta);

        // Quantization config extraction
        cfg.quant.quant_method = extract_string("quant_method", cfg.quant.quant_method);
        cfg.quant.format = extract_string("format", cfg.quant.format);
        cfg.quant.num_bits = extract_int("num_bits", cfg.quant.num_bits);
        cfg.quant.group_size = extract_int("group_size", cfg.quant.group_size);
        cfg.quant.symmetric = extract_bool("symmetric", cfg.quant.symmetric);

        return cfg;
    }
};

} // namespace aeon::core
