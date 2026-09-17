#pragma once

#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "infrastructure/core/aeon_loader.hpp"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::core {

struct V4TensorRequirement {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
};

class V4ModelContract {
public:
    static void validate(const DeepSeekV4Config& config, const AeonModelLoader& loader) {
        const auto layers = V4ModelSpec::resolve_layers(config);
        for (const auto& requirement : model_tensor_requirements(config)) {
            validate_tensor(requirement, loader);
        }
        for (const auto& layer : layers) {
            for (const auto& requirement : layer_tensor_requirements(config, layer)) {
                validate_tensor(requirement, loader);
            }
        }
    }

    static void validate_tensor_requirement(
        const V4TensorRequirement& requirement,
        const AeonModelLoader& loader
    ) {
        validate_tensor(requirement, loader);
    }

    static std::vector<V4TensorRequirement> model_tensor_requirements(const DeepSeekV4Config& config) {
        return {
            {"embed.weight", "F16", {config.vocab_size, config.hidden_size}},
            {"hc_head_base", "F32", {config.hc_mult}},
            {"hc_head_fn", "F32", {config.hc_mult, config.hc_mult * config.hidden_size}},
            {"hc_head_scale", "F32", {1}},
            {"head.weight", "F16", {config.vocab_size, config.hidden_size}},
            // The final RMSNorm's learned weight. It was missing from this list
            // even though `V4ModelResources::initialize` uploads it, so a
            // checkpoint without it would have passed `validate()` and then thrown
            // at upload with a less specific message. Found by the P4 budget
            // audit: summing this table disagreed with the measured upload by
            // exactly this tensor's 8,192 bytes.
            {"norm.weight", "F16", {config.hidden_size}}
        };
    }

    static std::vector<V4TensorRequirement> layer_tensor_requirements(
        const DeepSeekV4Config& config,
        const V4LayerSpec& layer
    ) {
        const std::string prefix = "layers." + std::to_string(layer.layer_id) + ".";
        std::vector<V4TensorRequirement> requirements = {
            {prefix + "attn_norm.weight", "F16", {config.hidden_size}},
            {prefix + "attn.wq_a.weight", "F16", {config.q_lora_rank, config.hidden_size}},
            {prefix + "attn.q_norm.weight", "F16", {config.q_lora_rank}},
            {prefix + "attn.wq_b.weight", "F16", {config.num_attention_heads * config.head_dim * 1, config.q_lora_rank}},
            {prefix + "attn.wkv.weight", "F16", {config.head_dim, config.hidden_size}},
            {prefix + "attn.kv_norm.weight", "F16", {config.head_dim}},
            {prefix + "attn.attn_sink", "F32", {config.num_attention_heads}},
            {prefix + "attn.wo_a.weight", "F16", {config.o_groups * config.o_lora_rank, config.hidden_size}},
            {prefix + "attn.wo_b.weight", "F16", {config.hidden_size, config.o_groups * config.o_lora_rank}},
            {prefix + "hc_attn_fn", "F32", {24, config.hc_mult * config.hidden_size}},
            {prefix + "hc_attn_base", "F32", {24}},
            {prefix + "hc_attn_scale", "F32", {3}},
            {prefix + "ffn_norm.weight", "F16", {config.hidden_size}},
            {prefix + "hc_ffn_fn", "F32", {24, config.hc_mult * config.hidden_size}},
            {prefix + "hc_ffn_base", "F32", {24}},
            {prefix + "hc_ffn_scale", "F32", {3}},
            {prefix + "ffn.shared_experts.w1.weight", "F16", {config.moe_intermediate_size, config.hidden_size}},
            {prefix + "ffn.shared_experts.w2.weight", "F16", {config.hidden_size, config.moe_intermediate_size}},
            {prefix + "ffn.shared_experts.w3.weight", "F16", {config.moe_intermediate_size, config.hidden_size}},
            {prefix + "ffn.gate.weight", "F16", {config.n_routed_experts, config.hidden_size}}
        };

        if (layer.layer_id < static_cast<uint32_t>(config.num_hash_layers)) {
            requirements.push_back({prefix + "ffn.gate.tid2eid", "I64", {config.vocab_size, config.num_experts_per_tok}});
        } else {
            requirements.push_back({prefix + "ffn.gate.bias", "F32", {config.n_routed_experts}});
        }

        if (layer.attention_kind == V4AttentionKind::CSA) {
            requirements.push_back({prefix + "attn.compressor.ape", "F32", {4, config.q_lora_rank}});
            requirements.push_back({prefix + "attn.compressor.norm.weight", "F16", {config.head_dim}});
            requirements.push_back({prefix + "attn.compressor.wgate.weight", "F16", {config.q_lora_rank, config.hidden_size}});
            requirements.push_back({prefix + "attn.compressor.wkv.weight", "F16", {config.q_lora_rank, config.hidden_size}});
            requirements.push_back({prefix + "attn.indexer.compressor.ape", "F32", {4, config.index_n_heads * 4}});
            requirements.push_back({prefix + "attn.indexer.compressor.norm.weight", "F16", {config.index_head_dim}});
            requirements.push_back({prefix + "attn.indexer.compressor.wgate.weight", "F16", {config.index_n_heads * 4, config.hidden_size}});
            requirements.push_back({prefix + "attn.indexer.compressor.wkv.weight", "F16", {config.index_n_heads * 4, config.hidden_size}});
            requirements.push_back({prefix + "attn.indexer.weights_proj.weight", "F16", {config.index_n_heads, config.hidden_size}});
            requirements.push_back({prefix + "attn.indexer.wq_b.weight", "F16", {config.index_n_heads * config.index_head_dim, config.q_lora_rank}});
        } else if (layer.attention_kind == V4AttentionKind::HCA) {
            requirements.push_back({prefix + "attn.compressor.ape", "F32", {128, config.head_dim}});
            requirements.push_back({prefix + "attn.compressor.norm.weight", "F16", {config.head_dim}});
            requirements.push_back({prefix + "attn.compressor.wgate.weight", "F16", {config.head_dim, config.hidden_size}});
            requirements.push_back({prefix + "attn.compressor.wkv.weight", "F16", {config.head_dim, config.hidden_size}});
        }
        return requirements;
    }

    // Byte size of one required tensor, from its declared dtype and shape.
    static size_t requirement_bytes(const V4TensorRequirement& requirement) {
        size_t elements = 1;
        for (int64_t dimension : requirement.shape) {
            if (dimension < 0) {
                throw std::invalid_argument(
                    "V4ModelContract: negative dimension for " + requirement.name);
            }
            elements *= static_cast<size_t>(dimension);
        }
        return elements * dtype_size(requirement.dtype);
    }

    // The bytes the graph actually **uploads to VRAM**.
    //
    // This is what the memory budget must reserve, and it is deliberately not
    // `AeonModelLoader::dense_file_size()`. The container holds two things the
    // graph never places on the device:
    //
    //   * `embed.weight` — 1,059,061,760 B. The embedding lookup reads the
    //     **host** mmap (`V4ModelResources::host_embed_table`, and `embed_token`
    //     copies one row per token), so no device copy of the table exists.
    //   * the `mtp.*` DSpark draft head — 1,041,848,220 B across 69 tensors,
    //     unused and deferred (composition plan §9). It is excluded automatically,
    //     because this function enumerates the *contract*, and the contract lists
    //     only what the graph binds.
    //
    // Together those are 1.957 GiB that the old file-size reservation over-counted.
    // Reserving them cost 148 Hot VRAM expert slots (675 instead of 823 at context
    // 256), which is the whole reason for this function.
    static size_t uploaded_dense_bytes(const DeepSeekV4Config& config) {
        size_t total = 0;
        for (const auto& requirement : model_tensor_requirements(config)) {
            if (requirement.name == "embed.weight") continue;  // host-resident
            total += requirement_bytes(requirement);
        }
        for (const auto& layer : V4ModelSpec::resolve_layers(config)) {
            for (const auto& requirement : layer_tensor_requirements(config, layer)) {
                total += requirement_bytes(requirement);
            }
        }
        return total;
    }

private:
    static size_t dtype_size(const std::string& dtype) {
        if (dtype == "F16") return 2;
        if (dtype == "F32") return 4;
        if (dtype == "I64") return 8;
        throw std::invalid_argument("V4ModelContract: unsupported expected dtype " + dtype);
    }

    static std::string shape_string(const std::vector<int64_t>& shape) {
        std::ostringstream output;
        output << '[';
        for (size_t index = 0; index < shape.size(); ++index) {
            if (index != 0) output << ',';
            output << shape[index];
        }
        output << ']';
        return output.str();
    }

    static void validate_tensor(const V4TensorRequirement& requirement, const AeonModelLoader& loader) {
        if (!loader.has_tensor(requirement.name)) {
            throw std::invalid_argument("V4ModelContract: missing required tensor " + requirement.name);
        }
        const auto& tensor = loader.get_tensor(requirement.name);
        if (tensor.dtype != requirement.dtype) {
            throw std::invalid_argument(
                "V4ModelContract: tensor " + requirement.name + " has dtype " + tensor.dtype +
                ", expected " + requirement.dtype);
        }
        if (tensor.shape != requirement.shape) {
            throw std::invalid_argument(
                "V4ModelContract: tensor " + requirement.name + " has shape " + shape_string(tensor.shape) +
                ", expected " + shape_string(requirement.shape));
        }
        size_t element_count = 1;
        for (const int64_t dimension : requirement.shape) {
            if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max() / element_count) {
                throw std::invalid_argument("V4ModelContract: invalid expected shape for " + requirement.name);
            }
            element_count *= static_cast<size_t>(dimension);
        }
        const size_t expected_bytes = element_count * dtype_size(requirement.dtype);
        if (tensor.byte_size != static_cast<int64_t>(expected_bytes)) {
            throw std::invalid_argument(
                "V4ModelContract: tensor " + requirement.name + " has byte size " +
                std::to_string(tensor.byte_size) + ", expected " + std::to_string(expected_bytes));
        }
    }
};

} // namespace aeon::core