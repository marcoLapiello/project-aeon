#include "architecture/deepseek_v4/core/v4_model_contract.hpp"
#include "infrastructure/core/model_manifest.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace {

void write_json_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u00" << "0123456789abcdef"[(character >> 4) & 0xf]
                           << "0123456789abcdef"[character & 0xf];
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    output << '"';
}

const char* environment_value(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? "unknown" : value;
}

void write_shape(std::ostream& output, const std::vector<int64_t>& shape) {
    output << '[';
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) output << ',';
        output << shape[index];
    }
    output << ']';
}

void write_string_array(std::ostream& output, const std::vector<std::string>& values) {
    output << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        write_json_string(output, values[index]);
    }
    output << ']';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: aeon_model_contract <model-dir> [--output <path>]\n";
        return 2;
    }

    try {
        const std::string model_dir = argv[1];
        std::string output_path = "-";
        if (argc == 4) {
            if (std::string(argv[2]) != "--output") throw std::invalid_argument("unknown option");
            output_path = argv[3];
        } else if (argc == 3) {
            throw std::invalid_argument("--output requires a path");
        }

        const auto config = aeon::core::DeepSeekV4Config::load_from_json(model_dir + "/config.json");
        const auto layers = aeon::core::V4ModelSpec::resolve_layers(config);
        aeon::core::AeonModelLoader loader;
        loader.open_model(model_dir);
        aeon::core::V4ModelContract::validate(config, loader);
        const auto manifest = aeon::core::AeonModelManifest::load_from_json(model_dir + "/model_manifest.json");
        const auto inventory = loader.dense_tensor_inventory();

        std::ofstream file;
        std::ostream* output = &std::cout;
        if (output_path != "-") {
            const std::filesystem::path parent = std::filesystem::path(output_path).parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
            file.open(output_path);
            if (!file.is_open()) throw std::runtime_error("failed to open output: " + output_path);
            output = &file;
        }

        *output << "{\n  \"stage\":\"stage0\",\n  \"metadata\":{\n"
                << "    \"aeon_commit\":";
        write_json_string(*output, environment_value("AEON_GIT_COMMIT"));
        *output << ",\n    \"model_config_sha256\":";
        write_json_string(*output, environment_value("AEON_MODEL_CONFIG_SHA256"));
        *output << ",\n    \"manifest_sha256\":";
        write_json_string(*output, environment_value("AEON_MANIFEST_SHA256"));
        *output << ",\n    \"tokenizer_sha256\":";
        write_json_string(*output, environment_value("AEON_TOKENIZER_SHA256"));
        *output << ",\n    \"source_snapshot_revision\":";
        write_json_string(*output, environment_value("AEON_SOURCE_SNAPSHOT_REVISION"));
        *output << ",\n    \"external_reference_revision\":";
        write_json_string(*output, environment_value("AEON_EXTERNAL_REFERENCE_REVISION"));
        *output << "\n  },\n  \"manifest\":{\n"
                << "    \"manifest_version\":" << manifest.manifest_version
                << ",\n    \"model_family\":";
        write_json_string(*output, manifest.model_family);
        *output << ",\n    \"architecture\":";
        write_json_string(*output, manifest.architecture);
        *output << ",\n    \"weight_backend\":";
        write_json_string(*output, manifest.weight_backend);
        *output << ",\n    \"dense_file_bytes\":" << manifest.dense_file_bytes
                << ",\n    \"num_layers\":" << manifest.num_layers
                << ",\n    \"experts_per_layer\":" << manifest.experts_per_layer
                << ",\n    \"dense_filename\":";
        write_json_string(*output, manifest.artifact.dense_filename);
        *output << ",\n    \"dense_format_version\":" << manifest.artifact.expected_dense_version
                << "\n  },\n  \"config\":{\n"
                << "    \"model_type\":";
        write_json_string(*output, config.model_type);
        *output << ",\n    \"num_hidden_layers\":" << config.num_hidden_layers
                << ",\n    \"compress_ratios\":";
        *output << '[';
        for (size_t index = 0; index < config.compress_ratios.size(); ++index) {
            if (index != 0) *output << ',';
            *output << config.compress_ratios[index];
        }
        *output << "],\n    \"index_head_dim\":" << config.index_head_dim
                << ",\n    \"index_n_heads\":" << config.index_n_heads
                << ",\n    \"index_topk\":" << config.index_topk
                << ",\n    \"compress_rope_theta\":" << config.compress_rope_theta
                << ",\n    \"o_groups\":" << config.o_groups
                << ",\n    \"num_nextn_predict_layers\":" << config.num_nextn_predict_layers
                << "\n  },\n  \"resolved_layers\":[\n";
        for (size_t index = 0; index < layers.size(); ++index) {
            const auto& layer = layers[index];
            *output << "    {\"layer_id\":" << layer.layer_id << ",\"attention_kind\":";
            write_json_string(*output, aeon::core::v4_attention_kind_name(layer.attention_kind));
            *output << ",\"compression_ratio\":" << layer.compression_ratio << '}';
            if (index + 1 != layers.size()) *output << ',';
            *output << '\n';
        }
        *output << "  ],\n  \"dense_tensors\":[\n";
        for (size_t index = 0; index < inventory.size(); ++index) {
            const auto& [name, tensor] = inventory[index];
            *output << "    {\"name\":";
            write_json_string(*output, name);
            *output << ",\"dtype\":";
            write_json_string(*output, tensor.dtype);
            *output << ",\"byte_size\":" << tensor.byte_size << ",\"shape\":";
            write_shape(*output, tensor.shape);
            *output << '}';
            if (index + 1 != inventory.size()) *output << ',';
            *output << '\n';
        }

        std::vector<std::string> mtp_tensors;
        std::vector<std::string> dspark_tensors;
        for (const auto& [name, tensor] : inventory) {
            (void)tensor;
            if (name.rfind("mtp.", 0) == 0) mtp_tensors.push_back(name);
            if (name.rfind("dspark.", 0) == 0) dspark_tensors.push_back(name);
        }
        *output << "  ],\n  \"auxiliary_tensor_names\":{\n    \"mtp\":";
        write_string_array(*output, mtp_tensors);
        *output << ",\n    \"dspark\":";
        write_string_array(*output, dspark_tensors);
        *output << "\n  }\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "aeon_model_contract: " << error.what() << '\n';
        return 1;
    }
}