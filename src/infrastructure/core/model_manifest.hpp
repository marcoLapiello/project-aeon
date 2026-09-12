#pragma once

#include "infrastructure/core/aeon_artifact.hpp"
#include "infrastructure/backend_registry/expert_backend.hpp"
#include "infrastructure/core/expert_format.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>

namespace aeon::core {

inline constexpr uint32_t AEON_MODEL_MANIFEST_VERSION = 1;

namespace detail {

inline std::size_t manifest_value_start(
    const std::string& text,
    const std::string& key
) {
    const std::string quoted_key = "\"" + key + "\"";
    const std::size_t key_pos = text.find(quoted_key);
    if (key_pos == std::string::npos) {
        throw std::invalid_argument("AeonModelManifest: missing field '" + key + "'");
    }
    const std::size_t colon_pos = text.find(':', key_pos + quoted_key.size());
    if (colon_pos == std::string::npos) {
        throw std::invalid_argument("AeonModelManifest: malformed field '" + key + "'");
    }
    const std::size_t value_pos = text.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (value_pos == std::string::npos) {
        throw std::invalid_argument("AeonModelManifest: missing value for field '" + key + "'");
    }
    return value_pos;
}

inline std::string manifest_string(const std::string& text, const std::string& key) {
    std::size_t pos = manifest_value_start(text, key);
    if (text[pos] != '"') {
        throw std::invalid_argument("AeonModelManifest: field '" + key + "' must be a string");
    }
    ++pos;
    std::string value;
    while (pos < text.size()) {
        const char current = text[pos++];
        if (current == '"') {
            return value;
        }
        if (current != '\\') {
            value += current;
            continue;
        }
        if (pos >= text.size()) {
            break;
        }
        const char escaped = text[pos++];
        switch (escaped) {
        case '"': value += '"'; break;
        case '\\': value += '\\'; break;
        case '/': value += '/'; break;
        case 'b': value += '\b'; break;
        case 'f': value += '\f'; break;
        case 'n': value += '\n'; break;
        case 'r': value += '\r'; break;
        case 't': value += '\t'; break;
        default:
            throw std::invalid_argument("AeonModelManifest: unsupported string escape");
        }
    }
    throw std::invalid_argument("AeonModelManifest: unterminated string field '" + key + "'");
}

inline uint64_t manifest_unsigned(const std::string& text, const std::string& key) {
    const std::size_t pos = manifest_value_start(text, key);
    std::size_t end = pos;
    while (end < text.size() && text[end] >= '0' && text[end] <= '9') {
        ++end;
    }
    if (end == pos) {
        throw std::invalid_argument("AeonModelManifest: field '" + key + "' must be unsigned");
    }
    try {
        return std::stoull(text.substr(pos, end - pos));
    } catch (const std::exception&) {
        throw std::invalid_argument("AeonModelManifest: invalid unsigned field '" + key + "'");
    }
}

inline ExpertFormatKind manifest_format_kind(const std::string& value) {
    if (value == "swizzled_w4a16") {
        return ExpertFormatKind::SWIZZLED_W4A16;
    }
    throw std::invalid_argument(
        "AeonModelManifest: unsupported expert format '" + value + "'");
}

} // namespace detail

struct AeonModelManifest {
    uint32_t manifest_version{AEON_MODEL_MANIFEST_VERSION};
    std::string model_family;
    std::string architecture;
    std::string weight_backend;
    AeonArtifactSpec artifact{};
    size_t dense_file_bytes{0};
    uint32_t num_layers{0};
    uint32_t experts_per_layer{0};

    static AeonModelManifest load_from_json(const std::string& manifest_path) {
        std::ifstream file(manifest_path);
        if (!file.is_open()) {
            throw std::runtime_error(
                "AeonModelManifest: failed to open manifest: " + manifest_path);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        const std::string text = buffer.str();

        AeonModelManifest manifest;
        manifest.manifest_version = static_cast<uint32_t>(
            detail::manifest_unsigned(text, "manifest_version"));
        manifest.model_family = detail::manifest_string(text, "model_family");
        manifest.architecture = detail::manifest_string(text, "architecture");
        manifest.weight_backend = detail::manifest_string(text, "weight_backend");
        manifest.artifact.dense_filename = detail::manifest_string(text, "dense_filename");
        manifest.artifact.experts_filename = detail::manifest_string(text, "experts_filename");
        manifest.artifact.index_filename = detail::manifest_string(text, "index_filename");
        manifest.artifact.expected_dense_version = static_cast<uint32_t>(
            detail::manifest_unsigned(text, "dense_format_version"));
        manifest.artifact.expert_format_kind = detail::manifest_format_kind(
            detail::manifest_string(text, "expert_format"));
        manifest.artifact.expected_expert_version = static_cast<uint32_t>(
            detail::manifest_unsigned(text, "expert_format_version"));
        manifest.artifact.expert_sector_size = static_cast<uint32_t>(
            detail::manifest_unsigned(text, "expert_sector_size"));
        manifest.artifact.expected_expert_payload_bytes = static_cast<size_t>(
            detail::manifest_unsigned(text, "expert_payload_bytes"));
        manifest.dense_file_bytes = static_cast<size_t>(
            detail::manifest_unsigned(text, "dense_file_bytes"));
        manifest.num_layers = static_cast<uint32_t>(
            detail::manifest_unsigned(text, "num_layers"));
        manifest.experts_per_layer = static_cast<uint32_t>(
            detail::manifest_unsigned(text, "experts_per_layer"));
        manifest.validate();
        return manifest;
    }

    void validate() const {
        if (manifest_version != AEON_MODEL_MANIFEST_VERSION) {
            throw std::invalid_argument(
                "AeonModelManifest: unsupported manifest version " +
                std::to_string(manifest_version));
        }
        if (model_family.empty() || architecture.empty() || weight_backend.empty()) {
            throw std::invalid_argument("AeonModelManifest: model identity is incomplete");
        }
        if (artifact.dense_filename.empty() || artifact.experts_filename.empty() ||
            artifact.index_filename.empty() ||
            artifact.expected_dense_version == 0 ||
            artifact.expert_format_kind == ExpertFormatKind::UNKNOWN ||
            artifact.expected_expert_version == 0 ||
            artifact.expert_sector_size == 0 ||
            (artifact.expert_sector_size & (artifact.expert_sector_size - 1)) != 0 ||
            artifact.expected_expert_payload_bytes == 0) {
            throw std::invalid_argument("AeonModelManifest: artifact identity is incomplete");
        }
        if (weight_backend != ExpertBackendRegistry::resolve(artifact.expert_format_kind).name) {
            throw std::invalid_argument(
                "AeonModelManifest: weight backend does not match expert format kind");
        }
        if (dense_file_bytes == 0 || num_layers == 0 || experts_per_layer == 0) {
            throw std::invalid_argument(
                "AeonModelManifest: dense size and catalog dimensions must be non-zero");
        }

        ExpertFormatDescriptor expected_format{
            artifact.expert_format_kind,
            artifact.expected_expert_version,
            artifact.expert_sector_size,
            num_layers,
            experts_per_layer,
            artifact.expected_expert_payload_bytes
        };
        expected_format.validate_catalog();
    }

    void validate_loaded(
        const ExpertFormatDescriptor& loaded_format,
        size_t loaded_dense_file_bytes
    ) const {
        validate();
        if (loaded_dense_file_bytes != dense_file_bytes) {
            throw std::runtime_error(
                "AeonModelManifest: dense file size does not match the manifest");
        }
        if (loaded_format.kind != artifact.expert_format_kind ||
            loaded_format.artifact_version != artifact.expected_expert_version ||
            loaded_format.sector_size != artifact.expert_sector_size ||
            loaded_format.payload_bytes != artifact.expected_expert_payload_bytes ||
            loaded_format.num_layers != num_layers ||
            loaded_format.experts_per_layer != experts_per_layer) {
            throw std::runtime_error(
                "AeonModelManifest: loaded artifact identity does not match the manifest");
        }
    }
};

} // namespace aeon::core