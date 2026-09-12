#pragma once

#include "core/aeon_artifact.hpp"
#include "core/expert_backend.hpp"
#include "core/expert_format.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace aeon::core {

inline constexpr uint32_t AEON_MODEL_MANIFEST_VERSION = 1;

struct AeonModelManifest {
    uint32_t manifest_version{AEON_MODEL_MANIFEST_VERSION};
    std::string model_family;
    std::string architecture;
    std::string weight_backend;
    AeonArtifactSpec artifact{};
    size_t dense_file_bytes{0};
    uint32_t num_layers{0};
    uint32_t experts_per_layer{0};

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