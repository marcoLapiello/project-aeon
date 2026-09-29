#pragma once

#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace aeon::core {

inline constexpr uint32_t AEON_DENSE_FORMAT_VERSION = 1;

struct AeonArtifactSpec {
    std::string dense_filename{"model_dense.aeon"};
    std::string experts_filename{"model_experts_swizzled.aeon"};
    std::string index_filename{"model_experts_swizzled.index"};
    uint32_t expected_dense_version{AEON_DENSE_FORMAT_VERSION};
    ExpertFormatKind expert_format_kind{ExpertFormatKind::SWIZZLED_W4A16};
    uint32_t expected_expert_version{AEON_SWIZZLED_EXPERT_FORMAT_VERSION};
    uint32_t expert_sector_size{AEON_SECTOR_SIZE};
    size_t expected_expert_payload_bytes{AEON_SWIZZLED_EXPERT_BYTES};
};

inline AeonArtifactSpec make_current_swizzled_artifact_spec() {
    return AeonArtifactSpec{};
}

inline bool is_current_swizzled_artifact_spec(const AeonArtifactSpec& artifact) noexcept {
    const auto current = make_current_swizzled_artifact_spec();
    return artifact.dense_filename == current.dense_filename &&
           artifact.experts_filename == current.experts_filename &&
           artifact.index_filename == current.index_filename &&
           artifact.expected_dense_version == current.expected_dense_version &&
           artifact.expert_format_kind == current.expert_format_kind &&
           artifact.expected_expert_version == current.expected_expert_version &&
           artifact.expert_sector_size == current.expert_sector_size &&
           artifact.expected_expert_payload_bytes == current.expected_expert_payload_bytes;
}

} // namespace aeon::core