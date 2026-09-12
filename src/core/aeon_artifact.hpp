#pragma once

#include "core/swizzled_expert_format.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace aeon::core {

struct AeonArtifactSpec {
    std::string dense_filename{"model_dense.aeon"};
    std::string experts_filename{"model_experts_swizzled.aeon"};
    std::string index_filename{"model_experts_swizzled.index"};
    ExpertFormatKind expert_format_kind{ExpertFormatKind::SWIZZLED_W4A16};
    uint32_t expected_expert_version{AEON_SWIZZLED_EXPERT_FORMAT_VERSION};
    uint32_t expert_sector_size{AEON_SECTOR_SIZE};
    size_t expected_expert_payload_bytes{AEON_SWIZZLED_EXPERT_BYTES};
};

inline AeonArtifactSpec make_current_swizzled_artifact_spec() {
    return AeonArtifactSpec{};
}

} // namespace aeon::core