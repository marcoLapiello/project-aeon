#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace aeon::core {

inline constexpr uint32_t AEON_SECTOR_SIZE = 4096;
inline constexpr size_t AEON_EXPERT_BYTES = 14155776;

enum class ExpertFormatKind : uint8_t {
    UNKNOWN = 0,
    SWIZZLED_W4A16 = 1
};

struct ExpertFormatDescriptor {
    ExpertFormatKind kind{ExpertFormatKind::UNKNOWN};
    uint32_t artifact_version{0};
    uint32_t sector_size{AEON_SECTOR_SIZE};
    uint32_t num_layers{0};
    uint32_t experts_per_layer{0};
    size_t payload_bytes{0};

    uint64_t total_experts() const noexcept {
        return static_cast<uint64_t>(num_layers) * experts_per_layer;
    }

    void validate_payload() const {
        if (sector_size == 0 || (sector_size & (sector_size - 1)) != 0 ||
            payload_bytes == 0 || payload_bytes % sector_size != 0) {
            throw std::invalid_argument(
                "ExpertFormatDescriptor: payload must be a non-zero multiple of sector_size");
        }
    }

    void validate_catalog() const {
        validate_payload();
        if (num_layers == 0 || experts_per_layer == 0 ||
            total_experts() > std::numeric_limits<size_t>::max() / 16) {
            throw std::invalid_argument(
                "ExpertFormatDescriptor: layer and expert counts must describe a valid catalog");
        }
    }
};

inline constexpr ExpertFormatDescriptor make_current_swizzled_expert_format(
    uint32_t num_layers = 0,
    uint32_t experts_per_layer = 0
) noexcept {
    return ExpertFormatDescriptor{
        ExpertFormatKind::SWIZZLED_W4A16,
        2,
        AEON_SECTOR_SIZE,
        num_layers,
        experts_per_layer,
        AEON_EXPERT_BYTES
    };
}

} // namespace aeon::core