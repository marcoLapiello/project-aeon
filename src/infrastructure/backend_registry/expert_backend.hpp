#pragma once

#include "infrastructure/core/expert_format.hpp"
#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace aeon::core {

struct ExpertBackendDescriptor {
    std::string_view name;
    ExpertFormatKind format_kind{ExpertFormatKind::UNKNOWN};
    uint32_t artifact_version{0};
    size_t payload_bytes{0};
    bool supports_v4_pipeline{false};

    void validate_format(const ExpertFormatDescriptor& format) const {
        if (format.kind != format_kind ||
            format.artifact_version != artifact_version ||
            format.payload_bytes != payload_bytes) {
            throw std::runtime_error(
                "Expert backend '" + std::string(name) +
                "' is incompatible with the loaded expert artifact");
        }
    }
};

class ExpertBackendRegistry {
public:
    static const ExpertBackendDescriptor& resolve(std::string_view name) {
        if (name == "swizzled_w4a16") {
            return swizzled_w4a16();
        }
        throw std::invalid_argument(
            "ExpertBackendRegistry: unsupported weight backend '" + std::string(name) + "'");
    }

    static const ExpertBackendDescriptor& resolve(ExpertFormatKind kind) {
        if (kind == ExpertFormatKind::SWIZZLED_W4A16) {
            return swizzled_w4a16();
        }
        throw std::invalid_argument(
            "ExpertBackendRegistry: no backend is registered for the requested expert format");
    }

    static const ExpertBackendDescriptor& resolve(
        std::string_view name,
        const ExpertFormatDescriptor& format
    ) {
        const auto& backend = resolve(name);
        backend.validate_format(format);
        return backend;
    }

    static const ExpertBackendDescriptor& resolve(const ExpertFormatDescriptor& format) {
        const auto& backend = resolve(format.kind);
        backend.validate_format(format);
        return backend;
    }

private:
    static const ExpertBackendDescriptor& swizzled_w4a16() {
        static const ExpertBackendDescriptor descriptor{
            "swizzled_w4a16",
            ExpertFormatKind::SWIZZLED_W4A16,
            AEON_SWIZZLED_EXPERT_FORMAT_VERSION,
            AEON_SWIZZLED_EXPERT_BYTES,
            true
        };
        return descriptor;
    }
};

} // namespace aeon::core