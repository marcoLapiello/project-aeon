#pragma once

#include "core/expert_format.hpp"

#include <cstddef>

namespace aeon::core {

inline constexpr uint32_t AEON_SWIZZLED_EXPERT_FORMAT_VERSION = 2;

inline constexpr size_t AEON_W1_PACKED_OFFSET = 0;
inline constexpr size_t AEON_W1_SCALE_OFFSET  = 4194304;
inline constexpr size_t AEON_W2_PACKED_OFFSET = 4718592;
inline constexpr size_t AEON_W2_SCALE_OFFSET  = 8912896;
inline constexpr size_t AEON_W3_PACKED_OFFSET = 9437184;
inline constexpr size_t AEON_W3_SCALE_OFFSET  = 13631488;

inline constexpr size_t AEON_W1_PACKED_BYTES = 2048 * 512 * sizeof(uint32_t);
inline constexpr size_t AEON_W1_SCALE_BYTES  = 2048 * 128 * sizeof(uint16_t);
inline constexpr size_t AEON_W2_PACKED_BYTES = 4096 * 256 * sizeof(uint32_t);
inline constexpr size_t AEON_W2_SCALE_BYTES  = 4096 * 64 * sizeof(uint16_t);
inline constexpr size_t AEON_W3_PACKED_BYTES = 2048 * 512 * sizeof(uint32_t);
inline constexpr size_t AEON_W3_SCALE_BYTES  = 2048 * 128 * sizeof(uint16_t);

inline constexpr size_t AEON_SWIZZLED_EXPERT_BYTES =
    AEON_W1_PACKED_BYTES + AEON_W1_SCALE_BYTES +
    AEON_W2_PACKED_BYTES + AEON_W2_SCALE_BYTES +
    AEON_W3_PACKED_BYTES + AEON_W3_SCALE_BYTES;

static_assert(AEON_SWIZZLED_EXPERT_BYTES == AEON_EXPERT_BYTES);

} // namespace aeon::core