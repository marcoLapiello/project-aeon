#pragma once

#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aeon::reference {

enum class SwizzledMatrixKind : uint8_t {
    W1,
    W2,
    W3
};

struct SwizzledMatrixLayout {
    size_t packed_offset;
    size_t scale_offset;
    int rows;
    int columns;
    int rows_per_wave;
    int lanes_per_row;
};

inline SwizzledMatrixLayout matrix_layout(SwizzledMatrixKind kind) {
    switch (kind) {
        case SwizzledMatrixKind::W1:
            return {
                core::AEON_W1_PACKED_OFFSET,
                core::AEON_W1_SCALE_OFFSET,
                2048,
                4096,
                4,
                8
            };
        case SwizzledMatrixKind::W2:
            return {
                core::AEON_W2_PACKED_OFFSET,
                core::AEON_W2_SCALE_OFFSET,
                4096,
                2048,
                8,
                4
            };
        case SwizzledMatrixKind::W3:
            return {
                core::AEON_W3_PACKED_OFFSET,
                core::AEON_W3_SCALE_OFFSET,
                2048,
                4096,
                4,
                8
            };
    }
    throw std::invalid_argument("v4_int4_reference: unsupported matrix kind");
}

inline float fp16_to_float(uint16_t bits) noexcept {
    const uint32_t sign = static_cast<uint32_t>(bits >> 15);
    const uint32_t exponent = static_cast<uint32_t>((bits >> 10) & 0x1F);
    const uint32_t fraction = static_cast<uint32_t>(bits & 0x03FF);

    float value = 0.0f;
    if (exponent == 0) {
        value = fraction == 0
            ? 0.0f
            : std::ldexp(static_cast<float>(fraction), -24);
    } else if (exponent == 0x1F) {
        value = fraction == 0
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(
            1.0f + static_cast<float>(fraction) / 1024.0f,
            static_cast<int>(exponent) - 15);
    }
    return sign == 0 ? value : -value;
}

inline uint16_t load_u16(const uint8_t* address) noexcept {
    uint16_t value = 0;
    std::memcpy(&value, address, sizeof(value));
    return value;
}

inline uint32_t load_u32(const uint8_t* address) noexcept {
    uint32_t value = 0;
    std::memcpy(&value, address, sizeof(value));
    return value;
}

inline int inverse_nibble_position(int source_position) noexcept {
    constexpr int permutation[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    for (int destination_position = 0; destination_position < 8; ++destination_position) {
        if (permutation[destination_position] == source_position) {
            return destination_position;
        }
    }
    return 0;
}

inline void decode_gemv(
    const uint8_t* expert_payload,
    SwizzledMatrixKind kind,
    const float* activation,
    float* output
) {
    if (expert_payload == nullptr || activation == nullptr || output == nullptr) {
        throw std::invalid_argument("v4_int4_reference: null GEMV input");
    }

    const SwizzledMatrixLayout layout = matrix_layout(kind);
    const int groups = layout.columns / 32;
    const int iterations = groups / layout.lanes_per_row;
    const size_t words_per_group = 4;
    const uint8_t* packed = expert_payload + layout.packed_offset;
    const uint8_t* scales = expert_payload + layout.scale_offset;

    for (int row = 0; row < layout.rows; ++row) {
        float dot = 0.0f;
        const int row_block = row / layout.rows_per_wave;
        const int row_in_block = row % layout.rows_per_wave;
        for (int column = 0; column < layout.columns; ++column) {
            const int group = column / 32;
            const int word_in_group = (column % 32) / 8;
            const int source_nibble = column % 8;
            const int iteration = group / layout.lanes_per_row;
            const int slice = group % layout.lanes_per_row;
            const int lane = row_in_block * layout.lanes_per_row + slice;
            const size_t storage_u4 =
                (static_cast<size_t>(row_block) * iterations + iteration) * 32 + lane;
            const size_t packed_word = storage_u4 * words_per_group + word_in_group;
            const uint32_t word = load_u32(packed + packed_word * sizeof(uint32_t));
            const int swizzled_nibble = inverse_nibble_position(source_nibble);
            const int quantized = static_cast<int>((word >> (4 * swizzled_nibble)) & 0x0F);
            const float scale = fp16_to_float(
                load_u16(scales + storage_u4 * sizeof(uint16_t)));
            dot += activation[column] * static_cast<float>(quantized - 8) * scale;
        }
        output[row] = dot;
    }
}

inline void decode_fp16_gemv(
    const uint8_t* weight_data,
    const uint16_t* activation,
    int rows,
    int columns,
    float* output
) {
    if (weight_data == nullptr || activation == nullptr || output == nullptr ||
        rows <= 0 || columns <= 0) {
        throw std::invalid_argument("v4_int4_reference: invalid FP16 GEMV input");
    }

    for (int row = 0; row < rows; ++row) {
        float dot = 0.0f;
        for (int column = 0; column < columns; ++column) {
            const size_t weight_index =
                (static_cast<size_t>(row) * columns + column) * sizeof(uint16_t);
            dot += fp16_to_float(activation[column]) *
                   fp16_to_float(load_u16(weight_data + weight_index));
        }
        output[row] = dot;
    }
}

inline void decode_routed_ffn(
    const uint8_t* expert_payload,
    const float* activation,
    float* output,
    float swiglu_limit = 10.0f
) {
    std::vector<float> gate(2048);
    std::vector<float> up(2048);
    std::vector<float> hidden(2048);
    decode_gemv(expert_payload, SwizzledMatrixKind::W1, activation, gate.data());
    decode_gemv(expert_payload, SwizzledMatrixKind::W3, activation, up.data());

    for (size_t index = 0; index < hidden.size(); ++index) {
        const float clamped_gate = std::min(gate[index], swiglu_limit);
        const float clamped_up = std::min(std::max(up[index], -swiglu_limit), swiglu_limit);
        hidden[index] =
            (clamped_gate / (1.0f + std::exp(-clamped_gate))) * clamped_up;
    }
    decode_gemv(expert_payload, SwizzledMatrixKind::W2, hidden.data(), output);
}

} // namespace aeon::reference