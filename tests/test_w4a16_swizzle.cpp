#include "backend/swizzled_w4a16/kernels/aeon_w4a16_swizzle.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct ShapeCase {
    const char* name;
    int N;
    int K;
    aeon::SwizzleCfg cfg;
};

uint32_t make_source_word(std::size_t word_index) {
    uint32_t word = 0;
    for (int nibble = 0; nibble < 8; ++nibble) {
        const uint32_t value = static_cast<uint32_t>((word_index * 3 + nibble * 5 + 1) % 16);
        word |= value << (4 * nibble);
    }
    return word;
}

void verify_case(const ShapeCase& shape) {
    const int groups = shape.K / 32;
    const std::size_t packed_words = static_cast<std::size_t>(shape.N) * shape.K / 8;
    const std::size_t scales = static_cast<std::size_t>(shape.N) * groups;

    std::vector<uint32_t> source_packed(packed_words);
    std::vector<uint32_t> swizzled_packed(packed_words);
    std::vector<uint32_t> restored_packed(packed_words);
    std::vector<float> source_scale(scales);
    std::vector<float> swizzled_scale(scales);
    std::vector<float> restored_scale(scales);

    for (std::size_t index = 0; index < source_packed.size(); ++index) {
        source_packed[index] = make_source_word(index);
    }
    for (std::size_t index = 0; index < source_scale.size(); ++index) {
        source_scale[index] = 0.03125f + static_cast<float>(index % 17) * 0.0078125f;
    }

    aeon::swizzle_w4a16(
        source_packed.data(), source_scale.data(),
        swizzled_packed.data(), swizzled_scale.data(),
        shape.N, shape.K, shape.cfg);
    aeon::unswizzle_w4a16(
        swizzled_packed.data(), swizzled_scale.data(),
        restored_packed.data(), restored_scale.data(),
        shape.N, shape.K, shape.cfg);

    assert(restored_packed == source_packed);
    assert(restored_scale == source_scale);

    int inverse_permutation[8]{};
    for (int position = 0; position < 8; ++position) {
        inverse_permutation[aeon::kNibblePerm[position]] = position;
    }

    const int iterations = groups / shape.cfg.lpr;
    const int source_words_per_row = shape.K / 8;
    for (int row = 0; row < shape.N; ++row) {
        for (int k = 0; k < shape.K; ++k) {
            const int group = k / 32;
            const int block = row / shape.cfg.rpw;
            const int row_in_block = row % shape.cfg.rpw;
            const int slice = group % shape.cfg.lpr;
            const int iteration = group / shape.cfg.lpr;
            const int lane = row_in_block * shape.cfg.lpr + slice;
            const std::size_t swizzled_u4 =
                (static_cast<std::size_t>(block) * iterations + iteration) * 32 + lane;
            const int word_in_group = (k % 32) / 8;
            const int nibble_in_word = k % 8;

            const uint32_t source_word =
                source_packed[static_cast<std::size_t>(row) * source_words_per_row +
                              static_cast<std::size_t>(group) * 4 + word_in_group];
            const int source_nibble = (source_word >> (4 * nibble_in_word)) & 0xF;
            const uint32_t swizzled_word = swizzled_packed[swizzled_u4 * 4 + word_in_group];
            const int swizzled_nibble =
                (swizzled_word >> (4 * inverse_permutation[nibble_in_word])) & 0xF;

            const float source_value =
                static_cast<float>(source_nibble - 8) * source_scale[static_cast<std::size_t>(row) * groups + group];
            const float swizzled_value =
                static_cast<float>(swizzled_nibble - 8) * swizzled_scale[swizzled_u4];
            assert(source_nibble == swizzled_nibble);
            assert(std::abs(source_value - swizzled_value) == 0.0f);
        }
    }

    std::cout << "[PASS] " << shape.name << " swizzle round-trip and dequantization identity" << std::endl;
}

} // namespace

int main() {
    verify_case({"W1/W3", 2048, 4096, aeon::kCfgW13});
    verify_case({"W2", 4096, 2048, aeon::kCfgW2});
    std::cout << "[PASS] W4A16 swizzle layout contract" << std::endl;
    return 0;
}