#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace aeon {

struct SwizzleCfg {
    int rpw;
    int lpr;
};

static constexpr SwizzleCfg kCfgW13{4, 8};
static constexpr SwizzleCfg kCfgW2{8, 4};
static constexpr int kNibblePerm[8] = {0, 2, 4, 6, 1, 3, 5, 7};

inline uint32_t permute_w4a16_word(uint32_t source) {
    uint32_t destination = 0;
    for (int position = 0; position < 8; ++position) {
        const uint32_t nibble = (source >> (4 * kNibblePerm[position])) & 0xFu;
        destination |= nibble << (4 * position);
    }
    return destination;
}

inline uint32_t unpermute_w4a16_word(uint32_t source) {
    uint32_t destination = 0;
    for (int position = 0; position < 8; ++position) {
        const uint32_t nibble = (source >> (4 * position)) & 0xFu;
        destination |= nibble << (4 * kNibblePerm[position]);
    }
    return destination;
}

inline void validate_w4a16_swizzle(int N, int K, SwizzleCfg cfg) {
    assert(cfg.rpw > 0 && cfg.lpr > 0);
    assert(cfg.rpw * cfg.lpr == 32);
    assert(N > 0 && K > 0);
    assert(N % cfg.rpw == 0);
    assert(K % 32 == 0);
    assert((K / 32) % cfg.lpr == 0);
}

template <typename scale_t>
void swizzle_w4a16(
    const uint32_t* source_packed,
    const scale_t* source_scale,
    uint32_t* destination_packed,
    scale_t* destination_scale,
    int N,
    int K,
    SwizzleCfg cfg
) {
    validate_w4a16_swizzle(N, K, cfg);

    const int groups = K / 32;
    const int iterations = groups / cfg.lpr;
    const int row_blocks = N / cfg.rpw;
    const int source_words_per_row = K / 8;

    for (int block = 0; block < row_blocks; ++block) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (int row = 0; row < cfg.rpw; ++row) {
                for (int slice = 0; slice < cfg.lpr; ++slice) {
                    const int source_row = block * cfg.rpw + row;
                    const int source_group = iteration * cfg.lpr + slice;
                    const int lane = row * cfg.lpr + slice;
                    const std::size_t destination_u4 =
                        (static_cast<std::size_t>(block) * iterations + iteration) * 32 + lane;
                    const std::size_t source_word =
                        static_cast<std::size_t>(source_row) * source_words_per_row +
                        static_cast<std::size_t>(source_group) * 4;

                    for (int word = 0; word < 4; ++word) {
                        destination_packed[destination_u4 * 4 + word] =
                            permute_w4a16_word(source_packed[source_word + word]);
                    }
                    destination_scale[destination_u4] =
                        source_scale[static_cast<std::size_t>(source_row) * groups + source_group];
                }
            }
        }
    }
}

template <typename scale_t>
void unswizzle_w4a16(
    const uint32_t* source_packed,
    const scale_t* source_scale,
    uint32_t* destination_packed,
    scale_t* destination_scale,
    int N,
    int K,
    SwizzleCfg cfg
) {
    validate_w4a16_swizzle(N, K, cfg);

    const int groups = K / 32;
    const int iterations = groups / cfg.lpr;
    const int row_blocks = N / cfg.rpw;
    const int destination_words_per_row = K / 8;

    for (int block = 0; block < row_blocks; ++block) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (int row = 0; row < cfg.rpw; ++row) {
                for (int slice = 0; slice < cfg.lpr; ++slice) {
                    const int destination_row = block * cfg.rpw + row;
                    const int destination_group = iteration * cfg.lpr + slice;
                    const int lane = row * cfg.lpr + slice;
                    const std::size_t source_u4 =
                        (static_cast<std::size_t>(block) * iterations + iteration) * 32 + lane;
                    const std::size_t destination_word =
                        static_cast<std::size_t>(destination_row) * destination_words_per_row +
                        static_cast<std::size_t>(destination_group) * 4;

                    for (int word = 0; word < 4; ++word) {
                        destination_packed[destination_word + word] =
                            unpermute_w4a16_word(source_packed[source_u4 * 4 + word]);
                    }
                    destination_scale[static_cast<std::size_t>(destination_row) * groups +
                                     destination_group] = source_scale[source_u4];
                }
            }
        }
    }
}

} // namespace aeon