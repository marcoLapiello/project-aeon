#pragma once

// Hyperparameters for DeepSeek-V4 attention, shared by the attention kernel
// modules and their callers. Included by `v4_attention.hpp`; each kernel module
// includes it directly so it reads on its own.

#include <cstdint>

namespace aeon::kernel {

// Hyperparameters for DeepSeek-V4 Attention
constexpr uint32_t DSV4_HIDDEN_SIZE = 4096;
constexpr uint32_t DSV4_NUM_HEADS = 64;
constexpr uint32_t DSV4_HEAD_DIM = 512;
constexpr uint32_t DSV4_ROPE_DIM = 64;
constexpr uint32_t DSV4_NOPE_DIM = DSV4_HEAD_DIM - DSV4_ROPE_DIM; // 448
constexpr uint32_t DSV4_Q_LORA_RANK = 1024;
constexpr uint32_t DSV4_O_GROUPS = 8;
constexpr uint32_t DSV4_O_LORA_RANK = 1024;
constexpr uint32_t DSV4_HEADS_PER_GROUP = DSV4_NUM_HEADS / DSV4_O_GROUPS; // 8
constexpr uint32_t DSV4_GROUP_HEADS_DIM = DSV4_HEADS_PER_GROUP * DSV4_HEAD_DIM; // 4096
constexpr uint32_t DSV4_TOTAL_O_LORA_DIM = DSV4_O_GROUPS * DSV4_O_LORA_RANK; // 8192
constexpr uint32_t DSV4_SLIDING_WINDOW = 128;
constexpr uint32_t DSV4_INDEX_N_HEADS = 64;
constexpr uint32_t DSV4_INDEX_HEAD_DIM = 128;
constexpr uint32_t DSV4_INDEX_TOPK = 512;
constexpr uint32_t DSV4_MAX_ATTENTION_KEYS = DSV4_SLIDING_WINDOW + DSV4_INDEX_TOPK;
constexpr float DSV4_ROPE_THETA = 10000.0f;
constexpr float DSV4_ATTN_SCALE = 0.04419417382415922f; // 1.0f / sqrtf(512.0f)

// ---------------------------------------------------------------------------
// RoPE and YaRN configuration: see `v4_rope.hpp` (`RopeTable`).

} // namespace aeon::kernel
