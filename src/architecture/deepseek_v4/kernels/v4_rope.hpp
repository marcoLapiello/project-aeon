#pragma once

// -----------------------------------------------------------------------------
// DeepSeek-V4 kept primitive: RoPE (Step 2.3).
//
// Extracted from the pre-rewrite attention header for the same reason RMSNorm
// was: a primitive must be gated on its own, outside the legacy graph, and there
// must be exactly one definition. `v4_attention.hpp` includes this header.
//
// The four properties this file must get right — each is a recorded trap:
//
//   1. TAIL, NOT HEAD. Only the LAST `rope_dim` (64) of each 512-wide head
//      rotates; the layout is `[nope (448) | rope (64)]`. The generic upstream
//      parent class rotates the FIRST `rotary_dim`; the DSV4 subclass overrides
//      it to rotate the last. Getting this wrong rotates the wrong dims and
//      passes every "does it run" check. (trap 27)
//   2. GPT-J INTERLEAVE, NOT NeoX. Adjacent pairs `(2k, 2k+1)` rotate together;
//      `is_neox_style=False`.
//   3. TWO BASES. `theta = compress_rope_theta (160000)` with YaRN when
//      `compress_ratio > 1`, else `rope_theta (10000)` plain. Two `RopeTable`
//      instances are built at model load, never one. (trap 7)
//   4. NO AMPLITUDE SCALING. Upstream sets `mscale = 0`, so the effective
//      magnitude factor is exactly 1.0. There is no `1 + 0.1*log(scale)` term.
//
// Gate: `tests/test_v4_rope_oracle.cpp`, against `reference/dsv4_oracle.hpp`.
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace aeon::kernel {

// ---------------------------------------------------------------------------
// RoPE and YaRN configuration
// ---------------------------------------------------------------------------
struct RopeTable {
    uint32_t max_seq_len{4096};
    uint32_t rope_dim{64};
    uint32_t half_rope{32}; // rope_dim / 2
    std::vector<float> cos_cache; // [max_seq_len, half_rope]
    std::vector<float> sin_cache; // [max_seq_len, half_rope]

    void init(
        uint32_t seq_len = 4096,
        float theta = 10000.0f,
        float factor = 1.0f,
        float beta_fast = 32.0f,
        float beta_slow = 1.0f,
        uint32_t orig_max_pos = 65536
    ) {
        max_seq_len = seq_len;
        cos_cache.resize(max_seq_len * half_rope);
        sin_cache.resize(max_seq_len * half_rope);

        for (uint32_t k = 0; k < half_rope; ++k) {
            float freq = 1.0f / std::pow(theta, (2.0f * k) / (float)rope_dim);
            if (factor > 1.0f) {
                // YaRN interpolates between the original and scaled
                // frequencies across the beta-derived correction range.
                constexpr float pi = 3.14159265358979323846f;
                const auto correction_dim = [this, theta, orig_max_pos](float rotations) {
                    return static_cast<float>(rope_dim) *
                        std::log(static_cast<float>(orig_max_pos) / (rotations * 2.0f * pi)) /
                        (2.0f * std::log(theta));
                };
                const float low = std::max(
                    0.0f, std::floor(correction_dim(beta_fast)));
                const float high = std::min(
                    static_cast<float>(half_rope - 1),
                    std::ceil(correction_dim(beta_slow)));
                if (low >= high) {
                    freq = static_cast<float>(k) < low ? freq : freq / factor;
                } else {
                    const float w = std::clamp(
                        (static_cast<float>(k) - low) / (high - low), 0.0f, 1.0f);
                    freq = (1.0f - w) * freq + w * (freq / factor);
                }
            }

            for (uint32_t pos = 0; pos < max_seq_len; ++pos) {
                float angle = pos * freq;
                cos_cache[pos * half_rope + k] = std::cos(angle);
                sin_cache[pos * half_rope + k] = std::sin(angle);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Device Kernels (RDNA3 Wave32)
// ---------------------------------------------------------------------------

// Forward GPT-J RoPE on the trailing `rope_dim` of [T, num_heads, head_dim].
//   out[2k] = x[2k]*cos - x[2k+1]*sin
//   out[2k+1] = x[2k]*sin + x[2k+1]*cos
__global__ void __launch_bounds__(32) v4_forward_rope_wave32_kernel(
    __half* __restrict__ vec,           // [num_tokens, num_heads, head_dim]
    const float* __restrict__ cos_cache,// [max_seq, half_rope]
    const float* __restrict__ sin_cache,// [max_seq, half_rope]
    int num_heads,
    int head_dim,                       // 512
    int nope_dim,                       // 448
    int half_rope                       // 32
) {
    int token_idx = blockIdx.y;
    int head_idx  = blockIdx.x;
    int k         = threadIdx.x; // 0..31 (one thread per frequency pair)

    if (k < half_rope) {
        int base_idx = token_idx * (num_heads * head_dim) + head_idx * head_dim + nope_dim + 2 * k;

        float c = cos_cache[token_idx * half_rope + k];
        float s = sin_cache[token_idx * half_rope + k];

        float x0 = __half2float(vec[base_idx + 0]);
        float x1 = __half2float(vec[base_idx + 1]);

        float rot0 = x0 * c - x1 * s;
        float rot1 = x0 * s + x1 * c;

        vec[base_idx + 0] = __float2half(rot0);
        vec[base_idx + 1] = __float2half(rot1);
    }
}

// Inverse GPT-J RoPE on the trailing `rope_dim`: the transpose of the forward
// rotation, equivalently forward with `sin` negated.
//   out[2k] = x[2k]*cos + x[2k+1]*sin
//   out[2k+1] = x[2k+1]*cos - x[2k]*sin
__global__ void __launch_bounds__(32) v4_inverse_rope_wave32_kernel(
    __half* __restrict__ vec,           // [num_tokens, num_heads, head_dim]
    const float* __restrict__ cos_cache,// [max_seq, half_rope]
    const float* __restrict__ sin_cache,// [max_seq, half_rope]
    int num_heads,                      // 64
    int head_dim,                       // 512
    int nope_dim,                       // 448
    int half_rope                       // 32
) {
    int token_idx = blockIdx.y;
    int head_idx  = blockIdx.x;
    int k         = threadIdx.x; // 0..31

    if (k < half_rope) {
        int base_idx = token_idx * (num_heads * head_dim) + head_idx * head_dim + nope_dim + 2 * k;

        float c = cos_cache[token_idx * half_rope + k];
        float s = sin_cache[token_idx * half_rope + k];

        float x0 = __half2float(vec[base_idx + 0]);
        float x1 = __half2float(vec[base_idx + 1]);

        float inv0 = x0 * c + x1 * s;
        float inv1 = x1 * c - x0 * s;

        vec[base_idx + 0] = __float2half(inv0);
        vec[base_idx + 1] = __float2half(inv1);
    }
}

// Single-position forward, for autoregressive decode on [num_heads, head_dim]
// and for any position in the table.
__global__ void __launch_bounds__(32) v4_forward_rope_at_pos_wave32_kernel(
    __half* __restrict__ vec,           // [num_heads, head_dim]
    const float* __restrict__ cos_cache,// [max_seq, half_rope]
    const float* __restrict__ sin_cache,// [max_seq, half_rope]
    int pos,
    int num_heads,
    int head_dim,                       // 512
    int nope_dim,                       // 448
    int half_rope                       // 32
) {
    int head_idx = blockIdx.x;
    int k        = threadIdx.x;

    if (k < half_rope) {
        int base_idx = head_idx * head_dim + nope_dim + 2 * k;

        float c = cos_cache[pos * half_rope + k];
        float s = sin_cache[pos * half_rope + k];

        float x0 = __half2float(vec[base_idx + 0]);
        float x1 = __half2float(vec[base_idx + 1]);

        float rot0 = x0 * c - x1 * s;
        float rot1 = x0 * s + x1 * c;

        vec[base_idx + 0] = __float2half(rot0);
        vec[base_idx + 1] = __float2half(rot1);
    }
}

// Single-position inverse, for the attention-output tail before the grouped
// projection (Step 2.5).
__global__ void __launch_bounds__(32) v4_inverse_rope_at_pos_wave32_kernel(
    __half* __restrict__ vec,           // [num_heads, head_dim]
    const float* __restrict__ cos_cache,// [max_seq, half_rope]
    const float* __restrict__ sin_cache,// [max_seq, half_rope]
    int pos,
    int num_heads,
    int head_dim,                       // 512
    int nope_dim,                       // 448
    int half_rope                       // 32
) {
    int head_idx = blockIdx.x;
    int k        = threadIdx.x;

    if (k < half_rope) {
        int base_idx = head_idx * head_dim + nope_dim + 2 * k;

        float c = cos_cache[pos * half_rope + k];
        float s = sin_cache[pos * half_rope + k];

        float x0 = __half2float(vec[base_idx + 0]);
        float x1 = __half2float(vec[base_idx + 1]);

        float inv0 = x0 * c + x1 * s;
        float inv1 = x1 * c - x0 * s;

        vec[base_idx + 0] = __float2half(inv0);
        vec[base_idx + 1] = __float2half(inv1);
    }
}

} // namespace aeon::kernel
