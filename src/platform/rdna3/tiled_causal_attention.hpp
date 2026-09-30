#pragma once

// -----------------------------------------------------------------------------
// G2 (RDNA3): tiled causal attention over a shared key row-set.
//
// The primitive answers one question for a whole tile of queries at once: given a
// **union** of key rows (with their positions) and a query tile (with its
// positions), attend each query over the union masked to its own causal window.
// It is the batched form of the per-query attention kernel, and it is deliberately
// model-agnostic: it knows about positions, a causal window, a per-head scale, and
// an optional per-head **bias**, and nothing else.
//
// ## Why one union and a mask
//
// A tile's queries share almost all of their keys — for a sliding window of `W`
// over a tile of `T` queries the union is `W + T - 1` rows, not `T × W`. Composing
// one union and masking each query into it is what lets a single launch serve the
// whole tile, which is the point: the per-query launch left the grid at one block
// per head. `tests/test_v4_tiled_attention_oracle.cpp` pins the reformulation in
// fp64 — masking the union is an **exact selection** of each query's window, not an
// approximation, because the surviving terms keep their order.
//
// ## The bias is a zero-value softmax candidate
//
// `bias[h]` is an optional extra score for head `h` that enters the max and the
// denominator but contributes **no value row** — the upstream reading of the DSV4
// attention sink as *"a virtual extra K with V=0"*. Naming it `bias` and not `sink`
// is deliberate: the G2 primitive must not know it is a sink, only that it is a
// scalar score with no value. A null `bias` is the no-candidate case and reduces to
// ordinary causal attention.
//
// ## Layout and bounds
//
// Queries and outputs are `[count, num_heads, head_dim]` with a **row pitch**
// (`q_stride`, `out_stride`) so a caller whose queries live inside a larger
// per-token buffer reads them in place. Keys and values are `[union_rows, stride]`.
// `head_dim` must be a multiple of the wave width (`32`) and at most `64 × 32`.
// Values may alias keys (MLA's `V = K`); the primitive does not assume it.
//
// Accumulation is fp32 and the result rounds to fp16 once. The softmax keeps the
// per-query kernel's summation order — an exact tree max, a strided `expf`, and an
// ascending denominator — so this kernel and the scalar one it generalizes differ
// only where the tile genuinely reorders, not in the arithmetic.
// -----------------------------------------------------------------------------

#include "platform/rdna3/wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace aeon::rdna3 {

constexpr int kCausalAttentionLanes = kWmmaLaneCount;
constexpr int kCausalAttentionMaxSlice = 64;              // head_dim / 32
constexpr int kCausalAttentionMaxHeadDim = kCausalAttentionMaxSlice * kCausalAttentionLanes;

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    !defined(__HIP_DEVICE_COMPILE__)

// One block per (head, query). The tile is the grid's `y` axis, so a chunk is one
// launch. `window <= 0` means full causal (no sliding restriction).
__global__ void __launch_bounds__(kCausalAttentionLanes)
causal_attention_fp16_wave32_kernel(
    const __half* __restrict__ q, int q_stride,
    const __half* __restrict__ keys, int key_stride,
    const __half* __restrict__ values, int value_stride,
    const int64_t* __restrict__ key_positions,
    const int64_t* __restrict__ query_positions,
    __half* __restrict__ out, int out_stride,
    int num_heads, int head_dim, int union_rows, int window,
    const float* __restrict__ bias, float scale) {
    extern __shared__ float scores[];

    const int head = blockIdx.x;
    const int query = blockIdx.y;
    const int lane = threadIdx.x;
    const int slice = head_dim / kCausalAttentionLanes;

    const int64_t query_position = query_positions[query];
    const int64_t window_first = window > 0
        ? (query_position >= window - 1 ? query_position - (window - 1) : 0)
        : 0;

    const __half* query_row =
        q + static_cast<size_t>(query) * q_stride + static_cast<size_t>(head) * head_dim;

    // The query is loop-invariant, so it is read once into registers rather than
    // re-read for every key row.
    float query_reg[kCausalAttentionMaxSlice];
    #pragma unroll 4
    for (int i = 0; i < slice; ++i) {
        query_reg[i] = __half2float(query_row[lane * slice + i]);
    }

    // Phase 1: masked scaled dot products.
    for (int row = 0; row < union_rows; ++row) {
        const int64_t key_position = key_positions[row];
        const bool valid = key_position >= window_first && key_position <= query_position;
        float dot = 0.0f;
        if (valid) {
            const __half* key_row = keys + static_cast<size_t>(row) * key_stride;
            #pragma unroll 4
            for (int i = 0; i < slice; ++i) {
                dot += query_reg[i] * __half2float(key_row[lane * slice + i]);
            }
            #pragma unroll
            for (int offset = kCausalAttentionLanes / 2; offset > 0; offset /= 2) {
                dot += __shfl_xor(dot, offset, kCausalAttentionLanes);
            }
        }
        if (lane == 0) scores[row] = valid ? dot * scale : -INFINITY;
    }
    __syncthreads();

    // Phase 2: softmax. The max is an exact tree, the `expf` is strided off the
    // redundant lockstep issue, and the denominator keeps its ascending walk — the
    // same three choices the per-query kernel makes, so the results stay comparable.
    const float head_bias = bias != nullptr ? bias[head] : -INFINITY;

    float thread_max = head_bias;
    for (int row = lane; row < union_rows; row += kCausalAttentionLanes) {
        thread_max = fmaxf(thread_max, scores[row]);
    }
    #pragma unroll
    for (int offset = kCausalAttentionLanes / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_xor(thread_max, offset, kCausalAttentionLanes));
    }
    const float maximum = thread_max;

    for (int row = lane; row < union_rows; row += kCausalAttentionLanes) {
        scores[row] = expf(scores[row] - maximum);
    }
    __syncthreads();

    // A masked row holds `-INFINITY`, so its `expf` is exactly `0`. The bias is one
    // term, added once, outside the reduction.
    float denominator = expf(head_bias - maximum);
    for (int row = 0; row < union_rows; ++row) {
        denominator += scores[row];
    }
    const float inverse_denominator = 1.0f / fmaxf(denominator, 1e-30f);
    __syncthreads();

    // Phase 3: weighted sum of values.
    __half* out_row =
        out + static_cast<size_t>(query) * out_stride + static_cast<size_t>(head) * head_dim;
    for (int d = lane * slice; d < (lane + 1) * slice; ++d) {
        float acc = 0.0f;
        for (int row = 0; row < union_rows; ++row) {
            const float weight = scores[row] * inverse_denominator;
            acc += weight * __half2float(values[static_cast<size_t>(row) * value_stride + d]);
        }
        out_row[d] = __float2half(acc);
    }
}

#endif  // gfx11 device pass, or any host pass

inline void dispatch_causal_attention_fp16(
    const __half* q, int q_stride,
    const __half* keys, int key_stride,
    const __half* values, int value_stride,
    const int64_t* key_positions,
    const int64_t* query_positions,
    __half* out, int out_stride,
    int count, int num_heads, int head_dim, int union_rows, int window,
    const float* bias, float scale, hipStream_t stream) {
    if (count <= 0 || num_heads <= 0 || union_rows <= 0) return;
    if (head_dim <= 0 || head_dim % kCausalAttentionLanes != 0 ||
        head_dim > kCausalAttentionMaxHeadDim) {
        throw std::invalid_argument("dispatch_causal_attention_fp16: unsupported head_dim");
    }
    if (q == nullptr || keys == nullptr || values == nullptr || out == nullptr ||
        key_positions == nullptr || query_positions == nullptr) {
        throw std::invalid_argument("dispatch_causal_attention_fp16: null buffer");
    }
    // A row pitch narrower than the row it holds would silently read a neighbour's
    // data, so it is rejected rather than clamped.
    if (q_stride < num_heads * head_dim || out_stride < num_heads * head_dim) {
        throw std::invalid_argument("dispatch_causal_attention_fp16: row pitch too small");
    }

    const dim3 grid(num_heads, count);
    const dim3 block(kCausalAttentionLanes);
    const size_t shared = static_cast<size_t>(union_rows) * sizeof(float);
    causal_attention_fp16_wave32_kernel<<<grid, block, shared, stream>>>(
        q, q_stride, keys, key_stride, values, value_stride,
        key_positions, query_positions, out, out_stride,
        num_heads, head_dim, union_rows, window, bias, scale);
}

} // namespace aeon::rdna3
