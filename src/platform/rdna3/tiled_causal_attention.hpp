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
// per-token buffer reads them in place. Keys and values are `[rows, stride]` with a
// stride per block. `head_dim` must be a multiple of the wave width (`32`) and at most
// `64 × 32`. Values may alias keys (MLA's `V = K`); the primitive does not assume it.
//
// ## Two blocks
//
// A caller supplies up to two key blocks. This is what lets one launch serve a layer
// whose attention is a *recent window plus an older compressed set*: block 0 carries
// the window (masked to `[query - W + 1, query]`), block 1 the compressed rows (window
// `0`, i.e. causal only). They share one softmax and one denominator, so the split is
// exact — it is the same arithmetic as reading one concatenated set. A caller with a
// single set passes it as block 0 and leaves block 1 empty.
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
// launch.
//
// Two key blocks, each with its **own window**: block 0 is the recent window (the
// sliding-window rows), block 1 is an optional older set with no sliding restriction
// (`window1 <= 0` is full causal) — the model's compressed rows, which lie outside
// the window but inside the causal past. Both share one softmax and one denominator.
// Block 1 may be empty (`rows1 == 0`), which reduces to plain sliding attention.
//
// Each block is `[rows, stride]` with its own row pitch. Values may alias keys, and
// block 1 may alias block 0's buffers when `rows1 == 0`.
__global__ void __launch_bounds__(kCausalAttentionLanes)
causal_attention_fp16_wave32_kernel(
    const __half* __restrict__ q, int q_stride,
    const __half* __restrict__ keys0, int key_stride0,
    const __half* __restrict__ values0, int value_stride0,
    const int64_t* __restrict__ positions0, int rows0, int window0,
    const __half* __restrict__ keys1, int key_stride1,
    const __half* __restrict__ values1, int value_stride1,
    const int64_t* __restrict__ positions1, int rows1, int window1,
    int64_t query_position_base, int64_t query_position_stride,
    __half* __restrict__ out, int out_stride,
    int num_heads, int head_dim,
    const float* __restrict__ bias, float scale) {
    extern __shared__ float scores[];

    const int head = blockIdx.x;
    const int query = blockIdx.y;
    const int lane = threadIdx.x;
    const int slice = head_dim / kCausalAttentionLanes;

    const int64_t query_position =
        query_position_base + static_cast<int64_t>(query) * query_position_stride;
    // A window of 0 (or less) is no window: the causal mask alone bounds the block.
    const int64_t window_first0 = window0 > 0
        ? (query_position >= window0 - 1 ? query_position - (window0 - 1) : 0)
        : 0;
    const int64_t window_first1 = window1 > 0
        ? (query_position >= window1 - 1 ? query_position - (window1 - 1) : 0)
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

    // Phase 1: masked scaled dot products, into one score array shared by both blocks.
    for (int row = 0; row < rows0; ++row) {
        const int64_t key_position = positions0[row];
        const bool valid = key_position >= window_first0 && key_position <= query_position;
        float dot = 0.0f;
        if (valid) {
            const __half* key_row = keys0 + static_cast<size_t>(row) * key_stride0;
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

    for (int row = 0; row < rows1; ++row) {
        const int64_t key_position = positions1[row];
        const bool valid = key_position >= window_first1 && key_position <= query_position;
        float dot = 0.0f;
        if (valid) {
            const __half* key_row = keys1 + static_cast<size_t>(row) * key_stride1;
            #pragma unroll 4
            for (int i = 0; i < slice; ++i) {
                dot += query_reg[i] * __half2float(key_row[lane * slice + i]);
            }
            #pragma unroll
            for (int offset = kCausalAttentionLanes / 2; offset > 0; offset /= 2) {
                dot += __shfl_xor(dot, offset, kCausalAttentionLanes);
            }
        }
        if (lane == 0) scores[rows0 + row] = valid ? dot * scale : -INFINITY;
    }
    __syncthreads();

    const int total_rows = rows0 + rows1;

    // Phase 2: softmax. The max is an exact tree, the `expf` is strided off the
    // redundant lockstep issue, and the denominator keeps its ascending walk — the
    // same three choices the per-query kernel makes, so the results stay comparable.
    const float head_bias = bias != nullptr ? bias[head] : -INFINITY;

    float thread_max = head_bias;
    for (int row = lane; row < total_rows; row += kCausalAttentionLanes) {
        thread_max = fmaxf(thread_max, scores[row]);
    }
    #pragma unroll
    for (int offset = kCausalAttentionLanes / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_xor(thread_max, offset, kCausalAttentionLanes));
    }
    const float maximum = thread_max;

    for (int row = lane; row < total_rows; row += kCausalAttentionLanes) {
        scores[row] = expf(scores[row] - maximum);
    }
    __syncthreads();

    // A masked row holds `-INFINITY`, so its `expf` is exactly `0`. The bias is one
    // term, added once, outside the reduction.
    float denominator = expf(head_bias - maximum);
    for (int row = 0; row < total_rows; ++row) {
        denominator += scores[row];
    }
    const float inverse_denominator = 1.0f / fmaxf(denominator, 1e-30f);
    __syncthreads();

    // Phase 3: weighted sum of values over both blocks. A masked row scored `0`, so it
    // contributes nothing without needing a second validity check.
    __half* out_row =
        out + static_cast<size_t>(query) * out_stride + static_cast<size_t>(head) * head_dim;
    for (int d = lane * slice; d < (lane + 1) * slice; ++d) {
        float acc = 0.0f;
        for (int row = 0; row < rows0; ++row) {
            acc += (scores[row] * inverse_denominator) *
                   __half2float(values0[static_cast<size_t>(row) * value_stride0 + d]);
        }
        for (int row = 0; row < rows1; ++row) {
            acc += (scores[rows0 + row] * inverse_denominator) *
                   __half2float(values1[static_cast<size_t>(row) * value_stride1 + d]);
        }
        out_row[d] = __float2half(acc);
    }
}

#endif  // gfx11 device pass, or any host pass

// The key blocks a launch reads. Block 1 is optional; an empty block 1 (or `rows == 0`
// on block 0) is allowed. `window <= 0` on a block means no sliding restriction.
struct CausalAttentionBlock {
    const __half* keys{nullptr};
    const __half* values{nullptr};   // may alias `keys` (MLA's V = K)
    const int64_t* positions{nullptr};
    int rows{0};
    int key_stride{0};
    int value_stride{0};
    int window{0};
};

inline void dispatch_causal_attention_fp16(
    const __half* q, int q_stride,
    const CausalAttentionBlock& block0,
    const CausalAttentionBlock& block1,
    int64_t query_position_base, int64_t query_position_stride,
    __half* out, int out_stride,
    int count, int num_heads, int head_dim,
    const float* bias, float scale, hipStream_t stream) {
    if (count <= 0 || num_heads <= 0) return;
    const int total_rows = block0.rows + block1.rows;
    if (total_rows <= 0) return;
    if (head_dim <= 0 || head_dim % kCausalAttentionLanes != 0 ||
        head_dim > kCausalAttentionMaxHeadDim) {
        throw std::invalid_argument("dispatch_causal_attention_fp16: unsupported head_dim");
    }
    if (q == nullptr || out == nullptr) {
        throw std::invalid_argument("dispatch_causal_attention_fp16: null buffer");
    }
    // Block 0 carries the required buffers even when it is empty; either block that
    // declares rows must name its key, value and position arrays.
    if (block0.keys == nullptr || block0.values == nullptr || block0.positions == nullptr) {
        throw std::invalid_argument("dispatch_causal_attention_fp16: null block 0");
    }
    if (block1.rows > 0 &&
        (block1.keys == nullptr || block1.values == nullptr || block1.positions == nullptr)) {
        throw std::invalid_argument("dispatch_causal_attention_fp16: null block 1");
    }
    // A row pitch narrower than the row it holds would silently read a neighbour's
    // data, so it is rejected rather than clamped.
    if (q_stride < num_heads * head_dim || out_stride < num_heads * head_dim) {
        throw std::invalid_argument("dispatch_causal_attention_fp16: row pitch too small");
    }

    // An empty block 1 must still pass a non-null pointer the kernel never dereferences
    // (the loops do not run), so the required block 0 stands in.
    const __half* keys1 = block1.keys != nullptr ? block1.keys : block0.keys;
    const __half* values1 = block1.values != nullptr ? block1.values : block0.values;
    const int64_t* positions1 = block1.positions != nullptr ? block1.positions : block0.positions;

    const dim3 grid(num_heads, count);
    const dim3 block(kCausalAttentionLanes);
    const size_t shared = static_cast<size_t>(total_rows) * sizeof(float);
    causal_attention_fp16_wave32_kernel<<<grid, block, shared, stream>>>(
        q, q_stride,
        block0.keys, block0.key_stride, block0.values, block0.value_stride,
        block0.positions, block0.rows, block0.window,
        keys1, block1.key_stride, values1, block1.value_stride,
        positions1, block1.rows, block1.window,
        query_position_base, query_position_stride, out, out_stride,
        num_heads, head_dim, bias, scale);
}

constexpr int kCausalAttentionWarps = 4;

// Split-keys causal attention: the same result as the one-warp kernel, but the block
// owns several warps that scan the key range in parallel and combine a partial
// `(max, sum, weighted value)` through shared memory.
//
// Why: one warp per `(head, query)` left the key scan serial — a five-step wave
// reduction for *every* key — which the profile showed as the kernel's dominant cost,
// and it left a single-token launch at `num_heads` blocks, too few to fill the GPU. Both
// are fixed here: a block is still `(head, query)` but `kCausalAttentionWarps` warps
// split its keys, so the grid does not change while the per-block work parallelises.
//
// The second block may be given **per query** (`per_query_keys`, one `int` index into
// block 1 per query). That is how a `CSA` layer's indexer top-k is served — each query
// selects its own compressed rows — without a second kernel or a shared-key contract.
// Masked or out-of-range indices contribute nothing.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    !defined(__HIP_DEVICE_COMPILE__)
template <int WARPS>
__global__ void __launch_bounds__(WARPS * kCausalAttentionLanes)
causal_attention_split_fp16_kernel(
    const __half* __restrict__ q, int q_stride,
    const __half* __restrict__ keys0, int key_stride0,
    const __half* __restrict__ values0, int value_stride0,
    const int64_t* __restrict__ positions0, int rows0, int window0,
    const __half* __restrict__ keys1, int key_stride1,
    const __half* __restrict__ values1, int value_stride1,
    const int64_t* __restrict__ positions1, int rows1, int window1,
    const int32_t* __restrict__ per_query_keys, int per_query_count,
    int64_t query_position_base, int64_t query_position_stride,
    __half* __restrict__ out, int out_stride,
    int num_heads, int head_dim, const float* __restrict__ bias, float scale) {
    extern __shared__ float smem[];

    const int rows1_effective = per_query_keys != nullptr ? per_query_count : rows1;
    const int total_rows = rows0 + rows1_effective;

    float* scores = smem;                                        // [total_rows]
    float* warp_max = scores + total_rows;                       // [WARPS]
    float* warp_sum = warp_max + WARPS;                          // [WARPS]
    float* warp_out = warp_sum + WARPS;                          // [WARPS, head_dim]
    float* global_reduce = warp_out + WARPS * head_dim;          // [2]

    const int head = blockIdx.x;
    const int query = blockIdx.y;
    const int lane = threadIdx.x & (kCausalAttentionLanes - 1);
    const int warp = threadIdx.x / kCausalAttentionLanes;
    const int slice = head_dim / kCausalAttentionLanes;

    const int64_t query_position =
        query_position_base + static_cast<int64_t>(query) * query_position_stride;
    const int64_t window_first0 = window0 > 0
        ? (query_position >= window0 - 1 ? query_position - (window0 - 1) : 0) : 0;
    const int64_t window_first1 = window1 > 0
        ? (query_position >= window1 - 1 ? query_position - (window1 - 1) : 0) : 0;

    const __half* query_row =
        q + static_cast<size_t>(query) * q_stride + static_cast<size_t>(head) * head_dim;
    float query_reg[kCausalAttentionMaxSlice];
    #pragma unroll 4
    for (int i = 0; i < slice; ++i) {
        query_reg[i] = __half2float(query_row[lane * slice + i]);
    }

    // The row of block 1 for key `index`, honouring the per-query selection.
    auto block1_row = [&](int index) { return per_query_keys != nullptr
        ? per_query_keys[query * per_query_count + index] : index; };
    auto key_position_at = [&](int k, int64_t& position) -> bool {
        if (k < rows0) {
            position = positions0[k];
            return position >= window_first0 && position <= query_position;
        }
        const int row = block1_row(k - rows0);
        if (row < 0 || row >= rows1) return false;
        position = positions1[row];
        return position >= window_first1 && position <= query_position;
    };
    auto value_ptr_at = [&](int k) -> const __half* {
        if (k < rows0) return values0 + static_cast<size_t>(k) * value_stride0;
        const int row = block1_row(k - rows0);
        // A masked selection (`-1`) or an out-of-range index has no row to read; the
        // caller skips it rather than forming a pointer past the buffer.
        if (row < 0 || row >= rows1) return nullptr;
        return values1 + static_cast<size_t>(row) * value_stride1;
    };
    auto key_ptr_at = [&](int k) -> const __half* {
        return k < rows0 ? keys0 + static_cast<size_t>(k) * key_stride0
                         : keys1 + static_cast<size_t>(block1_row(k - rows0)) * key_stride1;
    };

    // Phase 1: this warp's keys, stride `WARPS`.
    for (int k = warp; k < total_rows; k += WARPS) {
        int64_t position = 0;
        const bool valid = key_position_at(k, position);
        float dot = 0.0f;
        if (valid) {
            const __half* key_row = key_ptr_at(k);
            #pragma unroll 4
            for (int i = 0; i < slice; ++i) {
                dot += query_reg[i] * __half2float(key_row[lane * slice + i]);
            }
            #pragma unroll
            for (int offset = kCausalAttentionLanes / 2; offset > 0; offset /= 2) {
                dot += __shfl_xor(dot, offset, kCausalAttentionLanes);
            }
        }
        if (lane == 0) scores[k] = valid ? dot * scale : -INFINITY;
    }
    __syncthreads();

    const float head_bias = bias != nullptr ? bias[head] : -INFINITY;

    // The softmax loops must be **lane-strided over the warp's own keys**. A warp's
    // keys are `{warp, warp + W, …}`, but phase 1 computes each key's dot with all 32
    // lanes (a key needs a full wave reduction), so leaving the softmax loop strided by
    // warp alone would make every lane sum the *same* keys and the `shfl` reduction
    // would then multiply the count by the wave width. Striding each lane over a
    // disjoint slice of the warp's key list keeps the reduction correct.
    const int warp_keys = total_rows > warp
        ? (total_rows - warp + WARPS - 1) / WARPS : 0;
    auto warp_key = [&](int j) { return warp + j * WARPS; };

    // Phase 2a: max, reduced within each warp then across the warps.
    float local_max = head_bias;
    for (int j = lane; j < warp_keys; j += kCausalAttentionLanes) {
        local_max = fmaxf(local_max, scores[warp_key(j)]);
    }
    #pragma unroll
    for (int offset = kCausalAttentionLanes / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset, kCausalAttentionLanes));
    }
    if (lane == 0) warp_max[warp] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = head_bias;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) m = fmaxf(m, warp_max[w]);
        global_reduce[0] = m;
    }
    __syncthreads();
    const float maximum = global_reduce[0];

    // Phase 2b: `expf` and the denominator, same lane-stride.
    float local_sum = 0.0f;
    for (int j = lane; j < warp_keys; j += kCausalAttentionLanes) {
        const int k = warp_key(j);
        const float e = expf(scores[k] - maximum);
        scores[k] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int offset = kCausalAttentionLanes / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor(local_sum, offset, kCausalAttentionLanes);
    }
    if (lane == 0) warp_sum[warp] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = expf(head_bias - maximum);
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) s += warp_sum[w];
        global_reduce[1] = 1.0f / fmaxf(s, 1e-30f);
    }
    __syncthreads();
    const float inverse_denominator = global_reduce[1];

    // Phase 3: each warp accumulates its keys into a partial output row, combined after.
    for (int d = lane * slice; d < (lane + 1) * slice; ++d) {
        float acc = 0.0f;
        for (int k = warp; k < total_rows; k += WARPS) {
            const __half* value_row = value_ptr_at(k);
            if (value_row == nullptr) continue;
            acc += (scores[k] * inverse_denominator) * __half2float(value_row[d]);
        }
        warp_out[warp * head_dim + d] = acc;
    }
    __syncthreads();

    __half* out_row =
        out + static_cast<size_t>(query) * out_stride + static_cast<size_t>(head) * head_dim;
    for (int d = lane * slice; d < (lane + 1) * slice; ++d) {
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) acc += warp_out[w * head_dim + d];
        out_row[d] = __float2half(acc);
    }
}

#endif  // gfx11 device pass, or any host pass

template <int WARPS = kCausalAttentionWarps>
inline void dispatch_causal_attention_split_fp16(
    const __half* q, int q_stride,
    const CausalAttentionBlock& block0,
    const CausalAttentionBlock& block1,
    const int32_t* per_query_keys, int per_query_count,
    int64_t query_position_base, int64_t query_position_stride,
    __half* out, int out_stride,
    int count, int num_heads, int head_dim,
    const float* bias, float scale, hipStream_t stream) {
    if (count <= 0 || num_heads <= 0) return;
    const int rows1_effective = per_query_keys != nullptr ? per_query_count : block1.rows;
    const int total_rows = block0.rows + rows1_effective;
    if (total_rows <= 0) return;
    if (head_dim <= 0 || head_dim % kCausalAttentionLanes != 0 ||
        head_dim > kCausalAttentionMaxHeadDim) {
        throw std::invalid_argument("dispatch_causal_attention_split_fp16: unsupported head_dim");
    }
    if (q == nullptr || out == nullptr) {
        throw std::invalid_argument("dispatch_causal_attention_split_fp16: null buffer");
    }
    if (block0.keys == nullptr || block0.values == nullptr || block0.positions == nullptr) {
        throw std::invalid_argument("dispatch_causal_attention_split_fp16: null block 0");
    }
    if (rows1_effective > 0 &&
        (block1.keys == nullptr || block1.values == nullptr || block1.positions == nullptr)) {
        throw std::invalid_argument("dispatch_causal_attention_split_fp16: null block 1");
    }
    if (per_query_keys != nullptr && per_query_count <= 0) {
        throw std::invalid_argument("dispatch_causal_attention_split_fp16: empty selection");
    }
    if (q_stride < num_heads * head_dim || out_stride < num_heads * head_dim) {
        throw std::invalid_argument("dispatch_causal_attention_split_fp16: row pitch too small");
    }

    const __half* keys1 = block1.keys != nullptr ? block1.keys : block0.keys;
    const __half* values1 = block1.values != nullptr ? block1.values : block0.values;
    const int64_t* positions1 = block1.positions != nullptr ? block1.positions : block0.positions;

    const dim3 grid(num_heads, count);
    const dim3 block(WARPS * kCausalAttentionLanes);
    const size_t shared = (static_cast<size_t>(total_rows) +
                           2 * WARPS +
                           static_cast<size_t>(WARPS) * head_dim + 2) *
                          sizeof(float);
    causal_attention_split_fp16_kernel<WARPS><<<grid, block, shared, stream>>>(
        q, q_stride,
        block0.keys, block0.key_stride, block0.values, block0.value_stride,
        block0.positions, block0.rows, block0.window,
        keys1, block1.key_stride, values1, block1.value_stride,
        positions1, block1.rows, block1.window,
        per_query_keys, per_query_count,
        query_position_base, query_position_stride, out, out_stride,
        num_heads, head_dim, bias, scale);
}

constexpr int kCausalAttentionQueryTile = kWmmaTileM;   // 16

// WMMA QKᵀ for a **shared-key** query tile: the same arithmetic as the split kernel's
// scalar dot loop, but the `QKᵀ` tile is one `v_wmma_f32_16x16x16_f16_w32` per
// 16-key × 16-dim step instead of 16 per-key wave reductions.
//
// **Only QKᵀ is on the matrix cores; PV is not.** A `16 × 512` output tile is `8192`
// fp32 — `256` registers per lane at one wave, past gfx11's limit — so PV keeps the
// scalar form with a register-resident 16-query accumulator (`16` registers per lane,
// sweep below). That is a real ceiling, not a shortcut: a WMMA PV would need the output
// tile staged in LDS, which this primitive does not spend.
//
// One wave per `(head, query-tile)`. The tile's queries share the whole key union, so
// this serves `Sliding` and `HCA`; a `CSA` layer's per-query top-k has no shared key
// tile and stays on `causal_attention_split_fp16`.
//
// The 16×16 score tile has the instruction's lane map: lane `t`, slot `i` holds
// `C[2i + (t >> 4)][t & 15]`. Masking is applied as the tile is written to shared
// memory, so padding and out-of-window keys become `-INF` and the softmax ignores them.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    !defined(__HIP_DEVICE_COMPILE__)
__global__ void __launch_bounds__(kCausalAttentionLanes)
causal_attention_wmma_qk_fp16_kernel(
    const __half* __restrict__ q, int q_stride,
    const __half* __restrict__ keys0, int key_stride0,
    const __half* __restrict__ values0, int value_stride0,
    const int64_t* __restrict__ positions0, int rows0, int window0,
    const __half* __restrict__ keys1, int key_stride1,
    const __half* __restrict__ values1, int value_stride1,
    const int64_t* __restrict__ positions1, int rows1, int window1,
    int64_t query_position_base, int64_t query_position_stride,
    __half* __restrict__ out, int out_stride,
    int num_heads, int head_dim, int count,
    const float* __restrict__ bias, float scale) {
    extern __shared__ unsigned char raw[];
    const int total_rows = rows0 + rows1;
    const int tpad = (total_rows + kWmmaTileN - 1) & ~(kWmmaTileN - 1);
    float* scores = reinterpret_cast<float*>(raw);                   // [16][tpad]
    __half* probs = reinterpret_cast<__half*>(scores + 16 * tpad);   // [16][tpad]

    const int head = blockIdx.x;
    const int first_query = blockIdx.y * kCausalAttentionQueryTile;
    if (first_query >= count) return;
    const int lane = threadIdx.x;
    const int axis = wmma_lane_axis(lane);
    const int parity = wmma_lane_parity(lane);

    // Phase 1: QKᵀ over the union, one WMMA per (key tile, 16-dim step).
    for (int kt = 0; kt < tpad; kt += kWmmaTileN) {
        f32_vec8 acc = wmma_zero_accumulator();
        for (int kd = 0; kd < head_dim; kd += kWmmaTileK) {
            const int qrow = min(first_query + axis, count - 1);
            const f16_vec16 a = wmma_load_a_row(
                q + static_cast<size_t>(qrow) * q_stride +
                static_cast<size_t>(head) * head_dim + kd);
            const int kg = min(kt + axis, total_rows - 1);
            const __half* krow = kg < rows0
                ? keys0 + static_cast<size_t>(kg) * key_stride0
                : keys1 + static_cast<size_t>(kg - rows0) * key_stride1;
            const f16_vec16 b = wmma_load_a_row(krow + kd);
            acc = wmma_mma(a, b, acc);
        }
        // Apply the causal + window mask as the tile lands in shared memory.
        for (int i = 0; i < 8; ++i) {
            const int m = 2 * i + parity;
            const int kg = kt + axis;
            const int qidx = first_query + m;
            bool valid = false;
            if (qidx < count && kg < total_rows) {
                const int64_t qp = query_position_base +
                                   static_cast<int64_t>(qidx) * query_position_stride;
                const int64_t kp = kg < rows0 ? positions0[kg] : positions1[kg - rows0];
                const int w = kg < rows0 ? window0 : window1;
                const int64_t wf = w > 0 ? (qp >= w - 1 ? qp - (w - 1) : 0) : 0;
                valid = kp >= wf && kp <= qp;
            }
            scores[m * tpad + kg] = valid ? acc[i] * scale : -INFINITY;
        }
    }
    __syncthreads();

    // Phase 2: softmax per query row, one row per lane. The bias enters the max and the
    // denominator, exactly as the scalar kernels do.
    const float head_bias = bias != nullptr ? bias[head] : -INFINITY;
    if (lane < kCausalAttentionQueryTile) {
        const int m = lane;
        float mx = head_bias;
        for (int k = 0; k < tpad; ++k) mx = fmaxf(mx, scores[m * tpad + k]);
        float sum = expf(head_bias - mx);
        for (int k = 0; k < tpad; ++k) {
            const float p = expf(scores[m * tpad + k] - mx);
            scores[m * tpad + k] = p;
            sum += p;
        }
        const float inv = 1.0f / fmaxf(sum, 1e-30f);
        for (int k = 0; k < tpad; ++k) {
            probs[m * tpad + k] = __float2half(scores[m * tpad + k] * inv);
        }
    }
    __syncthreads();

    // Phase 3: PV, scalar and lane-strided. Each lane owns `head_dim / 32` output
    // dimensions and accumulates all 16 queries of the tile for each — `16` registers,
    // not the `256` a WMMA output tile of this width would need.
    const int slice = head_dim / kCausalAttentionLanes;
    for (int i = 0; i < slice; ++i) {
        const int d = lane * slice + i;
        float acc[kCausalAttentionQueryTile];
        #pragma unroll
        for (int m = 0; m < kCausalAttentionQueryTile; ++m) acc[m] = 0.0f;
        for (int k = 0; k < total_rows; ++k) {
            const __half* vrow = k < rows0
                ? values0 + static_cast<size_t>(k) * value_stride0
                : values1 + static_cast<size_t>(k - rows0) * value_stride1;
            const float v = __half2float(vrow[d]);
            #pragma unroll
            for (int m = 0; m < kCausalAttentionQueryTile; ++m) {
                acc[m] += __half2float(probs[m * tpad + k]) * v;
            }
        }
        #pragma unroll
        for (int m = 0; m < kCausalAttentionQueryTile; ++m) {
            const int qidx = first_query + m;
            if (qidx < count) {
                out[static_cast<size_t>(qidx) * out_stride +
                    static_cast<size_t>(head) * head_dim + d] = __float2half(acc[m]);
            }
        }
    }
}
#endif  // gfx11 device pass, or any host pass

inline void dispatch_causal_attention_wmma_qk_fp16(
    const __half* q, int q_stride,
    const CausalAttentionBlock& block0,
    const CausalAttentionBlock& block1,
    int64_t query_position_base, int64_t query_position_stride,
    __half* out, int out_stride,
    int count, int num_heads, int head_dim,
    const float* bias, float scale, hipStream_t stream) {
    if (count <= 0 || num_heads <= 0) return;
    const int total_rows = block0.rows + block1.rows;
    if (total_rows <= 0) return;
    // WMMA's K axis is 16 wide, so the head must be a multiple of it.
    if (head_dim <= 0 || head_dim % kWmmaTileK != 0) {
        throw std::invalid_argument("dispatch_causal_attention_wmma_qk_fp16: unsupported head_dim");
    }
    if (q == nullptr || out == nullptr) {
        throw std::invalid_argument("dispatch_causal_attention_wmma_qk_fp16: null buffer");
    }
    if (block0.keys == nullptr || block0.values == nullptr || block0.positions == nullptr) {
        throw std::invalid_argument("dispatch_causal_attention_wmma_qk_fp16: null block 0");
    }
    if (block1.rows > 0 &&
        (block1.keys == nullptr || block1.values == nullptr || block1.positions == nullptr)) {
        throw std::invalid_argument("dispatch_causal_attention_wmma_qk_fp16: null block 1");
    }
    if (q_stride < num_heads * head_dim || out_stride < num_heads * head_dim) {
        throw std::invalid_argument("dispatch_causal_attention_wmma_qk_fp16: row pitch too small");
    }

    const __half* keys1 = block1.keys != nullptr ? block1.keys : block0.keys;
    const __half* values1 = block1.values != nullptr ? block1.values : block0.values;
    const int64_t* positions1 = block1.positions != nullptr ? block1.positions : block0.positions;

    const int tpad = (total_rows + kWmmaTileN - 1) & ~(kWmmaTileN - 1);
    // The score tile is `16 x tpad` fp32 plus the same again as fp16 probabilities. A
    // launch past the workgroup's shared-memory limit does not fail loudly — `<<<>>>`
    // drops the error and the output stays stale — so it is refused here instead.
    const size_t shared = static_cast<size_t>(16) * tpad * (sizeof(float) + sizeof(__half));
    if (shared > 64 * 1024) {
        throw std::invalid_argument(
            "dispatch_causal_attention_wmma_qk_fp16: row-set exceeds LDS; use the split kernel");
    }
    const dim3 grid(num_heads, (count + kCausalAttentionQueryTile - 1) / kCausalAttentionQueryTile);
    causal_attention_wmma_qk_fp16_kernel<<<grid, kCausalAttentionLanes, shared, stream>>>(
        q, q_stride,
        block0.keys, block0.key_stride, block0.values, block0.value_stride,
        block0.positions, block0.rows, block0.window,
        keys1, block1.key_stride, values1, block1.value_stride,
        positions1, block1.rows, block1.window,
        query_position_base, query_position_stride, out, out_stride,
        num_heads, head_dim, count, bias, scale);
}

} // namespace aeon::rdna3