#pragma once

// -----------------------------------------------------------------------------
// G4 binding: the DSV4 tiled causal attention over a chunk's rows.
//
// Binds the G2 tile primitive (`platform/tiled_causal_attention.hpp`) to the model:
// it supplies the full-head scale, the per-head attention sink, and the layer class's
// key sets, and it names no WMMA, lane or tile detail. The two key blocks are exactly
// what the primitive was given for:
//
//   * **Sliding** — one block, the local union (the sliding window).
//   * **HCA** — two: the local union, plus **every committed compressed row** with
//     window `0` (causal only), because HCA attends the whole compressed set and its
//     per-position causality is enforced by the mask, not by a top-k.
//   * **CSA** — deliberately unsupported. Its indexer selects a *different* top-k per
//     query, so there is no shared compressed block to tile; it keeps the scalar path
//     until 4.2b.
//
// The local union is a whole tile's row-set, composed once by the caller. Each query
// masks into its own window, which `tests/test_v4_tiled_attention_oracle.cpp` pins as
// an exact selection before any kernel runs.
// -----------------------------------------------------------------------------

#include "platform/tiled_causal_attention.hpp"

#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "architecture/deepseek_v4/layer/v4_layer.hpp"

namespace aeon::core {

// Whether this layer's attention can be served by one tiled launch. All three classes
// now can: the shared-key classes (`Sliding`, `HCA`) pass the compressed set whole, and
// `CSA` passes its indexer's per-query selection as an index block.
inline bool attention_tile_supported(const V4Layer& layer) {
    const V4AttentionKind kind = layer.spec().attention_kind;
    return kind == V4AttentionKind::Sliding || kind == V4AttentionKind::HCA ||
           kind == V4AttentionKind::CSA;
}

// Switch for the tiled attention, in the chunk path and in decode. It defaults to **on**.
// Setting it false replays the per-token scalar kernel, which is slower and exists so a
// gate can compare the two in one process (the bit-exact chunk-vs-serial gates run with
// it off, since the tile is a reorder; the parity gate runs with it on).
inline bool& attention_tile_enabled() {
    static bool enabled = true;
    return enabled;
}

// One tiled attention launch over `count` consecutive rows starting at
// `query_position_base` (row `r` is at position `query_position_base + r`).
//
// `q`/`out` are the chunk workspace's contiguous per-row buffers, `q_stride` the
// distance between rows (`total_q`). `local_keys`/`local_positions`/`local_rows` are
// the composed **union** row-set for the tile; `committed` is the number of compressed
// entries materialized by the time the chunk's attention runs (all of them, since the
// whole chunk's pre-attention half has completed).
//
// `per_query_keys`/`per_query_count` are the CSA case: one `int` index into the
// compressed cache per (query, candidate), so each query attends its own indexer top-k.
// The shared-key classes pass `nullptr` and read the compressed set whole. `Sliding`
// reads neither.
inline void run_attention_tile(
    const V4Layer& layer,
    const half* q, int q_stride,
    half* out, int out_stride,
    int64_t query_position_base,
    int count,
    const half* local_keys, const int64_t* local_positions, int local_rows,
    uint32_t committed,
    const int32_t* per_query_keys, int per_query_count,
    hipStream_t stream) {
    constexpr int HEAD_DIM = kernel::DSV4_HEAD_DIM;

    CausalAttentionBlock block0;
    block0.keys = local_keys;
    block0.values = local_keys;              // MLA: V = K
    block0.positions = local_positions;
    block0.rows = local_rows;
    block0.key_stride = HEAD_DIM;
    block0.value_stride = HEAD_DIM;
    // The local window: a query keeps only the keys within `capacity` of its position,
    // which the mask enforces. It is the ring capacity, which equals the sliding window
    // once the sequence is long enough.
    block0.window = static_cast<int>(layer.local_cache_capacity());

    CausalAttentionBlock block1;
    if (layer.spec().attention_kind != V4AttentionKind::Sliding && committed > 0) {
        block1.keys = layer.d_compressed_key_cache;
        block1.values = layer.d_compressed_value_cache;
        block1.positions = layer.d_compressed_positions;
        block1.rows = static_cast<int>(committed);
        block1.key_stride = HEAD_DIM;
        block1.value_stride = HEAD_DIM;
        block1.window = 0;                   // causal only; older than the window, by design
    }

    aeon::dispatch_causal_attention_head_group_fp16<>(
        q, q_stride, block0, block1, per_query_keys, per_query_count,
        query_position_base, 1, out, out_stride,
        count, static_cast<int>(kernel::DSV4_NUM_HEADS), HEAD_DIM,
        layer.d_attn_sink, kernel::DSV4_ATTN_SCALE, stream);
}

// The same launch, but QKᵀ on the matrix cores. Only for the shared-key classes
// (`Sliding`, `HCA`): a `CSA` layer's per-query top-k has no key tile to share across the
// query tile, so it stays on the split-keys kernel above.
inline void run_attention_tile_wmma(
    const V4Layer& layer,
    const half* q, int q_stride,
    half* out, int out_stride,
    int64_t query_position_base,
    int count,
    const half* local_keys, const int64_t* local_positions, int local_rows,
    uint32_t committed,
    hipStream_t stream) {
    constexpr int HEAD_DIM = kernel::DSV4_HEAD_DIM;

    CausalAttentionBlock block0;
    block0.keys = local_keys;
    block0.values = local_keys;
    block0.positions = local_positions;
    block0.rows = local_rows;
    block0.key_stride = HEAD_DIM;
    block0.value_stride = HEAD_DIM;
    block0.window = static_cast<int>(layer.local_cache_capacity());

    CausalAttentionBlock block1;
    if (layer.spec().attention_kind == V4AttentionKind::HCA && committed > 0) {
        block1.keys = layer.d_compressed_key_cache;
        block1.values = layer.d_compressed_value_cache;
        block1.positions = layer.d_compressed_positions;
        block1.rows = static_cast<int>(committed);
        block1.key_stride = HEAD_DIM;
        block1.value_stride = HEAD_DIM;
        block1.window = 0;
    }

    aeon::dispatch_causal_attention_wmma_qk_fp16(
        q, q_stride, block0, block1, query_position_base, 1, out, out_stride,
        count, static_cast<int>(kernel::DSV4_NUM_HEADS), HEAD_DIM,
        layer.d_attn_sink, kernel::DSV4_ATTN_SCALE, stream);
}

} // namespace aeon::core
