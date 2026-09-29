#pragma once

// -----------------------------------------------------------------------------
// The layer body's attention phases.
//
// The first half of the body: everything through attention and the FFN RMSNorm.
// `run_layer_body_pre_attention` stops deliberately short of attention, because a
// chunk must run it for **every** token before it attends any — all of the
// chunk's keys have to be in the ring first. `run_layer_body_attention_and_norm`
// is the second phase. See `v4_layer_body_types.hpp` for the shared row and
// tables.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/layer/v4_layer_body_types.hpp"
#include "architecture/deepseek_v4/kernels/hc_sinkhorn.hpp"
#include "platform/ops/cast.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace aeon::core {

// Runs the pre-attention half for **one token**: everything up to, but not
// including, attention.
//
// On entry the row's `d_res_in` (float, `hc_mult × hidden`) holds the four HC
// residual streams. On return the token's rotated key is in the local ring, the
// compressor (and on CSA the indexer) has been fed, and a ratio boundary has
// materialized its compressed entry.
//
// Stopping here is deliberately the wrong place to stop for a single-token
// decode. The two halves are worth separating for exactly one reason: a chunk
// must interleave them, because **all** of the chunk's keys have to be in the
// ring before **any** of its queries attends, or an early query cannot see a late
// key. Splitting the body here is what lets `run_layer_body_chunk` do that while
// still running the same code as decode; `run_layer_body_decoding` simply calls
// this and then the attention half, which is why there is one body and not two.
//
// The layer class is read from `layer.spec().attention_kind`: a Sliding layer
// takes the local-only path and never touches the compressor or indexer tensors;
// CSA and HCA take the compressed path with the indexer on CSA only.
inline V4LayerBodyPre run_layer_body_pre_attention(
    V4Layer& layer,
    V4LayerBodyRow& scratch,
    const V4LayerBodyTables& tables,
    uint32_t token_id,
    uint32_t pos,
    hipStream_t stream,
    V4LayerBodyObserver& observer) {
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    constexpr int HC = 4;
    constexpr int HC_DIM = HC * H;          // 16384
    constexpr int HC_MULT3 = HC * (2 + HC);// 24
    constexpr int Q_LORA = kernel::DSV4_Q_LORA_RANK;
    constexpr int HEAD_DIM = kernel::DSV4_HEAD_DIM;
    constexpr int NUM_HEADS = kernel::DSV4_NUM_HEADS;
    constexpr int TOTAL_Q = NUM_HEADS * HEAD_DIM;
    constexpr int INDEXER_Q = kernel::DSV4_INDEX_N_HEADS * kernel::DSV4_INDEX_HEAD_DIM;

    const bool uses_compressed_rope = layer.spec().attention_kind != V4AttentionKind::Sliding;
    const V4LayerBodyTables::View rope = tables.for_layer(layer.spec().attention_kind);

    // -----------------------------------------------------------------
    // A. Hyper-Connections attention pre-mix & Sinkhorn
    // -----------------------------------------------------------------
    hipLaunchKernelGGL(
        kernel::hc_project_kernel,
        dim3(HC_MULT3), dim3(256), 0, stream,
        scratch.d_res_in, layer.d_hc_attn_fn, scratch.d_mixes_a,
        H, HC, 1e-6f);

    hipLaunchKernelGGL(
        kernel::hc_sinkhorn_normalize_kernel,
        dim3(1), dim3(32), 0, stream,
        scratch.d_mixes_a, layer.d_hc_attn_scale, layer.d_hc_attn_base,
        scratch.d_pre_a, scratch.d_post_a, scratch.d_comb_a,
        1e-6f, 1e-6f, 2.0f, 20);

    hipLaunchKernelGGL(
        kernel::hc_pre_combine_kernel,
        dim3((H / 4 + 255) / 256), dim3(256), 0, stream,
        scratch.d_res_in, scratch.d_pre_a, scratch.d_x_pre, H, HC);

    // -----------------------------------------------------------------
    // B. Attention RMSNorm
    // -----------------------------------------------------------------
    hipLaunchKernelGGL(
        kernel::rmsnorm_wave32_kernel,
        dim3(1), dim3(32), 0, stream,
        scratch.d_x_pre, layer.d_attn_norm, scratch.d_x_norm, H, 1e-6f);

    // -----------------------------------------------------------------
    // C. MLA projections — q_lora -> q_norm -> wq_b -> per-head norm; wkv -> kv_norm
    // -----------------------------------------------------------------
    hipLaunchKernelGGL(
        kernel::gemv_fp16_kernel,
        dim3(Q_LORA, 1), dim3(32), 0, stream,
        scratch.d_x_norm, layer.d_wq_a, scratch.d_qa, H);

    hipLaunchKernelGGL(
        kernel::rmsnorm_wave32_kernel,
        dim3(1), dim3(32), 0, stream,
        scratch.d_qa, layer.d_q_norm, scratch.d_qa_norm, Q_LORA, 1e-6f);

    hipLaunchKernelGGL(
        kernel::gemv_fp16_kernel,
        dim3(TOTAL_Q, 1), dim3(32), 0, stream,
        scratch.d_qa_norm, layer.d_wq_b, scratch.d_q, Q_LORA);

    // The per-head norm is WEIGHTLESS: the artifact has no tensor for it, and
    // upstream's fused q-norm/rope takes no weight argument.
    hipLaunchKernelGGL(
        kernel::rmsnorm_unit_wave32_kernel,
        dim3(NUM_HEADS), dim3(32), 0, stream,
        scratch.d_q, scratch.d_q, HEAD_DIM, 1e-6f);

    hipLaunchKernelGGL(
        kernel::gemv_fp16_kernel,
        dim3(HEAD_DIM, 1), dim3(32), 0, stream,
        scratch.d_x_norm, layer.d_wkv, scratch.d_kv, H);

    hipLaunchKernelGGL(
        kernel::rmsnorm_wave32_kernel,
        dim3(1), dim3(32), 0, stream,
        scratch.d_kv, layer.d_kv_norm, scratch.d_kv_norm_act, HEAD_DIM, 1e-6f);

    if (uses_compressed_rope) {
        const int ratio = layer.spec().compression_ratio;
        const int coefficient = ratio == 4 ? 2 : 1;
        const int compressor_width = coefficient * HEAD_DIM;

        hipLaunchKernelGGL(
            kernel::gemv_fp16_kernel,
            dim3(compressor_width, 1), dim3(32), 0, stream,
            scratch.d_x_norm, layer.d_compressor_wkv, scratch.d_compressor_kv, H);
        hipLaunchKernelGGL(
            kernel::gemv_fp16_kernel,
            dim3(compressor_width, 1), dim3(32), 0, stream,
            scratch.d_x_norm, layer.d_compressor_wgate, scratch.d_compressor_score, H);

        if (layer.spec().attention_kind == V4AttentionKind::CSA) {
            hipLaunchKernelGGL(
                kernel::gemv_fp16_kernel,
                dim3(INDEXER_Q, 1), dim3(32), 0, stream,
                scratch.d_qa_norm, layer.d_indexer_wq_b, scratch.d_indexer_query, Q_LORA);
            hipLaunchKernelGGL(
                kernel::gemv_fp16_kernel,
                dim3(kernel::DSV4_INDEX_N_HEADS, 1), dim3(32), 0, stream,
                scratch.d_x_norm, layer.d_indexer_weights_proj,
                scratch.d_indexer_weights_half, H);
            hipLaunchKernelGGL(
                kernel::gemv_fp16_kernel,
                dim3(coefficient * kernel::DSV4_INDEX_HEAD_DIM, 1), dim3(32), 0, stream,
                scratch.d_x_norm, layer.d_indexer_compressor_wkv,
                scratch.d_indexer_compressor_kv, H);
            hipLaunchKernelGGL(
                kernel::gemv_fp16_kernel,
                dim3(coefficient * kernel::DSV4_INDEX_HEAD_DIM, 1), dim3(32), 0, stream,
                scratch.d_x_norm, layer.d_indexer_compressor_wgate,
                scratch.d_indexer_compressor_score, H);
        }
    }

    V4AttentionTraceRecord* attention_trace =
        observer.begin_trace(layer, token_id, pos);
    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->block_residual_input, scratch.d_res_in, HC_DIM);
        trace_copy(observer, attention_trace->attention_hc_mixes, scratch.d_mixes_a, HC_MULT3);
        trace_copy(observer, attention_trace->attention_hc_pre_mix, scratch.d_pre_a, HC);
        trace_copy(observer, attention_trace->attention_hc_post_mix, scratch.d_post_a, HC);
        trace_copy(observer, attention_trace->attention_hc_comb_mix, scratch.d_comb_a, HC * HC);
        trace_copy(observer, attention_trace->attention_precombined_input, scratch.d_x_pre, H);
        trace_copy(observer, attention_trace->attention_normalized_input, scratch.d_x_norm, H);
        trace_copy(observer, attention_trace->query, scratch.d_q, TOTAL_Q);
        trace_copy(observer, attention_trace->local_key, scratch.d_kv_norm_act, HEAD_DIM);
        trace_copy(observer, attention_trace->local_value, scratch.d_kv_norm_act, HEAD_DIM);
        if (layer.spec().attention_kind != V4AttentionKind::Sliding) {
            const int coefficient = layer.spec().compression_ratio == 4 ? 2 : 1;
            const size_t compressor_width = static_cast<size_t>(coefficient * HEAD_DIM);
            trace_copy(observer, attention_trace->compressor_kv,
                       scratch.d_compressor_kv, compressor_width);
            trace_copy(observer, attention_trace->compressor_score,
                       scratch.d_compressor_score, compressor_width);
            if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                const size_t indexer_query_width = static_cast<size_t>(
                    kernel::DSV4_INDEX_N_HEADS * kernel::DSV4_INDEX_HEAD_DIM);
                const size_t indexer_width =
                    static_cast<size_t>(coefficient * kernel::DSV4_INDEX_HEAD_DIM);
                trace_copy(observer, attention_trace->indexer_query,
                           scratch.d_indexer_query, indexer_query_width);
                trace_copy(observer, attention_trace->indexer_weights,
                           scratch.d_indexer_weights_half, kernel::DSV4_INDEX_N_HEADS);
                trace_copy(observer, attention_trace->indexer_compressor_kv,
                           scratch.d_indexer_compressor_kv, indexer_width);
                trace_copy(observer, attention_trace->indexer_compressor_score,
                           scratch.d_indexer_compressor_score, indexer_width);
            }
        }
    }

    // -----------------------------------------------------------------
    // D. RoPE forward & local KV cache persistence
    // -----------------------------------------------------------------
    // Decode writes the ring slot for this position. A chunk has already pointed
    // these at its own key buffer, because a chunk cannot write the ring until its
    // queries are done: for a ring of `C` slots, the write for position `p` lands
    // in slot `p mod C`, which held position `p − C` — and `p − C` is the *first*
    // key of the window of every query in the chunk with a position below `p`.
    // Writing the chunk's keys into the ring first therefore evicts keys that
    // earlier queries in the same chunk still need, for any chunk length above
    // one. `v4_layer_body_batch.hpp` carries the proof.
    const uint32_t local_slot = pos % layer.local_cache_capacity();
    if (scratch.d_local_key_write == nullptr) {
        const size_t local_offset = static_cast<size_t>(local_slot) * HEAD_DIM;
        scratch.d_local_key_write = layer.d_local_key_cache + local_offset;
        scratch.d_local_value_write = layer.d_local_value_cache + local_offset;
        scratch.d_local_position_write = layer.d_local_positions + local_slot;
    }

    hipLaunchKernelGGL(
        kernel::v4_forward_rope_at_pos_wave32_kernel,
        dim3(NUM_HEADS), dim3(32), 0, stream,
        scratch.d_q, rope.cos, rope.sin, pos,
        NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2);

    hipLaunchKernelGGL(
        kernel::v4_forward_rope_at_pos_wave32_kernel,
        dim3(1), dim3(32), 0, stream,
        scratch.d_kv_norm_act, rope.cos, rope.sin, pos,
        1, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2);

    // Key and value are the *same* row. Both caches are written.
    CHECK_HIP(hipMemcpyAsync(scratch.d_local_value_write,
                             scratch.d_kv_norm_act, HEAD_DIM * sizeof(half),
                             hipMemcpyDeviceToDevice, stream));
    CHECK_HIP(hipMemcpyAsync(scratch.d_local_key_write,
                             scratch.d_kv_norm_act, HEAD_DIM * sizeof(half),
                             hipMemcpyDeviceToDevice, stream));
    {
        const int64_t absolute_position = static_cast<int64_t>(pos);
        CHECK_HIP(hipMemcpyAsync(scratch.d_local_position_write,
                                 &absolute_position, sizeof(absolute_position),
                                 hipMemcpyHostToDevice, stream));
    }
    layer.record_position(pos);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->rotated_query, scratch.d_q, TOTAL_Q);
        trace_copy(observer, attention_trace->rotated_local_key, scratch.d_kv_norm_act, HEAD_DIM);
        trace_copy(observer, attention_trace->local_key_cache,
                   layer.d_local_key_cache,
                   static_cast<size_t>(layer.state_layout().local_capacity) * HEAD_DIM);
        trace_copy(observer, attention_trace->local_value_cache,
                   layer.d_local_value_cache,
                   static_cast<size_t>(layer.state_layout().local_capacity) * HEAD_DIM);
        trace_copy(observer, attention_trace->local_positions,
                   layer.d_local_positions, layer.state_layout().local_capacity);
        attention_trace->local_valid_count = layer.local_valid_count_;
    }

    const int64_t absolute_position = static_cast<int64_t>(pos);

    // Every compressed-row count is derived from `pos` rather than read off the
    // layer, because a chunk advances the layer's counters to its *last* token
    // while an earlier token must still see its own. For a single token the two
    // are equal by construction.
    bool emitted_compressed = false;
    uint32_t emitted_index = 0;
    const uint32_t committed = uses_compressed_rope
        ? committed_entries_for(layer, pos, layer.spec().compression_ratio)
        : 0u;

    if (uses_compressed_rope) {
        const int ratio = layer.spec().compression_ratio;
        const int coefficient = ratio == 4 ? 2 : 1;
        const int compressor_width = coefficient * HEAD_DIM;
        const int partial_capacity =
            static_cast<int>(layer.state_layout().compressor_partial_capacity);

        // Compressor partial state. The APE add lives inside the kernel and
        // applies to `score` only.
        hipLaunchKernelGGL(
            kernel::v4_save_compressor_state_kernel,
            dim3(1), dim3(256), 0, stream,
            scratch.d_compressor_kv, scratch.d_compressor_score,
            layer.d_compressor_partial_kv, layer.d_compressor_partial_score,
            layer.d_compressor_partial_positions, layer.d_compressor_ape,
            absolute_position, ratio, partial_capacity, compressor_width);

        if (layer.spec().attention_kind == V4AttentionKind::CSA) {
            hipLaunchKernelGGL(
                kernel::v4_forward_rope_at_pos_wave32_kernel,
                dim3(kernel::DSV4_INDEX_N_HEADS), dim3(32), 0, stream,
                scratch.d_indexer_query, rope.cos, rope.sin, pos,
                kernel::DSV4_INDEX_N_HEADS, kernel::DSV4_INDEX_HEAD_DIM,
                kernel::DSV4_INDEX_HEAD_DIM - kernel::DSV4_ROPE_DIM,
                kernel::DSV4_ROPE_DIM / 2);
            kernel::v4_half_to_float_n_kernel<<<1, 128, 0, stream>>>(
                scratch.d_indexer_weights_half, scratch.d_indexer_weights,
                kernel::DSV4_INDEX_N_HEADS);

            hipLaunchKernelGGL(
                kernel::v4_save_compressor_state_kernel,
                dim3(1), dim3(256), 0, stream,
                scratch.d_indexer_compressor_kv, scratch.d_indexer_compressor_score,
                layer.d_indexer_partial_kv, layer.d_indexer_partial_score,
                layer.d_indexer_partial_positions, layer.d_indexer_compressor_ape,
                absolute_position, ratio, partial_capacity,
                coefficient * kernel::DSV4_INDEX_HEAD_DIM);
        }

        // Compressed entries fire only on the boundary of a ratio window.
        if ((pos + 1u) % static_cast<uint32_t>(ratio) == 0) {
            const int compressed_index = static_cast<int>(
                (pos + 1u) / static_cast<uint32_t>(ratio) - 1u);
            emitted_compressed = true;
            emitted_index = static_cast<uint32_t>(compressed_index);
            hipLaunchKernelGGL(
                kernel::v4_materialize_compressed_entry_kernel,
                dim3(1), dim3(512), 0, stream,
                layer.d_compressor_partial_kv, layer.d_compressor_partial_score,
                layer.d_compressor_partial_positions, layer.d_compressor_norm,
                layer.d_compressed_key_cache, layer.d_compressed_value_cache,
                layer.d_compressed_positions,
                tables.compressed_cos, tables.compressed_sin,
                absolute_position, ratio, partial_capacity, HEAD_DIM,
                compressor_width, compressed_index,
                kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM, 1e-6f);

            if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                hipLaunchKernelGGL(
                    kernel::v4_materialize_compressed_entry_kernel,
                    dim3(1), dim3(512), 0, stream,
                    layer.d_indexer_partial_kv, layer.d_indexer_partial_score,
                    layer.d_indexer_partial_positions, layer.d_indexer_compressor_norm,
                    layer.d_indexer_key_cache, layer.d_indexer_key_cache,
                    layer.d_indexer_positions,
                    tables.compressed_cos, tables.compressed_sin,
                    absolute_position, ratio,
                    static_cast<int>(layer.state_layout().indexer_partial_capacity),
                    kernel::DSV4_INDEX_HEAD_DIM,
                    coefficient * kernel::DSV4_INDEX_HEAD_DIM,
                    compressed_index,
                    kernel::DSV4_INDEX_HEAD_DIM - kernel::DSV4_ROPE_DIM,
                    kernel::DSV4_ROPE_DIM, 1e-6f);
            }
        }

        // The indexer runs on CSA only. HCA attends every committed compressed
        // row and has no indexer tensors at all.
        if (layer.spec().attention_kind == V4AttentionKind::CSA) {
            if (committed != 0) {
                hipLaunchKernelGGL(
                    kernel::v4_indexer_scores_kernel,
                    dim3((committed + 255u) / 256u), dim3(256), 0, stream,
                    scratch.d_indexer_query, scratch.d_indexer_weights,
                    layer.d_indexer_key_cache,
                    scratch.d_indexer_scores, static_cast<int>(committed),
                    kernel::DSV4_INDEX_N_HEADS, kernel::DSV4_INDEX_HEAD_DIM,
                    1.0f / std::sqrt(static_cast<float>(kernel::DSV4_INDEX_HEAD_DIM)),
                    1.0f / std::sqrt(static_cast<float>(kernel::DSV4_INDEX_N_HEADS)));
            }
            select_indexer_topk(layer, scratch.d_indexer_scores,
                                scratch.d_indexer_topk_indices,
                                static_cast<size_t>(committed), stream);
        }

        if (attention_trace != nullptr) {
            trace_copy(observer, attention_trace->compressor_partial_kv,
                       layer.d_compressor_partial_kv,
                       layer.state_layout().compressor_partial_vector_bytes() / sizeof(float));
            trace_copy(observer, attention_trace->compressor_partial_score,
                       layer.d_compressor_partial_score,
                       layer.state_layout().compressor_partial_vector_bytes() / sizeof(float));
            trace_copy(observer, attention_trace->compressor_partial_positions,
                       layer.d_compressor_partial_positions,
                       layer.state_layout().compressor_partial_capacity);
            trace_copy(observer, attention_trace->compressed_key_cache,
                       layer.d_compressed_key_cache,
                       static_cast<size_t>(layer.state_layout().compressed_capacity) * HEAD_DIM);
            trace_copy(observer, attention_trace->compressed_value_cache,
                       layer.d_compressed_value_cache,
                       static_cast<size_t>(layer.state_layout().compressed_capacity) * HEAD_DIM);
            trace_copy(observer, attention_trace->compressed_positions,
                       layer.d_compressed_positions,
                       layer.state_layout().compressed_capacity);
            attention_trace->compressor_partial_count = layer.compressor_partial_count_;
            attention_trace->compressed_entry_count = committed;
            attention_trace->indexer_candidate_count = committed;

            if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                trace_copy(observer, attention_trace->indexer_partial_kv,
                           layer.d_indexer_partial_kv,
                           layer.state_layout().indexer_partial_vector_bytes() / sizeof(float));
                trace_copy(observer, attention_trace->indexer_partial_score,
                           layer.d_indexer_partial_score,
                           layer.state_layout().indexer_partial_vector_bytes() / sizeof(float));
                trace_copy(observer, attention_trace->indexer_partial_positions,
                           layer.d_indexer_partial_positions,
                           layer.state_layout().indexer_partial_capacity);
                trace_copy(observer, attention_trace->indexer_key_cache,
                           layer.d_indexer_key_cache,
                           static_cast<size_t>(layer.state_layout().compressed_capacity) *
                               layer.state_layout().index_head_dim);
                trace_copy(observer, attention_trace->indexer_positions,
                           layer.d_indexer_positions,
                           layer.state_layout().compressed_capacity);
                trace_copy(observer, attention_trace->indexer_scores,
                           scratch.d_indexer_scores, committed);
                trace_copy(observer, attention_trace->indexer_topk_indices,
                           scratch.d_indexer_topk_indices,
                           layer.state_layout().index_topk);
            }
        }
    }

    V4LayerBodyPre pre;
    pre.trace = attention_trace;
    pre.emitted_compressed = emitted_compressed;
    pre.compressed_index = emitted_index;
    pre.committed = committed;
    return pre;
}

// The attention half for **one token**: attention over the class's row-set,
// followed by the output projection, the FFN and the residual.
//
// Splitting here is what lets a chunk write every key before any query runs. For
// a single token the split is invisible: `run_layer_body_decoding` calls both
// halves back to back with nothing in between.
//
// Everything in this half is what a token's selection does **not** depend on and
// what the router **does** depend on. The tail is split into phases — not
// duplicated — so a chunk can run each phase over every one of its tokens before
// the next begins: the router of every token must have run before the layer's
// union can be issued as one set, and the MoE of none of them may have run before
// it.
inline void run_layer_body_attention_and_norm(
    V4Layer& layer,
    V4LayerBodyRow& scratch,
    const V4LayerBodyTables& tables,
    uint32_t token_id,
    uint32_t pos,
    hipStream_t stream,
    V4LayerBodyObserver& observer,
    V4LayerBodyPre pre) {
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    constexpr int HC = 4;
    constexpr int HC_DIM = HC * H;          // 16384
    constexpr int HC_MULT3 = HC * (2 + HC);// 24
    constexpr int M_PAD = 16;
    constexpr int HEAD_DIM = kernel::DSV4_HEAD_DIM;
    constexpr int NUM_HEADS = kernel::DSV4_NUM_HEADS;
    constexpr int TOTAL_Q = NUM_HEADS * HEAD_DIM;
    constexpr int O_LORA = kernel::DSV4_O_LORA_RANK;
    constexpr int O_GROUPS = kernel::DSV4_O_GROUPS;
    constexpr int TOT_LORA = kernel::DSV4_TOTAL_O_LORA_DIM;

    V4AttentionTraceRecord* attention_trace = pre.trace;
    const V4LayerBodyTables::View rope = tables.for_layer(layer.spec().attention_kind);
    const int64_t absolute_position = static_cast<int64_t>(pos);

    // -----------------------------------------------------------------
    // E. Class-specific attention over the cached states
    // -----------------------------------------------------------------
    // The local rows come either from the ring itself (decode) or from the
    // caller's composed row-set (a chunk, whose own keys are not in the ring yet).
    // In both cases the rows are handed to the kernel with the row count as the
    // "capacity" and the positions alongside, so the kernel's own window filter —
    // `local_start <= key_position <= current_position` — is satisfied by every
    // row and the arithmetic is the same code with the same summation order.
    const half* local_keys = scratch.d_composed_keys != nullptr
        ? scratch.d_composed_keys : layer.d_local_key_cache;
    const int64_t* local_positions = scratch.d_composed_positions != nullptr
        ? scratch.d_composed_positions : layer.d_local_positions;
    const int local_rows = scratch.composed_rows > 0
        ? scratch.composed_rows : static_cast<int>(layer.local_cache_capacity());

    if (layer.spec().attention_kind == V4AttentionKind::Sliding) {
        hipLaunchKernelGGL(
            kernel::v4_cached_sliding_window_attn_wave32_kernel,
            dim3(NUM_HEADS), dim3(32), 0, stream,
            scratch.d_q, local_keys, local_keys,
            local_positions, layer.d_attn_sink, scratch.d_attn_out,
            static_cast<int>(pos), local_rows, kernel::DSV4_ATTN_SCALE);
    } else {
        const bool uses_indexer = layer.spec().attention_kind == V4AttentionKind::CSA;
        const int compressed_count = static_cast<int>(pre.committed);
        const int topk_count = uses_indexer
            ? std::min<int>(compressed_count, static_cast<int>(layer.state_layout().index_topk))
            : 0;
        hipLaunchKernelGGL(
            kernel::v4_cached_compressed_attention_wave32_kernel,
            dim3(NUM_HEADS), dim3(32), 0, stream,
            scratch.d_q, local_keys, local_keys,
            local_positions, layer.d_attn_sink,
            layer.d_compressed_key_cache, layer.d_compressed_value_cache,
            layer.d_compressed_positions,
            uses_indexer ? scratch.d_indexer_topk_indices : nullptr,
            scratch.d_attn_out, absolute_position,
            local_rows, compressed_count,
            topk_count, uses_indexer, kernel::DSV4_ATTN_SCALE);
    }

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->attention_output, scratch.d_attn_out, TOTAL_Q);
    }

    // Inverse RoPE on the attention output tail, before the grouped projection.
    hipLaunchKernelGGL(
        kernel::v4_inverse_rope_at_pos_wave32_kernel,
        dim3(NUM_HEADS), dim3(32), 0, stream,
        scratch.d_attn_out, rope.cos, rope.sin, pos,
        NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->inverse_rope_output, scratch.d_attn_out, TOTAL_Q);
    }

    // Grouped wo_a [G, R, D] then wo_b [hidden, G·R].
    hipLaunchKernelGGL(
        kernel::v4_grouped_wo_a_wave32_kernel,
        dim3(O_LORA, O_GROUPS, 1), dim3(32), 0, stream,
        scratch.d_attn_out, layer.d_wo_a, scratch.d_z_lora, 1);

    hipLaunchKernelGGL(
        kernel::gemv_fp16_kernel,
        dim3(H, 1), dim3(32), 0, stream,
        scratch.d_z_lora, layer.d_wo_b, scratch.d_attn_proj, TOT_LORA);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->grouped_output, scratch.d_attn_proj, H);
    }

    // -----------------------------------------------------------------
    // F. HC attention post expansion: res_mid = comb_a · res_in + post_a · attn_proj
    // -----------------------------------------------------------------
    kernel::float_to_half_kernel<<<(HC_DIM + 255) / 256, 256, 0, stream>>>(
        scratch.d_res_in, scratch.d_res_in_half, HC_DIM);

    hipLaunchKernelGGL(
        kernel::hc_post_kernel,
        dim3((H + 255) / 256, 1), dim3(256), 0, stream,
        scratch.d_attn_proj, scratch.d_res_in_half, scratch.d_post_a,
        scratch.d_comb_a, scratch.d_res_mid_half, H);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->attention_post_residual,
                   scratch.d_res_mid_half, HC_DIM);
    }

    kernel::half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, stream>>>(
        scratch.d_res_mid_half, scratch.d_res_mid, HC_DIM);

    // -----------------------------------------------------------------
    // G. HC FFN pre-mix & Sinkhorn
    // -----------------------------------------------------------------
    hipLaunchKernelGGL(
        kernel::hc_project_kernel,
        dim3(HC_MULT3), dim3(256), 0, stream,
        scratch.d_res_mid, layer.d_hc_ffn_fn, scratch.d_mixes_f,
        H, HC, 1e-6f);

    hipLaunchKernelGGL(
        kernel::hc_sinkhorn_normalize_kernel,
        dim3(1), dim3(32), 0, stream,
        scratch.d_mixes_f, layer.d_hc_ffn_scale, layer.d_hc_ffn_base,
        scratch.d_pre_f, scratch.d_post_f, scratch.d_comb_f,
        1e-6f, 1e-6f, 2.0f, 20);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->ffn_hc_post_mix, scratch.d_post_f, HC);
        trace_copy(observer, attention_trace->ffn_hc_comb_mix, scratch.d_comb_f, HC * HC);
    }

    hipLaunchKernelGGL(
        kernel::hc_pre_combine_kernel,
        dim3((H / 4 + 255) / 256), dim3(256), 0, stream,
        scratch.d_res_mid, scratch.d_pre_f, scratch.d_ffn_pre, H, HC);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->ffn_precombined_input, scratch.d_ffn_pre, H);
    }

    // -----------------------------------------------------------------
    // FFN RMSNorm
    // -----------------------------------------------------------------
    hipLaunchKernelGGL(
        kernel::rmsnorm_wave32_kernel,
        dim3(1), dim3(32), 0, stream,
        scratch.d_ffn_pre, layer.d_ffn_norm, scratch.d_ffn_norm_act, H, 1e-6f);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->ffn_normalized_input, scratch.d_ffn_norm_act, H);
    }

    // WMMA compatibility: row 0 replicated into the padded rows.
    for (int r = 1; r < M_PAD; ++r) {
        CHECK_HIP(hipMemcpyAsync(scratch.d_ffn_norm_act + r * H,
                                 scratch.d_ffn_norm_act, H * sizeof(half),
                                 hipMemcpyDeviceToDevice, stream));
    }
}

} // namespace aeon::core
