#pragma once

// -----------------------------------------------------------------------------
// Tier 2 — the decoder layer body (plan Steps 2.0 … 2.11).
//
// This is the composition of the Tier-1 primitives into one layer. It exists as
// its own module for two reasons the plan states explicitly:
//
//   1. **One body, not two.** "The layer body must be the same code as decode,
//      parameterised by chunk size. Two bodies is how the two paths drift."
//      Decode calls `run_layer_body_decoding` below; the batched prefill path is
//      meant to join it rather than keep its own copy. Nothing here is allowed to
//      acquire a decode-only special case.
//
//   2. **The composition is what a Tier-2 gate has to certify.** Tier 1 proved
//      each primitive; a layer fails in the wiring between them, and the wiring is
//      only testable if the test can drive the same code the runtime drives. So
//      the body takes its model-level inputs, its observers and its routed-expert
//      supply as *parameters*, and knows nothing about artifact tiers, telemetry
//      or the pipeline object.
//
// Two seams keep the numerics independent of the runtime around them:
//
//   * `V4LayerBodyObserver` — the attention trace. The gate passes the null
//     observer; the pipeline passes one that fills a trace record. Keeping the
//     field list here means there is exactly one description of a trace.
//
//   * `V4RoutedExpertExecutor` — the routed-expert supply system (index lookup,
//     Hot/Warm/Cold promotion, prefetch, leases, staging). All of it lives behind
//     this interface, so a layer body's arithmetic does not depend on which tier
//     the weights came from, and a gate can supply experts directly.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/v4_attention_trace.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline_scratch.hpp"
#include "architecture/deepseek_v4/kernels/hc_sinkhorn.hpp"
#include "architecture/deepseek_v4/kernels/moe_router.hpp"
#include "architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace aeon::core {

// The model-level tensors the layer body reads. Today that is only the RoPE
// tables; they are handed over explicitly so the body has no dependency on the
// model-resources object and a gate can hand it a table it built itself.
struct V4LayerBodyTables {
    const float* sliding_cos{nullptr};
    const float* sliding_sin{nullptr};
    const float* compressed_cos{nullptr};
    const float* compressed_sin{nullptr};

    struct View {
        const float* cos;
        const float* sin;
    };

    // Two bases, selected by layer class (plan 2.3): a Sliding layer rotates with
    // the plain base, a compressed layer with the YaRN-on-compressed base.
    View for_layer(V4AttentionKind kind) const noexcept {
        return kind == V4AttentionKind::Sliding
            ? View{sliding_cos, sliding_sin}
            : View{compressed_cos, compressed_sin};
    }
};

// Observer seam for the attention trace. `begin_trace` returns a record to fill
// or nullptr when this layer/position is not traced; `copy_to_host` performs the
// device→host copy. The trace field knowledge stays in the layer body.
class V4LayerBodyObserver {
public:
    virtual ~V4LayerBodyObserver() = default;

    virtual V4AttentionTraceRecord* begin_trace(V4Layer& layer,
                                                uint32_t token_id,
                                                uint32_t position) {
        (void)layer;
        (void)token_id;
        (void)position;
        return nullptr;
    }

    virtual void copy_to_host(void* destination, const void* source, size_t bytes) = 0;
};

// The gate's observer: traces nothing and copies nothing.
class V4NullLayerBodyObserver final : public V4LayerBodyObserver {
public:
    void copy_to_host(void*, const void*, size_t) override {}
};

template <typename T>
inline void trace_copy(V4LayerBodyObserver& observer, std::vector<T>& destination,
                       const T* source, size_t count) {
    destination.resize(count);
    if (count == 0) return;
    observer.copy_to_host(destination.data(), source, count * sizeof(T));
}

// Seam for the routed-expert supply system. The three hooks are ordered exactly
// as the runtime needs them, so moving this code behind an interface does not
// change when leases are released or when a prefetch is dispatched relative to
// the shared-expert pass.
class V4RoutedExpertExecutor {
public:
    virtual ~V4RoutedExpertExecutor() = default;

    // Runs once the routed ids are known and *before* any device work for the
    // MoE: release the previous layer's leases, record the routing profile,
    // release staging slots whose transfer has completed.
    virtual void on_routing_ready(uint32_t layer_id, uint32_t position,
                                  const std::vector<int32_t>& ids,
                                  const std::vector<float>& weights) {
        (void)layer_id;
        (void)position;
        (void)ids;
        (void)weights;
    }

    // Accumulate `Σ_k w_k · W2_k · clamped_swiglu(W1_k·x, W3_k·x)` into
    // `moe_accum`, which already holds the shared expert's output.
    //
    // Both operands are passed explicitly: `expert_input` is this token's FFN
    // RMSNorm output (so the executor never has to know which row of a batch
    // workspace the token occupies) and `expert_weights` is the per-token routing
    // weight vector the fused kernel scales by. That is what lets one body serve
    // a single-token decode and a chunk of tokens without the supply system
    // reaching into a shared scratch buffer. Must preserve the stream order: any
    // work it enqueues runs after the shared expert.
    virtual void accumulate_routed(uint32_t layer_id, uint32_t position,
                                   const half* expert_input,
                                   const float* expert_weights,
                                   half* moe_accum) = 0;

    // Runs after the routed experts have been consumed, so the executor can
    // remember which staging slots to release on the next layer.
    virtual void on_routed_consumed(uint32_t layer_id, uint32_t position) {
        (void)layer_id;
        (void)position;
    }
};

struct V4LayerBodyOutput {
    std::vector<int32_t> topk_indices;
    std::vector<float> topk_weights;
};

// Indexer candidate selection (plan 2.4.3, layer part). Descending score, ties
// broken to the **lower index** — the same rule as the router (trap 18) — and
// the degenerate case `candidates <= index_topk` selects every candidate with no
// padding.
//
// Note this is the one host-side sync left in the body. The plan requires it to
// become on-device for chunked batched prefill; that is item 19's second half and
// is recorded here rather than hidden, because it does not change any result and
// therefore cannot be seen by the equivalence gate.
inline void select_indexer_topk(const V4Layer& layer, const float* device_scores,
                                int32_t* device_topk, size_t candidate_count,
                                hipStream_t stream) {
    if (candidate_count == 0) return;

    std::vector<float> scores(candidate_count);
    CHECK_HIP(hipMemcpyAsync(scores.data(), device_scores,
                             candidate_count * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    std::vector<int32_t> order(candidate_count);
    for (size_t index = 0; index < candidate_count; ++index) {
        order[index] = static_cast<int32_t>(index);
    }
    const size_t topk = std::min(
        candidate_count, static_cast<size_t>(layer.state_layout().index_topk));
    std::vector<int32_t> selected(static_cast<size_t>(layer.state_layout().index_topk), -1);
    if (candidate_count <= static_cast<size_t>(layer.state_layout().index_topk)) {
        std::copy(order.begin(), order.end(), selected.begin());
    } else {
        std::stable_sort(order.begin(), order.end(),
                         [&scores](int32_t left, int32_t right) {
                             const float left_score = scores[static_cast<size_t>(left)];
                             const float right_score = scores[static_cast<size_t>(right)];
                             if (left_score != right_score) return left_score > right_score;
                             return left < right;
                         });
        std::copy_n(order.begin(), topk, selected.begin());
    }
    CHECK_HIP(hipMemcpyAsync(device_topk, selected.data(),
                             selected.size() * sizeof(int32_t),
                             hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
}

// How many compressed entries exist on or before `pos`. Every caller that used to
// read `layer.compressed_entry_count_` after `record_position(pos)` reads this
// instead, because a chunk advances the layer's counter to the chunk's *last*
// token while an earlier token's attention must still see its own count. For a
// single token the two are equal by definition, so the decode path is unchanged.
inline uint32_t committed_entries_for(const V4Layer& layer, uint32_t pos, int32_t ratio) {
    if (ratio <= 0 || !layer.state_layout().is_compressed()) return 0;
    const int64_t capacity = static_cast<int64_t>(layer.state_layout().compressed_capacity);
    return static_cast<uint32_t>(std::min<int64_t>(
        capacity, (static_cast<int64_t>(pos) + 1) / static_cast<int64_t>(ratio)));
}

// -----------------------------------------------------------------------------
// The per-token buffer view. **One body serves both a decode step and a chunk of
// tokens**, and this is the seam that makes that literal rather than aspirational:
//
//   * decode fills it from row 0 of `PipelineScratchBuffers`, which already has
//     the right shape — every workspace array in it is a single 16-row tile,
//     because the expert kernels read `ffn_norm_act`/`moe_accum` in 16-row WMMA
//     tiles and the pipeline replicates row 0 to fill them;
//   * a chunk fills it from row `r` of `V4LayerBodyBatchScratch`, where each token
//     owns its own tile — so one token's padding can never clobber another's
//     input.
//
// The four *state* buffers the indexer owns (query, weights, scores, selected
// top-k) come from the layer on the decode path, which is where they already
// lived, and from the batch workspace on the chunk path where they must be
// per-token. Nothing else in the body distinguishes the two.
// -----------------------------------------------------------------------------
struct V4LayerBodyRow {
    float* d_res_in{nullptr};
    float* d_res_mid{nullptr};
    half* d_res_in_half{nullptr};
    half* d_res_mid_half{nullptr};
    half* d_res_out_half{nullptr};

    float* d_mixes_a{nullptr};
    float* d_pre_a{nullptr};
    float* d_post_a{nullptr};
    float* d_comb_a{nullptr};
    float* d_mixes_f{nullptr};
    float* d_pre_f{nullptr};
    float* d_post_f{nullptr};
    float* d_comb_f{nullptr};

    half* d_x_pre{nullptr};
    half* d_x_norm{nullptr};
    half* d_qa{nullptr};
    half* d_qa_norm{nullptr};
    half* d_q{nullptr};
    half* d_kv{nullptr};
    half* d_kv_norm_act{nullptr};
    half* d_compressor_kv{nullptr};
    half* d_compressor_score{nullptr};

    half* d_indexer_query{nullptr};
    half* d_indexer_weights_half{nullptr};
    float* d_indexer_weights{nullptr};
    half* d_indexer_compressor_kv{nullptr};
    half* d_indexer_compressor_score{nullptr};
    float* d_indexer_scores{nullptr};
    int32_t* d_indexer_topk_indices{nullptr};

    half* d_attn_out{nullptr};
    half* d_z_lora{nullptr};
    half* d_attn_proj{nullptr};

    half* d_ffn_pre{nullptr};
    half* d_ffn_norm_act{nullptr};

    half* d_router_logits_half{nullptr};
    float* d_router_logits{nullptr};
    float* d_topk_weights{nullptr};
    int32_t* d_topk_indices{nullptr};
    int32_t* d_token_id{nullptr};

    half* d_shared_gate{nullptr};
    half* d_shared_up{nullptr};
    half* d_shared_swiglu{nullptr};

    half* d_moe_accum{nullptr};

    // ---- Where this token's rotated key (which is also its value, trap 6) is
    // written. Decode leaves these null and gets the ring slot for its own
    // position; a chunk points them at its own key buffer, because **a chunk must
    // not write the ring as it goes** — see the proof in `v4_layer_body_batch.hpp`.
    half* d_local_key_write{nullptr};
    half* d_local_value_write{nullptr};
    int64_t* d_local_position_write{nullptr};

    // ---- The local row-set attention reads, when it is not the ring itself.
    // Ascending in position, so the attention kernel's summation order is the same
    // whichever path built it. A chunk supplies this; decode leaves it null and
    // the ring is read directly (so the decode path is bit-for-bit what it was
    // before the batch workspace existed).
    const half* d_composed_keys{nullptr};
    const int64_t* d_composed_positions{nullptr};
    int32_t composed_rows{0};
};

// The single-token row over `PipelineScratchBuffers` plus the layer's own indexer
// state buffers. Every pointer is exactly what the pre-refactor body used, so the
// decode path is byte-for-byte the same computation.
inline V4LayerBodyRow decode_layer_body_row(PipelineScratchBuffers& scratch,
                                            V4Layer& layer) {
    V4LayerBodyRow row;
    row.d_res_in = scratch.d_res_in;
    row.d_res_mid = scratch.d_res_mid;
    row.d_res_in_half = scratch.d_res_in_half;
    row.d_res_mid_half = scratch.d_res_mid_half;
    row.d_res_out_half = scratch.d_res_out_half;

    row.d_mixes_a = scratch.d_mixes_a;
    row.d_pre_a = scratch.d_pre_a;
    row.d_post_a = scratch.d_post_a;
    row.d_comb_a = scratch.d_comb_a;
    row.d_mixes_f = scratch.d_mixes_f;
    row.d_pre_f = scratch.d_pre_f;
    row.d_post_f = scratch.d_post_f;
    row.d_comb_f = scratch.d_comb_f;

    row.d_x_pre = scratch.d_x_pre;
    row.d_x_norm = scratch.d_x_norm;
    row.d_qa = scratch.d_qa;
    row.d_qa_norm = scratch.d_qa_norm;
    row.d_q = scratch.d_q;
    row.d_kv = scratch.d_kv;
    row.d_kv_norm_act = scratch.d_kv_norm_act;
    row.d_compressor_kv = scratch.d_compressor_kv;
    row.d_compressor_score = scratch.d_compressor_score;

    // The indexer query is RoPE'd in place into the layer's own buffer — the
    // pre-refactor body copied the scratch buffer into it, which is a
    // device-to-device copy of the value that is already there.
    row.d_indexer_query = layer.d_indexer_query;
    row.d_indexer_weights_half = scratch.d_indexer_weights;
    row.d_indexer_weights = layer.d_indexer_weights;
    row.d_indexer_compressor_kv = scratch.d_indexer_compressor_kv;
    row.d_indexer_compressor_score = scratch.d_indexer_compressor_score;
    row.d_indexer_scores = layer.d_indexer_scores;
    row.d_indexer_topk_indices = layer.d_indexer_topk_indices;

    row.d_attn_out = scratch.d_attn_out;
    row.d_z_lora = scratch.d_z_lora;
    row.d_attn_proj = scratch.d_attn_proj;

    row.d_ffn_pre = scratch.d_ffn_pre;
    row.d_ffn_norm_act = scratch.d_ffn_norm_act;

    row.d_router_logits_half = scratch.d_router_logits_half;
    row.d_router_logits = scratch.d_router_logits;
    row.d_topk_weights = scratch.d_topk_weights;
    row.d_topk_indices = scratch.d_topk_indices;
    row.d_token_id = scratch.d_token_id;

    row.d_shared_gate = scratch.d_shared_gate;
    row.d_shared_up = scratch.d_shared_up;
    row.d_shared_swiglu = scratch.d_shared_swiglu;

    row.d_moe_accum = scratch.d_moe_accum;
    return row;
}

// What the pre-attention half hands to the attention half.
struct V4LayerBodyPre {
    V4AttentionTraceRecord* trace{nullptr};
    bool emitted_compressed{false};
    uint32_t compressed_index{0};
    uint32_t committed{0};   // compressed entries on or before this token
};

// Runs Steps 2.0 – 2.4.3 for **one token**: everything up to, but not including,
// attention.
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
// still running the same code as decode; `run_layer_body_decoding` below simply
// calls this and then the attention half, which is why there is one body and not
// two.
//
// The layer class is read from `layer.spec().attention_kind`: a Sliding layer
// takes the local-only path and never touches the compressor or indexer tensors
// (traps 4 and 33); CSA and HCA take the compressed path with the indexer on CSA
// only.
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
        kernel::v4_rmsnorm_wave32_kernel,
        dim3(1), dim3(32), 0, stream,
        scratch.d_x_pre, layer.d_attn_norm, scratch.d_x_norm, H, 1e-6f);

    // -----------------------------------------------------------------
    // C. MLA projections — q_lora -> q_norm -> wq_b -> per-head norm; wkv -> kv_norm
    // -----------------------------------------------------------------
    hipLaunchKernelGGL(
        kernel::v4_gemv_fp16_kernel,
        dim3(Q_LORA, 1), dim3(32), 0, stream,
        scratch.d_x_norm, layer.d_wq_a, scratch.d_qa, H);

    hipLaunchKernelGGL(
        kernel::v4_rmsnorm_wave32_kernel,
        dim3(1), dim3(32), 0, stream,
        scratch.d_qa, layer.d_q_norm, scratch.d_qa_norm, Q_LORA, 1e-6f);

    hipLaunchKernelGGL(
        kernel::v4_gemv_fp16_kernel,
        dim3(TOTAL_Q, 1), dim3(32), 0, stream,
        scratch.d_qa_norm, layer.d_wq_b, scratch.d_q, Q_LORA);

    // The per-head norm is WEIGHTLESS (plan 2.2): the artifact has no tensor for
    // it, and upstream's fused q-norm/rope takes no weight argument.
    hipLaunchKernelGGL(
        kernel::v4_rmsnorm_unit_wave32_kernel,
        dim3(NUM_HEADS), dim3(32), 0, stream,
        scratch.d_q, scratch.d_q, HEAD_DIM, 1e-6f);

    hipLaunchKernelGGL(
        kernel::v4_gemv_fp16_kernel,
        dim3(HEAD_DIM, 1), dim3(32), 0, stream,
        scratch.d_x_norm, layer.d_wkv, scratch.d_kv, H);

    hipLaunchKernelGGL(
        kernel::v4_rmsnorm_wave32_kernel,
        dim3(1), dim3(32), 0, stream,
        scratch.d_kv, layer.d_kv_norm, scratch.d_kv_norm_act, HEAD_DIM, 1e-6f);

    if (uses_compressed_rope) {
        const int ratio = layer.spec().compression_ratio;
        const int coefficient = ratio == 4 ? 2 : 1;
        const int compressor_width = coefficient * HEAD_DIM;

        hipLaunchKernelGGL(
            kernel::v4_gemv_fp16_kernel,
            dim3(compressor_width, 1), dim3(32), 0, stream,
            scratch.d_x_norm, layer.d_compressor_wkv, scratch.d_compressor_kv, H);
        hipLaunchKernelGGL(
            kernel::v4_gemv_fp16_kernel,
            dim3(compressor_width, 1), dim3(32), 0, stream,
            scratch.d_x_norm, layer.d_compressor_wgate, scratch.d_compressor_score, H);

        if (layer.spec().attention_kind == V4AttentionKind::CSA) {
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(INDEXER_Q, 1), dim3(32), 0, stream,
                scratch.d_qa_norm, layer.d_indexer_wq_b, scratch.d_indexer_query, Q_LORA);
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(kernel::DSV4_INDEX_N_HEADS, 1), dim3(32), 0, stream,
                scratch.d_x_norm, layer.d_indexer_weights_proj,
                scratch.d_indexer_weights_half, H);
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(coefficient * kernel::DSV4_INDEX_HEAD_DIM, 1), dim3(32), 0, stream,
                scratch.d_x_norm, layer.d_indexer_compressor_wkv,
                scratch.d_indexer_compressor_kv, H);
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
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
    // one. `v4_layer_body_batch.hpp` carries the proof; trap 39 records it.
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

    // Key and value are the *same* row (trap 6). Both caches are written.
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

        // Compressor partial state. The APE add lives inside the kernel
        // (plan 2.4.2, trap 26) and applies to `score` only.
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
        // row and has no indexer tensors at all (trap 33).
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

// Steps 2.4.4 – 2.11 for **one token**: attention over the class's row-set,
// followed by the output projection, the FFN and the residual.
//
// Splitting here is what lets a chunk write every key before any query runs. For
// a single token the split is invisible: `run_layer_body_decoding` below calls
// both halves back to back with nothing in between.
inline V4LayerBodyOutput run_layer_body_attention_tail(
    V4Layer& layer,
    V4LayerBodyRow& scratch,
    const V4LayerBodyTables& tables,
    uint32_t token_id,
    uint32_t pos,
    hipStream_t stream,
    V4RoutedExpertExecutor& experts,
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
    constexpr int INTER_DIM = 2048;

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

    // Inverse RoPE on the attention output tail, before the grouped projection
    // (plan 2.3 / 2.5, trap 8).
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
        kernel::v4_gemv_fp16_kernel,
        dim3(H, 1), dim3(32), 0, stream,
        scratch.d_z_lora, layer.d_wo_b, scratch.d_attn_proj, TOT_LORA);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->grouped_output, scratch.d_attn_proj, H);
    }

    // -----------------------------------------------------------------
    // F. HC attention post expansion: res_mid = comb_a · res_in + post_a · attn_proj
    // -----------------------------------------------------------------
    kernel::v4_float_to_half_kernel<<<(HC_DIM + 255) / 256, 256, 0, stream>>>(
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

    kernel::v4_half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, stream>>>(
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
    // 2.8 — FFN RMSNorm
    // -----------------------------------------------------------------
    hipLaunchKernelGGL(
        kernel::v4_rmsnorm_wave32_kernel,
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

    // -----------------------------------------------------------------
    // H. MoE router (2.9)
    // -----------------------------------------------------------------
    hipLaunchKernelGGL(
        kernel::v4_gemv_fp16_vec8_kernel,
        dim3(256, 1), dim3(32), 0, stream,
        scratch.d_ffn_norm_act, layer.d_gate_weight, scratch.d_router_logits_half, H);

    kernel::v4_half_to_float_n_kernel<<<(256 + 255) / 256, 256, 0, stream>>>(
        scratch.d_router_logits_half, scratch.d_router_logits, 256);

    // The hash branch indexes `tid2eid` by token id, so the id has to be on the
    // device. Staging it here rather than expecting the caller to have done it
    // removes a precondition that would be invisible at the call site.
    {
        const int32_t h_token = static_cast<int32_t>(token_id);
        CHECK_HIP(hipMemcpyAsync(scratch.d_token_id, &h_token, sizeof(int32_t),
                                 hipMemcpyHostToDevice, stream));
    }

    hipLaunchKernelGGL(
        kernel::moe_router_kernel,
        dim3(1), dim3(64), 0, stream,
        scratch.d_router_logits,
        layer.is_hash_layer ? nullptr : layer.d_gate_bias,
        layer.d_tid2eid, scratch.d_token_id,
        scratch.d_topk_weights, scratch.d_topk_indices,
        256, 6, 1.5f, true);

    V4LayerBodyOutput output;
    output.topk_weights.resize(6);
    output.topk_indices.resize(6);
    CHECK_HIP(hipMemcpyAsync(output.topk_weights.data(), scratch.d_topk_weights,
                             6 * sizeof(float), hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipMemcpyAsync(output.topk_indices.data(), scratch.d_topk_indices,
                             6 * sizeof(int32_t), hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->router_logits, scratch.d_router_logits, 256);
        attention_trace->routed_expert_indices.assign(
            output.topk_indices.begin(), output.topk_indices.end());
        attention_trace->routed_expert_weights.assign(
            output.topk_weights.begin(), output.topk_weights.end());
    }

    // The ids are known: the supply system now does its bookkeeping and then
    // dispatches. The ordering is deliberate — all of it happens before the
    // shared-expert pass so the device stays busy while the CPU submits I/O.
    experts.on_routing_ready(static_cast<uint32_t>(layer.layer_id), pos,
                             output.topk_indices, output.topk_weights);

    // -----------------------------------------------------------------
    // 2.10.4 — shared expert (always fires), accumulating into the cleared buffer
    // -----------------------------------------------------------------
    // Clear MoE accumulation buffer
    CHECK_HIP(hipMemsetAsync(scratch.d_moe_accum, 0, M_PAD * H * sizeof(half), stream));

    hipLaunchKernelGGL(
        kernel::v4_gemv_fp16_vec8_kernel,
        dim3(INTER_DIM, 1), dim3(32), 0, stream,
        scratch.d_ffn_norm_act, layer.d_shared_w1, scratch.d_shared_gate, H);
    hipLaunchKernelGGL(
        kernel::v4_gemv_fp16_vec8_kernel,
        dim3(INTER_DIM, 1), dim3(32), 0, stream,
        scratch.d_ffn_norm_act, layer.d_shared_w3, scratch.d_shared_up, H);

    {
        constexpr int swiglu_threads = 256;
        const int swiglu_blocks = (INTER_DIM + swiglu_threads - 1) / swiglu_threads;
        hipLaunchKernelGGL(
            kernel::v4_pipeline_swiglu_clamp_kernel,
            dim3(swiglu_blocks), dim3(swiglu_threads), 0, stream,
            scratch.d_shared_gate, scratch.d_shared_up, scratch.d_shared_swiglu,
            INTER_DIM, 10.0f);
    }

    hipLaunchKernelGGL(
        kernel::v4_gemv_fp16_vec8_kernel,
        dim3(H, 1), dim3(32), 0, stream,
        scratch.d_shared_swiglu, layer.d_shared_w2, scratch.d_moe_accum, INTER_DIM);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->shared_expert_output, scratch.d_moe_accum, H);
    }

    // -----------------------------------------------------------------
    // 2.10.3 — routed experts, supplied by the executor
    // -----------------------------------------------------------------
    experts.accumulate_routed(static_cast<uint32_t>(layer.layer_id), pos,
                              scratch.d_ffn_norm_act, scratch.d_topk_weights,
                              scratch.d_moe_accum);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->moe_output, scratch.d_moe_accum, H);
    }

    experts.on_routed_consumed(static_cast<uint32_t>(layer.layer_id), pos);

    // -----------------------------------------------------------------
    // I. HC FFN post expansion: res_out = comb_f · res_mid + post_f · moe_accum
    // -----------------------------------------------------------------
    hipLaunchKernelGGL(
        kernel::hc_post_kernel,
        dim3((H + 255) / 256, 1), dim3(256), 0, stream,
        scratch.d_moe_accum, scratch.d_res_mid_half, scratch.d_post_f,
        scratch.d_comb_f, scratch.d_res_out_half, H);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->post_ffn_residual,
                   scratch.d_res_out_half, HC_DIM);
    }

    // Hand the output back as the next layer's input.
    CHECK_HIP(hipMemcpyAsync(scratch.d_res_in_half, scratch.d_res_out_half,
                             HC_DIM * sizeof(half), hipMemcpyDeviceToDevice, stream));
    kernel::v4_half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, stream>>>(
        scratch.d_res_in_half, scratch.d_res_in, HC_DIM);

    return output;
}

// The single-token layer body: the pre-attention half immediately followed by the
// attention half, with nothing in between.
//
// This is the definition of "one body, not two". The chunked prefill path in
// `v4_layer_body_batch.hpp` calls the *same two functions*; it simply calls the
// first for every token in the chunk before it calls the second for any of them,
// which is the only structural difference between a chunk and a sequence of
// decode steps.
inline V4LayerBodyOutput run_layer_body_decoding(
    V4Layer& layer,
    PipelineScratchBuffers& scratch,
    const V4LayerBodyTables& tables,
    uint32_t token_id,
    uint32_t pos,
    hipStream_t stream,
    V4RoutedExpertExecutor& experts,
    V4LayerBodyObserver& observer) {
    V4LayerBodyRow row = decode_layer_body_row(scratch, layer);
    const V4LayerBodyPre pre = run_layer_body_pre_attention(
        layer, row, tables, token_id, pos, stream, observer);
    return run_layer_body_attention_tail(
        layer, row, tables, token_id, pos, stream, experts, observer, pre);
}

} // namespace aeon::core
