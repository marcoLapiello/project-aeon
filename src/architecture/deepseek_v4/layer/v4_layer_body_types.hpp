#pragma once

// -----------------------------------------------------------------------------
// The layer body's types and per-token buffer view.
//
// The shared vocabulary every layer-body phase reads: the model-level tables, the
// two seams (`V4LayerBodyObserver`, and the routed-expert executor which lives
// with the MoE phase in `v4_layer_body_moe.hpp`), the per-token row over the
// activation scratch, and the phase hand-off structs. Split out of
// `v4_layer_body.hpp` so the phase headers can share it without a cycle.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/layer/v4_activation_scratch.hpp"
#include "architecture/deepseek_v4/layer/v4_attention_trace.hpp"
#include "architecture/deepseek_v4/layer/v4_layer.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention_kernels.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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

    // Two bases, selected by layer class: a Sliding layer rotates with the plain
    // base, a compressed layer with the YaRN-on-compressed base.
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

struct V4LayerBodyOutput {
    std::vector<int32_t> topk_indices;
    std::vector<float> topk_weights;
};

// Indexer candidate selection (layer part). Descending score, ties broken to the
// **lower index** — the same rule as the router — and the degenerate case
// `candidates <= index_topk` selects every candidate with no padding.
//
// This used to be a host sort: D2H the scores, `hipStreamSynchronize`,
// `std::stable_sort`, H2D the indices, synchronize again — once per token per CSA
// layer. Both synchronizations drained the queue behind them, which is what left the
// GPU idle inside the attention phase. The selection now runs in one kernel
// (`v4_indexer_topk_kernel`), so the only ordering is the stream's own and no score
// reaches the host. The result is identical by construction: the same candidates, the
// same descending order, the same lower-index ties, the same `-1` padding — which is
// why the equivalence gates cannot see the change.
inline void select_indexer_topk(const V4Layer& layer, const float* device_scores,
                                int32_t* device_topk, size_t candidate_count,
                                hipStream_t stream) {
    if (candidate_count == 0) return;
    constexpr int kThreads = 256;
    const size_t shared_bytes = (candidate_count + 7) / 8;  // one bit per candidate
    hipLaunchKernelGGL(
        kernel::v4_indexer_topk_kernel, dim3(1), dim3(kThreads), shared_bytes, stream,
        device_scores, device_topk, static_cast<int>(candidate_count),
        static_cast<int>(layer.state_layout().index_topk));
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
//   * decode fills it from row 0 of `V4ActivationScratch`, which already has
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

    // ---- Where this token's rotated key (which is also its value) is written.
    // Decode leaves these null and gets the ring slot for its own position; a chunk
    // points them at its own key buffer, because **a chunk must not write the ring as
    // it goes** — see the proof in `v4_layer_body_batch.hpp`.
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

// The single-token row over `V4ActivationScratch` plus the layer's own indexer
// state buffers. Every pointer is exactly what the decode path uses, so the decode
// path is byte-for-byte the same computation.
inline V4LayerBodyRow decode_layer_body_row(V4ActivationScratch& scratch,
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

    // The indexer query is RoPE'd in place into the layer's own buffer.
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

} // namespace aeon::core
