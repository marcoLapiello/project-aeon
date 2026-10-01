#pragma once

// -----------------------------------------------------------------------------
// Chunked batched prefill over the *same* layer body.
//
// The layer body must be the same code as decode, parameterised by chunk size — two
// bodies is how the two paths drift. No arithmetic is added here: what is added is a
// per-token workspace, the ordering, and one thing that turns out to be unavoidable.
//
// The obvious chunking — run every token's pre-attention half, then every token's
// attention half — is **not equivalent to serial for any chunk length above one**,
// and the reason is the local ring, not the compressed path.
//
// Write every key first, then attend. The ring has `C` slots, position `p` lives in
// slot `p mod C`, and query `q` attends `[q − C + 1, q]`. For a chunk `[S, E]` with
// `E > S`, the write for `E` lands in slot `E mod C`, which held position `E − C` —
// and since `E − C >= S − C + 1`, that is a key the chunk's first query still needs.
// Every query below `E` loses the keys in `[q − C + 1, E − C]`.
//
// The fix: do not write the chunk's keys into the ring while the chunk is being
// processed. A token's key goes into a per-chunk key buffer; each query attends a
// **composed row-set** — the pre-chunk ring rows still inside its window plus the
// chunk's own rows up to and including itself — and the chunk's keys are committed to
// the ring once every query has run. (The reference does the same: the current chunk's
// freshly-computed K lives in a separate per-forward buffer not yet written to the
// sliding-window ring.)
//
// The comparison can be bit-exact because the composed rows are assembled in
// **ring-slot order** (`position mod C`) with the row count passed as the kernel's
// "capacity". Slot order matters and position order would not: decode hands the
// kernel the ring and lets it iterate slots `0 … C-1`, so for a wrapped window it sums
// in a *rotation* of position order. Composing in the same rotation makes both paths
// add identical `exp` terms over bit-identical keys, so a chunk and the same tokens
// run one at a time differ by nothing at all.
//
// The compressed path needs no special care: the compressor's partial ring is written
// once per token in position order inside phase 1, and each boundary materializes
// immediately, so a row is read by every boundary that needs it before any later
// token in the chunk can reach its slot. A chunk of any length is therefore safe.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/layer/v4_layer_body.hpp"
#include "architecture/deepseek_v4/layer/v4_attention_tile.hpp"
#include "infrastructure/profiling/phase_profiler.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::core {

// A per-token tile workspace for a chunk.
//
// Every buffer is `count` rows of one token's working set. The two buffers the
// WMMA expert kernels read as 16-row tiles — `ffn_norm_act` and `moe_accum` — get
// a full `kMPad`-row tile **per token**, because the body pads one token's row 0
// into rows 1..15 and a shared tile would let token r's padding overwrite token
// r+1's input.
//
// The four indexer buffers that live on `V4Layer` in the decode path are per-token
// here: a layer holds one indexer query, one weights vector, one score vector and
// one selected top-k, and a chunk needs one of each per token or a later token's
// selection would destroy an earlier token's.
class V4LayerBodyBatchScratch {
public:
    // The body's row cap. It is a **memory** decision and nothing else: the
    // composed row-set check in `compose_local_rows` is the only state bound, and it
    // is sized from this.
    //
    // The reference implementation clamps its equivalent at `128` (`V4_PREFILL_CHUNK`),
    // but that clamp comes from *its* batch kernels' contract, not from the math. Here
    // the only bound is the composed row-set. `256` is the layer's own expert count and
    // therefore the widest chunk whose `6C` requests dedup to a whole layer, which makes
    // it the natural ceiling. It is a *launch-count* knob: the sweep's byte cost does
    // not depend on it.
    static constexpr uint32_t kMaxTokens = 256;
    static constexpr uint32_t kMPad = 16;

    ~V4LayerBodyBatchScratch() { free(); }

    V4LayerBodyBatchScratch() = default;
    V4LayerBodyBatchScratch(const V4LayerBodyBatchScratch&) = delete;
    V4LayerBodyBatchScratch& operator=(const V4LayerBodyBatchScratch&) = delete;

    // `count` tokens, plus the layer's own capacities for the two buffers whose
    // length is a property of the layer (the indexer's candidate scores and its
    // selected top-k).
    void allocate(const V4Layer& layer, uint32_t count) {
        const auto& layout = layer.state_layout();
        allocate_capacity(layout.compressed_capacity, layout.index_topk,
                          layout.local_capacity, count);
    }

    // The same allocation from **explicit** capacities rather than a layer. A
    // layer-major window enters all 43 layers and the scratch is one buffer, so it has
    // to be sized for the worst case across them and allocated once at load from the
    // configured chunk `C`, rather than re-allocated as each layer is entered.
    // `allocate(layer, count)` is the per-layer form.
    void allocate_capacity(uint32_t compressed_capacity, uint32_t index_topk,
                           uint32_t local_capacity, uint32_t count) {
        free();
        if (count == 0 || count > kMaxTokens) {
            throw std::invalid_argument(
                "V4LayerBodyBatchScratch::allocate: unsupported token count");
        }
        token_count_ = count;
        index_scores_per_token_ = compressed_capacity;
        index_topk_per_token_ = index_topk;
        // The composed row-set is the pre-chunk ring plus the chunk's own rows. It
        // is sized from the row cap rather than from `count`, so the worst-case
        // `local_capacity` across the layers is what the allocation needs.
        max_composed_rows_ = static_cast<size_t>(local_capacity) + kMaxTokens;

        constexpr uint32_t H = kernel::DSV4_HIDDEN_SIZE;
        constexpr uint32_t HC_DIM = 4 * H;
        constexpr uint32_t Q_LORA = kernel::DSV4_Q_LORA_RANK;
        constexpr uint32_t HEAD_DIM = kernel::DSV4_HEAD_DIM;
        constexpr uint32_t TOTAL_Q = kernel::DSV4_NUM_HEADS * HEAD_DIM;
        constexpr uint32_t COMPRESSOR_WIDTH = 2 * HEAD_DIM;
        constexpr uint32_t INDEXER_Q =
            kernel::DSV4_INDEX_N_HEADS * kernel::DSV4_INDEX_HEAD_DIM;
        constexpr uint32_t INDEXER_WIDTH = 2 * kernel::DSV4_INDEX_HEAD_DIM;
        constexpr uint32_t TOT_LORA = kernel::DSV4_TOTAL_O_LORA_DIM;
        constexpr uint32_t INTER = 2048;

        const size_t rows = count;
        res_in_ = alloc_array<float>(rows * HC_DIM);
        res_mid_ = alloc_array<float>(rows * HC_DIM);
        res_in_half_ = alloc_array<half>(rows * HC_DIM);
        res_mid_half_ = alloc_array<half>(rows * HC_DIM);
        res_out_half_ = alloc_array<half>(rows * HC_DIM);

        mixes_a_ = alloc_array<float>(rows * 24);
        pre_a_ = alloc_array<float>(rows * 4);
        post_a_ = alloc_array<float>(rows * 4);
        comb_a_ = alloc_array<float>(rows * 16);
        mixes_f_ = alloc_array<float>(rows * 24);
        pre_f_ = alloc_array<float>(rows * 4);
        post_f_ = alloc_array<float>(rows * 4);
        comb_f_ = alloc_array<float>(rows * 16);

        x_pre_ = alloc_array<half>(rows * H);
        x_norm_ = alloc_array<half>(rows * H);
        qa_ = alloc_array<half>(rows * Q_LORA);
        qa_norm_ = alloc_array<half>(rows * Q_LORA);
        q_ = alloc_array<half>(rows * TOTAL_Q);
        kv_ = alloc_array<half>(rows * HEAD_DIM);
        kv_norm_act_ = alloc_array<half>(rows * HEAD_DIM);
        compressor_kv_ = alloc_array<half>(rows * COMPRESSOR_WIDTH);
        compressor_score_ = alloc_array<half>(rows * COMPRESSOR_WIDTH);

        indexer_query_ = alloc_array<half>(rows * INDEXER_Q);
        indexer_weights_half_ = alloc_array<half>(rows * kernel::DSV4_INDEX_N_HEADS);
        indexer_weights_ = alloc_array<float>(rows * kernel::DSV4_INDEX_N_HEADS);
        indexer_compressor_kv_ = alloc_array<half>(rows * INDEXER_WIDTH);
        indexer_compressor_score_ = alloc_array<half>(rows * INDEXER_WIDTH);
        indexer_scores_ = alloc_array<float>(rows * index_scores_per_token_);
        indexer_topk_ = alloc_array<int32_t>(rows * index_topk_per_token_);

        attn_out_ = alloc_array<half>(rows * TOTAL_Q);
        z_lora_ = alloc_array<half>(rows * TOT_LORA);
        attn_proj_ = alloc_array<half>(rows * H);

        ffn_pre_ = alloc_array<half>(rows * H);
        ffn_norm_act_ = alloc_array<half>(rows * kMPad * H);

        router_logits_half_ = alloc_array<half>(rows * 256);
        router_logits_ = alloc_array<float>(rows * 256);
        topk_weights_ = alloc_array<float>(rows * 6);
        topk_indices_ = alloc_array<int32_t>(rows * 6);
        token_id_ = alloc_array<int32_t>(rows);

        shared_gate_ = alloc_array<half>(rows * INTER);
        shared_up_ = alloc_array<half>(rows * INTER);
        shared_swiglu_ = alloc_array<half>(rows * INTER);

        moe_accum_ = alloc_array<half>(rows * kMPad * H);

        // The chunk's own keys, held here until every query has run, and the
        // composed row-set each query attends.
        chunk_keys_ = alloc_array<half>(rows * HEAD_DIM);
        chunk_positions_ = alloc_array<int64_t>(rows);
        composed_keys_ = alloc_array<half>(max_composed_rows_ * HEAD_DIM);
        composed_positions_ = alloc_array<int64_t>(max_composed_rows_);
    }

    void free() noexcept {
        release(res_in_);
        release(res_mid_);
        release(res_in_half_);
        release(res_mid_half_);
        release(res_out_half_);
        release(mixes_a_);
        release(pre_a_);
        release(post_a_);
        release(comb_a_);
        release(mixes_f_);
        release(pre_f_);
        release(post_f_);
        release(comb_f_);
        release(x_pre_);
        release(x_norm_);
        release(qa_);
        release(qa_norm_);
        release(q_);
        release(kv_);
        release(kv_norm_act_);
        release(compressor_kv_);
        release(compressor_score_);
        release(indexer_query_);
        release(indexer_weights_half_);
        release(indexer_weights_);
        release(indexer_compressor_kv_);
        release(indexer_compressor_score_);
        release(indexer_scores_);
        release(indexer_topk_);
        release(attn_out_);
        release(z_lora_);
        release(attn_proj_);
        release(ffn_pre_);
        release(ffn_norm_act_);
        release(router_logits_half_);
        release(router_logits_);
        release(topk_weights_);
        release(topk_indices_);
        release(token_id_);
        release(shared_gate_);
        release(shared_up_);
        release(shared_swiglu_);
        release(moe_accum_);
        release(chunk_keys_);
        release(chunk_positions_);
        release(composed_keys_);
        release(composed_positions_);
        token_count_ = 0;
        bytes_allocated_ = 0;
    }

    uint32_t token_count() const noexcept { return token_count_; }

    // Exact VRAM this scratch holds, reported at load and checked against the budget's
    // allowance there, so a configuration that would overrun fails with a named message
    // instead of silently eating into the Hot pool.
    size_t bytes() const noexcept { return bytes_allocated_; }

    // Per-row pitch of the indexer candidate scores, for the batched selection. Read
    // from the scratch and not the layer because a layer-major window sizes the scratch
    // for the widest layer in the stack, which can exceed a given layer's own capacity.
    uint32_t index_scores_stride() const noexcept { return index_scores_per_token_; }

    // The chunk's key row for local index `index` (0 = the chunk's first token),
    // written during the pre-attention phase instead of the ring.
    half* chunk_key(uint32_t index) const {
        constexpr uint32_t HEAD_DIM = kernel::DSV4_HEAD_DIM;
        return chunk_keys_ + static_cast<size_t>(index) * HEAD_DIM;
    }
    int64_t* chunk_position(uint32_t index) const { return chunk_positions_ + index; }

    // The composed row-set the attention kernel reads. Key and value are the same
    // rows, so one buffer serves both of the kernel's arguments.
    half* composed_keys() const { return composed_keys_; }
    int64_t* composed_positions() const { return composed_positions_; }
    size_t max_composed_rows() const { return max_composed_rows_; }

    // The `index`-th token's row view. Exactly the structure the decode path gets
    // from `decode_layer_body_row`, which is why one body serves both.
    V4LayerBodyRow row(uint32_t index) const {
        if (index >= token_count_) {
            throw std::out_of_range("V4LayerBodyBatchScratch::row: index out of range");
        }
        constexpr uint32_t H = kernel::DSV4_HIDDEN_SIZE;
        constexpr uint32_t HC_DIM = 4 * H;
        constexpr uint32_t HEAD_DIM = kernel::DSV4_HEAD_DIM;
        constexpr uint32_t TOTAL_Q = kernel::DSV4_NUM_HEADS * HEAD_DIM;
        constexpr uint32_t INDEXER_Q =
            kernel::DSV4_INDEX_N_HEADS * kernel::DSV4_INDEX_HEAD_DIM;
        const size_t r = index;

        V4LayerBodyRow out;
        out.d_res_in = res_in_ + r * HC_DIM;
        out.d_res_mid = res_mid_ + r * HC_DIM;
        out.d_res_in_half = res_in_half_ + r * HC_DIM;
        out.d_res_mid_half = res_mid_half_ + r * HC_DIM;
        out.d_res_out_half = res_out_half_ + r * HC_DIM;

        out.d_mixes_a = mixes_a_ + r * 24;
        out.d_pre_a = pre_a_ + r * 4;
        out.d_post_a = post_a_ + r * 4;
        out.d_comb_a = comb_a_ + r * 16;
        out.d_mixes_f = mixes_f_ + r * 24;
        out.d_pre_f = pre_f_ + r * 4;
        out.d_post_f = post_f_ + r * 4;
        out.d_comb_f = comb_f_ + r * 16;

        out.d_x_pre = x_pre_ + r * H;
        out.d_x_norm = x_norm_ + r * H;
        out.d_qa = qa_ + r * kernel::DSV4_Q_LORA_RANK;
        out.d_qa_norm = qa_norm_ + r * kernel::DSV4_Q_LORA_RANK;
        out.d_q = q_ + r * TOTAL_Q;
        out.d_kv = kv_ + r * HEAD_DIM;
        out.d_kv_norm_act = kv_norm_act_ + r * HEAD_DIM;
        out.d_compressor_kv = compressor_kv_ + r * 2 * HEAD_DIM;
        out.d_compressor_score = compressor_score_ + r * 2 * HEAD_DIM;

        out.d_indexer_query = indexer_query_ + r * INDEXER_Q;
        out.d_indexer_weights_half = indexer_weights_half_ + r * kernel::DSV4_INDEX_N_HEADS;
        out.d_indexer_weights = indexer_weights_ + r * kernel::DSV4_INDEX_N_HEADS;
        out.d_indexer_compressor_kv =
            indexer_compressor_kv_ + r * 2 * kernel::DSV4_INDEX_HEAD_DIM;
        out.d_indexer_compressor_score =
            indexer_compressor_score_ + r * 2 * kernel::DSV4_INDEX_HEAD_DIM;
        out.d_indexer_scores = indexer_scores_ + r * index_scores_per_token_;
        out.d_indexer_topk_indices = indexer_topk_ + r * index_topk_per_token_;

        out.d_attn_out = attn_out_ + r * TOTAL_Q;
        out.d_z_lora = z_lora_ + r * kernel::DSV4_TOTAL_O_LORA_DIM;
        out.d_attn_proj = attn_proj_ + r * H;

        out.d_ffn_pre = ffn_pre_ + r * H;
        out.d_ffn_norm_act = ffn_norm_act_ + r * kMPad * H;

        out.d_router_logits_half = router_logits_half_ + r * 256;
        out.d_router_logits = router_logits_ + r * 256;
        out.d_topk_weights = topk_weights_ + r * 6;
        out.d_topk_indices = topk_indices_ + r * 6;
        out.d_token_id = token_id_ + r;

        out.d_shared_gate = shared_gate_ + r * 2048;
        out.d_shared_up = shared_up_ + r * 2048;
        out.d_shared_swiglu = shared_swiglu_ + r * 2048;

        out.d_moe_accum = moe_accum_ + r * kMPad * H;

        // This token's key goes to the chunk buffer, not the ring. K and V are the
        // same row, so both pointers name the same place.
        out.d_local_key_write = chunk_key(index);
        out.d_local_value_write = chunk_key(index);
        out.d_local_position_write = chunk_position(index);
        return out;
    }

private:
    // Non-static so the allocation is accounted: `bytes()` is what the load-time check
    // and the budget report read.
    template <typename T>
    T* alloc_array(size_t count) {
        T* pointer = nullptr;
        const hipError_t err = hipMalloc(&pointer, count * sizeof(T));
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("V4LayerBodyBatchScratch: hipMalloc failed: ") +
                hipGetErrorString(err));
        }
        bytes_allocated_ += count * sizeof(T);
        return pointer;
    }

    template <typename T>
    static void release(T*& pointer) noexcept {
        if (pointer != nullptr) {
            (void)hipFree(pointer);
            pointer = nullptr;
        }
    }

    size_t bytes_allocated_{0};
    uint32_t token_count_{0};
    size_t index_scores_per_token_{0};
    size_t index_topk_per_token_{0};
    size_t max_composed_rows_{0};

    float* res_in_{nullptr};
    float* res_mid_{nullptr};
    half* res_in_half_{nullptr};
    half* res_mid_half_{nullptr};
    half* res_out_half_{nullptr};

    float* mixes_a_{nullptr};
    float* pre_a_{nullptr};
    float* post_a_{nullptr};
    float* comb_a_{nullptr};
    float* mixes_f_{nullptr};
    float* pre_f_{nullptr};
    float* post_f_{nullptr};
    float* comb_f_{nullptr};

    half* x_pre_{nullptr};
    half* x_norm_{nullptr};
    half* qa_{nullptr};
    half* qa_norm_{nullptr};
    half* q_{nullptr};
    half* kv_{nullptr};
    half* kv_norm_act_{nullptr};
    half* compressor_kv_{nullptr};
    half* compressor_score_{nullptr};

    half* indexer_query_{nullptr};
    half* indexer_weights_half_{nullptr};
    float* indexer_weights_{nullptr};
    half* indexer_compressor_kv_{nullptr};
    half* indexer_compressor_score_{nullptr};
    float* indexer_scores_{nullptr};
    int32_t* indexer_topk_{nullptr};

    half* attn_out_{nullptr};
    half* z_lora_{nullptr};
    half* attn_proj_{nullptr};

    half* ffn_pre_{nullptr};
    half* ffn_norm_act_{nullptr};

    half* router_logits_half_{nullptr};
    float* router_logits_{nullptr};
    float* topk_weights_{nullptr};
    int32_t* topk_indices_{nullptr};
    int32_t* token_id_{nullptr};

    half* shared_gate_{nullptr};
    half* shared_up_{nullptr};
    half* shared_swiglu_{nullptr};

    half* moe_accum_{nullptr};

    half* chunk_keys_{nullptr};
    int64_t* chunk_positions_{nullptr};
    half* composed_keys_{nullptr};
    int64_t* composed_positions_{nullptr};
};

// Assembles one query's local row-set: the pre-chunk ring rows still inside the
// query's window, then the chunk's own rows up to and including the query.
//
// **The rows are ordered by ring slot (`position mod capacity`), not by position.**
// That is not cosmetic: decode hands the attention kernel the ring and lets it iterate
// slots `0 … capacity-1`, so the order it sums the window in is slot order — which for
// a wrapped window is a rotation of position order. Composing the chunk's rows in the
// same slot order makes the two paths add the same `exp` terms in the same sequence,
// which turns `chunk ≡ serial` into an equality rather than a tolerance.
//
// The row-set has a **closed form**, so it is one gather launch rather than a loop of
// per-slot copies: output row `r` is slot `(first + r) mod capacity` holding position
// `first + r`. The loop this replaced submitted two `hipMemcpyAsync` (a key row and a
// position) per ring slot per query — measured at ~`45 s` of host time and the single
// largest cost of the swept prefill, because the CPU could not issue a chunk's
// attention while it was still submitting `C × capacity` tiny copies.
//
// `start_position` is the chunk's first position; `chunk_row` is the query's index
// within the chunk, so the query's own row is included and no later one is.
// Returns the number of rows.
inline uint32_t compose_local_rows(
    const V4Layer& layer,
    const V4LayerBodyBatchScratch& workspace,
    uint32_t start_position,
    uint32_t chunk_row,
    uint32_t query_position,
    hipStream_t stream) {
    constexpr uint32_t HEAD_DIM = kernel::DSV4_HEAD_DIM;
    const int capacity = static_cast<int>(layer.local_cache_capacity());
    const int64_t first = std::max<int64_t>(
        0, static_cast<int64_t>(query_position) - static_cast<int64_t>(capacity) + 1);
    // The window is at most `capacity` long, so the row count is the window length.
    const uint32_t rows =
        static_cast<uint32_t>(static_cast<int64_t>(query_position) - first + 1);

    if (rows > workspace.max_composed_rows()) {
        throw std::invalid_argument("compose_local_rows: row-set exceeds the workspace");
    }
    // A chunk query may only read a key the chunk has already written. The last row is
    // the query's own, so this is the only bound that can be violated.
    if (query_position >= start_position &&
        query_position - start_position > chunk_row) {
        throw std::logic_error(
            "compose_local_rows: a query cannot read a key that is not yet written");
    }

    constexpr int kThreads = 128;
    hipLaunchKernelGGL(
        kernel::v4_compose_local_rows_kernel, dim3(rows), dim3(kThreads), 0, stream,
        layer.d_local_key_cache, workspace.chunk_key(0), workspace.composed_keys(),
        workspace.composed_positions(), first, static_cast<int64_t>(start_position),
        capacity, static_cast<int>(HEAD_DIM));
    return rows;
}

// The **union** row-set for a tile of `tile_rows` queries starting at chunk row
// `first_chunk_row`: every key position the tile needs, `[earliest window start, last
// query]`, in position order. Each query then masks into its own window in the kernel.
//
// Unlike `compose_local_rows` — which orders one query's rows by ring slot so a chunk
// stays bit-identical to decode — this labels each row with its position and lets the
// mask do the work, which is what a shared tile needs.
inline uint32_t compose_tile_rows(
    const V4Layer& layer,
    const V4LayerBodyBatchScratch& workspace,
    uint32_t start_position,
    uint32_t first_chunk_row,
    uint32_t tile_rows,
    hipStream_t stream) {
    constexpr uint32_t HEAD_DIM = kernel::DSV4_HEAD_DIM;
    const int capacity = static_cast<int>(layer.local_cache_capacity());
    const int64_t first_query = static_cast<int64_t>(start_position) + first_chunk_row;
    const int64_t last_query = first_query + tile_rows - 1;
    const int64_t first = std::max<int64_t>(0, first_query - capacity + 1);
    const uint32_t rows = static_cast<uint32_t>(last_query - first + 1);
    if (rows > workspace.max_composed_rows()) {
        throw std::invalid_argument("compose_tile_rows: row-set exceeds the workspace");
    }

    constexpr int kThreads = 128;
    hipLaunchKernelGGL(
        kernel::v4_compose_union_rows_kernel, dim3(rows), dim3(kThreads), 0, stream,
        layer.d_local_key_cache, workspace.chunk_key(0), workspace.composed_keys(),
        workspace.composed_positions(), first, static_cast<int64_t>(start_position),
        capacity, static_cast<int>(HEAD_DIM));
    return rows;
}

// Phase 1 of a chunk, for one token, exposed so a gate can drive the phases
// separately and assert the property the ordering exists for: that after every
// token's pre-attention half the ring has not moved, and no query has run.
inline V4LayerBodyPre run_chunk_pre_attention(
    V4Layer& layer,
    V4LayerBodyBatchScratch& workspace,
    const V4LayerBodyTables& tables,
    uint32_t token_id,
    uint32_t position,
    uint32_t chunk_row,
    hipStream_t stream,
    V4LayerBodyObserver& observer) {
    V4LayerBodyRow row = workspace.row(chunk_row);
    return run_layer_body_pre_attention(layer, row, tables, token_id, position,
                                        stream, observer);
}

// Phase 1 for the whole chunk: the same stages as `run_chunk_pre_attention`, with each
// dense projection issued once over every row instead of once per row. Rows are
// independent until the tail (ring, compressor and indexer state advance in position
// order there), so hoisting the projections out of the per-row loop reorders no state.
//
// The tail's per-stage sub-regions are sampled at this stride: they fire once per
// (row, stage) and would otherwise dominate the run they measure. See
// `PhaseProfiler::set_sample_stride`.
inline constexpr size_t kTailSampleStride = 16;

// The chunk's per-head weightless query norm, batched out of the tail.
//
// The tail ran `rmsnorm_unit_wave32_kernel` once per row over the row's `num_heads`
// heads (`E1`, `396 ms` of prefill GPU). The chunk's `d_q` is one contiguous
// `count × num_heads × head_dim` block, so flattening it to `count × num_heads` rows of
// `head_dim` serves every head of every row in one launch — the kernel takes its row on
// `blockIdx.x`. It must run **before** `run_chunk_rope_kv_write_batch`, which rotates the
// same `d_q` in place.
inline void run_chunk_q_norm_batch(
    V4Layer& layer,
    V4LayerBodyBatchScratch& workspace,
    uint32_t count,
    hipStream_t stream) {
    if (count == 0) return;
    constexpr int NUM_HEADS = static_cast<int>(kernel::DSV4_NUM_HEADS);
    constexpr int HEAD_DIM = static_cast<int>(kernel::DSV4_HEAD_DIM);
    auto region = PhaseProfiler::instance().region("  E1 q-norm (batched)", stream);
    const V4LayerBodyRow base = workspace.row(0);
    hipLaunchKernelGGL(kernel::rmsnorm_unit_wave32_kernel, dim3(NUM_HEADS * count),
                       dim3(32), 0, stream, base.d_q, base.d_q, HEAD_DIM, 1e-6f);
}

// The chunk's query and key RoPE and its key write, batched out of the tail.
//
// The tail ran two rope launches (one for the query heads, one for the single key
// head), two identical key/value D2D copies and a per-row position H2D, once per row —
// `0.73 s` of prefill GPU at this context. The chunk's per-row buffers are contiguous
// (`d_q` at pitch `num_heads*head_dim`, `d_kv_norm_act` at `head_dim`), so one
// `v4_forward_rope_wave32_kernel` launch over `(heads, rows)` serves every row: the
// kernel indexes the table by `blockIdx.y`, so offsetting the table by the chunk's first
// position makes row `r` read position `start + r`. The chunk writes both key and value
// to the same buffer, so the two per-row copies collapse to one bulk copy.
//
// The per-row position write is dropped: the chunk's `chunk_positions_` is never read
// (the composed row-sets carry kernel-computed positions and the commit recomputes the
// ring slot), so writing it was dead work.
inline void run_chunk_rope_kv_write_batch(
    V4Layer& layer,
    V4LayerBodyBatchScratch& workspace,
    const V4LayerBodyTables& tables,
    uint32_t start_position,
    uint32_t count,
    hipStream_t stream) {
    if (count == 0) return;
    constexpr int HEAD_DIM = static_cast<int>(kernel::DSV4_HEAD_DIM);
    constexpr int NUM_HEADS = static_cast<int>(kernel::DSV4_NUM_HEADS);
    constexpr int NOPE_DIM = static_cast<int>(kernel::DSV4_NOPE_DIM);
    constexpr int HALF_ROPE = static_cast<int>(kernel::DSV4_ROPE_DIM) / 2;

    auto region =
        PhaseProfiler::instance().region("  E2 rope + kv write (batched)", stream);
    const V4LayerBodyTables::View rope = tables.for_layer(layer.spec().attention_kind);
    const V4LayerBodyRow base = workspace.row(0);
    const float* cos = rope.cos + static_cast<size_t>(start_position) * HALF_ROPE;
    const float* sin = rope.sin + static_cast<size_t>(start_position) * HALF_ROPE;

    hipLaunchKernelGGL(kernel::v4_forward_rope_wave32_kernel, dim3(NUM_HEADS, count),
                       dim3(32), 0, stream, base.d_q, cos, sin, NUM_HEADS, HEAD_DIM,
                       NOPE_DIM, HALF_ROPE);
    hipLaunchKernelGGL(kernel::v4_forward_rope_wave32_kernel, dim3(1, count), dim3(32),
                       0, stream, base.d_kv_norm_act, cos, sin, 1, HEAD_DIM, NOPE_DIM,
                       HALF_ROPE);

    CHECK_HIP(hipMemcpyAsync(workspace.chunk_key(0), base.d_kv_norm_act,
                             static_cast<size_t>(count) * HEAD_DIM * sizeof(half),
                             hipMemcpyDeviceToDevice, stream));
}

// One batched indexer scores launch for the whole chunk, then the per-row top-k.
//
// `v4_indexer_scores_kernel` was launched once per row with `ceil(committed / 256)` =
// one block; the profile measured it at `4.3 s` of prefill GPU — `60%` of pre-attention,
// `18%` of the whole prefill — against `0.13 s` for the top-k it feeds. The rows share
// the indexer weights and read the same committed key cache, so the chunk is one larger
// grid with a row pitch.
//
// The scores are read **after** the tail loop, once every compressed entry the chunk
// materializes is in the cache. A row ranks only the `committed` entries that predate
// it, so the entries later rows add are never read by an earlier row — which is what
// makes deferring the selection safe, and why the tail takes `defer_indexer_select`
// rather than the scores moving to another phase. The top-k stays per row: it is cheap,
// and its candidate count differs per row.
inline void run_chunk_indexer_select_batch(
    V4Layer& layer,
    V4LayerBodyBatchScratch& workspace,
    uint32_t start_position,
    const std::vector<V4LayerBodyPre>& pre,
    hipStream_t stream) {
    if (layer.spec().attention_kind != V4AttentionKind::CSA) return;
    const int ratio = layer.spec().compression_ratio;
    const uint32_t count = static_cast<uint32_t>(pre.size());
    if (ratio <= 0 || count == 0 || !layer.state_layout().is_compressed()) return;

    auto region = PhaseProfiler::instance().region("  E5 indexer (batched)", stream);

    constexpr int INDEX_HEADS = static_cast<int>(kernel::DSV4_INDEX_N_HEADS);
    constexpr int INDEX_DIM = static_cast<int>(kernel::DSV4_INDEX_HEAD_DIM);
    const int capacity = static_cast<int>(layer.state_layout().compressed_capacity);
    const V4LayerBodyRow base = workspace.row(0);
    // `committed` grows with position, so the last row bounds the grid.
    const int max_committed = static_cast<int>(pre[count - 1].committed);
    if (max_committed > 0) {
        const int tiles = (max_committed + 255) / 256;
        hipLaunchKernelGGL(
            kernel::v4_indexer_scores_batch_kernel, dim3(tiles, count), dim3(256), 0,
            stream, base.d_indexer_query, INDEX_HEADS * INDEX_DIM,
            base.d_indexer_weights, INDEX_HEADS, layer.d_indexer_key_cache,
            base.d_indexer_scores, static_cast<int>(workspace.index_scores_stride()),
            static_cast<int>(start_position), ratio, capacity, INDEX_HEADS, INDEX_DIM,
            1.0f / std::sqrt(static_cast<float>(INDEX_DIM)),
            1.0f / std::sqrt(static_cast<float>(INDEX_HEADS)));
    }
    for (uint32_t row = 0; row < count; ++row) {
        const V4LayerBodyRow view = workspace.row(row);
        select_indexer_topk(layer, view.d_indexer_scores, view.d_indexer_topk_indices,
                            static_cast<size_t>(pre[row].committed), stream);
    }
}

inline std::vector<V4LayerBodyPre> run_chunk_pre_attention_batch(
    V4Layer& layer,
    V4LayerBodyBatchScratch& workspace,
    const V4LayerBodyTables& tables,
    const uint32_t* token_ids,
    uint32_t start_position,
    uint32_t count,
    hipStream_t stream,
    V4LayerBodyObserver& observer) {
    std::vector<V4LayerBodyRow> rows(count);
    {
        // Sub-regions inside "pre-attention (per token)" so an attribution run can see
        // which stage holds the `~7 s` GPU, rather than counting launches from source.
        // The two projection rounds are already batched; the per-row stages are the
        // candidates, so the split is drawn along exactly that line.
        auto region = PhaseProfiler::instance().region("  A hc mix + norm", stream);
        for (uint32_t row = 0; row < count; ++row) {
            rows[row] = workspace.row(row);
        }
        run_pre_attention_mix_batch(layer, rows[0], static_cast<int>(count), stream);
    }
    {
        auto region = PhaseProfiler::instance().region("  B x-projections", stream);
        run_pre_attention_x_projections(layer, rows[0], static_cast<int>(count), stream);
    }
    {
        auto region = PhaseProfiler::instance().region("  C lora norm", stream);
        run_pre_attention_lora_norm_batch(layer, rows[0], static_cast<int>(count), stream);
    }
    {
        auto region = PhaseProfiler::instance().region("  D q-projections", stream);
        run_pre_attention_q_projections(layer, rows[0], static_cast<int>(count), stream);
    }

    std::vector<V4LayerBodyPre> pre(count);
    {
        auto region = PhaseProfiler::instance().region("  E pre-attn tail", stream);
        // The tail's sub-regions fire once per (row, stage); at that granularity the
        // event records cost more than they measure, so they are sampled and scaled
        // back. The coarse regions above are one per chunk and stay exact.
        PhaseProfiler::instance().set_sample_stride(kTailSampleStride);
        for (uint32_t row = 0; row < count; ++row) {
            pre[row] = run_pre_attention_tail(layer, rows[row], tables, token_ids[row],
                                              start_position + row, stream, observer,
                                              /*defer_indexer_select=*/true,
                                              /*defer_rope_kv_write=*/true,
                                              /*defer_q_norm=*/true);
        }
        PhaseProfiler::instance().set_sample_stride(1);
        // All rows' keys are in place, so the chunk's rope, key write and indexer are
        // each one batched launch instead of one per row. The q-norm runs first, because
        // it writes the same `d_q` the rope then rotates.
        run_chunk_q_norm_batch(layer, workspace, count, stream);
        run_chunk_rope_kv_write_batch(layer, workspace, tables, start_position, count,
                                      stream);
        run_chunk_indexer_select_batch(layer, workspace, start_position, pre, stream);
    }
    return pre;
}

// The router for the whole chunk, in one pass.
//
// This is the per-token `run_layer_body_router` with the token loop lifted out: one
// gate GEMV over `[C, H]`, one logit widening, one top-k launch for all `C` rows, then
// a single read-back of every token's selection. It is not a second algorithm — it
// calls the same `dispatch_router` the decode path does — but it removes the `C − 1`
// `hipStreamSynchronize` calls that the per-token loop paid to learn each token's
// top-k, which is what let the CPU run ahead of the GPU between rows.
//
// The rows are addressed by stride out of the chunk workspace rather than gathered
// first: `ffn_norm_act` is the row-0 prefix of each token's padded tile, and the
// logits/top-k rows are contiguous per token, so every buffer is read in place.
inline void run_layer_body_router_batch(
    V4Layer& layer,
    V4LayerBodyBatchScratch& workspace,
    const uint32_t* token_ids,
    uint32_t count,
    hipStream_t stream,
    V4LayerBodyObserver& observer,
    const std::vector<V4LayerBodyPre>& pre,
    std::vector<V4LayerBodyOutput>& outputs) {
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    constexpr int M_PAD = static_cast<int>(V4LayerBodyBatchScratch::kMPad);
    constexpr int kRouted = 6;
    constexpr int kExperts = 256;
    if (count == 0) return;

    const int tokens = static_cast<int>(count);
    const V4LayerBodyRow base = workspace.row(0);

    dispatch_router(layer, base.d_ffn_norm_act, M_PAD * H,
                    base.d_router_logits_half, base.d_router_logits,
                    token_ids, base.d_token_id,
                    base.d_topk_weights, base.d_topk_indices, tokens, stream);

    // One read-back for the chunk. The rows are `kRouted` apart, so each token's
    // selection is a contiguous slice of the same block.
    std::vector<float> weights(static_cast<size_t>(tokens) * kRouted);
    std::vector<int32_t> indices(static_cast<size_t>(tokens) * kRouted);
    CHECK_HIP(hipMemcpyAsync(weights.data(), base.d_topk_weights,
                             weights.size() * sizeof(float), hipMemcpyDeviceToHost,
                             stream));
    CHECK_HIP(hipMemcpyAsync(indices.data(), base.d_topk_indices,
                             indices.size() * sizeof(int32_t), hipMemcpyDeviceToHost,
                             stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    outputs.assign(static_cast<size_t>(tokens), V4LayerBodyOutput{});
    for (int row = 0; row < tokens; ++row) {
        V4LayerBodyOutput& output = outputs[static_cast<size_t>(row)];
        output.topk_weights.assign(weights.begin() + row * kRouted,
                                   weights.begin() + (row + 1) * kRouted);
        output.topk_indices.assign(indices.begin() + row * kRouted,
                                   indices.begin() + (row + 1) * kRouted);
        V4AttentionTraceRecord* trace = pre[static_cast<size_t>(row)].trace;
        if (trace != nullptr) {
            trace_copy(observer, trace->router_logits,
                       base.d_router_logits + static_cast<size_t>(row) * kExperts,
                       kExperts);
            trace->routed_expert_indices.assign(output.topk_indices.begin(),
                                                output.topk_indices.end());
            trace->routed_expert_weights.assign(output.topk_weights.begin(),
                                                output.topk_weights.end());
        }
    }
}

// Phase 2c's shared expert over the whole chunk, in the three weight reads the
// per-token path pays per token.
//
// The shared expert is a dense FFN that fires on every token, so a chunk reads its
// weights once per projection instead of once per token — the same refactor
// `project_dense` already gave the pre-attention projections, and the reason the dense
// path is the largest byte term. Below `kDenseGemmMinTokens` the WMMA GEMM loses to the
// vectorized GEMV (the crossover `project_dense` was measured at), so a sub-threshold
// chunk keeps the exact per-token sequence — which is also what keeps the chunk oracle
// bit-identical to serial for its small schedules.
//
// The per-token path is not traced here: the chunk that reaches the batched branch is
// production, which observes with a null observer, and a chunk below the threshold
// still emits its per-row traces through `run_layer_body_moe_shared_expert`.
inline void run_layer_body_moe_shared_expert_batch(
    V4Layer& layer,
    V4LayerBodyBatchScratch& workspace,
    const std::vector<V4LayerBodyPre>& pre,
    uint32_t count,
    hipStream_t stream,
    V4LayerBodyObserver& observer) {
    if (count == 0) return;
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    constexpr int INTER_DIM = 2048;
    constexpr int M_PAD = static_cast<int>(V4LayerBodyBatchScratch::kMPad);

    auto region = PhaseProfiler::instance().region("shared expert (batched)", stream);

    if (count < static_cast<uint32_t>(kDenseGemmMinTokens)) {
        for (uint32_t row = 0; row < count; ++row) {
            V4LayerBodyRow view = workspace.row(row);
            run_layer_body_moe_shared_expert(layer, view, stream, observer, pre[row]);
        }
        return;
    }

    const V4LayerBodyRow base = workspace.row(0);

    // One clear for every token's accumulator tile, where the per-token path cleared
    // each row's own tile.
    CHECK_HIP(hipMemsetAsync(base.d_moe_accum, 0,
                             static_cast<size_t>(count) * M_PAD * H * sizeof(half), stream));

    // w1 and w3 read the FFN RMSNorm out of row 0 of each token's padded tile and write
    // the per-token gate/up, once for the whole chunk.
    project_dense(base.d_ffn_norm_act, layer.d_shared_w1, base.d_shared_gate,
                  static_cast<int>(count), H, INTER_DIM, M_PAD * H, INTER_DIM, stream);
    project_dense(base.d_ffn_norm_act, layer.d_shared_w3, base.d_shared_up,
                  static_cast<int>(count), H, INTER_DIM, M_PAD * H, INTER_DIM, stream);

    // The gate/up/swiglu buffers are contiguous per token, so one elementwise launch
    // over the whole chunk replaces one per token.
    {
        constexpr int swiglu_threads = 256;
        const int elements = static_cast<int>(count) * INTER_DIM;
        hipLaunchKernelGGL(
            kernel::v4_swiglu_clamp_kernel,
            dim3((elements + swiglu_threads - 1) / swiglu_threads), dim3(swiglu_threads),
            0, stream, base.d_shared_gate, base.d_shared_up, base.d_shared_swiglu,
            elements, 10.0f);
    }

    // w2 writes row 0 of each token's padded accumulator tile — where the routed reduce
    // and the FFN post read the shared contribution.
    project_dense(base.d_shared_swiglu, layer.d_shared_w2, base.d_moe_accum,
                  static_cast<int>(count), INTER_DIM, H, INTER_DIM, M_PAD * H, stream);
}

// Runs one layer and a **chunk of tokens**.
//
// The order of operations is the whole content of this function:
//   1. phase 1 — every token's pre-attention half, in position order. Keys go to
//      the chunk buffer; the compressor and the compressed entries advance
//      exactly as they do serially, because those are per-token already.
//   2. phase 2 — every token's attention half, in position order. Each composes
//      its own row-set (the pre-chunk ring rows in its window plus the chunk's own
//      rows up to itself) and attends that.
//   3. commit — the chunk's keys are written into the ring, in position order,
//      once no query can need the rows they replace any more.
//
// Throws when the chunk does not fit the caller's workspace. It does **not** throw
// for a chunk larger than either ring: the local ring is not written until the commit
// phase and the compressor materializes each boundary as its token is processed, so
// neither ring bounds the chunk.
inline std::vector<V4LayerBodyOutput> run_layer_body_chunk(
    V4Layer& layer,
    V4LayerBodyBatchScratch& workspace,
    const V4LayerBodyTables& tables,
    const uint32_t* token_ids,
    uint32_t start_position,
    uint32_t count,
    hipStream_t stream,
    V4RoutedExpertExecutor& experts,
    V4LayerBodyObserver& observer) {
    constexpr uint32_t HEAD_DIM = kernel::DSV4_HEAD_DIM;
    if (count == 0) return {};
    if (count > V4LayerBodyBatchScratch::kMaxTokens || count > workspace.token_count()) {
        throw std::invalid_argument("run_layer_body_chunk: chunk exceeds the workspace");
    }
    // There is deliberately **no** check against the local ring or the compressor's
    // partial ring here, and that is a measured decision rather than an omission.
    //
    // Both rings look like they bound the chunk size, and both were guarded against
    // until they were tested. Neither does. A chunk's keys go to the chunk buffer,
    // not the ring, so the local ring is untouched until the commit phase and a query
    // whose window predates the chunk still reads it; and the compressor
    // **materializes each boundary immediately, in position order, inside phase 1**, so
    // a row is consumed by every boundary that needs it before any later token in the
    // chunk can reach its slot. For ratio 4 the last boundary that reads position `p`
    // is at most `p + window - 1`, which is strictly before `p + window`, the first
    // write that could share `p`'s slot. The margin is exactly zero — the ring is as
    // narrow as it can be and still correct — which is why the false constraint was
    // easy to believe and why removing it needed proof rather than argument.
    //
    // Measured on `gfx1100`: a chunk of 16 tokens through the Sliding, CSA and HCA
    // classes — larger than the local ring (10) *and* than the compressor ring (8) —
    // is bit-identical to the same tokens run one at a time, across every token and
    // the whole final state, with zero differing values. The real bounds are the
    // workspace (`kMaxTokens`, `count <= workspace.token_count()`) and the composed
    // row-set, which `compose_local_rows` checks against `max_composed_rows()`.
    // Raising `kMaxTokens` is therefore a memory decision, not a state-contract one.

    // Phase 1. Every key the chunk owns is produced before any query runs, and
    // none of them touches the ring.
    std::vector<V4LayerBodyPre> pre;
    {
        auto phase = PhaseProfiler::instance().region("pre-attention (per token)", stream);
        pre = run_chunk_pre_attention_batch(layer, workspace, tables, token_ids,
                                            start_position, count, stream, observer);
    }

    // Phase 2, in three sub-phases. This is the order a chunk needs and a single
    // token cannot express: **every** token's attention and normalization, then
    // every token's router, then one layer-wide dispatch, then every token's MoE.
    //
    // The reason is the dispatch. A layer's `6C` requests can only be issued as one
    // set — deduplicated, and overlapped — once all `C` selections are known, and the
    // selection is produced *after* attention. So the router has to be lifted out of
    // the per-token tail and run for the whole chunk first. This is the same
    // decomposition the reference batch path uses (attention for all, then a
    // whole-chunk MoE union).
    //
    // The phase split, the layer-wide dispatch, and the depth-sized staging arena are
    // one change, not three. The split alone, with the per-token dispatch, separates
    // `on_routing_ready` from `on_routed_consumed` and lets the staging arena hand the
    // same six slots to every token at a layer; the layer-wide dispatch alone, without
    // the split, cannot know all `C` selections before the first token's MoE.
    // `on_routing_ready_batch` is what makes the split correct.

    // Phase 2a — attention through the FFN RMSNorm, for every token. Each token
    // composes its own row-set, because the row-set is a property of the query's
    // position.
    std::vector<V4LayerBodyRow> views(count);
    {
        auto phase = PhaseProfiler::instance().region("attention+norm (per token)", stream);
        for (uint32_t row = 0; row < count; ++row) {
            views[row] = workspace.row(row);
        }

        if (attention_tile_enabled() && attention_tile_supported(layer)) {
            // One batched attention launch per sub-tile, then the per-token tail. The
            // sub-tile keeps the composed union tight: every query scans the whole
            // union masked to its own window, so a larger tile re-reads more masked
            // keys. Sliding and HCA share their compressed keys; CSA does not (its
            // indexer top-k is per query) and keeps the per-token path below.
            constexpr uint32_t kTile = 16;
            constexpr int TOTAL_Q = static_cast<int>(kernel::DSV4_NUM_HEADS) *
                                    static_cast<int>(kernel::DSV4_HEAD_DIM);
            const uint32_t committed = pre[count - 1].committed;
            for (uint32_t first = 0; first < count; first += kTile) {
                const uint32_t tile = std::min(kTile, count - first);
                const uint32_t rows =
                    compose_tile_rows(layer, workspace, start_position, first, tile, stream);
                // CSA selects a different compressed top-k per query; pass each row's
                // indexer selection as an index block. The shared-key classes read the
                // compressed set whole (or not at all, for Sliding).
                const bool csa = layer.spec().attention_kind == V4AttentionKind::CSA;
                const bool csa_live = csa && committed > 0;
                run_attention_tile(
                    layer, views[first].d_q, TOTAL_Q, views[first].d_attn_out, TOTAL_Q,
                    static_cast<int64_t>(start_position + first), static_cast<int>(tile),
                    workspace.composed_keys(), workspace.composed_positions(),
                    static_cast<int>(rows), committed,
                    csa_live ? views[first].d_indexer_topk_indices : nullptr,
                    csa_live ? static_cast<int>(layer.state_layout().index_topk) : 0,
                    stream);
            }
            for (uint32_t row = 0; row < count; ++row) {
                run_layer_body_attention_tail(layer, views[row], tables, start_position + row,
                                              stream, observer, pre[row]);
            }
        } else {
            for (uint32_t row = 0; row < count; ++row) {
                const uint32_t query_position = start_position + row;
                views[row].composed_rows = static_cast<int32_t>(compose_local_rows(
                    layer, workspace, start_position, row, query_position, stream));
                views[row].d_composed_keys = workspace.composed_keys();
                views[row].d_composed_positions = workspace.composed_positions();
                run_layer_body_attention_and_norm(layer, views[row], tables, token_ids[row],
                                                  query_position, stream, observer, pre[row]);
            }
        }
    }

    // Phase 2b — the router for every token, in one pass. The selections have to
    // reach the host before the layer's union can be issued as one set, and the
    // batched router pays for that with a single synchronisation for the whole chunk
    // instead of one per token.
    std::vector<V4LayerBodyOutput> outputs;
    {
        auto phase = PhaseProfiler::instance().region("router (batched)", stream);
        run_layer_body_router_batch(layer, workspace, token_ids, count, stream, observer,
                                    pre, outputs);
    }
    std::vector<std::vector<int32_t>> batch_ids(count);
    std::vector<std::vector<float>> batch_weights(count);
    for (uint32_t row = 0; row < count; ++row) {
        batch_ids[row] = outputs[row].topk_indices;
        batch_weights[row] = outputs[row].topk_weights;
    }

    // The one layer-wide dispatch: the chunk's `6C` requests as a deduplicated set.
    // Leases are released at the layer boundary by the caller
    // (`V4Graph::forward_window`).
    {
        auto phase = PhaseProfiler::instance().region("routing dispatch", stream);
        experts.on_routing_ready_batch(static_cast<uint32_t>(layer.layer_id),
                                       start_position, batch_ids, batch_weights);
    }

    // Phase 2c — the shared expert for every token, then **one** batched routed
    // accumulate over them all, then every token's FFN post.
    //
    // The chunk body always calls the batched accumulate: a synthetic executor inherits
    // the base-class default, which reproduces the per-token sequence exactly (so a
    // chunk stays bit-identical to serial), while the tiered executor may run the
    // grouped path behind its own switch. That keeps one code path in the body.
    run_layer_body_moe_shared_expert_batch(layer, workspace, pre, count, stream, observer);
    {
        constexpr int kMPad = static_cast<int>(V4LayerBodyBatchScratch::kMPad);
        constexpr int kH = static_cast<int>(kernel::DSV4_HIDDEN_SIZE);
        // The model's top-k; the workspace's `topk_weights`/`topk_indices` rows are
        // `[kRoutedSlots]`, so that is their per-token stride.
        constexpr int kRoutedSlots = 6;
        auto phase = PhaseProfiler::instance().region("routed experts (batched)", stream);
        experts.accumulate_routed_batch(
            static_cast<uint32_t>(layer.layer_id), start_position, count,
            views[0].d_ffn_norm_act, kMPad * kH,
            views[0].d_topk_indices, kRoutedSlots,
            views[0].d_topk_weights, kRoutedSlots,
            views[0].d_moe_accum, kMPad * kH);
    }
    {
        auto phase = PhaseProfiler::instance().region("moe post (per token)", stream);
        for (uint32_t row = 0; row < count; ++row) {
            if (pre[row].trace != nullptr) {
                trace_copy(observer, pre[row].trace->moe_output, views[row].d_moe_accum,
                           kernel::DSV4_HIDDEN_SIZE);
            }
            experts.on_routed_consumed(static_cast<uint32_t>(layer.layer_id),
                                       start_position + row);
            run_layer_body_moe_post(views[row], stream, observer, pre[row]);
        }
    }

    // Commit. Only now may the ring move: every query that could have needed a
    // pre-chunk row has run.
    const uint32_t capacity = layer.local_cache_capacity();
    {
        auto phase = PhaseProfiler::instance().region("commit (per token)", stream);
        for (uint32_t row = 0; row < count; ++row) {
            const uint32_t position = start_position + row;
            const uint32_t slot = position % capacity;
            CHECK_HIP(hipMemcpyAsync(
                layer.d_local_key_cache + static_cast<size_t>(slot) * HEAD_DIM,
                workspace.chunk_key(row), HEAD_DIM * sizeof(half),
                hipMemcpyDeviceToDevice, stream));
            CHECK_HIP(hipMemcpyAsync(
                layer.d_local_value_cache + static_cast<size_t>(slot) * HEAD_DIM,
                workspace.chunk_key(row), HEAD_DIM * sizeof(half),
                hipMemcpyDeviceToDevice, stream));
            const int64_t absolute = static_cast<int64_t>(position);
            CHECK_HIP(hipMemcpyAsync(layer.d_local_positions + slot, &absolute,
                                     sizeof(absolute), hipMemcpyHostToDevice, stream));
        }
    }
    return outputs;
}

} // namespace aeon::core
