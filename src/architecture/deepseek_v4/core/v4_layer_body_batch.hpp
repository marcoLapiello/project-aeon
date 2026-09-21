#pragma once

// -----------------------------------------------------------------------------
// Tier 3, item 19 — chunked batched prefill over the *same* layer body.
//
// The plan's requirement is one sentence: "The layer body must be the same code as
// decode, parameterised by chunk size. Two bodies is how the two paths drift."
// No arithmetic is added here. What is added is a per-token workspace, the
// ordering, and one thing that turns out to be unavoidable.
//
// ## The structural problem this file exists to solve
//
// The obvious chunking — run every token's pre-attention half, then every token's
// attention half — is **not equivalent to serial, for any chunk length above
// one**, and the reason is the local ring, not the compressed path.
//
// Write every key first, then attend. The local ring has `C` slots and position
// `p` lives in slot `p mod C`. Query `q` attends positions `[q − C + 1, q]`. In a
// chunk spanning `[S, E]` with `E > S`, the write for position `p ∈ [S, E]` lands
// in slot `p mod C`, which before the write held position `p − C`. Take `p = E`
// (the chunk's last write) and `q = S` (its first query): `E − C ≥ S − C + 1`
// whenever `E ≥ S + 1`, i.e. whenever the chunk has more than one token. So the
// chunk's last write evicts a key — the *oldest* key of the window — that its
// first query still needs. And it is not only the first query: every query below
// `E` loses the keys in `[q − C + 1, E − C]`.
//
// This is trap 39, and it is why "as written, the equivalence gate cannot pass" in
// Part III. A chunk of length 1 keeps equivalence, which is exactly what makes a
// chunk-1 run the right reference: the property under test is that batching does
// not change the answer, and at length 1 there is no batching.
//
// ## The fix, which is also the canonical design
//
// Do not write the chunk's keys into the ring while the chunk is being processed.
// A token's key goes into a per-chunk key buffer (which the body's
// `d_local_key_write` / `d_local_position_write` allow); each query then attends a
// **composed row-set** — the pre-chunk ring rows still inside its window plus the
// chunk's own rows up to and including itself — and the chunk's keys are committed
// to the ring once every query has run.
//
// This is the reference's own shape: "the current chunk's freshly-computed K lives
// in a separate per-forward `kv [total_tokens, D]` not yet written to the SWA
// ring", with the row-set assembled as `[compressed | swa positional]`
// [plan 2.4.4].
//
// ## Why the comparison can be bit-exact
//
// The composed rows are assembled in **ring-slot order** (`position mod C`) and the
// row count is passed to the attention kernel as its "capacity", so the kernel's own
// window filter (`local_start <= key_position <= current_position`) is satisfied by
// every row.
//
// Slot order matters and position order would not do. The decode path hands the
// kernel the ring itself and lets it iterate slots `0 … C-1`, so the order it sums
// the window in is slot order — and for a window that has wrapped, that is a
// *rotation* of position order, not position order. Composing in the same rotation
// makes both paths add the identical `exp` terms in the identical sequence over
// bit-identical keys, so a chunk and the same tokens run one at a time differ by
// nothing at all. The item-19 gate asserts that as equality rather than as a
// tolerance, because with the orders matched there is no numerical reason for any
// difference to exist. (Mutation M19-3 assembles descending slot order and is
// killed.)
//
// The compressed path needs no special care, and that is a measured claim rather
// than an obvious one. The compressor's partial ring is written once per token in
// position order inside phase 1, and each boundary **materializes immediately**, so
// a row is read by every boundary that needs it before any later token in the chunk
// can reach its slot. The ring is therefore exactly as wide as it must be (one
// window) and no wider, and a chunk of any length is safe — the compressed entries
// themselves are appended to a growing array rather than to a ring. This was
// previously guarded against, and the guard was wrong: item 19's gate now asserts
// the opposite, that a chunk larger than both rings is bit-identical to serial.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/v4_layer_body.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
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
    static constexpr uint32_t kMaxTokens = 16;
    static constexpr uint32_t kMPad = 16;

    ~V4LayerBodyBatchScratch() { free(); }

    V4LayerBodyBatchScratch() = default;
    V4LayerBodyBatchScratch(const V4LayerBodyBatchScratch&) = delete;
    V4LayerBodyBatchScratch& operator=(const V4LayerBodyBatchScratch&) = delete;

    // `count` tokens, plus the layer's own capacities for the two buffers whose
    // length is a property of the layer (the indexer's candidate scores and its
    // selected top-k).
    void allocate(const V4Layer& layer, uint32_t count) {
        free();
        if (count == 0 || count > kMaxTokens) {
            throw std::invalid_argument(
                "V4LayerBodyBatchScratch::allocate: unsupported token count");
        }
        token_count_ = count;
        const auto& layout = layer.state_layout();
        index_scores_per_token_ = layout.compressed_capacity;
        index_topk_per_token_ = layout.index_topk;
        // The composed row-set is the pre-chunk ring plus the chunk's own rows.
        max_composed_rows_ = static_cast<size_t>(layout.local_capacity) + kMaxTokens;

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
    }

    uint32_t token_count() const noexcept { return token_count_; }

    // The chunk's key row for local index `index` (0 = the chunk's first token),
    // written during the pre-attention phase instead of the ring.
    half* chunk_key(uint32_t index) const {
        constexpr uint32_t HEAD_DIM = kernel::DSV4_HEAD_DIM;
        return chunk_keys_ + static_cast<size_t>(index) * HEAD_DIM;
    }
    int64_t* chunk_position(uint32_t index) const { return chunk_positions_ + index; }

    // The composed row-set the attention kernel reads. Key and value are the same
    // rows (trap 6), so one buffer serves both of the kernel's arguments.
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
        // same row, so both pointers name the same place (trap 6).
        out.d_local_key_write = chunk_key(index);
        out.d_local_value_write = chunk_key(index);
        out.d_local_position_write = chunk_position(index);
        return out;
    }

private:
    template <typename T>
    static T* alloc_array(size_t count) {
        T* pointer = nullptr;
        const hipError_t err = hipMalloc(&pointer, count * sizeof(T));
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("V4LayerBodyBatchScratch: hipMalloc failed: ") +
                hipGetErrorString(err));
        }
        return pointer;
    }

    template <typename T>
    static void release(T*& pointer) noexcept {
        if (pointer != nullptr) {
            (void)hipFree(pointer);
            pointer = nullptr;
        }
    }

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
// That is not cosmetic. The decode path hands the attention kernel the ring itself
// and lets it iterate slots `0 … capacity-1`, so the order it sums the window in is
// slot order — which for a wrapped window is a rotation of the position order, not
// position order. Composing the chunk's rows in the same slot order makes the two
// paths add the same `exp` terms in the same sequence, which is what turns
// `chunk ≡ serial` into an equality rather than a tolerance. (The kernel's own
// window filter then passes every row: the window has exactly `rows` members and
// `local_start = current_position - rows + 1` is its first position.)
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
    const uint32_t capacity = layer.local_cache_capacity();
    const int64_t first = std::max<int64_t>(
        0, static_cast<int64_t>(query_position) - static_cast<int64_t>(capacity) + 1);

    half* destination = workspace.composed_keys();
    uint32_t rows = 0;
    for (uint32_t slot = 0; slot < capacity; ++slot) {
        // The one position in `[first, query_position]` congruent to this slot, if
        // the window reaches it. The window is at most `capacity` long, so there is
        // at most one.
        const int64_t offset = (static_cast<int64_t>(slot) - first) %
            static_cast<int64_t>(capacity);
        const int64_t position = first + (offset < 0 ? offset + capacity : offset);
        if (position > static_cast<int64_t>(query_position)) continue;

        const half* source = nullptr;
        if (position < static_cast<int64_t>(start_position)) {
            // A pre-chunk key: still in the ring, which the chunk has not touched.
            source = layer.d_local_key_cache + static_cast<size_t>(slot) * HEAD_DIM;
        } else {
            const uint32_t index =
                static_cast<uint32_t>(position - static_cast<int64_t>(start_position));
            if (index > chunk_row) {
                throw std::logic_error(
                    "compose_local_rows: a query cannot read a key that is not yet written");
            }
            source = workspace.chunk_key(index);
        }
        CHECK_HIP(hipMemcpyAsync(destination + static_cast<size_t>(rows) * HEAD_DIM,
                                 source, HEAD_DIM * sizeof(half),
                                 hipMemcpyDeviceToDevice, stream));
        // The positions live in device memory too, so they are written the same way.
        CHECK_HIP(hipMemcpyAsync(workspace.composed_positions() + rows, &position,
                                 sizeof(position), hipMemcpyHostToDevice, stream));
        ++rows;
    }
    if (rows > workspace.max_composed_rows()) {
        throw std::invalid_argument("compose_local_rows: row-set exceeds the workspace");
    }
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

// Runs Steps 2.0 – 2.11 for **one layer and a chunk of tokens**.
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
// for a chunk larger than either ring: the local ring is not written until the
// commit phase and the compressor materializes each boundary as its token is
// processed, so neither ring bounds the chunk. The measured proof is in item 19's
// gate and in the comment on the guard below.
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
    // not the ring (trap 39), so the local ring is untouched until the commit phase
    // and a query whose window predates the chunk still reads it; and the compressor
    // **materializes each boundary immediately, in position order, inside phase 1**, so
    // a row is consumed by every boundary that needs it before any later token in the
    // chunk can reach its slot. For ratio 4 the last boundary that reads position `p`
    // is at most `p + window - 1` (the r128 case:
    // `p + ratio - 1 < p + window`), which is strictly before `p + window`, the first
    // write that could share `p`'s slot. The margin is exactly zero — the ring is as
    // narrow as it can be and still correct — which is why the false constraint was
    // easy to believe and why removing it needed proof rather than argument.
    //
    // Measured (item 19's gate, `gfx1100`): a chunk of 16 tokens through the
    // Sliding, CSA and HCA classes — larger than the local ring (10) *and* than the
    // compressor ring (8) — is bit-identical to the same tokens run one at a time,
    // across every token and the whole final state, with zero differing values. The
    // real bounds are the workspace (`kMaxTokens`, `count <= workspace.token_count()`)
    // and the composed row-set, which `compose_local_rows` checks against
    // `max_composed_rows()`. Raising `kMaxTokens` is therefore a memory decision, not
    // a state-contract one.

    // Phase 1. Every key the chunk owns is produced before any query runs, and
    // none of them touches the ring.
    std::vector<V4LayerBodyPre> pre(count);
    for (uint32_t row = 0; row < count; ++row) {
        pre[row] = run_chunk_pre_attention(layer, workspace, tables, token_ids[row],
                                           start_position + row, row, stream, observer);
    }

    // Phase 2, in three sub-phases. This is the order a chunk needs and a single
    // token cannot express: **every** token's attention and normalization, then
    // every token's router, then one layer-wide dispatch, then every token's MoE.
    //
    // The reason is the dispatch (Step 6 D1). A layer's `6C` requests can only be
    // issued as one set — deduplicated, and overlapped — once all `C` selections
    // are known, and the selection is produced *after* attention. So the router has
    // to be lifted out of the per-token tail and run for the whole chunk first.
    // This is the same decomposition colibri's `coli_v4_block_window_batch_ref`
    // uses (attention for all, then a whole-chunk MoE union).

    // Phase 2a — attention through the FFN RMSNorm, for every token. Each token
    // composes its own row-set, because the row-set is a property of the query's
    // position.
    for (uint32_t row = 0; row < count; ++row) {
        const uint32_t query_position = start_position + row;
        V4LayerBodyRow view = workspace.row(row);
        view.composed_rows = static_cast<int32_t>(compose_local_rows(
            layer, workspace, start_position, row, query_position, stream));
        view.d_composed_keys = workspace.composed_keys();
        view.d_composed_positions = workspace.composed_positions();
        run_layer_body_attention_and_norm(layer, view, tables, token_ids[row],
                                          query_position, stream, observer, pre[row]);
    }

    // Phase 2b — the router, for every token. All `C` selections are known when
    // this ends, which is what makes the single dispatch below possible.
    std::vector<V4LayerBodyOutput> outputs(count);
    std::vector<std::vector<int32_t>> batch_ids(count);
    std::vector<std::vector<float>> batch_weights(count);
    for (uint32_t row = 0; row < count; ++row) {
        outputs[row] = run_layer_body_router(layer, workspace.row(row), token_ids[row],
                                             start_position + row, stream, observer,
                                             pre[row]);
        batch_ids[row] = outputs[row].topk_indices;
        batch_weights[row] = outputs[row].topk_weights;
    }

    // Phase 2b' — the layer-wide dispatch. One set of `6C` requests, deduplicated
    // to the layer's union, submitted together.
    experts.on_routing_ready_batch(static_cast<uint32_t>(layer.layer_id), start_position,
                                   batch_ids, batch_weights);

    // Phase 2c — shared expert, routed accumulate, HC FFN post, per token.
    for (uint32_t row = 0; row < count; ++row) {
        run_layer_body_moe_and_post(layer, workspace.row(row), start_position + row,
                                    stream, experts, observer, pre[row]);
    }

    // Commit. Only now may the ring move: every query that could have needed a
    // pre-chunk row has run.
    const uint32_t capacity = layer.local_cache_capacity();
    for (uint32_t row = 0; row < count; ++row) {
        const uint32_t position = start_position + row;
        const uint32_t slot = position % capacity;
        CHECK_HIP(hipMemcpyAsync(layer.d_local_key_cache + static_cast<size_t>(slot) * HEAD_DIM,
                                 workspace.chunk_key(row), HEAD_DIM * sizeof(half),
                                 hipMemcpyDeviceToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(layer.d_local_value_cache + static_cast<size_t>(slot) * HEAD_DIM,
                                 workspace.chunk_key(row), HEAD_DIM * sizeof(half),
                                 hipMemcpyDeviceToDevice, stream));
        const int64_t absolute = static_cast<int64_t>(position);
        CHECK_HIP(hipMemcpyAsync(layer.d_local_positions + slot, &absolute,
                                 sizeof(absolute), hipMemcpyHostToDevice, stream));
    }
    return outputs;
}

} // namespace aeon::core
