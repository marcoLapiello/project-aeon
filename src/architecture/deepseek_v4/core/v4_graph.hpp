#pragma once

// -----------------------------------------------------------------------------
// The forward pass — the order, and only the order.
//
// The graph owns *when* each op runs and which buffers it reads; it owns no weights,
// no state and no storage, because `V4ModelHost` owns those. Nothing here is a
// kernel: every line dispatches something already written and certified.
//
// The head end turns the model's four residual streams into logits:
//
//    token id -> embedding row, expanded across the `hc_mult` streams
//    `hc_head`  — collapse the streams to one 4096-wide vector
//    final RMSNorm — the one with a learned weight
//    LM head — a separate [129280, 4096] matrix, not the embedding
//
// Between them sits the 43-layer loop. `run_layer` is the loop body — one call to
// `run_layer_body_decoding`, which chains the residual itself (the body ends by
// writing `d_res_in` from its own `d_res_out`), so no layer is special-cased: the
// attention-class branch lives inside the body and the routed experts behind the
// executor. `forward_token` is that loop with the embedding in front and the head
// behind.
//
// `run_layer` is public so an fp64 reference can be fed the **device's own** per-step
// residual and routing rather than free-running and accumulating fp16 drift — a
// router near-tie flipped by drift would make two trajectories diverge for reasons
// that are not defects.
//
// Precision: `hc_head` and the final RMSNorm store fp16; the LM head accumulates in
// fp32 and stores fp16. `logits()` is fp16 and says so; widening happens in the
// sampler, where there is a consumer.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/v4_layer_body_batch.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "platform/ops/gemv.hpp"
#include "platform/ops/rmsnorm.hpp"
#include "platform/ops/cast.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::core {

class V4Graph {
public:
    explicit V4Graph(V4ModelHost& host) : host_(host) {}

    V4Graph(const V4Graph&) = delete;
    V4Graph& operator=(const V4Graph&) = delete;

    // --- the embedding lookup and the Hyper-Connections expansion -----------

    // The state entering the layer stack is `hc_mult × 4096` per token: the embedding
    // row is **broadcast** across the streams, and that shape is held until `hc_head`
    // collapses it.
    //
    // This is a broadcast, not four related vectors: the four copies carry the same
    // `hidden` halves and differ in nothing. A gate asserts that byte-for-byte, because
    // an implementation that wrote one row and left the other three stale would be
    // plausible at the first position and wrong forever after.
    void embed_token(uint32_t token_id, hipStream_t stream) {
        const uint32_t hidden = static_cast<uint32_t>(kernel::DSV4_HIDDEN_SIZE);
        const uint32_t hc_mult = static_cast<uint32_t>(host_.config().hc_mult);
        const uint32_t vocab = static_cast<uint32_t>(host_.config().vocab_size);

        // Refused rather than clamped: an out-of-range id indexes past the table, and
        // a table overrun that happens to produce finite numbers is a defect no later
        // comparison could attribute.
        if (token_id >= vocab) {
            throw std::out_of_range(
                "V4Graph::embed_token: token id " + std::to_string(token_id) +
                " is outside the vocabulary of " + std::to_string(vocab));
        }

        const half* row = host_.resources().host_embed_table +
                          static_cast<size_t>(token_id) * hidden;
        auto& scratch = host_.scratch();

        for (uint32_t stream_index = 0; stream_index < hc_mult; ++stream_index) {
            CHECK_HIP(hipMemcpyAsync(
                scratch.d_res_in_half + static_cast<size_t>(stream_index) * hidden,
                row, static_cast<size_t>(hidden) * sizeof(half),
                hipMemcpyHostToDevice, stream));
        }

        // The HC pre-mix reads fp32, so the broadcast is widened once here rather
        // than inside every downstream kernel.
        const int total = static_cast<int>(hc_mult * hidden);
        constexpr int kThreads = 256;
        kernel::half_to_float_kernel<<<(total + kThreads - 1) / kThreads, kThreads,
                                          0, stream>>>(
            scratch.d_res_in_half, scratch.d_res_in, total);
    }

    // --- the head end --------------------------------------------------------

    // Consumes the residual the layer stack left in `d_res_in` and produces logits.
    // Returns the device logits (`[vocab_size]`, fp16), which the caller reads or
    // hands to the sampler.
    const half* head_stage(hipStream_t stream) {
        const int hidden = kernel::DSV4_HIDDEN_SIZE;
        const int hc_mult = host_.config().hc_mult;
        const int vocab = host_.config().vocab_size;
        const float rms_eps = host_.config().rms_norm_eps;
        const float hc_eps = host_.config().hc_eps;

        auto& scratch = host_.scratch();
        const auto& resources = host_.resources();

        // `hc_head`. Unlike the pre-mix, the norm is **weightless**, the RMS is over
        // the **flattened** 16384, `hc_head_scale` is a scalar, and there is no
        // Sinkhorn and no comb, because with one output stream left there is nothing to
        // mix. One block of 32 lanes, as the kernel is written.
        hipLaunchKernelGGL(
            kernel::hc_head_wave32_kernel, dim3(1), dim3(32), 0, stream,
            scratch.d_res_in, resources.d_hc_head_fn, resources.d_hc_head_base,
            resources.d_hc_head_scale, scratch.d_hc_head_out,
            hidden, hc_mult, rms_eps, hc_eps);

        // The final RMSNorm. This one carries a learned weight (`norm.weight`), unlike
        // the weightless head norm above and the weightless per-head Q norm — which one
        // has a tensor is a per-site fact, not a house rule.
        hipLaunchKernelGGL(
            kernel::rmsnorm_wave32_kernel, dim3(1), dim3(32), 0, stream,
            scratch.d_hc_head_out, resources.d_final_norm, scratch.d_head_norm,
            hidden, rms_eps);

        // The LM head. A separate matrix: `tie_word_embeddings = False`, and both
        // `embed.weight` and `head.weight` exist as distinct [129280, 4096] tensors in
        // the artifact. One block per logit, fp32 accumulate, fp16 store.
        hipLaunchKernelGGL(
            kernel::gemv_fp16_vec8_kernel, dim3(vocab), dim3(32), 0, stream,
            scratch.d_head_norm, resources.d_lm_head, scratch.d_logits, hidden);

        return scratch.d_logits;
    }

    // A token id in, logits out — the whole tail of the graph, whose input is
    // text-facing and whose output the sampler could consume.
    const half* forward_head(uint32_t token_id, hipStream_t stream) {
        embed_token(token_id, stream);
        return head_stage(stream);
    }

    // --- the 43-layer loop ---------------------------------------------------

    // One layer, one token: the graph's unit of composition. It hands the layer body
    // its layer, the model's RoPE tables and the host's expert executor, and returns
    // the body's routed selection.
    //
    // It does not advance the residual, write a position, or pick a branch. The body
    // chains `d_res_in` from its own `d_res_out`, records the position itself, and reads
    // `attention_kind` in one place; a driver repeating any of that would be a second
    // body.
    V4LayerBodyOutput run_layer(uint32_t layer_id, uint32_t token_id, uint32_t position,
                                hipStream_t stream) {
        const V4LayerBodyTables tables = host_.tables();
        return run_layer_body_decoding(
            host_.layer(layer_id), host_.scratch(), tables, token_id, position, stream,
            host_.executor(), observer_);
    }

    // The whole forward pass for one token at one position: the embedding, 43
    // calls to `run_layer`, then the head. Returns the device logits, fp16 and
    // `[vocab_size]`, exactly as `head_stage` leaves them.
    //
    // The layering is deliberate: the embedding enters the first layer through
    // `d_res_in`, and every later layer's input is the previous layer's output
    // because the body chained it. `num_layers()` is the model's own count, so a
    // 43-layer checkpoint runs 43 times with no constant in this file.
    const half* forward_token(uint32_t token_id, uint32_t position, hipStream_t stream) {
        embed_token(token_id, stream);

        const uint32_t layers = host_.num_layers();
        for (uint32_t layer = 0; layer < layers; ++layer) {
            (void)run_layer(layer, token_id, position, stream);
        }

        const half* logits = head_stage(stream);

        // The token boundary, and it is a precondition rather than an implementation
        // detail: a lease grants no ordering, so it must be held for as long as compute
        // reading that slot may be in flight. Sampling must read the logits back, so the
        // caller has a compute-stream boundary here for free — which is what makes this
        // release safe.
        host_.release_expert_leases();

        return logits;
    }

    // --- the layer-major prefill window --------------------------------------

    // Gathers `count` token rows into the carry, broadcast across the `hc_mult`
    // streams and widened to fp32 in one pass.
    //
    // `embed.weight` is host-resident (on the device it would cost ~74 Hot slots), so
    // the rows are gathered on the host into one contiguous broadcast and uploaded with
    // a **single** copy — a gather over the chunk's ids, not 4 H2D copies per row. The
    // broadcast is the embedding's shape unchanged: every stream carries the same halves.
    void embed_window(const uint32_t* token_ids, uint32_t count, hipStream_t stream) {
        const uint32_t hidden = static_cast<uint32_t>(host_.config().hidden_size);
        const uint32_t hc_mult = static_cast<uint32_t>(host_.config().hc_mult);
        const uint32_t hc_dim = hc_mult * hidden;
        const uint32_t vocab = static_cast<uint32_t>(host_.config().vocab_size);

        embed_staging_.resize(static_cast<size_t>(count) * hc_dim);
        for (uint32_t row = 0; row < count; ++row) {
            // Refused rather than clamped: an out-of-range id indexes past the table,
            // and a finite number from the wrong row is unattributable.
            const uint32_t token_id = token_ids[row];
            if (token_id >= vocab) {
                throw std::out_of_range(
                    "V4Graph::embed_window: token id " + std::to_string(token_id) +
                    " is outside the vocabulary of " + std::to_string(vocab));
            }
            const half* source = host_.resources().host_embed_table +
                                 static_cast<size_t>(token_id) * hidden;
            half* destination = embed_staging_.data() + static_cast<size_t>(row) * hc_dim;
            for (uint32_t stream_index = 0; stream_index < hc_mult; ++stream_index) {
                std::memcpy(destination + static_cast<size_t>(stream_index) * hidden, source,
                            static_cast<size_t>(hidden) * sizeof(half));
            }
        }

        CHECK_HIP(hipMemcpyAsync(host_.prefill_carry_half(), embed_staging_.data(),
                                 static_cast<size_t>(count) * hc_dim * sizeof(half),
                                 hipMemcpyHostToDevice, stream));

        const int total = static_cast<int>(count * hc_dim);
        constexpr int kThreads = 256;
        kernel::half_to_float_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            host_.prefill_carry_half(), host_.prefill_carry(), total);
    }

    // Layer-major prefill over a **window** of `count` tokens.
    //
    // The nested order is window -> layer -> chunk: for each layer, every chunk of the
    // window runs before the next layer is entered, so a layer's routed-expert set is
    // fetched once per window instead of once per chunk. The residual cannot live in
    // the per-layer chunk workspace — that is recycled as layers advance — so it is
    // carried in the host-owned buffer sized `count x hc_dim`.
    //
    // `chunk` is the body chunk `C`: the rows in flight per body invocation, at most
    // `V4LayerBodyBatchScratch::kMaxTokens`. `count` is the window `W`. Neither is
    // derived, and `chunk == count` is the degenerate schedule with one body invocation
    // per layer.
    const half* forward_window(const uint32_t* token_ids, uint32_t start_position,
                               uint32_t count, uint32_t chunk, hipStream_t stream) {
        if (count == 0) {
            throw std::invalid_argument("V4Graph::forward_window: an empty window");
        }
        if (chunk == 0 || chunk > V4LayerBodyBatchScratch::kMaxTokens) {
            throw std::invalid_argument(
                "V4Graph::forward_window: the chunk must be in [1, " +
                std::to_string(V4LayerBodyBatchScratch::kMaxTokens) + "]");
        }
        // A chunk issues its `6C` routed requests as one deduplicated set, each distinct
        // expert held in its own staging slot while in transit. Dedup can only collapse
        // onto the layer's own experts, so the real demand is `min(6C,
        // experts_per_layer)`. Checked against the **chunked window's** capacity rather
        // than the live arena, because the arena is cut to decode's smaller shape between
        // windows and only widened when a window begins.
        const uint32_t staging_needed = std::min<uint32_t>(
            static_cast<uint32_t>(host_.config().num_experts_per_tok) * chunk,
            static_cast<uint32_t>(host_.config().n_routed_experts));
        const uint32_t staging_capacity = host_.batch_staging_capacity();
        if (staging_needed > staging_capacity) {
            throw std::invalid_argument(
                "V4Graph::forward_window: the chunk needs " + std::to_string(staging_needed) +
                " staging slots but the chunked window's corridor holds " +
                std::to_string(staging_capacity) +
                " — raise AeonRuntimeConfig::prefill_chunk");
        }
        const uint32_t hc_dim = static_cast<uint32_t>(host_.config().hc_mult) *
                                static_cast<uint32_t>(host_.config().hidden_size);

        host_.ensure_prefill_carry(count);
        embed_window(token_ids, count, stream);

        const uint32_t layers = host_.num_layers();
        const V4LayerBodyTables tables = host_.tables();
        const uint32_t workspace_tokens = std::min(chunk, count);

        // The window is the swept prefill whenever the sweep is enabled and the Hot pool
        // can hold a whole layer; below the prompt-length gate it is the route-aware
        // cached supply instead. Both are the same layer-major window — only the expert
        // supply differs. The window length `count` is what the gate reads.
        host_.prefill_begin(count);

        for (uint32_t layer = 0; layer < layers; ++layer) {
            // The layer's whole set must be resident before its body runs: the
            // router lives inside the body, so its selection is not known earlier,
            // and the sweep loaded the set in layer order precisely because it is
            // the whole layer rather than a prediction.
            host_.prefill_before_layer(layer);
            host_.ensure_batch_scratch(layer, workspace_tokens);
            V4LayerBodyBatchScratch& workspace = host_.batch_scratch();

            for (uint32_t offset = 0; offset < count; offset += chunk) {
                const uint32_t span = std::min(chunk, count - offset);
                copy_carry_to_workspace(workspace, offset, span, hc_dim, stream);
                (void)run_layer_body_chunk(
                    host_.layer(layer), workspace, tables, token_ids + offset,
                    start_position + offset, span, stream, host_.executor(), observer_);
                copy_workspace_to_carry(workspace, offset, span, hc_dim, stream);
            }

            // The layer boundary, which the state forces rather than a policy choosing
            // it: anything wider would lease the whole model. A lease grants no ordering,
            // so handing it back needs a compute-stream boundary here.
            CHECK_HIP(hipStreamSynchronize(stream));
            host_.release_expert_leases();
            // The layer is dead the moment it retires — a window visits each layer
            // once — so the sweep releases its whole set and refills the room from
            // the next layers in order. LRU has nothing to rank here.
            host_.prefill_after_layer(layer);
        }

        host_.prefill_end();

        // The head reads the last position's residual, which is where the serial
        // path leaves it too (`scratch().d_res_in`).
        copy_carry_row_to_scratch(count - 1, hc_dim, stream);
        return head_stage(stream);
    }

    // --- read access, for the gate and for whatever runs above -----------------

    const float* residual() const noexcept { return host_.scratch().d_res_in; }
    const half* hc_head_output() const noexcept { return host_.scratch().d_hc_head_out; }
    const half* head_norm() const noexcept { return host_.scratch().d_head_norm; }
    const half* logits() const noexcept { return host_.scratch().d_logits; }

    V4ModelHost& host() noexcept { return host_; }
    const V4ModelHost& host() const noexcept { return host_; }

private:
    V4ModelHost& host_;

    // Host staging for the batched embedding gather. One contiguous broadcast of
    // the window's rows, uploaded once; a member so the allocation is reused rather
    // than made per call.
    std::vector<half> embed_staging_;

    // The carry <-> workspace copies. Both directions move one token's `hc_dim`
    // residual, both halves, device to device.
    void copy_carry_to_workspace(V4LayerBodyBatchScratch& workspace, uint32_t offset,
                                 uint32_t span, uint32_t hc_dim, hipStream_t stream) {
        for (uint32_t row = 0; row < span; ++row) {
            const V4LayerBodyRow view = workspace.row(row);
            const size_t source = static_cast<size_t>(offset + row) * hc_dim;
            CHECK_HIP(hipMemcpyAsync(view.d_res_in_half, host_.prefill_carry_half() + source,
                                     static_cast<size_t>(hc_dim) * sizeof(half),
                                     hipMemcpyDeviceToDevice, stream));
            CHECK_HIP(hipMemcpyAsync(view.d_res_in, host_.prefill_carry() + source,
                                     static_cast<size_t>(hc_dim) * sizeof(float),
                                     hipMemcpyDeviceToDevice, stream));
        }
    }

    void copy_workspace_to_carry(V4LayerBodyBatchScratch& workspace, uint32_t offset,
                                 uint32_t span, uint32_t hc_dim, hipStream_t stream) {
        for (uint32_t row = 0; row < span; ++row) {
            const V4LayerBodyRow view = workspace.row(row);
            const size_t destination = static_cast<size_t>(offset + row) * hc_dim;
            CHECK_HIP(hipMemcpyAsync(host_.prefill_carry_half() + destination, view.d_res_in_half,
                                     static_cast<size_t>(hc_dim) * sizeof(half),
                                     hipMemcpyDeviceToDevice, stream));
            CHECK_HIP(hipMemcpyAsync(host_.prefill_carry() + destination, view.d_res_in,
                                     static_cast<size_t>(hc_dim) * sizeof(float),
                                     hipMemcpyDeviceToDevice, stream));
        }
    }

    void copy_carry_row_to_scratch(uint32_t row, uint32_t hc_dim, hipStream_t stream) {
        auto& scratch = host_.scratch();
        const size_t source = static_cast<size_t>(row) * hc_dim;
        CHECK_HIP(hipMemcpyAsync(scratch.d_res_in, host_.prefill_carry() + source,
                                 static_cast<size_t>(hc_dim) * sizeof(float),
                                 hipMemcpyDeviceToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(scratch.d_res_in_half, host_.prefill_carry_half() + source,
                                 static_cast<size_t>(hc_dim) * sizeof(half),
                                 hipMemcpyDeviceToDevice, stream));
    }

    // The body requires an observer; a real one that traces a forward pass is not yet
    // written. The null observer copies nothing, so the hot path is the arithmetic and
    // nothing else.
    V4NullLayerBodyObserver observer_;
};

} // namespace aeon::core
