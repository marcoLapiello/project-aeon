#pragma once

// -----------------------------------------------------------------------------
// The forward pass — the order, and only the order.
//
// This is the component the composition plan calls **G2**. It owns *when* each op
// runs and which buffers it reads; it owns no weights, no state and no storage,
// because `V4ModelHost` owns those. Nothing here is a kernel: every line is a
// dispatch of something already written, already certified, and named in the
// plan's Step list.
//
// P1 builds the **head end** of the graph — the last three ops of the forward pass,
// which turn the model's four residual streams into logits:
//
//    Step 1   token id -> embedding row, expanded across the `hc_mult` streams
//    Step 3   `hc_head` — collapse the streams to one 4096-wide vector
//    Step 4   final RMSNorm — the one with a learned weight
//    Step 5   LM head — a separate [129280, 4096] matrix, not the embedding
//
// The 43-layer loop between them is P2's, and it is the only thing missing
// between `forward_head` and a model forward pass. That is why `forward_head` is
// named for what it is rather than `forward_token`: it takes a real token id and
// produces real logits from the real head, but it has no layers in it, so calling
// it a forward pass would be the kind of naming this rewrite is meant to stop.
//
// What P1 must not drag in. The head stage reads four tensors — `embed.weight`,
// `hc_head_fn` / `base` / `scale`, `norm.weight`, `head.weight` — so it needs the
// host's steps 1–9 and none of the tiering. The expert pools, the registry and the
// executor are P2's, and adding them here before the loop exists would be exactly
// the "build it because it is easy" this plan forbids.
//
// Precision. `hc_head` and the final RMSNorm both store fp16, and the LM head
// accumulates in fp32 and stores fp16 (plan Part I §1: the head is fp16, the
// accumulate is fp32). The sampler (P3) widens to fp32 for its softmax and for the
// logit-processor seam; widening here would be a second copy of the logits with no
// consumer, which is why `logits()` is fp16 and says so.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "architecture/deepseek_v4/kernels/v4_gemv.hpp"
#include "architecture/deepseek_v4/kernels/v4_norm.hpp"
#include "architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace aeon::core {

class V4Graph {
public:
    explicit V4Graph(V4ModelHost& host) : host_(host) {}

    V4Graph(const V4Graph&) = delete;
    V4Graph& operator=(const V4Graph&) = delete;

    // --- Step 1 — the embedding lookup and the Hyper-Connections expansion ----

    // The state entering the layer stack is `hc_mult × 4096` per token: the
    // embedding row is **broadcast** across the streams, and that shape is held
    // until `hc_head` collapses it (plan Step 1).
    //
    // This is a broadcast and not four related vectors, so the four copies carry
    // the same `hidden` halves and differ in nothing. A gate can and should assert
    // that byte-for-byte, because an implementation that wrote one row and left
    // the other three stale would be numerically plausible at the very first
    // position and wrong forever after.
    void embed_token(uint32_t token_id, hipStream_t stream) {
        const uint32_t hidden = static_cast<uint32_t>(kernel::DSV4_HIDDEN_SIZE);
        const uint32_t hc_mult = static_cast<uint32_t>(host_.config().hc_mult);
        const uint32_t vocab = static_cast<uint32_t>(host_.config().vocab_size);

        // Refused rather than clamped: an out-of-range id indexes past the table,
        // and a table overrun that happens to produce finite numbers is a defect
        // no later comparison could attribute. (Same rule as trap 40.)
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
        kernel::v4_half_to_float_kernel<<<(total + kThreads - 1) / kThreads, kThreads,
                                          0, stream>>>(
            scratch.d_res_in_half, scratch.d_res_in, total);
    }

    // --- Steps 3–5 — the head end --------------------------------------------

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

        // Step 3 — `hc_head`. Not Step 2.0's pre-mix: the norm is **weightless**,
        // the RMS is over the **flattened** 16384, `hc_head_scale` is a scalar, and
        // there is no Sinkhorn and no comb, because with one output stream left
        // there is nothing to mix. One block of 32 lanes, as the kernel is written.
        hipLaunchKernelGGL(
            kernel::hc_head_wave32_kernel, dim3(1), dim3(32), 0, stream,
            scratch.d_res_in, resources.d_hc_head_fn, resources.d_hc_head_base,
            resources.d_hc_head_scale, scratch.d_hc_head_out,
            hidden, hc_mult, rms_eps, hc_eps);

        // Step 4 — the final RMSNorm. This one *does* carry a learned weight
        // (`norm.weight`), unlike the weightless head norm above and unlike the
        // weightless per-head Q norm — the three are the model's only three, and
        // which one has a tensor is a per-site fact, not a house rule.
        hipLaunchKernelGGL(
            kernel::v4_rmsnorm_wave32_kernel, dim3(1), dim3(32), 0, stream,
            scratch.d_hc_head_out, resources.d_final_norm, scratch.d_head_norm,
            hidden, rms_eps);

        // Step 5 — the LM head. A separate matrix: `tie_word_embeddings = False`,
        // and both `embed.weight` and `head.weight` exist as distinct
        // [129280, 4096] tensors in the artifact. One block per logit, fp32
        // accumulate, fp16 store.
        hipLaunchKernelGGL(
            kernel::v4_gemv_fp16_vec8_kernel, dim3(vocab), dim3(32), 0, stream,
            scratch.d_head_norm, resources.d_lm_head, scratch.d_logits, hidden);

        return scratch.d_logits;
    }

    // P1's unit of work: a token id in, logits out — the whole tail of the graph,
    // and the first thing in the rewrite whose input is text-facing and whose
    // output the sampler could consume.
    const half* forward_head(uint32_t token_id, hipStream_t stream) {
        embed_token(token_id, stream);
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
};

} // namespace aeon::core
