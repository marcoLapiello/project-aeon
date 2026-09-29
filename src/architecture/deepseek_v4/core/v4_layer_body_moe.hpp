#pragma once

// -----------------------------------------------------------------------------
// The layer body's MoE phases and the routed-expert supply seam
// (plan Steps 2.9 – 2.11).
//
// The second half's routing half: the router that writes the per-token top-k ids
// and weights, the shared expert plus routed accumulate, and the HC FFN post. The
// `V4RoutedExpertExecutor` interface lives here with the phases that drive it.
// See `v4_layer_body_types.hpp` for the shared row and tables.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/v4_layer_body_types.hpp"
#include "architecture/deepseek_v4/kernels/hc_sinkhorn.hpp"
#include "architecture/deepseek_v4/kernels/moe_router.hpp"
#include "architecture/deepseek_v4/kernels/v4_swiglu_clamp.hpp"
#include "platform/ops/cast.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace aeon::core {

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

    // A **layer-wide** dispatch: the `6C` requests of a chunk issued as one set
    // (Step 6 D1). The default is the per-token sequence, so an executor with
    // nothing to batch — a gate's synthetic one, for instance — needs no override,
    // and `C = 1` reproduces `on_routing_ready` exactly. A real supply overrides it
    // to deduplicate the layer's union and submit the transfers together, which is
    // the whole point: one set of reads instead of `C` serialized ones.
    virtual void on_routing_ready_batch(uint32_t layer_id,
                                        uint32_t first_position,
                                        const std::vector<std::vector<int32_t>>& ids,
                                        const std::vector<std::vector<float>>& weights) {
        for (size_t index = 0; index < ids.size(); ++index) {
            on_routing_ready(layer_id, first_position + static_cast<uint32_t>(index),
                             ids[index], weights[index]);
        }
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

// Phase 2b — the router (step H). Writes the per-token top-k ids and weights and
// returns them, because a chunk-wide dispatch needs every token's selection on the
// host before it can issue the layer's union as one set.
inline V4LayerBodyOutput run_layer_body_router(
    V4Layer& layer,
    V4LayerBodyRow& scratch,
    uint32_t token_id,
    uint32_t pos,
    hipStream_t stream,
    V4LayerBodyObserver& observer,
    V4LayerBodyPre pre) {
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    constexpr int M_PAD = 16;
    V4AttentionTraceRecord* attention_trace = pre.trace;
    (void)M_PAD;

    // -----------------------------------------------------------------
    // H. MoE router (2.9)
    // -----------------------------------------------------------------
    hipLaunchKernelGGL(
        kernel::gemv_fp16_vec8_kernel,
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
    return output;
}

// Phase 2c — the shared expert, the routed accumulate, and the HC FFN post.
//
// The dispatch (`on_routing_ready`) is deliberately **not** here: a chunk issues
// the layer's union once for all of its tokens, between 2b and 2c, so the caller
// owns that call. Decode issues it per token, exactly where it always did.
inline void run_layer_body_moe_and_post(
    V4Layer& layer,
    V4LayerBodyRow& scratch,
    uint32_t pos,
    hipStream_t stream,
    V4RoutedExpertExecutor& experts,
    V4LayerBodyObserver& observer,
    V4LayerBodyPre pre) {
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    constexpr int HC = 4;
    constexpr int HC_DIM = HC * H;
    constexpr int INTER_DIM = 2048;
    constexpr int M_PAD = 16;
    V4AttentionTraceRecord* attention_trace = pre.trace;

    // -----------------------------------------------------------------
    // 2.10.4 — shared expert (always fires), accumulating into the cleared buffer
    // -----------------------------------------------------------------
    // Clear MoE accumulation buffer
    CHECK_HIP(hipMemsetAsync(scratch.d_moe_accum, 0, M_PAD * H * sizeof(half), stream));

    hipLaunchKernelGGL(
        kernel::gemv_fp16_vec8_kernel,
        dim3(INTER_DIM, 1), dim3(32), 0, stream,
        scratch.d_ffn_norm_act, layer.d_shared_w1, scratch.d_shared_gate, H);
    hipLaunchKernelGGL(
        kernel::gemv_fp16_vec8_kernel,
        dim3(INTER_DIM, 1), dim3(32), 0, stream,
        scratch.d_ffn_norm_act, layer.d_shared_w3, scratch.d_shared_up, H);

    {
        constexpr int swiglu_threads = 256;
        const int swiglu_blocks = (INTER_DIM + swiglu_threads - 1) / swiglu_threads;
        hipLaunchKernelGGL(
            kernel::v4_swiglu_clamp_kernel,
            dim3(swiglu_blocks), dim3(swiglu_threads), 0, stream,
            scratch.d_shared_gate, scratch.d_shared_up, scratch.d_shared_swiglu,
            INTER_DIM, 10.0f);
    }

    hipLaunchKernelGGL(
        kernel::gemv_fp16_vec8_kernel,
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
    kernel::half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, stream>>>(
        scratch.d_res_in_half, scratch.d_res_in, HC_DIM);
}

} // namespace aeon::core
