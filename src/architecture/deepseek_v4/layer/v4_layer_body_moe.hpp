#pragma once

// -----------------------------------------------------------------------------
// The layer body's MoE phases and the routed-expert supply seam.
//
// The second half's routing half: the router that writes the per-token top-k ids
// and weights, the shared expert plus routed accumulate, and the HC FFN post. The
// `V4RoutedExpertExecutor` interface lives here with the phases that drive it.
// See `v4_layer_body_types.hpp` for the shared row and tables.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/layer/v4_layer_body_types.hpp"
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

    // A **layer-wide** dispatch: the `6C` requests of a chunk issued as one set.
    // The default is the per-token sequence, so an executor with nothing to batch —
    // a gate's synthetic one, for instance — needs no override, and `C = 1`
    // reproduces `on_routing_ready` exactly. A real supply overrides it to
    // deduplicate the layer's union and submit the transfers together.
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

    // The chunk-wide form: the same accumulate for `token_count` tokens in one call,
    // so a batched executor can run the routed experts as grouped GEMMs over the
    // whole chunk instead of per token.
    //
    // The activation is passed as a batch with a row stride, not a set of pointers:
    // the layer body's workspace stores each token's row inside its own 16-row padded
    // tile, so the stride (`16 x H`) is what lets the kernel read them in place. Both
    // the activations and the outputs are strided the same way.
    //
    // The default is the per-token sequence, so an executor with nothing to batch —
    // a gate's synthetic one — needs no override, and `token_count == 1` reproduces
    // `accumulate_routed`.
    virtual void accumulate_routed_batch(uint32_t layer_id, uint32_t first_position,
                                         uint32_t token_count,
                                         const half* batch_input, int input_stride,
                                         const int* batch_ids, int ids_stride,
                                         const float* batch_weights, int weights_stride,
                                         half* batch_output, int output_stride) {
        (void)batch_ids;
        (void)ids_stride;
        for (uint32_t token = 0; token < token_count; ++token) {
            accumulate_routed(layer_id, first_position + token,
                              batch_input + static_cast<size_t>(token) * input_stride,
                              batch_weights + static_cast<size_t>(token) * weights_stride,
                              batch_output + static_cast<size_t>(token) * output_stride);
        }
    }

    // Runs after the routed experts have been consumed, so the executor can
    // remember which staging slots to release on the next layer.
    virtual void on_routed_consumed(uint32_t layer_id, uint32_t position) {
        (void)layer_id;
        (void)position;
    }
};

// Enqueues the router's device work for `count` tokens. It is the whole device half of
// routing — the gate projection, the logit widening and the top-k selection — and
// nothing more: the read-back is the caller's, because a single token needs its
// selection at once while a chunk wants all `C` of them counted together.
//
// Every buffer is addressed as `base + token * stride`, so one launch sweeps the whole
// batch. That is the reason `count` is a parameter at all: the same sequence serves a
// decode step (`count == 1`, rows `H` apart in one padded tile) and a chunk (`count == C`,
// rows `kMPad * H` apart in the layered workspace) with no second kernel set and no
// gather into a dense copy.
inline void dispatch_router(
    const V4Layer& layer,
    const half* activation, int activation_stride,
    half* logits_half, float* logits,
    const uint32_t* host_token_ids, int32_t* device_token_ids,
    float* topk_weights, int32_t* topk_indices,
    int count, hipStream_t stream) {
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    constexpr int kExperts = 256;
    constexpr int kTopK = 6;

    hipLaunchKernelGGL(
        kernel::gemv_fp16_vec8_kernel,
        dim3(kExperts, count), dim3(32), 0, stream,
        activation, layer.d_gate_weight, logits_half, H, activation_stride);

    const int n = count * kExperts;
    kernel::v4_half_to_float_n_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        logits_half, logits, n);

    // The hash branch indexes `tid2eid` by token id, so the ids have to be on the
    // device. Staging them here rather than expecting the caller to have done it
    // removes a precondition that would be invisible at the call site.
    CHECK_HIP(hipMemcpyAsync(device_token_ids, host_token_ids,
                             static_cast<size_t>(count) * sizeof(uint32_t),
                             hipMemcpyHostToDevice, stream));

    hipLaunchKernelGGL(
        kernel::moe_router_kernel,
        dim3(count), dim3(64), 0, stream,
        logits,
        layer.is_hash_layer ? nullptr : layer.d_gate_bias,
        layer.d_tid2eid, device_token_ids,
        topk_weights, topk_indices,
        kExperts, kTopK, 1.5f, true);
}

// The router. Writes the per-token top-k ids and weights and returns them, because
// a chunk-wide dispatch needs every token's selection on the host before it can
// issue the layer's union as one set.
//
// Decode's form of the router: one token, one read-back. The chunk's form
// (`run_layer_body_router_batch`) enqueues the same `dispatch_router` with `count = C`
// and pays the synchronisation once for the whole chunk.
inline V4LayerBodyOutput run_layer_body_router(
    V4Layer& layer,
    V4LayerBodyRow& scratch,
    uint32_t token_id,
    uint32_t pos,
    hipStream_t stream,
    V4LayerBodyObserver& observer,
    V4LayerBodyPre pre) {
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    V4AttentionTraceRecord* attention_trace = pre.trace;
    (void)pos;

    dispatch_router(layer, scratch.d_ffn_norm_act, H,
                    scratch.d_router_logits_half, scratch.d_router_logits,
                    &token_id, scratch.d_token_id,
                    scratch.d_topk_weights, scratch.d_topk_indices, 1, stream);

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

// The shared expert, the routed accumulate, and the HC FFN post.
//
// The dispatch (`on_routing_ready`) is deliberately **not** here: a chunk issues
// the layer's union once for all of its tokens, between the router and this phase,
// so the caller owns that call. Decode issues it per token, exactly where it always
// did.
//
// The first and last phases are exposed separately so a chunk can run every token's
// shared expert, then one batched routed accumulate over them all, then every token's
// post. Decode runs the three in sequence through `run_layer_body_moe_and_post`.

// Clears the accumulation buffer and runs the shared expert into it, for one token.
inline void run_layer_body_moe_shared_expert(
    V4Layer& layer,
    V4LayerBodyRow& scratch,
    hipStream_t stream,
    V4LayerBodyObserver& observer,
    V4LayerBodyPre pre) {
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    constexpr int INTER_DIM = 2048;
    constexpr int M_PAD = 16;
    V4AttentionTraceRecord* attention_trace = pre.trace;

    CHECK_HIP(hipMemsetAsync(scratch.d_moe_accum, 0, M_PAD * H * sizeof(half), stream));

    hipLaunchKernelGGL(
        kernel::gemv_fp16_vec8_kernel,
        dim3(INTER_DIM, 1), dim3(32), 0, stream,
        scratch.d_ffn_norm_act, layer.d_shared_w1, scratch.d_shared_gate, H, H);
    hipLaunchKernelGGL(
        kernel::gemv_fp16_vec8_kernel,
        dim3(INTER_DIM, 1), dim3(32), 0, stream,
        scratch.d_ffn_norm_act, layer.d_shared_w3, scratch.d_shared_up, H, H);

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
        scratch.d_shared_swiglu, layer.d_shared_w2, scratch.d_moe_accum, INTER_DIM, INTER_DIM);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->shared_expert_output, scratch.d_moe_accum, H);
    }
}

// The HC FFN post expansion and the hand-back to the next layer's input, one token.
inline void run_layer_body_moe_post(
    V4LayerBodyRow& scratch,
    hipStream_t stream,
    V4LayerBodyObserver& observer,
    V4LayerBodyPre pre) {
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    constexpr int HC_DIM = 4 * H;
    V4AttentionTraceRecord* attention_trace = pre.trace;

    hipLaunchKernelGGL(
        kernel::hc_post_kernel,
        dim3((H + 255) / 256, 1), dim3(256), 0, stream,
        scratch.d_moe_accum, scratch.d_res_mid_half, scratch.d_post_f,
        scratch.d_comb_f, scratch.d_res_out_half, H);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->post_ffn_residual,
                   scratch.d_res_out_half, HC_DIM);
    }

    CHECK_HIP(hipMemcpyAsync(scratch.d_res_in_half, scratch.d_res_out_half,
                             HC_DIM * sizeof(half), hipMemcpyDeviceToDevice, stream));
    kernel::half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, stream>>>(
        scratch.d_res_in_half, scratch.d_res_in, HC_DIM);
}

inline void run_layer_body_moe_and_post(
    V4Layer& layer,
    V4LayerBodyRow& scratch,
    uint32_t pos,
    hipStream_t stream,
    V4RoutedExpertExecutor& experts,
    V4LayerBodyObserver& observer,
    V4LayerBodyPre pre) {
    constexpr int H = kernel::DSV4_HIDDEN_SIZE;
    V4AttentionTraceRecord* attention_trace = pre.trace;

    // Shared expert (always fires), accumulating into the cleared buffer.
    run_layer_body_moe_shared_expert(layer, scratch, stream, observer, pre);

    // Routed experts, supplied by the executor.
    experts.accumulate_routed(static_cast<uint32_t>(layer.layer_id), pos,
                              scratch.d_ffn_norm_act, scratch.d_topk_weights,
                              scratch.d_moe_accum);

    if (attention_trace != nullptr) {
        trace_copy(observer, attention_trace->moe_output, scratch.d_moe_accum, H);
    }

    experts.on_routed_consumed(static_cast<uint32_t>(layer.layer_id), pos);

    // HC FFN post expansion: res_out = comb_f · res_mid + post_f · moe_accum.
    run_layer_body_moe_post(scratch, stream, observer, pre);
}

} // namespace aeon::core
