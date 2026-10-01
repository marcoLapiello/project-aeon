#pragma once

// -----------------------------------------------------------------------------
// The decoder layer body.
//
// This is the composition of the layer's primitives into one layer. It exists as
// its own module for two reasons:
//
//   1. **One body, not two.** The layer body must be the same code as decode,
//      parameterised by chunk size — two bodies is how the two paths drift. Decode
//      calls `run_layer_body_decoding` below; the batched prefill path joins it
//      rather than keeping its own copy. Nothing here may acquire a decode-only
//      special case.
//
//   2. **The composition is what the gate has to certify.** A layer fails in the
//      wiring between its primitives, and the wiring is only testable if the test
//      can drive the same code the runtime drives. So the body takes its model-level
//      inputs, its observers and its routed-expert supply as *parameters*, and knows
//      nothing about artifact tiers, telemetry or the model host.
//
// Two seams keep the numerics independent of the runtime around them:
//
//   * `V4LayerBodyObserver` — the attention trace. The runtime passes the **null**
//     observer (`V4Graph` holds one, and nothing installs another), so the trace is
//     gate-only instrumentation; keeping the field list here means there is exactly
//     one description of a trace.
//
//   * `V4RoutedExpertExecutor` — the routed-expert supply system (index lookup,
//     Hot/Warm/Cold promotion, prefetch, leases, staging). All of it lives behind
//     this interface, so a body's arithmetic does not depend on which tier the
//     weights came from, and a gate can supply experts directly.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/layer/v4_layer_body_types.hpp"
#include "architecture/deepseek_v4/layer/v4_layer_body_attention.hpp"
#include "architecture/deepseek_v4/layer/v4_layer_body_moe.hpp"
#include "infrastructure/profiling/phase_profiler.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

namespace aeon::core {

// The three phases back to back: the whole attention tail for **one token**, in
// the order decode has always run it. A chunk drives the phases itself — every
// token's attention, then every token's router, then one dispatch, then every
// token's MoE — and that ordering is the only difference between a chunk and a
// sequence of decode steps.
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
    {
        auto phase = PhaseProfiler::instance().region("attention+norm (decode)", stream);
        run_layer_body_attention_and_norm(layer, scratch, tables, token_id, pos, stream,
                                          observer, pre);
    }
    V4LayerBodyOutput output;
    {
        auto phase = PhaseProfiler::instance().region("router (decode)", stream);
        output = run_layer_body_router(layer, scratch, token_id, pos, stream, observer, pre);
    }
    experts.on_routing_ready(static_cast<uint32_t>(layer.layer_id), pos,
                             output.topk_indices, output.topk_weights);
    {
        auto phase = PhaseProfiler::instance().region("moe (decode)", stream);
        run_layer_body_moe_and_post(layer, scratch, pos, stream, experts, observer, pre);
    }
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
    V4ActivationScratch& scratch,
    const V4LayerBodyTables& tables,
    uint32_t token_id,
    uint32_t pos,
    hipStream_t stream,
    V4RoutedExpertExecutor& experts,
    V4LayerBodyObserver& observer) {
    V4LayerBodyRow row = decode_layer_body_row(scratch, layer);
    V4LayerBodyPre pre;
    {
        auto phase = PhaseProfiler::instance().region("pre-attention (decode)", stream);
        pre = run_layer_body_pre_attention(layer, row, tables, token_id, pos, stream, observer);
    }
    return run_layer_body_attention_tail(
        layer, row, tables, token_id, pos, stream, experts, observer, pre);
}

} // namespace aeon::core
