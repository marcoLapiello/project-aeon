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
//      or the model host.
//
// Two seams keep the numerics independent of the runtime around them:
//
//   * `V4LayerBodyObserver` — the attention trace. The runtime passes the **null**
//     observer (`V4Graph` holds one, and nothing installs another), so the trace is
//     **gate-only instrumentation**: `test_v4_layer_body_lifecycle` is the one
//     observer that captures a record, and it does so to compare a chunked body
//     against a serial one. Keeping the field list here means there is exactly one
//     description of a trace.
//
//   * `V4RoutedExpertExecutor` — the routed-expert supply system (index lookup,
//     Hot/Warm/Cold promotion, prefetch, leases, staging). All of it lives behind
//     this interface, so a layer body's arithmetic does not depend on which tier
//     the weights came from, and a gate can supply experts directly.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Split note (2026-09-27). The body's types, its attention phases and its MoE
// phases now live in focused headers — `v4_layer_body_types.hpp`,
// `v4_layer_body_attention.hpp`, `v4_layer_body_moe.hpp` — and this header is the
// orchestrator that sequences them: the per-token attention tail and the single
// decode body. The phase split itself is unchanged; only the module boundaries
// moved.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/v4_layer_body_types.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body_attention.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body_moe.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

namespace aeon::core {

// The three phases back to back: the whole attention tail for **one token**, in
// the order decode has always run it. A chunk drives the phases itself — every
// token's 2a, then every token's 2b, then one dispatch, then every token's 2c —
// and that ordering is the only difference between a chunk and a sequence of
// decode steps.
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
    run_layer_body_attention_and_norm(layer, scratch, tables, token_id, pos, stream,
                                      observer, pre);
    const V4LayerBodyOutput output =
        run_layer_body_router(layer, scratch, token_id, pos, stream, observer, pre);
    experts.on_routing_ready(static_cast<uint32_t>(layer.layer_id), pos,
                             output.topk_indices, output.topk_weights);
    run_layer_body_moe_and_post(layer, scratch, pos, stream, experts, observer, pre);
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
    const V4LayerBodyPre pre = run_layer_body_pre_attention(
        layer, row, tables, token_id, pos, stream, observer);
    return run_layer_body_attention_tail(
        layer, row, tables, token_id, pos, stream, experts, observer, pre);
}

} // namespace aeon::core
