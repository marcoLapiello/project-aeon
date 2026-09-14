# Project Aeon V4 Review and Fix Report

**Date:** 2026-09-14
**Source review:** [AEON_V4_REVIEW_AND_FIX_PLAN.md](AEON_V4_REVIEW_AND_FIX_PLAN.md)
**Status:** Paused after bounded validation; latest test-acceptance corrections recorded below

## Executive Summary

The external review contains several valuable defects, but it also assumes a different runtime and reference behavior in a few important places. The selected Aeon artifact is an F16 dense model package, and the live pipeline already validates the required dense tensor dtype, shape, and byte size through [V4ModelContract](../../../src/architecture/deepseek_v4/core/v4_model_contract.hpp) before uploading weights.

The most valuable semantic finding was the local KV value-frame mismatch. Independent SGLang DeepSeek-V4 code rotates the KV vector before storing it, uses the rotated vector as both attention key and value, and inverse-rotates the attention output before `wo_a`. Aeon was storing the raw local value while still inverse-rotating the output. The serial and batched local-value writes now use the rotated vector, and the CPU oracle was aligned with that behavior.

The earlier long full-block trace failure was initially misclassified as cumulative device corruption. The reported `layer 0 HC mixes` error actually occurred later, during the real-prompt trace: the test hardcoded embedding token ID `1` while the real prompt begins with token ID `0`. The trace acceptance criterion was therefore comparing valid production output against the wrong expected embedding. After recording token provenance in each trace and correcting the expectation, the narrow 19-token real-prompt trace passes all 43 layers. A full rerun after these test corrections remains pending; no claim of coherent end-to-end generation is made yet.

## Changes Applied

### Confirmed correctness and safety fixes

1. **Router barrier:** moved the shared routing-weight reduction outside the `tid < top_k` predicate so all 64 threads reach both barriers in [moe_router.hpp](../../../src/architecture/deepseek_v4/kernels/moe_router.hpp).
2. **W4A16 dispatcher validation:** replaced silent shape and expert-count returns with `std::invalid_argument` failures in [aeon_w4a16_swizzled_gemv.hpp](../../../src/backend/swizzled_w4a16/kernels/aeon_w4a16_swizzled_gemv.hpp), [aeon_moe_fused_w13.hpp](../../../src/backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp), and [aeon_moe_fused_w2.hpp](../../../src/backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp).
3. **HC pre-combine barriers:** moved out-of-range exits after the shared-memory barrier in both HC pre-combine kernels in [hc_sinkhorn.hpp](../../../src/architecture/deepseek_v4/kernels/hc_sinkhorn.hpp).
4. **Staging lifetime:** moved final staging-slot release in [v4_pipeline.hpp](../../../src/architecture/deepseek_v4/core/v4_pipeline.hpp) below the compute-stream synchronization and argmax readback.
5. **Local KV value frame:** changed serial and batched local-value cache writes in [v4_pipeline.hpp](../../../src/architecture/deepseek_v4/core/v4_pipeline.hpp) to store post-RoPE KV, matching the rotated local-key cache and the independent reference path.
6. **Oracle alignment:** updated [v4_attention_oracle.hpp](../../../src/architecture/deepseek_v4/reference/v4_attention_oracle.hpp) so local key and value entries share the rotated frame.
7. **Configuration-discarding overload:** removed the unused two-argument `V4ModelResources::initialize` overload from [v4_model_resources.hpp](../../../src/architecture/deepseek_v4/core/v4_model_resources.hpp). The production pipeline uses the parsed model config explicitly.
8. **Trace record lifetime guard:** added a compute-stream synchronization before trace records are cleared and reallocated. This did not resolve the cumulative full-trace failure, but it removes one plausible asynchronous host-buffer lifetime hazard.
9. **Trace token provenance:** added the input `token_id` to `V4AttentionTraceRecord` and populated it from the serial pipeline so full-block checks use the actual embedding row.
10. **HC acceptance staging:** separated scalar HC projection and pre-mix checks from pre-combine and RMSNorm checks. Downstream FP16 expectations now use the captured production pre-mix after that pre-mix has independently passed scalar parity.
11. **FP16 tolerance correction:** integrated half-output checks now use an explicit absolute-plus-relative tolerance (`0.02 + 0.002 * max(1, |expected|)`) so a one-quantum FP16 difference at large activations is not treated as a semantic failure. Float32 projection/state tolerances remain unchanged.

All pre-existing user and debugging changes in the worktree were preserved.

## Review Findings That Were Reclassified

### Already covered by the repository

- The runtime loads the model config, validates the supported DeepSeek-V4 contract, and validates all required dense tensors before resource and layer initialization. The current package contains 1,271 dense tensors and declares F16 dense weights, not BF16.
- The selected model contract validates required dtype, shape, and byte size, including embedding, LM-head, attention, HC, router, compressor, indexer, and shared-expert tensors.
- The converter copies dense tensor bytes and performs a layout-only swizzle for already packed routed experts. Existing real expert parity and W4A16 tests cover the current artifact representation.

The lower-level upload helper is still not self-validating if someone bypasses `V4ModelContract`, so the reviewer\'s robustness recommendation remains reasonable for a future API cleanup. It is not evidence of the current BF16-as-FP16 failure in this artifact.

### Contradicted by the independent reference

- **Main RoPE factor:** pure sliding layers intentionally use unscaled main RoPE. Compressed C4/C128 layers use the compressed YaRN table with factor 16. The current `factor=1.0` main table is therefore consistent with the independent SGLang DeepSeek-V4 implementation.
- **YaRN ramp units:** the independent reference uses the same full-dimension correction range with a pair-index ramp as the current implementation. The proposed extra `0.5` correction is not applied.
- **YaRN mscale:** the independent reference path uses `1/sqrt(512)` attention scale and does not apply the proposed additional `mscale^2` factor in this DeepSeek-V4 attention path. No speculative temperature change was made.
- **Inverse RoPE:** the inverse output rotation is intentional. The actual mismatch was that Aeon cached raw local values instead of the rotated KV vector that the reference uses.

### Still unresolved or intentionally deferred

- FP32 LM-head logits and argmax remain a plausible numerical-quality improvement, but were not mixed into this investigation before the cumulative trace failure is explained.
- FP32 accumulation in the deterministic routed-expert path remains a valid numerical cleanup candidate.
- HC stream initialization, Sinkhorn orientation/alpha, chat-template conventions, tokenizer parity, and other P2 items still require independent reference or golden-output evidence.
- A full coherent generation comparison against an independent accelerator-backed reference remains open.

## Validation Results

### Passing focused checks

- W4A16 swizzle, GEMV, dual GEMV, fused W13, and fused W2: **5/5 passed**.
- HC Sinkhorn and pre-combine device test: **passed**.
- CPU attention oracle: **passed**.
- Real attention oracle: **passed**.
- Model contract: **passed**.
- Layer-state device and class-attention device tests: **passed**.
- Touched-file diagnostics: no errors reported.
- `git diff --check`: passed.

### Superseded failing result

The earlier `test_v4_stage4_trace` run was redirected to avoid flooding the
terminal and produced:

```text
[PASS] Stage 4 HIP/oracle trace layer 0 Sliding through position 131
[PASS] Stage 4 HIP/oracle trace layer 2 CSA through position 131
[PASS] Stage 4 HIP/oracle trace layer 3 HCA through position 131
what(): layer 0 HC mixes mismatch at layer 0, position 0:
max_abs=88.590942, tolerance=0.002000
```

The message was misleading because the test continued after the all-layer
position-0 trace and later replayed a real prompt whose first token was `0`.
`compare_layer_zero_attention_input()` expected token `1`, so this was a test
oracle defect, not evidence of cumulative device corruption.

### Latest bounded result

After adding token provenance and correcting the HC checks, the narrow
`real-only` diagnostic ran the exact 19-token formatted prompt through all 43
layers and passed every layer trace through position 18. The relevant evidence
was:

- Layers 0, 1, and 2 passed before the HC tolerance correction exposed the
	next issue at layer 3.
- At layer 7, scalar HC mix error was `0.000177383`, scalar-to-production
	pre-mix error was `2.14577e-06`, and the pre-combine difference was one FP16
	quantum (`79.1875` versus `79.25`).
- The dtype-aware FP16 criterion then passed all 43 real-prompt layer traces.

The full `test_v4_stage4_trace` target has not yet been rerun after these final
acceptance-test corrections. The result is therefore a corrected narrow gate,
not Stage 4 closure.

## Recommended Next Investigation

The next session should first rerun the full `test_v4_stage4_trace` target with
the corrected token-aware and FP16-aware acceptance checks. Then:

1. Review and remove or formalize the temporary command-line isolation modes
	and HC diagnostic prints in `test_v4_stage4_trace.cpp`.
2. Review the experimental all-stream/per-step synchronization and synchronous
	trace-copy changes in `v4_pipeline.hpp`; the earlier race hypothesis was not
	the cause of the reported `88.590942` failure and these changes were not
	independently justified by the corrected acceptance result.
3. Only after the full corrected trace passes, use the existing tensor-level
	trace to investigate final logits and coherent generation.

## Acceptance-Criteria Audit

The concern about accumulating technical debt through circular tests is
partially confirmed. The current suite contains useful independent component
checks, but it also contains several green tests that prove only internal
consistency or structural progress. Those evidence classes must not be
reported as model correctness.

### What the green tests actually establish

- `test_v4_real_dense_parity` compares HIP dense kernels with a scalar host
	decode over the selected artifact. This is useful kernel and dtype evidence,
	but it is not a checkpoint-level reference or a full-model output check.
- `test_v4_real_expert_parity` compares the native expert kernels with a scalar
	INT4 decode of the same payload. It validates the packed representation and
	kernels, not routing semantics or model behavior.
- `test_v4_attention_oracle` and `test_v4_real_attention_oracle` validate the
	host oracle's state transitions and chunk/serialization equivalence. They do
	not execute HIP production attention and cannot detect a shared semantic
	assumption in both the oracle and production path.
- `test_v4_prefill_state` compares one-shot, serialized, chunked, and hybrid
	Aeon execution. This is a valuable metamorphic test, but all compared paths
	can preserve the same wrong full-model behavior.
- `test_v4_stage4_dispatch` checks token range, layer classification, cache
	counts, and boundary positions. It is a structural integration smoke, not a
	correctness gate.
- `bench_full_model` and the text-generation unit test measure execution or
	wrapper control flow. Neither supplies a trusted model output.

The full-block part of `test_v4_stage4_trace` is stronger than those tests,
but it still has gaps: later-layer expectations begin with production-captured
inputs, routed expert IDs and weights are consumed from the production trace,
and some downstream HC checks reuse `cpu_sinkhorn_and_mix` or `cpu_hc_post`.
There is no explicit assertion that layer N+1 receives layer N's captured
post-FFN residual. A chain can therefore be locally self-consistent without
proving the inter-layer contract.

The previous `88.590942` layer-0 HC mix check must not be treated as evidence
of cumulative state or device corruption. Its expected path used the wrong
embedding row for the real prompt. The corrected check now derives the row from
the recorded token ID. The broader audit conclusion still stands: green
component, metamorphic, and structural tests are not equivalent to trusted
full-model correctness.

### Required evidence tiers going forward

1. Component parity: HIP kernels versus scalar implementations using explicit
	 numeric tolerances.
2. State-machine parity: production traces versus an independently implemented
	 state oracle, including reset and repeated-run isolation.
3. Integrated boundary parity: explicit continuity between adjacent layer
	 trace records, independent router selection/weight reconstruction, and
	 final residual checks.
4. Trusted-reference parity: identical artifact, token IDs, positions, and
	 greedy settings compared at selected intermediate tensors and final logits.
5. Behavioral evidence: exact formatted prompt, generated IDs, stop reason,
	 and a task-level known-answer or compatible-reference comparison.

A green result at one tier must not be described as closing a higher tier.
Until the full corrected trace and trusted-reference comparison are resolved,
the engine has no evidence-backed coherent-generation acceptance.

## Session Addendum: 2026-09-14

This session directly tested the concern that passing tests may encode the
wrong acceptance criteria. The long trace failure exposed exactly that failure
mode: the test itself supplied the wrong token to its independent layer-0
embedding calculation. The follow-up HC decomposition then exposed a second
criteria problem: a fixed absolute FP16 threshold rejected one half-precision
quantization step at large activation magnitude even when the scalar mix and
pre-mix values matched.

The current evidence is consequently classified as follows:

- **Corrected test oracle:** trace records now carry input token IDs.
- **Corrected numeric criterion:** integrated FP16 checks use a declared
	absolute-plus-relative tolerance; scalar FP32 projection checks remain
	strict.
- **Passing narrow gate:** all 43 layers on the 19-token real prompt through
	position 18 passed in `real-only` mode.
- **Still open:** full corrected Stage 4 rerun, trusted-reference logits and
	intermediate parity, and coherent known-answer generation.

Temporary diagnostic/probe code remains in the worktree because the session
was stopped immediately after this bounded result, as requested. No commit or
branch was created.

No commit or branch was created.
