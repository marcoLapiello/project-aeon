# llama.cpp DeepSeek-V4 Batch Prefill Analysis

Status: current analysis, 2026-09-14

Reference checkout: `/home/marcolap/aeon-references/llama.cpp`

Reference revision: `9cf3bf256b5a50a971a636c36dfe974387140687`

Scope: read-only comparison of the local llama.cpp C++ DeepSeek-V4 batch
prefill path with Aeon's Stage 5 `prefill_batched()` implementation. No source
files in either checkout were changed as part of this analysis.

## Executive conclusion

llama.cpp already contains the missing architectural pattern for true
multi-token DeepSeek-V4 prefill. It does not call a single-token decode method
for every prompt token. Instead, it evaluates a token batch through one graph
and supplies that graph with an explicit per-batch state plan.

The important reusable idea is not a particular ggml operator. It is the
separation between:

1. logical batch metadata and microbatch scheduling;
2. a batch-level plan for local/compressed state reads and writes; and
3. a graph whose tensors retain the token dimension and apply causal masks per
   query row.

Aeon already batches many dense projections, HC operations, shared-expert
operations, and router logits. However, its critical stateful path remains
serialized. `prefill_batched_chunk()` loops over tokens for local-cache writes,
compressor mutation, compressed-entry materialization, indexer scoring and
selection, attention, expert supply, routed W2, and cleanup. The current
`Batched` result therefore means that the batch entry point was selected, not
that the complete transformer layer executes as a true batch.

The clearest Stage 5 gap is an Aeon equivalent of llama.cpp's DSV4
`comp_plan`. Aeon needs a batch state plan that computes causal visibility and
all persistent-state dependencies before launching the batch kernels. The
existing padded `M=16` scratch allocation is useful storage, but it is not by
itself evidence of true batched prefill.

## Reference surfaces

The comparison used these llama.cpp files:

- [Public batch contract](../../../../aeon-references/llama.cpp/include/llama.h): `llama_batch`, per-token `pos`, `seq_id`, `logits`, and `llama_decode()`.
- [Batch allocation and splitting](../../../../aeon-references/llama.cpp/src/llama-batch.cpp): `llama_batch_allocr::split_simple()` and related logical-batch to `llama_ubatch` conversion.
- [Decode and graph execution](../../../../aeon-references/llama.cpp/src/llama-context.cpp): `llama_context::decode()` and `process_ubatch()`.
- [DSV4 cache interface](../../../../aeon-references/llama.cpp/src/llama-kv-cache-dsv4.h): `llama_dsv4_comp_state`, `llama_kv_cache_dsv4`, and `comp_plan`.
- [DSV4 cache implementation](../../../../aeon-references/llama.cpp/src/llama-kv-cache-dsv4.cpp): `dsv4_build_comp_plan()`, `llama_kv_cache_dsv4::init_batch()`, and persistent-state planning.
- [Graph input binding](../../../../aeon-references/llama.cpp/src/llama-graph.cpp): `dsv4_set_comp_inputs()`, `dsv4_set_kq_mask()`, and DSV4 graph input setup.
- [DeepSeek-V4 graph](../../../../aeon-references/llama.cpp/src/models/deepseek4.cpp): batched layer construction, compressor updates, `build_lid_top_k()`, `build_csa_lid_attention()`, and `build_hca_attention()`.
- [Batched MoE graph builder](../../../../aeon-references/llama.cpp/src/llama-graph.cpp): `build_moe_ffn()` and `build_lora_mm_id()` expert selection/aggregation.
- [Batched public example](../../../../aeon-references/llama.cpp/examples/batched/batched.cpp): prompt submission as one batch and later parallel sequence batches.
- [DSV4 rollback validation](../../../../aeon-references/llama.cpp/tests/test-recurrent-state-rollback.cpp): state restoration and replay across a ratio-4 compressor boundary.
- [DSV4 backend operation tests](../../../../aeon-references/llama.cpp/tests/test-backend-ops.cpp): batched HC and top-k-MoE operation coverage.

The corresponding Aeon surfaces are:

- [Stage 5 execution plan](../../execution/active/MODEL_CORRECTNESS_EXECUTION_PLAN.md): serialized/chunk-equivalence gate and the open true-prefill task.
- [V4 pipeline](../../../src/architecture/deepseek_v4/core/v4_pipeline.hpp): `step()`, `prefill()`, `prefill_batched()`, `prefill_batched_chunk()`, and `select_indexer_topk()`.
- [Batch scratch buffers](../../../src/architecture/deepseek_v4/core/v4_pipeline_scratch.hpp): `PipelineBatchScratchBuffers` and token-major temporary storage.
- [V4 attention kernels](../../../src/architecture/deepseek_v4/kernels/v4_attention.hpp): single-row compressor save, compressed-entry materialization, indexer scoring, and local-plus-compressed attention kernels.
- [Stage 5 state test](../../../tests/test_v4_prefill_state.cpp): current batch entry-point assertion, serialized baseline, chunk-boundary comparisons, and reset checks.
- [Stage 4 trace test](../../../tests/test_v4_stage4_trace.cpp): production-to-oracle attention tracing used as the semantic reference boundary.
- [V4 layer state](../../../src/architecture/deepseek_v4/core/v4_layer.hpp): persistent local, compressor, compressed, and indexer state ownership.
- [V4 attention trace](../../../src/architecture/deepseek_v4/core/v4_attention_trace.hpp): state and intermediate tensors currently captured for serial trace validation.

## What llama.cpp does

### 1. It separates logical batches from executable microbatches

The public batch carries one position, sequence-ID set, and output flag per
token. `llama_context::decode()` initializes the logical batch, prepares memory,
splits it into `llama_ubatch` objects when needed, and sends each ubatch through
`process_ubatch()`.

`process_ubatch()` builds or reuses a graph whose shape includes
`ubatch.n_tokens`. It binds the ubatch inputs and calls graph execution with a
batched flag when more than one token is present. Splitting is a scheduling
boundary, not a semantic boundary: the state plan makes each ubatch equivalent
to processing its positions in order.

This distinction is useful for Aeon. `requested_batch_size` should determine
the maximum executable batch, while the state plan should determine how a chunk
is evaluated. A chunk may be split for resource limits without falling back to
the single-token `step()` path.

### 2. It plans compressed state before graph execution

`dsv4_build_comp_plan()` produces one plan for each compressed state family.
The plan includes:

- `state_pos`: APE row IDs, `position % ratio`;
- `n_visible`: completed compressed rows visible to each query token;
- `state_read_idxs`: source rows for completed compression windows;
- `state_write_idxs` and `state_write_pos`: compressed-cache destinations and
  their RoPE positions;
- `state_persist_src_idxs` and `state_persist_dst_idxs`: the latest current
  batch rows that must survive into persistent compressor state;
- restore/snapshot indices for recurrent-state rollback; and
- a fixed-width graph view with masked padding rows.

For a contiguous single sequence, current-batch rows are addressable as graph
scratch rows. Older rows resolve to persistent state slots. This is the key
mechanism that lets a later token in the same batch consume a prior token's
compressor result without requiring a host-side token loop.

The planner handles both compression contracts:

- C4A/CSA uses two contiguous groups of overlapping previous/current source
  rows for each completed block.
- C128A/HCA uses one non-overlapping group of source rows.

The graph receives the plan arrays through `dsv4_set_comp_inputs()`. Persistent
state is restored into a graph-visible view, current-token state is computed in
the graph, completed rows are written to compressed caches, and only the latest
state rows are copied back to the persistent state object.

### 3. It expresses causal visibility as masks, not control flow

The compressed attention input contains a per-query visibility count. The
DSV4 mask writer fills each query row with zero for visible compressed entries
and negative infinity for entries that are not yet complete or are beyond the
query's causal boundary.

The DeepSeek-V4 graph then:

1. computes Q and KV for all token columns;
2. computes compressor rows for all token columns;
3. materializes completed compressed rows using the plan indices;
4. computes indexer queries, weights, and scores for all CSA query rows;
5. selects top-k per query row;
6. combines raw local keys with compressed keys; and
7. applies the per-query raw/compressed masks in attention.

`build_csa_lid_attention()` constructs the CSA top-k mask and concatenates it
with the raw local mask. `build_hca_attention()` does the same for HCA without
the Lightning Indexer branch. The graph does not need to call a serial
attention function once per query token.

### 4. It batches expert routing and expert outputs

`build_moe_ffn()` treats the activation as `[hidden, tokens]`. Router logits,
selection probabilities, selected expert IDs, and expert weights all retain the
token dimension. `build_lora_mm_id()` evaluates the selected expert rows and
returns expert outputs that are then weighted and summed per token.

Aeon's storage problem is different because experts arrive through the
Hot/Warm/Cold coordinator, but the dataflow is still applicable: determine the
expert requests for the whole chunk, acquire the union of required experts,
group or pack token rows by expert, execute the grouped W13/W2 work, and scatter
weighted outputs back to the token rows.

## Aeon implementation comparison

### Already useful

Aeon has a real token-major batch substrate. `PipelineBatchScratchBuffers`
allocates rows for residuals, HC matrices, Q/KV projections, compressor
projections, indexer projections, attention output, FFN activations, shared
expert output, and router output. Several launches in
`prefill_batched_chunk()` already use `batch_count` as a grid dimension.

That work can remain the front half of a true batch path. It should not be
discarded or replaced with an external runtime.

### Clear gaps

1. **No batch state plan.** Aeon has persistent per-layer state, but no object
   computes per-token compressed visibility, current-batch source indices,
   compressed-cache write indices, or persistent-state copy dependencies before
   execution.

2. **State mutation is serial.** In
   `prefill_batched_chunk()`, the loop beginning at the per-token attention
   section writes local cache rows, saves compressor state, materializes
   compressed entries, and updates indexer state one token at a time. This is
   semantically ordered work, but the ordering is currently implemented as host
   control flow rather than as planned device dependencies.

3. **The compressor kernels are single-row kernels.**
   `v4_save_compressor_state_kernel()` accepts one KV row, one score row, one
   absolute position, and one persistent destination. It cannot represent the
   current batch as graph-local rows or gather prior rows from the same batch.

4. **Compressed attention is single-query.**
   `v4_cached_compressed_attention_wave32_kernel()` accepts one query, one
   current position, one compressed count, and one top-k array. It has no
   per-query visibility mask or `[batch, topk]` index input.

5. **Indexer top-k is explicitly serialized.**
   `select_indexer_topk()` copies one score vector to the host, synchronizes the
   compute stream, performs one stable sort, copies one selected-index vector
   back, and synchronizes again. It is called inside the token loop.

6. **Expert supply is serialized per token.** Router outputs are batch-shaped,
   but the subsequent loop performs expert prefetch, lease acquisition, fused
   W13/W2 execution, stream synchronization, staging release, lease release,
   and telemetry for each token independently.

7. **The public execution label is too strong.** The `Batched` result is
   returned when `prefill_batched_chunk()` is selected for a multi-token input.
   It does not currently assert that attention, state planning, indexer
   selection, or expert execution avoided token serialization.

8. **No per-token position/mask contract reaches the attention kernel.** Aeon
   passes a scalar position to each invocation. This makes the current path
   correct only because it processes one query at a time; it cannot express
   llama.cpp's `n_visible` or causal compressed-cache masks for a token batch.

## What to reuse and what not to copy

### Reuse the architecture

Aeon should adapt these llama.cpp ideas:

- a logical prompt span split into bounded executable chunks;
- explicit absolute positions for every row;
- one batch plan per layer and state family;
- graph-local current-batch rows plus persistent state rows;
- gather/index arrays for C4 overlap and C128 reduction;
- per-query raw/compressed visibility masks;
- per-query CSA top-k indices;
- batch-wide expert request collection and grouped execution; and
- state snapshot comparisons after each chunk.

### Keep Aeon-specific choices

Aeon should not copy llama.cpp's tensor or weight assumptions wholesale:

- Aeon's version-2 swizzled symmetric INT4 expert artifact remains the weight
  contract.
- The existing Hot/Warm/Cold supply coordinator remains the residency owner.
- The first implementation can use conservative HIP kernels and a host-side
  batched top-k while semantics are being proven.
- The current serialized `step()` path should remain available as a fallback
  and oracle comparison path.
- Llama.cpp's multi-sequence and rollback machinery is not required for the
  first contiguous single-sequence Stage 5 milestone, but the plan should not
  prevent adding it later.

## Recommended Aeon design

Start with one contiguous sequence and one executable chunk. Add an
architecture-owned plan, for example:

```text
V4BatchStatePlan
  positions[B]
  state_pos[B]
  n_visible[B]
  local_cache_slots[B]
  state_source_indices
  state_read_indices
  state_write_indices
  state_write_positions
  persistent_copy_sources
  persistent_copy_destinations
  indexer_candidate_counts[B]
  indexer_topk_indices[B, 512]
```

The first semantic implementation can retain host-generated arrays and
conservative device kernels. The important change is that all token-dependent
state relationships are represented in the plan rather than hidden in a
`for (token ...)` loop.

Suggested sequence:

1. Build and validate the plan for positions before launching a layer. Cover
   positions around 4 and 128 boundaries, including incomplete windows.
2. Batch local KV writes and retain absolute positions for every row.
3. Add graph-local compressor state. Gather current-batch and persistent rows
   according to the plan, then commit only the required final state rows.
4. Add per-query compressed visibility and a batch attention kernel. A host
   stable top-k once per layer is acceptable temporarily if it processes all
   query rows in one transfer.
5. Add batched indexer scoring and `[batch, topk]` selected indices.
6. Batch expert supply by the union of expert IDs in the chunk, then group
   token rows by expert for W13/W2 and scatter weighted results.
7. Keep a serialized fallback for unsupported batch shapes and compare every
   resulting state snapshot and next-token output against it.

The first implementation does not need to optimize all operations at once. It
does need to remove the semantic dependence on observing one token's final
layer output before starting the next token in the same chunk.

## Validation implications for Stage 5

The existing [Stage 5 state test](../../../tests/test_v4_prefill_state.cpp)
should distinguish these cases explicitly:

- serialized `prefill()` baseline;
- true batch size 2 or 3, crossing neither boundary;
- batch sizes 4 and 8, crossing a C4 boundary;
- a batch crossing the first HCA boundary near position 127/128;
- an unaligned chunk split such as 3, 7, or 127;
- a batch whose CSA candidate count is below 512; and
- a batch whose CSA candidate count exceeds 512.

For each case compare:

- absolute-position metadata;
- local ring contents and valid positions;
- compressor partial rows and positions;
- compressed entry counts, positions, and values;
- indexer candidate counts and every per-query top-k row;
- final layer state snapshots; and
- next-token logits or greedy token IDs.

The llama.cpp rollback test is a useful additional pattern: take a state
snapshot, replay a suffix that crosses a compression boundary, and compare the
replay against a separately initialized context. This can expose state-copy
bugs that a final-logit-only comparison hides.

## Evidence boundary

This document records source inspection, not a claim that llama.cpp and Aeon
are numerically identical. Llama.cpp's DSV4 graph is a strong implementation
reference for batch planning, causal visibility, and state persistence, but
Aeon's INT4 artifact, HIP kernels, memory ownership, and deterministic
accumulation remain separate contracts.

The current Stage 5 serialized/chunk-equivalence gate remains valid evidence
for the fallback path. A `Batched` result should be promoted to evidence for a
true batch path only after the state plan, per-query masks, batched compressed
attention, indexer rows, and batch-wide expert execution are traceable and
validated against the serialized path.