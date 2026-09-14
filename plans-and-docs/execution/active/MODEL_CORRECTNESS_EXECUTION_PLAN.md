# DeepSeek-V4 Flash Model Correctness Execution Plan

**Date:** 2026-09-14
**Status:** Open; Stages 0-6 serial correctness complete; Stage 5 serialized/chunk equivalence and hybrid batched-prefill subgates complete; Stage 7 trusted-reference parity is active but its independent reference-runtime lane is pending; fully batched stateful optimization remains separate
**Target:** `DeepSeek-V4-Flash-0731-INT4-W4A16` on the native `.aeon` artifact and AMD RDNA3/gfx1100  
**Scope:** Restore mathematically faithful base-decoder execution, then prove it against an independent reference before resuming placement or performance work.

## 1. Executive decision

The current runtime is a stable DeepSeek-V4-shaped prototype, not yet a faithful
implementation of the selected checkpoint. The decisive gap is not a missing
benchmark or a final API comparison: the production path executes a 128-token
sliding-window approximation for every layer, while the checkpoint declares a
layer-specific sliding/compressed/indexed attention schedule.

This plan is the canonical implementation sequence for closing that gap. It
supersedes the correctness portions of the historical Phase 1 checklist while
preserving the useful storage, expert-format, mHC, routing, and native text
work already completed.

The plan has four governing rules:

1. Recover the checkpoint contract before adding kernels or performance work.
2. Implement a deterministic CPU reference/oracle for cache and attention state
   before relying on HIP output.
3. Make the runtime reject unsupported architecture contracts rather than
   silently running all layers as sliding-window attention.
4. Do not use routing profiles, placement results, or end-to-end throughput as
   model evidence until the final correctness gates in this document pass.

The first completion target is the **base causal decoder**: exact next-token
logits and greedy generation for the 43 decoder layers. MTP, DSpark auxiliary
heads, prefix caching, and serving optimizations are separate follow-up work;
the runtime must not claim those features merely because their tensors exist.

## 2. Definition of done

The base-decoder correctness gate is closed only when all of the following are
true:

- The selected model configuration is parsed into an explicit, validated V4
  execution specification.
- Layers 0-42 are classified from `compress_ratios`, with no implicit fallback:
  - layers 0-1: sliding-window;
  - layers 2, 4, ..., 42: ratio-4 CSA with Lightning Indexer;
  - layers 3, 5, ..., 41: ratio-128 HCA;
  - entries 43-45: recognized as auxiliary schedule entries and not used as
    base decoder layers.
- Every required compressor, indexer, attention, normalization, HC, router,
  shared-expert, and output tensor is present with the expected dtype and
  shape before device execution begins.
- A CPU float32 oracle reproduces the reference layer semantics for sliding,
  C4A/CSA, and C128A/HCA attention, including cache state and causal boundary
  behavior.
- One-shot prefill, arbitrarily chunked prefill, and serialized token append
  produce equivalent cache state and next-token outputs.
- HIP implementations match the CPU oracle at the declared FP16/FP32
  tolerances for layer 0, layer 2, and layer 3 representative traces.
- The independent INT4 decoder and the complete routed expert path have passed
  a real-tensor parity check; local GPU-versus-local-mirror tests alone do not
  count.
- A full 43-layer run on identical formatted token IDs agrees with a trusted
  compatible reference at the agreed intermediate and output checkpoints.
- The native text path records the exact prompt IDs, generated IDs, logits/top-k
  evidence where available, and stop reason. Existing generated text is not
  treated as proof without this evidence.
- The routing profiler remains locked until the correctness evidence is stored
  and the routing plan is updated to reference it.

Performance is explicitly not part of this gate. The first correct path may be
serialized, conservative, and slower than the current approximation.

## 3. Frozen checkpoint contract

The authoritative package is:

```text
models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/
```

The selected configuration declares:

| Contract | Value |
| --- | ---: |
| Base decoder layers | 43 |
| Hidden size | 4096 |
| Vocabulary | 129280 |
| Query heads | 64 |
| KV heads | 1 |
| Head dimension | 512 |
| NoPE dimension | 448 |
| RoPE dimension | 64 |
| Query low-rank dimension | 1024 |
| Output low-rank dimension | 1024 |
| Output groups | 8 |
| Sliding window | 128 |
| Routed experts | 256 |
| Shared experts | 1 |
| Routed experts per token | 6 |
| Hash-routed layers | 0-2 |
| HC streams | 4 |
| HC Sinkhorn iterations | 20 |
| RMS/HC epsilon | `1e-6` |
| SwiGLU limit | 10.0 |
| Routed scaling factor | 1.5 |
| Indexer heads | 64 |
| Indexer head dimension | 128 |
| Indexer top-k | 512 |
| Maximum position embeddings | 1048576 |
| Original position limit | 65536 |
| Main RoPE theta | 10000 |
| Compressed RoPE theta | 160000 |
| YaRN factor | 16 |
| YaRN beta fast/slow | 32 / 1 |
| MTP layers declared | 1 |

### 3.1 Layer schedule

The selected `compress_ratios` array has 46 entries. Only the first 43 entries
belong to the base decoder:

```text
layer 0-1:       0, 0
layer 2-42:      4, 128, 4, 128, ..., 4, 128, 4
entries 43-45:  0, 0, 0   (auxiliary schedule entries)
```

The exact array must be parsed and validated from the package rather than
reconstructed from this summary. A mismatch in schedule length, layer value,
or layer-class mapping is a startup error.

### 3.2 Shared attention preparation

The base attention path has the following logical dataflow. The existing
tensor names and layouts may be retained where they are proven equivalent, but
the implementation must be compared against the reference path:

```text
four HC residual streams
    -> attention HC pre-mix and Sinkhorn
    -> attention RMSNorm
    -> q low-rank projection, q RMSNorm, q expansion
    -> per-head unit RMSNorm on Q
    -> shared 512-wide KV projection and KV RMSNorm
    -> layer-specific cache insertion and attention
    -> inverse RoPE on attention output
    -> grouped W_o_a, then W_o_b
    -> HC attention post-expansion
    -> FFN HC pre-mix and RMSNorm
    -> shared expert plus six routed experts
    -> HC FFN post-expansion
```

The reference implementation fuses some projections, while Aeon currently
binds tensors such as `attn.wq_a`, `attn.wq_b`, and `attn.wkv` separately. This
is acceptable only after intermediate tensors are shown equivalent.

### 3.3 Sliding-window attention

Sliding layers use a causal local window of at most 128 positions, including
the current position. The local cache must retain absolute positions even if
the physical storage is a ring. The attention sink, query/key RoPE treatment,
and output inverse-RoPE must follow the selected reference implementation.

The current `d_kv_cache: [max_seq_len, 512]` allocation is not an adequate
model-wide cache contract. It can remain as an initial implementation detail
for a sliding layer only if its reset, position, and window semantics are
explicitly tested.

### 3.4 Ratio-4 CSA and Lightning Indexer

Every even layer from 2 through 42 is a ratio-4 compressed sparse attention
layer. It has both local sliding state and long-range compressed/indexed state.

The reference flow is:

```text
hidden state
    -> ratio-4 compressor KV and score projections
    -> persistent compressor partial state plus APE[position % 4]
    -> overlapping compressed entry at a completed boundary
    -> compressed KV normalization and compressed RoPE

hidden/query state
    -> indexer query projection [64, 128]
    -> indexer weights projection [64]
    -> indexer score against compressed indexer keys
    -> causal top-512 compressed-entry selection
    -> sparse attention over local plus selected long-range entries
```

The local reference compressor uses `coff = 2` for ratio 4. At a completed
position it gathers the two overlapping four-token regions, so the compressor
state spans eight token states. It adds the learned APE row selected by
`position % 4` to the score state before the boundary reduction. The exact
causal boundary and warm-up behavior must come from the reference code and the
oracle tests, not from a simplified non-overlapping four-token average.

The indexer is a separate scoring path. It is not ordinary full attention
followed by truncation. Top-k ordering and tie behavior must be deterministic;
when the valid candidate count is below 512, every valid candidate is selected
and the remaining output slots are marked invalid.

### 3.5 Ratio-128 HCA

Every odd layer from 3 through 41 is a ratio-128 heavily compressed attention
layer. It has local sliding state and a non-overlapping compressed state:

```text
128 token states -> one completed compressed entry
```

An incomplete 128-token region must not become visible as a completed causal
entry unless the reference implementation explicitly says so. There is no
CSA Lightning Indexer on this branch. The compressor state must survive chunk
boundaries and decode steps.

### 3.6 RoPE and compressed RoPE

RoPE is interleaved/GPT-J style and applies to the trailing 64 channels of the
512-wide head. The layer class controls the table:

- sliding layers use main `rope_theta = 10000` and no long-context YaRN
  scaling (`factor = 1` for this branch);
- ratio-4 and ratio-128 compressed layers use
  `compress_rope_theta = 160000` and the configured YaRN parameters;
- compressed entries use the reference compressed position convention;
- inverse RoPE on attention output uses the same layer-class position table.

The current single table initialized with `factor = 1.0` is not sufficient.
Separate main/compressed tables or an equivalent class-aware table provider are
required.

### 3.7 HC, MoE, quantization, and output

The current HC and routing paths have useful silicon tests, but the tests must
be connected to real checkpoint-layer parity before they count toward the full
gate:

- mHC operates on four streams with 20 Sinkhorn iterations;
- each MoE layer combines one shared expert and six routed experts;
- layers 0-2 use checkpoint-provided `tid2eid` hash routing;
- later layers use the configured sqrt-softplus score and bias correction;
- routed experts use the selected artifact's symmetric group-32 INT4 contract:
  low-nibble-first packed words and `(nibble - 8) * FP16 scale`;
- W1/W3 are `[2048, 4096]`, W2 is `[4096, 2048]`, and expert execution is
  `activation @ weight.T`;
- the current expert artifact remains version-2 swizzled and must not be
  replaced or guessed as FP4 merely because the config labels it `fp4`;
- the final base path is HC head reduction, final RMSNorm, untied `head.weight`
  projection, and greedy argmax over 129280 logits.

The checkpoint also contains `mtp.*` and DSpark-related metadata. Those are
not part of the first base-decoder gate. They must either be implemented in a
later milestone or explicitly reported as unsupported rather than silently
implying full release feature coverage.

## 4. Current implementation and evidence boundary

| Area | Current source | Confirmed state | Consequence |
| --- | --- | --- | --- |
| Configuration | `src/architecture/deepseek_v4/core/config.hpp` | Parses scalar defaults but not the compression schedule, indexer fields, compressed theta, nested YaRN data, or auxiliary metadata | The runtime cannot derive the real layer classes from the checkpoint. |
| Layer state | `src/architecture/deepseek_v4/core/v4_layer.hpp` | Owns one full-resolution `[max_seq_len, 512]` cache per layer | No compressor, compressed pool, indexer cache, boundary state, or ring metadata exists. |
| Pipeline dispatch | `src/architecture/deepseek_v4/core/v4_pipeline.hpp` | Calls `v4_cached_sliding_window_attn_wave32_kernel` for every layer | Layers 2-41 execute the wrong attention mechanism. |
| Dense binding | `src/architecture/deepseek_v4/core/v4_dense_weight_binding.hpp` | Binds current attention/HC/router/shared-expert tensors only | Compressor and indexer tensors are present in the artifact but unused. Missing tensors fail open by becoming null pointers. |
| RoPE | `src/architecture/deepseek_v4/core/v4_model_resources.hpp` | One theta-10000 table with factor 1.0 | Compressed-layer RoPE and configured YaRN are not executed. |
| Prefill | `V4Pipeline::prefill()` and `prefill_batched()` in `v4_pipeline.hpp` | Serial prefill is the reference path; hybrid batch prefill batches dense/HC/FFN/router work but advances cache, compressor, indexer, attention, and routed experts per token | The hybrid path is state-equivalent to serial execution for the validated 132-token case. A batch state plan, per-query visibility masks, batched stateful attention, and batch-wide routed-expert execution remain open. |
| Attention tests | `tests/test_v4_attention.cpp` | Synthetic SWA, RoPE round trip, grouped projection, and local CPU mirror | Does not test C4A, C128A, real checkpoint state, or layer schedule. |
| HC/router tests | `tests/test_hc_sinkhorn.cpp`, `tests/test_moe_router.cpp` | GPU versus hand-written CPU mirrors | Does not prove checkpoint/reference graph parity. |
| Expert tests | `tests/test_w4a16_swizzled_gemv.cpp` and fused expert tests | GPU versus local INT4 decoder | Does not by itself prove source checkpoint semantics or full-layer parity. |
| Final output | `v4_pipeline.hpp` and `v4_attention.hpp` | Base HC head, final norm, untied LM head, and argmax are wired | MTP is absent; output is only meaningful after preceding layers are correct. |

The native text result and current routing profiles remain plumbing evidence.
They must not be presented as proof of selected-checkpoint correctness.

## 5. Reference sources and evidence discipline

### 5.1 In-repository references

- [DeepSeek-V4 Flash versus Aeon comparison](../../analysis/current/DEEPSEEK_V4_FLASH_AEON_COMPARISON.md)
  is the current diagnosis and records the selected 0731 schedule correction.
- [DeepSeek-V4 architecture notes](../../analysis/current/deepseek_v4_flash_architecture.md)
  records the model geometry, cache categories, and causal requirements.
- [VLLM RDNA3 and DeepSeek-V4 reference analysis](../../analysis/current/VLLM_RDNA3_DEEPSEEK_V4_REFERENCE_ANALYSIS.md)
  maps the external reference attention, compressor, indexer, prefill, and
  hardware paths to Aeon ownership boundaries.
- [Native text-in/text-out plan](TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md)
  owns formatter/tokenizer parity and the external behavioral gate.
- [Routing profile plan](ROUTING_PROFILE_AND_PLACEMENT_STUDY.md) owns the
  profiler contract and must remain correctness-gated.
- [Codebase map](../../status/CODEBASE_MAP.md) identifies production files,
  validation targets, and non-production diagnostics.
- [Performance ledger](../../status/PERFORMANCE_LEDGER.md) is the only place
  for authoritative silicon measurements.

Primary local implementation surfaces:

- `src/architecture/deepseek_v4/core/config.hpp`
- `src/architecture/deepseek_v4/core/v4_layer.hpp`
- `src/architecture/deepseek_v4/core/v4_dense_weight_binding.hpp`
- `src/architecture/deepseek_v4/core/v4_model_resources.hpp`
- `src/architecture/deepseek_v4/core/v4_pipeline.hpp`
- `src/architecture/deepseek_v4/core/v4_pipeline_scratch.hpp`
- `src/architecture/deepseek_v4/kernels/v4_attention.hpp`
- `src/architecture/deepseek_v4/kernels/hc_sinkhorn.hpp`
- `src/architecture/deepseek_v4/kernels/moe_router.hpp`
- `src/architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp`
- `src/infrastructure/core/aeon_loader.hpp`
- `scripts/convert_safetensors_to_aeon.py`

### 5.2 External reference checkouts

The following are source references only. They must not become Aeon runtime
dependencies. The inspected vLLM checkout was at revision `94848ed` during the
reference review; record a new revision if it changes before implementation.

```text
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/attention.py
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/compressor.py
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/common/rope.py
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/common/ops/save_partial_states.py
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
/home/marcolap/aeon-references/vllm/vllm/v1/attention/backends/mla/indexer.py
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/amd/model.py
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/amd/rocm.py
```

The vLLM V4 graph is primarily an FP8/MXFP4-oriented model path. Reuse its
attention, cache, compressor, indexer, and boundary semantics; do not reuse its
weight-format assumptions as a substitute for Aeon's validated symmetric INT4
artifact.

The selected model source snapshot and formatter are under:

```text
models/DeepSeek-V4-Flash-0731-INT4-W4A16/
  models--yiminyuan--DeepSeek-V4-Flash-0731-INT4-W4A16/
    snapshots/64700592cadaf205fe0c13202061ff4b45afbfd0/
      encoding/encoding_dsv4.py
      encoding/test_encoding_dsv4.py
      encoding/README.md
      README.md
```

Use the source snapshot, checked-in config, tensor metadata, and vLLM source
as a triangulated reference. If they disagree, stop and record the discrepancy
before writing a kernel.

## 6. Step-by-step execution sequence

### Stage 0: Freeze evidence and create the contract boundary

**Objective:** Make the selected checkpoint and current approximation explicit
before changing execution.

Tasks:

1. Record the Aeon commit, artifact manifest identity, model config hash,
   source snapshot revision, tokenizer hash, and external reference revision in
   the correctness test metadata.
2. Enumerate every dense tensor in `model_dense.aeon` with name, dtype, byte
   size, and shape. Produce a machine-readable inventory for the selected
   checkpoint; do not rely only on a hand-maintained list.
3. Identify the exact checkpoint names and shapes for:
   - ratio-4 compressor KV/score projections, APE, and normalization;
   - ratio-128 compressor KV/score projections, APE, and normalization;
   - ratio-4 indexer query projection, indexer weight projection, and indexer
     compressor tensors;
   - `mtp.*` and DSpark tensors;
   - all current attention, HC, router, shared-expert, norm, embedding, and LM
     head tensors.
4. Extend `DeepSeekV4Config` with the fields needed to describe the execution
   contract, including `compress_ratios`, `index_head_dim`, `index_n_heads`,
   `index_topk`, `compress_rope_theta`, `o_groups`, MTP metadata, and the
   relevant YaRN fields. Parse nested objects and arrays structurally; do not
   add more fragile scalar substring searches for architecture-defining data.
5. Add an explicit layer descriptor, for example `Sliding`, `CSA`, or `HCA`,
   derived from the validated schedule. Keep this descriptor in the V4
   architecture layer, not in the architecture-neutral expert supply code.
6. Add startup validation for schedule length, supported ratios, model
   dimensions, indexer dimensions, RoPE settings, tensor presence, tensor
   dtype, and tensor shape.

Acceptance criteria:

- A config test reads the checked-in package and asserts the complete schedule,
  not only scalar dimensions.
- Mutating `compress_ratios`, `index_topk`, `compress_rope_theta`, or a required
  tensor entry causes a clear startup/test failure.
- A valid package never reaches `V4Pipeline::step()` with a required compressor
  or indexer pointer null.
- The runtime prints or exposes the resolved layer class for at least layers 0,
  2, 3, 41, and 42.
- No change is made to Hot/Warm/Cold ownership or expert artifact bytes.

Suggested validation surfaces:

```text
tests/test_config_parser.cpp
tests/test_v4_model_contract.cpp       (new)
src/architecture/deepseek_v4/core/config.hpp
src/architecture/deepseek_v4/core/v4_model_spec.hpp  (new or equivalent)
src/architecture/deepseek_v4/core/v4_dense_weight_binding.hpp
```

### Stage 0 implementation record (2026-09-13)

Stage 0 is complete for the selected Aeon package. The runtime now parses the
configuration through a structured JSON value tree, resolves the 43 base layers
to 2 Sliding, 21 CSA, and 20 HCA descriptors, retains dense directory shapes,
and rejects missing, mismatched, or incorrectly shaped required tensors before
device layer initialization. The class-specific compressor and CSA indexer
weights are bound fail-closed from the resolved descriptor.

The reproducible evidence command is:

```text
scripts/record_v4_stage0_metadata.sh \
  models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon \
  build/correctness/stage0_model_contract.json
```

The recorded package contains 1,271 dense tensors, 43 resolved base layers, and
the following frozen identities:

| Evidence | Value |
| --- | --- |
| Aeon commit | `6b4a20ffc424a73a3627608f4dbec20625d9e0fb` |
| Source snapshot | `64700592cadaf205fe0c13202061ff4b45afbfd0` |
| vLLM reference | `94848eda600a07c28675f5753a11b2c212c146ed` |
| MTP tensor count | 72 |
| DSpark tensor count | 0 |

The MTP tensors are inventoried but remain outside the base-decoder execution
contract. No DSpark tensors are present in the selected dense artifact.

### Stage 1 implementation record (2026-09-13)

The independent weight and dense-operation gate is complete for the selected
Aeon package. The new host reference boundary in
`src/architecture/deepseek_v4/reference/v4_int4_reference.hpp` decodes the
serialized version-2 expert layout from logical row/group coordinates, including
the swizzled nibble permutation, symmetric group-32 values, and FP16 scales. It
does not call the production swizzle or GEMV helpers.

The model-backed tests are:

```text
tests/test_v4_real_expert_parity.cpp
tests/test_v4_real_dense_parity.cpp
```

The expert test samples expert IDs `0, 1, 7, 31, 127, 255` from layers 0, 21,
and 42. It compares independent CPU W1/W2/W3 vectors, clamped SwiGLU, and a
six-expert weighted routed FFN against the native swizzled Wave32 path using
fixed FP16 activations. The dense test covers layers 0, 2, and 3, including
attention projections, grouped `wo_a`, `wo_b`, RMSNorms, shared experts, the
router, both HC projection/Sinkhorn paths, the final RMSNorm, and all 129,280
LM-head rows.

The declared test tolerances are fixed rather than selected per layer:

| Comparison | Tolerance |
| --- | ---: |
| Expert/dense FP16 projection | `0.005 + 0.001 * max(1, abs(reference))` |
| Routed FFN/SwiGLU | `0.01 + 0.002 * max(1, abs(reference))` |
| HC and RMSNorm outputs | `0.002 + 0.001 * max(1, abs(reference))` |

On the Radeon RX 7900 XTX (`gfx1100`), both tests passed. Maximum observed
errors were approximately `2.24e-4` for sampled expert projections,
`5.55e-6` for weighted routed FFN output, `4.87e-4` for dense RMSNorm, and
`1.94e-3` for the full LM head. The source converter's existing
`verify_swizzled_expert_payload()` remains the source-to-artifact bit-exact
conversion check; the new test additionally exercises artifact-to-host and
artifact-to-VRAM views through the native kernels.

Evidence identities for this run:

| Evidence | Value |
| --- | --- |
| Aeon baseline commit | `ab908040289d146f809cbaee5434b10a6ad527e8` |
| Model config SHA-256 | `3911161a028fa2818b22ffff82dc2dad212aafeff32a66698ef7d154dea9d6a1` |
| Model manifest SHA-256 | `717bc27a36329524a9e716da9aecfc2d942c482e32e2b3fe23345b58b78144fe4` |
| Source snapshot | `64700592cadaf205fe0c13202061ff4b45afbfd0` |
| vLLM reference | `94848eda600a07c28675f5753a11b2c212c146ed` |

This closes the weight/dequantization and dense-operation gate only. It does
not establish layer-semantic correctness, compressed cache behavior, prefill
equivalence, or full-model parity. Stage 2 was the next workstream at the time
of this record; its completed CPU-oracle gate is recorded below.

### Stage 1: Close the independent weight and dense-operation gate

**Objective:** Ensure that attention work is not debugging a hidden weight or
quantization mismatch.

Tasks:

1. Keep the current version-2 swizzled artifact and loader unchanged as the
   storage baseline.
2. Implement or retain a separate reference decoder for real W1/W2/W3 tensors
   using the source contract:
   - packed `uint32` words;
   - eight low-nibble-first values per word;
   - symmetric value `(nibble - 8)`;
   - FP16 scale per group of 32 input values;
   - row-major `[out, in]` weights and `activation @ weight.T`.
3. Select real experts from early, middle, and late layers. Compare all three
   projections and the complete routed FFN output against the native swizzled
   path for fixed FP16 activations.
4. Compare dense projections required by the attention trace: attention norm,
   `wq_a`, q norm, `wq_b`, `wkv`, KV norm, grouped `wo_a`, `wo_b`, HC tables,
   router, shared expert, final norm, and LM head.
5. Keep reference decoding independent from the current GPU test helper. A
   test that copies the same dequantization loop into CPU code is not sufficient
   evidence by itself.

Acceptance criteria:

- Real W1/W2/W3 output vectors agree with the independent reference before they
  are mixed into a transformer state.
- Dense projection outputs agree for fixed activation fixtures within the
  declared FP16/FP32 tolerance; all threshold choices are recorded in the test
  metadata.
- The full expert payload remains bit-identical through source, `.aeon`, host
  staging, and VRAM views.
- Any quantization discrepancy blocks later attention work and is recorded as a
  separate failure, rather than being absorbed by a relaxed full-model test.

Relevant existing tests:

- `tests/test_w4a16_swizzle.cpp`
- `tests/test_w4a16_swizzled_gemv.cpp`
- `tests/test_w4a16_swizzled_dual_gemv.cpp`
- `tests/test_aeon_moe_fused_w13.cpp`
- `tests/test_aeon_moe_fused_w2.cpp`
- `tests/test_aeon_loader.cpp`
- `tests/test_aeon_swizzled_loader.cpp`

### Stage 2: Build the CPU attention and cache oracle

**Objective:** Define exact state transitions before porting them to HIP.

Implement a deterministic float32 reference module under the V4 architecture
or test-reference boundary. It must expose state, not only final output, so a
failure can identify a compressor, cache, indexer, or attention mismatch.

The oracle must own or expose:

- layer class and compression ratio;
- absolute sequence position;
- local 128-token K/V state and valid-position metadata;
- ratio-4 compressor partial state, score state, APE state, and compressed KV
  entries;
- ratio-128 compressor partial state and compressed KV entries;
- ratio-4 indexer key state, query state, candidate count, top-k indices, and
  indexer weights;
- RoPE table identity and position mapping;
- attention sink and causal masks;
- the output of each attention branch before inverse RoPE and output projection.

Implement the reference control flow from the external source:

1. Insert the current token's local KV state with its absolute position.
2. For a compressed layer, produce KV and score states from the hidden state.
3. Add the APE row selected by `position % compress_ratio` to score state.
4. Materialize a compressed entry only at the correct completed boundary.
5. For C4A, use the overlapping two-window/8-token boundary semantics.
6. For C128A, use the non-overlapping 128-token semantics.
7. Build the C4A indexer query and candidate scores independently from normal
   attention.
8. Select at most 512 causal compressed entries with deterministic ordering.
9. Execute local, C4A, or C128A attention according to the layer descriptor.
10. Return both the attention output and the updated state snapshot.

Required oracle tests:

- Positions 0, 1, 3, 4, 7, 8, 127, 128, 129, and 131.
- Prompt lengths immediately below, at, and above 4 and 128 boundaries.
- Chunk splits at aligned and unaligned positions, including `[1, 3, 4, 7,
  16, 127, 128]`.
- Short context where C4A has fewer than 512 valid candidates.
- Context with more than 512 valid C4A candidates and deterministic ties.
- Reset followed by a second generation must not observe stale state.

Acceptance criteria:

- One-shot, chunked, and serialized append produce identical float32 state
  snapshots and next-token outputs within the oracle tolerance.
- Incomplete C4/C128 windows are not visible early.
- Absolute positions remain correct after local ring-slot reuse.
- C4A top-k indices match the independent reference selection, including short
  context and tie cases.
- The oracle can dump a compact trace for layers 0, 2, and 3 without loading
  the complete model into GPU memory.

Suggested files:

```text
src/architecture/deepseek_v4/reference/v4_attention_oracle.hpp
src/architecture/deepseek_v4/reference/v4_attention_oracle.cpp
tests/test_v4_attention_oracle.cpp
tests/test_v4_prefill_state.cpp
```

If the repository keeps the first reference implementation header-only, keep it
isolated from production HIP kernels and document the boundary.

### Stage 2 implementation record (2026-09-13)

The initial CPU oracle boundary is implemented in
`src/architecture/deepseek_v4/reference/v4_attention_oracle.hpp` with focused
coverage in `tests/test_v4_attention_oracle.cpp`. It is host-only and accepts
caller-supplied float32 query, local K/V, compressor, and indexer projection
fixtures; it does not depend on the production HIP kernels or the complete
model artifact.

The oracle now exposes and serializes the absolute position, 128-slot local
K/V ring with absolute positions, C4 overlapping and C128 non-overlapping
partial compressor rows, score plus APE state, normalized compressed entries,
class-specific RoPE identities and positions, C4 indexer query/weight/key
state, candidate counts, deterministic top-k indices, attention sink, and
attention output before inverse RoPE. C4 short-context selection retains all
valid candidates in ascending index order; larger candidate sets use
descending score with ascending index tie-breaking.

The reproducible validation commands are:

```text
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R '^(test_v4_attention_oracle|test_v4_real_attention_oracle|test_v4_attention|test_v4_model_contract)$'
```

On the current build, the full repository build and all four selected tests
passed. The synthetic oracle covers positions through 131, the real 512/513
C4 candidate boundary, local ring-slot reuse, reset, aligned and unaligned
chunk splits, and serialized state continuation. The model-backed oracle loads
the selected `.aeon` dense artifact and exercises real layer 0 Sliding, layer
2 C4A, and layer 3 C128A projections through positions 0-131, including
independent compressed-entry reconstruction, class-aware RoPE divergence at
position 65536, chunk equivalence, and serialized continuation. Its CTest
timeout is 120 seconds; the current run completes in approximately 11 seconds.

This closes the Stage 2 CPU attention/cache oracle gate. Stage 3 remains next:
production layer-state ownership, strict persistent cache initialization,
class-aware device resources, and memory accounting must now be implemented
against this oracle. HIP-versus-oracle parity, true production prefill, and
trusted-reference parity remain later gates and are not implied by this Stage 2
completion.

### Stage 3: Add class-aware layer ownership and strict tensor binding

**Objective:** Make cache and weight ownership match the checkpoint schedule.

Tasks:

1. Extend `V4Layer` with an immutable `V4LayerSpec` containing layer ID,
   attention class, compression ratio, and required dimensions.
2. Replace the single implicit cache contract with explicit state ownership:
   - sliding layer: local cache and absolute-position metadata;
   - C4A layer: local cache, ratio-4 compressor state, compressed KV state,
     indexer state, valid counts, and top-k workspace;
   - C128A layer: local cache, ratio-128 compressor state, compressed KV state,
     and valid counts.
3. Add reset/clear methods that reset all state for a new generation. Do not
   clear only the old `d_kv_cache`.
4. Extend `V4DenseWeightBinding` with exact compressor/indexer bindings for
   the selected checkpoint. Store the required tensor shapes next to the
   binding contract or validate them during upload.
5. Change optional `upload_tensor()` behavior for architecture-defining tensors:
   missing required tensors must throw a named error. Optional MTP/DSpark
   tensors may remain absent only when the selected execution mode explicitly
   disables them.
6. Extend `V4ModelResources` with class-aware main and compressed RoPE tables.
   The table builder must receive `rope_theta`, `compress_rope_theta`, YaRN
   factor, beta values, original context, and the interleaved layout.
7. Extend scratch ownership for C4A/C128A state and indexer top-k metadata. Do
   not hide large persistent state in transient per-token scratch buffers.

Acceptance criteria:

- Initialization reports the expected class counts: 2 sliding layers, 21 C4A
  layers, and 20 C128A layers.
- A real model initializes every required layer tensor without null pointers or
  shape reinterpretation.
- `reset_generation_state()` clears local, compressed, and indexer state and
  reproduces the first run exactly.
- Main and compressed RoPE tables differ where the checkpoint requires them;
  positions 0 and 65536 are covered by a deterministic unit test.
- The memory report accounts for the new state categories instead of treating
  the old full-resolution cache as the model contract.

### Stage 3 implementation record (2026-09-13)

The production layer-ownership slice is complete. `V4Layer` now keeps its
resolved layer descriptor private after initialization and owns an explicit
128-slot local key/value ring with absolute-position metadata. CSA and HCA
layers additionally own their class-specific compressor partial rows,
compressed key/value entries, and valid-count metadata; CSA layers also own
indexer keys, partial rows, query/weight/score state, candidate positions, and
deterministic top-k workspace. The legacy `d_kv_cache` name remains only as a
non-owning alias to the local key cache; the current sliding path uses the ring
and its absolute-position metadata.

`V4ModelResources` now builds and uploads separate main and compressed RoPE
tables from the parsed theta and YaRN settings. The pipeline scratch object
owns bounded per-token compressor/indexer projections and top-k metadata, while
large persistent state remains in each layer. The sliding kernel now filters
ring slots by absolute position, and `reset_generation_state()` clears local,
compressed, compressor, indexer, position, and top-k state together. The memory
budget reports local K/V, compressed K/V, compressor, indexer, metadata, RoPE,
and total attention-state categories using the same layout used for allocation.

The focused validation targets are:

```text
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R '^(test_v4_attention|test_v4_attention_oracle|test_v4_real_attention_oracle|test_v4_model_contract|test_v4_layer_state|test_v4_layer_state_device|test_dynamic_expert_pool|test_hot_warm_cold_pipeline)$'
```

On the Radeon RX 7900 XTX (`gfx1100`), the full build and all eight focused
tests passed. The model-backed real-oracle test completed in approximately
10.4 seconds, the context-256 43-layer pipeline initialization and one-token
smoke completed in approximately 8.5 seconds, and the device state test
verified ring-slot reuse at positions 0 and 128 plus reset sentinels. The
resolved schedule is 2 Sliding, 21 CSA, and 20 HCA; the earlier 3/20/20 text
was an acceptance-text error, not a checkpoint change.

This closes Stage 3 layer-state ownership, resource allocation, reset, and
memory-accounting work. It does not establish C4A/HCA production dispatch,
HIP-versus-oracle parity, true batched prefill, or trusted-reference parity.
Stage 4 is next: serial dispatch through the class-specific state machine.

### Stage 4: Implement correct serial decode semantics first

**Objective:** Replace the all-layer sliding fallback with a correct, simple
single-token execution path before attempting throughput-oriented prefill.

Tasks:

1. Preserve `V4Pipeline::step(token_id, pos, phase)` as the first integration
   surface, but dispatch each layer through its `V4LayerSpec`.
2. Keep the current SWA path as the layer-0/1 branch only, after comparing
   its K/V insertion, sink handling, RoPE, and inverse-RoPE order with the
   oracle.
3. Add the C128A branch for layers 3, 5, ..., 41. Verify that a compressed
   entry is created only when its causal region completes.
4. Add the C4A branch for layers 2, 4, ..., 42. Verify local plus selected
   compressed attention and the Lightning Indexer in separate trace points.
5. Keep top-k selection in a simple deterministic implementation first. A
   slower host or one-block device selection is acceptable while semantics are
   being proven.
6. Compare each branch after every major operation: query, KV insertion,
   compressor state, compressed entry, indexer scores/indices, attention output,
   inverse-RoPE output, and grouped output projection.

Acceptance criteria:

- Layer 0, layer 2, and layer 3 each pass the oracle trace on real checkpoint
  tensors for positions before and after their first compression boundary.
- A token beyond position 128 changes C4A/HCA state and output according to the
  compressed branch, while a sliding layer retains only its local causal
  window.
- No layer 2-41 execution reaches the old all-layer SWA kernel by accident.
- The existing native generation API still returns a token and preserves its
  public behavior shape, but its output is now labeled as the corrected path
  only after this stage passes.

### Stage 4 implementation record (2026-09-13)

The serial production dispatch slice is implemented in
`V4Pipeline::step()`. Sliding layers retain the existing ring attention branch;
CSA layers now run their compressor and Lightning Indexer projections, write
APE-adjusted partial rows, materialize overlapping ratio-4 entries at causal
boundaries, perform stable host-side top-k ordering, and attend over local plus
selected compressed entries. HCA layers run the same persistent compressor
state path with non-overlapping ratio-128 materialization and attend over all
completed compressed entries. Local values are copied before class-specific
RoPE so the cache keeps separate key and value semantics.

The conservative HIP primitives are in
`src/architecture/deepseek_v4/kernels/v4_attention.hpp`; they keep compressor
state in float32, normalize and rotate completed entries before FP16 storage,
and use a bounded shared score buffer for the 128-token local ring plus the
512-entry CSA selection. Top-k scores are read back for deterministic ordering
while the semantic path is being proven; this is intentionally not a
performance implementation.

The production trace boundary is implemented in
`src/architecture/deepseek_v4/core/v4_attention_trace.hpp` and
`V4Pipeline::enable_attention_trace()`. It is disabled by default and captures
projection inputs, rotated query/local cache state, compressor partial rows,
compressed entries, indexer state and scores, deterministic top-k indices,
attention output before and after inverse RoPE, and grouped output projection.
`tests/test_v4_stage4_trace.cpp` replays the captured layer inputs through the
independent CPU oracle and checks layers 0, 2, and 3 through position 131,
including local ring reuse, the ratio-4 boundaries, the first ratio-128
boundary, post-position-128 state, CSA top-k, and grouped output projection.

Two semantic discrepancies exposed by this trace were fixed before the gate
was accepted: production YaRN now uses the configured beta-derived correction
range, and short-context CSA selection retains all valid candidates in
ascending index order when the candidate count is below `index_topk`.

The reproducible Stage 4 validation commands are:

```text
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R '^(test_v4_attention|test_v4_attention_oracle|test_v4_real_attention_oracle|test_v4_model_contract|test_v4_layer_state|test_v4_layer_state_device|test_v4_class_attention_device|test_v4_stage4_dispatch|test_v4_stage4_trace|test_dynamic_expert_pool|test_hot_warm_cold_pipeline)$'
```

The earlier Stage 4 pass record is superseded by the bounded rerun documented
in `AEON_V4_REVIEW_AND_FIX_REPORT.md`. The focused component and oracle tests
still pass, but `test_v4_stage4_trace` now fails after the dedicated layer 0,
layer 2, and layer 3 runs during the subsequent all-layer position-0 check:
layer 0 HC mixes differ by `88.590942` with a `0.002` tolerance. A fresh
one-token all-layer diagnostic passes, so the serial trace gate is open again
until repeated-reset corruption is isolated. Stage 4 must not be described as
closed, and the Stage 5 equivalence results remain metamorphic evidence rather
than trusted-reference model correctness.

### Stage 5: Implement stateful and hybrid prefill

**Objective:** Make prompt processing mathematically equivalent regardless of
how the prompt is chunked, then add a hybrid batched path without changing
semantics. Fully batched stateful attention and routed-expert execution are
separate performance work.

Tasks:

1. Add an explicit prefill API that accepts a token span and absolute starting
   position. It may initially use a conservative serialized implementation,
   but it must update all layer states through the same state machine.
2. Add a batched/chunked prefill path for multiple prompt tokens. Keep the
   current `step()` API as the decode and reference fallback path.
3. Ensure compressor state crosses chunks. A chunk boundary must never reset a
   partial C4A or C128A window.
4. Ensure router token IDs, phase labels, and expert requests retain the
   original absolute positions during prefill.
5. Compare the next-token logits after:
   - one-shot prefill;
   - chunks of 1 token;
   - aligned chunks of 4 and 128 tokens;
   - unaligned chunks such as 3, 7, and 127 tokens.
6. Only after semantic equivalence passes, optimize prefill GEMM and expert
   batching. Do not let padded `M=16` scratch allocation be mistaken for true
   batched execution.

Acceptance criteria:

- All prefill chunkings produce the same per-layer state snapshot and next-token
  logits within the fixed tolerance.
- The first generated token is identical whether it follows one-shot or
  serialized prefill.
- A reset between prompts removes all previous local, compressed, indexer, and
  routing state.
- The batched path has a traceable fallback to the serialized oracle path for
  unsupported shapes; no silent semantic substitution is allowed.

### Stage 5 serialized/chunk-equivalence implementation record (2026-09-13)

The first Stage 5 subgate is complete. `V4Pipeline::prefill()` now accepts a
token span and absolute starting position, rejects gaps and context overflow,
and advances the same state machine used by `step()`. `generate()` uses this
API for prompt processing. `snapshot_generation_state()` exposes local,
compressor, compressed, indexer, and top-k state so chunk boundaries can be
checked directly rather than inferred from the final token alone.

`tests/test_v4_prefill_state.cpp` validates a 132-token real-model prompt
through position 131 with repeated one-shot replay, serialized one-token
append, aligned 4-token and 128-token chunks, and boundary splits at
`[1, 3, 4, 7, 16, 127, 128, 132]`. It compares metadata exactly, FP16 state
at `0.02`, FP32 state at `0.002` (indexer scores at `0.1`), logits at `0.02`,
and greedy token IDs exactly. Reset clears all persistent attention state and a
nonzero starting position is rejected.

The model-backed test uses the proven 35 GiB Warm profile by default on the
64 GiB host and accepts a GiB command-line argument; `0` remains the explicit
cold-only control. The selected package initialized with 675 Hot slots, 2,642
Warm slots, and 7,691 Cold slots on the RX 7900 XTX. This is a test resource
policy, not an attention-semantic requirement: the earlier `warm_host_bytes =
0` setting forced every non-Hot expert through the SSD path and made repeated
equivalence runs unnecessarily slow.

The fused six-expert W2 path uses floating-point `atomicAdd`, which is not
replay-stable enough for exact chunk-equivalence evidence. The test therefore
enables the new opt-in deterministic routed-expert accumulation mode, which
uses the existing single-expert W2 GEMV and sequential accumulation kernels;
the production default remains the fused path. A per-launch W2 completion
counter reset is also issued on the compute stream for both modes.

Validation on Radeon RX 7900 XTX (`gfx1100`):

```text
cmake --build build --parallel
./build/bin/test_v4_prefill_state
ctest --test-dir build --output-on-failure -R '^(test_aeon_moe_fused_w2|test_v4_prefill_state|test_v4_stage4_dispatch|test_v4_stage4_trace|test_dynamic_expert_pool|test_hot_warm_cold_pipeline|test_text_generation)$'
```

The Stage 5 test and all affected regressions passed. This closes the
serialized, aligned, and unaligned state-equivalence subgate and the explicit
serialized-fallback contract.
For the complete five-schedule correctness run, the explicit cold-only control
took `283.49 s` and the 35 GiB Warm profile took `205.55 s` on the same device,
a `27.5%` reduction. This is a resource-policy measurement for test runtime,
not a model-throughput result.

The public `prefill_batched()` contract now makes that boundary explicit. It
accepts a requested batch size, reports whether execution used the hybrid
batched path or `SerializedFallback`, and can reject fallback when a caller
requires a real multi-token entry point. Multi-token requests use a
fixed-capacity 16-row batch path and split larger requests into contiguous
chunks; single-token requests or an explicit batch size of one retain the
serialized fallback. Dense/HC/MLA/FFN projections, shared-expert work, and
router logits use batch rows, while local cache, compressor, compressed-entry
materialization, indexer top-k, attention, routed expert supply, routed W1/W3,
routed W2, and cleanup remain ordered per token. The new path uses the
existing deterministic per-expert W2 accumulation when that correctness mode
is enabled; it is a semantic implementation, not a fully parallel batch
performance claim.

### Stage 5 hybrid batched prefill implementation record (2026-09-14)

The production batch path is implemented in `V4Pipeline::prefill_batched()`
and `prefill_batched_chunk()`. `PipelineBatchScratchBuffers` owns fixed-capacity
device rows for embeddings, HC mixes, MLA projections, compressor/indexer
projections, FFN activations, router results, and shared-expert output. New
Wave32 batch kernels cover HC projection/pre-combine and clamped SwiGLU. The
existing batch-indexed GEMV and grouped output kernels are used with explicit
`grid.y` token rows and Wave32 blocks.

The causal attention state machine still advances each token in order inside a
chunk. That ordering is required for local ring insertion, C4 overlap,
C128 boundary materialization, Lightning Indexer candidate selection, and
absolute-position metadata. Routed expert supply remains correctness-first and
waits for each token's residency and W2 accumulation before releasing its
leases; this avoids silently treating the existing six-expert storage path as
a multi-token expert kernel.

`tests/test_v4_prefill_state.cpp` now requires `V4PrefillExecutionPath::Batched`
for a 16-token request, verifies that fallback can be disallowed for a
supported request, and compares the batch result against the serial baseline
for a 132-token prompt through position 131. The comparison covers exact
metadata and positions, fixed FP16/FP32 state tolerances, logits, greedy token
IDs, and the continuation decode step. The request passed on the Radeon RX
7900 XTX (`gfx1100`) with the 35 GiB Warm profile.

Validation:

```text
cmake --build build --parallel
./build/bin/test_v4_prefill_state
```

This closes the hybrid batched-prefill semantic subgate. It does not close a
fully batched llama.cpp-style stateful path: batch state planning, per-query
compressed visibility, batched compressed attention, batched indexer selection,
batch-wide expert acquisition/grouping, and larger single-launch capacities
remain future optimization work. Trusted-reference parity remains the next
correctness gate.

### Full-model validation execution policy (2026-09-14)

The complete 43-layer checkpoint must not be run through the float32 CPU oracle.
That oracle remains intentionally narrow: selected layer 0/2/3 traces,
compressor/indexer boundary transitions, dense-operation fixtures, and state
serialization. It is a semantic diagnostic, not a second implementation of
the full 170 GiB decoder.

Complete-model evidence is collected on the target accelerator through the
opt-in [`record_v4_gpu_evidence`](../../../tools/record_v4_gpu_evidence.cpp)
tool. It runs the existing HIP pipeline with deterministic routed-expert
accumulation and records the exact input IDs, model/runtime policy, GPU
identity, prefill execution path, greedy token IDs, top-k logits, half-logit
FNV-1a checksums, and timing. `--include-logits` additionally writes the full
129,280-element logit vectors for selected-reference comparison. The tool is
not a correctness claim by itself and is deliberately not registered as a
default CTest because model initialization and expert residency are expensive.

The trusted-reference gate therefore has two execution lanes:

1. Compare selected layer/state checkpoints against the host oracle and an
  independent reference implementation on a short deterministic fixture.
2. Run the complete decoder on an accelerator and compare final logits/top-k
  and greedy IDs using the same token IDs, model revision, and runtime
  contract. If an independent accelerator-backed reference is unavailable,
  the full-model trusted-reference gate remains open; a CPU fallback must not
  be substituted merely to produce a result.

  The first recorder run completed on the Radeon RX 7900 XTX (`gfx1100`) with
  the 35 GiB Warm policy and the exact input IDs `[1, 101, 2054, 300]`. The
  prefill used the reported `Batched` path, produced greedy token `982`, and
  took `1618.20 ms`; the continuation decode produced token `875` in
  `341.57 ms`. All 129,280 logits were finite. The JSON artifact was written to
  `build/correctness/v4_gpu_evidence.json`. These numbers are a workflow smoke
  result, not a performance-ledger entry or an independent parity result.

  The local `ds4` checkout is a candidate for the independent accelerator lane:
  revision `6289c516273979173abbc062209a81dd3706b804` contains a native ROCm
  DeepSeek-V4 implementation and a Safetensors-to-GGUF converter. It is not a
  drop-in reader for the Aeon artifact: its runtime consumes project-generated
  GGUF, defaults to `gfx1151`, and no compatible GGUF is currently present in
  the workspace. Before using it for parity, convert the same 0731 source
  snapshot with a pinned DS4 template, build for `gfx1100`, verify tensor names,
  shapes, and quantization against the selected source, and record that artifact
  identity beside the Aeon evidence.

### Stage 6: Port the validated branches to HIP on silicon

**Objective:** Implement the smallest correct Wave32 kernels and compare them
against the CPU oracle before tuning them.

Port in this order:

1. Class-aware RoPE table lookup and per-position q/KV preparation.
2. Local sliding cache insertion and attention, including separate K/V inputs
   if required by the reference trace.
3. C128 compressor state update, boundary reduction, normalization, RoPE, and
   compressed cache insertion.
4. C4 compressor state update with overlap and APE state.
5. Indexer query projection, compressed-key scoring, causal top-k selection, and
   selected-entry attention.
6. Class-aware output projection and HC integration.

Implementation constraints:

- Keep a CPU-oracle path available behind a test/debug option until the full
  model gate closes.
- Use float32 accumulation where the reference requires it; convert to FP16
  only at an explicit storage boundary.
- Do not optimize top-k, cache paging, compression quantization, or prefetch
  overlap in the same change as the first semantic port.
- Preserve Wave32 launch assumptions and validate synchronization around shared
  state and cache writes.
- Treat indexer/compressed cache quantization as a later optimization. The first
  semantic HIP path may use FP16/FP32 state if memory permits the test fixture.

Acceptance criteria:

- `test_v4_attention` is split or extended so SWA, C4A, and C128A each compare
  HIP output against the oracle rather than only against a local kernel mirror.
- Device traces match host traces for cache valid counts, compressed entries,
  indexer top-k indices, attention outputs, and output projections.
- Boundary tests run on the target gfx1100 device and pass after repeated
  launches, not only once after process startup.
- HIP errors, invalid positions, invalid top-k indices, and missing cache state
  fail explicitly during correctness tests.

### Stage 6 implementation record (2026-09-14)

Stage 6 is complete for the serial correctness path. The class-aware main and
compressed RoPE resources, local SWA branch, C4A compressor/indexer branch,
C128A compressor branch, deterministic top-k path, grouped output projection,
and HC integration are implemented in the production HIP pipeline. The
Stage 4 device trace already exercised these branches on `gfx1100` against the
independent CPU oracle for layers 0, 2, and 3 through position 131, including
cache metadata, compressor boundaries, compressed entries, indexer selection,
attention outputs, and grouped projection.

This closes the Stage 6 acceptance boundary for semantic serial execution. It
does not claim a fully batched stateful graph, batch-wide compressed attention,
or batch-wide routed-expert execution; those remain performance work. The
active correctness gate is now Stage 7: complete-model parity against an
independent accelerator-backed reference.

### Stage 7: Complete layer and full-model parity

**Objective:** Prove that corrected attention composes with the existing mHC,
MoE, quantized experts, final head, and text contract.

**Status:** Active. Selected-layer HIP/oracle parity is complete; the complete
43-layer trusted-reference comparison and its evidence artifact remain open.

**Boundary clarification (2026-09-14):** Stage 7 does not add another Aeon
production engine and does not change the `.aeon` runtime contract. Aeon remains
the only production path and continues to consume the native `.aeon` artifact.
An external implementation such as vLLM or DS4 is used only as an offline
reference process to emit comparison values; it is not linked, vendored, or
required by Aeon at runtime. The reference may read the original source
Safetensors because that is its input contract, while Aeon continues to be
tested against the resulting values from its `.aeon` artifact.

**Current blocker:** the Aeon-side GPU evidence recorder is available, but the
workspace currently has only the vLLM/DS4 source references, not an executable
independent reference environment for this checkpoint. vLLM's Python runtime
dependencies are absent, and DS4 requires a separately generated GGUF with a
different quantization/input contract. This is an evidence-environment gap,
not a missing Aeon Stage 5 or Stage 6 implementation.

**Minimum package to close Stage 7:**

1. A pinned external reference revision and a runnable source-checkpoint
  configuration, preferably on an accelerator rather than through a complete
  CPU decode.
2. Identical formatted token IDs, absolute positions, greedy settings, source
  checkpoint identity, and base-decoder-only mode on both paths.
3. Aeon GPU captures for the selected layer-0/2/3 checkpoints and final logits;
  the existing attention trace covers the attention state, while
  `record_v4_gpu_evidence` covers the complete-model final output.
4. Reference values at those selected checkpoints plus the final logits/top-k,
  followed by one comparison report recording max error, token agreement, and
  any quantization or numerical differences.

Until item 1 exists, Stage 7 is correctly marked active but blocked at the
external-reference lane. No second production engine is required to unblock it.

### Stage 7 native end-to-end smoke record (2026-09-14)

Before waiting on an external reference, the native `.aeon` text path was run
on the Radeon RX 7900 XTX with the verified DSV4 formatter and tokenizer, a
35 GiB Warm policy, and the prompt `What is 2 + 2? Answer with just the
number.` The model initialized all 43 layers, completed prefill and decode,
and passed through EOS-aware detokenization, but the behavioral gate failed:

- Default fused routed-expert accumulation produced generated IDs `[1]` and an
  empty response because EOS ranked first.
- Replay-stable deterministic accumulation produced `[1999, 344, 270, 20, 1]`,
  decoded as `What is the2`, which is not a coherent answer.
- Explicit Thinking mode with deterministic accumulation again produced
  `[1]` and an empty response.

The exact formatted prompt IDs were recorded in the diagnostic output. A
deterministic replay of those IDs through `record_v4_gpu_evidence` reproduced
the non-EOS top token, separating the fused atomic W2 instability from the
remaining full-block semantic failure. The formatter and EOS IDs match the
source encoding contract; this is therefore a failing model-composition smoke,
not a text-wrapper failure.

The post-layer trace boundary is now implemented for the full block. Use the
all-layer position-0 capture and the representative long traces to identify
the first layer-specific attention/output discrepancy before attempting the
complete trusted-reference comparison.
Do not unlock routing placement or claim coherent generation until this smoke
passes with a non-empty, task-correct response.

Build a trace harness that can compare the same token IDs and positions at
stable boundaries. At minimum capture:

```text
embedding
HC attention pre-mix
attention normalized input
Q and KV projections
local cache insertion
compressor state and compressed entries
indexer scores and selected indices
attention output
grouped output projections
HC attention post state
FFN normalized input
router logits, expert IDs, and weights
shared/routed expert output
HC FFN post state
final head input
logits and greedy argmax
```

Required comparison order:

1. One layer-0 SWA trace.
2. One layer-2 C4A trace.
3. One layer-3 C128A trace.
4. A complete 3-layer prefix containing layers 0-2.
5. The complete 43-layer base decoder on a short deterministic token-ID
   sequence.
6. Longer positions crossing 128 and compressed boundaries.

The trusted reference may be a one-layer or selected-layer harness built from
the checked-in source/reference implementation when the full checkpoint cannot
fit in the local reference runtime. A text-only external API comparison is
useful but cannot replace intermediate tensor or logit evidence.

### Stage 7 full-block trace implementation record (2026-09-14)

The production trace now supports a compact all-layer position-0 capture in a
single GPU pass. `test_v4_stage4_trace` validates every layer 0-42 against
independent host calculations for the real layer-specific tensors at these
boundaries: HC attention input, FFN normalized input, router logits, shared
expert output, combined routed MoE output, and HC FFN post residual. The same
test continues to validate the long cache/compressor/indexer attention traces
through position 131 for representative layers 0, 2, and 3.

This closes the selected full-block composition subgate across all 43 layers;
it does not prove all-layer compressed-attention boundary semantics or produce
a coherent text answer. The native text smoke remains failing, so the next
repair target is the first layer-specific attention/output discrepancy outside
the representative long traces, not another generic HC or MoE test.

The canonical user-facing path is explicit: `generate_until_stop()` processes
prompt and decode tokens through `prefill()`/`step()`. `prefill_batched()` is an
opt-in hybrid path used for semantic comparison and future performance work;
it is not called by native text generation. On the exact 19-token formatted
arithmetic prompt, serialized and hybrid prefill produced identical final and
continuation half-logit checksums and greedy IDs, so there is no evidence of
state overlap or double processing between the two paths.

Acceptance criteria:

- CPU oracle and independent reference agree on the selected layer traces.
- HIP and CPU oracle agree within the fixed dtype tolerance. Thresholds are
  declared before the run and are not changed per failing layer.
- For identical model revision, token IDs, and greedy settings, the full-model
  logits/top-k outputs agree at the recorded checkpoints. Exact token IDs are
  required when both paths use the same quantized artifact and deterministic
  arithmetic; otherwise any divergence is explained and recorded.
- The three most informative failure classes are distinguishable: attention
  state, weight/dequantization, and output/routing composition.
- Existing Hot/Warm/Cold tests still pass, demonstrating that the corrected
  attention state did not invalidate expert residency ownership.

### Stage 8: Text contract and external behavioral gate

**Objective:** Confirm that the corrected base model is being exercised with the
right prompt format and stopping behavior.

Follow [TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md](TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md):

1. Use the native DSV4 formatter and tokenizer artifact, not an assumed ChatML
   format.
2. Record exact formatted prompt IDs, mode, tokenizer/formatter hashes, model
   revision, generated IDs, decoded text, and stop reason.
3. Compare identical formatted IDs with a trusted compatible reference. Use
   deterministic greedy settings where possible.
4. Prefer logits/top-k/token IDs over prose-only comparison. An external API
   match is behavioral evidence, not activation or routing parity.
5. Include a long answer, arithmetic/exact-format tasks, multi-turn context,
   and a case that crosses a local/compressed attention boundary.

Acceptance criteria:

- Formatter/tokenizer parity is exact for the existing fixtures.
- The corrected native path reaches EOS or the declared context/generation
  limit without feeding EOS back into the model.
- Any divergence from the trusted reference is classified as prompt contract,
  weight/quantization, attention state, numerical tolerance, or provider/model
  revision rather than recorded only as a different sentence.
- The external gate is recorded with provider/model revision and request
  parameters. It does not unlock profiling on prose similarity alone.

### Stage 9: Unlock dependent work and handle deferred features

After Stages 0-8 pass:

1. Update `ROUTING_PROFILE_AND_PLACEMENT_STUDY.md` to mark the correctness
   evidence and exact text contract used by the profile corpus.
2. Re-run a small routing instrumentation smoke test, then discard or label
   all pre-correction profiles as invalid for placement decisions.
3. Build separate profile and held-out corpora using the verified formatter.
4. Resume Phase 2 cold-tier and placement measurements only after the corrected
   model is the measured model.
5. Create a separate MTP execution plan for `num_nextn_predict_layers = 1`,
   `mtp.*`, and DSpark metadata. Do not fold speculative execution into the
   base correctness patch.
6. Add prefix caching only after the cache object can serialize and restore
   local state, compressor boundaries, compressed entries, indexer metadata,
   and absolute positions as one transaction.

Acceptance criteria:

- `profile_routing` metadata identifies the correctness plan/gate, model
  revision, tokenizer/formatter hash, layer schedule, and corrected runtime.
- No placement conclusion uses a pre-correction trace.
- MTP and prefix-cache support are either implemented and tested in their own
  records or explicitly reported as unavailable.

## 7. Test and validation matrix

### CPU-only or host-reference tests

- `test_config_parser`: scalar values, nested YaRN values, and complete array
  schedule.
- `test_v4_model_contract`: layer classification, required tensor inventory,
  and rejection of incompatible schedules.
- `test_v4_attention_oracle`: SWA/C4A/C128A state transitions and top-k.
- `test_v4_prefill_state`: one-shot/chunked/serialized equivalence.
- independent real-tensor dequantization and dense projection tests.

### HIP component tests

Retain and run the existing regression surfaces while extending them with the
new oracle comparisons:

```text
test_v4_attention
test_hc_sinkhorn
test_moe_router
test_swiglu_clamp
test_w4a16_swizzle
test_w4a16_swizzled_gemv
test_w4a16_swizzled_dual_gemv
test_aeon_moe_fused_w13
test_aeon_moe_fused_w2
```

### Model-backed tests

Add or extend focused targets for:

```text
test_v4_layer_parity
test_v4_full_model_correctness
test_v4_generation_correctness
```

Each model-backed test must state context size, artifact identity, model
revision, residency policy, token IDs, generation settings, and whether it is
checking a CPU oracle, HIP trace, or external reference.

### Suggested local commands

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R 'test_(config_parser|v4_attention|hc_sinkhorn|moe_router|w4a16|aeon_moe)'
ctest --test-dir build --output-on-failure -R 'test_(v4_|hot_warm_cold_pipeline|text_generation)'
```

Run model-backed HIP tests on the intended gfx1100 device. Record any unavailable
external reference separately; an unavailable reference blocks the final claim
of full correctness but does not block CPU-oracle and component work.

## 8. Artifacts and reporting required at each gate

Every completed stage must leave compact, reproducible evidence:

- config and tensor inventory hashes;
- reference checkout revision;
- test command and hardware identity;
- token IDs and absolute positions;
- layer class and compression ratio;
- cache valid counts and boundary decisions;
- top-k indexer selections where applicable;
- max absolute error, RMS error, and exact token/top-k agreement;
- whether weights were CPU reference, native HIP, or external reference;
- failure classification and next action when a gate fails.

Update [PERFORMANCE_LEDGER.md](../../status/PERFORMANCE_LEDGER.md) only for
meaningful silicon measurements. Correctness traces belong in the focused test
artifacts or a dedicated correctness output directory, not in benchmark prose.

## 9. Non-goals and stop conditions

This plan does not authorize:

- changing the version-2 swizzled expert format;
- replacing symmetric INT4 with an inferred FP4/MXFP4 decoder;
- adding Python, PyTorch, vLLM, or an external service as a runtime dependency;
- optimizing storage, expert placement, or multi-GPU scheduling before the base
  model gate;
- replacing CSA/HCA with full attention or a larger sliding window as a claimed
  correctness workaround;
- treating a readable generated answer as proof of model parity;
- adding universal model abstractions before the V4 state machine is proven.

Stop and resolve the discrepancy when:

- the checkpoint tensor names/shapes do not match the assumed compressor or
  indexer contract;
- vLLM/source/config disagree on a boundary or position convention;
- a required tensor is absent from the `.aeon` dense inventory;
- CPU oracle and independent reference disagree before HIP is involved;
- top-k selection differs even when attention output happens to look plausible;
- prefill chunking changes a compressed state or next-token result;
- a performance change makes correctness failures disappear without identifying
  the semantic cause.

## 10. Relationship to existing plans

- [PHASE_2_EXECUTION_PLAN.md](PHASE_2_EXECUTION_PLAN.md) remains paused for
  cold-tier, layout, host-pressure, and latency-hiding work.
- [TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md](TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md)
  remains the owner of tokenizer, formatter, EOS, and external behavioral
  comparison work.
- [ROUTING_PROFILE_AND_PLACEMENT_STUDY.md](ROUTING_PROFILE_AND_PLACEMENT_STUDY.md)
  remains locked until this plan's final gate passes.
- [BACKEND_GENERALIZATION_EXECUTION_PLAN.md](BACKEND_GENERALIZATION_EXECUTION_PLAN.md)
  remains the owner of artifact/backend boundaries. This plan consumes the
  current swizzled backend; it does not generalize the V4 graph.
- [DOCUMENTATION_STATUS.md](../../status/DOCUMENTATION_STATUS.md) and
  `AGENTS.md` must be updated when this plan changes state.