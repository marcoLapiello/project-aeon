# Parallel GPTQ-Aeon Backend Analysis

**Date:** 2026-09-11
**Status:** Current analysis and architecture decision record
**Scope:** Feasibility and implementation strategy for
[sokada4/DeepSeek-V4-Flash-0731-GPTQ-Int4-V1](https://huggingface.co/sokada4/DeepSeek-V4-Flash-0731-GPTQ-Int4-V1)
as a sibling Aeon backend. This document does not investigate vLLM kernels.

## Executive decision

The GPTQ checkpoint should be treated as a valid candidate for a **parallel
GPTQ-Aeon backend**, not as a source that must be converted into the current
Aeon quantization contract and not as a reason to replace the current backend.

The proposed shape is:

```text
                         V4 model semantics
                 mHC, routing, attention, generation
                                  |
                    shared pipeline and supply contracts
          tokenizer, leases, Hot/Warm/Cold policy, I/O, telemetry
                    /                              \
       current compressed-tensors backend       GPTQ backend
       group32, symmetric, no qzeros            group128, GPTQ planes
       current W4A16 kernels                    GPTQ-aware kernels
       current .aeon artifacts                  GPTQ-Aeon artifacts
```

The current model remains the regression baseline. The GPTQ path gets its own
artifact format, dense-weight representation, expert-payload layout, and
quantized linear dispatch. The two paths share only contracts that have
independent meaning from the quantization format.

This is the conservative boundary for Project Aeon:

1. Preserve the current backend and its bit-exact artifact tests.
2. Add a GPTQ-preserving converter and an offline CPU/reference decoder.
3. Generalize the tiered expert supply chain around opaque, described payloads.
4. Specialize dense and expert weight access behind a backend interface.
5. Add GPTQ kernels and integrate them only after tensor-level and layer-level
   correctness gates pass.

The active [Warm-tier repair and supply telemetry plan](../../execution/active/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md)
remains the next production-runtime priority. The GPTQ work can begin with
offline inventory, conversion, and reference tests without changing that path.

## 1. Checkpoint evidence

The alternative repository is a quantized version of the same base model:

- Architecture: `DeepseekV4ForCausalLM` and `deepseek_v4`.
- Decoder layers: 43.
- Hidden size: 4096.
- Routed experts: 256 per layer.
- Routed experts per token: 6.
- Shared experts: 1 per layer.
- Vocabulary size: 129,280.
- Model repository commit observed during the investigation:
  `4375393f25f4d901712d9eebd2da610cfa27980e`.
- Safetensors shards: 36.
- Safetensors tensor inventory: 134,198 entries in the index.
- Published repository storage: approximately 152.8 GB.

Its quantization metadata is:

```text
method/checkpoint_format: GPTQ
bits: 4
group_size: 128
sym: true
desc_act: false
pack_dtype: int32
lm_head: false
quantizer: GPTQModel 7.3.4
```

The index contains 33,325 `qweight` matrices and matching `qzeros`, `scales`,
and `g_idx` tensors:

| Category | Matrices | Calculation |
| --- | ---: | --- |
| Routed experts | 33,024 | `43 * 256 * 3` |
| Shared experts | 129 | `43 * 3` |
| Attention projections | 172 | `43 * 4` |
| Total non-expert GPTQ matrices | 301 | `129 + 172` |

The routed names are structurally mappable but not name-compatible with the
current converter:

```text
model.layers.<L>.mlp.experts.<E>.gate_proj.{qweight,qzeros,scales,g_idx}
model.layers.<L>.mlp.experts.<E>.up_proj.{qweight,qzeros,scales,g_idx}
model.layers.<L>.mlp.experts.<E>.down_proj.{qweight,qzeros,scales,g_idx}
```

The current source instead uses names such as:

```text
layers.<L>.ffn.experts.<E>.w1.weight_packed
layers.<L>.ffn.experts.<E>.w1.weight_scale
```

The logical mapping is direct:

| V4 role | Current Aeon role | GPTQ role |
| --- | --- | --- |
| Gate projection | `w1` | `gate_proj` |
| Down projection | `w2` | `down_proj` |
| Up projection | `w3` | `up_proj` |

The checkpoint retains model-level and architectural tensors outside the
GPTQ linear set. The embedding header is BF16, the config declares BF16 as the
base dtype, and the LM head is explicitly unquantized. The first backend
implementation must therefore make the BF16-to-device policy explicit rather
than assuming that every non-GPTQ tensor is FP16.

## 2. Why a separate path is feasible

The user-visible model graph does not change because of GPTQ. The checkpoint
still represents the same V4 layer geometry, router topology, and expert
identity space. GPTQ changes how linear weights are stored and evaluated.

The current runtime combines those concerns in a few concrete classes. For
example:

- [v4_layer.hpp](../../../src/core/v4_layer.hpp) owns V4 layer state and also
  hardcodes FP16 device pointers for attention and shared-expert projections.
- [vram_expert_pool.hpp](../../../src/core/vram_expert_pool.hpp) hardcodes the
  current six-plane expert layout and `AEON_EXPERT_BYTES`.
- [aeon_loader.hpp](../../../src/core/aeon_loader.hpp) assumes the current
  dense container and fixed expert record size.
- [v4_pipeline.hpp](../../../src/core/v4_pipeline.hpp) selects current
  quantized expert pointers directly and hardcodes the current dense byte count
  in the dynamic memory budget path.

Those are implementation boundaries, not model limitations. A sibling path
can introduce a GPTQ loader and backend while leaving the current path alone.
The first version does not need a template explosion or a universal tensor
abstraction. It needs a small number of explicit contracts at the points where
the storage system and V4 graph meet.

## 3. GPTQ layout versus current Aeon layout

The current routed expert format is:

```text
W1 packed:  [2048, 512] uint32       4,194,304 bytes
W1 scales:  [2048, 128] fp16           524,288 bytes
W2 packed:  [4096, 256] uint32       4,194,304 bytes
W2 scales:  [4096,  64] fp16           524,288 bytes
W3 packed:  [2048, 512] uint32       4,194,304 bytes
W3 scales:  [2048, 128] fp16           524,288 bytes
                                      ----------------
                                      14,155,776 bytes
```

Its dequantization contract is symmetric group32:

```text
weight(k) = (nibble(k) - 8) * scale(output, k / 32)
```

The verified GPTQ shard headers show the following physical shapes for one
routed expert:

| Projection | `qweight` | `qzeros` | `scales` | `g_idx` |
| --- | --- | --- | --- | --- |
| Gate/up | `[512, 2048] I32` | `[32, 256] I32` | `[32, 2048] F16` | `[4096] I32` |
| Down | `[256, 4096] I32` | `[16, 512] I32` | `[16, 4096] F16` | `[2048] I32` |

The shapes show the GPTQ packed orientation: the input dimension is packed by
8 in the first dimension of `qweight`, rather than the current row-major
`[out, in / 8]` representation. The exact nibble and zero-point semantics
must still be established by an independent decoder test; `desc_act=false`
does not justify silently discarding `g_idx`.

Preserving all four GPTQ planes gives this initial per-expert size:

| Projection | Bytes |
| --- | ---: |
| Gate | 4,374,528 |
| Up | 4,374,528 |
| Down | 4,366,336 |
| **Total** | **13,115,392** |

That is exactly 3,202 4 KiB sectors. Compared with the current 3,456-sector
record, it saves 1,040,384 bytes, or approximately 7.35 percent per routed
expert. The earlier rough estimate of 13,279,232 bytes was incorrect; this
document uses the sizes calculated from the actual shard headers.

The initial GPTQ artifact should retain `g_idx` even if it later proves to be
sequential. Removing it is an optimization after semantic validation, not a
format assumption. The same rule applies to `qzeros`: the source metadata says
`sym=true`, but the checkpoint physically stores zero tensors, so the backend
must validate the reference interpretation rather than infer it from the
metadata label.

## 4. Why group size 32 is not required

Group size 32 was relevant only to reuse the current kernels and current
expert-pool layout. It is not a requirement of Aeon, RDNA3, or the V4 model.

For `K=4096`:

```text
group32  -> 128 groups per output row
group128 ->  32 groups per output row
```

For `K=2048`:

```text
group32  -> 64 groups per output row
group128 -> 16 groups per output row
```

The GPTQ path can therefore keep group128 and use kernels that understand:

1. GPTQ packed orientation;
2. group128 scales and zero points;
3. `g_idx` lookup or its validated sequential specialization;
4. the checkpoint's actual int32 packing order; and
5. FP16 activation with the required accumulation precision.

Converting group128 to group32 would be a separate requantization or
dequantize-and-requantize operation. It would add work, could change numerical
behavior, and would throw away the reason for selecting this checkpoint. It is
only justified if the existing group32 kernels are the chosen compatibility
path and the resulting quality and performance are acceptable.

## 5. Shared versus specialized architecture

The intended decomposition has three boundaries, not one universal runtime
abstraction.

### 5.1 DeepSeek-V4 architecture layer

This layer is model-family-specific and should be shared by both quantization
backends:

- V4 hidden and head geometry;
- four-stream mHC and Sinkhorn behavior;
- router scoring, hash layers, top-6 selection, and routing weights;
- shared and routed MoE graph semantics;
- sliding, compressed, and indexed attention layer schedule;
- KV/compressor/indexer state ownership;
- RoPE and YaRN configuration;
- tokenizer, chat formatting, generation, EOS, and MTP policy.

The current `V4Pipeline` and `V4Layer` are the starting point for this layer,
but `V4Layer` currently mixes graph state with concrete FP16 weight pointers.
The target is a V4 layer state object that owns architecture state and asks a
weight backend to execute each linear operation.

### 5.2 Quantization and weight backend layer

This layer is format-specific:

- source tensor name mapping;
- dense tensor plane descriptors;
- device storage type and dtype conversion;
- linear projection dispatch;
- routed expert plane layout;
- dequantization and accumulation rules;
- backend-specific kernel geometry;
- model-level dense byte accounting.

The current backend provides the existing symmetric group32 W4A16 behavior.
The GPTQ backend provides group128 GPTQ behavior. Neither backend should
pretend that the other backend's tensor descriptors are interchangeable.

### 5.3 Generic expert supply layer

This layer is format-agnostic as long as one expert can be represented as a
sector-aligned opaque payload:

- expert identity `(layer_id, expert_id)`;
- persistent owner: Hot, Warm, or Cold;
- transfer and publication state;
- leases and slot reclamation;
- direct I/O submission and completion;
- pinned staging ownership;
- Hot-to-Warm demotion and Warm-to-Hot promotion;
- source-tier byte counters and exposed-wait telemetry;
- memory-budget accounting from a runtime format descriptor.

The current Warm plan already defines the required ownership and transfer
invariants. The generic supply layer must not inspect whether bytes contain
`weight_scale`, `qzeros`, or `g_idx`.

## 6. Proposed contracts

The names below are design targets, not an instruction to implement every type
at once.

### 6.1 Artifact manifest

Each native model directory should have an explicit manifest, for example:

```text
aeon_manifest.json
  model_family: deepseek_v4
  backend: current_w4a16 | gptq_w4a16
  artifact_version
  num_layers
  experts_per_layer
  expert_payload_bytes
  sector_size
  dense_payload_bytes
  source_quantization
```

The manifest prevents the runtime from silently feeding a GPTQ artifact to the
current loader. It also removes the hardcoded dense byte count currently used
by the dynamic memory-budget path.

### 6.2 Expert format descriptor

The generic registry and I/O path should consume a descriptor containing at
least:

```text
artifact version
payload byte length
sector size
layer count
expert count per layer
```

The current backend descriptor reports 14,155,776 bytes. The first GPTQ
descriptor reports 13,115,392 bytes. The registry, direct reader, host pool,
staging arena, and budget engine use this value for capacity and offsets; the
backend owns the meaning of bytes inside the payload.

### 6.3 Weight backend

The V4 graph should request operations by semantic role, such as:

```text
attention.q_a
attention.q_b
attention.kv
attention.o_a
attention.o_b
shared_expert.gate
shared_expert.up
shared_expert.down
routed_expert.gate/up/down
router
```

The current backend resolves those roles to FP16 or current group32 views.
The GPTQ backend resolves them to GPTQ descriptors and dispatches the matching
device kernel. The graph should not call `get_data_ptr<uint32_t>()` or assume
that a layer projection is a raw `half*`.

### 6.4 Layer and pipeline composition

A practical incremental shape is composition rather than a fully templated
pipeline:

```text
V4LayerState
  architecture state, cache state, router state, norm/HC state

V4WeightBackend
  backend-specific dense bindings and linear dispatch

TieredExpertStore
  backend-described expert payloads and residency state

V4Pipeline
  orchestrates V4LayerState, V4WeightBackend, and TieredExpertStore
```

The first GPTQ implementation may use a sibling `GptqV4Pipeline` or an
explicit backend selection inside `V4Pipeline` if that keeps the current path
stable. The long-term target is one V4 orchestration path with backend
composition, but that refactor should follow two working backends rather than
precede them.

## 7. GPTQ-Aeon artifact proposal

The converter should continue to split source tensors into a dense artifact and
an expert artifact, but it should preserve GPTQ planes instead of translating
them:

```text
<model-dir>/
  aeon_manifest.json
  config.json
  tokenizer files
  model_dense_gptq.aeon
  model_experts_gptq.aeon
  model_experts_gptq.index
```

The dense directory should describe each tensor or linear module with plane
records. A GPTQ linear record needs offsets, byte sizes, dtypes, shapes, and
the relationship between `qweight`, `qzeros`, `scales`, and `g_idx`. An
unquantized BF16 or FP16 tensor remains an ordinary dense record.

The expert file should store one fixed-size record per `(layer, expert)` in a
documented order, initially:

```text
gate.qweight  gate.qzeros  gate.scales  gate.g_idx
down.qweight  down.qzeros  down.scales  down.g_idx
up.qweight    up.qzeros    up.scales    up.g_idx
```

The exact order is a format decision and may instead follow gate/up/down
kernel consumption. The important properties are:

- every record starts on a 4096-byte boundary;
- every record has a fixed, manifest-declared byte length;
- the index maps `(layer, expert)` to offset and length;
- the converter verifies all expected tensors and shapes;
- the converter performs no silent dequantization or requantization;
- payload samples are byte-exact against the source Safetensors tensors;
- an independent CPU decoder verifies numerical semantics.

The existing converter can remain the current-backend converter. A separate
`convert_gptq_to_aeon.py` is preferable initially because it makes the two
format assumptions explicit and avoids adding GPTQ conditionals to a lossless
converter whose current contract is already validated.

## 8. Implementation sequence

### Stage 0: protect the current baseline

- Keep `model_dense.aeon`, `model_experts.aeon`, and the version-2 swizzled
  artifact unchanged.
- Keep current loader and kernel tests as regression tests.
- Do not make the current `UnifiedVRAMExpertPool` accept multiple layouts by
  guessing from byte counts.

### Stage 1: offline GPTQ inventory and reference decoder

- Download or stream the complete GPTQ index and shard headers.
- Validate the 43-layer, 256-expert inventory and all expected tensor counts.
- Check every expert's gate/up/down shape and dtype.
- Inspect `g_idx` values across representative layers and projections.
- Implement a small independent CPU GPTQ dequantizer for test fixtures.
- Compare representative projection outputs against a trusted GPTQ reference.

This stage should not require HIP or alter the production pipeline.

### Stage 2: GPTQ-preserving native artifacts

- Implement the separate converter and manifest.
- Write dense GPTQ plane records and fixed-size GPTQ expert records.
- Verify offsets, sector alignment, tensor counts, and byte identity.
- Add a loader smoke test that checks manifest/backend rejection and sample
  plane addresses.
- Measure actual dense and expert artifact sizes before changing the memory
  budget model.

### Stage 3: generalize the supply boundary

After the active Warm-tier invariants are stable, parameterize the shared
storage path by the expert format descriptor:

- `AEON_EXPERT_BYTES` becomes a descriptor value at runtime;
- host and staging pools allocate the descriptor's payload length;
- direct I/O uses the descriptor's record length and index;
- memory budgeting uses actual dense and expert sizes;
- telemetry reports source-tier bytes independent of format.

The current backend should be adapted first and run through its existing test
suite before the GPTQ backend is connected.

### Stage 4: GPTQ device backend

- Add GPTQ dense descriptors and device ownership.
- Add a correctness-first GPTQ linear kernel for one projection.
- Validate gate, up, and down routed projections independently.
- Add shared-expert and attention projections.
- Add the fused routed path only after the single-projection path is correct.
- Handle BF16 model-level tensors either with explicit conversion at load time
  or with native BF16 kernels; make the choice measurable and documented.

The kernel investigation itself is a separate work item and is intentionally
outside this document.

### Stage 5: V4 pipeline integration

- Select the backend from `aeon_manifest.json` or an explicit CLI option.
- Keep tokenizer and chat formatting shared.
- Keep routing and expert IDs shared.
- Route all linear operations through the selected backend.
- Share the tiered expert registry and telemetry after its descriptor
  generalization is proven.
- Reject an artifact/backend mismatch before allocating device memory.

### Stage 6: controlled comparison

Compare current and GPTQ backends with identical:

- formatted prompt text and token IDs;
- layer count and context size;
- Hot/Warm capacity;
- initial residency state;
- generation settings;
- hardware and stream configuration.

Record logits or hidden-state error, output tokens, dense bytes resident,
expert bytes by source tier, exposed GPU wait, transfer time, TTFT, and decode
tokens per second. Do not attribute a result to GPTQ kernels if the run also
changed Warm ownership or attention semantics.

## 9. Performance hypothesis

The GPTQ checkpoint is not automatically faster. Its likely performance value
comes from two separate effects.

### 9.1 Direct expert traffic reduction

The first GPTQ-preserving record is approximately 7.35 percent smaller than
the current expert record. At six routed requests across 43 layers:

```text
current: 258 * 14,155,776 = 3,652,190,208 bytes per all-cold token
GPTQ:   258 * 13,115,392 = 3,383,771,136 bytes per all-cold token
```

That saves approximately 268 MB per all-cold token. It is useful for NVMe and
PCIe service, but it is not by itself a transformative improvement. A GPTQ
kernel may also perform more metadata work because it must honor zero points,
group128 scales, and possibly `g_idx`.

### 9.2 Dense footprint and Hot capacity

The alternative quantizes 301 non-routed matrices that the current runtime
keeps as FP16. This can reduce the resident dense footprint by several GiB,
subject to GPTQ metadata, alignment, BF16 conversion, and runtime allocations.
That may allow materially more routed experts in Hot VRAM and reduce the
number of Warm or Cold requests.

This is the stronger performance hypothesis for Aeon's workload:

```text
GPTQ dense weights
        -> smaller resident dense allocation
        -> larger Hot expert pool
        -> fewer Warm/Cold expert services
        -> lower exposed supply wait
```

It is only a hypothesis until the actual GPTQ dense artifact is measured and
the same routing trace is replayed at the resulting Hot/Warm capacities.

Expected regimes:

| Regime | Expected result before measurement |
| --- | --- |
| Fixed all-Hot expert set | GPTQ may be neutral or slower initially because of dequantization overhead. |
| Same Hot/Warm capacities, misses present | Expert byte reduction helps modestly; kernel cost may offset part of it. |
| GPTQ dense footprint enables many more Hot slots | This is the most promising regime for end-to-end improvement. |
| LM-head dominated step | Benefit is limited because the LM head remains unquantized. |

The current Warm-tier repair must be measured separately. A GPTQ backend must
not be credited for a speedup caused by fixing the existing draining-Warm
state machine.

## 10. Correctness and risk gates

The following gates are mandatory before a performance conclusion:

1. **GPTQ tensor semantics.** Independent dequantization matches the trusted
   reference for qweight, qzeros, scales, and g_idx.
2. **Projection parity.** Gate, up, and down outputs match the CPU reference
   for real tensors and representative FP16 activations.
3. **Dense parity.** Quantized attention and shared-expert projections pass
   the same layer-level comparison.
4. **BF16 policy.** Embedding, norm, HC, router, and LM-head dtype handling is
   explicit and numerically tested.
5. **Artifact integrity.** All source tensors are present exactly once, all
   records are aligned, and backend mismatch is rejected.
6. **Supply integrity.** Hot/Warm/Cold ownership, leases, demotions, and
   staging remain correct with the new payload length.
7. **V4 architecture parity.** Attention schedule, compressed/indexed state,
   RoPE, and routing semantics are held constant when comparing backends.
8. **Performance attribution.** Hot hits, Warm hits, Cold bytes, exposed wait,
   and kernel time are reported separately.

The existing [DeepSeek-V4 comparison](DEEPSEEK_V4_FLASH_AEON_COMPARISON.md)
continues to govern the broader model-correctness gate. A GPTQ backend does not
remove the need to implement and validate the selected V4 attention schedule;
it only changes the weight execution backend.

## Final assessment

The GPTQ checkpoint is a credible alternative for Aeon, and a separate path is
architecturally cleaner than forcing either the checkpoint or the current
runtime into the other's quantization contract.

The right division is:

```text
DeepSeek-V4 graph and semantics: shared
Hot/Warm/Cold supply and telemetry: generalized and shared
GPTQ/current weight storage and kernels: specialized
```

Group128 should be preserved in the GPTQ backend. Group32 conversion is only a
compatibility fallback, not a requirement. The main potential win is not the
7.35 percent smaller expert record; it is the possibility that GPTQ dense
weights free enough VRAM to expand Hot residency. That opportunity should be
validated with a GPTQ-preserving artifact and controlled measurements before
any decision to retire or alter the current backend.
