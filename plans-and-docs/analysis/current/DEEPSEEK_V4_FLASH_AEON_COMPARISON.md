# DeepSeek-V4-Flash 0731 versus Aeon Runtime Comparison

Status: current analysis, 2026-09-11

## Executive conclusion

Project Aeon is pursuing a viable deployment direction, but the current
runtime is not yet an implementation of the selected DeepSeek-V4-Flash-0731
architecture. It is a V4-shaped single-token approximation with a strong,
model-specific MoE, mHC, weight-format, and storage foundation.

The distinction matters:

- The routed expert artifact is structurally compatible with the selected
  checkpoint. The source tensors are symmetric pack-quantized INT4 with
  group size 32, not a generic floating-point FP4 encoding. The Aeon converter
  preserves those packed bytes and FP16 scales, and the current dequantizer
  uses the reference `uint4b8` convention: low-nibble-first and `(q - 8) *
  scale`.
- The dense transformer geometry is substantially specialized for V4. The
  implementation includes the 4-stream mHC path, partial RoPE, shared 512-wide
  KV state, grouped output projection, hash routing, score-based top-6 routing,
  the shared expert, and the routed INT4 experts.
- The controlling correctness gap is attention. The production path has one
  contiguous `[max_seq_len, 512]` cache and calls the same 128-token sliding
  attention kernel for every layer. It does not implement the selected
  checkpoint's ratio-4 CSA compressor/indexer or ratio-128 HCA compressor.
- The configured YaRN factor and compressed-attention RoPE base are not applied
  by the current pipeline. True batched prefill, prefix caching, and the MTP
  component are also absent.

Therefore the current effort is not useless, but the present full-model text
output is not evidence of model correctness. The reusable work is mostly below
the attention-policy boundary. The next correctness milestone must be a
layer-class reference comparison, not another storage or expert-cache
optimization.

## 1. Checkpoint facts

The comparison uses the checked-in converted artifact and its source snapshot:

- 43 base decoder layers, hidden size 4096, 64 query heads, one shared KV head,
  and head dimension 512.
- Query low-rank dimension 1024, output low-rank dimension 1024, and eight
  output groups.
- Four mHC streams, 256 routed experts, one shared expert, six routed experts
  per token, and three hash-routed initial layers.
- Routed expert tensors are `I32` packed weights plus `F16` scales. W1/W3 are
  `[2048, 512]` packed words with `[2048, 128]` scales; W2 is `[4096, 256]`
  packed words with `[4096, 64]` scales.
- Non-expert tensors are predominantly F16, with F32 mHC tables, sink values,
  and router biases. This selected artifact is not an FP8 dense checkpoint.
- The source snapshot contains approximately 14.66 GiB of dense payload and
  155.25 GiB of routed packed expert payload. The latter is why a tiered
  storage design is necessary even though each individual expert is compact.

The authoritative local configuration is
[config.json](../../../models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/config.json).
The source tensor inventory also contains compressor and indexer tensors on
the ratio-4 and ratio-128 layers, and contains `mtp.*` tensors. Those tensors
are not merely theoretical configuration fields.

### Important schedule correction

The attached architecture summary describes layer 42 as sliding-window
attention. The selected 0731 `config.json` has 46 compression entries. The
first 43, which apply to the base decoder, are:

| Layers | Checkpoint ratio | Required class |
| --- | ---: | --- |
| 0-1 | 0 | Sliding-window |
| 2, 4, ..., 42 | 4 | CSA plus Lightning Indexer |
| 3, 5, ..., 41 | 128 | HCA |
| 43-45 | 0 | Trailing auxiliary entries, not base decoder layers |

This is a release-specific difference that must be resolved from the selected
checkpoint rather than from a family-level summary. The current C++ config
parser does not represent `compress_ratios`, `index_topk`, `index_head_dim`, or
the auxiliary schedule fields. See
[config.hpp](../../../src/architecture/deepseek_v4/core/config.hpp) and the source snapshot config.

## 2. Quantized model compatibility

### What is correct

The conversion/storage path is well grounded for the routed experts.

[convert_safetensors_to_aeon.py](../../../scripts/convert_safetensors_to_aeon.py)
copies the packed expert tensors and scales without dequantizing or
requantizing them. It writes the six source planes in a fixed contiguous
payload and checks the expected 14,155,776-byte size. The native loader and
[vram_expert_pool.hpp](../../../src/backend/swizzled_w4a16/core/vram_expert_pool.hpp) preserve that
same layout for NVMe, pinned staging, and VRAM.

The current W4A16 kernels assume:

```text
packed tensor: [out, in / 8] uint32
scale tensor:  [out, in / 32] fp16
q(k) = ((word >> (4 * (k % 8))) & 0xf) - 8
weight(k) = q(k) * scale(k / 32)
operation = activation @ weight.T
```

That is consistent with the independent local reference decoder for symmetric
compressed-tensors `pack-quantized` INT4. The shape and byte-size contracts are
also correct for W1, W2, and W3. The swizzled artifact changes layout only and
has a round-trip verification script.

The current tests establish useful silicon facts:

- packed-to-dequantized GEMM/GEMV agrees with the repository's CPU decoder;
- a real checkpoint W1 tensor can pass through the GPU kernel;
- the converted artifact is byte-identical to sampled source tensors;
- the routed expert output remains stable enough for the existing pipeline
  regressions.

### What is not yet proven

The tests do not yet establish end-to-end numerical parity with the selected
DeepSeek-V4 reference implementation. The CPU reference in
[test_w4a16_swizzled_gemv.cpp](../../../tests/test_w4a16_swizzled_gemv.cpp) repeats the same
nibble and scale assumptions as the GPU kernel. That proves GPU-versus-local
decoder agreement, not checkpoint-versus-reference-model agreement.

The remaining quantization gate is small and concrete: select several real
experts from W1/W2/W3, dequantize them with an independent reference path,
run the same FP16 activation through both implementations, and compare the
full output vector before the result is mixed into the transformer. Then run
one complete reference layer and compare hidden states and logits. This gate
should be completed before interpreting routing or placement measurements as
model evidence.

The label `expert_dtype: fp4` in the checkpoint config should not be used to
infer an FP4 floating-point encoding for this artifact. The actual source
metadata is symmetric integer pack-quantization. A future MXFP4/NVFP4 release
would be a different binary contract and must not be accepted by the current
loader just because it is also described as FP4.

## 3. What the engine implements correctly

The engine is not generic inference logic with only a V4 name. The following
parts are deliberately tied to this model:

| V4 requirement | Current status | Assessment |
| --- | --- | --- |
| 4096-wide hidden state and 43 layers | Implemented | Geometry matches the selected base model. |
| 64 Q heads, one 512-wide shared KV state | Implemented for current path | Tensor shapes and projections match, but the layer-specific cache policy is missing. |
| 448 non-RoPE plus 64 RoPE channels | Implemented | Current kernels rotate the trailing 64 channels. |
| q low-rank 1024 and grouped o projection | Implemented | `wq_a`, q norm, `wq_b`, grouped `w_o_a`, and `w_o_b` are present. |
| Four mHC streams | Implemented | mHC projection, Sinkhorn normalization, pre/post mixing, and head reduction exist. |
| 20 Sinkhorn iterations and 1e-6 epsilons | Implemented | The production launches use the V4 values. |
| One shared expert plus six routed experts | Implemented | The shared FP16 path and routed W4A16 path are separate. |
| First three hash layers | Implemented | `tid2eid` is loaded and used for layers 0-2. |
| Later sqrt-softplus routing with bias correction | Implemented | Router scores, non-hash bias, renormalization, and 1.5 scaling are present. |
| 10.0 SwiGLU clamp | Implemented | The production expert path clamps gate and up values. |
| Hot/Warm/Cold expert residency | Implemented as a runtime strategy | This is deployment machinery, not a replacement for model semantics. |

The owning implementations are
[v4_layer.hpp](../../../src/architecture/deepseek_v4/core/v4_layer.hpp),
[moe_router.hpp](../../../src/architecture/deepseek_v4/kernels/moe_router.hpp),
[hc_sinkhorn.hpp](../../../src/architecture/deepseek_v4/kernels/hc_sinkhorn.hpp), and
[v4_pipeline.hpp](../../../src/architecture/deepseek_v4/core/v4_pipeline.hpp).

## 4. Where the engine is currently an approximation

### 4.1 Attention is the major correctness blocker

[v4_attention.hpp](../../../src/architecture/deepseek_v4/kernels/v4_attention.hpp) contains a correct
local implementation of the currently chosen approximation: causal
128-token attention over a single 512-wide cached vector, with an attention
sink and a weighted sum that treats the cached vector as the value as well as
the key.

The production layer owns only:

```text
d_kv_cache: [max_seq_len, 512] fp16
```

and [v4_pipeline.hpp](../../../src/architecture/deepseek_v4/core/v4_pipeline.hpp) invokes
`v4_cached_sliding_window_attn_wave32_kernel` for every layer. There is no
per-layer dispatch based on `compress_ratio`.

For the selected checkpoint, the missing behavior is:

- Ratio-4 overlapping compressor state and compressed KV/score state.
- Ratio-4 Lightning Indexer query projection, scoring, and top-512 selection.
- Ratio-128 non-overlapping compressor state and causal compressed attention.
- Separate local SWA state and compressed/indexer state with persistent state
  across chunk boundaries.
- The layer-specific cache metadata and causal boundary rules needed for
  prefill and decode.
- The compressor/indexer weights that are present in the source tensor
  inventory but are not loaded by `V4Layer::init_with_loader`.

This is not a performance-only omission. Replacing CSA/HCA with sliding-window
attention changes which historical tokens can affect hidden states and logits.
The current 43-layer text turn proves that the implemented approximation is
stable enough to produce text; it does not prove that it is the selected model.

### 4.2 RoPE is only correct for the short, unscaled path

The pipeline initializes the RoPE table with factor `1.0`, even though the
checkpoint declares YaRN factor 16, original context 65,536, and a separate
compressed RoPE base of 160,000. The current implementation also has one
ordinary theta-10,000 table and no compressed-attention position transform.

This may be harmless for a narrow short-context smoke test, but it is not
correct at the advertised long context and is not sufficient for the
compressor/indexer path.

### 4.3 Prefill is a repeated decode loop

`generate()` calls `step()` once per prompt token. That preserves causal state
for the current sliding-only approximation, but it is not a batched prefill
implementation and it cannot correctly build compressor state independently
per chunk until the compressor state machine exists. It also leaves the
current expert transfer path in a single-token regime.

### 4.4 MTP and prefix caching are absent

The checkpoint contains MTP-related tensors and declares one next-token
prediction layer. The current pipeline uses only the base decoder and one LM
head. This does not prevent ordinary one-token generation, but it means the
engine is not feature-complete for the release's speculative/MTP serving
contract.

Prefix caching is not the immediate blocker. It should be added after the
per-layer compressed state is correct, because a prefix cache must preserve
local cache entries, compressor boundaries, compressed states, and indexer
metadata together. Adding it to the current single 512-wide cache would only
cache the approximation.

## 5. Hardware and scaling walls

### 5.1 Current one-GPU cache wall

The testbed has 23.98 GB per RX 7900 XTX. The converted dense backbone is
approximately 14.66 GiB before runtime allocations. The current full-resolution
KV cache costs:

```text
43 layers * context tokens * 512 values * 2 bytes
```

At the advertised 1,048,576-token context this is approximately 43.0 GiB,
before scratch buffers, RoPE tables, hot experts, or safety headroom. The
current cache design therefore cannot reach the advertised context on one
card. The memory-budget gate correctly rejects such a configuration; increasing
the allocation or changing the default would not solve it.

Implementing the model's compressed caches changes this conclusion. Local
128-token state plus ratio-4 and ratio-128 compressed state can fit in a much
smaller footprint, although the ratio-4 indexer and its metadata add real
memory and compute. This is a required architectural rewrite, not a reason to
discard the dense loader or expert pool.

### 5.2 Cold expert traffic wall

Each routed expert is 14.155776 MB. A token can request 258 routed experts
across 43 layers, or about 3.65 GB of expert payload if every request is cold.
The current hot pool is only roughly 660-700 experts after the dense backbone,
KV cache, scratch, and headroom are resident. A 64 GB host cannot hold all
155.25 GiB of routed payload in Warm memory either.

This creates a real single-GPU throughput ceiling for novel expert routes:

- NVMe bandwidth and scattered-file latency can dominate even when expert
  kernels are very fast.
- PCIe upload bandwidth can be overlapped, but it cannot be eliminated for
  experts that are not resident.
- More LRU slots help only until the route working set is covered. The current
  locality measurements already show that history-based speculative routing is
  close to the LRU ceiling on the measured workload.

This is a performance wall, not a correctness wall. The Hot/Warm/Cold design,
physical repacking, deadline-aware layer residency, batching, and eventually
multi-GPU placement can still make the system useful. It does mean that a
single-card, high-throughput, arbitrary-prompt target cannot be justified by
expert-kernel speed alone.

### 5.3 What RDNA3 does and does not block

The hardware has Wave32, FP16 WMMA, enough VRAM bandwidth for the dense path,
and validated asynchronous PCIe and direct-I/O mechanisms. There is no
fundamental RDNA3 instruction-set wall for:

- FP16 dense projections;
- symmetric INT4 dequantization into FP16/FP32 accumulation;
- mHC and Sinkhorn operations;
- compressor state updates;
- top-k/indexer logic implemented as custom Wave32 kernels.

The absence of native FP8/FP4 matrix acceleration is a performance concern,
but it is not a mismatch with this particular expert artifact: the routed
weights are INT4 and the dense backbone is mostly FP16. The harder hardware
problem is bandwidth and latency for sparse expert movement, plus the work of
implementing and tuning sparse attention without the CUDA FlashMLA kernels.

Four cards do not automatically remove the wall. The host has 96 GB aggregate
VRAM, but duplicating the 14.66 GiB dense backbone on every card consumes most
of that capacity. A useful multi-GPU design must shard layers or expert
residency deliberately and manage activation/P2P traffic; the current
single-GPU pipeline does not yet do that.

## 6. Decision table

| Question | Answer | Confidence |
| --- | --- | --- |
| Will the current routed INT4 artifact dequantize with the intended values? | Yes, for this checkpoint's symmetric pack-quantized INT4 contract. | High for tensor-level semantics; medium for full-model numerical parity. |
| Is the current converter lossless? | Yes for the packed tensors, scales, and dense source bytes it serializes. | High. |
| Does the current runtime execute the selected V4 architecture? | Only partially. MoE, mHC, routing, and base projection geometry are specialized; CSA/HCA/indexer attention is absent. | High. |
| Can the current sliding-only runtime be extended into a real V4 engine? | Yes. The dense loader, layer ownership, mHC path, expert registry, storage tiers, and kernel/toolchain work remain useful. | High. |
| Can the current one-card cache design support 1M tokens? | No. Not with a full-resolution 512-wide KV row for every layer. | High. |
| Does RDNA3 make the project impossible? | No. It makes sparse attention and cold expert service an engineering/performance challenge. | Medium-high. |
| Are current end-to-end generated texts proof of correctness? | No. They validate plumbing and the implemented approximation only. | High. |

## 7. Required gates before further optimization

1. **Independent quantization gate.** Compare real W1/W2/W3 dequantization and
   full output vectors against an independent reference implementation, then
   compare one complete routed FFN layer.
2. **Layer-class attention oracle.** Add deterministic CPU or reference-backed
   tests for layer 0 SWA, layer 2 CSA, and layer 3 HCA. Compare compressor
   state, local cache, selected indexer entries, attention output, and hidden
   state before optimizing HIP kernels.
3. **Config-driven layer specification.** Parse and validate the full
   `compress_ratios` schedule, compressed RoPE settings, indexer dimensions,
   and MTP metadata. Refuse a checkpoint when required tensors are absent
   instead of silently running sliding attention for all layers.
4. **Correct cache ownership.** Give each layer class local cache plus the
   required compressor/indexer state, with explicit causal boundary behavior
   across prefill chunks and decode steps.
5. **RoPE parity gate.** Apply the selected YaRN and compressed-RoPE rules and
   compare positions below and above the 65,536-token original context.
6. **Only then measure placement and cold-tier strategy.** Routing profiles,
   expert hotlists, and storage-layout conclusions are meaningful only after
   the route and attention outputs come from the correct model.

## Final assessment

The project has a sound systems direction and a credible path to a real
DeepSeek-V4-Flash runtime. The effort that should be protected is the
checkpoint-preserving expert format, unified residency system, mHC/routing
implementation, Wave32 kernel foundation, and hardware measurement discipline.

The effort that must be reclassified is the current full-model inference
result: it is a successful V4-shaped prototype, not yet a faithful V4-Flash
engine. The critical path is now clear. Implementing and validating
layer-specific CSA/HCA/indexer state is the condition that decides whether the
existing work becomes a real model runtime; it is not necessary to throw away
the storage or MoE work, and the hardware does not by itself make that next
step impossible.