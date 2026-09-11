# vLLM RDNA3 and DeepSeek-V4 Reference Analysis

**Date:** 2026-09-11
**Status:** Current reference map and implementation guidance
**Scope:** Local vLLM source inspection for AMD RDNA3/gfx1100 W4A16/GPTQ kernels and DeepSeek-V4 Flash prefill, attention, sparse MLA, indexer, and KV-cache paths. This document records source locations and applicability to Project Aeon. It does not add vLLM as an Aeon runtime dependency and does not investigate vLLM kernels outside the local checkout.

## Executive conclusion

The local vLLM checkout contains two useful but separate bodies of reference code:

```text
vLLM RDNA3 GPTQ/W4A16 kernels
    -> directly useful for Aeon GPTQ weight execution on RX 7900 XTX

vLLM DeepSeek-V4 ROCm attention and KV-cache paths
    -> useful for V4 attention semantics, cache metadata, prefill/decode structure,
       and sparse indexer behavior
```

The complete vLLM DeepSeek-V4 ROCm path is not a drop-in solution for the
current Aeon workload. The model-side vLLM implementation is organized around
DeepSeek-V4 FP8/MXFP4-era artifacts, an FP8-oriented `fp8_ds_mla` cache, and
ROCm/AITER or Triton sparse-attention paths. The native RDNA3 GPTQ kernels are
separate operations and are not selected by the DeepSeek-V4 model module.

The practical interpretation for Aeon is:

```text
Reuse directly or closely port:
  GPTQ packing, zero-point handling, group128 dequantization,
  RDNA3 dot2/WMMA geometry, and fused W4A16 MoE scheduling ideas.

Adapt carefully:
  expert pointers, residency, routed-expert batching, V4 Q/KV processing,
  sparse attention metadata, and cache insertion semantics.

Do not make foundational dependencies:
  AITER FP8/MXFP4 paths, gfx950-only optimizations, or generic vLLM
  paged attention as a substitute for DeepSeek-V4 MLA/CSA/HCA semantics.
```

The completed [Warm-tier repair and supply telemetry plan](../../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md)
is the current runtime baseline for persistent residency and source-tier
telemetry. The GPTQ work can use this document for offline format and kernel
reference work without changing that runtime path.

## 1. Reference checkout and evidence boundary

The inspected checkout is:

```text
/home/marcolap/aeon-references/vllm
```

Reference revision:

```text
94848ed 2026-09-09 20:37:10 +0800 [Bugfix] Fix unreachable None guard in Molmo2 get_candidate_target_fps (#55893)
```

The checkout was clean during inspection. No vLLM files were modified. The
findings below are source-based; no vLLM build, pytest run, or performance
benchmark was executed as part of this documentation pass.

The target hardware and execution constraints are:

```text
GPU:       AMD Radeon RX 7900 XTX
ISA:       gfx1100 / RDNA3
Wave mode: Wave32
Use case:  single GPU, batch-1 autoregressive generation with tiered experts
Constraint: no native FP8 or FP4 acceleration on this device
```

The exact vLLM revision matters because the checkout contains active hardware-
specific dispatch and model paths. Update the revision and re-audit this
document if the reference checkout changes materially.

## 2. vLLM architecture and build gates

### 2.1 ROCm architecture detection

The platform-level architecture selectors are in:

```text
/home/marcolap/aeon-references/vllm/vllm/platforms/rocm.py
```

Important symbols:

```text
_ON_GFX1100
_ON_GFX1X
_ON_RDNA
on_gfx1100()
on_gfx11()
on_gfx1x()
on_rdna()
use_rocm_custom_paged_attention()
```

The file also identifies the RX 7900 XTX PCI device ID:

```text
"0x744c": "AMD_Radeon_RX7900XTX"
```

The important distinction is between exact gfx1100 gating and broader gfx1x
gating:

- The native GPTQ RDNA3 extension is built and registered only for `gfx1100`.
- The hybrid skinny W4A16 path uses broader `gfx11/gfx12` checks.
- Some generic ROCm attention code uses `gfx11/gfx12`, but its supported head
  sizes and cache contract do not make it a DeepSeek-V4 implementation.

### 2.2 Native extension build and registration

The build rules are in:

```text
/home/marcolap/aeon-references/vllm/CMakeLists.txt
```

The relevant build block:

```text
VLLM_ROCM_HAS_GFX1100
VLLM_ROCM_GFX1100
csrc/rocm/q_gemm_rdna3.cu
csrc/rocm/q_gemm_rdna3_wmma.cu
csrc/rocm/moe_q_gemm_rdna3.cu
```

`VLLM_ROCM_GFX1100` is added when the selected GPU architecture list matches
`gfx1100`. The three RDNA3 GPTQ source files are then added to `_rocm_C`.

The public operation declarations and implementations are in:

```text
/home/marcolap/aeon-references/vllm/csrc/rocm/torch_bindings.cpp
/home/marcolap/aeon-references/vllm/csrc/rocm/ops.h
/home/marcolap/aeon-references/vllm/vllm/_custom_ops.py
```

The operations are:

```text
gptq_gemm_rdna3(...)
gptq_gemm_rdna3_wmma(...)
moe_gptq_gemm_rdna3(...)
paged_attention(...)
```

The GPTQ operation registrations are inside the `#ifdef VLLM_ROCM_GFX1100`
block. The generic `paged_attention` operation is registered separately.

## 3. RDNA3 GPTQ/W4A16 reference kernels

### 3.1 GPTQ dequantization and packing primitives

Primary source:

```text
/home/marcolap/aeon-references/vllm/csrc/rocm/qdq_4_rdna3.cuh
```

Important symbols:

```text
shuffle_4bit_8()
prep_zero_scale_fp16()
dequant_4bit_8_fp16()
prep_zero_scale_bf16()
dequant_4bit_8_bf16()
prep_zero_scale_bf16_f32()
dequant_4bit_8_bf16_f32()
```

The file documents the ExLlama/GPTQ nibble shuffle. An int32 containing eight
4-bit values is rearranged into an even/odd interleaved form so that one mask
can produce matching `half2` or BF16 pairs:

```text
source logical values: q0 q1 q2 q3 q4 q5 q6 q7
shuffled nibble order: q0 q2 q4 q6 q1 q3 q5 q7
```

The FP16 path uses the familiar `1024 + nibble` bit trick. The BF16 path cannot
use the same upper-nibble shift because BF16 has a narrower mantissa. vLLM
therefore uses a separate BF16 conversion path and, on RDNA3, often widens the
calculation to FP32 because gfx11 lacks a native packed BF16 FMA instruction.

The key implication for Aeon is that this source provides a proven RDNA3
implementation strategy for GPTQ group-wise dequantization, but it does not
establish the source checkpoint's zero-point convention automatically. The
checkpoint's `sym=true` metadata and physical `qzeros` tensors must still be
validated by an independent decoder.

### 3.2 Scalar/decode GPTQ GEMM

Primary source:

```text
/home/marcolap/aeon-references/vllm/csrc/rocm/q_gemm_rdna3.cu
```

Important symbols and regions:

```text
gemm_q4_kernel_rdna3()
launch_gemm_q4_for_mcount()
launch_gemm_q4()
gptq_gemm_rdna3()
```

The kernel's tuned geometry is:

```text
THREADS_X     = 256 threads = 8 Wave32 waves
BLOCK_KN_SIZE = 256 K elements
Each thread computes 4 output columns
```

The kernel splits K over `gridDim.z`, uses LDS for activation tiles when
needed, issues vectorized weight loads, dequantizes four int32 words, and
accumulates in FP32. The public launcher selects `M_COUNT` templates for
small matrices:

```text
M = 1       -> M_COUNT = 1
M = 2..3    -> M_COUNT = 2
M = 4..7    -> M_COUNT = 4
M >= 8      -> M_COUNT = 8, unless the WMMA entry point is selected first
```

The RDNA3 path writes FP16 or BF16 output through packed CAS-loop atomics when
K is split. This is a hardware-specific detail worth preserving conceptually:
gfx11 does not provide the native packed FP16/BF16 atomic add used by newer
AMD architectures, so vLLM uses `atomicCAS` on packed 32- or 64-bit words.

The public input contract is:

```text
a:            [M, K] half or bfloat16
b_q_weight:   [K/8, N] uint32, already shuffled
b_qzeros:     [groups, N/8] uint32, packed 4-bit zero points
b_scales:     [groups, N] half or bfloat16
output:       [M, N] same dtype as a
```

The kernel assumes sequential group traversal. It has no `g_idx` argument.
That is directly relevant to the GPTQ checkpoint: `desc_act=false` may imply a
sequential layout, but Aeon must inspect and prove the `g_idx` values before
adopting a kernel that does not consume them.

The wrapper checks that `N` is a multiple of 8 and derives the group count from
`b_qzeros`. The `use_v2_format` flag controls whether the GPTQv1 `+1`
zero-point offset is applied.

### 3.3 WMMA/prefill GPTQ GEMM

Primary source:

```text
/home/marcolap/aeon-references/vllm/csrc/rocm/q_gemm_rdna3_wmma.cu
```

Important symbols:

```text
compute_wmma_k_split()
compute_wmma_k_split_mn()
gemm_q4_wmma_kernel_16x16_1w()
gemm_q4_wmma_kernel_32x16_2w()
gemm_q4_wmma_kernel_64x16_4w()
gemm_q4_wmma_kernel_64x32_4w()
gemm_q4_wmma_kernel_64x64_4w()
gemm_q4_wmma_kernel_128x64_k16()
gemm_q4_wmma_kernel_128x64_k32()
launch_gemm_q4_wmma_64x64_4w()
gptq_gemm_rdna3_wmma()
```

The file deliberately isolates the WMMA translation unit from the scalar
kernel. Its comments record a prior hipcc interaction in which putting both
paths in one translation unit affected the scalar kernel's generated code.
That is a useful engineering warning for Aeon: keep decode and prefill kernel
translation units separable until the local compiler behavior is measured.

The path uses native RDNA3 Wave32 intrinsics:

```text
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32()
__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32()
```

The public `gptq_gemm_rdna3()` dispatches to WMMA for sufficiently large
prefill matrices. In the inspected source, the high-level choice is:

```text
BF16 M >= 16 -> WMMA path
FP16 M >= 64 -> WMMA path
otherwise    -> scalar path
```

The WMMA launcher then chooses progressively larger tiles as M grows. The
smallest path is one Wave32 for a 16x16 output tile; larger paths use two or
four waves and larger M/N tiles. K splitting increases occupancy and uses
packed CAS accumulation when multiple K partitions write the same output tile.

This is the strongest vLLM reference for a future Aeon true-prefill path. The
current Aeon generation loop is decode-shaped and generally operates at
`M=1`, so the scalar path is the first relevant target.

### 3.4 Fused RDNA3 GPTQ MoE

Primary source:

```text
/home/marcolap/aeon-references/vllm/csrc/rocm/moe_q_gemm_rdna3.cu
```

Important symbols:

```text
moe_gemm_q4_kernel_rdna3()
launch_moe_gemm_q4()
dispatch_moe_gemm_q4()
moe_gptq_gemm_rdna3()
```

The fused kernel combines:

```text
sorted token/expert assignments
    -> expert-specific W4A16 dequantized dot products
    -> optional top-k weighting
    -> direct output accumulation
```

Its input contract includes:

```text
a:                       [M, K] or [M*top_k, K]
c:                       pre-zeroed output
b_q_weight:              [E, K/8, N] uint32
b_scales:                [E, groups, N]
b_qzeros:                [E, groups, N/8]
sorted_token_ids:        aligned routing assignments
expert_ids:              expert per token block
num_tokens_post_padded:  aligned token count
block_size_m:            1, 2, 4, or 8
output_topk:             optional direct reduction to original token rows
```

The Python integration is:

```text
/home/marcolap/aeon-references/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna3.py
```

Important symbols:

```text
CompressedTensorsWNA16RDNA3MoEMethod
_rdna3_fused_moe()
```

The Python path uses `BLOCK_SIZE_M=1` for decode and `BLOCK_SIZE_M=4` for
small prefill. It performs a fused W1/Gate-Up operation, activation, and a W2
operation that can reduce the top-k expert results directly into the original
output rows.

The associated correctness test is:

```text
/home/marcolap/aeon-references/vllm/tests/kernels/quantization/test_rdna3_moe_w4a16.py
```

Important limitation for Aeon: vLLM stores all experts in contiguous device
arrays with shape `[E, K/8, N]`. Aeon's Hot/Warm/Cold design intentionally
keeps only a changing subset resident in VRAM. The vLLM kernel math and token
sorting are reusable, but its all-experts-resident pointer/stride assumption
must be replaced with either:

1. a gather step that places the six selected experts into a compact device
   workspace; or
2. an Aeon-specific pointer-bundle kernel that accepts six resident expert
   records.

### 3.5 Generic RDNA W4A16 hybrid path

The broader RDNA path is split across:

```text
/home/marcolap/aeon-references/vllm/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py
/home/marcolap/aeon-references/vllm/csrc/rocm/skinny_gemms_int4.cu
/home/marcolap/aeon-references/vllm/vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py
```

Important symbols:

```text
RDNAHybridW4A16LinearKernel
_rdna_hybrid_w4a16_apply_impl
triton_w4a16_skinny_fmt_gemm
wvSplitK_int4_g
wvSplitK_int4_hf_sml_
wvSplitK_int4_hf_
```

The hybrid dispatcher uses a HIP skinny kernel for small M and a Triton W4A16
kernel for larger M. It supports group sizes:

```text
32, 64, 128
```

The skinny implementation supports symmetric and asymmetric zero-point paths.
The Triton implementation uses an ExLlama-shuffled weight buffer and clamps
its K tile to the group size so a tile never crosses a scale-group boundary.

This path is a useful fallback and group128 validation reference. It is less
attractive as the first Aeon GPTQ target than the exact gfx1100 GPTQ kernel
because Aeon is specifically targeting `gfx1100`, already uses native HIP,
and needs to preserve tiered expert records rather than a PyTorch tensor
layout.

## 4. GPTQ contract comparison with Aeon

### 4.1 vLLM RDNA3 GPTQ contract

The vLLM native RDNA3 kernel expects an ExLlama-shuffled, output-column-packed
layout:

```text
qweight: [K/8, N] uint32
qzeros:  [groups, N/8] uint32
scales:  [groups, N] FP16 or BF16
```

The source-side Python transformation is in:

```text
/home/marcolap/aeon-references/vllm/vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py
```

Important transformation steps:

```text
permute_param_layout_()
ops.gptq_shuffle()
pack_quantized_values_into_int32()
```

### 4.2 GPTQ checkpoint under investigation

The local GPTQ checkpoint evidence is recorded in
[GPTQ_AEON_PARALLEL_BACKEND_ANALYSIS.md](GPTQ_AEON_PARALLEL_BACKEND_ANALYSIS.md).
Its relevant source shapes are:

```text
Gate/up qweight: [512, 2048] int32
Gate/up qzeros:  [32, 256] int32
Gate/up scales:  [32, 2048] float16
Gate/up g_idx:   [4096] int32

Down qweight:    [256, 4096] int32
Down qzeros:     [16, 512] int32
Down scales:     [16, 4096] float16
Down g_idx:      [2048] int32
```

The logical dimensions match vLLM's `[K/8, N]` qweight orientation for the
three V4 expert projections. The binary contract still requires independent
validation of:

- whether the checkpoint's qweight nibble order needs `gptq_shuffle`;
- whether qzeros use GPTQv1's stored-zero-plus-one convention;
- whether `sym=true` changes the interpretation despite physical qzeros;
- whether every `g_idx` is sequential for `desc_act=false`; and
- whether scale dtype and orientation remain uniform across dense and expert
  tensors.

The first Aeon GPTQ artifact should preserve all source planes, including
`g_idx`, even if the first kernel specialization proves that the values are
sequential. A descriptor can then mark a validated sequential specialization
without deleting the source evidence.

### 4.3 Current Aeon contract

The current Aeon backend is documented by:

```text
/home/marcolap/project-aeon/src/kernel/w4a16_gemm.hpp
/home/marcolap/project-aeon/src/kernel/aeon_w4a16_swizzle.hpp
/home/marcolap/project-aeon/src/kernel/aeon_w4a16_swizzled_gemv.hpp
/home/marcolap/project-aeon/src/core/aeon_loader.hpp
/home/marcolap/project-aeon/src/core/vram_expert_pool.hpp
```

It assumes:

```text
symmetric INT4
constant zero = 8
no qzeros plane
no g_idx plane
group size = 32
source-oriented [out, K/8] expert packed layout
fixed 14,155,776-byte expert payload
```

The current swizzled path is not a GPTQ decoder. Its nibble permutation and
Wave32 lane layout are an Aeon-specific group32 backend contract. The GPTQ
backend should therefore use a separate artifact version and descriptor rather
than teaching the current loader to guess between formats.

## 5. DeepSeek-V4 model and attention reference paths

### 5.1 Platform-specific model selection

The model selector is:

```text
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/__init__.py
```

On ROCm it imports:

```text
vllm.models.deepseek_v4.amd.model
vllm.models.deepseek_v4.amd.rocm
vllm.models.deepseek_v4.amd.mtp
```

The AMD model graph is in:

```text
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/amd/model.py
```

Important symbols:

```text
DeepseekV4MLP
DeepseekV4MoE
DeepseekV4DecoderLayer
DeepseekV4ForCausalLM
```

The AMD layer uses `DeepseekV4ROCMAiterMLAAttention` and the vLLM fused MoE
layer. It also contains AMD-specific preparation hooks such as weight
preshuffling and AITER dispatch.

### 5.2 Why the complete vLLM V4 quantization path is not Aeon's target

The V4 quantization configuration is:

```text
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/quant_config.py
```

Important symbol:

```text
DeepseekV4FP8Config
```

The file describes DeepSeek-V4 checkpoints in terms of FP8 block-quantized
dense/attention weights and `expert_dtype` values of `fp4` or `fp8`. Its FP4
expert path is MXFP4-oriented, not the GPTQ integer group128 contract under
investigation.

This is the critical separation:

```text
vLLM DeepSeek-V4 model path: FP8/MXFP4 model contracts
vLLM RDNA3 GPTQ path:        GPTQ INT4 W4A16 contracts
```

The RDNA3 GPTQ operations are not automatically selected by the AMD V4 model
module. Aeon must compose the V4 graph semantics with its own GPTQ backend.

## 6. DeepSeek-V4 Q/KV preparation and cache insertion

### 6.1 Base V4 attention layer

The common attention implementation is:

```text
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/attention.py
```

Important symbols:

```text
_resolve_dsv4_kv_cache_dtype()
DeepseekV4Attention
DeepseekV4Attention.forward()
DeepseekV4Attention._fused_qnorm_rope_kv_insert()
```

The base layer owns the V4 geometry and cache selection. It separates:

```text
448 NoPE dimensions
64 RoPE dimensions
head_dim = 512
```

It also handles the distinction between compressed layers and sliding-window
only layers through `compress_ratio` and the associated cache modules.

### 6.2 Native fused Q-norm/RoPE/KV insertion

The native fused source is:

```text
/home/marcolap/aeon-references/vllm/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu
```

Important symbols:

```text
fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel()
fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid()
fusedDeepseekV4FullCacheKernel()
fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert()
fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert()
fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert()
```

The corresponding operation registration is in:

```text
/home/marcolap/aeon-references/vllm/csrc/libtorch_stable/torch_bindings.cpp
```

The fused operation combines some of the following work depending on cache
mode:

```text
Q RMSNorm
GPT-J/interleaved RoPE on the final 64 dimensions
FP8 quantization of the 448 NoPE dimensions
BF16 storage of the 64 RoPE dimensions
paged cache insertion
```

The kernel is a useful reference for operation fusion and exact V4 RoPE/cache
boundaries. Its FP8 path is not evidence that RX 7900 XTX has native FP8
acceleration.

### 6.3 DeepSeek-V4 cache layout and quantization helpers

The cache metadata and cache tensor class are in:

```text
/home/marcolap/aeon-references/vllm/vllm/v1/attention/backends/mla/sparse_swa.py
```

Important symbols:

```text
DeepseekV4SWACache
DeepseekSparseSWABackend
DeepseekSparseSWAMetadata
DeepseekSparseSWAMetadataBuilder
```

The V4 SWA cache uses a block size of 64 tokens in the inspected path. For the
`fp8_ds_mla` layout, each token has:

```text
448 FP8 NoPE values
128 bytes of BF16 RoPE values
8 UE8M0 scale bytes
584 bytes total logical token state
```

The quantization and gather helpers are in:

```text
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py
```

Important symbols:

```text
quantize_and_insert_k_kernel()
quantize_and_insert_k_cache()
DequantizeAndGatherKCacheKernel
 dequantize_and_gather_k_cache()
combine_topk_swa_indices()
```

The cache helper documents the paged layout as:

```text
block data:
  token data: 448 FP8 bytes + 128 BF16 bytes per token
  scale area: 8 uint8 scale bytes per token
  block padding/alignment as determined by the cache stride
```

The `DequantizeAndGatherKCacheKernel` reconstructs BF16 K rows for sparse
prefill. This is valuable for modeling cache bandwidth and metadata movement,
but Aeon should not make FP8 cache storage a prerequisite for its first GPTQ
backend.

### 6.4 Indexer query preparation

The DeepSeek-V4 sparse indexer sources are:

```text
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
/home/marcolap/aeon-references/vllm/vllm/v1/attention/backends/mla/indexer.py
```

Important symbols include:

```text
FusedIndexerQRopeQuantTritonKernel
FusedIndexerQRopeMxFp4TritonKernel
DeepseekV4IndexerBackend
BuildPrefillChunkMetadataKernel
build_c128a_topk_metadata()
```

The indexer path applies RoPE to index queries, quantizes them for the index
scoring path, and builds top-k metadata for compressed cache entries. The
MXFP4 indexer path is gated to Blackwell-class capabilities and is not relevant
to RX 7900 XTX. The FP8/triton structure remains useful as an algorithmic
reference for a future native or software-emulated Aeon indexer.

## 7. ROCm DeepSeek-V4 prefill and decode paths

### 7.1 AMD attention wrapper

The AMD wrapper is:

```text
/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/amd/rocm.py
```

Important symbols:

```text
DeepseekV4ROCMAiterMLASparseBackend
DeepseekV4ROCMAiterMLAAttention
DeepseekV4ROCMAiterMLASparseMetadataBuilder
DeepseekV4ROCMAiterSparseSWAMetadataBuilder
DeepseekV4ROCMAiterMLAAttention.forward_mqa()
DeepseekV4ROCMAiterMLAAttention._forward_prefill()
DeepseekV4ROCMAiterMLAAttention._forward_decode()
```

### 7.2 Prefill sequence

`_forward_prefill()` in `amd/rocm.py` performs the following high-level work:

```text
1. Gather/dequantize compressed K rows when the layer has a compressed cache.
2. Gather/dequantize the sliding-window K rows.
3. Combine compressed top-k indices with the local window indices.
4. Process requests in bounded prefill chunks.
5. Execute sparse attention over ragged per-query indices.
```

The ROCm sparse attention operation wrappers are:

```text
/home/marcolap/aeon-references/vllm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
```

Important symbols:

```text
rocm_sparse_attn_prefill()
_rocm_sparse_attn_prefill_ragged_triton()
_rocm_sparse_attn_prefill_triton()
_sparse_attn_prefill_ragged_kernel()
```

The core prefill Triton kernel is `_sparse_attn_prefill_ragged_kernel()`. It
loads a query, walks a ragged list of selected KV slots, computes scaled
Q/K scores, applies online softmax, supports an attention sink, and accumulates
selected K/V rows.

This is the most useful prefill reference for V4 semantics. It is not a native
RDNA3 HIP kernel and should not be copied as an assumption that Triton sparse
MLA is already optimized for gfx1100.

### 7.3 Decode sequence

`_forward_decode()` in `amd/rocm.py`:

```text
1. Maps local top-k compressed indices to global paged-cache slots.
2. Builds ragged compressed and local-window index arrays.
3. Calls rocm_sparse_attn_decode().
```

The decode wrappers and kernels are:

```text
rocm_sparse_attn_decode()
_rocm_sparse_attn_decode_ragged_triton()
_rocm_sparse_attn_decode_triton()
_sparse_attn_decode_ragged_kernel()
_sparse_attn_decode_partial_kernel()
_sparse_attn_decode_reduce_kernel()
```

The source location for all of these is:

```text
/home/marcolap/aeon-references/vllm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
```

The generic `_sparse_attn_decode_ragged_kernel()` is the fallback for
architectures that are not gfx942 or gfx950. The split-K partial/reduce path
is selected only for those tuned CDNA architectures; the inspected gfx1100
path should therefore be treated as a correctness reference and a baseline
algorithm, not as a proven high-performance RX 7900 XTX implementation.

The decode kernel separately accumulates the 448 NoPE and 64 RoPE dimensions,
reads the paged FP8/BF16 cache layout, applies online softmax, and supports an
attention sink. This separation is important when mapping it to Aeon's current
single 512-wide FP16 cache.

## 8. Generic ROCm paged attention: useful boundary, wrong V4 substitute

The generic ROCm paged-attention implementation is:

```text
/home/marcolap/aeon-references/vllm/csrc/rocm/attention.cu
```

Important symbols:

```text
paged_attention_ll4mi_QKV_mfma16_kernel()
paged_attention_ll4mi_QKV_mfma4_kernel()
paged_attention_ll4mi_reduce_kernel()
paged_attention_custom_launcher()
paged_attention_custom_launcher_navi()
paged_attention()
```

The source has explicit Navi/RDNA handling. `paged_attention_custom_launcher_navi()`
contains the note that Navi does not support FP8 in this path. The generic
launcher supports head sizes 64 and 128 and ordinary paged K/V layouts.

This is not the correct kernel family for DeepSeek-V4 Flash because:

```text
DeepSeek-V4 latent head dimension: 512
DeepSeek-V4 cache policy: SWA plus ratio-4/ratio-128 compressed state
DeepSeek-V4 attention: sparse MLA with indexer/top-k metadata
```

Do not replace V4 sparse MLA with this generic paged attention implementation.
The useful parts are its Wave32/MFMA organization, partitioned softmax, block
tables, and reduction patterns for a future non-sparse cache experiment.

## 9. Direct source map

This table is intended as a quick navigation index for future work. Paths are
absolute because the vLLM checkout is a sibling reference repository rather
than part of the Aeon workspace.

| Area | Source file | Symbols or contents | Aeon use |
| --- | --- | --- | --- |
| gfx detection | `/home/marcolap/aeon-references/vllm/vllm/platforms/rocm.py` | `_ON_GFX1100`, `on_gfx1100`, `on_gfx1x` | Reproduce explicit gfx1100 capability checks in offline/runtime selection. |
| build gate | `/home/marcolap/aeon-references/vllm/CMakeLists.txt` | `VLLM_ROCM_GFX1100`, RDNA3 source list | Verify which source is compiled for gfx1100. |
| operation registration | `/home/marcolap/aeon-references/vllm/csrc/rocm/torch_bindings.cpp` | GPTQ and MoE op schemas | Identify public tensor contracts. |
| operation declarations | `/home/marcolap/aeon-references/vllm/csrc/rocm/ops.h` | `gptq_gemm_rdna3`, `moe_gptq_gemm_rdna3` | Mirror API boundaries in an Aeon-specific backend, not PyTorch bindings. |
| GPTQ dequant | `/home/marcolap/aeon-references/vllm/csrc/rocm/qdq_4_rdna3.cuh` | shuffle, zero/scale prep, FP16/BF16 dequant | First device-side GPTQ reference. |
| decode GPTQ | `/home/marcolap/aeon-references/vllm/csrc/rocm/q_gemm_rdna3.cu` | scalar W4A16, M_COUNT, dot2, CAS accumulation | Highest priority kernel reference for Aeon M=1. |
| prefill GPTQ | `/home/marcolap/aeon-references/vllm/csrc/rocm/q_gemm_rdna3_wmma.cu` | Wave32 WMMA tiles and K splitting | Future M>=16 prefill path. |
| fused GPTQ MoE | `/home/marcolap/aeon-references/vllm/csrc/rocm/moe_q_gemm_rdna3.cu` | sorted experts, fused W4A16, output reduction | Future batched/resident-expert path. |
| skinny INT4 | `/home/marcolap/aeon-references/vllm/csrc/rocm/skinny_gemms_int4.cu` | gfx1x small-M group32/64/128 kernels | Alternative decode/prefill fallback and group-size reference. |
| RDNA3 linear dispatch | `/home/marcolap/aeon-references/vllm/vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py` | `RDNA3W4A16LinearKernel` | Weight transformation and capability validation. |
| RDNA hybrid dispatch | `/home/marcolap/aeon-references/vllm/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py` | small-M HIP vs larger-M Triton | Batch-regime reference. |
| fused MoE Python | `/home/marcolap/aeon-references/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna3.py` | `CompressedTensorsWNA16RDNA3MoEMethod` | Routing and fused two-pass MoE orchestration. |
| GPTQ linear tests | `/home/marcolap/aeon-references/vllm/tests/kernels/quantization/test_rdna3_w4a16.py` | real format construction and CPU reference | Correctness-test template. |
| GPTQ MoE tests | `/home/marcolap/aeon-references/vllm/tests/kernels/quantization/test_rdna3_moe_w4a16.py` | dense-vs-fused and top-k reduction | Fused MoE test template. |
| V4 model selector | `/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/__init__.py` | ROCm AMD module selection | Locate platform-specific V4 entry point. |
| V4 AMD model | `/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/amd/model.py` | decoder layer, MoE, weight preparation | V4 graph and routing reference. |
| V4 quant config | `/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/quant_config.py` | `DeepseekV4FP8Config` | Explicitly identify FP8/MXFP4 mismatch. |
| V4 AMD attention wrapper | `/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/amd/rocm.py` | prefill/decode, cache metadata, ROCm dispatch | Attention orchestration reference. |
| V4 base attention | `/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/attention.py` | cache dtype, qnorm/RoPE insertion | Shared V4 attention semantics. |
| V4 fused Q/KV insert | `/home/marcolap/aeon-references/vllm/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | qnorm, RoPE, cache writes | Kernel fusion and cache-boundary reference. |
| V4 cache helpers | `/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py` | quantize/insert and gather/dequantize | Paged cache metadata and prefill gather reference. |
| V4 indexer quant | `/home/marcolap/aeon-references/vllm/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | index query RoPE/quantization | Future CSA indexer reference; MXFP4 branch is not RX7900XTX-targeted. |
| V4 indexer metadata | `/home/marcolap/aeon-references/vllm/vllm/v1/attention/backends/mla/indexer.py` | C4/C128 metadata and slot mapping | Sparse metadata reference. |
| V4 SWA metadata | `/home/marcolap/aeon-references/vllm/vllm/v1/attention/backends/mla/sparse_swa.py` | block size, layer types, prefill/decode metadata | Per-layer cache scheduling reference. |
| ROCm sparse attention | `/home/marcolap/aeon-references/vllm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | ragged prefill/decode kernels | V4 attention algorithm and fallback path. |
| generic paged attention | `/home/marcolap/aeon-references/vllm/csrc/rocm/attention.cu` | MFMA paged attention and Navi launcher | Low-priority generic reference; not V4 sparse MLA. |
| fused operation registration | `/home/marcolap/aeon-references/vllm/csrc/libtorch_stable/torch_bindings.cpp` | V4 qnorm/cache operation schemas | Trace native op registration. |

## 10. Reuse, adaptation, and exclusion matrix

| Reference | Decision | Reason |
| --- | --- | --- |
| `qdq_4_rdna3.cuh` | Reuse conceptually, port selectively | Correct RDNA3 GPTQ nibble/dequant primitives; validate zero semantics first. |
| `q_gemm_rdna3.cu` | Highest-priority kernel reference | It directly targets gfx1100 W4A16 GPTQ decode and FP16/BF16 activations. |
| `q_gemm_rdna3_wmma.cu` | Later adaptation | Relevant to true prefill; separate translation-unit strategy is worth preserving. |
| `moe_q_gemm_rdna3.cu` | Adapt around residency | Its fused routing/reduction is valuable, but its all-experts-contiguous layout conflicts with Hot/Warm/Cold. |
| `skinny_gemms_int4.cu` | Fallback/reference | Supports group128 and small M, but the exact GPTQ RDNA3 path is a cleaner first target. |
| `rdna3_w4a16.py` | Use as format/validation reference | Shows GPTQ shuffle, qzeros synthesis, capability checks, and testable shape constraints. |
| `rdna_hybrid_w4a16.py` | Use for regime comparison | Demonstrates decode-versus-prefill dispatch and group-size handling. |
| `test_rdna3_w4a16.py` | Reproduce test structure in C++ | It compares the GPU result against an independent FP32 dequantized reference. |
| `test_rdna3_moe_w4a16.py` | Reproduce fused-MoE gates | It checks dense-vs-fused parity and direct top-k reduction. |
| V4 AMD `model.py` | Reuse V4 graph concepts | The model graph, routing, and mHC organization are useful; quantization backend is different. |
| V4 AMD `rocm.py` | Adapt semantics/metadata | Prefill/decode and compressed/local cache separation are useful; gfx950 tuning is not a gfx1100 result. |
| `rocm_aiter_mla_sparse.py` | Algorithmic reference only | Triton ragged sparse attention is useful for correctness; performance is not established on gfx1100. |
| `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | Adapt selectively | Q/KV/RoPE fusion is useful, but its FP8-DS-MLA cache format is not the first Aeon cache target. |
| `attention.cu` | Do not use as V4 replacement | Generic paged attention has different head/cache assumptions and explicitly lacks Navi FP8 support. |
| AITER/FlyDSL/MXFP4 paths | Exclude from first RX7900XTX plan | They depend on FP8/MXFP4 contracts or newer gfx950/Blackwell capabilities. |

## 11. Recommended Aeon implementation sequence

### Stage 1: independent GPTQ reference

Use the existing GPTQ analysis and the vLLM tests as the source map for:

1. inventorying all qweight/qzeros/scales/g_idx tensors;
2. checking whether `g_idx` is sequential for representative dense and expert
   matrices;
3. resolving GPTQv1 versus GPTQv2 zero-point behavior;
4. implementing an independent CPU dequantizer; and
5. comparing real gate, up, and down projection outputs.

The CPU reference must not call vLLM or reuse the GPU bit trick. It should
operate directly on the source Safetensors planes.

### Stage 2: GPTQ-preserving artifact

Create a separate converter and manifest that preserves the source GPTQ planes
and records:

```text
backend = gptq_w4a16
bits = 4
group_size = 128
qweight/qzeros/scales/g_idx descriptors
source tensor names and shapes
zero-point convention
packing convention
validated g_idx mode
```

The expert supply path should treat each GPTQ expert record as an opaque
sector-aligned payload. The current `AEON_EXPERT_BYTES` and six-plane pool
views must not be reused by byte-count inference.

### Stage 3: native RDNA3 scalar backend

Implement one correctness-first native HIP projection for `M=1`:

```text
activation FP16
qweight [K/8,N] GPTQ packed data
qzeros [G,N/8]
scales [G,N]
FP32 accumulation
FP16 output
```

Use the vLLM RDNA3 scalar kernel as a reference for:

- Wave32 block geometry;
- `__builtin_amdgcn_fdot2` use;
- group128 transitions;
- packed qzero reads; and
- K-split output accumulation.

The first Aeon kernel should be independently tested on real GPTQ planes
before it is connected to tiered expert streaming.

### Stage 4: V4 projection and MoE integration

Add attention/shared-expert GPTQ bindings only after routed projection parity
passes. Then adapt the fused MoE structure around Aeon residency:

```text
router selects six expert IDs
    -> leases/supply path resolves six resident records
    -> compact pointer bundle or gather workspace
    -> fused W1/W3 and activation
    -> fused W2 and weighted accumulation
```

Do not make the kernel responsible for NVMe reads, Warm ownership, or cache
state transitions. Those remain owned by the generic supply layer described in
[GPTQ_AEON_PARALLEL_BACKEND_ANALYSIS.md](GPTQ_AEON_PARALLEL_BACKEND_ANALYSIS.md).

### Stage 5: V4 attention correctness

Use the vLLM V4 paths to define the missing Aeon attention contracts:

1. per-layer compression ratio and layer class;
2. local sliding-window cache;
3. ratio-4 compressed state and indexer metadata;
4. ratio-128 compressed state;
5. prefill chunk state continuity;
6. decode ragged selection and attention-sink behavior; and
7. BF16/FP16 cache policy for RX 7900 XTX.

Keep the first cache implementation in BF16/FP16 unless measurement shows
software FP8 conversion is worth the added format and numerical complexity.

### Stage 6: controlled comparison

Compare the current group32 backend and GPTQ group128 backend with identical:

```text
formatted token IDs
routing and layer schedule
Hot/Warm capacities and initial residency
KV-cache policy
hardware and stream configuration
```

Report projection error, layer/logit error, Hot/Warm/Cold traffic, exposed
supply wait, kernel time, TTFT, and decode throughput separately.

## 12. Correctness gates and open risks

The vLLM source map does not close any of these gates:

1. **GPTQ planes:** qweight nibble order, qzeros convention, scales, and g_idx
   must match an independent CPU reference.
2. **Group128:** the source checkpoint must be executed as group128; conversion
   to group32 is only a compatibility fallback.
3. **BF16 policy:** vLLM has separate FP16 and BF16 device paths. Aeon must
   choose and measure its dense non-GPTQ dtype policy explicitly.
4. **Residency:** the vLLM fused MoE kernels assume contiguous device expert
   tensors. Aeon must preserve leases and publication state while supplying
   only the selected experts.
5. **Prefill:** the vLLM WMMA path is a matrix path; Aeon's current prompt loop
   is repeated single-token execution and is not evidence of true prefill.
6. **V4 attention:** generic sliding-window or ordinary paged attention is not
   equivalent to the selected ratio-4/ratio-128 sparse MLA schedule.
7. **KV cache:** FP8-DS-MLA storage is a vLLM format choice, not a hardware
   capability of the RX 7900 XTX.
8. **Performance attribution:** vLLM source structure gives no Aeon end-to-end
   result. Kernel, PCIe, NVMe, Warm, and attention costs must be measured in
   the native runtime.

The current reference tests to mirror are:

```text
/home/marcolap/aeon-references/vllm/tests/kernels/quantization/test_rdna3_w4a16.py
/home/marcolap/aeon-references/vllm/tests/kernels/quantization/test_rdna3_moe_w4a16.py
/home/marcolap/aeon-references/vllm/tests/kernels/quantization/test_rdna_hybrid_w4a16.py
/home/marcolap/aeon-references/vllm/tests/kernels/attention/test_rocm_triton_attn_dsv4.py
/home/marcolap/aeon-references/vllm/tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py
```

They are references for test shape and semantics, not dependencies or evidence
that the complete vLLM path is optimized for Aeon's hardware/workload.

## Final assessment

The local vLLM checkout gives Aeon a strong answer to the kernel question:

```text
Yes: there is a native RDNA3/gfx1100 GPTQ W4A16 path worth using as a reference.
Yes: there is a fused RDNA3 GPTQ MoE path worth adapting.
Yes: there are detailed V4 prefill, sparse attention, indexer, and KV-cache
     implementations worth mining for semantics and metadata.
No: the complete vLLM DeepSeek-V4 FP8/MXFP4 ROCm path is not the first target
    for an RX 7900 XTX GPTQ deployment.
No: generic ROCm paged attention should not replace V4 sparse MLA.
```

The durable architecture remains:

```text
V4 graph and attention semantics: shared and model-specific
GPTQ/current W4A16 storage and kernels: specialized backends
Hot/Warm/Cold supply, leases, I/O, and telemetry: generalized shared layer
```

The immediate implementation target is an independent CPU GPTQ decoder and a
GPTQ-preserving group128 artifact. Native RDNA3 M=1 projection correctness
comes next; fused resident-expert MoE and true prefill follow after that.
