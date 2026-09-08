# V4 Pipeline Modularization Analysis

## Purpose

`src/core/v4_pipeline.hpp` is the current integration point for the DeepSeek-V4
runtime, but it has grown into a 1,500-line header containing several distinct
ownership and execution concerns. This document records the duplication and
modularization findings before structural changes are made.

The goal is to reduce parallel implementations and clarify ownership without
changing inference behavior.

## Current Responsibilities

`v4_pipeline.hpp` currently contains:

- Four pipeline-local HIP utility kernels.
- `VRAMExpertSlot` and `HostExpertSource` data structures.
- `V4Layer`, including dense layer weights, KV cache, local expert sources,
  local VRAM LRU state, loading, and cleanup.
- `PipelineScratchBuffers`, including allocation and cleanup for the complete
  token execution arena.
- `V4Pipeline`, including model loading, model-level device resources, streams,
  memory-budget setup, expert-tier orchestration, one-token execution, greedy
  sampling, generation, and cleanup.

These are separate concerns even though they currently participate in one
header-only implementation.

## Duplication Findings

### Transformer block representation

There is substantial conceptual overlap between `V4Layer` in
`src/core/v4_pipeline.hpp` and the following types in
`src/core/v4_block.hpp`:

- `DeepSeekV4BlockWeights`
- `DeepSeekV4BlockDeviceContext`
- `cpu_v4_block_forward`

Both paths describe the same attention and FFN topology, including:

- Hyper-Connections weights.
- Attention and FFN normalization weights.
- MLA projections (`wq_a`, `wq_b`, `wkv`, `wo_a`, and `wo_b`).
- Attention sink values.
- RoPE resources.
- Intermediate activation buffers.

This is not currently a direct implementation duplicate. `v4_block.hpp` is a
CPU/reference-oriented path, while `V4Layer` is part of the production
token-at-a-time GPU pipeline and also owns KV-cache and expert-tier state.
However, the model block is represented twice, which creates a maintenance
risk when tensor names, dimensions, or block behavior change.

Recommended direction: define shared logical block metadata or weight naming
constants, while keeping separate CPU-reference and GPU-runtime resource
representations. Do not merge the two execution paths prematurely.

### Scratch-buffer ownership

`PipelineScratchBuffers` in `v4_pipeline.hpp` and
`DeepSeekV4BlockDeviceContext` in `v4_block.hpp` independently allocate many
of the same categories of activation buffers:

- `d_x_pre` and `d_x_norm`.
- `d_qa`, `d_qa_norm`, `d_q`, and `d_kv`.
- `d_kv_norm_act`, `d_attn_out`, `d_z_lora`, and `d_attn_proj`.
- `d_ffn_pre` and `d_ffn_norm_act`.

The layouts are not identical. The current pipeline uses a token-at-a-time
arena padded to 16 rows for WMMA compatibility, while the block context is a
more general batched context. The duplication is therefore conceptual and
ownership-related rather than safely mergeable by simple renaming.

Recommended direction: establish shared shape/constants and separate scratch
types for reference and production execution. Extract the pipeline scratch
type from `v4_pipeline.hpp` before attempting deeper unification.

### Expert residency and caching

`V4Layer` contains a complete local expert-cache mechanism:

- `VRAMExpertSlot`.
- `free_slots_`.
- `lru_list_`.
- `lru_map_`.
- `acquire_expert_slot()`.
- Host/mmap expert pointer binding.
- Host-to-device expert streaming.

The same general responsibility is implemented by the production tier
components:

- `src/core/vram_expert_pool.hpp` for the global hot VRAM pool.
- `src/core/host_expert_pool.hpp` for the warm host pool.
- `src/core/expert_registry.hpp` for global residency, tier transitions, and
  LRU tracking.

The local path and global path are currently intentional modes, but they have
parallel concepts for slot ownership, streaming, residency, eviction, and
statistics. This is the most important duplication to resolve before adding
asynchronous prefetching: two cache implementations will otherwise require
two separate prefetch and correctness paths.

Recommended direction: define a common expert-device view and residency
manager boundary. The local per-layer cache and the global multi-layer pool can
remain separate implementations behind that boundary until the global path is
complete.

### Model initialization

`V4Pipeline::init()`, `init_aeon()`, and `init_dynamic_global()` repeat most of
the following work:

- HIP stream creation.
- RoPE table initialization and device upload.
- Embedding binding.
- LM-head allocation and upload.
- Hyper-Connections head allocation and upload.
- Final normalization allocation and upload.
- Scratch allocation.
- Layer construction.

The model-source differences and global-pool setup are real, but the shared
resource setup should not be copied across three initialization functions.

Recommended direction: introduce a loader/resource initialization helper that
accepts a model-source abstraction or narrowly shared loader operations. Keep
the three public initialization entry points for compatibility.

### Hyper-Connections pre-mix calculation

The production `step()` method manually performs host-side Hyper-Connections
projection work before launching the shared Sinkhorn kernel:

1. Copy the residual from device to host.
2. Compute RMS on the CPU.
3. Compute the projection against the HC function matrix on the CPU.
4. Copy the mixes back to the device.
5. Launch `hc_sinkhorn_normalize_kernel`.

The CPU reference path in `v4_block.hpp` uses the shared helper
`kernel::cpu_sinkhorn_and_mix` from `kernel/hc_sinkhorn.hpp` instead. This is
not a byte-for-byte duplicate, but the algorithmic responsibility is split
between two implementations.

This should be treated separately from the first mechanical extraction. The
host round trips are also a likely performance bottleneck and should be removed
only with a correctness check against the CPU reference.

## Logic That Is Already Reused Correctly

The main attention, routing, and quantized GEMM algorithms are not duplicated
inside the pipeline. `step()` dispatches existing implementations from the
kernel headers:

- `kernel/v4_attention.hpp` for RMSNorm, RoPE, attention, projections, and the
  HC head.
- `kernel/hc_sinkhorn.hpp` for Sinkhorn and HC post-expansion operations.
- `kernel/moe_router.hpp` for MoE routing.
- `kernel/w4a16_gemm.hpp` for INT4-W4A16 WMMA GEMM.

The four pipeline-local utility kernels are small and legitimate, but they do
not need to live in the pipeline header:

- Clamped SwiGLU.
- Weighted expert-output accumulation.
- FP16-to-FP32 conversion.
- FP32-to-FP16 conversion.

They should move to a focused kernel header such as
`src/kernel/v4_pipeline_ops.hpp`.

## Resource-Management Duplication

HIP resources are manually allocated and released in several independent
types:

- `V4Layer::free()`.
- `PipelineScratchBuffers::free()`.
- `V4Pipeline::free_all()`.
- `GlobalVRAMExpertPool::free()`.
- `HostExpertPool::free()`.
- `DeepSeekV4BlockDeviceContext::free()`.

`CHECK_HIP` is also defined in both `v4_pipeline.hpp` and
`vram_expert_pool.hpp`. The production expert-pool classes already have clearer
move-only ownership semantics than `V4Layer` and `PipelineScratchBuffers`.

Recommended direction: make extracted resource owners consistently move-only
RAII types, but avoid broad error-handling changes during the first split.

## Proposed Modularization Order

The safest sequence is mechanical first, architectural second:

1. Extract the four pipeline utility kernels into
   `src/kernel/v4_pipeline_ops.hpp`.
2. Extract `PipelineScratchBuffers` into
   `src/core/v4_pipeline_scratch.hpp`.
3. Extract `VRAMExpertSlot`, `HostExpertSource`, and `V4Layer` into
   `src/core/v4_layer.hpp`.
4. Preserve the existing local-cache fallback while defining the common expert
   residency boundary needed by the global path.
5. Extract model-level resources and shared initialization helpers.
6. Extract the `step()` implementation into a focused execution component.
7. Keep `generate()` as the public orchestration API until the execution split
   is stable.
8. Revisit `v4_block.hpp` after the production types have stable ownership and
   shape contracts.

Each early step should preserve public member names where practical because
current tests inspect `pipeline.layers` and `pipeline.expert_registry_`
directly.

## Architectural Conclusion

`v4_pipeline.hpp` is duplicating meaningful *responsibilities* from other
headers, but it is not generally duplicating the core GPU algorithms. The main
problems are parallel representations of a transformer block, two expert-cache
architectures, repeated initialization, and repeated manual resource cleanup.

The correct next move is therefore modularization with ownership boundaries,
not a broad rewrite. The first extraction should be behavior-neutral and
focused on moving types and utility kernels. Expert residency unification and
the removal of host-side HC synchronization should follow only after targeted
validation is in place.