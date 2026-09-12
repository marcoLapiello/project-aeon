# V4 Pipeline Modularization Analysis

*Status: historical pre-refactor analysis. The first three extraction and ownership steps are implemented; use [CODEBASE_MAP.md](../../status/CODEBASE_MAP.md) and [DOCUMENTATION_STATUS.md](../../status/DOCUMENTATION_STATUS.md) for the current structure.*

## Purpose

`src/architecture/deepseek_v4/core/v4_pipeline.hpp` was the original integration point for the DeepSeek-V4
runtime and had grown into a roughly 1,500-line header containing several distinct
ownership and execution concerns. This document records the duplication findings
and the outcome of the initial structural changes.

The goal is to reduce parallel implementations and clarify ownership without
changing inference behavior.

## Pre-refactor Responsibilities

Before the refactor, `v4_pipeline.hpp` contained:

- Four pipeline-local HIP utility kernels.
- `VRAMExpertSlot` and `HostExpertSource` data structures.
- `V4Layer`, including dense layer weights, KV cache, local expert sources,
  local VRAM LRU state, loading, and cleanup.
- `PipelineScratchBuffers`, including allocation and cleanup for the complete
  token execution arena.
- `V4Pipeline`, including model loading, model-level device resources, streams,
  memory-budget setup, expert-tier orchestration, one-token execution, greedy
  sampling, generation, and cleanup.

The current split is `src/architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp` for pipeline utility
kernels, `src/architecture/deepseek_v4/core/v4_pipeline_scratch.hpp` for scratch ownership,
`src/architecture/deepseek_v4/core/v4_layer.hpp` for layer-local structures,
`src/architecture/deepseek_v4/core/v4_model_resources.hpp` for model-level device resources, and
`src/architecture/deepseek_v4/core/v4_expert_supply.hpp` for transfer and prefetch coordination.
Production residency is handled by `UnifiedVRAMExpertPool`, `HostExpertPool`, and `ExpertRegistry`.

These were separate concerns even though they originally participated in one
header-only implementation; the current ownership split is described above.

## Duplication Findings

### Transformer block representation

There is substantial conceptual overlap between `V4Layer` in
`src/architecture/deepseek_v4/core/v4_pipeline.hpp` and the following types in
`src/architecture/deepseek_v4/core/v4_block.hpp`:

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

- `src/backend/swizzled_w4a16/core/vram_expert_pool.hpp` for the global hot VRAM pool.
- `src/infrastructure/core/host_expert_pool.hpp` for the warm host pool.
- `src/infrastructure/core/expert_registry.hpp` for global residency, tier transitions, and
  LRU tracking.

The global tier path is now the production path used by the `.aeon` pipeline.
The old layer-local cache remains only as compatibility/reference structure where
needed; new tiering work should extend the global ownership model rather than add
another cache implementation.

### Model initialization

The historical `init_aeon()` and `init_dynamic_global()` entry points repeated
model-resource setup and allowed callers to override the model layer count. They
were removed after this analysis: `V4Pipeline::initialize()` now loads the
architecture from `config.json`, derives the layer count, evaluates the runtime
policy, and constructs the single production Hot/Warm/Cold path.

### Hyper-Connections execution gap (resolved)

The original production `step()` path split HC execution across the CPU and
GPU. It copied the residual to the host, computed the RMS and 24-value
projection on the CPU, copied the mixes back, ran the existing GPU Sinkhorn
kernel, copied the four pre-mix values back, combined the four residual
streams on the CPU, and copied the resulting hidden vector to the GPU.

This was not evidence that CPU execution was the intended architecture. It was
an incremental implementation gap. The existing GPU HC kernels covered only:

- `hc_sinkhorn_normalize_kernel` for the 4x4 Sinkhorn normalization and
  pre/post/comb coefficient generation.
- `hc_post_kernel` for the post-expansion residual update.

The initial 16,384-to-24 projection and the pre-combination reduction did not
have production GPU kernels.

A device-side replacement with `hc_project_kernel` and `hc_pre_combine_kernel`
was then parallelized across 24 Wave32 blocks with vectorized loads/stores. The
result passed numerical tests and recovered approximately 123 tok/s on the
two-layer zero-miss benchmark. The HC CPU round trips are therefore no longer
an active blocker; the measurements are recorded in [PERFORMANCE_LEDGER.md](../../status/PERFORMANCE_LEDGER.md).

## Logic That Is Already Reused Correctly

The main attention, routing, and quantized GEMM algorithms are not duplicated
inside the pipeline. `step()` dispatches existing implementations from the
kernel headers:

- `kernel/v4_attention.hpp` for RMSNorm, RoPE, attention, projections, and the
  HC head.
- `kernel/hc_sinkhorn.hpp` for Sinkhorn and HC post-expansion operations.
- `kernel/moe_router.hpp` for MoE routing.
- `kernel/w4a16_gemm.hpp` for INT4-W4A16 WMMA GEMM.

The four pipeline-local utility kernels were small and legitimate, but did not
need to live in the pipeline header:

- Clamped SwiGLU.
- Weighted expert-output accumulation.
- FP16-to-FP32 conversion.
- FP32-to-FP16 conversion.

They now live in `src/architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp`.

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

## Modularization Outcome and Remaining Follow-up

The mechanical sequence was completed as follows:

1. [x] Extract the pipeline utility kernels into
   `src/architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp`.
2. [x] Extract `PipelineScratchBuffers` into
   `src/architecture/deepseek_v4/core/v4_pipeline_scratch.hpp`.
3. [x] Extract `VRAMExpertSlot`, `HostExpertSource`, and `V4Layer` into
   `src/architecture/deepseek_v4/core/v4_layer.hpp`.
4. [x] Standardize production tiering on the global VRAM pool and registry;
   remove the dual-cache split and hardcoded slot assumptions.
5. [ ] Consider a later model-resource initialization helper if repeated setup
   becomes a maintenance problem.
6. [ ] Revisit a deeper `step()` extraction only if it removes real complexity;
   the current header boundary is not an active correctness blocker.
7. [x] Keep `generate()` as the public orchestration API.
8. [ ] Revisit shared shape/metadata contracts with `v4_block.hpp` only when a
   concrete duplication affects an active change.

The extraction preserved public member names where practical because current
tests inspect `pipeline.layers` and `pipeline.expert_registry_` directly.

## Architectural Conclusion

`v4_pipeline.hpp` originally duplicated meaningful *responsibilities* from
other headers, but not the core GPU algorithms. The high-value cleanup is now
complete: utility kernels, scratch ownership, layer structure, global expert
residency, and HC synchronization have focused ownership boundaries. Further
splitting is optional maintenance work and should be justified by a concrete
change rather than treated as an unfinished phase.