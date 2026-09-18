# Project Aeon Codebase Map

> **Rewrite note (2026-09-18).** The DeepSeek-V4 graph rewrite on branch `rewrite/graph-v2` has
> landed: P1–P4 are green and the engine runs the model end-to-end (text in, text out). The
> **pre-rewrite graph was deleted on 2026-09-18** — `core/v4_pipeline.hpp`,
> `reference/v4_int4_reference.hpp`, `reference/v4_attention_oracle.hpp`, the superseded
> `text/dsv4_chat_formatter.*`, both gated parity tests, `tools/profile_routing.cpp` and
> `tools/record_v4_gpu_evidence.cpp`, and the `AEON_ENABLE_LEGACY_V4_GRAPH` gate itself. The
> "current engine path" below is the **rebuilt** graph; the storage, streaming, artifact-format
> and kernel sections were always current. See [DOCUMENTATION_STATUS.md](DOCUMENTATION_STATUS.md).

*Status: current runtime map, audited 2026-09-12.*

This document distinguishes the current inference engine from validation programs,
hardware spikes, and offline tooling. A file being built by CMake does not by itself
mean that it is part of the production runtime.

## Current engine path

The current integration point is the rewrite: `src/architecture/deepseek_v4/core/v4_engine.hpp`
binds the text front end to the graph, `core/v4_graph.hpp` runs the ordered forward
(embed → 43 × layer → head → LM head), and `core/v4_model_host.hpp` owns everything resident
(loader, config, spec, contract, budget, resources, scratch, streams, 43 layers, expert pools,
registry, staging, supply, executor). Model-level GPU allocations are owned by
`v4_model_resources.hpp`; architecture-neutral Hot/Warm/Cold transfer state, prefetch,
direct-I/O materialization, and supply telemetry are owned by
`infrastructure/core/tiered_expert_supply.hpp`, with `v4_expert_supply.hpp` retaining the V4
request adapter.

The production-facing implementation is:

- `src/architecture/deepseek_v4/core/config.hpp` - DeepSeek-V4 model configuration.
- `src/platform/rdna3/device.hpp` - RDNA3/HIP device selection and GPU utilities.
- `src/architecture/deepseek_v4/core/v4_engine.hpp` - the text binding: tokenizer + prompt encoder + generation loop + graph.
- `src/architecture/deepseek_v4/core/v4_graph.hpp` - the ordered forward pass (embed, 43 layers, head, LM head).
- `src/architecture/deepseek_v4/core/v4_model_host.hpp` - what is resident: the whole assembly.
- `src/architecture/deepseek_v4/core/v4_sampler.hpp` - the logit-processor seam and the sampler.
- `src/architecture/deepseek_v4/core/v4_layer_body.hpp` - the single layer body (decode and chunk share it).
- `src/architecture/deepseek_v4/core/v4_model_resources.hpp` - RoPE caches and model-level resident weights.
- `src/architecture/deepseek_v4/core/v4_expert_supply.hpp` - six-expert V4 request adapter and Aeon payload-source mapping.
- `src/infrastructure/core/tiered_expert_supply.hpp` - architecture-neutral tiered payload movement and transfer lifecycle.
- `src/infrastructure/core/aeon_loader.hpp` - native `.aeon` dense and expert container access.
- `src/architecture/deepseek_v4/core/memory_budget.hpp` - V4 VRAM/host feasibility calculations and startup checks.
- `src/backend/swizzled_w4a16/core/vram_expert_pool.hpp` - current backend's Tier 1 hot expert pool.
- `src/infrastructure/core/host_expert_pool.hpp` - shared Tier 2 warm expert pool.
- `src/infrastructure/core/expert_registry.hpp` - shared expert residency and usage tracking.
- `src/infrastructure/core/prefetch_staging.hpp` - shared bounded pinned staging and HIP event ownership.
- `src/infrastructure/io/direct_io_reader.hpp` - shared validated batched `io_uring`/`O_DIRECT` cold reads.
- `src/architecture/deepseek_v4/text/` and `src/infrastructure/text/` - DSV4 and generic text support.
- `src/architecture/deepseek_v4/kernels/` - V4 attention, routing, and Hyper-Connections kernels.
- `src/backend/swizzled_w4a16/kernels/` - current W4A16 swizzle, GEMV, and fused expert kernels.

`V4Engine` is the runtime entry point (via `tools/aeon_chat.cpp`). It drives the graph,
which loads the model architecture configuration, evaluates the runtime memory
policy, and opens the dedicated `O_DIRECT` descriptor. The cold request path uses
`DirectIOReader` and the staging arena before upload on `sdma_cold_stream`; mapped
access remains available inside the loader for native `.aeon` ownership and
validation. The bounded Hot/Warm/Cold path is implemented, while physical layout,
cold-cache measurement, and latency-hiding acceptance remain open in
[PHASE_2_EXECUTION_PLAN.md](../execution/active/PHASE_2_EXECUTION_PLAN.md).

## Active validation (default `ctest` — 44 tests)

These targets are built and run on every branch. They validate the artifact format,
the storage tiers, the W4A16 kernels, the text front end, the kept model-side
components (contract, router, HC Sinkhorn), the Tier-1/2/3 oracle gates, and the
graph rewrite's P1–P4 gates.

- `tests/test_aeon_loader.cpp`
- `tests/test_aeon_swizzled_loader.cpp`
- `tests/test_dynamic_expert_pool.cpp`
- `tests/test_model_direct_io.cpp`
- `tests/test_dsv4_tokenizer.cpp`
- `tests/test_text_generation.cpp`
- `tests/test_swiglu_clamp.cpp`
- `tests/test_hc_sinkhorn.cpp`
- `tests/test_moe_router.cpp`
- `tests/test_v4_model_contract.cpp`
- `tests/test_w4a16_swizzle.cpp`
- `tests/test_w4a16_swizzled_gemv.cpp`
- `tests/test_w4a16_swizzled_dual_gemv.cpp`
- `tests/test_aeon_moe_fused_w2.cpp`
- `tests/test_expert_registry_warm_state.cpp`
- `tests/test_supply_telemetry.cpp`
- `tests/test_routing_profile.cpp`

`tests/bench_model_direct_io.cpp` is a benchmark, not a correctness test. Its
results belong in `PERFORMANCE_LEDGER.md` only when a run completes with clearly
recorded conditions.

## Deleted: legacy graph (2026-09-18)

The pre-rewrite graph and everything that existed only to drive or certify it were
deleted, because the graph they implement is a *different computation* (the ledger
marks its measurements invalid) and leaving it invites mistaking a green legacy run
for coverage. Removed: `core/v4_pipeline.hpp`, `reference/v4_int4_reference.hpp`,
`reference/v4_attention_oracle.hpp`, `text/dsv4_chat_formatter.{hpp,cpp}`,
`tests/test_v4_real_expert_parity.cpp`, `tests/test_v4_real_dense_parity.cpp`,
`tests/test_dsv4_chat_formatter.cpp`, `tools/profile_routing.cpp`,
`tools/record_v4_gpu_evidence.cpp`, `cmake/AeonLegacyGraph.cmake`, and the
`AEON_ENABLE_LEGACY_V4_GRAPH` option. Also removed as dead code: the pipeline-only
`PipelineBatchScratchBuffers`, `hc_project_batched_kernel`,
`hc_pre_combine_batched_kernel`, `v4_pipeline_swiglu_clamp_batched_kernel`, and
`V4_ARGMAX_BLOCKS`. All of it is recoverable from git history and from `main`.

`tools/aeon_model_contract.cpp` was **promoted** into the default build (it uses
only kept components), and `v4_pipeline_accumulate_expert_kernel` was **kept** as
the fp16 reproducibility control that the deterministic gate fixtures use.

## Manual diagnostics and benchmarks

These targets are intentionally outside the default CTest suite because they are
manual hardware diagnostics or measurements:

- `src/smoke.cpp` / `smoke_check` - compiler and HIP toolchain sanity check.
- `tools/aeon_info.cpp` / `aeon_info` - hardware and topology inspection.
- `tests/bench_model_direct_io.cpp` - model-backed cold-read request-shape benchmark.

## Offline scripts

- `scripts/convert_safetensors_to_aeon.py` is the current production model preparation
  tool. It creates `model_manifest.json`, `model_dense.aeon`,
  `model_experts_swizzled.aeon`, and the version-2 swizzled expert index.
- `scripts/prepare_dsv4_tokenizer.py` prepares the native tokenizer artifact used by
  the text path.

## Cleanup rule

Code outside the current runtime path must either protect a current contract,
measure an open gate, or carry unique historical evidence. Otherwise delete it
and update the map and execution records in the same change.
