# Project Aeon Codebase Map

*Status: current runtime map, audited 2026-09-12.*

This document distinguishes the current inference engine from validation programs,
hardware spikes, and offline tooling. A file being built by CMake does not by itself
mean that it is part of the production runtime.

## Current engine path

The current integration point is `src/architecture/deepseek_v4/core/v4_pipeline.hpp`. It is header-only and
owns the V4 execution schedule and generation API. Model-level GPU allocations
are owned by `v4_model_resources.hpp`; Hot/Warm/Cold transfer state, prefetch,
direct-I/O materialization, and supply telemetry are owned by
`v4_expert_supply.hpp`.

The production-facing implementation is:

- `src/architecture/deepseek_v4/core/config.hpp` - DeepSeek-V4 model configuration.
- `src/platform/rdna3/device.hpp` - RDNA3/HIP device selection and GPU utilities.
- `src/architecture/deepseek_v4/core/v4_pipeline.hpp` - multi-layer V4 execution schedule and generation orchestration.
- `src/architecture/deepseek_v4/core/v4_model_resources.hpp` - RoPE caches and model-level resident weights.
- `src/architecture/deepseek_v4/core/v4_expert_supply.hpp` - Hot/Warm/Cold transfer state, prefetch, and supply telemetry coordination.
- `src/architecture/deepseek_v4/core/v4_block.hpp` - transformer block composition.
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

`V4Pipeline::initialize()` is the sole runtime entry point. It discovers the native
manifest, loads the model architecture configuration, evaluates the runtime memory
policy, and opens the dedicated `O_DIRECT` descriptor. The cold request path uses
`DirectIOReader` and the staging arena before upload on `sdma_cold_stream`; mapped
access remains available inside the loader for native `.aeon` ownership and
validation. The bounded Hot/Warm/Cold path is implemented, while physical layout,
cold-cache measurement, and latency-hiding acceptance remain open in
[PHASE_2_EXECUTION_PLAN.md](../execution/active/PHASE_2_EXECUTION_PLAN.md).

## Active production validation

These targets exercise the current `.aeon` and three-tier direction and should be
kept prominent while Phase 2 is in progress:

- `tests/test_aeon_loader.cpp`
- `tests/test_aeon_swizzled_loader.cpp`
- `tests/test_dynamic_expert_pool.cpp`
- `tests/bench_full_model.cpp`
- `tests/test_model_direct_io.cpp`
- `tests/test_hot_warm_cold_pipeline.cpp`
- `tests/test_dsv4_tokenizer.cpp`
- `tests/test_dsv4_chat_formatter.cpp`
- `tests/test_text_generation.cpp`
- `tools/profile_routing.cpp` / `profile_routing`
- `tools/aeon_chat.cpp` / `aeon_chat`

`tests/bench_full_model.cpp` is a benchmark, not a correctness test. Its results
belong in `PERFORMANCE_LEDGER.md` only when a run completes with clearly recorded
conditions.

## Regression coverage for completed work

These targets validate pieces that are already part of the engine and should remain
as regression tests, even though they are not runtime binaries:

- `tests/test_config_parser.cpp`
- `tests/test_swiglu_clamp.cpp`
- `tests/test_hc_sinkhorn.cpp`
- `tests/test_w4a16_swizzle.cpp`
- `tests/test_w4a16_swizzled_gemv.cpp`
- `tests/test_w4a16_swizzled_dual_gemv.cpp`
- `tests/test_aeon_moe_fused_w13.cpp`
- `tests/test_aeon_moe_fused_w2.cpp`
- `tests/test_moe_router.cpp`
- `tests/test_v4_attention.cpp`
- `tests/test_v4_block.cpp`

These are not noise: they protect the completed Phase 1 implementation and provide
smaller failure surfaces when the integrated pipeline changes.

## Historical and infrastructure spikes

These targets validate foundational hardware behavior or an earlier toy model. They
are useful when changing the relevant subsystem, but they do not need to be treated
as part of every normal engine run:

- `src/smoke.cpp` / `smoke_check` - compiler and HIP toolchain sanity check.
- `tools/aeon_info.cpp` / `aeon_info` - hardware and topology inspection.
- `tests/test_wmma_tile.cpp` and `tests/bench_wmma_gemm.cpp` - Phase 0 WMMA checks.
- `tests/test_direct_io.cpp` - standalone direct-I/O primitive check.
- `tests/bench_async_overlap.cpp` - compute/SDMA overlap benchmark.
- `tests/test_toy_moe_layer.cpp` - earlier toy cache pipeline.

These should remain available as explicit diagnostic targets until the corresponding
production subsystem is stable. They should eventually be separated from the default
validation set, rather than deleted casually.

## Offline scripts

- `scripts/convert_safetensors_to_aeon.py` is the current production model preparation
  tool. It creates `model_manifest.json`, `model_dense.aeon`,
  `model_experts_swizzled.aeon`, and the version-2 swizzled expert index.
- `scripts/prepare_rdna.py` is an older synthetic formatter for the original toy
  `AEON` layout. It does not create the current `AEON_DENSE`/`AEON_EXPERTS` format
  and is retained only for historical diagnostics.

## Cleanup rule

Do not delete a spike merely because it is not on the current runtime path. First
move it out of the default build or mark it as an explicit diagnostic target, then
delete it only after its result has been captured in the relevant plan or ledger.
