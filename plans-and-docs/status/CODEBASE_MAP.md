# Project Aeon Codebase Map

*Status: current runtime map, audited 2026-09-10.*

This document distinguishes the current inference engine from validation programs,
hardware spikes, and offline tooling. A file being built by CMake does not by itself
mean that it is part of the production runtime.

## Current engine path

The current integration point is `src/core/v4_pipeline.hpp`. It is header-only and
pulls in the model configuration, device setup, transformer block, Aeon loader,
Safetensors loader, VRAM and host expert pools, registry, and all active kernels.

The production-facing implementation is:

- `src/core/config.hpp` - model and runtime configuration.
- `src/core/device.hpp` - HIP device selection and GPU utilities.
- `src/core/v4_pipeline.hpp` - multi-layer inference and memory-tier orchestration.
- `src/core/v4_block.hpp` - transformer block composition.
- `src/core/aeon_loader.hpp` - native `.aeon` dense and expert container access.
- `src/core/safetensors_loader.hpp` and `src/core/safetensors.hpp` - source-format loading and parsing.
- `src/core/memory_budget.hpp` - VRAM/host feasibility calculations and startup checks.
- `src/core/vram_expert_pool.hpp` - Tier 1 hot expert pool.
- `src/core/host_expert_pool.hpp` - Tier 2 warm expert pool.
- `src/core/expert_registry.hpp` - expert residency and usage tracking.
- `src/core/prefetch_staging.hpp` - bounded pinned staging and HIP event ownership.
- `src/io/direct_io_reader.hpp` - validated batched `io_uring`/`O_DIRECT` cold reads.
- `src/text/` - native tokenizer, DSV4 formatter, and text-generation support.
- `src/kernel/*.hpp` - attention, W4A16 GEMM, routing, and Hyper-Connections kernels.

`V4Pipeline::init_aeon()` opens both the mapped expert container and a dedicated
`O_DIRECT` descriptor. The cold request path uses `DirectIOReader` and the
staging arena before upload on `sdma_cold_stream`; mapped access remains available
as a source for comparison and legacy paths. The bounded Hot/Warm/Cold path is
implemented, while physical layout, cold-cache measurement, and latency-hiding
acceptance remain open in [PHASE_2_EXECUTION_PLAN.md](../execution/active/PHASE_2_EXECUTION_PLAN.md).

## Active production validation

These targets exercise the current `.aeon` and three-tier direction and should be
kept prominent while Phase 2 is in progress:

- `tests/test_aeon_loader.cpp`
- `tests/test_aeon_pipeline.cpp`
- `tests/test_dynamic_expert_pool.cpp`
- `tests/bench_dynamic_pool.cpp`
- `tests/bench_full_model.cpp`
- `tests/test_model_direct_io.cpp`
- `tests/test_async_prefetch.cpp`
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
- `tests/test_safetensors_parser.cpp`
- `tests/test_swiglu_clamp.cpp`
- `tests/test_hc_sinkhorn.cpp`
- `tests/test_w4a16_wmma.cpp`
- `tests/test_moe_router.cpp`
- `tests/test_v4_moe_layer.cpp`
- `tests/test_v4_attention.cpp`
- `tests/test_v4_block.cpp`
- `tests/test_v4_pipeline.cpp`
- `tests/bench_v4_generation.cpp`

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
  tool. It creates `model_dense.aeon`, `model_experts.aeon`, and the expert index.
- `scripts/prepare_rdna.py` is an older synthetic formatter for the original toy
  `AEON` layout. It does not create the current `AEON_DENSE`/`AEON_EXPERTS` format
  and is retained only for historical diagnostics.

## Cleanup rule

Do not delete a spike merely because it is not on the current runtime path. First
move it out of the default build or mark it as an explicit diagnostic target, then
delete it only after its result has been captured in the relevant plan or ledger.
