# AGENTS.md — Project Aeon System & Agent Context

## 1. Project Purpose & High-Level Context
**Project Aeon** is a high-performance, bare-metal Mixture-of-Experts (MoE) inference engine built from scratch in C++20 and native HIP for consumer AMD hardware (primarily RDNA3 / `gfx1100`, scalable across multi-GPU rigs).

Aeon solves the memory wall for massive MoE models (e.g., DeepSeek-V4 architectures) on consumer workstations by combining:
1. **Bare-metal RDNA3 execution**: Wave32 execution mode, AI Matrix Accelerators (WMMA), and direct HIP/AMDGCN instruction dispatch without CUDA or framework overhead.
2. **Three-Tier Hierarchical Storage & Streaming**: VRAM (Hot LRU) $\leftarrow$ Host DDR (Warm pinned staging) $\leftarrow$ NVMe SSD (Cold asynchronous Direct I/O via Linux `io_uring` with 4KB sector alignment).
3. **Double-Buffered Expert-Batched Prefill & Pipeline Parallelism**: Eliminating I/O latency stalls by overlapping compute with asynchronous DMA/storage transfers.

---

## 2. Foundational Documentation
Always consult these authoritative documents for deep technical specifics:
- [Vision & Architecture Roadmap](plans-and-docs/PROJECT_AEON_VISION.md): Architectural pillars, multi-GPU topology, hardware target philosophy.
- [Technical Blockers & Risk Analysis](plans-and-docs/AEON_TECHNICAL_BLOCKERS_AND_RISKS.md): Analysis of cold-miss traps (Colibri trap), prefetch horizons, ROCm SDMA jitter, and engineering mitigations.
- [Specialization Strategy & Entropy Analysis](plans-and-docs/AEON_SPECIALIZATION_AND_ENTROPY_ANALYSIS.md): Rationale for Phase 1 single-model focus (DeepSeek MoE) and empirical activation entropy formulas.
- [Research & Inspiration Survey](plans-and-docs/AEON_RESEARCH_AND_INSPIRATION.md): Prior art review (hipfire, zinc, Flash-MoE, llama.cpp PR #25294).
- [Phase 0 Execution Plan](plans-and-docs/PHASE_0_EXECUTION_PLAN.md): Step-by-step micro-plan for foundational spikes and hardware validation.
- [Phase 1 Execution Plan](plans-and-docs/PHASE_1_EXECUTION_PLAN.md): Micro-execution plan for single-GPU Core Runtime on real INT4-W4A16 weights.
- [Phase 2 Execution Plan](plans-and-docs/PHASE_2_EXECUTION_PLAN.md): Micro-execution plan for Single-GPU 3-Tier Storage & Memory Hierarchy Optimization.
- [Performance & Accuracy Ledger](plans-and-docs/PERFORMANCE_LEDGER.md): Empirical benchmark ledger recording test conditions, throughput, latencies, and cache behaviors across major milestones.

### Local Reference Implementations
The primary external source references are maintained as shallow, default-branch checkouts outside this repository. They are for source comparison only, not Aeon build or runtime dependencies:
- [llama.cpp reference checkout](../aeon-references/llama.cpp): portable runtime, expert streaming, quantization, KV state, and serving paths.
- [FreeToken reference checkout](../aeon-references/freetoken): bandwidth-adaptive CPU/GPU execution, expert caching, prefill streaming, and agent-facing serving.
- [Colibri reference checkout](../aeon-references/colibri): VRAM/RAM/NVMe tiering, routing-aware placement, direct I/O, prefetch, and persistent KV state.
- [vLLM reference checkout](../aeon-references/vllm): paged memory, prefix/KV caching, scheduling, resource management, and production serving.
- [SGLang reference checkout](../aeon-references/sglang): radix/HiCache, chunked prefill, MoE scheduling, disaggregation, and AMD paths.

Update a reference checkout with `git -C <directory> pull --ff-only` and record its commit SHA whenever an implementation decision depends on a specific revision.

---

## 3. Progress Tracking & State of Execution
*Keep this section up-to-date at the end of every significant task or session.*

### Past (Completed)
- [x] Initialized the Git repository and validated the target hardware/toolchain: 4x RX 7900 XTX (`gfx1100`), 64 GB host RAM, ROCm 7.2.2, and native `hipcc`.
- [x] Established shallow local reference checkouts for llama.cpp, FreeToken, Colibri, vLLM, and SGLang under `/home/marcolap/aeon-references/` for comparative source research.
- [x] Completed the architectural research, model specialization decisions, and Phase 0 foundations. The project now has a working HIP/CMake base, hardware discovery, Wave32 WMMA validation, direct-I/O primitives, SDMA overlap checks, and the original toy MoE cache path. See [Phase 0 Execution Plan](plans-and-docs/PHASE_0_EXECUTION_PLAN.md).
- [x] Completed Phase 1: a single-GPU DeepSeek-V4 INT4-W4A16 runtime with configuration and Safetensors loading, fused kernels, MoE routing, sliding-window attention, transformer blocks, and multi-layer autoregressive generation. See [Phase 1 Execution Plan](plans-and-docs/PHASE_1_EXECUTION_PLAN.md).
- [x] Completed Phase 2 Spike 0: lossless Safetensors-to-`.aeon` repacking with separate dense and routed-expert containers, 4096-byte alignment, expert indexing, and bit-exact verification. See [Phase 2 Execution Plan](plans-and-docs/PHASE_2_EXECUTION_PLAN.md) and [Performance & Accuracy Ledger](plans-and-docs/PERFORMANCE_LEDGER.md).
- [x] Completed the implementation portion of Phase 2 Spike 1: dynamic memory budgeting, startup feasibility checks, a unified VRAM expert pool, Host RAM expert staging, expert residency tracking, and full-model silicon benchmarks. The original contiguous three-tier preload was blocked by host-memory pressure; the later bounded segmented Warm-tier integration is recorded below.
- [x] Completed Phase 2 Pipeline Modularization Step 1: carved out HIP utility kernels to `src/kernel/v4_pipeline_ops.hpp`, scratch activation arena to `src/core/v4_pipeline_scratch.hpp`, and layer structure to `src/core/v4_layer.hpp`, cutting the monolithic `v4_pipeline.hpp` from ~1,500 down to 920 lines with full silicon test verification.
- [x] Completed Phase 2 Pipeline Modularization Step 2: eliminated the dual-cache split, retired per-layer local LRU caches, standardized all pipelines on `UnifiedVRAMExpertPool` + `ExpertRegistry`, and removed hardcoded slot counts in favor of dynamic runtime configuration. Passed bit-exact tests on silicon (`test_dynamic_expert_pool`, `test_aeon_pipeline`, `test_v4_pipeline`).
- [x] Connected a bounded Hot/Warm/Cold runtime path: segmented Warm Host storage, direct `io_uring` population for initial Hot and Warm residents, VRAM-to-host demotion, safe Warm-to-Hot promotion, and multi-layer staging reuse. The 43-layer silicon smoke test passes with 676 Hot slots, 8 Warm slots, and direct Cold misses; full-capacity performance measurement remains open.

### Present (In Progress)
- [ ] **[Phase 2 Execution Plan](plans-and-docs/PHASE_2_EXECUTION_PLAN.md) — Single-GPU 3-Tier Storage & Memory Hierarchy Optimization**:
  - [x] Spike 0: Surgical Safetensors-to-`.aeon` Model Repacking & Weight Verification.
  - [x] Spike 1: Dynamic memory budgeting & Global Unified VRAM Expert Pool.
  - [ ] Pipeline Architecture Cleanup & Refactoring:
    - [x] Step 1: Mechanical modularization (extract ops, scratch buffers, layer context).
    - [x] Step 2: Eliminate dual-cache split (retire per-layer local LRU cache, standardize on Unified VRAM Pool + ExpertRegistry, scrub hardcoded slot numbers).
    - [x] Step 3: Purge host-side HC synchronization roundtrips in token step loop.
      - Resolved blocking performance gap: redesigned `hc_project_kernel` with 24 parallel Wave32 blocks and `float4` vectorized loads ($1,188\ \mu\text{s} \to 9.2\ \mu\text{s}$, $129\times$ kernel speedup) and vectorized `hc_pre_combine_kernel`.
      - Exceeded baseline: decode throughput accelerated from 49 tok/s to **122.9 tok/s** on 2 layers with bit-exact CPU reference parity on physical silicon.
  - [x] Spike 2: Dual-stream asynchronous SDMA prefetching & PCIe latency hiding:
    - [x] Micro-Step 2.1: Lookahead Routing & Prefetch Horizon Pipeline.
    - [x] Micro-Step 2.2: Double-Buffered Asynchronous SDMA Transfer Stream (`PrefetchStagingArena`).
    - [x] Micro-Step 2.3: Overlap Verification & Latency Hiding Benchmark on Silicon (`test_async_prefetch` passing with +71.5% decode speedup under cold misses).
      - Note: Inter-layer lookahead confirmed that single-threaded CPU `memcpy` from unpinned `mmap` backing pages ($85\text{ MB/step}$) bottlenecks prefetching, making Spike 3 Direct I/O the critical unlock.
  - [ ] Spike 3: Linux `io_uring` Direct I/O NVMe Cold Tier integration.
    - [x] Implemented batched `io_uring` requests, validated expert offsets, dedicated `O_DIRECT` model descriptor, and staging-slot ownership cleanup.
    - [x] Corrected synchronous regular-file submissions by enabling `IOSQE_ASYNC` and splitting expert payloads into 4 MiB aligned subreads.
    - [x] Integrated direct cold reads into the native pipeline and passed model parity plus silicon generation regressions.
    - [x] Replaced contiguous Warm Host allocation with segmented slabs and populated bounded Hot/Warm residents through direct I/O; added Hot-to-Warm DMA demotion and safe multi-layer staging reuse.
    - [x] Ran the full 43-layer benchmark with 606 direct-populated Warm slots (`7.99 GiB`): 555 Warm hits and 601 Cold misses, with valid output at `3.98 tok/s`.
    - [x] Ran the target 35 GiB Warm profile with 2,654 direct-populated slots: 684 Warm hits and 472 Cold misses, with valid output at `3.95 tok/s`; the run completed but increased observed swap usage by approximately `0.7 GiB`.
    - [x] Removed the synchronous Hot-to-Warm demotion wait and validated the controlled A/B: `4.37 tok/s` with Warm disabled versus `5.15 tok/s` with the 35 GiB Warm profile; output remained identical.
    - [x] Executed Expert Review Step 1 ([Expert Performance Review](plans-and-docs/EXPERT_PERFORMANCE_REVIEW.md)): deleted Hot-to-Warm D2H demotion from the request path, warm hits now upload directly from pinned segments (no staging memcpy), and shared-expert kernels enqueue before CPU staging dispatch. M16 A/B: `4.36` vs `5.11 tok/s` — throughput unchanged despite 3× less warm-hit traffic, proving the loop is latency-bound by just-in-time dispatch, not bandwidth-bound.
    - [ ] Reduce host-memory pressure, reach the `>= 6.0 GB/s` model-backed throughput target, and complete controlled Tier 2 end-to-end measurements; remaining work is dominated by physical expert placement and storage-layout optimization.
    - [x] Executed Expert Review Step 3 measurement (M17): per-layer n-token-union top-6 coverage instrumentation (`AEON_MEASURE_LOCALITY=1`) showed gated-layer coverage saturates at 3.60/6 (60%) for n=2 — equal to the already-measured 59.3% VRAM hot-hit rate. LRU residency at 664 slots already harvests all historical routing locality; history-based speculative prefetch would add ≤1% hits, so it was rejected. The residual ~2.4 cold misses/layer are genuinely novel experts. (Instrumentation removed after the conclusion was recorded.)
    - [x] Executed Expert Review Step 4 (M18): rewrote the routed-expert W4A16 decode path as a warp-per-row fused INT4 GEMV (`w4a16_gemv_kernel`, coalesced `uint4` streams, FP32 dual-accumulator FMA, shuffle reduction) — `138.8 → 13.9 µs` per GEMM (10×, ~340 GB/s), bit-exact vs CPU FP32 reference, golden token 295 preserved. End-to-end: warm off `4.36 → 5.25 tok/s`, warm 35 GiB `5.11 → 5.79 tok/s`, TTFT `2755 → 1835 ms`.
    - [x] Executed Expert Review Step 2 (M19): re-laid-out `UnifiedVRAMExpertPool` to a single contiguous device allocation with per-slot 13.5 MiB regions byte-identical to the `.aeon` layout (full-expert H2D = 1 `hipMemcpyAsync` instead of 6) and split DMA streams (`sdma_cold_stream` for io_uring uploads, `sdma_stream` for warm/safetensors H2D). Warm 35 GiB `5.79 → 5.88 tok/s`, warm off `5.25 → 5.39 tok/s`, tokens identical to M18; all regressions pass, golden token 295 preserved.
    - [x] Executed Expert Review Step 5 (M20): eliminated the per-layer router-logits D2H/sync/CPU/H2D round-trip (device half→float kernel), added a two-phase GPU argmax over the 129,280-logit head (first-max-wins tie-break identical to CPU; replaces 258 KB D2H + CPU scan with a 4-byte readback), and a vectorized `uint4` FP16 GEMV for router/shared-expert/LM-head projections. Warm 35 GiB `5.88 → 7.10 tok/s` (+20.7%), warm off `5.39 → 6.32 tok/s` (+17.3%); tokens returned to the M15 sequence `[237, 223 ×7]`; all regressions pass, golden token 295 preserved.
    - [ ] **Next (highest priority)**: Review Step 6 — contiguous `.aeon` repack (fallocate, frequency-ordered) toward the ≥6 GB/s cold-tier target; the step is now dominated by the exposed just-in-time cold-miss read path (`141 ms/token`, ~1,077 cold misses/run).

### Future (Upcoming Next)
- [ ] **Phase 3 — Multi-GPU Pipeline Parallelism**:
  - 4-card stage partitioning across P2P PCIe links and 1F1B micro-batching.

---

## 4. Project Rules & Engineering Conventions

### Code Architecture & Runtime
1. **Production Runtime Zero-Dependency Policy**: The inference runtime engine must remain pure C++20 / native HIP with no runtime dependencies on Python or PyTorch.
2. **Standard Ecosystem Compatibility**: Never invent custom lossy quantization formats. Ingest standard community formats (GGUF, Safetensors).
3. **Hardware Precision Discipline**: Golden reference tests must verify that GPU GEMM / dequant kernels match reference precision within standard FP16/BF16 tolerances ($\epsilon < 10^{-3}$).
4. **Direct I/O Discipline**: All streaming file reads must be strictly 4096-byte aligned (`O_DIRECT` compliant) to eliminate kernel page-cache contention and buffer copies.

### Development Process & Git Conventions
1. **Incremental Micro-Steps**: Advance through small, verifiable steps. Never implement broad abstractions before underlying hardware primitives are verified on silicon.
2. **Hardware-Grounded Verification**: Test and benchmark on physical hardware (`gfx1100`) at every step.
3. **Commit Messages**: Follow standard conventional commits format (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`).
4. **Maintenance of AGENTS.md**: Update the "Progress Tracking & State of Execution" section whenever milestones or micro-steps transition between Past, Present, and Future.
5. **Empirical Milestone Logging**: For every significant milestone or architectural transition, log the exact test conditions, throughput (tok/s), latencies (TTFT, decode step ms), and cache metrics in [Performance & Accuracy Ledger](plans-and-docs/PERFORMANCE_LEDGER.md). Do not log noise for small code edits; log meaningful, comparable system-level milestones to provide clear before-and-after tracking on the path to production.
6. **Strategic codebase searching**: When you need to collect any info from the codebase or search for specific code or entities, use the search subagent tool.
7. **Modular, Scalable and Mantainable**: avoid growing monolitic files with mixed concerns, extract those concerns in separate smaller and focused modules, reuse and improve existing modules, avoid duplications and redundancies.