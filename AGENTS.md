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

---

## 3. Progress Tracking & State of Execution
*Keep this section up-to-date at the end of every significant task or session.*

### Past (Completed)
- [x] Initialized the Git repository and validated the target hardware/toolchain: 4x RX 7900 XTX (`gfx1100`), 64 GB host RAM, ROCm 7.2.2, and native `hipcc`.
- [x] Completed the architectural research, model specialization decisions, and Phase 0 foundations. The project now has a working HIP/CMake base, hardware discovery, Wave32 WMMA validation, direct-I/O primitives, SDMA overlap checks, and the original toy MoE cache path. See [Phase 0 Execution Plan](plans-and-docs/PHASE_0_EXECUTION_PLAN.md).
- [x] Completed Phase 1: a single-GPU DeepSeek-V4 INT4-W4A16 runtime with configuration and Safetensors loading, fused kernels, MoE routing, sliding-window attention, transformer blocks, and multi-layer autoregressive generation. See [Phase 1 Execution Plan](plans-and-docs/PHASE_1_EXECUTION_PLAN.md).
- [x] Completed Phase 2 Spike 0: lossless Safetensors-to-`.aeon` repacking with separate dense and routed-expert containers, 4096-byte alignment, expert indexing, and bit-exact verification. See [Phase 2 Execution Plan](plans-and-docs/PHASE_2_EXECUTION_PLAN.md) and [Performance & Accuracy Ledger](plans-and-docs/PERFORMANCE_LEDGER.md).
- [x] Completed the implementation portion of Phase 2 Spike 1: dynamic memory budgeting, startup feasibility checks, a unified VRAM expert pool, Host RAM expert staging, expert residency tracking, and full-model silicon benchmarks. The full three-tier preload remains blocked by host-memory pressure and is documented in the Phase 2 plan.
- [x] Completed Phase 2 Pipeline Modularization Step 1: carved out HIP utility kernels to `src/kernel/v4_pipeline_ops.hpp`, scratch activation arena to `src/core/v4_pipeline_scratch.hpp`, and layer structure to `src/core/v4_layer.hpp`, cutting the monolithic `v4_pipeline.hpp` from ~1,500 down to 920 lines with full silicon test verification.
- [x] Completed Phase 2 Pipeline Modularization Step 2: eliminated the dual-cache split, retired per-layer local LRU caches, standardized all pipelines on `UnifiedVRAMExpertPool` + `ExpertRegistry`, and removed hardcoded slot counts in favor of dynamic runtime configuration. Passed bit-exact tests on silicon (`test_dynamic_expert_pool`, `test_aeon_pipeline`, `test_v4_pipeline`).

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
    - [ ] Reach the `>= 6.0 GB/s` model-backed throughput target and complete the Tier 2 warm-cache end-to-end measurement; remaining work is dominated by physical expert placement and storage-layout optimization.

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