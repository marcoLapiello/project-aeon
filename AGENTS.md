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
- [Performance & Accuracy Ledger](plans-and-docs/PERFORMANCE_LEDGER.md): Empirical benchmark ledger recording test conditions, throughput, latencies, and cache behaviors across major milestones.

---

## 3. Progress Tracking & State of Execution
*Keep this section up-to-date at the end of every significant task or session.*

### Past (Completed)
- [x] Initialized Git repository on `main` branch and linked remote `https://github.com/marcoLapiello/project-aeon.git`.
- [x] Completed architectural research, blocker analysis, and model specialization strategy.
- [x] Verified host hardware: AMD Ryzen Threadripper PRO 3975WX (32C/64T), 64 GB DDR, 4x AMD Radeon RX 7900 XTX (96 GB VRAM total, `gfx1100`), ROCm 7.2.2 toolchain with `hipcc`, Linux kernel 7.0.
- [x] Established non-re-inventing philosophy: No PyTorch in production runtime; ingest standard model formats (GGUF/Safetensors); Python for offline toolchain/tests only.
- [x] **[Phase 0 Execution Plan](plans-and-docs/PHASE_0_EXECUTION_PLAN.md) — Foundations & Hardware Validation**:
  - Spike 1: CMake & Ninja build system with `hipcc` targeting RDNA3 Wave32 mode; hardware topology inspection tool (`tools/aeon_info.cpp`) verifying 4x RX 7900 XTX devices with 100% full bidirectional P2P access.
  - Spike 2: Native Wave32 WMMA micro-kernel with CPU reference validation (`tests/test_wmma_tile.cpp`); tiled block GEMM benchmark achieving ~25.6 TFLOP/s and 870 us per 2048-dim matrix on silicon (`tests/bench_wmma_gemm.cpp`).
  - Spike 3: 4KB sector-aligned memory allocator and Linux `io_uring` Direct I/O reader achieving 6.33 GB/s from NVMe (`src/io/`, `tests/test_direct_io.cpp`); concurrent compute + SDMA transfer test proving non-blocking PCIe DMA transfers at 24.9 GB/s with 0% compute jitter (`tests/bench_async_overlap.cpp`).
  - Spike 4: Offline 4KB sector-aligned `.aeon` format packer (`scripts/prepare_rdna.py`); single-layer toy MoE pipeline validating dynamic Tier 1 VRAM LRU caching and asynchronous Tier 2 Host DDR SDMA swaps (`tests/test_toy_moe_layer.cpp`).

### Present (In Progress)
- [ ] **[Phase 1 Execution Plan](plans-and-docs/PHASE_1_EXECUTION_PLAN.md) — Single-GPU Core Runtime for DeepSeek-V4-Flash**:
  - [x] Spike 1.1: DeepSeek-V4 C++20 configuration & metadata parser (`src/core/config.hpp`, `tests/test_config_parser.cpp`).
  - [x] Spike 1.2: Zero-dependency Safetensors header parser (`src/core/safetensors.hpp`, `tests/test_safetensors_parser.cpp`).
  - [x] Spike 2.1: Wave32 RMSNorm and fused SwiGLU with `swiglu_limit = 10.0` clamping (`tests/test_swiglu_clamp.cpp`).
  - [x] Spike 2.2: Hyper-Connections (HC) 4-stream Sinkhorn normalization and residual expansion kernels on silicon (`src/kernel/hc_sinkhorn.hpp`, `tests/test_hc_sinkhorn.cpp`).
  - [x] Spike 3: Fused INT4 $\to$ FP16 Wave32 Dequantization-GEMM kernel targeting `gfx1100` WMMA (`src/kernel/w4a16_gemm.hpp`, `tests/test_w4a16_wmma.cpp`).
  - [x] Spike 4: Dual-mode MoE routing (hash layers 0–2 + `sqrtsoftplus` layers 3–42) with Tier 1/2 dynamic expert streaming (`src/kernel/moe_router.hpp`, `tests/test_moe_router.cpp`, `tests/test_v4_moe_layer.cpp`).
  - [x] Spike 5: Sliding-window attention ($W=128$) and end-to-end `DeepSeekV4Block` single-layer silicon validation (`src/kernel/v4_attention.hpp`, `src/core/v4_block.hpp`, `tests/test_v4_attention.cpp`, `tests/test_v4_block.cpp`).
  - [x] Spike 6: Multi-layer pipeline execution & autoregressive generation benchmark (`src/core/safetensors_loader.hpp`, `src/core/v4_pipeline.hpp`, `tests/test_v4_pipeline.cpp`, `tests/bench_v4_generation.cpp`).

### Present (In Progress)
- [ ] **Phase 1 Complete**: Single-GPU core runtime for DeepSeek-V4 verified end-to-end on real INT4-W4A16 Safetensors shards.

### Future (Upcoming Next)
- [ ] **Phase 2 — Multi-GPU Pipeline Parallelism**:
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
