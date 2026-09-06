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

---

## 3. Progress Tracking & State of Execution
*Keep this section up-to-date at the end of every significant task or session.*

### Past (Completed)
- [x] Initialized Git repository on `main` branch and linked remote `https://github.com/marcoLapiello/project-aeon.git`.
- [x] Completed architectural research, blocker analysis, and model specialization strategy.
- [x] Verified host hardware: AMD Ryzen Threadripper PRO 3975WX (32C/64T), 64 GB DDR, 4x AMD Radeon RX 7900 XTX (96 GB VRAM total, `gfx1100`), ROCm 7.2.2 toolchain with `hipcc`, Linux kernel 7.0.
- [x] Established non-re-inventing philosophy: No PyTorch in production runtime; ingest standard model formats (GGUF/Safetensors); Python for offline toolchain/tests only.
- [x] Micro-Step 1.1: Root `CMakeLists.txt` configured for `hipcc`, C++20, and `gfx1100` Wave32 mode.
- [x] Micro-Step 1.2: Hardware inspection utility (`tools/aeon_info.cpp`) enumerating 4x RX 7900 XTX devices, CUs, VRAM, and full P2P peer access matrix.
- [x] Micro-Step 2.1: Single-tile WMMA HIP kernel test with CPU reference validation (`tests/test_wmma_tile.cpp`).
- [x] Micro-Step 2.2: Tiled block GEMM benchmark achieving ~25.6 TFLOP/s and 870 us per 2048-dim expert on silicon (`tests/bench_wmma_gemm.cpp`).
- [x] Micro-Step 3.1: 4096-byte sector-aligned memory allocator and `io_uring` direct reader (`src/io/`).

### Present (In Progress)
- [ ] **Phase 0 Spike 3 — Asynchronous I/O & SDMA Transfer Overlap**:
  - [ ] Micro-Step 3.2: Concurrent compute + SDMA transfer jitter test.

### Future (Upcoming Next)
- [ ] **Phase 0 Spike 4 — Sector-Aligned Storage & Single-Layer Toy MoE Pipeline**:
  - Offline format packer (`prepare_rdna.py`) and single-layer cached execution test.

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
