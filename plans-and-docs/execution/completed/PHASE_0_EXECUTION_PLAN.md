# Phase 0: Foundations & Hardware Validation Micro-Plan

This actionable plan breaks down the foundational work into discrete, verifiable micro-steps. Each step produces working, testable code verified directly on the local 4x AMD Radeon RX 7900 XTX (`gfx1100`) rig before moving to the next.

---

## Spike 1: Build System & Hardware Discovery

### Micro-Step 1.1: CMake & HIP Build Scaffold
- **Goal:** Set up a clean, modern CMake (3.28+) and Ninja build system targeting AMD ROCm 7.2 with `hipcc`.
- **Target Architecture:** `gfx1100` (RDNA3).
- **Compiler Flags:** `-std=c++20 -O3 --offload-arch=gfx1100 -mwavefrontsize64=false` (ensures 32-lane Wave32 compilation).
- **Verification:** Run `cmake -B build -G Ninja && ninja -C build` and ensure clean generation with zero warnings.
- **Git Commit:** `build: configure initial cmake build system for rocm hipcc and gfx1100`

### Micro-Step 1.2: Hardware Topology Inspection Utility (`tools/aeon_info.cpp`)
- **Goal:** Write a standalone diagnostic executable querying the ROCm runtime via HIP.
- **Functionality:**
  - Enumerate all detected GPUs (expected: 4x RX 7900 XTX).
  - Print device names, GFX architectures, compute unit (CU) counts, total VRAM, and memory bus width.
  - Query PCIe bus IDs and check peer-to-peer (P2P) access compatibility between all pairs of GPUs.
  - Verify Wave32 default execution mode (`warpSize == 32`).
- **Verification:** Run `./build/bin/aeon_info` on the machine and inspect output against expected hardware specs.
- **Git Commit:** `feat(tools): add aeon_info hardware topology and rocm inspection tool`

---

## Spike 2: Bare-Metal Wave32 WMMA Compute Kernel

### Micro-Step 2.1: Single-Tile WMMA Smoke Test (`tests/test_wmma_tile.cpp`)
- **Goal:** Verify that the compiler generates native RDNA3 Wave32 WMMA matrix instructions and that arithmetic output matches CPU reference math.
- **Kernel Implementation:**
  - Execute a single 16x16x16 tile multiplication ($D = A \times B + C$) using `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (or FP16 input variant).
  - Launch exactly 1 wavefront of 32 threads.
- **Verification:** Compare device output with a bit-accurate CPU reference calculation; assert absolute difference $\epsilon < 10^{-4}$.
- **Git Commit:** `test(kernels): add single-tile wave32 wmma hip smoke test with cpu validation`

### Micro-Step 2.2: Tiled Block GEMM Benchmark (`tests/bench_wmma_gemm.cpp`)
- **Goal:** Implement a 2D block-tiled GEMM kernel ($M=1024, N=1024, K=1024$) using shared memory (LDS) and WMMA Wave32 instructions.
- **Metrics Collected:**
  - Execution time (microseconds via `hipEventElapsedTime`).
  - Sustained TFLOP/s achieved vs. theoretical peak of RX 7900 XTX.
  - Verification against CPU or reference matrix multiplication.
- **Verification:** Run benchmark across different tile sizes to identify optimal LDS configurations on `gfx1100`.
- **Git Commit:** `perf(kernels): implement block-tiled wave32 wmma gemm benchmark`

---

## Spike 3: Asynchronous I/O & SDMA Transfer Overlap

### Micro-Step 3.1: Sector-Aligned Allocator & Direct I/O Reader (`src/io/`)
- **Goal:** Create a lightweight, high-performance reader using Linux `io_uring` with `O_DIRECT`.
- **Implementation:**
  - Allocate host memory aligned to 4096-byte boundaries (`posix_memalign` / `aligned_alloc`).
  - Read a test file directly from NVMe (`nvme0n1`) into aligned memory bypassing kernel page cache.
  - Measure read throughput in GB/s.
- **Verification:** Read a synthetic 500 MB file with `O_DIRECT` and verify data integrity with MD5/SHA256 checksum.
- **Git Commit:** `feat(io): implement 4kb sector-aligned direct io reader using io_uring`

### Micro-Step 3.2: Concurrent Compute & SDMA Transfer Stress Test (`tests/bench_async_overlap.cpp`)
- **Goal:** Validate that background PCIe DMA transfers (`hipMemcpyAsync` on dedicated stream) do NOT serialize or degrade foreground WMMA compute kernel execution.
- **Implementation:**
  - Stream 1: Continuously launch the WMMA GEMM kernel on GPU 0.
  - Stream 2 (SDMA queue): Concurrently push 100 MB buffers from pinned host memory to GPU VRAM.
  - Measure kernel latency with and without concurrent transfer to quantify ROCm bus jitter.
- **Verification:** Assert that compute kernel execution duration increases by no more than an acceptable margin ($< 5\text{--}10\%$).
- **Git Commit:** `test(runtime): verify compute and sdma transfer overlap without serialization`

---

## Spike 4: Sector-Aligned Storage & Single-Layer Toy MoE Pipeline

### Micro-Step 4.1: Offline Model Formatter (`scripts/prepare_rdna.py`)
- **Goal:** Python tool to pack mock or real MoE weights into Aeon's `.aeon` format with 4096-byte padding and layout aligned for Wave32 register swizzling.
- **Verification:** Generate synthetic 64-expert weights file and verify all tensor headers and data offsets align to $4096 \times n$.
- **Git Commit:** `feat(scripts): add offline 4kb-aligned expert weight serialization tool`

### Micro-Step 4.2: Single-Layer Toy MoE Runtime (`tests/test_toy_moe_layer.cpp`)
- **Goal:** End-to-end integration of Tier 1 (VRAM LRU pool) and Tier 2 (Pinned Host RAM) with top-K routing.
- **Workflow:**
  - Simulate a stream of 50 tokens with top-6 routing across 64 experts.
  - Hit path: Compute resident experts immediately.
  - Miss path: Asynchronously fetch missing expert from host RAM via SDMA.
  - Measure total latency per token across varying cache hit rates (100%, 80%, 50%).
- **Verification:** Confirm correct output math and benchmark token processing rate under simulated routing.
- **Git Commit:** `feat(runtime): implement single-layer toy moe pipeline with dynamic expert caching`
