# Project Aeon: Technical Blockers, Failure Modes & Engineering Mitigations

## Executive Summary
This document provides a critical engineering analysis of the potential blockers, structural traps, and failure modes that could jeopardize Project Aeon. Many offloading and streaming experiments (such as Colibri) achieve theoretical feasibility on paper but degrade to single-digit or sub-token-per-second throughput (1–2 tok/s) in real-world workloads.

To ensure Project Aeon remains a viable, high-throughput production engine (targeting 15–35+ tok/s), each identified bottleneck is analyzed alongside concrete engineering mitigations.

The local [Colibri reference checkout](../../aeon-references/colibri) and [FreeToken reference checkout](../../aeon-references/freetoken) provide the implementation comparisons for the storage-tier and CPU/GPU coprocessing risks discussed below.

---

## 1. Blocker 1: The "Cold Miss Avalanche" (The Colibri Trap)

### Mechanism of Failure
Naive offloading engines stream weights reactively. When the router selects $K$ active experts for Layer $L$, any expert not resident in memory forces the engine to halt GPU computation, issue I/O requests, wait for data arrival over PCIe/NVMe, and only then proceed with computation. 

Because matrix multiplication on RDNA3 compute units completes in hundreds of microseconds, an unmitigated 10–20 ms storage latency causes the GPU compute engines to sit idle 95–98% of the execution time.

### Quantitative Reality Check
Assuming an MoE architecture activating 8 experts per layer across 60 layers, with each 4-bit expert sized at ~150 MB:
* Activation per layer: $8 	imes 150\text{ MB} = 1.2\text{ GB}$ of weights.
* In a worst-case scenario with a 0% cache hit rate, generating a single token requires reading:
  $$60 \times 1.2\text{ GB} = 72\text{ GB per token}$$
* Even on an ultra-fast PCIe Gen4 NVMe array delivering an aggregate bandwidth of 14 GB/s, reading 72 GB takes:
  $$\frac{72\text{ GB}}{14\text{ GB/s}} \approx 5.14\text{ seconds per token (0.19 tok/s)}$$

### Engineering Mitigations
1. **Enforce High Hit-Rate Floor (>75–85%):**
   * The hardware target possesses **96 GB of VRAM and 64 GB of Host DDR** (aggregate ~150 GB usable RAM).
   * Models must be partitioned so that all dense attention layers, shared experts, and high-frequency routed experts reside permanently in fast memory.
   * If a model's activation entropy is completely uniform across all experts, pure streaming is fundamentally unviable; Aeon must detect and warn against models lacking activation locality.
2. **Never Stream Synchronously from SSD for Immediate Next Layer:**
   * Tier 3 (SSD) data transfers must be strictly speculative and asynchronous. If an expert is completely absent from both VRAM and DDR during the active layer pass, the system must trigger host CPU computation fallback rather than stalling the GPU pipeline.

---

## 2. Blocker 2: Cross-Layer Prefetch Prediction Horizons

### Mechanism of Failure
In standard transformer architectures, the router for Layer $L+1$ computes expert activation probabilities using the hidden state generated at the output of Layer $L$:
* The identity of required experts for Layer $L+1$ is **physically unknown** until Layer $L$ finishes computing.
* Layer $L$ compute duration on an RX 7900 XTX is roughly **0.5 to 1.5 ms**.
* NVMe storage read latency typically ranges between **0.2 to 0.8 ms** before sustained I/O throughput begins.
* A single-layer look-ahead horizon ($L \to L+1$) is insufficient to mask the combined I/O and PCIe transfer latency of a cold SSD miss.

### Engineering Mitigations
1. **Multi-Layer Speculative Prefetching:**
   * Track empirical expert co-activation matrices and sequence-level activation histories to predict expert selection 2–4 layers ahead of the current execution frontier.
2. **Dynamic Host CPU Coprocessing ($q^*$ Fallback):**
   * Incorporate the [FreeToken](../../aeon-references/freetoken) coprocessing paradigm: when an expert is missing from GPU memory, dispatch the token hidden state to the Threadripper Pro host CPU via pinned memory.
   * Compute the missing expert's forward pass using AVX-512 while the GPU continues executing resident experts, merging the output tensors prior to the residual addition.

---

## 3. Blocker 3: Linux Page Cache Contention and `io_uring` Alignment

### Mechanism of Failure
Standard file operations (`mmap`, `read`, `fread`) route I/O through the Linux kernel's page cache:
* Streaming tens of gigabytes per minute induces massive page cache eviction churn, driving CPU utilization to 100% on memory management overhead.
* Frequent Translation Lookaside Buffer (TLB) shootdowns stall execution threads.
* Double-buffering incurs redundant memory copies: NVMe $\to$ Kernel Page Buffer $\to$ User Space $\to$ Pinned Memory $\to$ GPU VRAM.

### Engineering Mitigations
1. **Linux `io_uring` with `O_DIRECT`:**
   * Bypass the OS kernel page cache entirely.
   * Pre-register memory buffers using `io_uring_register_buffers()` to eliminate page-pinning overhead on every I/O transaction.
2. **Strict Storage Sector Alignment in Binary Formats:**
   * `O_DIRECT` requires file offsets, memory addresses, and read lengths to be strictly aligned to physical storage block boundaries (typically 4096 bytes).
   * The offline layout conversion tool (`prepare_rdna.py`) must inject deterministic padding between tensor headers and expert weights to guarantee 4KB boundary alignment.

---

## 4. Blocker 4: Consumer ROCm Driver Latency and Transfer Jitter

### Mechanism of Failure
Unlike enterprise-grade compute platforms, consumer RDNA3 implementations under ROCm can exhibit driver-level serialization and latency spikes:
* Issuing hundreds of small `hipMemcpyAsync` calls across multiple HIP streams can trigger lock contention and internal mutex bottlenecks inside `libamdhip64.so`.
* If a background DMA transfer queue contends with the primary compute queue executing matrix kernels, compute execution can stall intermittently.

### Engineering Mitigations
1. **Coarse-Grained Memory Staging:**
   * Avoid fine-grained, per-tensor allocations. Pre-allocate large contiguous memory arenas (512 MB to 2 GB slabs) at engine initialization.
2. **Dedicated Hardware SDMA Engine Utilization:**
   * Route memory traffic explicitly through hardware System DMA (SDMA) copy engines rather than relying on driver-managed compute-queue transfers.
   * Implement explicit ring-buffer double-buffering to ensure compute and data transfer channels operate completely orthogonal to one another.

---

## 5. Blocker 5: Quantization Dequant Overhead on RDNA3 WMMA

### Mechanism of Failure
Enterprise compute accelerators (e.g., NVIDIA H100, AMD MI300X) feature specialized hardware units capable of ingesting packed sub-byte formats directly.
* Consumer RDNA3 (`gfx1100`) WMMA hardware natively supports standard formats (FP16, BF16, INT8, INT4), but lacks direct hardware execution for non-standard quantization schemes (e.g., GGML `Q4_K_M`, `Q5_K_S`).
* If custom HIP kernels must execute extensive arithmetic unpacking, shift logic, and scale interpolations in software, register pressure spikes, thread occupancy drops, and memory bandwidth is wasted on decompression rather than tensor math.

### Engineering Mitigations
1. **Pre-Swizzled Hardware-Aligned Layouts:**
   * Quantized formats must match RDNA3's Wave32 register layout during the offline conversion stage.
   * Pack INT4 nibbles such that simple vector shift and mask instructions yield WMMA-compliant input registers without requiring intermediate register spilling.
2. **Fused Dequantization-GEMM Kernels:**
   * Fuse dequantization directly into the shared memory load phase of the matrix multiplication kernel, avoiding extra global memory round-trips.

---

## 6. Risk and Mitigation Summary Matrix

| Failure Mode | Severity | Likelihood | Impact on System | Direct Engineering Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **The Colibri Trap (Synchronous Cold Misses)** | **Critical** | Medium | GPU idle 95% of time; throughput collapses to < 1 tok/s. | Maintain >80% hit rate in 150 GB VRAM+DDR; never block synchronously on SSD. |
| **Short Prefetch Window ($L \to L+1$)** | **High** | High | Storage latency exceeds layer compute duration. | Speculative multi-layer look-ahead + Host CPU AVX-512 ($q^*$) fallback. |
| **ROCm Driver Async Transfer Jitter** | **Medium** | High | Background DMA transfers serialize compute kernels. | Dedicated hardware SDMA queues + coarse-grained memory arenas. |
| **`io_uring` Direct I/O Alignment Failures** | **High** | Medium | `EINVAL` returns cause fatal fallback to slow buffered I/O. | Strict 4096-byte padding enforced by offline converter format. |
| **RDNA3 Register Spilling on Dequant** | **Medium** | Low | High register pressure lowers occupancy and cuts FLOPs. | Pre-swizzle weights offline to align directly with Wave32 WMMA instructions. |
