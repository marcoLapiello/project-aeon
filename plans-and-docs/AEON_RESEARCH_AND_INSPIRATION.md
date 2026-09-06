# Project Aeon: Global Competitive Landscape & Prior Art Survey

## Executive Summary
This document synthesizes findings from a comprehensive global survey into the state of local LLM inference engines, focusing on two distinct technical branches:
1. **AMD / RDNA-Native Inference Engines** (bare-metal ISA, bypassing generic wrappers and CUDA-centric layers).
2. **MoE Expert Streaming & Tiered Caching** (streaming weights across VRAM, Host DDR, and NVMe SSDs via direct I/O).

The primary takeaway: The broader open-source ecosystem is independently solving individual pieces of this problem, but remains deeply fragmented. Project Aeon stands to become the first engine to unify custom RDNA3 execution with a multi-tier, look-ahead asynchronous streaming runtime.

---

## 1. AMD & RDNA-Focused Inference Engines

Historically, non-NVIDIA inference has relied on translation layers (ZLUDA), heavy general-purpose frameworks (PyTorch/ROCm), or monolithic multi-backend engines (llama.cpp/GGML). Recently, independent engineering efforts have emerged to bypass these abstractions entirely.

### 1.1 hipfire (`hipfire.dev`)
* **Repository / Project:** Standalone binary written in Rust with hand-tuned HIP C++ kernels.
* **Architecture:**
  * Uses direct dynamic loading (`dlopen`) of AMD’s runtime library (`libamdhip64.so`), stripping out PyTorch and Python runtime overhead entirely.
  * Maintains an extensive library of ~300+ custom HIP kernels tuned per AMD ISA target (`gfx1100` for RX 7900 series, `gfx1151` for Strix Halo, `gfx1201` for RDNA4).
  * Optimizes memory allocation patterns and tensor layouts directly for RDNA Wave32 dual-issue compute units.
* **Performance:** Reports 1.7× to 2.1× higher throughput compared to generic Ollama/ROCm distributions on consumer RX 7900 XTX hardware.
* **Limitations / Gaps:** Designed strictly for in-memory execution. Does not feature dynamic SSD offloading, NVMe streaming, or sparse MoE tiering.

### 1.2 zinc (`github.com/zolotukhin/zinc`)
* **Repository / Project:** LLM inference engine written entirely in Zig.
* **Architecture:**
  * Targets Apple Silicon, Vulkan, and AMD ROCm/HIP with minimal static binaries.
  * Completely removes the legacy architectural cruft of GGML/llama.cpp to prioritize decode latency and fast cold-starts.
* **Limitations / Gaps:** Classical monolithic memory model; does not address sub-tensor or expert-level dynamic scheduling.

### 1.3 ROCmFPX & hipEngine
* **Focus:** Custom low-bit quantization layouts tailored to AMD compute.
* **Relevance:** Confirms that generic GGUF bit-packing structures waste instruction cycles on RDNA during dequantization. Highlights the necessity of pre-swizzled weight layouts tailored to RDNA3 WMMA (Wave Matrix Multiply-Accumulate) instructions.

---

## 2. Dynamic MoE Streaming & Multi-Tier Caching

Running large Mixture-of-Experts (MoE) models on memory-constrained systems has spurred innovative caching and offloading experiments.

### 2.1 llama.cpp PR #25294: Stream MoE Routed Experts from Disk
* **Author / Reference:** `freedomljc` (Open-source contribution to `llama.cpp`).
* **Technical Innovation:**
  * Implements an asynchronous worker pool that streams only active routed experts on-demand from NVMe storage.
  * Bypasses OS page cache using Linux `O_DIRECT` to eliminate double-buffering and CPU cache thrashing.
  * Introduces "Wave-Partitional Prefill" to chunk long context sequences so that prompt prefill remains viable even when weights exceed memory.
* **Reported Benchmarks:** Yielded up to **5.3× prefill speedups** and **2.4× decode speedups** over standard OS-level `mmap` offloading.
* **Limitations / Gaps:** Single-device focus. Multi-GPU pipelining and true 3-tier staging (NVMe $	o$ DDR $	o$ VRAM) were left as unaddressed future work.

### 2.2 Flash-MoE & TinyGiant (llama.cpp Discussion #27149)
* **Focus:** Selective, expert-aware sparse I/O.
* **Technical Insight:**
  * Demonstrated that reading only 8 active experts out of 128 yields an **11.4× reduction in I/O bandwidth demand** compared to reading monolithic layer blocks.
  * Successfully demonstrated execution of a 397B MoE model on a 48GB workstation at ~4.4 tokens/second.
* **Limitations / Gaps:** Flash-MoE is tightly coupled to Apple Silicon unified memory (Unaligned Metal buffers and unified memory controller) and does not map directly to discrete multi-GPU PCIe environments.

### 2.3 Micro-Expert-Router & MoEpic (arXiv:2509.08342)
* **Focus:** Tiered expert caching and vertical tensor splitting.
* **Technical Innovation:**
  * *Micro-Expert-Router* prototyped Rust-based `io_uring` fixed-buffer direct I/O for expert extraction.
  * *MoEpic* introduced vertical splitting: keeping high-salience expert layers permanently in VRAM while streaming low-salience components dynamically.

---

## 3. Comparative Architecture Matrix

| Capability / Architecture | hipfire / zinc | llama.cpp PR #25294 | Flash-MoE | **Project Aeon** |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Language & Runtime** | Rust / Zig (Lean runtime) | C++ (GGML monolithic) | Swift / Metal | **C++20 / Direct HIP & LLVM** |
| **Target GPU Architecture** | Dedicated AMD RDNA3/4 | Broad / Generic | Apple Silicon | **Dedicated AMD RDNA3 (`gfx1100`)** |
| **Low-Level Compute Kernels** | Handwritten Wave32 HIP | Generic GGML kernels | Metal Shaders | **Wave32 WMMA + Pre-swizzling** |
| **Direct NVMe I/O Engine** | None (In-memory only) | `O_DIRECT` worker pool | OS mmap | **Linux `io_uring` with registered buffers** |
| **3-Tier Hierarchy (VRAM/RAM/SSD)** | No | No (Disk $	o$ RAM/VRAM) | No (Unified RAM) | **Yes (Dynamic hot/warm/cold pools)** |
| **Multi-GPU Pipelining** | Basic sequential | No (Single GPU only) | No (Single SoC only) | **Yes (1F1B + Async Look-Ahead)** |
| **Host CPU Coprocessing ($q^*$)** | No | No | No | **Yes (AVX-512 fallback for cache misses)** |

---

## 4. Conclusion & Strategic Positioning

The state of the art confirms the foundational thesis of Project Aeon:
1. **The compute bottleneck is solved by native AMD kernels:** Projects like *hipfire* prove that stripping away CUDA translation layers and writing directly to RDNA3 primitives produces immediate 1.7×–2.1× performance gains.
2. **The memory bottleneck is solved by sparse expert streaming:** Projects like *llama.cpp PR #25294* and *Flash-MoE* prove that dynamic expert selection reduces bandwidth pressure by an order of magnitude.

**Aeon's Unique Niche:** Unifying these two breakthroughs into a production-grade, distributed pipeline engine designed specifically for multi-GPU AMD consumer workstations.
