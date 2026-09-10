# Project Aeon: RDNA-Native MoE Inference Engine
*Architectural Vision, Strategic Roadmap, and System Principles*

*Status: strategic vision. This document records long-term principles and aspirations, including multi-GPU scaling and future runtime capabilities; it is not a current implementation checklist. Use [DOCUMENTATION_STATUS.md](../../status/DOCUMENTATION_STATUS.md) and the active execution plans for present-state evidence.*

---

## 1. Executive Summary & Problem Statement

Modern open-weights Large Language Models have converged heavily on **Mixture-of-Experts (MoE)** architectures (e.g., DeepSeek, Qwen, and MiniMax families). While these models achieve frontier-grade intelligence, their parameter counts—ranging from 100B to over 600B—place them well out of reach of standard consumer hardware when evaluated with conventional serving frameworks.

Today's inference landscape presents a severe structural divide:
- **Enterprise Datacenter Engines** (*vLLM, SGLang, TensorRT-LLM*) are heavily optimized for massive clusters of NVIDIA enterprise accelerators interconnected via multi-hundred-gigabyte NVLink fabrics. Consumer AMD Radeon (RDNA) architectures are treated as secondary or unsupported.
- **Universal Edge Engines** (*llama.cpp*) prioritize cross-platform portability across dozens of hardware targets (CPUs, mobile chips, Apple Silicon, diverse GPUs). Consequently, they cannot make deep, architecture-specific micro-optimizations or adopt aggressive asynchronous memory hierarchies.
- **Specialized Research Runtimes** (*FreeToken, KTransformers*) pioneer dynamic weight-streaming paradigms, but remain tightly bound to NVIDIA CUDA primitives, proprietary kernel libraries, and single-GPU topologies.

**Project Aeon** is an initiative to build a dedicated, greenfield inference engine written from scratch exclusively for **consumer AMD RDNA3 (and future RDNA-family) hardware**. By discarding legacy vendor abstractions, Aeon leverages the distinct silicon traits of RDNA3—such as Wave32 execution, AI matrix accelerators (WMMA), and large Infinity Caches—to enable running frontier MoE models on consumer setups.

---

## 2. Core Vision & Hardware Target Philosophy

### Hardware Accessibility with Elastic Scaling
The guiding principle of Project Aeon is **broad consumer accessibility with zero-overhead upward scalability**:
1. **The Baseline Target:** A standard consumer PC equipped with a **single AMD Radeon RX 7000-series GPU (e.g., 7900 XT/XTX or 7800 XT)**, a standard desktop CPU, consumer DDR5/DDR4 system memory, and a single NVMe SSD. On this baseline, the engine must deliver smooth, interactive token generation for MoE models that substantially exceed the card's onboard VRAM.
2. **Elastic Scaling:** When deployed on higher-tier workstations—featuring multi-GPU configurations, high-core CPUs with abundant PCIe lanes (e.g., AMD Threadripper platforms), multi-channel system RAM, and multi-drive NVMe arrays—the runtime must scale its throughput linearly without code modifications or architectural changes.

---

## 3. Inspirations & Cross-Engine Concept Synthesis

Project Aeon synthesizes proven paradigms from leading inference runtimes into a cohesive, RDNA-tailored runtime:

| Project | Key Concept Adopted | Strategic Value to Aeon |
| :--- | :--- | :--- |
| [**FreeToken**](../../../../aeon-references/freetoken) | **Dynamic Expert Caching & Bandwidth-Adaptive Scheduling ($q^*$)** | Treating VRAM not as a static weight container, but as an active dynamic cache for sparse MoE experts, while offloading excess transfer demands to host compute. |
| [**SGLang**](../../../../aeon-references/sglang) | **RadixTree KV State Tracking** | Preserving token prefix state hierarchically across multi-turn interactions and agentic workflows, eliminating redundant prompt evaluation passes. |
| [**vLLM**](../../../../aeon-references/vllm) | **Paged Memory Management & Asynchronous Scheduling** | Non-contiguous memory allocation for attention contexts to eliminate VRAM fragmentation and enable deterministic execution queues. |
| [**Colibri**](../../../../aeon-references/colibri) & "LLM in a Flash" | **Direct-I/O Asynchronous SSD Streaming** | Treating high-speed NVMe flash memory as an active third tier in the memory hierarchy via OS-bypass direct I/O interfaces. |
| [**DwarfStar (ds4)**](../../../../aeon-references/ds4) | **DeepSeek-V4-specific kernels, profile-derived expert hotlists & agent serving** | Comparing model-specific routing placement, SSD expert streaming, KV/prefix reuse, and native tool-using workflows against Aeon's specialized runtime. |
| **DeepSeek (DS4 / DualPipe)** | **Multi-Head Latent Attention (MLA) & Overlapped Dispatch** | Native handling of compressed latent KV representations and concurrent dispatch/compute synchronization pipelines. |
| [**llama.cpp**](../../../../aeon-references/llama.cpp) / Unsloth | **Quantization Modalities & Container Ecosystem** | Ingesting widely adopted, community-curated low-bit quantization layouts while maintaining structural independence from runtime compute implementations. |

---

## 4. Architectural Pillars

### A. Three-Tier Hierarchical Storage Architecture
Rather than viewing model execution through a binary "fits in VRAM vs. runs on CPU" lens, Aeon models system resources as an active three-tier streaming continuum:
1. **Tier 1 (Hot - VRAM):** Hosts permanent core structures (dense attention, router networks, shared experts, and active KV cache blocks) alongside a dynamically managed LRU pool for routed experts.
2. **Tier 2 (Warm - Host System RAM):** Pinned, page-locked staging buffers holding secondary expert candidates and prefetch queues.
3. **Tier 3 (Cold - NVMe Flash Storage):** Houses the long-tail repository of inactive experts, ingested via asynchronous, OS-bypass Direct I/O (`io_uring`).

Data flows uni-directionally from Cold to Warm to Hot, ensuring that the GPU compute engine interacts solely with high-speed memory spaces while transfer operations run fully in the background.

### B. Decoupled Prefill and Decode Pipelines
Recognizing that prompt evaluation (prefill) and token generation (decode) exhibit opposing operational profiles:
- **Prefill Optimization:** Addresses the dense nature of prompt processing through double-buffered streaming rings. Asynchronous DMA transfers continuously stage upcoming layer weights while the matrix cores compute the current layer, masking streaming latency behind computational work.
- **Decode Optimization:** Exploits token-level MoE sparsity through localized LRU hit tracking, fetching missed experts on non-blocking transfer channels and dynamically dispatching overflow compute to idle CPU cores when transfer queues saturate.

### C. Native Pipeline Parallelism & Hybrid Topologies
For configurations with multiple GPUs, Aeon prioritizes **Pipeline Parallelism (PP)** over Tensor Parallelism. In the absence of proprietary high-speed inter-GPU links (such as NVLink), Pipeline Parallelism keeps communication minimal—passing only small intermediate activation vectors across PCIe slots while giving subsequent pipeline stages natural look-ahead time to stage upcoming expert weights into local VRAM.

### D. RDNA3 Microarchitectural Focus
Aeon intentionally foregoes generic compute abstractions in favor of direct alignment with AMD’s Navi 31/32 hardware characteristics:
- Native dispatch targeting **Wave32** wavefront execution to eliminate register spilling.
- Direct targeting of RDNA3 **AI Matrix Accelerators (WMMA)** for mixed-precision and low-bit matrix operations.
- Structuring activation and routing operations to maximize hit rates inside RDNA3's dedicated **on-die Infinity Cache**.

---

## 5. Deep-Dive: Distributed Topologies & Parallelism Strategy

A critical design challenge in multi-GPU consumer environments is choosing between **Tensor Parallelism (TP)**, **Expert Parallelism (EP)**, and **Pipeline Parallelism (PP)**. Aeon establishes a clear, mathematically grounded strategy:

### Why Pure Tensor Parallelism (TP) Is Deprioritized on Consumer Hardware
Tensor Parallelism slices every individual weight matrix across all devices, requiring multiple synchronized `All-Reduce` collective operations on every single transformer block. In enterprise servers, this relies on multi-hundred-gigabyte NVLink interconnects. On consumer PCIe platforms:
- **Bus Contention:** Running low-latency `All-Reduce` collectives across PCIe slots while concurrently streaming missing expert weights from host RAM or SSD creates severe queue congestion.
- **Shard Fragmentation:** In 4-way TP, an 80 MB expert is fragmented into four 20 MB slices, transforming clean, contiguous DMA transfers into fragmented multi-device synchronizations.

### Expert Parallelism (EP) vs. Pipeline Parallelism (PP): Trade-Offs & Realities
- **The Pitfall of Global EP in Layer Pipelines:** Slicing experts globally across GPUs while distributing layers sequentially creates cross-device routing stalls. If a layer executing on GPU 0 requires an expert assigned to GPU 3, intermediate token activations must halt mid-layer to execute over PCIe, collapsing pipeline throughput.
- **The Single-Token Straggler Effect:** In single-user decode ($M=1$), tokens select only a few experts ($k pprox 6	ext{--}10$). Due to non-uniform routing, one GPU may receive multiple expert assignments while others receive zero or one, causing 75% of the hardware to idle waiting on the slowest card (the straggler penalty).
- **The Advantage of Pure Pipeline Parallelism (PP):**
  - **Zero Inter-Layer Contention:** Only a tiny intermediate activation vector (kilobytes) passes between cards at stage boundaries.
  - **Zero Weight Duplication:** Experts belong strictly to specific layers. GPU 0 only caches experts for its assigned layers (e.g., Layers 0–14); GPU 1 only caches experts for Layers 15–29. Each card's local LRU VRAM pool is effectively $4	imes$ deeper, dramatically increasing local cache hit rates without duplicating a single byte.
  - **Look-Ahead I/O Masking:** While GPU 0 computes its stage, downstream GPUs use their execution bubble to pull predicted missing experts from NVMe/RAM into VRAM, completely masking I/O latency.
  - **1F1B Micro-Batching:** When serving concurrent tokens, speculative verification streams, or micro-batches, the pipeline bubble collapses, driving all GPUs to full concurrent saturation.
- **The Evolutionary Path: Localized Hybrid PP $	imes$ EP:** For larger multi-GPU clusters, Aeon supports clustering cards into stages (e.g., 2 pipeline stages of 2 GPUs each). Inside each stage, GPUs share attention layers and partition the stage's expert pool via localized high-speed peer-to-peer (P2P) transfers, avoiding global cross-pipeline coordination.

---

## 6. Bare-Metal AMD Toolchain: Beyond the "CUDA Monopoly"

A foundational premise of Project Aeon is that **GPUs execute machine code (ISA), not CUDA.** The widespread belief that deep learning requires CUDA translation layers is an ecosystem artifact, not a hardware constraint.

### Native Execution Architecture
- **No Translation or Emulation:** Aeon bypasses CUDA entirely. Code is authored in native C++/HIP and compiled via AMD's official LLVM compiler toolchain (`hipcc`/Clang) directly into native RDNA3 machine instructions targeting the `gfx1100` architecture.
- **Direct Silicon Primitives:** The runtime invokes RDNA3-native matrix hardware instructions directly (e.g., `__builtin_amdgcn_wmma_*`), matching AMD's specific 32-lane wave vector structure.
- **Complementary Tooling:** In addition to hand-tuned C++/HIP kernels, Aeon leverages AMD Composable Kernel (CK) building blocks and OpenAI Triton's native AMDGCN backend, ensuring peak matrix utilization without proprietary NVIDIA dependencies.

---

## 7. Storage Tier Scaling: Asynchronous NVMe Striping

To extend memory headroom beyond physical VRAM and host RAM, Aeon incorporates flash storage directly into the execution graph:

- **Striping Over Duplication:** Rather than holding redundant full model copies on multiple drives (which leaves individual expert transfer speeds capped at single-drive limits), Aeon stripes expert data across available NVMe drives.
- **Hardware-Level Sector Alignment:** All expert blocks are mapped to exact 4,096-byte (4KB) boundaries, matching NVMe logical block formats and enabling true zero-copy OS-bypass direct I/O via Linux `io_uring` with `O_DIRECT`.
- **Parallel Channel Ingestion:** With multi-drive configurations on platforms offering dedicated PCIe lanes (such as AMD Threadripper Pro), independent `io_uring` instances stream separate expert payloads in parallel, driving aggregate flash bandwidth into the 20–28 GB/s regime—nearing DDR-channel speeds.

---

## 8. Host Compute Coprocessing: The Bandwidth-Adaptive Slow-Path

Rather than letting host CPUs sit idle during GPU execution, Aeon implements a bandwidth-adaptive compute policy inspired by FreeToken's $q^*$ formulation:

- **PCIe Congestion Bypass:** When a layer encounters multiple cache misses simultaneously, attempting to push all missing weights across the PCIe bus can queue-block the GPU's compute stream.
- **CPU AVX Coprocessing:** High-priority misses stream to the GPU LRU cache, while overflow misses execute directly in host RAM via the host CPU threadpool (leveraging AVX-512 vector units).
- **Elementwise Convergence:** The CPU computes its assigned expert GEMMs in-place out of host DDR memory and returns only the final hidden-state delta vector (a few kilobytes) over PCIe, where it is summed with the GPU's output. This dynamic load-balancing smooths out tail latencies and maximizes total platform compute.

---

## 9. Ecosystem & Quantization Strategy

To ensure day-one utility and prevent ecosystem isolation:
- **No Proprietary Quantization Format:** Aeon rejects the overhead of developing an isolated quantization format. It adopts standard low-bit modalities widely distributed across the open-source ecosystem (e.g., GGUF, Unsloth dynamic quants).
- **Offline Structural Reorganization:** Aeon introduces a lightweight preprocessing workflow (`prepare_rdna.py`). This utility inspects standard model distributions, deconstructs monolithic MoE tensors into discrete, individually addressable expert blocks, aligns them to 4KB storage hardware sectors, and reorganizes weight memory layouts to match RDNA3 Wave32 register swizzling.
- **Turnkey Integration:** Users retain access to public model hubs and existing quantization pipelines while benefiting from hardware-specific data ordering at runtime.

---

## 10. Strategic Scope & Boundaries

### What Aeon Is:
- A high-performance, single-node inference engine engineered for local AI practitioners, researchers, and personal agentic workflows.
- A platform optimized to deliver interactive, low-latency generation speeds on models previously considered unrunnable on consumer hardware.
- A lean, hardware-specialized runtime designed to maximize efficiency per dollar on AMD consumer silicon.

### What Aeon Is Not (Initial Non-Goals):
- **Massive Multi-Tenant Cloud Server:** Aeon will not prioritize hyperscale concurrency or thousands of simultaneous user streams, which fundamentally conflict with sparse weight-streaming techniques.
- **Cross-Vendor Compatibility Layer:** Aeon will make zero architectural compromises to accommodate NVIDIA CUDA, Intel XPU, or mobile SoC constraints.
- **A Model Training Framework:** The engine's runtime primitives are exclusively optimized for forward-pass evaluation and generation.

---

## 11. Phased Project Roadmap

1. **Phase 1: Single-GPU Core Runtime**
   - Implement the foundational asynchronous memory manager and direct storage ingestion interface (`io_uring`).
   - Develop native RDNA3 low-bit GEMM execution primitives targeting Wave32 WMMA instructions.
   - Establish the dynamic LRU expert cache for single-GPU offloaded execution.

2. **Phase 2: Advanced Memory Hierarchy & Prefill Pipeline**
   - Integrate double-buffered layer prefetching to optimize Time-To-First-Token (TTFT).
   - Implement hierarchical prefix caching (RadixTree) to accelerate iterative agentic interactions.
   - Deploy bandwidth-adaptive host CPU compute routing ($q^*$) for cache-miss load balancing.

3. **Phase 3: Multi-Hardware Elastic Scaling**
   - Implement low-overhead PCIe Pipeline Parallelism across multi-GPU topologies with 1F1B micro-batching.
   - Add multi-channel NVMe sharding to support striping across multiple storage devices.
   - Provide automated system profiling to calibrate cache boundaries and transfer queues to any detected hardware configuration.