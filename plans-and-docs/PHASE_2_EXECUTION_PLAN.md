# Phase 2: Single-GPU 3-Tier Storage & Memory Hierarchy Optimization
*Architecture Target: DeepSeek-V4-Flash-0731 (INT4-W4A16 on AMD RDNA3 / gfx1100)*

---

## 1. Executive Summary & Architectural Rationale

Following the silicon verification of Phase 1 (Spikes 1–6), where all mathematical primitives, fused INT4 WMMA kernels, 4-stream Hyper-Connections Sinkhorn, sliding-window attention with sink, and multi-layer state chaining were proven on physical AMD Radeon RX 7900 XTX hardware, **Phase 2 focuses strictly on maximizing the physical I/O and memory throughput of the single-GPU inference runtime**.

As revealed in our empirical performance analysis ([plans-and-docs/PERFORMANCE_LEDGER.md](plans-and-docs/PERFORMANCE_LEDGER.md)):
- Fused INT4 compute time is only **$140\ \mu\text{s}$ per expert** ($0.84\text{ ms}$ for 6 active experts).
- Memory transfers over PCIe 4.0 x16 ($81\text{ MB}$ per token layer) dominate latency if accessed naively or synchronously.
- DeepSeek MLA's single compressed KV head ($head\_dim=512$) requires only **$1.34\text{ GB}$ for a 32,768-token context across all 43 layers**, leaving nearly **$12\text{ GB}$ of VRAM available for dynamic expert caching**.

Rather than prematurely distributing an unoptimized single-GPU engine across multiple cards, Phase 2 implements and rigorously benchmarks the full **3-Tier Storage & Memory Hierarchy** on a single card:
$$\text{Tier 1: Hot VRAM Pool (LRU)} \longleftrightarrow \text{Tier 2: Warm Pinned Host DDR} \longleftrightarrow \text{Tier 3: Cold NVMe SSD (io\_uring Direct I/O)}$$

Phase 2 is partitioned into four distinct, decoupled Spikes:
0. **Spike 0**: Surgical Safetensors-to-`.aeon` Model Repacking & Weight Verification.
1. **Spike 1**: Dynamic VRAM Budgeting & Global Unified Expert Pool.
2. **Spike 2**: Dual-Stream Asynchronous SDMA Prefetching & Latency Hiding.
3. **Spike 3**: Linux `io_uring` Direct I/O NVMe Cold Tier Integration.

---

## 2. Micro-Spike Breakdown & Verification Gates

### Spike 0: Surgical Safetensors-to-`.aeon` Model Repacking & Weight Verification
*Objective: Unbundle and isolate the dense backbone from the 11,008 routed experts into strictly 4096-byte sector-aligned binary containers, ensuring production-ready layouts and 100% bit-exact numerical parity before cache engine optimization.*

- **Micro-Step 0.1: Surgical Model Splitting & Sector-Aligned Serializer (`scripts/prepare_rdna.py`)**
  - Ingest the 34 sharded Safetensors files of `DeepSeek-V4-Flash-0731-INT4-W4A16`.
  - Isolate all dense weights into `model_dense.aeon` (~9.24 GB): Attention projections ($W_q, W_{kv}, W_o$), RMSNorms, Hyper-Connections Sinkhorn tables, Shared Experts, and Router gate weights.
  - Isolate the 11,008 routed experts (43 layers $\times$ 256 experts) into `model_experts.aeon` (~135 GB): Each expert FFN ($W_1, W_2, W_3$ packed INT4 + FP16 scales) is written as an isolated contiguous block starting at a strictly 4096-byte aligned file offset (`O_DIRECT` compliant).
  - Generate a compact binary index table `model_experts.index` mapping `(layer_id, expert_id)` to `(uint64_t file_offset, uint64_t byte_length)`.
- **Micro-Step 0.2: Bit-Exact Numerical Verification Suite**
  - Implement a verification test comparing the parsed `.aeon` weights against the original Safetensors weights on silicon.
  - Validate that every INT4 nibble and FP16 scale is 100% bit-identical with zero precision loss ($\epsilon = 0.0$).
  - *Verification:* Silicon test confirming bit-level identical tensor hashes across dense components and sample routed experts.

---

### Spike 1: Dynamic Memory Budgeting & Global Unified VRAM Expert Pool
*Objective: Eliminate rigid per-layer slot allocations; maximize cache hit rate under real Zipfian MoE activation entropy by dynamically sharing VRAM capacity across layers.*

- **Micro-Step 1.1: Exact Memory Budgeting & Dynamic Slot Allocator**
  - Allocate the 32k context KV Cache buffer upfront ($1.34\text{ GB}$).
  - Permanently pin all dense weights (attention projections, RoPE, Sinkhorn, shared experts, LM head) in VRAM ($9.24\text{ GB}$).
  - Dynamically calculate remaining VRAM and allocate a **Global Unified Expert Slot Pool** ($\approx 880$ INT4-W4A16 slots, $\approx 11.9\text{ GB}$).
- **Micro-Step 1.2: Global Multi-Layer LRU Eviction & Mapping Policy**
  - Implement a thread-safe global LRU cache index tracking `(layer_id, expert_id)` pairs to physical VRAM slot indices.
  - Allow layers with high activation frequency or lower entropy to dynamically hold more slots than inactive layers.
  - *Verification:* Silicon test comparing cache hit rate of global pool vs. static 8-slot per-layer baseline on real token sequences. Log hit rates and step latencies in [plans-and-docs/PERFORMANCE_LEDGER.md](plans-and-docs/PERFORMANCE_LEDGER.md).

---

### Spike 2: Dual-Stream Asynchronous SDMA Prefetching & Latency Hiding
*Objective: Overlap compute and PCIe transfers so that cold-miss expert streaming runs concurrently with active attention and resident expert execution.*

- **Micro-Step 2.1: Lookahead Routing & Prefetch Horizon Pipeline**
  - While Layer $L$ is executing its attention and resident shared expert pass, trigger routing calculation for Layer $L+1$.
  - Identify missing experts for Layer $L+1$ ahead of execution time.
- **Micro-Step 2.2: Double-Buffered Asynchronous SDMA Transfer Stream**
  - Dispatch non-blocking PCIe DMA transfers on a dedicated HIP SDMA stream concurrently with Layer $L$'s compute stream.
  - Synchronize via HIP events (`hipEventRecord`, `hipStreamWaitEvent`) immediately before Layer $L+1$'s routed MoE execution.
  - *Verification:* Measure PCIe transfer overlap efficiency using `hipEventElapsedTime`; verify that $>80\%$ of PCIe transfer latency is hidden behind compute without compute kernel jitter.

---

### Spike 3: Linux `io_uring` Direct I/O NVMe Cold Tier Integration
*Objective: Complete the 3-tier chain by connecting cold NVMe SSD storage directly to Host DDR staging via Linux `io_uring` with zero kernel page-cache contention.*

- **Micro-Step 3.1: Linux `io_uring` Direct I/O Reader Integration (Tier 3 $\to$ Tier 2)**
  - Integrate `src/io/direct_io_reader.hpp` into the runtime pipeline targeting `model_experts.aeon`.
  - Implement asynchronous direct streaming of cold experts from NVMe into pinned host DDR staging buffers with `O_DIRECT`.
  - Maintain a dynamic Tier 2 warm cache in host RAM ($\approx 48\text{ GB}$, $\approx 3,550$ warm experts) feeding Tier 1 VRAM.
- **Micro-Step 3.2: End-to-End 3-Tier Pipeline Validation**
  - Silicon test measuring end-to-end 3-tier streaming throughput from NVMe $\to$ Host RAM $\to$ GPU VRAM, validating data integrity and measuring throughput in GB/s.

---

## 3. Milestone Verification & Success Criteria

1. **VRAM Hit Rate**: Global dynamic VRAM pool achieves $\ge 40\%$ hit rate (up from $2.4\%$) on multi-token sequences.
2. **Transfer Latency Hiding**: Asynchronous SDMA prefetching hides $\ge 80\%$ of PCIe transfer time behind attention/shared-expert compute.
3. **NVMe Direct I/O**: Direct I/O reader streams cold experts from NVMe at $\ge 6.0\text{ GB/s}$ directly into pinned host buffers.
4. **Empirical Ledger Logging**: Update [plans-and-docs/PERFORMANCE_LEDGER.md](plans-and-docs/PERFORMANCE_LEDGER.md) at the end of each spike with exact before-and-after throughputs, latencies, and cache statistics.
