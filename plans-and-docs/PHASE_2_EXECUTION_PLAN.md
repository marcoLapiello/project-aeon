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

### Spike 1: Dynamic Memory Budgeting, Feasibility Gating & Global Unified VRAM Expert Pool
*Objective: Eliminate rigid per-layer slot allocations; enforce strict startup hardware feasibility gating; maximize cache hit rate under real Zipfian MoE activation entropy by dynamically sharing VRAM capacity across all 43 layers.*

- **Micro-Step 1.1: Runtime Configuration & Hard Feasibility Gate (`src/core/memory_budget.hpp`)**
  - Define `AeonRuntimeConfig` accepting user-specified `context_size` ($T \in [1, \text{max\_position\_embeddings}]$) and `host_ram_bytes`.
  - Enforce internal safety constraints:
    - Fixed VRAM headroom: $\mathbf{300\text{ MB}}$ to prevent OS desktop compositor/GTT memory migration.
    - Fixed Host RAM safety cap: $\mathbf{80\%}$ of physical system RAM (`sysinfo` / `sysconf`).
  - Calculate required VRAM components:
    $$\text{VRAM}_{\text{kv}} = T \times L_{\text{layers}} \times d_{\text{kv}} \times 2\text{ bytes}$$
    $$\text{VRAM}_{\text{min\_active}} = 2 \times K \times \text{AEON\_EXPERT\_BYTES} \quad (2 \times 6 \times 14{,}155{,}776\text{ B} \approx 162\text{ MB})$$
  - Hard Startup Feasibility Gate:
    $$\text{VRAM}_{\text{dense}} + \text{VRAM}_{\text{kv}}(T) + \text{VRAM}_{\text{scratch}} + \text{VRAM}_{\text{min\_active}} \le \text{VRAM}_{\text{total}} - 300\text{ MB}$$
    If violated, cleanly reject initialization with detailed diagnostics (displaying current allocation breakdown, available VRAM, and maximum allowable context length $T_{\text{max}}$).
  - Compute dynamic Hot VRAM capacity ($S_{\text{hot}}$ slots) and Warm Host DDR capacity ($S_{\text{warm}}$ slots).

- **Micro-Step 1.2: Global Unified VRAM Expert Pool (`src/core/vram_expert_pool.hpp`)**
  - Allocate a single, unified flat VRAM slab of $S_{\text{hot}}$ expert slots ($\approx 880$ slots on 24 GB card with $32\text{k}$ context).
  - Flatten weight allocations into contiguous arrays `d_w1_packed`, `d_w1_scale`, `d_w2_packed`, `d_w2_scale`, `d_w3_packed`, `d_w3_scale` indexed by physical `slot_idx \in [0, S_{\text{hot}}-1]`.
  - Provide asynchronous DMA transfer methods to load and evict experts to/from physical slot indices without per-layer fragmentation.

- **Micro-Step 1.3: Host-Side Dynamic Expert Registry (`src/core/expert_registry.hpp`)**
  - Maintain a lightweight Host CPU catalog ($\approx 528\text{ KB}$) tracking all 11,008 experts ($43 \times 256$).
  - For each expert $(L, E)$, track:
    - Current tier: `Tier::HOT_VRAM`, `Tier::WARM_HOST`, or `Tier::COLD_NVME`.
    - Physical `slot_idx` in the corresponding tier pool.
    - Online activation statistics: `activation_count`, `last_step_used`, and exponential moving average (EMA) activation frequency.
  - Startup Initialization:
    - Default policy: round-robin interleaving across layers into Hot VRAM and Warm DDR pools.
    - Optional prior policy: ingest empirical calibration entropy table (`entropy_prior.bin`) if present.
  - Multi-tier eviction policy: Priority score combining decayed activation frequency and recency to prevent cache pollution from transient tokens.

- **Micro-Step 1.4: Pipeline Integration & Verification (`tests/test_dynamic_expert_pool.cpp`)**
  - Wire `AeonMemoryBudgetEngine`, `GlobalVRAMExpertPool`, and `ExpertRegistry` into `V4Pipeline`.
  - Benchmark on physical silicon:
    - Test feasibility validation (boundary checks with small, valid, and over-budget context sizes).
    - Measure cache hit rate, eviction overhead, and token latency across varying sequence lengths.
    - Record findings in [plans-and-docs/PERFORMANCE_LEDGER.md](plans-and-docs/PERFORMANCE_LEDGER.md).

#### Open Issues & Known Blockers for Next Session (Recorded 2026-09-07)
1. **Tier 2 Warm Host DDR Preload Bottleneck & Memory Pressure**:
   - **Page-Cache & Swap Contention**: Populating 3,325 warm experts (~43.8 GB) synchronously via `mmap` / `memcpy` caused severe memory pressure, pushing active process pages into the Linux swap partition and risking OOM termination (`Getötet` / exit code 137).
   - **Extremely Slow Pre-population**: Single-threaded `mmap` page-faulting across a 145 GB disk file is completely unviable for large allocations. Warm expert pre-loading must be converted to sector-aligned Direct I/O (`O_DIRECT` / `pread` / `io_uring`) instead of sequential `memcpy` over `mmap`.
   - **Host RAM Budget Configuration**: User configuration for `host_ram_bytes` must be explicitly capped to safe bounds (e.g., target ~35 GB instead of maximum hardware limits) so that user-space buffers never compete with the Linux OS, window compositor, or kernel buffers.
2. **`hipHostMalloc` Fallback**:
   - `hipHostMalloc` failed with OOM when requesting ~43.8 GB in a single contiguous pinned slab, falling back to `posix_memalign`. The staging pool needs segmented slab allocation or explicit capacity limits tailored to available unpinned/pinned memory limits (`ulimit -l`).

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
