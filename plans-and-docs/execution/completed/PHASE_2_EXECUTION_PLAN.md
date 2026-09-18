# Phase 2: Single-GPU 3-Tier Storage & Memory Hierarchy Optimization
*Architecture Target: DeepSeek-V4-Flash-0731 (INT4-W4A16 on AMD RDNA3 / gfx1100)*

*Status: completed record. Spikes 0-3 were implemented. The runtime that consumed this work was subsequently rebuilt; the targets that were never reached (storage layout, host-memory pressure, model-backed throughput) are not restated here. See [DOCUMENTATION_STATUS.md](../../status/DOCUMENTATION_STATUS.md) for the project-wide status classification.*

---

## 1. Executive Summary & Architectural Rationale

Following the silicon verification of Phase 1 (Spikes 1–6), where all mathematical primitives, fused INT4 WMMA kernels, 4-stream Hyper-Connections Sinkhorn, sliding-window attention with sink, and multi-layer state chaining were proven on physical AMD Radeon RX 7900 XTX hardware, **Phase 2 focuses strictly on maximizing the physical I/O and memory throughput of the single-GPU inference runtime**.

As revealed in our empirical performance analysis ([PERFORMANCE_LEDGER.md](../../status/PERFORMANCE_LEDGER.md)):
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

- **Micro-Step 0.1: Surgical Model Splitting & Sector-Aligned Serializer (`scripts/convert_safetensors_to_aeon.py`)**
  - Ingest the 34 sharded Safetensors files of `DeepSeek-V4-Flash-0731-INT4-W4A16`.
  - Isolate all dense weights into `model_dense.aeon` (~9.24 GB): Attention projections ($W_q, W_{kv}, W_o$), RMSNorms, Hyper-Connections Sinkhorn tables, Shared Experts, and Router gate weights.
  - Isolate the 11,008 routed experts (43 layers $\times$ 256 experts) into `model_experts_swizzled.aeon` (~135 GB): Each expert FFN ($W_1, W_2, W_3$ packed INT4 + FP16 scales) is written in the version-2 Wave32 swizzled layout as an isolated contiguous block starting at a strictly 4096-byte aligned file offset (`O_DIRECT` compliant).
  - Generate a compact binary index table `model_experts_swizzled.index` mapping `(layer_id, expert_id)` to `(uint64_t file_offset, uint64_t byte_length)`.
- **Micro-Step 0.2: Bit-Exact Numerical Verification Suite**
  - Implement a verification test comparing the parsed `.aeon` weights against the original Safetensors weights on silicon.
  - Validate that every INT4 nibble and FP16 scale is 100% bit-identical with zero precision loss ($\epsilon = 0.0$).
  - *Verification:* Silicon test confirming bit-level identical tensor hashes across dense components and sample routed experts.

---

### Spike 1: Dynamic Memory Budgeting, Feasibility Gating & Global Unified VRAM Expert Pool
*Objective: Eliminate rigid per-layer slot allocations; enforce strict startup hardware feasibility gating; maximize cache hit rate under real Zipfian MoE activation entropy by dynamically sharing VRAM capacity across all 43 layers.*

- **Micro-Step 1.1: Runtime Configuration & Hard Feasibility Gate (`src/architecture/deepseek_v4/core/memory_budget.hpp`)**
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

- **Micro-Step 1.2: Global Unified VRAM Expert Pool (`src/backend/swizzled_w4a16/core/vram_expert_pool.hpp`)**
  - Allocate a single, unified flat VRAM slab of $S_{\text{hot}}$ expert slots ($\approx 880$ slots on 24 GB card with $32\text{k}$ context).
  - Flatten weight allocations into contiguous arrays `d_w1_packed`, `d_w1_scale`, `d_w2_packed`, `d_w2_scale`, `d_w3_packed`, `d_w3_scale` indexed by physical `slot_idx \in [0, S_{\text{hot}}-1]`.
  - Provide asynchronous DMA transfer methods to load and evict experts to/from physical slot indices without per-layer fragmentation.

- **Micro-Step 1.3: Host-Side Dynamic Expert Registry (`src/infrastructure/core/expert_registry.hpp`)**
  - Maintain a lightweight Host CPU catalog ($\approx 528\text{ KB}$) tracking all 11,008 experts ($43 \times 256$).
  - For each expert $(L, E)$, track:
    - Current tier: `Tier::HOT_VRAM`, `Tier::WARM_HOST`, or `Tier::COLD_NVME`.
    - Physical `slot_idx` in the corresponding tier pool.
    - Online activation statistics: `activation_count`, `last_step_used`, and exponential moving average (EMA) activation frequency.
  - Startup Initialization:
    - Default policy: round-robin interleaving across layers into Hot VRAM and Warm DDR pools.
    - Optional prior policy: ingest empirical calibration entropy table (`entropy_prior.bin`) if present.
    - Compare frequency-ordered placement against the profile-derived expert hotlist in the local [DwarfStar (ds4) reference](../../../../aeon-references/ds4); treat it as a placement prior, not as a substitute for Aeon's raw routing-entropy measurements.
  - Multi-tier eviction policy: Priority score combining decayed activation frequency and recency to prevent cache pollution from transient tokens.

- **Micro-Step 1.4: Pipeline Integration & Verification (`tests/test_dynamic_expert_pool.cpp`)**
  - Wire `AeonMemoryBudgetEngine`, `GlobalVRAMExpertPool`, and `ExpertRegistry` into the model host.
  - Benchmark on physical silicon:
    - Test feasibility validation (boundary checks with small, valid, and over-budget context sizes).
    - Measure cache hit rate, eviction overhead, and token latency across varying sequence lengths.
    - Record findings in [PERFORMANCE_LEDGER.md](../../status/PERFORMANCE_LEDGER.md).

---

### Spike 2: Dual-Stream Asynchronous SDMA Prefetching & Latency Hiding
*Objective: Overlap compute and PCIe transfers so that cold-miss expert streaming runs concurrently with active attention and resident expert execution.*

- **Micro-Step 2.1: Lookahead Routing & Prefetch Horizon Pipeline**
  - While Layer $L$ is executing its attention and resident shared expert pass, trigger routing calculation for Layer $L+1$.
  - Identify missing experts for Layer $L+1$ ahead of execution time.
- **Micro-Step 2.2: Double-Buffered Asynchronous SDMA Transfer Stream (`src/infrastructure/core/prefetch_staging.hpp`)**
  - Implemented 12-slot ($170\text{ MB}$) pinned host arena via `hipHostMalloc` (`hipHostMallocPortable`), bypassing OS page-faults and unpinned memory thrashing.
  - Dispatch non-blocking PCIe DMA transfers on dedicated HIP SDMA stream concurrently with compute stream.
  - Synchronize via non-blocking HIP event barriers (`hipEventRecord`, `hipStreamWaitEvent`) immediately before routed MoE execution.
- **Micro-Step 2.3: Overlap Verification & Latency Hiding Benchmark on Silicon (historical two-layer checkpoint)**
  - Silicon verification under severe cold-miss conditions (12 VRAM slots, 94% miss rate).
  - TTFT improved by $+36.2\%$ ($111.4\text{ ms} \to 71.1\text{ ms}$) and decode throughput accelerated by $+71.5\%$ ($19.1\text{ tok/s} \to 32.8\text{ tok/s}$).
  - *Key Finding*: Inter-layer lookahead plateaued at $33.1\text{ tok/s}$ because single-threaded CPU `memcpy` from unpinned `mmap` backing pages into pinned staging buffers ($85\text{ MB/step}$) creates a synchronous host memory bus bottleneck, directly affirming the need for Spike 3 Direct I/O.

---

### Spike 3: Linux `io_uring` Direct I/O NVMe Cold Tier Integration
*Objective: Complete the 3-tier chain by connecting cold NVMe SSD storage directly to Host DDR staging via Linux `io_uring` with zero kernel page-cache contention, eliminating synchronous CPU `memcpy` stalls from the streaming pipeline.*

- **Micro-Step 3.1: Linux `io_uring` Direct I/O Reader Integration (Tier 3 $\to$ Tier 2)**
  - Integrate `src/infrastructure/io/direct_io_reader.hpp` into the runtime pipeline targeting `model_experts_swizzled.aeon`.
  - Replace `mmap` + CPU `memcpy` expert retrieval with asynchronous direct streaming of cold experts from NVMe into pinned host DDR staging buffers (`PrefetchStagingArena`) using `O_DIRECT`.
  - Maintain a dynamic Tier 2 warm cache in host RAM ($\approx 35\text{ GB}$) feeding Tier 1 VRAM without triggering OS page-cache bloat or swap thrashing.
- **Micro-Step 3.2: End-to-End 3-Tier Pipeline Validation**
  - Silicon test measuring end-to-end 3-tier streaming throughput from NVMe $\to$ Host RAM $\to$ GPU VRAM, validating data integrity and measuring throughput in GB/s.

#### Spike 3 Implementation Checkpoint (2026-09-08)

- [x] Added batched `io_uring` submission/completion handling, strict 4KB request validation, and default `IOSQE_ASYNC` execution in `src/infrastructure/io/direct_io_reader.hpp`.
- [x] Added a dedicated `O_DIRECT` descriptor plus validated expert locations to `AeonModelLoader`.
- [x] Added staging-slot ownership states and matching HIP/`posix_memalign` cleanup; direct and legacy host fills now share the release lifecycle.
- [x] Integrated cold expert reads into the native pipeline prefetch boundary while preserving Hot VRAM and Warm Host source paths.
- [x] Added `tests/test_model_direct_io.cpp` and the `test_model_direct_io` CTest target; the gate now validates 24 aligned 4 MiB subreads for six experts.
- [x] Silicon checks passed after the async/chunked fix: generic direct I/O (`6.43 GB/s`), model-backed six-expert payload parity (`3.17 GiB/s`), native pipeline golden token `69146`, and async cold-miss generation (`32.87 tok/s`).
- [x] Root cause isolated: baseline `io_uring_enter` submission took about `45 ms` for six expert reads while completion waits were sub-millisecond; `IOSQE_ASYNC` removed that synchronous submission behavior. A 4 MiB direct-read sweep was also materially faster than 14-16 MiB requests on this device.
- [x] Historical 12-slot native source A/B produced identical tokens and measured direct `32.31/32.57 tok/s` versus mmap-source `33.57/33.22 tok/s`. Replacing the source fill alone did not improve the then-current end-to-end critical path.

#### Spike 3 Tier 2 Integration Checkpoint (2026-09-08, historical pre-M16 behavior)

- [x] Replaced the single contiguous `HostExpertPool` allocation with sector-aligned 64-expert segments and matching per-segment HIP-pinned or `posix_memalign` cleanup.
- [x] Added bounded direct-I/O batch loading for initial Hot VRAM and Warm Host residents. The dynamic-global path now fills both tiers from the `O_DIRECT` descriptor instead of synchronously faulting expert payloads from mmap.
  - [x] Added the initial VRAM-to-host expert downloads for Hot-to-Warm demotion and corrected Warm-to-Hot promotion ordering so host slots could not be overwritten before their payload was staged. The later M16 change removed demotion from the request path.
- [x] Corrected staging-slot release across multi-layer execution; the 43-layer path can reuse the fixed double-buffer arena without invalid state transitions.
- [x] Added a full-model Hot/Warm/Cold smoke test: 43 layers, context 256, 676 Hot slots, 8 Warm slots, and 10,324 Cold slots. Silicon run completed one valid step with token `295`, 12 Hot hits, 0 Warm hits, and 24 Cold misses.
- [x] Measured a repeat Warm-serving workload on the full 43-layer model: the 8 GiB profile served 555 requests from Warm Host and 601 from Cold NVMe during the measured generation.
- [x] Ran the full configured Warm capacity: 2,654 slots (`34.99 GiB`) populated through direct I/O and used by the 43-layer benchmark. The run completed successfully; host swap usage rose by approximately `0.7 GiB`.
- [x] Removed the synchronous Hot-to-Warm demotion wait: VRAM-to-host DMA is now event-tracked per Warm slot and only waited on when the payload is consumed or the slot is reused. Controlled 0 GiB versus 35 GiB runs measure `4.37 tok/s` versus `5.15 tok/s` with identical output and service counts.
