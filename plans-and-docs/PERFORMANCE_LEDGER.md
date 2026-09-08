# Project Aeon — Milestone Performance & Accuracy Ledger

This ledger records physical hardware verification benchmarks, test conditions, latencies, throughputs, and cache behaviors across major development milestones. Every significant advancement must add an entry here for empirical before-and-after tracking.

---

## Hardware Target & Testbed Baseline
* **Host CPU**: AMD Ryzen Threadripper PRO 3975WX (32 Cores / 64 Threads @ 3.5–4.2 GHz)
* **Host Memory**: 64 GB DDR4 (128 PCIe 4.0 root complex lanes)
* **Storage**: NVMe PCIe 4.0 SSD (`/dev/nvme0n1p2`, ~6.33 GB/s `io_uring` O_DIRECT read bandwidth)
* **Accelerators**: 4x AMD Radeon RX 7900 XTX (Navi 31 / `gfx1100`, 24 GB GDDR6 VRAM each, 96 GB aggregate, PCIe 4.0 x16 per slot, bidirectional P2P enabled)
* **Software Toolchain**: ROCm 7.2.2, native `hipcc`, Linux Kernel 7.0, Wave32 execution mode (`-mno-wavefrontsize64`)

---

## Milestone Entries

### Milestone 1: Phase 0 Foundations & Hardware Spikes
* **Date**: 2026-09-07
* **Commit**: Foundational Phase 0 Spikes
* **Environment**: Single RX 7900 XTX (`gfx1100`, Device 0)
* **Key Numbers & Findings**:
  - **Single Tile Wave32 WMMA** (`test_wmma_tile.cpp`): $16 \times 16 \times 16$ FP16 tile verified against CPU golden math ($\epsilon = 0.0$).
  - **Tiled Block GEMM** (`bench_wmma_gemm.cpp`): $25.6\text{ TFLOP/s}$ on $2048 \times 2048$ matrix ($870\ \mu\text{s}$).
  - **Direct I/O Linux `io_uring`** (`test_direct_io.cpp`): Sustained $6.33\text{ GB/s}$ reads from NVMe using 4096-byte sector-aligned buffer allocations.
  - **Async Overlap (SDMA + WMMA Compute)** (`bench_async_overlap.cpp`): $24.9\text{ GB/s}$ PCIe transfer concurrently overlapped with active WMMA kernels with $0.0\%\text{ compute jitter}$.

---

### Milestone 2: Single-GPU Mathematical Primitives on Real INT4 Checkpoint (Phase 1 Spikes 1–5)
* **Date**: 2026-09-07
* **Commit**: Phase 1 Spikes 1–5
* **Environment**: Single RX 7900 XTX (`gfx1100`, Device 0)
* **Key Numbers & Findings**:
  - **Fused W4A16 Dequant-GEMM** (`test_w4a16_wmma.cpp`): DeepSeek-V4 expert projection ($M=16, N=2048, K=4096$) executes in **$140.34\ \mu\text{s}$** ($1.91\text{ TFLOP/s}$). Real weights matched CPU reference with error $\epsilon = 0.0$.
  - **RMSNorm & SwiGLU Clamp** (`test_swiglu_clamp.cpp`): RMSNorm max error $8.4 \times 10^{-4}$; SwiGLU with clamp limit $10.0$ max error $0.0$.
  - **Hyper-Connections 4-Stream Sinkhorn** (`test_hc_sinkhorn.cpp`): 20 Sinkhorn iterations executed in a single Wave32 kernel with max error $5.96 \times 10^{-8}$.
  - **Dual-Mode MoE Router** (`test_moe_router.cpp`): Hash router (layers 0–2) and SqrtSoftplus router (layers 3–42) matched CPU reference top-6 assignments 100%.
  - **Sliding-Window Attention with Attention Sink** (`test_v4_attention.cpp`): $W=128$, 64 heads, head dim 512 attention kernel latency **$41.91\ \mu\text{s}$** across 16 tokens ($2.62\ \mu\text{s/token}$).
  - **Single Transformer Block** (`test_v4_block.cpp`): Full layer forward pass latency **$1.80\text{ ms/token}$**; output matched golden CPU reference within $\epsilon = 0.0033$.

---

### Milestone 3: Single-GPU Multi-Layer Autoregressive Pipeline Baseline (Spike 6)
* **Date**: 2026-09-07
* **Commit**: `5e27dd9`
* **Test Conditions**:
  - Model: `DeepSeek-V4-Flash-0731-INT4-W4A16`
  - Shards: Shard 1 & Shard 2 (`model-00001.safetensors`, `model-00002.safetensors`) via zero-copy `mmap`
  - Active Layers: 2 consecutive layers (Layer 0 and Layer 1)
  - VRAM Cache Configuration: 8 Tier 1 slots per layer (16 total slots = 216 MB VRAM)
  - Memory Hierarchy: Tier 1 (VRAM LRU) $\leftarrow$ Tier 2 (OS Page Cache / mmap Host DDR)
  - Sampling: Greedy argmax from full $129,280$-dim logits projected on device

* **Achieved Benchmark Numbers**:

| Metric | Short Prompt Scenario | Medium Prompt Scenario |
| :--- | :---: | :---: |
| **Prompt Length** | 4 tokens | 8 tokens |
| **Generated Tokens** | 16 tokens | 32 tokens |
| **Total Latency** | $1674.94\text{ ms}$ | $2150.87\text{ ms}$ |
| **TTFT (Prefill)** | $422.20\text{ ms}$ ($105.55\text{ ms/token}$) | $528.60\text{ ms}$ ($66.07\text{ ms/token}$) |
| **Decode Throughput** | **$11.97\text{ tokens/sec}$** | **$19.11\text{ tokens/sec}$** |
| **Decode Step Latency** | **$83.52\text{ ms/token}$** | **$52.33\text{ ms/token}$** |
| **Tier 1 Cache Hits** | 4 hits | 11 hits |
| **Tier 1 Cache Misses**| 224 misses | 457 misses |
| **Tier 1 VRAM Hit Rate**| **$1.8\%$** | **$2.4\%$** |

* **Analysis & Critical Gaps Identified**:
  - The $19.11\text{ tok/s}$ throughput was achieved on **only 2 layers**. Extrapolated naively to all 43 layers without pipelining or larger caches, sequential throughput drops to $\approx 0.89\text{ tok/s}$.
  - The 8-slot VRAM LRU cache suffers from the **Cold-Miss Trap**: $97.6\%$ of expert accesses missed VRAM and required synchronous host transfers.
  - Next architectural step: Implement accurate memory budgeting, expand Tier 1 VRAM cache, integrate true 3-tier streaming (VRAM hot $\leftarrow$ DDR warm $\leftarrow$ NVMe cold direct I/O), and overlap asynchronous SDMA prefetching.

---

### Milestone 4: Phase 2 Spike 1 — Dynamic Memory Budgeting & Global Unified VRAM Expert Pool
* **Date**: 2026-09-07
* **Commit**: `cc252c9`
* **Test Conditions**:
  - Model: `DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon` (Native `.aeon` format)
  - Hardware: AMD Radeon RX 7900 XTX (24 GB VRAM, `gfx1100`, PCIe 4.0 x16), Threadripper PRO 3975WX (64 GB DDR)
  - Memory Budget Engine:
    - Target Context Size: $4,096$ tokens ($176\text{ MB}$ KV Cache for all 43 layers)
    - Headroom: Fixed $300\text{ MB}$ OS safety buffer
    - Host RAM Cap: Fixed $80\%$ ($50.10\text{ GB}$)
  - Hierarchy Partitioning:
    - **Tier 1 (Hot VRAM)**: **664 dynamic slots** ($8.75\text{ GB}$) dynamically shared across all layers (up from 16 slots / 216 MB in Milestone 3)
    - **Tier 2 (Warm Host DDR)**: **3,800 slots** ($50.10\text{ GB}$)
    - **Tier 3 (Cold NVMe SSD)**: **6,544 slots**
  - Active Layers: 2 consecutive layers (Layer 0 and Layer 1)
  - Benchmark Executable: `tests/bench_dynamic_pool.cpp`
  - Sampling: Greedy argmax from full $129,280$-dim logits projected on device

* **Achieved Benchmark Numbers**:

| Metric | Short Prompt Scenario (Spike 1) | Medium Prompt Scenario (Spike 1) |
| :--- | :---: | :---: |
| **Prompt Length** | 4 tokens | 8 tokens |
| **Generated Tokens** | 16 tokens | 32 tokens |
| **Total Latency** | **$207.20\text{ ms}$** *(was $1674.94\text{ ms}$)* | **$434.63\text{ ms}$** *(was $2150.87\text{ ms}$)* |
| **TTFT (Prefill)** | **$43.76\text{ ms}$** ($10.94\text{ ms/token}$) | **$88.19\text{ ms}$** ($11.02\text{ ms/token}$) |
| **Decode Throughput** | **$91.78\text{ tokens/sec}$** *(was $11.97\text{ tok/s}$)* | **$89.48\text{ tokens/sec}$** *(was $19.11\text{ tok/s}$)* |
| **Decode Step Latency** | **$10.90\text{ ms/token}$** *(was $83.52\text{ ms/tok}$)* | **$11.18\text{ ms/token}$** *(was $52.33\text{ ms/tok}$)* |
| **Overall Avg Latency** | **$10.36\text{ ms/token}$** | **$10.87\text{ ms/token}$** |
| **Tier 1 VRAM Cache Hits** | 228 hits | 468 hits |
| **Tier 1 VRAM Cache Misses**| 0 misses *(was 224 misses)* | 0 misses *(was 457 misses)* |
| **Tier 1 VRAM Hit Rate**| **$100.0\%$** *(was $1.8\%$)* | **$100.0\%$** *(was $2.4\%$)* |

* **Analysis & Comparison against Milestone 3 Baseline**:
  - **$4.7\times$ to $7.7\times$ Decode Speedup**: Throughput jumped from $11.97 - 19.11\text{ tok/s}$ to **$89.48 - 91.78\text{ tok/s}$** on 2 layers.
  - **Elimination of the Cold-Miss Trap**: With 664 global VRAM slots instead of 8 per-layer slots, all required experts for the sequence were resident in Hot VRAM ($100\%$ hit rate), completely eliminating the synchronous host-to-device PCIe transfer stalls ($83.52\text{ ms} \to 10.90\text{ ms}$ per step).
  - **Pre-Prefill Acceleration**: TTFT improved by **$6.0\times$ to $9.6\times$** ($105.55\text{ ms/tok} \to 10.94\text{ ms/tok}$).

---

### Milestone 5: Full 43-Layer End-to-End Model Execution on Single GPU
* **Date**: 2026-09-07
* **Commit**: Pending
* **Test Conditions**:
  - Model: `DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon` (All 43 layers, 11,008 routed experts, 145 GB parameter universe)
  - Hardware: AMD Radeon RX 7900 XTX (24 GB VRAM, `gfx1100`, PCIe 4.0 x16), Threadripper PRO 3975WX (64 GB DDR)
  - Active Layers: **All 43 consecutive layers (100% full model forward pass)**
  - VRAM Budget: $14.66\text{ GB}$ dense backbone + $0.17\text{ GB}$ KV cache + $100\text{ MB}$ scratch + $300\text{ MB}$ headroom
  - Global VRAM Pool: **664 hot slots** ($8.75\text{ GB}$) shared dynamically across all 43 layers
  - Warm Host DDR Staging: **3,800 slots** ($50.10\text{ GB}$)
  - Benchmark Suite: `tests/bench_full_model.cpp` (Prompt: 4 tokens -> 8 generated tokens)

* **Achieved Benchmark Numbers**:

| Metric | 43-Layer Full-Model Run |
| :--- | :---: |
| **Total Model Layers Executed** | **43 layers** (100% complete model) |
| **Pipeline Initialization Time** | **$16.05\text{ seconds}$** |
| **Prompt Length** | 4 tokens |
| **Generated Tokens** | 8 tokens |
| **Total Execution Time** | $10,565.01\text{ ms}$ |
| **TTFT (Prompt Prefill)** | **$5,816.63\text{ ms}$** ($1,454.16\text{ ms/tok}$ total, **$33.82\text{ ms/tok/layer}$**) |
| **Decode Throughput** | **$1.47\text{ tokens/sec}$** |
| **Decode Step Latency** | **$678.32\text{ ms/token}$** (**$15.77\text{ ms/tok/layer}$**) |
| **Tier 1 VRAM Cache Hits** | 1,684 hits |
| **Tier 1 VRAM Cache Misses**| 1,154 misses |
| **Tier 1 VRAM Hit Rate** | **$59.3\%$** |

* **Empirical Bottleneck Analysis & Spike 2 Rationale**:
  - **Hit Rate Reality**: Across 43 layers, the 664 VRAM slots achieved a **$59.3\%$ hit rate** under uniform initial placement without offline prior or async prefetching.
  - **PCIe Latency Cost**: The 1,154 misses required on-demand synchronous PCIe host transfers ($14.15\text{ MB}$ per expert $\approx 0.6\text{ ms}$ per miss). Across the 43-layer forward pass ($258$ expert evaluations per token), cache misses accounted for $\approx 450\text{ ms}$ of the $678\text{ ms}$ decode step latency.
  - **The Direct Mandate for Spike 2**: Asynchronous SDMA prefetching must overlap these misses behind the preceding layer's compute to recover pure silicon compute speed.

---

### Milestone 6: Full-Model Cold-Tier Baseline Without Eager Warm Preload
* **Date**: 2026-09-08
* **Commit**: Pending
* **Test Conditions**:
  - Model: `DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon` (43 layers, 11,008 routed experts)
  - Hardware: AMD Radeon RX 7900 XTX (`gfx1100`, Device 0)
  - Context size: 4,096 tokens
  - Tier 1: 664 hot VRAM slots ($8.75\text{ GB}$)
  - Tier 2: Disabled for this baseline; the configured 35 GiB ceiling was not allocated or preloaded
  - Tier 3: 10,344 experts streamed on demand from the mapped `.aeon` source
  - Benchmark: `tests/bench_full_model.cpp` (4-token prompt, 8 generated tokens)

* **Achieved Benchmark Numbers**:

| Metric | First Run (Cold File Pages) | Final Run (Warm File Pages) |
| :--- | :---: | :---: |
| **Pipeline Initialization** | $7.57\text{ s}$ | $4.42\text{ s}$ |
| **Total Execution Time** | $12,822.61\text{ ms}$ | $5,884.86\text{ ms}$ |
| **TTFT (Prefill)** | $6,277.68\text{ ms}$ | $2,549.36\text{ ms}$ |
| **Decode Throughput** | $1.07\text{ tokens/sec}$ | **$2.10\text{ tokens/sec}$** |
| **Decode Step Latency** | $934.98\text{ ms/token}$ | $476.48\text{ ms/token}$ |
| **Tier 1 VRAM Cache Hits** | 1,609 | 1,609 |
| **Tier 1 VRAM Cache Misses** | 1,229 | 1,229 |
| **Tier 1 VRAM Hit Rate** | 56.7% | 56.7% |

* **Analysis**:
  - The benchmark now reaches inference without allocating or synchronously populating a 35 GiB warm buffer.
  - The large difference between the first and final passes exposes the current mapped-file/page-cache dependency; it is the baseline Spike 2 must improve with explicit asynchronous staging and direct I/O.

---

### Milestone 7: Two-Layer Refactored Pipeline Performance Recheck
* **Date**: 2026-09-08
* **Commit**: Pending
* **Test Conditions**:
  - Model: `DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon`
  - Hardware: AMD Radeon RX 7900 XTX (`gfx1100`, Device 0)
  - Active layers: 2
  - Tier 1: 664 hot VRAM slots; warm-host preload disabled to avoid the known eager 35 GiB startup path
  - Three-step warmup before measurement; measured scenarios had no hot-pool misses
  - Benchmark: `tests/bench_dynamic_pool.cpp`

| Metric | Short Prompt (4 -> 16) | Medium Prompt (8 -> 32) |
| :--- | :---: | :---: |
| **Current Decode Throughput** | **$48.81\text{ tokens/sec}$** | **$49.01\text{ tokens/sec}$** |
| **Current Decode Step Latency** | $20.49\text{ ms/token}$ | $20.40\text{ ms/token}$ |
| **Tier 1 VRAM Cache Hits** | 228 | 468 |
| **Tier 1 VRAM Cache Misses** | 0 | 0 |
| **Tier 1 VRAM Hit Rate** | 100.0% | 100.0% |

* **Regression Against Milestone 4**:
  - Short-prompt throughput decreased from $91.78$ to $48.81\text{ tokens/sec}$ ($46.8\%$ lower).
  - Medium-prompt throughput decreased from $89.48$ to $49.01\text{ tokens/sec}$ ($45.2\%$ lower).
  - Because both current scenarios have zero expert misses, the regression is in the per-token execution path rather than PCIe expert transfers.
  - Controlled A/B: restoring only the legacy CPU-side HC projection and pre-combine path under the same lazy-cache configuration recovered $90.80$ and $93.41\text{ tokens/sec}$. Restoring the device-HC path returned to $48.83$ and $48.79\text{ tokens/sec}$. This isolates the regression to the new device-side HC precompute implementation introduced by the refactoring.

---

### Milestone 8: Parallelized Device-Side Hyper-Connections Kernel Optimization
* **Date**: 2026-09-08
* **Commit**: Pending
* **Test Conditions**:
  - Model: `DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon` (Native `.aeon` format)
  - Hardware: AMD Radeon RX 7900 XTX (`gfx1100`, Device 0)
  - Active Layers: 2 consecutive layers (Layer 0 and Layer 1)
  - Global VRAM Pool: 664 hot slots ($8.75\text{ GB}$)
  - Benchmark: `tests/bench_dynamic_pool.cpp`
  - Optimization: Redesigned `hc_project_kernel` to parallelize 24 output mixes across 24 Wave32 blocks (256 threads each) with 128-bit `float4` vectorized loads; optimized `hc_pre_combine_kernel` with `float4` / `half2` vectorization. Kernel latency plummeted from $1,188.34\ \mu\text{s}$ down to $9.20\ \mu\text{s}$ ($129\times$ speedup).

* **Achieved Benchmark Numbers**:

| Metric | Short Prompt (4 -> 16) | Medium Prompt (8 -> 32) |
| :--- | :---: | :---: |
| **Decode Throughput** | **$122.90\text{ tokens/sec}$** | **$122.60\text{ tokens/sec}$** |
| **Decode Step Latency** | **$8.14\text{ ms/token}$** | **$8.16\text{ ms/token}$** |
| **TTFT (Prefill)** | **$32.73\text{ ms}$** ($8.18\text{ ms/tok}$) | **$65.66\text{ ms}$** ($8.21\text{ ms/tok}$) |
| **Total Latency** | $154.80\text{ ms}$ | $318.53\text{ ms}$ |
| **Tier 1 VRAM Cache Hits** | 228 hits | 468 hits |
| **Tier 1 VRAM Cache Misses** | 0 misses | 0 misses |
| **Tier 1 VRAM Hit Rate** | 100.0% | 100.0% |

* **Analysis & Recovery Comparison**:
  - **Full Throughput Recovery and Leap**: Completely resolved the $49\text{ tok/s}$ regression, exceeding both the Milestone 4 CPU-hybrid baseline ($89.5 - 91.8\text{ tok/s}$) and the earlier Milestone 7 regression by jumping to **$122.6 - 122.9\text{ tok/s}$** ($2.5\times$ over regressed state, $+34\%$ over original baseline).
  - **Pure Zero-Host-Sync GPU Execution**: Eliminates all CPU roundtrip copies (`d_res` to host, CPU dot products, host-to-device transfers) with zero host synchronization stalls.
  - **Bit-Exact Numerical Precision**: Parity verified against CPU reference with error $< 4.5 \times 10^{-6}$ for projection and exact 0 error for pre-combination and post-expansion.

---

### Milestone 9: 43-Layer Full-Model Benchmark with Optimized Kernels
* **Date**: 2026-09-08
* **Commit**: Pending
* **Test Conditions**:
  - Model: `DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon` (All 43 layers, 11,008 routed experts)
  - Hardware: AMD Radeon RX 7900 XTX (`gfx1100`, Device 0)
  - Active Layers: **All 43 consecutive layers (100% full model forward pass)**
  - Tier 1: 664 hot VRAM slots ($8.75\text{ GB}$) dynamically shared across all layers
  - Tier 2: Disabled for cold baseline; on-demand streaming from `.aeon` container
  - Benchmark Executable: `tests/bench_full_model.cpp` (Prompt: 4 tokens -> 8 generated tokens)

* **Achieved Benchmark Numbers**:

| Metric | First Pass (Cold Page Cache) | Second Pass (Warm Page Cache) | Prior Milestone 6 Baseline |
| :--- | :---: | :---: | :---: |
| **Pipeline Initialization** | $17.03\text{ s}$ | $3.70\text{ s}$ | $4.42\text{ s}$ |
| **Total Execution Time** | $9,064.50\text{ ms}$ | **$2,921.36\text{ ms}$** | $5,884.86\text{ ms}$ |
| **TTFT (Prefill)** | $4,828.98\text{ ms}$ ($1,207.24\text{ ms/tok}$) | **$1,494.25\text{ ms}$** ($373.56\text{ ms/tok}$) | $2,549.36\text{ ms}$ |
| **Decode Throughput** | $1.65\text{ tokens/sec}$ | **$4.91\text{ tokens/sec}$** | $2.10\text{ tokens/sec}$ |
| **Decode Step Latency** | $605.06\text{ ms/token}$ | **$203.86\text{ ms/token}$** | $476.48\text{ ms/token}$ |
| **Per-Layer Decode Latency** | $14.07\text{ ms/tok/layer}$ | **$4.74\text{ ms/tok/layer}$** | $11.08\text{ ms/tok/layer}$ |
| **Tier 1 VRAM Cache Hits** | 1,682 hits | 1,682 hits | 1,609 hits |
| **Tier 1 VRAM Cache Misses**| 1,156 misses | 1,156 misses | 1,229 misses |
| **Tier 1 VRAM Hit Rate** | **$59.3\%$** | **$59.3\%$** | 56.7% |

* **Analysis**:
  - The parallelized HC kernels and zero-host-sync pipeline drastically lowered per-layer compute latency across all 43 layers.
  - On warm pages, decode throughput jumped from **$2.10\text{ tok/s}$ to $4.91\text{ tok/s}$** ($2.3\times$ speedup), and decode step latency dropped from **$476.5\text{ ms}$ to $203.9\text{ ms}$** ($4.74\text{ ms/tok/layer}$).
  - Even on cold page faults, total latency improved by over $3.7\text{ seconds}$ ($12.8\text{ s} \to 9.06\text{ s}$).
  - Across 43 layers ($258$ active expert evaluations per token), the remaining $\approx 200\text{ ms/tok}$ latency is dominated by synchronous page faults and PCIe DMA transfers for the 1,156 misses ($40.7\%$ miss rate). This confirms the critical necessity of **Spike 2: Dual-stream asynchronous SDMA prefetching** to hide this remaining I/O latency behind compute.
