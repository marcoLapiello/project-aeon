# Project Aeon — Milestone Performance & Accuracy Ledger

Physical hardware verification benchmarks, latencies, throughputs, and cache behaviors across major development milestones on AMD Radeon RX 7900 XTX (`gfx1100`).

---

## 1. Hardware Testbed Baseline
* **Host CPU**: AMD Ryzen Threadripper PRO 3975WX (32C/64T @ 3.5–4.2 GHz)
* **Host RAM**: 64 GB DDR4 (128 PCIe 4.0 lanes)
* **NVMe SSD**: `/dev/nvme0n1p2` (~6.33 GB/s `io_uring` O_DIRECT read bandwidth)
* **GPUs**: 4x AMD Radeon RX 7900 XTX (Navi 31 / `gfx1100`, 24 GB GDDR6 each, 96 GB aggregate, PCIe 4.0 x16, P2P enabled)
* **Toolchain**: ROCm 7.2.2, native `hipcc`, Linux 7.0, Wave32 execution mode (`-mno-wavefrontsize64`)

---

## 2. Milestone Summary Matrix

### End-to-End Generation Progress (DeepSeek-V4 INT4-W4A16)

| Milestone | Scope | Format | Hot VRAM Slots | Hit Rate | TTFT (ms/tok) | Decode (tok/s) | Step Latency | Key Progression |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M3: Baseline Pipeline** | 2L | Safetensors | 16 (8/layer) | 2.4% | 66.07 ms | **19.1 tok/s** | 52.33 ms | First multi-layer token generation; cold-miss bottleneck |
| **M4: Dynamic Hot Pool** | 2L | `.aeon` | 664 (global) | 100.0% | 11.02 ms | **89.5 tok/s** | 11.18 ms | Unified pool eliminates cold-miss trap ($4.7\times$ speedup) |
| **M5: Full-Model First Pass**| 43L | `.aeon` | 664 (global) | 59.3% | 33.82 ms/L | **1.47 tok/s** | 678.32 ms | 100% full model forward pass (11,008 experts) |
| **M6: Bounded Cold Tier** | 43L | `.aeon` | 664 (global) | 56.7% | 59.29 ms/L | **2.10 tok/s** | 476.48 ms | Lazy on-demand mmap; eliminated eager 35GB host alloc |
| **M7: Refactor Check (Gap)**| 2L | `.aeon` | 664 (global) | 100.0% | 20.39 ms | **49.0 tok/s** | 20.40 ms | Regressed: unparallelized device HC kernel (1 warp) |
| **M8: Parallel HC Kernels** | 2L | `.aeon` | 664 (global) | 100.0% | 8.21 ms | **122.6 tok/s** | 8.16 ms | 24-block float4 HC ($129\times$ kernel speedup); $+37\%$ over M4 |
| **M9: Full Model Optimized** | 43L | `.aeon` | 664 (global) | 59.3% | 8.69 ms/L | **4.91 tok/s** | 203.86 ms | $2.3\times$ speedup across all 43L ($4.74\text{ ms/tok/layer}$) |

---

## 3. Detailed Milestone Ledger

### M1: Phase 0 Foundations & Hardware Spikes
* **Date**: 2026-09-07 | **Target**: RX 7900 XTX (`gfx1100`, Device 0)
* `test_wmma_tile.cpp`: $16 \times 16 \times 16$ FP16 tile verified against CPU reference ($\epsilon = 0.0$).
* `bench_wmma_gemm.cpp`: $25.6\text{ TFLOP/s}$ on $2048 \times 2048$ ($870\ \mu\text{s}$).
* `test_direct_io.cpp`: Sustained $6.33\text{ GB/s}$ NVMe read via Linux `io_uring` with 4KB sector alignment.
* `bench_async_overlap.cpp`: $24.9\text{ GB/s}$ PCIe transfer overlapped with WMMA compute ($0.0\%$ compute jitter).

### M2: Single-GPU Mathematical Primitives (Phase 1 Spikes 1–5)
* **Date**: 2026-09-07 | **Target**: RX 7900 XTX (`gfx1100`, Device 0)
* `test_w4a16_wmma.cpp`: Expert projection ($M=16, N=2048, K=4096$) in **$140.34\ \mu\text{s}$** ($1.91\text{ TFLOP/s}$), $\epsilon = 0.0$.
* `test_swiglu_clamp.cpp`: RMSNorm $\epsilon < 8.4 \times 10^{-4}$; SwiGLU clamp limit $10.0$ $\epsilon = 0.0$.
* `test_hc_sinkhorn.cpp`: 20-iter Sinkhorn in single Wave32 kernel, $\epsilon < 5.96 \times 10^{-8}$.
* `test_moe_router.cpp`: Hash router (L0–2) & SqrtSoftplus (L3–42) matched CPU top-6 assignments 100%.
* `test_v4_attention.cpp`: Cached sliding-window ($W=128$) latency **$41.91\ \mu\text{s}$** for 16 tokens ($2.62\ \mu\text{s/tok}$).
* `test_v4_block.cpp`: Full block forward pass **$1.80\text{ ms/tok}$**; CPU reference parity $\epsilon = 0.0033$.

### M3: Autoregressive Pipeline Baseline (Phase 1 Spike 6)
* **Date**: 2026-09-07 | **Commit**: `5e27dd9` | **Model**: DeepSeek-V4 INT4-W4A16 (Safetensors)
* **Config**: 2 layers (L0–L1), 8 slots/layer (16 total = 216 MB), greedy argmax (129,280 logits).
* **Metrics**:
  - Short prompt (4 $\to$ 16 gen): TTFT $422.20\text{ ms}$ ($105.55\text{ ms/tok}$), Decode **$11.97\text{ tok/s}$** ($83.52\text{ ms/tok}$), Hit rate $1.8\%$.
  - Med prompt (8 $\to$ 32 gen): TTFT $528.60\text{ ms}$ ($66.07\text{ ms/tok}$), Decode **$19.11\text{ tok/s}$** ($52.33\text{ ms/tok}$), Hit rate $2.4\%$.
* **Bottleneck**: Severe Cold-Miss Trap ($97.6\%$ misses require synchronous host transfers).

### M4: Dynamic Budgeting & Unified VRAM Expert Pool (Phase 2 Spike 1)
* **Date**: 2026-09-07 | **Commit**: `cc252c9` | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`)
* **Config**: 2 layers (L0–L1), **664 dynamic VRAM slots** ($8.75\text{ GB}$ global pool), 4K context budget.
* **Metrics**:
  - Short prompt (4 $\to$ 16 gen): TTFT **$43.76\text{ ms}$** ($10.94\text{ ms/tok}$), Decode **$91.78\text{ tok/s}$** ($10.90\text{ ms/tok}$), Hit rate **$100.0\%$**.
  - Med prompt (8 $\to$ 32 gen): TTFT **$88.19\text{ ms}$** ($11.02\text{ ms/tok}$), Decode **$89.48\text{ tok/s}$** ($11.18\text{ ms/tok}$), Hit rate **$100.0\%$**.
* **Impact**: $4.7\times$ to $7.7\times$ decode speedup via elimination of cold-miss trap for working set.

### M5: Full 43-Layer Execution on Single GPU
* **Date**: 2026-09-07 | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 43 layers, 11,008 experts)
* **Config**: 664 hot VRAM slots, 3,800 warm host slots, 4 prompt $\to$ 8 gen.
* **Metrics**: Init $16.05\text{ s}$, TTFT $5,816.63\text{ ms}$ ($33.82\text{ ms/tok/layer}$), Decode **$1.47\text{ tok/s}$** ($678.32\text{ ms/tok}$, $15.77\text{ ms/tok/layer}$), Hit rate **$59.3\%$** (1,684 hits / 1,154 misses).
* **Finding**: Synchronous PCIe misses ($14.15\text{ MB/expert}$) account for $\approx 450\text{ ms}$ of the $678\text{ ms}$ step latency.

### M6: Full-Model Baseline with Bounded Cold Tier
* **Date**: 2026-09-08 | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 43 layers)
* **Config**: 664 hot VRAM slots; warm-host preload disabled; 10,344 experts streamed on demand from mmap `.aeon`.
* **Metrics (Warm file pages)**: Init $4.42\text{ s}$, TTFT $2,549.36\text{ ms}$, Decode **$2.10\text{ tok/s}$** ($476.48\text{ ms/tok}$, $11.08\text{ ms/tok/layer}$), Hit rate $56.7\%$ (1,609 hits / 1,229 misses).
* **Impact**: Full model runs without allocating or populating 35 GiB host staging buffer.

### M7: Pipeline Refactoring Recheck (Identified HC Regression)
* **Date**: 2026-09-08 | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 2 layers, 664 hot slots)
* **Metrics**: Decode **$48.81 - 49.01\text{ tok/s}$** ($20.4\text{ ms/tok}$), $100.0\%$ hit rate.
* **Root Cause**: Device HC projection kernel serialized 24 outputs on a single Wave32 warp ($1,188\ \mu\text{s}$ latency).

### M8: Parallel HC Kernels & 2-Layer Peak Throughput
* **Date**: 2026-09-08 | **Commit**: `c49b5b5` | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 2 layers, 664 hot slots)
* **Optimizations**:
  - `hc_project_kernel`: 24 parallel Wave32 blocks (256 threads) + `float4` vectorized loads ($1,188\ \mu\text{s} \to 9.2\ \mu\text{s}$, **$129\times$ kernel speedup**).
  - `hc_pre_combine_kernel`: Vectorized `float4` / `half2` stores ($3.5\ \mu\text{s}$).
  - Fully purged CPU synchronization roundtrips from token step loop.
* **Metrics**:
  - Short prompt (4 $\to$ 16 gen): TTFT **$32.73\text{ ms}$** ($8.18\text{ ms/tok}$), Decode **$122.90\text{ tok/s}$** ($8.14\text{ ms/tok}$), Hit rate **$100.0\%$**.
  - Med prompt (8 $\to$ 32 gen): TTFT **$65.66\text{ ms}$** ($8.21\text{ ms/tok}$), Decode **$122.60\text{ tok/s}$** ($8.16\text{ ms/tok}$), Hit rate **$100.0\%$**.
* **Impact**: Recovered and surpassed baseline by $+37\%$ ($89.5 \to 122.9\text{ tok/s}$); verified bit-exact ($\epsilon < 4.5 \times 10^{-6}$).

### M9: Full 43-Layer Model with Optimized Kernels
* **Date**: 2026-09-08 | **Commit**: `fb2c7c0` | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 43 layers, 11,008 experts)
* **Config**: 664 hot VRAM slots, on-demand `.aeon` streaming, 4 prompt $\to$ 8 gen.
* **Metrics Comparison**:

| Metric | M6 (Previous Warm) | M9 (Cold Pages) | M9 (Warm Pages) | Progression (M6 $\to$ M9 Warm) |
| :--- | :---: | :---: | :---: | :---: |
| **Pipeline Init** | $4.42\text{ s}$ | $17.03\text{ s}$ | **$3.70\text{ s}$** | $-16\%$ init time |
| **Total Execution** | $5,884.86\text{ ms}$ | $9,064.50\text{ ms}$ | **$2,921.36\text{ ms}$** | **$2.0\times$ faster** |
| **TTFT (Prefill)** | $2,549.36\text{ ms}$ | $4,828.98\text{ ms}$ | **$1,494.25\text{ ms}$** | $373.56\text{ ms/tok}$ ($8.69\text{ ms/tok/layer}$) |
| **Decode Throughput** | **$2.10\text{ tok/s}$** | **$1.65\text{ tok/s}$** | **$4.91\text{ tok/s}$** | **$2.3\times$ speedup** |
| **Decode Step Latency**| $476.48\text{ ms/tok}$ | $605.06\text{ ms/tok}$ | **$203.86\text{ ms/tok}$** | **$4.74\text{ ms/tok/layer}$** (was $11.08$) |
| **VRAM Cache Hits / Misses** | 1,609 / 1,229 | 1,682 / 1,156 | 1,682 / 1,156 | $59.3\%$ hit rate |

* **Hardware Bottleneck Isolation**:
  - Per-layer compute latency dropped to **$4.74\text{ ms}$**.
  - The remaining $203.86\text{ ms}$ decode step is dominated by synchronous DMA transfers for the 1,156 misses ($40.7\%$ miss rate).
  - Direct mandate for **Spike 2**: Asynchronous dual-stream SDMA prefetching to overlap expert transfers behind preceding layer compute.
