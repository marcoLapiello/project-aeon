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
| **M10: Dual-Stream SDMA Prefetch** | 2L (Miss-stressed) | `.aeon` | 12 (pinned) | 6.2% | 71.10 ms | **32.8 tok/s** | 30.51 ms | Overlapped SDMA DMA with compute; $+71\%$ decode speed under severe cold-misses |

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

### M10: Dual-Stream Asynchronous SDMA Prefetching & Latency Hiding (Phase 2 Spike 2)
* **Date**: 2026-09-08 | **Commit**: `HEAD` | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 2 layers, 12 slots)
* **Optimization & Architecture**:
  - Implemented `PrefetchStagingArena`: 12-slot ($170\text{ MB}$) double-buffered pinned host RAM arena (`hipHostMallocPortable`), bypassing OS page faults and swap thrashing.
  - Decoupled `compute_stream` and `sdma_stream` with fine-grained HIP event synchronization (`hipEventRecord`, `hipStreamWaitEvent`).
  - Asynchronous PCIe transfers of missing routed experts execute concurrently with Shared Expert forward pass ($W_1, W_3, \text{SwiGLU}, W_2$).
* **Stress-Test Metrics (12 VRAM slots, 93.8% cold miss rate)**:
  - TTFT (Prefill): **$71.10\text{ ms}$** (down from $111.43\text{ ms}$, **$36.2\%$ faster**).
  - Decode Throughput: **$32.78\text{ tok/s}$** ($30.51\text{ ms/tok}$), compared to $19.11\text{ tok/s}$ synchronous baseline (**$+71.5\%$ speedup** under severe cold-miss pressure).
  - Cache Stats: $14\text{ hits} / 208\text{ misses}$ ($6.2\%$ hit rate) while sustaining smooth execution without blocking CPU roundtrips.
  - Bit-exact output verified on silicon: token `69146` at step 0 matching golden reference.

### M11: Direct I/O Cold Expert Integration Checkpoint (Phase 2 Spike 3)
* **Date**: 2026-09-08 | **Status**: Implementation checkpoint, target not yet met | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 2 layers)
* **Storage path**:
  - Added batched `io_uring` submission and completion harvesting with strict 4KB buffer, offset, and length validation.
  - Opened `model_experts.aeon` through a dedicated `O_DIRECT` descriptor and routed cold expert payloads directly into the existing staging arena before HIP SDMA upload.
  - Added explicit staging ownership transitions and matching cleanup for HIP-pinned and `posix_memalign` allocations.
* **Validation**:
  - Generic 128 MiB direct-I/O fixture: **$6.42\text{ GB/s}$**, bit-exact.
  - Six real expert blocks (81 MiB total): **$3.17\text{ GiB/s}$** after forced asynchronous submission and 4 MiB aligned subreads, bit-exact against `AeonModelLoader`.
  - Native pipeline: golden token `69146`, valid six-token generation, **$31.73\text{ tok/s}$** in the chunked-read regression run.
  - Cold-miss async pipeline: **$32.87\text{ tok/s}$** with 12 VRAM slots and 214 misses per layer in the final run.
* **Open work**:
  - The original `1.65\text{ GiB/s}` result was partly an implementation defect: `IORING_OP_READ` submissions executed synchronously inside `io_uring_enter` unless `IOSQE_ASYNC` was set. The corrected path reaches `3.17\text{ GiB/s}` for the same six experts.
  - The remaining gap to $\ge 6.0\text{ GB/s}$ is workload/layout dependent: a 1 GiB sequential read from the model reaches about `$5.88\text{ GiB/s}$`, while the six routed experts are scattered across a file with `1,552` physical extents. A fresh 128 MiB probe file had only `6` extents.
  - Unit clarification: each expert is `14,155,776` bytes, exactly `3,456` sectors, which is `13.5 MiB` or `14.155776 MB` decimal. The apparent `13.5 MB` versus `14.15 MB` discrepancy is binary versus decimal notation, not a format change.
  - End-to-end A/B (`bench_async_prefetch_io_modes`, identical 12-slot native pipeline): direct I/O measured `32.31` and `32.57 tok/s`; mmap source measured `33.57` and `33.22 tok/s`, with identical generated tokens. The mmap case warms its page-cache pages during the warmup step, while O_DIRECT bypasses that cache, so this is a steady-state behavior comparison rather than a cold-cache equivalence test.
  - Tier 2 warm-cache population remains disabled for this checkpoint because the prior contiguous preload caused host-memory pressure and swap contention.

### M12: Direct Warm-Tier Population & Bounded 3-Tier Integration (Phase 2 Spike 3)
* **Date**: 2026-09-08 | **Status**: Bounded integration passed; capacity/performance measurement open | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 43 layers)
* **Configuration**: Context 256; 676 Hot VRAM slots (`8.91 GiB`); 8 segmented Warm Host slots (`105.47 MiB`); 10,324 Cold NVMe slots.
* **Implementation**:
  - Initial Hot and Warm residents are populated through bounded 4 MiB `O_DIRECT`/`io_uring` batches; the dynamic-global preload no longer reads routed experts through the expert mmap.
  - Hot-to-Warm demotion uses VRAM-to-host DMA, and Warm-to-Hot promotion snapshots the warm payload before registry slot reuse.
  - Multi-layer staging ownership was corrected so the fixed 12-slot arena can be reused across all 43 layers.
* **Silicon validation**:
  - `test_hot_warm_cold_pipeline` passed on RX 7900 XTX with valid token `295`, 12 Hot hits, 0 Warm hits, and 24 Cold misses in one 43-layer step.
  - Focused Phase 2 group passed `5/5`: model direct-I/O parity, native pipeline, dynamic pool, asynchronous prefetch, and bounded hot/warm/cold integration.
* **Interpretation**: Direct I/O has now unlocked a real bounded Hot -> Warm -> Cold runtime path. The one-step workload did not select one of the eight preloaded Warm experts, so Warm-hit latency is not yet measured. Full-capacity Warm preload, controlled cold-cache comparison, storage-layout optimization, and the `>= 6.0 GB/s` model-backed target remain open.

### M13: Full-Model Direct Warm-Tier Benchmark (Phase 2 Spike 3)
* **Date**: 2026-09-08 | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 43 layers, 11,008 routed experts)
* **Configuration**: Context 4096; 664 Hot VRAM slots (`8.75 GiB`); 606 Warm Host slots (`7.99 GiB`); 9,738 Cold NVMe slots. Warm pool was populated through bounded 4 MiB `O_DIRECT`/`io_uring` reads.
* **Initialization**: `8.59 s`.
* **Measured generation**: 4 prompt tokens -> 8 generated tokens; output `[237, 223, 223, 223, 223, 223, 223, 223]`.
* **Performance**: TTFT `2,455.05 ms`; total measured execution `4,215.70 ms`; decode `3.98 tok/s` (`251.51 ms/tok`).
* **Tier service counts**: 1,682 Hot hits; 555 Warm Host hits; 601 Cold NVMe misses. Warm served `48.0%` of the lower-tier requests in this run.
* **Interpretation**: The significant Warm pool is now serving the full 43-layer inference path. This is a functional integration milestone, not the final performance target: the run still exposes 601 Cold requests, and the full approximately 35 GiB Warm configuration, controlled cold-cache A/B, storage-layout optimization, and the `>= 6.0 GB/s` model-backed target remain open.

### M14: Full-Capacity Warm-Tier Benchmark (Phase 2 Spike 3)
* **Date**: 2026-09-08 | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 43 layers, 11,008 routed experts)
* **Configuration**: Context 4096; 664 Hot VRAM slots (`8.75 GiB`); 2,654 Warm Host slots (`34.99 GiB`); 7,690 Cold NVMe slots. Warm pool was populated through bounded 4 MiB `O_DIRECT`/`io_uring` reads.
* **Initialization**: `24.05 s`; all 2,654 Warm slots allocated and populated successfully.
* **Measured generation**: 4 prompt tokens -> 8 generated tokens; output `[237, 223, 223, 223, 223, 223, 223, 223]`.
* **Performance**: TTFT `2,408.78 ms`; total measured execution `4,179.34 ms`; decode `3.95 tok/s` (`252.92 ms/tok`).
* **Tier service counts**: 1,682 Hot hits; 684 Warm Host hits; 472 Cold NVMe misses. Warm served `59.2%` of the lower-tier requests in this run.
* **Host-memory observation**: The process exited cleanly and returned host memory, but observed swap usage increased by approximately `0.7 GiB` during the run. The target capacity is therefore operational but still requires memory-pressure tuning before being treated as the default profile.
* **Interpretation**: Full-capacity Warm integration is complete. This pre-scheduling-fix run exposed the cost of synchronously demoting every evicted Hot expert into Warm. Remaining work is controlled cold-cache A/B measurement, reducing host pressure, physical expert-layout optimization, and reaching the `>= 6.0 GB/s` model-backed target.

### M15: Asynchronous Hot-to-Warm Demotion Validation (Phase 2 Spike 3)
* **Date**: 2026-09-08 | **Status**: Correctness and controlled Warm benefit validated | **Model**: DeepSeek-V4 INT4-W4A16 (`.aeon`, 43 layers)
* **Root cause fixed**: The first 35 GiB Warm implementation synchronously waited for a 14.15 MiB VRAM-to-host demotion on every non-Hot request. That made the Warm profile slower despite serving 684 Warm requests.
* **Fix**: Hot-to-Warm device-to-host transfers are now enqueued on `sdma_stream` with per-host-slot HIP events. The CPU waits only when a pending Warm payload is consumed or its host slot is about to be overwritten.
* **Controlled A/B**: Identical 43-layer workload, context 4096, 664 Hot slots, 4 prompt tokens -> 8 generated tokens.
  - Warm disabled (`0 GiB`): `4.37 tok/s`, 1,682 Hot hits, 1,156 Cold misses.
  - Warm enabled (`35 GiB`): `5.15 tok/s`, 1,682 Hot hits, 684 Warm hits, 472 Cold misses.
  - Generated token sequences were identical: `[237, 223, 223, 223, 223, 223, 223, 223]`.
* **Interpretation**: Warm is now demonstrably active and beneficial: `+17.8%` decode throughput versus the direct-cold control. The stable `59.3%` Hot hit rate is expected because Hot capacity and the routed access sequence are unchanged; Warm capacity changes lower-tier service cost, not Hot residency capacity.
