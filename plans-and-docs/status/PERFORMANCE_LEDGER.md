# Project Aeon - Performance & Accuracy Ledger

Authoritative silicon record for the AMD Radeon RX 7900 XTX (`gfx1100`). This
ledger keeps historical evidence, but only compares measurements that share an
explicit comparison key and pass the entry gate.

* **Last normalized**: 2026-09-11
* **Scope**: latency, throughput, storage supply, cache behavior, and numerical
  correctness across the Aeon milestones.
* **Status rule**: `[x] Invalidate for comparison` excludes the headline result
  from cross-entry comparisons; the raw observation remains useful historical
  evidence and its reason must be short and explicit.

## 1. Recording contract

- Compare only entries with the same **class**, **comparison key**, model
  artifact, workload, cache state, and instrumentation. The index below is a
  routing aid, not permission to pool unlike runs.
- Every card reports `n` and a statistic when the measurement is variable.
  Single-run values remain historical trend evidence, not statistical claims.
- Use `--` for a value that was not measured and `N/A` when it does not apply.
- Keep one-sentence conclusions in the ledger. Put implementation rationale,
  traces, and long investigation notes in the linked plan or review.

### Run card template

```markdown
### Mxx: Short milestone name
- **Run**: `YYYY-MM-DD`; commit or artifact; model/scope
- **Class / comparison key**: `E2E | Kernel | I/O | Primitive | Integration | Analysis` / `key`
- **Platform**: `baseline` or the complete deviation from the baseline below
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: prompt, context, layers, cache state, `n`, statistic
- **Metrics**: class-required metrics with units; control/baseline and delta
- **Correctness / service**: output gate, error, hit/miss or source-tier counts
- **Conclusion / next gate**: one sentence
- **Evidence**: test, benchmark, report, or artifact path
```

### Required metrics by class

| Class | Required fields |
| :--- | :--- |
| `E2E` | TTFT, decode throughput, decode step latency, service counts/hit rate, `n`/statistic, output gate |
| `Kernel` | median or p50 latency, throughput/effective bandwidth, launch shape, max error, `n` |
| `I/O` | throughput, request/extent conditions, payload parity, completion behavior, `n` |
| `Primitive` | shape, latency/throughput, reference error or tolerance, pass/fail |
| `Integration` | exercised path, pass/fail, service counts, explicitly untested path |
| `Analysis` | sample/corpus, coverage or observed distribution, decision, limitation |

## 2. Hardware testbed

| Field | Baseline |
| :--- | :--- |
| Host | AMD Ryzen Threadripper PRO 3975WX, 32C/64T, 3.5-4.2 GHz; 64 GB DDR4; 128 PCIe 4.0 lanes |
| GPU | 4x AMD Radeon RX 7900 XTX, Navi 31 / `gfx1100`, 24 GB GDDR6 each, PCIe 4.0 x16, P2P enabled |
| Storage | `/dev/nvme0n1p2`; approximately 6.33 GB/s sequential `io_uring` O_DIRECT read bandwidth |
| Toolchain | ROCm 7.2.2, native `hipcc`, Linux 7.0, Wave32 (`-mno-wavefrontsize64`) |

## 3. Comparison index

| Comparison key | Intended use | Entries |
| :--- | :--- | :--- |
| `primitive-fixture` | Same named component test and shape only | M1-M2 |
| `e2e-2L` | Two-layer `.aeon` generation; prompt and slot count must still match | M3-M4, M7-M8 |
| `e2e-2L-miss-stress` | Two-layer cold-miss stress with 12 pinned slots | M10 |
| `e2e-43L-legacy` | Early 43-layer runs; cache/page state differs, so trend only | M5-M6, M9 |
| `tier-integration` | Hot/Warm/Cold path integration and capacity gates, not a latency series | M12-M14 |
| `e2e-43L-A/B` | Context 4096, 4 prompt -> 8 generated, 664 Hot slots, matched Warm control | M15-M16, M18-M20 |
| `routing-locality` | History-based routing coverage; no placement claim | M17 |
| `direct-io` | Direct-I/O fixtures and model-backed payload reads | M11 |
| `native-text` | Native text turns; prompt, context, telemetry, and model variant differ | M21-M22, M25-M26 |
| `kernel-stage1` | Isolated synthetic swizzled/fused expert kernels | M23-M24 |

## 4. Milestone cards

### M1: Phase 0 Foundations & Hardware Spikes
- **Run**: `2026-09-07`; component fixtures
- **Class / comparison key**: `Primitive / primitive-fixture`; compare each named shape only
- **Platform**: `baseline`, Device 0
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: WMMA tile and GEMM, aligned direct I/O, and PCIe/compute overlap probes; `n=1` per fixture
- **Metrics**: WMMA tile error `0.0`; GEMM `25.6 TFLOP/s` at `870 us` for `2048 x 2048`; NVMe `6.33 GB/s`; overlap `24.9 GB/s` with `0.0%` compute jitter
- **Correctness / service**: FP16 tile and aligned direct-I/O payload checks passed
- **Conclusion / next gate**: Hardware and storage primitives are usable as baselines; these numbers are not end-to-end inference measurements
- **Evidence**: `test_wmma_tile.cpp`, `bench_wmma_gemm.cpp`, `test_direct_io.cpp`, `bench_async_overlap.cpp`

### M2: Single-GPU Mathematical Primitives (Phase 1 Spikes 1–5)
- **Run**: `2026-09-07`; component fixtures
- **Class / comparison key**: `Primitive / primitive-fixture`; compare identical test shape only
- **Platform**: `baseline`, Device 0
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: W4A16 projection, RMSNorm/SwiGLU, Sinkhorn, router, cached attention, and block-forward checks; `n=1` per fixture
- **Metrics**: projection `140.34 us` (`1.91 TFLOP/s`); RMSNorm error `<8.4e-4`; Sinkhorn error `<5.96e-8`; router top-6 match `100%`; attention `41.91 us` for 16 tokens (`2.62 us/token`); block `1.80 ms/token`, error `0.0033`
- **Correctness / service**: all named CPU-reference and assignment checks passed their recorded thresholds
- **Conclusion / next gate**: Mathematical primitives passed; full-model parity remains a separate gate
- **Evidence**: `test_w4a16_wmma.cpp`, `test_swiglu_clamp.cpp`, `test_hc_sinkhorn.cpp`, `test_moe_router.cpp`, `test_v4_attention.cpp`, `test_v4_block.cpp`

### M3: Autoregressive Pipeline Baseline (Phase 1 Spike 6)
- **Run**: `2026-09-07`; commit `5e27dd9`; DeepSeek-V4 INT4-W4A16, Safetensors
- **Class / comparison key**: `E2E / e2e-2L`; prompt length and slot count are part of the key
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: 2 layers (L0-L1), 8 slots/layer (16 total, 216 MB), greedy argmax; short `4 -> 16` and medium `8 -> 32`; `n=1` per prompt
- **Metrics**: short TTFT `422.20 ms`, decode `11.97 tok/s`, step `83.52 ms`, hit `1.8%`; medium TTFT `528.60 ms`, decode `19.11 tok/s`, step `52.33 ms`, hit `2.4%`
- **Correctness / service**: `97.6%` misses required synchronous host transfers
- **Conclusion / next gate**: Safetensors expert placement exposed the cold-miss trap; use the same workload key for the `.aeon` pool comparison
- **Evidence**: Phase 1 Spike 6 benchmark record

### M4: Dynamic Budgeting & Unified VRAM Expert Pool (Phase 2 Spike 1)
- **Run**: `2026-09-07`; commit `cc252c9`; DeepSeek-V4 INT4-W4A16, `.aeon`
- **Class / comparison key**: `E2E / e2e-2L`; compare with M3 only after matching artifact and slot configuration
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: 2 layers (L0-L1), 664 dynamic Hot slots (`8.75 GB`), 4K context; short `4 -> 16` and medium `8 -> 32`; `n=1` per prompt
- **Metrics**: short TTFT `43.76 ms`, decode `91.78 tok/s`, step `10.90 ms`, hit `100.0%`; medium TTFT `88.19 ms`, decode `89.48 tok/s`, step `11.18 ms`, hit `100.0%`
- **Correctness / service**: complete working set stayed resident; no cold-miss transfer in the measured sequences
- **Conclusion / next gate**: Unified Hot capacity removed the working-set miss trap; M8 is the comparable kernel/refactor follow-up
- **Evidence**: Phase 2 Spike 1 benchmark record

### M5: Full 43-Layer Execution on Single GPU
- **Run**: `2026-09-07`; DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers, 11,008 experts
- **Class / comparison key**: `E2E / e2e-43L-legacy`; cache state and prompt are part of the key
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: 664 Hot slots, 3,800 Warm slots, `4 -> 8` tokens; `n=1`
- **Metrics**: init `16.05 s`; TTFT `5,816.63 ms` (`33.82 ms/token/layer`); decode `1.47 tok/s` (`678.32 ms/token`); service `1,684 Hot / 1,154 Cold` (`59.3%` hit)
- **Correctness / service**: synchronous PCIe misses of `14.15 MB/expert` account for approximately `450 ms` of the step
- **Conclusion / next gate**: First complete 43-layer pass worked, but cold service dominated the baseline
- **Evidence**: full-model benchmark record

### M6: Full-Model Baseline with Bounded Cold Tier
- **Run**: `2026-09-08`; DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers
- **Class / comparison key**: `E2E / e2e-43L-legacy`; warm file pages, so do not compare with cold-page runs
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: 664 Hot slots; Warm preload disabled; 10,344 experts streamed from mmap; `4 -> 8`; `n=1`
- **Metrics**: init `4.42 s`; TTFT `2,549.36 ms`; decode `2.10 tok/s` (`476.48 ms/token`, `11.08 ms/token/layer`); service `1,609 Hot / 1,229 Cold` (`56.7%` hit)
- **Correctness / service**: full model ran without allocating or populating a 35 GiB host staging buffer
- **Conclusion / next gate**: Bounded cold storage removed the eager host allocation; cache/page state must be recorded for later comparisons
- **Evidence**: bounded cold-tier benchmark record

### M7: Pipeline Refactoring Recheck (Identified HC Regression)
- **Run**: `2026-09-08`; DeepSeek-V4 INT4-W4A16, `.aeon`
- **Class / comparison key**: `E2E / e2e-2L`; regression check against the 2-layer pool path
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: 2 layers, 664 Hot slots; `n=1`
- **Metrics**: decode `48.81-49.01 tok/s`; step approximately `20.4 ms`; hit `100.0%`
- **Correctness / service**: HC projection serialized 24 outputs on one Wave32 warp; kernel latency `1,188 us`
- **Conclusion / next gate**: Refactoring exposed a device-side HC regression; parallelize the projection before judging pipeline changes
- **Evidence**: pipeline refactoring recheck

### M8: Parallel HC Kernels & 2-Layer Peak Throughput
- **Run**: `2026-09-08`; commit `c49b5b5`; DeepSeek-V4 INT4-W4A16, `.aeon`
- **Class / comparison key**: `E2E / e2e-2L`; same 2-layer working-set family as M4/M7
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: 2 layers, 664 Hot slots; short `4 -> 16` and medium `8 -> 32`; `n=1` per prompt
- **Metrics**: short TTFT `32.73 ms`, decode `122.90 tok/s`, step `8.14 ms`, hit `100.0%`; medium TTFT `65.66 ms`, decode `122.60 tok/s`, step `8.16 ms`, hit `100.0%`; HC kernel `1,188 -> 9.2 us`
- **Correctness / service**: bit-exact within `4.5e-6`; `hc_pre_combine_kernel` `3.5 us`; CPU synchronization roundtrips removed
- **Conclusion / next gate**: Parallel HC recovered and exceeded M4 throughput by approximately `37%`; full-model impact remained open
- **Evidence**: `test_aeon_pipeline`, `bench_full_model.cpp`

### M9: Full 43-Layer Model with Optimized Kernels
- **Run**: `2026-09-08`; commit `fb2c7c0`; DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers
- **Class / comparison key**: `E2E / e2e-43L-legacy`; M6 control, cold-page run, and warm-page run are separate conditions
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: 664 Hot slots, on-demand `.aeon`, `4 -> 8`; `n=1` per condition
- **Metrics**:
  | Condition | Init | TTFT | Decode / step | Service |
  | :--- | ---: | ---: | :--- | :--- |
  | M6 warm control | `4.42 s` | `2,549.36 ms` | `2.10 tok/s` / `476.48 ms` | `1,609 / 1,229` Hot/Cold |
  | M9 cold pages | `17.03 s` | `4,828.98 ms` | `1.65 tok/s` / `605.06 ms` | `1,682 / 1,156` Hot/Cold |
  | M9 warm pages | `3.70 s` | `1,494.25 ms` | `4.91 tok/s` / `203.86 ms` | `1,682 / 1,156` Hot/Cold |
- **Correctness / service**: warm-page compute fell to `4.74 ms/token/layer`; `40.7%` misses still used synchronous DMA
- **Conclusion / next gate**: Optimized kernels doubled full-model throughput in warm pages; asynchronous miss hiding became the next gate
- **Evidence**: full-model optimized benchmark

### M10: Dual-Stream Asynchronous SDMA Prefetching & Latency Hiding (Phase 2 Spike 2)
- **Run**: `2026-09-08`; commit `HEAD`; DeepSeek-V4 INT4-W4A16, `.aeon`
- **Class / comparison key**: `E2E / e2e-2L-miss-stress`; compare only with the same 12-slot cold-miss workload
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: 2 layers, 12 pinned Hot slots, 12-slot (`170 MB`) pinned staging arena, `93.8%` cold misses; `n=1`
- **Metrics**: TTFT `71.10 ms` versus `111.43 ms` (`36.2%` faster); decode `32.78 tok/s` versus synchronous `19.11 tok/s` (`+71.5%`); step `30.51 ms`; service `14 Hot / 208 Cold` (`6.2%` hit)
- **Correctness / service**: token `69146` at step 0 matched the golden reference; compute and SDMA streams used HIP event barriers
- **Conclusion / next gate**: Dual-stream transfer overlap improved the severe-miss workload; direct-I/O source integration was the next gate
- **Evidence**: `test_async_prefetch.cpp`, `PrefetchStagingArena`

### M11: Direct I/O Cold Expert Integration Checkpoint (Phase 2 Spike 3)
- **Run**: `2026-09-08`; implementation checkpoint; DeepSeek-V4 INT4-W4A16, `.aeon`, 2 layers
- **Class / comparison key**: `I/O / direct-io`
- **Platform**: `baseline`
- [x] **Invalidate for comparison** | **Reason**: direct-I/O versus mmap A/B warmed page cache during warmup and is not cold-cache equivalent
- **Workload / configuration**: aligned `io_uring` reads, dedicated `O_DIRECT` descriptor, 4 MiB subreads, 12-slot native pipeline; six real experts and a 128 MiB fixture
- **Metrics**: fixture `6.42 GB/s`; six experts (`81 MiB`) `3.17 GiB/s`; native chunked-read `31.73 tok/s`; cold-miss async `32.87 tok/s` with 214 misses/layer; original synchronous-submission result `1.65 GiB/s`
- **Correctness / service**: payloads bit-exact against `AeonModelLoader`; staging ownership and HIP/`posix_memalign` cleanup passed; each expert is `14,155,776` bytes (`3,456` sectors)
- **Conclusion / next gate**: Direct-I/O integration works, but physical extent layout, cold-cache equivalence, and the `>=6.0 GB/s` model-backed target remain open
- **Evidence**: `test_model_direct_io.cpp`, `bench_async_prefetch_io_modes`

### M12: Direct Warm-Tier Population & Bounded 3-Tier Integration (Phase 2 Spike 3)
- **Run**: `2026-09-08`; DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers
- **Class / comparison key**: `Integration / tier-integration`
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: context `256`; `676` Hot (`8.91 GiB`), `8` segmented Warm (`105.47 MiB`), `10,324` Cold; bounded 4 MiB direct-I/O batches
- **Metrics**: one 43-layer step: `12 Hot / 0 Warm / 24 Cold`; valid token `295`; focused Phase 2 group `5/5`
- **Correctness / service**: Hot/Warm/Cold ownership and reusable 12-slot staging passed; no Warm-hit latency was measured
- **Conclusion / next gate**: Bounded three-tier integration passed; full-capacity Warm, cold-cache comparison, layout, and throughput gates remained open
- **Evidence**: `test_hot_warm_cold_pipeline`, focused Phase 2 CTest group

### M13: Full-Model Direct Warm-Tier Benchmark (Phase 2 Spike 3)
- **Run**: `2026-09-08`; DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers, 11,008 experts
- **Class / comparison key**: `E2E / tier-integration`
- **Platform**: `baseline`
- [x] **Invalidate for comparison** | **Reason**: single run with no matched control; retain as Warm-service integration evidence
- **Workload / configuration**: context `4096`; `664` Hot (`8.75 GiB`), `606` Warm (`7.99 GiB`), `9,738` Cold; `4 -> 8`; `n=1`
- **Metrics**: init `8.59 s`; TTFT `2,455.05 ms`; total `4,215.70 ms`; decode `3.98 tok/s` (`251.51 ms/token`); service `1,682 Hot / 555 Warm / 601 Cold`; Warm handled `48.0%` of lower-tier requests
- **Correctness / service**: output `[237, 223, 223, 223, 223, 223, 223, 223]`; bounded direct-I/O preload completed
- **Conclusion / next gate**: Full-model Warm service works, but this run cannot establish performance without repeated matched controls
- **Evidence**: full-model Warm-tier benchmark

### M14: Full-Capacity Warm-Tier Benchmark (Phase 2 Spike 3)
- **Run**: `2026-09-08`; DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers, 11,008 experts
- **Class / comparison key**: `E2E / tier-integration`
- **Platform**: `baseline`
- [x] **Invalidate for comparison** | **Reason**: single run had no matched control and increased swap by approximately 0.7 GiB
- **Workload / configuration**: context `4096`; `664` Hot (`8.75 GiB`), `2,654` Warm (`34.99 GiB`), `7,690` Cold; `4 -> 8`; `n=1`
- **Metrics**: init `24.05 s`; TTFT `2,408.78 ms`; total `4,179.34 ms`; decode `3.95 tok/s` (`252.92 ms/token`); service `1,682 Hot / 684 Warm / 472 Cold`; Warm handled `59.2%` of lower-tier requests
- **Correctness / service**: all Warm slots allocated and populated; output `[237, 223, 223, 223, 223, 223, 223, 223]`; process exited cleanly
- **Conclusion / next gate**: Full-capacity Warm was functional but host pressure and synchronous demotion prevented it from being a reliable performance baseline
- **Evidence**: full-capacity Warm-tier benchmark

### M15: Asynchronous Hot-to-Warm Demotion Validation (Phase 2 Spike 3)
- **Run**: `2026-09-08`; DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers
- **Class / comparison key**: `E2E / e2e-43L-A/B`; context and token counts match M16 and M18-M20
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: context `4096`, `664` Hot slots, `4 -> 8`; Warm `0 GiB` versus `35 GiB`; `n=1` per variant
- **Metrics**: Warm off `4.37 tok/s`, `229.1 ms/token`, `1,682 Hot / 1,156 Cold`; Warm on `5.15 tok/s`, `194.2 ms/token`, `1,682 Hot / 684 Warm / 472 Cold`; decode delta `+17.8%`
- **Correctness / service**: generated IDs identical: `[237, 223, 223, 223, 223, 223, 223, 223]`; asynchronous D2H used per-slot HIP events
- **Conclusion / next gate**: Warm became beneficial once demotion stopped blocking the request path; repeated A/B runs are still needed for uncertainty bounds
- **Evidence**: controlled Warm A/B benchmark

### M16: Demotion-Free Warm Path & Pre-Staging Shared-Expert Enqueue (Expert Review Step 1)
- **Run**: `2026-09-08`; DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers; [review](../analysis/historical/EXPERT_PERFORMANCE_REVIEW.md)
- **Class / comparison key**: `E2E / e2e-43L-A/B`; same workload as M15
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: context `4096`, `664` Hot slots, `4 -> 8`; Warm `0` versus `35 GiB`; `n=1` per variant
- **Metrics**: Warm off `4.36 tok/s` (`229.30 ms/token`), `1,682 Hot / 1,156 Cold`; Warm on `5.11 tok/s` (`195.82 ms/token`), `1,682 Hot / 157 Warm / 999 Cold`; warm traffic reduced from approximately `42 MB` to `14 MB` per hit without a throughput gain over M15
- **Correctness / service**: regression suite passed; golden token `295`; generated IDs matched M15; shared-expert enqueue moved ahead of prefetch
- **Conclusion / next gate**: Removing request-path demotion confirmed a latency-bound just-in-time miss path; cross-token prefetch and repeated A/B measurement became the next gates
- **Evidence**: linked review; `test_model_direct_io`, `test_aeon_pipeline`, `test_dynamic_expert_pool`, `test_async_prefetch`, `test_hot_warm_cold_pipeline`

### M17: Routing-Locality Measurement — Historical Speculation Rejected (Expert Review Step 3)
- **Run**: `2026-09-08`; [review](../analysis/historical/EXPERT_PERFORMANCE_REVIEW.md); DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers
- **Class / comparison key**: `Analysis / routing-locality`
- **Platform**: `baseline`
- [x] **Invalidate for comparison** | **Reason**: repetitive output and no held-out corpus; do not use as a general placement benchmark
- **Workload / configuration**: `AEON_MEASURE_LOCALITY=1`; Warm `35 GiB`; context `4096`; `4 -> 8`; `400` gated layer pairs
- **Metrics**: previous-set union coverage: `n=1 3.38/6 (56%)`, `n=2 3.60/6 (60%)`, `n=3 3.60/6 (60%)`, `n=4 3.44/6 (57%)`; full `6/6` rates `17.5/22.8/26.0/27.8%`; Hot hit `59.3%` at `5.10 tok/s`
- **Correctness / service**: instrumentation is env-gated and zero-cost when disabled; all regressions passed; tokens matched M16
- **Conclusion / next gate**: History-based speculation was not justified for this run; diverse prompts and held-out data are required before a placement decision
- **Evidence**: `V4Pipeline` locality observer and linked review

### M18: W4A16 Decode GEMV Rewrite — 10× Routed-Expert GEMM Speedup (Expert Review Step 4)
- **Run**: `2026-09-08`; [review](../analysis/historical/EXPERT_PERFORMANCE_REVIEW.md); DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers
- **Class / comparison key**: `E2E / e2e-43L-A/B`; supporting kernel result is tagged in the metrics
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: context `4096`, `4 -> 8`, 664 Hot slots, Warm off versus `35 GiB`; `n=1` per variant; decode uses `M=1` GEMV
- **Metrics**: W1/W3 `138.8 -> 13.9 us` (`10.0x`, `340 GB/s`); W2 `13.9 us` (`338 GB/s`); Warm off `4.36 -> 5.25 tok/s` (`+20.4%`); Warm on `5.11 -> 5.79 tok/s` (`+13.3%`), TTFT `2,755 -> 1,835 ms`
- **Correctness / service**: kernel max diff `0` versus CPU FP32; regression suite and golden token `295` passed; output `[237, 201, 1778, ...]` was deterministic and tier-independent, but differs from earlier near-tie argmax output
- **Conclusion / next gate**: Routed-expert GEMV time fell from approximately `108 ms` to `11 ms/token`, exposing the just-in-time miss path as the next bottleneck
- **Evidence**: [w4a16_gemm.hpp](../../src/kernel/w4a16_gemm.hpp), `test_w4a16_wmma`, linked review

### M19: Contiguous Per-Slot VRAM Layout & DMA Stream Split (Expert Review Step 2)
- **Run**: `2026-09-08`; [review](../analysis/historical/EXPERT_PERFORMANCE_REVIEW.md); DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers
- **Class / comparison key**: `E2E / e2e-43L-A/B`; same workload as M18
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: context `4096`, `4 -> 8`, 664 Hot slots, Warm off versus `35 GiB`; `n=1` per variant
- **Metrics**: Warm `35 GiB` `5.79 -> 5.88 tok/s` (`172.6 -> 170.0 ms/token`), TTFT `1,835 -> 1,803 ms`; Warm off `5.25 -> 5.39 tok/s` (`190.4 -> 185.6 ms/token`)
- **Correctness / service**: one contiguous `13.5 MiB` region per slot and one H2D copy replaced six; separate cold SDMA stream; regression group `5/5` passed; output matched M18
- **Conclusion / next gate**: Relayout and stream split delivered `+1.5-2.7%`; synchronous cold-read exposure remained dominant
- **Evidence**: `UnifiedVRAMExpertPool`, `test_dynamic_expert_pool`, `test_aeon_pipeline`, `test_hot_warm_cold_pipeline`, `test_async_prefetch`, `test_w4a16_wmma`

### M20: Per-Layer CPU-Stall Removal & GPU Argmax (Expert Review Step 5)
- **Run**: `2026-09-08`; commit `HEAD`; [review](../analysis/historical/EXPERT_PERFORMANCE_REVIEW.md); DeepSeek-V4 INT4-W4A16, `.aeon`, 43 layers
- **Class / comparison key**: `E2E / e2e-43L-A/B`; same workload as M19
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: context `4096`, `4 -> 8`, 664 Hot slots, Warm off versus `35 GiB`; `n=1` per variant
- **Metrics**: Warm `35 GiB` `5.88 -> 7.10 tok/s` (`170.0 -> 140.8 ms/token`), TTFT `1,803 -> 1,801 ms`; Warm off `5.39 -> 6.32 tok/s` (`185.6 -> 158.2 ms/token`)
- **Correctness / service**: device-side router conversion and GPU argmax removed 43 per-layer drains and 258 KB/token CPU readback; golden token `295` and regression group passed; output returned to `[237, 223 x7]`
- **Conclusion / next gate**: CPU-stall removal cut approximately `29 ms/token`; cold-read latency and novel experts remained the limiting path
- **Evidence**: [v4_attention.hpp](../../src/kernel/v4_attention.hpp), [v4_pipeline.hpp](../../src/core/v4_pipeline.hpp), [v4_pipeline_scratch.hpp](../../src/core/v4_pipeline_scratch.hpp), linked review

### M21: Native Text-In/Text-Out Frontend and 43-Layer Smoke
- **Run**: `2026-09-09`; DeepSeek-V4 INT4-W4A16, `.aeon`
- **Class / comparison key**: `E2E / native-text-smoke`
- **Platform**: `baseline`
- [x] **Invalidate for comparison** | **Reason**: smoke output was `? ?`; no external behavior or activation-parity reference
- **Workload / configuration**: 43 layers, context `512`, 675 Hot slots, Warm disabled; chat prompt `What is 2+2?`; `n=1`
- **Metrics**: TTFT `4,010.18 ms`; decode `5.57 tok/s`; IDs `[33, 539, 33, 539]`; stop `max_new_tokens`
- **Correctness / service**: tokenizer, formatter, generation limits, and detokenization ran without Python; native prompt IDs recorded in the source report
- **Conclusion / next gate**: Native plumbing worked, but this is not a model-quality or placement-data measurement
- **Evidence**: tokenizer preparation, `aeon_chat`, native text smoke

### M22: Corrected Native Inference and Complete Text Turn
- **Run**: `2026-09-09`; DeepSeek-V4 INT4-W4A16, `.aeon`
- **Class / comparison key**: `E2E / native-text-simple-turn`
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: 43 layers, context `1024`, 674 Hot slots, Warm disabled; chat prompt `What is the capital of France?`; EOS-bounded; `n=1`
- **Metrics**: TTFT `4,084.13 ms`; decode `3.04 tok/s`; IDs `[671, 6102, 294, 8760, 344, 2619, 51119, 42499, 1]`; stop `eos`
- **Correctness / service**: response `The capital of France is **Paris**.`; router conversion, GPU argmax, query RMSNorm, and gate correction fixes passed the simple-turn gate
- **Conclusion / next gate**: Complete native text turn works, but broader reference parity and compressed/indexed attention remain open
- **Evidence**: native text generation and pipeline regression records

### M23: Stage 1 Swizzled and Fused Expert Kernel Measurement
- **Run**: `2026-09-10`; target RX 7900 XTX (`gfx1100`)
- **Class / comparison key**: `Kernel / kernel-stage1`
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: synthetic weights, version-2 swizzled layout, isolated HIP-event timing; excludes model load, NVMe, staging, routing, and cache misses; `n=101` trace iterations
- **Metrics**: W1/W3 `12.822 -> 11.152 us` (`1.150x`, `368 -> 423 GB/s`); W2 `13.077 -> 10.926 us` (`1.197x`, `361 -> 432 GB/s`); dual W1/W3 `29.012 -> 15.816 us` (`1.834x`); fused W1/W3 `147.557 -> 39.775 us` (`3.710x`); fused W2 `107.486 -> 28.652 us` (`3.751x`)
- **Correctness / service**: individual and dual-launch max diff `0.000`; fused W1/W3 `0.016`, fused W2 `0.002`; launch trace reduced fused paths to 101 kernels; GL2C counters unsupported, so no counter claim
- **Conclusion / next gate**: Isolated kernels improved, but full-model impact requires a controlled Hot/Warm run; the `M=16` WMMA check at `149.234 us` is not a decode comparison
- **Evidence**: `bench_aeon_moe_fused_w13`, `bench_aeon_moe_fused_w2`, `rocprofv2 --kernel-trace`

### M24: Activation-Staging A/B Experiment
- **Run**: `2026-09-10`; target RX 7900 XTX (`gfx1100`)
- **Class / comparison key**: `Kernel / kernel-stage1`
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: same synthetic inputs, geometry, outputs, and FP32 accumulation as M23; direct versus `STAGE_ACTIVATION=true`; W1/W3 `n=5` alternating trials, W2 repeated
- **Metrics**: W1/W3 direct `37.170 us` median (range `37.052-37.202`) versus staged `41.354 us` (`40.526-42.040`), ratio `0.899x`; W2 first `26.045` versus `25.937 us` (`1.004x`), repeat `25.361` versus `24.988 us` (`1.015x`)
- **Correctness / service**: all direct/staged output max differences `0.000`; staged LDS allocation `8 KiB` for W1/W3 and `4 KiB` for W2
- **Conclusion / next gate**: Do not enable activation staging; W1/W3 is slower and W2 is neutral within variation
- **Evidence**: fused kernel A/B benchmark

### M25: Persistent Warm Refill and Supply Telemetry Closure
- **Run**: `2026-09-11`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, `.aeon`, 43 layers; [A/B report](../execution/active/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_AB_REPORT.md)
- **Class / comparison key**: `E2E / native-text-warm-A/B`
- **Platform**: `baseline`; RX 7900 XTX, ROCm 7.2.2, Linux 7.0.0-31-generic
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: context `256`; 676 Hot, 2,642 Warm at `35 GiB`, 12 staging slots, queue depth `2`; prompt `What is 2 + 2? Answer briefly.`; max 8 new tokens; `n=5` per variant after fixed startup
- **Metrics**:
  | Variant | Cold bytes/decode token | Service | TTFT / decode |
  | :--- | ---: | :--- | :--- |
  | Warm `0` | `2,052,587,520` | `0 Warm / 145 Cold` | `5,156.91 ms` / `2.62 tok/s` |
  | Warm, refill off | `1,854,406,656` | `14 Warm / 131 Cold` | control |
  | Warm, refill on | `1,429,733,376` | `44 Warm / 101 Cold` | `4,907.82 ms` / `3.11 tok/s` |
- **Correctness / service**: repaired Q1-Q3 TTFT `4,890.84-4,926.32 ms`, decode `3.10-3.13 tok/s`; all 15 runs output `[22, 1]` and stopped on `eos`; CTest `19/19`; VmSwap delta `0`; D2H `778,567,680` bytes and `55/55` submissions/completions in the captured artifact
- **Conclusion / next gate**: Persistent refill reduced measured Cold supply cost without CPU waiting or output changes; the artifact predates demotion-accounting correction, and host pressure/model correctness remain separate gates
- **Evidence**: linked A/B report and source-tier JSONL telemetry

### M26: Full Real-Prompt Hot/Warm/Cold Chat Comparison
- **Run**: `2026-09-11`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, `.aeon`, 43 layers
- **Class / comparison key**: `E2E / native-text-real-prompt`
- **Platform**: `baseline`; RX 7900 XTX, ROCm 7.2.2, Linux 7.0.0-31-generic
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: native tokenizer/formatter; prompt `What is the capital of France?`; context `1024`; EOS-bounded greedy generation; `n=3` per variant; wall time includes Warm preload
- **Metrics**:
  | Variant | TTFT median (range) | Decode median (range) | Wall / service |
  | :--- | ---: | ---: | :--- |
  | Hot/Cold | `4,470.38 ms` (`4,357.54-4,581.78`) | `2.74 tok/s` (`2.72-2.77`) | `15.00 s`; `1,068 Hot / 996 Cold` |
  | Hot/Warm/Cold | `3,757.83 ms` (`3,683.64-3,796.14`) | `3.56 tok/s` (`3.56-3.57`) | `24.36 s`; `1,068 Hot / 401 Warm / 595 Cold` |
  | Hot/Cold, telemetry off | `4,222.51 ms` (`4,217.88-4,388.50`) | `2.87 tok/s` (`2.85-2.88`) | control for instrumentation |
- **Correctness / service**: all six instrumented runs output `[671, 6102, 294, 8760, 344, 2619, 51119, 42499, 1]` and stopped on `eos`; Warm reduced Cold bytes/token `1,762,394,112 -> 1,052,835,840` (`40.3%`); persistent Warm allocation `37,399,560,192` bytes; unpinned allocation and VmSwap delta `0`
- **Conclusion / next gate**: Within this instrumented A/B, Warm reduced TTFT `15.9%`, step latency `23.0%`, and increased decode `29.9%`; telemetry adds approximately `5.9%` TTFT and `4.5%` decode cost, so cross-day M22 comparison is only a regression signal
- **Evidence**: three-run comparison artifact and source-tier telemetry
