# Project Aeon - Performance & Accuracy Ledger

Authoritative silicon record for the AMD Radeon RX 7900 XTX (`gfx1100`).

* **Last normalized**: 2026-09-18
* **Scope**: latency, throughput, storage supply, cache behavior, and numerical correctness.
* **Status rule**: `[x] Invalidate for comparison` excludes the headline result from cross-entry comparisons.

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
| `kernel-stage1` | Isolated synthetic swizzled/fused expert kernels | M23-M24 |
| `e2e-43L-text` | Text in/text out at 43 layers; single-token decode, no batching | M28 |
| `supply-telemetry` | Per-phase, per-tier supply request/byte counters | M29 |
| `budget-cap` | Hot VRAM expert-pool cap feasibility and behavior | M30 |
| `tier-invariance` | Two runs, different tiers, byte-identical logits | M31 |
| `starved-pool` | Hot-pool cap forcing the emergency drain; logits still exact | M32 |
| `demotion-ab` | Demotion-queue capacity 2 vs 6; drops, Warm, NVMe, D2H | M33 |

## 4. Milestone cards

### M1: Phase 0 Foundations & Hardware Spikes
- **Run**: `2026-09-07`; component fixtures
- **Class / comparison key**: `Primitive / primitive-fixture`; compare each named shape only
- **Platform**: `baseline`, Device 0
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: WMMA tile and GEMM, aligned direct I/O, and PCIe/compute overlap probes; `n=1` per fixture
- **Metrics**: WMMA tile error `0.0`; GEMM `25.6 TFLOP/s` at `870 us` for `2048 x 2048`; NVMe `6.33 GB/s`; overlap `24.9 GB/s` with `0.0%` compute jitter
- **Correctness / service**: FP16 tile and aligned direct-I/O payload checks passed
- **Conclusion / next gate**: Baselines; not end-to-end inference numbers
- **Evidence**: Phase 0 primitive probes (retired)

### M2: Single-GPU Mathematical Primitives (Phase 1 Spikes 1–5)
- **Run**: `2026-09-07`; component fixtures
- **Class / comparison key**: `Primitive / primitive-fixture`; compare identical test shape only
- **Platform**: `baseline`, Device 0
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: W4A16 projection, RMSNorm/SwiGLU, Sinkhorn, router, cached attention, and block-forward checks; `n=1` per fixture
- **Metrics**: projection `140.34 us` (`1.91 TFLOP/s`); RMSNorm error `<8.4e-4`; Sinkhorn error `<5.96e-8`; router top-6 match `100%`; attention `41.91 us` for 16 tokens (`2.62 us/token`); block `1.80 ms/token`, error `0.0033`
- **Correctness / service**: all named CPU-reference and assignment checks passed their recorded thresholds
- **Conclusion / next gate**: Primitive fixtures passed; full-model parity is a separate gate
- **Evidence**: `test_w4a16_swizzle.cpp`, `test_w4a16_swizzled_gemv.cpp`, `test_swiglu_clamp.cpp`, `test_hc_sinkhorn.cpp`, `test_moe_router.cpp`. The `test_v4_attention.cpp` and block fixtures that produced the latency figures are retired.

### M23: Stage 1 Swizzled and Fused Expert Kernel Measurement
- **Run**: `2026-09-10`; target RX 7900 XTX (`gfx1100`)
- **Class / comparison key**: `Kernel / kernel-stage1`
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: synthetic weights, version-2 swizzled layout, isolated HIP-event timing; excludes model load, NVMe, staging, routing, and cache misses; `n=101` trace iterations
- **Metrics**: W1/W3 `12.822 -> 11.152 us` (`1.150x`, `368 -> 423 GB/s`); W2 `13.077 -> 10.926 us` (`1.197x`, `361 -> 432 GB/s`); dual W1/W3 `29.012 -> 15.816 us` (`1.834x`); fused W1/W3 `147.557 -> 39.775 us` (`3.710x`); fused W2 `107.486 -> 28.652 us` (`3.751x`)
- **Correctness / service**: individual and dual-launch max diff `0.000`; fused W1/W3 `0.016`, fused W2 `0.002`; launch trace reduced fused paths to 101 kernels; GL2C counters unsupported, so no counter claim
- **Conclusion / next gate**: Isolated kernels only; full-model impact not measured. The `M=16` WMMA check at `149.234 us` is not a decode comparison
- **Evidence**: `bench_aeon_moe_fused_w13`, `test_aeon_moe_fused_w13` (retired); `test_aeon_moe_fused_w2` (built); `rocprofv2 --kernel-trace`

### M24: Activation-Staging A/B Experiment
- **Run**: `2026-09-10`; target RX 7900 XTX (`gfx1100`)
- **Class / comparison key**: `Kernel / kernel-stage1`
- **Platform**: `baseline`
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: same synthetic inputs, geometry, outputs, and FP32 accumulation as M23; direct versus `STAGE_ACTIVATION=true`; W1/W3 `n=5` alternating trials, W2 repeated
- **Metrics**: W1/W3 direct `37.170 us` median (range `37.052-37.202`) versus staged `41.354 us` (`40.526-42.040`), ratio `0.899x`; W2 first `26.045` versus `25.937 us` (`1.004x`), repeat `25.361` versus `24.988 us` (`1.015x`)
- **Correctness / service**: all direct/staged output max differences `0.000`; staged LDS allocation `8 KiB` for W1/W3 and `4 KiB` for W2
- **Conclusion / next gate**: Activation staging is slower on W1/W3 and neutral on W2; do not enable
- **Evidence**: fused kernel A/B benchmark (retired)

### M28: 43-Layer Text-In/Text-Out at `≈3 tok/s`
- **Run**: `2026-09-17`; branch `rewrite/graph-v2`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, 43 layers, 11,008 experts
- **Class / comparison key**: `E2E / e2e-43L-text`
- **Platform**: `baseline`, Device 0 only; RX 7900 XTX, ROCm 7.2.2, `gfx1100`, Wave32
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: acceptance prompt `What is the capital of France?` (`11` prompt tokens) and `8` single-turn thinking-mode prompts; artifact sampling policy (`T=1.0`, `top_p=1.0`); `n=1` per prompt
- **Metrics**:

  | Configuration | Hot / Warm / Cold | TTFT | Decode | Step |
  | :--- | :--- | ---: | ---: | ---: |
  | context `256`, Warm `0` | `675 / 0 / 7207` | `4,487 ms` | `2.7 tok/s` | `~370 ms` |
  | context `32768`, Warm `40 GiB` | `779 / 3022 / 7207` | `4,573.81–11,509.82 ms` | `3.42–3.64 tok/s` | `274.73–292.40 ms` |

  Prefill `302.89–340.26 ms/token`; decode step `274.73–292.40 ms` (`≈1.13×` decode per prompt token). VRAM held `23.75 GiB = 99.06%` at context `32768` (`rocm-smi --showpids`), `VmSwap 0`. Budget correction: Hot `675 → 809`, VRAM held `21.99 → 23.76 GiB` at context `256`.
- **Correctness / service**: output `The capital of France is **Paris**.`, stop `eos`, `9` tokens; per-tier counts `--` (supply not instrumented on this path)
- **Conclusion**: decode `≈3 tok/s` in all configurations; Hot/Cold-only vs Hot/Warm/Cold moves it less than the run-to-run spread. Prefill not amortized. Warm bound by physical RAM (`≈40 GiB`), not the `52.62 GB` budget cap.
- **Evidence**: `tools/aeon_chat.cpp`, `rocm-smi --showpids`

### M29: Supply telemetry live — tier configuration moves the measured bytes
- **Run**: `2026-09-21`; branch `main`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, 43 layers
- **Class / comparison key**: `Analysis / supply-telemetry`
- **Platform**: `baseline`, Device 0 only
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: acceptance prompt `What is the capital of France?`, context `256`, `--greedy`, `8` generated tokens, `n=1`; run A `--warm-gib 0`, run B `--warm-gib 40`; `--supply-telemetry` JSONL
- **Metrics**: per `phase_summary` row (`request_count`, bytes):

  | Run | Phase / tier | req | bytes |
  | :--- | :--- | ---: | ---: |
  | A Warm `0` | prefill cold | `1516` | `21.46 GB` NVMe |
  | A Warm `0` | decode cold | `747` | `10.57 GB` NVMe (`1.51 GB/tok`) |
  | B Warm `40` | decode cold | `434` | `6.14 GB` NVMe |
  | B Warm `40` | decode **warm** | `313` | `4.43 GB` host (`logical_bytes_from_warm`) |

  Run A decode is Cold-only; run B serves `42%` of decode requests from Warm and cuts NVMe bytes `42%` (`10.57 → 6.14 GB`). Demotion, run B decode: `747` attempts, `254` dropped (`34%`).
- **Correctness / service**: identical output in both runs (`The capital of France is **Paris**.`); `Hot` occupancy steady `809/809`; Warmup phase empty as expected (preloads bypass the supply)
- **Conclusion / next gate**: Step 1 gate met — `phase_summary` rows carry `request_count > 0` and the tier bytes move with the tier configuration; `logical_bytes_from_warm` is `0` in A and `> 0` in B
- **Evidence**: `tools/aeon_chat.cpp` `--supply-telemetry`, `/tmp/aeon-telemetry/run-{a-warm0,b-warm40}.jsonl` (Step 1 of [Expert Streaming Execution Plan](../execution/active/EXPERT_STREAMING_EXECUTION_PLAN.md))

### M30: Hot-slot pressure knob — the starved pool is reachable
- **Run**: `2026-09-21`; branch `main`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, 43 layers
- **Class / comparison key**: `Analysis / budget-cap`
- **Platform**: `baseline`, Device 0 only
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: acceptance prompt `What is the capital of France?`, context `256`, `--greedy`, `n=1`; `--max-hot-slots 12` vs the derived `809`; `--supply-telemetry` JSONL
- **Metrics**: at `--max-hot-slots 12` the report shows `Tier 1: Hot VRAM : 12 slots (0.16 GB)` and `[Hot cap] requested 12 slots, applied 12 slots`; `Feasibility Status: [FEASIBLE / APPROVED]`. Prefill telemetry: `2838` cold requests / `40.17 GB` NVMe for an `11`-token prompt ≈ `3.65 GB/token`, versus `1.79 GB/token` at the derived `809` slots. Unit test: cap `12 → 12`, cap `3 → 6` (floored), cap above derived `→ 817` (unchanged).
- **Correctness / service**: one token completes at the cap; `is_feasible` true; output intact
- **Conclusion / next gate**: Step 2 gate met — the cap is honored, feasibility holds, and the graph runs starved; the derived pool on this GPU never reaches this regime, so this knob is what makes Step 4's drain gate testable
- **Evidence**: `tools/aeon_chat.cpp` `--max-hot-slots`, `tests/test_dynamic_expert_pool.cpp` (Test 2b), `/tmp/aeon-telemetry/run-c-cap12.jsonl`

### M31: Tier-invariance gate — the answering tier does not change the number
- **Run**: `2026-09-21`; branch `main`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, 43 layers
- **Class / comparison key**: `Analysis / tier-invariance`
- **Platform**: `baseline`, Device 0 only
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: acceptance prompt `What is the capital of France?`, context `32768`, `--greedy`, `8` generated tokens, `n=1`; run A `--warm-gib 0` (Hot+Cold), run B `--warm-gib 40` (Hot+Warm); both `--dump-logits` and `--supply-telemetry`
- **Metrics**: logits dumps **byte-identical** (`cmp`, `4,654,080 B` each); generated token ids identical; total request counts identical across runs — prefill `1521` (`A`: `1521` Cold; `B`: `1029` Cold + `492` Warm), decode `760` (`A`: `760` Cold; `B`: `437` Cold + `323` Warm). `logical_bytes_from_warm`: A `0`, B `11.54 GB`. Demotion drops in B: decode `156/437` (`36%`)
- **Correctness / service**: `registry.invariants_hold()` true in both; `outstanding_leases == 0` in both; output `The capital of France is **Paris**.` in both
- **Conclusion / next gate**: Step 3 gate met — the tier that answered did not change a single byte of the logits, so the supply is numerically invisible; the identical request counts confirm the comparison is of *answering tier*, not *workload*
- **Evidence**: `scripts/expert_tier_invariance.sh`, `/tmp/aeon-tier-invariance.dh1W11/{A,B}.{logits.bin,telemetry.jsonl}`

### M32: Starved-pool gate — the emergency drain runs and stays exact
- **Run**: `2026-09-21`; branch `main`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, 43 layers
- **Class / comparison key**: `Analysis / starved-pool`
- **Platform**: `baseline`, Device 0 only
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: acceptance prompt `What is the capital of France?`, context `32768`, `--greedy`, `8` generated tokens, `n=1`; reference `--warm-gib 0` (derived `779` slots), starved `--warm-gib 0 --max-hot-slots 12`; both `--dump-logits`
- **Metrics**: reference `forced_drains=0`; starved `forced_drains=378`, `staging_in_use=0` in both. Starved prefill `2838` Cold requests / `40.17 GB` NVMe; decode `1806` / `25.57 GB` NVMe — versus the uncapped Run A's `10.76 GB` decode, i.e. `2.4×` the NVMe traffic
- **Correctness / service**: logits **byte-identical** to the uncapped reference (`cmp`); `registry.invariants_hold()` true; `outstanding_leases == 0`; staging arena fully drained
- **Warm-enabled starved run (mid-eviction hazard probe)**: `--max-hot-slots 12 --warm-gib 40`, first the `8`-token prompt then a `96`-token generation. Both exit `0` — the `"request-path CPU synchronization is forbidden"` throw was **not reached**. The 96-token run: `forced_drains=2478`, `staging_in_use=0`, `invariants_hold=true`, all demotion drops `queue_pressure`. Starved+Warm decode NVMe `16.52 GB` versus starved Warm-off `25.57 GB` (`-35%`). Logits identical to *all three* other configurations (uncapped Warm-off, uncapped Warm-on, starved Warm-off) — **four configurations, one logits file**.
- **Conclusion / next gate**: Step 4 gate met — the emergency valve, eviction under lease pressure, and the async-demotion path all executed (`378` drains) with no number changed and no leak; the cost of starvation is `2.4×` decode NVMe traffic, which is the sweep's target. The mid-eviction hazard is a **narrow race** (≈2% of a layer) that this exposure did not reach, not a structural impossibility.
- **Evidence**: `scripts/expert_starved_pool.sh`, `/tmp/aeon-starved-pool.9uqTbu/{ref,starved}.log`, `/tmp/aeon-starved-warm/{starved-warm,long}.{log,telemetry.jsonl}`

### M33: Demotion-queue A/B — capacity 6 removes every drop and cuts decode NVMe 40%
- **Run**: `2026-09-21`; branch `main`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, 43 layers
- **Class / comparison key**: `Analysis / demotion-ab`
- **Platform**: `baseline`, Device 0 only
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: essay prompt, context `32768`, `--greedy`, `24` generated tokens, `n=1`, `--warm-gib 40`; arm q2 (`--demotion-queue 2`, the default) vs arm q6 (`--demotion-queue 6`)
- **Metrics** (whole run, then decode phase):

  | Field | q2 | q6 | Δ |
  | :--- | ---: | ---: | ---: |
  | `logical_bytes_from_warm` (all) | `32.74 GB` | `43.05 GB` | `+31%` |
  | `bytes_from_nvme` (all) | `48.60 GB` | `38.29 GB` | `−21%` |
  | `demotion_drops` (all) | `2312` | **`0`** | `−100%` |
  | decode Warm | `17.65 GB` | `25.54 GB` | `+45%` |
  | decode NVMe | `19.75 GB` | `11.86 GB` | **`−40%`** |
  | decode drops | `979` | `0` | `−100%` |
  | decode D2H | `23.54 GB` | `37.36 GB` | `+59%` |
  | `demotion_queue_depth_max` | `2` | `6` | — |

  **Warm service (the same run, request counts):** decode Warm `1247 → 1804` requests (`+45%`), Cold `1395 → 838` (`−40%`); prefill Warm `1066 → 1237`. Warm hit rate decode `21.0% → 30.4%`. Total request counts are identical across arms (`5934`/row), so this is a like-for-like tier shift.

  🔶 **Correction (2026-09-21, after a realistic run).** The cost/benefit line below is a *bandwidth* estimate and was **not confirmed in wall-clock**. A 1024-token non-greedy run (`3.40 → 3.22 tok/s`, TTFT `16.25 → 17.32 s`) is a wash within the run-to-run spread. The reason is the **max-of-six**: a layer waits on its slowest fetch, so what matters is `P(any cold) = 1 − (1 − p)⁶`, which moved only `80% → 60%` — a predicted ≈`5%` step-time gain, below noise. **Decode is supply-latency-bound, not bandwidth-bound**, so a byte reduction on the non-critical path does not show up as throughput. Treat the `−40%` NVMe as a resource win, not a latency win.
- **Correctness / service**: logits **byte-identical** across arms (`cmp`); both arms exit `0`
- **Conclusion / next gate**: Step 5 gate met — the larger queue converts every dropped demotion into Warm service (`drops 979 → 0`; Warm hits `+45%`), cutting decode NVMe `40%`. **Throughput effect not established** (`n=1`, predicted ≈`5%` ≈ noise); the queue is a resource/cleanliness win. Capacity `6` sufficed for **both** phases (`queue_depth_max = 6`), so `12` adds nothing here. **The per-layer outcome distribution measured on a `512`-token run (queue `6`): decode `all_hot 8.4%`, `warm_no_cold 46.3%`, `has_cold 45.3%` (`21973` dispatches `= 511 × 43`)** — nearly half of decode layers touch Cold and only `8.4%` are all-Hot, confirming the max-of-six explanation for the flat throughput (§6.10 thesis 1)
- **Evidence**: `scripts/expert_demotion_queue_ab.sh`, `/tmp/aeon-demotion-ab.MAvbnY/{q2,q6}.{log,telemetry.jsonl,logits.bin}`, `/tmp/aeon-q6-essay.log`, `/tmp/aeon-layers.log`
