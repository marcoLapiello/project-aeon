# Project Aeon - Performance & Accuracy Ledger

Authoritative silicon record for the AMD Radeon RX 7900 XTX (`gfx1100`).

* **Last normalized**: 2026-09-22.
* **Scope**: end-to-end runs through `aeon_chat`, and — as edge cases — directly measured performance values (a benchmark, a machine reference rate) and cache/supply analyses derived from an end-to-end run.
* **Not recorded here**: per-test correctness results and isolated synthetic-kernel microbenchmarks. Those live in the tests themselves.
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
- **Class / comparison key**: `E2E | Benchmark | I/O | Analysis` / `key`
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
| `Benchmark` | throughput, effective bandwidth, bytes moved, `n` |
| `I/O` | throughput, request/extent conditions, completion behavior |
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
| `hardware-baseline` | Machine reference rates (NVMe, GEMM, PCIe/compute overlap) | M1 |
| `e2e-43L-text` | Text in/text out at 43 layers | M28 |
| `supply-telemetry` | Per-phase, per-tier supply request/byte counters | M29 |
| `budget-cap` | Hot VRAM expert-pool cap feasibility and behavior | M30 |
| `tier-invariance` | Two runs, different tiers, byte-identical logits | M31 |
| `starved-pool` | Hot-pool cap forcing the emergency drain; logits still exact | M32 |
| `demotion-ab` | Demotion-queue capacity 2 vs 6 | M33 |
| `staging-depth` | Staging arena contention; depth lever viability | M34 |
| `routing-reuse` | Decode reuse-distance (ideal-LRU) vs measured Hot hit rate | M35 |
| `routing-opt` | Belady-OPT vs ideal-LRU: policy headroom | M36 |
| `prefill-ab` | Serial vs swept prefill at two prompt lengths, tok/s and bytes | M42 |
| `prefill-config` | Window/chunk as user settings; workspace derived and allocated at load | M43 |
| `registry-audit-cost` | Per-request `validate_invariants()`: dispatch cost and its removal | M43 |

## 4. Milestone cards

### M1: Hardware baseline — NVMe, GEMM, and PCIe/compute overlap
- **Run**: `2026-09-07`; Device 0
- **Class / comparison key**: `I/O / hardware-baseline`
- **Platform**: `baseline`, Device 0
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: aligned direct-I/O read, a large WMMA GEMM, and a PCIe/compute overlap probe; `n=1` per fixture
- **Metrics**: NVMe sequential `io_uring` `O_DIRECT` read `6.33 GB/s`; GEMM `25.6 TFLOP/s` (`2048 x 2048`, `870 us`); overlap `24.9 GB/s` with `0.0%` compute jitter
- **Correctness / service**: aligned direct-I/O payload and FP16 tile checks passed
- **Conclusion / next gate**: the machine's reference rates; `6.33 GB/s` is the single-drive rate used by every prefill cost model in the plan
- **Evidence**: Phase 0 hardware probes (retired)

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
- **Correctness / service**: logits **byte-identical** across arms (`cmp`); both arms exit `0`
- **Conclusion / next gate**: Step 5 gate met — the larger queue converts every dropped demotion into Warm service (`drops 979 → 0`; Warm hits `+45%`), cutting decode NVMe `40%`. **Throughput effect not established** (`n=1`, predicted ≈`5%` ≈ noise); the queue is a resource/cleanliness win, not a latency win. Capacity `6` sufficed for **both** phases (`queue_depth_max = 6`), so `12` adds nothing here. Per-layer outcome distribution on a `512`-token run (queue `6`): decode `all_hot 8.4%`, `warm_no_cold 46.3%`, `has_cold 45.3%` (`21973` dispatches `= 511 × 43`) — nearly half of decode layers touch Cold and only `8.4%` are all-Hot, which is why a byte reduction on the non-critical path does not become a speed win.
- **Evidence**: `scripts/expert_demotion_queue_ab.sh`, `/tmp/aeon-demotion-ab.MAvbnY/{q2,q6}.{log,telemetry.jsonl,logits.bin}`, `/tmp/aeon-q6-essay.log`, `/tmp/aeon-layers.log`

### M34: Staging depth — not a bottleneck in the single-token path
- **Run**: `2026-09-21`; branch `main`; analysis of the M33 A/B telemetry (no new silicon run)
- **Class / comparison key**: `Analysis / staging-depth`
- **Platform**: `baseline`, Device 0 only
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: the M33 `q2`/`q6` 24-token runs at context `32768`, `--warm-gib 40`; decode phase
- **Metrics**: staging-using decode transfers ≈`2642`; `staging_reuse_wait_ns` sum `69.37 s` (q2) / `75.66 s` (q6) ⇒ mean slot idle **`26.3 / 28.6 ms`**. Slot addressing is fixed: `staging_offset = (layer % 2) * 6` (`v4_expert_supply.hpp`), two banks by layer parity, no free-list
- **Correctness / service**: no run has ever thrown a staging state-transition error; `staging_in_use == 0` at every measured end (M32)
- **Conclusion / next gate**: **Premise refuted.** Staging is not contended — its slots idle ≈ one layer period and never block; `staging_reuse_wait_ns` is a misnamed *idle* counter, not a wait. Raising `TOTAL_STAGING_SLOTS` cannot help the single-dispatch path. Depth becomes a lever only when dispatches overlap (chunked prefill), where Step 0 D4's `banks × depth` sizing applies
- **Evidence**: `/tmp/aeon-demotion-ab.MAvbnY/{q2,q6}.telemetry.jsonl`, `src/architecture/deepseek_v4/core/v4_expert_supply.hpp` (`staging_offset`), `src/infrastructure/core/prefetch_staging.hpp` (`take_reuse_delay_ns`)

### M35: Routing reuse distance — the recency policy is already at its ceiling
- **Run**: `2026-09-21`; branch `main`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, 43 layers; `aeon_chat --profile-routing`
- **Class / comparison key**: `Analysis / routing-reuse`
- **Platform**: `baseline`, Device 0 only
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: essay prompt, context `32768`, non-greedy, `512` generated tokens, `--warm-gib 40`, queue `6`; layers 0–2 excluded (hash router)
- **Metrics**: `122640` learned-layer decode requests, `6531` compulsory (`5.3%`). Ideal-LRU hit rate by capacity: `258 → 31.3%`, `779 → 60.5%`, `1558 → 74.0%`, `3022 → 85.7%`, `11008 → 94.7%` (ceiling `= 1 − compulsory`). **Measured Hot hit rate `60.6%` vs ideal-LRU at capacity `779` `60.5%`**
- **Correctness / service**: unit test `test_routing_reuse` (hand-computed stack distances, hash-layer exclusion, measured-hit tracking) passes; run exits `0`
- **Conclusion / next gate**: The current global-LRU policy already achieves the ideal-LRU hit rate at its capacity — **no implementation headroom, so LRU is not thrashing**. The consequent lever is **capacity/coverage** (curve is steep `258→3022`). Whether a *different policy* (frequency/OPT) beats ideal-LRU is M36
- **Evidence**: `--profile-routing`, `tests/test_routing_reuse.cpp`, `/tmp/aeon-reuse.log`

### M36: Belady-OPT — a non-recency policy has ~17 points of headroom at the Hot capacity
- **Run**: `2026-09-21`; branch `main`; same workload as M35 (`512` decode tokens, `122640` learned-layer requests, queue `6`, `--warm-gib 40`)
- **Class / comparison key**: `Analysis / routing-opt`
- **Platform**: `baseline`, Device 0 only
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: `aeon_chat --profile-routing`, layers 0–2 excluded; Belady-OPT simulated offline over the same stream
- **Metrics**: hit rate by capacity (`ideal-LRU` → `OPT`): `258: 31.3 → 59.8%`, **`779: 60.5 → 77.4%`**, `1558: 74.0 → 85.9%`, `3022: 85.7 → 92.1%`, `11008: 94.7 → 94.7%`
- **Correctness / service**: `test_routing_reuse` also covers OPT — hand-computed `6/14` vs LRU `0/14` on an LRU-pessimal cycling trace, plus the structural `OPT ≥ ideal-LRU` invariant
- **Conclusion / next gate**: **Recency is at its ceiling, but a non-recency policy is not.** OPT beats ideal-LRU by `+16.9` points at the Hot capacity (and `+28.5` at `258`). This is an **oracle upper bound**, so it is the maximum a policy can win at `779`, not the expected win. The practically capturable fraction needs the Phase 2 static ranking in the [Routing Profile and Placement Study](../execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md)
- **Evidence**: `--profile-routing` (`opt_hit` column), `tests/test_routing_reuse.cpp`, `/tmp/aeon-opt.log`

### M42: The prefill A/B, and the double-buffered load
- **Run**: `2026-09-22`; branch `main`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, 43 layers, 256 experts each; `bench_prefill_ab` + `scripts/prefill_ab.sh`
- **Class / comparison key**: `Benchmark / prefill-ab`
- **Platform**: `baseline`, Device 0 only, single NVMe (`6.33 GB/s` measured ceiling used throughout)
- **Workload / configuration**: one arm per process (`serial` = `forward_token` per prompt token; `swept` = one `forward_window`, `prefill_sweep = true`), pristine host each, context `2048`, body chunk derived (`min(68, arena 42, pool 134) = 42`), `809` Hot slots, no Warm
- **Metrics**:

  | N | arm | seconds | tok/s | NVMe | GiB/1k tok |
  | ---: | :--- | ---: | ---: | ---: | ---: |
  | 256 | serial | 103.3 | 2.48 | 394.8 GiB | 1542 |
  | 256 | swept, before overlap | 68.1 | 3.76 | 145.1 GiB | 567 |
  | 256 | **swept, double-buffered** | **44.8** | **5.71** | 145.1 GiB | 567 |
  | 512 | serial | 211.2 | 2.42 | 786.5 GiB | 1536 |
  | 512 | swept, before overlap | 97.1 | 5.27 | 145.1 GiB | 290 |
  | 512 | **swept, double-buffered** | **73.6** | **6.96** | 145.1 GiB | 290 |

  The sweep's byte count is **constant** (`145.1 GiB` = one model read) while serial's grows at `1.54 GiB/token`. Speed-up vs serial: `2.30x` at N=256, `2.88x` at N=512, rising with N. The overlap (blocked-inside-`materialize` `34.6 s → 6.2 s` at N=512; `io_s` submission `11.6 s`; `lookahead = 1`) is worth `+52%` (N=256) and `+32%` (N=512).
- **Correctness / service**: byte-exactness asserted in the same run (swept logits `0 differing of 258560` vs serial `forward_token`); registry invariants hold and no lease leaks in any run
- **Conclusion / next gate**: Three findings: (1) the batched prefill is faster than serial, by more the longer the prompt; (2) one layer of double-buffer is enough because a layer set is `~0.55 s` of drive against `~1.4 s` of compute; (3) with the load hidden, `~56 s` of `73.6 s` at N=512 is body compute (`144 ms/token` vs colibri's `31 ms/token`). The last figure's attribution is superseded by M43, which found a per-request registry audit inside it.
- **Evidence**: `tests/bench_prefill_ab.cpp`, `scripts/prefill_ab.sh`, `src/architecture/deepseek_v4/core/v4_prefill_sweep.hpp` (`dispatch_ahead`, `materialize_layer`)

### M43: The prefill configuration, and the registry audit that was eating the prefill
- **Run**: `2026-09-22`; branch `main`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, 43 layers, 256 experts each; `aeon_chat` end-to-end
- **Class / comparison key**: `E2E / prefill-config`, `E2E / registry-audit-cost`
- **Platform**: `baseline`, Device 0 only
- [ ] **Invalidate for comparison** | **Reason**: `--`
- **Workload / configuration**: three end-to-end `aeon_chat` runs at context `2048`, greedy, `max-new-tokens 256`, Warm `35 GiB` unless noted. The same paragraph repeated N times as the prompt.

  | run | W | C | Warm | prompt tok | gen tok | stop | TTFT | decode |
  | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
  | 1 | 512 | 128 | 0 | 338 | 234 | eos | `53.3 s` | `2.27 tok/s` |
  | 2 before | 512 | 128 | 35 GiB | 338 | 234 | eos | `52.4 s` | `3.04 tok/s` |
  | **2 after** | 512 | 128 | 35 GiB | 338 | 234 | eos | **`39.4 s`** | **`3.58 tok/s`** |
  | 3 | 1024 | 256 | 35 GiB | 1004 | 153 | eos | `123.7 s` | `3.26 tok/s` |

- **Metrics**:
  1. **The registry audit was the prefill's hidden cost.** With the audit off, run 2's dispatch preparation fell **`io_ms` `10 162 -> 450 ms`**, TTFT **`52.4 -> 39.4 s` (`−25%`)**, decode **`3.04 -> 3.58 tok/s` (`+18%`)**. `submit_ms` stayed at `61` and `sqes` at `67 884`, so the I/O itself was never the cost — only the bookkeeping around it. The decode gain is the same bug on the decode path (258 reservations per token).
  2. **`C` is flat, confirmed a third time.** `C = 128 -> 256` gave no throughput (`117 -> 123 ms/prompt-token`) and cost `+87 MiB` scratch and `13` fewer Hot slots (`802 -> 789`).
  3. **The sweep's load is independent of both knobs.** `load_ms` `4 694` (W=512,C=128) vs `4 776` (W=1024,C=256); one pass, `43` layer loads, `11008` experts, `lookahead = 1` in both.
  4. **TTFT is linear in prompt tokens at `~120 ms/token`**: `117` (338 tok) vs `123` (1004 tok). Both runs are one pass, so there is no per-window overhead left to amortize.
- **Correctness / service**: all three runs `eos`, `registry.invariants_hold=true`, `outstanding_leases=0`, `staging_in_use=0`; the replies are coherent (each run correctly detects the repeated paragraph and summarizes it). The audit still runs **unconditionally at every boundary**; `--validate-registry` restores the per-operation audit for a debugging run.
- **Conclusion / next gate**: **Step 7's settings are exposed and the workspace is real.** The finding that matters is the audit: **the prefill was never as compute-bound as M42's subtraction suggested** — a per-request whole-registry scan was `10.2 s` of a `39 s` prefill and `258` scans per decode token. The remaining `~120 ms/prompt-token` is linear in the prompt and is the open prefill target.
- **Evidence**: `src/infrastructure/core/expert_registry.hpp` (`set_validate_each_request`, `checked_validate`), `core/memory_budget.hpp`, `core/v4_model_host.hpp` (`allocate_prefill_workspace`), `tools/aeon_chat.cpp` (`--prefill-window`, `--prefill-chunk`, `--validate-registry`, `[Prefill workspace]`, `[Prefill sweep]`)

### M44: The routed-cached bank, and the measured prefill-supply crossover
- **Run**: `2026-09-23`; branch `main`; DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon, 43 layers, 256 experts each; `bench_prefill_ab` + `scripts/prefill_ab.sh`
- **Class / comparison key**: `Benchmark / prefill-supply`, `Analysis / prefill-gate`
- **Platform**: `baseline`, Device 0 only, single NVMe; context `2048`, body chunk `C = 16`, no Warm
- **Workload / configuration**: one arm per process, pristine host each. `serial` = `forward_token` per prompt token; `swept` = `forward_window` with the gate forced to 1; `routed` = the same window with the gate forced above any prompt, so the route-aware cached bank supplies experts. Both batched arms are the same layer-major window; only the supply differs.
- **Metrics** (tok/s, NVMe GiB):

  | N | serial | swept | routed | winner |
  | ---: | ---: | ---: | ---: | :--- |
  | 32 | 2.52 / 57.4 | 1.02 / 141.6 | **3.14 / 38.6** | routed |
  | 64 | 2.64 / 105.3 | 2.18 / 141.6 | **4.32 / 50.8** | routed |
  | 128 | 2.63 / 206.1 | 4.15 / 141.6 | **5.20 / 65.9** | routed |
  | 160 | — | 5.03 / 141.6 | **5.61 / 70.8** | routed |
  | 192 | — | **6.34 / 141.6** | 5.90 / 73.8 | swept |
  | 256 | — | **7.38 / 141.6** | 6.17 / 79.5 | swept |
  | 512 | — | **8.04 / 141.6** | 7.09 / 91.8 | swept |

  The swept byte count is **constant** (`141.6 GiB`, one model read); the routed bank reads the per-layer union, which grows with `N` but stays below the sweep through `N = 512` (`0.111 GiB/token`) because disjoint unions saturate near `E`. The sweep nevertheless *wins on throughput* from `N ≈ 192`: its bulk sequential whole-layer reads beat the routed path's per-expert reservations and scattered reads even while reading more bytes.
- **Correctness / service**: `test_v4_prefill_sweep` (swept, 16 checks) and `test_v4_routed_prefill` (routed, 18 checks, at the derived `813`-slot pool and the floor `256`-slot pool) are byte-identical to serial; the bounded drain `512 + 301 = 813` and `256 + 0 = 256`; the switch-point Hot set is restored in both; Warm is unchanged; `invariants_hold()`, no lease leak, staging drained. `test_v4_engine` (38 checks) and the CLI run a short prompt end-to-end on the routed path.
- **Conclusion / next gate**: **The supply strategy is a prompt-length switch, and the crossover is `≈0.7 E` (`≈176` tokens).** Below it the routed bank is the better strategy — at `N = 32` it is `3×` the sweep and the sweep is `2.5×` worse than serial — and above `N ≈ 192` the sweep wins. The plan's provisional `E/4` gate was refuted by this measurement and corrected to `3 E / 4`, the round fraction above the crossover, on the safe side because the sweep is a throughput optimisation the routed path never needs for correctness. **Budget and restore are shared by both strategies** (one `begin_prefill_stream`/`end_prefill_stream` pair, differing only by `PrefillAlloc`): the drain is bounded to `2E` or `E` worst-LRU residents, the rest are preserved, and the drained set is reloaded through the normal cold path at `prefill_end`.
- **Evidence**: `tests/bench_prefill_ab.cpp`, `scripts/prefill_ab.sh`, `tests/test_v4_routed_prefill.cpp`, `tests/test_v4_prefill_sweep.cpp`, `src/infrastructure/core/expert_registry.hpp` (`PrefillAlloc`, `restore_set`, `resident_at_prefill_begin`), `src/architecture/deepseek_v4/core/v4_prefill_sweep.hpp`, `core/v4_model_host.hpp` (`restore_prefill_residents`, `prefill_sweep_min_tokens`), `core/memory_budget.hpp` (`prefill_sweep_min_tokens`)
