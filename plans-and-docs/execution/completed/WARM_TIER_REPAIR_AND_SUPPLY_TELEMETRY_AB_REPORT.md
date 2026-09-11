# Warm-Tier Repair and Supply Telemetry A/B Report

**Date:** 2026-09-11
**Plan:** `WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md`
**Schema:** Supply telemetry JSONL v1
**Status:** Stage 4 silicon evidence complete; second-pass structural corrections recorded

## Run Card

- **Model:** `DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon`
- **Native artifacts:** `model_dense.aeon` (15,744,996,344 bytes), `model_experts.aeon` (155,826,782,208 bytes), `model_experts.index` (176,160 bytes), `tokenizer.aeon` (4,596,329 bytes)
- **Config SHA-256:** `3911161a028fa2818b22ffff82dc2dad212aafeff32a66698ef7d154dea9d6a1`
- **Generation config SHA-256:** `5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0`
- **Source base:** `5252b97ed1b74b6e10fb16692d525e75aa45e8eb`
- **Uncommitted source fingerprint:** `7583fd4c10d344604d4ed686efaccc14da768792279888cfb07fb564fbf9c415`
- **Device:** AMD Radeon RX 7900 XTX, `gfx1100`
- **Host:** AMD Ryzen Threadripper PRO 3975WX, 64 GB RAM
- **Runtime:** ROCm 7.2.2, Linux 7.0.0-31-generic, Wave32
- **Workload:** prompt `What is 2 + 2? Answer briefly.`
- **Prompt SHA-256:** `175cdc5ea254141821eb79e777565558f2f152554a707b51761496c9541d9926`
- **Pipeline:** 43 layers, context 256, maximum 8 new tokens, greedy generation, seed not applicable
- **Measured interval:** one initialized request per run; startup Warm preload is the fixed initialization procedure and there was no additional inference warm-up request
- **Capacity:** 676 Hot slots, 2,642 Warm slots at 35 GiB, 12 transient staging slots, demotion queue capacity 2
- **Runs:** five independent runs per variant; all completed without infrastructure failure

## Variants

| Variant | Warm allocation | Startup preload | Refill |
| --- | ---: | --- | --- |
| `warm0` | 0 GiB | disabled | disabled by capacity |
| `demotion-free` | 35 GiB | enabled | disabled with `--no-warm-refill` |
| `repaired` | 35 GiB | enabled | enabled by default |

## Exact Output Identity

Every run in all three variants produced generated IDs `[22, 1]` and stop reason `eos`. No generated-ID or stop-condition divergence was observed.

## Decode Results

Values are median with inclusive Q1 and Q3 in brackets across five runs. Byte values are per generated Decode token because each run emitted one Decode token.

| Metric | `warm0` | `demotion-free` | `repaired` |
| --- | ---: | ---: | ---: |
| TTFT (ms) | 5754.09 [5748.11, 5755.35] | 5156.91 [5144.04, 5191.31] | 4907.82 [4890.84, 4926.32] |
| Decode throughput (tok/s) | 2.52 [2.51, 2.52] | 2.62 [2.62, 2.62] | 3.11 [3.10, 3.13] |
| Cold NVMe bytes/token | 2,052,587,520 [same, same] | 1,854,406,656 [same, same] | 1,429,733,376 [same, same] |
| Warm requests/token | 0 | 14 | 44 |
| Cold requests/token | 145 | 131 | 101 |
| D2H refill bytes/token | 0 | 0 | 778,567,680 |
| Demotion submissions/completions | 0 / 0 | 0 / 0 | 55 / 55 |
| Demotion drops | 0 | 145 | 90 |
| Valid Warm slots, Decode min/max | 0 / 0 | 2,211 / 2,225 | 2,626 / 2,642 |
| Pending logical operations max | 6 | 6 | 6 |
| Demotion queue depth max | 0 | 0 | 2 |
| Optional demotion wait (ns) | 0 | 0 | 0 |
| Peak RSS (bytes) | 13,273,690,112 | 50,674,688,000 | 50,691,203,072 |
| VmSwap delta (bytes) | 0 | 0 | 0 |

The repaired path reduces median Decode Cold NVMe bytes by 22.9% versus the demotion-free Warm control and preserves output identity. Median TTFT and Decode latency are both below the demotion-free control. Warm service is nonzero in every repaired run, and valid Warm occupancy remains stable after the preloaded startup state is consumed.

## Structural Evidence

- `test_expert_registry_warm_state` passes the required 10,000-transition trace with seed `0xAE0F`, exact hand-sequence counters, zero-valid-slot lazy startup, unique logical pending-operation accounting, D2H failure rollback, and request failure rollback.
- The same registry test now exercises a real `PrefetchStagingArena` host-to-device event dependency, event completion, and failure release, in addition to the deterministic registry trace.
- `test_supply_telemetry` passes the versioned JSONL contract, including phase/source summaries, transfer events, occupancy, RSS, and VmSwap fields.
- The full configured CTest suite passes 19/19 after the final policy correction.
- Production `aeon_chat --supply-telemetry <path> --run-id <id>` emits phase/source JSONL. D2H and H2D transfer events carry physical source/destination slots; operation phase and request source tier are saved until completion.
- Optional D2H admission uses stream/event dependencies and nonblocking event queries. The request path contains no CPU synchronization on an optional demotion event; `optional_demotion_wait_ns` is zero in every measured run.
- Warm capacity can be allocated with `--no-warm-preload`; this is content-lazy rather than allocation-lazy: the configured host capacity is allocated, startup Warm payload reads are skipped, and normal refill later publishes Warm entries and produces Warm hits.

## Second-Pass Corrections

- The request path no longer synchronizes `pending->h2d_event` when a request
	encounters a demotion-pending catalog entry. It uses a nonblocking event query
	and never introduces a CPU wait for optional D2H admission.
- The registry no longer applies a layer-order restriction to otherwise safe Hot
	victims. Queue-pressure and unavailable-Warm-destination rejections are
	retained as explicit drop reasons and are emitted in transfer telemetry.
- Registered fallback host segments are included in the pinned-slot count, so
	`warm_pinned_bytes` and `warm_unpinned_bytes` match the transfer mechanism.
- `demotion_attempts` now counts every candidate considered when Warm admission
	is enabled. The five-run artifact above was captured before this semantic
	correction: its `55 / 55` value is demotion submissions/completions, while
	the `90` drops are rejected candidates. New JSONL runs use the corrected
	candidate-attempt definition.

## Decision

Stage 4 passes for the fixed workload and Stage 5 documentation evidence is complete. Persistent Warm refill remains the default. The 35 GiB profile is operational on this host, but its approximately 47 GiB peak process RSS includes mapped model pages and should remain subject to host-pressure monitoring. The historical A/B counter label is preserved explicitly above; subsequent telemetry uses the corrected candidate-attempt contract. Broader cold-layout optimization and model-correctness work remain separate open tracks; this report does not claim DeepSeek compressed/indexed attention parity.
