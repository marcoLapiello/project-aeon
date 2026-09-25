# Supply-Chain Hot-Path — Execution Plan

*Created 2026-09-24; restated as a spec 2026-09-25. Owner: [SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md](../../analysis/current/SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md). **This plan is the action list; the analysis doc owns the evidence.** Scope: how fast the bytes arrive, not which bytes.*

---

## 0. The pipeline model (the spec this plan implements)

### 0.1 Units

- **layer-block** = one layer's expert set = `E` payloads (`E = experts_per_layer`).
- **vram_blocks** — decided **at load** from the budget (`hot_slots = total_vram − dense − KV − headroom − scratch`): `hot_slots ≥ 2E → 2`, else `1`.
- **staging_blocks** — the pinned staging arena in layer-blocks. **Must be 2 in both cases**: one block is the read destination, one is the copy source.
- **One block is always the compute block** (`L`). It is not pipeline capacity.

### 0.2 Target steady state

| Scenario | Blocks | Steady state during `body(L)` |
| :--- | :--- | :--- |
| **hot ≥ 2E** | 2 vram + 2 staging | `L` computing │ `L+1` in VRAM │ `L+2` copying in │ `L+3` reading |
| **hot < 2E** | 1 vram + 2 staging | `L` computing │ `L+1` copying in │ `L+2` reading |

### 0.3 Requirements (normative — every step is judged against these)

| # | Requirement |
| :-- | :--- |
| **R1** | The **only** physical ordering: a VRAM block must not be overwritten while the GPU reads it as weights. Enforced by a **stream event**, never by a host synchronize. |
| **R2** | The **default** staging arena is **2 layer-blocks** (`2E`) in **both** scenarios — it must not be derived from `vram_blocks` (`1` vram block or `2`). R6 then requires the design to still *work* below that default. |
| **R3** | Every stage advance is triggered by a **completion event**, not by the layer boundary: **slot-free → issue one read into it**; **read-event → enqueue that expert's copy**; **copy-event → release that expert's staging slot**; **compute-event (a stream marker at the end of `body(L)`) → release layer `L`'s VRAM block**. The copy *fills* VRAM; it never frees it. Today the **first** link does not exist: reads are a single wave at the layer boundary. |
| **R4** | No host synchronization on the supply path. The per-layer sync in `v4_graph` may remain only for what genuinely needs it (the router readback), never to order supply. |
| **R5** | Lookahead is **derived from the free blocks at runtime**; no hardcoded depth. |
| **R6** | Depth is a **memory budget, not a schedule**: the slot count must change how much RAM the corridor spends, **never** how fast it runs. Today it changes speed, because the count selects the algorithm — see the table below. (Gate A.) |

**Why the depth changes speed today (measured).** At `1E` the arena is **one room with two doors**: `read(L) → copy(L) → drain all → read(L+1)`, strictly serial across layers. At `2E` it is **two rooms** (`staging_base = (layer % 2) * E`): bank `L%2` copies `L` while bank `(L+1)%2` reads `L+1`. The `+13%` at `512` is that cross-layer overlap and nothing else — **`2E` is the only mechanism the current code has for overlapping read-from-SSD with copy-into-VRAM.** That is the defect: the arena size is selecting the algorithm.

**Depth is a gradient, and only the top rung is proven.** `< E` is a **hypothesis, not a demonstrated outcome**:

| Arena | Overlap it buys | Status |
| :--- | :--- | :--- |
| **`2E`** | copy `L` ∥ read `L+1` | **measured** (`+13%`) |
| **`E`** | plausible *with* P2.3 — if each copy runs as its read lands, the slots are already free at the boundary | **unproven** |
| **`< E`** | additionally needs **paced reads** (one per slot freed, R3's first link) | **unproven, and blocked by the per-layer park** |

**Why the floor is `E` today, and it is not the prompt gate.** `V4Graph::forward_window` refuses a chunk needing more than the arena holds: `staging_needed = min(6C, E)`. Separately, a swept dispatch binds **every non-resident expert of the layer**, so the sweep's real demand is `E` — the `6C` guard *under*-estimates it and only passes because `C` is large enough that `6C > E`. The prompt-length gate (`prefill_sweep_min_tokens = 3E/4 = 192`, configurable) is **not** the cause: the gate test runs `N = 256 > 192`, so the sweep engages and the chunk/staging guard then bites.

**Scenario 2 (`hot < 2E`) has a hard limit regardless of staging size.** With one VRAM block, `L`'s copies cannot start until `body(L-1)` ends, so its staging slots do not free until mid-body and `L+1`'s reads cannot be issued at the boundary. `2E` staging still buys read(`L+2`) ∥ copy(`L+1`), but the copy itself stays partly exposed. Do not claim scenario 1's steady state here.
| **R7** | The work is resource-invariant: identical token/layers/experts at every depth. (Gate B.) |
| **R8** | No hardware constant (bandwidth, latency, ratio) may select behaviour. Pinned memory is reported, never assumed. |

### 0.4 Already right — do not rebuild

- Per-expert read→copy pipelining inside a layer (`materialize` enqueues copy *i* when read *i* lands).
- The VRAM frontier already adapts to the pool (measured: holds as many whole layers as fit; frontier `44`).
- Consumer-side ordering (`accumulate_routed` waits per-expert on the transfer's event) — why Phase 1 works.

### 0.5 The defect this phase fixes

Today's default arena is **`1 × E`**, a single room: `read(L) → copy(L) → drain all → read(L+1)`, serial across layers. The `2 × E` opt-in gives two rooms and the cross-layer overlap, worth a measured `+13%` — so the arena **size** is selecting the **algorithm** (R6). The default must become `2E`, in **both** scenarios and independent of `vram_blocks` (R2); and in scenario 2 (`hot < 2E`) the win is inherently smaller, because one VRAM block cannot host `L` and `L+1` at once (R6).

---

## 1. Instrumentation

Landed — use, do not rebuild:

- `TieredExpertSupply` counters (`io_wait_ns`, `h2d_enqueue_ns`, `h2d_drain_ns`, `h2d_drain_calls`, `dispatch_cpu_ns`, `direct_io_submit_ns`, `reset_transfer_counters()`), exposed on `V4ExpertSupplyCoordinator` / `V4ModelHost`.
- `tests/bench_supply_split.cpp` + `scripts/supply_split.sh` — the exposed-load split, one model load.
- `tests/test_v4_staging_depth.cpp` — the two portability gates (A, B).
- `V4ModelHost::resize_staging_slots(slots)` — set the arena depth at runtime.

**To add (P2.1):** a block-occupancy readout — per layer, staging slots in use and VRAM blocks reserved-but-empty — so the pipeline's fill is observable, not argued.

---

## 2. Phase 1 — swept H2D overlapped with its compute *(done)*

The swept layer's H2D no longer blocks the host before the body: `materialize_layer` settles reads and holds a bank, `after_layer` reclaims it, and ordering is the **consumer's**.

| Warm | N | `h2d_drain` | `wall_s` |
| ---: | ---: | ---: | ---: |
| 0 | 256 | `4.57 → 0.006 s` | `33.811 → 30.574` |
| 0 | 512 | `5.36 → 0.006 s` | `63.593 → 59.087` |
| 30 | 256 | `4.23 → 0.006 s` | `32.913 → 29.400` |
| 30 | 512 | `4.25 → 0.006 s` | `62.280 → 59.055` |

`io_wait` and decode unmoved. **Blocker:** needs `banks = 2` (`+3.44 GiB` pinned), which swaps the box at the production Warm shape → ships **opt-in** (`prefill_sweep_staging_banks`, default `1`) until Phase 2. Evidence: analysis §10.

---

## 3. Phase 2 — The corridor as a demand-driven pipeline *(mandatory)*

Implements the §0 spec. Each step is independently verifiable and independently committable.

### P2.1 — `staging_blocks = 2` in both scenarios, + block-occupancy readout — ✅ **done**

| | |
| :--- | :--- |
| **Status** | ✅ **shipped 2026-09-25.** `2E` is now unconditional (the config knob is gone) and the corridor's fill is observable. |
| **Fixes** | §0.5 — scenario 2 (`hot < 2E`) currently has **zero** overlap. |
| **Where** | `memory_budget.hpp` (`staging_slot_count`), `prefetch_staging.hpp` (`StateCounts`), `tiered_expert_supply.hpp` / `v4_expert_supply.hpp` (passthrough), `v4_prefill_sweep.hpp` (`BlockOccupancy`, per-layer sample), `v4_model_host.hpp` (`sweep_occupancy()`), `tests/test_v4_staging_depth.cpp` (Gate C) |
| **Change** | `staging_slot_count` returns `max(base, 2E)` whenever `prefill_sweep` is on; `prefill_sweep_staging_banks` is **removed** (a settable depth lets a resource select the algorithm — R6). The A/B and the gate now change depth through `resize_staging_slots`. New: `PrefetchStagingArena::StateCounts` (free / reading / copying) sampled once per layer into `V4PrefillSweep::occupancy_samples()`. |
| **Requirement** | R2 ✅ `staging_blocks` does not depend on `vram_blocks`. |
| **Requirement** | R8 ✅ `transient_staging_bytes` == `slots × payload_bytes`, still asserted by `bench_supply_split`. |
| **Verify** | ✅ **Gate C added and passing**: `0/43` layers overlapped at `1E` (drain `4.9 s`) vs **`42/43` at `2E`** (drain `0.006 s`) — the parking lot and the corridor, measured. Byte-exactness gates pass at the new default (`16`/`18`/`10`/`38`, 0 failures, token unchanged). |

### P2.2 — Completion-driven staging release — ✅ **done**

| | |
| :--- | :--- |
| **Status** | ✅ **shipped 2026-09-25.** The release is attached to the copy's own completion event. **Measured effect on throughput: none on its own** — see below. |
| **Fixes** | Holding each staging slot for a whole body when its copy took ~0.1 s. |
| **Where** | `prefetch_staging.hpp` (`release_if_copying`), `tiered_expert_supply.hpp` (the reaper's success path + `staging_released_on_completion()`), `v4_prefill_sweep.hpp` (reap **before** reclaim in `after_layer`/`materialize_layer`) |
| **Change** | The reaper releases each slot the moment **its own** `h2d_event` fires, via `release_if_copying` (idempotent, so decode's `on_routed_consumed` still releases safely). The sweep reaps **before** its block reclaim, so the completion path is primary and `release_streamed_staging` is the fallback. |
| **Requirement** | R3 ✅ Release is triggered by the copy event, not the layer boundary. |
| **Requirement** | In-flight accounting stays exact: no reuse while the read *or* the copy is outstanding (`SlotState`, unbypassed). |
| **Verify** | ✅ `in_use_slots() == 0` at `prefill_end`; byte-exactness gates pass (`16`/`18`/`10`/`38`); **`staging_released_on_completion` = `10198` at `2E`, `0` at `1E`** — the release really does come from the completion path. |
| **Honest result** | **No wall-time change** (`30.913` vs `30.641 s`, within noise; `drain_s` `0.000`) — **as expected; a throughput claim was never made for this step.** P2.2 is the **structural prerequisite** for P2.3 (copy on read-completion) and P2.4 (lookahead from free blocks): the arena is now a completion-drained free-list in which each expert's slot is individually tracked and releaseable. It does **not** by itself lower the `E` floor (that is the read wave). |

### P2.3 — Copy into VRAM as soon as a read lands — ✅ **done**

| | |
| :--- | :--- |
| **Status** | ✅ **shipped 2026-09-25.** Real win: `−2.7 s` at `1E`, and Gate A's spread cut from `1.133×` to `1.055×`. |
| **Fixes** | `L+1` sitting in staging with its VRAM block empty until the boundary. |
| **Where** | `direct_io_reader.hpp` (`try_completion`), `tiered_expert_supply.hpp` (`materialize_available`, `enqueue_expert_copy`, `copies_pumped()`), `v4_expert_supply.hpp` (`pump_layer_prefetch`), `v4_prefill_sweep.hpp` (`pump`), `v4_expert_executor.hpp` (`set_supply_pump`), `v4_model_host.hpp` (wiring) |
| **Change** | A **non-blocking** materialize: drain everything the CQ already holds, then enqueue each expert's copy as soon as **its own** reads have all landed; leave the rest. Driven by the executor's **per-token hook** — the only host activity inside a body — which is what makes "as its read lands" possible without breaking the per-layer park. |
| **Requirement** | R1 ✅ The copy is a stream-ordered `upload_from_host_expert` on `sdma_cold_stream_`, as before; no host sync added. |
| **Requirement** | R3 ✅ Copies advance on read completion, not the boundary. |
| **Requirement** | R4 ✅ `try_completion` is a pure CQ peek — no syscall, no wait. |
| **Verify** | ✅ Byte-exactness gates (`16`/`18`/`10`/`38`), token unchanged; `copies_pumped` = `8345` at `1E`, `8238` at `2E`. |
| **Result** | `1E`: `34.896 → 32.204 s` (`drain 4.708 → 1.613 s`); `384`: `35.027 → 32.350 s`; `2E`: `30.913 → 30.654 s` (already overlapped, so neutral). **Gate A spread `1.133× → 1.055×`.** |
| **Why `1E` gained** | With the pump, `L+1`'s copies run **during `body(L)`** as its reads land, and P2.2's completion release frees the slots immediately — so even one staging block now pipelines read→copy within a body. The boundary drain shrinks to the residue (`1.6 s` of `4.7 s`). |

### P2.4 — Derive the lookahead from free blocks

| | |
| :--- | :--- |
| **Fixes** | The hardcoded one-layer lookahead (`pending_valid_`/`resident_valid_`). |
| **Where** | `v4_prefill_sweep.hpp` — `dispatch_ahead` / `before_layer` |
| **Change** | Replace the single pending/resident pair with a queue sized by **how many blocks are free**, up to the VRAM frontier. Depth becomes dynamic. |
| **Requirement** | R5. The depth is computed at runtime from free blocks; no constant. |
| **Requirement** | R3. A layer enters the queue when a block frees, not on a fixed schedule. |
| **Verify** | Scenario 1 reaches the §0.2 steady state (`L+1` in VRAM while `L` computes); byte-exactness gates; no SQ-full throw (ring check). |

### P2.5 — The portability gates must pass *(acceptance)*

| | |
| :--- | :--- |
| **Where** | `tests/test_v4_staging_depth.cpp` |
| **Gate A** | ❌ fails today, but **much closer**: spread `1.055×` (was `1.144×` before P2.3). **Target:** flat across a wide depth range. **Reaching `< E` is a hypothesis** (R6's gradient), not a committed outcome — the blocker is paced reads. |
| **Gate B** | ✅ passes today. **Must keep passing.** |
| **Gate C** | ✅ **added with P2.1**, passes: the default `2E` shape shows `42/43` layer-bodies with a read and a copy in flight at once; `1E` shows `0/43`. |
| **Command** | `./build/bin/test_v4_staging_depth` (default `64 128 192 256 384 512`) |

---

## 4. Phase 3 — Dispatch bookkeeping *(low priority; independent; do last)*

Measured at **`≈1%`** of both prefill (`610 ms/window`) and decode (`3.4 ms/token`). Real, small, and **independent of Phases 1–2** — it may be done any time, or skipped. Every fix is local and mechanical. Note Phase 1 gave it a reason to exist after all: with the deferred drain more transfers are in flight at once, and `disp_ms` rose from `698` to `856 ms` (`Warm 30`, `N = 512`, analysis §10.3).

| # | Item | Where | Change | Requirement |
| :-- | :--- | :--- | :--- | :--- |
| P3.1 | `O(1)` pending-transfer count | `src/infrastructure/core/expert_registry.hpp` — `pending_transfer_count()` (~L939) | Maintain `uint32_t operations_in_flight_`; return `operations_in_flight_ - pending_demotion_count` | Increment on `NONE → {IO_PENDING, PROMOTION_PENDING, DEMOTION_PENDING}`; decrement on the return to `NONE` in `complete_request`, `fail_request`, `complete_demotion`, `drop_demotion`, `fail_demotion`, and both `warm_shadow` branches. **Exact equality with the old set-based result is the gate.** |
| P3.2 | Skip the no-demotion catalog scan | `src/infrastructure/core/tiered_expert_supply.hpp` — `schedule_demotion` (~L570) | Guard `if (expert_registry_->pending_demotion_count == 0) return;` before the loop | Correct because a dropped victim always increments that counter at reservation and decrements it at drop/fail. In the sweep the branch is unreachable anyway. |
| P3.3 | `O(1)` reap | `src/infrastructure/core/tiered_expert_supply.hpp` — `reap_registry_transfers` (~L700) | Swap-and-pop instead of erase-at-index + `continue` without incrementing | The registry is order-independent (`ensure`/`find` are by `operation_id`), so reordering is safe. Must not skip entries. |
| P3.4 | `operation_id → index` map | `src/infrastructure/core/tiered_expert_supply.hpp` — `find_registry_transfer` / `ensure_registry_transfer` | Maintain `std::unordered_map<uint64_t, size_t>` beside `registry_transfers_` | Must be updated on every insert **and** on every swap-and-pop. Only worth doing with P3.3 (they touch the same structure). |

**Gate:** no behaviour change — all existing supply/registry tests pass unchanged, and `disp_ms` in both tables falls (measure with `scripts/supply_split.sh`).

---

## 5. Guardrails — checked per step, not at the end

1. **Byte-exactness is the correctness gate.** Both supplies byte-identical to the serial reference: `test_v4_prefill_sweep`, `test_v4_routed_prefill`, `test_v4_prefill_window`, `test_v4_engine`. Do **not** run the whole suite (project rule).
2. **Registry invariants hold.** `invariants_hold() == true`, no lease leak at any boundary, `end_prefill_stream` does not throw.
3. **Staging fully reclaimed.** `in_use_slots() == 0` after `prefill_end()`.
4. **No host synchronization on the supply path** (R4) and no new per-expert HIP events.
5. **No hardware constant selects behaviour** (R8). If a step needs a number, it must be derived from free blocks at runtime.
6. **Pinned memory reported, never assumed.**
7. **Both Warm configurations.** Sweep gates run at `AEON_WARM_GIB=0` **and** `=30`; a win that only holds cold is not a win. (Not `35`: it swaps the box — analysis §10.4.)
8. **Both strategies unaffected.** Routed bank and decode must not regress; re-measure decode in the same runs.
9. **One commit per step**, measured before/after in the message. Conventional prefix (`perf:`, `refactor:`, `test:`).
10. **Ledger rule.** Performance ledger **only** for end-to-end measurement of real prompts; a hardware probe does not qualify.
11. **One heavy process at a time.** Never run a bench, a test, and a build concurrently.
12. **Portability gates re-run per step.** `./build/bin/test_v4_staging_depth`: **Gate B must pass** and **Gate A must not regress**. Faster at one depth while more depth-dependent is a regression.

---

## 6. Order and stop conditions

```
Phase 1  ✅ DONE — H2D overlapped (3.2–4.5 s/window); memory cost blocks shipping
   │
   ▼
Phase 2  MANDATORY — the corridor becomes a demand-driven pipeline
   │   P2.1  ✅ DONE  arena is 2E unconditionally + corridor-fill readout (Gate C)
   │   P2.2  ✅ DONE  release each staging slot on its copy event (R3); no throughput change alone
   │   P2.3  ✅ DONE  copy into VRAM as each read lands, pumped per token (R1/R3): −2.7 s at 1E
   │   P2.4  lookahead derived from free blocks                          (R5)  ← next
   │   P2.5  gates: A must pass over a wide range incl. < E; B/C must hold
   │   └─ ABORT if any step cannot hold byte-exactness → keep it opt-in, record negative
   ▼
Phase 3  optional, independent (dispatch bookkeeping)
```

At each gate: **update the analysis doc**, then commit. Never start the next step with a stale doc.

---

## 7. Excluded — refuted or out of scope *(do not re-open without new evidence)*

| Item | Why it is not in this plan |
| :--- | :--- |
| **C4 — NVMe directly into VRAM** | Measured `NOT_SUPPORTED` (analysis §9): `EFAULT` on `pread(O_DIRECT)` and on the production `io_uring` path. No userspace workaround. |
| **`banks = 2` as a *schedule*** | The count must not select behaviour (R6). It stays only as the arena's depth once P2.1 makes that depth a pure budget. |
| **Batched H2D sync (§4.4a)** | Refuted payoff: the syncs absorb copy time. Subsumed by P2.3. |
| **Duplicate per-expert HIP event (§4.4b)** | Bounded by `h2d_enqueue`. Solve only as a side effect of P2.3, never as its own task. |
| **Single H2D copy (C5)** | Submission is not the cost; bandwidth is already at the PCIe ceiling. |
| **`O(catalog)` scans as a *throughput* item** | Measured `≈1%` — hence Phase 3 (hygiene). |
| **Decode NVMe wait (`36–63%`/token)** | **Out of scope** — its remedy is *which bytes are resident*. Hand-off to the [routing profile and placement study](ROUTING_PROFILE_AND_PLACEMENT_STUDY.md). |
| **`validate_invariants` per request** | Already off by default; boundary audits kept (analysis §4.6). |
| **`LayerPrefetchState::sync_state`** | µs scale; a clarity smell, not a cost (analysis §4.5). |

---

## 8. Definition of done

- **Phase 1:** ✅ `h2d_drain → ≈0`, `wall_s` down `3.2–4.5 s` at Warm 0 and Warm 30, byte-exact. **Caveat:** needs `banks = 2` (`+3.44 GiB` pinned) → **opt-in** until Phase 2.
- **Phase 2:** the §0 spec holds — R1–R8 met, with:
  - **R2:** ✅ the arena is `2E` in both scenarios (scenario 2 has overlap); the knob is gone.
  - **R5:** lookahead derived from free blocks, no constant.
  - **Gate A** flat across a **wide** depth range; **`< E` is a hypothesis** (blocked by paced reads + the park), so treat `E` as the first real target and `< E` as the stretch goal. **Gate B and Gate C** must hold.
  - the Phase 1 win (`h2d_drain → ≈0`, `wall` down) survives; pinned cost fits the production Warm shape.
- **Phase 3 (if done):** `disp_ms` down in both tables, no behaviour change.
- **Throughout:** no regression in decode or the routed bank; measurements reproducible from `scripts/supply_split.sh` and `./build/bin/test_v4_staging_depth`; **one heavy process at a time**.
