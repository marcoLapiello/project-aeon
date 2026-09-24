# Supply-Chain Hot-Path — Execution Plan

*Created 2026-09-24. Owner: [SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md](../../analysis/current/SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md). **This plan is the action list; the analysis doc owns the evidence and the argument.** Do not repeat the analysis here — read it for the *why* of any step.*

## 0. What this plan is, and what is excluded

Every item below is either **measured as a real cost** or **provably wasteful** in the source analysis, and **not refuted**. Items the measurements refuted are excluded (§7) and must not be re-opened without new evidence.

Scope is unchanged from the analysis: *how fast the bytes arrive*. The *which bytes arrive* question (decode residency) is a hand-off (§7), not work here.

### The measured cost picture this plan acts on

| Measured site | Size | Status |
| :--- | :--- | :--- |
| Swept-layer H2D exposed in front of the body | `4.1–5.4 s/window` (`6.5–8.4%`) | **Target of Phase 1 — removed (§10): `h2d_drain → 0.006 s`, `wall −3.2…−4.5 s`** |
| `dispatch` per-request bookkeeping (scans) | `610 ms/window`, `3.4 ms/token` (`≈1%`) | Phase 3 (low priority) — now with a reason: `+0.15 s` when transfers are in flight simultaneously |
| `h2d_enqueue` (H2D submission) | `47–72 ms/window` | Excluded — too small |
| `submit` (`io_uring_enter`) | `23–31 ms/window` | Excluded — too small |
| Swept-layer read wait (`io_wait`) | `0.27–2.07 s/window` | Excluded — already hidden |

### Pre-existing instrumentation (already landed; use it, do not rebuild)

- `TieredExpertSupply`: `io_wait_ns`, `h2d_enqueue_ns`, `h2d_drain_ns`, `h2d_drain_calls`, `dispatch_cpu_ns`, `direct_io_submit_ns`, `reset_transfer_counters()`.
- Exposed on `V4ExpertSupplyCoordinator` and `V4ModelHost` (`supply_io_wait_ns()`, `supply_h2d_drain_ns()`, `supply_dispatch_cpu_ns()`, `reset_supply_transfer_counters()`, …).
- Bench: `tests/bench_supply_split.cpp` + `scripts/supply_split.sh` (one model load, gate-selected strategy, greedy decode, prints the split).
- Probe: `tools/aeon_c4_probe.cpp` (C4, closed).

---

## 1. Objective and the hypothesis under test

**Objective.** Remove the swept prefill's exposed H2D drain from the pre-body critical path: `h2d_drain → ≈0` for the swept arm, with a corresponding fall in `wall_s`.

**Hypothesis (coupled — this is the key point).** Two changes are required *together*; neither works alone. **(Both confirmed by measurement, with one correction to the second half.)**

1. **Deferred drain.** The sweep must stop *host-blocking* on the H2D before the body, so the copy overlaps it. **Correction from the measurement:** the ordering must be the **consumer's** (the body's MoE dispatch joins the pending transfer and `accumulate_routed` waits on its per-expert event), **not** a compute-stream wait issued by the driver — a whole-body `hipStreamWaitEvent` recovers zero, because it puts the copy back in front of attention. What the driver must do is *nothing*.
2. **Staging headroom.** Deferring the drain holds the layer's staging slots through the body, so `dispatch_ahead(L+1)` has nowhere to read. The arena must have enough slots that the next layer's reads proceed while this layer's upload drains.

> Why this matters for the record: the analysis (§8.4 item 4, §5.2 C1) refuted `banks = 2` *on its own* — it only speeds reads, which are already hidden. That refutation holds **only while the drain stays synchronous before the body**. **Phase 1 succeeded, so §5.2 C1 has been reworded (analysis §10.3, §8.6): the second bank is right when coupled with the deferred drain, and wrong alone.** The coupling is also what makes Phase 2 mandatory — the bank is unaffordable at the production Warm shape.

**Expected ceiling, stated up front.** `≈4–5 s` of a `≈62–64 s` swept window, i.e. **`≈7–8%` of the swept prefill**. The swept prefill is one of several phases; this is a bounded win, not a step change. Contention is expected to be negligible (the copy writes `3.44 GiB/layer` at `26 GiB/s` ≈ `2.7%` of the card's VRAM bandwidth), which is why the overlap is worth trying — but it is unproven until measured.

---

## 2. Phase 1 — Overlap the swept layer's H2D with its compute

> **Status: complete (2026-09-24).** All four steps landed. **Gate outcome: PASS at Warm 0 and Warm 30; the mechanism is proven, but the memory cost blocks shipping it as-is** — see P1.4. Evidence and argument: [analysis §10](../../analysis/current/SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md).
>
> **One correction to this plan's own design, from the measurement.** The plan had P1.1 order the compute stream behind the copies (`hipStreamWaitEvent` per slot). Measured, that recovers **nothing** — it moves the copy from in front of the compute to in front of attention. The working form is to order **nothing** on the driver side and let the body's own MoE dispatch join each pending transfer and wait on its per-expert event (`accumulate_routed` already does this for decode). Where it says "the compute-stream wait (P1.1) is what guarantees it" below, read: *the consumer's per-expert wait*.

**Order matters: do the steps in sequence; each has its own verification and its own commit.**

### P1.1 — Add a non-blocking staging drain path (supply layer)

| | |
| :--- | :--- |
| **Status** | ✅ **done.** `release_streamed_staging` is left intact; the sweep now simply **defers** it (`finish_streamed_batch` at `after_layer`) and orders nothing on the compute stream. The consumer-side ordering already existed in `V4TieredExpertExecutor::accumulate_routed`. |
| **Where** | `src/infrastructure/core/tiered_expert_supply.hpp`, `v4_expert_supply.hpp`, `v4_prefill_sweep.hpp` |
| **What** | Add a **stream-ordered** variant that, instead of `hipEventSynchronize` per slot: (a) makes the given consumer stream wait on each in-flight slot's gate event (`hipStreamWaitEvent`), and (b) leaves slot reclamation to the caller. Keep `release_streamed_staging` intact for the existing path. |
| **Requirement** | No host synchronization on the sweep path. This is a hard project rule for the request path; the sweep is not the request path, but the same discipline applies — the whole point is to stop the host blocking. |
| **Requirement** | The gate event must be the one already recorded on `sdma_cold_stream_` (`prefetch_staging_->events[slot]`). Do **not** add a second event (that is §4.4b, excluded). |
| **Verify** | ✅ `h2d_drain_calls` stops incrementing on the sweep path; the drain counter is `≈0` in every swept row (§10.2). |

### P1.2 — Size the arena for two layers *(temporary, deliberate over-allocation)*

| | |
| :--- | :--- |
| **Status** | ✅ **done**, and **demoted to an opt-in.** `runtime.prefill_sweep_staging_banks` ships with default `1` (the engine's original memory shape); `2` is the A/B switch. The default was `2` for one run and it swapped the box at the production Warm shape — see P1.4. |
| **Where** | `src/architecture/deepseek_v4/core/memory_budget.hpp` (`staging_slot_count`, shared by the budget and the arena); `v4_model_host.hpp` (`sweep_staging_banks_`) |
| **What** | When `prefill_sweep` is on, size the arena to `2 × experts_per_layer` instead of `experts_per_layer`. Add a runtime-config override so it can be set to `1` (today) or `2` (experiment) without a rebuild. |
| **Requirement** | ✅ The budget report and the arena call one `staging_slot_count` helper, and `bench_supply_split` asserts they agree. |
| **Requirement** | `+3.44 GiB` pinned (`6.75 GiB` total) when set to `2`. ✅ Confirmed by the header print: `banks=2 staging_slots=512 staging=6.75 GiB`. |
| **Verify** | ✅ Arena constructs, budget prints the doubled staging, a swept window runs. |

### P1.3 — Defer the drain past the body boundary (sweep layer)

| | |
| :--- | :--- |
| **Status** | ✅ **done.** `materialize_layer` now settles reads and holds a `resident_state_` bank; `after_layer` reclaims it and reaps the registry (which promotes the completed uploads out of `PROMOTION_PENDING` before `release_layer` refuses a live transfer). |
| **Where** | `src/architecture/deepseek_v4/core/v4_prefill_sweep.hpp` — `materialize_layer`, `after_layer`, `end`, `reclaim_resident_staging` |
| **What** | Split the current `materialize_layer` into: **(a) settle reads + enqueue H2D** (runs in `before_layer(L)`, as now), and **(b) reclaim staging slots** — moved out of `before_layer(L)` to the next boundary. With the compute stream ordered behind the copy's event (P1.1), the body may launch while the copy is in flight. |
| **Requirement** | ✅ Ordering is non-negotiable and it holds: the body's MoE dispatch joins the pending transfer and `accumulate_routed` waits on its per-expert event before the weights are read. Verified by the byte-exactness gates and by `test_v4_prefill_window` (window == serial, bit-exact). |
| **Requirement** | ✅ `dispatch_ahead(L+1)` still issues at `before_layer(L)`; P1.2's headroom is what makes its slots free. |
| **Requirement** | ✅ At `end()` the arena drains fully (`in_use_slots() == 0`); `test_v4_prefill_sweep` and `test_v4_routed_prefill` assert it. |
| **Verify** | ✅ Byte-exactness gates pass; `h2d_drain_calls` no longer increments on the sweep path. |

### P1.4 — Measure, and gate

| | |
| :--- | :--- |
| **Status** | ✅ **measured. Gate verdict: PASS on the mechanism, BLOCKED on memory.** |
| **Command** | `AEON_WARM_GIB=<w> [AEON_SWEEP_BANKS=2] bash scripts/supply_split.sh 256 512` |
| **Result** | `h2d_drain 4.57/5.36 s → 0.006 s`; `wall_s` `33.811 → 30.574` (N=256) and `63.593 → 59.087` (N=512) at Warm 0; `32.913 → 29.400` and `62.280 → 59.055` at Warm 30. `io_wait` and decode unmoved. |
| **Gate — success** | ✅ Met: `h2d_drain → ≈0` **and** `wall_s` down `≈3.2–4.5 s` at `N = 512`, at **both** Warm sizes, byte-exactness and invariants holding. |
| **Gate — abort** | ⚠️ **Not** triggered by the wall clock — the overlap pays — but `banks = 2` needs `+3.44 GiB` pinned, and at the production Warm shape that pushed the reference box into swap (~50 GiB used, 5 GiB swap) and the run could not complete. Hence the default is `1` and the win is conditional on Phase 2. |
| **Also required** | ✅ Wall/tok-s recorded above; `bench_prefill_ab` not needed. |

**At this gate:** ✅ the analysis doc is updated — §1, §3, §5.2 C1, §7 rows 7/8, §8.4 item 4, §8.6, and the new **§10** — to match the measured outcome. Phase 1's mechanism is proven and its memory cost is the reason Phase 2 is mandatory.

---

## 3. Phase 2 — Make the overlap shippable *(mandatory — this is how Phase 1 ships)*

Phase 1 proved the win and showed its price: `+3.44 GiB` of pinned staging, which does not fit the production Warm shape (it swapped the 62 GiB box at Warm 35). This phase keeps the win and removes the over-allocation, which is what makes it compatible with the open [host-memory pressure investigation](../../analysis/current/HOST_MEMORY_PRESSURE_INVESTIGATION.md). **It is no longer "only if Phase 1 passed" — it is required to ship Phase 1's result.**

### P2.1 — Per-expert slot recycling (the rolling corridor, C2)

| | |
| :--- | :--- |
| **Where** | `src/infrastructure/core/tiered_expert_supply.hpp` (`dispatch` / the new drain), `v4_prefill_sweep.hpp` (`dispatch_ahead` / the new settle step) |
| **What** | Recycle each staging slot the moment **its own** expert's copy completes, and let the next layer's reads refill that slot — instead of holding all `E` slots for the whole layer and freeing them as a block. Reads *and* copies are already per-expert (`13.5 MiB`, `4` chunks), so this is a granularity change, not a redesign. |
| **Target** | Headroom reduced from one full bank (`E` slots) to a **bounded, reported surplus** (`S` slots, sized ~ the read-vs-copy pipeline depth). Pick `S` small (start `≤16`), and report it. |
| **Requirement** | The in-flight accounting must stay exact: a slot may not be reused while its copy is outstanding. This is what `SlotState` already encodes — do not bypass it. |
| **Requirement** | Pinned memory must be **measured and reported**, not assumed: `S × payload_bytes` on top of `E × payload_bytes`. |
| **Verify** | Byte-exactness gates; `in_use_slots() == 0` at `prefill_end`; the wall/`h2d_drain` from P1.4 **survive** (this is the acceptance test — the memory optimisation must not cost the win). |

### P2.2 — Re-check the submission ring

| | |
| :--- | :--- |
| **Where** | `src/architecture/deepseek_v4/core/v4_model_host.hpp` — `io_queue_depth` sizing (~L803–812); `src/infrastructure/io/direct_io_reader.hpp` — the SQ-full guard |
| **What** | Confirm whether P2.1 ever has **two layers' reads** outstanding at once. If it does, `io_queue_depth` must cover `2 × requests_per_expert × E`, and the CQ (`2 × SQ`) must be re-checked. |
| **Requirement** | Do not rely on the `submit_read` guard firing as the safety mechanism — size the ring correctly and keep the guard as an assertion. |
| **Verify** | `direct_io_submit_calls` / `direct_io_requests_submitted` consistent; no SQ-full throw across a full window. |

---

## 4. Phase 3 — Dispatch bookkeeping *(low priority; independent; do last)*

Measured at **`≈1%`** of both prefill (`610 ms/window`) and decode (`3.4 ms/token`). Real, small, and **independent of Phases 1–2** — it may be done any time, or skipped. Every fix is local and mechanical.

| # | Item | Where | Change | Requirement |
| :-- | :--- | :--- | :--- | :--- |
| P3.1 | `O(1)` pending-transfer count | `src/infrastructure/core/expert_registry.hpp` — `pending_transfer_count()` (~L939) | Maintain `uint32_t operations_in_flight_`; return `operations_in_flight_ - pending_demotion_count` | Increment on `NONE → {IO_PENDING, PROMOTION_PENDING, DEMOTION_PENDING}`; decrement on the return to `NONE` in `complete_request`, `fail_request`, `complete_demotion`, `drop_demotion`, `fail_demotion`, and both `warm_shadow` branches. **Exact equality with the old set-based result is the gate.** |
| P3.2 | Skip the no-demotion catalog scan | `src/infrastructure/core/tiered_expert_supply.hpp` — `schedule_demotion` (~L570) | Guard `if (expert_registry_->pending_demotion_count == 0) return;` before the loop | Correct because a dropped victim always increments that counter at reservation and decrements it at drop/fail. In the sweep the branch is unreachable anyway. |
| P3.3 | `O(1)` reap | `src/infrastructure/core/tiered_expert_supply.hpp` — `reap_registry_transfers` (~L700) | Swap-and-pop instead of erase-at-index + `continue` without incrementing | The registry is order-independent (`ensure`/`find` are by `operation_id`), so reordering is safe. Must not skip entries. |
| P3.4 | `operation_id → index` map | `src/infrastructure/core/tiered_expert_supply.hpp` — `find_registry_transfer` / `ensure_registry_transfer` | Maintain `std::unordered_map<uint64_t, size_t>` beside `registry_transfers_` | Must be updated on every insert **and** on every swap-and-pop. Only worth doing with P3.3 (they touch the same structure). |

**Gate:** no behaviour change — all existing supply/registry tests pass unchanged, and `disp_ms` in both tables falls (measure with `scripts/supply_split.sh`).

---

## 5. Guardrails — apply to every step above

These are not optional and are checked per phase, not at the end.

1. **Byte-exactness is the correctness gate.** Both supplies must remain byte-identical to the serial reference. Run: `build/bin/test_v4_prefill_sweep`, `test_v4_routed_prefill`, `test_v4_prefill_window`, `test_v4_engine`. Do **not** run the whole suite (project rule).
2. **Registry invariants hold.** `invariants_hold() == true` and no lease leaks at every boundary; `end_prefill_stream` must not throw.
3. **Staging is fully reclaimed.** `in_use_slots() == 0` after `prefill_end()`.
4. **No request-path host synchronization** and no new per-expert HIP events (§4.4b is excluded).
5. **Pinned memory is reported, never assumed.** Every phase that changes it must print the allocated bytes.
6. **Both Warm configurations.** Any gate that touches the sweep runs at `AEON_WARM_GIB=0` **and** `=35`; a win that only holds cold is not a win.
7. **Both strategies unaffected.** The routed bank and decode must not regress; they share `dispatch`/`materialize`, so re-measure decode in the same runs.
8. **One commit per phase**, with the measured before/after in the message. Conventional-commit prefix (`perf:`, `refactor:`, …).
9. **Ledger rule.** Record in the performance ledger **only** if the change moves the **end-to-end** path and the measurement is of the engine running real prompts. A hardware/capability probe (like C4) does **not** qualify.

---

## 6. Order, dependencies, and stop conditions

```
Phase 1  (P1.1 → P1.2 → P1.3 → P1.4 gate)      ← ✅ DONE: win proven (3.2–4.5 s), memory cost blocks shipping
   │
   ▼
Phase 2  (P2.1 → P2.2)                          ← MANDATORY: the shipping form of Phase 1's win
   │            └─ ABORT if the memory fix loses the Phase-1 win → keep banks=2 as opt-in
   ▼
Phase 3  (P3.1 → P3.2 → P3.3 → P3.4)            ← optional, independent, low priority
```

At each gate: **update the analysis doc**, then commit. Do not start the next phase with a stale doc.

---

## 7. Excluded — refuted or out of scope *(do not re-open without new evidence)*

| Item | Why it is not in this plan |
| :--- | :--- |
| **C4 — NVMe directly into VRAM** | Measured `NOT_SUPPORTED` (analysis §9): `EFAULT` on `pread(O_DIRECT)` **and** on the production `io_uring` path. `get_user_pages` cannot pin a BAR VMA; no userspace workaround. |
| **`banks = 2` alone** | Refuted (§8.4): reads are already `98%` hidden, so a second bank alone recovers nothing. *Coupling caveat:* Phase 1 tests it **combined with** a deferred drain — see §1. |
| **Batched H2D sync (§4.4a)** | Refuted payoff: collapsing `10 723` syncs to one saves `≈0`; the syncs absorb copy time. Subsumed by Phase 1 if the copy is overlapped. |
| **Duplicate per-expert HIP event (§4.4b)** | Bounded by `h2d_enqueue` (`47–72 ms/window`). Solve only as a side effect of P1.1, never as its own task. |
| **Single H2D copy (C5)** | Submission is not the cost (`h2d_enqueue` small); the bandwidth is already at the PCIe ceiling. |
| **`O(catalog)` scans as a *throughput* item** | Measured `≈1%` — hence Phase 3 (hygiene), not a Phase-1 lever. |
| **Decode NVMe wait (`36–63%`/token)** | **Out of scope** — its remedy is *which bytes are resident*. Hand-off to the [routing profile and placement study](ROUTING_PROFILE_AND_PLACEMENT_STUDY.md). |
| **`validate_invariants` per request** | Already off by default; the boundary audits are kept (analysis §4.6). |
| **`LayerPrefetchState::sync_state`** | µs scale; a clarity/duplication smell, not a cost (§4.5). |

---

## 8. Definition of done

- **Phase 1:** ✅ `h2d_drain → ≈0` and `wall_s` down `3.2–4.5 s` at both Warm 0 and Warm 30, with byte-exactness and invariants holding. **Caveat:** the win needs `banks = 2` (`+3.44 GiB` pinned), which does not fit the production Warm shape, so it ships **opt-in** until Phase 2.
- **Phase 2:** the same win with staging headroom bounded to a small reported surplus, not a full bank; pinned memory reported; `prefill_sweep_staging_banks` back to its default `1` and the second bank no longer needed.
- **Phase 3 (if done):** `disp_ms` down in both tables, no behaviour change. (Phase 1 gave it a reason: more transfers in flight made the `O(catalog)` scans cost `+0.15 s/window`.)
- **Throughout:** no regression in decode or the routed bank; every measurement reproducible from `scripts/supply_split.sh`. Run **one heavy process at a time** — the Warm-shaped runs are pinned-memory heavy and must not overlap with builds, tests, or other benches.
