# Supply-Chain Hot-Path — Execution Plan (completed)

*Created 2026-09-24; restated as a spec 2026-09-25; **completed 2026-09-27**. Owner: [SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md](../../analysis/current/SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md) — this plan is the action list, the analysis doc owns the evidence. Scope: how fast the bytes arrive, not which bytes.*

**Phase 1 and Phase 2 are done**, with the §0 spec holding (R1–R8) and every step measured on silicon. Only **Phase 3 (dispatch bookkeeping)** is open, deferred as low value (§3). Ledger: M42–M46. Analysis: §8–§19.

---

## 0. The pipeline model (the spec this plan implements)

### 0.1 Units

- **layer-block** = one layer's expert set = `E` payloads (`E = experts_per_layer`).
- **vram_blocks** — decided **at load** from the budget (`hot_slots = total_vram − dense − KV − headroom − scratch`): `hot_slots ≥ 2E → 2`, else `1`.
- **staging_blocks** — the pinned staging arena in layer-blocks. **2 in both cases**: one block is the read destination, one is the copy source.
- **One block is always the compute block** (`L`). It is not pipeline capacity.

### 0.2 Target steady state

| Scenario | Blocks | Steady state during `body(L)` |
| :--- | :--- | :--- |
| **hot ≥ 2E** | 2 vram + 2 staging | `L` computing │ `L+1` in VRAM │ `L+2` copying in │ `L+3` reading |
| **hot < 2E** | 1 vram + 2 staging | `L` computing │ `L+1` copying in │ `L+2` reading |

*Scenario 2 has a hard limit regardless of staging size: with one VRAM block, `L`'s copies cannot start until `body(L-1)` ends, so its staging slots do not free until mid-body and `L+1`'s reads cannot be issued at the boundary. `2E` staging still buys read(`L+2`) ∥ copy(`L+1`), but the copy itself stays partly exposed — do not claim scenario 1's steady state here.*

### 0.3 Requirements (normative — every step was judged against these)

| # | Requirement | Verdict |
| :-- | :--- | :--- |
| **R1** | The **only** physical ordering: a VRAM block must not be overwritten while the GPU reads it as weights. Enforced by a **stream event**, never by a host synchronize. | ✅ |
| **R2** | The staging arena is **2 layer-blocks** (`2E`) in **both** scenarios — not derived from `vram_blocks`. R6 then requires the design to work below that default. | ✅ |
| **R3** | Every stage advance is triggered by a **completion event**, not the layer boundary: slot-free → issue one read into it; read-event → enqueue that expert's copy; copy-event → release that expert's staging slot; compute-event → release layer `L`'s VRAM block. The copy *fills* VRAM; it never frees it. The **first** link did not exist at plan time — reads were a single wave at the boundary. | ✅ |
| **R4** | No host synchronization on the supply path. The `v4_graph` per-layer sync may remain only for what genuinely needs it (the router readback). | ✅ |
| **R5** | Lookahead is **derived from the free blocks at runtime**; no hardcoded depth. | ✅ |
| **R6** | Depth is a **memory budget, not a schedule**: the slot count changes how much RAM the corridor spends, **never** how fast it runs (Gate A). | ✅ |
| **R7** | The work is resource-invariant: identical token/layers/experts at every depth (Gate B). | ✅ |
| **R8** | No hardware constant (bandwidth, latency, ratio) may select behaviour. Pinned memory is reported, never assumed. | ✅ |

**The defect R6 named, and how it closed.** At plan time `1E` was the default and the arena was **one room with two doors** — `read(L) → copy(L) → drain all → read(L+1)`, strictly serial across layers — while `2E` (`staging_base = (layer % 2) * E`) gave two rooms and the cross-layer overlap, worth a measured `+13%` at `N = 512`. So the arena *size* was selecting the *algorithm*. P2.3's within-body pump removed that dependence (`1E` gained `2.7 s`), and P2.6 made depth a pure budget: **`2E…6E` all land in `29.83–30.39 s`, Gate A `1.009×`**. `< E` remains **unreached** and is a portability item, not a throughput one — on this pool VRAM residency is the limiter (§4).

### 0.4 Already right — do not rebuild

- Per-expert read→copy pipelining inside a layer (`materialize` enqueues copy *i* when read *i* lands).
- The VRAM frontier already adapts to the pool (measured: holds as many whole layers as fit; frontier `44`).
- Consumer-side ordering (`accumulate_routed` waits per-expert on the transfer's event) — why Phase 1 works.

---

## 1. Instrumentation (landed — use, do not rebuild)

- `TieredExpertSupply` counters (`io_wait_ns`, `h2d_enqueue_ns`, `h2d_drain_ns`, `h2d_drain_calls`, `dispatch_cpu_ns`, `direct_io_submit_ns`, `reset_transfer_counters()`), exposed on `V4ExpertSupplyCoordinator` / `V4ModelHost`.
- `tests/bench_supply_split.cpp` + `scripts/supply_split.sh` — the exposed-load split, one model load.
- `tests/test_v4_staging_depth.cpp` — the portability gates (A, B) and Gate C (corridor occupancy).
- `PrefetchStagingArena::StateCounts` (free / reading / copying), sampled per layer into `V4PrefillSweep::occupancy_samples()` — the corridor's fill is observed, not argued.
- `V4ModelHost::resize_staging_slots(slots)` — set the arena depth at runtime.

---

## 2. Phase 1 and Phase 2 — what each step did and what it measured

### Phase 1 — swept H2D overlapped with its compute ✅

The swept layer's H2D no longer blocks the host before the body: `materialize_layer` settles reads and holds a bank, `after_layer` reclaims it, and ordering is the **consumer's**.

| Warm | N | `h2d_drain` | `wall_s` |
| ---: | ---: | ---: | ---: |
| 0 | 256 | `4.57 → 0.006 s` | `33.811 → 30.574` |
| 0 | 512 | `5.36 → 0.006 s` | `63.593 → 59.087` |
| 30 | 256 | `4.23 → 0.006 s` | `32.913 → 29.400` |
| 30 | 512 | `4.25 → 0.006 s` | `62.280 → 59.055` |

`io_wait` and decode unmoved. Blocker at the time: a second bank (`+3.44 GiB` pinned) swapped the box at the production Warm shape, so it shipped **opt-in** until Phase 2 absorbed it. Evidence: analysis §10.

### Phase 2 — the corridor as a demand-driven pipeline ✅

Each step is independently verifiable and was independently committed. Key figures below; the analysis section named is the evidence.

| Step | Change | Measured | R | Evidence |
| :--- | :--- | :--- | :-- | :--- |
| **P2.1** arena `2E` unconditionally + corridor-fill readout | `staging_slot_count` returns `2E` whenever `prefill_sweep` is on; the settable-banks knob **removed** (a settable depth lets a resource select the algorithm); `StateCounts` + per-layer `BlockOccupancy` added | Gate C added: `0/43` layers overlapped at `1E` (drain `4.9 s`) vs **`42/43` at `2E`** (`0.006 s`) | R2, R8 | §12 |
| **P2.2** completion-driven staging release | The reaper releases each slot the moment **its own** `h2d_event` fires (`release_if_copying`, idempotent); the sweep reaps **before** its block reclaim | **No wall-time change** (`30.913` vs `30.641 s`) — a structural prerequisite, never a throughput claim. `staging_released_on_completion` = `10198` at `2E`, `0` at `1E` | R3 | §13 |
| **P2.3** copy into VRAM as a read lands | Non-blocking materialize: drain what the CQ holds, enqueue each expert's copy when **its own** reads land. Driven by the executor's per-token hook — the only host activity inside a body | `1E`: `34.896 → 32.204 s` (drain `4.708 → 1.613 s`); `2E` neutral. **Gate A spread `1.133× → 1.055×`** | R1, R3, R4 | §14 |
| **P2.4** lookahead derived from free blocks | The single-slot lookahead became a **queue**, length `min(free VRAM blocks, free staging blocks)` recomputed at each boundary; already-resident layers are skipped without spending budget | Spread `1.131× → ~1.04×`; **Gate A passes**. Limiter is **VRAM, not staging**: mid-window `vram_free` `14–16 sl` vs `staging_free` `7 sl` (`1E`) / `192 sl` (`2E`), because `466–539` of `797` slots (`58–68%`) are preserved decode residents | R5, R3 | §15 |
| **P2.5** portability gates (acceptance) | `tests/test_v4_staging_depth.cpp` | Gate A passes (`1.014–1.027×`, both Warm shapes; was `1.144×`); Gate B holds; Gate C holds. `< E` unreached | R6, R7 | §11 |
| **P2.6** decouple the read leg from VRAM; serialize the read waves | (a) A cold read reserves **no** VRAM destination; `attach_vram_destination` supplies one when the copy can run, and `enqueue_expert_copy` returns `false` on a full pool, leaving the expert staged to retry. (b) **One read wave in flight**, the next issued by `advance_reads()` the instant the previous lands | `2E…6E` all `29.83–30.39 s`; **Gate A `1.009×` PASS** (was `1.382×` FAIL); **Gate C `42/43` at every depth** (was `0/43` at `1E`, `21/43` at `3E`+). The `3E` pathology (`41.9 s`) was **two waves in front of the drive**, not depth | R3, R5, R6 | §16, §17 |
| **P2.7** extend the staged-only deferral to the Warm shadow | One admission rule for both sources: a read **or** a Warm hand-off takes a staging slot immediately and a VRAM slot only when its copy can run. A deferred Warm hand-off borrows its slot without a payload copy (`mark_ready`) and uploads from the pinned host slot | `Warm 30` default arena: **abort → `29.093 s`, `io_wait 0.436 s`**. `16 GiB` Warm sweep **bit-exact** (`0 differing of 258560`), Warm unchanged (`0 of 1213`) | R3, R5, R6 | §18 |
| **P2.8** one pinned region, cut to **three** corridor requirements | `warm_host_bytes` became the **total**; Warm and the corridor share one `ExpertHostRegion` via non-owning views, and the boundary is cut to the running phase. Decode needs `2×6`; a routed window `min(6C, E)`; a swept one `blocks × E`. A surrendered Warm slot is demoted, **recorded**, and re-admitted at `prefill_end` | Decode Warm **`1640 → 1884` slots** (`162 MiB` corridor vs `3456 MiB`); routed `1640`; swept `1128`. **TTFT `158.6 → 162.1 s` (`+2.2%`)** — the swept window now borrows `756` Warm experts instead of `512` and re-reads them | R2, R5, R6 | §19 |

**Cross-step note (P2.4 → P2.8).** The `1E`/`2E` gap fell to `~3%` once P2.3's pump carried the overlap, and P2.6 showed the arena size does not move throughput at all. The arena is therefore chosen **on memory alone**; `prefill_sweep_staging_blocks` (default `2`) is the user's pinned-memory knob, and the earlier question of reverting to `1E` was closed by keeping `2E`.

---

## 3. Deferred — Phase 3, dispatch bookkeeping *(low value)*

Measured at **`≈1%`** of both prefill (`610 ms/window`) and decode (`3.4 ms/token`). Independent of Phases 1–2 and never required by them. **Deferred 2026-09-27 as low value**: every item is local and mechanical, so it stays here as a ready list rather than an open obligation. (Phase 1 did give it a reason to exist: with the deferred drain more transfers are in flight at once, and `disp_ms` rose from `698` to `856 ms` at `Warm 30`, `N = 512`, analysis §10.3.)

| # | Item | Where | Change |
| :-- | :--- | :--- | :--- |
| P3.1 | `O(1)` pending-transfer count | `expert_registry.hpp` — `pending_transfer_count()` | Maintain `uint32_t operations_in_flight_`; return `operations_in_flight_ - pending_demotion_count`. Increment on `NONE → {IO_PENDING, PROMOTION_PENDING, DEMOTION_PENDING}`; decrement on the return to `NONE` in `complete_request`, `fail_request`, `complete_demotion`, `drop_demotion`, `fail_demotion`, and both `warm_shadow` branches. **Exact equality with the old set-based result is the gate.** |
| P3.2 | Skip the no-demotion catalog scan | `tiered_expert_supply.hpp` — `schedule_demotion` | Guard `if (pending_demotion_count == 0) return;` before the loop. Correct because a dropped victim always increments that counter at reservation and decrements it at drop/fail. Unreachable in the sweep anyway. |
| P3.3 | `O(1)` reap | `tiered_expert_supply.hpp` — `reap_registry_transfers` | Swap-and-pop instead of erase-at-index + `continue` without incrementing. The registry is order-independent (`ensure`/`find` are by `operation_id`), so reordering is safe; must not skip entries. |
| P3.4 | `operation_id → index` map | `tiered_expert_supply.hpp` — `find_registry_transfer` / `ensure_registry_transfer` | `std::unordered_map<uint64_t, size_t>` beside `registry_transfers_`, updated on every insert **and** every swap-and-pop. Only worth doing with P3.3. |

**Gate if resumed:** no behaviour change — all supply/registry tests pass unchanged, and `disp_ms` falls in both tables (`scripts/supply_split.sh`).

---

## 4. Excluded — refuted or out of scope *(do not re-open without new evidence)*

| Item | Why it is not in this plan |
| :--- | :--- |
| **C4 — NVMe directly into VRAM** | Measured `NOT_SUPPORTED` (analysis §9): `EFAULT` on `pread(O_DIRECT)` and on the production `io_uring` path. No userspace workaround. |
| **`banks = 2` as a *schedule*** | The count must not select behaviour (R6). It stays only as the arena's depth. |
| **Batched H2D sync (§4.4a)** | Refuted payoff: the syncs absorb copy time. Subsumed by P2.3. |
| **Duplicate per-expert HIP event (§4.4b)** | Bounded by `h2d_enqueue`. Solve only as a side effect of P2.3, never as its own task. |
| **Single H2D copy (C5)** | Submission is not the cost; bandwidth is already at the PCIe ceiling. |
| **`O(catalog)` scans as a *throughput* item** | Measured `≈1%` — hence Phase 3 (hygiene), deferred. |
| **Decode NVMe wait (`36–63%`/token)** | **Out of scope** — its remedy is *which bytes are resident*. Handed to the [routing profile and placement study](../active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md). |
| **A larger staging arena as a *speed* lever** | Measured `2E…6E` all `29.83–30.39 s` (analysis §17.4); the drive is the limiter, so depth is a budget only (R6). |
| **Bounding the read queue by free VRAM** | Tried in §16.4 and **refuted in §17**: it *is* the coupling the decoupling removes (the cartridge box must not be sized by the rifle's magazine wells). The real `3E` defect was two read waves in front of the drive. |
| **`validate_invariants` per request** | Already off by default; boundary audits kept (analysis §4.6). |
| **`LayerPrefetchState::sync_state`** | µs scale; a clarity smell, not a cost (analysis §4.5). |
| **The 4-block pipeline** (`L` computing │ `L+1` resident │ `L+2` copying │ `L+3` reading) | Needs **four** layers of VRAM state live. The limit is **residency, not the corridor** — a *which-bytes* question, not a supply question. |

---

## 5. Definition of done — met

- **Phase 1:** ✅ `h2d_drain → ≈0`, `wall_s` down `3.2–4.5 s` at Warm 0 and Warm 30, byte-exact. The second-bank cost that blocked shipping was absorbed by Phase 2.
- **Phase 2:** ✅ the §0 spec holds — **R1–R8 met**:
  - **R2** ✅ arena is `2E` in both scenarios; no settable-depth knob.
  - **R3** ✅ every stage advance is a completion event, including R3's first link (a read issued as a slot frees).
  - **R5** ✅ lookahead derived from free blocks, no constant.
  - **R6** ✅ **Gate A passes** (`1.009×`); depth is a pure budget — `2E…6E` within `1%`.
  - **R7** ✅ **Gate B** holds; **Gate C** `42/43` at every depth.
- **Throughout:** no regression in decode or the routed bank. The byte-exactness gates are `test_v4_prefill_sweep`, `test_v4_routed_prefill`, `test_v4_prefill_window`, `test_v4_engine`; the portability gates are `./build/bin/test_v4_staging_depth`; the exposed-load split is `scripts/supply_split.sh`. Full gate set at close: `16`/`18`/`10`/`38`/`7`/`11`/`12`/`29` checks, 0 failures.

**Process rules this plan used** (kept for reference; the project-wide form is in [AGENTS.md](../../../AGENTS.md)): byte-exactness is the correctness gate; registry invariants and `in_use_slots() == 0` hold at every boundary; no host sync on the supply path; no hardware constant selects behaviour; both Warm configurations; one commit per step with before/after measured; ledger only for end-to-end runs of real prompts; one heavy process at a time.
