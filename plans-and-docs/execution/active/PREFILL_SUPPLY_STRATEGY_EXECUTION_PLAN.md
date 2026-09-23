# Prefill Supply Strategy Execution Plan

*Status: active. Opened 2026-09-23. Refines Step 6–7 of [EXPERT_STREAMING_EXECUTION_PLAN.md](EXPERT_STREAMING_EXECUTION_PLAN.md); supersedes the supply proposals of [PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md](../../analysis/current/PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md) §5.3, §8, §9.4, §9.6 (its multi-GPU material is out of scope here).*

**Subject.** One layer-major batched prefill, two expert-supply strategies chosen by a visible prompt-length gate, and a Hot-pool **restore** that both strategies share.

**Scope.** The supply strategy inside the layer-major window and the registry state that backs it. Numerics are certified elsewhere and are not re-opened; the storage layer's own targets, session/prefix state, and KV precision are not this plan's. Measurements go to the [Performance Ledger](../../status/PERFORMANCE_LEDGER.md).

---

## 1. The requirement

The layer-major batched prefill (`V4Graph::forward_window`) is the single prefill path. Its expert supply has two strategies, selected by prompt length:

| Window `N` | Supply strategy | Rationale |
| :--- | :--- | :--- |
| `N ≥ gate` | **Sweep** — load each layer's whole set in layer order, lookahead at `L+1` | The layer set is known; blind whole-layer load is near-optimal when the chunk touches ~`E` experts. |
| `N < gate` | **Routed** — route-aware, cached, per-chunk union | A short prompt touches a small distinct set; a whole-layer load over-reads it. |

Both strategies share one requirement:

> **The Hot pool must be left as it was found.** Whatever each strategy drains at prefill entry, the same experts are resident again when the prefill ends, so decode resumes on the allocation it had before — the precondition for the future LFU policy.

---

## 2. Vocabulary and the two budgets

| Symbol | Meaning |
| :--- | :--- |
| `E` | `experts_per_layer` (256) — one layer's set, the unit of every budget below. |
| `H` | `registry.vram_capacity` — the Hot pool, derived from hardware and config at boot. |
| `N` | the window length (prompt tokens in one layer-major pass). |
| **Sweep drain** | slots freed at prefill entry for the sweep's layer sets. |
| **Routed bank** | slots the routed strategy may occupy — **always `E`**, no lookahead. |
| `restore_set_` | the gids drained at entry; the set re-admitted at exit. |

**All budgets are expressed in units of `E`, never absolute slots.** No step may hard-code a slot count, a byte count, or a measured rate; the numbers in the analysis doc are configuration-specific and are not requirements.

### The minimum viable pool

The engine runs three modes; the smallest pool that admits all three is `E`:

| Mode | Working set held at once | Minimum `H` |
| :--- | :--- | :--- |
| serial decode | `6` leases + a reclaimable victim | `6` |
| routed batched prefill (below gate) | one chunk's distinct set `min(6C, E)` | `≤ E` |
| swept prefill (above gate) | one whole layer set `E` (single-buffered) | `E` |

Below `E` the engine still decodes and still runs the batched body, but the sweep is infeasible — which `V4PrefillSweep::is_feasible()` already encodes. This plan does not change that floor.

### The drain rule (configuration-independent)

Free only what the strategy needs, and no more:

| `H` | Sweep drain | Consequence |
| :--- | :--- | :--- |
| `H ≥ 2E` | `2E` | double buffer preserved (lookahead at `L+1` runs) |
| `E ≤ H < 2E` | `E` | single-buffered; **still produces identical output** — `dispatch_ahead` becomes a no-op and each layer loads at its own boundary |
| `H < E` | — | sweep infeasible; the gate forces the routed strategy |

The routed bank is `E` under every `H`. On the floor (`H = E`) the bank is the whole pool; on a larger pool it leaves the preserved residents untouched.

---

## 3. The registry expansion (shared by both strategies)

Four additions to `ExpertRegistry`, no strategy-specific variants:

1. **`resident_at_prefill_begin`** — a per-catalog-entry mark, set on every `HOT_VRAM` resident when the prefill opens and cleared when one is drained, so a mark means exactly *preserved resident*. Only Hot residents can carry it: a Warm shadow exists **only during** frozen prefill and every one is cleared at entry (`begin_prefill_stream()` resets `warm_shadow`/`shadow_vram_slot`), so at prefill open there are no shadows to mark, and the mark cannot be derived from or applied to them.
2. **`restore_set_`** — the gids the prefill drained at entry, recorded in drain order. By construction these are the Hot residents that were drained — never shadows, which do not exist at entry.
3. **A release that spares pre-existing residents** — the per-layer release frees only residents **admitted during the prefill**, never one marked in (1). Without this the partial drain is decorative: the layer release would free the preserved experts anyway and the restore would have saved nothing.
4. **An end that re-admits `restore_set_`** — reads the drained gids back from Cold and clears the marks. The routed strategy and the sweep share this end.

The end invariant is **weakened in bounded mode**: today `end_prefill_stream()` requires Hot empty; in bounded mode Hot is expected to hold preserved residents plus the restored set. One begin/end pair, the mode decides the invariant — not two pairs.

---

## 4. The steps

Each step states its requirement, its gate, and its files. A step is done when its gate is green and its evidence is in the ledger.

### Step 1 — The registry restore infrastructure

**Requirement.** Implement the four additions of §3 as registry state and API: mark residents on prefill entry, record `restore_set_`, spare marked residents in the per-layer release, and re-admit `restore_set_` on exit. No strategy changes yet — this step only makes begin/end able to preserve and restore a set.

**Gate.**
- `invariants_hold()` passes at every boundary with marks set and cleared.
- A drain-then-restore round trip leaves the catalog byte-for-byte identical to before (same owners, same slots, same LRU order for preserved experts).
- A **full** drain (`drain_slots` covering the pool) preserves nothing: `preserved_resident_count() == 0`, `restore_set()` holds every resident, and every mark is cleared after `end_prefill_stream()`.
- Unit test: construct a registry, drain a bounded subset, release a layer that overlaps the preserved residents, assert the preserved experts survive and the prefill-admitted ones go, then end and assert the pre-prefill set is resident after the restore reload.

**Files.** `src/infrastructure/core/expert_registry.hpp`; `tests/test_expert_registry_warm_state.cpp`.

---

### Step 2 — Bounded drain for the sweep

**Requirement.** `V4PrefillSweep::begin()` drains `2E` when `H ≥ 2E`, else `E` — the worst-LRU residents first — instead of the whole pool; the remainder is preserved (Step 1). The sweep's per-layer release spares preserved residents, and `V4ModelHost::prefill_end()` reloads `restore_set_` through the normal cold path so the pool returns to its switch-point set.

**Gate.**
- Sweep output is byte-identical to serial (existing sweep correctness test), for `H ≥ 2E` and for `E ≤ H < 2E`.
- `sweep_lookahead_depth()` reports `1` at `H ≥ 2E` and `0` at `E ≤ H < 2E` — the degradation is measured, not assumed.
- After `prefill_end()`, the Hot set equals the pre-prefill set (`--dump-logits` run plus a residency comparison).
- No leak: `invariants_hold()`, `outstanding_leases() == 0`, `staging_in_use == 0`.

**Files.** `src/infrastructure/core/expert_registry.hpp`, `src/architecture/deepseek_v4/core/v4_prefill_sweep.hpp`, `src/architecture/deepseek_v4/core/v4_model_host.hpp`, `tests/test_v4_prefill_sweep.cpp`.

---

### Step 3 — The gate setting and the strategy switch

**Requirement.** Add `AeonRuntimeConfig::prefill_sweep_min_tokens`, defaulting to `E / 4` (≈64), **user-configurable** via `--prefill-sweep-min-tokens <n>`. `V4Graph::forward_window` passes the window length to `V4ModelHost::prefill_begin(count)`; the host selects **sweep** when `count ≥ gate` and **routed** otherwise. The gate is the only condition — no hidden threshold, and the selected strategy is recorded and reportable.

**Gate.**
- At `count ≥ gate` the sweep engages (`prefill_sweep_engaged() == true`); at `count < gate` it does not.
- Setting `0` selects the sweep for every window; setting a value above the prompt length forces the routed strategy.
- Below the gate, layer-major structure is unchanged (window → layer → chunk) — only the supply differs.

**Files.** `src/architecture/deepseek_v4/core/memory_budget.hpp` (config), `src/architecture/deepseek_v4/core/v4_model_host.hpp` (`prefill_begin(count)`), `src/architecture/deepseek_v4/core/v4_graph.hpp` (`forward_window`), `tools/aeon_chat.cpp` (flag).

---

### Step 4 — The routed-cached below-gate supply

**Requirement.** Below the gate the window runs the route-aware cached supply with:
- a **bank of `E` slots** — admission capped so one layer's working set cannot churn the pool;
- **cross-chunk residency at the layer boundary** — the chunk union is accumulated through the existing `on_routing_ready_batch` and held until the layer retires (the layer-major guarantee the routed path currently lacks);
- **`ensure_pool_headroom` suppressed** while the bank is active: on the floor pool (`H = E`) the valve fires on the first chunk and would evict the union mid-layer;
- **Warm frozen with shadow copies**, as in the sweep — a Warm hit is a host→device copy that does not transfer ownership;
- the **shared release and restore** of Steps 1–2.

**Gate.**
- Byte-identical to serial greedy ids and logits, through the full layer-major window.
- Residency guarantee: `batch_distinct() < batch_draws()` and no expert of a layer is re-fetched after its first chunk (telemetry: no repeat `bytes_from_nvme` for the same gid within a layer).
- Warm preserved: Warm's resident set and LRU are unchanged across the prefill; `shadow_copies()` rises when Warm is populated.
- No leak: the invariants and counters of Step 2's fourth bullet.
- Works on the floor pool (`--max-hot-slots E`) and on the starved pool (`--max-hot-slots 6` still decodes; routed path at `H = E`).

**Files.** `src/architecture/deepseek_v4/core/v4_expert_executor.hpp`, `src/architecture/deepseek_v4/core/v4_expert_supply.hpp`, `src/architecture/deepseek_v4/core/v4_model_host.hpp`, `src/infrastructure/core/tiered_expert_supply.hpp`.

---

### Step 5 — One end, one invariant

**Requirement.** Collapse the routed and swept begin/end onto the single registry pair of §3 so both modes share the same entry drain, the same sparing release, the same restore, and the same (weakened) end invariant. Remove any strategy-specific teardown left from Step 4.

**Gate.** A sweep-then-decode run and a routed-then-decode run both end with the identical pre-prefill Hot set; `invariants_hold()` at `prefill_end()` under both modes; no second `end` path remains.

**Files.** `src/architecture/deepseek_v4/core/v4_model_host.hpp`, `src/architecture/deepseek_v4/core/v4_prefill_sweep.hpp`.

---

### Step 6 — The measurement

**Requirement.** Measure the three arms — serial, swept, routed — at fixed `C`, `W`, and config, one pristine process each, across `N ∈ {gate/2, gate, 2·gate, 256, 512}`. Report the crossover and confirm the gate default is on the correct side of it. This is the measurement the plan exists to justify; it does not set a performance target, it locates the switch.

**Gate.** A table with, per arm and `N`: seconds, tok/s, NVMe bytes, and the restore cost. The gate default is confirmed or corrected from the table (a correction changes the default, not the mechanism).

**Files.** `tests/bench_prefill_ab.cpp` (add the `routed` arm), `scripts/prefill_ab.sh`.

---

### Step 7 — Ledger and documents

**Requirement.** Record the milestone in the Performance Ledger; update `AGENTS.md` if a milestone row transitions; mark the superseded analysis sections per the header.

**Gate.** The ledger entry names the gate default, the drain rule, and the restore result.

---

## 5. Rules that bind this work

| Rule | What it forbids |
| :--- | :--- |
| **Budgets in units of `E`** | Hard-coding a slot count, a byte count, or a measured rate anywhere in the mechanism. |
| **The minimum viable pool is a contract** | Adding a mode that needs more than `E` slots without stating it as a new floor and rejecting the boot below it. |
| **No hidden thresholds** | Any eligibility rule that cannot be read from configuration and reported. |
| **Correctness gates first** | Exercising the routed or swept supply on a path not already proven byte-identical to serial. |
| **Restore is a requirement, not an optimization** | Shipping a prefill that leaves Hot different from how it was found. |
| **A preserved resident must be released by nobody** | The per-layer release freeing a `resident_at_prefill_begin` expert. |
| **Warm is frozen; a shadow is not ownership** | Promotion/demotion during prefill, or a shadow surviving past `prefill_end()`. |
| **Anti-circularity** | Grading a mode against an oracle derived from that mode's own helpers. |

---

## 6. Open work

| Item | State | Detail |
| :--- | :--- | :--- |
| **Restore overlap** | deferred | The restore read (`restore_set_` × payload) is on the prefill's critical path; overlapping it with the first decode layer is a later optimization, not part of the mechanism. |
| **LFU restore source** | future | The restored set is the pre-prefill set; under LFU the frequency ranking will make it the correct set to restore. No work now. |
| **Routed bank width** | closed | Always `E`; there is no lookahead in the routed strategy. |
| **Multi-GPU** | out of scope | Not this plan's subject. |
