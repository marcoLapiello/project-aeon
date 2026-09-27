# Monolith Module Split — Execution Plan

*Created 2026-09-27. Owner: this document — it is the action list. Evidence for the measurements below is the file inventory and churn count taken at creation (see §1). Scope: **structural split only**. No behaviour change, no performance claim, no new abstraction.*

**Goal:** break the files that accreted during the last implementations into focused modules, per [AGENTS.md](../../../AGENTS.md) §3 *Code Architecture & Runtime* rule 7 (modular, no monolithic mixed-concern files) and the [codebase map](../../status/CODEBASE_MAP.md) cleanup rule.

**Non-goals:** new abstractions, unifying the oracle with kernels, moving code across the `infrastructure` / `architecture` / `backend` / `platform` boundaries, or any change a test can observe. A split that alters a single emitted number is out of scope here.

**Priority note.** `reference/dsv4_oracle.hpp` is the largest file but is **not** our priority — we almost never work in it, so its pain is read-only and deferred to **Tier D**. The tiers below are ordered by *where we actually edit*, not by line count.

---

## 1. The target set (measured at creation)

All of `src/` is header-only (no `.cpp` in the engine or registry directories). Lines = `wc -l`; fan-in = files that include it; churn = commits touching it in the last ~3 months.

| File | Lines | Fan-in | Churn | Tier | What it mixes |
| :--- | ---: | ---: | ---: | :-- | :--- |
| `architecture/deepseek_v4/core/v4_model_host.hpp` | 1496 | 13 | **37** | **A** | composition root: 15 assembly steps + all accessors |
| `architecture/deepseek_v4/core/memory_budget.hpp` | 744 | 11 | **23** | **A** | runtime config + report + budget engine |
| `architecture/deepseek_v4/core/v4_layer_body.hpp` | 1177 | 13 | 6 | **B** | executor + 6 body-phase free functions |
| `infrastructure/core/tiered_expert_supply.hpp` | 1249 | 2 | 13 | **B** | transfer lifecycle + IO + HIP events + telemetry |
| `infrastructure/core/expert_registry.hpp` | 1778 | 8 | 13 | **C** | one class + 6 sub-concerns + a 190-line validator |
| `architecture/deepseek_v4/core/v4_prefill_sweep.hpp` | 673 | 1 | 13 | **C** | one class; lookahead/depth logic separable |
| `architecture/deepseek_v4/kernels/v4_attention.hpp` | 754 | 12 | 10 | **C** | attention kernels + argmax + grouped-Wo + HC-head + CPU refs |
| `infrastructure/core/routing_profile.hpp` | 885 | 1 | 4 | **C** | JSON/atomic-IO helpers + aggregate + store |
| `architecture/deepseek_v4/core/v4_layer_body_batch.hpp` | 689 | 4 | 7 | **D** | scratch class + chunk orchestration |
| `architecture/deepseek_v4/reference/dsv4_oracle.hpp` | 2754 | 20 (tests) | 17 | **D** | ~12 unrelated fp64 reference families |

**Tiering:** **A** = decompose the composition root, where recent churn concentrates. **B** = medium risk, extract coherent pieces behind a thin orchestrator. **C** = real refactors, single class split by collaborators, gated on silicon. **D** = deferred / read-only pain.

**Shared technique — umbrella headers.** Where a split would otherwise churn callers, keep the original filename as a one-screen umbrella that includes the new modules. Callers are untouched; the split is a pure header reorganisation. Use this for **every** item below unless stated otherwise.

---

## 2. Tier A — the composition root and its config *(do first)*

The two files with the highest churn. This is where the "grown during the last implementations" claim is literally true.

### A1. `memory_budget.hpp` → three units (fan-in 11) — ✅ done 2026-09-27
- [x] `aeon_runtime_config.hpp` — `AeonRuntimeConfig`, `StagingSlotCounts`, `staging_slot_counts`, `staging_slot_count`, the scratch-allowance helpers (and the constants they derive from).
- [x] `memory_budget_report.hpp` — `MemoryBudgetReport`, `prefill_carry_bytes`, `AttentionStateMemory` (promoted from a nested type; internal-only, no external reference).
- [x] `memory_budget_engine.hpp` — `MemoryBudgetEngine` (`evaluate` overloads) only.
- [x] `memory_budget.hpp` becomes the umbrella. **Gate:** `test_v4_engine`, `test_v4_staging_depth`, and the budget cross-check in `test_v4_layer_body_lifecycle` pass unchanged. *(`test_v4_staging_depth` failed once on a timing gate under load and passed on rerun; the split is a byte-identical move, so this was environmental.)*

### A2. `v4_model_host.hpp` → owner + controllers (fan-in 13)
The header itself says it is the G1 composition root; the target is to shrink it to genuine ownership and accessors. Extract one controller per commit. Drawn in dependency order — the self-contained workspace first, the coupled lifecycle last — so each step lands and is gated on its own:
- [x] `v4_prefill_workspace.hpp` — `V4PrefillWorkspace` owns the residual carry, the worst-case batch scratch, and the window/chunk it was sized for. It takes the layer vector and model config as arguments, so it needs no host reference. *Done 2026-09-27: gates `test_v4_engine`, `test_v4_graph_body`, `test_v4_prefill_sweep` pass unchanged.*
- [x] `v4_host_partition.hpp` — `V4HostPartition` owns the phase slot arithmetic and the boundary move (`apply`) and the arena re-depth (`resize`); it edits its collaborators as bound services rather than a host back-reference. *Done 2026-09-27: gates `test_v4_staging_depth`, `test_v4_prefill_sweep`, `test_v4_graph_body`, `test_v4_engine` pass unchanged.*
- [ ] `v4_prefill_controller.hpp` — `prefill_begin` / `prefill_before_layer` / `prefill_after_layer` / `prefill_end`, `restore_prefill_warm`, and the sweep accessors (`prefill_sweep`, `sweep_occupancy`, `sweep_load_ns`, `sweep_io_ns`, `prefill_sweep_engaged*`).
- [ ] Leave the host with: streams, layers/scratch/resources accessors, registry/pool/telemetry accessors, `initialize`, `free`.
- [ ] `v4_model_host.hpp` stays the umbrella. **Gate:** `test_v4_graph_body`, `test_v4_graph_head`, `test_v4_engine`, `test_v4_staging_depth` pass unchanged; `aeon_chat` token stream identical to the pre-split run.

---

## 3. Tier B — coherent pieces behind an orchestrator

### B1. `v4_layer_body.hpp` → types + phases (fan-in 13)
- [ ] `v4_layer_body_types.hpp` — `V4LayerBodyTables`, `V4LayerBodyObserver`, `V4NullLayerBodyObserver`, `trace_copy`, `V4LayerBodyRow`, `V4LayerBodyPre`, `V4LayerBodyOutput`, `select_indexer_topk`, `committed_entries_for`, `decode_layer_body_row`.
- [ ] `v4_layer_body_attention.hpp` — `run_layer_body_pre_attention`, `run_layer_body_attention_and_norm`, `run_layer_body_attention_tail`.
- [ ] `v4_layer_body_moe.hpp` — `V4RoutedExpertExecutor`, `run_layer_body_router`, `run_layer_body_moe_and_post`.
- [ ] Leave `v4_layer_body.hpp` as `run_layer_body_decoding` + the umbrella. Do **not** change the phase split itself — it exists so a chunk can interleave (plan note at header line ~797).
- [ ] **Gate:** `test_v4_layer_body_oracle`, `..._serial_oracle`, `..._compressed_oracle`, `..._chunk_oracle`, `test_v4_expert_executor` pass unchanged; `test_v4_graph_body` token stream identical.

### B2. `tiered_expert_supply.hpp` → collaborator seams (fan-in 2, but 1249 lines)
Low fan-in, so low caller churn; the value is readability of the hot path.
- [ ] Move `PayloadLocation` / `PayloadSource` / `PayloadRequest` / `PayloadTransfer` / `PayloadBatch` / `PendingTransfer` to `tiered_expert_supply_types.hpp`.
- [ ] Separate the module from any remaining telemetry surface once §0 of the [supply-chain plan](../completed/SUPPLY_CHAIN_HOT_PATH_EXECUTION_PLAN.md) §1 counters are the only telemetry left (do not move counters that a deferred Phase 3 item still edits).
- [ ] **Gate:** `test_v4_expert_tiering`, `test_supply_telemetry`, `test_v4_staging_depth` pass unchanged.

---

## 4. Tier C — real refactors *(gated on silicon; do one at a time)*

Single classes split by extracting collaborators. Each has a real regression surface, so each lands alone and is verified on `gfx1100`.

### C1. `expert_registry.hpp` → registry + partition + residency + validation (fan-in 8)
- [ ] `expert_registry_types.hpp` — enums and POD structs (`ExpertTier`, `ExpertOperation`, `ExpertGpuTransfer`, `ExpertDemotionDropReason` + name helper, `ExpertPublication`, `ExpertSlotState`, `ExpertRequestKind`, `ExpertCatalogEntry`, `ExpertDemotionReservation`, `ExpertRequestReservation`).
- [ ] `warm_partition.hpp` — `admit_warm`, `grow_host_capacity`, `shrink_host_capacity`, `release_host_tail`, `take_free_host_slot`, `rebuild_free_host_slots`, `usable_host_capacity`, `host_restore_set` / `clear_host_restore_set`, `release_reserved_host_slot`, `find_reserved_host_slot`.
- [ ] `prefill_residency.hpp` — `begin_prefill_stream` / `end_prefill_stream` / `prefill_streaming`, `release_layer`, `release_shadow_residency` + `release_shadow_residencies`, `shadow_touch`, `set_warm_frozen` / `warm_frozen`, shadow accessors.
- [ ] `expert_registry_validation.hpp` — `invariants_hold`, `checked_validate`, `validate_invariants` (~190 lines), `validate_lru`.
- [ ] **Gate:** `test_expert_registry_warm_state`, `test_dynamic_expert_pool`, `test_routing_reuse`, `test_v4_expert_tiering` pass unchanged, on hardware.

### C2. `v4_prefill_sweep.hpp` → sweep + lookahead policy (fan-in 1)
- [ ] Extract the derived-depth / read-ahead policy (`derived_lookahead_capacity`, `read_lookahead_capacity`, `set_read_ahead_max`, `update_frontier`, `dispatch_ahead`, `advance_reads`, `reads_in_flight`) into a policy object the sweep drives.
- [ ] **Constraint:** the corridor requirements R1–R8 of the [supply-chain plan](../completed/SUPPLY_CHAIN_HOT_PATH_EXECUTION_PLAN.md) §0.3 still hold — carry them, unchanged, as the acceptance. **Gate:** `test_v4_staging_depth` Gates A/B/C pass unchanged.

### C3. `v4_attention.hpp` → kernels by concern (fan-in 12)
- [ ] `v4_attention_kernels.hpp` — sliding / cached-sliding / cached-compressed attention, compressor state + materialization, indexer scores.
- [ ] `v4_argmax.hpp` — the two argmax phase kernels (LM head; currently misfiled here).
- [ ] `v4_grouped_wo.hpp` — `v4_grouped_wo_a_wave32_kernel`, `v4_half_to_float_n_kernel`.
- [ ] `v4_hc_head_kernel.hpp` — `hc_head_wave32_kernel`.
- [ ] **Flag — anti-circularity.** The `cpu_rmsnorm` / `cpu_sliding_window_attention` references sit in the same header as the kernels they certify. Move them to a `_reference.hpp` and confirm the gate in `test_v4_attention_sink_oracle` still compares independent code ([AGENTS.md](../../../AGENTS.md) §3 rule 5, *Anti-circularity*). Investigate, do not silently relocate.
- [ ] **Gate:** `test_v4_attention_sink_oracle`, `test_v4_mla_oracle` pass unchanged, on hardware.

### C4. `routing_profile.hpp` → IO helpers split (fan-in 1)
- [ ] `routing_profile_json.hpp` — `namespace routing_profile_detail` (parser helpers, `json_escape`, `write_string` / `read_string`, `write_value` / `read_value`, `fnv1a_*`).
- [ ] Keep prompt parsing, `RoutingProfileAggregate`, `RoutingProfileStore` in the main header. **Gate:** `test_routing_profile`, `test_routing_reuse` pass unchanged.

---

## 5. Tier D — deferred / read-only pain

Explicitly **not** scheduled. Listed so the inventory is complete and so a future reader does not "discover" them as new work.

- [ ] **`v4_layer_body_batch.hpp`** (fan-in 4) — split the `V4LayerBodyBatchScratch` class from `compose_local_rows` / `run_chunk_pre_attention` / `run_layer_body_chunk`. Do only when next edited.
- [ ] **`reference/dsv4_oracle.hpp`** (2754 lines, fan-in 20 tests) — the largest file, and **deprioritised: we almost never work in it.** If ever split, it is a mechanical free-function bundle (`oracle_support`, `oracle_rope`, `oracle_hc`, `oracle_attention`, `oracle_router`, `oracle_swizzled`, `oracle_expert`, `oracle_layer_body`) behind an umbrella, with **zero test edits**. It is the safest split in the repo precisely because it is read-only — which is also why it can wait.

---

## 6. Ordering and rules

**Sequence:** A1 → A2 → B1 → B2 → then C1, C2, C3, C4 one at a time. A first, because it is where the churn is and it de-risks the rest. D never, unless the file is edited for another reason.

**Per-step rules (inherited, non-negotiable):**

- **One split per commit**, conventional-commit subject (`refactor:`), umbrella preserved so callers do not move in the same change.
- **No behaviour change.** The only permitted diff is `#include` lines and file locations. If a split needs a code edit, it is a separate task.
- **Selective tests only.** Run the gates named for that step; do not run the full suite ([AGENTS.md](../../../AGENTS.md) §3 rule 8). Hardware gates (`gfx1100`) for every Tier C step.
- **Anti-circularity** ([AGENTS.md](../../../AGENTS.md) §3 rule 5) — a split must never make a kernel and its oracle share a helper, and any existing co-location (C3) is a finding to report, not to preserve silently.
- **Update the [codebase map](../../status/CODEBASE_MAP.md)** when a split introduces a new engine-path file, in the same change.
- **No ledger entry.** This plan changes no measurement; [PROJECT_STATUS.md](../../status/PROJECT_STATUS.md) §3 does not move until a tier completes, at which point a single row records the structural milestone.

**Done when:** every Tier A and B item is checked and each Tier C item is either checked or explicitly deferred here.

---

## 7. Checklist summary

| Tier | Item | State |
| :--- | :--- | :--- |
| A1 | `memory_budget.hpp` → 3 units | ✅ |
| A2 | `v4_model_host.hpp` → owner + 3 controllers | ◐ workspace, partition done |
| B1 | `v4_layer_body.hpp` → types + phases | ☐ |
| B2 | `tiered_expert_supply.hpp` → types + seams | ☐ |
| C1 | `expert_registry.hpp` → registry + partition + residency + validation | ☐ |
| C2 | `v4_prefill_sweep.hpp` → sweep + lookahead policy | ☐ |
| C3 | `v4_attention.hpp` → kernels by concern (+ anti-circularity check) | ☐ |
| C4 | `routing_profile.hpp` → IO helpers split | ☐ |
| D | `v4_layer_body_batch.hpp`, `dsv4_oracle.hpp` — deferred, on-edit only | — |
