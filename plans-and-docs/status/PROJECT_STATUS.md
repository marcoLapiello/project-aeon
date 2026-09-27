# Project Aeon — Project Status

**Audited: 2026-09-27, on branch `main`.**

This is the **single progress-tracking document**. It owns what is done, what is in flight, what is open, and what is deliberately future. Read it with [AGENTS.md](../../AGENTS.md), which owns the stable context — purpose, external references, and engineering rules.

Numbers do not live here. Measured results belong to the [Performance & Accuracy Ledger](PERFORMANCE_LEDGER.md), design rationale to the analysis document that owns it, and step-by-step implementation to the plan.

---

## 1. Where the project is

Project Aeon is **rewriting its DeepSeek-V4 inference graph**. The storage, streaming, artifact-format and kernel layers are kept; the graph that composes them is rebuilt from a verified specification — [DSV4 Inference Pipeline Plan](../execution/completed/DSV4_INFERENCE_PIPELINE_PLAN.md), the authority for graph semantics, whose every `[V]` claim cites readable reference code and whose every remaining unknown names the gate that settles it.

**Why.** An audit of the runtime against the selected checkpoint found structural errors in the graph — not tuning gaps: a missing Hyper-Connections comb scale, a missing compressor APE term, and HCA layers running indexer selection they do not have. The tests of the time did not catch them because several compared a kernel against an oracle derived from the same helper (the anti-circularity rule). Measurements taken against that graph describe a different computation and are invalid; the ledger records the deletion. The pre-rewrite graph and its gate were deleted on 2026-09-18, ~5,600 lines.

| Field | Value |
| :--- | :--- |
| Branch | `main` |
| Default `ctest` | 50 tests |

---

## 2. How the documentation is organised

Four folders under `plans-and-docs/`, one job each:

- **`execution/`** — the plans. `active/` holds plans with open gates the project still depends on; `completed/` holds plans whose gates are met; `superseded/` holds plans replaced by a different approach, kept for chronology.
- **`analysis/`** — the reasoning and the evidence. `current/` holds the live analyses; `historical/` holds completed reviews and pre-rewrite documents.
- **`status/`** — this document, the [Performance & Accuracy Ledger](PERFORMANCE_LEDGER.md) (the authoritative silicon record), and the [codebase map](CODEBASE_MAP.md).
- **`reference/`** — strategy, vision, and prior art. Rationale only, never a checklist.

The documents that matter for a milestone are linked from its row in §3. This section deliberately does **not** inventory them: the folder listing is one command away, and an inventory is a second place for a link to go stale.

**Historical documents are marked, not deleted.** A replaced plan moves to `superseded/`, an analysis to `historical/`, annotated with what replaced it. Such a document may name targets and files that no longer exist — correct for a record, but **not** for instructions. The codebase map is authoritative for what actually builds.

---

## 3. Progress

One row per milestone, pointing at the document that owns its detail.

### Past

Oldest to newest. As the list grows, the two oldest rows fuse into one, so the top row stays a compaction of the earliest work.

| Milestone | State | Detail |
| :--- | :--- | :--- |
| Phase 0–2: foundations, primitives, three-tier storage stack (oldest rows, fused) | complete | [Phase 0](../execution/completed/PHASE_0_EXECUTION_PLAN.md), [Phase 1](../execution/completed/PHASE_1_EXECUTION_PLAN.md), [Phase 2](../execution/completed/PHASE_2_EXECUTION_PLAN.md), [Warm-tier repair](../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md) |
| Native text path: tokenizer, prompt formatter, EOS-aware generation, CLI | complete | [TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md](../execution/completed/TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md) |
| Specification and oracle harness: Tier 0 research, Step 0 prompt encoding, Tier 1 primitives, Step 3 `hc_head` | verified | [DSV4 Inference Pipeline Plan](../execution/completed/DSV4_INFERENCE_PIPELINE_PLAN.md) §Tier 0–1, §Step 0, §Step 3 |
| Tier 2 — the layer body, the three attention classes, the serial loop (items 16–18) | closed | pipeline plan §Tier 2 |
| Tier 3 — chunked prefill and the long-context lifecycle (items 19–20), real-scale state | done | pipeline plan §Tier 3 |
| Tier 4 — streaming and tiering, state restore (items 21–22) | certified | pipeline plan §Tier 4; [session analysis](../analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md) |
| Graph rewrite: composition P0–P4, the 43-layer text-in/text-out path | acceptance criterion met | [DSV4 Graph Composition Plan](../execution/completed/DSV4_GRAPH_COMPOSITION_PLAN.md) §7; ledger M28 |
| Prefill supply strategy: prompt-length gate, bounded Hot drain, routed bank, Hot-set restore | complete | [PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md](../execution/completed/PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md); ledger M44/M44b |
| Supply-chain hot path: the corridor as a demand-driven pipeline | complete | [SUPPLY_CHAIN_HOT_PATH_EXECUTION_PLAN.md](../execution/completed/SUPPLY_CHAIN_HOT_PATH_EXECUTION_PLAN.md); ledger M42–M46; evidence §8–§19 of [the analysis](../analysis/current/SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md) |

### Present

| Work in progress | State | Detail |
| :--- | :--- | :--- |
| Expert streaming and chunked prefill | in progress | [EXPERT_STREAMING_EXECUTION_PLAN.md](../execution/active/EXPERT_STREAMING_EXECUTION_PLAN.md) — §6 open work: the `W`/`C` window sweep, the prefill body's `~120 ms/prompt-token`, the sub-`E` waved ring, the cold→warm fill path, the placement policy, and prefix reuse. Its prefill supply half is complete: [Prefill Supply Strategy](../execution/completed/PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md); its corridor half is complete: [Supply-Chain Hot Path](../execution/completed/SUPPLY_CHAIN_HOT_PATH_EXECUTION_PLAN.md) |
| Session state and swap | open | [SESSION_STATE_AND_SWAP_ANALYSIS.md](../analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md) |
| Host-memory pressure | open investigation | [HOST_MEMORY_PRESSURE_INVESTIGATION.md](../analysis/current/HOST_MEMORY_PRESSURE_INVESTIGATION.md) |
| Routing profile and placement study | open | [ROUTING_PROFILE_AND_PLACEMENT_STUDY.md](../execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md) |
| Kernel compute path | open | [KERNEL_COMPUTE_PATH_ANALYSIS.md](../analysis/current/KERNEL_COMPUTE_PATH_ANALYSIS.md) — the compute half of "batched prefill" is unbuilt |
| Backend generalization: factory and second backend | open gates | [BACKEND_GENERALIZATION_EXECUTION_PLAN.md](../execution/active/BACKEND_GENERALIZATION_EXECUTION_PLAN.md) |

### Future

Directions explicitly defined as future — researched or discussed, not yet admitted to Present.

| Direction | Why it waits | Detail |
| :--- | :--- | :--- |
| Prefix matcher (block table, cache key, radix search, eviction) | needs an assembled graph and a fork workload | composition plan §9; session analysis |
| MTP / DSpark draft head (`num_nextn_predict_layers=1`) | speculative decoding, not the base forward pass | composition plan §9; pipeline plan §Tier 0.2e |
| Multi-GPU pipeline parallelism (Phase 3) | out of scope for this revision | composition plan §9; [vision](../reference/strategy/PROJECT_AEON_VISION.md) |
| Tool use beyond prompt encoding | frontend work; needs the logit-processor seam plus a tool workload | composition plan §9 |
| KV fp8/E4M3 versus bf16 store | a delta that must be measured, not assumed | pipeline plan Gates 9/10 |
| Expert-placement policy (routing-aware hotlists) | scheduling optimization, not a correctness requirement | expert streaming analysis |
| Throughput targets (chunked-prefill amortization) | speed and correctness are separate gates | expert streaming analysis |

---

## 4. Open gates, settled empirically rather than by reading

Pointers only. Each gate is specified, with its procedure and its result, in the document that owns it.

1. **KV fp8/E4M3 vs bf16 storage delta** — open. (pipeline plan Gates 9/10)
2. **MoE routed-expert accumulation order** — partially settled.
3. ~~Indexer Hadamard rotation~~ — **settled: do not apply it** (pipeline plan Gate 11).
4. ~~Local-window prefix-reuse boundary~~ — **settled at item 20**: the compressed store never evicts inside the declared context, so only the local ring is window-bounded and must be **replayed**, not restored.

**Graph correctness is closed.** Tiers 0–4 and composition P0–P4 are done, and the acceptance criterion is met — one command, conversation in, text out, on the artifact's real weights through Hot/Warm/Cold.

**Kept infrastructure** — cold-tier characterization, physical `.aeon` placement, host-memory pressure, the model-backed `>= 6.0 GB/s` target, kernel occupancy tuning, the routing placement study, the explicit backend factory, and Phase 3 multi-GPU.

---

## 5. Maintenance

Inherited from [AGENTS.md](../../AGENTS.md) §3, *Development Process & Git Conventions* rule 4, because an unbounded progress log stops being read.

- **Update §3 whenever a milestone transitions.** State a milestone as one row and point at the document that owns its detail. When a milestone advances, **change its row — do not append a narrative**. Detail added here is duplication that will drift.
- **Fuse the oldest rows** rather than letting **Past** grow. The top row is a compaction of the earliest work; a new milestone fuses the two rows below it.
- **Keep the document the same size.** A milestone entering **Present** is paid for by a row leaving it, not by the document growing.
- **Do not add a fourth place to record state.** Measurements go in the ledger, the implementation sequence in the plan, progress here, and stable context in AGENTS.md.
- **Update §1's figures** (branch, test count) when they change, rather than leaving them to drift.
- **Do not re-introduce a document inventory.** §2 explains the structure; the folder listing is the inventory.
