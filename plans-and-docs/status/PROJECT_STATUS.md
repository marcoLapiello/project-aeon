# Project Aeon — Project Status

**Audited: 2026-09-29, on branch `main`.**

This is the **single progress-tracking document**. It owns what is done, what is in flight, what is open, and what is deliberately future. Read it with [AGENTS.md](../../AGENTS.md), which owns the stable context — purpose, external references, and engineering rules.

Numbers do not live here. Measured results belong to the [Performance & Accuracy Ledger](PERFORMANCE_LEDGER.md), design rationale to the analysis document that owns it, and step-by-step implementation to the plan.

---

## 1. Where the project is

Project Aeon runs **DeepSeek-V4-Flash-0731 end-to-end on consumer AMD RDNA3** (`gfx1100`): text in, text out, with the `145.12 GiB` routed-expert store streamed from NVMe through the Warm host tier into VRAM and the `12.71 GiB` dense backbone resident. The acceptance criterion — one command, a conversation in, text out — is met.

It got there by **rebuilding the inference graph** against a verified specification rather than repairing the original. An audit of the runtime against the selected checkpoint found structural errors — a missing Hyper-Connections comb scale, a missing compressor APE term, and HCA layers running indexer selection they do not have — and several tests of the time could not have caught them, because they compared a kernel against an oracle derived from the same helper. **Measurements taken before 2026-09-18 describe that other computation and are invalid**; the ledger records the deletion of the pre-rewrite graph, ~5,600 lines.

| Field | Value |
| :--- | :--- |
| Branch | `main` |
| Default `ctest` | 53 tests |

---

## 2. How the documentation is organised

Four folders under `plans-and-docs/`, one job each:

- **`execution/`** — the plans. `active/` holds plans with open gates the project still depends on; `completed/` holds plans whose gates are met; `superseded/` holds plans replaced by a different approach, kept for chronology.
- **`analysis/`** — the reasoning and the evidence. `current/` holds the live analyses; `historical/` holds completed reviews and pre-rewrite documents.
- **`status/`** — this document, the [Performance & Accuracy Ledger](PERFORMANCE_LEDGER.md) (the authoritative silicon record), and the [codebase map](CODEBASE_MAP.md).
- **`reference/`** — strategy, vision, and prior art. Rationale only, never a checklist.

The documents that matter for a milestone are linked from its row in §3. This section deliberately does **not** inventory them: the folder listing is one command away, and an inventory is a second place for a link to go stale.

**Historical documents are marked, not deleted.** A replaced plan moves to `superseded/`, an analysis to `historical/`, annotated with what replaced it. Such a document may name targets and files that no longer exist — correct for a record, but **not** for instructions. The codebase map is authoritative for what actually builds.

The **four concern groups (G1–G4)** and the **comment-writing rule** are stable context, owned by [AGENTS.md](../../AGENTS.md) §3. They are stated there once and not restated here.

---

## 3. Progress

One row per milestone, pointing at the document that owns its detail.

### Past

Oldest to newest. As the list grows, the two oldest rows fuse into one, so the top row stays a compaction of the earliest work.

| Milestone | State | Detail |
| :--- | :--- | :--- |
| Foundations (oldest rows, fused): build and hardware baseline, the three-tier storage stack, the native text path, and the DeepSeek-V4 specification and oracle harness (implementation order Tiers 0–4, through the layer body, the three attention classes, chunked prefill, the long-context lifecycle, streaming and tiering, and state restore) | complete / verified | [Phase 0](../execution/completed/PHASE_0_EXECUTION_PLAN.md), [Phase 1](../execution/completed/PHASE_1_EXECUTION_PLAN.md), [Phase 2](../execution/completed/PHASE_2_EXECUTION_PLAN.md), [Warm-tier repair](../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md), [native text path](../execution/completed/TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md), [DSV4 Inference Pipeline Plan](../execution/completed/DSV4_INFERENCE_PIPELINE_PLAN.md) — every `[V]` claim cites readable reference code |
| Graph rewrite: composition P0–P4, the 43-layer text-in/text-out path | acceptance criterion met | [DSV4 Graph Composition Plan](../execution/completed/DSV4_GRAPH_COMPOSITION_PLAN.md) §7; ledger M28 |
| Prefill supply strategy: prompt-length gate, bounded Hot drain, routed bank, Hot-set restore | complete | [PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md](../execution/completed/PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md); ledger M44/M44b |
| Supply-chain hot path: the corridor as a demand-driven pipeline | complete | [SUPPLY_CHAIN_HOT_PATH_EXECUTION_PLAN.md](../execution/completed/SUPPLY_CHAIN_HOT_PATH_EXECUTION_PLAN.md); ledger M42–M46; evidence §8–§19 of [the analysis](../analysis/historical/SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md) |
| Structural consolidation: monolith module split, concern relocation into **G1–G4**, and the plan-reference comment pass across the tree | complete | [Module Split Plan](../execution/completed/MONOLITH_MODULE_SPLIT_EXECUTION_PLAN.md), [Relocation Execution Plan](../execution/completed/RELOCATION_EXECUTION_PLAN.md); grouping rationale in [the analysis](../analysis/historical/CONCERN_GROUPING_AND_RELOCATION_ANALYSIS.md). No measurement and no behaviour change; the four groups are now a standing rule ([AGENTS.md](../../AGENTS.md) §3 rule 6) |

### Present

| Work in progress | State | Detail |
| :--- | :--- | :--- |
| Prefill body cost (`~120 ms/prompt-token`) | open — **next investigation** | Linear in the prompt and measured through `aeon_chat`; neither supply nor the registry audit, both of which are accounted for. Must be split into body vs supply before it anchors a target. [KERNEL_COMPUTE_PATH_ANALYSIS.md](../analysis/current/KERNEL_COMPUTE_PATH_ANALYSIS.md) — the compute half of "batched prefill" is unbuilt |
| Expert streaming — residual | open | [EXPERT_STREAMING_EXECUTION_PLAN.md](../execution/active/EXPERT_STREAMING_EXECUTION_PLAN.md) billed deliverables are complete (telemetry, the tier-invariance and starved-pool gates, the demotion-queue A/B, the layer-major sweep), and its supply and corridor halves are in **Past**. What remains is §6: the `W`/`C` window sweep — every run to date is one pass (`W ≥ N`), so the `⌈N/W⌉` regime is unmeasured — and the cold→warm fill path. A **sub-`E` staging ring** is unreached but low value: the depth question is closed (the arena is a budget, not a lever) and it would buy nothing while VRAM residency is the limiter |
| Session state and swap — prefix reuse | open, not started | Every turn re-prefills from token 0. A product requirement rather than an optimization. [SESSION_STATE_AND_SWAP_ANALYSIS.md](../analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md) |
| Host-memory pressure | open investigation | Why a `45 GiB` Warm tier does not finish loading on a `62.62 GiB` host: cause unidentified, the dense-page hypothesis excluded, and the deciding measurement (`--warm-gib 45 --no-warm-preload`, freshly booted) not yet completed. The honest ceiling is `≈40 GiB` Warm. [HOST_MEMORY_PRESSURE_INVESTIGATION.md](../analysis/current/HOST_MEMORY_PRESSURE_INVESTIGATION.md) |
| Routing profile and placement study | Phase 1 complete, Phase 2 scoped | Reuse-distance and Belady-OPT measured (ledger M35/M36); Phase 2 decides whether frequency-informed placement can help. The aggregation library exists; the driver needs adjusting to the rebuilt runtime. [ROUTING_PROFILE_AND_PLACEMENT_STUDY.md](../execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md) |

### Future

Directions explicitly defined as future — researched or discussed, not yet admitted to Present.

| Direction | Why it waits | Detail |
| :--- | :--- | :--- |
| Backend generalization: the factory and a second backend | its remaining stages are **contingent on a second backend existing**, and the plan's own non-goal forbids the shared abstraction before two concrete backends demonstrate it | [BACKEND_GENERALIZATION_EXECUTION_PLAN.md](../execution/active/BACKEND_GENERALIZATION_EXECUTION_PLAN.md) — the descriptor, manifest, backend selection, dense binding and source boundaries are already implemented |
| Prefix matcher (block table, cache key, radix search, eviction) | needs an assembled graph and a fork workload | composition plan §9; session analysis |
| MTP / DSpark draft head (`num_nextn_predict_layers=1`) | speculative decoding, not the base forward pass | composition plan §9; pipeline plan §Tier 0.2e |
| Multi-GPU pipeline parallelism (Phase 3) | out of scope for this revision | composition plan §9; [vision](../reference/strategy/PROJECT_AEON_VISION.md). The one worked analysis is §7 of the [archived prefill/multi-GPU review](../analysis/historical/PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md) — start there if this is picked up |
| Tool use beyond prompt encoding | frontend work; needs the logit-processor seam plus a tool workload | composition plan §9 |
| KV fp8/E4M3 versus bf16 store | a delta that must be measured, not assumed | pipeline plan Gates 9/10 |
| Expert-placement policy (routing-aware hotlists) | scheduling optimization, not a correctness requirement | expert streaming analysis |
| Throughput targets (chunked-prefill amortization) | speed and correctness are separate gates | expert streaming analysis |

---

## 4. Measurement gates, settled empirically rather than by reading

Not everything can be decided by reading code, and these are the questions that cannot. Each is specified, with its procedure and its result, in the document that owns it.

| Gate | State |
| :--- | :--- |
| KV fp8/E4M3 versus bf16 storage delta | open (pipeline plan Gates 9/10) |
| MoE routed-expert accumulation order | partially settled |
| Indexer Hadamard rotation | **settled: do not apply it** (pipeline plan Gate 11) |
| Local-window prefix-reuse boundary | **settled at item 20**: the compressed store never evicts inside the declared context, so only the local ring is window-bounded and must be **replayed**, not restored |

**Correctness is settled; the gates above are measurements.** Tiers 0–4 and composition P0–P4 are certified and the acceptance criterion is met, so nothing on the correctness axis is undecided — what remains open in this project is measurement, performance, and product scope.

Also open, tracked where they belong: cold-tier characterization, physical `.aeon` placement, the model-backed `>= 6.0 GB/s` target, and kernel occupancy tuning.

---

## 5. Maintenance

Inherited from [AGENTS.md](../../AGENTS.md) §3, *Development Process & Git Conventions* rule 4, because an unbounded progress log stops being read.

- **Update §3 whenever a milestone transitions.** State a milestone as one row and point at the document that owns its detail. When a milestone advances, **change its row — do not append a narrative**. Detail added here is duplication that will drift.
- **Fuse the oldest rows** rather than letting **Past** grow. The top row is a compaction of the earliest work; a new milestone fuses the two rows below it.
- **Keep the document the same size.** A milestone entering **Present** is paid for by a row leaving it, not by the document growing.
- **Do not add a fourth place to record state.** Measurements go in the ledger, the implementation sequence in the plan, progress here, and stable context in AGENTS.md.
- **Update §1's figures** (branch, test count) when they change, rather than leaving them to drift.
- **Do not re-introduce a document inventory.** §2 explains the structure; the folder listing is the inventory.
