# Project Aeon — Documentation Status

**Audited: 2026-09-18, on branch `rewrite/graph-v2`.**

This file is the navigation point for project state. It is deliberately short: detailed numbers belong in [PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md), design rationale in the reference documents, and step-by-step implementation in the plan linked below. [AGENTS.md](../../AGENTS.md) carries the engineering rules.

---

## 1. Where the project is

Project Aeon is **rewriting its DeepSeek-V4 inference graph**. The storage, streaming, artifact-format and kernel layers are kept; the graph that composes them is rebuilt from a verified specification — [DSV4 Inference Pipeline Plan](../execution/completed/DSV4_INFERENCE_PIPELINE_PLAN.md), the authority for graph semantics, whose every `[V]` claim cites readable reference code and whose every remaining unknown names the gate that settles it.

**Why.** An audit of the runtime against the selected checkpoint found structural errors in the graph — not tuning gaps: a missing Hyper-Connections comb scale, a missing compressor APE term, and HCA layers running indexer selection they do not have. The tests of the time did not catch them because several compared a kernel against an oracle derived from the same helper (the anti-circularity rule). Measurements taken against that graph describe a different computation and are invalid; the ledger records the deletion.

### Current execution state

The graph is **built and speaks**. Each row points at the document that owns the detail.

| Milestone | State | Detail |
| :--- | :--- | :--- |
| Tier 0 — specification research (phases 0.1–0.2f) | done | plan §Tier 0 |
| Step 0 — prompt encoding | verified, 4/4 golden vectors byte-identical | plan §Step 0 |
| Step 3 — `hc_head` | verified; the last graph op that had no code, no oracle and no gate | plan §Step 3 |
| Tier 1 — oracle harness and all 11 primitives | complete, mutation-tested | plan §Tier 1 |
| Tier 2 — items 16–18: layer body, three attention classes, serial loop | closed | plan §Tier 2 |
| Tier 3 — items 19–20: chunked prefill, long-context lifecycle | done; item 19's throughput half open | plan §Tier 3 |
| Tier 4 — item 21: streaming and tiering | certified | plan §Tier 4 |
| Tier 4 — item 22: state restore (R3) | certified; R4 is the matcher's half | plan §Tier 4; [session analysis](../analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md) |
| Tier 4 — item 23: the generating loop | closed by P0–P4 | [composition plan](../execution/completed/DSV4_GRAPH_COMPOSITION_PLAN.md) §7 |
| P0–P4 — executor seam, head end, 43-layer driver, sampler, text binding | built; acceptance criterion met | composition plan §7 |
| Real-scale state — window 128 and `index_topk` 512 together | certified | plan §Tier 3 |
| Pre-rewrite graph and its gate | deleted 2026-09-18, ~5,600 lines | [CODEBASE_MAP.md](CODEBASE_MAP.md) |

| Field | Value |
| :--- | :--- |
| Branch | `rewrite/graph-v2` (`main` is the pre-rewrite state, untouched) |
| Default `ctest` | 44 tests |

Gate results, tolerances, mutation tallies and their findings belong to the plan and the ledger. This table is a pointer, not a record.

---

## 2. Document map

### Live — guides current work (`analysis/current/`)

| Document | Purpose |
| :--- | :--- |
| [deepseek_v4_flash_architecture.md](../analysis/current/deepseek_v4_flash_architecture.md) | Orientation overview of the model family. Not authoritative — defer to the plan. |
| [EXPERT_STREAMING_AND_CHUNKED_PREFILL_ANALYSIS.md](../analysis/current/EXPERT_STREAMING_AND_CHUNKED_PREFILL_ANALYSIS.md) | **Next work.** Expert transfer on the live path: tiering under miss pressure, then chunked prefill with a chunk-wide expert dispatch. |
| [SESSION_STATE_AND_SWAP_ANALYSIS.md](../analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md) | **Open.** Session aggregate, registry, residency seam, cold-tier store, R4. Session swap before the prefix matcher. |
| [HOST_MEMORY_PRESSURE_INVESTIGATION.md](../analysis/current/HOST_MEMORY_PRESSURE_INVESTIGATION.md) | **Open investigation.** Why a large Warm tier never finishes loading on a `62.62 GiB` host. Records the hypotheses that were **refuted** (so they are not retried), the dense-page release that was kept, and the one measurement that would split the problem. No root cause yet. |

### Execution records (`execution/`)

| Folder | Meaning |
| :--- | :--- |
| `active/` | Plans with open gates that the project still depends on. |
| `completed/` | Plans whose execution gates are complete. |
| `superseded/` | Plans replaced by a different approach; retained for chronology only. |

| Document | State | Purpose |
| :--- | :--- | :--- |
| [DSV4_INFERENCE_PIPELINE_PLAN.md](../execution/completed/DSV4_INFERENCE_PIPELINE_PLAN.md) | Complete | **The specification.** Evidence-tagged, cited step-by-step graph procedure with gates, the earning order, anti-circularity and mutation-testing rules. Authoritative for semantics. |
| [DSV4_GRAPH_COMPOSITION_PLAN.md](../execution/completed/DSV4_GRAPH_COMPOSITION_PLAN.md) | Complete | **The composition.** One ordered text-in/text-out path, every step mapped to the component that implements it, and what does not exist yet. P1–P4 built the graph and met the acceptance criterion; every gap it opened is closed. |
| [PHASE_2_EXECUTION_PLAN.md](../execution/completed/PHASE_2_EXECUTION_PLAN.md) | Complete | Single-GPU 3-tier storage and memory hierarchy: repacking, dynamic budget and pools, async prefetch, NVMe direct I/O. |
| [TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md](../execution/completed/TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md) | Complete | Native text path: tokenizer, prompt formatter, EOS-aware generation, native CLI. |
| [EXPERT_STREAMING_EXECUTION_PLAN.md](../execution/active/EXPERT_STREAMING_EXECUTION_PLAN.md) | **Active** | **Next work.** The routed-expert supply: telemetry, then the tier-invariance and starved-pool gates, then the prefill expert sweep. Supersedes the analysis document as the working document; §6 keeps the prefill supply chain open by design. |
| [ROUTING_PROFILE_AND_PLACEMENT_STUDY.md](../execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md) | Open | Multi-prompt routing profiling: measurement boundary, aggregation, durable output. Driver draft to be adjusted to the rebuilt runtime. Decides whether decode routing concentrates, which the plan's §6.3 turns on. |
| [BACKEND_GENERALIZATION_EXECUTION_PLAN.md](../execution/active/BACKEND_GENERALIZATION_EXECUTION_PLAN.md) | Open | Descriptor-driven artifacts and manifest implemented; factory and second-backend gates remain. |
| [MODEL_CORRECTNESS_EXECUTION_PLAN.md](../execution/superseded/MODEL_CORRECTNESS_EXECUTION_PLAN.md) | **Superseded** | The staged in-place repair approach, replaced by the plan. Retained for chronology. |
| [WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md](../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md) | Complete | Persistent Warm ownership, asynchronous refill, source-tier telemetry. See the [closure report](../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_AB_REPORT.md). |
| [PHASE_0_EXECUTION_PLAN.md](../execution/completed/PHASE_0_EXECUTION_PLAN.md) | Complete | Build, hardware, I/O, overlap, and toy-cache foundations. |
| [PHASE_1_EXECUTION_PLAN.md](../execution/completed/PHASE_1_EXECUTION_PLAN.md) | Complete | Single-GPU runtime gates. Its correctness portion is superseded by the plan. |

### Status

| Document | Purpose |
| :--- | :--- |
| [PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md) | Authoritative silicon record. **Read its banner**: the pre-rewrite model-path entries were **deleted 2026-09-18**, so only component, storage and machine measurements survive in §4 and everything on the rebuilt graph is in §5. |
| [CODEBASE_MAP.md](CODEBASE_MAP.md) | Source-tree map. Its "current engine path" section describes the pre-rewrite runtime and is being superseded as the rewrite lands. |

### Historical and reference (rationale only — not checklists)

`analysis/historical/` holds completed reviews, design analyses, and the pre-rewrite planning documents (`AEON_V4_REVIEW_AND_FIX_*`, the supply-chain and backend analyses, the vLLM reference map, and the llama.cpp prefill analysis). `reference/strategy/` and `reference/prior-art/` hold vision, risk, and external research. None of these are current implementation evidence.

---

## 3. Open gates

Pointers only. Each gate is specified, with its procedure and its result, in the document that owns it.

**Graph correctness is closed.** Tiers 0–4 and the composition plan's P0–P4 are done, and the acceptance criterion is met (one command, conversation in, text out, on the artifact's real weights through Hot/Warm/Cold).

**Open work** — both extracted from the composition plan on 2026-09-18:

- [Expert Streaming and Chunked Prefill](../execution/active/EXPERT_STREAMING_EXECUTION_PLAN.md): the routed-expert supply — telemetry, the tier-invariance and starved-pool gates, the demotion-queue A/B, then the layer-major prefill sweep with a chunk-wide deduplicated expert dispatch, a Warm tier frozen across the prefill, and a layer-ordered draining sweep (**all built**; the sweep's throughput half remains). Carries item 21's concurrency remainder and item 19's throughput half. Its working document supersedes [the analysis](../analysis/current/EXPERT_STREAMING_AND_CHUNKED_PREFILL_ANALYSIS.md), whose §6 (the prefill supply chain) stays deliberately open.
- [Session State and Swap](../analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md): the session aggregate, registry, residency seam, cold-tier store and R4. Session swap precedes the prefix **matcher**, which is deliberately deferred along with MTP and multi-GPU.

**Measurement gates, settled empirically rather than by reading:**

1. **KV fp8/E4M3 vs bf16 storage delta** — open.
2. **MoE routed-expert accumulation order** — partially settled.
3. ~~Indexer Hadamard rotation~~ — **settled: do not apply it** (plan Gate 11).
4. ~~Local-window prefix-reuse boundary~~ — **settled at item 20**: the compressed store never evicts inside the declared context, so only the local ring is window-bounded and must be **replayed**, not restored.

**Kept infrastructure** — cold-tier characterization, physical `.aeon` placement, host-memory pressure, the model-backed `>= 6.0 GB/s` target, kernel occupancy tuning, the routing placement study, the explicit backend factory, and Phase 3 multi-GPU.

**Host-memory pressure** is an active investigation: a `45 GiB` Warm tier does not finish `initialize()` on the `62.62 GiB` host, and releasing the dense mapping's page cache (`13.68 GiB`, verified) did not fix it. The refuted hypotheses, the one kept change and the measurement that would split the problem are in [HOST_MEMORY_PRESSURE_INVESTIGATION.md](../analysis/current/HOST_MEMORY_PRESSURE_INVESTIGATION.md).

---

## 4. Conventions

- **Update this file and `AGENTS.md` §3 whenever a milestone transitions.** They are the two documents a new session should read first.
- **Mark superseded documents explicitly** rather than deleting them; move them to `execution/superseded/` or `analysis/historical/`.
- **Do not add a fourth place to record state.** Measurements go in the ledger, the implementation sequence in the plan, navigation here.
- **Target availability.** The rewrite deleted several pre-rewrite test targets (`bench_full_model`, `test_hot_warm_cold_pipeline`, the `test_v4_stage*` / `attention*` / `layer_state*` families, `test_aeon_moe_fused_w13`, `bench_aeon_moe_fused_w13`). Older plans and ledger evidence lines still name them, which is correct for historical records but **not** for instructions. [CODEBASE_MAP.md](CODEBASE_MAP.md) is authoritative for what actually builds.
