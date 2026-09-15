# Project Aeon — Documentation Status

**Audited: 2026-09-15, on branch `rewrite/graph-v2`.**

This file is the navigation point for project state. It is deliberately short:
detailed numbers belong in [PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md), design
rationale in the reference documents, and step-by-step implementation in the plan
linked below. [AGENTS.md](../../AGENTS.md) carries the engineering rules.

---

## 1. Where the project is

Project Aeon is **rewriting its DeepSeek-V4 inference graph**. The storage,
streaming, artifact-format, and kernel layers are kept; the graph that composes
them is being rebuilt from a verified specification.

**Why.** An audit of the existing runtime against the selected checkpoint found
structural errors in the graph — not tuning gaps. Concretely: a missing
Hyper-Connections comb scale, a missing compressor APE term, and HCA layers
running indexer selection they do not have. The existing tests did not catch them
because several compared a kernel against an oracle derived from the same helper
(see the anti-circularity rule in the plan). Measurements taken against that graph
describe a different computation and are marked invalid in the ledger.

**How.** The rewrite is driven by a single document, written with an
evidence-tagging convention and an explicit reference hierarchy, so that every
claim is either cited or flagged as unverified:

> ### ➡️ [Inference Pipeline Plan](../analysis/current/inference_pipeline_plan.md) — the specification

It is the authority for graph semantics. Every `[V]` claim in it cites readable
reference code; every remaining unknown names the gate that settles it.

### Current execution state

| Field | Value |
| :--- | :--- |
| Branch | `rewrite/graph-v2` (`main` is the pre-rewrite state, untouched) |
| Build gate | `AEON_ENABLE_LEGACY_V4_GRAPH` — **OFF by default** |
| Default `ctest` | 29 infrastructure/backend/text/kept-component/Tier-1 tests |
| Legacy `ctest` | 31 (the 29 plus 2 gated parity anchors) |
| Research | Phases 0.1–0.2f complete; the whole forward pass is re-cited |
| Step 0 | **Verified** — the artifact's own encoder is ported and matches all 4 golden vectors byte-for-byte |
| Tier 1 | Certified: **RMSNorm**, **RoPE** (both bases), **MLA Q/KV**, **HC + Sinkhorn** (found a transposed comb index in the plan), **attention + sink + softmax**, **compressor + APE** (both ratio classes), **indexer + top-k** (found the ReLU missing from the kernel), **grouped output projection** (per-group reduction asserted, not assumed), **MoE router** (all four traps shown load-bearing; `tid2eid` checked against the artifact), **routed expert** (format + asymmetric clamp rule, re-checked on a real payload; Gate 14 partially settled). Next: shared expert |

---

## 2. Document map

### Live — guides current work (`analysis/current/`)

| Document | Purpose |
| :--- | :--- |
| [inference_pipeline_plan.md](../analysis/current/inference_pipeline_plan.md) | **The specification.** Evidence-tagged, cited step-by-step graph procedure with gates. Authoritative for semantics. |
| [checkpoint_verification_plan.md](../analysis/current/checkpoint_verification_plan.md) | **Checkpoint & artifact integrity.** Validates the input: structural audit, the prompt-encoder oracle (the artifact ships its own encoder + golden vectors), repack round-trip, streaming integrity. Deliberately does **not** compare against the original FP4 checkpoint — the graph is the goal, and the artifact is swappable. |
| [deepseek_v4_flash_architecture.md](../analysis/current/deepseek_v4_flash_architecture.md) | Orientation overview of the model family. Not authoritative — defer to the plan. |

### Execution records (`execution/`)

| Folder | Meaning |
| :--- | :--- |
| `active/` | Plans with open gates that the project still depends on. |
| `completed/` | Plans whose execution gates are complete. |
| `superseded/` | Plans replaced by a different approach; retained for chronology only. |

| Document | State | Purpose |
| :--- | :--- | :--- |
| [PHASE_2_EXECUTION_PLAN.md](../execution/active/PHASE_2_EXECUTION_PLAN.md) | Paused | Cold-tier, storage-layout, and host-pressure work. Spikes 0–2 complete; Spike 3 has open acceptance gates. |
| [TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md](../execution/active/TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md) | Open | Native text path is implemented and kept. Its attention-correctness gate now belongs to the rewrite. |
| [ROUTING_PROFILE_AND_PLACEMENT_STUDY.md](../execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md) | Open; gated | Routing observer and durable profiler implemented; representative data and evaluation gated by correctness. |
| [BACKEND_GENERALIZATION_EXECUTION_PLAN.md](../execution/active/BACKEND_GENERALIZATION_EXECUTION_PLAN.md) | Open | Descriptor-driven artifacts and manifest implemented; factory and second-backend gates remain. |
| [MODEL_CORRECTNESS_EXECUTION_PLAN.md](../execution/superseded/MODEL_CORRECTNESS_EXECUTION_PLAN.md) | **Superseded** | The staged in-place repair approach, replaced by the plan. Retained for chronology. |
| [WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md](../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md) | Complete | Persistent Warm ownership, asynchronous refill, source-tier telemetry. See the [closure report](../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_AB_REPORT.md). |
| [PHASE_0_EXECUTION_PLAN.md](../execution/completed/PHASE_0_EXECUTION_PLAN.md) | Complete | Build, hardware, I/O, overlap, and toy-cache foundations. |
| [PHASE_1_EXECUTION_PLAN.md](../execution/completed/PHASE_1_EXECUTION_PLAN.md) | Complete | Single-GPU runtime gates. Its correctness portion is superseded by the plan. |

### Status

| Document | Purpose |
| :--- | :--- |
| [PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md) | Authoritative silicon record. **Read its banner before comparing any `E2E` entry** — pre-rewrite model-path measurements are marked invalid. |
| [CODEBASE_MAP.md](CODEBASE_MAP.md) | Source-tree map. Its "current engine path" section describes the pre-rewrite runtime and is being superseded as the rewrite lands. |

### Historical and reference (rationale only — not checklists)

`analysis/historical/` holds completed reviews, design analyses, and the
pre-rewrite planning documents (`AEON_V4_REVIEW_AND_FIX_*`, the supply-chain and
backend analyses, the vLLM reference map, and the llama.cpp prefill analysis).
`reference/strategy/` and `reference/prior-art/` hold vision, risk, and external
research. None of these are current implementation evidence.

---

## 3. Open gates

**Correctness (the active work).** Execute the plan: build the independent oracle
and gate harness first, then the Tier 1 primitives, then the layer body, then
compare identical formatted inputs, intermediate checkpoints, and final logits
against a trusted compatible reference before any placement work.

Four gates are settled empirically, not by reading. **One is now closed; three remain:**

1. ~~Indexer Hadamard rotation — apply or not (must be symmetric over Q/K).~~ **SETTLED at Gate 11: do not apply it.** Two-sided is a no-op to `3.6e-16`; one-sided shifts scores by `1.46`. See 2.4.3.
2. KV fp8/E4M3 vs bf16 storage delta.
3. MoE routed-expert accumulation order.
4. Local-window prefix-reuse boundary behaviour.

**Kept infrastructure.** Cold-tier characterization, physical `.aeon` layout,
host-memory pressure, the model-backed `>= 6.0 GB/s` target, kernel occupancy
tuning, the placement study, the explicit backend factory, and Phase 3 multi-GPU
all remain open and are unaffected by the correctness rewrite.

---

## 4. Conventions

- **Update this file and `AGENTS.md` §3 whenever a milestone transitions.** They
  are the two documents a new session should read first.
- **Mark superseded documents explicitly** rather than deleting them; move them to
  `execution/superseded/` or `analysis/historical/`.
- **Do not add a fourth place to record state.** Measurements go in the ledger,
  the implementation sequence in the plan, navigation here.
- **Target availability.** The rewrite deleted several pre-rewrite test targets
  (`bench_full_model`, `test_hot_warm_cold_pipeline`, the `test_v4_stage*` /
  `attention*` / `layer_state*` families, `test_aeon_moe_fused_w13`,
  `bench_aeon_moe_fused_w13`). Older plans and ledger evidence lines still name
  them, which is correct for historical records but **not** for instructions.
  [CODEBASE_MAP.md](CODEBASE_MAP.md) is authoritative for what actually builds.
