# Session State, Residency, and Swap

*Status: analysis. Extracted 2026-09-18 from the graph composition plan. Not started.*

**Subject:** carrying a conversation's *state* across turns without recomputing it — the session
aggregate, a session registry with a residency seam, a cold-tier session store, and the rule that
bounds a reused prefix. It does not cover the prefix **matcher**.

**Why this matters here, and not as a frontend feature.** On tiered hardware, recomputing the full
context every turn is not viable: the dominant cost is re-streaming expert weights off disk, not
attention. Prefix reuse is therefore a **foundational design constraint** of this engine, not an
optimization. It was decided in the [DSV4 Inference Pipeline Plan](../../execution/completed/DSV4_INFERENCE_PIPELINE_PLAN.md)
§6 and is implemented here.

**Provenance.** This document was extracted from
[DSV4_GRAPH_COMPOSITION_PLAN.md](../../execution/completed/DSV4_GRAPH_COMPOSITION_PLAN.md), where it was
the plan's trailing phase ("P7 — session swap") with its four build gaps. The composition plan is the
record of the graph's construction; this document is the record of what was left open when it closed.
The phase name `P7` is retired. The inference pipeline plan §6 owns the state-contract *requirements*
(R1–R6) and is not restated here.

---

## 1. What was actually built, and what is missing

The state **layout contract** and **byte-exact restore** exist; the accounting exists; the session
aggregate is largely written. What has no consumer yet is everything that carries a *whole session*
and moves it.

| Piece | State | Evidence |
| :--- | :--- | :--- |
| Layer state layout — separately addressable, absolute-position addressed | exists | `V4LayerStateLayout` |
| Byte-exact restore (R3) | **certified** | `restore_state` + `tests/test_v4_state_restore.cpp`, 146 checks, 6/6 mutations |
| Session-state accounting | exists | `MemoryBudgetEngine::attention_state_memory()` sums all 43 layers; `evaluate()` binary-searches the largest context that fits |
| Session aggregate | **a lift, not new work** | `V4PipelineStateSnapshot` was already `current_seq_len` + a vector of layer snapshots |
| Non-token inputs in the aggregate | **missing** | thinking mode, reasoning effort, active tool set, response format exist only as `dsv4_prompt_encoder` parameters |
| Session registry + residency seam | **missing** | needs a consumer, which is now the engine |
| Cold-tier session store with a cap | **missing** | |
| R4 — declining a boundary older than the local window | **missing** | needed only once matching exists |

---

## 2. Two mechanisms are conflated under "prefix caching", and only one is needed soon

| | **Session swap** | **Prefix matching** |
| :--- | :--- | :--- |
| Question | "restore session 7" | "find the longest computed prefix of these *new* tokens" |
| Needs | session identity + `restore_state` | hashing, block table, radix search, eviction |
| Pays when | always — a multi-turn conversation, or an agent loop extending its own context | **different** sessions share a prefix: a common system prompt, or a sub-agent forked from a parent |

The near-term goal is **session swap**. Turn `N+1` is turn `N`'s state plus a delta, so the state that
is wanted is known **by identity** — there is nothing to search for, and therefore no key, no block
table and no matching. Matching is *not* worthless (the sub-agent fork is a genuine consumer workload,
where a child's prefix *is* its parent's context), but it is not first, and building a radix tree
before there is a graph to serve would be designing against an assumption.

**R4 is not needed for session swap.** A session resumed at its own last position restores a ring
covering exactly `[N−C+1, N]`, so nothing is stale. R4 fires only for an *edited* prefix or a *matched*
boundary shorter than the entry — both the matcher's.

---

## 3. The work

| Item | What | Where | Gate |
| :--- | :--- | :--- | :--- |
| **Session aggregate + identity** | `current_seq_len` + 43 × `V4LayerStateSnapshot` + the **non-token inputs** (thinking mode, reasoning effort, active tool set, response format). | `core/v4_session.hpp` | R3 at session granularity: snapshot → reset → restore → continue, bit-identical |
| **Session registry + residency seam (VRAM-only first)** | A resident session's state stays in VRAM; an inactive session's may leave. Needs a consumer, which is now the engine. | `core/v4_session.hpp` | two sessions alternating: each continues bit-identically and only one is resident |
| **Cold-tier session store with a GiB cap** | An inactive session's home is NVMe (sector-aligned, so it reuses the `O_DIRECT` path), never warm RAM, which the experts already over-subscribe. | `infrastructure/io/` | round-trip byte-exactness on the `O_DIRECT` path |
| **R4 — declining a boundary older than the local window** | The local ring is not reconstructible; a matched prefix shorter than the entry must replay the last `C` tokens rather than serve a stale ring. | `core/v4_session.hpp` | a boundary outside the window is **refused**, not served |

---

## 4. Constraints this work must respect

Carried in from the inference pipeline plan §6, not re-derived here:

- **The state is four independent, position-addressed pieces per layer**, never one monolith
  accumulated from position 0: the local KV ring, the compressed entries, the compressor partial
  state, and the indexer K cache plus its partial state. Hyper-Connections and the residual stream are
  per-token and recomputed, so they are not cached.
- **A reuse boundary may fall mid-ratio-window**, which is why the compressor partial state must be
  persisted per token.
- **The cache key must include every non-token input that changes the graph** — which is why tool use
  and prefix caching are coupled for us.
- **A resident session's state stays in VRAM**; what leaves is a *whole* inactive session, never a
  slice of a live one. Its home is the cold tier, written on eviction/swap-out rather than per turn.
- **A decode step is not bit-reproducible** unless the deterministic MoE accumulation is used, so any
  byte-exact restore claim must state which accumulation it requires (trap 38).
- **Never clamp a position** (trap 40): a wrapped compressed store is invisible to the kernel's own
  position guard, so `record_position` must refuse an out-of-range position rather than clamp it.

---

## 5. Deferred, and named so it is not an implicit "later"

| Deferred | Why | Revisits when |
| :--- | :--- | :--- |
| The prefix **matcher** (block table, cache key, radix search, eviction) | its parameters are measurements of an assembled graph; session swap needs no key | session swap is green and a fork workload exists |
