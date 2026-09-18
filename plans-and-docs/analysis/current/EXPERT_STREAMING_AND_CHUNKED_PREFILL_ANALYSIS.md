# Expert Streaming and Chunked Prefill Plan

*Status: active. Extracted 2026-09-18 from the graph composition plan, whose build phases it continues.*

**Subject:** the routed-expert path of the live graph — its transfer strategy, and the prefill path
that the graph does not yet have. Two work items:

- **W1 — tiering under miss pressure on the live path.** No new mechanism; the *measurement* and the
  *pressure*. The graph must stream experts while it runs and produce the same logits.
- **W2 — chunked prefill through the host.** A prefill path over the certified chunk body, with the
  expert dispatch widened from one token to a chunk.

**Scope:** the expert supply and dispatch on the assembled graph. This plan does not re-open the
graph's numerics (certified elsewhere), session/prefix state, or the storage layer's own targets.

**Provenance.** This document was extracted from the composition plan's own two trailing work items —
tiering on the live path and chunked prefill — together with the build gaps and the rules those work
items owned. The composition plan is the record of the graph's construction; this plan is the record
of what was left open when it closed. Those work items were named `P5` and `P6` there; the names are
retired, and the work items below are named for what they do. Design has not been expanded here —
that is a later session's work.

---

## 1. Order of work, and why the tiering gate comes first

The question *"should chunked prefill precede the tiering gate?"* has an obvious-sounding answer
that is wrong once the layering is written down.

The premise is correct and is a fact about the model, not a preference: **prefill and decode want
different expert-transfer strategies.** The chunk path in the tree is already evidence of it —
`run_layer_body_chunk` (`core/v4_layer_body_batch.hpp:570-592`) loops `run_layer_body_attention_tail`
per row, so a chunk **batches attention and leaves the MoE serialized per token**. Prefill is
expert-bound (`6 × 14,155,776 B × 43 = 3.65 GB` per token), so that serialization amortizes ~5% of
the work. The strategy must change at W2.

The premise is right; the conclusion does not follow, because the differing strategy sits **above**
the layer W1 certifies, and the two must not be collapsed into the one word "tiering":

| | Phase-independent — **W1** | Phase-dependent — **W2** |
| :--- | :--- | :--- |
| What it is | the **mechanism**: Hot←Warm←Cold promotion, `O_DIRECT` cold reads, LRU demotion, staging recycling, the read/supply overlap | the **strategy**: how many tokens one request set carries, when leases are scoped and released, how slots are sized |
| Where it lives | `TieredExpertSupply`, `ExpertRegistry`, `PrefetchStagingArena`, `V4ExpertSupplyCoordinator` — none of which knows how many tokens are in flight | the executor's members: `state_`/`current_layer_` hold **one** dispatch (`ids.size() == 6`), `leases_` spans a whole token, `ensure_pool_headroom()` is thresholded on one layer's worth, `TOTAL_STAGING_SLOTS = 2 × 6` |
| W2's effect | none — the tiers are the same | the dispatch shape becomes a set of `6C` requests |

**So W2 is not a second delivery path and does not unexercise W1's mechanism.** It is an
optimization of *dispatch granularity* over the same proven mechanism. Three reasons keep W1 first:

1. **The chunk path already runs through W1's mechanism — correctly, just slowly.** Each row hits
   the same seam; the tiering is exercised, merely not under miss pressure. W2 changes the *shape* of
   a request, not the *place* a request is answered.
2. **Attribution.** The composition plan's own rule: a head defect produces plausibly-scaled logits;
   and tiering bugs and numerical bugs have identical symptoms. Building a new batched interface on a
   streaming path whose safety under misses is undemonstrated is two unknowns at once, and the
   failure signature is the lease hazard — **a plausible number, produced from wrong weights, with
   nothing recording that it became wrong**.
3. **Cost.** W1 adds no code; W2 adds an interface. Certifying the simpler request shape first is the
   more attributable experiment, and it is the one that can be run today.

---

## 2. W1 — tiering under miss pressure on the live path

**Build:** nothing new — the host already routes through the supply. What is added is the
*measurement* and the *pressure*: run with `hot_vram_slots` small enough that every layer misses,
and with a warm tier that is not preloaded.

**Gate:** the item-21 properties re-derived **through the graph** rather than one round at a time:
cold reads counted, staging slots returned, `forced_drains()` recorded, `invariants_hold()` at the
end, and the logits **bit-identical** to a run with all experts resident. This is the gate the
inference plan lists as item 21's remainder — streaming while the graph runs.

**Unblocks:** the engine purpose. Before this, the graph is correct but is not the engine.

### 2.1 The gate's claims must be split, or it certifies decode's strategy as *the* strategy

| W1 asserts | Status after W2 |
| :--- | :--- |
| the logits are bit-identical to an all-resident run; `invariants_hold()`; cold reads counted; staging slots returned; `forced_drains()` recorded | **phase-neutral invariants** — W2 must preserve every one |
| leases released at the **token** boundary | **decode-specific** — becomes per-chunk (or per-token-within-chunk); W2 re-derives |
| `TOTAL_STAGING_SLOTS = 12` returned | the *invariant* survives; the **number** is derived from `C` at W2 |
| the "distinct requests" reuse metric (the P2 gate's `767 of 1032`) | **decode-specific** — within-chunk dedup changes what the number *means* |
| one `state_` / `current_layer_` | **decode-specific** — becomes a set of `6C` requests |

The lower four are **decode-specific and named so**: W2 re-derives them rather than "regressing" a
property that was never meant to be phase-general.

### 2.2 One action before the gate is trusted

Write down the phase-parameterized dispatch shape — how `on_routing_ready` / `accumulate_routed` take
a *set* of tokens rather than one, how leases are scoped, how staging is sized from `C` — **before**
W1 certifies the mechanism *through* it, so there is no throwaway strategy to unwind at W2.

### 2.3 Diagnostics owned by this work item

| Work item | What | Where | Gate |
| :--- | :--- | :--- | :--- |
| **The layer-body observer for the new graph** | The body takes an observer; the only production one lived inside the pre-rewrite `V4Pipeline`. The null one runs but leaves no diagnostic path. | `core/v4_graph.hpp` (observer adapter) | none needed; must not perturb the hot path (assert identical output traced vs null) |
| **Expert-timing / telemetry / routing-counter wiring** | Diagnostics only; the substrate exists and is unused by the rewrite. | `core/v4_model_host.hpp` | none required |

---

## 3. W2 — chunked prefill through the host

**Build:** `V4Graph::forward_chunk` over `run_layer_body_chunk`; the batched embedding.

**Gate:** the item-19 equality gate re-run through the new host (`chunk ≡ serial`, exact), then —
separately — throughput, whose blocker was measured false and whose real work is batched
projections.

### 3.1 Reconnaissance — measured 2026-09-17, before the work starts

Everything below was read or computed from the tree at `6b999bd` plus the then-uncommitted budget
audit. Nothing here changes behaviour.

**The rewrite has no prefill path.** `V4Engine::chat` renders the prompt and drives
`text::generate_token_ids`, whose step is `V4Graph::forward_token` — **one token, one position, one
call**. Prompt tokens go through the identical path as generated tokens. The measured TTFT of
`4,487 ms` for an 11-token prompt is `~410 ms × 11`, i.e. serial, and it is not a bug: the chunk
driver is unbuilt and W2 is the work item that builds it.

**Two batch scratch types exist, and the host allocates neither.**

| Type | Cap | Allocated by | Used by |
| :--- | ---: | :--- | :--- |
| `PipelineBatchScratchBuffers` (`v4_pipeline_scratch.hpp:334`) | 16 | `V4Pipeline` (legacy, since deleted) | the pre-rewrite graph only |
| `V4LayerBodyBatchScratch` (`v4_layer_body_batch.hpp:99`) | 16 | **nobody** | the item-19 gate only |

So `M = 16` inside `PipelineScratchBuffers` is a decode-path allocation sized for a batch path the
rewrite does not have yet. The consequence is that the budget's single scratch line is wrong in both
directions:

| | Bytes | MiB |
| :--- | ---: | ---: |
| `PIPELINE_SCRATCH_BYTES` (reserved) | 104,857,600 | 100.00 |
| real decode scratch (`PipelineScratchBuffers`) | 4,903,616 | 4.68 |
| **real batch scratch (16 tokens)** | **11,630,400** | **11.09** |

The reservation over-counts decode by ~95 MiB **and does not cover the prefill scratch at all**.
Two of the batch scratch's eleven MiB are `ffn_norm_act_` and `moe_accum_`, each carrying an extra
`kMPad = 16` multiplier **on top of** `rows = 16` — 2 MiB apiece. W2 must therefore reserve
**two derived lines** (decode + batch), not one literal.

**The staging arena is decode-shaped.** `TOTAL_STAGING_SLOTS = NUM_BUFFERS (2) ×
EXPERTS_PER_HORIZON (6) = 12` — *double-buffer one token's top-6*. A 16-token chunk routes to up to
`16 × 6 = 96` expert requests per layer, which is 8 waves against 12 slots.

**But the arena is not what breaks it, and this is the finding that matters.** The expert seam is
strictly **per token**: `V4RoutedExpertExecutor::accumulate_routed(layer_id, position, expert_input,
expert_weights, moe_accum)` takes *one* token's FFN-norm row and *one* token's six weights, and
`run_layer_body_chunk` loops `for row … run_layer_body_attention_tail(...)`. The executor therefore
sees **six experts at a time**, and `6 ≤ 12`, so the tiering stays correct. What follows is the
opposite of a correctness bug and worse than one: **the chunk batches attention and leaves the MoE
serialized per token.**

That matters because prefill is expert-bound, not attention-bound:

- `6 experts × 14,155,776 B × 43 layers = 3.65 GB` of weights streamed **per token**;
- at the measured `~6.3 GB/s` `O_DIRECT` ([PERFORMANCE_LEDGER](../../status/PERFORMANCE_LEDGER.md) M1)
  that is `≈ 0.58 s/token`;
- the attention half of a token is tens of milliseconds.

So chunked prefill wired the way the seam exists today would amortize roughly **5%** of the work.
The honest statement is that W2 is **not** "call `run_layer_body_chunk`".

**The smaller pieces:** the batched token embedding (decode uses 4 small H2D copies of the row; a
chunk needs a gather over the chunk's ids on the device) and the on-device indexer top-k
(`select_indexer_topk` does one D2H + sync and one H2D + sync per CSA token, which the inference
plan forbids in a prefill; it changes no value, so no equivalence gate can see it, and the target is
**zero syncs**).

### 3.2 What W2 must build — each item is a decision, not a mechanical step

1. **The chunk driver** plus allocating `V4LayerBodyBatchScratch`: the 11.09 MiB that is currently
   unaccounted for in the budget.
2. **A chunk-wide expert dispatch.** The seam needs a batch form (`on_routing_ready` /
   `accumulate_routed` over `C` tokens) so a chunk's `6C` requests are **issued as a set** rather
   than one token at a time. This is an interface change to the thing the composition plan calls
   "the image of one token", and it is the item that actually unlocks prefill throughput.
3. **Sizing the staging arena from the chunk size**, which is a real tradeoff rather than a bigger
   constant: `2 × 6 × C = 192` slots at `C = 16` is **2.72 GB of pinned host RAM**. Slots recycle
   once the payload is resident in VRAM, so 192 is an upper bound rather than a requirement — but
   the current design binds one slot per request **for the request's whole lifetime**, so the
   reservation would approach it. The number must be derived from chunk size and the
   latency/bandwidth product, never hardcoded.
4. **Deduplicating experts within a chunk.** 16 tokens draw ~96 requests from 256 experts, so
   collisions are likely and every hit saves a full 14 MB fetch. This also changes what "distinct
   requests" means for the W1 reuse measurement (the P2 gate measured **767 of 1032** for *decode*).
5. **A deliberate chunk size.** `kMaxTokens = 16` is currently inherited from the workspace, and it
   bounds both the batch scratch and the staging demand. A 1000-token prompt is then 63 chunks with
   no cross-chunk expert reuse and no prefix cache. Raising it is a memory decision (the in-code
   note says so) and it should be made openly rather than inherited.

### 3.3 Open item — the budget's scratch and staging lines are literals

Two constants in `memory_budget.hpp` are not measurements:

- `PIPELINE_SCRATCH_BYTES = 100 MiB`, a fixed literal. It should be derived from
  `PipelineScratchBuffers` (decode) **plus** a batch term for the configured chunk size, so the
  reservation cannot drift from the allocations and so prefill is not silently un-budgeted.
- the staging line writes `12` as a literal instead of `PrefetchStagingArena::TOTAL_STAGING_SLOTS`.
  If `NUM_BUFFERS` ever changed, the budget and the arena's real allocation would disagree with no
  error. It must use the arena's own constant, and after W2, the chunk-derived value.

Both are folded into item 1's work rather than fixed now, because the correct form of the batch term
depends on the chunk size this work item chooses. (The rest of the budget audit landed on 2026-09-17
— uploaded-dense accounting, usable-VRAM planning, and the host cap — and is recorded in
[PERFORMANCE_LEDGER](../../status/PERFORMANCE_LEDGER.md) **M28**, which also carries the
before/after VRAM measurement: **`21.99 GiB → 23.76 GiB`** of the card, `675 → 809` Hot slots.)

---

## 4. Rules that bind this work

| Rule | What it forbids here |
| :--- | :--- |
| **Accumulation order is slot order** | A within-chunk expert dedup that permutes which slot a token's k-th expert occupies: the fixed-order reduce sums in slot order, fp32 is not associative, so W1's bit-identity claim would silently break. This is a correctness constraint, not bookkeeping, and it must be found before W2's design is fixed, not after. |
| **Tiering is mechanism, not strategy** | Holding one request shape as *the* shape: the seam is per-token today and prefill needs a set of `6C`. The phase-parameterized dispatch shape is written before W1's gate, so no strategy is thrown away at W2. |

Carried in from the inference plan, not re-decided here: item 19's throughput half (batched
projections — the alleged chunk cap was measured false), the indexer top-k host round-trip
(countable today, target zero), and streaming under concurrency (W1).

---

## 5. Revisit conditions

| Deferred | Why | Revisits when |
| :--- | :--- | :--- |
| Expert-placement policy (routing-aware hotlists) | a scheduling optimization, not a correctness requirement | W1 is green |
| Throughput targets | speed and correctness are two different gates | W2 |

The KV fp8/E4M3 vs bf16 storage decision is **not** this plan's; it is settled at the inference
plan's KV-precision gates.
