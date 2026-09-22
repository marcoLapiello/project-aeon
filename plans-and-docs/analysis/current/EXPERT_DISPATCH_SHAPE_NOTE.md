# Expert Dispatch Shape — the `C = 1` case of a general batch

*Status: open note for [Expert Streaming Execution Plan](../../execution/active/EXPERT_STREAMING_EXECUTION_PLAN.md) **Step 0**. No code. Written 2026-09-21.*

## 1. What this note fixes

The expert dispatch is written for **exactly one token**. The whole path — `on_routing_ready`, `LayerPrefetchState`, `dispatch_layer_prefetch`, the staging arena — carries a literal `6` (one token's routed experts) and a literal `12` (two such batches, double-buffered, in the arena). Those literals are **the `C = 1` case of a batch of `C` tokens**, not facts about the system.

This note writes the general form once, parameterized by `C`, so the correctness gates (Steps 3–5) certify the general shape's `C = 1` instance rather than a shape Step 6 must later unwind. **It decides shapes only and tunes nothing.**

## 2. The parameter

`C` = the number of tokens in one dispatch = **1 in decode, the chunk size in prefill.**

Every decision below reduces to today's behavior at `C = 1`.

## 3. Decisions

### D1 — Dispatch unit

The seam carries a **batch of requests**: `C` tokens × 6 routed experts = `6C` requests, submitted as a set. Today `C = 1`, one batch per layer.

- `on_routing_ready(layer, first_position, ids[C][6], weights[C][6])`
- `accumulate_routed(layer, first_position, inputs[C], weights, accum[C])`

The `C = 1` signatures are today's.

### D2 — Dedup before dispatch

Within a layer's batch, collapse `6C` requests to the layer's **distinct** global expert IDs. Stage each distinct expert **once**; every token that selected it leases the same VRAM slot.

**Binding (§7):** dedup changes **which copy is read**, never the **slot-sum order**. The fixed-order fp32 reduce sums in slot order; dedup must not permute which slot a token's k-th expert occupies. Asserted, not assumed.

### D3 — Lease scope

A lease is a claim on a VRAM slot that excludes it from being chosen as an eviction victim. It grants **no ordering**. The fixed rule: **a lease is held while compute reading that slot may still be in flight.** The earliest safe release is therefore *forced by mechanics*, not chosen by policy:

| Config | Earliest safe release | Why |
| :--- | :--- | :--- |
| **`C = 1` (decode)** | the token boundary | free — sampling's logit read-back is already a compute boundary (today's policy) |
| **`C > 1` (prefill)** | the **layer boundary** | the chunk's rows at a layer are consumed together; the next layer is a compute boundary |

**Chunk-wide leases are infeasible:** 43 layers × up to 256 distinct experts = up to **11,008 leases = the whole model**, leaving nothing evictable. So `C > 1` *must* release per-layer.

Releasing at the layer boundary restores the no-reader-in-flight precondition, which requires a **compute-stream drain there** (as `ensure_pool_headroom` already does). Per-layer release therefore costs ≈43 drains per chunk — a **cost to measure, not assume**.

**Prefill and decode get different handling as a consequence of this rule, not as an added fork:** the same safety rule yields token scope for `C = 1` and layer scope for `C > 1`.

### D4 — Staging sizing

`slots = banks × depth`, with `banks = 2` (double-buffering).

**NOT `2 × 6 × C`.** At `C = 256` that is 3072 slots = **40.5 GiB pinned**, which cannot fit on a 62.62 GiB host that already holds Warm.

- **Ceiling:** the layer's **deduplicated** distinct set (≤256). Dedup is a precondition of the ceiling being finite.
- **Target:** the concurrency depth that saturates `io_uring` (tens of slots); found by the Step 7 sweep.
- Staging holds transfers **in transit**; a completed expert frees its slot as it lands in VRAM.

**Measured, `2026-09-21` (ledger M34): there is no depth to sweep today.** In the single-dispatch path the arena is not a free-list pool — `staging_offset = (layer % 2) * 6` addresses two *fixed* banks by layer parity, so nothing ever waits for a slot. The per-transfer metric named `staging_reuse_wait_ns` is idle time since the slot last freed (mean `26–29 ms` ≈ one layer period), so slots sit idle rather than starve. `depth` only becomes a real parameter once **more than one layer's dispatch overlaps** — which is exactly what Steps 6–7 introduce, and the reason this sizing question is deferred to them rather than resolved now.

**Built for the batch path, `2026-09-22` (ledger M38).** The arena's slot count is now a construction parameter, not the `12` literal: the host sizes it to the **ceiling** `max(12, 6C)` from `AeonRuntimeConfig::prefill_chunk` (`C = 1` keeps the decode shape), and the `io_uring` submission depth is sized the same way because `dispatch()` queues a whole batch's reads before its single `submit_pending_reads()`. A layer-wide batch assigns each distinct expert the staging index of its position in the distinct set, so the slots are structurally collision-free. Reaching the *smaller* target depth above — and the waving that a depth below the distinct count requires — is Step 7.

### D5 — Two independent budgets

The **expert sweep's residency** (one layer's ~255 experts in **VRAM**) and the **staging arena** (**pinned host**) are separate. Never summed. Conflating them produced D4's 40 GiB error.

### D6 — In-flight dispatches

Today exactly one token's dispatch is tracked at a time (`state_`, `staging_in_use_`). General form: **one layer's batch**, tracked per dispatch. Stated explicitly so `C > 1` does not silently require N concurrent dispatch states.

### D7 — Literal audit

Every constant that *is* the `C = 1` case, and its general form:

| Literal | Where | `C = 1` | General form |
| :--- | :--- | :--- | :--- |
| `EXPERTS_PER_HORIZON = 6` | `prefetch_staging.hpp` | 6 | distinct experts in flight; ≤ `6C`, bounded by 256 |
| `NUM_BUFFERS = 2` | `prefetch_staging.hpp` | 2 | 2 (banks), unchanged |
| `TOTAL_STAGING_SLOTS = 2 × 6 = 12` | `prefetch_staging.hpp` | 12 | `banks × depth` (D4); now a runtime ctor parameter `max(12, 6C)` |
| `staging_offset = (layer % 2) * 6` | `v4_expert_supply.hpp` | per-token offset | per-batch distinct-set position (`0…D-1`) in the batch dispatch |
| `std::array<…, 6>` in `LayerPrefetchState` | `v4_expert_supply.hpp` | 6 | `6C` (deduped ≤256) |
| `ROUTED_EXPERTS = 6` | `dispatch_layer_prefetch` | 6 | per-token `k` (unchanged); the request *count* becomes `6C` |
| `staging_in_use_` | `v4_expert_executor.hpp` | one token | one layer's batch |

## 4. Out of scope

The note makes the shapes parameterizable; it does **not** choose among strategies. As of `2026-09-21` two of those are no longer open here — they were decided in Step 6, and are recorded for reference only:

- **Prefill iteration order** (chunk-major vs layer-major) — **decided: layer-major within a window** (Step 6 D-a). This note's `C` is the body chunk; the layer-major span `W` is a separate knob (Step 6 §6b).
- **Warm admission** (approaches A–E) — **decided: A, Warm frozen** (Step 6 D-b).
- **The numeric `depth`** — Step 7's sweep.
- **Prefix reuse** (§6.6) — a product dependency, not this note's.
