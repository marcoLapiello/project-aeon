# Expert Streaming Execution Plan

*Status: active. Opened 2026-09-19. Continues [Expert Streaming and Chunked Prefill Analysis](../../analysis/current/EXPERT_STREAMING_AND_CHUNKED_PREFILL_ANALYSIS.md), which this plan supersedes as the working document.*

**Subject:** the routed-expert supply on the assembled graph — proving it lossless under pressure, then making it fast in the phase that can actually be made fast.

**What this plan is not.** The graph's numerics are certified elsewhere and are not re-opened here. The storage layer's own targets, session/prefix state, and KV precision are not this plan's.

---

## 1. The goal, in plain English

> **The expert supply must be invisible.** Whether an expert was answered from VRAM, from pinned host memory, or from NVMe must not change a single number. Under pressure it must never lose, duplicate, or leak a slot. Once that holds, it must move as few bytes per token as the hardware allows.

That is four guarantees, each with an instrument that already exists or is named below.

| Guarantee | Plain meaning | Measured by |
| :--- | :--- | :--- |
| **Tier-invariance** | the tier that answered did not change the result | two runs, identical logits |
| **No corruption** | never evict a leased or in-flight slot; never double-book a slot | `registry.invariants_hold()` |
| **No duplication** | one expert, one owner; a repeat request joins the in-flight transfer | request-kind counters |
| **Bounded cost** | know the bytes and the wait before optimizing them | `SupplyTelemetry` |

**Correctness and speed are two gates, and correctness comes first.** A tiering defect and a numerical defect have the same signature: a plausible number, produced from the wrong weights, with nothing recording that it went wrong. The batched dispatch (Step 6) is only safe on a path already proven correct under pressure (Steps 3–5).

---

## 2. Vocabulary

These four words are used precisely. Two of them were confused in earlier drafts.

| Term | Meaning | Cost |
| :--- | :--- | :--- |
| **Fetch** | Cold → Hot. The bytes are read from NVMe into **transient staging**, then copied to VRAM. | 13.5 MiB NVMe read **+** 13.5 MiB H2D |
| **Promotion** | Warm → Hot. The bytes are copied host → device. | 13.5 MiB H2D |
| **Demotion** | Hot → Warm. The bytes are copied device → host. | 13.5 MiB D2H |
| **Release** | the pool forgets an expert. **No copy** — the NVMe file is the canonical copy. | free |
| **Deduplication** | within a chunk, many tokens select the same expert; fetch it **once** and let every token read the resident copy. | saves the duplicate reads |

**Transient staging is not Warm.** A cold payload passes through a pinned host buffer on its way to VRAM, but nothing *owns* it there and the buffer is recycled immediately. This distinction is load-bearing: it is why a cold fetch costs two copies rather than one, and why a cold expert must never be routed through the Warm tier merely because host memory is physically involved.

**Staging is a corridor, and every tier that needs a pinned landing zone walks through it.** It is not cold's passage alone: a Cold read always lands there (the `O_DIRECT` buffer must be pinned), an **unpinned** Warm segment uses it as a bounce buffer, and a demotion may borrow a slot for its D2H. A **pinned** Warm expert is the only traffic that skips it — its Promotion is a single H2D. Pinning is a property of the `HostExpertPool`'s **segment**, fixed at allocation: a segment whose `hipHostMalloc` *and* `hipHostRegister` both failed is unpinned for the process lifetime, and experts demoted into it never become pinned later (`pinned_slot_count()` / `unpinned_slot_count()` measure the split). An unpinned Warm hit still beats Cold — it pays a DDR memcpy (≈1.1 ms) and no disk read — but it loses Warm's advantage, not its advantage over Cold.

**Cold → Warm does not exist.** Warm is filled by exactly two paths: the startup preload, and demotion. Nothing ever promotes a Cold expert into Warm, so a Warm hit always means the expert was resident there before the run reached it.

And two strategies, which are **not** the same thing:

| Strategy | What it does | Verdict |
| :--- | :--- | :--- |
| **Candidate staging** | fetch experts *predicted* for future layers, ranked by a routing profile | **Weak in both phases.** Hiding a cold read needs ≈543 ms of lead; one layer computes in ≈30 ms. The pool holds 3 layers, so a horizon of ≈18 is required — it does not fit. |
| **Expert sweep** | when a chunk touches nearly every expert in a layer, fetch the layer's set once, use it for the whole chunk, then move to the next layer | **Strong in prefill.** Needs no prediction and no lead time: for `C = 256` the set is ~255 of 256 experts, which is known, not guessed. |

---

## 3. The phase split

This is the plan's central structural decision: **prefill and decode get different strategies**, because they have different working sets.

| | Prefill (large chunk) | Decode (batch 1) |
| :--- | :--- | :--- |
| Experts per layer | ~255 of 256 — **known in advance** | 6 of 256 — **unknown until that layer's router runs** |
| Working set | one layer's full set, ~3.44 GB (255 slots) | 258 experts scattered across 43 layers |
| Strategy | **expert sweep** + dedup | on-demand + LRU + demotion |
| On eviction | **release** (no copy) | **demote** (copy to Warm) |
| Why | a swept expert is not reused — every layer needs a different set, and demoting it would copy 3.6 GB per layer for nothing | a demoted expert may be reused within the token stream; demotion preserves "natural selection" |

**Why decode keeps demotion and prefill does not.** The Warm tier is meant to accumulate the experts that get reused. Demotion is what feeds it. In prefill the sweep visits each layer once and never returns, so a demoted expert is dead weight; in decode the same experts recur across tokens, so demotion pays. Measured break-even: a demotion costs ≈0.54 ms and saves ≈2.13 ms of NVMe — it pays whenever more than ~25% of demoted experts are reused before Warm evicts them. (The staging→VRAM copy that ends a fetch is common to both paths, so it cancels out of the comparison.) **That reuse rate is unmeasured, and Step 5 measures it.**

The sizes above, and the dense-versus-streamed split they rest on, are derived and checked in [Appendix A](#appendix-a--model-composition-verified). Two facts from it bear on this section: the routed experts are **92%** of the model, and the dense backbone is **257 MiB per layer** in all 43 layers — including layers 0 and 1, which are dense in the same sense as every other layer and differ only in their attention class.

---

## 4. The steps

Each step states its **requirement**, its **gate**, and its **files**. A step is done when its gate is green and its evidence is recorded in the ledger.

### Step 0 — The dispatch-shape note (no code)

**Requirement.** Write down how the expert seam becomes *batched*: how `on_routing_ready` / `accumulate_routed` take a **set** of tokens rather than one, how leases are scoped, and how the staging arena is sized. One page, in the analysis document's own folder.

**Why.** `TOTAL_STAGING_SLOTS = 2 × 6 = 12`, one dispatch in flight, and per-token leases are decode-shaped facts today. Writing the batched shape down first means the certify-able constants are written as the `C = 1` case of a parameterized form, not as facts that must be unwound at Step 6.

**The findings it must pin** (`C` = tokens per dispatch; `C = 1` reproduces today exactly). The note is [EXPERT_DISPATCH_SHAPE_NOTE.md](../analysis/current/EXPERT_DISPATCH_SHAPE_NOTE.md).

| # | Finding | Question it answers |
| :--- | :--- | :--- |
| **D1** | **Dispatch unit** — a batch of `6C` requests submitted as a set | shape of `on_routing_ready` / `accumulate_routed`; `C = 1` = today |
| **D2** | **Dedup before dispatch** | collapse `6C` to the layer's distinct set; must not permute slot-sum order |
| **D3** | **Lease scope** — forced by the safety rule, not chosen: token boundary at `C = 1`, **layer boundary** at `C > 1`; chunk-wide leases are infeasible (≤11,008 = whole model) | how long a slot stays un-evictable |
| **D4** | **Staging sizing** — `banks × depth`, **not** `2 × 6 × C` (40.5 GiB at `C = 256`); ceiling = deduplicated distinct set, target = disk-saturation depth | how many staging slots |
| **D5** | **Two independent budgets** — sweep residency (VRAM) vs staging (pinned host), never summed | where the "40 GiB" error came from |
| **D6** | **In-flight dispatches** — one layer's batch, tracked per dispatch | how many dispatch states exist |
| **D7** | **Literal audit** — every `C = 1` constant and its general form | what Step 6 must parameterize |

**Gate.** None. This is the one action that must precede Step 3's gate, so no throwaway strategy is certified as *the* strategy.

---

### Step 1 — Turn the telemetry on

**Requirement.** The telemetry is already plumbed — `V4ModelHost` owns it and `TieredExpertSupply` already calls `record_request`, `record_timing`, `observe_occupancy` and `record_demotion_*`. What has **no caller** is the switch. Wire it:

- `AeonRuntimeConfig`: add `supply_telemetry_path` (empty = off) and `run_id`.
- `V4ModelHost::initialize`: call `enable_jsonl` **before** `initialize_experts`; `free()` calls `disable()`.
- `V4Engine`: `set_phase(Prefill)` for prompt tokens, `set_phase(Decode)` for generation, `record_decode_token()` per generated token.
- `tools/aeon_chat.cpp`: restore `--supply-telemetry <path>` and `--run-id <id>`.

**Known limitation, to be recorded in the document.** The Hot and Warm **preload** reads go through `read_experts_direct_blocking`, not through the supply. Those bytes will never appear in the telemetry. The Warmup phase stays empty by construction; only supply-mediated traffic is measurable.

**Gate.** The M28 prompt produces `phase_summary` rows with `request_count > 0`, and `bytes_from_nvme` / `bytes_from_host` move with the tier configuration. The figures are then checked against the derived expectation: **≈240 evictions, ≈154 demotion drops, ≈2.29 GB cold per decode token.**

**Files.** `core/memory_budget.hpp`, `core/v4_model_host.hpp`, `core/v4_engine.hpp`, `tools/aeon_chat.cpp`.

**Status: done** (`2026-09-21`). Sink wired (`AeonRuntimeConfig::supply_telemetry_path` / `run_id`; `V4ModelHost::initialize` opens it before `initialize_experts`, `free()` flushes it; `V4Engine` labels each dispatch's phase and counts decode tokens; `--supply-telemetry` / `--run-id` restored). Gate recorded in the ledger as **M29**: two runs of the M28 prompt (`--warm-gib 0` vs `40`) produce `phase_summary` rows with `request_count > 0`, and the tier bytes move — Warm serves `42%` of decode requests, cutting NVMe bytes `10.57 → 6.14 GB`, with `logical_bytes_from_warm` `0 → 4.43 GB`.

---

### Step 2 — The pressure knob

**Requirement.** Add `AeonRuntimeConfig::max_hot_vram_slots` (0 = derived, the default). Applied in `MemoryBudgetEngine::evaluate` as `min(derived, cap)`, **floored at 6** so the graph stays runnable.

**Why it is needed.** `hot_vram_slots` is derived, never user-set. At context 32768 it is 779, and the emergency valve (`ensure_pool_headroom`) only arms below ~264 slots — so **the drain path is never exercised on the live graph**, and neither is eviction-under-lease-pressure. Step 4's gate asks for exactly that. Without this knob, half the mechanism is uncertified.

**Gate.** `--verbose` reports the capped count, `is_feasible` stays true, and one token completes at a cap of 12. `--max-hot-slots <n>` is the flag.

**Files.** `core/memory_budget.hpp`, `tools/aeon_chat.cpp`.

**Status: done** (`2026-09-21`). `--max-hot-slots <n>` caps the derived pool at `min(derived, n)`, floored at 6; the floored/bounded behavior is pinned in `tests/test_dynamic_expert_pool.cpp` (cap 12 → 12, cap 3 → 6, cap-above-derived → unchanged). Gate recorded in the ledger as **M30**: at `--max-hot-slots 12` the report reads `12 slots`, `is_feasible` holds, and a token completes — and an 11-token prompt reads `40.17 GB` from NVMe against `1.79 GB/token` uncapped, which is the starved regime Step 4's drain gate needs.

---

### Step 3 — The tier-invariance gate

**Requirement.** Two runs, same context, same prompt, both `--greedy`:

| Run | Configuration | Answering tier |
| :--- | :--- | :--- |
| **A** | `--context-size 32768 --warm-gib 0` | Hot and Cold only |
| **B** | `--context-size 32768 --warm-gib 40` | Hot and Warm |

Same context means **the same Hot pool**, so the only thing that differs is *which tier answers the non-Hot misses*. That is the variable under test.

**Why `--greedy`.** It sets temperature 0, so the sampler takes the most likely token instead of drawing from the distribution. That removes the RNG from the comparison entirely: the token sequence becomes a deterministic function of the logits, so assertion (i) *follows from* (ii) rather than being an independent claim. If the tokens still differ, the cause is the logits and not the sampling — which is the attribution this gate needs.

**Assert:** (i) identical generated token ids; (ii) byte-identical fp16 logits at every position; (iii) `registry.invariants_hold()` at the end; (iv) `outstanding_leases() == 0` at the end; (v) `logical_bytes_from_warm` is 0 in A and > 0 in B.

**Corrected claim.** The analysis document states this gate as "logits bit-identical to a run with all experts resident". **That is not realizable**: 11,008 experts × 14,155,776 B ≈ 156 GB. The real claim is the one above — *the tier that answered did not change the number* — which is testable and is what actually matters.

**Deliverable.** `scripts/expert_tier_invariance.sh` and a logits dump (`--dump-logits <path>`) so (ii) is checkable.

**Files.** new script; `tools/aeon_chat.cpp` (dump flag).

**Status: done** (`2026-09-21`). `--dump-logits <path>` writes each position's raw fp16 logits; `scripts/expert_tier_invariance.sh` runs A (Warm `0`) and B (Warm `40`) at context `32768`, both greedy, and asserts (i)–(v). All pass; recorded in the ledger as **M31**. The strongest single result: the logits files are **byte-identical** (`cmp`, 4,654,080 B each), and the **total request counts are identical** across runs — prefill `1521` = `1029` Cold + `492` Warm in B, decode `760` = `437` + `323` — so the only thing that changed is *which tier answered*, not *what was asked*.

---

### Step 4 — The starved-pool gate

**Requirement.** Re-run Run A at `max_hot_vram_slots = 12` (two layers' worth). Every layer now forces a drain.

**Assert:** `forced_drains() > 0`; the staging arena returns to zero in-use slots; the logits still match Run A; `invariants_hold()` passes.

**Gap to close (annotated 2026-09-21).** `forced_drains()` is an executor accessor with **no telemetry field and no CLI path**, so this assertion has no way to read the value from a run today. Step 4 must add either a `forced_drains` field to `SupplyTelemetry`/`phase_summary` or a test-level accessor on `V4ModelHost`, before the gate can be evaluated. Also note the cap floor: at `max_hot_vram_slots = 12` the pool holds exactly two layers' worth and the drain path is reachable (M30), but the earlier plan text said "two layers' worth" while the derived arm point is ~264 slots — the cap is the only way to reach the regime.

**Gap closed** (`2026-09-21`, Step 4 implementation). `V4ModelHost::forced_drains()` and `V4ModelHost::staging_in_use_slots()` were added, and the unconditional `[Invariants]` line now reports `forced_drains` and `staging_in_use` alongside the registry and lease figures — a test-level accessor path, which is the option that did not enlarge `SupplyTelemetry`.

**Why this is the step that matters.** It is the only configuration that exercises the emergency valve, the eviction-under-lease-pressure path, and a newly identified hazard:

> A demotion is asynchronous. If a later layer's router asks for an expert that is **mid-eviction**, the registry cannot serve it. If the D2H has completed it is re-fetched from Warm; if it is **still in flight the request path throws** (`"request-path CPU synchronization is forbidden"`). This is a latent failure mode with no measurement behind it. Step 1's telemetry, and this gate, are what quantify it.

**Measured** (`2026-09-21`, M32). The Warm-enabled starved configuration was run to reach this path: `--max-hot-slots 12 --warm-gib 40`, first the 8-token acceptance prompt, then a `96`-token generation. Neither threw. The second ran `2478` forced drains and thousands of demotion attempts — all `queue_pressure` drops — with `invariants_hold()` true and `staging_in_use=0`. **The path is real (the throw is unconditional at `tiered_expert_supply.hpp:210`) but was not reached at this exposure.**

**Structural reason it is rare** (read from the code, not assumed). The throw needs a request for an expert whose catalog entry is `DEMOTION_PENDING` **and** whose H2D event is not yet complete. A demotion costs ≈`0.54 ms`; `reap_registry_transfers` runs at every layer dispatch **and** inside the `reserve_request` retry loop, while one layer computes in ≈`30 ms`. The in-flight window is therefore roughly `2%` of a layer, and reaping closes it before the next layer asks. The hazard is a **narrow race, not a structural impossibility** — a workload that widens the window (a much smaller Warm pool, or a demotion queue large enough to keep many evictions in flight) could still reach it, and it remains worth a targeted test rather than being declared dead.

**Files.** new script (`scripts/expert_starved_pool.sh`); host accessors `forced_drains()` / `staging_in_use_slots()` plus the extended `[Invariants]` line.

**Status: done** (`2026-09-21`). `scripts/expert_starved_pool.sh` runs the uncapped reference and the `--max-hot-slots 12` starved run, asserting `forced_drains > 0`, `staging_in_use == 0`, byte-identical logits, `invariants_hold()`, and zero leases. Recorded in the ledger as **M32**. All pass: the uncapped run reports `forced_drains=0`, the capped run `forced_drains=378`, `staging_in_use=0`, and the logits are **byte-identical** — so the emergency valve, eviction under lease pressure, and the async-demotion path all ran, and none changed a number. Cost: starved decode reads `2.4×` the NVMe bytes.

---

### Step 5 — The demotion-queue A/B

**Requirement.** `TieredExpertSupply::DEFAULT_DEMOTION_QUEUE_CAPACITY` is **2**. One layer dispatches up to 6 experts and each may evict a victim, with the bookkeeping cleared once per layer. The derived expectation for decode is therefore:

| Per decode token | Count | Bytes |
| :--- | ---: | ---: |
| Evictions (demotion attempts) | ≈240 | — |
| Demoted to Warm | ≈86 | 1.16 GB |
| **Dropped** (`QUEUE_PRESSURE`) | **≈154** | 2.08 GB of lost demotion value |

Two thirds of every eviction is discarded. A/B **2 vs 6** (the exact decode-match is 6; 12 only helps prefill, and Steps 6–7 derive the prefill value from `C`).

**Assert:** `logical_bytes_from_warm` rises; `bytes_from_nvme` falls; `demotion_drops` by reason falls; the logits are unchanged from Run A.

**Cost to weigh.** Raising the queue raises D2H traffic from ≈1.2 GB to ≈3.5 GB per token, on the demotion stream, competing with the downloads. **It pays only if reuse exceeds ~25%** — which is the number this step produces. This is also the best available explanation for M28's anomaly (a 40 GiB Warm tier moved decode less than the run-to-run spread); with a queue of 2, Warm is a near-arbitrary resident sample rather than the naturally selected one.

**Files.** `infrastructure/core/tiered_expert_supply.hpp`, `core/memory_budget.hpp` (config), scripts.

**Status: done** (`2026-09-21`). Added `AeonRuntimeConfig::demotion_queue_capacity` (0 = derived from `enable_warm_refill`, so `--no-warm-refill` still disables demotion) and `--demotion-queue <n>`; the applied value is reported on the `[Invariants]` line. `scripts/expert_demotion_queue_ab.sh` runs capacity 2 vs 6 at `--warm-gib 40`. Recorded in the ledger as **M33**; all four assertions pass.

The ladder (decode, 23 tokens): **`logical_bytes_from_warm` `17.65 → 25.54 GB`**, **NVMe `19.75 → 11.86 GB` (−40%)**, **drops `979 → 0`**, logits byte-identical. Decode Warm requests rose `1247 → 1804` (`+45%`), the warm hit rate `21.0% → 30.4%`. The cost is real and the plan's estimate was low: decode D2H rises `23.54 → 37.36 GB` (`+59%`).

🔶 **But the byte win did not become a speed win.** A realistic `1024`-token non-greedy run at the same config moved `3.40 → 3.22 tok/s` (TTFT `16.25 → 17.32 s`) — a wash inside the run-to-run spread. My initial `+30 ms/token` extrapolation assumed decode is bandwidth-bound; it is not. The cause is `P(any cold)` gating each layer (see §6.10 thesis 1). So the honest statement is: **the queue-6 default removes dropped demotions and cuts NVMe bytes (`−40%`) — a resource/cleanliness win — with no established throughput effect.**

**Two corrections to record.** (1) The plan said "12 only helps prefill"; at capacity `6` the drops are **0 in both phases** (`queue_depth_max = 6`, i.e. the queue never overflowed), so `12` would add nothing for this workload — the interesting question becomes the *depth* at which it stops being enough, not the phase. (2) The plan's "≈1.2 → 3.5 GB per token" D2H estimate is off by roughly an order of magnitude against this measurement (`1.02 → 1.62 GB/token` decode). (3) The card's original "so it pays" conclusion was a bandwidth model, not a measurement; it is corrected in ledger M33 and superseded by §6.10.

---

### Step 6 — Layer-major prefill through the host

**Status: decided (2026-09-21); core built (2026-09-21); item 4 built (2026-09-22); item 5 built (2026-09-22); item 6 built (2026-09-22); item 7 built (2026-09-22); outcome 5 measured (2026-09-22).** The strategy questions this step previously held open — iteration order, Warm policy, residency — are settled by the decisions below. The **window driver is built and outcome 1 is met** (ledger M37): a layer-major pass is byte-identical to serial. **Item 4 and D4 landed together (2026-09-22, ledger M38):** the chunk issues its `6C` routed requests as one deduplicated, layer-wide set through a depth-sized staging arena, and the same byte-exact equality holds through that dispatch (dedup `1909` distinct of `4128` draws). **Item 5 landed (2026-09-22, ledger M39):** the prefill phase freezes Warm — a Warm-resident expert is copied into VRAM without transferring ownership, and Warm's resident set is **identical** before and after a prefill (`291 → 291`) while a control run with the freeze off changes `332` experts. **Item 6 landed (2026-09-22, ledger M40):** prefill is a swept, layer-ordered stream — Hot is **drained** on entry (`0` residents at the switch), the frontier holds whole layer sets in computation order, each layer is bulk-released as it retires, Hot is **empty** on exit, and Warm is untouched across the sweep; the window is still byte-identical to serial, `11008` experts streamed in `43` loads. **Item 7 landed (2026-09-22, ledger M41):** the engine's prefill *is* the window — the prompt is handed to `forward_window` as one unit through a `PromptPrefill` step of `text::generate_token_ids`, so the swept supply is on the production path; the sweep engages only when a window clears its own over-fetch rule (§6c item 6). §6 keeps the discussion that led here; it is no longer the specification for this step.

#### 6a. The decisions (not to be re-litigated)

| # | Decision | Why | Arbiter |
| :--- | :--- | :--- | :--- |
| **D-a** | **Iteration order — layer-major within a bounded window.** For each window of `W` tokens, visit layers 0…42, processing the window's tokens in body chunks of `C ≤ W`. Chunk-major is the degenerate `W = C`. | Fetches each layer's expert set **once per window**, not once per chunk. | `colibri/c/deepseek_v4.c` (segment loop) |
| **D-b** | **Warm is frozen during prefill (policy A).** No promotion, no demotion; a Warm-resident expert is read as a **non-destructive copy** (a shadow residency), and a swept expert's eviction is a **release** (no copy). | Promotion is a **move**: the Warm slot is returned to the free list on completion, so reading Warm during the sweep **destroys** the decode set. Release leaks nothing. | `expert_registry.hpp` completion path; §2 |
| **D-c** | **The registry seam is kept.** The sweep goes through the same `on_routing_ready` / `accumulate_routed` path, batched, with leases released at the **layer boundary** (D3). | Preserves the certified invariants and the dedup-slot-order rule (§7) rather than bypassing them with a second path. | [Step 0 note](../analysis/current/EXPERT_DISPATCH_SHAPE_NOTE.md) D1–D3 |
| **D-d** | **The sweep reuses the Hot pool.** No dedicated prefill bank. | VRAM is binding — dense `12.71 GiB` + KV on a `24 GiB` card. Colibri affords a separate `2.2 GB` transient bank only because its dense set is `6.3 GB`. | [Appendix A](#appendix-a--model-composition-verified) |

> **Naming.** This window is not any of the three existing "segments": not `HostExpertPool::SEGMENT_SLOTS` (host storage), not the attention compressor's two-segment overlap, not a pinned host segment. It is the **layer-major span**. Colibri calls the same quantity a *prefill segment* (`V4_PREFILL_SEGMENT`); "window" is used here only to avoid the collision.

#### 6b. The three knobs, and why they are separate

| Knob | Bounds | Effect of raising it |
| :--- | :--- | :--- |
| **`C`** — body chunk | the batch scratch (`V4LayerBodyBatchScratch`, MiB) | fewer GEMM launches; ~flat in throughput |
| **`W`** — layer-major window | the residual carry (`W × 64 KB`) | fewer sweeps; fewer expert bytes |
| **`N`** — prompt length | — | more sweeps unless `W` grows with it |

The residual is **bounded by the window, not the prompt**, so no VRAM is reserved for a hypothetical carry and there is no pre-flight cliff (superseding §6.5's binary guard):

$$\text{expert bytes} \approx \left\lceil \frac{N}{W} \right\rceil \times 156\,\text{GB} \quad (W \gtrsim 256 = \text{one ring}), \qquad \text{residual} = \min(N, W) \times 64\,\text{KB}$$

A prompt longer than one window is simply `⌈N/W⌉` windows, each re-sweeping. Costs in the table below use the **single-drive** rate (6.33 GB/s); colibri's own figures are on a 2× NVMe mirror and must not be imported directly.

| Configuration (prompt 1000 tokens) | Sweeps | Expert bytes | Time | Residual |
| :--- | ---: | ---: | ---: | ---: |
| chunk-major, `C = 16` (today) | 63 | 3.1 TB | 485 s | 1 MB |
| chunk-major, `C = 256` | 4 | 621 GB | 98 s | 16 MB |
| **layer-major, `W ≥ 1024`** | **1** | **156 GB** | **25 s** | **64 MB** |

**156 GB is the floor** — the whole model read once. `W ≥ N` reaches it. The residual column is the *in-place* carry; until D-e below is verified it is ping-pong, i.e. doubled.

#### 6c. Requirement (the build)

1. **`V4Graph::forward_chunk`** over `run_layer_body_chunk`, plus a batched token embedding (a device-side gather of the chunk's ids, not per-row H2D copies). **[built — `forward_window` + `embed_window`]**
2. **Host allocates `V4LayerBodyBatchScratch`**, sized from `C` — the host allocates neither batch scratch type today. **[built — `ensure_batch_scratch`]**
3. **Windowed layer-major driver:** for each window, for each layer, for each body chunk within the window. **[built — `V4Graph::forward_window` + a host-owned residual carry]**
4. **Chunk-wide expert dispatch:** `on_routing_ready` / `accumulate_routed` over `C` tokens so the `6C` requests are issued **as a set** (D1); dedup to the layer's distinct set (D2) without permuting slot-sum order; leases released at the **layer boundary** (D3). **[built — `2026-09-22`, ledger M38]** The chunk's phase 2 is the split the item needs: attention-and-norm for every token, then router for every token (the selections on the host), then **one** `on_routing_ready_batch`, then MoE for every token. `dispatch_layer_prefetch_batch` deduplicates the `6C` requests to the layer's distinct set, stages each once, and returns a per-token index map; the executor's single `accumulate_routed` resolves each token's `k`-th expert through that map, so the fixed-order reduce still sums in `k` order and dedup changes *which copy is read*, never the order. De-duplication measured at `1909` distinct of `4128` draws over the 16-token window, with `window ≡ serial` still byte-exact.
   > **Item 4 and D4 are one change, not two — measured `2026-09-21`, built `2026-09-22`.** The routing is produced *inside* the attention tail (after attention), so issuing the `6C` requests as a set requires lifting the router out into a phase of its own: attention for every token, then router for every token, then one dispatch, then MoE for every token (the decomposition colibri's `coli_v4_block_window_batch_ref` uses). Attempting that phase split **alone**, with the per-token dispatch, is *worse than not doing it*: it separates `on_routing_ready` from `on_routed_consumed`, and the staging arena — whose slots were addressed by **layer parity, six per layer** — then hands the same six slots to every token at that layer at once. **The phase split, the layer-wide dispatch and the depth-sized free-list arena therefore landed together (2026-09-22).**

   **D4, as built.** The arena's slot count is a construction parameter, not the `NUM_BUFFERS × 6` literal: `PrefetchStagingArena` is resized to the **ceiling** `6C` of the configured prefill chunk (`AeonRuntimeConfig::prefill_chunk`, default `1` = decode's shape, so the certified path is untouched), and the direct reader's `io_uring` submission depth is sized the same way — `dispatch()` queues a whole batch's reads before its single `submit_pending_reads()`, so a queue shallower than `6C × chunks-per-expert` would overflow instead of stream. A layer-wide batch assigns each distinct expert the staging index of its position in the distinct set `0…D-1`, and a chunk that would need more slots than the arena owns is refused at `forward_window` rather than colliding two transfers silently. Sizing to the ceiling means no transfer ever waits for a slot; the **smaller** concurrency depth (D4's `banks × depth` target) and the waving that reaching it requires are the Step 7 sweep.
5. **Warm-frozen policy (D-b):** a prefill-phase switch that disables promotion **and** demotion; eviction is release. Verified by outcome 3. **[built — `2026-09-22`, ledger M39]**

   **The mechanism, and why it is not just "skip Warm".** A promotion is a **move**: `complete_request` frees the source host slot, so the expert leaves Warm. A prefill touches every expert, so a sweep that promoted would empty the tier — the outcome 3 failure mode. A bypass that re-reads Warm-resident experts from **NVMe** preserves the set but forfeits Warm's bandwidth, which the plan's own §6.7 rationale says is only necessary *because* the read would otherwise be destructive.

   So the prefill takes a **non-destructive copy**: `ExpertRegistry` gains a `warm_frozen` mode in which a Warm-resident expert is loaded into VRAM as a **shadow residency** — the catalog entry keeps `owner == WARM_HOST` and its host slot, and the VRAM copy is recorded in a `shadow_vram_slot` field with its own LRU. Eviction under the frozen mode is a **release**: the victim is a shadow copy when one is available, else an ordinary Hot resident, and no demotion is ever attempted. `set_warm_frozen(false)` releases every idle shadow and returns the VRAM to decode's pool; a shadow that is not reached in time is reclaimed by ordinary eviction, so nothing strands.

   This is an **explicit extension of the registry's ownership model**, not a loosening of it: `validate_invariants`'s VRAM bijectivity now admits exactly two owners for a slot — a Hot expert, or a Warm expert's declared shadow — and validates the shadow LRU and the per-expert shadow map against the catalog. The host drives the switch from `set_supply_phase` (`freeze_warm_during_prefill`, on by default per D-b).
6. **The layer-ordered sweep (the residency model).** Prefill and decode are two **different allocation strategies**, so the switch between them is a hard switch, and prefill streams in **layer order** rather than evicting by recency. **[built — `2026-09-22`, ledger M40]**

   **Why not LRU here.** Eviction by recency ranks candidates that will be reused. Inside a window *nothing* is reused — layer `L` is visited once and its whole set then dies at once — so the only correct release is the whole layer and the only correct admission order is layer order. LRU is not a slower way to do this; it is the wrong instrument. (This supersedes the earlier "prefetch `L+1`" phrasing, which described the *in-flight* depth rather than the *residency* policy.)

   **The switch.** `begin_prefill_stream()` drains the Hot pool outright — no decode resident survives, because not one of them is in the plan the sweep follows — and leaves Warm and its LRU ranking untouched for the whole prefill. `end_prefill_stream()` requires Hot to be empty again, which the per-layer release guarantees by construction, and clears the mode so decode resumes on the Warm set the prefill never disturbed.

   **The order.** `V4PrefillSweep` (`core/v4_prefill_sweep.hpp`) holds the frontier. Before layer `L` runs it guarantees `L`'s whole set is resident; after `L` retires it bulk-releases `L` and refills the freed slots with the next unvisited layers **in computation order** until one no longer fits. So the Hot pool is a sliding window over layer sets — `L, L+1, L+2, …` up to capacity, a partial layer at the frontier — and when `L` retires its slots are the room the frontier advances into. The lookahead loads a layer **whole, not a routing prediction**: the router lives inside the body after attention, so `L+1`'s *selection* is unknowable while `L` computes, but its *set* is the whole layer, which is known.

   **Why a whole layer, and why the arena is sized to it.** A swept load is one layer's `experts_per_layer` distinct experts in one batch, so `PrefetchStagingArena` is sized to the layer (and the `io_uring` depth to `experts_per_layer × chunks-per-expert`). Nothing evicts during streaming — allocation is free-list only and release is by layer — so the guard in `ensure_pool_headroom` is a no-op there, and the sweep releases its staging slots itself (`finish_streamed_batch`).

   **Overlap is a separate, bounded concern.** The loads are issued and materialized in layer order; overlapping them with compute is the throughput half (outcome 5), bounded by staging depth (D4). Physics caps it regardless: a layer's set is ≈`3.44 GiB` ≈ `0.54 s` at `6.33 GB/s` against ≈`30 ms` of compute, so prefill is transfer-bound and one layer in flight already saturates the drive. **Measured (2026-09-22, M41): the drive is *not* saturated.** A `103`-token swept prefill streams `11008` experts (`148 GiB`) in `49.2 s` — `3.0 GB/s`, or `2.1 tok/s` — against §6b's `156 GB / 25 s` at the single-drive rate. Loads are serialized with their materialization and with compute; the overlap is still unbuilt, and it is now the measured size of outcome 5's gap.

   **The over-fetch rule — the sweep is engaged per window, not per config.** The sweep fetches a layer **whole**; the layer-major window *without* the sweep fetches each layer's distinct set as its body chunks ask for it, and dedup keeps that set far below the layer while the window is narrow — measured at `44` distinct of `96` draws for a `16`-token window (M38), i.e. `≈46%` of `6W`. Loading `256` for a window that would have asked for `44` is a `5.8x` over-fetch, paid in the currency prefill is bound by. So `V4PrefillSweep::worth` engages the sweep only above `6W ≥ 2 x experts_per_layer` — where the measured distinct set is already `≈235` of `256`, so the whole-layer load is a small and shrinking over-fetch. The rule is derived from the model and the window and carries no tuned constant; a window below it still runs layer-major, deduplicated, per chunk, it simply does not pre-load whole layers. **Measured (M41):** the engine's `103`-token prompt sweeps (`43` loads, `11008` experts) and its `11`-token prompt does not (`0` loads), while both stay byte-identical to serial.

7. **The engine's prefill is the window.** `V4Engine::chat` hands the whole prompt to `V4Graph::forward_window` as one unit and leaves the rest to decode. The generation loop is unchanged — `text::generate_token_ids` gained a `PromptPrefill` step (`[prompt] -> first token`) beside its per-token `TokenStep`, so the loop still owns the EOS stop, the cap and the context limit, and the engine supplies a **prefill mechanism** rather than a second loop. `AeonRuntimeConfig::prefill_window` (`0` = the whole prompt, `--prefill-window`) bounds `W`, so a prompt longer than one window is `⌈N/W⌉` layer-major passes; the body chunk `C` is **derived** from the arena, the Hot pool and the body's row cap, so a starved configuration degrades the chunk instead of colliding transfers. **[built — `2026-09-22`, ledger M41]**
8. **D-e — verify the in-place residual.** §6.3 asserts each token's residual update reads only its own row, so a write-back target is safe (`W` rather than `2W`). **Verified.** The body chains `d_res_in_half = d_res_out_half` per row and reads only that row's own residual; M37 asserts the carry round-trips the window byte-exactly, so the carry is `min(N,W) × 64 KB`, not double.

**Attention does not bound the window.** From the kernel constants, `DSV4_MAX_ATTENTION_KEYS = 640`: 23 of 43 layers are hard-capped and only HCA grows, at `context / 128`. Attention is **linear in `W`** and ≈1000× smaller than the sweep. The window's real ceiling is the residual (VRAM) and the batch scratch.

#### 6d. Gate (outcomes)

| # | Outcome | Instrument |
| :--- | :--- | :--- |
| **1** | **Equality** — item 19 re-run through the new host: `chunk ≡ serial`, exact (greedy token ids identical **and** logits byte-identical). **Met (`2026-09-21`, M37; re-met through the layer-wide dispatch `2026-09-22`, M38):** the window's final logits differ in `0` of `258560` bytes and the residual in `0` of `65536`, at body chunks `16` and `5`; `10` checks, `0` failures. Item 4's dedup is non-vacuous in the same run (`1909` distinct of `4128` draws) and the last dispatch covers all `16` tokens, so the equality is through the batch dispatch and not the per-token one. | `test_v4_prefill_window` |
| **2** | **Tier-invariance preserved** — the M31 comparison still holds on prefill (Hot+Cold vs Hot+Warm → identical logits), proving Warm-frozen changed no number. | `scripts/expert_tier_invariance.sh` |
| **3** | **Warm preserved (the decisive check of D-b)** — Warm's resident set is unchanged across a prefill; the first decode tokens serve from Warm at the pre-prefill rate, with `logical_bytes_from_warm > 0`. **Met (`2026-09-22`, M39):** `291 → 291` resident experts, `0` differing of `291`; the prefill took `51` non-destructive copies (`721,944,576 B` logically from Warm, so it read Warm rather than bypassing to NVMe); leaving the phase released every shadow and left Warm still at `291`; the control window with the freeze off changed `332` experts. `11` checks, `0` failures. | `test_v4_warm_frozen_prefill` |
| **4** | **No leak** — `invariants_hold()`, `outstanding_leases() == 0`, `staging_in_use == 0` at end; prefill `demotion_attempts == 0`, `forced_drains == 0`. | `[Invariants]` line |
| **5** | **Throughput** — prefill tok/s against the measured baseline, and bytes/token against §6b's prediction. **Met (`2026-09-22`, M41):** the engine's `103`-token prompt swept in `49.2 s` (`2.1 tok/s`) streaming `11008` experts (`148 GiB`, `3.0 GB/s`) in `43` loads — §6b's one-sweep floor reached, but at **half** the single-drive rate, because load and compute are still serialized. The `11`-token prompt took the non-swept windowed path, which streams a layer's deduplicated distinct set (`1909` distinct of `4128` draws) rather than the whole layer. | ledger |
| **6** | **The sweep switch** — Hot drained on entry, whole layer sets streamed in computation order, Hot empty on exit, Warm untouched; byte-identical to serial throughout. **Met (`2026-09-22`, M40):** `0` Hot residents at the switch, `11008` experts in `43` loads, frontier `2` layers deep, `0` Hot residents and `0` shadows at the end, Warm `0` differing across the sweep, logits `0` differing of `258560`; `13` checks, `0` failures. | `test_v4_prefill_sweep` |

> **A harness rule this step established.** A gate that reads device buffers must **synchronize the stream first**. The forward paths enqueue on the non-default, non-blocking compute stream, and a plain `hipMemcpy` does not order against it, so a read can return the previous run's buffer. M37's first version did this and reported a fabricated `76%` divergence that cost an investigation and pointed at the wrong component. The body's own `hipStreamSynchronize` sits *before* its final stages, so it does not cover them.

**Files.** `core/v4_graph.hpp`, `core/v4_layer_body_batch.hpp`, `core/v4_expert_executor.hpp`, `core/v4_expert_supply.hpp`, `core/v4_model_host.hpp`, `core/v4_prefill_sweep.hpp`, `core/v4_engine.hpp`, `infrastructure/text/text_generation.hpp`, `core/memory_budget.hpp`, `tools/aeon_chat.cpp`.

**Arbiters.** `aeon-references/colibri/c/deepseek_v4.c` (window/chunk loop), `c/deepseek_v4_bank_pair.h` (double-buffered sweep). Both are DSV4-only paths. Colibri's `~0.35 s/layer` is a **2× NVMe** figure — on one drive it is `~0.7 s/layer` (`≈30 s` per full sweep), which is the rate every time above uses.

---

### Step 7 — Window and chunk: sweep, then expose

**Requirement.** Neither knob carries a target in advance: `C` (body chunk, bounds the scratch) and `W` (layer-major window, bounds the residual) — Step 6 §6b. `kMaxTokens = 16` is inherited and fixes both today. Sweep them and measure, then **expose both as user settings** so the engine is configurable for other hardware rather than tuned for this GPU.

**Expected shape, to confirm or refute.** The window is the lever and the chunk is nearly flat: on colibri, `V4_PREFILL_CHUNK` 128 vs 64 differed by ~3% end to end, while the window count dominated. Aeon's silicon decides.

**Consequence for the budget.** `PIPELINE_SCRATCH_BYTES = 100 MiB` is a literal that is wrong in both directions today: it over-counts decode scratch (4.68 MiB) by ~95 MiB **and does not cover the batch scratch at all**. Replace it with **three derived lines** — decode, a batch term from the configured `C`, and a residual-carry term from the configured `W` — and make the staging line use the arena's own constant.

**Gate.** The sweep's throughput vs `W` and vs `C`, recorded in the ledger; the budget lines track the configured `W` and `C` with no literal.

**Status: partly done (`2026-09-22`, M41).** `W` is a setting (`AeonRuntimeConfig::prefill_window`, `--prefill-window`; `0` = the whole prompt) and `C` is **derived** per call from the staging arena, the Hot pool and the body's row cap — so neither is the inherited `kMaxTokens` literal any more. Still open: the *sweep* of the two (the throughput-vs-`W`/`C` table) and the three derived budget lines that replace `PIPELINE_SCRATCH_BYTES`. Outcome 5's first engine measurement (M41: `W = N = 103`, `C = 16`, `2.1 tok/s`, `3.0 GB/s`) is the baseline that sweep is measured against.

**Files.** `core/memory_budget.hpp` (derived lines), `tools/aeon_chat.cpp` (the settings).

---

### Step 8 — The layer-body observer (diagnostics, deferrable)

**Requirement.** The body takes an observer, but the only production one lived inside the pre-rewrite `V4Pipeline`; the null one runs and leaves no diagnostic path. Write the adapter.

**Why it is last.** Nothing in Steps 1–7 depends on it: the counters come from the supply and the registry, both already instrumented. Pull it forward only when a defect needs per-op attribution.

**Gate.** None required; it must not perturb the hot path — assert identical output traced vs null.

**Files.** `core/v4_graph.hpp`.

---

## 5. Corrections to record

Corrections to the analysis document and the historical supply-chain analysis. These are settled; they are not to be re-litigated.

| Claim | Correction |
| :--- | :--- |
| "W1 adds no code" | False, but small: ≈40 lines to reach the existing telemetry. |
| "Logits bit-identical to an all-resident run" | Not realizable (156 GB). The claim is **tier-invariance** — same logits, different answering tier. |
| "Layer rotation" | Ambiguous name. The operation rotates **experts through the pools** in layer order. Renamed **expert sweep**. |
| Candidate staging | **Weak in both phases.** Needs ≈543 ms of lead against ≈30 ms per layer. The pool covers 3 layers, the horizon needs ≈18. |
| Expert sweep | **Strong in prefill.** Needs no prediction — the set is known. |
| Prefill and Warm | The sweep **consumes** Warm when it promotes a Warm-resident expert. Open — see §6. |
| `enable_warm_refill` | The name lies. It sets the demotion queue to 2 instead of 0. **Nothing proactively demotes a Hot expert.** |
| Demotion queue = 2 | Not a small tightness: ~2/3 of all evictions are dropped, every token. **M33:** at capacity `6` the drops are `0` in both phases; decode NVMe falls `40%` for a D2H cost that nets ≈`30 ms/token` saved. |
| "12 only helps prefill" | False as stated. Capacity `6` already reached `queue_depth_max = 6` with **0** drops in **both** phases, so `12` adds nothing for this workload; the question is the depth at which the queue overtops, not the phase. |
| Demotion D2H "≈1.2 → 3.5 GB per token" | Measured decode D2H is `1.02 GB/token` (q2) → `1.62 GB/token` (q6) — about an order of magnitude below the estimate. |
| `moving_frequency` | Recorded on every activation, **never read**. Victim selection is plain LRU. |
| No Hot slot reclaimable | Not "the pool is full of needed experts". It is the **lease count**: up to 258 slots are leased per token (43 × 6), so a pool below ~264 can have every slot un-evictable. |
| Layer 42 is a Sliding layer | False. It is **CSA** (ratio 4). The class counts are **2 Sliding / 21 CSA / 20 HCA**, not 3/20/20 — the alternating pattern runs to layer 42 inclusive. The implementation and its tests were always right (they read `compress_ratios`); only `deepseek_v4_flash_architecture.md` was wrong, and it is corrected. |
| The dense backbone is the first two layers | False — see [Appendix A](#appendix-a--model-composition-verified). Every one of the 43 layers has a full dense set; only the 256 routed experts per layer stream. |
| "Size the staging arena from `C`" | False. Staging is a **pipeline buffer** for transfers in transit, sized from transfer-concurrency depth (`banks × depth`). The chunk's deduplicated distinct set (≤256/layer) is its **ceiling**, not its size: `2 × 6 × C` at `C = 256` is 40.5 GiB pinned, which cannot fit. Dedup is a precondition of the ceiling being finite. **Built (`2026-09-22`):** the arena is `max(12, 6C)` — the decode shape at `C = 1`, the deduped ceiling above it — and the `io_uring` depth is sized to match; reaching the smaller target depth is Step 7. |
| "The sweep is always the prefill" | False, and it was measured. The sweep fetches a whole layer, so for a window that would have asked for less — a `16`-token window draws `1909` distinct of `4128` (M38), `≈46%` of `6W` — it over-fetches by `5.8x`, in the one currency prefill is bound by. **Corrected (`2026-09-22`, M41):** the sweep is engaged per window, above `6W ≥ 2 x experts_per_layer` (`V4PrefillSweep::worth`); below it the window still runs layer-major with the chunk-wide dedup, it just does not pre-load whole layers. |
| `--dump-logits` covers every position | False since the prompt became one window (M41). The window computes the **last** position's head — that is what makes it a window — so the dump now holds one row per window plus one row per decode token, not one per prompt token. A byte-compare of two runs still covers the same positions in the same order, which is all the tier-invariance gate asks of it (M31 re-verified). |

---

## 6. The supply chain during prefill — the record

**The strategy is decided in [Step 6](#step-6--layer-major-prefill-through-the-host) (D-a–D-d, 2026-09-21).** This section is retained as the evidence that led there, not as the specification. Subsection status:

| § | Subject | Status |
| :--- | :--- | :--- |
| 6.1 | the question | answered — the split is by phase (§3) |
| 6.2 | the re-read / iteration order | **decided** — layer-major within a window (D-a) |
| 6.3 | the residual cost | current, generalized — the carry is bounded by `W`, not `N` (Step 6 §6b) |
| 6.4 | scratch vs residual | current |
| 6.5 | the fallback guard | **superseded** — a bounded window has no cliff (Step 6 §6b) |
| 6.6 | prefix reuse | open, **not deferrable** |
| 6.7 | Warm admission | **decided A** (D-b); B rejected, C–E future |
| 6.8 | does routing concentrate | open — Phase 2 of the routing study |
| 6.9 | what must not be assumed | standing |
| 6.10 | the throughput theses | open — still the leading explanation |

### 6.1 The question

Prefill and decode alternate on every turn. A large prefill touches all 11,008 experts, so it rearranges both pools completely, and decode never gets a long enough run to establish the "natural selection" the Warm tier exists to capture. **The mechanism is sound and the workload never lets it operate.**

### 6.2 The re-read — an iteration-order artifact

The driver loops layers, and a chunk's tokens are processed inside each layer. Call this **chunk-major**. For a prompt longer than one chunk, every ring is therefore re-entered once per chunk, and its experts are re-fetched every time.

| Prompt 1000 tokens, chunk 256 | Ring entries | Expert bytes |
| :--- | ---: | ---: |
| **Chunk-major** (today's structure) | 4 passes × 43 rings = 172 | **592 GB** |
| **Layer-major** | 43 (once each) | **148 GB** |

A ring is 255 experts ≈ 3.44 GB. The **444 GB difference is ~70 s at 6.33 GB/s**, and it exists purely because we return to ring 0 after 43 rings have evicted it.

**Layer-major** swaps the loops: for each layer, process every chunk before moving on. Chunk `c` at layer `L` needs layer `L`'s KV for positions before `start(c)`, which chunks `0..c-1` wrote on this same pass — legal, and position order stays monotonic. **The same fetch count as a full-length chunk, without the full-length chunk's scratch.**

Ragged tails need no special case: the chunk is a batch size inside a layer, not a partition of the prompt. The last iteration simply has fewer rows.

### 6.3 Layer-major: what it costs, and where it stops

Layer-major keeps the **residual stream for every token** alive across all 43 layers, instead of only the chunk in flight. The residual is **fp32 and 4 streams wide**: `4 × 4096 × 4 B = 64 KB` per token.

| Prompt tokens | Residual | In Hot-slot equivalents |
| ---: | ---: | ---: |
| 340 | 22 MB | < 1 slot |
| 4096 | 268 MB | 19 slots |
| 32768 | **2.0 GiB** | **152 slots** |

Slot-equivalents are `residual ÷ 14,155,776`. Against a 779-slot Hot pool, spending a quarter of it (~195 slots) buys a single layer-major pass up to **~42,000 tokens**. **Generalized (Step 6 §6b):** the span is a chosen window `W`, not the prompt — the carry is capped at `W × 64 KB`, and a prompt longer than one window pays `⌈N/W⌉` windows rather than falling back to chunk-major.

**Two buffers, or one.** The existing body pings-pongs `res_in → res_out`, which for a persistent residual means `2N` (4.0 GiB at 32768, 304 slots). Writing back **in place** would halve it to `N` — and in-place is safe by reasoning, because the residual has **no cross-token dependence**: each token's residual update reads only its own row. That is a property to verify, not assume, and it needs the body to accept a write-back target.

### 6.4 The scratch and the residual are different things

They must not be conflated — their lifetimes differ by three orders of magnitude:

| | Scratch (`V4LayerBodyBatchScratch`) | Residual carry |
| :--- | :--- | :--- |
| Holds | per-op temporaries for the rows in flight | the tensor being transformed |
| Lifetime | one op, one layer | the whole prefill |
| Sized from | the row count (chunk size) | the prompt token count |

So the residual is **not** scratch, and cannot be: scratch is recycled per layer, and the residual must survive all 43.

**The scratch already sizes itself from a row count** (`allocate(count)` derives `rows`), so dynamic sizing exists. What is missing is the *accounting*: the host allocates neither batch scratch type today, and `PIPELINE_SCRATCH_BYTES` is a 100 MiB literal that neither covers the batch scratch nor tracks the chunk size. That is Step 7's derived-lines work, and it is independent of layer-major — layer-major only changes the *carry*, not the workspace.

### 6.5 The fallback is pre-flight, not on-the-fly

**Superseded by Step 6 §6b (2026-09-21).** This subsection assumed layer-major over the *whole* prompt, which forced an exact binary guard between layer-major and chunk-major at prefill entry. Bounding the layer-major span to a **window** `W` removes the cliff: the residual is a function of `W`, which is *chosen*, not of `N`, which is *given*, so no allocation can fail after work has begun. The open decision is therefore the window size (Step 7's sweep), not a guard.

*Retained from the original:* the choice still belongs at prefill entry — a mid-prefill failure has no clean recovery — and in-place versus ping-pong still halves or doubles the carry (now `W × 64 KB` vs `2W × 64 KB`).

### 6.6 Prefix reuse is a dependency, and the two are complementary

**Today, `N` is the entire conversation, not the new turn.** `V4Engine::chat` renders the full message list, calls `reset_generation_state()`, and re-prefills from token 0 — every turn. So a fifth-turn conversation pays a full-context prefill for the fifth time.

That is not a performance nicety on this engine, it is a product requirement: with a disk-bound supply, re-prefilling 32k tokens per turn is minutes of TTFT. [Session State and Swap](../../analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md) is the work item, and multi-turn agentic use at large context is the target that makes it mandatory rather than optional.

The two are **complementary**: layer-major's cost scales with the *unreused* portion of the prompt, and prefix reuse is exactly what shrinks that portion. With reuse in place, `N` is the new turn's delta — small — and layer-major becomes viable almost unconditionally. Without it, layer-major is limited to the ~42k-token ceiling of §6.3.

### 6.7 The Warm admission approaches

These govern **what Warm holds**, a separate axis from §6.2's **iteration order**.

| Approach | Mechanism | Status |
| :--- | :--- | :--- |
| **A. Sweep releases, bypasses Warm** | the sweep never promotes from or writes to Warm, so Warm survives prefill intact | **Decided (Step 6 D-b), built `2026-09-22` (M39).** The built mechanism reads Warm **non-destructively** rather than bypassing it to NVMe: a copy that leaves the catalog entry Warm-owned. This is strictly better than the bypass the wording assumed and is what the wording's own rationale endorses ("promotion is a *move*, so reading Warm during the sweep would destroy the decode set" — the *move* is the problem, not the read). Measured: `721,944,576 B` served from Warm across one 16-token prefill at zero NVMe cost. |
| **B. Sweep consumes Warm** | the sweep promotes Warm-resident experts like any other request | **Rejected.** Faster prefill (host RAM ≈25 GB/s vs NVMe ≈6.3 GB/s), but it does not merely *drain* Warm as a cost of speed — it *deletes* the entries decode depends on. One ~18 s prefill saving does not repay `~275 ms × generated tokens` of lost decode warmth. |
| **C. Admit by decode priority** | as slots free, refill Warm with the experts decode is most likely to route to, rather than round-robin | needs the routing profile; the profile is a separate open study |
| **D. Cold → Warm fill** | read early into Warm, use much later | **not a transfer saving** — Cold → Warm → Hot moves the same bytes as Cold → Hot. It buys only *persistence*. Buildable; the condition is §6.3's — it matters only when the Hot pool cannot hold a ring. |
| **E. Tail refill** | during the last chunk, stop sweeping and refill Warm | **cannot restore Warm** — ≈3000 slots is 40 GB ≈ 6.4 s of NVMe, far longer than the tail. Only a fraction of what frees up is recoverable. |

**A and B were the real trade, and it was roughly balanced** — preserving ~3022 warm slots might save prefill ~18 s, while a warm decode start saves ~275 ms per token until Warm is re-established. **Resolved for A** (Step 6 D-b): the asymmetry is that B's cost is charged *per decode token* while its saving is charged *once per prefill*.

**C is the interesting one**, and it is the surviving half of the historical document's rolling idea — with prediction applied to *what to admit to Warm*, not to *what to prefetch*, because only the former has time to act.

### 6.8 The measurement that decides it

**Does decode routing concentrate?** If the same experts recur across tokens, every approach above has something to preserve, and C is worth building. If routing is flat across 11,008 experts, none of them helps, Warm cannot be made useful by any admission policy, and decode's speed comes from coverage alone.

One known headwind: **layers 0–2 use a hash router keyed on token id**, not a learned one, so their routing is flat and non-repeating by construction. Those three layers can never benefit from Warm.

The instrument exists but is unpopulated: `ExpertCatalogEntry::moving_frequency` and `activation_count` are recorded and unused, and `RoutingCounter` collects per-layer phase-specific selection counts but is not wired into the rewrite. The open [Routing Profile and Placement Study](ROUTING_PROFILE_AND_PLACEMENT_STUDY.md) is where this is answered.

### 6.9 What must not be assumed

- That Warm helps decode. M28's evidence is **negative and unexplained** — a 40 GiB Warm tier moved decode less than the run-to-run spread. Step 5 tests one explanation.
- That a bigger Warm pool helps. Coverage is bounded by physics (a 62.62 GiB host), and the tier is pinned and therefore unevictable.
- That candidate staging helps anywhere. §2 argues it does not.
- That layer-major is free. §6.3 gives its cost and its ceiling; neither has been measured.

### 6.10 Two theses on why throughput is flat (added 2026-09-21)

M33 raised Warm service `45%` and cut decode NVMe `40%`, yet a realistic `1024`-token run moved `3.40 → 3.22 tok/s` — a wash. These two theses **are the leading explanation and both are testable**. They also correct an earlier guess in this plan's own analysis that decode is compute-bound: the arithmetic refutes it (≈`44 GFLOP`/token against a `~123 TFLOP/s` card is `<0.2%` of peak; 6 experts × 43 layers ≈ `8.6 GFLOP` of expert work). **Decode is supply-latency-bound.**

**Thesis 1 — the max-of-six gates the layer.** A layer issues 6 concurrent fetches and waits on the *slowest*. So its latency is governed by `P(any cold)`, not the cold *count*:

$$P(\text{any cold}) = 1 - (1 - p)^6$$

At the M33 measured cold-request rates, `p` fell `23.5% → 14.1%`, which moves `P(any cold)` only `80% → 60%`. Feeding a latency model (`L_cold ≈ 2.13 ms`, `L_warm ≈ 0.54 ms`) gives `1.80 → 1.48 ms/layer`, i.e. ≈`5%` of a ≈`290 ms` step — **below the run-to-run spread**. It also predicts the shape M28 and M33 both show: throughput settles at a *median between warm-only and cold-only*, because the vast majority of layers still touch Cold while a minority are fully resident. **Consequence:** shrinking the cold-miss *rate* has sharply diminishing returns; only driving `P(any cold)` toward `0` (coverage) or cutting the cold *latency* itself moves throughput.

**Thesis 2 — global, phase-locked LRU thrashes on a cyclic access pattern.** The pools are global with no per-layer partition, and victim selection is **plain global LRU** — `reserve_vram_destination` scans `hot_vram_lru` from `rbegin()`, and `reserve_warm_destination` does the same on `warm_host_lru`; no layer awareness anywhere (`moving_frequency` is recorded and never read, plan §5). Decode's access pattern is a cycle: `L0₆, L1₆, …, L42₆, L0₆, …`, and **LRU is pessimal for a cycle — it evicts the element next needed.** Hot holds only `779/258 ≈ 3.0` tokens' worth, so it sits *just above* the working set, the worst case for cyclic LRU; Warm (`3022/258 ≈ 11.7`) partly buffers it, which is why hits are `21–30%` rather than `0`. Worse, both pools cycle with the **same period**, so their evictions are in phase and **reinforce** the thrash rather than damping it. The remedy is a frequency-aware (LFU-like) retention that lets "natural selection" place experts by workload — which only pays if routing concentrates. **§6.8 is therefore the deciding input for thesis 2 as well.**

**What to measure, in cost order.**

1. **Per-layer outcome distribution** — the fraction of layer dispatches answered all-Hot / Warm-no-Cold / any-Cold. Directly confirms or kills thesis 1's shape (predicting ≈`80%`/`60%` any-Cold at `p≈23.5%`/`14.1%`). **Measured** (`2026-09-21`, `--diagnostic`, `512`-token run, queue `6`): decode `all_hot=8.4%`, `warm_no_cold=46.3%`, **`has_cold=45.3%`** (`21973` dispatches = `511 × 43`); prefill `10.2% / 28.5% / 61.3%`. **Thesis 1 confirmed in shape:** almost half of decode layers still touch Cold and only `8.4%` are answered entirely from Hot — which is exactly why a `45%` Warm-service rise moved throughput by nothing. (Note measured `has_cold` `45.3%` is *below* the `60%` predicted from the A/B's cold-request rate, and `all_hot` above it — the six selections are not independent draws, so the naive `(1−p)^6` over-predicts cold exposure.)
2. **Routing concentration** (§6.8) — decides whether thesis 2's LFU direction has anything to preserve. **Measured (`2026-09-21`, Phase 1, `512`-token decode, `122640` learned-layer requests).** Ideal-LRU hit rate by capacity: `258 → 31.3%`, **`779 → 60.5%`**, `1558 → 74.0%`, `3022 → 85.7%`, `11008 → 94.7%` (ceiling, `= 1 − compulsory 5.3%`). **Measured Hot hit rate `60.6%` vs ideal-LRU at the same `779` capacity `60.5%` — the current policy already achieves the recency ceiling.** So there is **no implementation headroom**: LRU is not thrashing and is not leaving recency-locality on the table (thesis 2's *mechanism*, as stated, is refuted for a same-capacity recency policy). What remains open is the *policy* question — whether a frequency/future-aware policy beats ideal-LRU. **Phase 1b measured it (`2026-09-21`): yes, there is real headroom.** Belady-OPT (an oracle that knows the future) alongside ideal-LRU:

| Capacity | ideal-LRU | Belady-OPT | gap |
| ---: | ---: | ---: | ---: |
| 258 | 31.3% | 59.8% | +28.5 |
| **779** (Hot) | **60.5%** | **77.4%** | **+16.9** |
| 1558 | 74.0% | 85.9% | +11.9 |
| 3022 (Warm) | 85.7% | 92.1% | +6.4 |
| 11008 | 94.7% | 94.7% | 0 |

So the two statements are both true and must be stated together: **the recency policy has no headroom** (measured = ideal-LRU → LRU is not thrashing, thesis 2's *mechanism* refuted), **but a non-recency policy does** (OPT beats ideal-LRU by ≈17 points at the Hot capacity → thesis 2's *direction* is viable). OPT is an oracle, so `+16.9` is the **maximum** any policy can win at `779`, not the expected win; how much is practically capturable needs the static frequency ranking (Phase 2). Together with thesis 1, a `+16.9`-point cut in Hot misses is the lever that would shrink `P(any cold)`. **Phase 2 is scoped in the [Routing Study §10](./ROUTING_PROFILE_AND_PLACEMENT_STUDY.md#10-phase-2--is-the-headroom-real-and-is-it-capturable):** gate **P2-a** (per-layer skew — is frequency concentrated at all?) and gate **P2-b** (does a ranking generalize to held-out prompts?). The target policy is **dynamic LFU-with-decay** (experts still migrate; only the victim rule changes), not static pinning.
3. **Staging depth** — `TOTAL_STAGING_SLOTS = 12` is exactly `2 layers × 6`, a thin pipeline buffer; the A/B telemetry's dominant term is `staging_reuse_wait`. Raising it (Step 0 D4: size by *concurrency depth*) is independent of both theses and cheap. **Measured (`2026-09-21`) — premise refuted, no live lever.** The arena is **not** a free-list pool: `staging_offset = (layer % 2) * 6` addresses two *fixed* banks by layer parity, so no dispatch ever waits for a slot and depth cannot reduce anything. The metric named `staging_reuse_wait_ns` is *idle* time since the slot last freed (`now − available_since_`), not a wait: it means the mean slot idles **`26–29 ms`** — about one layer period — before reuse. Slots sit idle, not starved. **Depth becomes a lever only when more than one layer's dispatch overlaps** (chunked prefill / prefetch-ahead, Step 6), which is exactly when Step 0 D4's `banks × depth` sizing applies. Deferred to Step 6, not sweepable now.

---

## 7. Rules that bind this work

| Rule | What it forbids |
| :--- | :--- |
| **Accumulation order is slot order** | Any dedup that permutes which slot a token's k-th expert occupies. The fixed-order reduce sums in slot order and fp32 is not associative. Dedup changes *which copy is read*, never the order it is summed in — a property to assert, not assume. |
| **A lease grants no ordering** | Releasing a slot while compute reading it may be in flight. The token boundary is a precondition, not an implementation detail (trap 41). |
| **Never clamp a position** | Trap 40. An out-of-range position is refused, never clamped. |
| **Anti-circularity** | A test must not compare a kernel against an oracle derived from that kernel's own helper. |
| **Tiering is mechanism, not strategy** | Holding one request shape as *the* shape. Step 0 exists to prevent exactly this. |
| **Staging is a pipeline buffer, not a working-set store** | Sizing it from the chunk size or the layer's expert set. It holds transfers **in transit**; the sweep's residency is a separate **VRAM** budget, never summed with it. |
| **Prefill and decode are two strategies, not one parameter** | A resident from one phase crossing into the other, and a recency policy (LRU) applied to prefill. Prefill drains on entry, streams in layer order, and is empty on exit; decode owns recency and demotion. |
| **A shadow residency is legal only while Warm is frozen** | A second ownership existing in decode. Asserted in `invariants_hold()`, not merely intended. |

---

## 8. Revisit conditions

| Deferred | Revisits when |
| :--- | :--- |
| **Layer-major window `W` and body chunk `C`** | the Step 7 sweep completes (the order itself is decided: Step 6 D-a; `W` and `C` are now settings/derived, M41) |
| **In-place residual write-back** (§6.3, Step 6 D-e) | Step 6 verifies no cross-token dependence; until then the carry is ping-pong (`2W × 64 KB`) |
| **Load/compute overlap in the sweep** | the Step 7 sweep has a baseline: M41 measures `3.0 GB/s` against the drive's `6.33 GB/s`, so half the prefill time is available to an overlap that is still unbuilt |
| **Cold → Warm fill path** (§6.7 D) | the Hot pool cannot hold a ring — i.e. contexts beyond §6.3's ceiling |
| Expert-placement policy (routing-aware hotlists) | Steps 3–5 are green **and** §6.8 is answered |
| Warm admission by decode priority (approach C) | §6.8 shows routing concentrates |
| Prefix reuse (session state and swap) | **not deferrable** — a product requirement, and the input that makes §6.3 viable at full context |
| Prefix matcher, MTP, multi-GPU | unchanged from the composition plan |

---

## Appendix A — Model composition (verified)

The premise the supply-chain discussion rests on: **what is resident and what streams.** Every figure below is computed from `V4ModelContract::uploaded_dense_bytes` and the checkpoint's own `config.json`; the total reproduces the recorded `13,643,885,660 B` exactly.

| | Bytes | Size | Lifetime |
| :--- | ---: | ---: | :--- |
| **Dense backbone** (uploaded) | 13,643,885,660 | **12.71 GiB** | VRAM, permanent |
| `embed.weight` | 1,059,061,760 | 0.99 GiB | **host RAM**, read per token |
| **Routed experts** (11,008) | 155,826,782,208 | **145.12 GiB** | streamed from NVMe |

The routed experts are **92% of the model**. The dense backbone is 8.7%.

### Model-level tensors (outside the layer stack)

| Tensor | Dtype / shape | Size |
| :--- | :--- | ---: |
| `head.weight` (LM head) | F16 [129280, 4096] | 1010.0 MiB |
| `hc_head_fn` | F32 [4, 16384] | 0.25 MiB |
| `hc_head_base`, `hc_head_scale` | F32 | 20 B |
| `norm.weight` (final RMSNorm) | F16 [4096] | 8 KiB |
| **subtotal** | | **1,010.25 MiB** |

### Per layer — identical in all 43 layers

| Group | Tensors | Size |
| :--- | :--- | ---: |
| **Attention** | `wq_a` `q_norm` `wq_b` `wkv` `kv_norm` `attn_sink` `wo_a` `wo_b` | 204.0 MiB |
| **Shared expert** | `w1` `w2` `w3` — the always-firing FFN | 48.0 MiB |
| **Hyper-connections + norms** | `hc_attn_*`, `hc_ffn_*`, `attn_norm`, `ffn_norm` | 3.0 MiB |
| **Router** | `ffn.gate.weight` [256, 4096] | 2.0 MiB |
| **per-layer total** | | **257.0 MiB** |

### Per-layer additions, by class

| Class | Layers | Extra tensors | Size each |
| :--- | ---: | :--- | ---: |
| **CSA** (ratio 4) | 21 | compressor ×4 + indexer ×6 | 36.52 MiB |
| **HCA** (ratio 128) | 20 | compressor ×4 | 8.25 MiB |
| **Sliding** (ratio 0) | 2 | none | 0 |
| **Hash router** | 3 (layers 0–2) | `ffn.gate.tid2eid` I64 [129280, 6] | 5.92 MiB |

### The layer schedule

`compress_ratios` has **46** entries; the first 43 describe layers, the last 3 are auxiliary zeros.

| Layers | Class | Count |
| :--- | :--- | ---: |
| 0, 1 | Sliding | 2 |
| 2, 4, 6, … **42** | **CSA** | **21** |
| 3, 5, … 41 | HCA | 20 |

The alternating pattern ends on **CSA at layer 42**, not on a sliding layer. `V4ModelSpec::validate_config` enforces `layer % 2 == 0 ? 4 : 128` for `layer >= 2`.

### The hash router

Layers 0–2 select experts from `ffn.gate.tid2eid`, a `[129280, 6]` table **indexed by token id** — `out_indices[k] = table[token_id][k]`. There is no top-k selection; `ffn.gate.weight` still runs, but only to compute the *weights* of the already-chosen six.

Consequences for the supply: the choice is **independent of context and position**, so the same token always draws the same six experts at that layer, and different tokens draw unrelated ones. That defeats cross-turn reuse for those three layers by construction. (`deepseek_v4_flash_architecture.md` describes these as the only layers without `ffn.gate.bias`, which only layers 3–42 carry.)

### `embed.weight` and `head.weight` are not duplicates

They are distinct `[129280, 4096]` F16 matrices — `tie_word_embeddings` is `false`, and `test_v4_graph_head.cpp` asserts they differ **from the bytes**. `embed.weight` is held host-side deliberately (it is read one row per token, and placing it on the device would cost ~74 Hot slots); the rest of the dense container's page cache is released after upload by `release_dense_pages_except("embed.weight")`.
