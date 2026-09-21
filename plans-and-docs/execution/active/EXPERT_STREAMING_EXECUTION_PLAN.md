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

The ladder (decode, 23 tokens): **`logical_bytes_from_warm` `17.65 → 25.54 GB`**, **NVMe `19.75 → 11.86 GB` (−40%)**, **drops `979 → 0`**, logits byte-identical. The cost is real and the plan's estimate was low: decode D2H rises `23.54 → 37.36 GB` (`+59%`). At the measured stream rates (NVMe `6.33 GB/s`, D2H ≈`25 GB/s`) that is `−1.25 s` of NVMe against `+0.55 s` of D2H — a net ≈`0.7 s` over 23 tokens, ≈`30 ms/token`, roughly `10%` of the measured decode step. The implied demotion reuse is ≈`57%`, well above the plan's `~25%` break-even.

**Two corrections to record.** (1) The plan said "12 only helps prefill"; at capacity `6` the drops are **0 in both phases** (`queue_depth_max = 6`, i.e. the queue never overflowed), so `12` would add nothing for this workload — the interesting question becomes the *depth* at which it stops being enough, not the phase. (2) The plan's "≈1.2 → 3.5 GB per token" D2H estimate is off by roughly an order of magnitude against this measurement (`1.02 → 1.62 GB/token` decode).

---

### Step 6 — Chunked prefill through the host

**Requirement.** Build the prefill path:

1. **`V4Graph::forward_chunk`** over `run_layer_body_chunk`, plus the batched token embedding (a gather over the chunk's ids on the device, not 4 H2D copies per row).
2. **Allocate `V4LayerBodyBatchScratch`** (≈11 MiB at `C = 16`) — the host allocates neither batch scratch type today.
3. **A chunk-wide expert dispatch**: `on_routing_ready` / `accumulate_routed` over `C` tokens, so a chunk's `6C` requests are issued **as a set**.
4. **Deduplicate within the chunk** — collapse `6C` requests to the layer's distinct expert set.
5. **Size the staging arena from transfer-concurrency depth**, never from `C` and never hardcoded: `banks × depth` (`banks = 2`), with the layer's **deduplicated** distinct set as the ceiling and disk-saturation depth as the target. See the [Step 0 note](../analysis/current/EXPERT_DISPATCH_SHAPE_NOTE.md) and §7.

**Why a large chunk is the lever.** A layer's expert set is fetched once and used by every token in the chunk. For a 1000-token prompt:

| Chunk | Passes | Total expert bytes | At 6.33 GB/s |
| ---: | ---: | ---: | ---: |
| 16 (today's `kMaxTokens`) | 63 | 3.1 TB | 485 s |
| 256 | 4 | 621 GB | 98 s |
| 1024 | 1 | **156 GB** | **25 s** |

**156 GB is the floor** — the entire model read once. Chunk 1024 reaches it for this prompt.

**Attention does not bound the chunk.** Measured from the kernel constants: `DSV4_MAX_ATTENTION_KEYS = 128 + 512 = 640`. 23 of 43 layers are hard-capped; only the HCA class grows, and at `context / 128`. Attention work per chunk is therefore **linear in `C`**, and ≈1000× smaller than the expert transfer. **The real ceiling is the batch scratch and the staging budget.**

**Iteration order is a separate decision, and it is open.** Chunk-major re-fetches every ring once per chunk; layer-major (§6.2) fetches each once. §6.3 prices it and §6.5 gives the fallback rule. This step should not fix the order until that is settled.

**Gate.** The item-19 equality gate re-run through the new host at the chosen chunk size (`chunk ≡ serial`, exact), then — separately — throughput.

**Files.** `core/v4_graph.hpp`, `core/v4_layer_body_batch.hpp`, `core/v4_expert_executor.hpp`, `core/v4_expert_supply.hpp`, `core/memory_budget.hpp`.

---

### Step 7 — Chunk size: sweep, then expose

**Requirement.** No chunk-size target is set in advance. `kMaxTokens = 16` is currently inherited and bounds both the batch scratch and the staging demand. Sweep the size and measure; **expose the winner as a user setting**, so the engine is configurable for other hardware rather than tuned for this GPU.

**Consequence for the budget.** `PIPELINE_SCRATCH_BYTES = 100 MiB` is a literal that is wrong in both directions today: it over-counts decode scratch (4.68 MiB) by ~95 MiB **and does not cover the batch scratch at all**. Replace it with **two derived lines** — decode plus a batch term computed from the configured chunk size — and make the staging line use the arena's own constant.

**Gate.** The sweep's throughput curve, recorded in the ledger; the budget lines track the configured chunk size with no literal.

**Files.** `core/memory_budget.hpp` (derived lines), `tools/aeon_chat.cpp` (the setting).

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
| "Size the staging arena from `C`" | False. Staging is a **pipeline buffer** for transfers in transit, sized from transfer-concurrency depth (`banks × depth`). The chunk's deduplicated distinct set (≤256/layer) is its **ceiling**, not its size: `2 × 6 × C` at `C = 256` is 40.5 GiB pinned, which cannot fit. Dedup is a precondition of the ceiling being finite. |

---

## 6. OPEN — the supply chain during prefill

**This section is deliberately unresolved.** It records the problem, the candidate approaches, and the measurements that would decide between them. It is not a checklist.

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

Slot-equivalents are `residual ÷ 14,155,776`. Against a 779-slot Hot pool, spending a quarter of it (~195 slots) buys layer-major up to **~42,000 tokens**. Beyond that the residual starts dismantling the expert pool, and chunk-major is the fallback.

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

The decision needs no runtime detection, and that is better than detecting on the fly — a mid-prefill failure has no clean recovery. At prefill entry the engine already knows all three inputs: the token count `N`, the residual requirement `N × 64 KB`, and the budget. The choice is one computed branch:

```text
residual_bytes = N × 64 KB  (in place)  or  2N × 64 KB  (ping-pong)
layer-major  if residual_bytes ≤ affordable headroom
chunk-major  otherwise
```

The guard must be **exact**, because the two failure directions are both bad: too aggressive and the allocation fails after work has begun; too conservative and every long prompt silently pays the re-read.

### 6.6 Prefix reuse is a dependency, and the two are complementary

**Today, `N` is the entire conversation, not the new turn.** `V4Engine::chat` renders the full message list, calls `reset_generation_state()`, and re-prefills from token 0 — every turn. So a fifth-turn conversation pays a full-context prefill for the fifth time.

That is not a performance nicety on this engine, it is a product requirement: with a disk-bound supply, re-prefilling 32k tokens per turn is minutes of TTFT. [Session State and Swap](../../analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md) is the work item, and multi-turn agentic use at large context is the target that makes it mandatory rather than optional.

The two are **complementary**: layer-major's cost scales with the *unreused* portion of the prompt, and prefix reuse is exactly what shrinks that portion. With reuse in place, `N` is the new turn's delta — small — and layer-major becomes viable almost unconditionally. Without it, layer-major is limited to the ~42k-token ceiling of §6.3.

### 6.7 The Warm admission approaches

These govern **what Warm holds**, a separate axis from §6.2's **iteration order**.

| Approach | Mechanism | Status |
| :--- | :--- | :--- |
| **A. Sweep releases, bypasses Warm** | the sweep never promotes from or writes to Warm, so Warm survives prefill intact | needs a measurement: bypassing costs NVMe reads the Warm tier could have served |
| **B. Sweep consumes Warm** | the sweep promotes Warm-resident experts like any other request | faster prefill (host RAM ≈25 GB/s vs NVMe ≈6.3 GB/s); drains Warm |
| **C. Admit by decode priority** | as slots free, refill Warm with the experts decode is most likely to route to, rather than round-robin | needs the routing profile; the profile is a separate open study |
| **D. Cold → Warm fill** | read early into Warm, use much later | **not a transfer saving** — Cold → Warm → Hot moves the same bytes as Cold → Hot. It buys only *persistence*. Buildable; the condition is §6.3's — it matters only when the Hot pool cannot hold a ring. |
| **E. Tail refill** | during the last chunk, stop sweeping and refill Warm | **cannot restore Warm** — ≈3000 slots is 40 GB ≈ 6.4 s of NVMe, far longer than the tail. Only a fraction of what frees up is recoverable. |

**A and B are the real trade, and it is roughly balanced:** preserving ~3022 warm slots might save prefill ~18 s; a warm decode start saves ~275 ms per token until Warm is re-established. Which wins is a measurement.

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

---

## 8. Revisit conditions

| Deferred | Revisits when |
| :--- | :--- |
| **Prefill iteration order (layer-major vs chunk-major)** | §6.3's cost and ceiling are measured against a real prompt, and the in-place residual write-back is verified |
| **Cold → Warm fill path** (§6.7 D) | the Hot pool cannot hold a ring — i.e. contexts beyond §6.3's ceiling |
| Expert-placement policy (routing-aware hotlists) | Steps 3–5 are green **and** §6.8 is answered |
| Warm admission by decode priority (approach C) | §6.8 shows routing concentrates |
| Chunk size as a tuned constant | the Step 7 sweep is complete |
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
