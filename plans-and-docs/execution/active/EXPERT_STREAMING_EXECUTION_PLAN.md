# Expert Streaming Execution Plan

*Status: active. Opened 2026-09-19. Supersedes [Expert Streaming and Chunked Prefill Analysis](../../analysis/current/EXPERT_STREAMING_AND_CHUNKED_PREFILL_ANALYSIS.md) as the working document.*

> The prefill supply half of this plan — the prompt-length gate, the bounded drain with Hot-set restore, and the routed bank below the gate — is **complete** in [Prefill Supply Strategy](../completed/PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md) (ledger M44/M44b). What remains open here is §6.

**Subject:** the routed-expert supply on the assembled graph — proving it lossless under pressure, then making it fast in the phase that can be made fast.

**Scope.** The graph's numerics are certified elsewhere and are not re-opened here. The storage layer's own targets, session/prefix state, and KV precision are not this plan's. Measurements are recorded in the [Performance Ledger](../../status/PERFORMANCE_LEDGER.md); this document is the implementation sequence and the open work.

---

## 1. The goal

> **The expert supply must be invisible.** Whether an expert was answered from VRAM, from pinned host memory, or from NVMe must not change a single number. Under pressure it must never lose, duplicate, or leak a slot. Once that holds, it must move as few bytes per token as the hardware allows.

| Guarantee | Plain meaning | Measured by |
| :--- | :--- | :--- |
| **Tier-invariance** | the tier that answered did not change the result | two runs, identical logits |
| **No corruption** | never evict a leased or in-flight slot; never double-book a slot | `registry.invariants_hold()` |
| **No duplication** | one expert, one owner; a repeat request joins the in-flight transfer | request-kind counters |
| **Bounded cost** | know the bytes and the wait before optimizing them | `SupplyTelemetry` |

**Correctness and speed are two gates, and correctness comes first.** A tiering defect and a numerical defect have the same signature — a plausible number produced from the wrong weights, with nothing recording it — so the batched dispatch (Step 6) is only safe on a path already proven correct under pressure (Steps 3–5).

---

## 2. Vocabulary

| Term | Meaning | Cost |
| :--- | :--- | :--- |
| **Fetch** | Cold → Hot. NVMe read into **transient staging**, then H2D into VRAM. | 13.5 MiB read **+** 13.5 MiB H2D |
| **Promotion** | Warm → Hot. Host → device copy. | 13.5 MiB H2D |
| **Demotion** | Hot → Warm. Device → host copy. | 13.5 MiB D2H |
| **Release** | the pool forgets an expert. **No copy** — the NVMe file is canonical. | free |
| **Deduplication** | within a chunk, many tokens select the same expert; fetch it **once** and let every token read the resident copy. | saves the duplicate reads |

**Transient staging is not Warm.** A cold payload passes through a pinned host buffer on its way to VRAM, but nothing *owns* it there and the buffer is recycled immediately — which is why a cold fetch costs two copies, and why a cold expert is never routed through Warm merely because host memory is physically involved.

**Staging is a corridor every tier walks through.** A Cold read always lands there (the `O_DIRECT` buffer must be pinned), an **unpinned** Warm segment uses it as a bounce buffer, and a demotion may borrow a slot. Only a **pinned** Warm expert skips it — its Promotion is a single H2D. Pinning is fixed per `HostExpertPool` segment at allocation; an unpinned Warm hit still beats Cold (a DDR memcpy, ≈1.1 ms, no disk read).

**Cold → Warm does not exist.** Warm is filled by exactly two paths — the startup preload and demotion — so a Warm hit always means the expert was resident before the run reached it.

| Strategy | What it does | Verdict |
| :--- | :--- | :--- |
| **Candidate staging** | fetch experts *predicted* for future layers, ranked by a routing profile | **Weak in both phases.** Hiding a cold read needs far more lead time than one layer's compute, and the pool cannot cover the horizon. |
| **Expert sweep** | when a chunk touches nearly every expert in a layer, fetch the layer's set once, use it for the whole chunk, then move to the next layer | **Strong in prefill, above the prompt-length gate.** No prediction and no lead time — the set is known. Below the gate a blind whole-layer load over-reads the prompt's smaller distinct set, so the **routed bank** supplies instead; see [Prefill Supply Strategy](../completed/PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md). |

---

## 3. The phase split

The plan's central structural decision: **prefill and decode get different strategies**, because they have different working sets.

| | Prefill (large chunk) | Decode (batch 1) |
| :--- | :--- | :--- |
| Experts per layer | ~255 of 256 — **known in advance** | 6 of 256 — **unknown until that layer's router runs** |
| Working set | one layer's full set, ~3.44 GB (255 slots) | 258 experts scattered across 43 layers |
| Strategy | **expert sweep** + dedup | on-demand + LRU + demotion |
| On eviction | **release** (no copy) | **demote** (copy to Warm) |
| Why | a swept expert is not reused — demoting it would copy 3.6 GB per layer for nothing | a demoted expert may be reused within the token stream; demotion feeds Warm's "natural selection" |

The routed experts are **92%** of the model and the dense backbone is **257 MiB per layer in all 43 layers** — including layers 0–1, which are dense in the same sense as every other layer and differ only in attention class. Derived and checked in [Appendix A](#appendix-a--model-composition-verified).

---

## 4. The steps

Each step states its requirement, its gate, and its files. A step is done when its gate is green and its evidence is in the ledger.

### Step 0 — The dispatch-shape note — **done**

**Requirement.** Write down how the expert seam becomes *batched*: how `on_routing_ready` / `accumulate_routed` take a **set** of tokens rather than one, how leases are scoped, and how the staging arena is sized. One page: [EXPERT_DISPATCH_SHAPE_NOTE.md](../../analysis/current/EXPERT_DISPATCH_SHAPE_NOTE.md).

**Why first.** `TOTAL_STAGING_SLOTS = 2 × 6 = 12`, one dispatch in flight, and per-token leases are decode-shaped facts today. Writing the batched shape down first means the certifiable constants are written as the `C = 1` case of a parameterized form, not as facts that must be unwound at Step 6.

**The findings it pins** (`C` = tokens per dispatch; `C = 1` reproduces today exactly):

| # | Finding |
| :--- | :--- |
| **D1** | **Dispatch unit** — a batch of `6C` requests submitted as a set; `C = 1` = today. |
| **D2** | **Dedup before dispatch** — collapse `6C` to the layer's distinct set; must not permute slot-sum order. |
| **D3** | **Lease scope** — forced by the safety rule: token boundary at `C = 1`, **layer boundary** at `C > 1`. |
| **D4** | **Staging sizing** — `banks × depth`, **not** `2 × 6 × C`; ceiling = the deduplicated distinct set, target = disk-saturation depth. |
| **D5** | **Two independent budgets** — sweep residency (VRAM) vs staging (pinned host), never summed. |
| **D6** | **In-flight dispatches** — one layer's batch, tracked per dispatch. |
| **D7** | **Literal audit** — every `C = 1` constant and its general form. |

**Gate.** None. This is the one action that must precede Step 3's gate, so no throwaway strategy is certified as *the* strategy.

---

### Step 1 — Turn the telemetry on — **done**

**Requirement.** The telemetry is already plumbed (`V4ModelHost` owns it; `TieredExpertSupply` calls `record_request`, `record_timing`, `observe_occupancy`, `record_demotion_*`). What had no caller was the switch: `AeonRuntimeConfig::supply_telemetry_path` / `run_id`; `V4ModelHost::initialize` opens it **before** `initialize_experts` and `free()` flushes it; `V4Engine` labels each dispatch's phase and counts decode tokens; `--supply-telemetry` / `--run-id` on the CLI.

**Known limitation.** The Hot and Warm **preload** reads go through `read_experts_direct_blocking`, not through the supply, so those bytes never appear in the telemetry and the Warmup phase is empty by construction. Only supply-mediated traffic is measurable.

**Gate.** The acceptance prompt produces `phase_summary` rows with `request_count > 0`, and `bytes_from_nvme` / `bytes_from_host` move with the tier configuration.

**Files.** `core/memory_budget.hpp`, `core/v4_model_host.hpp`, `core/v4_engine.hpp`, `tools/aeon_chat.cpp`.

---

### Step 2 — The pressure knob — **done**

**Requirement.** Add `AeonRuntimeConfig::max_hot_vram_slots` (0 = derived, the default), applied in `MemoryBudgetEngine::evaluate` as `min(derived, cap)`, **floored at 6** so the graph stays runnable. `--max-hot-slots <n>`.

**Why it is needed.** `hot_vram_slots` is derived, never user-set, and the emergency valve (`ensure_pool_headroom`) only arms below ~264 slots — so the drain path is never exercised on the live graph. Step 4's gate asks for exactly that; without this knob, half the mechanism is uncertified.

**Gate.** `--verbose` reports the capped count, `is_feasible` stays true, and one token completes at a cap of 12.

**Files.** `core/memory_budget.hpp`, `tools/aeon_chat.cpp`; behavior pinned in `tests/test_dynamic_expert_pool.cpp`.

---

### Step 3 — The tier-invariance gate — **done**

**Requirement.** Two runs, same context (`32768`), same prompt, both `--greedy`:

| Run | Configuration | Answering tier |
| :--- | :--- | :--- |
| **A** | `--warm-gib 0` | Hot and Cold only |
| **B** | `--warm-gib 40` | Hot and Warm |

Same context means the same Hot pool, so the only thing that differs is *which tier answers the non-Hot misses* — the variable under test. `--greedy` removes the RNG, so identical tokens follow from identical logits rather than being an independent claim.

**Assert:** (i) identical generated token ids; (ii) byte-identical fp16 logits at every position; (iii) `registry.invariants_hold()`; (iv) `outstanding_leases() == 0`; (v) `logical_bytes_from_warm` is 0 in A and > 0 in B.

**Corrected claim.** "Logits bit-identical to a run with all experts resident" is not realizable (11,008 × 14,155,776 B ≈ 156 GB). The real, testable claim is the one above: *the tier that answered did not change the number*.

**Deliverable.** `scripts/expert_tier_invariance.sh` + `--dump-logits <path>`.

**Files.** new script; `tools/aeon_chat.cpp` (dump flag).

---

### Step 4 — The starved-pool gate — **done**

**Requirement.** Re-run Run A at `max_hot_vram_slots = 12`, so every layer forces a drain.

**Assert:** `forced_drains() > 0`; the staging arena returns to zero in-use slots; the logits still match Run A; `invariants_hold()`.

**Exposure added.** `V4ModelHost::forced_drains()` / `staging_in_use_slots()`, reported on the unconditional `[Invariants]` line.

**Why this is the step that matters.** It is the only configuration that exercises the emergency valve, eviction under lease pressure, and a latent hazard: *a demotion is asynchronous, and a router that asks for an expert mid-eviction whose H2D is not yet complete makes the request path throw.* The hazard is a **narrow race** (the reaper runs at every layer dispatch and inside the reserve retry loop, so the in-flight window is ≈2% of a layer), not a structural impossibility; a targeted test may still be warranted.

**Files.** `scripts/expert_starved_pool.sh`; host accessors + the extended `[Invariants]` line.

---

### Step 5 — The demotion-queue A/B — **done**

**Requirement.** `DEFAULT_DEMOTION_QUEUE_CAPACITY` was **2**, so one layer's evictions overflowed the queue and roughly two thirds of every eviction was dropped. A/B **2 vs 6** at `--warm-gib 40`: `AeonRuntimeConfig::demotion_queue_capacity` (0 = derived from `enable_warm_refill`) and `--demotion-queue <n>`.

**Assert:** `logical_bytes_from_warm` rises; `bytes_from_nvme` falls; `demotion_drops` falls; the logits are unchanged from Run A.

**Outcome.** Capacity `6` removes **every** drop in **both** phases, cutting decode NVMe `40%` for a D2H cost that did **not** become throughput — the win is a resource/cleanliness one (`n=1`, predicted effect below the run-to-run spread). Capacity `12` adds nothing for this workload.

**Files.** `infrastructure/core/tiered_expert_supply.hpp`, `core/memory_budget.hpp` (config), `scripts/expert_demotion_queue_ab.sh`.

---

### Step 6 — Layer-major prefill through the host — **built**

Prefill streams in layer order through the real host, with the swept supply on the production path.

**Decisions (settled, not to be re-litigated).**

| # | Decision | Why |
| :--- | :--- | :--- |
| **D-a** | **Layer-major within a bounded window `W`.** For each window, visit layers 0…42, each in body chunks `C ≤ W`. Chunk-major is the degenerate `W = C`. | Fetches each layer's expert set **once per window**, not once per chunk. Arbiter: colibri `c/deepseek_v4.c` (segment loop). |
| **D-b** | **Warm is frozen during prefill.** No promotion, no demotion; a Warm-resident expert is read as a **non-destructive copy** (a shadow residency), and a swept expert's eviction is a **release**. | Promotion is a **move** — it returns the Warm slot to the free list and would destroy the decode set. Release leaks nothing. |
| **D-c** | **The registry seam is kept.** The sweep goes through `on_routing_ready` / `accumulate_routed`, batched, leases released at the **layer boundary**. | Preserves the certified invariants and the dedup slot-order rule (§5) instead of adding a second path. |
| **D-d** | **The sweep reuses the Hot pool.** No dedicated prefill allocation. | VRAM is binding — dense `12.71 GiB` + KV on a `24 GiB` card. The **routed bank** below the gate reserves one layer's worth *within* the same pool rather than allocating a separate one, so no second allocation exists either; see [Prefill Supply Strategy](../completed/PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md). |

> **Naming.** The window is not any of the three existing "segments" (`HostExpertPool::SEGMENT_SLOTS`, the attention compressor's overlap, a pinned host segment). It is the **layer-major span**; colibri calls the same quantity a *prefill segment*. "Window" is used to avoid the collision.

**The three knobs, and why they are separate.**

| Knob | Bounds | Effect of raising it |
| :--- | :--- | :--- |
| **`C`** — body chunk | the batch scratch (`V4LayerBodyBatchScratch`) | fewer GEMM launches; ~flat in throughput |
| **`W`** — layer-major window | the residual carry (`W × 64 KB`) | fewer sweeps; fewer expert bytes |
| **`N`** — prompt length | — | more sweeps unless `W` grows with it |

The residual is **bounded by the window, not the prompt**, so no VRAM is reserved for a hypothetical carry and there is no pre-flight cliff:

$$\text{expert bytes} \approx \left\lceil \frac{N}{W} \right\rceil \times 156\,\text{GB} \quad (W \gtrsim 256 = \text{one ring}), \qquad \text{residual} = \min(N, W) \times 64\,\text{KB}$$

`156 GB` (the whole model read once) is the floor; `W ≥ N` reaches it. A longer prompt is `⌈N/W⌉` windows, each re-sweeping. Use the **single-drive** rate (`6.33 GB/s`); colibri's figures are on a `2×` NVMe mirror.

**The build (all items built).**

1. `V4Graph::forward_window` over the layer body, plus a device-side batched embedding (`embed_window`).
2. The host allocates the batch scratch from `C` and the residual carry from `W` **at load** (`allocate_prefill_workspace`), worst case across all 43 layers.
3. Windowed layer-major driver: window → layers 0…42 → body chunks, with a host-owned residual carry.
4. **Chunk-wide deduplicated expert dispatch:** attention-and-norm for every token, then router for every token, then **one** `on_routing_ready_batch` over the layer's distinct set, then MoE for every token. Dedup resolves each token's `k`-th expert through a per-token index map, so the fixed-order fp32 reduce still sums in `k` order — dedup changes *which copy is read*, never the order. Leases release at the layer boundary.
5. **Warm-frozen policy (D-b).** A Warm-resident expert is loaded into VRAM as a **shadow residency**: the catalog entry stays `WARM_HOST`, the VRAM copy is recorded in a `shadow_vram_slot` with its own LRU; eviction is a release of the shadow (or of an ordinary Hot resident), never a demotion. `invariants_hold()` admits exactly two owners per slot — a Hot expert, or a Warm expert's declared shadow.
6. **The layer-ordered sweep** (`V4PrefillSweep`). `begin()` frees only what the pass needs — `2E` when the pool holds two layer sets, else `E`, worst-LRU first — and the registry **preserves** the rest; the frontier holds whole layer sets in computation order; each layer's prefill-admitted set is bulk-released as it retires; `end_prefill_stream()` refuses any prefill-admitted resident left. The drained set is recorded in `restore_set()` and reloaded at `prefill_end` (the restore requirement of [Prefill Supply Strategy](../completed/PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md)). Warm and its LRU ranking are untouched for the whole prefill. LRU is not used in prefill — nothing inside a window is reused, so the only correct release is the whole layer. The sweep loads a layer **whole** (its set is known), not a routing prediction. The next layer's reads are **dispatched before the current body runs**, so the previous transfer is complete when awaited; when the pool cannot hold two layers, `dispatch_ahead` is a no-op and the loads fall back to synchronous. Engaged by the visible `AeonRuntimeConfig::prefill_sweep` (default on) **and** the prompt-length gate `prefill_sweep_min_tokens` (default `3 E / 4`) — **no hidden threshold**.
7. **The engine's prefill is the window.** `V4Engine::chat` hands the whole prompt to `forward_window` as one unit via a `PromptPrefill` step of `text::generate_token_ids`; the generation loop still owns the EOS stop, the cap and the context limit. `--prefill-window <W>` (default `4096`) and `--prefill-chunk <C>` (default `64`) are user settings read at load; a prompt longer than `W` runs `⌈N/W⌉` windows, a shorter one is a single window at `C = min(C, N)`.
8. **In-place residual write-back verified.** The body reads only each row's own residual, so the carry is `min(N,W) × 64 KB`, not doubled.

**Outcomes (gate).**

| # | Outcome | Status |
| :--- | :--- | :--- |
| **1** | **Equality** — `chunk ≡ serial`, exact (identical greedy ids **and** byte-identical logits + residual), including through the layer-wide deduplicated dispatch. | **met** |
| **2** | **Tier-invariance preserved** — the Step 3 comparison still holds through the windowed/Warm-frozen prefill. | **met** |
| **3** | **Warm preserved** — Warm's resident set is unchanged across a prefill, and the prefill reads it rather than bypassing to NVMe. | **met** |
| **4** | **No leak** — `invariants_hold()`, `outstanding_leases() == 0`, `staging_in_use == 0` on every run. | **met** |
| **5** | **Throughput** — prefill tok/s against a measured baseline, and bytes/token against the predicted one model read. | **met** |
| **6** | **The sweep switch** — a bounded drain on entry, whole layer sets streamed in computation order, the switch-point Hot set restored on exit, Warm untouched, byte-identical throughout. | **met** |

Outcome 5, measured one pristine process per arm (`bench_prefill_ab`):

| N | serial | swept (double-buffered) | ratio | swept bytes |
| ---: | ---: | ---: | ---: | ---: |
| 256 | `2.48 tok/s` | **`5.71 tok/s`** | **`2.30x`** | constant |
| 512 | `2.42 tok/s` | **`6.96 tok/s`** | **`2.88x`** | constant |

The swept byte count is **constant at `145.1 GiB`** (one model read) while serial grows at `1.54 GiB/token`, so the advantage rises with the prompt.

**Files.** `core/v4_graph.hpp`, `core/v4_layer_body_batch.hpp`, `core/v4_expert_executor.hpp`, `core/v4_expert_supply.hpp`, `core/v4_model_host.hpp`, `core/v4_prefill_sweep.hpp`, `core/v4_engine.hpp`, `infrastructure/text/text_generation.hpp`, `core/memory_budget.hpp`, `tools/aeon_chat.cpp`.

**Arbiter.** `aeon-references/colibri/c/deepseek_v4.c` (window/chunk loop), `c/deepseek_v4_bank_pair.h` (double-buffered sweep). Colibri's `~0.35 s/layer` is a `2×` NVMe figure; on one drive it is `~0.7 s/layer`.

---

### Step 7 — Window and chunk: sweep, then expose — **settings/budget done, window sweep open**

**Requirement.** Neither knob carries a target in advance: `C` bounds the scratch, `W` bounds the residual. Both are exposed as user settings so the engine is configurable for other hardware rather than tuned for this GPU; what remains is to **sweep** them.

**Expected shape, to confirm or refute.** The window is the lever and the chunk is nearly flat (colibri's `V4_PREFILL_CHUNK` 128 vs 64 differed ~3% end to end, while the window count dominated).

**The budget consequence.** `PIPELINE_SCRATCH_BYTES = 100 MiB` was a literal wrong in both directions. It is replaced by **three derived terms** — a decode allowance, a batch term from the configured `C`, and a residual-carry term from the configured `W` — each reported separately, with the two allowances checked against the real allocations at load. The staging line is derived (`max(12, min(6C, experts_per_layer))` slots) instead of the literal `12`.

**Warm and staging are independent budgets (D5).** The budget no longer subtracts the staging figure from `warm_host_bytes`: staging is a corridor sized by `C`, Warm is expert residency sized by the request, and neither derives from the other. The requested Warm budget is honoured in full, and the **total** (Warm + staging + reserve) is checked against the host ceiling.

**Status.** Settings and budget closed (all four items above). The **chunk half of the sweep is measured flat** (`128 → 256` bought nothing while costing `+87 MiB` scratch), confirming the expected shape. **Still open:** the **window half** of the sweep — every measurement to date is one pass (`W ≥ N`), so the `⌈N/W⌉` regime has never executed.

**Files.** `core/memory_budget.hpp` (derived lines), `tools/aeon_chat.cpp` (the settings).

---

### Step 8 — The layer-body observer — **open (deferrable)**

**Requirement.** The body takes an observer, but the only production one lived inside the pre-rewrite `V4Pipeline`; the null one runs and leaves no diagnostic path. Write the adapter.

**Why it is last.** Nothing in Steps 1–7 depends on it — the counters come from the supply and the registry, both already instrumented. Pull it forward only when a defect needs per-op attribution.

**Gate.** None required; it must not perturb the hot path — assert identical output traced vs null.

**Files.** `core/v4_graph.hpp`.

---

## 5. Rules that bind this work

| Rule | What it forbids |
| :--- | :--- |
| **Accumulation order is slot order** | Any dedup that permutes which slot a token's k-th expert occupies. The fixed-order reduce sums in slot order and fp32 is not associative; dedup changes *which copy is read*, never the order it is summed in. |
| **A lease grants no ordering** | Releasing a slot while compute reading it may be in flight. The token/layer boundary is a precondition, not an implementation detail. |
| **Never clamp a position** | An out-of-range position is refused, never clamped. |
| **Anti-circularity** | A test must not compare a kernel against an oracle derived from that kernel's own helper. |
| **Tiering is mechanism, not strategy** | Holding one request shape as *the* shape. |
| **Staging is a pipeline buffer, not a working-set store** | Sizing it from the chunk size or the layer's expert set, or summing it with the sweep's VRAM residency. It holds transfers **in transit**. |
| **Prefill and decode are two strategies, not one parameter** | A resident from one phase crossing into the other, and a recency policy (LRU) applied to prefill. Prefill frees only what it needs on entry, streams in layer order, and restores the switch-point set on exit; decode owns recency and demotion. |
| **A shadow residency is legal only while Warm is frozen** | A second ownership existing in decode. Asserted in `invariants_hold()`, not merely intended. |
| **A gate that reads device buffers must synchronize the stream first** | Comparing device buffers without ordering against the non-default compute stream — a plain `hipMemcpy` can return the *previous* run's buffer. |

---

## 6. Open work

| Item | State | Detail |
| :--- | :--- | :--- |
| **Window half of the `W`/`C` sweep** | **open** | every run to date is one pass (`W ≥ N`, including the `W = 1024` M44b matrix); the `⌈N/W⌉` regime and the cost of a second pass are unmeasured. |
| **Prefill body cost `~120 ms/prompt-token`** | **open — next investigation** | linear in the prompt, measured through `aeon_chat`; not supply and not the registry audit. The `≈31 ms/token` reference is unsourced for this checkpoint (see M44's note) and must be split into body vs supply before it anchors a target. |
| **Staging arena's `banks × depth` target** | **open** | at the dedup ceiling because a layer-wide dispatch assigns every distinct expert at once (`3.38 GiB` pinned at `C = 256`); the smaller waved ring is unbuilt. |
| **Prefix reuse (session state and swap)** | **open — not deferrable** | a product requirement: today every turn re-prefills from token 0. |
| **Routing concentration / frequency policy** | **open** | decides whether frequency-informed placement can help; owned by the [Routing Profile and Placement Study](ROUTING_PROFILE_AND_PLACEMENT_STUDY.md) Phase 2. |
| **Cold → Warm fill path** | open | matters only when the Hot pool cannot hold a layer. |
| **Expert-placement policy (routing-aware hotlists)** | open | Steps 3–5 green and routing concentration answered. |
| **Warm admission by decode priority** | open | routing concentration answered. |
| **Step 8 — layer-body observer** | open, deferrable | needed only when a defect requires per-op attribution. |
| **Prefix matcher, MTP, multi-GPU** | future | unchanged from the composition plan. |

---

## Appendix A — Model composition (verified)

The premise the supply-chain discussion rests on: **what is resident and what streams.** Every figure is computed from `V4ModelContract::uploaded_dense_bytes` and the checkpoint's own `config.json`; the total reproduces the recorded `13,643,885,660 B`.

| | Bytes | Size | Lifetime |
| :--- | ---: | ---: | :--- |
| **Dense backbone** (uploaded) | 13,643,885,660 | **12.71 GiB** | VRAM, permanent |
| `embed.weight` | 1,059,061,760 | 0.99 GiB | **host RAM**, read per token |
| **Routed experts** (11,008) | 155,826,782,208 | **145.12 GiB** | streamed from NVMe |

The routed experts are **92%** of the model; the dense backbone is **8.7%**.

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

The alternating pattern ends on **CSA at layer 42**, not a sliding layer; `V4ModelSpec::validate_config` enforces `layer % 2 == 0 ? 4 : 128` for `layer >= 2`.

### The hash router

Layers 0–2 select experts from `ffn.gate.tid2eid`, a `[129280, 6]` table **indexed by token id** (`out_indices[k] = table[token_id][k]`). There is no top-k selection; `ffn.gate.weight` still runs, but only to compute the *weights* of the already-chosen six. The choice is **independent of context and position**, so the same token always draws the same six experts and different tokens draw unrelated ones — which defeats cross-turn reuse for those layers by construction.

### `embed.weight` and `head.weight` are not duplicates

They are distinct `[129280, 4096]` F16 matrices — `tie_word_embeddings` is `false`, and `test_v4_graph_head.cpp` asserts they differ **from the bytes**. `embed.weight` is held host-side deliberately (read one row per token; on the device it would cost ~74 Hot slots); the rest of the dense container's page cache is released after upload by `release_dense_pages_except("embed.weight")`.
