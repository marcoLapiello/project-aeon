# Project Aeon — Documentation Status

**Audited: 2026-09-16, on branch `rewrite/graph-v2`.**

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
| Default `ctest` | 39 infrastructure/backend/text/kept-component/Tier-1/Tier-2/Tier-3/Tier-4/Step-3 tests |
| Legacy `ctest` | 41 (the 39 plus the 2 gated parity anchors) |
| Research | Phases 0.1–0.2f complete; the whole forward pass is re-cited |
| Step 0 | **Verified** — the artifact's own encoder is ported and matches all 4 golden vectors byte-for-byte |
| Step 3 | **Verified — the last gap in the graph closed.** `hc_head` was the **only graph op with no code, no oracle and no gate**; it did not appear anywhere in `reference/dsv4_oracle.hpp` and nothing in `tests/` touched it, yet it is the last operation before the LM head, so an error there produces plausibly-scaled logits and fluent-looking garbage. The oracle grew `hc_head_reduce`; `tests/test_v4_hc_head_oracle.cpp` certifies `hc_head_wave32_kernel` against it on real `hc_head_fn`/`base`/`scale` at the real shapes. **25 checks, 0 failures.** The kernel matches to `2.3e-4 … 4.3e-4` of peak (the fp16 store), and the four ways the head is not 2.0's pre-mix are each asserted as a *discriminating* check rather than by construction: the norm is **weightless** (a supplied weight moves the output 11x the noise floor), the RMS is over the **flattened** dim (33x on stream-asymmetric input), `hc_head_scale` is a **scalar `[1]`** against the layer scales' `[3]`, and there is **no Sinkhorn or comb** (a single-stream residual gives `out == pre[3]·x[3]` exactly). **No defect found.** A property discovered while writing it: `mixes` is **scale-invariant**, so no scaling of the residual can saturate a sigmoid — the first probe tried exactly that and failed; `out`'s degree-1 homogeneity is asserted instead, and the `rms_eps` term is shown to be what breaks it. **6 of 6 mutations killed**, one of them (the dropped `hc_eps`) caught *only* by a saturation probe needing an **absolute** floor. Also **fixed in this step: the routed-expert accumulation** — see the note below. |
| Tier 1 | **COMPLETE — all 11 primitives certified, then mutation-tested.** RMSNorm, RoPE (both bases), MLA Q/KV, HC + Sinkhorn, attention + sink + softmax, compressor + APE, indexer + top-k, grouped output projection, MoE router, routed expert, shared expert. Four real findings: a transposed comb index in the plan, the indexer ReLU missing from the kernel, the plan's normalization-guard claim being wrong, and the combine-order claim describing only the unfused path. **Mutation testing then found two gate defects that review and a green suite had both missed** (see the plan's "Mutation testing" section): the clamp-rule gates could not see a symmetrically-clamped kernel, and the RMSNorm gate could not see a deleted `eps`. 10 mutations: 8 killed, 1 provably equivalent, 0 unclassified. |
| Tier 2 | **Items 16–18 done — all three attention classes, and the serial loop.** `core/v4_layer_body.hpp` is the single layer body (Steps 2.0–2.11 for one token) and `reference/dsv4_oracle.hpp::layer_body` its composed fp64 reference, branching on the attention class exactly as the device does. Item 16 covers the Sliding class on real `layers.0` weights (`tests/test_v4_layer_body_oracle.cpp`); item 17 covers **CSA (ratio 4)** and **HCA (ratio 128)** on real `layers.2` / `layers.3` weights (`tests/test_v4_layer_body_compressed_oracle.cpp`), including the compressor, the APE-adjusted partial ring, the materialized compressed entry, the indexer and the row-set rule; item 18 (`tests/test_v4_layer_body_serial_oracle.cpp`) drives a three-layer stack for **136 tokens across 34 CSA boundaries and one HCA boundary** with the residual carried by the device itself, and compares the **whole accumulated state**. Every checkpoint within `~1e-3` of its own peak against a `4e-3` tolerance. **13 of 13 mutations killed** (4 wiring + 1 redundant in item 16; 5 in item 17; 3 in item 18). Item 16 found a defect **in its own oracle**; item 17 found that the router ids cannot be compared against an fp64 oracle's inputs (**trap 37**) and made the fp16 decode ~5× faster; item 18 found that the device's serial decode is **not bit-reproducible** (**trap 38**) and that a state-evolution error is **invisible to a per-step gate** (M18-2: 577 failures in item 18, zero in item 16). Next: item 19 (chunked batched prefill, gate `chunk ≡ serial`). |
| Tier 3 | **Item 19 done as a structural gate; its two non-structural halves are open.** `core/v4_layer_body_batch.hpp` drives the *same* two half-bodies decode calls (`run_layer_body_pre_attention` / `run_layer_body_attention_tail`) and holds the chunk's keys **outside** the local ring, with a per-query composed row-set ordered by ring slot; `tests/test_v4_layer_body_chunk_oracle.cpp` requires **exact equality — 0 differing values** — over 130 tokens × three attention classes × three chunk schedules, comparing each token's residual, router logits, ids and weights plus the whole final state (ring keys/positions, all 32 CSA + 1 HCA committed entries, the compressor's partial ring), and ties the one-at-a-time run to the Tier-2 certified decode body so the equality is not circular. **The finding the plan had missed: the obvious chunking is wrong (trap 39)** — writing the chunk's keys into the local ring as it goes is not equivalent to serial at any chunk length above one, because the write for the chunk's last token evicts the oldest key of its own first token's window. That is the **local** ring, not the compressed path the plan's original finding blamed. **Trap 38 was answered explicitly**: the gate requires the deterministic MoE accumulation, which is why the equality is exact rather than a tolerance. **Open, and named as such:** chunked-prefill *throughput* (a separate gate by the plan's own rule — the composition is a per-query loop of device-to-device copies and the body is still per-token) and the indexer top-k's per-token host round-trip, which Part III forbids in a prefill but which changes no value, so no equivalence gate can see it. **4 of 4 mutations killed** — and **M19-4 survived the first sweep, which was a gate defect**: section C compares the chunk path against itself at a different chunk length, so a mistake applied to *both* sides (the commit position) was invisible; section C2 now compares the **final state** against the Tier-2 certified decode body, whose other side does not share the chunk driver. That is the method's second failure mode (a metric that cannot see the difference), the same shape as Tier 1's M9/M9b. **Item 20 (long-context lifecycle):** `tests/test_v4_layer_body_lifecycle.cpp` runs a three-layer stack for **260 tokens at the model's own window (128)** — every earlier gate shrank the window and named the shrinkage as uncovered — and compares against **closed forms and invariants rather than an fp64 oracle**, because items 16–18 already own the arithmetic: the ring's contents are *predicted* from the token count before they are read. **52 checks, 0 failures, 5.6 s.** The ring's slot assignment holds for all 384 slots, an **unfilled** slot perturbs attention by **exactly `0.0`** while the oldest in-window row moves it by `26.9×` peak, each class's **first** compressed entry is byte-identical 256 steps later (with the local ring wrapping twice in between), and the row-set rule holds past the window (HCA reads every committed entry; CSA's 8-of-65 selection is a real selection and swapping in a rejected candidate moves attention by `1.63`). **The finding is that the capacity refusal in `V4Layer::record_position` is load-bearing, not decorative (trap 40):** a wrapped compressed store is invisible to the kernel's own position guard *and* to the committed count — a hand-built wrapped store passes every guard and the count reads `4` for both — so only the positions' closed form distinguishes it, and the row-set would silently become "the newest `K`" instead of "every committed entry". A wrapped store needs a position past the declared context (`pos ≥ 1023` against a 512 context), which is why it cannot arise by accident. **6 of 6 mutations killed**, and **M20-6 forced a repair of the gate's own probe** (section E had been querying an HCA store with `ratio = 4`, so its count check passed on a mismatched pair). **Next: item 19's two open halves, then Tier 4 item 22.** |
| Tier 4 | **Item 21 done — streaming and tiering are transparent to the numerics.** `tests/test_v4_expert_tiering.cpp` is the first gate to drive `TieredExpertSupply` itself rather than a leg of it: one expert delivered **Cold** (`O_DIRECT` into a `PrefetchStagingArena` slot — four io_uring requests, because the `14 155 776`-byte payload is a non-integral `4` chunks of the reader's `4 MiB`), **Hot** (a repeat request that must claim no staging slot and move no bytes) and **Warm** (an LRU demotion by D2H into a pinned host slot, then a promotion back from it). Every delivery is compared byte-for-byte against the **mmapped** container — a different I/O route from all three — and the three are bit-identical to each other and to the artifact. The pool is deliberately **saturated** (the registry populates every Hot slot at construction), so every cold miss must evict a resident, which is what makes the demotion and promotion legs reachable without contriving one. **29 checks, 0 failures, 0.75 s.** Each delivery's *tier* is asserted as well as its bytes, because a silently dropped demotion would answer from Cold and pass a byte check while measuring nothing. The cold payload is checked **twice** — in the staging slot and in the destination VRAM slot — which is what localises a failure to a leg: **5 of 5 mutations killed, each by a different check** (the `O_DIRECT` offset by the staging check; the wrong staging slot by the resident check *with the staging check still green*; the D2H and H2D ends of the warm pair separately). Repairs during the sweep were both probe defects: with the warm pool non-empty and VRAM saturated, warm-pool *counts* cannot measure consumption — the promotion's own eviction fills a slot — so the instrument became the host slot itself. Not covered, named there: checkpoint-plan **Stage D.2 concurrency**, which needs a forward pass streaming experts while the graph runs. **Next: item 22.** |
| Item 22 (R3) | **Certified — the state can now be *restored*, not merely captured.** `V4Layer::snapshot_state()` had existed with nothing to put a snapshot back, so the state contract had no executable meaning. `restore_state` is implemented with **size validation** (a snapshot from a differently shaped layer throws rather than writing past a buffer), and `tests/test_v4_state_restore.cpp` certifies it against **a run the layer never stopped**: tokens `0 … N+K−1` straight through versus prefix → snapshot → reset → restore → continuation, requiring the continuation's tokens and the final state to be **bit-identical**. Boundaries are genuinely mid-ratio-window — CSA at prefix 10 (`10 % 4 == 2`) *and* at 12 (on a boundary), HCA at 140 (past its entry at 127) — and the gate asserts each label against the boundary arithmetic, so "mid-window" cannot silently become "on a boundary". **146 checks, 0 failures, 5.5 s.** Section C makes it load-bearing: a cleared snapshot moves 30/30 tokens, a zeroed ring 7, lost partial positions 29. **6 of 6 mutations killed** (RS-1…RS-6). Not covered: **R4** (declining a boundary outside the local window — the matcher's half, which needs the cache key and block table), tier placement (R5), and the non-token cache-key inputs (trap 23). |
| Correction | **The committed tree was red, and two documented blockers were false.** `ctest` does not build: when `db0d3ed` was committed, the binaries for items 16/17/18 were **stale**, so the suite read `38/38` while that commit's *source* failed — rebuilding it gave **21 failing lines in item 16 alone** (`moe_out` at `1.32×peak`). The cause was in that commit: it extracted `swizzled_w2_row_dot` **with** its `__shfl_xor` reduction but left the caller's reduction in place, so the routed contribution was multiplied by `LPR` (a silent `×4`) — visible only to an oracle comparison, never to a run-vs-run check, and it affected the deterministic path too. Fixed by removing the caller's duplicate: `test_aeon_moe_fused_w2` now reads `0.000976562` (one fp16 ulp) where it read `8.55427`. Separately, the claim that the chunk size was **capped at 8** by the compressor's partial ring — which the plan used to justify deferring item 19's throughput half and to motivate item 22 — was **measured false**: a chunk of **16**, larger than both rings, is bit-identical to serial, because a boundary materializes before any later token can reach its slot. Item 19(a) now needs only batched projections. |

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

**Correctness (the active work).** Execute the plan: the oracle and gate harness are
built, all eleven Tier-1 primitives are certified and mutation-tested, and **Tier 2 items 16–18
are done** — the single layer body (`core/v4_layer_body.hpp`) is gated against the
composed fp64 reference on real weights for **all three attention classes**: Sliding, CSA
(ratio 4) and HCA (ratio 128), the last two including the compressor, the APE, the indexer
and the row-set rule, and the serial loop carries the device's own residual across 34 CSA
boundaries and one HCA boundary. **Tier 3 is now complete**: item 19's structural gate
(`core/v4_layer_body_batch.hpp` + `tests/test_v4_layer_body_chunk_oracle.cpp`) drives the same two
half-bodies decode calls a chunk at a time, holding the chunk's keys outside the local ring and
composing a per-query row-set, and requires **exact equality** (0 differing values) between a
chunked run and the same tokens one at a time — across three attention classes, three schedules and
the whole final state; and item 20 (`tests/test_v4_layer_body_lifecycle.cpp`) runs **260 tokens at
the model's own window (128)** and asserts the lifecycle against **closed forms rather than a
reference**, showing that the compressed store never evicts inside the declared context (its
capacity is exactly that context's entry count) while a *wrapped* store would be undetectable from
its contents and would silently turn "every committed entry" into "the newest `K`" (**trap 40**) —
which is why `V4Layer::record_position` must **refuse** an out-of-range position rather than clamp
it. Item 19's *throughput* half and the indexer top-k's host round-trip remain open and named as
such; neither is a correctness gap. **Tier 4 item 21 is now certified**: `test_v4_expert_tiering`
drives `TieredExpertSupply` itself — one expert delivered Cold (`O_DIRECT` through the staging
arena), Hot (a repeat request that must move no bytes) and Warm (LRU demotion into a pinned host
slot, then a promotion back) — and requires every delivery to be **bit-identical, `14 155 776` bytes
of `14 155 776`**, against the mmapped container, which is a different route from all three. Each
leg's *tier* is asserted as well as its bytes, because a silently dropped demotion would answer from
Cold and pass a byte check while measuring the wrong tier. What remains of the plan's sequence is
Tier 4 — **item 22** (prefix cache
manager: restore byte-exact, and a boundary outside the local window detected rather than served
stale) and **item 23** (generation loop). Separately, the comparison of identical formatted inputs,
intermediate checkpoints and final logits against a **trusted compatible reference** (the
patched-RDNA vLLM run) is a real and motivated gate, but it has **no number in the plan's sequence**
— `DOCUMENTATION_STATUS.md` and earlier notes called it "item 21", which the plan assigns to
streaming/tiering. It must be given a number or named explicitly before it is scheduled. The last
three gates settled three properties the remaining work must respect: a decode step is **not
bit-reproducible** (trap 38), so `chunk ≡ serial` and byte-exact prefix restore each have to state
which MoE accumulation they require — item 19 requires the deterministic one; the local ring is
**not reconstructible** from anything else (trap 39), so a prefix boundary older than the window
must be handled by *replaying* the last `C` tokens rather than restoring a ring; and the compressed
store never evicts inside the context (item 20), so only the local ring needs that replay (trap 40).

Four gates are settled empirically, not by reading. **One is closed, one is now half-settled; two remain:**

1. ~~Indexer Hadamard rotation — apply or not (must be symmetric over Q/K).~~ **SETTLED at Gate 11: do not apply it.** Two-sided is a no-op to `3.6e-16`; one-sided shifts scores by `1.46`. See 2.4.3.
2. KV fp8/E4M3 vs bf16 storage delta.
3. MoE routed-expert accumulation order.
4. ~~Local-window prefix-reuse boundary behaviour.~~ **Half settled at Tier 3 item 20.** Within the declared context the compressed store **never evicts** — its capacity is exactly that context's entry count, verified for both ratios — so a reused prefix's *compressed* state is always fully present and only the **local ring** is window-bounded and must be **replayed**, not restored. The converse is what makes the refusal structural rather than tidy: a wrapped compressed store is invisible to the kernel's own position guard *and* to the committed count, so exceeding the capacity would silently serve the newest `K` entries instead of every committed one (**trap 40**). What remains open is the *prefix-reuse* half proper — detecting a boundary older than the window and declining it rather than serving a stale ring — which is **Tier 4 item 22**.

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
