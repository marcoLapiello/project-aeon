# AGENTS.md — Project Aeon System & Agent Context

## 1. Project Purpose & High-Level Context
**Project Aeon** is a bare-metal Mixture-of-Experts inference engine in C++20 and native HIP for
consumer AMD RDNA3 (`gfx1100`), scalable across multi-GPU rigs. It attacks the memory wall for very
large MoE models on consumer workstations with three mechanisms: **bare-metal RDNA3 execution**
(Wave32, WMMA, direct HIP/AMDGCN dispatch — no CUDA, no framework overhead); a **three-tier
hierarchy** (VRAM Hot ← host-DDR Warm ← NVMe Cold, via `io_uring` `O_DIRECT` at 4 KiB alignment);
and **double-buffered expert-batched prefill** overlapped with asynchronous DMA/storage.

---

## 2. Documentation and References

**Navigation — read this first.** [Documentation Status](plans-and-docs/status/DOCUMENTATION_STATUS.md)
owns the plan inventory, the document map (live / active / completed / superseded), the open gates,
and the rules for historical documents. **This file does not duplicate that inventory.**

**The specification** for the DeepSeek-V4 graph rewrite on branch `rewrite/graph-v2` — the storage,
streaming, artifact-format and kernel layers are kept; the graph that composes them is rebuilt:
- [Inference Pipeline Plan](plans-and-docs/analysis/current/inference_pipeline_plan.md) — **the
  authority for graph semantics.** Evidence-tagged (`[V]`/`[?]`/`[I]`), cited step-by-step procedure
  with per-step gates, a reference hierarchy, and the anti-circularity and mutation-testing rules.
- [Checkpoint & Artifact Integrity Plan](plans-and-docs/analysis/current/checkpoint_verification_plan.md)
  — proves the **input** is sound (structural audit, prompt-encoder oracle, repack round-trip,
  streaming integrity), so that a graph failure is a graph failure. Companion to the specification;
  it does not cover the graph.
- [Graph Composition Plan](plans-and-docs/analysis/current/graph_composition_plan.md) — **the
  composition**: one ordered text-in/text-out path, every step mapped to the component that
  implements it and where it lives, and the definitive list of what does not exist yet. Owns the
  graph's build phases (P1–P7) and the acceptance criterion. It consumes the specification and
  re-derives nothing.
- [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md) — authoritative silicon
  measurements. **Read its banner**: the pre-rewrite model-path entries were **deleted 2026-09-18**
  (they measured a *different computation*, not a slower one), so §4 holds only component, storage
  and machine measurements, and §5 holds everything measured on the rebuilt graph.

Execution records (active, completed, superseded) are indexed in the status document. Do not use an
old checklist or review as current implementation evidence.

### Local Reference Implementations
The primary external source references are maintained as shallow, default-branch checkouts outside this repository. They are for source comparison only, not Aeon build or runtime dependencies:
- [llama.cpp reference checkout](../aeon-references/llama.cpp): portable runtime, expert streaming, quantization, KV state, and serving paths.
- [FreeToken reference checkout](../aeon-references/freetoken): bandwidth-adaptive CPU/GPU execution, expert caching, prefill streaming, and agent-facing serving.
- [Colibri reference checkout](../aeon-references/colibri): VRAM/RAM/NVMe tiering, routing-aware placement, direct I/O, prefetch, and persistent KV state.
- [DwarfStar (ds4) reference checkout](../aeon-references/ds4): DeepSeek-V4-specific kernels, profile-derived expert hotlists, SSD streaming, KV/prefix caching, and native agent serving.
- [vLLM reference checkout](../aeon-references/vllm): paged memory, prefix/KV caching, scheduling, resource management, and production serving. **Also the canonical DeepSeek-V4 prompt encoder** — `vllm/tokenizers/deepseek_v4_encoding.py::encode_messages` — since the checkpoint ships no `chat_template`.
- [SGLang reference checkout](../aeon-references/sglang): radix/HiCache, chunked prefill, MoE scheduling, disaggregation, and AMD paths. Its readable `srt/layers/attention/dsv4/**` and `kernels/ops/attention/dsv4/**` are model-specific and are the preferred arbiter for DSV4 attention and indexer semantics.

Update a reference checkout with `git -C <directory> pull --ff-only` and record its commit SHA whenever an implementation decision depends on a specific revision.

---

## 3. Progress Tracking & State of Execution
*Status: 2026-09-17, branch `rewrite/graph-v2`.*

**This section is an index, not a record.** One row per milestone, pointing at the document that owns
the detail: the [Inference Pipeline Plan](plans-and-docs/analysis/current/inference_pipeline_plan.md) (per-item gate results, traps 1–44, mutation
tables, open unknowns), the [Documentation Status](plans-and-docs/status/DOCUMENTATION_STATUS.md) (plan inventory and document
boundaries), and the [Performance Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md) (silicon measurements and its validity banner).
**Do not restate a finding, gate result, tolerance, trap, or measurement here — link to it.** When a
milestone transitions, change its row.

### The graph rewrite (active)
Executing Part V of the plan. Tiers 0–3 are complete; Tier 4 is under way.

| Stage | State | Detail |
| :--- | :--- | :--- |
| Tier 0 — specification research (phases 0.1–0.2f) | done | plan §Tier 0 |
| Step 0 — prompt encoding | done | plan §Step 0 |
| Tier 1 — oracle harness + all 11 primitives | done, mutation-tested | plan §Tier 1 |
| Tier 2 — items 16–18: layer body, all three attention classes, serial loop | **closed** | plan §Tier 2 |
| Tier 3 — items 19–20: chunked prefill, long-context lifecycle | done *(item 19's throughput half blocked)* | plan §Tier 3 |
| Tier 4 — items 21–23: streaming/tiering, prefix cache, generating loop | item 21 done; item 22's restore half (R3) done; **23 closed** — executor seam, head end (P1), 43-layer driver (P2), sampler (P3) and text binding (P4) all built | plan §Tier 4 / [Graph Composition Plan](plans-and-docs/analysis/current/graph_composition_plan.md) |

`core/v4_layer_body.hpp` is the single layer body; decode, chunked prefill and every Tier-2/3 gate
call it. The pre-rewrite graph (`core/v4_pipeline.hpp`) and its gate were **deleted on 2026-09-18**,
together with the two gated real-weight parity anchors, the two tools that drove it, the superseded
`dsv4_chat_formatter`, and the orphaned `v4_attention_oracle` — roughly 5,600 lines, all of it
unreachable from the default build (verified by a trial deletion + build). Default `ctest`: **44 tests**.

**Next: P5 — tiering on the live path.** **P0–P4 are green and the graph speaks**: `V4ModelHost`builds the whole assembly (`core/v4_model_host.hpp` — loader → config → spec → contract → budget →
resources → scratch → streams → 43 layers → Hot/Warm expert pools → registry → staging → tiered
supply → production `V4TieredExpertExecutor`), `V4Graph` runs `embed_token` → 43 ×
`run_layer_body_decoding` → `hc_head` → final norm → LM head (**34 checks, 0 failures, 5/5 mutations**,
`forward_token` reproducing the gate's own per-layer loop at **0 differing of 517 120** fp16 logits),
`core/v4_sampler.hpp` turns those logits into a token behind a **logit-processor seam** (**57 checks,
6/6 mutations, plus one named equivalent**), and `core/v4_engine.hpp` binds the tokenizer, the prompt
encoder and the generation loop to them (**28 checks, 7/7 mutations**) — so `aeon_chat`, now in the
**default** build, takes a conversation and returns text:

```
What is the capital of France?  ->  The capital of France is **Paris**.
```

EOS-reached, 43 layers, on the artifact's real weights through Hot/Warm/Cold — the plan's acceptance
criterion, and the same sentence the pre-rewrite graph produced.

**A resource-accounting correction landed with P4 (ledger M28).** The budget reserved
`dense_file_size()` — the whole container — while the graph uploads only what the contract
enumerates, so **1.957 GiB** was reserved for VRAM never touched (`embed.weight`, read from the host
mmap, and the unused `mtp.*` draft head). Reserving it cost 148 Hot expert slots. The budget now
derives dense bytes from `V4ModelContract::uploaded_dense_bytes`, plans against `min(free, total)`
rather than nominal VRAM, and caps host RAM at `total − 10 GiB` rather than 70% of total. Measured:
**21.99 → 23.76 GiB** held (`rocm-smi --showpids`), **675 → 809** Hot slots. Note
`rocm-smi --showmeminfo` numbers devices differently from `--showpids`; use the latter for a
per-process figure.

**P6's reconnaissance is recorded in the composition plan before the phase starts**, because it
found that the plan's own framing of that phase was too small. The rewrite has **no prefill path**
(the prompt goes through the same one-token-at-a-time call as decode), the host allocates **neither**
batch scratch type, and the expert seam is **strictly per token** — so a chunk batches attention and
leaves the MoE serialized, which is ~95% of prefill's cost (3.65 GB of expert weights streamed per
token). What P6 must build is an interface change, not just a driver. The detailed list, the measured
numbers, and the two budget literals still to be derived are in the composition plan's P6 section.


What P5 adds is not a component but **pressure**: run with `hot_vram_slots` small enough that every
layer misses and with a warm tier that is not preloaded, and re-derive the item-21 properties
**through the graph** — cold reads counted, staging slots returned, `forced_drains()` recorded,
`invariants_hold()` at the end, and the logits bit-identical to a run with all experts resident. That
is item 21's `Stage D.2` remainder, and it is named so the engine purpose is not mistaken for done.
The full ordered list and each phase's gate are in the composition plan.

**Two traps found in the last two phases, both about the *instrument* rather than the arithmetic.**
**Trap 44 (P3):** the gate had to assert that the seam runs *before* the truncations, and the first
check written for it — "a promoted token survives `top_k = 1`" — passes on **both** orderings, so it
was a passing check that discriminated nothing; it was replaced by what the processor **sees**. The
general form: **for an ordering claim, assert an observable of the moved step, not a downstream
consequence both orderings produce.** The sweep is what found it, which is why the sweep runs before
the gate is trusted. **At P4 the same shape appeared one level up:** the "the history is in the
context" claim was first made on the *drawn token*, and at `T=1` both contexts drew the same one —
the honest instrument is the **logits** (129 251 of 129 280 differ), because a peaked distribution can
draw the same token from two different distributions.

**A cost trap found at P2, worth knowing before touching a model-level gate.** The new gate takes
~3 minutes, and essentially all of it is the *reference*, not the device: the same 172 layer-steps
cost the device 1.6 s. The fp64 reference materialises each real expert's three matrices as `double`
— 600 MB per expert — so it is memory-bound, and the per-element swizzle address arithmetic that
looks expensive is not (hoisting it bought `1.4x`). The reuse measurement is in the gate: 767 of
1032 requests were distinct `(layer, expert)` pairs, so caching decoded experts cannot pay either.
The same shape as trap 42: **attribute an instrument's cost by measurement, not by inspection.**
The three graph gates now have three distinct cost shapes, which is the useful part: P2 is **180 s of
fp64 reference**, P3 is **8 s of pure transform** (no model in the loop at all), and P4 is **~110 s of
device** — a text-in/text-out run has to run the model, and the model costs what it costs.

**Two items are explicit non-goals for the near term** and are named so they are not mistaken for
gaps: the prefix **matcher** (22b) and **R4**. Session swap needs neither.

**Needs a decision — a numbering conflict.** "Item 21" currently names two different gates: the plan's
Tier 4 item 21 is streaming/tiering, while `plans-and-docs/status/DOCUMENTATION_STATUS.md` and older notes use it for the comparison
against a **trusted compatible reference** (the patched-RDNA vLLM run). That comparison is a real,
separately motivated gate but has **no number in the plan's sequence** — give it one before
scheduling it.

### Other states
- **Superseded —** [Model Correctness Execution Plan](plans-and-docs/execution/superseded/MODEL_CORRECTNESS_EXECUTION_PLAN.md):
  the staged in-place repair, replaced by the plan. Retained for chronology; do not execute.
- **Completed and still valid —** Phase 0–2 foundations; modular runtime ownership; artifact and
  backend contracts; native text front end; the Stage-1 expert-kernel path (swizzled layout plus
  fused W1/W3 and W2 kernels, now the default routed-expert path); warm-tier repair and supply
  telemetry ([closure report](plans-and-docs/execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_AB_REPORT.md)).
- **Paused —** Phase 2 continuation (cold-tier, storage layout, placement, latency hiding), pending
  the graph rewrite and the host-pressure tradeoff.

### Open gates
Pointers only; each is specified in the plan's "Open unknowns" table or the ledger.
- **Graph correctness —** Tier 4 item 23's remaining half: the sampler (P3) and the text binding (P4).
  The 43-layer driver (P2) is closed. (See *Next* above.)
- **Measurement gates —** KV fp8/E4M3 vs bf16 storage; MoE routed-expert accumulation order
  (partially settled); local-window prefix-reuse boundary (half settled by item 20). The indexer
  Hadamard is **settled — do not apply it** (plan Gate 11).
- **Kept infrastructure —** cold-tier characterization, physical `.aeon` placement, host-memory
  pressure, the model-backed `>= 6.0 GB/s` target, kernel occupancy tuning, the routing placement
  study, the explicit backend factory, and Phase 3 multi-GPU.
- **Before comparing any number —** read the ledger banner: the pre-rewrite model-path entries were
  deleted because they measured a *different computation*, not merely a slower one. §4 keeps only
  what the graph change cannot move; §5 is the post-rewrite record.

---

## 4. Project Rules & Engineering Conventions

### Code Architecture & Runtime
1. **Production Runtime Zero-Dependency Policy**: The inference runtime engine must remain pure C++20 / native HIP with no runtime dependencies on Python or PyTorch.
2. **Standard Ecosystem Compatibility**: Never invent custom lossy quantization formats. Ingest standard community formats (GGUF, Safetensors).
3. **Hardware Precision Discipline**: Golden reference tests must verify that GPU GEMM / dequant kernels match reference precision within standard FP16/BF16 tolerances ($\epsilon < 10^{-3}$).
4. **Direct I/O Discipline**: All streaming file reads must be strictly 4096-byte aligned (`O_DIRECT` compliant) to eliminate kernel page-cache contention and buffer copies.

### Development Process & Git Conventions
1. **Incremental Micro-Steps**: Advance through small, verifiable steps. Never implement broad abstractions before underlying hardware primitives are verified on silicon.
2. **Hardware-Grounded Verification**: Test and benchmark on physical hardware (`gfx1100`) at every step.
3. **Commit Messages**: Follow standard conventional commits format (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`).
4. **Maintenance of AGENTS.md**: Update the "Progress Tracking & State of Execution" section whenever milestones or micro-steps transition between Past, Present, and Future but keep in mind that this is an entry-point not a detailed record - more details are documented in the related plans and documents.
5. **The specification is authoritative**: For DeepSeek-V4 graph semantics, [Inference Pipeline Plan](plans-and-docs/analysis/current/inference_pipeline_plan.md) governs. Do not implement a graph op from memory, from this file, or from an unsourced reference. If the plan lacks a citation for something being implemented, add the citation or tag it `[?]` first.
6. **Anti-circularity**: a test must not compare a kernel against an oracle derived from that kernel's own helper — that proves self-consistency, not correctness. New graph tests compare against an independently written reference.
7. **No second graph**: the pre-rewrite graph is gone (`AEON_ENABLE_LEGACY_V4_GRAPH` and
`core/v4_pipeline.hpp` were deleted 2026-09-18; recover them from git history or `main` if a
specific old value must be re-derived). There is exactly one graph, and a re-introduced legacy
path is a defect, not a convenience.
8. **Empirical Milestone Logging**: For every significant milestone or architectural transition, log the exact test conditions, throughput (tok/s), latencies (TTFT, decode step ms), and cache metrics in [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md). Do not log noise for small code edits; log meaningful, comparable system-level milestones to provide clear before-and-after tracking on the path to production.
9. **Modular, Scalable and Maintainable**: avoid growing monolithic files with mixed concerns, extract those concerns in separate smaller and focused modules, reuse and improve existing modules, avoid duplications and redundancies.