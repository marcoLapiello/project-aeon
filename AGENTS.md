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
- [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md) — authoritative silicon
  measurements. **Read its banner before comparing any `E2E` entry** — pre-rewrite model-path
  measurements are marked invalid, and they describe a *different computation*, not a slower one.

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
*Status: 2026-09-16, branch `rewrite/graph-v2`.*

**This section is an index, not a record.** One row per milestone, pointing at the document that owns
the detail: the [Inference Pipeline Plan](plans-and-docs/analysis/current/inference_pipeline_plan.md) (per-item gate results, traps 1–40, mutation
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
| Tier 4 — items 21–23: streaming/tiering, prefix cache, generating loop | item 21 done; 22–23 not started | plan §Tier 4 |

`core/v4_layer_body.hpp` is the single layer body; decode, chunked prefill and every Tier-2/3 gate
call it. It is deliberately **not** wired into `core/v4_pipeline.hpp`, which is the pre-rewrite graph
and stays behind `AEON_ENABLE_LEGACY_V4_GRAPH`. Default `ctest`: **36 tests** (legacy: 37).

**Next: item 22's state layout.** The compressor's partial state is still a fixed ring, which caps
`run_layer_body_chunk` at a chunk of 8 and thereby blocks item 19's throughput half; Part I §6.2
requires position-addressed state for that *and* for mid-ratio-window reuse. Item 19's other half —
the indexer top-k's per-token host round-trip, target zero — is countable today and needs no baseline.
Reasoning and the full remaining sequence: plan item 19, item 22, and the open-unknowns table.

**Needs a decision — a numbering conflict.** "Item 21" currently names two different gates: the plan's
Tier 4 item 21 is streaming/tiering, while `plans-and-docs/status/DOCUMENTATION_STATUS.md` and older notes use it for the comparison
against a **trusted compatible reference** (the patched-RDNA vLLM run). That comparison is a real,
separately motivated gate but has **no number in the plan's sequence** — give it one before
scheduling it.

### Other states
- **Superseded —** [Model Correctness Execution Plan](plans-and-docs/execution/superseded/MODEL_CORRECTNESS_EXECUTION_PLAN.md):
  the staged in-place repair, replaced by the plan. Retained for chronology; do not execute.
- **Completed and still valid —** Phase 0–2 foundations; modular runtime ownership; artifact and
  backend contracts; native text front end; the Stage-1 expert-kernel path
  ([report](plans-and-docs/analysis/historical/EXPERT_KERNELS_REVIEW_stage-1_IMPLEMENTATION_REPORT.md));
  warm-tier repair and supply telemetry ([closure report](plans-and-docs/execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_AB_REPORT.md)).
- **Paused —** Phase 2 continuation (cold-tier, storage layout, placement, latency hiding), pending
  the graph rewrite and the host-pressure tradeoff.

### Open gates
Pointers only; each is specified in the plan's "Open unknowns" table or the ledger.
- **Graph correctness —** Tier 4 items 22–23 (see *Next* above).
- **Measurement gates —** KV fp8/E4M3 vs bf16 storage; MoE routed-expert accumulation order
  (partially settled); local-window prefix-reuse boundary (half settled by item 20). The indexer
  Hadamard is **settled — do not apply it** (plan Gate 11).
- **Kept infrastructure —** cold-tier characterization, physical `.aeon` placement, host-memory
  pressure, the model-backed `>= 6.0 GB/s` target, kernel occupancy tuning, the routing placement
  study, the explicit backend factory, and Phase 3 multi-GPU.
- **Before comparing any number —** read the ledger banner: pre-rewrite model-path measurements
  describe a *different computation*, not merely a slower one.

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
7. **The legacy graph is gated**: `AEON_ENABLE_LEGACY_V4_GRAPH` (default `OFF`) controls the pre-rewrite graph, its tests, and its tools. Leave it off. Enable it only to re-derive a specific value from the old path, and never treat a green legacy run as coverage.
8. **Empirical Milestone Logging**: For every significant milestone or architectural transition, log the exact test conditions, throughput (tok/s), latencies (TTFT, decode step ms), and cache metrics in [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md). Do not log noise for small code edits; log meaningful, comparable system-level milestones to provide clear before-and-after tracking on the path to production.
9. **Modular, Scalable and Maintainable**: avoid growing monolithic files with mixed concerns, extract those concerns in separate smaller and focused modules, reuse and improve existing modules, avoid duplications and redundancies.