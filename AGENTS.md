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
- [DSV4 Inference Pipeline Plan](plans-and-docs/execution/completed/DSV4_INFERENCE_PIPELINE_PLAN.md) — **the
  authority for graph semantics.** Evidence-tagged (`[V]`/`[?]`/`[I]`), cited step-by-step procedure
  with per-step gates, a reference hierarchy, and the anti-circularity and mutation-testing rules.
- [DSV4 Graph Composition Plan](plans-and-docs/execution/completed/DSV4_GRAPH_COMPOSITION_PLAN.md) — **the
  composition**: one ordered text-in/text-out path, every step mapped to the component that
  implements it and where it lives, and the definitive list of what does not exist yet. Owns the
  graph's build phases (P1–P4, all built) and the acceptance criterion. It consumes the specification
  and re-derives nothing.
- [Expert Streaming and Chunked Prefill Analysis](plans-and-docs/analysis/current/EXPERT_STREAMING_AND_CHUNKED_PREFILL_ANALYSIS.md) —
  the next work: tiering under miss pressure on the live path, then chunked prefill with a chunk-wide
  expert dispatch. Extracted from the composition plan.
- [Session State and Swap Analysis](plans-and-docs/analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md) —
  the session aggregate, registry, residency seam, cold-tier store and R4. Session swap before the
  prefix matcher. Extracted from the composition plan.
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
*Status: 2026-09-18, branch `rewrite/graph-v2`.*

**This is an index, not a record.** One row per milestone, pointing at the document that owns the
detail: the [DSV4 Inference Pipeline Plan](plans-and-docs/execution/completed/DSV4_INFERENCE_PIPELINE_PLAN.md)
(per-item gate results, traps 1–44, mutation tables), the
[DSV4 Graph Composition Plan](plans-and-docs/execution/completed/DSV4_GRAPH_COMPOSITION_PLAN.md)
(the build phases P0–P4 and the acceptance criterion), the
[Documentation Status](plans-and-docs/status/DOCUMENTATION_STATUS.md) (milestone table, open gates,
document map) and the [Performance Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md) (silicon
measurements). **Do not restate a finding, gate result, tolerance, trap or measurement here — link
to it.**

### The graph rewrite (complete)
Part V of the plan is executed end to end, and so is the composition plan's P0–P4.

| Stage | State | Detail |
| :--- | :--- | :--- |
| Tier 0 — specification research (phases 0.1–0.2f) | done | plan §Tier 0 |
| Step 0 — prompt encoding; Step 3 — `hc_head` | verified | plan §Step 0, §Step 3 |
| Tier 1 — oracle harness + all 11 primitives | done, mutation-tested | plan §Tier 1 |
| Tier 2 — layer body, all three attention classes, serial loop | closed | plan §Tier 2 |
| Tier 3 — chunked prefill, long-context lifecycle | done | plan §Tier 3 |
| Tier 4 — item 21: streaming and tiering | certified | plan §Tier 4 |
| Tier 4 — item 22: state restore (R3) | certified | plan §Tier 4 |
| Tier 4 — item 23: the generating loop | closed by P0–P4 | [composition plan](plans-and-docs/execution/completed/DSV4_GRAPH_COMPOSITION_PLAN.md) §7 |
| P0–P4 — executor seam, head end, 43-layer driver, sampler, text binding | built; acceptance criterion met | composition plan §7 |

`core/v4_layer_body.hpp` is the single layer body; decode, chunked prefill and every Tier-2/3 gate
call it. `aeon_chat`, now in the **default** build, takes a conversation and returns text on the
artifact's real weights through Hot/Warm/Cold — the acceptance criterion. The pre-rewrite graph
(`core/v4_pipeline.hpp`) and its gate were **deleted on 2026-09-18** (~5,600 lines, verified
unreachable by a trial deletion + build). Default `ctest`: **44 tests**.

**Next: tiering under miss pressure, then chunked prefill.** Both are owned by the
[Expert Streaming and Chunked Prefill Analysis](plans-and-docs/analysis/current/EXPERT_STREAMING_AND_CHUNKED_PREFILL_ANALYSIS.md),
cut out of the composition plan on 2026-09-18. Session state and swap is the other extraction, in the
[Session State and Swap Analysis](plans-and-docs/analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md).

### Other states
- **Superseded —** [Model Correctness Execution Plan](plans-and-docs/execution/superseded/MODEL_CORRECTNESS_EXECUTION_PLAN.md):
  the staged in-place repair, replaced by the plan. Retained for chronology; do not execute.
- **Completed and still valid —** Phase 0–2 foundations; modular runtime ownership; artifact and
  backend contracts; native text front end; the Stage-1 expert-kernel path (swizzled layout plus
  fused W1/W3 and W2 kernels, now the default routed-expert path); warm-tier repair and supply
  telemetry ([closure report](plans-and-docs/execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_AB_REPORT.md)).
- **Paused —** Phase 2 continuation (cold-tier, storage layout, placement, latency hiding), pending
  the host-pressure tradeoff.

### Open gates
See [Documentation Status §3](plans-and-docs/status/DOCUMENTATION_STATUS.md) for the single list. The
named non-goals for the near term are the prefix **matcher** and **R4**; session swap needs neither.

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
4. **Maintenance of AGENTS.md**: Update the "Progress Tracking & State of Execution" section whenever milestones or micro-steps transition between Past, Present, and Future. This is an entry point, not a detailed record: state a milestone as one row and point at the document that owns its detail. When a milestone advances, **change its row — do not append a narrative**. Detail added here is duplication that will drift.
5. **The specification is authoritative**: For DeepSeek-V4 graph semantics, [DSV4 Inference Pipeline Plan](plans-and-docs/execution/completed/DSV4_INFERENCE_PIPELINE_PLAN.md) governs. Do not implement a graph op from memory, from this file, or from an unsourced reference. If the plan lacks a citation for something being implemented, add the citation or tag it `[?]` first.
6. **Anti-circularity**: a test must not compare a kernel against an oracle derived from that kernel's own helper — that proves self-consistency, not correctness. New graph tests compare against an independently written reference.
7. **No second graph**: the pre-rewrite graph is gone (`AEON_ENABLE_LEGACY_V4_GRAPH` and
`core/v4_pipeline.hpp` were deleted 2026-09-18; recover them from git history or `main` if a
specific old value must be re-derived). There is exactly one graph, and a re-introduced legacy
path is a defect, not a convenience.
8. **Empirical Milestone Logging**: For every significant milestone or architectural transition, log the exact test conditions, throughput (tok/s), latencies (TTFT, decode step ms), and cache metrics in [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md). Do not log noise for small code edits; log meaningful, comparable system-level milestones to provide clear before-and-after tracking on the path to production.
9. **Modular, Scalable and Maintainable**: avoid growing monolithic files with mixed concerns, extract those concerns in separate smaller and focused modules, reuse and improve existing modules, avoid duplications and redundancies.