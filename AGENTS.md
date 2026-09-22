# AGENTS.md — Project Aeon System & Agent Context

## 1. Project Purpose & High-Level Context

**Project Aeon** is a bare-metal Mixture-of-Experts inference engine in C++20 and native HIP for consumer AMD RDNA3 (`gfx1100`), scalable across multi-GPU rigs. It attacks the memory wall for very large MoE models on consumer workstations with three mechanisms: **bare-metal RDNA3 execution** (Wave32, WMMA, direct HIP/AMDGCN dispatch — no CUDA, no framework overhead); a **three-tier hierarchy** (VRAM Hot ← host-DDR Warm ← NVMe Cold, via `io_uring` `O_DIRECT` at 4 KiB alignment); and **double-buffered expert-batched prefill** overlapped with asynchronous DMA/storage.

---

## 2. Documentation and References

**Navigation** [Documentation Status](plans-and-docs/status/DOCUMENTATION_STATUS.md) owns the plan inventory, the document map (live / active / completed / superseded), the open gates, and the rules for historical documents. **This file does not duplicate that inventory.**

### Local Reference Implementations

The primary external source references are maintained as shallow, default-branch checkouts outside this repository. They are for source comparison only, not Aeon build or runtime dependencies:

- [llama.cpp](../aeon-references/llama.cpp): portable runtime, expert streaming, quantization, KV state, and serving paths.
- [FreeToken](../aeon-references/freetoken): bandwidth-adaptive CPU/GPU execution, expert caching, prefill streaming, and agent-facing serving.
- [Colibri](../aeon-references/colibri): VRAM/RAM/NVMe tiering, routing-aware placement, direct I/O, prefetch, and persistent KV state.
- [DwarfStar (ds4)](../aeon-references/ds4): DeepSeek-V4-specific kernels, profile-derived expert hotlists, SSD streaming, KV/prefix caching, and native agent serving.
- [vLLM](../aeon-references/vllm): paged memory, prefix/KV caching, scheduling, resource management, and production serving. **Also the canonical DeepSeek-V4 prompt encoder** — `vllm/tokenizers/deepseek_v4_encoding.py::encode_messages` — since the checkpoint ships no `chat_template`.
- [SGLang](../aeon-references/sglang): radix/HiCache, chunked prefill, MoE scheduling, disaggregation, and AMD paths. Its readable `srt/layers/attention/dsv4/**` and `kernels/ops/attention/dsv4/**` are model-specific and are the preferred arbiter for DSV4 attention and indexer semantics.

Update a reference checkout with `git -C <directory> pull --ff-only` and record its commit SHA whenever an implementation decision depends on a specific revision.

---

## 3. Progress Tracking & State of Execution

*Status: 2026-09-18.*

### Past

Oldest to newest. Each new milestone fuses the two older rows into one, so the top row stays a compaction of the earliest work.

| Milestone | State | Detail |
| :--- | :--- | :--- |
| Phase 0–2: foundations, primitives, three-tier storage stack (oldest rows, fused) | complete | [Phase 0](plans-and-docs/execution/completed/PHASE_0_EXECUTION_PLAN.md), [Phase 1](plans-and-docs/execution/completed/PHASE_1_EXECUTION_PLAN.md), [Phase 2](plans-and-docs/execution/completed/PHASE_2_EXECUTION_PLAN.md), [Warm-tier repair](plans-and-docs/execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md) |
| Native text path: tokenizer, prompt formatter, EOS-aware generation, CLI | complete | [TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md](plans-and-docs/execution/completed/TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md) |
| Specification and oracle harness: Tier 0 research, Step 0 prompt encoding, Tier 1 primitives, Step 3 `hc_head` | verified | [DSV4 Inference Pipeline Plan](plans-and-docs/execution/completed/DSV4_INFERENCE_PIPELINE_PLAN.md) §Tier 0–1, §Step 0, §Step 3 |
| Graph rewrite: Tiers 2–4 and composition P0–P4, the 43-layer text-in/text-out path | acceptance criterion met | [DSV4 Graph Composition Plan](plans-and-docs/execution/completed/DSV4_GRAPH_COMPOSITION_PLAN.md) §7; pipeline plan §Tier 2–4; ledger M28 |

### Present

| Work in progress | State | Detail |
| :--- | :--- | :--- |
| Expert streaming and chunked prefill | in progress | [EXPERT_STREAMING_EXECUTION_PLAN.md](plans-and-docs/execution/active/EXPERT_STREAMING_EXECUTION_PLAN.md) (Steps 0–5 done; Step 6 core and item 4/D4 done — Warm-frozen and the double-buffered sweep remain) |
| Session state and swap | open | [SESSION_STATE_AND_SWAP_ANALYSIS.md](plans-and-docs/analysis/current/SESSION_STATE_AND_SWAP_ANALYSIS.md) |
| Host-memory pressure | open investigation | [HOST_MEMORY_PRESSURE_INVESTIGATION.md](plans-and-docs/analysis/current/HOST_MEMORY_PRESSURE_INVESTIGATION.md) |
| Routing profile and placement study | open | [ROUTING_PROFILE_AND_PLACEMENT_STUDY.md](plans-and-docs/execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md) |
| Backend generalization: factory and second backend | open gates | [BACKEND_GENERALIZATION_EXECUTION_PLAN.md](plans-and-docs/execution/active/BACKEND_GENERALIZATION_EXECUTION_PLAN.md) |

### Future

Directions we have explicitly defined as future — researched or discussed, not yet admitted to Present.

| Direction | Why it waits | Detail |
| :--- | :--- | :--- |
| Prefix matcher (block table, cache key, radix search, eviction) | needs an assembled graph and a fork workload | composition plan §9; session analysis |
| MTP / DSpark draft head (`num_nextn_predict_layers=1`) | speculative decoding, not the base forward pass | composition plan §9; pipeline plan §Tier 0.2e |
| Multi-GPU pipeline parallelism (Phase 3) | out of scope for this revision | composition plan §9; [vision](plans-and-docs/reference/strategy/PROJECT_AEON_VISION.md) |
| Tool use beyond prompt encoding | frontend work; needs the logit-processor seam plus a tool workload | composition plan §9 |
| KV fp8/E4M3 versus bf16 store | a delta that must be measured, not assumed | pipeline plan Gates 9/10 |
| Expert-placement policy (routing-aware hotlists) | scheduling optimization, not a correctness requirement | expert streaming analysis |
| Throughput targets (chunked-prefill amortization) | speed and correctness are separate gates | expert streaming analysis |

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
3. **Commit Messages**: Follow conventional commits format (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`).
4. **Maintenance of AGENTS.md**: Update the "Progress Tracking & State of Execution" section whenever milestones transition. This is an entry point, not a record: state a milestone as one row and point at the document that owns its detail. When a milestone advances, **change its row — do not append a narrative**. Detail added here is duplication that will drift.
5. **Anti-circularity**: A test must not compare a kernel against an oracle derived from that kernel's own helper — that proves self-consistency, not correctness. New graph tests compare against an independently written reference.
6. **Empirical Milestone Logging**: For every significant milestone or architectural transition, log the exact test results in the [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md). Do not log noise for small code edits; log meaningful, comparable system-level milestones to provide clear before-and-after tracking on the path to production.
7. **Modular, Scalable and Maintainable**: Avoid growing monolithic files with mixed concerns. Extract them into separate, focused modules; reuse and improve existing ones; avoid duplication and redundancy.
8. **Avoid Running The Entire Test Suite**: running the entire test suite takes many minutes and should be avoided unless many breaking-changes were applied. Run only the very necessary tests selectively depending on the changes that were applied.
