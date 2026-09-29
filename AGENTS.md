# AGENTS.md — Project Aeon System & Agent Context

## 1. Project Purpose & High-Level Context

**Project Aeon** is a bare-metal Mixture-of-Experts inference engine in C++20 and native HIP for consumer AMD RDNA3 (`gfx1100`), scalable across multi-GPU rigs. It attacks the memory wall for very large MoE models on consumer workstations with three mechanisms: **bare-metal RDNA3 execution** (Wave32, WMMA, direct HIP/AMDGCN dispatch — no CUDA, no framework overhead); a **three-tier hierarchy** (VRAM Hot ← host-DDR Warm ← NVMe Cold, via `io_uring` `O_DIRECT` at 4 KiB alignment); and **double-buffered expert-batched prefill** overlapped with asynchronous DMA/storage.

---

## 2. Documentation and References

**Status** [PROJECT_STATUS.md](plans-and-docs/status/PROJECT_STATUS.md) is the **single progress-tracking document** — what is done, in flight, open and future. Read it for state. **This file owns stable context only** (purpose, references, engineering rules) and is not updated per milestone.

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

## 3. Project Rules & Engineering Conventions

### Code Architecture & Runtime

1. **Production Runtime Zero-Dependency Policy**: The inference runtime engine must remain pure C++20 / native HIP with no runtime dependencies on Python or PyTorch.
2. **Standard Ecosystem Compatibility**: Never invent custom lossy quantization formats. Ingest standard community formats (GGUF, Safetensors).
3. **Hardware Precision Discipline**: Golden reference tests must verify that GPU GEMM / dequant kernels match reference precision within standard FP16/BF16 tolerances ($\epsilon < 10^{-3}$).
4. **Direct I/O Discipline**: All streaming file reads must be strictly 4096-byte aligned (`O_DIRECT` compliant) to eliminate kernel page-cache contention and buffer copies.
5. **Comment Discipline**: A comment explains *why* the code is the way it is — the constraint, the hazard, the measurement that forced it — and nothing else.
   - **Never reference a planning artifact.** No execution plan, step, stage, tier, phase, gate label, trap/item number, milestone id, ledger entry, spike, or any other identifier minted by a planning document. Such a reference reads as noise (it usually does not even name its source) and it is stale the moment the code is refactored or moved, which is continuously. Explain the concept directly instead.
   - **Delete stale comments; do not preserve them.** If a comment no longer matches the code, remove it. A comment that has drifted is worse than no comment, because it is trusted.
   - **Compact what remains.** Keep the reasoning, cut the length: same information in markedly fewer lines. Prefer one sentence to a paragraph, and one paragraph to a section. Comments are read far more often than they are written, so their length is a recurring cost.
   - **Do not restate the code.** If the next line already says it, the comment is not needed.
6. **Concern Separation (G1–G4)**: Every file under `src/` belongs to exactly one concern, and dependencies point one way only. A file changes for exactly one reason, and it lives in the group that owns that reason.
   - **G1 Engine — `src/infrastructure/`**: the engine's own strategy — budget allocation, cache and supply policy, prefill strategy, sampling, streams, the text loop, routing instrumentation. It changes when *we* make a design decision, and it is **model-agnostic**: G1 must never include G2's or G4's model types (`src/architecture/`, model-specific symbols).
   - **G2 GPU architecture — `src/platform/`**: how an operation is *executed* on a specific GPU — wave width, lane mapping, tile shape, intrinsics. It changes with the GPU.
   - **G3 Weight format — `src/backend/`**: how weights are *stored and decoded*. It changes with the quant/artifact format, independently of the GPU.
   - **G4 Model architecture — `src/architecture/`**: what the model *computes* — attention sinking, Hyper-Connections, MLA, the router rule, RoPE, the tokenizer. It changes with the checkpoint.

   **Dependency direction** (verified from the `#include` graph): `G4 → G3/G2`, `G1 → G3/G2`; G2 and G3 depend on nothing above them. When adding or extending code, put it in the group whose *change reason* matches. **Cross a boundary only through a neutral seam** — an interface, or a scalar/shape parameter the caller supplies — never by pulling a lower group's model types upward. If a change would make `infrastructure/` include `architecture/`, or would put a model-specific name or constant into `platform/` or `backend/`, the design is wrong: introduce the seam instead. A move, a split or a rename is its own step with no behaviour change, and it must not carry a "while I'm here" edit.

### Development Process & Git Conventions

1. **Incremental Micro-Steps**: Advance through small, verifiable steps. Never implement broad abstractions before underlying hardware primitives are verified on silicon.
2. **Hardware-Grounded Verification**: Test and benchmark on physical hardware (`gfx1100`) at every step.
3. **Commit Messages**: Follow conventional commits format (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`).
4. **Maintenance of PROJECT_STATUS.md**: [PROJECT_STATUS.md](plans-and-docs/status/PROJECT_STATUS.md) is the **single progress-tracking document** and the one that changes as work advances. Update its §3 whenever milestones transition, and keep it the same size: state a milestone as one row, **change its row rather than appending a narrative**, and let the oldest rows fuse so **Past** stays a compaction rather than a log. Detail added there is duplication that will drift. **This file (AGENTS.md) is stable** — change it only when the purpose, the references, or these rules change.
5. **Anti-circularity**: A test must not compare a kernel against an oracle derived from that kernel's own helper — that proves self-consistency, not correctness. New graph tests compare against an independently written reference.
6. **Empirical Milestone Logging**: For every significant milestone or architectural transition, log the exact test results in the [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md). Do not log noise for small code edits; log meaningful, comparable system-level milestones to provide clear before-and-after tracking on the path to production.
7. **Modular, Scalable and Maintainable**: Avoid growing monolithic files with mixed concerns. Extract them into separate, focused modules; reuse and improve existing ones; avoid duplication and redundancy.
8. **Avoid Running The Entire Test Suite**: running the entire test suite takes many minutes and should be avoided unless many breaking-changes were applied. Run only the very necessary tests selectively depending on the changes that were applied.
