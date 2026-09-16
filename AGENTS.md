# AGENTS.md — Project Aeon System & Agent Context

## 1. Project Purpose & High-Level Context
**Project Aeon** is a high-performance, bare-metal Mixture-of-Experts (MoE) inference engine built from scratch in C++20 and native HIP for consumer AMD hardware (primarily RDNA3 / `gfx1100`, scalable across multi-GPU rigs).

Aeon solves the memory wall for massive MoE models (e.g., DeepSeek-V4 architectures) on consumer workstations by combining:
1. **Bare-metal RDNA3 execution**: Wave32 execution mode, AI Matrix Accelerators (WMMA), and direct HIP/AMDGCN instruction dispatch without CUDA or framework overhead.
2. **Three-Tier Hierarchical Storage & Streaming**: VRAM (Hot LRU) $\leftarrow$ Host DDR (Warm pinned staging) $\leftarrow$ NVMe SSD (Cold asynchronous Direct I/O via Linux `io_uring` with 4KB sector alignment).
3. **Double-Buffered Expert-Batched Prefill & Pipeline Parallelism**: Eliminating I/O latency stalls by overlapping compute with asynchronous DMA/storage transfers.

---

## 2. Documentation and References
Use [Documentation Status](plans-and-docs/status/DOCUMENTATION_STATUS.md) for the current plan inventory, open gates, and historical-document boundaries. It is the navigation point; read it before any other planning document.

**The project is rewriting its DeepSeek-V4 inference graph.** The storage, streaming, artifact-format, and kernel layers are kept; the graph that composes them is being rebuilt on branch `rewrite/graph-v2`. The specification is:
- [Inference Pipeline Plan](plans-and-docs/analysis/current/inference_pipeline_plan.md): **the authority for graph semantics.** Evidence-tagged (`[V]`/`[?]`/`[I]`), cited step-by-step procedure with per-step gates, a reference hierarchy, and an anti-circularity rule. Every claim is cited or flagged unverified.
- [Checkpoint & Artifact Integrity Plan](plans-and-docs/analysis/current/checkpoint_verification_plan.md): proves the **input** is sound (structural audit, prompt-encoder oracle, repack round-trip, streaming integrity) so a graph failure is a graph failure. Companion to the specification; it does not cover the graph.

Current execution records:
- [Phase 2 Execution Plan](plans-and-docs/execution/active/PHASE_2_EXECUTION_PLAN.md): paused single-GPU Hot/Warm/Cold runtime and remaining cold-tier work.
- [Native Text-In/Text-Out Plan](plans-and-docs/execution/active/TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md): native frontend status and correctness gates.
- [Routing Profile Study](plans-and-docs/execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md): profiler contract and placement-study gates.
- [Backend Generalization Execution Plan](plans-and-docs/execution/active/BACKEND_GENERALIZATION_EXECUTION_PLAN.md): descriptor-driven artifact and expert-supply boundary; manifest and second-backend gates.
- [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md): authoritative silicon measurements. **Read its banner before comparing any `E2E` entry** — pre-rewrite model-path measurements are marked invalid.

Superseded records (retained for chronology; do not execute):
- [Model Correctness Execution Plan](plans-and-docs/execution/superseded/MODEL_CORRECTNESS_EXECUTION_PLAN.md): the staged in-place repair approach, replaced by the Inference Pipeline Plan.

Completed execution records:
- [Warm-Tier Repair and Supply Telemetry Plan](plans-and-docs/execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md): persistent Warm ownership, asynchronous refill, and source-tier telemetry; see the [closure report](plans-and-docs/execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_AB_REPORT.md).

Completed plans and historical rationale remain available through the status index. Do not use an old checklist or review as current implementation evidence.

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
*Status: 2026-09-15, branch `rewrite/graph-v2`. Keep this summary current; put detailed measurements and historical execution notes in the linked documents.*

### Present: the graph rewrite
- [ ] **Rewrite the DeepSeek-V4 inference graph** from [Inference Pipeline Plan](plans-and-docs/analysis/current/inference_pipeline_plan.md). The storage, streaming, artifact, and kernel layers are **kept**; the graph that composes them is rebuilt. An audit found structural graph errors — a missing Hyper-Connections comb scale (`hc_scale[2]`), a missing compressor APE term (`score += ape[pos % ratio]`), and HCA layers running indexer selection they do not have — plus tests that could not catch them because they were circular.
- [x] **Specification research complete (phases 0.1–0.2f).** The whole forward pass is re-cited against readable reference code with an evidence convention and a reference hierarchy: Hyper-Connections, MLA/sink, both RoPE bases, compressor window + APE + store quantization, indexer scope, router bias placement, attention composition (local + compressed row-sets), prompt encoding, embedding, LM head, sampling. 33 known traps recorded.
- [x] **Build separated and gated.** CMake split into focused modules; `AEON_ENABLE_LEGACY_V4_GRAPH` (default OFF) gates every target that depends on the pre-rewrite graph, so the old tests cannot be compiled or run by accident.
- [x] **Obsolete tests removed.** The pre-rewrite layer-schedule tests and the circular kernel checks were deleted; the two independent-oracle parity tests stay gated as tier-0 anchors. Default `ctest`: 30 tests; legacy: 32.
- [x] **Step 0 — prompt encoding verified.** The artifact ships its own encoder plus four golden vectors; it is ported into `text/dsv4_prompt_encoder.{hpp,cpp}` and now reproduces all four **byte-for-byte** (was 1/4 before the port). One implementation: `Dsv4ChatFormatter` delegates to the encoder.
- [x] **Tier 1 oracle harness + all eleven certified primitives — COMPLETE.** `reference/dsv4_oracle.hpp` is the host-only, fp64, kernel-free reference layer (RMSNorm, the two-class RoPE spec, dense projections, MLA, Hyper-Connections, attention + sink, the compressor + APE, the indexer + top-k + Hadamard, the grouped output projection, the MoE router, the swizzled W4A16 expert format + clamped SwiGLU, and the dense shared-expert FFN). `kernels/v4_norm.hpp`, `kernels/v4_rope.hpp`, and `kernels/v4_gemv.hpp` are kept primitives extracted so each can be gated on its own; `v4_attention.hpp` includes all three. Each gate runs the oracle's own self-checks first, states an explicit tolerance, and asserts discriminating properties rather than only closeness; gates 12–15 earn their results by construction — the grouped projection by perturbing one group and requiring the other seven to stay bit-identical, the router by showing all four of its traps load-bearing, the routed expert by showing the zero point, the nibble permutation and the asymmetric clamp each load-bearing before re-checking on real artifact bytes, and the shared expert by measuring the combine on the device. **Gate 14 partially settled** (32 identical 6-expert atomic accumulations are bit-identical; bounds one configuration, not the order in general). **Four real findings, none of which came from review:** a transposed comb index in this plan's 2.0 prose (trap 34); a **missing ReLU in `v4_indexer_scores_kernel`** (trap 11) that had survived a real-weight parity test because that test never called the kernel; the plan's claim that we implement the fused router normalization guard (we use a fourth form, `Σ+1e-20`, inert below a logit of −96); and the combine-order claim, which describes only the reference's *unfused* path while ours mirrors its fused one. Full detail in [Inference Pipeline Plan](plans-and-docs/analysis/current/inference_pipeline_plan.md) §2.9–2.10.
- [x] **Mutation testing of the Tier 1 gates.** For each certified property, the specific wrong variant was injected into the *kernel* and the gate required to go red. Ten mutations: **eight killed, one provably equivalent, zero unclassified.** Two gate defects were found that a green suite and code review had both missed — the clamp-rule gates could not see a symmetrically-clamped kernel (in either the standalone or the fused form), and the RMSNorm gate could not see a deleted `eps`. Both gates are repaired and both mutations now fail. This corrects a claim the item-14 gate report made: it is recorded in the plan, not quietly fixed. The equivalent mutation verifies trap 35 by experiment. Procedure recorded in the plan under "Mutation testing — the second rule".
- [ ] **Next: Tier 2 item 17 — the CSA layer class, then HCA.** Item 16 is done: `core/v4_layer_body.hpp` is the single layer body (Steps 2.0–2.11, one token), `reference/dsv4_oracle.hpp::layer_sliding_body` is its composed fp64 reference, and `tests/test_v4_layer_body_oracle.cpp` gates the composition on the artifact's real `layers.0` weights across ten positions. Every checkpoint lands within `1.1e-3` of its own peak (the fp16 store and nothing else). Five wiring mutations were injected: four killed, one shown redundant. The gate also found a defect **in its own oracle** — a fp16 tensor widened arithmetically instead of decoded, now fixed and guarded by a fixture check. Two findings recorded: **trap 36** (position 0 makes every RoPE the identity, so a single-token gate is blind to base class, rotation and its inverse) and the MoE accumulation clear being redundant. Full detail in [Inference Pipeline Plan](plans-and-docs/analysis/current/inference_pipeline_plan.md) §Part V Tier 2.
- [ ] **Tier 2 remains:** the CSA and HCA layer classes (item 17), then serial multi-token decode across compressor boundaries (item 18). The layer body is deliberately **not** wired into `core/v4_pipeline.hpp`: that file is the pre-rewrite graph, gated off, and the rewrite must not depend on it. The body takes its RoPE tables, an attention-trace observer, and a routed-expert executor as parameters, so the same code serves decode, batched prefill and the gate.

### Superseded (retained for chronology, do not execute)
- [Model Correctness Execution Plan](plans-and-docs/execution/superseded/MODEL_CORRECTNESS_EXECUTION_PLAN.md) — Stages 0–6 ran against the runtime and produced useful evidence (contract parsing, INT4 parity, CPU oracles, class-aware device state, serial dispatch), but the staged in-place approach could not catch the structural errors above. Its Stage 7 is not resumed.

### Completed and still valid
- [x] Phase 0–2 foundations: single-GPU runtime gates; lossless Safetensors-to-`.aeon` conversion with sector-aligned indexing and bit-exact verification; dynamic Hot/Warm/Cold storage, direct `io_uring`, asynchronous SDMA staging, residency management.
- [x] Modular runtime ownership: pipeline operations, scratch, layer state, V4 model resources, and architecture-neutral `TieredExpertSupply`; V4 dense binding; opaque descriptor-driven payload storage; infrastructure/architecture/backend/platform source boundaries.
- [x] Artifact and backend contracts: versioned manifest/sidecar validation, backend selection, swizzled-view guards.
- [x] Native text front end: tokenizer, DSV4 formatting, EOS-aware generation, detokenization, `aeon_chat`, and resumable routing-profile aggregation.
- [x] Stage 1 expert-kernel path: version-2 swizzled W4A16 artifacts, Wave32 GEMV, fused W1/W3 and W2 kernels, native conversion, and silicon validation. See the [implementation report](plans-and-docs/analysis/historical/EXPERT_KERNELS_REVIEW_stage-1_IMPLEMENTATION_REPORT.md).
- [x] Warm-tier repair and supply telemetry: persistent Warm ownership, event-ordered refill, lazy/eager preload, transactional transfer cleanup, pinned fallback, source-tier JSONL telemetry, and controlled silicon A/B. See [the closure report](plans-and-docs/execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_AB_REPORT.md).

### Paused work
- [ ] **Phase 2 continuation:** broad cold-tier, storage-layout, placement, and latency-hiding work remains paused while the graph rewrite and the 35 GiB host-pressure tradeoff are characterized. The Phase 2 document remains the historical execution record for its completed spikes and open gates.

### Open gates
- [ ] **Graph correctness:** the oracle and gate harness plus all eleven Tier-1 primitives are done; what remains is the plan's next tier — compose the primitives into the layer body and gate the composition, then compare formatted inputs, intermediate checkpoints, and final logits against a trusted compatible reference before any placement work.
- [ ] **Four measurement gates** (specified in the plan; settled on silicon, not by reading): ~~indexer Hadamard rotation~~ (**closed — do not apply it**); KV fp8/E4M3 vs bf16 storage delta; MoE routed-expert accumulation order (**partially settled**: 32 identical 6-expert `atomicAdd` accumulations are bit-identical and the sum matches the weighted per-expert sum to `max_rel < 5e-7`, which bounds one configuration but not the effect of the order in general); local-window prefix-reuse boundary behaviour. See the ledger banner for why pre-rewrite numbers cannot be compared.
- [ ] **Cold-tier performance:** characterize cold-cache and steady-state behavior, reduce host-memory pressure, improve physical `.aeon` placement, and test whether the exposed just-in-time miss path needs a new scheduling or CPU-fallback design. The model-backed `>= 6.0 GB/s` target remains open.
- [ ] **Swizzled full-model performance:** validate representative 43-layer generation under controlled Hot/Warm conditions, collect useful rocprof counters, and tune occupancy/register pressure beyond the isolated six-expert benchmarks. A residency invariant violation would produce invalid/stale results or a device memory fault rather than trigger an automatic fallback.
- [ ] **Kernel quality:** the three audit-flagged gaps are now audited and certified at Tier 1 — the HC comb scale (the plan's prose was wrong, not the kernel), the compressor APE, and indexer scope — plus a fourth the gates found on their own, the indexer ReLU. Two more certified properties are structural rather than numeric: the grouped output projection's per-group reduction, shown load-bearing by perturbing one group and requiring the other seven to stay bit-identical; and the shared expert's combine, measured on the device to apply the shared contribution exactly once. What remains unprofiled or conservative: the batched prefill path, the ordered routed-expert path (note its fp16 accumulator, which is *less* accurate than the atomic path's fp32 one), and the fused atomic W2 path.
- [ ] **Routing placement study:** collect representative profile and held-out corpora with the verified text contract, then evaluate frequency-informed placement against dynamic LRU.
- [ ] **Backend specialization:** add an explicit backend factory, semantic dispatch, and a second working weight backend before introducing a universal V4 linear-dispatch abstraction.
- [ ] **Phase 3:** multi-GPU pipeline parallelism and 1F1B scheduling remain future work.

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
4. **Maintenance of AGENTS.md**: Update the "Progress Tracking & State of Execution" section whenever milestones or micro-steps transition between Past, Present, and Future.
5. **The specification is authoritative**: For DeepSeek-V4 graph semantics, [Inference Pipeline Plan](plans-and-docs/analysis/current/inference_pipeline_plan.md) governs. Do not implement a graph op from memory, from this file, or from an unsourced reference. If the plan lacks a citation for something being implemented, add the citation or tag it `[?]` first.
6. **Anti-circularity**: a test must not compare a kernel against an oracle derived from that kernel's own helper — that proves self-consistency, not correctness. New graph tests compare against an independently written reference.
7. **The legacy graph is gated**: `AEON_ENABLE_LEGACY_V4_GRAPH` (default `OFF`) controls the pre-rewrite graph, its tests, and its tools. Leave it off. Enable it only to re-derive a specific value from the old path, and never treat a green legacy run as coverage.
8. **Empirical Milestone Logging**: For every significant milestone or architectural transition, log the exact test conditions, throughput (tok/s), latencies (TTFT, decode step ms), and cache metrics in [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md). Do not log noise for small code edits; log meaningful, comparable system-level milestones to provide clear before-and-after tracking on the path to production.
9. **Modular, Scalable and Maintainable**: avoid growing monolithic files with mixed concerns, extract those concerns in separate smaller and focused modules, reuse and improve existing modules, avoid duplications and redundancies.