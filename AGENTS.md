# AGENTS.md — Project Aeon System & Agent Context

## 1. Project Purpose & High-Level Context
**Project Aeon** is a high-performance, bare-metal Mixture-of-Experts (MoE) inference engine built from scratch in C++20 and native HIP for consumer AMD hardware (primarily RDNA3 / `gfx1100`, scalable across multi-GPU rigs).

Aeon solves the memory wall for massive MoE models (e.g., DeepSeek-V4 architectures) on consumer workstations by combining:
1. **Bare-metal RDNA3 execution**: Wave32 execution mode, AI Matrix Accelerators (WMMA), and direct HIP/AMDGCN instruction dispatch without CUDA or framework overhead.
2. **Three-Tier Hierarchical Storage & Streaming**: VRAM (Hot LRU) $\leftarrow$ Host DDR (Warm pinned staging) $\leftarrow$ NVMe SSD (Cold asynchronous Direct I/O via Linux `io_uring` with 4KB sector alignment).
3. **Double-Buffered Expert-Batched Prefill & Pipeline Parallelism**: Eliminating I/O latency stalls by overlapping compute with asynchronous DMA/storage transfers.

---

## 2. Documentation and References
Use [Documentation Status](plans-and-docs/status/DOCUMENTATION_STATUS.md) for the current plan inventory, open gates, and historical-document boundaries.

Current execution records:
- [Warm-Tier Repair and Supply Telemetry Plan](plans-and-docs/execution/active/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md): next implementation priority for persistent Warm ownership, asynchronous refill, and source-tier telemetry.
- [Phase 2 Execution Plan](plans-and-docs/execution/active/PHASE_2_EXECUTION_PLAN.md): paused single-GPU Hot/Warm/Cold runtime and remaining cold-tier work.
- [Native Text-In/Text-Out Plan](plans-and-docs/execution/active/TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md): native frontend status and correctness gates.
- [Routing Profile Study](plans-and-docs/execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md): profiler contract and placement-study gates.
- [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md): authoritative silicon measurements.
- [Expert Performance Review Conclusions](plans-and-docs/analysis/historical/EXPERT_PERFORMANCE_REVIEW_CONCLUSIONS.md): historical latency diagnosis and measurement rationale.

Completed plans and historical rationale remain available through the status index. Do not use an old checklist or review as current implementation evidence.

### Local Reference Implementations
The primary external source references are maintained as shallow, default-branch checkouts outside this repository. They are for source comparison only, not Aeon build or runtime dependencies:
- [llama.cpp reference checkout](../aeon-references/llama.cpp): portable runtime, expert streaming, quantization, KV state, and serving paths.
- [FreeToken reference checkout](../aeon-references/freetoken): bandwidth-adaptive CPU/GPU execution, expert caching, prefill streaming, and agent-facing serving.
- [Colibri reference checkout](../aeon-references/colibri): VRAM/RAM/NVMe tiering, routing-aware placement, direct I/O, prefetch, and persistent KV state.
- [DwarfStar (ds4) reference checkout](../aeon-references/ds4): DeepSeek-V4-specific kernels, profile-derived expert hotlists, SSD streaming, KV/prefix caching, and native agent serving.
- [vLLM reference checkout](../aeon-references/vllm): paged memory, prefix/KV caching, scheduling, resource management, and production serving.
- [SGLang reference checkout](../aeon-references/sglang): radix/HiCache, chunked prefill, MoE scheduling, disaggregation, and AMD paths.

Update a reference checkout with `git -C <directory> pull --ff-only` and record its commit SHA whenever an implementation decision depends on a specific revision.

---

## 3. Progress Tracking & State of Execution
*Status: 2026-09-11. Keep this summary current; put detailed measurements and historical execution notes in the linked documents.*

### Completed milestones
- [x] Phase 0 foundations and the Phase 1 single-GPU runtime gates are implemented. Phase 1 remains bounded by the open full-model correctness work described in the text plan.
- [x] Phase 2 Spike 0: lossless Safetensors-to-`.aeon` repacking, sector alignment, indexing, and bit-exact verification.
- [x] Phase 2 Spike 1: runtime feasibility budgeting, unified Hot VRAM pool, residency registry, and silicon validation.
- [x] Phase 2 Spike 2: asynchronous SDMA staging and overlap validation under cold misses.
- [x] Phase 2 Spike 3 bounded integration: direct `io_uring` cold reads, segmented Warm Host storage, Hot/Warm/Cold promotion, and 43-layer regression coverage.
- [x] Pipeline modularization Steps 1-3: extracted pipeline operations/scratch/layer ownership, unified the production expert cache path, and removed the HC/router CPU round trips.
- [x] Native text milestone: tokenizer, DSV4 formatter, EOS-aware generation, detokenization, `aeon_chat`, and a complete simple 43-layer text turn.
- [x] Routing profiler plumbing: optional observation, resumable aggregation, complete rankings, compact summaries, and regeneration mode.
- [x] Stage 1 parallel expert-kernel path: version-2 W4A16 swizzled artifact, vectorized Wave32 GEMV, fused six-expert W1/W3 plus clamped SwiGLU, fused W2 FP32 accumulation, opt-in pipeline/CLI integration, and silicon validation. The version-1 path remains the default; the standalone [Stage 1 implementation report](plans-and-docs/analysis/current/EXPERT_KERNELS_REVIEW_stage-1_IMPLEMENTATION_REPORT.md) records the exact proposal differences. The runtime resolves six complete expert payloads into VRAM and waits for transfer events before launch; the kernels do not handle non-resident experts or storage/cache misses.

### Current priority
- [ ] **Warm-tier repair and supply telemetry:** implement persistent Warm ownership, asynchronous Hot-to-Warm refill, explicit transfer/lease states, and source-tier byte and exposed-wait measurements. Follow [WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md](plans-and-docs/execution/active/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md).

### Paused work
- [ ] **Phase 2 continuation:** broad cold-tier, storage-layout, placement, and latency-hiding work is paused until the focused Warm-tier infrastructure plan closes. The existing Phase 2 document remains the historical execution record for completed spikes and open gates.

### Open gates
- [ ] **Model correctness:** validate compressed/indexed attention for layers 2-42 and compare identical formatted inputs and outputs with a trusted compatible reference before using traces for placement.
- [ ] **Warm-tier foundation:** close the active [Warm-tier repair and supply telemetry plan](plans-and-docs/execution/active/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md) before resuming broader cold-tier performance work.
- [ ] **Cold-tier performance:** after the Warm-tier foundation, characterize cold-cache and steady-state behavior, reduce host-memory pressure, improve physical `.aeon` placement, and test whether the exposed just-in-time miss path needs a new scheduling or CPU-fallback design. The model-backed `>= 6.0 GB/s` target remains open.
- [ ] **Swizzled full-model performance:** validate representative 43-layer generation with the version-2 artifact under controlled Hot/Warm conditions, collect useful rocprof performance counters, and tune occupancy/register pressure beyond the isolated six-expert benchmarks. A residency invariant violation would produce invalid/stale results or a device memory fault rather than trigger an automatic fallback. The baseline version-1 path remains the default fallback.
- [ ] **Routing placement study:** collect representative profile and held-out corpora with the verified text contract, then evaluate frequency-informed placement against dynamic LRU.
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
5. **Empirical Milestone Logging**: For every significant milestone or architectural transition, log the exact test conditions, throughput (tok/s), latencies (TTFT, decode step ms), and cache metrics in [Performance & Accuracy Ledger](plans-and-docs/status/PERFORMANCE_LEDGER.md). Do not log noise for small code edits; log meaningful, comparable system-level milestones to provide clear before-and-after tracking on the path to production.
6. **Strategic codebase searching**: When you need to collect any info from the codebase or search for specific code or entities, use the search subagent tool.
7. **Modular, Scalable and Mantainable**: avoid growing monolitic files with mixed concerns, extract those concerns in separate smaller and focused modules, reuse and improve existing modules, avoid duplications and redundancies.