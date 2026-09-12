# Project Aeon Documentation Status

Status audited on 2026-09-11 against `HEAD` (`b7998dc`, `feat: complete warm-tier refill and supply telemetry`) and the current source tree.

This file is the navigation point for project state. Detailed benchmark numbers belong in [PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md); design rationale belongs in the reference and vision documents; this file and [AGENTS.md](../../AGENTS.md) should stay concise.

## Directory layout

- `status/`: project navigation, current runtime map, and the living benchmark ledger.
- `execution/active/`: plans with open acceptance gates or remaining implementation work.
- `execution/completed/`: implementation plans whose execution gates are complete; unresolved correctness or product gates are called out from the active plans.
- `analysis/current/`: current technical conclusions and reviews that guide the next measurement or implementation decision.
- `analysis/historical/`: completed reviews and design analyses retained for rationale and chronology.
- `reference/strategy/`: vision, risk, and specialization documents.
- `reference/prior-art/`: external research and comparative implementation studies.

## Execution documents

| Document | State | Purpose |
| --- | --- | --- |
| [WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md](../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md) | Complete | Persistent Warm ownership, asynchronous refill, transfer safety, corrected demotion accounting, source-tier telemetry, and five-run silicon A/B are complete; configured capacity is eager while preload publication is content-lazy; see the [closure report](../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_AB_REPORT.md). |
| [PHASE_2_EXECUTION_PLAN.md](../execution/active/PHASE_2_EXECUTION_PLAN.md) | Paused | Paused single-GPU cold-tier, storage-layout, and memory-pressure work. Spikes 0-2 are complete; Spike 3 has a bounded implementation but open acceptance gates. |
| [TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md](../execution/active/TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md) | Open | Native text path is implemented; external behavioral comparison and longer-context attention correctness remain open. |
| [ROUTING_PROFILE_AND_PLACEMENT_STUDY.md](../execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md) | Open; evidence gated | Routing observer and durable profiler are implemented; representative profile/held-out data and placement evaluation remain gated by model correctness. |
| [BACKEND_GENERALIZATION_EXECUTION_PLAN.md](../execution/active/BACKEND_GENERALIZATION_EXECUTION_PLAN.md) | Open; foundation implemented | Descriptor-driven native artifacts, opaque expert supply, and current-backend guarding are implemented; manifest, dense binding, and second-backend gates remain open. |
| [PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md) | Living record | Authoritative silicon results, regressions, and milestone measurements. |
| [CODEBASE_MAP.md](CODEBASE_MAP.md) | Current map | Describes the production runtime, validation targets, and legacy diagnostics. |

## Current analysis

| Document | State | Purpose |
| --- | --- | --- |
| [EXPERT_SUPPLY_CHAIN_AND_ROLLING_RESIDENCY_ANALYSIS.md](../analysis/current/EXPERT_SUPPLY_CHAIN_AND_ROLLING_RESIDENCY_ANALYSIS.md) | Current analysis | Records the registry, Hot/Warm/Cold residency findings and the rationale for the deferred rolling-residency direction. Implementation sequencing is recorded in the completed Warm-tier plan. |
| [DEEPSEEK_V4_FLASH_AEON_COMPARISON.md](../analysis/current/DEEPSEEK_V4_FLASH_AEON_COMPARISON.md) | Current analysis | Compares the selected 0731 checkpoint contract with Aeon quantization, attention, runtime, and hardware behavior. |
| [GPTQ_AEON_PARALLEL_BACKEND_ANALYSIS.md](../analysis/current/GPTQ_AEON_PARALLEL_BACKEND_ANALYSIS.md) | Current analysis | Records the feasibility, shared-versus-specialized boundary, artifact strategy, and performance gates for a parallel GPTQ-Aeon backend. |
| [VLLM_RDNA3_DEEPSEEK_V4_REFERENCE_ANALYSIS.md](../analysis/current/VLLM_RDNA3_DEEPSEEK_V4_REFERENCE_ANALYSIS.md) | Current reference analysis | Maps the local vLLM gfx1100 GPTQ/W4A16, fused MoE, DeepSeek-V4 prefill, sparse-attention, indexer, and KV-cache sources to Aeon reuse and adaptation decisions. |
| [EXPERT_KERNELS_REVIEW.md](../analysis/current/EXPERT_KERNELS_REVIEW.md) | Current review | Records the pending kernel-geometry, quant-layout, bottleneck, and interface questions for the next kernel investigation. |
| [EXPERT_KERNELS_REVIEW_stage-1_IMPLEMENTATION_REPORT.md](../analysis/historical/EXPERT_KERNELS_REVIEW_stage-1_IMPLEMENTATION_REPORT.md) | Historical implementation report | Compares the external Stage 1 proposal with the implemented version-2 path and records the measured GPU-side effects that led to the v2-only promotion. |

## Completed execution records

- [PHASE_0_EXECUTION_PLAN.md](../execution/completed/PHASE_0_EXECUTION_PLAN.md) is complete for the build, hardware, I/O, overlap, and toy-cache foundations.
- [PHASE_1_EXECUTION_PLAN.md](../execution/completed/PHASE_1_EXECUTION_PLAN.md) is complete for the implemented single-GPU runtime gates. It is not a claim that the full model correctness gate is closed: compressed/indexed attention for layers 2-42 and external parity remain open in the text plan.
- Phase 2 Spike 0 (lossless `.aeon` repacking), Spike 1 (budgeting and unified hot pool), and Spike 2 (asynchronous SDMA overlap) are complete. Their evidence is summarized in [PHASE_2_EXECUTION_PLAN.md](../execution/active/PHASE_2_EXECUTION_PLAN.md) and the ledger.
- Phase 2 Spike 3 has reached bounded Hot/Warm/Cold integration, direct-I/O population, and a measured 35 GiB Warm profile. The model-backed `>= 6.0 GB/s` target, cold-cache/queue-saturation comparison, host-pressure tuning, and scheduling latency remain open.
- The [Warm-tier repair plan](../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md) is complete. Its repaired default reduced median Decode Cold NVMe bytes by 22.9% versus the demotion-free Warm control across five runs per variant, with identical generated IDs and EOS stop behavior. The second-pass corrections remove the optional-demotion CPU synchronization, remove the undocumented layer filter, distinguish drop reasons, include registered host fallback in pinned accounting, and document the content-lazy/eager-capacity boundary; broader cold-tier and host-pressure work remains open.
- Native tokenizer, DSV4 formatting, EOS-aware generation, and `aeon_chat` are implemented and have passed the simple 43-layer text turn. This is the native milestone, not external model-behavior parity.
- Routing counting, resumable aggregation, rankings, compact summaries, and profile regeneration are implemented. The existing pilot remains a plumbing artifact until the correctness gate passes.
- The version-2 swizzled expert path is now the sole native runtime and conversion path. Loader, pipeline, tests, and active tooling no longer expose the retired baseline format or kernels.

## Open work

1. **Correctness:** compare identical formatted IDs and outputs against a trusted compatible reference; implement and validate compressed/indexed attention for longer-context layers 2-42.
2. **Cold tier:** characterize cold-cache and steady-state behavior, improve physical `.aeon` placement/extent layout, reduce host-memory pressure, and decide whether further scheduling or CPU fallback work is justified by measurements.
3. **Swizzled full-model performance:** characterize representative 43-layer version-2 generation under controlled Hot/Warm conditions, collect useful rocprof counters, and tune occupancy/register pressure beyond the isolated kernel measurements.
4. **Placement study:** collect profile and held-out corpora with the verified text contract, then compare measured placement against dynamic LRU. Do not use the existing pilot for placement decisions.
5. **Scaling:** Phase 3 multi-GPU pipeline parallelism remains future work.

## Historical and reference documents

These documents remain useful, but they are not current checklists:

- [V4_PIPELINE_MODULARIZATION_ANALYSIS.md](../analysis/historical/V4_PIPELINE_MODULARIZATION_ANALYSIS.md) records the pre-refactor duplication analysis and completed extraction decisions.
- [EXPERT_PERFORMANCE_REVIEW.md](../analysis/historical/EXPERT_PERFORMANCE_REVIEW.md) records the M15-era source review and the implemented M16-M20 follow-up steps; use the conclusions document and ledger for current status.
- [EXPERT_PERFORMANCE_REVIEW_CONCLUSIONS.md](../analysis/historical/EXPERT_PERFORMANCE_REVIEW_CONCLUSIONS.md) records the historical latency diagnosis that led to the current Warm-tier and supply-chain sequencing.
- [PROJECT_AEON_VISION.md](../reference/strategy/PROJECT_AEON_VISION.md), [AEON_TECHNICAL_BLOCKERS_AND_RISKS.md](../reference/strategy/AEON_TECHNICAL_BLOCKERS_AND_RISKS.md), and [AEON_SPECIALIZATION_AND_ENTROPY_ANALYSIS.md](../reference/strategy/AEON_SPECIALIZATION_AND_ENTROPY_ANALYSIS.md) preserve strategic rationale and hypotheses.
- [AEON_RESEARCH_AND_INSPIRATION.md](../reference/prior-art/AEON_RESEARCH_AND_INSPIRATION.md) and [REFERENCE_EXPERT_CACHING_AND_COLD_STREAMING_ANALYSIS.md](../reference/prior-art/REFERENCE_EXPERT_CACHING_AND_COLD_STREAMING_ANALYSIS.md) preserve external research and comparative implementation evidence.
