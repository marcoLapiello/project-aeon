# Project Aeon Documentation Status

Status audited on 2026-09-10 against `HEAD` (`843f993`, `feat: complete native text turn`) and the current source tree.

This file is the navigation point for project state. Detailed benchmark numbers belong in [PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md); design rationale belongs in the reference and vision documents; this file and [AGENTS.md](../../AGENTS.md) should stay concise.

## Directory layout

- `status/`: project navigation, current runtime map, and the living benchmark ledger.
- `execution/active/`: plans with open acceptance gates or remaining implementation work.
- `execution/completed/`: implementation plans whose execution gates are complete; unresolved correctness or product gates are called out from the active plans.
- `analysis/current/`: current technical conclusions and reviews that guide the next measurement or implementation decision.
- `analysis/historical/`: completed reviews and design analyses retained for rationale and chronology.
- `reference/strategy/`: vision, risk, and specialization documents.
- `reference/prior-art/`: external research and comparative implementation studies.

## Active execution documents

| Document | State | Purpose |
| --- | --- | --- |
| [PHASE_2_EXECUTION_PLAN.md](../execution/active/PHASE_2_EXECUTION_PLAN.md) | Open | Active single-GPU cold-tier, storage-layout, and memory-pressure work. Spikes 0-2 are complete; Spike 3 has a bounded implementation but open acceptance gates. |
| [TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md](../execution/active/TEXT_IN_TEXT_OUT_IMPLEMENTATION_PLAN.md) | Open | Native text path is implemented; external behavioral comparison and longer-context attention correctness remain open. |
| [ROUTING_PROFILE_AND_PLACEMENT_STUDY.md](../execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md) | Open | Routing observer and durable profiler are implemented; representative profile/held-out data and placement evaluation are still gated. |
| [PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md) | Living record | Authoritative silicon results, regressions, and milestone measurements. |
| [CODEBASE_MAP.md](CODEBASE_MAP.md) | Current map | Describes the production runtime, validation targets, and legacy diagnostics. |

## Current analysis

| Document | State | Purpose |
| --- | --- | --- |
| [EXPERT_PERFORMANCE_REVIEW_CONCLUSIONS.md](../analysis/current/EXPERT_PERFORMANCE_REVIEW_CONCLUSIONS.md) | Current analysis | Explains the remaining latency-bound cold-miss problem and the measurements needed before another scheduling policy is added. |
| [EXPERT_KERNELS_REVIEW.md](../analysis/current/EXPERT_KERNELS_REVIEW.md) | Current review | Records the pending kernel-geometry, quant-layout, bottleneck, and interface questions for the next kernel investigation. |
| [EXPERT_KERNELS_REVIEW_stage-1_IMPLEMENTATION_REPORT.md](../analysis/current/EXPERT_KERNELS_REVIEW_stage-1_IMPLEMENTATION_REPORT.md) | Current implementation report | Compares the external Stage 1 proposal with the implemented version-2 path, records measured GPU-side effects, and documents the remaining full-model gate. |

## Completed execution records

- [PHASE_0_EXECUTION_PLAN.md](../execution/completed/PHASE_0_EXECUTION_PLAN.md) is complete for the build, hardware, I/O, overlap, and toy-cache foundations.
- [PHASE_1_EXECUTION_PLAN.md](../execution/completed/PHASE_1_EXECUTION_PLAN.md) is complete for the implemented single-GPU runtime gates. It is not a claim that the full model correctness gate is closed: compressed/indexed attention for layers 2-42 and external parity remain open in the text plan.
- Phase 2 Spike 0 (lossless `.aeon` repacking), Spike 1 (budgeting and unified hot pool), and Spike 2 (asynchronous SDMA overlap) are complete. Their evidence is summarized in [PHASE_2_EXECUTION_PLAN.md](../execution/active/PHASE_2_EXECUTION_PLAN.md) and the ledger.
- Phase 2 Spike 3 has reached bounded Hot/Warm/Cold integration, direct-I/O population, and a measured 35 GiB Warm profile. The model-backed `>= 6.0 GB/s` target, cold-cache/queue-saturation comparison, host-pressure tuning, and scheduling latency remain open.
- Native tokenizer, DSV4 formatting, EOS-aware generation, and `aeon_chat` are implemented and have passed the simple 43-layer text turn. This is the native milestone, not external model-behavior parity.
- Routing counting, resumable aggregation, rankings, compact summaries, and profile regeneration are implemented. The existing pilot remains a plumbing artifact until the correctness gate passes.

## Open work

1. **Correctness:** compare identical formatted IDs and outputs against a trusted compatible reference; implement and validate compressed/indexed attention for longer-context layers 2-42.
2. **Cold tier:** characterize cold-cache and steady-state behavior, improve physical `.aeon` placement/extent layout, reduce host-memory pressure, and decide whether further scheduling or CPU fallback work is justified by measurements.
3. **Stage 1 end-to-end validation:** compare the version-2 swizzled path with the version-1 baseline under controlled Hot/Warm conditions, while preserving the runtime residency invariant and checking broader output parity.
4. **Placement study:** collect profile and held-out corpora with the verified text contract, then compare measured placement against dynamic LRU. Do not use the existing pilot for placement decisions.
5. **Scaling:** Phase 3 multi-GPU pipeline parallelism remains future work.

## Historical and reference documents

These documents remain useful, but they are not current checklists:

- [V4_PIPELINE_MODULARIZATION_ANALYSIS.md](../analysis/historical/V4_PIPELINE_MODULARIZATION_ANALYSIS.md) records the pre-refactor duplication analysis and completed extraction decisions.
- [EXPERT_PERFORMANCE_REVIEW.md](../analysis/historical/EXPERT_PERFORMANCE_REVIEW.md) records the M15-era source review and the implemented M16-M20 follow-up steps; use the conclusions document and ledger for current status.
- [PROJECT_AEON_VISION.md](../reference/strategy/PROJECT_AEON_VISION.md), [AEON_TECHNICAL_BLOCKERS_AND_RISKS.md](../reference/strategy/AEON_TECHNICAL_BLOCKERS_AND_RISKS.md), and [AEON_SPECIALIZATION_AND_ENTROPY_ANALYSIS.md](../reference/strategy/AEON_SPECIALIZATION_AND_ENTROPY_ANALYSIS.md) preserve strategic rationale and hypotheses.
- [AEON_RESEARCH_AND_INSPIRATION.md](../reference/prior-art/AEON_RESEARCH_AND_INSPIRATION.md) and [REFERENCE_EXPERT_CACHING_AND_COLD_STREAMING_ANALYSIS.md](../reference/prior-art/REFERENCE_EXPERT_CACHING_AND_COLD_STREAMING_ANALYSIS.md) preserve external research and comparative implementation evidence.
