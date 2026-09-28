# Relocation Execution Plan

**Date:** 2026-09-28
**Status:** Open. Not started.
**Scope:** Execute the file movements described by the [Concern Grouping and Relocation Analysis](../../analysis/current/CONCERN_GROUPING_AND_RELOCATION_ANALYSIS.md), one small commit per step, with no behaviour change. This plan covers only the steps that are ready or nearly ready now; the genuinely mixed files are deferred to their own steps (§6).

**Companion document.** The analysis owns *what the four groups are and where each file belongs*. This plan owns *the order and the verification*. Where they disagree, the analysis wins on classification and this plan wins on sequence.

---

## 1. Frozen decisions

| # | Decision | Resolution |
| :--- | :--- | :--- |
| 1 | G1 destination | `src/infrastructure/core/` — continue to use the flat `core/` folder; subfolder reorganisation comes later. |
| 2 | Reusable-kernel destination (G2) | `src/platform/ops/`, renamed neutrally (`rmsnorm.hpp`, `gemv.hpp`, `argmax.hpp`). |
| 3 | G2 policy | `platform/` is the GPU-specific home. Backend files split **only** where GPU-specific logic can be safely extracted while keeping the backend logic strictly format-specific. |
| 4 | Naming | Any file that is not DSV4-specific loses the `v4_`/`dsv4_` prefix when it moves. Class names are **not** renamed in a move step (that is separate churn). |
| 5 | Umbrella technique | **Retired.** A move rewrites include paths in all includers and deletes the old path. No old-path re-including stubs are left behind. |
| 6 | `supports_v4_pipeline` | Rename the neutral-registry flag to a capability name (`supports_fused_moe_experts`); let `v4_model_host` decide whether that capability is enough. Independent, low-priority step. |

---

## 2. House rules

There are two step kinds, each with one rule.

> **MOVE step:** the only permitted diff is `#include` paths and file location. Every code line is identical before and after.
> **SPLIT step:** code may move between files and a seam may appear, but behaviour is unchanged and the seam is the only new thing.

Both are verified by the same recipe (§5). A move must never carry a rename, a refactor, or a "while I'm here" fix; if it needs one, that is a separate step.

---

## 3. Why the ready set is small

Most G1 files are engine *strategy* written against model types. Measured from the include graph:

- **Model-free (ready to move):** `v4_device_streams.hpp` (includes only HIP), `aeon_runtime_config.hpp` (includes only `prefetch_staging.hpp`), and `v4_sampler.hpp` (includes only the argmax kernel).
- **Model-coupled (must be decoupled first):** `memory_budget*` (include `config.hpp`), `v4_prefill_*` (include `v4_expert_supply.hpp`, `v4_layer*`), `v4_host_partition.hpp` (includes `v4_expert_supply.hpp`), `v4_engine.hpp` (includes the graph and host).

Moving a model-coupled file into `infrastructure/` today would make `infrastructure/` include `architecture/`, inverting the one-way dependency the analysis found. Those files therefore need a **seam** (a neutral input/interface) before they can move; that is a SPLIT step and is deferred (§6).

This is expected, not a setback: it is the same decouple-then-move shape the backend plan already uses, and it confirms G1 is real — the code *is* engine logic — while showing exactly where the model type leaks into it.

---

## 4. Steps

Ordered so that each commit builds and passes, leaves the tree consistent, and does the lowest-risk work first. Moves are sequenced by include fan-in (low first) so each step touches few files.

### Stage 1 — Reusable kernels to G2 (`src/platform/ops/`)

| Step | Action | From | To | Fan-in |
| :--- | :--- | :--- | :--- | ---: |
| **K1** | MOVE + rename | `architecture/deepseek_v4/kernels/v4_argmax.hpp` | `platform/ops/argmax.hpp` | 1* |
| **K2** | MOVE + rename | `architecture/deepseek_v4/kernels/v4_norm.hpp` | `platform/ops/rmsnorm.hpp` | 4 |
| **K3** | MOVE + rename | `architecture/deepseek_v4/kernels/v4_gemv.hpp` | `platform/ops/gemv.hpp` | 5 |

K1 is first because the sampler's only model include is the argmax kernel, so moving it unblocks the sampler (S2). K2 and K3 are independent. `v4_pipeline_ops.hpp` is **excluded** — it carries the model's SwiGLU clamp and stays a deferred split (§6).

\* `v4_argmax.hpp` has one includer inside the kernel tree (`v4_attention.hpp` umbrella) plus test users.

### Stage 2 — The ready G1 moves (`src/infrastructure/core/`)

| Step | Action | From | To | Fan-in |
| :--- | :--- | :--- | :--- | ---: |
| **S1** | MOVE + rename | `architecture/deepseek_v4/core/aeon_runtime_config.hpp` | `infrastructure/core/runtime_config.hpp` | 5 |
| **S2** | MOVE + rename | `architecture/deepseek_v4/core/v4_sampler.hpp` | `infrastructure/core/sampler.hpp` | 5 |
| **S3** | MOVE + rename | `architecture/deepseek_v4/core/v4_device_streams.hpp` | `infrastructure/core/device_streams.hpp` | 3 |

S2 depends on K1 (its only model include is the argmax kernel). S1 and S3 are independent.

### Stage 3 — Neutral-registry cleanliness (independent)

| Step | Action | Detail |
| :--- | :--- | :--- |
| **R1** | Rename a field | `ExpertBackendDescriptor::supports_v4_pipeline` → `supports_fused_moe_experts`. Touch-points: the declaration and both `resolve()` descriptors in `expert_backend.hpp`, the reader in `v4_model_host.hpp:126`, and the assertion in `test_aeon_swizzled_loader.cpp:40`. |

R1 is not a file move and can run at any time. It removes a model name from the neutral backend contract.

---

## 5. Verification protocol (every step)

1. **Proof of move** — normalized line-multiset diff: strip comments (`sed -E 's://.*$::'`), trim, drop blanks, `sort`, then `comm -23` / `comm -13`. For a MOVE, the code-line multiset is identical (only paths differ). For a SPLIT, the only new lines are the seam.
2. **Brace balance** — `grep -o '{' | wc -l` vs `'}'` before and after. This is the companion check that catches a truncation the multiset diff cannot see (repeated `}` lines mask a small loss).
3. **Build** — `aeon_chat` plus every target affected by the step.
4. **Selective tests** (AGENTS.md rule 8) — the tests that cover the moved area, e.g.:
   - Stage 1: `test_v4_norm_oracle`, `test_v4_mla_oracle`, `test_v4_attention_sink_oracle`.
   - Stage 2: `test_v4_sampler`, `test_v4_engine`.
   - Stage 3: build only (`test_aeon_swizzled_loader` asserts the renamed field).
5. **No logic edits** — do not touch kernel or class bodies while moving them (AGENTS.md rule 5, anti-circularity, is not at risk in a move and must stay that way).

Header-only note: none of the Stage 1–2 files appear in CMake (headers are not listed), so **no CMake change is needed** for header moves. Only `dsv4_tokenizer.cpp`, `dsv4_prompt_encoder.cpp`, and `text_generation.cpp` are listed, and none of them move here.

---

## 6. Deferred (design at the step, not now)

These are the model-coupled G1 files and the genuinely mixed files. Each needs a **seam design** before it can move; that design is written when the step is started, per the decision to keep the split strategy open until we get there.

| Deferred item | Why it waits | Nature of the seam |
| :--- | :--- | :--- |
| `memory_budget.hpp`, `memory_budget_engine.hpp`, `memory_budget_report.hpp` | Include `config.hpp` / `V4ModelSpec` | Neutral model-dimensions input instead of `DeepSeekV4Config` |
| `v4_prefill_controller/sweep/lookahead/workspace.hpp` | Include `v4_expert_supply.hpp`, `v4_layer*` | Neutral supply-adapter interface; template on the model type |
| `v4_host_partition.hpp` | Includes `v4_expert_supply.hpp` | Same supply-adapter seam |
| `v4_expert_executor.hpp` (mixed) | Model MoE execution + tiered-supply mechanics + backend dispatch | Lease/tier policy vs. V4 dispatch |
| `v4_model_host.hpp` (mixed, 1227 lines) | Model residency + engine assembly | **The one hard step.** Engine-owned assembly type; defer to last |
| `v4_layer_body_batch.hpp` (mixed) | Engine prefill strategy + model row-set assembly | Strategy/progression extraction |
| `v4_pipeline_ops.hpp` (split) | Carries the model's SwiGLU clamp | Split casts/accumulate (reusable) from clamped SwiGLU (model) |

Once the seams exist, each deferred file moves by the ordinary MOVE recipe into the same destinations (`infrastructure/core/` for G1; the backend GPU halves to `platform/` per decision 3).

---

## 7. Done when

- No G1 file remains under `architecture/deepseek_v4/`; the ready set is in `infrastructure/core/`.
- The reusable kernels live in `platform/ops/` under neutral names.
- The neutral backend registry no longer names the model.
- Every step left the build and its selective tests green, and each MOVE was proven by the multiset diff plus brace balance.

Deferred items are explicitly **not** part of this plan's "done" — they are tracked in §6 and executed when their seams are designed.

---

## 8. Non-goals

- No behaviour change, no performance claim, no new abstraction beyond the seams the deferred steps require.
- No class renaming (only file names).
- No subfolder reorganisation of `infrastructure/core/` (later).
- No kernel logic edits while moving.
