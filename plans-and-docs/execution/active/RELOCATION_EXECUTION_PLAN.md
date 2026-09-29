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

There are three step kinds, each with one rule.

> **MOVE step:** the only permitted diff is `#include` paths and file location. Every code line is identical before and after.
> **SPLIT step:** code may move between files and a seam may appear, but behaviour is unchanged and the seam is the only new thing.
> **RENAME step:** a mechanical symbol rename across all call sites (`src`, `tests`, `tools`), word-boundary and exact. No definition, signature, or behaviour changes; the diff is the same token on a different spelling.

Both MOVE and SPLIT are verified by the same recipe (§5). A move must never carry a rename, a refactor, or a "while I'm here" fix; if it needs one, that is a separate step.

**Naming consistency (the case the original rule missed).** A file that moves out of the model tree must not keep the model's name — neither in its file name (handled by the MOVE) nor in the symbols it *defines*. A generic type defined in a moved file (`V4Sampler`, `V4DeviceStreams`, `V4PrefillSweep`, `v4_gemv_fp16_kernel`, …) is renamed to its neutral spelling in a dedicated RENAME step. Because a symbol has call sites everywhere — **including inside model files** — the rename necessarily touches `v4_` files, but only as *usages*: no model-specific symbol is ever renamed. The audit for a RENAME is that the diff contains only the renamed tokens (word-boundary), and that every changed line in a model file is a call site of the renamed generic symbol, never a model-specific definition.

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

### Stage 1 — Reusable kernels to G2 (`src/platform/ops/`) — ✅ complete

| Step | Action | From | To | Fan-in | Commit |
| :--- | :--- | :--- | :--- | ---: | :--- |
| **K1** | MOVE + rename | `architecture/deepseek_v4/kernels/v4_argmax.hpp` | `platform/ops/argmax.hpp` | 1* | `01eb3cd` |
| **K2** | MOVE + rename | `architecture/deepseek_v4/kernels/v4_norm.hpp` | `platform/ops/rmsnorm.hpp` | 4 | `0d31224` |
| **K3** | MOVE + rename | `architecture/deepseek_v4/kernels/v4_gemv.hpp` | `platform/ops/gemv.hpp` | 5 | `3e313a1` |

All three landed with the normalized code-line multiset identical to the pre-move version (a single comment-title line corrected per file), brace balance intact, a full-tree build, and the seven kernel/graph gates green (`test_v4_{norm,mla,grouped_wo,shared_expert,attention_sink}_oracle`, `test_v4_graph_head`, `test_v4_sampler`).

K1 is first because the sampler's only model include is the argmax kernel, so moving it unblocks the sampler (S2). K2 and K3 are independent. `v4_pipeline_ops.hpp` is **excluded** — it carries the model's SwiGLU clamp and stays a deferred split (§6).

\* `v4_argmax.hpp` has one includer inside the kernel tree (`v4_attention.hpp` umbrella) plus test users.

### Stage 2 — The ready G1 moves (`src/infrastructure/core/`) — ✅ complete

| Step | Action | From | To | Fan-in | Commit |
| :--- | :--- | :--- | :--- | ---: | :--- |
| **S1** | MOVE + rename | `architecture/deepseek_v4/core/aeon_runtime_config.hpp` | `infrastructure/core/runtime_config.hpp` | 5 | `9082ebd` |
| **S2** | MOVE + rename | `architecture/deepseek_v4/core/v4_sampler.hpp` | `infrastructure/core/sampler.hpp` | 5 | `3c452b4` |
| **S3** | MOVE + rename | `architecture/deepseek_v4/core/v4_device_streams.hpp` | `infrastructure/core/device_streams.hpp` | 3 | `f2dcb43` |

All three landed with the normalized code-line multiset identical to the pre-move version (S2 differs by exactly the include swap that made it model-free). S1 also removed a stray untracked duplicate of the former kernel file `v4_gemv.hpp` that the editor restored at the old path. The heavy gates `test_v4_engine`, `test_v4_graph_body`, `test_v4_sampler`, `test_v4_expert_executor`, and `test_v4_expert_tiering` all pass.

### Stage 3 — Neutral-registry cleanliness (independent) — ✅ complete

| Step | Action | Detail | Commit |
| :--- | :--- | :--- | :--- |
| **R1** | Rename a field | `ExpertBackendDescriptor::supports_v4_pipeline` → `supports_fused_moe_experts`. Touch-points: the descriptor and its comment in `expert_backend.hpp`, the reader in `v4_model_host.hpp:126`, and the assertion in `test_aeon_swizzled_loader.cpp:40`. | `10f713c` |

R1 removes a model name from the neutral backend contract. Verified: full rename, no residual old name, `test_aeon_swizzled_loader` passes.

### Stage 4 — Split the mixed kernel file (independent) — ✅ complete

| Step | Action | Detail | Commit |
| :--- | :--- | :--- | :--- |
| **P1** | SPLIT + rename | `v4_pipeline_ops.hpp` mixed three unrelated things. The two FP16↔FP32 casts and the MoE accumulate pair are model-agnostic, so they moved to `platform/ops/cast.hpp` (`half_to_float_kernel`, `float_to_half_kernel`) and `platform/ops/moe_accumulate.hpp` (`moe_accumulate_fixed_order_kernel`, plus the superseded fp16 control `moe_accumulate_expert_kernel`). The clamped SwiGLU is a DSV4 choice — `swiglu_limit` is a config knob — so the residual file was renamed `v4_swiglu_clamp.hpp` and its kernel `v4_pipeline_swiglu_clamp_kernel` → `v4_swiglu_clamp_kernel`. Every kernel body is byte-identical to the pre-split version; only the location and the name changed. Verified: `test_v4_expert_oracle`, `test_v4_shared_expert_oracle`, `test_v4_moe_accum_oracle`, `test_v4_expert_executor`, `test_v4_graph_head`, `test_swiglu_clamp`, `test_v4_layer_body_serial_oracle`. | `9e7bb2d` |

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
| ~~`memory_budget.hpp`, `memory_budget_engine.hpp`, `memory_budget_report.hpp`~~ | ✅ **Done** (`c53e5f8`): moved to `infrastructure/core/` behind the neutral `ModelMemoryGeometry` seam (`model_memory_geometry.hpp`; V4 adapter `v4_memory_geometry.hpp`). Behaviour unchanged; `test_dynamic_expert_pool`, `test_v4_prefill_window`, `test_v4_engine` pass. | — |
| ~~`v4_prefill_sweep.hpp`, `v4_prefill_lookahead.hpp`~~ | ✅ **Done**: moved to `infrastructure/core/` as `prefill_sweep.hpp` / `prefill_lookahead.hpp` behind the neutral `LayerBatchSupply` port (`layer_batch_supply.hpp`); the V4 adapter `V4ExpertSupplyCoordinator` implements it, and its `LayerPrefetchState` now derives from the neutral `LayerBatchState`. All 7 correctness checks in `test_v4_staging_depth` pass (Gate A's wall-time flatness is the known load-sensitive flake). | — |
| ~~`v4_prefill_controller.hpp`~~ | ✅ **Done**: moved to `infrastructure/core/prefill_controller.hpp` as `PrefillController`. Its remaining coupling was a single `V4TieredExpertExecutor*`, used only for `release_leases()`/`outstanding_leases()`; that two-method contract is now the neutral `ExpertLeaseHolder` interface (`infrastructure/core/expert_lease_holder.hpp`), which the executor implements. `supply`/`partition` became `LayerBatchSupply*`/`HostPartition*`. The **executor itself stays** (it is genuine model MoE). All lifecycle gates pass. | — |
| `v4_prefill_workspace.hpp` (partly) | ✅ **Done for the engine half** (`6985b9a`): the residual carry — the only engine-owned buffer — is now the neutral `infrastructure/core/prefill_carry.hpp` (`PrefillCarry`), sized by a scalar residual width and delegating only the model's width back to the workspace. The **file itself stays** in the model tree: its other buffer, `V4LayerBodyBatchScratch`, is written in the model's `DSV4_*` dimensions and sized worst-case across the model's layers, and `v4_graph.hpp` needs its concrete type. Prefill gates pass (`test_v4_staging_depth` Gate A spread is the known load-sensitive flatness flake, green on re-run). | — |
| ~~`v4_host_partition.hpp`~~ | ✅ **Done**: moved to `infrastructure/core/host_partition.hpp` as `HostPartition`; its `V4ExpertSupplyCoordinator*` became the neutral `LayerBatchSupply*`. `test_v4_prefill_window`, `test_v4_warm_frozen_prefill`, `test_v4_staging_depth`, `test_v4_engine` pass. | — |
| `v4_expert_executor.hpp` (mixed) | Model MoE execution + tiered-supply mechanics + backend dispatch | **Re-assessed: stays.** The MoE shape (six experts, clamped SwiGLU, fixed-order accumulation), the backend kernel dispatch, and the scratch are all model-specific; only its lease *contract* was generic and is now `ExpertLeaseHolder`. The lease *policy* (`ensure_pool_headroom`) remains executor-local, entangled with dispatch victim selection — extract only if a second model needs it. |
| `v4_model_host.hpp` (mixed) | Model residency + engine assembly | **The one hard step**, done as increments. **Step 1 — ✅ direct expert I/O** (`e147905`): the `io_uring` reader, the in-flight completion map, the request-id counter and the batched blocking read moved to `infrastructure/core/expert_direct_io.hpp` (`ExpertDirectIO`), a verbatim move. No seam is introduced: `AeonModelLoader` is itself infrastructure (`.aeon` is the engine's own container), so the reader taking the loader is a model → infrastructure dependency, which is the correct direction. **Step 2 — ✅ tier load/restore** (`81b4e03`): `preload_hot`, `preload_warm` and the drained-resident restore moved to `infrastructure/core/expert_tier_loader.hpp` (`ExpertTierLoader`), bound to the neutral collaborators through a `Services` set. Verbatim move; a second architecture reuses the load/restore path. **Step 3 — ✅ tier state ownership** (`2b7b757`): the neutral tier machinery (registry, Warm pool, staging arena + pinned region, Warm/staging partition, supply telemetry, routing-reuse profiler, direct I/O, bulk loader) is now one engine-owned bundle, `infrastructure/core/expert_tier_state.hpp` (`ExpertTierState`); the host holds `tier_` and reaches them as `tier_.X`. Ownership only — construction stayed in the host. The Hot payload pool is deliberately *not* a member of the bundle yet: see Step 3b. Name is `ExpertTierState` because `ExpertTier` is already the HOT/WARM/COLD enum. **Remaining — Step 4 (optional):** move the neutral *construction* (`steps 10–12` of `initialize_experts`) into an `ExpertTierState::initialize`, leaving the host to wire the V4 supply/executor. This is a further regroup whose only payoff is that the model host shrinks; behavioural risk is low but the churn is real, so it is optional. **Step 3b (small, correctness-of-design):** the Hot pool belongs in the tier too. The pool is an engine concept — `ExpertPayloadPool` (`infrastructure/core/expert_payload_pool.hpp`) owns the VRAM allocation and slot management, and every neutral consumer already takes `ExpertPayloadPool*`; what lives in the backend is only the *byte view* of a slot (`UnifiedVRAMExpertPool` adds the swizzled offset accessors the V4 kernels read). So the tier should hold the **neutral** `ExpertPayloadPool` and the host — the one backend-aware composition point — should construct the format's concrete pool and hand it in. Step 3 left it out only to avoid an `infrastructure` → `backend` include, which is a constraint on the *concrete type*, not a statement that the pool is format-owned. Note: `ExpertPayloadPool` has no virtual destructor, so if the tier *owns* it that needs adding (a non-owning reference is fine as-is). |
| `v4_engine.hpp` (ENG, but binds model text) | §3 lists it as model-coupled and the analysis classes it ENG (MOVE → `infrastructure`), but it was missing from this table — a gap closed here. It binds the model's tokenizer, prompt encoder, and graph; its includes are `v4_graph.hpp` + `v4_model_host.hpp`. | **Waits on the `v4_model_host` seam:** once the engine assembly and a model-binding seam exist, this becomes an engine-owned text binding over them. |
| `v4_layer_body_batch.hpp` (mixed) | Re-assessed: **stays.** Flagged as "engine prefill strategy + model row-set", but every symbol is the model's chunked layer body — the scratch is `DSV4_*`-shaped, `compose_local_rows` reads the DSV4 SWA ring (trap 39), and `run_layer_body_chunk` drives the model's phase functions in the order DSV4 semantics force. The prefill *strategy* is engine-owned and already moved (`prefill_controller` picks sweep-vs-routed by the gate; `prefill_sweep` owns the layer-major order); `v4_graph::forward_window` holds only the model-side loop that applies it over this model's layers. No model-free fragment to lift. | — |
| ~~`v4_pipeline_ops.hpp` (split)~~ | ✅ **Done**: split into `platform/ops/cast.hpp` (casts), `platform/ops/moe_accumulate.hpp` (the MoE reduce pair), and the renamed model file `v4_swiglu_clamp.hpp` (the clamped SwiGLU). See Stage 4. | — |

Once the seams exist, each deferred file moves by the ordinary MOVE recipe into the same destinations (`infrastructure/core/` for G1; the backend GPU halves to `platform/` per decision 3).

**Naming follow-up:** ~~the sweep class is still named `V4PrefillSweep`~~ ✅ **Done** — the RENAME step below neutralised the symbols defined in the moved files.

### RENAME step — neutralised symbols in moved files — ✅ complete

| Symbol (old → new) | Defined in | Kind |
| :--- | :--- | :--- |
| `V4Sampler` → `Sampler`, `V4SamplerConfig` → `SamplerConfig`, `V4SplitMix64` → `SplitMix64`, `V4LogitProcessor` → `LogitProcessor` | `infrastructure/core/sampler.hpp` | types |
| `V4DeviceStreams` → `DeviceStreams` | `infrastructure/core/device_streams.hpp` | type |
| `V4PrefillSweep` → `PrefillSweep` | `infrastructure/core/prefill_sweep.hpp` | type |
| `v4_gemv_fp16_kernel` → `gemv_fp16_kernel`, `v4_gemv_fp16_vec8_kernel` → `gemv_fp16_vec8_kernel` | `platform/ops/gemv.hpp` | kernels |
| `v4_rmsnorm_wave32_kernel` → `rmsnorm_wave32_kernel`, `v4_rmsnorm_unit_wave32_kernel` → `rmsnorm_unit_wave32_kernel` | `platform/ops/rmsnorm.hpp` | kernels |
| `v4_argmax_fp16_partial_kernel` → `argmax_fp16_partial_kernel`, `v4_argmax_partial_reduce_kernel` → `argmax_partial_reduce_kernel` | `platform/ops/argmax.hpp` | kernels |
| `v4_half_to_float_kernel` → `half_to_float_kernel`, `v4_float_to_half_kernel` → `float_to_half_kernel` | `platform/ops/cast.hpp` | kernels |
| `v4_moe_accumulate_fixed_order_kernel` → `moe_accumulate_fixed_order_kernel`, `v4_pipeline_accumulate_expert_kernel` → `moe_accumulate_expert_kernel` | `platform/ops/moe_accumulate.hpp` | kernels |
| `v4_pipeline_swiglu_clamp_kernel` → `v4_swiglu_clamp_kernel` (model-specific; only the stale `pipeline` token dropped) | `v4_swiglu_clamp.hpp` | kernel |

26 files touched, all as usages. Audited: no model-specific symbol (`V4ModelHost`, `V4Graph`, `V4Layer*`, `V4Attention*`, …) was renamed; the only such tokens in the diff are unchanged context on comment lines that also carried a renamed generic symbol. Builds; the seven gates pass.

---

## 7. Done when

**Ready scope (Stages 1–4) — ✅ met.**

- The reusable kernels live in `platform/ops/` under neutral names (`rmsnorm`, `gemv`, `argmax`, `cast`, `moe_accumulate`).
- The ready G1 set is in `infrastructure/core/` (`runtime_config`, `sampler`, `device_streams`).
- The neutral backend registry no longer names the model.
- Every step left the build and its selective tests green, and each MOVE was proven by the multiset diff plus brace balance.

Deferred items are explicitly **not** part of this plan's "done" — they are tracked in §6 and executed when their seams are designed.

---

## 8. Non-goals

- No behaviour change, no performance claim, no new abstraction beyond the seams the deferred steps require.
- No class renaming (only file names).
- No subfolder reorganisation of `infrastructure/core/` (later).
- No kernel logic edits while moving.
