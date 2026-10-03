# Modularity audit — G1–G4 coupling after the kernel work

Date `2026-10-02`. Question: can a new GPU architecture, a new model architecture or a new weight format (e.g. GPTQ) be added as a plugin, without touching working paths? Method: include-direction and name searches over `src/` (`rdna3`, `swizzled`/`w4a16`, `deepseek`/`v4_`, `AEON_ARCH`, wave-width intrinsics). Rule under test: `G4 → G3/G2`, `G1 → G3/G2`, and G2/G3 depend on nothing above them.

## Verdict

| Plugin | Ready? | Why |
| :--- | :--- | :--- |
| New GPU architecture | **Mostly** | Selectors (`platform/dot2.hpp`, `dense_gemm.hpp`, `moe_grouped_ffn.hpp`, `tiled_causal_attention.hpp`) resolve on `AEON_ARCH_*`; G2 includes nothing upward. Blocked by wave-32 hard-coding (item 3). |
| New model architecture | **Partly** | G1 includes no `architecture/` code. The engine still names the one expert format (item 2). |
| New weight format | **No** | G4's executor and kernel bindings use the INT4 types directly (item 1). |

## Clean

- `platform/` includes nothing from `architecture/`, `backend/` or `infrastructure/`.
- `AEON_ARCH_*` appears only inside `platform/`.
- No `__builtin_amdgcn`, `__ballot`, `warpSize` or inline asm outside `platform/`.
- `infrastructure/` includes no `architecture/` file.
- `backend/` reaches `platform/` only through the neutral `platform/dot2.hpp`.

## Fixed

- `platform/rdna3/device.hpp` held arch-neutral HIP device selection and was included directly by G4 (`v4_model_host.hpp`). Moved to `platform/device.hpp`, all includes updated (commit `da80671`).

## Open

| # | Coupling | Where | Consequence |
| :--- | :--- | :--- | :--- |
| 1 | **G4 uses the INT4 format directly**: `SwizzledW13ExpertPtrs`, `SwizzledW2ExpertPtrs`, `kAeonSwizzledMaxExperts`, `SwizzledW4A16Feed`, `vram_expert_pool.hpp` | `v4_expert_executor.hpp`, `moe_grouped_batch.hpp`, `moe_grouped_dispatch.hpp`, `moe_gemv_dispatch.hpp`, `v4_model_host.hpp` | A new format edits the DSV4 executor. Largest gap. |
| 2 | **G1 hard-codes the current format**: `make_current_swizzled_expert_format()` as default; `ExpertFormatKind` enum and swizzled spec in the engine | `expert_payload_pool`, `host_expert_pool`, `expert_host_region`, `prefetch_staging`, `memory_budget_engine`, `aeon_loader`, `aeon_artifact`, `expert_format.hpp` | A new format edits the engine's pools and enum; only `expert_backend.hpp` (the registry) is a legitimate registration point. |
| 3 | **Wave width 32 hard-coded** (`__shfl_xor(…, 32)`) in G4 kernels (`hc_sinkhorn`, `v4_attention_kernels`, `v4_grouped_wo`, `v4_hc_head_kernel`, `v4_rope`, and the `*_wave32_*` kernels in `v4_layer_body_*`, `v4_graph`) and in `platform/ops/{rmsnorm,gemv}.hpp`, which have no selector | G4, `platform/ops` | A wave64 GPU edits model kernels; `platform/ops` is RDNA3 code in a neutral folder. |
| 4 | **G3 depends on G1**: `swizzled_expert_format.hpp` → `expert_format.hpp`; `vram_expert_pool.hpp` → `expert_payload_pool.hpp` | `backend/swizzled_w4a16/core/` | Reverse of the rule. Acceptable only if G1 owns the interface and G3 implements it; document that or move the interface to a neutral place. |
| 5 | Naming: `d_swizzled_*` scratch buffers in the model | `v4_activation_scratch.hpp` | Cosmetic. |

## Proposed order (one behaviour-preserving step each)

1. Format-neutral expert-kernel interface (weight-pointer table, max experts per launch) consumed by the G4 executor; no `Swizzled*` names in G4.
2. Replace the engine's `make_current_swizzled_*` defaults with a descriptor chosen from the registry; the format enum stops being engine-owned.
3. Wave-width reductions behind a `platform/` selector, as done for `fdot2`; give `platform/ops` an arch folder.
4. Settle item 4 either way, then item 5.

Gates per step: `test_v4_grouped_wmma_oracle`, `test_v4_expert_executor`, `test_moe_grouped_batch_parity`, `test_v4_layer_body_chunk_oracle`, `test_aeon_loader`; end-to-end gates only after step 1 and 2 together.
