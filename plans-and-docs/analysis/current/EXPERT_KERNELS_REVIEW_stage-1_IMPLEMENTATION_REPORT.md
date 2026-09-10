# Expert Kernels Review - Stage 1 Implementation Report

**Purpose:** Give the external reviewer a complete implementation and measurement summary without requiring a source-tree read-through.

## 1. Executive status

| Path | Artifact | Runtime | Selection | Default |
|---|---|---|---|---|
| Baseline | `model_experts.aeon` + `model_experts.index` (index version 1) | Original `dispatch_w4a16_gemm` path and per-expert W1/W3/SwiGLU/W2/accumulation launches | `!swizzled_moe_enabled_` | Yes |
| Stage 1 alternative | `model_experts_swizzled.aeon` + `model_experts_swizzled.index` (index version 2) | Swizzled GEMV plus fused six-expert W1/W3 and W2 kernels | `--swizzled-experts` / `use_swizzled_experts=true` | No |

The version-1 artifact and runtime path remain intact and are not overwritten. The loader selects the artifact explicitly and rejects an unexpected index version. The new path is implemented for native `.aeon` inference; the Safetensors initialization path remains baseline-only.

Stage 1 changes the routed expert execution used by the current single-token path. It does not implement true batched prefill and does not change dense FP16 projections, attention, the router, the shared expert, or the LM head.

Relevant implementation files: [swizzle layout](../../../src/kernel/aeon_w4a16_swizzle.hpp), [swizzled GEMV](../../../src/kernel/aeon_w4a16_swizzled_gemv.hpp), [fused W1/W3](../../../src/kernel/aeon_moe_fused_w13.hpp), [fused W2](../../../src/kernel/aeon_moe_fused_w2.hpp), [repacker](../../../scripts/repack_aeon_experts_swizzled.py), [loader](../../../src/core/aeon_loader.hpp), and [pipeline integration](../../../src/core/v4_pipeline.hpp).

## 2. Proposal comparison

### 2.1 Implemented 1:1

| Proposal item | Actual implementation |
|---|---|
| Wave-oriented W4A16 destination layout | `RPW=4,LPR=8` for W1/W3 and `RPW=8,LPR=4` for W2; both use `RPW*LPR=32` and `ITERS=16`. Destination indexing is `(row block, iteration, lane)`. |
| Nibble permutation | Exact permutation `[0,2,4,6,1,3,5,7]` for packed words. Scales follow the same lane/group transpose. |
| Payload size and tensor offsets | Each expert remains `14,155,776` bytes with the original six-subtensor offsets. Element counts and 4096-byte alignment are unchanged. |
| Offline repacking | The separate repacker reads version 1, applies the W1/W3 or W2 configuration, writes version 2, and verifies selected real experts by inverse transformation. The version-1 input is never replaced. |
| Vectorized dequantized dot product | `uint4` loads, `half2` activation pairs, `fdot2` on `gfx1100`, FP32 accumulation, and FP16 output. Each lane owns one output row; `LPR` lanes reduce with Wave32 XOR shuffles. |
| Iteration prefetch | Current `uint4`/scale values remain in registers while the next iteration is loaded before the current iteration is computed. |
| Dual W1/W3 work | The dual GEMV path shares activation traversal and computes two accumulators. The fused W1/W3 kernel extends the same dataflow across six experts. |
| Two fused routed-expert dispatches | One dispatch handles all six W1/W3 projections plus SwiGLU; one dispatch handles all six W2 projections plus weighted accumulation. |
| Fused W2 finalization | FP32 accumulation is finalized to FP16 by the last expert block using per-row-block counters and a device fence. Counters are initialized once and rearmed by the kernel. |
| Device-side top-k weights | Fused W2 consumes the existing FP32 normalized six-element device weight array; no new host upload is introduced. |
| Resident-expert assumption | Transfer completion events are waited on before either fused dispatch. The kernels assume the six selected experts are resident, as proposed. |

### 2.2 Adjusted or modified

| Area | Proposal | Actual implementation | Reason or effect |
|---|---|---|---|
| Signed INT4 unpack | The sample subtracts `1024.0f` after constructing `1024+n`. | The implementation subtracts `1032.0f`. | `0x6400` encodes `1024+n`; subtracting `1032` produces `n-8`, preserving `(nibble - 8) * scale`. This is a correctness correction. |
| Activation staging | The proposed fused kernel stages `x` in `__shared__` memory. | Activation pairs are read directly from the activation pointer; no shared activation tile is staged. | Same tested numerical dataflow, but the LDS staging strategy was not implemented 1:1 and remains a tuning difference. |
| Pointer bundle | One bundle contains W1, W3, and W2 pointers. | Production uses separate `SwizzledW13ExpertPtrs` and `SwizzledW2ExpertPtrs`. | Same by-value pointer passing and no per-layer pointer upload; interfaces stay kernel-specific. |
| SwiGLU | The proposal marks its formula as a placeholder with a symmetric gate clamp and no up clamp. | The fused path uses the existing production formula in Section 3. | Preserves the model's established semantics rather than copying the placeholder. |
| Shared-expert contribution | The proposal describes routed W2 weighted accumulation only. | Fused W2 accepts `initial_output` and adds the shared-expert result once, on expert zero, before finalization. | Preserves the current MoE output contract. |
| Versioning and placement | A version field is suggested; the byte order is described as the only artifact change. | Version 2 is carried by the companion index and selected filenames. The new repacker also writes experts sequentially with a new index. | Prevents v1/v2 mixing and provides contiguous physical expert placement while preserving per-expert size and alignment. |
| Kernel organization | The proposal presents one generic `gemv_tile<..., DUAL>` helper. | Common `swizzled_group_dot` logic is shared by separate single/dual GEMV and fused kernel headers. | Same dataflow, adapted to the existing header-defined kernel organization. |

## 3. Exact production SwiGLU formula

For FP32 gate and up accumulators `g` and `u`, with `L=10.0`:

$$
g' = \min(g, L), \qquad
u' = \min(\max(u, -L), L)
$$

$$
\operatorname{SwiGLU}(g,u) =
\left(\frac{g'}{1+\exp(-g')}\right)u'
$$

The result is converted to FP16. There is no lower clamp on the gate; the up value is clamped symmetrically. This formula is shared by the existing `v4_pipeline_swiglu_clamp_kernel` and the fused W1/W3 kernel.

Therefore the proposal placeholder differs in two ways: it lower-clamps the gate and leaves the up value unclamped.

## 4. Measured Stage 1 effects

**Environment:** RX 7900 XTX (`gfx1100`), Wave32, ROCm 7.2.2, synthetic weights, isolated HIP-event timing. These measurements exclude model loading, NVMe reads, staging, cache misses, routing, and the rest of the transformer pipeline.

| Workload | Baseline | Stage 1 | Effect | Output check |
|---|---:|---:|---:|---:|
| Single W1/W3 GEMV, `N=2048,K=4096` | `12.822 us` / `368 GB/s` | `11.152 us` / `423 GB/s` | `1.150x` | max diff `0.000` |
| Single W2 GEMV, `N=4096,K=2048` | `13.077 us` / `361 GB/s` | `10.926 us` / `432 GB/s` | `1.197x` | max diff `0.000` |
| W1/W3 pair, two launches versus dual launch | `29.012 us` | `15.816 us` | `1.834x` | max diff `0.000` |
| Six-expert W1/W3 plus SwiGLU | `147.557 us` / 18 launches | `39.775 us` / 1 launch | `3.710x` | max diff `0.016` |
| Six-expert W2 plus weighted accumulation | `107.486 us` / 12 launches | `28.652 us` / 1 launch | `3.751x` | max diff `0.002` |

The W1/W3 fused baseline is twelve swizzled GEMV launches plus six existing SwiGLU launches, so it isolates dispatch/fusion after layout conversion. The W2 fused baseline uses the source-layout current GEMV plus accumulation, so its result combines layout and fusion effects. The fused differences are FP16 output differences against the separate-launch baselines and remain below the focused test thresholds.

The historical WMMA check (`M=16,N=2048,K=4096`) reran at `149.234 us` (`1.799 TFLOP/s`). It is not a direct Stage 1 decode comparison because the new routed path operates at `M=1`.

## 5. Dispatch trace and profiler response

The 101-iteration kernel trace showed the expected launch reduction:

- Fused W1/W3: `101` fused launches versus `1,212` swizzled GEMV plus `606` SwiGLU launches.
- Fused W2: `101` fused launches versus `606` GEMV plus `606` accumulation launches.

`rocprofv2 --kernel-trace` confirmed Wave32 dispatches and the expected launch reduction. The installed v2 counter backend rejected the requested GL2C group as unsupported and emitted zero SQ counter values. No `FETCH_SIZE`, `MemUnitStalled`, occupancy, or counter-derived bandwidth claim is made. The bandwidth values in Section 4 are effective bandwidth calculated from isolated HIP-event timing.

## 6. Routing contract

- The runtime always passes `top_k=6`, allocates six weights and indices, and launches the fused kernels with `expert_count=6`.
- Score-based routing cannot duplicate an expert because each selected score is masked before the next selection.
- The three current hash tables for layers 0-2 were exhaustively checked. All `129,280` rows in each table contain six distinct expert IDs.
- The hash kernel relies on that model-table contract and does not perform runtime deduplication.
- Routed weights are FP32, normalized, and scaled by the model's `1.5` routed scaling factor before fused W2 accumulation.

## 7. Correctness and end-to-end status

- Host swizzle and inverse-swizzle tests pass for both W1/W3 and W2 shapes, including dequantization identity.
- Silicon GEMV, dual GEMV, fused W1/W3, and fused W2 tests pass. The fused W2 test also verifies that counters are rearmed to zero after finalization.
- Version-2 loader validation confirms the separate artifact, index version, dimensions, byte count, and sequential expert offsets.
- The opt-in two-layer pipeline smoke test completes a valid token on silicon.
- A full-model baseline/swizzled text A/B produced identical generated IDs and the same English response in the prior validation run.
- No controlled 43-layer throughput comparison isolates Stage 1 yet. The current full-model text run remains approximately `3 tok/s` and is dominated by cold expert service and just-in-time transfer/dispatch latency. The measurements above establish GPU-side improvement, not an end-to-end speedup claim.
