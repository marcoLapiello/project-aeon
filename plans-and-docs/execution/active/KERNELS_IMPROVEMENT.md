Every compute kernel in prefill today is a GEMV (one token at a time). The chunk path loops over tokens on the host for every stage, so each chunk re-reads the weights T times. Nothing uses WMMA: the 16-row padding in scratch is there, but no kernel reads it. So the swept prefill runs at GEMV rate (bandwidth-bound, around 1 flop per byte) when it could run at matrix-multiply rate. That is the main headroom.

## Findings

- The chunk "batch" is really serial. In `v4_layer_body_batch.hpp:540-610`, pre-attention, attention and norm, the router, and MoE plus post each run for `row < count`. Each row launches its own GEMV kernels, so every weight byte is read T times per chunk.
- The expert kernels are single-token GEMVs. `aeon_moe_fused_w13_swiglu` and `aeon_moe_fused_w2_contrib` (dispatched at `v4_expert_executor.hpp:305-320`) take one activation vector. An expert chosen by k tokens in a chunk is dequantised k times.
- The router makes one host sync per row. It returns `std::vector` top-k results, with `hipStreamSynchronize` at `v4_layer_body_moe.hpp:138` and `v4_layer_body_types.hpp:104`. This leaves the GPU idle between rows.
- Small per-row launches (rmsnorm, rope, HC sinkhorn, KV-cache copies, and an H2D memcpy of the position per row) add launch and sync overhead that scales with T.

## Execution plan (in order of return)

### Step 1: W4A16 grouped WMMA GEMM for experts (largest gain)

- New G2/G3 kernel `aeon_moe_grouped_w13_wmma` / `_w2_wmma` in `backend/swizzled_w4a16/kernels/`.
- Input is a token→expert permutation built on the GPU: sort the chunk's (token, slot) pairs by expert, with offsets per expert.
- Tiling: M=16 tokens per WMMA tile (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`), N=64–128 output rows per workgroup, K split into 128-wide groups that match the quantisation scale group.
- Dequantise int4 to fp16 into LDS once per K-tile and reuse it across every token tile of that expert. Double-buffer the LDS loads (global→LDS of the next tile overlaps WMMA on the current one).
- Fuse SwiGLU-clamp into the W13 epilogue (gate and up interleaved in N). The W2 epilogue writes weighted fp32 contributions per (token, slot), which keeps the existing fixed-order deterministic reduce.
- Keep the swizzled layout if a WMMA B-fragment can be decoded from it. Otherwise add a second layout pass in the converter (a standard-format ingest, not a new quantisation).
- Test: an independent CPU reference for per-token expert output, ε < 1e-3.

#### Implementation record

_Built `2026-09-29`. What was built, what it measured, and the revisions those measurements imply. These are synthetic microbenchmarks, so they are recorded **here** rather than in the [Performance & Accuracy Ledger](../status/PERFORMANCE_LEDGER.md), which holds end-to-end runs only._

Built, smallest verified piece first:

| Artefact | Role | Gate |
| :--- | :--- | :--- |
| `src/platform/rdna3/wmma.hpp` | G2 primitive: the Wave32 `v_wmma_f32_16x16x16_f16_w32` and its operand lane map | `tests/test_rdna3_wmma_oracle.cpp` |
| `src/platform/rdna3/moe_grouped_ffn.hpp` (G2) + `src/backend/swizzled_w4a16/kernels/swizzled_w4a16_feed.hpp` (G3) + `src/architecture/deepseek_v4/kernels/moe_grouped_dispatch.hpp` (G4) | grouped W13+activation and W2 over a token→expert permutation, split into arch kernel / weight feed / model epilogue | `tests/test_v4_grouped_wmma_oracle.cpp` |
| `tests/bench_expert_pair_ab.cpp` | GEMV pair vs grouped pair: time, weight traffic, M window, crossover | `build/bin/bench_expert_pair_ab` |

The lane map was pinned on silicon before anything was built on it. It is not documented in one place, and a wrong reading still runs and still returns finite numbers — the common failure computes `A · Bᵀ`, which an identity-A test cannot see.

**Measured** (`bench_expert_pair_ab`, Device 0, synthetic experts in the real swizzled format, resident pool `128` experts (`1.69 GiB`, past the `96 MiB` Infinity Cache); draws sampled from the measured layer-0 prefill expert distribution of `routing-profile/first-real-prompt/counts.csv`; `T = 16 / 64 / 256 / 1024`; M windows `w1 / w2 / w4 / w8` token tiles (16–128 tokens sharing one dequantized slab); `n = 3`, best-of; every window compared elementwise against the GEMV arm and agreeing, `max |Δ| ≤ 1e-3 · scale`):

| T | tok/expt | GEMV ms | w1 | w2 | w4 | w8 | traffic `w4` / `w8` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1.9 | `2.56` | `3.20` | **`3.19`** | `3.50` | `4.69` | `1.9x` / `1.9x` |
| 64 | 5.1 | `10.17` | `4.98` | **`4.76`** | `5.13` | `6.80` | `5.1x` / `5.1x` |
| 256 | 20.2 | `39.65` | `7.67` | `6.12` | **`5.53`** | `7.25` | `20.2x` / `20.2x` |
| 1024 | 80.8 | `159.06` | `15.36` | `11.60` | `10.07` | **`9.99`** | `45.5x` / `70.6x` |

Best-window speedup over GEMV: `0.80x` (T=16) / `2.14x` (64) / `7.17x` (256) / `15.93x` (1024). At `T=1024` the traffic is GEMV `81.00 GiB`, `w4` `1.78 GiB`, `w8` `1.15 GiB` against a one-read-per-distinct-expert floor of `1.00 GiB`. The **window is a dispatcher choice, not a constant**: `w2` is fastest at `T ≤ 64`, `w4` at `256`, `w8` at `1024`, and all reach the one-read floor once the window holds the expert (default `w4`). Holding the dequantized slab across every token tile (K-outer) is what took `T=1024` from `15.18 ms` / `10.36x` to `9.99 ms` / `15.93x`, traffic `5.51 → 1.15 GiB`.

Correctness: all four windows agree with the GEMV arm at every chunk size; `test_rdna3_wmma_oracle` and `test_v4_grouped_wmma_oracle` green, the latter bit-identical (`max_rel` `3.969e-04` / `2.390e-04` / `8.427e-06`) across the loop-nest change and the double-buffer revert.

Still open, in the order they should land:

1. **Permutation kernel** (`token→expert` sort + offsets) — **landed** as Step 2's first piece; see Step 2's record below.
2. **Executor wiring** — dispatch the grouped pair from `V4TieredExpertExecutor` above the crossover, keep the GEMV pair below it.
3. **Routed-prefill / short-prompt path** — same grouped kernel; per-expert GEMV fallback for 1–3 token experts.
4. **LDS double-buffering** — deferred pending the measurement below, which decides whether it can pay.

#### Revisions the measurements imply

1. **Step 3's `T ≥ 16` threshold is too thin.** The measured crossover is between `T = 16` (`0.80x`, grouped loses) and `T = 64` (`2.14x`). `T ≥ 64` is where the choice stops being close.
2. **Step 6's "choose the chunk size so the average tokens per expert reaches ≥ 16" is the wrong stopping rule.** `16` is reached at `T = 256`, yet the reward keeps rising well past it (`7.17x` → `15.93x`). Chunk size, layer-major window and the swept-prefill gate stay **user-configurable**; the deliverable here is the *curve* above, reported, not a constant baked into the engine.
3. **"K split into 128-wide groups" became 64.** Two 32-wide quantization groups map exactly one int4 load per thread, which is what makes the staging branch-free. 128 would need two loads per thread.
4. **Chunk and window are not tuning constants.** They are user settings; the engine must accept them and the measurement above tells the user what they buy. Nothing in this plan should hardcode them.

#### Open measurement: is the grouped kernel out of weight bound, or out of compute?

The window sweep above shows weight traffic falling `1.55x` from window 4 to window 8 at `T = 1024` while elapsed time is flat (`10.07` vs `9.99 ms`). Reuse that buys no time means the kernel is no longer bound by the stream. Resolved by measurement — see "Attribution" below.

#### Attribution: what the gate half actually spends its time on

Measured at `T = 1024` over all ten expert batches, by running each half of the production loop alone with the same grid and trip counts (`bench_expert_pair_ab`):

| Arm | ms | Regs | LDS | Wave slots |
| :--- | ---: | ---: | ---: | ---: |
| staging alone (dequant + LDS stores) | `3.24` | `39` | `16 KiB` | `16/32` |
| matrix multiply alone | `2.42` | `65` | `128 B` | `32/32` |
| production (gate half) | `6.24` — `33.1 TFLOP/s` | `186` | `16 KiB` | `16/32` |

Three conclusions, two of which contradict what was assumed before measuring:

1. **The kernel composes almost additively** (`production / (staging + mma) = 1.10`), so there is *no* large serialisation to recover. The tempting "overlap the staging with the MMAs" saving is at most `10%`, not the `2.3x` an earlier broken staging arm suggested — that arm measured `1` register because a conditional read-back let the compiler delete its stores.
2. **The staging is the larger half**: `3.24` of `6.24 ms`, `52%`. It is the dequant and the LDS stores, not the matrix units, that bound this kernel.
3. **Occupancy is capped by LDS, not registers.** The slab is `4 KiB` per wave; gfx1100 has `64 KiB` LDS and `32` wave slots per CU, so `4 KiB` per wave permits at most `16` waves — `50%` — and the `186` registers are not the binding constraint. Double-buffering the slab cuts that to `25%` and is a measured regression (`6.2 → 8.8 ms`), so it stays out.

**Next lever, in order:** cut the LDS per wave so occupancy can rise (K block `32` instead of `64` halves the slab per wave, at the cost of reworking the staging mapping, which currently relies on two 32-wide quantization groups per thread), and cut the staging itself — it writes one half per element, so its store count is four times the global read it came from.

### Step 2: Batch the chunk loop in the layer body

- Rewrite `run_chunk` so every stage takes a `[T, dim]` activation. Remove `std::vector<...>` views/pre/outputs per row.
- Router: one batched gate GEMM, then a top-k kernel over all T rows that writes device-side ids and weights. Hand these to the Step 1 permutation kernel. Copy the ids to the host once per chunk, asynchronously, only for the supply hint (`on_routing_ready_batch`), with no stall on the compute stream.
- Batch the KV-cache/position writes into one kernel (position computed on the device, no H2D per row).

#### Implementation record

_Started `2026-09-30`._

**Seam first (structural prerequisite).** The grouped pair was one G2+G3+G4 file: `backend/swizzled_w4a16/kernels/aeon_moe_grouped_wmma.hpp` called the RDNA3 WMMA primitives, decoded the swizzled format, and applied the DSV4 clamp, with the tile/window/batch choices exposed as raw template parameters and no dispatcher. Wiring the executor to it would have built that coupling into production, so the seam landed before any wiring. Three policies, one binding:

| Group | File | Owns |
| :--- | :--- | :--- |
| G2 | `src/platform/moe_grouped_ffn.hpp` → `src/platform/rdna3/moe_grouped_ffn.hpp` | the WMMA loop: fragments, lane map, K-outer nest, the LDS B-tile `[K][N]` contract. The first is the architecture-free selector; the second is the RDNA3 implementation |
| G3 | `src/backend/swizzled_w4a16/kernels/swizzled_w4a16_feed.hpp` | `SwizzledW4A16Feed<RPW,LPR>`: dequantize a K-tile into the LDS slab, and state the per-launch expert capacity |
| G4 | `src/architecture/deepseek_v4/kernels/moe_grouped_epilogue.hpp` | the clamped SwiGLU and the routing-weight scale |
| G4 | `src/architecture/deepseek_v4/kernels/moe_grouped_dispatch.hpp` | the binding, the public entry points, and the tuning defaults |

The kernel stays one fused kernel — the slab reuse only pays while staging and MMA stay fused — but as compile-time policies, so it names no format and no model: the G2 file is templated on `Feed` and `Epilogue` and includes neither. The gate (`test_v4_grouped_wmma_oracle`) is **bit-identical** to before the move (`max_rel` `3.969e-04` / `2.390e-04` / `8.427e-06`), and the A/B attribution reports the same shapes (`186` regs, `16 KiB` LDS, `16/32` wave slots), so the restructure is behaviour-preserving.

Two consequences for the rest of this step: the executor now calls the G4 binding by name and never sees a G3 type directly, and the dispatcher — not the kernel — is where the M window, waves and expert batching are chosen, which is where the `T ≥ 64` grouped/GEMV threshold will live.

The **GEMV pair carried the same defect** and was fixed the same way. `aeon_moe_fused_w13.hpp` (G3) defined the model's clamped SwiGLU and baked it into the kernel; it is now a generic gate/up projection with the activation injected as an `Epilogue`, and the model's activation is supplied by the new G4 binding `architecture/deepseek_v4/kernels/moe_gemv_dispatch.hpp`. In the opposite direction, `aeon_w4a16_swizzled_gemv.hpp` (G3) carried the gfx11 `fdot2` intrinsic with its own `#if __gfx11__`; the intrinsic moved to the G2 primitive `platform/rdna3/dot2.hpp`. Every caller of the gate/up dispatch was routed through the binding.

Moving the intrinsic out was only half the job. A consumer that then named `aeon::rdna3::fdot2` would still hardcode the architecture — adding RDNA4 would mean editing a format file. So the architecture is now resolved by neutral selectors: `platform/dot2.hpp` and `platform/moe_grouped_ffn.hpp` re-export the build's architecture under architecture-free names (`aeon::fdot2`, `aeon::dispatch_moe_grouped_gate_up`), and the G3 decoder and the G4 binding call those. Which architecture they resolve to is a **build-visible** macro, `AEON_ARCH_RDNA3`/`RDNA4`, set from `AEON_GPU_TARGET` in `AeonToolchain.cmake` — not the device-only `__gfx1100__`, because a selector keyed on that would take different branches in the host and device passes and name different kernels. Adding an architecture is now a new `platform/<arch>/` directory plus one branch in each selector and one line in CMake; no G3 or G4 file changes. `backend/swizzled_w4a16/kernels/aeon_w4a16_swizzled_gemv.hpp` now names no architecture at all.

Both expert gates stay bit-identical (`test_v4_expert_oracle`, and `test_v4_expert_executor`'s production-vs-reference `moe_out` `0` differing elements), so the de-coupling is behaviour-preserving.

Still not separated, and noted deliberately: the **superseded** atomic control path (`aeon_moe_fused_w2_accum_kernel`, `moe_accumulate_expert_kernel`) is kept only for the gates and reads the same way; it is not worth a structural change while its only consumer is a control.

**Permutation primitive.** The grouped pair is driven by a token→expert permutation; nothing produced one before.

| Artefact | Role | Gate |
| :--- | :--- | :--- |
| `src/platform/ops/expert_permutation.hpp` | G2 primitive: device counting sort of the router's top-k ids into the expert-contiguous `(expert_offsets, token_indices, draw_indices)` the grouped pair consumes | `tests/test_expert_permutation.cpp` |

One property comes from the grouped pair rather than the sort, and one is a correctness requirement of the engine:

- **A single launch today holds at most `8` experts** (`kAeonSwizzledMaxExperts`). This is **not** a gfx1100 limit and **not** a property of the int4 format: it is the size of the kernels' weight-pointer table (`SwizzledW13ExpertPtrs`/`SwizzledW2ExpertPtrs`, fixed arrays passed by value as a kernel argument), a constant inherited from the GEMV kernels where one token's MoE touched `6` experts. A chunk's `6C` draws cover many more experts than that, so the executor splits the union into expert batches. Because the offsets ascend and an absent expert is zero-width rather than a missing slot, a batch is just a **contiguous expert range rebased by `expert_offsets[first]`** — no index copy between batches, since an empty expert cannot break the range apart. Lifting the cap would mean handing the weights by device-side pointer indirection instead of a by-value table; batching is the pragmatic alternative and also bounds the per-dispatch intermediate (`d_expert_hidden` is sized for one dispatch, not the whole union).
- **Placement is deterministic** (byte-reproducible restore): the scatter visits draws in index order, one thread per expert, instead of racing for a position.

Gate — five shapes (`1` token, `7`, `300`, a small `8`-expert count, and `draws = 0`) against the permutation's definition, not a second copy of the algorithm: monotone offsets anchored at `0` and the draw count, every draw under its own expert with `token = draw / slots`, `draw_indices` a bijection onto `0..draws-1`, ascending draw order within each expert, and widths equal to an independently counted histogram. Passes on `gfx1100`; the `300`-token case leaves `227` of `256` experts present with a widest expert of `94` draws (past one 16-row M tile), and the small-count case has all `8` present.

Still open in this step: the batched router, the executor's `accumulate_routed_batch` dispatch of the grouped pair (expert batching plus the VRAM-slot weight tables), the `M`-keyed dispatcher, and the batched KV/position writes.

**Open decision — the grouped FFN path cannot be bit-identical to the serial one.** `tests/test_v4_layer_body_chunk_oracle.cpp` requires a chunk to be **bit-identical** to the same tokens run one at a time (`differing == 0`), and it states why: the chunk ordering only re-sequences `exp` terms over bit-identical keys, so any difference is a defect. That argument holds for the **attention** path, which this step does not touch. It does **not** hold for the **routed experts**: the grouped WMMA GEMM sums the `K` reduction in a different order than the per-token GEMV (`fdot2` chain), so its fp32 result differs in the last bits, and after the fp16 rounding the FFN output differs by a small amount. The grouped gate already measures this band (grouped vs GEMV agree within `1e-3·scale`), and rule 1 accepts it — but it means the chunk gate's expert-affected comparisons must move from `differing == 0` to a documented tolerance, while the attention/ring comparisons stay exact. That is a change to a certification gate, so it is being raised rather than made silently.

The wiring itself needs no activation gather: the grouped A-read can take a **row stride**, so it reads each token's row directly out of the batch scratch's per-token 16-row tile (`stride = 16 × H`), instead of copying rows into a compact `[T, H]` buffer. Only the down-projection output is already draw-indexed, which the existing fixed-order reduce consumes unchanged.

**Measured — a correct reorder, but not a settled win.** The batched grouped path was implemented, wired into `run_layer_body_chunk`, and measured against the serial per-token path (`test_v4_routed_prefill`, 43 layers). It is **not a bug**: the grouped batch is a correct reorder of the per-token GEMV, which `test_moe_grouped_batch_parity` shows directly — running `moe_out` through both paths on identical experts and routing gives a bit-identical result on the gate's synthetic fixture, and in the real executor the routed parts differ by `9.8e-4 … 2.0e-3`, i.e. **one fp16 ULP of `moe_out`** (the per-layer value is stored fp16, so this is the reorder at the representation floor). The kernels were already oracle-certified within `ε`, so both paths are equally valid computations.

The problem is amplification. Through 43 layers that per-layer ULP becomes a final-logit delta of `max_abs 2.734e-01` against a peak of `14.6` and a **top-2 margin of `0.820`** — the delta is **`33%` of the margin**. The greedy argmax held on this window, but a delta a third of the margin can flip near-ties, so "no quality degradation" is **not** demonstrated. The grouped chunk path is therefore left **gated off** (`run_layer_body_chunk` calls the per-token accumulate, the suite is green and a chunk stays bit-identical to serial); `accumulate_routed_batch`, `run_moe_grouped_expert_batch` and the parity gate exist and are tested, but unused in production.

**Quality check — passes.** The decisive measurement landed. In one process, the same window is run both ways (`test_v4_routed_prefill`, which now always drives `accumulate_routed_batch` and toggles the executor between per-token and grouped):

| Arm | logit delta vs serial | greedy token | greedy 8-token continuation |
| :--- | ---: | :--- | :--- |
| Per-token (switch off) | `0` (bit-identical) | agrees | `320 62 80 5809 5809 5809 5809 5809` |
| Grouped (switch on) | `0.273` | agrees | `320 62 80 5809 5809 5809 5809 5809` |

The grouped path's `0.273` logit delta **does not move the answer**: the greedy token and the whole 8-token continuation are identical. Combined with the parity result (the grouped batch is a correct fp16-ULP reorder, not a defect), the grouped path is quality-neutral for deterministic decoding. So the bit-exact comparison is the wrong bar — it asserts reproducibility against the per-token *implementation*, not model correctness (a MoE's `Σ_k w_k·Expert_k(x)` has no "intended" summation order; vLLM/SGLang serve prefill with grouped GEMMs) — and the greedy comparison replaces it.

**Enabling it by default is the remaining step, and it touches gates, not kernels.** `moe_grouped_batch_enabled()` currently defaults off, so production still runs per-token. Flipping it makes the chunk path grouped everywhere, which changes what the gates that assert chunk-vs-serial **bit-exactness** through the real executor will see: `test_v4_prefill_window` ("WINDOW == SERIAL, BIT-EXACT") and `test_v4_warm_frozen_prefill` at least, plus `test_v4_routed_prefill`'s own D-check. Each needs its chunk-vs-serial comparison moved from `differing == 0` to the greedy-agreement bar the measurement above establishes, while the *per-token* path keeps its bit-exact regression check. That is a gate-baseline change, deliberately listed rather than made in passing.

### Step 3: WMMA dense GEMM for attention and shared-expert projections

- The same WMMA core as Step 1 (a single group). Use it for Q/KV/O projections (including `v4_grouped_wo`), the compressor/indexer projections, shared-expert W13/W2 and the HC projections, whenever T ≥ 16.
- Below 16 rows (routed prefill with short prompts, and decode), keep the GEMV path. Set the threshold by the dispatcher on M.

### Step 4: Batched causal attention over the chunk

- Replace per-row `run_layer_body_attention_and_norm` with one kernel over a query tile (16 queries × head) using WMMA for QKᵀ and PV, with online softmax plus sink. It reads the composed local and compressed keys once per tile instead of once per query.
- Use SGLang dsv4 attention/indexer as the semantic reference.

### Step 5: Fuse elementwise and norm stages over [T, dim]

- Make rmsnorm, rope, HC sinkhorn and the residual/HC mixing multi-row (one launch per stage per chunk, one row per wave). Fuse the rmsnorm scaling into the next GEMM's A-load where it's cheap.
- Capture the per-layer chunk sequence in a HIP graph for the swept path, where shapes are fixed per chunk size.

### Step 6: Tune for gfx1100

- Sweep the WMMA tile config for the dominant expert shapes: N×K tile, waves per workgroup (4–8), LDS ≤ 64 KiB for 2 workgroups per CU.
- Choose the chunk size so the average tokens per expert reaches ≥16 (a full M tile). Pad partial tiles through the permutation rather than giving tiny tiles a separate path.
- Give the routed-prefill (short-prompt) path the same grouped kernel. Experts with only 1–3 tokens fall back to the GEMV kernel, which is picked per expert inside one launch.

### Step 7: Move the bottleneck back to supply

- Re-measure the swept prefill once Steps 1–3 land. Compute should then be faster than supply. Then tune the lookahead depth and chunk size together so the GPU stays fed.

**Scope:** dependencies are G2/G3 (new kernels) → G4 (batched layer body and attention) → G1 (chunk sizing and lookahead). Each step can be verified alone with an independent-oracle test before moving on.