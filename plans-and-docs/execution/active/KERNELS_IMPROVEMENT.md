At the time this plan was written, every compute kernel in prefill was a GEMV (one token at a time): the chunk path looped over tokens on the host for every stage, so each chunk re-read the weights T times, and nothing used WMMA — the 16-row padding in scratch was there, but no kernel read it. So the swept prefill ran at GEMV rate (bandwidth-bound, around 1 flop per byte) when it could run at matrix-multiply rate.

The steps below are numbered in the order they were *written*, not the order the time is. A measured phase profile of the swept prefill (below) later showed that Steps 1–2 address ~2.3% of it, while two per-token loops — the attention half and the pre-attention half — are 94%. The numbering is kept for the record; **the work order is the profile's**, and Step 4a is inserted on that basis.

**Scope: throughput in general, not one arm.** The swept prefill is profiled first because it has no supply constraint, so its host/GPU split is clean to read; that is a *measurement* choice, not a statement about where the work belongs. The routed prefill and decode share the same per-token body — the same attention kernel, the same GEMVs, the same HC/norm stages — so a fix to those helps every arm. Steps 3–5 therefore target the kernels and launches that all three pay for, and each step's record states its effect on each arm it touches. Decode's own bottleneck (the batched per-token sequence) is tracked separately.

## Measured phase profile (the ordering this plan should follow)

`aeon_chat --phase-profile`, swept prefill, 666-token prompt, `W=4096 C=256` (129 chunk bodies over 43 layers). Host = CPU time issuing the phase; GPU = `hipEvent` span on the compute stream. `n = 1`, one prompt.

| Phase | host ms | gpu ms | gpu share |
| :--- | ---: | ---: | ---: |
| **attention+norm (per token)** | `37,271` | `46,954` | **`70.7%`** |
| **pre-attention (per token)** | `13,057` | `15,718` | **`23.7%`** |
| router (batched) | `13,756` | `62` | `0.1%` |
| routed experts (batched, Step 1) | `1,035` | `1,433` | `2.2%` |
| routing dispatch | `443` | `461` | `0.7%` |
| shared expert (per token) | `227` | `1,149` | `1.7%` |
| moe post / commit | `449` | `653` | `0.8%` |
| **total** | **`66,238`** | **`66,430`** | |

Three things it settles:

1. **The two per-token loops are 94% of prefill.** `run_chunk_pre_attention` and `run_layer_body_attention_and_norm` are still `for (row …)` loops — one token's full pipeline at a time. Steps 3–5 replace them; that is where the time is.
2. **Steps 1–2 bought ~2.3% directly.** The grouped expert pair is `2.2%` and the batched router's device work `0.1%`. The router's `13.8 s` of host time is *not* issuance — it is its read-back `hipStreamSynchronize` draining the attention backlog queued ahead of it. The launch/queue cost of a phase is only visible as host time, which is why the profile reports both columns.
3. **The same drain, per token per CSA layer, is inside the 70.7%.** `select_indexer_topk` synchronizes the stream **twice** to sort the indexer's candidates on the host, once per token for every CSA layer (~`38k` drains per window). That is Step 4a.

**Then the profile was used to split its own largest number.** `attention+norm` divided into the attention kernel, the HC/norm tail, and `compose_local_rows` — and the last was ~`45 s` host / ~`23 s` GPU, the largest single cost of the prefill. It assembled each query's local row-set with two `hipMemcpyAsync` per ring slot per query (~`256` submissions per token). The row-set has a closed form (`row r` = slot `r`, position `first + ((r − first) mod capacity)`), so it is now **one gather launch**; row order must be slot order, not position order, because decode iterates the ring in slot order and for a wrapped window the two are a rotation — composing by position is a softmax summation-order change, which `test_v4_layer_body_chunk_oracle` caught (`313` differing) before it was fixed.

Result of the three steps so far (`aeon_chat`, 666-token prompt, `W=4096 C=256`, swept, `n = 1`):

| | M47 baseline | after Steps 1–2 | after 4a | after the compose gather |
| :--- | ---: | ---: | ---: | ---: |
| TTFT | `79.4 s` | `73.3 s` | `70.8 s` | **`50.0 s`** |
| ms / prompt-token | `121.4` | `110.1` | `106.3` | **`75.1`** |
| `attention+norm` host | — | `37.3 s` | `48.3 s` | **`1.5 s`** |
| attention kernel, GPU | — | — | `22.8 s` | `22.8 s` |
| pre-attention, GPU | — | `15.7 s` | `14.7 s` | `14.6 s` |

Prefill is now **GPU-bound**: after the host stalls are gone, the two GPU numbers that matter are the attention kernel (`22.8 s`) and pre-attention (`14.6 s`). That is what Steps 3–4 reduce, and it is why they are the next work rather than more launch surgery.

## Findings
- The chunk "batch" is serial: pre-attention, attention and norm, and the router each run per row (`row < count`), and each row launches its own kernels. The routed experts were per-token too, so an expert chosen by `k` tokens in a chunk was dequantised `k` times — addressed by Step 1/2 (the grouped pair and the permutation).
- The router makes one host sync per row, returning `std::vector` top-k results, so the GPU idles between rows.
- Small per-row launches (rmsnorm, rope, HC sinkhorn, KV-cache copies, an H2D memcpy of the position per row) add launch and sync overhead that scales with T.

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

_Built `2026-09-29`. Synthetic microbenchmarks, recorded here rather than in the [Performance & Accuracy Ledger](../status/PERFORMANCE_LEDGER.md), which holds end-to-end runs only._

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

**Attribution — where the gate half's time goes** (`bench_expert_pair_ab`, `T = 1024`, each half run alone with the same grid and trip counts):

| Arm | ms | Regs | LDS | Wave slots |
| :--- | ---: | ---: | ---: | ---: |
| staging (dequant + LDS stores) | `3.24` | `39` | `16 KiB` | `16/32` |
| matrix multiply | `2.42` | `65` | `128 B` | `32/32` |
| production (gate half) | `6.24` — `33.1 TFLOP/s` | `186` | `16 KiB` | `16/32` |

The halves compose almost additively (`production/(staging+mma) = 1.10`), so there is no serialisation to recover; and **the staging is the larger half** (`52%`). Occupancy is LDS-capped, not register-capped — the `4 KiB`/wave slab permits `16` of `32` waves, and double-buffering it halves that and is a measured regression (`6.2 → 8.8 ms`), so it is out. Next lever: cut LDS per wave (K block `32` instead of `64`, which needs reworking the two-groups-per-thread staging) and cut the staging itself (it writes one half per element, so its store count is `4x` the global read it came from).

**Corrections to the plan, from the measurements:**

1. Step 3's `T ≥ 16` threshold is too thin: the crossover is between `T = 16` (`0.80x`, grouped loses) and `T = 64` (`2.14x`). Wire `T ≥ 64`.
2. K is split `64`-wide, not `128`: two 32-wide quantization groups map to one int4 load per thread, which is what makes the staging branch-free.
3. Chunk size and window are **user settings**, not tuning constants; the deliverable is the curve above, not a baked-in value. This also supersedes Step 6's "reach ≥ 16 tokens per expert" rule — `16` is reached at `T = 256` yet the reward keeps rising to `15.93x`.

**Remaining:** the routed-prefill / short-prompt path (same grouped kernel, with a per-expert GEMV fallback for 1–3-token experts), and the `M`-keyed dispatcher that picks grouped above the crossover (see Step 2).

### Step 2: Batch the chunk loop in the layer body

- Rewrite `run_chunk` so every stage takes a `[T, dim]` activation. Remove `std::vector<...>` views/pre/outputs per row.
- Router: one batched gate GEMM, then a top-k kernel over all T rows that writes device-side ids and weights. Hand these to the Step 1 permutation kernel. Copy the ids to the host once per chunk, asynchronously, only for the supply hint (`on_routing_ready_batch`), with no stall on the compute stream.
- Batch the KV-cache/position writes into one kernel (position computed on the device, no H2D per row).

#### Implementation record

_Started `2026-09-30`._

**Seam first (structural prerequisite).** The grouped pair was one G2+G3+G4 file under the G3 path (`aeon_moe_grouped_wmma.hpp`), calling the RDNA3 WMMA primitives, decoding the swizzled format and applying the DSV4 clamp, with the tile/window/batch choices as raw template parameters and no dispatcher. Wiring the executor to it would have built that coupling into production, so the seam landed first. The kernel stays one **fused** kernel — the slab reuse only pays while staging and MMA stay fused — but as compile-time policies, so its G2 file includes neither the format nor the model, and the executor reaches it through one G4 binding:

| G2 (kernel) | G3 (feed) | G4 (epilogue + binding) |
| :--- | :--- | :--- |
| `platform/rdna3/moe_grouped_ffn.hpp` — the WMMA loop, templated on the two policies | `backend/swizzled_w4a16/kernels/swizzled_w4a16_feed.hpp` — dequantize a K-tile into the LDS slab, and state the per-launch expert capacity | `kernels/moe_grouped_epilogue.hpp` (the clamp) + `kernels/moe_grouped_dispatch.hpp` (the binding and the tuning defaults) |

The **GEMV pair carried the same defect** and got the same fix: the SwiGLU baked into `aeon_moe_fused_w13.hpp` (G3) is now an injected `Epilogue` supplied by `kernels/moe_gemv_dispatch.hpp`, and the gfx11 `fdot2` moved out of the G3 decoder into the G2 primitive `platform/rdna3/dot2.hpp`. Architecture is resolved by neutral selectors (`platform/dot2.hpp`, `platform/moe_grouped_ffn.hpp`) keyed on a **build-visible** `AEON_ARCH_*` macro — not the device-only `__gfx1100__`, which would differ between the host and device passes — so adding an architecture is a new `platform/<arch>/` plus a selector branch, and no G3 or G4 file changes. Behaviour-preserving: `test_v4_grouped_wmma_oracle` bit-identical (`8.427e-06`), `test_v4_expert_oracle` unchanged, `test_v4_expert_executor` `moe_out` `0` differing elements. The superseded atomic control path deliberately keeps its coupling; its only consumer is a gate.

**Permutation primitive.** The grouped pair is driven by a token→expert permutation; nothing produced one before.

| Artefact | Role | Gate |
| :--- | :--- | :--- |
| `src/platform/ops/expert_permutation.hpp` | G2 primitive: device counting sort of the router's top-k ids into the expert-contiguous `(expert_offsets, token_indices, draw_indices)` the grouped pair consumes | `tests/test_expert_permutation.cpp` |

One property comes from the grouped pair rather than the sort, and one is a correctness requirement of the engine:

- **A single launch today holds at most `8` experts** (`kAeonSwizzledMaxExperts`). This is **not** a gfx1100 limit and **not** a property of the int4 format: it is the size of the kernels' weight-pointer table (`SwizzledW13ExpertPtrs`/`SwizzledW2ExpertPtrs`, fixed arrays passed by value as a kernel argument), a constant inherited from the GEMV kernels where one token's MoE touched `6` experts. A chunk's `6C` draws cover many more experts than that, so the executor splits the union into expert batches. Because the offsets ascend and an absent expert is zero-width rather than a missing slot, a batch is just a **contiguous expert range rebased by `expert_offsets[first]`** — no index copy between batches, since an empty expert cannot break the range apart. Lifting the cap would mean handing the weights by device-side pointer indirection instead of a by-value table; batching is the pragmatic alternative and also bounds the per-dispatch intermediate (`d_expert_hidden` is sized for one dispatch, not the whole union).
- **Placement is deterministic** (byte-reproducible restore): the scatter visits draws in index order, one thread per expert, instead of racing for a position.

Gate — five shapes (`1` token, `7`, `300`, a small `8`-expert count, and `draws = 0`) against the permutation's definition, not a second copy of the algorithm: monotone offsets anchored at `0` and the draw count, every draw under its own expert with `token = draw / slots`, `draw_indices` a bijection onto `0..draws-1`, ascending draw order within each expert, and widths equal to an independently counted histogram. Passes on `gfx1100`; the `300`-token case leaves `227` of `256` experts present with a widest expert of `94` draws (past one 16-row M tile), and the small-count case has all `8` present.

**Numerical effect of the grouped chunk path — resolved.** The chunk body now always drives `accumulate_routed_batch`; the tiered executor runs the per-token sequence unless `moe_grouped_batch_enabled()` is set, so one process can compare both paths. The grouped path is a correct reorder, not a defect: `test_moe_grouped_batch_parity` runs `moe_out` through both paths on identical experts and routing — bit-identical on the synthetic fixture, and one fp16 ULP of `moe_out` (`9.8e-4 … 2.0e-3`) in the real executor, which is where a reorder lands once the per-layer value is stored fp16.

Through 43 layers that per-layer ULP becomes a final-logit delta of `2.73e-1` against a top-2 margin of `0.820` — a third of the margin — so a bit-exact chunk comparison fails by construction. But it does not move the answer: `test_v4_routed_prefill` runs the same window both ways and the greedy token and the whole 8-token continuation are identical (`320 62 80 5809 ×5`). The bit-exact bar was the wrong one — it asserts reproducibility against the per-token *implementation*, not model correctness, and a MoE's `Σ_k w_k·Expert_k(x)` has no intended summation order (vLLM/SGLang serve prefill with grouped GEMMs). The greedy comparison replaces it. The A-read needs no gather: it takes a row stride and reads each token's row directly out of the batch scratch's per-token 16-row tile.

**Per-token router removed.** The router ran once per row and ended in a `hipStreamSynchronize` to read that row's top-k back, so phase 2b drained the compute queue `C` times per layer — the stall that keeps the CPU from running ahead of the GPU. The router's *device* work was already batched (`moe_router_kernel` is one block per token); only the read-back was per token. The device half is now `dispatch_router(count)`: one gate GEMV over `[count, H]`, one logit widening and one top-k launch for the whole chunk, every buffer addressed by row stride so nothing is gathered first (`ffn_norm_act` is the row-0 prefix of each token's padded tile, `16·H` apart; logits and top-k are contiguous per token). The chunk reads all `C` selections back with **one** synchronisation; decode calls the same dispatcher with `count = 1` and keeps its single-row read-back. New G2 shape: the GEMV primitive takes the activation row pitch (`x_stride`), so a batched caller whose rows sit inside a larger per-token buffer reads them in place; a contiguous caller passes `in_dim`.

Measured (`bench_prefill_ab routed 128`, bank arm, chunk `128`, window `1024`, warm `0`, pristine host per run): `24.181 s` → `23.426 s` (`5.29` → `5.46 tok/s`). The run reads `66.6 GiB` over NVMe, which does not move, so this isolates the ~`0.75 s` of compute the `43 × 127` removed queue drains cost. Gates unchanged: `test_v4_layer_body_chunk_oracle` / `_serial_oracle` / `_compressed_oracle` (`0` differing), `test_v4_prefill_window` `10/10`, `test_v4_routed_prefill` `20/20`, `test_v4_expert_executor`, `test_v4_mla_oracle`, `test_v4_shared_expert_oracle`, `test_v4_grouped_wo_oracle`, `test_v4_graph_head`, `test_v4_engine` green.

**Remaining.** Open in this step: the `M`-keyed dispatcher (grouped above the crossover), and the batched KV/position writes. Separately, **enabling the grouped path by default touches gates, not kernels**: flipping `moe_grouped_batch_enabled()` to default-on makes the chunk path grouped everywhere, so the gates that assert chunk-vs-serial bit-exactness through the real executor — `test_v4_prefill_window` ("WINDOW == SERIAL, BIT-EXACT"), `test_v4_warm_frozen_prefill`, and `test_v4_routed_prefill`'s own D-check — must move their chunk comparison to the greedy-agreement bar, while the per-token path keeps its bit-exact regression check.

### Step 3: WMMA dense GEMM for attention and shared-expert projections

- The same WMMA core as Step 1 (a single group). Use it for Q/KV/O projections (including `v4_grouped_wo`), the compressor/indexer projections, shared-expert W13/W2 and the HC projections, whenever T ≥ 16.
- Below 16 rows (routed prefill with short prompts, and decode), keep the GEMV path. Set the threshold by the dispatcher on M.

### Step 4: Batched causal attention over the chunk

- Replace per-row `run_layer_body_attention_and_norm` with one kernel over a query tile (16 queries × head) using WMMA for QKᵀ and PV, with online softmax plus sink. It reads the composed local and compressed keys once per tile instead of once per query.
- Use SGLang dsv4 attention/indexer as the semantic reference.

### Step 4a: Move the indexer top-k on-device (interposed — see the profile)

`select_indexer_topk` (`v4_layer_body_types.hpp`) selects the CSA layer's `index_topk = 512` compressed rows **on the host**: D2H the candidate scores, `hipStreamSynchronize`, `std::stable_sort` on the CPU, H2D the chosen indices, `hipStreamSynchronize` again. It runs once per token for every CSA layer, so a window pays ~`38k` queue drains — and the drains are why the attention phase (70.7%) has the GPU idle inside it. The profile's `13.8 s` host line for the *router* is the same mechanism at `0.1%` the frequency.

- One G2 kernel that reads the candidate scores and writes the `index_topk` indices, so no score leaves the device and no host sort runs. **Same selection, same descending order, same lower-index ties** — it changes no number, which is why the equivalence gates cannot see it.
- Input and output buffers already exist and are already the right size (`indexer_scores`, `indexer_topk`); the only addition is per-row shared-memory scratch, not VRAM.
- Keep the degenerate branch exact: `candidates <= index_topk` selects every candidate in ascending index order with no padding.
- Test: against the same definition the host version implements (order, ties, the degenerate branch), not against a copy of the kernel.

#### Implementation record

_Done `2026-09-30`._

`v4_indexer_topk_kernel` (G4, in `v4_attention_kernels.hpp`): one block per row, iterative max-extraction over the unselected candidates with a block reduction. The comparison takes the higher score and, on an exact tie, the lower index — a total order, so the result does not depend on the reduction tree and matches the host's descending `stable_sort` exactly. The mask is a `ceil(candidates/8)`-byte bitmap in shared memory, so the kernel adds no VRAM; the caller's score buffer is not modified. `select_indexer_topk` keeps its name and signature and now only launches the kernel, so the call site is unchanged.

Measured (`aeon_chat --phase-profile`, 666-token prompt, `W=4096 C=256`, swept, `n = 1`):

| Phase | host before → after | gpu before → after |
| :--- | :--- | :--- |
| pre-attention (per token) | `13,057` → **`944`** | `15,718` → `14,657` |
| attention+norm (per token) | `37,271` → `48,343` | `46,954` → `46,944` |
| **total** | `66,238` → **`65,145`** | `66,430` → `65,339` |
| **TTFT** | `71.9 s` → **`70.8 s`** | |

**The honest reading: this removed the stall but not the bottleneck.** Pre-attention's *issuing* cost fell `13.0 s → 0.9 s` — the two drains were real and are gone. But the freed CPU immediately ran ahead into the attention loop, whose own per-token issuance then became the exposed critical path (`37.3 → 48.3 s`); the window is bounded by total host time, so prefill moved only ~`1.5%`. This is the profile's third point restated: the pre-attention sync was expensive *in isolation* and never on the critical path while the attention loop's ~`860k` launches (`666 × 43 × ~30`) were.

Gates: `test_v4_real_scale_state` (the real `index_topk = 512` as a strict selection, `0` differing), `test_v4_indexer_oracle`, `test_v4_layer_body_serial_oracle` / `_compressed_oracle`, `test_v4_engine` (`38/0`) green.

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