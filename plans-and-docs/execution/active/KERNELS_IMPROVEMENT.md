Every compute kernel in prefill today is a GEMV (one token at a time). The chunk path loops over tokens on the host for every stage, so each chunk re-reads the weights T times. Nothing uses WMMA: the 16-row padding in scratch is there, but no kernel reads it. So the swept prefill runs at GEMV rate (bandwidth-bound, around 1 flop per byte) when it could run at matrix-multiply rate. That is the main headroom.

---

## Implementation record

_Updated `2026-09-29`. The plan above is unmodified; this section records what has been built, what it measured, and the revisions those measurements imply. Milestone numbers live in the [Performance & Accuracy Ledger](../status/PERFORMANCE_LEDGER.md) as `M48`._

### Step 1 — kernels built and gated; now believed compute-bound

Built, smallest verified piece first:

| Artefact | Role | Gate |
| :--- | :--- | :--- |
| `src/platform/rdna3/wmma.hpp` | G2 primitive: the Wave32 `v_wmma_f32_16x16x16_f16_w32` and its operand lane map | `tests/test_rdna3_wmma_oracle.cpp` |
| `src/backend/swizzled_w4a16/kernels/aeon_moe_grouped_wmma.hpp` | G3: grouped W13+SwiGLU and W2 over a token→expert permutation | `tests/test_v4_grouped_wmma_oracle.cpp` |
| `tests/bench_expert_pair_ab.cpp` | GEMV pair vs grouped pair: time, weight traffic, M window, crossover | `M48` |

The lane map was pinned on silicon before anything was built on it. It is not documented in one place, and a wrong reading still runs and still returns finite numbers — the common failure computes `A · Bᵀ`, which an identity-A test cannot see.

**Measured (`M48`, `T = 16 / 64 / 256 / 1024`, real layer-0 routing):**

| T | tok/expt | GEMV ms | grouped ms | speedup | traffic (GEMV → grouped → floor) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1.9 | `2.56` | `3.19` | `0.80x` | `1.27 → 0.67 → 0.67 GiB` |
| 64 | 5.1 | `10.17` | `4.76` | `2.14x` | `5.06 → 0.99 GiB` |
| 256 | 20.2 | `39.65` | `5.53` | `7.17x` | `20.25 → 1.69 GiB` |
| 1024 | 80.8 | `159.06` | `9.99` | `15.93x` | `81.00 → 1.15 GiB` (floor `1.00`) |

Still open, in the order they should land:

1. **Permutation kernel** (`token→expert` sort + offsets). Its only consumer is Step 2's batched router, so it should land with that rather than as a standalone.
2. **Executor wiring** — dispatch the grouped pair from `V4TieredExpertExecutor` above the crossover, keep the GEMV pair below it.
3. **Routed-prefill / short-prompt path** — same grouped kernel; per-expert GEMV fallback for 1–3 token experts.
4. **LDS double-buffering** — deferred pending the measurement below, which decides whether it can pay.

### Revisions the measurements imply

1. **Step 3's `T ≥ 16` threshold is too thin.** The measured crossover is between `T = 16` (`0.80x`, grouped loses) and `T = 64` (`2.14x`). `T ≥ 64` is where the choice stops being close.
2. **Step 6's "choose the chunk size so the average tokens per expert reaches ≥ 16" is the wrong stopping rule.** `16` is reached at `T = 256`, yet the reward keeps rising well past it (`7.17x` → `15.93x`). Chunk size, layer-major window and the swept-prefill gate stay **user-configurable**; the deliverable here is the *curve* above, reported, not a constant baked into the engine.
3. **"K split into 128-wide groups" became 64.** Two 32-wide quantization groups map exactly one int4 load per thread, which is what makes the staging branch-free. 128 would need two loads per thread.
4. **Chunk and window are not tuning constants.** They are user settings; the engine must accept them and the measurement above tells the user what they buy. Nothing in this plan should hardcode them.

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

### Step 2: Batch the chunk loop in the layer body

- Rewrite `run_chunk` so every stage takes a `[T, dim]` activation. Remove `std::vector<...>` views/pre/outputs per row.
- Router: one batched gate GEMM, then a top-k kernel over all T rows that writes device-side ids and weights. Hand these to the Step 1 permutation kernel. Copy the ids to the host once per chunk, asynchronously, only for the supply hint (`on_routing_ready_batch`), with no stall on the compute stream.
- Batch the KV-cache/position writes into one kernel (position computed on the device, no H2D per row).

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

---

## Open measurement: is the grouped kernel out of weight bound, or out of compute?

The `M48` window sweep shows weight traffic falling `1.55x` from window 4 to window 8 at `T = 1024` while elapsed time is flat (`10.07` vs `9.99 ms`). Reuse that buys no time means the kernel is no longer bound by the stream. Resolved by measurement — see "Attribution" below.

## Attribution: what the gate half actually spends its time on

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