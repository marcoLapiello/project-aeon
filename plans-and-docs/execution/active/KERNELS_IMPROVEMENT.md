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