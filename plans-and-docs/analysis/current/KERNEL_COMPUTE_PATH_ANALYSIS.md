# Kernel Compute-Path Analysis — where the body's time could go

*Status: open analysis. Written 2026-09-22. A first-pass audit of the compute kernels on the prefill body's critical path, in answer to: where are the compute improvement opportunities. Source read: `aeon_moe_fused_w13.hpp`, `aeon_moe_fused_w2.hpp`, `aeon_w4a16_swizzled_gemv.hpp`, `aeon_w4a16_swizzle.hpp`, `v4_gemv.hpp`, `v4_pipeline_ops.hpp`, `v4_norm.hpp`, `v4_attention.hpp`, `v4_layer_body.hpp`, `v4_layer_body_batch.hpp`. No code changed.*

**Subject.** The kernels that run inside a layer body — the routed MoE, the shared expert, the dense projections, and the batch driver that sequences them. This is the *how much compute work is there* question, complementing the [supply hot-path audit](SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md) (how fast bytes arrive) and the [prefill supply review](PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md) (which bytes arrive).

**Scope.** Structure and intensity of the compute as written. **Nothing here is measured** — every figure is an estimate from the code and the model contract, and §6 names the profiling that would settle it. This document does **not** propose an implementation sequence; it names candidates and their ordering.

---

## 1. The verdict

The prefill body batches its **attention** and its **router**, but not its **MoE compute**: the routed experts and the shared expert still run once per token. On a `gfx1100` part that leaves the routed path at roughly **a quarter of the machine's ridge-point intensity** — i.e. hard memory-bound, re-reading the same expert weights once per token that selects them.

The most valuable single change is therefore **batching the token dimension in the MoE**, which the plan has already scoped ("MoE for every token") but not built. Tensor cores are a real second lever but are secondary *while* the path is memory-bound.

| # | Finding | Leverage | Prerequisite |
| :--- | :--- | :--- | :--- |
| **K1** | Routed MoE (and shared expert) run per-token; weights re-read per token | **highest** | none — it completes Step 6's stated shape |
| **K2** | No matrix-core instructions in the tree | medium, gated | K1 (batching must raise intensity first) |
| **K3** | int4 unpack costs ~2 ALU ops per weight element | low, gated | K1, K2 |
| **K4** | Dense/shared path is 1-token scalar GEMV + per-token memset | medium | none (same fix shape as K1) |

---

## 2. K1 — the prefill MoE is per-token GEMV

`run_layer_body_chunk` (`v4_layer_body_batch.hpp`) does the right thing for the first two phases — attention for every token, then the router for every token — and then does **one** layer-wide supply dispatch:

```cpp
experts.on_routing_ready_batch(layer.layer_id, start_position, batch_ids, batch_weights);
for (uint32_t row = 0; row < count; ++row) {          // ← once per token
    run_layer_body_moe_and_post(layer, views[row], start_position + row, stream, experts, observer, pre[row]);
}
```

So dedup collapses `6C` requests to the layer's distinct set for the **H2D** (each distinct expert is uploaded once — Step 6 D2 working), but `accumulate_routed` still runs **per token** and launches the GEMV kernels once per token. Each token that selects an expert re-reads that expert's packed weights from VRAM.

**The intensity arithmetic.** Per token on the routed path, with `H = 4096`, `I = 2048`, `k = 6`, `W4A16` at ~`12 MiB/expert` (packed + scales):

$$\text{weight bytes} = 6 \times 12\,\text{MiB} = 72\,\text{MiB}$$
$$\text{FLOPs} = 6 \times 2\,(I\!\cdot\!H + I\!\cdot\!H + H\!\cdot\!I) = 6 \times 2\,(3 \cdot 2048 \cdot 4096) \approx 302\ \text{MFLOP}$$
$$\text{intensity} = \frac{302\ \text{MFLOP}}{72\ \text{MiB}} \approx 4\ \text{FLOP/byte}$$

A `gfx1100` part (`7900 XTX`-class) sits near a `~62 FLOP/byte` ridge point (`~60 TFLOP fp16 ÷ ~960 GB/s`). So the routed compute runs at roughly **`1/15` of the ridge** — bandwidth-bound by a wide margin.

**Why batching fixes it.** A grouped dispatch over the layer's distinct set processes all tokens that share an expert in one pass, so each expert's `12 MiB` serves up to `C` rows instead of one. Effective intensity scales toward `C` (bounded by the dedup ratio `6C/D(C)`), and the path moves from bandwidth-bound toward compute-bound — which is the regime where K2's tensor cores start to pay.

**Launch storm.** `run_layer_body_moe_and_post` is ~`10` kernel launches + 1 `hipMemsetAsync` + 1 D2D `hipMemcpyAsync` per token per layer. At `C = 64` over 43 layers that is `~27,000` launches for a single chunk. A grouped dispatch would collapse the per-token launches to per-layer ones.

**Honest scale caveat.** Both estimates above come out *small* against the plan's `≈120 ms/prompt-token`: the VRAM re-read is `~72 MiB × 64 ÷ 1 TB/s ≈ 4.6 ms` per layer, and `27,000` launches at `~1 µs` is `~27 ms` per chunk — neither obviously explains `120 ms`. So either VRAM is far slower for these access patterns than the ~1 TB/s figure, or a cost I have not traced dominates (attention phases, the indexer/compressor, per-token serialization on one stream). **This is why §6 proposes attribution before implementation.**

## 3. K2 — no matrix cores in the tree

A search of `src/` for `amdgcn_wmma`, `amdgcn_mfma`, `v_mmac`, `wmma_f32`, `wmma_bf16` returns **one** AMD matrix-adjacent intrinsic: `__builtin_amdgcn_fdot2` (`aeon_w4a16_swizzled_gemv.hpp`), a 2-wide vector dot — ALU work, not tensor cores. The dense path (`v4_gemv.hpp`) is scalar `fmaf` over `uint4`-loaded halves. No fragment path, no `v_mmac`.

AGENTS.md advertises *"Wave32, WMMA, direct HIP/AMDGCN dispatch."* On the evidence of the code, the **Wave32 and direct-intrinsic** parts are present; the **WMMA** part is not present in either the expert or the dense path.

**Ordering:** because K1 shows the path is memory-bound, a WMMA path would raise an ALU ceiling we are not currently hitting. It becomes materially valuable *after* batching pushes intensity toward the ridge, and it requires a fragment layout that the current nibble swizzle (`kNibblePerm`) is not built for. Flag it, do not chase it first.

## 4. K3 — int4 unpack spends ALU on decode

`swizzled_group_dot` runs, per `uint4` (32 K-values): 16 `unpack2` + 16 `fdot2`. Each `unpack2` is the magic-number trick — shift, mask, or, `hsub2`:

```cpp
const uint32_t bits = ((word >> (4 * pair)) & 0x000F000Fu) | 0x64006400u;
return __hsub2(*reinterpret_cast<const half2*>(&bits), __float2half2_rn(1032.0f));
```

That is roughly `~64` integer ops + `16` `fdot2` per 16 bytes of weight — a good trick, but ALU spent decoding rather than multiplying. A 256-entry LUT (Colibri's `e8_table` style) or a pre-dequantized fp16 cache would trade the int work for a table load.

**Ordering:** lowest priority. While memory-bound (K1), ALU-side savings buy nothing. Revisit only once K1 and K2 have moved the path nearer the ridge.

## 5. K4 — the dense/shared-expert path is also 1-token and scalar

`run_layer_body_moe_and_post` runs the shared expert (`ffn.shared_experts.{w1,w3,w2}`, `48 MiB` fp16) as **three separate 1-token GEMVs**:

```cpp
dim3(INTER_DIM, 1), dim3(32)   // v4_gemv_fp16_vec8_kernel, token grid = 1
```

plus a `v4_pipeline_swiglu_clamp_kernel` pass and a full `hipMemsetAsync(d_moe_accum, 0, M_PAD*H)` **every token**. The shared expert fires for every token unconditionally, so its `48 MiB` is re-read per token per layer: at `C = 64`, `48 MiB × 64 × 43 ≈ 132 GiB` per chunk — the same structural issue as K1 at smaller scale, and the same fix (batch the token dimension; clear `moe_accum` once per chunk, not per token).

The three GEMVs and the SwiGLU are also **separate** launches that could fuse the way the routed path already fuses gate/up/SwiGLU (`aeon_moe_fused_w13_swiglu_kernel`).

## 6. Attribution before optimization

The estimates in §2 and §5 are too small, on their own, to explain the plan's `≈120 ms/prompt-token` — which is itself unsourced against the reference ([prefill supply review](PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md) §6). Before writing a grouped-MoE kernel, **attribute a chunk's time per phase** so the work lands on a real cost:

| Phase | What to measure |
| :--- | :--- |
| Pre-attention (RMSNorm, projections, RoPE, compressor, indexer) | per-token vs batched split |
| Attention (three classes) | the sliding/CSA/HCA kernels, and how much is memory vs launch |
| Router | the grouped selection and its host round-trip |
| Shared expert | the three GEMVs + Swiglu + per-token memset |
| Routed MoE | GEMV kernel time vs the re-read, per token |
| Non-kernel | launch count, stream syncs, D2D copies |

Only after that should K1 (and K4) be implemented, and K2/K3 be reconsidered.

## 7. What this analysis does not claim

- **It does not claim the kernels are wrong or unoptimised** — `fdot2`, uint4 loads, and the nibble swizzle are deliberate and reasonable. The claim is structural: they are invoked **per token** where they could serve a batch.
- **It does not claim tensor cores are the answer.** The path is memory-bound; tensor cores are gated behind K1.
- **It does not claim any number here is measured.** All are reasoned from the code and the model contract and marked so. §6 exists to make them measurable before anything is built.
- **It does not re-open numerics.** The fixed-order fp32 reduce and the anti-circularity rules are untouched by any of this.

## 8. Disposition

| Item | Where it goes |
| :--- | :--- |
| K1 — batched routed MoE | Completes the plan's own Step 6 shape ("MoE for every token"); the compute half of the batched prefill. |
| K4 — batched shared expert + fused SwiGLU + per-chunk memset | Same shape as K1; candidate to land with it. |
| Phase attribution (§6) | **First action**; feeds the plan's open `≈120 ms/prompt-token` investigation and the re-sourcing of the reference figure. |
| K2 — matrix cores | Deferred behind K1; revisit when intensity is near the ridge. |
| K3 — int4 unpack | Lowest priority; gated behind K1/K2. |
