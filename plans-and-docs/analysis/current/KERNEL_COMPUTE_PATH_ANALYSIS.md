# Kernel Compute-Path Analysis — where the body's time could go

*Status: open analysis. Written 2026-09-22. Revised 2026-09-23 after a full read of both kernel families, the layer body and batch driver, the expert executor, the model contract and the converter. Source read: `aeon_moe_fused_w13.hpp`, `aeon_moe_fused_w2.hpp`, `aeon_w4a16_swizzled_gemv.hpp`, `aeon_w4a16_swizzle.hpp`, `vram_expert_pool.hpp`, `swizzled_expert_format.hpp`, `expert_format.hpp`, `v4_gemv.hpp`, `v4_pipeline_ops.hpp`, `v4_norm.hpp`, `v4_rope.hpp`, `v4_attention.hpp`, `hc_sinkhorn.hpp`, `moe_router.hpp`, `v4_layer_body.hpp`, `v4_layer_body_batch.hpp`, `v4_expert_executor.hpp`, `v4_layer_state.hpp`, `v4_model_spec.hpp`, `v4_model_contract.hpp`, `v4_prefill_sweep.hpp`, `v4_graph.hpp`, `scripts/convert_safetensors_to_aeon.py`. No code changed.*

**Subject.** Every kernel that can run on a layer body's critical path — the routed MoE, the shared expert, the dense projections, the attention classes, the hyper-connection ops, and the glue that sequences them. This is the *where does the body's time go* question, complementing the [supply hot-path audit](SUPPLY_CHAIN_HOT_PATH_ANALYSIS.md) (how fast bytes arrive) and the [prefill supply review](PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md) (which bytes arrive).

**Scope.** Structure, cost and sequencing of the compute as written. **Nothing here is measured** — every figure is reasoned from the code and the model contract, and §4 names the two experiments that would settle it. §6 gives an ordered plan to work through point by point; each finding below carries an id (`F1`…) so it can be addressed on its own.

---

## 1. How the kernels divide

Three families, and the third — the glue — is where the time is.

| Family | Where | Members | What it is |
| :--- | :--- | :--- | :--- |
| **A — Backend / format** | `src/backend/swizzled_w4a16/` | 4 device kernels (`w13_swiglu`, `w2_accum`, `w2_contrib`, `swizzled_gemv`/`dual`), 2 host swizzle transforms, 2 dispatchers | int4-W4A16 decode + MAC. Knows the quantisation, not the model. |
| **B — Model / architecture** | `src/architecture/deepseek_v4/kernels/` | ~20 kernels: the HC set (`hc_project`, `hc_sinkhorn_normalize`, `hc_pre_combine`, `hc_post`, `hc_head`), `rmsnorm`×2, `rope`×4, `gemv`×2, the three attention classes, `save_compressor_state`, `materialize_compressed_entry`, `indexer_scores`, `moe_router`, `argmax`×2 | DSV4 semantics. |
| **C — Glue** | inside `v4_layer_body.hpp` / `v4_layer_body_batch.hpp` | `half_to_float`, `float_to_half`, `hipMemsetAsync`, ~20 `hipMemcpyAsync`, 3 host round-trips | Neither a primitive nor a model kernel — data motion and sequencing. |

A token through one **CSA** layer issues roughly:

| Phase | Kernels | Memcpy / memset | Host syncs |
| :--- | ---: | ---: | ---: |
| Pre-attention (A) | 21 | 3 | 2 (indexer top-k) |
| Attention → FFN norm (E–G) | 12 | 15 (M_PAD) | 0 |
| Router (H) | 4 | 3 | 1 |
| MoE + post | 8 | 2 | 0 |
| **Total** | **≈74** | **≈23** | **3** |

Sliding layers are ~10 launches lighter (no compressor/indexer); HCA ~8 lighter. Over 43 layers: **≈3,100 launches and ≈130 host synchronisations per prompt token.** Family C accounts for roughly half the launches and for every stall.

## 2. The verdict

The body is **issuance-bound**: it is bound by how fast work can be submitted and synchronised, not by arithmetic and not by DRAM. Three floors, against a measured ≈`2.8 ms` for one token through one layer — the plan's `~120 ms/prompt-token` is this figure × 43, and it reconciles exactly with the sweep's *"~1.4 s of compute [per layer]"* at a 512-token window.

- **Compute floor** — ≈700 MFLOP per token-layer (routed 302 + shared 50 + MLA/indexer/compressor ≈264 + attention ≈84), ≈30 GFLOP per token over 43 layers.
  $$30\ \text{GFLOP} / 60\ \text{TFLOP/s} \approx 0.5\ \text{ms per token}$$
- **Bandwidth floor** — the dense fp16 weights per CSA layer are **≈294 MiB** (`wq_b` 64 + `wo_a` 64 + `wo_b` 64 + shared 48 + `wq_a`/`wkv`/compressor 28 + indexer 20 + gate/misc 2.5), plus **≈81 MiB** of resident routed experts (`6 × 13.5`), ≈375 MiB total.
  $$375\ \text{MiB} \times 43 / 960\ \text{GB/s} \approx 16\ \text{ms per token}$$
- **Measured** — ≈120 ms per token.

Compute is ≈2% of the measured time and DRAM ≈13%. **≈85% is unaccounted for by any kernel's arithmetic or its weight traffic** — it is launch submission and host synchronisation. At `120 ms ÷ ≈3,180 launch slots` that is ≈`38 µs` per slot; a ROCm launch is ≈`5–10 µs`, so even a fully serialised queue of free launches is ≈25 ms, and the remainder is the cost of the per-token `hipStreamSynchronize` calls draining the queue so the CPU cannot run ahead.

**Consequence for the levers.** Arithmetic intensity cannot rank fixes here: a path running at 13% of its bandwidth bound *and* 1% of its compute bound is on neither roofline. Reducing submissions and synchronisations is the first lever; everything downstream becomes visible only after it. The routed MoE *is* run per token where it could be per layer (F6) — that remains true — but its byte saving is `6C/D(C) ≈ 1.9×` at the plan's `C = 64` (not `C×`), and it sits inside a measured 120 ms that compute and bandwidth already fail to explain. It is a middle-priority item, not the first.

---

## 3. Findings

Grouped by the decision each one feeds. Each has a stable id so it can be taken on its own.

| # | Finding | Leverage | Prerequisite |
| :--- | :--- | :--- | :--- |
| **F1** | Path is issuance-bound; syncs are per-token where they could be per-layer | **highest** | none |
| **F2** | `M_PAD` 16-row replication is dead work | high (pure deletion) | none |
| **F3** | Batch row-set assembly is `O(C × window)` tiny D2D copies and serialises the chunk | high (batch path) | none |
| **F4** | Legacy atomic W13 path has a zeroing race | correctness (gate-only) | none |
| **F5** | Dense MLA path is the largest byte term and has true `C×` headroom | **highest byte lever** | F1 |
| **F6** | Routed MoE is per-token; win is launch collapse + ~1.9× bytes at `C=64` | medium | F1 |
| **F7** | Attention softmax executed 32×; per-slot wave reductions | medium | none |
| **F8** | `hc_project` recomputes the same RMS 24× per site | medium | none |
| **F9** | Dense GEMV is scalar fp32 ALU on fp16 data | medium, gated | F5 |
| **F10** | HC residual carried in fp16 across 43 layers | correctness to measure | none |
| **F11** | No matrix cores in the tree | ≈0 today | F1, F5 |
| **F12** | int4 unpack spends ALU on decode | low, gated | F1, F5 |
| **F13** | Expert layout contract lives in three places; converter verifier is circular | seam integrity | none |
| **F14** | Model shape constants duplicated as kernel constants | seam integrity | none |
| **F15** | Large kernarg blobs passed by value per launch | low | F1 |
| **F16** | Shared expert runs as 3 GEMVs + SwiGLU in 4 launches | low | F6 |
| **F17** | Config declares `expert_dtype: fp4` while `quant.type: int` | correctness (silent-wrongness door) | none |

### F1 — the path is issuance-bound

`run_layer_body_chunk` (`v4_layer_body_batch.hpp`) batches attention and the router, then does **one** layer-wide supply dispatch:

```cpp
experts.on_routing_ready_batch(layer.layer_id, start_position, batch_ids, batch_weights);
for (uint32_t row = 0; row < count; ++row) {          // ← once per token
    run_layer_body_moe_and_post(layer, views[row], start_position + row, stream, experts, observer, pre[row]);
}
```

Dedup collapses `6C` requests to the layer's distinct set for the **H2D** — each distinct expert uploads once, Step 6 D2 working — but `accumulate_routed` still runs per token. More importantly, the batch's own structure is undermined by synchronisation: `run_layer_body_router` ends in `hipStreamSynchronize`, and phase 2b calls it **per token**, so the layer-wide dispatch is preceded by `C` queue drains. The sync only has to happen once, after all `C` selections are known. `select_indexer_topk` (`v4_layer_body.hpp`) syncs **twice** per token per CSA layer and additionally heap-allocates three `std::vector`s and runs a host `std::stable_sort` over up to 512 floats inside the body's hot loop.

*Why it dominates:* it is what stops the CPU running ahead; every kernel optimisation downstream is invisible while it stands.
*Verify:* count `hipStreamSynchronize` per token per layer and time the syncs (§4, Experiment A).

### F2 — the `M_PAD` replication is dead work

`run_layer_body_attention_and_norm` replicates row 0 of `d_ffn_norm_act` into rows 1..15, labelled *"WMMA compatibility"*. There is no WMMA, and every consumer reads only row 0: `swizzled_group_dot` indexes `activation + group·32` for `group < ITERS·LPR = 128`, i.e. offsets `[0, 4096)` — exactly one row — and every gate reads `upload_and_read(0, …, kHidden)`. Consequences:

- 15 dead D2D copies × 43 layers = **645 dead submissions per token**;
- `d_ffn_norm_act` and `d_moe_accum` are `16×` over-allocated per token in `V4LayerBodyBatchScratch` (`rows × kMPad × H`), ≈64 MiB wasted at `C = 256`;
- `hipMemsetAsync(d_moe_accum, 0, M_PAD·H)` clears 4× what any consumer touches.

*Why it matters:* it is a backend-format assumption (16-row tiles) taxing the model body after the assumption stopped being true — the two families competing, concretely.
*Verify:* delete the padding loop and the padding, run the layer-body oracles; nothing should change.

### F3 — batch row-set assembly is ≈16k tiny copies per layer

`compose_local_rows` (`v4_layer_body_batch.hpp`) submits two `hipMemcpyAsync` per ring slot per query — one 1 KiB key row and one 8-byte position:

$$C \times 128 \times 2 \approx 16{,}000\ \text{submissions per layer at } C=64$$

That is ≈215× the rest of the body's copy count, **and it serialises the chunk**: it writes into the single shared `workspace.composed_keys()` buffer, so query `r+1`'s copies cannot start until query `r`'s attention kernel has consumed it.

*Where:* `compose_local_rows`.
*Verify:* count submissions per layer in the batch path; replace with a gather kernel or indirect positional indexing and measure.

### F4 — legacy atomic W13 zeroing race

`aeon_moe_fused_w13_swiglu_kernel` (`aeon_moe_fused_w13.hpp`) zeroes `output_f32` from the `blockIdx.y == 0` blocks **inside the same launch** whose other `y` blocks `atomicAdd` into it, with no cross-block ordering. Production passes `output_f32 = nullptr`, so the committed path is safe, but any gate using that path is unsound.

*Action:* forbid `output_f32 != nullptr`, or zero in a separate launch.

### F5 — the dense MLA path is the largest byte term

Per CSA token-layer the dense projections are **≈294 MiB** against the routed path's **≈81 MiB** — 3.6× larger — and they are re-read per token with **no dedup possible**, because dense weights are shared by all tokens. Their `C×` headroom is therefore *true* headroom, unlike the routed path's `6C/D(C)`:

$$294\ \text{MiB} \times C \;\rightarrow\; 294\ \text{MiB}$$

*Where:* `v4_gemv.hpp` (`v4_gemv_fp16_kernel`, `v4_gemv_fp16_vec8_kernel`), driven from `v4_layer_body.hpp` for `wq_a`, `wq_b`, `wkv`, the compressor, the indexer and `wo_b`; `v4_grouped_wo_a_wave32_kernel` for `wo_a`.
*Fix shape:* a **true GEMM** — a kernel that holds a weight tile once and multiplies it against all `C` tokens — not a grouped GEMV, because the vec8 kernel re-reads `W` per token by construction.
*Verify:* batch one projection type end to end at `C = 64`; compare bytes and time.

### F6 — the routed MoE is per-token

Each token that selects an expert re-reads that expert's packed weights from VRAM. The byte saving from batching is bounded by the dedup ratio, not by `C`:

$$\text{byte reduction} = \frac{6C}{D(C)}, \qquad D(C) \approx 256\left(1 - e^{-6C/256}\right)$$

| chunk `C` | `D(C)` (uniform) | byte reduction | note |
| ---: | ---: | ---: | :--- |
| 1 | 6 | 1.0× | baseline |
| 64 | ≈199 | **1.9×** | the plan's default chunk |
| 256 | 256 | **6.0×** | the per-layer ceiling (`kMaxTokens`) |

*Real value:* launch collapse — ≈10 launches + 1 memset + 1 D2D copy per token per layer → per layer — and L2 residency of the reused expert, not the byte count at `C = 64`.
*Where:* `v4_layer_body_batch.hpp` phase 2c; `v4_expert_executor.hpp` (`accumulate_routed`); `aeon_moe_fused_w13.hpp`, `aeon_moe_fused_w2.hpp`.

### F7 — attention softmax runs 32× redundantly

In both `v4_cached_sliding_window_attn_wave32_kernel` and `v4_cached_compressed_attention_wave32_kernel` (`v4_attention.hpp`), phase 1 stores the scores to shared memory, then **every lane independently** loops all keys doing `fmaxf` and `expf`. For CSA with `local(128) + compressed(512) = 640` keys that is `640 × 32 = 20{,}480` `expf` per (head, token) where 640 suffice — a ~32× overcompute on a multi-instruction transcendental. Phase 1 also runs one full 5-step wave reduction *per key slot* serially (up to 640 reductions).

*Fix shape:* lane-strided softmax over the shared score array; register-block the phase-1 reduction over multiple slots.
*Verify:* the attention oracles are byte-comparable, so a change here must keep them exact.

### F8 — `hc_project` recomputes the same RMS 24× per site

`hc_project_kernel` (`hc_sinkhorn.hpp`) launches `grid(24)` — one block per mix row — and every block recomputes the full 16,384-element sum-of-squares and re-reads the 64 KiB residual. Once per site, twice per layer per token: a 24× redundant recompute and ≈1.5 MiB of redundant L2 traffic per site. At `C = 64` that is ≈8 GiB per window — comparably large to a third of the dense weight traffic.

*Fix shape:* compute the RMS once (one block or a two-pass launch) and pass it in.
*Verify:* `test_v4_hc_oracle.cpp`.

### F9 — dense GEMV is scalar fp32 ALU on fp16 data

`v4_gemv_fp16_kernel` and `v4_gemv_fp16_vec8_kernel` (`v4_gemv.hpp`) use `__half22float2` + scalar `fmaf` — no `fdot2`, no `__hfma2`, no half2 arithmetic. The AMD `fdot2` intrinsic exists in the tree but only in the backend kernels.

*Why gated:* it is ALU work on a path currently at ≈1% of its compute bound; it becomes valuable only once F1/F5 move intensity.

### F10 — HC residual is carried in fp16 across layers

`hc_post_kernel` writes `residual_out` as `__half`, and each layer round-trips `float→half` (`res_in`), `half→float` (`res_mid`), `half→float` (`res_out→res_in`) — three conversions plus a D2D copy per token-layer. The CPU reference (`cpu_hc_post`) is `float` throughout. Whether the fp16 carry matches the oracle **across 43 chained layers** is a depth-accumulating precision question that a per-primitive gate cannot see.

*Verify:* drive ≥43 chained layers against the fp64 reference and measure drift versus depth, not at one layer.

### F11 — no matrix cores

`amdgcn_wmma`, `amdgcn_mfma`, `v_mmac`, `wmma_f32`, `wmma_bf16` are absent from `src/`; the only AMD matrix-adjacent intrinsic is `__builtin_amdgcn_fdot2` (a 2-wide vector dot — ALU, not tensor core). AGENTS.md advertises *"Wave32, WMMA, direct HIP/AMDGCN dispatch"*; the Wave32 and direct-intrinsic parts are present, the WMMA part is not — in either the expert or the dense path.

*Ordering:* at ≈1% of the compute bound a tensor-core path is unobservable. It becomes valuable only after F1/F5 push intensity toward the ridge, and it needs a fragment layout the current nibble swizzle (`kNibblePerm`) is not built for.

### F12 — int4 unpack spends ALU on decode

`swizzled_group_dot` (`aeon_w4a16_swizzled_gemv.hpp`) runs, per `uint4` (32 K-values): 16 `unpack2` + 16 `fdot2`. Each `unpack2` is the magic-number trick — shift, mask, or, `hsub2`:

```cpp
const uint32_t bits = ((word >> (4 * pair)) & 0x000F000Fu) | 0x64006400u;
return __hsub2(*reinterpret_cast<const half2*>(&bits), __float2half2_rn(1032.0f));
```

That is roughly `~64` integer ops + `16` `fdot2` per 16 bytes of weight — a good trick, but ALU spent decoding rather than multiplying. A 256-entry LUT (Colibri's `e8_table` style) or a pre-dequantised fp16 cache would trade the int work for a table load.

*Ordering:* lowest. It applies to the backend family (≈20% of the traffic) and only pays once the path nears the ridge.

### F13 — the expert layout contract lives in three places

The byte offsets (`swizzled_expert_format.hpp`), the swizzle config (`aeon_w4a16_swizzle.hpp`), and the converter's `EXPERT_TENSOR_LAYOUTS` + `NIBBLE_PERM` (`scripts/convert_safetensors_to_aeon.py`) all encode the same layout. The C++ constants are bound to each other by `static_assert`s; the Python is not. Worse, `verify_swizzled_expert_payload` unswizzles with the same constants it swizzled with, so **it is circular** — it proves self-consistency, not agreement with the C++ kernel. The only non-circular binding is the C++ oracle gate (`test_v4_expert_oracle.cpp`).

*Action:* generate the Python layout from the C++ header, or make the verifier test the boundary (read the kernel's constants, or replay the C++ unswizzle) rather than itself.

### F14 — model shape constants duplicated as kernel constants

`kernel::DSV4_*` in `v4_attention.hpp` mirror `DeepSeekV4Config`, which `V4ModelSpec::validate_config` checks separately. They agree today; nothing prevents drift.

*Action:* a `static_assert` tie (or a single generated header) between the two.

### F15 — large kernarg blobs

`SwizzledW13ExpertPtrs` is 8 experts × 4 pointers = **256 bytes passed by value per launch** (plus 128 B for W2). At ≈3,100 expert launches per token this is pure launch-path cost. A device-resident pointer table indexed by a small struct would remove it.

*Where:* `aeon_moe_fused_w13.hpp`, `aeon_moe_fused_w2.hpp`; call site `v4_expert_executor.hpp`.

### F16 — shared expert launches are separate

The shared expert runs in `run_layer_body_moe_and_post` (`v4_layer_body.hpp`) as `w1` GEMV, `w3` GEMV, a SwiGLU pass, and a `w2` GEMV — four launches that could fuse the way the routed path already fuses gate/up/SwiGLU into `aeon_moe_fused_w13_swiglu_kernel`. It also fires for every token unconditionally (48 MiB re-read per token — the same shape as F6 at smaller scale), and its `moe_accum` is cleared with a full `M_PAD`-length memset every token.

*Fix shape:* one fused launch per shared expert; clear `moe_accum` once per chunk rather than per token; batch the token dimension alongside F6.

### F17 — config self-contradicts on expert precision

`expert_dtype: "fp4"` (`config.hpp`) sits alongside `quant.type: "int"` and a symmetric `group_size: 32`. The kernel decodes two's-complement int4 with zero point 1032 (`((word >> 4p) & 0x000F000F) | 0x64006400`, then `__hsub2(…, 1032)`), which matches the `INT4-W4A16` checkpoint. The two decodings differ numerically — E2M1 is lossy and unsigned — so a loader that ever honoured `expert_dtype` would silently mis-scale.

*Action:* assert two's-complement int4 in the loader/contract, and reconcile or remove the `fp4` field.

## 4. Attribution before optimization — the two experiments that decide the rest

§2 leaves ≈85% of a measured `120 ms/token` unattributed. The work below should not begin until it is attributed, because the attribution is what sets the order. Two cheap experiments settle it, in this order.

**Experiment A — the launch and sync ledger (decisive, no code changes).** Count `hipLaunchKernelGGL` / `hipMemcpyAsync` / `hipMemsetAsync` / `hipStreamSynchronize` per phase, and time only the syncs. Prediction, stated so it can be refuted: ≈3,100 launches and ≈130 syncs per token, with the syncs accounting for the majority of the 120 ms. If it holds, F1 is first and the rest follows; if the syncs are cheap, the cost is launch submission, and F1/F2/F3 are still jointly first.

**Experiment B — hoist the syncs (small, local, gate-visible).** Move the indexer top-k on-device and replace the router's per-token readback with one readback per layer, after phase 2b. This changes no arithmetic — the oracles should stay exact — and isolates the issuance question from everything else.

Then attribute per phase, to place the remaining work on a real cost:

| Phase | What to measure |
| :--- | :--- |
| Pre-attention (RMSNorm, projections, RoPE, compressor, indexer) | per-token vs batched split; submissions |
| Attention (three classes) | the sliding/CSA/HCA kernels; memory vs transcendental vs launch |
| Router | selection cost and its host round-trip |
| Shared expert | 3 GEMVs + SwiGLU + per-token memset |
| Routed MoE | GEMV time vs the re-read, per token |
| Non-kernel | launch count, stream syncs, D2D copies, row-set assembly |

*Caveat.* Every figure in this document is analytic; nothing was run. The two facts most worth confirming on silicon are (a) the per-`hipStreamSynchronize` cost, which is what makes the ≈85% land on syncs rather than launch overhead, and (b) that the sweep's `~1.4 s/layer` is per 512-token window — the whole per-token-layer arithmetic in §2 depends on it, and it reconciles the plan's 120 ms exactly, which is strong but indirect evidence.

## 5. What this analysis does not claim

- **It does not claim the kernels are wrong or unoptimised** — `fdot2`, uint4 loads, and the nibble swizzle are deliberate and reasonable. The claim is structural: they are invoked **per token** where they could serve a batch.
- **It does not claim tensor cores are the answer.** The path is issuance-bound; tensor cores are gated behind F1/F5.
- **It does not claim any number here is measured.** All are reasoned from the code and the model contract and marked so. §4 exists to make them measurable before anything is built.
- **It does not re-open numerics.** The fixed-order fp32 reduce and the anti-circularity rules are untouched by any of this.

## 6. Recommendations — how to proceed

Ordered. Each step names the finding it discharges, the gate that must stay green, and how to verify. Work top to bottom; do not start a step before its prerequisite's verification passes.

**Step 1 — Attribute (do not optimise yet).** Run Experiment A (§4). Discharges nothing; decides the order below.
*Gate:* none — measurement only.

**Step 2 — Hoist the synchronisation (F1).** On-device indexer top-k; one router readback per layer. This is the change that makes everything below it visible.
*Gate:* `test_v4_layer_body_oracle`, `test_v4_layer_body_chunk_oracle`; result must stay byte-identical. *Verify:* Experiment B.

**Step 3 — Delete the dead replication (F2).** Remove the `M_PAD` padding loop, shrink `ffn_norm_act`/`moe_accum` to one row per token, fix the memset length.
*Gate:* the same oracles. *Verify:* zero-value diff; VRAM accounting drops.

**Step 4 — Fix the batch row-set assembly (F3).** Replace the per-row memcpy with a gather kernel or indirect positional indexing.
*Gate:* item 19's bit-exact chunk-vs-serial equality. *Verify:* submissions per layer collapse from ≈16k.

**Step 5 — Batch the dense MLA projections (F5).** A true GEMM holding a weight tile across `C` — the biggest single byte term and the only one with `C×` headroom.
*Gate:* the projection oracles; batch-vs-serial equality. *Verify:* bytes and time for one projection type at `C = 64`.

**Step 6 — Close the seam and correctness items that need no measurement.** F4 (forbid the racy `output_f32` path), F17 (assert int4), F13 (de-duplicate the layout contract; de-circularise the converter verifier), F14 (tie the shape constants).
*Gate:* the existing oracle gates and the loader contract.

**Step 7 — Leaner arithmetic.** F7 (softmax 32×, phase-1 reductions), F8 (`hc_project` RMS 24×), F16 (fuse the shared expert).
*Gate:* the corresponding oracles, kept exact.

**Step 8 — Measure the fp16 residual at depth (F10).** Drive ≥43 chained layers against the fp64 reference; decide fp16 vs fp32 from the drift-vs-depth curve, not from a single-layer gate.

**Step 9 — Grouped / batched routed MoE (F6).** Worth doing for launch collapse and L2 residency; expect ≈1.9× bytes at `C = 64`, not `C×`.
*Gate:* the expert oracle and the fixed-order accumulation contract.

**Step 10 — Deferred, gated behind the above.** F9 (half2 GEMV) once F5/F1 raise intensity; F11 (tensor cores) once intensity nears the ridge; F12 (int4 LUT); F15 (kernarg blobs).

| Finding | Disposition |
| :--- | :--- |
| F1 | Step 2 — first; the gate everything else waits on |
| F2 | Step 3 — pure deletion, no numerics |
| F3 | Step 4 — batch path only |
| F4, F13, F14, F17 | Step 6 — correctness and seam, no measurement needed |
| F5 | Step 5 — the highest byte lever |
| F6, F16 | Step 9 — batched MoE and shared expert |
| F7, F8 | Step 7 — contained ALU/L2 waste |
| F10 | Step 8 — measure, then decide |
| F9, F11, F12, F15 | Step 10 — deferred, gated |
