# Inference Pipeline Plan
## DeepSeek-V4-Flash-0731 — Full Text-In / Text-Out Procedure

**Target hardware:** AMD Radeon RX 7900 XTX (RDNA 3, gfx1100)
**Model:** the `.aeon` artifact (`models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon`), byte-identical repack of the `yiminyuan` safetensors checkpoint
**Declared architecture:** `model_type: deepseek_v4`, `architectures: ["DeepseekV4ForCausalLM"]`, 43 layers
**Quantization:** W4A16 — INT4 weights (group size 32, symmetric), fp16/bf16 activations
**Memory model:** Three-tier — hot (VRAM), warm (host RAM), cold (SSD)

This is a conceptual plan. No code. Each step states what must happen numerically, what precision it happens in, and what the kernel must guarantee.

> **Revision note (2nd revision).** This document was rewritten after an audit found it
> described a *generic* transformer in several places rather than this architecture. The
> first revision omitted Hyper-Connections entirely, placed Sinkhorn on the wrong operand,
> described vanilla MHA instead of MLA, specified a single RoPE base, and stated the
> indexer as INT8. Those are corrected below. Corrections are marked `[corrected]` where
> the change is semantic, so the churn stays visible.

### Evidence convention (binding)

Every factual claim in this document carries one tag. **A claim with no tag is to be treated as unverified, not as a statement of fact.**

- `[V]` — verified against a cited source (file:line, or a config/artifact field).
- `[?]` — plausible but not verified. **Must be settled empirically at the step that depends on it.**
- `[I]` — inferred / from memory. Treat as a hypothesis.

A `[V]` tag is only valid if the cited source is authoritative for **RDNA 3 + this W4A16 artifact**. Cite the tag's source and its platform caveat together.

### Reference hierarchy (and what each may be used for)

| Rank | Source | Authoritative for | NOT authoritative for |
|---|---|---|---|
| 1 | `aeon-references/vllm` @ `94848ed`, `vllm/models/deepseek_v4/**` and `vllm/model_executor/layers/mhc.py` | **Graph semantics only** — op order and formulas | formats, kernels, batching, quantization, runtime |
| 2 | our `config.json` + `.aeon` index | **Parameters** — dims, thetas, ratios, limits | op semantics |
| 3 | `aeon-references/ds4` @ `6289c51` | **Executable cross-check** (RDNA, full CPU graph) | model identity — it serves multiple families; comments are not proof of which graph a path belongs to |
| 4 | our own kernels / format layer | storage layout, swizzle | numerical correctness (circular — see Part V) |

**Platform caveat (mandatory):** vLLM's DeepSeek-V4 ROCm path is gated on `_ON_GFX950` (MI350/CDNA4) and its quant config accepts only `fp8` / `deepseek_v4_fp8` / Quark-MXFP4-OCP `[V vllm/platforms/rocm.py:226, vllm/models/deepseek_v4/quant_config.py:142-165]`. There is **no `rdna`/`gfx11` path** in that tree `[V grep]`. Our checkpoint is `compressed-tensors`/`pack-quantized`/int4 `[V config.json]`, which does not match. Therefore vLLM is a **graph-semantics reference only** — never cite it for storage, kernels, or batching.

> **Hyper-Connections were re-cited (Tier 0.1, done).** The HC steps — **2.0, 2.6, 2.7,
> Step 3** — were re-read from `vllm/model_executor/layers/mhc.py` and
> `vllm/model_executor/kernels/mhc/torch.py` + `triton.py`, and agree with the `ds4`
> cross-check. `[V]` tags are now confirmed. **One error was found and fixed:**
> `hc_sinkhorn_eps` is added to **every** denominator, not row denominators only.
> Confirmed parameters: `hc_post_mult_value = 2.0` (hardcoded constant, not a config key),
> `sinkhorn_repeat = 20`, `hc_pre_eps = hc_sinkhorn_eps = config.hc_eps = 1e-6`.

> **Remaining soft spot.** The state layout described in Step 7 is inherited from our current
> per-request ring implementation. The canonical design is **paged / block-addressable** —
> see Part I §6, which is a design constraint, not an optional optimization.

> **Tier 0.2 (done) — `ds4`-only citations re-checked against readable DeepSeek-V4 code.**
> The router, compressor, SwiGLU and indexer were re-read from `vllm/.../dsv4_topk.py`,
> `vllm/.../activation.py`, `vllm/.../fused_compress_quant_cache.py`,
> `vllm/.../fused_indexer_q.py`, and `sglang/.../attention/dsv4/**`. Results:
> router/SwiGLU/compressor **confirmed**; indexer ReLU **confirmed required**;
> `tid2eid` is `[vocab, 6]`. A useful side effect: for several ops the **readable sglang
> DSV4 kernels are a better arbiter than ds4** because they are model-specific and not
> obfuscated inside a compiled library.

> **Tier 0.2b (done) — MLA, sink, RoPE, grouped output.** Re-read from
> `sglang/.../models/deepseek_v4.py`, `sglang/.../kernels/ops/attention/dsv4/**`, and
> `vllm/.../mhc`-adjacent ops. **Confirmed:** Q-norm between `wq_a`/`wq_b`; per-head norm
> over `head_dim=512`; `wkv`→`kv_norm`; **K=V** via a single `unified_kv` tensor; sink as a
> *"virtual K with V=0"* entering max+denominator only; `wo_a` as `[G,R,D]` with
> `einsum("tgd,grd->tgr")`.
> **Corrected:** the Tier 0.2 statement that the indexer Hadamard "does not appear anywhere"
> was **wrong** — the Hadamard *is* in the DSV4 tree (`dsa_indexer.py:192-204`, ROCm-registered)
> but is **logit-preserving**, so it is a quantization-conditioning choice, not a missing graph op.
> Neither of this plan's two earlier claims about it was accurate; the truthful statement is
> in 2.4.3. A second item (the MoE router's normalization guard) remains a practice-level
> choice rather than a semantics question.

> **Tier 0.2c (done) — two-base RoPE, YaRN, compressor window, and a missing op.** Re-read
> from `vllm/.../deepseek_v4/common/rope.py`, `vllm/.../rotary_embedding/deepseek_scaling_rope.py`
> (`DeepseekV4ScalingRotaryEmbedding`), `vllm/.../common/ops/fused_compress_quant_cache.py`,
> `vllm/.../common/ops/save_partial_states.py`, plus the `ds4` cross-check. **All RoPE and
> compressor-window claims confirmed** — tail-64 rotation, GPT-J interleave, two bases,
> YaRN-on-compressed-only, *no amplitude scaling*, compressed position `pos+1−ratio`, window
> `(1+overlap)·ratio`, per-dim softmax, RMSNorm. **One semantic op was found missing from the
> plan entirely: the compressor APE** — a learned absolute-position embedding added to the
> *score* branch before the window softmax. It is a real trained tensor in our checkpoint (62
> of them, fp32), so it is **required**; it is now in 2.4.2. Also resolved: APE's native layout
> is `[ratio, coff·head_dim]` (vLLM + checkpoint order), **not** ds4's transposed
> `[coff·head_dim, ratio]`. One near-miss avoided: the *generic* rope class rotates the first
> `rotary_dim`; DSV4 uses the subclass that rotates the **last** `rotary_dim` — citing the
> parent would have rotated the wrong 64 dims.

---

## Part I — Foundational Decisions

### 1. Precision Policy

| Component | Precision | Rationale |
|---|---|---|
| Routed expert weights | INT4 (group 32) | `[V config.json]` Storage and bandwidth. Dequantized on the fly. |
| Dense backbone weights | fp16 (or bf16) | `[V contract]` Resident in VRAM. Small relative to experts. |
| Activations | fp16 or bf16 | Must not be quantized to a *narrow integer* format; attention/compressor outliers collapse. |
| Attention logits accumulation | fp32 | `[V]` Required for numerical stability (reference accumulates in f32). |
| KV cache | fp16/bf16 **or** fp8 E4M3 | **`[corrected]`** Canonical is UE8M0-block-scaled fp8, but bf16 is a valid supported alternative `[V vllm attention.py:124-128]`. Our choice is an explicit storage decision, not a fidelity requirement. |
| Indexer weights | fp16/bf16 | **`[corrected]`** Not INT8. |
| Indexer K cache | fp8 (UE8M0) | **`[corrected]`** The indexer K cache is *always* quantized in the canonical graph — fp8 by default, mxfp4 Blackwell-only `[V vllm indexer.py:49-65]`. Not INT8. |
| Norms, router gates, embeddings, `lm_head` | fp16/bf16 | `[V contract]` Not quantized. |

**The dequantization rule:** INT4 weights are dequantized to fp16/bf16 **in registers**, immediately before the matmul. This is not a performance penalty because the workload is bandwidth-bound, not compute-bound. The cost of the dequantization arithmetic is hidden behind the memory latency of streaming the next expert.

**Storage format ≠ hardware instruction.** gfx1100 has no native FP8/FP4 *instructions* `[V]`. The canonical graph's fp8 steps are **software-emulated block round-trips** (amax → power-of-two scale → clamp → round-to-nearest, after an explicit bf16 cast), pure ALU. The canonical path applies them at three points: indexer Q, and both KV store paths `[V fused_indexer_q.py, fused_compress_quant_cache.py]`. Reproducing them is cheap and cheap-to-verify; whether our engine *must* is a fidelity decision whose delta must be **measured at a gate**, not assumed.

**Design consequence:** every matmul is fp16/bf16 (or an INT4-dequantizing equivalent). fp8 appears only as **software round-trips** at the store boundaries above.

### 2. Hardware Leverage

RDNA 3 (gfx1100) provides:
- `v_wmma_f32_16x16x16_f16` — fp16 matrix multiply with fp32 accumulate.
- `v_wmma_f32_16x16x16_bf16` — bf16 equivalent.
- `v_wmma_i32_16x16x16_iu8` — INT8 matrix multiply with int32 accumulate. Available, but **`[corrected]` is not used for the indexer** — the canonical indexer path is fp8, not INT8.
- No native FP8 or FP4 conversion instructions. The emulation above is plain ALU.

### 3. Memory Tiering Policy

- **Hot (VRAM):** dense backbone, KV cache, indexer weights and cache, currently-active experts, router gates, norms, embeddings, `lm_head`.
- **Warm (host RAM):** the full expert set, or the most probable subset based on activation profiling.
- **Cold (SSD):** the full expert set, memory-mapped or read on demand.

**Streaming rule:** experts move cold → warm → hot. The dense backbone and KV cache never leave VRAM. The expert index file maps every expert to its exact byte position, enabling direct seeks without reading the whole shard.

**Future optimization:** activation probability profiling. Track which experts fire for which token distributions, and bias allocation so that high-probability experts live in faster tiers. This is a scheduling optimization, not a correctness requirement — implement it only after the pipeline is numerically correct.

### 4. Scope

**In scope (base decoder):** embedding → 43 layers → HC head → final norm → `lm_head` → sampling, for prefill and decode.

**Explicitly out of scope for this revision** (present in config, not required for base-decoder correctness) `[V config.json]`:
- MTP / DSpark next-token head (`num_nextn_predict_layers=1`, `dspark_target_layer_ids=[40,41,42]`).
- Multi-GPU pipeline parallelism.

Declaring these out of scope is deliberate: an implicit "we'll handle it later" is how scope silently grows. They may be revisited once the base decoder is green.

**In scope and mandatory:** Hyper-Connections. It was missing from the first revision; it applies to **every** layer, not a subset.

### 5. Earning Order (binding — this is what failed last time)

The first implementation was prototyped by iteratively discovering components and validating them in the wrong succession, so errors accumulated faster than they could be isolated. This section is as important as Part II.

**Order A — dataflow** (Part II) answers *what happens*.
**Order B — earning** answers *in what succession we are allowed to believe it*.

Invariants. Each is a hard gate; do not advance past a red one:

1. **All-in-VRAM before streaming.** No tiering until numerics are correct.
2. **Reference-derived oracles, never our own kernels.** An oracle copied from our implementation proves only self-consistency. This circularity is why the current tests pass on wrong logic.
3. **One op before a layer; one layer class before another.** Certify in dependency order.
4. **Single token before sequences; serial before batched.**
5. **Sliding before CSA before HCA.** Simplest class first.
6. **No new capability while an earlier gate is red.** Do not "fix forward".
7. **Every step has an explicit gate** that produces an artifact which either matches the reference or does not. "It looks plausible" is not a gate.

### 6. State Contract & Prefix Reuse (design constraint — decide now, implement in Tier 4)

For an agent-serving engine on tiered hardware, recomputing the full context every turn is not viable: the dominant cost is re-streaming expert weights off disk, not attention. Prefix reuse is therefore a **foundational design constraint**, not a frontend feature. It is decided now and implemented in Tier 4.

#### 6.1 What "the KV cache" actually is here

It is not one buffer. It is a **composition of independent, position-addressed pieces**, per layer:

| Piece | Size bound | Present on |
|---|---|---|
| Local KV ring (single shared head, 512-wide) | `sliding_window = 128` `[V config]` | all layers |
| Compressed entries | `ceil(len / ratio)`, ratio ∈ {4,128} `[V config]` | ratio ≠ 0 layers |
| Compressor partial state (kv, score) | one in-progress ratio window | ratio ≠ 0 layers |
| Indexer K cache + indexer compressor partial state | `ceil(len/4)` / one window | CSA (ratio 4) layers |

Hyper-Connections and the residual stream are **per-token and recomputed** for new tokens, so they are not cached. Cached state is exactly the four pieces above.

#### 6.2 How the references solve it (they differ, and the differences are instructive)

| Engine | Granularity | Cache key includes | Tiering | DSV4-specific handling |
|---|---|---|---|---|
| vLLM `[V cache_utils.py, compressor.py, common/ops/save_partial_states.py]` | **Paged**, `slot_mapping`, `cache_block_size=64`; compressor owns a separate `CompressorStateCache(state_dim = 2·coff·head_dim)` | block hashes | GPU/CPU/NVMe offload | `save_partial_states` persists the compressor's partial **kv+score** per token into the paged state cache |
| sglang `[V mem_cache/radix_cache.py, test_dsv4_swa_radix_retract.py]` | **Radix tree**, prefix length rounded down to `page_size`; also holds `extra_key` and `cache_salt` | token ids + `extra_key` + `cache_salt` | HiCache multi-tier | **SWA tombstones**: a cached leaf older than the window cannot serve SWA attention; early leaf release + retract interact badly |
| llama.cpp `[V src/llama-kv-cache-dsv4.h/.cpp]` | **Per-sequence, per-class**: separate `iswa` / `csa` / `hca` / `lid` caches and separate comp states; `state_write`/`state_read`/`seq_cp` per `seq_id` | sequence id + flags | backend buffers | `PARTIAL` state mode (`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`); `comp_size = ceil(kv/ratio)` |
| ds4 `[V ds4_kvstore.h]` | **Session/prompt-level file store**, byte-prefix match | SHA of tokens **+ `ext_flags`** (tool map, responses/thinking visible, session title) | disk, with eviction scoring | stores the exact tokens *and* the graph state together |
| colibri `[V c/kv_prefix.h]` | **Token-id ledger** (`fed`, `len`, `cap`) | token ids, plus a **`tainted`** flag for "state consumed something token ids cannot describe" | VRAM/RAM/NVMe | reuse requires a true prefix match and ≥1 new token |

**The convergences — these are the actual requirements:**

1. State must be **independent, per-class, and addressable at a boundary** — never a monolithic per-request ring accumulated from position 0. vLLM pages it, sglang pages it, llama.cpp splits it per class and per sequence.
2. **The reuse boundary must be allowed to fall mid-ratio-window.** vLLM persists the compressor partial state per token; llama.cpp has a `PARTIAL` mode. Without this, a prefix whose length is not a multiple of `ratio` cannot be reused.
3. **The cache key must include every non-token input that changes the graph.** ds4's `ext_flags`, sglang's `extra_key`/`cache_salt`, colibri's `tainted`. In one reference these flags are literally *tool map* and *thinking visibility* — this is why tool use and prefix caching are coupled for us (see 6.4).
4. **Local SWA state is window-bounded and not reusable beyond the window** — the long-range memory lives in the compressed entries. sglang handles this with tombstones; we must at minimum detect and decline the reuse rather than serve a stale local ring. This is the DSV4-specific hazard that makes a naive radix cache incorrect.

#### 6.3 Requirements on our state layout (binding)

- **R1.** Local KV, compressed entries, compressor partial, and indexer state are **separately addressable** (block/slot indexed), not one contiguous per-request allocation.
- **R2.** Each piece is addressed by **absolute position** and is restorable at any token boundary, including mid-ratio-window (requires persisting partial state).
- **R3.** A **restore must be byte-exact** with respect to never having evicted the state. This is a gate, not an aspiration.
- **R4.** The engine must **detect** when a candidate reuse boundary is outside the local window and handle it explicitly (rebuild the local ring; never serve a stale one).
- **R5.** All pieces must be **movable across VRAM / host / NVMe**. Long-lived prefix state belongs in warm/cold tier; the existing DMA + `io_uring` staging path should be reused rather than adding a second one. `[I]` **Our own constraint makes this concrete:** cold-tier reads are `O_DIRECT` at 4096-byte sector granularity, so the state block size should be chosen to be sector-aligned (or an exact multiple/divisor of a sector), otherwise state eviction to NVMe cannot use the existing direct-I/O path and will need an extra bounce buffer. This is a synthesis of the reference designs with our tiering rules, not something taken directly from a reference.
- **R6.** The state manager is **deferred to Tier 4**; only the *layout contract* above is required now.

#### 6.4 Non-deferrable consequences for tool use

Because the cache key must include non-token graph inputs, and because agent workloads reuse long system prompts and tool schemas, two seams must exist from the start even though tool support itself is deferred:

- **Exact chat/template tokenization**, including the tool-definition sections; and a way to record the **non-token inputs that affect the graph** (e.g. which tool set is active, thinking visibility) so they can participate in the cache key. `[V ds4 ext_flags]`
- **A logit-processor seam at sampling** — constrained/structured output (tool-call JSON) is a mask applied to the logits before sampling. If sampling is a closed argmax with no hook, adding constraints later means touching the pipeline.

Everything else about tool use (schema formatting, parser, turn orchestration) is genuinely deferrable.

---

## Part II — The Forward Pass, Step by Step

### Step 0 — Tokenization

- Input text → token IDs via the model's tokenizer. Host-side.
- **Not deferrable:** the exact chat template, including tool-definition sections. Tool schemas are part of the prompt token stream, so a template that is *almost* right silently changes every prefix and defeats prefix caching.
- **Not deferrable:** produce, alongside the tokens, a record of the **non-token inputs that affect the graph** (active tool set, thinking visibility, etc.) so they can enter the prefix cache key `[V ds4_kvstore.h ext_flags]`. See Part I §6.4.
- **Gate:** known prompt → token ids byte-identical to the reference tokenizer; template round-trips.

### Step 1 — Embedding Lookup

- Token IDs → embedding vectors. Table is unquantized fp16 `[V contract: embed.weight F16 [129280, 4096]]`.
- **`[corrected]` Output is *not* a plain `[batch, seq, hidden]` tensor.** The state entering the layer stack is `4 × 4096` per token: the embedding row is **replicated across the 4 Hyper-Connection streams** `[V config hc_mult=4]`.
- **Gate:** all 4 streams byte-identical to the checkpoint row for a known token id.
- **Kernel:** gather (sharded table), then replicate.

### Step 2 — Per-Layer Loop (repeated 43 times)

Steps 2.0 through 2.11 execute once per layer. **Hyper-Connections run on every layer.** The only structural branch is the attention class (Sliding / CSA / HCA), which is driven by `compress_ratios` `[V config]`.

#### 2.0 — Hyper-Connections pre-mix + Sinkhorn `[corrected — was missing entirely]`

> **Resolved (Tier 0.1).** HCG semantics below were re-read from the authoritative
> `vllm/model_executor/layers/mhc.py` (`MHCPreOp`/`MHCPostOp`/`HCHeadOp`) and
> `vllm/model_executor/kernels/mhc/torch.py` (`mhc_pre_torch`, `mhc_post_torch`). They agree
> exactly with the `ds4` cross-check. `[V]` tags in 2.0 / 2.6 / 2.7 / Step 3 are now
> confirmed. **One error was found and fixed: `hc_sinkhorn_eps` is added to *every*
> denominator, not row denominators only.**

The residual state is 4 streams. Before each sublayer, a control vector is computed and split into three sets of mixing weights.

- **Project:** `mixes = (x_flat @ fnᵀ) · rsqrt(sumsq(x_flat)/(hc_mult·hidden) + rms_eps)`, where `x_flat` is the flattened `4 × 4096 = 16384` residual and `fn` is `hc_attn_fn [24, 16384]`. RMS is over the flattened dim, not per stream `[V mhc_pre_torch; V ds4 hc_split_sinkhorn_one]`. (Scaling-then-project and project-then-scale are algebraically identical; the reference scales the projection output.)
- **Pre-mix:** `pre[j] = sigmoid(mixes[j]·scale[0] + base[j]) + hc_pre_eps`, `j∈[0,4)` `[V]`.
- **Post-mix:** `post[j] = sigmoid(mixes[j+4]·scale[1] + base[j+4]) · hc_post_mult_value`, `j∈[0,4)` `[V]`. **`hc_post_mult_value = 2.0`** — a hardcoded constant in the model, not a config field `[V vllm amd/model.py:709 hc_post_alpha = 2.0; V ds4 hc_post_alpha=2.0f]`.
- **Comb logits:** stored as `mixes[2·hc + 4·dst + src]`, i.e. index = `8 + 4·output + contraction` `[V mhc_pre_torch mixes[:,2*hc:].view(T,hc,hc); V ds4 c[src + dst*n_hc]]`.
- **Sinkhorn:** softmax over the **source (contraction)** axis, add `hc_sinkhorn_eps`; normalize over the **output** axis with `hc_sinkhorn_eps`; then `(sinkhorn_repeat − 1)` iterations of (normalize over source, normalize over output), **each denominator adding `hc_sinkhorn_eps`** `[V mhc_pre_torch; V ds4]`. `sinkhorn_repeat = hc_sinkhorn_iters = 20` `[V config]`.
- **Pre-combine:** `layer_input = Σ_j pre[j] · residual[j, :]` `[V]`.

> `[corrected]` **Sinkhorn-Knopp belongs here, on the 4×4 comb matrix — not on the attention/compressor output.** The first revision placed it in the attention block; that was wrong.

- **Ensure:** `hc_pre_eps = hc_sinkhorn_eps = hc_eps = 1e-6` for our model `[V config hc_eps; V vllm passes hc_eps for both]`; **`eps` in every denominator** (initial softmax add, initial normalize, and both denominators of every iteration — *not* row-only); comb stored as `8 + 4·output + contraction`; `20` iterations, not fewer.
- **Gate:** `mixes(24)`, `pre(4)`, `post(4)`, `comb(16)` match reference; comb is doubly stochastic within 1e-4.
- **Kernel:** projection (reduction + dot), then iterative normalization (fp32).
- **Note (optimization, not semantics):** the reference *may* fuse an RMSNorm onto `layer_input` `[V vllm _apply_mhc_norm / norm_weight]`. That is equivalent to the separate step 2.1 and is an optimization only — do **not** treat it as license to drop 2.1.

#### 2.1 — Attention RMSNorm

- `x_norm = x / sqrt(mean(x²) + eps) · weight` over 4096; fp32 accumulate `[V]`.
- **Gate:** matches fp32 reference within ~1e-3 relative.

#### 2.2 — MLA Q/KV Projections `[corrected — was described as vanilla QKV]`

> **Re-cited (Tier 0.2b).** Confirmed against `sglang/.../models/deepseek_v4.py:719-743,1030-1044`
> and `vllm/.../kernels/mhc`-adjacent ops. Order, norm widths, and the shared-KV head all check out.

This is **Multi-head Latent Attention with a low-rank Q path and a single shared KV head**, not `W_q/W_k/W_v`.

- **Q path:** `q_lora = wq_a(x)` (4096→1024) → **`q_norm` RMSNorm over `q_lora_rank` (1024)** → `q = wq_b(q_lora)` (1024→64×512) → **per-head RMSNorm over `head_dim` (512)** `[V sglang deepseek_v4.py:719-734 `wq_a`/`q_norm = RMSNorm(q_lora_rank)`/`wq_b`; V :1030-1044 `q=wq_a(x); q_norm(q); q=wq_b(q); fused_q_norm_rope(...)`]`. The per-head norm operates over the **last dim = head_dim**, confirmed by `head_dim = q_input.shape[-1]` in the fused kernel `[V dsv4/elementwise.py:152]`.
- **KV path:** `kv = wkv(x)` (4096→512) → **`kv_norm` RMSNorm over `head_dim` (512)** `[V sglang deepseek_v4.py:726-743, 1072-1080]`.
- **`[corrected]` There is no separate V.** A single 512-wide KV tensor per position is used as **both key and value**. Confirmed structurally: the attention kernel takes one `unified_kv [total_pages, D]` with `D = head_dim` and produces `out [N, H, D]` with no separate value pointer `[V sglang dsv4/unified_kv_kernels/paged_prefill.py:25-41,66-71]`.
- **Ensure:** q-norm sits **between** `wq_a` and `wq_b`; the per-head norm is **after** `wq_b` and over 512 (not 1024); two distinct Q-path norms plus one KV norm.
- **Gate:** `q_lora`, `q_lora_norm`, `q`, `kv` each match reference.

#### 2.3 — RoPE `[corrected — one base, now two]` `[Tier 0.2c: re-cited]`

- **Partial, on the tail.** Only the **last 64** of each 512-wide head rotates; layout is `[nope (448) | rope (64)]`. The DSV4 module is the *subclass* `DeepseekV4ScalingRotaryEmbedding`, which overrides the generic parent to use `query_rot = query[..., -rotary_dim:]` and `query_pass = query[..., :-rotary_dim]` — documented as *"Applies RoPE to the last rotary_dim"* `[V deepseek_scaling_rope.py:230-238, 262-263, 289-291]`. `ds4` agrees (`n_nope = head_dim - n_rot; tail = x + h*head_dim + n_nope`) `[V ds4:11073-11074, 11082]`. **Trap:** the *generic* parent `DeepseekScalingRotaryEmbedding.forward_static` slices `query[..., :rotary_dim]` (the **first** 64) — using it by mistake rotates the wrong dims. Confirm against the subclass.
- **Interleaved (GPT-J), not NeoX.** `is_neox_style=False`, `cos.repeat_interleave(2)`, `rotate_gptj` `[V rope.py:50; V deepseek_scaling_rope.py:272-279]`; `ds4` rotates adjacent pairs `tail[i], tail[i+1]` in steps of 2 `[V ds4:11088-11094]`.
- **Two bases by layer class.** `rope_theta = compress_rope_theta` when `compress_ratio > 1`, else `rope_theta` `[V rope.py:28-30]`; `ds4` keys on `compress_ratio != 0` `[V ds4:11100-11103]`. Identical for our `{0, 4, 128}` (only 0 is ≤ 1). Here: **10000** for ratio 0, **160000** for 4/128.
- **YaRN only on compressed layers.** ratio > 1 → `deepseek_yarn` with `factor=16, beta_fast=32, beta_slow=1`; sliding/ratio-0 layers → `factor=1.0` (plain RoPE) `[V rope.py:31-44]`; `ds4` sets `ext_factor=1` only when compressed `[V ds4:11106-11112]`.
- **`[corrected]` No amplitude scaling — now triple-cited.** vLLM sets `mscale = 0` and `mscale_all_dim = 0` with the comment *"Disable mscale"*, so `yarn_get_mscale(·,0) = 1` and the effective `mscale = 1.0` `[V rope.py:38-39; V deepseek_scaling_rope.py:14-18, 48-52]`. `ds4` says it in words — *"DeepSeek V4 reference RoPE uses interpolation without that magnitude change, so pass the inverse factor here and let the helper cancel itself out"* — and divides out `1 + 0.1·log(1/freq_scale)` `[V ds4:11133-11144]`.
- **Compressed entries rotate at the window start: `comp_pos = pos + 1 − ratio`**, not at `pos` `[V ds4:13410, 13498]`; vLLM states the same as `(positions // compress_ratio) * compress_ratio` `[V compressor.py "position used" comment; V fused_compress_quant_cache.py:206]`. The two agree because the entry only exists when `(pos+1) % ratio == 0`, where the floor equals `pos+1−ratio`.
- **Inverse RoPE** on the attention-output tail **before** the grouped projection. The DSV4 rope `forward` takes an explicit `inverse` flag that negates `sin` `[V deepseek_scaling_rope.py:249-252, 281-284]`; `ds4` passes `inverse=true` for the output `[V ds4:13993, 14490]`.
- **Gate:** forward∘inverse = identity; tables match reference at two positions, one of them past `original_max_position_embeddings = 65536`; **the rotated slice is the last 64 dims, not the first.**

#### 2.4 — Attention

**2.4.1 — Score, sink, and softmax (all layers)**
- Scores `S = q·k / sqrt(512)`, fp32 accumulate `[V]`.
- **Attention sink** `[V Tier 0.2b]`: a per-head fp32 logit `attn_sink[H]`, described by the reference as *"a virtual extra K with V=0"*. Concretely `m_final = max(m_i, sink)`, `l_final = l_i·alpha + exp(sink − m_final)`, and *"the sink itself contributes 0 to acc since V_sink = 0"* `[V sglang dsv4/unified_kv_kernels/paged_prefill.py:194-203]`. So it enters the **max and the denominator only**, and contributes **no value** — exactly as this plan stated. The alternative implementation (a zero *value* row with the sink logit) is numerically identical.
- Sink is padded to `padded_heads` when the head count is padded (vLLM fills padding with `-inf`, sglang with `0`) — a host-side padding detail only `[V vllm attention.py:235-238; V sglang deepseek_v4.py:824-834]`.
- Mask disallowed keys with `-INF` (not zero) so they are excluded from both max and denominator.
- **Gate:** matches reference at pos 0, pos < window, pos > window.

**2.4.2 — Compressor (layers with ratio ≠ 0: 2..42)**

> **Re-cited (Tier 0.2 + 0.2c).** Confirmed against the readable kernels
> `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` (`compress_norm_rope_store_triton`)
> and `vllm/models/deepseek_v4/common/ops/save_partial_states.py`, plus `vllm/.../compressor.py`
> and the `ds4` cross-check. Tier 0.2c found **one whole op the plan was missing: APE** (below).

- `fused_wkv_wgate` produces `kv` and `score`, each width `coff·head_dim` where `coff = 1 + overlap` and `overlap = (ratio == 4)` — so `coff = 2` for ratio 4, else `1` `[V compressor.py:153-154, `fused_wkv_wgate`; V ds4:13382-13384]`.
- **`[new — Tier 0.2c]` APE — a learned absolute-position embedding added to the *score* branch.** `ape` is a real trained parameter of shape `[ratio, coff·head_dim]` `[V compressor.py:190-197]`. Each token's score row gets `score += ape[pos % ratio]`, applied **after** the `wkv_wgate` projection and **before** the window softmax `[V save_partial_states.py kernel: `ape_row = position % COMPRESS_RATIO; store(score + ape)`; V ds4:13386-13388 `sc_cur[j] += tensor_2d_value(model, ape, j, pos_mod)`]`. It is added to **score only — never to `kv`**. Omitting it shifts every compressed row and is silent.
  - **Present in our checkpoint, therefore required:** `layers.N.attn.compressor.ape` (fp32) and `layers.N.attn.indexer.compressor.ape` (fp32), 62 tensors total; measured shapes e.g. `[4,1024]` and `[4,256]` (ratio 4, `coff=2`) and `[128,512]` (ratio 128, `coff=1`) `[V safetensors header]`.
  - **Layout `[ratio, coff·head_dim]`, indexed by `pos % ratio`.** `ds4` stores/consumes it *transposed* as `[coff·head_dim, ratio]` (`tensor_expect_layout(..., comp_width, ratio, ...)`, indexed `(dim, pos_mod)`) `[V ds4:5320, 13387]` because the GGUF path transposes. The checkpoint and vLLM are row-major `[ratio, width]`. **Ingest the checkpoint order; do not copy ds4's transpose.** Keep fp32 (checkpoint dtype) when adding.
- **State layout.** ratio 128: `ratio` rows of width `head_dim`, current token at row `pos % ratio`. ratio 4 (overlap): width `2·head_dim`, `2·ratio` rows, current token at row `ratio + pos % ratio`; the pool reads the first half (offset `0`) for the `ratio` **older** tokens and the second half (offset `head_dim`) for the `ratio` **newer** ones `[V fused_compress_quant_cache.py:216; V ds4:13289-13338 compressor_pool_decode_state]`. After emitting, the halves are rotated `[V ds4:13422-13433]` — part of the R2 state contract.
- The pool window is exactly `(1 + overlap)·ratio` positions ending at the current position: `start = pos − (1+overlap)·ratio + 1` `[V fused_compress_quant_cache.py:206-207; V ds4:13298-13299]`. For ratio 4 that is 8 positions; for ratio 128, 128.
- The **overlap** is realized by `head_offset = (t >= ratio) · head_dim` — the second segment lives at `[head_dim, 2·head_dim)` of the same row `[V fused_compress_quant_cache.py:216]`. This confirms the two-segment layout the plan described.
- `score = softmax(score, dim=window)` then `compressed = Σ_window kv · score` — **per-dimension softmax over the window**, matching ds4 `[V fused_compress_quant_cache.py:241-255]`.
- `normed = compressed · rsqrt(Σcompressed²/head_dim + eps) · norm_weight` `[V:257-262]`.
- Boundary: the entry is produced only when `(pos + 1) % ratio == 0` `[V:201-202]`. Position for RoPE is the compressed position, not `pos` (see 2.3).
- **`[new]` Store-time quantization.** The canonical path then does `normed → bf16 → fp32` (an explicit round-trip "to match reference"), UE8M0-quantizes to E4M3, and stores **only the non-RoPE 448 dims as fp8**, with the RoPE part handled separately `[V fused_compress_quant_cache.py:288-345]`. This is a second place where an fp8 round-trip is part of the graph (see Step 7).
- **Gate:** for a scripted sequence, compressed rows/positions match reference; the boundary fires exactly when `(pos+1) % ratio == 0`; the two-segment overlap is populated from the correct offsets; **the `ape[pos % ratio]` add is applied to `score` and only `score`**, with the checkpoint's `[ratio, width]` order.

**2.4.3 — Lightning Indexer (ratio == 4 layers only: even layers 2..42)** `[corrected]`

> **Re-cited (Tier 0.2b).** ReLU is **required**; indexer Q and K are **fp8/UE8M0**.
>
> **Correction of a correction — the Hadamard.** In Tier 0.2 this plan stated "Hadamard does
> not appear anywhere; that claim was wrong for this model". **That was itself wrong.**
> `hadamard_transform` *is* present in DeepSeek-V4 code: `rotate_activation` in
> `sglang/.../qsa/dsa_indexer.py:192-204`, `fused_q_indexer_rope_hadamard_quant`
> (registered for **CUDA and ROCm**, not just NPU), and `hip_compress_fused_norm_rope_hadamard_inplace`
> with `hadamard_scale = head_dim**-0.5` — matching ds4's `0.08838834764831845f` exactly.
> **The accurate statement** is neither of the two earlier ones: the Hadamard is a
> **logit-preserving orthonormal rotation applied symmetrically to indexer Q and K before
> quantization**. sglang says so explicitly — *"Fusion drops the (logit-preserving) Hadamard
> rotation"* `[V dsa_indexer.py:396]` — and ships both a fused (no-Hadamard) and a legacy
> (Hadamard) path that it treats as equivalent. Therefore it is **not graph semantics**; it
> exists to condition the dynamic range before fp8 rounding. If we apply it, we must apply it
> to **both** Q and K or the scores change. Decision deferred to the Gate 11 measurement;
> record whichever we choose.

- **Query:** from `qr_norm` via `indexer.wq_b` (1024→64×128), GPT-J RoPE on the **last 64** of each 128-wide head `[V fused_indexer_q.py:126-148; V sglang]`.
- **Query is FP8-quantized:** after RoPE, a **bf16 round-trip** then UE8M0 block-scaled E4M3 (`amax → 2^ceil(log2(amax/448)) → clamp ±448`) `[V fused_indexer_q.py:150-178]`. The bf16 cast is explicitly "to match reference numerics".
- **Optional pre-quantization rotation:** Hadamard-128, scale `1/sqrt(128)`, applied to **both** Q and K. Logit-preserving; **must be symmetric** `[V dsa_indexer.py:192-204, :396]`.
- **Weights:** `weights_proj @ x` (4096→64), then scaled by `softmax_scale · n_heads^-0.5 · q_scale` where `softmax_scale = 128^-0.5` — i.e. `1/sqrt(128·64)`, applied inside the head-gate scale `[V dsa_indexer.py:387-389 `weights = weights_raw * n_heads**-0.5; return weights.unsqueeze(-1) * q_scale * softmax_scale`; V fused_indexer_q.py:190-204]`.
- **Score (order matters):**
  ```
  dot[c,h]   = q_h · k_c,h
  score[c]   = kv_scale[c] · Σ_h weight[h] · relu(dot[c,h])
  ```
  **ReLU is applied to the per-head dot *before* multiplying by the head weight, then summed** — *not* `relu(Σ weight·dot)`. The whole sum is then multiplied by the per-candidate KV dequantization scale `[V sglang dsv4/indexer.py:118-124; V qsa/dsa_indexer.py:43; V dsa/tilelang_kernel.py:245]`.
- **Indexer K cache is FP8/UE8M0** — `head_dim` bytes = 128 fp8 + 4 fp32 scales per head; mxfp4 is Blackwell-only `[V vllm attention.py:964-975]`.
- `top_k = 512` `[V config index_topk=512]`. Invalid/padded candidates take score `0` (not `-INF`) `[V sglang dsv4/indexer.py:133]`.
- **Gate:** selected index set matches reference **exactly** for scripted inputs; the Hadamard choice is recorded as an explicit, measured decision.

**2.4.4 — Paged / chunked attention over local + compressed rows**
- Attend over the last `min(pos+1, 128)` local rows plus the selected compressed rows `[V config sliding_window=128]`.
- **Gate:** state and output match the serial reference across a ratio boundary.

#### 2.5 — Output Projection `[corrected — grouped, not a single matmul]`

> **Re-cited (Tier 0.2b).** Confirmed against `sglang/.../models/deepseek_v4.py:405-440`.

- **Inverse RoPE** first (2.3).
- `wo_a` is **grouped**: shaped `[G, R, D]` = `[o_groups, o_lora_rank, group_dim]`, applied as `einsum("tgd,grd->tgr", o, wo_a)` where `o` is `[T, G, D]`. Result `[T, G, R]` = flattened `[T, G·R]` `[V sglang deepseek_v4.py:405-440 docstring + einsum]`. With our dims: `G=8`, `R=1024`, `group_dim = 8 heads × 512 = 4096` — matching the `wo_a [8192, 4096]` contract.
- `wo_b` maps `G·R = 8192` → 4096 `[V contract]`.
- **Ensure:** group assignment is **8 contiguous heads** per group; the stored `wo_a` row-major layout must match the `[G, R, D]` interpretation (group-major), i.e. `wo_a[g]` is a `[1024, 4096]` block, not an interleaved slice.
- **Gate:** low-rank tensor (8192) and final 4096 match reference.

#### 2.6 — Hyper-Connections attention post-mix `[corrected — was a plain residual add]`

- The residual is **not** a single-stream add. `res_mid[dst,h] = post_a[dst]·attn_proj[h] + Σ_src comb_a[src,dst]·res_in[src,h]` — the contraction runs over the **source (residual) index**, and the output is indexed by `dst` `[V mhc_post_torch einsum "...ij,...ih->...jh"; V ds4]`.
- **Ensure:** the contraction index is the residual stream, not the output stream. Transposing it is a silent error.
- **Gate:** `res_mid` matches reference; state remains 4 × 4096.

#### 2.7 — HC FFN pre-mix + Sinkhorn

- Same structure as 2.0, using `hc_ffn_fn/base/scale`, producing `pre_f`, `post_f`, `comb_f` and the FFN input `ffn_pre` `[V vllm MHCPreOp reused for the ffn sublayer; V ds4 hc_pre_from_state_one_scratch]`.
- **Gate:** `post_f`, `comb_f`, `ffn_pre` match reference.

#### 2.8 — FFN RMSNorm

- RMSNorm over 4096, fp32 accumulate.
- **Gate:** matches fp32 reference.

#### 2.9 — MoE Router `[corrected — scoring and layer branching were wrong]`

> **Re-cited (Tier 0.2).** Confirmed against
> `vllm/model_executor/layers/fused_moe/router/dsv4_topk.py` and
> `vllm/models/deepseek_v4/nvidia/model.py`.

- **Logits:** `logits = gate_weight @ x` → 256 `[V contract ffn.gate.weight F16 [256,4096]]`.
- **Score:** `weights[e] = sqrt(softplus(logits[e]))`, computed stably as
  `sqrt(x > 20 ? x : log(1+exp(x)))` `[V dsv4_topk.py:82; V config scoring_func=sqrtsoftplus]`.
- **Selection (either branch):**
  - **`[corrected]` Layers 0–2 use hash routing, not top-k.** `is_hash_moe = layer_index < num_hash_layers`; expert ids come from `ffn.gate.tid2eid[token_id]`, and those layers have **no** bias tensor `[V nvidia/model.py:810-835 comment "hash MoE doesn't use e_score_correction_bias"; V config num_hash_layers=3; V contract]`. The table is shaped `(vocab_size, num_experts_per_tok)` — **`[vocab, 6]`** `[V nvidia/model.py:820]`.
  - Layers ≥3: `selection[e] = weights[e] + e_score_correction_bias[e]`, take top-6 by `selection` `[V dsv4_topk.py:83; V config topk_method=noaux_tc]`.
  - **Tie-break: lowest expert index wins** — `expert_id = min(where(current == max, offsets, NUM_EXPERTS))` `[V dsv4_topk.py:91-92]`.
  - NaN selection scores are replaced by `-1e30` `[V dsv4_topk.py:86]`.
- **Weighting:** the weight stored for each selected expert is the **unbiased** `weights[id]`, *not* the biased selection score `[V dsv4_topk.py:93-97 "selected_weight = weights[...]"` separate from `current = weights + bias`]`.
- **Normalize then scale:** `selected_weights *= routed_scaling_factor / (Σ selected_weights)` `[V dsv4_topk.py:102-105]`. Note the reference guards with `Σ>0 ? Σ : 1` whereas ds4 used a `6.1e-5` floor — equivalent in practice because `sqrt(softplus) ≥ 0`, but record which we implement.
- **Ensure:** `num_experts == 256`, `top_k == 6` `[V dsv4_topk.py can_use_dsv4_topk]`.
- **Gate:** selected ids **and** weights match reference exactly, for one hash layer and one top-k layer. `tid2eid` orientation is verified against our artifact (`[vocab,6]` expected).

#### 2.10 — Expert Computation

**2.10.1 — Expert Fetch**
- Consult the index file for the expert's byte position; promote cold → warm → hot as needed.
- **Kernel:** none. Memory management; must be async and overlapped with compute.

**2.10.2 — Dequantization**
- INT4 → fp16/bf16, group size 32; `w = q · scale` with signed 4-bit `q` (bias −8).
- **Critical:** nibble order and scale layout must match the packing format. Verified in Stage D of the verification plan.

**2.10.3 — Expert Matmul `[corrected — SwiGLU is clamped]`**
- Per expert: `gate = w1 @ x`, `up = w3 @ x`, then **`hidden = silu(clamp(gate, max=limit)) · clamp(up, min=−limit, max=+limit)`** with `limit = 10` `[V vllm activation.py:241-242 SiluAndMulWithClamp; V config swiglu_limit=10]`.
- **`[V]` Re-cited and confirmed exactly:** the clamp is **asymmetric** — the gate is clamped **only above**, the up branch is clamped **both sides**. Getting this wrong shifts activations rather than breaking them loudly.
- Then `down = w2 @ hidden` `[V]`.
- Accumulate in fp32.

**2.10.4 — Shared Expert**
- Always fires; unquantized fp16 `[V contract ffn.shared_experts.* F16]`. Same clamped-SwiGLU as 2.10.3.

**2.10.5 — Combine**
- `shared + Σ_k weight_k · down_k`. **`[?]` Confirm the accumulation order matches the reference** — a different order changes fp rounding and can differ from the reference beyond tolerance.

#### 2.11 — Hyper-Connections FFN post-mix

- `res_out[j,h] = post_f[j]·moe_out[h] + Σ_i comb_f[i,j]·res_mid[i,h]`; this becomes the next layer's `res_in` `[V]`.
- **Gate:** `res_out` matches reference; state still 4 × 4096.

#### Note on Layers 0 and 1 `[corrected]`

Layers 0 and 1 have **no compressor and no indexer** (`compress_ratios[0]=compress_ratios[1]=0`) `[V config; V artifact]`.

`[corrected]` They **do** run Hyper-Connections, including the HC Sinkhorn. The first revision said "no Sinkhorn step" on layers 0–1 — that was a consequence of placing Sinkhorn in the attention block; with Sinkhorn restored to HC (2.0), it applies to all 43 layers.

**Gate:** assert the branch explicitly — a Sliding layer must never read compressor/indexer tensors.

### Step 3 — HC Head Reduction `[corrected — was missing]`

- The final state is 4 streams. Reduce to a single 4096 vector:
  `mixes = hc_head_fn @ rmsnorm_without_weight(flatten(x), rms_eps)`, then `pre[j] = sigmoid(mixes[j]·hc_head_scale + hc_head_base[j]) + hc_eps`, then `out[h] = Σ_j pre[j]·x[j,h]` `[V vllm kernels/mhc/triton.py hc_head_reduce_triton_kernel; V contract hc_head_fn F32 [4,16384], hc_head_base F32 [4], hc_head_scale F32 [1]]`.
- **Ensure:** the head RMSNorm has **no learned weight** (unlike 2.1); `hc_head_scale` is a scalar `[1]` broadcasting over the 4 gates; `hc_eps` is added after the sigmoid.
- **Gate:** matches reference; output is 4096.

### Step 4 — Final RMSNorm + LM Head

- Final RMSNorm over 4096 `[V contract norm.weight]`.
- `logits = lm_head @ hidden` → 129280 `[V contract head.weight F16 [129280,4096]]`; fp32 accumulate.
- **Gate:** logits match reference; top-1 token matches.

### Step 5 — Sampling

- Apply temperature, top-k, top-p as configured.
- **Not deferrable:** expose a **logit-processor seam** — a hook that may mask or bias the logits *before* sampling. Tool-call and structured-output grammar constraints are implemented as logit masks. A hardcoded argmax with no hook forces a pipeline change later.
- Softmax in fp32.
- Sample or take argmax.
- **Kernel:** reduction plus sampling. Host-side is acceptable for a first implementation.

### Step 6 — Detokenization

- Token ID → text.
- Host-side.

### Step 7 — KV / compressed state update `[corrected]`

- **`[corrected]` There is one shared KV head, and key and value are the same tensor.** A single 512-wide row is stored per position, not a K and a V `[V config num_key_value_heads=1; V ds4]`.
- The row is stored **after** RoPE for the local ring; compressed entries are produced every `ratio` tokens (2.4.2) and rotate at the compressed position.
- **Local ring:** `sliding_window = 128` `[V config]`. **Compressed ring:** capacity `ceil(ctx / ratio)` `[V our layout]`.
- **`[corrected]` Layout must obey the state contract in Part I §6**, not the current per-request contiguous ring. Concretely: the local KV, compressed entries, compressor partial state (kv, score), and indexer state are **separately addressable, position-addressed pieces** that can be individually restored at any token boundary — including mid-ratio-window. Getting this wrong is the single most expensive thing to retrofit, because prefix reuse and chunked prefill over a reused prefix both depend on it.
- **Storage dtype** is a decision, not a fidelity fact: fp16/bf16, or canonical UE8M0-block-scaled fp8 (halves footprint) — and you can store fp8 while dequantizing to fp16 in registers for the GEMM, getting both memory and canonicality `[V vllm]`. Whichever is chosen, the *values* the attention kernel consumes must be dequantized consistently.
- **`[?→strong]` The fp8/E4M3 round-trip is applied by the canonical path at *two* points:** the compressed entry store (`fused_compress_quant_cache.py:288-345`, only the non-RoPE 448 dims, after an explicit bf16 round-trip) and the raw-KV store. Prior evidence (~ds4 hardcodes it; some vLLM paths accept bf16) is now outweighed by a readable canonical kernel. **Treat it as required until a gate shows otherwise**, but keep it a gate: our engine may legitimately choose bf16/both, and the numerical delta must be measured, not assumed.
- **Critical:** the cache layout must match what the attention kernel expects; a transposed or offset cache corrupts only on long sequences.

---

## Part III — Prefill vs Decode

### Prefill (chunked batched — mandatory)

True chunked batched prefill is a hard requirement, not an optimization. Without it, daily-use performance is not meaningful.

- Process a chunk of tokens together, with causal masking within the chunk.
- **Every** stage must be batched: projections, attention, compressor, indexer, and the MoE. Attention is `[chunk, cache+chunk]`; the expert path batches the token→expert pairs.
- **No per-token host synchronization.** Any device→host copy inside the layer loop serializes the whole chunk.
- **The layer body must be the same code as decode**, parameterised by chunk size. Two bodies is how the two paths drift.
- The compressor/indexer run over the chunk and their ring boundaries must be respected mid-chunk.

**Gate:** `prefill(chunk)` must produce byte-equivalent state, logits, and token id to running the same tokens serially — **then**, separately, meet a throughput target. Structural equivalence and speed are two different gates; do not conflate them.

> **Finding about the current implementation (checkable):** `prefill_batched` currently batches only the dense projections and HC; attention, compressor, indexer, and the MoE run in a per-token loop, and the indexer top-k does a device→host copy plus a stream synchronize per CSA layer. As written, the equivalence gate above cannot pass. This is a structural problem with the layer body's shape, not a tuning issue.

### Decode

- One token at a time; attention is `[1, cache]`.
- **Bandwidth-bound** — expert streaming dominates.
- The indexer selects which compressed blocks to attend to.

**Design consequence:** both regimes must share one layer body; only the *scheduling* differs. Prefill wants large batched matmuls; decode wants efficient expert streaming and minimal per-token overhead. The expert fetch path (2.10.1) is the critical path in decode.

---

## Part IV — Kernel Inventory

| # | Kernel | Precision | Hardware path | Notes |
|---|---|---|---|---|
| 1 | Embedding gather + HC replicate | fp16/bf16 | memory | Output is 4 × 4096 |
| 2 | RMSNorm | fp32 accumulate | reduction | Stable on long seq |
| 3 | **HC project** | fp32 | reduction + dot | 16384 → 24 mixes; RMS over flattened dim |
| 4 | **HC Sinkhorn** | fp32 | iterative | **On the 4×4 comb matrix**, 20 iterations |
| 5 | **HC pre-combine / post-mix / head** | fp32→fp16 | elementwise | `[corrected]` replaces the plain residual add |
| 6 | **MLA q path** | fp16/bf16 | WMMA/GEMV | `wq_a` → q-norm(1024) → `wq_b` → per-head norm(512) |
| 7 | **MLA kv path** | fp16/bf16 | WMMA/GEMV | `wkv` → kv-norm; **K and V are the same tensor** |
| 8 | RoPE forward | fp16/bf16 | elementwise | Partial (tail 64); **two bases** by layer class |
| 9 | RoPE inverse | fp16/bf16 | elementwise | Tail 64 of output, before grouped proj |
| 10 | Attention scores + sink | fp32 accumulate | WMMA + softmax | Sink enters max + denominator only, no value |
| 11 | Softmax | fp32 | reduction | Mask with −INF, not 0 |
| 12 | Compressor | fp16/bf16 | WMMA + pooling | ratio ≠ 0 layers; per-dim softmax pooling; **`[Tier 0.2c]` `score += ape[pos%ratio]`** |
| 13 | Indexer | **fp8 (E4M3/UE8M0)** | WMMA (dequant to fp16) | `[corrected]` not INT8; **`[V]` ReLU required, per-head, before weighting** |
| 14 | Top-k selection | int32 | sort/select | Must match reference exactly; must be on-device for prefill |
| 15 | Grouped output projection | fp16/bf16 WMMA | `v_wmma_f32_16x16x16_f16` | `[corrected]` 8 groups × 1024 → `wo_b` |
| 16 | Router | fp16/bf16 | small matmul | `sqrt(softplus)`; hash branch for layers < 3 |
| 17 | Expert fetch | — | memory | Async, overlapped |
| 18 | Dequantization | INT4 → fp16/bf16 | registers | Fused with matmul; signed −8 bias |
| 19 | Expert matmul | fp16/bf16 WMMA | `v_wmma_f32_16x16x16_f16` | Fused with dequant; **clamped SwiGLU** |
| 20 | Shared expert | fp16/bf16 WMMA | `v_wmma_f32_16x16x16_f16` | Not quantized; same clamp |
| 21 | LM head | fp16/bf16 WMMA | `v_wmma_f32_16x16x16_f16` | fp32 accumulate |
| 22 | Sampling | fp32 | reduction | Host-side acceptable |

---

## Part V — Implementation Order (the earning order)

Build and certify in this order. Each item's gate must be green before the next is started.

**Tier 0 — Ground truth (no GPU).**
1. ~~Re-cite Hyper-Connections from `vllm/model_executor/layers/mhc.py`.~~ **DONE.** HC semantics confirmed; one error (sinkhorn eps placement) found and fixed. See 2.0.
2. ~~Re-cite the `ds4`-only claims (router, compressor, SwiGLU, indexer, MLA, sink, grouped output).~~ **DONE.** See Tier 0.2 / 0.2b notes. Remaining: settle the Hadamard choice (Gate 11) and the KV fp8-vs-bf16 delta (Gates 9/10).
3. **Reference derivation.** For each op, derive an independent oracle from the architecture (rank-1 reference hierarchy), **never** from our kernels. Handles the remaining `[?]` items below.
4. **Dequantization decode.** Verify against an independent decoder. Exact match required. (Our format layer already round-trips `[V our tests]` — reuse, re-validate.)

**Tier 1 — Primitives (GPU, one at a time).**
5. **RMSNorm** — vs fp64 reference.
6. **RoPE forward and inverse** — separately; two bases; forward∘inverse = identity.
7. **MLA q path and kv path** — including both intermediate norms.
8. **HC project + Sinkhorn** — verify doubly-stochastic convergence *before* composing it with anything.
9. **Attention score + sink + softmax** — verify at pos 0, within window, beyond window.
10. **Compressor** — verify pooling and boundary firing.
11. **Indexer + top-k** — exact index match. **Verifies the confirmed ReLU and fp8/UE8M0 quantization, and settles the Hadamard choice.**
12. **Grouped output projection** — low-rank and final.
13. **Router** — exact ids and weights, one hash layer and one top-k layer. **Verifies `tid2eid == [vocab,6]` against our artifact.**
14. **Expert matmul with fused dequant and clamped SwiGLU** — vs independent decoder.
15. **Shared expert** — separately.

**Tier 2 — Composition.**
16. **One full layer, Sliding class.** Verify `res_out` for a single token.
17. **Full layer, CSA class.** Then **HCA class**.
18. **Serial multi-token decode.** Verify state evolution across compressor boundaries.

**Tier 3 — Sequence.**
19. **Chunked batched prefill** with one shared layer body. Gate: chunk ≡ serial.
20. **Long-context lifecycle** — ring reuse and boundary compression past context capacity.

**Tier 4 — Integration (only after Tier 3 is fully green).**
21. **Streaming / tiering.** Gate: expert bytes bit-exact across Hot/Warm/Cold.
22. **Prefix cache manager.** Block table, cache key (tokens **+ non-token graph inputs**), matching, eviction; state pieces placed across tiers. Gate: **restore is byte-exact** with respect to never having evicted, and a candidate boundary outside the local window is detected rather than served stale (Part I §6.3 R3–R4).
23. **Generation loop.** Coherent output; logits agree with reference over several steps.

**Do not build the streaming system before the numerics are correct.** Streaming bugs and numerical bugs produce identical symptoms, and debugging both at once is intractable.

### Open unknowns — settle at the named gate, do not assume now

| Unknown | Where settled | Why it is listed |
|---|---|---|
| ~~Hyper-Connections semantics~~ | **Resolved Tier 0.1** | Confirmed against `mhc.py` + `kernels/mhc/torch.py`. Found and fixed one real error: `hc_sinkhorn_eps` applies to **every** denominator. |
| ~~Compressor APE~~ | **Resolved Tier 0.2c: REQUIRED** | The plan omitted it. `score += ape[pos % ratio]`, `[ratio, coff·head_dim]` fp32, added to **score only**. Present as 62 trained tensors in our checkpoint. |
| ~~Indexer ReLU~~ | **Resolved Tier 0.2: REQUIRED** | `relu` on the **per-head dot before weighting**, then `Σ_h w_h·relu(dot)` `[V sglang dsv4/indexer.py:121; qsa/dsa_indexer.py:43; cutedsl_fp8_paged_mqa_logits.py:43]`. My earlier retraction was wrong; my original assertion was right. |
| **Indexer Hadamard rotation** | **Open (measured at Gate 11)** | Real and in the DSV4 tree, but **logit-preserving** — a pre-quantization conditioning choice, not graph semantics. Both sglang paths (with/without) are valid. Decide by measuring score/top-k agreement; **must be symmetric over Q and K if used**. This item has now been mis-stated in three directions; it is listed here to stop further flip-flopping. |
| ~~`tid2eid` orientation~~ | **Resolved: `[vocab, 6]`** | Reference declares `(config.vocab_size, config.num_experts_per_tok)` `[V nvidia/model.py:820]`; still verify against our artifact at Gate 13. |
| KV fp8/E4M3 round-trip required vs optional | Gates 9 / 10 | **Strengthened toward required:** the canonical compressor kernel applies bf16+FP8/UE8M0 at two store points `[V fused_compress_quant_cache.py:288-345]`. Still a gate, because bf16 is a supported alternative and the delta must be measured. |
| MoE combine accumulation order | Gate 14 | Affects fp rounding vs reference tolerance. |
| Local-window reuse boundary behavior | Tier 4 gate 20 | Whether a reused prefix whose boundary predates the local window must rebuild the ring, or whether compressed state fully covers attention. sglang tombstones such leaves; confirm the correct handling for our state rather than assuming. |

### Anti-circularity rule

An oracle that shares code with the kernel tests only self-consistency. **Every Tier-1 gate must compare against an oracle written from the architecture description, independently of the kernel under test.** This is the specific defect that allowed the previous implementation's tests to pass on wrong logic.

---

## Part VI — Known Traps

These are the specific things that will break this model if implemented naively.

1. **Hyper-Connections are not optional and not a subset.** All 43 layers run a 4-stream HC pre-mix, Sinkhorn, and post-mix before and after each sublayer. Omitting it is a structural omission, not a numerical detail.
2. **Sinkhorn operates on the HC 4×4 comb matrix**, not on attention or compressor output. `[corrected]`
3. **Sinkhorn needs exactly 20 iterations** `[V config]`, with `eps` in the specific places (pre-mix add, row denominators, column denominators). Fewer iterations leave the comb not doubly stochastic.
4. **Layers 0–1 are compressor/indexer-free but still run HC.** Branch only the compressor/indexer, never the HC. `[corrected]`
5. **The Q path is low-rank with two norms.** `wq_a → q-norm(1024) → wq_b → per-head norm(512)`. Collapsing this into one matmul produces plausible but wrong output.
6. **There is no separate V.** A single shared KV head; attention uses the same tensor as key and value.
7. **Two RoPE bases.** Compressed layers use a different theta plus YaRN, with no amplitude scaling. Using one base everywhere is silently wrong.
8. **Inverse RoPE on the tail of the attention output**, before the grouped projection. Easy to omit, hard to diagnose.
9. **Attention sink enters the denominator only.** It must contribute no value vector, and must be included in the max.
10. **Indexer storage and quantization are fp8/UE8M0** — not INT8 `[V fused_indexer_q.py; V attention.py:964-975]`.
11. **Indexer ReLU is required and easy to omit.** `score[c] = kv_scale[c]·Σ_h w_h·relu(q_h·k_c,h)` — the ReLU is on the **per-head dot, before weighting**, not on the sum `[V sglang dsv4/indexer.py:121]`. At least three independent DSV4 implementations apply it.
12. **The Hadamard is real but logit-preserving.** A Hadamard-128 rotation (scale `1/sqrt(128)`) conditions indexer Q and K before quantization; the reference explicitly calls it *"logit-preserving"* `[V dsa_indexer.py:396]`. It is **not** required for semantics, **but if applied it must be applied to both Q and K** — one-sided application silently changes every score. Do not repeat this plan's earlier error of asserting it mandatory, *nor* its second error of asserting it absent. `[V dsa_indexer.py:192-204; V fused_q_indexer_rope_hadamard_quant (CUDA+ROCm)]`
13. **Hash routing for layers < 3.** These layers use a token-id lookup (`tid2eid`, shaped `[vocab,6]`) and have no bias tensor. Applying biased top-k to them reads a tensor that is not there.
14. **Router selection is biased, weighting is unbiased**, then normalized and scaled by 1.5; ties break to the lowest expert index.
15. **SwiGLU is clamped**: `max` only on the gate, both sides on the up branch. `[corrected]`
16. **Chunked prefill must be truly batched** with no per-token host sync, and must share the decode layer body.
17. **Nibble order and scale layout must match the packing format.** Verify with a synthetic tensor before trusting the loader.
18. **Top-k tie-breaking must match the reference exactly.** A different tie-break changes which experts fire.
19. **KV/state layout must match the attention kernel's expectation.** Corruption only shows on long sequences.
20. **The dense backbone and KV cache must never leave VRAM.** If they do, the streaming system is thrashing.
21. **Do not test a kernel against an oracle copied from that kernel.** The test must be derived from the architecture independently, or it proves nothing.
22. **The local state is window-bounded.** A cached prefix whose boundary predates the local sliding window cannot be served from a stale local ring — the long-range memory lives in the compressed entries. Detect and decline; do not silently reuse. (sglang calls these tombstones.)
23. **A cache key of token ids alone is insufficient.** Non-token inputs that change the graph (active tool set, thinking visibility) must participate in the key, or two different graphs will share state. `[V ds4 ext_flags; V colibri tainted]`
24. **Do not make the state a contiguous per-request allocation from position 0.** Prefix reuse and chunked prefill over a reused prefix both require individually addressable, position-addressed pieces. This is expensive to retrofit.
25. **Prefix reuse is not merely an optimization here.** On a tiered engine where the dominant cost is re-streaming experts from NVMe, recomputing the context each turn dominates everything; the state contract in Part I §6 is a correctness-of-design issue, not a tuning knob.
26. **The compressor has a learned APE added to `score`.** `score += ape[pos % ratio]`, shape `[ratio, coff·head_dim]`, fp32, added **before** the window softmax and **only to `score`** — not to `kv`. It is a real trained tensor in our checkpoint (62 of them). Omitting it, adding it to `kv`, or transposing it (`ds4` keeps it transposed) all shift every compressed row silently. `[Tier 0.2c]`
27. **DSV4 RoPE rotates the *last* 64 dims.** The DSV4 subclass `DeepseekV4ScalingRotaryEmbedding` overrides the generic parent to slice `[..., -rotary_dim:]`; the parent slices `[..., :rotary_dim]`. Reaching for the wrong one rotates the wrong dims and passes every "does it run" check. `[Tier 0.2c]`
