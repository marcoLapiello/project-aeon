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
| 1b | **The artifact's own `encoding/encoding_dsv4.py`** + 4 golden vectors | **Prompt encoding (Step 0) only** — see 2 below | the graph |
| 2 | our `config.json` + `.aeon` index | **Parameters** — dims, thetas, ratios, limits | op semantics |
| 3 | `aeon-references/ds4` @ `6289c51` | **Executable cross-check** (RDNA, full CPU graph) | model identity — it serves multiple families; comments are not proof of which graph a path belongs to |
| 4 | our own kernels / format layer | storage layout, swizzle | numerical correctness (circular — see Part V) |

**Platform caveat (mandatory):** upstream vLLM's DeepSeek-V4 ROCm path is gated on `_ON_GFX950` (MI350/CDNA4) and its quant config accepts only `fp8` / `deepseek_v4_fp8` / Quark-MXFP4-OCP `[V vllm/platforms/rocm.py:226, vllm/models/deepseek_v4/quant_config.py:142-165]`. There is **no `rdna`/`gfx11` path** in that tree `[V grep]`. Our checkpoint is `compressed-tensors`/`pack-quantized`/int4 `[V config.json]`, which does not match. Therefore upstream vLLM is a **graph-semantics reference only** — never cite it for storage, kernels, or batching.

> **Correction to the caveat above — a patched RDNA reference *does* exist.** The checkpoint card
> documents this exact W4A16 artifact running end to end on 4 `gfx1030` dies with `--dtype float16`:
> coherent greedy generation, perplexity 2.42 (prose) / 1.72 (code), and a 3,011-token prompt
> exercising the sparse attention indexer `[V checkpoint README]`. It requires *a vLLM build carrying
> the RDNA2 DeepSeek-V4 patches*, not upstream vLLM. So the accurate statement is **"no upstream
> gfx11 path"**, not "no gfx11 reference" — this is a candidate trusted-compatible reference for the
> correctness gate, for the same weights.

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
> in 2.4.3. A second item (the MoE router's normalization guard) was left soft here and is
> **resolved in Tier 0.2d** below — it is a non-semantic robustness detail, not a graph choice.

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

> **Tier 0.2d (done) — the MoE router, closed.** Re-read against the **naive** reference
> `vllm/tests/kernels/moe/test_topk_softplus_sqrt.py::_torch_topk_softplus_sqrt` — a better
> arbiter than the fused kernel because it carries no robustness shortcuts — plus the fused
> kernel, its dispatch, `nvidia/model.py`, and `ds4`. **No semantic errors.** Findings:
> the checkpoint names the correction bias **`ffn.gate.bias`** (renamed on load to
> `e_score_correction_bias`); it is added to **post-softplus scores**, not logits; there is
> **no group-limited routing** (`n_group`/`topk_group` absent); `norm_topk_prob=True`; the
> hash branch (layers 0–2) takes ids straight from `tid2eid` with no bias and no top-k. The
> earlier "normalization guard" soft spot is now **resolved as non-semantic**: the naive
> reference has no guard, the fused kernel uses `Σ>0?Σ:1`, `ds4` floors at `2⁻¹⁴`, and all
> three agree because `sqrt(softplus) > 0`. Our existing `moe_router.hpp` already matches the
> reference on every point — recorded as positive evidence, not a gate substitute.

> **Tier 0.2e (done) — the non-attention tail: prompt encoding, embedding, HC head, LM head,
> sampling, attention scale.** Re-read from `vllm/.../tokenizers/deepseek_v4_encoding.py`,
> `vllm/.../kernels/mhc/torch.py` + `triton.py`, `vllm/.../models/deepseek_v4/attention.py`,
> `vllm/.../models/deepseek_v4/nvidia/model.py`, `vllm/.../layers/mhc.py`, `generation_config.json`,
> `tokenizer_config.json`, and the safetensors headers. **Confirmed:** the attention scale is
> `head_dim**-0.5` = `1/sqrt(512)` (two references); the HC head reduction is exactly as 2.0/Step 3
> state (RMSNorm **without weight**, `hc_head_fn [4,16384]`, `hc_head_base [4]`, `hc_head_scale [1]`);
> the LM head is a separate `[129280,4096]` matrix (`tie_word_embeddings=False`); the combine order is
> routed-sum-then-`+= shared`; the embedding is expanded to 4 HC streams before layer 0.
> **Two real gaps found:** (1) the **HC comb logits carry a third scale** `hc_scale[2]` (checkpoint
> `hc_attn_scale` is `[3]`), which the plan omitted — now in 2.0; (2) **Step 0's chat template is not
> in `tokenizer_config.json`** — it is code in `deepseek_v4_encoding.py::encode_messages`, and our
> formatter implements only its basic skeleton. Step 0 is now specified against that reference.
> **Resolved:** the sampling defaults (`T=1, top_p=1, do_sample=true`) and the attention-scale `[?]`.

> **Tier 0.2f (done) — the attention composition (2.4.4).** Re-read from
> `sglang/.../dsv4/unified_kv_kernels/paged_prefill.py`, `.../dsv4/sparse_prefill_utils.py`
> (`combine_topk_swa_indices`), and `vllm/.../deepseek_v4/common/ops/{cache_utils,sparse_mla}.py`.
> **Confirmed:** the local window is `min(pos+1,128)`; the two KV sources (paged `unified_kv`
> prefix + per-forward `kv` extend) are summed under one order-invariant online softmax; the
> sink finalization is `m_final=max(m_i,sink)`, `l_final=l_i·α+exp(sink−m_final)`, `out=(acc·α)/max(l_final,1e−30)`.
> **One real error found and fixed:** 2.4.4 said compressed layers attend “the selected compressed
> rows”. **Only ratio-4 CSA layers select (indexer top-512). Ratio-128 HCA layers have *no
> indexer* and attend *all* committed compressed rows** (width `≥ seq_len/ratio`, capped at 8192).
> Ratio-0 layers attend local rows only. The checkpoint corroborates it: `attn.indexer.*` tensors
> exist only on ratio-4 layers. This is now stated explicitly in 2.4.4 and trap 33.

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

**Artifact portability (binding design constraint).** The goal is a graph that runs this model correctly
and can accept a **better W4A16 artifact later with little effort**. Two consequences:
- Op semantics come from the config and this specification. Anything that is a property of *this*
  checkpoint — its quantization scheme, expert payload layout, tensor naming — stays behind the
  descriptor/backend boundary, never hardcoded in the graph.
- **Do not fit tolerances to this artifact's fidelity floor.** The transcode is lossy (SNR min
  22.71 dB, median 25.69 dB `[V checkpoint README]`). A tolerance chosen merely to accommodate it
  would hide the same bug in the next checkpoint. Interpret a parity result against that floor;
  do not target it.

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

### Step 0 — Tokenization & Prompt Encoding `[Tier 0.2e: template located]`

> **Tier 0.2e.** The canonical DeepSeek-V4 prompt encoder is **code, not data**: the artifact ships
> its **own** revision (`encoding/encoding_dsv4.py`, 760 lines) plus four golden vectors — use that
> copy, not vLLM's sibling revision (648 lines; it differs). `tokenizer_config.json` has **no**
> `chat_template` field, and neither does the repack. The template must be reproduced from the
> artifact's encoder, and **Stage B of the** [checkpoint plan](checkpoint_verification_plan.md)
> **is its oracle** — this step must not be implemented from memory.

- Input text → token IDs via the model's tokenizer. Host-side.
- **Special tokens and defaults** `[V tokenizer_config.json; V encoding.py:21-29]`: `bos = "<｜begin▁of▁sentence｜>"` (id 0), `eos = pad = "<｜end▁of▁sentence｜>"` (id 1), `USER = "<｜User｜>"`, `ASSISTANT = "<｜Assistant｜>"`, `LATEST_REMINDER = "<｜latest_reminder｜>"`. **There is no system role token** — system/developer content is emitted bare (`system_msg_template = "{content}"`) `[V encoding.py:50,292]`.
- **BOS is conditional and off by default.** `add_bos_token: False` `[V tokenizer_config.json]`; the encoder prepends BOS only when `add_default_bos_token and len(context) == 0` `[V encoding.py:589]` — a fresh conversation gets it, a prefix-cache continuation does not.
- **Thinking mode is a parameter, not a fixed template.** `thinking_mode ∈ {"chat","thinking"}`; in thinking mode `<think>`/`</think>` wrap the reasoning, and assistant history renders `reasoning + thinking_end + content + eos` `[V encoding.py:373-381,407-414]`.
- **`[new]` Reasoning-effort prefix.** In thinking mode at index 0 the encoder prepends `REASONING_EFFORT_PROMPTS[effort]`, default `"low"` — **which is the empty string** `[V encoding.py:72-85,282-288]`. `high`/`max` inject a long literal instruction block. The default graph is unaffected by omitting it, but `reasoning_effort` must be a **parameter**, or `high`/`max` prompts silently differ.
- **`[new]` Tools force thinking to be kept.** `if any(m.get("tools")): effective_drop_thinking = False` `[V encoding.py:591-593]`. Dropping reasoning while rendering a tool-using prompt is a silent prefix change.
- **`[new]` `_drop_thinking_messages` semantics.** Keep `{user, system, tool, latest_reminder}` and everything at/after the last user; strip `reasoning` from earlier assistant messages; **drop** earlier `developer` messages entirely `[V encoding.py:641-648]`.
- **Extended surface our current formatter omits** (it matches the basic skeleton only): `developer` role, `latest_reminder` messages, `task` classification tokens (`<｜action｜>` …), tool-call/tool-result rendering (`<｜DSML｜invoke name=…>`, `tool_output_template = "<tool_result>{content}</tool_result>"`), `response_format_template`, and the tools→keep-thinking rule `[V encoding.py:26-70,142-190,302,343,394-414; V src/architecture/deepseek_v4/text/dsv4_chat_formatter.cpp]`.
  - **`[DONE — Step 0 verified]`** Implemented in `text/dsv4_prompt_encoder.{hpp,cpp}` and passing the artifact's four golden vectors **byte-for-byte (4/4)**. The first run scored 1/4 with the basic skeleton already correct and the gap exactly the list above, which is what an oracle is for. See [checkpoint_verification_plan.md §B.5](checkpoint_verification_plan.md).
- **Not deferrable:** the exact template including tool sections, and the record of **non-token inputs that affect the graph** (thinking mode, reasoning effort, active tool set, response format) so they enter the prefix cache key `[V ds4_kvstore.h ext_flags]`. See Part I §6.4.
- **Gate:** our formatter renders the artifact's four golden vectors (`encoding/tests/test_input_{1..4}.json` → `test_output_{1..4}.txt`) **byte-identically**, in both `chat` and `thinking` modes, with `reasoning_effort` default and non-default; the template round-trips; and the rendered text tokenizes to the same ids as the reference tokenizer. **This is the Step 0 oracle — see [checkpoint_verification_plan.md](checkpoint_verification_plan.md) Stage B.**

### Step 1 — Embedding Lookup

- Token IDs → embedding vectors. Table is unquantized fp16 `[V contract: embed.weight F16 [129280, 4096]]`.
- **`[corrected]` Output is *not* a plain `[batch, seq, hidden]` tensor.** The state entering the layer stack is `4 × 4096` per token: the embedding row is **expanded across the 4 Hyper-Connection streams**, and that shape is held until `hc_head` collapses it `[V nvidia/model.py:1379-1385 “V4 expands the token embedding to hc_mult streams before the first decoder layer and keeps that shape until hc_head() collapses it”; V config hc_mult=4]`.
- **Ensure:** the expansion is a broadcast (`expand`), so all 4 streams are identical — and it happens **before** layer 0's HC pre-mix.
- **Gate:** all 4 streams byte-identical to the checkpoint row for a known token id.
- **Kernel:** gather (sharded table), then broadcast.

> `[Tier 0.2e]` **MTP layers are present but out of scope.** The checkpoint also carries `mtp.0/1/2.*` (a Multi-Token-Prediction draft stack) `[V checkpoint]`. It is speculative-decoding machinery, not part of the base 43-layer forward pass; note it exists so the loader does not mistake it for a missing main-model tensor.

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
- **Comb logits:** the comb is a 4×4 matrix `C[contraction, output]`, flattened `C[i·hc + j]`, where **`i` indexes the incoming residual stream being read and `j` the outgoing stream being written**. Logits are `C[i,j] = mixes[2·hc + i·hc + j]·hc_scale[2] + hc_base[2·hc + i·hc + j]` `[V kernels/mhc/torch.py:75-77 `comb_logits = mixes[:, 2*hc_mult:].view(num_tokens, hc_mult, hc_mult) * hc_scale[2] + hc_base[2*hc_mult:].view(1, hc_mult, hc_mult)`; V ds4]`.
  - **`[corrected — Tier 1]` The index order is `contraction` then `output`, not `output` then `contraction`.** This plan previously said *"index = `8 + 4·output + contraction`"*, which is the **transpose** of the reference. The reference settles it in `mhc_post_torch`: the expansion is `torch.einsum("...ij,...ih->...jh", comb_res_mix, residual)`, so the comb's **first** axis is contracted against the residual's stream axis and its **second** axis becomes the output stream `[V mhc/torch.py:96-108]`. 2.7 below was already consistent with this; only 2.0's prose was wrong. Trap 34 records it.
  - **`[new — Tier 0.2e]` `hc_scale` is `[3]`, not `[2]`:** `[0]` pre, `[1]` post, `[2]` **comb**. Our checkpoint exposes `hc_attn_scale`/`hc_ffn_scale` as `F32 [3]` `[V checkpoint]`. **Omitting the `scale[2]`/`base` transform on the comb** silently changes the comb sharpness and hence every residual mix.
- **Sinkhorn:** softmax over the **output (`j`) axis**, add `hc_sinkhorn_eps`; normalize over the **contraction (`i`) axis** with `hc_sinkhorn_eps`; then `(sinkhorn_repeat − 1)` iterations of (normalize over the output axis, then over the contraction axis), **each denominator adding `hc_sinkhorn_eps`** `[V mhc_pre_torch: `softmax(comb_logits, dim=-1)`, then `/(sum(dim=-2)+eps)`, then alternating `dim=-1`/`dim=-2`]`. `sinkhorn_repeat = hc_sinkhorn_iters = 20` `[V config]`.
  - **`[corrected — Tier 1]`** This plan previously said "softmax over the source (contraction) axis; normalize over the output axis", which **swaps the two axes** relative to the reference. Verified empirically at Tier 1: the kernel matches the einsum reading to `4.0e-4` while the transposed reading differs from it by `0.52`.
  - **`[new — Tier 1]` The fixed point is not exactly doubly stochastic; it is offset by `hc_sinkhorn_eps`.** Because every normalization divides by `(sum + eps)` and the sum of a normalized row is 1, each pass multiplies entries by `1/(1+eps)`. Measured: at the fixed point the row and column sums are `1 − eps` (i.e. `max|sum − 1| = 1.0e-6` for `eps = 1e-6`), and uniform logits give a comb of `1/4·(1 − eps/4)` rather than exactly `1/4`. So "doubly stochastic within 1e-4" below is really "within `eps`" — the deviation has a floor set by `hc_sinkhorn_eps`, not by the iteration count. A gate that demanded exact double stochasticity would fail a correct implementation.
- **Pre-combine:** `layer_input = Σ_j pre[j] · residual[j, :]` `[V]`.

> `[corrected]` **Sinkhorn-Knopp belongs here, on the 4×4 comb matrix — not on the attention/compressor output.** The first revision placed it in the attention block; that was wrong.

- **Ensure:** `hc_pre_eps = hc_sinkhorn_eps = hc_eps = 1e-6` for our model `[V config hc_eps; V vllm passes hc_eps for both]`; **`eps` in every denominator** (initial softmax add, initial normalize, and both denominators of every iteration — *not* row-only); comb stored as `8 + 4·output + contraction`; `20` iterations, not fewer.
- **Gate:** `mixes(24)`, `pre(4)`, `post(4)`, `comb(16)` match reference; comb is doubly stochastic within 1e-4.

> **Gate result (Tier 1, item 8) — CERTIFIED, and the gate found a real error (in this plan).** `tests/test_v4_hc_oracle.cpp`, 14 lines green. Projection `9.4e-8`, pre/post/comb `5–7e-8` relative (fp32, essentially exact); the fp16 stages at `2.6e-4` / `4.0e-4`, i.e. one fp16 ulp.
>
> **The error was in 2.0's prose, not in the kernel.** The plan described the comb's flat index as `8 + 4·output + contraction`; the reference is `8 + 4·contraction + output` — the transpose. The gate settles it empirically: the kernel matches the einsum reading to `4.0e-4` while the transposed reading differs from the kernel by **`0.52`**, three orders of magnitude apart. A transposed comb is still doubly stochastic and still yields plausible residuals, which is exactly why a closeness-only test could not have caught this. `cpu_hc_post`/`hc_post_kernel` were already correct (they use `comb_mix[hci, hco]`, contraction first) and agree with 2.7. Corrected in place; see trap 34.
>
> **Also certified as load-bearing**, so a two-scale or short-iteration implementation cannot pass: reading the comb with `hc_scale[1]` instead of `[2]` changes it by `0.41`; 1 Sinkhorn iteration leaves a doubly-stochastic deviation of `0.18` versus `1.0e-6` at 20; `pre_mix` carries `+eps` while `post_mix` carries none (they land at `1e-6` and `0` when the sigmoid is driven down); and `post_mult = 2.0` is real (post-mix reaches `1.96 > 1`).
>
> **A gate-side bug this caught.** The oracle's closed-form self-check initially asserted that uniform logits yield exactly `1/4`. They do not — they yield `1/4·(1 − eps/4)`, because each normalization divides by `(1+eps)`. The self-check is now pinned to the exact expected value `eps/hc_mult`, which is stronger than a loose bound.
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

> **Gate result (Tier 1, item 7) — CERTIFIED.** `kernels/v4_gemv.hpp` (the dense projection primitive, extracted) + the MLA references in `reference/dsv4_oracle.hpp` + `tests/test_v4_mla_oracle.cpp`. This is a **composition** gate: MLA is not one kernel, it is a wiring of GEMV + RMSNorm + the per-head norm, and the failure mode is the wiring. The gate replays the pipeline's exact kernel order at the real dimensions (`4096→1024→32768`, `4096→512`) and compares every intermediate the plan names. Measured: max error `2.0e-4 … 4.6e-4` of peak, i.e. the fp16 store and nothing else.
>
> **The per-head Q norm is WEIGHTLESS — now cited two ways.** Upstream's `fused_q_norm_rope(q_input, q_output, eps, freqs_cis, positions)` takes **no weight argument** `[V sglang .../dsv4/elementwise.py:144-155; V .../deepseek_v4.py:1044]`, and the checkpoint declares exactly `attn.wq_a` / `attn.q_norm` / `attn.wq_b` / `attn.wkv` / `attn.kv_norm` — there is **no per-head norm tensor to load** `[V v4_model_contract.hpp:60-64]`. The pipeline already uses the weightless kernel, so contract, reference and kernel agree. An implementation that loads a weight there would be reading a tensor the artifact does not contain.
>
> **Three discriminating checks, not just closeness.** (a) The mid-path norm is **load-bearing**: dropping it changes `q` by `1.36` (135% of peak), so "norm between `wq_a` and `wq_b`" is distinguishable from "no norm" — which is what makes trap 5 testable. (b) The norm is **per head**: after it, every one of the 64 heads has `|mean(x²) − 1| ≤ 8.6e-5`, which is false if it were applied across all 32768 at once. (c) `kv` is exactly `head_dim` wide — one row, both roles (trap 6).

#### 2.3 — RoPE `[corrected — one base, now two]` `[Tier 0.2c: re-cited]`

- **Partial, on the tail.** Only the **last 64** of each 512-wide head rotates; layout is `[nope (448) | rope (64)]`. The DSV4 module is the *subclass* `DeepseekV4ScalingRotaryEmbedding`, which overrides the generic parent to use `query_rot = query[..., -rotary_dim:]` and `query_pass = query[..., :-rotary_dim]` — documented as *"Applies RoPE to the last rotary_dim"* `[V deepseek_scaling_rope.py:230-238, 262-263, 289-291]`. `ds4` agrees (`n_nope = head_dim - n_rot; tail = x + h*head_dim + n_nope`) `[V ds4:11073-11074, 11082]`. **Trap:** the *generic* parent `DeepseekScalingRotaryEmbedding.forward_static` slices `query[..., :rotary_dim]` (the **first** 64) — using it by mistake rotates the wrong dims. Confirm against the subclass.
- **Interleaved (GPT-J), not NeoX.** `is_neox_style=False`, `cos.repeat_interleave(2)`, `rotate_gptj` `[V rope.py:50; V deepseek_scaling_rope.py:272-279]`; `ds4` rotates adjacent pairs `tail[i], tail[i+1]` in steps of 2 `[V ds4:11088-11094]`.
- **Two bases by layer class.** `rope_theta = compress_rope_theta` when `compress_ratio > 1`, else `rope_theta` `[V rope.py:28-30]`; `ds4` keys on `compress_ratio != 0` `[V ds4:11100-11103]`. Identical for our `{0, 4, 128}` (only 0 is ≤ 1). Here: **10000** for ratio 0, **160000** for 4/128.
- **YaRN only on compressed layers.** ratio > 1 → `deepseek_yarn` with `factor=16, beta_fast=32, beta_slow=1`; sliding/ratio-0 layers → `factor=1.0` (plain RoPE) `[V rope.py:31-44]`; `ds4` sets `ext_factor=1` only when compressed `[V ds4:11106-11112]`.
- **`[corrected]` No amplitude scaling — now triple-cited.** vLLM sets `mscale = 0` and `mscale_all_dim = 0` with the comment *"Disable mscale"*, so `yarn_get_mscale(·,0) = 1` and the effective `mscale = 1.0` `[V rope.py:38-39; V deepseek_scaling_rope.py:14-18, 48-52]`. `ds4` says it in words — *"DeepSeek V4 reference RoPE uses interpolation without that magnitude change, so pass the inverse factor here and let the helper cancel itself out"* — and divides out `1 + 0.1·log(1/freq_scale)` `[V ds4:11133-11144]`.
- **Compressed entries rotate at the window start: `comp_pos = pos + 1 − ratio`**, not at `pos` `[V ds4:13410, 13498]`; vLLM states the same as `(positions // compress_ratio) * compress_ratio` `[V compressor.py "position used" comment; V fused_compress_quant_cache.py:206]`. The two agree because the entry only exists when `(pos+1) % ratio == 0`, where the floor equals `pos+1−ratio`.
- **Inverse RoPE** on the attention-output tail **before** the grouped projection. The DSV4 rope `forward` takes an explicit `inverse` flag that negates `sin` `[V deepseek_scaling_rope.py:249-252, 281-284]`; `ds4` passes `inverse=true` for the output `[V ds4:13993, 14490]`.
- **Gate:** forward∘inverse = identity; tables match reference at two positions, one of them past `original_max_position_embeddings = 65536`; **the rotated slice is the last 64 dims, not the first.**

> **Gate result (Tier 1, item 6) — CERTIFIED.** `kernels/v4_rope.hpp` (forward/inverse, batched and single-position) plus the two-class spec in `reference/dsv4_oracle.hpp`. Measured: tables match to `3.0e-8` at position 1; forward and inverse match the oracle to `4.9e-4` (= one fp16 ulp); the nope region `[0,448)` is **bit-identical** after a forward rotation, which is what makes trap 27 impossible to pass silently; device forward∘inverse returns the input to within one fp16 ulp of the row.
>
> **One quantified finding worth carrying forward.** Both our kernel and the reference store the tables in **fp32** and compute `angle = pos * freq` in fp32. Beyond `original_max_position_embeddings` the fp32 angle has an ulp of `2^-7 ≈ 7.8e-3` at position 65537, so the two tables differ by up to **`3.4e-3` in cos/sin before any kernel error is involved** (measured: `2.4e-3` sliding, `3.4e-3` compressed over all positions; `8.6e-4` / `1.1e-3` at position 65537 alone). This is a precision floor, not a defect — it bounds how tightly long-context logits can ever agree with a reference that rounds the same way, and it is the reason an fp64 table comparison at large positions must be judged in **absolute** terms. Rotating to fp64 tables would remove it and is a candidate change, but it is **not** part of this revision: the reference does not do it.

#### 2.4 — Attention

**2.4.1 — Score, sink, and softmax (all layers)**
- Scores `S = q·k / sqrt(512)`, fp32 accumulate. The scale is `head_dim**-0.5` with `head_dim = 512` (the **full** head, nope+rope), **not** the index head dimension `[V vllm attention.py:210-231 `self.head_dim=config.head_dim; self.scale=self.head_dim**-0.5`; V sglang deepseek_v4.py:665 `self.softmax_scale=self.head_dim**-0.5`]`. The indexer uses a separate `128**-0.5` (2.4.3).
- **`[new — Tier 0.2e]`** This plan previously asserted the scale as `[V]` without a citation; it is now cited at both references and is a plain `1/sqrt(head_dim)` — no extra low-rank MLA scale factor.
- **Attention sink** `[V Tier 0.2b]`: a per-head fp32 logit `attn_sink[H]`, described by the reference as *"a virtual extra K with V=0"*. Concretely `m_final = max(m_i, sink)`, `l_final = l_i·alpha + exp(sink − m_final)`, and *"the sink itself contributes 0 to acc since V_sink = 0"* `[V sglang dsv4/unified_kv_kernels/paged_prefill.py:194-203]`. So it enters the **max and the denominator only**, and contributes **no value** — exactly as this plan stated. The alternative implementation (a zero *value* row with the sink logit) is numerically identical.
- Sink is padded to `padded_heads` when the head count is padded (vLLM fills padding with `-inf`, sglang with `0`) — a host-side padding detail only `[V vllm attention.py:235-238; V sglang deepseek_v4.py:824-834]`.
- Mask disallowed keys with `-INF` (not zero) so they are excluded from both max and denominator.
- **Gate:** matches reference at pos 0, pos < window, pos > window.

> **Gate result (Tier 1, item 9) — CERTIFIED.** `tests/test_v4_attention_sink_oracle.cpp`, 19 lines green. Kernel under test: `v4_sliding_window_attn_wave32_kernel`, the one launch where scores + sink + softmax + value combination are self-contained. Measured `2.6e-4 … 3.1e-4` relative to peak (fp16 output), at all four positions: `pos 0` (1 key), `pos 5` (6 keys), `pos 127` (first full window), `pos 149` (mid-window, oldest keys excluded).
>
> **The window boundary is exact, and this is the strongest line in the gate.** Perturbing key row 0 to a magnitude of 50 changes the output at `pos 5` by `65x` and at `pos 127` by `227x`, but at `pos 149` by **exactly `0.0`**: at pos 149 the window is keys `22..149`, so key 0 is not read at all. That is the difference between `min(pos+1, 128)` and "every key up to pos", and no closeness test distinguishes them.
>
> **The sink is a pure rescaling, and that is now an asserted invariant.** Because the sink contributes no value, `out_with_sink = out_without_sink × c` for a single scalar `c = l_i / l_final`, elementwise. The gate asserts the ratio is constant across all 32,768 elements (spread `0.0`); a value contribution would make it vary per element, and a dropped sink would make it 1. Supporting lines: a large negative sink is bit-identical to *no* sink (`max_abs = 0.0`), a large positive sink absorbs the mass (`peak +60 = 2.7e-25` versus `0.22` unsunk), and the output stays finite at both extremes.
>
> **Scale re-confirmed:** the index-head scale `1/sqrt(128)` would change the output by `0.41`, the rope-part scale `1/sqrt(64)` by `0.76` — so the gate distinguishes the full-head scale from both wrong candidates.
>
> **HONEST LIMITATION — `sink in the max` is not testable from the output.** The plan (following upstream) says the sink enters the max as well as the denominator. That is a numerical-robustness property: it keeps every exponent `≤ 0` when the sink dominates. But in exactly the regime where it matters, all mass sits on the sink and the output is zero **whether or not the sink was included in the max** — so no output comparison can discriminate it. This gate asserts finiteness at the extremes and does not claim more. Recorded as trap 35 so it is not mistaken for coverage later.
>
> **A gate-side bug this caught, worth keeping in mind.** The first run *failed* the `sink = +60` line with `max_rel = 1.0`. That was not a kernel defect: both the reference and the kernel produce values at the fp32 noise floor there (`2.7e-25`), and a relative comparison of two zeroes is meaningless. The line now uses an absolute bound against the un-sunk scale. This is the second time in this tier that a **relative** tolerance was the wrong instrument for a quantity whose magnitude collapses — the first was `cos`/`sin` near a zero crossing in RoPE. Prefer absolute bounds whenever the quantity is signed or can collapse toward zero.

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

> **Gate result (Tier 1, item 10) — CERTIFIED, both ratio classes.** `tests/test_v4_compressor_oracle.cpp`, 32 lines green across `ratio 4` (coeff 2, window 8, overlap) and `ratio 128` (coeff 1, window 128). Both kernels are exercised: `v4_save_compressor_state_kernel` per token and `v4_materialize_compressed_entry_kernel` per boundary. Final row measured at `3.3e-4 … 4.3e-4` relative (fp16 store).
>
> **The APE is where the plan says is, and this is now proven three ways.** `partial_kv` is compared against a plain widening of the input and matches **exactly** (`max_abs = 0.0`), so no APE can leak into the kv branch — that is the bit-exact form of "score only". `partial_score` matches `score + ape[position % ratio]` to `1.7e-7`. And the row index is confirmed to be a genuine modulo: positions `p` and `p + ratio` produce **identical** rows, while `p` and `p + 1` differ by `0.99`. The layout used is the checkpoint's `[ratio, coeff·head_dim]`, not `ds4`'s transpose.
>
> **The compression is per dimension, and the gate asserts both segments are read.** For each output dimension `d`, the softmax runs over the window's scores at dimension `d` and combines the kv values at dimension `d`. For `ratio 4` this is not optional bookkeeping: zeroing the second segment changes the row by `2.87`, proving the overlap is actually consumed. Dropping half the window changes it by `2.87` (ratio 4) / `0.03` (ratio 128) — note the ratio-128 figure, which is why the check drops *half* the window rather than one entry: a single missing entry among 128 moves the average by only `4e-3`.
>
> **Structural fact discovered while gating: the first compressed entry of a ratio-4 layer is half-empty.** The window is `(1+overlap)·ratio = 8` wide, but the earliest possible emission is `pos = ratio − 1 = 3`, so the window starts at `-4`. Exactly the leading `ratio` offsets are unwritten — i.e. the **whole older segment is absent**, and that first entry is built from the newer `ratio` tokens alone. Including the absent offsets changes the row by `0.48`, so the kernel's skip is load-bearing and not a no-op. **For ratio 128 there is no such case**: the first entry is emitted at `pos = 127` with a window starting at exactly `0`, so nothing is ever skipped. A gate that asserted truncation for ratio 128 would be asserting a false property (this one did, on its first run).
>
> **Launch contract worth recording.** `v4_materialize_compressed_entry_kernel` maps `dimension = threadIdx.x` and guards with `dimension < head_dim`, so it must be launched with **at least `head_dim` (512) threads**. Launched with 256 the upper dimensions are silently never written — which is exactly how this gate failed initially, and it is the kind of bug that would look like a numeric disagreement rather than a launch error. `v4_save_compressor_state_kernel` is strided and has no such constraint.

**2.4.3 — Lightning Indexer (CSA layers only — the ratio-4 layers: even indices 2..42)** `[corrected]` `[Tier 0.2f: scope confirmed]`

> **Scope (Tier 0.2f).** The indexer runs **only** on ratio-4 (CSA) layers. Ratio-128 (HCA)
> layers have a compressor but **no indexer** — they attend all committed compressed rows
> directly (2.4.4), and the checkpoint carries no `attn.indexer.*` tensors for them
> `[V sparse_mla.py:252-260; V cache_utils.py:938; V safetensors header]`.

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

> **Gate result (Tier 1, item 11) — CERTIFIED, and the gate found a real bug in a kept kernel.** `tests/test_v4_indexer_oracle.cpp`, 11 lines green. Scores now match the oracle to `2.4e-7` and the top-k selection matches **exactly** (`0 of 512` differ).
>
> **`v4_indexer_scores_kernel` was missing the ReLU.** It computed `Σ_h w_h·(q_h·k_c,h)` with no rectification. On the first run the gate reported `max_rel = 0.98` on the scores and **65 of 512 wrong top-k indices** — a wrong selection of which compressed rows attention reads, which would have surfaced only as a subtle quality regression. Fixed in place; trap 11 updated to record that it was found in *our* code, not merely documented.
>
> **Why it survived every existing test, stated precisely.** The legacy `test_v4_real_dense_parity` loads real checkpoint weights and still passed with the ReLU missing. That is not a circularity failure — it is a **coverage** failure: that test calls exactly four kernels (`v4_gemv_fp16_kernel`, `v4_gemv_fp16_vec8_kernel`, `v4_grouped_wo_a_wave32_kernel`, `v4_rmsnorm_wave32_kernel`) and never touches the indexer or top-k. A green real-weight test says nothing about a kernel it does not call. (Verified by inspection: no `indexer`/`topk` reference anywhere in that file.)
>
> **The gate proves the ReLU is load-bearing before relying on it.** 50% of the per-head dots in the test data are negative, the with-ReLU and without-ReLU score vectors differ by `0.98`, and the two selections differ in `65 of 512` indices. Without that, a passing score comparison could not distinguish a correct indexer from one that omits the ReLU.
>
> **Top-k details verified:** descending order, ties to the **lower index** (trap 18), and the degenerate case `candidates <= top_k` selects all of them with no padding.

> ### Gate 11 — the Hadamard rotation: **SETTLED (do not apply it)**
>
> This item had been mis-stated three times, so it is now settled by **measurement** rather than argument. The gate applies a normalized `1/sqrt(128)` Sylvester–Hadamard to each 128-wide indexer head and compares:
>
> | Configuration | Scores | Top-k |
> |---|---|---|
> | Hadamard on **both** Q and K | unchanged, `3.6e-16` | unchanged, `0 of 512` |
> | Hadamard on **Q only** | differ by `1.46` | differ, `180 of 512` |
> | Orthogonality check (Gram matrix preserved) | deviation `0.0` | — |
>
> So the plan's framing is correct and now demonstrated: the rotation is **logit-preserving** because it is orthogonal, and it is harmless only when applied **symmetrically**. One-sided application is the failure mode, and it is a *large* one.
>
> **Decision and its basis.** Our engine does **not** apply it, for two independent reasons: (1) the reference itself drops it in its fused path, labelled *"(logit-preserving)"* `[V dsa_indexer.py:395-398]`; and (2) the rotation exists to condition values before an **fp8 round-trip**, and our indexer stores its K cache in fp16 — there is nothing to condition. If the indexer K store is ever moved to fp8, this decision must be revisited, and the rotation must then be applied to **both** Q and K.
>
> The remaining indexer-store difference is recorded rather than hidden: the canonical path keeps the indexer K cache in **fp8 (UE8M0)** with a **per-key scale** (`kv_scale[c]` in the reference's score), and neither is present in our fp16 path. That delta belongs to the KV-precision gate (Gates 9/10), not here.

**2.4.4 — Attention composition: local rows + compressed rows** `[Tier 0.2f: re-cited]`

> **Tier 0.2f.** The composition was re-read from the readable kernels
> `sglang/.../kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py`
> (`_sparse_attn_v4_paged_prefill_kernel`), `sglang/.../layers/attention/dsv4/sparse_prefill_utils.py`
> (`combine_topk_swa_indices`), and `vllm/.../models/deepseek_v4/common/ops/cache_utils.py` +
> `sparse_mla.py`. **One real error found and fixed: the plan implied every compressed layer
> *selects* compressed rows. Only CSA does; HCA attends all of them and has no indexer.**

- **Three attention row-sets, one per layer class** `[V cache_utils.py:938-939 `topk_width = active_topk_width if compress_ratio == 128 else index_topk`; V :1478-1484]`:
  - **ratio 0 — Sliding (layers 0, 1):** local SWA rows only. `topk = 0`; the compressed branch is a no-op `[V :939 `topk = 0 if compress_ratio == 1 else …`; V sparse_prefill_utils.py:31-32]`.
  - **ratio 4 — CSA (indexer layers):** local SWA rows **+ the indexer-selected `index_topk = 512`** compressed rows. `topk_width = index_topk` `[V :938; V config index_topk=512]`. Indices are **request/sequence-local** `[V :1484 `decode_compressed_indices_are_local=compress_ratio == 4`]`.
  - **`[corrected]` ratio 128 — HCA:** local SWA rows **+ ALL committed compressed rows — there is no indexer and no top-k selection.** The width is `active_topk_width = min(max(next_pow2(ceil(seq_len / ratio)), 128), c128a_max_compressed)`, and the code asserts `active_topk_width ≥ seq_len // ratio`, i.e. it always covers **every** compressed entry so far `[V sparse_mla.py:252-260]`. The cap is `c128a_max_compressed = ceil(ceil(max_model_len / 128) / 128)·128` = **8192** for our `max_position_embeddings = 1048576` `[V sparse_mla.py:39,158-170]`. Indices are **global** `[V :1484 `has_decode_compressed_lens=compress_ratio == 128`]`.
  - Structural confirmation from the checkpoint: `attn.indexer.*` tensors exist **only** on ratio-4 layers `[V safetensors header — `layers.11.attn.compressor.ape [128,512]` has no sibling `indexer.compressor.ape`; `layers.10` has both]`. Running the indexer on a ratio-128 layer would read tensors that do not exist.
- **Local rows are exactly `min(pos+1, 128)`.** `swa_start = max(pos − (WINDOW_SIZE−1), 0)`, `swa_len = pos − swa_start + 1` `[V cache_utils.py:893-894]`. (`left_add`/`right`/`image_width` widen this only for the vision variant, which is not our graph.)
- **Composition of the two sources.** One buffer `unified_kv [total_pages, D]` holds **both** the SWA ring (slots `[0, swa_pages)`) and the compressed pages (`[swa_pages, total_pages)`); the current chunk's freshly-computed K lives in a separate per-forward `kv [total_tokens, D]` **not yet written to the SWA ring** `[V paged_prefill.py:13-33 docstring]`. Per query, the selected rows are an int32 list `combined_indices = [ compressed indices (rebased) | swa positional indices (rebased) ]`, with `-1` marking padding/skips `[V sparse_prefill_utils.py:11-12,31; V combine kernel]`. The kernel then sums the two regions sequentially under **one shared online-softmax accumulator**, which is **order-invariant** — region order and index order are not semantically load-bearing `[V paged_prefill.py:45-47 docstring “Order of regions does not affect correctness”]`.
- **Score and mask in the kernel:** `scores = (q·kᵀ)·softmax_scale`, masked to `-3.4e38` for invalid slots/heads, then online max/rescale; `softmax_scale = head_dim**-0.5` (2.4.1) `[V paged_prefill.py:145-190]`.
- **Sink finalization (exact form).** After the loops, `m_final = max(m_i, sink)`, `alpha = exp(m_i − m_final)`, `l_final = l_i·alpha + exp(sink − m_final)`, `out = (acc·alpha) / max(l_final, 1e−30)`, and `out = 0` wherever `l_final ≤ 0` `[V paged_prefill.py:194-206]`. This is the 2.4.1 sink rule implemented exactly — sink enters the max and the denominator, contributing zero to the numerator.
- **Gate:** for a scripted sequence across a ratio boundary, (a) the ratio-0/4/128 row-sets are each correct — in particular **HCA attends every compressed row and never calls the indexer**; (b) the local row count is `min(pos+1,128)`; (c) output matches the serial reference; (d) the online-softmax result is **bit-stable under index permutation** (a cheap invariant that catches accumulation bugs).

#### 2.5 — Output Projection `[corrected — grouped, not a single matmul]`

> **Re-cited (Tier 0.2b).** Confirmed against `sglang/.../models/deepseek_v4.py:405-440`.

- **Inverse RoPE** first (2.3).
- `wo_a` is **grouped**: shaped `[G, R, D]` = `[o_groups, o_lora_rank, group_dim]`, applied as `einsum("tgd,grd->tgr", o, wo_a)` where `o` is `[T, G, D]`. Result `[T, G, R]` = flattened `[T, G·R]` `[V sglang deepseek_v4.py:405-440 docstring + einsum]`. With our dims: `G=8`, `R=1024`, `group_dim = 8 heads × 512 = 4096` — matching the `wo_a [8192, 4096]` contract.
- `wo_b` maps `G·R = 8192` → 4096 `[V contract]`.
- **Ensure:** group assignment is **8 contiguous heads** per group; the stored `wo_a` row-major layout must match the `[G, R, D]` interpretation (group-major), i.e. `wo_a[g]` is a `[1024, 4096]` block, not an interleaved slice.
- **Gate:** low-rank tensor (8192) and final 4096 match reference.

> **Gate result (Tier 1, item 12) — CERTIFIED.** `reference/dsv4_oracle.hpp::grouped_wo_a` + `tests/test_v4_grouped_wo_oracle.cpp`, 14 lines green. `z` matches the einsum to `2.9e-4` (fp16 inputs, one fp16 ulp); `attn_proj` matches the oracle to `3.1e-4` fed the kernel's own `z` and `4.2e-4` over the full chain.
>
> **The group structure is asserted by construction, not by closeness.** Three checks make a flat or shared implementation impossible to pass:
>  * Reading the same bytes as an interleaved `[R, G, D]` moves the result by `1.38` — so "group-major" is falsifiable here, not an assumption.
>  * Perturbing group 0's eight contiguous heads leaves **all 14 336 elements of groups 1–7 bit-identical**, and moves 1024 of 1024 of group 0. A shared weight block, a global reduction over the whole 32 768-wide row, and a wrong group stride all violate this; a plain closeness check does not.
>  * Perturbing the *last* element of group 0 (head 7, dim 511) still moves group 0 (1008 of 1024), so the reduction reaches the whole 4096-wide block rather than truncating at the first head.
>
> **`wo_b` is certified twice, and its orientation is shown load-bearing.** Fed the kernel's own `z` it isolates the 8192→4096 map (`3.1e-4`); fed the oracle `z` it certifies the chain (`4.2e-4`). Reading `wo_b` transposed differs by `1.52` — the plan states "no transpose anywhere in the path", and this is the line that would fail if that stopped being true.
>
> **Corroboration, not certification.** The legacy real-weight `test_v4_real_dense_parity` *does* exercise this pair (`run_grouped_wo_a` → `run_fp16_gemv` on `attn.wo_b.weight`, lines 429–432) and passes with real checkpoint weights. That is consistent with the above but is not the evidence — it shares the layer's `[8192, 4096]` interpretation, so it cannot falsify the layout question the way the interleaved measurement does. (The reverse also held: that same test's coverage gap is why the indexer ReLU survived — see item 11.)

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

#### 2.9 — MoE Router `[corrected — scoring and layer branching were wrong]` `[Tier 0.2d: re-cited]`

> **Re-cited (Tier 0.2 + 0.2d).** Confirmed against the **naive reference**
> `vllm/tests/kernels/moe/test_topk_softplus_sqrt.py::_torch_topk_softplus_sqrt` (the arbiter for
> semantics and tie-break), the fused kernel `vllm/.../router/dsv4_topk.py`, the dispatch
> `vllm/.../router/fused_topk_bias_router.py`, `vllm/models/deepseek_v4/nvidia/model.py`, and the
> `ds4` cross-check. **No semantic errors found.** Two facts the plan was missing are added:
> the **checkpoint tensor name** and the **absence of group routing**.

- **Logits:** `logits = gate_weight @ x` → 256, computed in **fp32** (`router_logits_dtype=float32`) `[V contract ffn.gate.weight F16 [256,4096]; V nvidia/model.py:979]`.
- **Score:** `scores[e] = sqrt(softplus(logits[e]))`, `softplus` with the default threshold 20 — i.e. `sqrt(x > 20 ? x : log(1+exp(x)))` `[V test_topk_softplus_sqrt.py:32 `F.softplus(...).sqrt()`; V dsv4_topk.py:82; V config scoring_func=sqrtsoftplus]`.
- **Selection — two mutually exclusive branches:**
  - **`[corrected]` Layers 0–2 use hash routing, not top-k.** `is_hash_moe = layer_index < num_hash_layers` `[V nvidia/model.py:810]`. IDs are taken **directly** from a table: `topk_ids = tid2eid[input_ids]` — **no bias, no top-k, no score comparison** `[V test:38 `hash_indices_table[input_ids.long()]`; V ds4:11540-11545]`. Those layers have **no bias tensor** `[V nvidia/model.py:812-813 comment "hash MoE doesn't use e_score_correction_bias"; **V checkpoint: `ffn.gate.bias` present only on layers 3..42, absent on 0..2**]`. The table is `(vocab_size, num_experts_per_tok)`; **measured in our artifact as `I64 [129280, 6]`** `[V safetensors header; V nvidia/model.py:820 *expected* [vocab,6]]`.
  - Layers ≥3: `selection[e] = scores[e] + e_score_correction_bias[e]`, then top-6 by `selection` `[V test:69-75; V dsv4_topk.py:83; V config topk_method=noaux_tc]`.
  - **`[new — Tier 0.2d]` No group-limited routing.** `n_group` and `topk_group` are **absent** from `config.json`, so this is a **flat top-6 over all 256 experts**, not the DeepSeek-V3 `noaux_tc` grouped variant `[V config.json: fields absent; V the flat kernel has no group term]`. Do **not** add `n_group`/`topk_group` grouping.
  - **Tie-break: lowest expert index wins.** Reference: stable descending `argsort`, so equal scores retain ascending index order `[V test:75-77; V dsv4_topk.py:91-92]`.
  - NaN selection scores → `-1e30` is a **fused-kernel robustness guard only**, absent from the naive reference `[V dsv4_topk.py:86; V absent in test]`.
- **Weighting (both branches):** the stored weight is the **unbiased** `scores[id]` — in the hash branch it is `scores` of the table-selected experts `[V test:80 gather; V ds4:11620-11626; V our moe_router.hpp hash branch]`.
- **Normalize then scale:** the reference divides by `Σ` **then** multiplies by `routed_scaling_factor = 1.5` `[V test:77-80; V config norm_topk_prob=True, routed_scaling_factor=1.5]`. (Mathematically `w·f/Σ` ≡ `(w/Σ)·f`; the plan's single-expression form is equivalent.) **Guard:** the naive reference has **no** guard; the fused kernel uses `Σ>0 ? Σ : 1`; `ds4` floors at `6.103515625e-5` (= `2⁻¹⁴`) `[V dsv4_topk.py:102-103; V ds4:11626]`. Since `sqrt(softplus) > 0` always, the three agree in practice — **record which we implement**. `[corrected — Tier 1]` This plan previously said "ours uses the fused `Σ>0?Σ:1`". It does not: **both our CPU and device paths use `Σ + 1e-20`** `[V moe_router.hpp:91, 184]`. That is a *fourth* form, and it is not bit-equivalent to the other three — it differs when `Σ ≲ 1e-20`, i.e. when a score falls below ~`1.6e-21`, i.e. when a logit falls below about **−96**. No such logit is reachable from fp16 weights, so the divergence is inert; the gate records the threshold rather than claiming an equivalence that does not hold.
  - *Order note:* in the hash branch the expert ids keep the **table's column order** (not score-sorted); only the top-k branch is score-ordered. Irrelevant to the weighted sum, but it changes the id *sequence* if compared positionally.
- **`[new — Tier 0.2d]` The checkpoint tensor is named `ffn.gate.bias`**, not `e_score_correction_bias — the reference renames it on load: `{".ffn.gate.bias": ".ffn.gate.e_score_correction_bias"}` `[V nvidia/model.py:1704]`. It is F32 `[256]` `[V safetensors header]`. A loader that treats `ffn.gate.bias` as a **linear logit bias** (adding it before `softplus`) is silently wrong — it must be added to the **post-softplus scores**. Our loader already maps it correctly (loaded as `d_gate_bias`, nulled for hash layers) `[V v4_dense_weight_binding.hpp:144; V v4_pipeline.hpp:1024]`.
- **Ensure:** `num_experts == 256`, `top_k == 6` `[V dsv4_topk.py can_use_dsv4_topk]`.
- **Gate:** selected ids **and** weights match the naive reference exactly, for one hash layer (0–2) and one top-k layer (≥3); ids compared **as a set and positionally**; `tid2eid` `[129280,6]` verified against our artifact; and specifically that the bias is added **after** `softplus`.

> **Gate result (Tier 1, item 13) — CERTIFIED. No defect found.** `reference/dsv4_oracle.hpp` gains `softplus` / `router_score` / `router_topk` / `router_hash`; `tests/test_v4_router_oracle.cpp`, **23 lines green**. Weights match the fp64 oracle to `1.3e-7` (the router is fp32, so this is a rounding bound, not an fp16 one) and the ids match **positionally**, not merely as a set, on all tokens.
>
> **All four of the plan's traps are shown load-bearing before the pass is trusted.**
>  * *Bias placement.* The wrong variant — bias on the logit before softplus, the shape a loader gets if it reads `ffn.gate.bias` as a linear bias — **changes the routing on 3 of 8 tokens** and moves the weights by `0.12`. The kernel sits `0.12` away from it and `1.3e-7` from the correct rule. Dropping the bias entirely changes **8 of 8** tokens, so the bias is not a no-op the pass ignores.
>  * *Flatness.* With six top logits planted in six different 32-expert groups, the DeepSeek-V3 grouped variant (top-2 groups) selects a **different set** and its weights sit `0.66` away. The kernel agrees with the flat rule positionally. `config.json` is also read as text to confirm `n_group` and `topk_group` are **absent** — the structural evidence the plan cites.
>  * *Tie-break.* Four experts tied at the maximum select in ascending index order (`7, 70, 180` first).
>  * *Hash branch.* Ids are reproduced **in the table's unsorted column order** (a score-sorting implementation would be caught), the weights are the unbiased scores, and a `+100` bias leaves the ids unchanged — the branch genuinely ignores it.
>
> **`tid2eid` is verified against the artifact, not a synthetic table.** `layers.0.ffn.gate.tid2eid` is `I64 [129280, 6]`; the table exists on layers 0–2 and `ffn.gate.bias` on layers 3–42, with **no overlap** across all 43 layers; and replaying the kernel in hash mode at **real token ids from `profiling-prompts/first-prompt.jsonl`** reproduces the artifact's own table rows exactly (`0 of 96` ids wrong, values in `[1, 252]`).
>
> **A gate-side bug worth recording.** The first run assigned a 64-row synthetic table but passed token ids `11, 22, 33`: the kernel indexes the table by **token id**, so it read past the end of the allocation and produced a hash id mismatch and a `max_rel = 0.54` weight error. The kernel was correct; the fixture was not. The same first run silently under-sized the shared device buffers (8 tokens) while section C used 16, which surfaced as a launch failure rather than as data corruption. Both are fixture defects — record them so a future failure at this address is recognised as a test bug, not a kernel bug.

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

> **Gate result (Tier 1, item 14) — CERTIFIED. No defect found.** `reference/dsv4_oracle.hpp` gains the swizzled W4A16 format (`swizzled_shape` / `swizzled_address` / `swizzled_decode` / `swizzled_encode`), `clamped_swiglu` with a `ClampMode`, and `expert_ffn` / `expert_gate_up`; `tests/test_v4_expert_oracle.cpp`, **22 lines green**. The gate is split by *rule*, not by kernel:
>
> **A. The dequantization format.** W1 through `aeon_w4a16_swizzled_gemv_kernel` matches an independent scalar decoder to `3.3e-4` of peak. Both format rules are shown load-bearing by computing the wrong reading: the unsigned-magnitude read (no `−8` zero point) differs by `1.65`, and reading the nibbles in column order instead of through the permutation differs by `1.32`. Without those two lines a pass would only show that two implementations of the same rule agree.
>
> **B. The clamped SwiGLU rule.** The device matches the asymmetric clamp to `9.1e-5`. The symmetric reading (gate clamped both sides) differs by `4.5e-3` absolute, and removing the clamp by `1500` — so the clamp is not decorative. The **asymmetry itself** is stated as a measurement, not inferred from "outputs differ": `gate = −40` is *not* clamped (`silu(−40) ≈ −1.7e-16`) while `up = −40` *is* (`0`), and the symmetric rule gives `−4.54e-4` for the same input. The probe data hits all three edges (16 above `+10`, 32 `|up| > 10`, 16 below `−10`) so the test cannot pass vacuously.
>
> **`[corrected by mutation testing]` The asymmetry claim above was initially WRONG, and this gate printed PASS while a symmetrically-clamped kernel went undetected.** Every assertion in this section was either oracle-vs-oracle (the `ClampMode` fork) or a relative comparison whose floor was the probe peak of ~1600, while the whole asymmetric/symmetric difference is capped at `silu(−limit)·limit = 4.5e-3`. Two kernels were affected — this one and the fused W13 path. The gate now restricts comparison to the entries where the two rules actually disagree (`gate < −limit`) and requires the device to match the asymmetric oracle there by an absolute margin; measured separation after repair is `0.000000` vs `0.004540`, and injecting either mutation now fails the gate. The general rule is in "Mutation testing" above: **a gate that prints PASS is not evidence until a wrong kernel has been shown to fail it.**
>
> **C. The composed routed FFN.** Fused W13 + SwiGLU + W2 matches the fp64 oracle on hidden (`3.1e-4`) and on the 4096-wide output (`3.5e-4` moderate, `2.6e-4` at 8× activation).
>
> **D. Real artifact bytes.** `layers.3` expert 17: W1 matches the decoder to `2.9e-4` and the full FFN to `4.0e-4`, with 87% of W1 weights non-zero (a degenerate payload would pass every comparison trivially). This section is what makes A–C more than an internal consistency check — A–C build their payloads with an encoder written from the same format description as the decoder, so that pair is self-consistent by construction, and only artifact bytes prove the description matches what the converter wrote.
>
> **E. Accumulation order — Gate 14, partially settled by measurement.** The plan asks whether the intra-routed order matters. It is `atomicAdd`, so the order is the scheduler's, not the source's — and 32 identical runs of the 6-expert accumulation are **bit-identical** (`0 of 31` differ after fp16 rounding), and the 6-expert sum equals the weighted sum of six single-expert runs to `max_rel < 5e-7`. So on this silicon, at this expert count, the order is stable and is not a tolerance risk. **This does not close Gate 14**: it bounds one configuration, not the effect of the order in general, and says nothing about larger expert counts or contended scheduling.
>
> **Two gate-side bugs, recorded because both are easy to repeat.** (1) The first run compared the symmetric-clamp difference **relatively** and failed on an absolute difference of `4.5e-3` against a vector peak of `~1500` — a relative bound is the wrong instrument when the differing quantity is tiny and signed, the same lesson as the attention gate. (2) That same run asserted "the clamp is load-bearing" from `gate > +10` counts alone. The two rules **agree** on gate values above the limit; only a gate below `−10` separates them. The gate now asks the two questions separately and only asserts the asymmetry when the data reaches that edge — which, at 8× activation scale, it does (337 values below `−10`).
>
> **Not covered here:** the fused **atomic W2** path's behaviour under contention, and the ordered routed-expert path — both remain in the plan's kernel-quality open list.

> **Gate result (Tier 1, item 15) — CERTIFIED. No defect found.** `reference/dsv4_oracle.hpp` gains `dense_ffn` (deliberately a separate function from `expert_ffn`, not a shared helper — the two differ in weight format, in routing, and nothing else); `tests/test_v4_shared_expert_oracle.cpp`, **22 lines green**, and with it **Tier 1 is complete**.
>
> **A. Structure, against the artifact.** `n_shared_experts == 1`; all three tensors are `F16` at exactly `rows × cols × 2` bytes — unquantized, not one swizzled expert payload; the shapes are `[2048, 4096]` / `[4096, 2048]`; and the weights exist on **every** sampled layer (0, 2, 3, 10, 42), which is what "always fires" means structurally. No routing, no bias, no table.
>
> **B/C. Numerics.** All three projections and the composed FFN match the oracle on synthetic weights (`3.9e-4`) and on the artifact's own `layers.3` tensors (`2.1e-4` to `4.4e-4`), with 99% of `w1` non-zero. `dense_ffn` is pinned to a hand-computed closed-form value in the oracle self-check, so a wrong oracle cannot certify a wrong kernel.
>
> **The clamp on real weights — a measurement worth carrying forward.** The plan says the shared path takes the same `activation_clamp` as the routed path. It does, but **at the shared expert's nominal operating point the clamp is inert**: at unit-RMS activation (which is what this tensor consumes — it reads an RMSNorm output) the pre-activations peak at `4.77` on the gate and `4.71` on the up branch, both well inside `±10`, and removing the clamp changes the output by **exactly `0.0`**. Sweeping the activation to 8× does engage it (84 gate values above `+10`, 674 `|up| > 10`, 224 below `−10`, no-clamp delta `324`), and there the asymmetric edge is reachable (symmetric delta `4.5e-3`). So the clamp is *correct* on the shared path and *dormant* in normal operation — both facts now recorded, because "the clamp is engaged" was the first run's assertion and it was wrong. (The same lesson as item 14, which is why the scale sweep exists here.)
>
> **D. The combine, measured on the device.** Driving the accumulation kernel the way the pipeline does gives `combined == routed + shared` to `1.4e-4`, and the two one-sided failure modes are excluded: distance from `routed + 2·shared` is `0.49` and from `routed`-only is `3.45`, so the shared contribution enters **exactly once**. `[corrected — Tier 1]` The plan asserted that "our 2.10.4 note is consistent with" routed-then-shared. It is not, in fp order terms: the pipeline passes the shared output as **`initial_output`** to the atomic W2 accumulation, so shared is folded in as an addend of the accumulate rather than added to a completed sum `[V v4_pipeline.hpp:1199-1206]`. That mirrors the reference's **own fused path**, where `shared_l1_weights`/`shared_l2_weights` are passed *into* the MoE kernel `[V model.py:722-732]` rather than the unfused `final_hidden_states += shared_output` `[V model.py:1024-1031]`. The term set is identical; only the rounding order differs, and our choice is one the reference itself makes. The plan's "order is fixed" claim is therefore true of the unfused path and should not be read as a constraint on ours.
>
> **One more precision observation, not covered by this gate.** The deterministic accumulation path stores its accumulator in **fp16** and rounds on every one of the six accumulation steps `[V v4_pipeline_ops.hpp:47-58]`, while the atomic path accumulates in fp32 and rounds once. The deterministic path is therefore strictly *less* accurate, which is the opposite of what the option name suggests. Recorded with the routed-expert open items; the `deterministic_expert_accumulation_` flag is not exercised by any Tier-1 gate.

**2.10.5 — Combine**
- **Order is fixed: routed sum first, shared expert added after.** The reference computes `final_hidden_states = experts(...)` and then `final_hidden_states += shared_output` `[V nvidia/model.py:1020-1031]` — i.e. `out = Σ_k weight_k · down_k`, then `out += shared`. `[corrected — Tier 1]` This describes the **unfused** path only. The reference also has a fused path that passes the shared expert's L1/L2 weights *into* the MoE kernel `[V model.py:722-732]`, and ours mirrors that one (shared is the `initial_output` of the atomic accumulation). The term set is identical; only the fp rounding order differs. Do not read "order is fixed" as a constraint on our implementation.
- **`[?]` The intra-routed accumulation order** (top-k slot order vs expert-id order) is still a measurement item: a different order changes fp rounding and can exceed tolerance. **Gate 14.**

#### 2.11 — Hyper-Connections FFN post-mix

- `res_out[j,h] = post_f[j]·moe_out[h] + Σ_i comb_f[i,j]·res_mid[i,h]`; this becomes the next layer's `res_in` `[V]`.
- **Gate:** `res_out` matches reference; state still 4 × 4096.

#### Note on Layers 0 and 1 `[corrected]`

Layers 0 and 1 have **no compressor and no indexer** (`compress_ratios[0]=compress_ratios[1]=0`) `[V config; V artifact]`. They attend the local window only (2.4.4).

`[corrected]` They **do** run Hyper-Connections, including the HC Sinkhorn. The first revision said "no Sinkhorn step" on layers 0–1 — that was a consequence of placing Sinkhorn in the attention block; with Sinkhorn restored to HC (2.0), it applies to all 43 layers.

**Gate:** assert the branch explicitly — a Sliding layer must never read compressor/indexer tensors.

### Step 3 — HC Head Reduction `[corrected — was missing]`

- The final state is 4 streams. Reduce to a single 4096 vector:
  `mixes = hc_head_fn @ rmsnorm_without_weight(flatten(x), rms_eps)`, then `pre[j] = sigmoid(mixes[j]·hc_head_scale + hc_head_base[j]) + hc_eps`, then `out[h] = Σ_j pre[j]·x[j,h]` `[V vllm kernels/mhc/triton.py hc_head_reduce_triton_kernel; V contract hc_head_fn F32 [4,16384], hc_head_base F32 [4], hc_head_scale F32 [1]]`.
- **Ensure:** the head RMSNorm has **no learned weight** (unlike 2.1); `hc_head_scale` is a scalar `[1]` broadcasting over the 4 gates; `hc_eps` is added after the sigmoid.
- **Gate:** matches reference; output is 4096.

### Step 4 — Final RMSNorm + LM Head

- Final RMSNorm over 4096 with a learned weight `[V contract norm.weight F16 [4096]]`.
- `logits = head_weight @ hidden` → 129280; fp32 accumulate `[V contract head.weight F16 [129280,4096]]`.
- **Ensure the head is a separate matrix, not tied to the embedding.** `tie_word_embeddings = False`, and both `embed.weight` and `head.weight` exist as distinct `[129280, 4096]` F16 tensors in our checkpoint `[V config; V checkpoint]`.
- **Gate:** logits match reference; top-1 token matches.

### Step 5 — Sampling

- Apply temperature, top-k, top-p as configured. **Defaults from our artifact:** `do_sample = true`, `temperature = 1.0`, `top_p = 1.0` `[V generation_config.json]` — at these values sampling is *untruncated*, so a first implementation may legitimately start at argmax, but the real defaults are `T=1, top_p=1`.
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

#### 7.1 — Long-context lifecycle: ring reuse, capacity, and why the two are not the same question `[Tier 3 item 20, specified]`

All four cached pieces (Part I §6.1) are **rings**, but their lifecycle has two regimes that must not be confused. One reuses continuously and correctly; the other must never reuse at all, and nothing inside it can say so.

**(a) The local ring reuses continually, and that is not an error condition.** `C = sliding_window = 128`. Slot `s ∈ [0, C)` holds the most recent position `p ≤ pos` with `p ≡ s (mod C)`, and the position is stored **alongside** the row (`d_local_positions[s]`), so the kernel reads the *position*, never the write order. A query at `pos` attends exactly `[max(0, pos − C + 1), pos]` `[V cache_utils.py:892-894 swa_start = max(pos − (WINDOW_SIZE−1) − left_add, 0); swa_len = pos + right − swa_start + 1; V 2.4.4]`, and the kernel filters on the recorded positions, so a slot whose position fails the window test is masked regardless of how stale its bytes are. Invalid slots are seeded with a position that cannot pass that test (`-1`, i.e. all-ones) `[V v4_layer.hpp reset_generation_state; V cache_utils.py:1425 `slot_ids = tl.where(offset < swa_len, slot_ids, -1)` — the same convention]`. Reuse past the window is therefore *the definition* of how much local context exists; the long-range memory lives in the compressed entries.

**(b) The compressed store must never reuse, and its wrap is undetectable from its contents.** Capacity `K = ceil(max_seq / ratio)`, which is **exactly** the number of entries the declared context can produce: at `pos = max_seq − 1` the count is `(pos+1)/ratio = max_seq/ratio ≤ K`, with equality whenever `ratio | max_seq`. Entry `i` (emitted only on a boundary, `i = (pos+1)/ratio − 1`) lives in slot `i mod K` and records its own position `(i+1)·ratio − 1` `[V fused_compress_quant_cache.py:201-202 boundary; V 2.3 compressed RoPE position]`. The count a query sees is `min(K, (pos+1)/ratio)` — the number of *populated slots*, which is the reference's own count formula `[V sparse_mla.py:404-405 `num_compressed = (position + 1) // compress_ratio`, then `tl.minimum(num_compressed, max_compressed_tokens)`]`.

> **If the ring ever wrapped, the row-set would silently become a sliding window over compressed entries — the *newest* `K` instead of every committed entry — and nothing in the arithmetic would look wrong.** That is why the capacity is a derived invariant rather than a tuning choice: the reference sizes `c128a_max_compressed` from `max_model_len` and asserts `active_topk_width ≥ max_seq_len // ratio` `[V sparse_mla.py:158-170, 259]`, so its own count clamp is unreachable in range.
>
> **Our engine enforces the same invariant by refusing**: `V4Layer::record_position` throws for `position ≥ max_seq_len` `[V v4_layer.hpp:211-216]`. The refusal is load-bearing rather than decorative, because **the wrap cannot be detected from the entries themselves** — every populated slot records a position `≤ pos`, so no `compressed_positions[i] ≤ current_position` guard can tell a live row from a slot whose older occupant was overwritten. Do not replace the throw with a clamp or a modulo, and note that a clamp would be *worse* than the ring: the reference's `tl.minimum` truncates to the **oldest** `K` entries while a ring keeps the newest. **Trap 40.**

**(c) The compressor partial ring is bounded by the window, not the context, so it can never collide with (b).** `coefficient · ratio` slots (8 at ratio 4, 128 at ratio 128), with the current token at `pos mod capacity`, and the materializer reading the window ending at `pos` (2.4.2) `[V fused_compress_quant_cache.py:216; V c128_cleanup.py:27-28 `slot = (seq_len + draft_offset) % ring_size`, and note it clears with kv `0.0` / score `-inf` — the masking convention for an invalid slot]`.

**(d) The two regimes are independent in both directions.** Materializing a compressed entry must not touch the local ring, and reusing a local slot must not touch the compressed store. A shared buffer or an off-by-one slot would otherwise surface only as a long-context quality loss, never as a failure.

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

> **Finding about the pre-rewrite implementation (checkable, and it was right about the gate):** the old `prefill_batched` batched only the dense projections and HC; attention, compressor, indexer, and the MoE ran in a per-token loop, and the indexer top-k did a device→host copy plus a stream synchronize per CSA layer. As written, the equivalence gate above could not pass. The rewrite's answer is item 19, below — and it turned out there was a second, sharper reason the gate could not pass, which no amount of batching the *arithmetic* would have fixed.

> **Gate result (Tier 3, item 19) — CERTIFIED, and the gate found the ordering the naive implementation gets wrong.**
> `tests/test_v4_layer_body_chunk_oracle.cpp`; the default suite is **34 tests**. The chunked path is
> `core/v4_layer_body_batch.hpp`; the body it drives is the *same* `run_layer_body_pre_attention` /
> `run_layer_body_attention_tail` pair the decode path calls, which is what "one body, not two" now
> literally means.
>
> **THE FINDING — writing the chunk's keys into the ring first is not equivalent to serial, at any
> chunk length above one (trap 39).** The obvious chunking is: run every token's pre-attention half,
> then every token's attention half. It is wrong, and wrong for a reason that has nothing to do with
> the compressed path that the plan's original finding was about.
>
> The local ring has `C` slots and position `p` lives in slot `p mod C`. Query `q` attends
> `[q − C + 1, q]`. In a chunk spanning `[S, E]` with `E > S`, the write for `p ∈ [S, E]` lands in slot
> `p mod C`, which before the write held `p − C`. Take `p = E`, the chunk's last write, and `q = S`, its
> first query: `E − C ≥ S − C + 1` whenever `E ≥ S + 1`. So **the chunk's last write evicts the oldest
> key of its own first query's window** — and every query below `E` loses the keys in
> `[q − C + 1, E − C]`. The measurement that proves it is cheap: with a 10-slot ring and a 5-token
> chunk, the naive order moves `attn_proj`, `moe_out` and the residual stream. `E = S` (a chunk of
> one) is the only case where the eviction is not inside the chunk, which is exactly why **"serial"
> in `chunk ≡ serial` must mean the same tokens one at a time** — the property under test is that
> batching changes nothing, and a chunk of one has no batching in it.
>
> **The fix is the canonical design, and the reference already has this shape.** A token's key goes to
> a per-chunk key buffer (the body takes `d_local_key_write` for this); each query attends a
> **composed row-set** — the pre-chunk ring rows in its window plus the chunk's own rows up to and
> including itself — and the chunk's keys are committed to the ring only once every query has run.
> That is what the reference does: *"the current chunk's freshly-computed K lives in a separate
> per-forward `kv [total_tokens, D]` **not yet written to the SWA ring**"*, with the row-set assembled
> as `[compressed | swa positional]` (2.4.4). The plan's Part III requirement — "the compressor/indexer
> run over the chunk and their ring boundaries must be respected mid-chunk" — was about the compressor;
> the *local ring* had the same constraint and it was not written down.
>
> **The composed rows are ordered by ring slot, not by position, and that is what makes the gate an
> equality.** The decode path hands the attention kernel the ring itself and lets it iterate slots
> `0 … C-1`, so the order it sums the window in is slot order — for a wrapped window, a *rotation* of
> position order. Composing in the same order makes both paths accumulate the identical `exp` terms in
> the identical sequence over bit-identical keys, so the comparison can be exact and the gate does not
> need a tolerance at all.
>
> **Measured: zero differing values anywhere.** Three classes × three chunk schedules × 130 tokens,
> comparing the residual stream, the router logits, the selected ids and the routing weights per token,
> plus the whole final state — ring keys and positions, every committed compressed entry with its
> position, and the compressor's partial ring. Every line reads `bit-identical`, and the gate reports
> a single total: **0**. The prompt is 130 tokens, so **HCA (ratio 128) commits a real compressed
> entry** at position 127 and **CSA commits 32**, with boundaries falling mid-chunk in every schedule
> and the 10-slot window wrapping thirteen times.
>
> **The three comparisons, and what each one catches.** (i) `chunk {5,5,3}` etc. vs the same tokens one
> at a time — batching changes nothing. (ii) `{6,6,1}` and `{4,4,4,1}` against `{1,…}` — the answer is
> schedule-independent, with `{4,4,4,1}` putting a chunk exactly on CSA's ratio so a boundary lands on
> a chunk edge and `{6,6,1}` using the largest chunk the window allows. (iii) `chunk path at count 1`
> vs **`run_layer_body_decoding`** — this is what keeps (i) and (ii) from being circular: the
> one-at-a-time chunk run is tied to the **Tier-2 certified decode body**, so the equality is between
> the batch path and the certified path, not between two copies of the same new code.
>
> **Section B asserts the ordering directly rather than only its consequence.** After every token's
> pre-attention half — no query run, nothing committed — the ring's keys and positions are compared
> against a snapshot taken before the chunk and must be **unchanged**, while the chunk buffer is shown
> to be non-zero. Mutation M19-2 (the false version: write the ring as the chunk goes) is precisely
> this and fails here.
>
> **What is NOT covered, named so it is not mistaken for coverage.** **Throughput.** The plan says
> structural equivalence and speed are separate gates, and this composition is a per-query loop of
> ~C device-to-device copies; correct, and not fast. Turning it into one gather kernel is its own step
> with its own measurement. **The indexer's host round-trip** — `select_indexer_topk` still copies the
> candidate scores to the host and synchronizes once per CSA token, which the plan forbids in a
> prefill. It changes no value, so this gate cannot see it by construction; it is now recorded in the
> open-unknowns table as item 19's own remainder rather than left implied by "item 19 done". And the
> usual: synthetic experts, a shrunk window (10) and `index_topk` (3), no tiering.


### Decode

- One token at a time; attention is `[1, cache]`.
- **Bandwidth-bound** — expert streaming dominates.
- The indexer selects which compressed blocks to attend to.

**Design consequence:** both regimes must share one layer body; only the *scheduling* differs. Prefill wants large batched matmuls; decode wants efficient expert streaming and minimal per-token overhead. The expert fetch path (2.10.1) is the critical path in decode.

---

## Part IV — Kernel Inventory

| # | Kernel | Precision | Hardware path | Notes |
|---|---|---|---|---|
| 1 | Embedding gather + HC broadcast | fp16/bf16 | memory | Output is 4 × 4096 (embedding expanded to `hc_mult` streams) |
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
| 13 | Indexer | **fp8 (E4M3/UE8M0)** | WMMA (dequant to fp16) | `[corrected]` not INT8; **`[V]` ReLU required, per-head, before weighting**; **CSA (ratio-4) layers only — HCA has no indexer** |
| 14 | Top-k selection | int32 | sort/select | Must match reference exactly; must be on-device for prefill |
| 15 | Grouped output projection | fp16/bf16 WMMA | `v_wmma_f32_16x16x16_f16` | `[corrected]` 8 groups × 1024 → `wo_b`; **`[Tier 1]` per-group reduction is load-bearing** |
| 16 | Router | fp16 logits → fp32 | small matmul | `sqrt(softplus)`; bias (`ffn.gate.bias`) added to **scores**; hash branch for layers < 3; flat top-6, no groups; **`[Tier 1]` all four traps shown load-bearing** |
| 17 | Expert fetch | — | memory | Async, overlapped |
| 18 | Dequantization | INT4 → fp16/bf16 | registers | Fused with matmul; signed −8 bias; **`[Tier 1]` zero point and nibble permutation both shown load-bearing** |
| 19 | Expert matmul | fp16/bf16 WMMA | `v_wmma_f32_16x16x16_f16` | Fused with dequant; **clamped SwiGLU (asymmetric)** |
| 20 | Shared expert | fp16/bf16 WMMA | `v_wmma_f32_16x16x16_f16` | Not quantized; same clamp; **`[Tier 1]` clamp is dormant at nominal activation scale** |
| 21 | LM head | fp16/bf16 WMMA | `v_wmma_f32_16x16x16_f16` | fp32 accumulate; **separate matrix, not tied to embedding** |
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
5. ~~**RMSNorm** — vs fp64 reference.~~ **DONE.** `reference/dsv4_oracle.hpp` (host-only, fp64, kernel-free) + `kernels/v4_norm.hpp` + `tests/test_v4_norm_oracle.cpp`. Weighted and unit forms both pass; max relative error 4.9e-4 = one fp16 ulp, i.e. the kernel is exact and only the fp16 store rounds. The gate also asserts the oracle against a closed-form host-only case, so a wrong oracle cannot certify a wrong kernel.
6. ~~**RoPE forward and inverse** — separately; two bases; forward∘inverse = identity.~~ **DONE.** `kernels/v4_rope.hpp` + `tests/test_v4_rope_oracle.cpp`, 18 lines green. Both bases certified (theta 10000 plain, 160000 YaRN factor 16); the gate asserts the two bases are distinguishable, that `[0,448)` is bit-identical after rotation (trap 27), and that forward∘inverse returns the input. See the gate result under 2.3 for the fp32 table-precision bound.
7. ~~**MLA q path and kv path** — including both intermediate norms.~~ **DONE.** `kernels/v4_gemv.hpp` + `tests/test_v4_mla_oracle.cpp`, 9 lines green at full dimensions. Composition gate: replays the pipeline's kernel order and compares `q_lora`, `q_lora_norm`, `q` (after the per-head norm), `kv` (before and after its norm), plus the three discriminating checks described under 2.2. Also settles that the per-head Q norm is weightless.
8. ~~**HC project + Sinkhorn** — verify doubly-stochastic convergence *before* composing it with anything.~~ **DONE.** `tests/test_v4_hc_oracle.cpp`, 14 lines green. Verified doubly stochastic (to `eps`, see 2.0), the 20-iteration count, the third `hc_scale` entry, the asymmetric eps placement, and — the finding that matters — the **comb index convention**, where the kernel matches the upstream einsum to `4.0e-4` while the transposed reading differs by `0.52`. A plan-prose error was found and corrected; see trap 34.
9. ~~**Attention score + sink + softmax** — verify at pos 0, within window, beyond window.~~ **DONE.** `tests/test_v4_attention_sink_oracle.cpp`, 19 lines green at pos 0 / 5 / 127 / 149. Asserts the exact window boundary (an out-of-window key has **exactly zero** influence), the sink as a pure scalar rescale, and the full-head scale against both wrong candidates. Trap 35 records the one property that cannot be tested from the output.
10. ~~**Compressor** — verify pooling and boundary firing.~~ **DONE.** `tests/test_v4_compressor_oracle.cpp`, 32 lines green at both ratio classes (4 and 128), covering both kernels. Verified: APE is a `score`-only term (kv bit-exact), the APE row is `position % ratio`, the per-dimension softmax, both overlap segments, the RoPE position, the truncated first entry, and the `blockDim >= head_dim` launch contract.
11. ~~**Indexer + top-k** — exact index match. **Verifies the confirmed ReLU and fp8/UE8M0 quantization, and settles the Hadamard choice.**~~ **DONE.** `tests/test_v4_indexer_oracle.cpp`, 11 lines green; top-k matches exactly. **The gate found the ReLU missing from `v4_indexer_scores_kernel`** (`max_rel 0.98`, 65 of 512 wrong indices) and it is now fixed — trap 11. **Gate 11 is settled by measurement: do not apply the Hadamard** (two-sided is a no-op to `3.6e-16`, one-sided shifts scores by `1.46` and 180 of 512 indices). The indexer-K fp8 + per-key scale delta is deferred to the KV-precision gates.
12. ~~**Grouped output projection** — low-rank and final.~~ **DONE.** `tests/test_v4_grouped_wo_oracle.cpp`, 14 lines green. `z` matches the einsum to `2.9e-4`, `attn_proj` to `4.2e-4` over the chain. The gate asserts the **per-group structure** rather than only closeness: the interleaved weight reading differs by `1.38`, perturbing group 0 leaves groups 1–7 bit-identical while moving all of group 0, and the group-0 reduction is shown to reach its last head. See the gate result under 2.5.
13. ~~**Router** — exact ids and weights, one hash layer and one top-k layer. **Verifies `tid2eid == [vocab,6]` against our artifact.**~~ **DONE.** `tests/test_v4_router_oracle.cpp`, 23 lines green. Ids match positionally and weights to `1.3e-7`. Every trap in 2.9 is shown load-bearing: bias-before-softplus routes 3 of 8 tokens differently, dropping the bias changes 8 of 8, the grouped variant picks a different set, and ties go to the lowest index. `tid2eid` is `I64 [129280, 6]` on layers 0–2 with `ffn.gate.bias` on 3–42, verified against the artifact and replayed at real token ids. **No defect found.** Corrected the plan's claim about our normalization guard (it is `Σ+1e-20`, not `Σ>0?Σ:1`).
14. ~~**Expert matmul with fused dequant and clamped SwiGLU** — vs independent decoder.~~ **DONE.** `tests/test_v4_expert_oracle.cpp`, 22 lines green. Format, clamp rule and composed FFN all certified, then re-checked on a real `layers.3` expert payload. The signed zero point and the nibble permutation are each shown load-bearing (`1.65` and `1.32`); the clamp asymmetry is proven by a constructed probe (`gate = −40` unclamped, `up = −40` clamped). **No defect found.** Partially settles Gate 14 by measurement: 32 identical 6-expert atomic accumulations are bit-identical.
15. ~~**Shared expert** — separately.~~ **DONE.** `tests/test_v4_shared_expert_oracle.cpp`, 22 lines green. Structure checked against the artifact (F16, exact byte counts, present on every sampled layer), numerics on synthetic and real `layers.3` tensors (`2.1e-4` to `4.4e-4`), and the combine measured on the device to apply shared exactly once. **No defect found.** Two findings recorded: the clamp is **inert at the shared expert's nominal activation scale** (peaks `4.77`/`4.71` against a limit of `10`, no-clamp delta `0.0`) and engages only when scaled to 8×; and the pipeline folds shared into the accumulation as `initial_output`, which mirrors the reference's fused path rather than its unfused `+=` order. **Tier 1 is complete.**

**Tier 2 — Composition.**
16. ~~**One full layer, Sliding class.** Verify `res_out` for a single token.~~ **DONE.** `core/v4_layer_body.hpp` (the layer body, Steps 2.0–2.11, one token) + `reference/dsv4_oracle.hpp::layer_sliding_body` (the composed fp64 reference) + `tests/test_v4_layer_body_oracle.cpp`. Device `res_out` and every named intermediate match the oracle on the artifact's real `layers.0` weights, for ten positions (the local ring is shrunk to 6 so the wrap is exercised). Every checkpoint is within **`1.1e-3` of its own peak**, which is the fp16 store and nothing else. Three mutations killed, one equivalent; and the gate found **a defect in the oracle itself** — a fp16 tensor widened with `std::vector<double>(uint16_t*, …)`, which converts the bit pattern arithmetically (`0x3C00` → `15360`) instead of decoding it. See the gate result below. **Next: item 17.**
17. ~~**Full layer, CSA class.** Then **HCA class**.~~ **DONE.** One body, all three classes. `reference/dsv4_oracle.hpp::layer_body` (generalized from `layer_sliding_body`, so the oracle mirrors the device's single structural branch) + `tests/test_v4_layer_body_compressed_oracle.cpp`, driving the artifact's real `layers.2` (CSA, ratio 4) and `layers.3` (HCA, ratio 128) weights across a sequence that crosses ratio boundaries. Every named intermediate matches — the compressor projections, the **APE-adjusted** partial ring row, the materialized compressed entry, the **indexer** scores and its top-k, the **row-set counts**, `attn_proj`, `ffn_norm`, `moe_out` and `res_out` — and the row-set rule itself (**trap 33**) is asserted directly: CSA reads the indexer-selected rows, HCA reads **every** committed compressed row and never touches the indexer. Five mutations, **all five killed**. See the gate result below. The shared gate scaffolding was extracted to `tests/support/v4_layer_body_gate.hpp`.
18. ~~**Serial multi-token decode.** Verify state evolution across compressor boundaries.~~ **DONE.** `tests/test_v4_layer_body_serial_oracle.cpp` drives a three-layer stack — Sliding (layer 0), CSA (layer 2), HCA (layer 3) — for **136 tokens with the residual carried by the device itself**, across 34 CSA boundaries and one HCA boundary. It removes the simplification both earlier tiers made: nothing writes `d_res_in` from the oracle, and the reference is driven by the device's own residual trajectory *and* its own discrete expert selection. The whole accumulated state — every local-ring slot, every committed compressed entry, and every position — is compared at the end, and the boundary's entry is shown to survive bit-identical. **A finding items 16/17 structurally could not reach:** the device's serial decode is **not bit-reproducible** (the default MoE `atomicAdd` order is undefined, and a router near-tie amplifies the drift), which is recorded as trap 38 and matters directly for the byte-exact prefix-reuse gate. See the gate result below.

> **Gate result (Tier 2, item 16) — CERTIFIED, and the gate found a defect in its own oracle.**
> `tests/test_v4_layer_body_oracle.cpp`, **31 lines green**; the default suite is **31 tests**.
>
> **What was built, and why it is structured this way.** The layer body is a new module
> (`core/v4_layer_body.hpp`) rather than a copy of an existing loop. It is not wired into
> `core/v4_pipeline.hpp`: that file is the **pre-rewrite** graph, gated off behind
> `AEON_ENABLE_LEGACY_V4_GRAPH` and slated for deletion, so the rewrite must not acquire a
> dependency on it. The body instead takes its model-level inputs (the two RoPE bases), an
> **observer** (the attention trace) and a **routed-expert executor** (the whole tiered
> supply system: index lookup, Hot/Warm/Cold promotion, prefetch, leases, staging) as
> parameters. One body is then shared by decode, batched prefill and this gate, which is
> what Part III requires — and the seam is what makes the gate possible at all, because it
> lets the gate supply experts directly instead of standing up the storage system.
>
> **A: the oracle is pinned to a closed form.** `hc_post` with an identity comb and unit
> post-mix reproduces `res[j][h] = res_in[j][h] + layer_out[h]` exactly (`0.0` differ), and
> an asymmetric comb is shown to separate the contraction/output reading by `0.25`.
>
> **B: the composition, on real weights.** Ten tokens at positions 0–9 through one
> `layers.0` Sliding layer, comparing `x_pre`, `x_norm`, `q_rot`, `kv_rot`, `attn_proj`,
> `ffn_norm`, `moe_out` and `res_out`, plus exact equality of the six routed ids. Measured
> worst case per checkpoint, as a fraction of that checkpoint's own peak:
>
> | checkpoint | worst error (× peak) |
> | :--- | ---: |
> | `x_pre` (HC attention pre-combine) | `4.9e-4` |
> | `x_norm` (attention RMSNorm) | `8.9e-4` |
> | `q_rot` (MLA + per-head norm + RoPE) | `2.0e-2` → `2.6e-3` of peak |
> | `kv_rot` (single shared K=V row) | `1.6e-2` → `2.0e-3` of peak |
> | `attn_proj` (attention + inverse RoPE + grouped wo) | `6.2e-2` → `2.3e-3` of peak |
> | `ffn_norm` | `7.4e-2` → `1.1e-3` of peak |
> | `moe_out` (routed + shared) | `5.0e-2` → `1.0e-3` of peak |
> | `res_out` (the layer output) | `3.2e-2` → `1.1e-3` of peak |
>
> The tolerance is `3e-3` of peak, i.e. ~3× the measured floor. **The comparison basis is a
> deliberate choice and it is not `max_rel`.** The chain stores every stage in fp16, so the
> honest question is "how far off, as a fraction of this tensor's scale". `max_rel` is
> dominated by elements sitting on its floored denominator on a tensor whose values span
> three decades, which is why the raw `max_rel` column reads `2e-2` while the same data is
> `2.6e-3` of peak. This is the third time in the rewrite that a relative tolerance was the
> wrong instrument; prefer peak-relative whenever the quantity spans decades.
>
> **The routed ids are exact, by construction.** On a hash layer the ids come from the
> artifact's own `tid2eid`, in table-column order, so a score-sorted or bias-applied router
> cannot reproduce them.
>
> **C: the composition is load-bearing.** Four properties are shown to move the layer output
> before any pass is trusted: the attention RMSNorm (`25%` of the q-lora peak when skipped —
> measured at `q_lora`, not at `q`, because the per-head norm re-normalises and would damp
> the difference to `2.7%`, too weak to catch a missing norm); the attention path itself
> (`35%` of peak when zeroed); the shared expert's presence in the combine (item 15's
> property, re-observed at layer scale); and the grouped projection's group-major layout
> (`1.92` when read interleaved).
>
> **What is NOT covered, named so it is not mistaken for coverage.** The compressed classes
> (item 17); the real 128-token local-window rollover (the ring is shrunk to 6 so the wrap
> is reachable in ten tokens — a faithful mini-model, since the window length is a launch
> parameter, but not the model's own window); and any tiering (the gate supplies experts
> directly).
>
> **A defect the gate found in its own oracle, and how.** The oracle read every fp16 weight
> through `std::vector<double>(fp16_bits, fp16_bits + n)` — which **converts** each
> `uint16_t` arithmetically, so the `0.045` norm weight became `15360` (its bit pattern
> `0x3C00` read as a number) and the layer's checkpoints came out at `3.5e4` where the device
> said `0.096`. `half_bits_to_doubles` now does the decode, and a **fixture check** was added
> that compares every host-side weight pointer against the device's uploaded copy — that is
> the line that would have caught it in one run instead of three. The lesson is narrower than
> "check your inputs": the failure was legible *only* because the report prints each
> checkpoint's peak. An absolute or relative error alone could not distinguish "the oracle is
> 3.5e4 out" from "the device is broken".

> **Gate result (Tier 2, item 17) — CERTIFIED, both compressed classes, and the class
distinction is the thing that was actually tested.**
> `tests/test_v4_layer_body_compressed_oracle.cpp`; the default suite is **32 tests**.
>
> **What was built.** The item-16 oracle was generalized rather than duplicated:
> `layer_sliding_body` became `layer_body`, whose only structural branch is the attention
> class — the same shape as `core/v4_layer_body.hpp`, so oracle and device branch in the same
> place. The compressor (`CompressorRing`, the two-segment window, the APE, the materializer),
> the indexer (query from `q_lora_norm`, head weights, a second compressor, the score path,
> the two-branch top-k), and the merged local+compressed row-set under one sink softmax were
> added. The scaffolding the two Tier-2 gates share moved to
> `tests/support/v4_layer_body_gate.hpp`.
>
> **CSA (layer 2) and HCA (layer 3), on the artifact's real weights.** CSA runs 20 tokens at
> ratio 4 with the local window shrunk to 4 and `index_topk` shrunk to 3, so the selection is
> **non-degenerate** (more candidates than slots) inside a short run; HCA runs **257** tokens
> at ratio 128, which is the minimum that commits two compressed entries. Worst case per
> checkpoint, as a fraction of that checkpoint's own peak:
>
> | checkpoint | CSA worst | HCA worst |
> | :--- | ---: | ---: |
> | `x_norm` (attention RMSNorm) | `9.4e-4` | `6.2e-4` |
> | `q_rot` (MLA + per-head norm + RoPE, **compressed base**) | `1.2e-3` | `6.6e-4` |
> | `kv_rot` (single shared K=V row) | `1.3e-3` | `5.2e-4` |
> | `compressor_kv` / `compressor_score` (raw projections) | `7.4e-4` | `4.7e-4` |
> | `partial_kv` / `partial_score` (**APE-adjusted** ring row) | `7.8e-4` | `5.0e-4` |
> | `compressed entry` (materialized row) | `4.3e-4` | `6.0e-4` |
> | `indexer scores` | `2.4e-3` (abs floor `5e-4`) | — (no indexer) |
> | `attn_proj` (attention + inverse RoPE + grouped wo) | `1.9e-3` | `1.2e-3` |
> | `ffn_norm` | `1.0e-3` | `8.2e-4` |
> | `moe_out` (routed + shared) | `2.3e-3` | `1.1e-3` |
> | `res_out` (the layer output) | `8.0e-4` | `5.2e-4` |
> | `router logits` | `5.9e-4` | `5.6e-4` |
>
> Tolerances are `3e-3` of peak (`4e-3` post-MoE), i.e. ~2–4× the measured floor, which is the
> fp16 store and nothing else. Every one of them is the same peak-relative basis item 16
> established, for the same reason.
>
> **A: the oracle's compressed rules are pinned closed-form.** The ring adds the APE to
> `score` and to **nothing else** — `kv` is compared against a plain widening of the input and
> matches **exactly** (`0.0`), which is the bit-exact form of "score only". The APE row index is
> shown to be a genuine modulo (`p` and `p+ratio` identical, `p` and `p+1` differing by `1.0`).
> And the fp16-decode fast path is checked against the `ldexp` definition it replaced over the
> full 16-bit space: **bit-exact, `0.0`** (see the performance note below).
>
> **B: the row-set rule, which IS trap 33, is asserted rather than assumed.** For each token the
> gate checks that a boundary fires on exactly `(pos+1) % ratio == 0` and matches the device's
> own committed count; that the local row count is `min(pos+1, window)`; and that the
> **compressed row count** equals the class's rule — `min(committed, index_topk)` for CSA,
> `committed` for HCA. On CSA, the indexer's returned top-k is compared **element-by-element**
> against `select_indexer_topk`'s output, which is itself a re-derivation of the device's own
> two-branch rule (ascending when the candidates fit, score-sorted otherwise) in the same order
> the device writes it.
>
> **C: the class distinction is shown load-bearing, as the thing the comparison is for.**
> Attention is recomputed by the oracle on the run's final state with the row-set altered:
> dropping the compressed rows entirely moves `attn_out` by `104%` of peak (CSA) / `44%` (HCA),
> so the compressed branch is genuinely in the row-set; **HCA's "every row"** is probed by
> dropping the *oldest* committed entry, which moves it by `16%` — a row a top-k would be free
> to skip; and **CSA's selection** is probed by swapping one selected entry for a candidate the
> indexer rejected, which moves it by `69%`. These three are what separate the classes, and
> each is visible in the output rather than argued.
>
> **A real finding, recorded rather than hidden: `router logits` disagree with the oracle at
> ~`1.2e-3`, which is 2–3× every other checkpoint's error and enough to flip a near-tie.** The
> gate began by requiring the six routed ids to match the oracle's, and **8 of 257 HCA tokens
> failed** — with their weights matching to `8e-3`, the signature of a near-tie resolved
differently rather than a routing bug. The cause is precision, not semantics: the oracle's
> fm64 GEMV and the device's fp16 GEMV legitimately differ at ~1e-4 on the logits — and the
> selection is `score + bias` over 256 experts, where the 6th and 7th experts are frequently
> within that gap. Putting the **fp64** derived ids back made 5 tokens fail with a `gap` of
> `1.6–1.8e-4`. The gate now compares the **logits** on the peak-relative basis (where the
> precision belongs) and compares the **selection rule** against the ids re-derived from the
> **device's own logits** — with a `1e-6` tie tolerance for the fp32-vs-fp64 `softplus_sqrt`.
> That is a test of the *rule* (bias after softplus, flat top-6, ties to the lower index), which
> is what a gate can honestly assert; the logits' precision is asserted on its own line. **This
> is the same lesson as trap 36 in a new guise: pick the comparison whose instrument matches the
> quantity.**
>
> **Five mutations, all killed.** M17-1 (CSA ignores the indexer and reads every compressed
> row — **trap 33 exactly**) killed with 27 red lines; M17-2 (HCA reads only the newest
> compressed row) killed with 81; M17-3 (compressed layers rotated with the sliding base) killed
> with 1021; M17-4 (compressor partial state saved at the wrong position, so the APE row and the
> ring slot are both wrong) killed with 577, `partial_score`/`partial_kv` at **100%** of peak;
> M17-5 (indexer top-k ordered ascending) killed with 32. **M17-1 is the important one**: the
> one property that separates CSA from HCA is now demonstrably visible to the gate.
>
> **A performance note that is also a correctness note.** The first version of this gate spent
> essentially all its time inside the oracle's `half_bits_to_double`, which called `ldexp` with a
> runtime exponent ~180M times per token (~3 s/token, and the 257-token HCA run could not
> finish). `half_bits_to_double` now assembles the double's bit pattern (`f64(e-15+1023, m<<42)`) —
> provably the same value, no rounding — and is checked **bit-exact against `ldexp` over the full
> 16-bit range** by the gate rather than trusted; plus the six fixed routed payloads are decoded
> once through a new `decode_expert_weights` seam instead of per token. The result is
> **3 s → 0.6 s per token** (the remaining cost is the genuine 1.15 GFLOP/token of fp64 matvec)
> and a **2m14s** gate. The `expert_ffn(payload, …)` entry point is unchanged, so item 16 still
> exercises the swizzle walk on the artifact's real experts; only the multi-token gate opts into
> the decoded cache.
>
> **Parallelized (2026-09-16) — the gate is now `54 s`.** The oracle's two fp64 reductions
> (`matvec`'s `o` axis and `grouped_wo_a`'s flattened `(t,g,r)` axis) are latency-bound dependent-add
> chains, so one core reaches only ~2 GFLOP/s no matter the SIMD width; spreading the outer axis over
> cores is where the time is, at no numerical cost (`acc` is a per-element local, `y[o]` has a single
> writer, the accessor and `x` are read-only, so every element executes the identical instruction
> sequence it did serially). Measured in isolation, the oracle's own op mix goes `119 s → 5.9 s` on
> 32 threads. **The gate only reaches `137 s → 54 s`, and that gap is the finding worth keeping:**
> ~48 s of what remains does not respond to host threads at all, and `OMP_WAIT_POLICY=passive`
> removes the CPU burn (3019% → 471%) without moving the wall clock — so the floor is the gate's own
> device half (per-token kernel launches and synchronised checkpoint readbacks), not the oracle.
> Oracle parallelization is exhausted here; further gains would need the test to stop driving the
> device one token and one readback at a time. Every reported checkpoint value is **byte-identical**
> between a 1-thread and a 64-thread run. See `aeon_enable_openmp` and `AEON_OPENMP_THREADS`.
>
> **What is NOT covered, named so it is not mistaken for coverage.** Serial state evolution
> **without** re-seeding — the residual is still written from the oracle between steps, so this
> measures one layer's composition and not a long loop's drift (item 18); the real 128-token
> local window (shrunk to 4 so the wrap is reachable in a few tokens — a faithful mini-model,
> since the window length is a launch parameter, but not the model's own); the indexer's own
> saturated regime on HCA (there is no indexer there to saturate); the fp8/UE8M0 indexer and
> compressed stores (Gates 9/10); and any tiering (the experts are synthetic and supplied
> directly).

> **Gate result (Tier 2, item 18) — CERTIFIED, and the gate found the first property in this
> rewrite that only a serial loop can see.**
> `tests/test_v4_layer_body_serial_oracle.cpp`; the default suite is **33 tests**.
>
> **The simplification this gate removes, stated exactly.** Items 16 and 17 both end every step
> with a device→host round-trip back into the device: `to_half(oracle.res_out)` is written into
> `scratch.d_res_in`. That is the right instrument for one layer's arithmetic — it makes the
> comparison a *composition* measurement — and it also means the device's own output never
> re-enters the device's own state. The loop is then a sequence of independent steps with a
> fixed external input, so no error can accumulate and no state bug can express itself as drift.
>
> **What replaces it.** A three-layer stack in decode order — Sliding (layer 0), CSA (layer 2),
> HCA (layer 3) — for **136 tokens**, with:
> * the device's residual chain entirely its own. Layer L's output is layer L+1's input, and the
>   last layer's output is the next token's input. Nothing writes `d_res_in` (or `d_res_in_half`)
>   from the reference at any point.
> * the reference driven by the device's own state *and* its own discrete output. The residual it
>   is handed each step is read out of the device's buffers, and its MoE **combine** is driven by
>   the device's own ids and weights through a new `routed_ids_override` seam. This is trap 37's
>   principle generalized from the indexer to the router and to the state: a reference that
>   supplies its own inputs measures input divergence and calls it a defect.
> * 136 is not a round number. HCA (ratio 128) commits its first compressed entry at position
>   **127**, so a run that crosses an HCA boundary has a hard floor of 129 tokens; the extra
>   seven put the boundary behind the loop instead of at its end.
>
> **Worst case per checkpoint over 408 layer-steps, as a fraction of that checkpoint's own peak:**
>
> | checkpoint | worst |
> | :--- | ---: |
> | `x_norm` (attention RMSNorm) | `7.1e-4` |
> | `kv_rot` (local ring row, every wrap) | `9.8e-4` |
> | `attn_proj` (attention + inverse RoPE + grouped wo) | `9.9e-4` |
> | `moe_out` (routed + shared) | `9.7e-4` |
> | `res_out` (**the chained residual**) | `8.5e-4` |
> | `router logits` | `9.8e-4` |
> | `compressor partial row` (APE-adjusted, 34 wraps on CSA) | `9.8e-5` |
>
> Tolerance is `4e-3` of peak throughout — a **4× margin** over the measured floor, which is the
> fp16 store and nothing else. (Item 17's `router logits` at `1.2e-3` is, with the reference now
> driven by the device's logits rather than racing them, down at `9.8e-4`.)
>
> **C: the whole accumulated state, and closed forms that need no oracle at all.** At the end of
> the run the gate compares *every* piece of state, not just the slot the last token wrote: the
> whole local ring (`6.2e-4` / `8.1e-4` / `5.5e-4` of peak for the three layers), all **34**
> committed CSA entries and the **1** HCA entry (`4.5e-4` / `2.0e-4`), and every position. The
> positions are additionally checked against **closed forms**: slot *s* of a `capacity`-wide ring
> must hold the largest written position congruent to *s*, which is asserted for both the local
> ring and the compressor partial ring on all three layers, and the compressed entries must sit
> at `(i+1)·ratio − 1`. Those four checks would be satisfied by nothing except a state that
> evolved correctly. The HCA entry materialized at position 127 is then re-read at the end of the
> loop and required to be **bit-identical** (`max|delta| = 0.0`) — the loop's state contract in
> one line, and the line that a state recomputed or cleared per step would fail while passing
> every per-step comparison. Finally, that entry is shown **load-bearing** in the final attention
> row-set (dropping it moves `attn_out` by `26%` of peak).
>
> **A: the residual hand-back is pinned before the loop relies on it.** With the sublayer silent
> (`post = 0`, `comb = I`) `hc_post` returns its input **exactly** (`0.0`), which is the property
> the chain depends on; and the compressor ring's slot is shown to be `position mod capacity`
> with the position recorded rather than the write count — the state-evolution contract, asserted
> without any weights.
>
> **FINDING — the device's serial decode is not bit-reproducible, and this is the first thing in
> the rewrite that could not have been found any other way.** The default routed-expert path
> accumulates with `atomicAdd` (`aeon_moe_fused_w2_accum_kernel`), whose order across the six
> experts is undefined, so two runs of the same binary differ by ~`1e-7` in `moe_out`. Over 408
> sequential steps that drift compounds, and any router step whose 6th and 7th candidates sit
> inside the drift lands on either side. This gate **measured 0–2 such steps per run** with a
> worst selection-value gap of `8.6e-5`, varying between runs, and produced a `moe_out` difference
> of up to **`7.0e-3`** of peak — above the `4e-3` tolerance. That is what the third check of
> section C measures (`drift = 0.0313`, non-zero by construction) and what the
> `routed_ids_override` seam neutralizes: the comparison is now arithmetic against the device's
> own selection, while the selection *rule* is asserted separately against the device's own
> logits — so a flaky near-tie cannot masquerade as a defect, and a rule error still cannot hide.
> Items 16 and 17 are structurally blind to all of this: they re-seed the residual every step, so
> their trajectory never accumulates and their near-ties are decided once. The pipeline already
> carries a deterministic alternative (`deterministic_expert_accumulation_`, a per-expert GEMV
> plus `v4_pipeline_accumulate_expert_kernel`), and **the gate now requires it** — see the item-19
> mutation note below for why that decision was taken rather than left open. The finding stands
> and its measured quantities are unchanged; what changed is that the gate no longer *asserts*
> against the nondeterministic path, because doing that made it fail roughly one run in ten, and
> never on the near-tie step: once the device's trajectory and the oracle's diverge, the oracle's
> *state* — the ring keys written in earlier steps — is no longer the device's, so a later
> `attn_proj` disagreement is a consequence of the divergence rather than a defect.
> **Trap 38.**
>
> **What is NOT covered, named so it is not mistaken for coverage.** The real 128-token local
> window and the real `index_topk = 512` (the window is shrunk to 4 and the top-k to 3, both
> launch parameters — items 16/17 do the same); the routed experts' own arithmetic (synthetic
> per-slot payloads, as in item 17 — Tier 1 and item 16 own it); any tiering; and the *batched*
> path entirely (item 19).
>
> **A performance note.** 408 layer-steps, **203 s** single-threaded and **`77 s`** now that the
> oracle reductions are host-parallel (2026-09-16), registered with a 600-second timeout. The
> single-threaded cost is the oracle's fp64 matvecs, not the device; it is the price of comparing
> ~4 300 checkpoints against an independent reference, and it stays a single gate rather than being
> traded for coverage. As on item 17, the residue after parallelization is the gate's device half —
> a `TIMEOUT 600` is still the right bound, but the gate is no longer close to it.

**Tier 3 — Sequence.**
19. ~~**Chunked batched prefill** with one shared layer body. Gate: chunk ≡ serial.~~ **DONE as a
    structural gate — see the result above and trap 39.** `core/v4_layer_body_batch.hpp` drives the
    *same* two half-bodies decode calls, with the chunk's keys held outside the ring and a per-query
    composed row-set; the gate is **exact equality** (0 differing values) against the same tokens run
    one at a time, across three classes, three schedules, 130 tokens, and the whole final state, with
    the serial run itself tied to the Tier-2 decode body. **What remains of item 19 is its two
    non-structural halves, both explicitly open — and the first of them is *blocked*, not merely
    unmeasured.**

    * **(a) Throughput — blocked on the chunk size, which is capped at 8.** `run_layer_body_chunk`
      refuses a chunk longer than the compressor's partial ring, because
      `v4_save_compressor_state_kernel` writes that state at `position % partial_capacity` into a
      fixed ring of `coefficient · ratio` slots (**8** for CSA). Two tokens more than 8 apart inside
      one chunk would therefore share a slot, and a boundary reading the earlier token's row would
      silently get the later token's — so the guard throws rather than corrupt. This is Part I §6's
      requirement not yet met: *"the reuse boundary must be allowed to fall mid-ratio-window"*
      requires **position-addressed** partial state, which §6.2 and trap 24 named as the expensive
      retrofit. **A usable chunk size (256–512) is unreachable until that state contract exists, and
      the state contract is item 22's.** Second, and independent of the cap: **a per-token body
      cannot show a chunk-size benefit at all.** Chunk 1 and chunk 8 execute the same code the same
      number of times — the only difference is that keys are held outside the ring and each query
      composes its own row-set — so nothing is amortized and tok/s can only be flat or worse.
      Batching the projections is an *implementation* step, not a measurement one. So (a) needs
      **both** the position-addressed partial state and batched projections before any speed number
      means anything; measuring before them would report the cost of a body that is deliberately
      per-token.
      *Method, settled but not yet used:* hold every expert of the resident layers in VRAM so expert
      transfer is zero, isolating compute + composition — a **floor**, never comparable to model
      throughput, and one that must **verify** the zero (count host-to-device expert copies at load
      and require the count not to move during a timed region) rather than assume it.
    * **(b) The indexer top-k's per-token host round-trip** in `select_indexer_topk`, which the plan
      forbids in a prefill and which no equivalence gate can see because it changes no value. It is
      **two** `hipStreamSynchronize` calls (one after the scores' D2H, one after the indices' H2D),
      once per token per CSA layer whose candidates are non-empty. Unlike (a) this half is
      **countable now and needs no baseline**: the target is zero, and the count is derived from the
      schedule rather than measured.

    **Consequence for the sequence, and it is a sequencing correction:** *item 22's state layout is
    the next real step for prefill speed, not item 19's measurement.* Item 19's structural gate is
    complete and its throughput half cannot be closed first. The compressor partial state is also
    the one piece of §6.1's four that is still a ring rather than position-addressed — the local ring
    is a ring by design (item 20), the compressed store never evicts within the context (item 20),
    and only this one blocks a capability.

20. ~~**Long-context lifecycle** — ring reuse and boundary compression past context capacity.~~ **DONE — see the result below.** **Specified in 7.1.** The gate is **structural and at the real dimensions** — window `128`, ratios `4` and `128` — because every earlier gate shrank them (items 16–19 all note the shrinkage as uncovered), and it compares against **closed forms and invariants**, not an fp64 oracle, since items 16–18 already own the arithmetic on real weights:
    * **A — the local ring's slot assignment is predicted, not merely observed.** After a run longer than two windows, slot `s` must hold the largest position `p ≤ pos` with `p ≡ s (mod C)`, and `local_valid_count == min(pos+1, C)`. Nothing is compared to a reference: the ring's *contents* are stated in advance.
    * **B — reuse evicts exactly what the window excludes.** Perturbing a key whose position is outside a query's window changes that query's attention output by **exactly `0.0`**; perturbing one inside it changes it materially. This is the Tier-1 attention gate's exact-boundary instrument, driven through the whole body at the real window.
    * **C — the two stores are independent, bit-exactly.** A compressed entry materialized *before* the local wrap is byte-identical after it, and vice versa: writing the ring does not move a committed entry, and materializing an entry does not move the ring. Entry positions match the closed form `(i+1)·ratio − 1`, and the compressor partial ring's slots `pos mod capacity`.
    * **D — the row-set past the window is the class rule.** HCA reads every committed entry (dropping the oldest must move `attn_out`), CSA reads the indexer's selection; the counts are `min(K, (pos+1)/ratio)`.
    * **E — capacity is exact and the refusal is load-bearing (trap 40).** `record_position` throws at `max_seq_len` and does not below it; `K = ceil(max_seq/ratio)` makes the count clamp a no-op in range; and — the discriminating part — a hand-built **wrapped** store is shown to pass the kernel's own `compressed_positions[i] ≤ pos` guard while its row-set is the newest `K` entries rather than all committed ones. That is what makes the throw, not the guard, the thing that keeps compression from degrading silently into a sliding window.

    **Mutations** (per the second rule): the local ring stops rotating; the ring positions are not recorded; the refusal becomes a clamp; `committed_entries_for` loses its `min`; the compressed entry records `pos` instead of the boundary. **All of these must fail before the gate is trusted.** — **6 of 6 killed; see the item-20 mutation table.**

> **Gate result (Tier 3, item 20) — CERTIFIED, and the gate's finding is that the refusal in `V4Layer::record_position` is load-bearing rather than decorative.**
> `tests/test_v4_layer_body_lifecycle.cpp`; the default suite is **35 tests**.
>
> **The instrument is deliberately different from items 16–19, and that is the point.** Those gates certify the *arithmetic* against an fp64 oracle on real weights, and they all shrink the window (to 6, 10, 4) and the index top-k so a wrap fits inside a short run — each of them names the shrinkage as uncovered. This gate spends its whole budget on the real window instead, and compares against **closed forms and invariants** rather than a reference: the ring's contents are **predicted** from the token count before they are read. **52 checks, 0 failures**, in **5.6 s**.
>
> **B — the real window, and reuse as the definition of it.** 260 tokens through three layers at the model's own window of 128: slot `s` must hold the largest position `p ≤ 259` with `p ≡ s (mod C)`, checked for **all 384 slots** across the three classes, and the oldest surviving row must be exactly `pos − C + 1 = 132`. The ring is therefore *predicted*, not observed — a full ring is the sliding window, not an accident. The exact-boundary behaviour is then measured through the production kernel: perturbing an **unfilled** slot (holding the `-1` sentinel, outside every query's window) moves `attn_out` by **exactly `0.0`**, while perturbing the oldest in-window row moves it by `26.9×` the output peak. The row that occupied a slot *before* the wrap is gone rather than merely masked.
>
> **C — the two stores are independent, bit-exactly.** CSA commits 65 entries and HCA 2, both equal to the closed form `kTokens/ratio` — an **integer floor, not a ceiling**: the third HCA entry needs position 383. Every committed entry records `(i+1)·ratio − 1` (0 mismatches of 65, 0 of 2), the compressor partial ring is `coefficient·ratio` wide (8 and 128) with slots at `pos mod capacity`, and the entry materialized at each class's **first boundary** is **byte-identical 256 steps later** — with the local ring wrapping twice in between, which is the statement that local reuse never disturbs the compressed store. The ring as it stood *at* that boundary is also the closed form, so materializing an entry does not write into the local ring.
>
> **D — the row-set past the window is the class rule.** The local row-set is the full 128 at the last position; Sliding commits nothing; HCA's compressed row-set is **every** committed entry (dropping the oldest moves attention by `8.3%` of peak); CSA's is the indexer's top-k, with **65 candidates for 8 slots** — a real selection, which is why 65 is the number that makes this non-degenerate. Dropping one selected entry moves it by `4.5%`, and swapping a selected entry for a candidate the indexer **rejected** moves it by `1.63` absolute. That last check is what keeps the selection from being decorative.
>
> **E — the finding: the capacity refusal is load-bearing, and trap 40 is why.** The layer accepts `max_seq_len − 1` and **throws** at `max_seq_len`; the capacity equals the context's own entry count exactly (128 and 4 for the two ratios, `K·ratio == max_seq`), so at the last legal position the ring is full *and* nothing has been evicted.
>
> The discriminating measurement builds two compressed stores for **HCA's own** ratio and capacity (`ratio = 128`, `K = 4`) and gives the kernel the same row-set `[0, K)` for both. One holds entries 4..7 (a wrapped ring), the other entries 0..3 (what a clamp to capacity would keep — the reference's own `tl.minimum` truncates here). **Every populated slot in both passes the kernel's own `compressed_positions[i] ≤ current_pos` guard.** The keys are bit-identical so the weights cancel; only the values differ. The outputs differ by **`0.615` of peak**, and nothing exposed by the state distinguishes the two except the positions' closed form: `committed_entries_for` reads **4** for both. So the guard cannot see the eviction, the count cannot see it, and the *positions are the only thing that can*. The refusal to take a position past capacity is therefore the sole mechanism keeping HCA on "every committed entry" rather than "the newest K".
>
> **A wrapped store needs a position past the declared context**, which is exactly why this cannot arise by accident: `K = 4` slots at `ratio = 128` cover 4 entries, so a wrap requires `pos ≥ 8·128 − 1 = 1023` while the context ends at 511. The gate states that rather than assuming it: section A proves `K·ratio == max_seq` for both ratios, so the count clamp inside `committed_entries_for` is **dead code in range** — it is defence, not behaviour.
>
> **One gate-side defect found and repaired during the sweep, worth recording.** Section E's first probe queried an HCA store (capacity 4) with `ratio = 4`, so its "the reported count is the whole ring" line was exercising the count clamp on a **mismatched pair** and would have passed for the wrong reason. Mutation M20-6 originally had no honest home; after the repair (probe at the layer's own `ratio = 128`, `current_position = 1023`) the clamp is genuinely load-bearing and M20-6 is killed for the right cause. The general lesson is the one item 16 recorded from the other direction: **build a probe from the object's own parameters, or it measures your convenience values rather than the lifecycle.**
>
> **A second, smaller gate-side defect, also worth recording because it looked like a kernel bug.** The first version of the value pattern was `(d % 11) - 5` computed on a `uint32_t` — *unsigned* arithmetic, which wraps to ~4.3e9 for `d % 11 < 5`, after which `__float2half` saturated **exactly those entries** to `+inf` while the rest of the vector was correct. The symptom was maximally misleading: the output was `+inf` at precisely the entries the pattern made "negative", which reads as a sign-handling defect in the attention kernel. It was a probe bug, and the production kernel was never involved.
>
> **What is NOT covered, named so it is not mistaken for coverage.** The real `index_topk = 512` — 260 tokens commit only 65 CSA entries, so a real top-k would select all of them and the selection would degenerate; the top-k is shrunk to 8 and the *window* is left at the model's own 128, which is the opposite trade from items 16–19. No checkpoint is compared against a reference at all, so this gate says nothing about precision — Tier 1 and items 16–18 own that. The routed experts are six synthetic payloads. Positions beyond `max_seq_len` are refused rather than exercised, so the gate measures the *refusal*, not what a clamped engine would do. And no tiering and no prefix restore (Tier 4 item 22).

**Tier 4 — Integration (only after Tier 3 is fully green).**
21. ~~**Streaming / tiering.** Gate: expert bytes bit-exact across Hot/Warm/Cold.~~ **DONE — see the result below.**

> **Gate result (Tier 4, item 21) — CERTIFIED, and this is the first gate to drive `TieredExpertSupply` itself rather than a leg of it.**
> `tests/test_v4_expert_tiering.cpp`; the default suite is **36 tests**. **29 checks, 0 failures**, in **0.75 s**.
>
> **What already existed, and why it was not this.** Two tests cover the storage legs in isolation:
> `test_model_direct_io.cpp` proves a standalone `O_DIRECT` read equals the mmapped bytes (the NVMe
> leg), and `test_dynamic_expert_pool.cpp` proves one hand-driven `hipMemcpy` into a VRAM slot is
> bit-exact (the H2D leg). Neither drives `TieredExpertSupply`, which is the code that decides *which
> tier answers a request, which staging slot an I/O lands in, which host slot a demotion writes to,
> and which stream a copy is enqueued on*. A defect in any of those decisions is invisible to both,
> and it presents as the checkpoint plan's triage entry **"identical artifact, different result on
> reload"** — the one symptom that cannot be attributed to the graph.
>
> **The artifact's own dimensions, deliberately saturated.** The pool is 8 VRAM slots and 8 host
> slots, and `ExpertRegistry` populates *every* Hot slot at construction (`populate_round_robin`), so
> there is no free VRAM slot from the first request on: every cold miss must evict a resident. That is
> the production steady state, and it is what makes the demotion and promotion legs reachable without
> contriving one. The payload is `14 155 776` bytes = `3456` sectors = a **non-integral** `4` chunks
> of the reader's `4 MiB`, so each expert is four io_uring requests and a chunk-offset or
> per-request-length mistake is reachable.
>
> **One expert, three routes, one reference.** Expert `(layer 0, expert 1)` is delivered **Cold**
> (io_uring `O_DIRECT` into a staging slot, then H2D on the cold SDMA stream), then **Hot** (a repeat
> request that must move no bytes), then **Warm** (an LRU demotion by a D2H into a pinned host slot,
> then a promotion back from that slot). Every leg is compared against the **mmapped** container
> (`AeonModelLoader::get_expert_data`) — page cache, not the io_uring ring, and not any of the three
> routes above. All three deliveries are **bit-identical to each other and to the artifact**: 0 bytes
> differ of `14 155 776`, on every comparison.
>
> **The cold payload is checked twice, and that is what makes a failure diagnosable.** The staging
> slot is compared *before* the H2D is read back, and the destination VRAM slot after, so the two legs
> are separately observable. The mutation table below shows this working: M21-1 (the `O_DIRECT`
> offset) is caught by the staging check, M21-2 (the wrong staging slot) by the resident check **with
> the staging check still green**. That is not redundancy — it is the difference between a two-line
> diagnosis and a bisection.
>
> **Each delivery's *tier* is asserted, not only its bytes.** The cold request must report four I/O
> requests and a cold-miss count; the repeat request must claim no staging slot; the final request
> must be answered from the host pool (`hits_warm` +1 with **no** additional cold miss). Without that,
> a **silently dropped demotion** would send the expert back to Cold and the byte comparison would
> pass while measuring the wrong tier entirely. That is the same failure mode as the capacity clamp in
> `committed_entries_for` (item 20) and the survivor in M20-6, in a different container.
>
> **Two probe defects found and repaired during the sweep, both of the same family, plus a third
> choice worth stating.** The warm pool is *not* empty in section E — section C's three cold misses
> already demoted three residents into it — so absolute counts became deltas. And "the promotion
> consumed its warm slot" could not be expressed as a warm-pool count at all: with VRAM saturated the
> promotion also evicts a resident, and that eviction immediately fills a warm slot, so the two
> cancel. The instrument is now the host slot itself, `registry.host_slots[warm_slot] == -1` — the
> slot the promotion read from, returned to the pool — which is the fact rather than a proxy for it.
> **A count that two effects move in opposite directions measures neither.** The third: the victim is
> **predicted from `registry.hot_vram_lru.back()`** and the residents are enumerated **from the
> catalog**, rather than re-deriving `populate_round_robin`'s loop order — the M20-6 lesson, applied
> before it could bite.
>
> **What is NOT covered, named so it is not mistaken for coverage.**
> **Concurrency — checkpoint plan Stage D.2.** Requests are dispatched and materialized one round at a
> time. "No expert is read while partially written under concurrent access" is a separate property, it
> needs a forward pass streaming experts while the graph runs, and the rewritten graph has no caller;
> the overlap the supply path already implements is covered by the warm-tier A/B report and the supply
> telemetry, not here. **The two staged warm sub-paths:** `TieredExpertSupply` prefers a direct H2D
> from the pinned host slot and falls back to staging a copy through the arena only when
> `HostExpertPool::is_slot_pinned` is false, and on this silicon `hipHostMalloc` succeeds — so only
> the direct path runs. **The expert's numerics:** this gate never dequantizes; it compares the
> *packed payload bytes*, which is the thing a tier can corrupt and a kernel cannot (Tier 1 and items
> 14–18 own the rest). And **the eviction policy's quality** — the registry's LRU is exercised, but
> whether it chooses *well* is item 22's and the routing-placement study's question.

22. **Prefix cache manager.** Block table, cache key (tokens **+ non-token graph inputs**), matching, eviction; state pieces placed across tiers. Gate: **restore is byte-exact** with respect to never having evicted, and a candidate boundary outside the local window is detected rather than served stale (Part I §6.3 R3–R4).
    **This item now also gates item 19's throughput half**: the compressor's partial state is the one piece of §6.1's four still stored as a ring (`position % coefficient·ratio`), and that ring is what caps the chunk size at 8. Making it **position-addressed** is what allows a usable chunk (256–512), so the §6 layout contract should be treated as the prerequisite for prefill speed rather than as a Tier-4-optional refactor. It is also the piece a mid-ratio-window restore needs, which is the same requirement seen from the reuse side.
23. **Generation loop.** Coherent output; logits agree with reference over several steps.

**Do not build the streaming system before the numerics are correct.** Streaming bugs and numerical bugs produce identical symptoms, and debugging both at once is intractable.

### Open unknowns — settle at the named gate, do not assume now

| Unknown | Where settled | Why it is listed |
|---|---|---|
| ~~Hyper-Connections semantics~~ | **Resolved Tier 0.1** | Confirmed against `mhc.py` + `kernels/mhc/torch.py`. Found and fixed one real error: `hc_sinkhorn_eps` applies to **every** denominator. |
| ~~Compressor APE~~ | **Resolved Tier 0.2c: REQUIRED** | The plan omitted it. `score += ape[pos % ratio]`, `[ratio, coff·head_dim]` fp32, added to **score only**. Present as 62 trained tensors in our checkpoint. |
| ~~Attention softmax scale~~ | **Resolved Tier 0.2e: `1/sqrt(512)`** | `head_dim**-0.5` over the full 512-wide head, cited in both vLLM and sglang `[V attention.py:231; V deepseek_v4.py:665]`. |
| ~~Chat template location~~ | **Resolved Tier 0.2e: code, not data** | `tokenizer_config.json` has no `chat_template`; the encoder is `deepseek_v4_encoding.py::encode_messages`. Step 0 now specifies it, and flags the surface our formatter omits. |
| ~~HC comb scale~~ | **Resolved Tier 0.2e: `hc_scale[2]`** | The comb logits are scaled and biased before softmax; `hc_scale` is `[3]`. The plan previously applied only two scales. |
| ~~Attention row-set composition~~ | **Resolved Tier 0.2f** | Three classes: ratio 0 = local only; ratio 4 (CSA) = local + indexer top-512; **ratio 128 (HCA) = local + *all* committed compressed rows, no indexer**. Two KV sources merge under one order-invariant online softmax. |
| ~~Indexer ReLU~~ | **Resolved Tier 0.2: REQUIRED** | `relu` on the **per-head dot before weighting**, then `Σ_h w_h·relu(dot)` `[V sglang dsv4/indexer.py:121; qsa/dsa_indexer.py:43; cutedsl_fp8_paged_mqa_logits.py:43]`. My earlier retraction was wrong; my original assertion was right. |
| **Indexer Hadamard rotation** | **SETTLED at Gate 11 — do not apply it** | Measured, not argued. A normalized Hadamard is orthogonal, so a **two-sided** rotation leaves scores identical (`3.6e-16`) and the top-k identical (`0 of 512`), while a **one-sided** rotation changes scores by `1.46` and flips `180 of 512` indices. The reference drops it in its fused path and labels it *"(logit-preserving)"*; its only purpose is conditioning values before an fp8 round-trip, and our indexer K is fp16. **Revisit if the indexer K store moves to fp8 — and then apply it to BOTH Q and K.** |
| ~~`tid2eid` orientation~~ | **Resolved: `[vocab, 6]`** | Reference declares `(config.vocab_size, config.num_experts_per_tok)` `[V nvidia/model.py:820]`; still verify against our artifact at Gate 13. |
| KV fp8/E4M3 round-trip required vs optional | Gates 9 / 10 | **Strengthened toward required:** the canonical compressor kernel applies bf16+FP8/UE8M0 at two store points `[V fused_compress_quant_cache.py:288-345]`, and the checkpoint card states `--kv-cache-dtype` resolves to `fp8_ds_mla`, *"the only layout these backends implement"* `[V checkpoint README]`. Still a gate, because bf16 is a supported alternative and the delta must be measured. |
| MoE combine accumulation order | Gate 14 | **Shared-vs-routed order is now settled** (routed sum, then `+= shared` `[V nvidia/model.py:1020-1031]`). The **intra-routed** slot order was measured at the item-14 gate: it is `atomicAdd`, yet 32 identical 6-expert runs are **bit-identical** and the sum matches the weighted per-expert sum to `max_rel < 5e-7`. **Partially settled** — bounded for one configuration on this silicon, not in general. |
| Local-window reuse boundary behavior | **Half settled at Tier 3 item 20 (7.1); the prefix half stays open for Tier 4 item 22** | Whether a reused prefix whose boundary predates the local window must rebuild the ring, or whether compressed state fully covers attention. sglang tombstones such leaves; confirm the correct handling for our state rather than assuming. **Item 19 sharpened this**: the local ring cannot be reconstructed from anything else, so a prefix boundary older than the window means the ring must be *rebuilt by replaying the last `C` tokens*, not restored. That is a cost the reuse decision has to weigh, and it is the same constraint trap 39 describes from the batching side. **Item 20 settled the other half, by measurement**: within the declared context the compressed store **never evicts** — its capacity is exactly the context's own entry count, verified for both ratios — so a reused prefix's *compressed* state is always fully present and only the **local ring** is window-bounded and must be replayed (7.1(a)/(b)). It also showed the converse, which is the part that makes the refusal structural: a wrapped compressed store is invisible to the kernel's own position guard *and* to the committed count, so if the capacity were ever exceeded the engine would silently serve a sliding window of compressed entries rather than fail (trap 40). |
| **Indexer top-k on-device** | item 19's other remaining half — **countable now, needs no baseline** | `select_indexer_topk` copies the candidate scores to the host and synchronizes the stream **twice** (once after the scores' D2H, once after the indices' H2D), once per token per CSA layer with non-empty candidates. Part III forbids per-token host synchronization in a prefill — *"any device→host copy inside the layer loop serializes the whole chunk"*. It changes no value, so the `chunk ≡ serial` gate cannot see it by construction, and it is not a correctness gap. The target is **zero**, and the count is derived from the schedule rather than measured — so unlike the throughput half this one needs neither a baseline nor a faster body. |
| **Streaming under concurrency (checkpoint plan Stage D.2)** | Tier 4 item 21's remainder — **no gate yet** | "Run a forward pass while experts stream in; verify no expert is read while partially written, and that index positions stay valid under concurrent access." The item-21 gate drives `TieredExpertSupply` one round at a time, which is what byte-exactness needs; this property needs a forward pass streaming experts *while the graph runs*, and the rewritten graph has no caller. It is a **correctness** half that no existing gate covers — the overlap the supply path already implements is a *throughput* property, measured by the warm-tier A/B report and the supply telemetry, not this. Listed so a green item 21 is not read as covering it. |
| **Chunked-prefill throughput** | **blocked — item 19's throughput half; unblocked by item 22's state contract** | The structural gate is green; speed is a separate gate by the plan's own rule. **It is blocked, not merely unmeasured.** (i) The chunk size is capped at **8** by the compressor's partial ring (`position % coefficient·ratio` into a fixed ring), so a usable chunk size needs the **position-addressed** partial state that Part I §6.2 requires and item 22 owns. (ii) Independently, a per-token body cannot show a chunk-size benefit at all: chunk 1 and chunk N run the same code the same number of times, so nothing amortizes. Batching the projections is the implementation step that would create a win to measure. `compose_local_rows` is a per-query loop of up to `C` device-to-device copies. Method when it is measurable: all experts of the resident layers held in VRAM, giving a **compute+composition floor** with storage verified at zero — never comparable to model throughput. |

### Anti-circularity rule

An oracle that shares code with the kernel tests only self-consistency. **Every Tier-1 gate must compare against an oracle written from the architecture description, independently of the kernel under test.** This is the specific defect that allowed the previous implementation's tests to pass on wrong logic.

### Mutation testing — the second rule (added after Tier 1)

The anti-circularity rule removes one failure mode: the oracle agreeing with the kernel because they are the same code. It does **not** remove two others:

1. the oracle and the plan both being *my* reading of the reference, so a consistent misreading passes every gate;
2. a gate whose **tolerance or metric cannot see** the difference it is supposedly certifying.

Rule 2 is not hypothetical — it happened, and only mutation testing found it. The procedure is therefore now part of the method: **for each certified property, inject the specific wrong variant into the *kernel* and require the gate to go red.** A mutation that survives is either (a) a gate defect to repair, or (b) an equivalent mutation to be named as such. It is never ignored.

Ten mutations were run across nine kernels (2026-09-15, `gfx1100`). **Eight were killed, one is provably equivalent, and zero remain unclassified.**

| # | Mutation (injected into the kernel) | Gate | Result |
| :-- | :--- | :--- | :--- |
| M0 | `v4_indexer_scores_kernel`: ReLU removed | indexer | **Killed** — `max_rel 0.98`, 65/512 indices (control; reproduces the shipped bug) |
| M1 | `v4_rmsnorm_*`: `eps` deleted | norm | **SURVIVED** → gate repaired → **killed** |
| M2 | `v4_rope_*`: rotate head, not tail | rope | **Killed** — including the `[0,448)` bit-identical assertion |
| M4 | `v4_sliding_window_attn_*`: sink removed from the max | attention | **Equivalent** (see below) |
| M5 | `hc_sinkhorn`: comb uses `hc_scale[1]` instead of `[2]` | HC | **Killed** — `comb[16] max_abs 0.27` |
| M6 | `v4_save_compressor_state`: APE term dropped | compressor | **Killed** — 4 lines red |
| M7 | `v4_grouped_wo_a_*`: `wo_a` read interleaved | grouped out | **Killed** — `max_rel 1.38` |
| M8 | `moe_router_kernel`: bias on the logit, not the score | router | **Killed** — 8/8 tokens re-route |
| M9 | `v4_pipeline_swiglu_clamp_kernel`: symmetric gate clamp | expert §B | **SURVIVED** → gate repaired → **killed** |
| M9b | `aeon_swiglu_clamped` (fused W13): symmetric gate clamp | expert §C | **SURVIVED** → gate repaired → **killed** |

**Tier 2 mutations (2026-09-16, `gfx1100`, layer body).** Five mutations were injected
into the *layer body's wiring* — not into a kernel, because Tier 1 already owns the
kernels and the thing a layer gate must be able to see is a mis-wiring. **Four were
killed, one is redundant by construction.** 

| # | Mutation (injected into `run_layer_body_decoding`) | Result |
| :-- | :--- | :--- |
| M16-1 | Sliding layer rotated with the **compressed** RoPE base | **Killed** — `q_rot`/`kv_rot`/`attn_proj`/`res_out` at `5.4e-2` of peak vs a `3e-3` tolerance |
| M16-2 | Attention and FFN sublayers swap their HC parameter sets | **Killed** — `x_pre` at **100%** of peak, from position 0 |
| M16-3 | FFN pre-mix reads `pre_a` instead of `pre_f` | **Killed** — `ffn_norm` at `61%` of peak |
| M16-4 | Inverse RoPE on the attention output **omitted** (trap 8) | **Killed** — `attn_proj` at `47%` of peak |
| M16-5 | MoE accumulation buffer **not cleared** before the shared expert | **SURVIVED — redundant, not a gate defect** (see below) |

**M16-1 and M16-4 exposed the same real limitation, and it is now trap 36.** Both mutations
pass at **position 0** and fail from position 1 on. At position 0 the rotation angle is
`0 · inv_freq = 0` for *every* frequency and *every* base, so `cos = 1, sin = 0` and both
the forward and inverse rotation are the identity regardless of which table was used. A gate
that samples only the first token is therefore **blind to every RoPE question there is** —
the base class, the rotation, and its inverse. The multi-position sweep is not padding; it
is the only thing that makes those three properties observable.

**M16-5 survived, and the honest classification is "provably redundant", not "equivalent
mutation hiding a defect".** Removing the clear changes nothing because the shared expert's
W2 GEMV **writes** row 0 of the accumulation buffer before the routed accumulate reads it,
and the remaining `M_PAD − 1` rows that the clear covers are never read from that buffer.
The experiment is the gate itself: under the mutation, **ten consecutive tokens run with no
clear anywhere** and every one of them still matches, so the stale content demonstrably
cannot reach the output. This is recorded rather than repaired, because there is no assertion
to strengthen — the clear is simply not load-bearing. (It is also not free: it clears sixteen
rows where one is consumed. Left as-is; it is a kernel-inventory question, not a semantics
one.)

**Procedure note for the next tier.** Two of the five mutations were only visible *after*
position 0, and one of those was in the plan's own trap list while the other was in the
gate's own reference implementation. The pattern worth carrying forward: **choose mutation
data that reaches every branch the property lives in** — a position sweep for anything
RoPE-shaped, a scaled activation for anything clamp-shaped (item 14's lesson), a
low-RMS input for anything epsilon-shaped (M1's lesson). And **print the peak**: the oracle
defect in item 16 was legible only because the report showed each checkpoint's own scale.

**Tier 2 item 17 mutations (2026-09-16, `gfx1100`, the compressed classes).** Five mutations,
injected into `run_layer_body_decoding`'s class branch, the compressor save, or the indexer's
top-k. **All five killed.**

| # | Mutation | Result |
| :-- | :--- | :--- |
| M17-1 | CSA **ignores the indexer** and reads every compressed row | **Killed** — 27 lines; the row-set count and `attn_proj` both go red. This is trap 33 injected deliberately. |
| M17-2 | HCA reads only the **newest** compressed row | **Killed** — 81 lines; `attn_proj`/`res_out` and the row-set count |
| M17-3 | Compressed layers rotate with the **sliding** RoPE base | **Killed** — 1021 lines, from position 1 (trap 36 again) |
| M17-4 | Compressor partial state saved at the **wrong position** | **Killed** — 577 lines; `partial_score` and `partial_kv` at **100%** of peak |
| M17-5 | Indexer top-k ordered **ascending** | **Killed** — 32 lines; the element-by-element top-k comparison |

**M17-4 is worth reading as a small lesson in its own right.** Its first injection deleted the
`v4_save_compressor_state_kernel` launch outright and the *build* failed rather than the gate —
which proves nothing about the gate at all. The useful form of the same idea is a semantic
mis-wiring: offsetting the saved position by one makes both the APE row (`pos % ratio`) and the
ring slot (`pos % capacity`) different, and the gate reports **100% of peak** on exactly the two
lines that exist to see it. **A mutation that does not compile is not a killed mutation.**

**Tier 2 item 18 mutations (2026-09-16, `gfx1100`, the serial loop).** Three mutations, injected
into `run_layer_body_decoding`'s position and state handling — the places a single-token gate
cannot reach. **All three killed, and the first two are the point of the tier.**

| # | Mutation | Result |
| :-- | :--- | :--- |
| M18-1 | The local ring stops rotating (every token writes slot 0) | **Killed by both gates** — item 16 from position 1 (45 lines, `kv_rot`/`attn_proj`/`res_out` at 100%/82%/61% of peak); the serial gate with 1485 lines |
| M18-2 | `record_position` always told position 0 — the layer's state never advances | **Killed by the serial gate (577 lines); item 16 is COMPLETELY GREEN (0 failures)** |
| M18-3 | The compressed entry materialized at window position 0 instead of the boundary | **Killed by the serial gate** — 539 lines, `partial_score`/`partial_kv`/`compressed entry` |

**M18-2 is the strongest argument the tier has produced, and it is the reason the tier exists.**
Telling the layer that every token is at position 0 corrupts the committed-entry count and the
local valid count, so the compressed classes attend no rows at all after the first step — and
**item 16 does not see it, at all**. That is not a gap in item 16's assertions: a Sliding layer
has no compressed rows and its attention kernel takes `pos` as a launch parameter rather than
reading the counter, so the quantity the mutation corrupts is *not in that gate's input space*.
A per-step gate can only ever test the step it was handed. **State that fails to evolve is
invisible to any comparison that re-seeds its input between steps** — which is precisely the
structural blindness item 18 was added to remove, now demonstrated rather than asserted.

**M18-1 killing both gates is the useful control**: it shows the two gates are not redundant in
*either* direction. The ring rotation is visible per step (item 16) and catastrophic across the
loop (item 18, 1485 lines); the position counter is visible only across the loop. Neither gate
subsumes the other.

**Tier 3 item 19 mutations (2026-09-16, `gfx1100`, the chunk composition).** Four mutations,
injected into `compose_local_rows` and the commit loop — the ordering and the row-set, which is
where a chunk can differ from serial. **All four killed, and the fourth is the one that found a
defect in the gate rather than in the code.**

| # | Mutation | Result |
| :-- | :--- | :--- |
| M19-1 | Every chunk query reads the chunk's **first** key instead of its own | **Killed** — 10 lines; `every token` and the final state both red |
| M19-2 | **The false version**: the chunk writes the ring as it goes (no deferral) | **Killed** — 31 lines, including section B's "the ring is untouched by phase 1" |
| M19-3 | Composed rows in **descending** slot order (not the kernel's order) | **Killed** — 4 lines |
| M19-4 | The commit records a position **one below** the token's | **SURVIVED** → gate repaired → **killed** (7 lines) |

**M19-2 is the tier's finding, injected deliberately**: it is trap 39 exactly, and it is killed by
the ordering assertion in section B (the ring must be unchanged after every pre-attention half)
*and* by the value comparisons — which is what makes section B a check on the property rather than
on its symptom.

**M19-4 SURVIVED the whole of section C, and that is a gate defect worth recording.** Section C
compares the chunk path against the same path at `count = 1`, so a mistake applied to *both* sides
is invisible: the commit loop is the same code in both runs, so a wrong ring position was written
identically twice and compared equal. Section C2 was comparing tokens only. The repair is to compare
**the final state as well** in C2 — the one comparison whose other side is
`run_layer_body_decoding`, the Tier-2 certified body, which does not share the chunk driver at all.
M19-4 now fails on `decode body state: ring positions`. This is the second failure mode of the
method (a gate whose *metric* cannot see the difference it certifies) rather than the first
(a shared oracle), and it is the same shape as M9/M9b in Tier 1: the gate was green and wrong.

**Tier 3 item 20 mutations (2026-09-16, `gfx1100`, the lifecycle containers).** Six mutations,
injected into the ring's slot arithmetic, the recorded positions, the capacity refusal, and the
capacity derivation. **All six killed, and one of them forced a repair to the gate's own probe.**

| # | Mutation | Result |
| :-- | :--- | :--- |
| M20-1 | The local ring stops rotating (every token writes slot 0) | **Killed** — 6 lines; all 128 slots mismatch on all three layers, and the post-wrap oldest row reads `-1` |
| M20-2 | The ring records half the token's position | **Killed** — 6 lines; 128 slot mismatches and `B3`'s post-wrap row reads 66 |
| M20-3 | The compressed entry records a position one below its boundary | **Killed** — 2 lines; `65 of 65` CSA and `2 of 2` HCA entries wrong |
| M20-4 | The capacity refusal becomes a silent clamp | **Killed** — 1 line; `refused=no` |
| M20-5 | The compressed capacity is one slot short of the context's entry count | **Killed** — 3 lines; `K=127` vs `128`, `K=3` vs `4`, and the count check diverges |
| M20-6 | `committed_entries_for` drops its capacity clamp | **Killed** — 1 line; the wrapped case reads `8` where it must read `4` |

**Every mutation is killed for a single, named cause, and that is the useful property.** M20-1 and
M20-2 both produce "128 mismatched slots", but different ones: M20-1 leaves the *slots* unwritten
while M20-2 writes every slot and corrupts the *position*, and `B3`'s post-wrap row distinguishes
them (`-1` versus `66`). M20-5 is killed by the capacity identity, not by a behavioural difference —
which is the right instrument for a layout invariant.

**M20-6 is the one that changed the gate.** It was expected to be *equivalent*: the clamp
`min(capacity, (pos+1)/ratio)` is a no-op at every position inside the declared context, since the
capacity is exactly the context's entry count (section A). It nevertheless went red — and the reason
was a defect in the probe, not a defect in the gate's logic: section E was querying an HCA store
(capacity `4`) with `ratio = 4`, so the clamp fired on a **mismatched pair** and the check passed
for a reason unrelated to the lifecycle. Rebuilt at HCA's own `ratio = 128` with
`current_position = 1023`, the clamp is genuinely load-bearing (it is what makes the wrapped case
read `4` rather than `8`) and M20-6 dies for the right cause. **A passing assertion whose inputs are
inconsistent with the object under test is the same failure mode as M19-4 and M9/M9b: the gate was
green and wrong.** Rebuilding the probe from the layer's own parameters is the repair, and it is
cheaper to state than to notice.

**Tier 4 item 21 mutations (2026-09-16, `gfx1100`, the tiering plumbing).** Five mutations, injected
into `TieredExpertSupply`'s transfer plumbing rather than into the test — the `O_DIRECT` offset, the
staging binding, the H2D destination, and both ends of the demotion / promotion pair. **All five
killed, and each is killed by a *different* check, which is the property that matters: the gate
localises a failure to a leg instead of only reporting that something is wrong.**

| # | Mutation | Killed by |
| :-- | :--- | :--- |
| M21-1 | The cold `O_DIRECT` read starts one sector late | **The staging check** — `differ=13 441 457` of `14 155 776`; the NVMe leg is verified before the H2D, so the failure lands there first and then propagates to the host-slot and promotion checks |
| M21-2 | The cold H2D uploads staging slot `0` instead of the request's own slot | **The resident check only** — `differ=13 435 889`, with the staging check **green**: the read was right and the copy was wrong |
| M21-3 | The demotion D2H writes to the host slot *after* the reserved one | **The host-slot check** — `differ=13 432 258`; the promotion and section F inherit it |
| M21-4 | The warm promotion reads the host slot *after* the reserved one | **The promotion check only** — `differ=13 433 802`; the D2H check stays green |
| M21-5 | The cold H2D lands in the VRAM slot after the one the registry reserved | **The resident check** — `differ=13 434 043`; the warm chain inherits it |

**M21-1 versus M21-2 is the useful pair, and so is M21-3 versus M21-4.** Both members of each pair
corrupt the same delivery, and the *same line of the report* tells them apart: M21-1 fails the staging
check because the bytes that came out of io_uring were already wrong, while M21-2 passes it because
io_uring was right and the H2D took the wrong source; M21-3 fails the host-slot (D2H) check while
M21-4 passes it and fails only the promotion (H2D) check. Verifying each copy at *both* ends is
therefore not redundancy — it is what turns a two-line diagnosis into a bisection.

*A note on the sweep tooling, so a reader is not misled.* The mutation script restores the header
after each injection but does not rebuild, so the binary left in `build/bin` when the sweep ends is
the last mutant. Three consecutive runs of it read `FAIL — 3 failed`, which is correct and is **not**
a stability result; the unmutated source rebuilt cleanly and passed 3 of 3.

**Item 18's accumulation decision, taken here because trap 38 required it.** The item-18 gate now
runs with `executor.deterministic = true`, the same choice item 19 makes. It had been driving the
default `atomicAdd` path deliberately, to surface trap 38 — and it did, but the consequence was a
gate that failed roughly **one run in ten**, and never on the near-tie step itself: once the
device's trajectory and the oracle's diverge, the oracle's *state* (the ring keys written in
earlier steps) is no longer the device's, so a later `attn_proj` disagreement is a consequence of
the divergence rather than a defect. Trap 38 says in as many words that every loop gate must state
which accumulation it requires. The nondeterminism remains a **measurement** in the gate
(`near_tie`, `selection_mismatch_steps`, `worst_selection_gap`, `drift`) and its recorded
quantities are unchanged; it is no longer inferred from an intermittently red line.

**The M9/M9b finding was the important one, and it contradicted a claim this plan's own gate report made.** Item 14's gate was reported as certifying the asymmetric clamp rule. It did not: every assertion it made was either *oracle-vs-oracle* (the `ClampMode` fork) or a **relative** comparison whose floor was the probe's peak of ~1600, while the entire asymmetric/symmetric difference is bounded by `silu(−limit)·limit` = `4.5e-3`. A kernel that clamped the gate symmetrically passed the gate completely — in both the standalone and the fused form. The claim was wrong, and review would not have caught it because the gate printed `PASS`.

The repair is a **targeted comparison**: select only the entries where the two rules actually disagree (`gate < −limit`) and require the device to match the asymmetric oracle there, by an absolute margin. A `max_abs` over the whole vector cannot express this, because it is dominated by entries the two rules treat identically. Measured after repair: `vs asymmetric = 0.000000` / `0.000002`, `vs symmetric = 0.004540` — roughly 2000× separation, and each check trips only on its own kernel.

**M1 was a coverage gap, not a wrong assertion.** At unit-scale input the `eps` moves `inv_rms` by ~5e-7 relative, below one fp16 ulp, so an eps-free kernel is *bit-identical* — the gate was correct and the data could not exercise it. The repair adds a low-RMS regime (input scaled by 2⁻¹⁰, making `mean(x²) ≈ eps`), where the difference is a factor of ~√2. Re-measured: `max_abs 0.89` versus `2.4e-4` before.

**M4 is a provably equivalent mutation, and that is itself a result: it confirms trap 35 by experiment rather than assertion.** Removing the sink from the max can only change a result when `sink > max(score)`. In that regime the output is `exp(s_i − sink)·v_j / (1 + Σ exp(...) − sink)` either way — algebraically identical — and when the exponent overflows, the numerator stays finite while the denominator goes to `inf`, so the output is `0` and the correct path also returns `≈0`. There is no input for which the two differ observably. The property is real (it prevents the exponent from going positive) and it is genuinely untestable from the output. Do not "fix" a gate to catch it; record that it cannot be caught.

---

## Part VI — Known Traps

These are the specific things that will break this model if implemented naively.

1. **Hyper-Connections are not optional and not a subset.** All 43 layers run a 4-stream HC pre-mix, Sinkhorn, and post-mix before and after each sublayer. Omitting it is a structural omission, not a numerical detail.
2. **Sinkhorn operates on the HC 4×4 comb matrix**, not on attention or compressor output. `[corrected]`
3. **Sinkhorn needs exactly 20 iterations** `[V config]`, with `eps` in the specific places (pre-mix add, row denominators, column denominators). Fewer iterations leave the comb not doubly stochastic.
4. **Layers 0–1 are compressor/indexer-free but still run HC.** Branch only the compressor/indexer, never the HC. `[corrected]` And the indexer is not merely ratio-≠0: it is **CSA-only** (ratio 4). HCA (ratio 128) compresses but does **not** index. `[Tier 0.2f]`
5. **The Q path is low-rank with two norms.** `wq_a → q-norm(1024) → wq_b → per-head norm(512)`. Collapsing this into one matmul produces plausible but wrong output.
6. **There is no separate V.** A single shared KV head; attention uses the same tensor as key and value.
7. **Two RoPE bases.** Compressed layers use a different theta plus YaRN, with no amplitude scaling. Using one base everywhere is silently wrong.
8. **Inverse RoPE on the tail of the attention output**, before the grouped projection. Easy to omit, hard to diagnose.
9. **Attention sink enters the denominator only.** It must contribute no value vector, and must be included in the max.
10. **Indexer storage and quantization are fp8/UE8M0** — not INT8 `[V fused_indexer_q.py; V attention.py:964-975]`.
11. **Indexer ReLU is required and easy to omit.** `score[c] = kv_scale[c]·Σ_h w_h·relu(q_h·k_c,h)` — the ReLU is on the **per-head dot, before weighting**, not on the sum `[V sglang dsv4/indexer.py:121]`. At least three independent DSV4 implementations apply it. **It is not hypothetical: `v4_indexer_scores_kernel` shipped without it, and the Tier-1 gate is what caught it** (`max_rel = 0.98`, 65 of 512 top-k indices wrong). A missing ReLU produces a plausible-looking score, and the real-weight dense parity test never called this kernel, so nothing else would have flagged it. `[Tier 1]`
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
28. **The router's correction bias is stored as `ffn.gate.bias` but is NOT a linear bias.** The reference renames it `{".ffn.gate.bias": ".ffn.gate.e_score_correction_bias"}` `[V nvidia/model.py:1704]`. It is added to the **post-`softplus` scores** (`selection = scores + bias`), never to the logits before `softplus`. Treating it as `nn.Linear.bias` shifts every routing score. Present only on layers 3..42, F32 `[256]`. `[Tier 0.2d]`
29. **There is no group-limited routing.** `n_group`/`topk_group` are absent from the config; selection is a **flat top-6 over all 256 experts**. Do not introduce the DeepSeek-V3 grouped variant. `[Tier 0.2d]`
30. **`hc_scale` has three entries, and the comb uses the third.** `pre` uses `hc_scale[0]`, `post` uses `hc_scale[1]`, and the comb logits use `hc_scale[2]` (+ its own `hc_base` slice) **before** the softmax/Sinkhorn `[V kernels/mhc/torch.py:75-77]`. Our `hc_attn_scale`/`hc_ffn_scale` are `F32 [3]`. Applying only two scales silently changes comb sharpness. `[Tier 0.2e]`
31. **The chat template is code, not a model file.** `tokenizer_config.json` has **no** `chat_template`; the canonical encoder is `vllm/.../tokenizers/deepseek_v4_encoding.py::encode_messages`. It carries a `thinking_mode`, a `reasoning_effort` prefix (default `"low"` = empty), conditional BOS (`add_bos_token: False`), no system role token, DSML tool-call/tool-result rendering, `latest_reminder`/`developer`/`task` messages, and a **tools→keep-thinking** rule. An "almost right" template silently changes every prefix. `[Tier 0.2e]`
32. **The attention softmax scale is `1/sqrt(head_dim)` with `head_dim = 512`** — the full head, not the 64-wide RoPE part and not the indexer's 128. Both vLLM and sglang set `softmax_scale = head_dim**-0.5` `[V attention.py:231; V deepseek_v4.py:665]`. `[Tier 0.2e]`
33. **Only CSA (ratio 4) layers have an indexer. HCA (ratio 128) layers attend *all* committed compressed rows with no selection.** The width is `active_topk_width ≥ seq_len/ratio` (capped at 8192) `[V sparse_mla.py:252-260; V cache_utils.py:938]`. Running an indexer on a ratio-128 layer reads `attn.indexer.*` tensors that **do not exist** in the checkpoint — only ratio-4 layers carry them. The three classes are: ratio 0 = local only, ratio 4 = local + indexer top-512, ratio 128 = local + all compressed. `[Tier 0.2f]`
34. **The HC comb's flat index is `8 + 4·contraction + output` — contraction first.** The comb's **first** axis is the incoming residual stream being contracted; its **second** axis is the outgoing stream. Evidenced by the reference expansion `torch.einsum("...ij,...ih->...jh", comb_res_mix, residual)` — the comb's first axis is summed against the residual's stream axis `[V mhc/torch.py:96-108]`. This module's 2.0 prose previously stated the opposite, and the correction was made at Tier 1 only because the gate computed both readings: the kernel matches the einsum to `4.0e-4`, the transposed reading is off by `0.52`. **A transposed comb is still doubly stochastic and still produces plausible residuals**, so no closeness test can detect it. Note also that both 2.0 and 2.7 use comb indices, and 2.7 was already right — when the two disagree, that is the signal to re-read the reference. `[Tier 1]`
35. **`sink in the max` is a robustness property, not an observable one — do not mistake a passing output comparison for coverage of it.** The sink must be in the max so that `exp(sink − m) ≤ 1` when the sink dominates; otherwise the exponent is positive and can overflow. But when the sink dominates, the output is zero *either way*, so a gate cannot tell the two implementations apart from the result. Assert finiteness instead, and do not claim the max placement is verified. `[Tier 1]`
36. **At position 0 every RoPE is the identity, for every base — so position 0 cannot test RoPE at all.** The angle is `pos · inv_freq`, which is `0` at `pos = 0` for every frequency, so `cos = 1, sin = 0` and both the forward rotation and its inverse leave the vector untouched **whichever table was passed**. A gate that only tests the first token is blind to the wrong base class (numeric theta or YaRN), to a rotation applied to the head instead of the tail, and to an omitted inverse rotation — all three at once. Two Tier-2 mutations demonstrated this: the wrong RoPE base and a deleted inverse RoPE each **passed at position 0** and failed from position 1 (`5.4e-2` and `47%` of peak). Sweep positions. Note this is a *gate-design* trap, not a graph trap: the graph is correct here, the test was not. `[Tier 2]`
37. **A discrete selection cannot be checked against a continuous oracle's inputs — re-derive it from the device's own.** The MoE router's ids are the `top-6` of `sqrt(softplus(logit)) + bias`; a *near-tie* (6th and 7th within the ~`1.2e-3` by which the device's fp16 GEMV and an fp64 oracle disagree on the logits) is then legitimately ordered either way, and the ids differ while the weights match to `8e-3`. Item 17 initially required the ids to equal the oracle's and **8 of 257** HCA tokens failed for this reason alone. The honest test compares the **logits** peak-relative (where precision belongs) and the **selection rule** against the ids re-derived from the **device's own** logits (where the rule belongs), with a `1e-6` allowance for the fp32-vs-fp64 activation. A gate that conflates the two cannot say whether a disagreement is a rule error or a rounding difference. `[Tier 2]`
38. **A serial decode is not bit-reproducible, and a reference must therefore be driven by the device's own discrete outputs *and* its own state — at every level, not just the top.** The default routed-expert path accumulates with `atomicAdd`, whose order across the six experts is undefined, so `moe_out` differs between two runs of the same binary by ~`1e-7`; over a 408-step loop that compounds, and any router step whose 6th and 7th candidates sit in the drift flips its expert set. Item 18 measured **0–2 such steps per run**, varying between runs, with a `moe_out` difference up to **`7.0e-3`** of peak — **above the `4e-3` tolerance**, i.e. a nondeterministic *failure* of an unconditioned comparison. Two consequences, both now built in: (i) the reference's combine is driven by the device's ids and weights (`routed_ids_override`) while the *rule* is asserted against the device's own logits, so a near-tie cannot masquerade as a defect and a rule error still cannot hide; (ii) a gate that measures the *loop* must compare accumulated **state** — whole rings, all committed entries, all positions — not only the step it just produced, because a per-step comparison is complete at the step level and therefore structurally blind to accumulation. Items 16 and 17 could not have seen any of this: they re-seed the residual every step, so their trajectory never accumulates. The deterministic alternative already in the pipeline (`deterministic_expert_accumulation_`) removes the amplification at its source; Gate 22's byte-exact prefix restore and item 19's `chunk ≡ serial` gate must both decide explicitly which accumulation they require. **Item 19 answered that for itself: requiring the deterministic path, so `chunk ≡ serial` can be an equality rather than a tolerance** — see item 19's gate result. **Item 18's own gate was then moved onto that same path**, because asserting against the atomic one made it fail about one run in ten; with the fixed order its residue on the oracle-vs-device selection counter is `2 of 408` steps at a **worst gap of exactly `0.0`** — exact ties whose *ids are reordered* (the positional comparison counts that, the selection values are equal), not drift. That is worth keeping distinct: a gap that is non-zero means a rule disagreement, and a gap that *grows* across steps means drift has returned. `[Tier 2]`
39. **A chunk must not write its keys into the local ring as it goes — the write for the chunk's last token evicts the oldest key of its own first token's window.** The local ring has `C` slots and position `p` lives in slot `p mod C`. Query `q` attends `[q − C + 1, q]`. In a chunk spanning `[S, E]` with `E > S`, the write for `p` lands in the slot that held `p − C`; take `p = E` and `q = S` and `E − C ≥ S − C + 1` holds **whenever the chunk has more than one token**. So "run every token's pre-attention half, then every token's attention half" is not equivalent to serial at *any* chunk length above one — and it is the **local** ring, not the compressed path, even though the plan's original finding about `prefill_batched` was phrased entirely in terms of the compressor, the indexer and the MoE. The canonical fix is the reference's own shape: hold the chunk's keys in a separate per-forward buffer (*"not yet written to the SWA ring"* `[V paged_prefill.py:13-33]`), give each query a composed row-set of the pre-chunk ring rows in its window plus the chunk's own rows up to itself, and commit to the ring afterwards. **Order the composed rows by ring slot, not by position**: the decode path iterates slots `0 … C−1`, so for a wrapped window the kernel's summation order is a rotation of position order, and matching it is what makes `chunk ≡ serial` an equality (item 19 measured **0 differing values**, 130 tokens × 3 classes × 3 schedules) instead of a tolerance. A corollary worth stating separately, because it is easy to get wrong when writing the reference: **`chunk ≡ serial` must mean the same tokens one at a time, not the decode body's `step()`** — a chunk of one is the only length at which batching is absent, which is what makes the property well-posed. `[Tier 3]`

40. **A compressed-entry ring's wrap is silent: every populated slot records a position `≤` the query position, so no position guard can detect eviction.** Entry `i` lives in slot `i mod K`; after a wrap the slot's occupant is a *newer* entry whose recorded position is still perfectly legal, so a guard of the form `compressed_positions[i] ≤ current_position` — which our attention kernel has — cannot distinguish a live row from one whose predecessor was overwritten. The row-set then becomes a sliding window over compressed entries (the newest `K`) instead of every committed entry, and attention simply forgets its long-range context without raising anything. The capacity is derived so this is unreachable within the declared context (`K = ceil(max_seq/ratio)` is exactly what that context produces, and the reference asserts `active_topk_width ≥ max_seq_len // ratio` `[V sparse_mla.py:158-170, 259]`), and our engine refuses a position at or beyond capacity `[V v4_layer.hpp:211-216]`. **That refusal is the only mechanism distinguishing "all committed entries" from "the newest `K`", so it must not be relaxed to a clamp or a modulo** — a clamp would be *worse* here, because the reference's own `tl.minimum(num_compressed, max_compressed_tokens)` `[V sparse_mla.py:405]` truncates to the **oldest** `K` entries while a ring keeps the newest, and the two differ by a large fraction of `attn_out`'s peak (item 20 measures it on a hand-built wrapped store). `[Tier 3]`
