# Project Aeon - DeepSeek-V4 Engine: Final Review & Fix Plan

**Type:** external static code review (no code executed, no tests run)
**Target symptom:** coherent-looking but wrong / incoherent generation
**Date:** 2026-09-14

---

## 0. How to use this document

Work strictly top-down through Section 2 (Stage 0), then Section 3 (Stage 1), then
Section 4 (Stage 2 bisection). Do **not** apply Section 5 (P2 convention items)
speculatively - they can only be resolved against the DeepSeek-V4 reference
implementation, and applying two of them wrongly at once produces partial
cancellation, which is exactly the "fix one thing, break another" pattern the team
has been hitting.

Grades:

| Grade | Meaning |
|---|---|
| P0 | Confirmed defect, can produce wrong output or memory corruption. Fix unconditionally. |
| P1 | Confirmed defect, limited blast radius, or latent UB that works today by accident. |
| P2 | Convention question. Code is self-consistent; only a reference diff can settle it. |
| P3 | Robustness / diagnostics / performance. Not a correctness bug. |

---

## 1. Top 10 by (confidence x risk)

| # | ID | Finding | Grade | Confidence | Effort |
|---|---|---|---|---|---|
| 1 | P0-6 | `upload_tensor` does no dtype/shape validation - BF16 checkpoint read as FP16 is undetectable | P0 | High | 2 h |
| 2 | P0-1 | Divergent `__syncthreads()` in `moe_router_kernel` - UB, corrupts routing weights | P0 | Certain | 15 min |
| 3 | P0-7 | Main RoPE table built with hardcoded `factor=1.0`; config-discarding `initialize()` overload | P0 | Certain | 1 h |
| 4 | P0-4 | Silent `return` on shape mismatch in all four W4A16 dispatchers - MoE becomes a no-op | P0 | Certain | 30 min |
| 5 | P0-2 | Inverse RoPE applied to values that were never rotated | P0 | High | needs P2-2 |
| 6 | P2-1 | RoPE pairing: interleaved (code) vs `rotate_half` (HF reference) | P2 | Unresolved | needs reference |
| 7 | P1-2 | YaRN `mscale` missing entirely - every attention distribution ~28% too flat | P1 | High | 30 min |
| 8 | P1-6 | FP16 logits + first-index-wins argmax - systematic bias toward low token IDs | P1 | High | 1 h |
| 9 | P1-1 | YaRN ramp compares full-dim index against pair index (off by 2x) | P1 | Certain | 15 min |
| 10 | P2-4 | HC stream init replicates embedding into all 4 streams - layer-0 magnitude 2-4x | P2 | Unresolved | needs reference |

---

## 2. STAGE 0 - Make failures loud (~0.5 day, zero semantic change)

No behaviour changes. Run the failing prompt immediately afterwards - it is
entirely possible one of these asserts fires and ends the investigation.

### 0.1 P0-6 (CRITICAL) - validate dtype and shape on every tensor upload

**File:** `v4_dense_weight_binding.hpp`, `v4_model_resources.hpp`

Current code is a blind byte copy that can never fault and can never detect a
mismatch:

```cpp
const auto& tensor = loader.get_tensor(name);
CHECK_HIP(hipMalloc(device_ptr, tensor.byte_size));
CHECK_HIP(hipMemcpy(*device_ptr, tensor.data, tensor.byte_size, hipMemcpyHostToDevice));
```

Three silent failure modes:

1. **dtype.** Some members are `half*` (`d_attn_norm`, `d_wq_b`, `d_wkv`, ...),
   others `float*` (`d_attn_sink`, `d_compressor_ape`, `d_hc_attn_fn/base/scale`,
   `d_hc_ffn_*`, `d_gate_bias`). Nothing checks the checkpoint agrees.
   **DeepSeek checkpoints are natively BF16.** BF16 bits reinterpreted as FP16
   produce finite, small, structured-looking garbage - not NaNs. This matches the
   reported symptom profile better than anything else in this document.
   Symmetrically, BF16 read as `float` glues two adjacent values into one.
2. **shape.** A transposed or mis-sized `wq_b` / `wkv` / `ape` is accepted and then
   read with the kernel's assumed strides.
3. **presence-only gating.** `has_tensor()` is the only check. The `tid2eid` path
   calls `upload_tensor` (which throws when missing) and *then* re-checks
   `has_tensor` - dead code implying the author believed it optional.

`V4ModelResources` adds: `loader.get_data_ptr<half>("embed.weight")` is an
unchecked host reinterpret, and `head.weight` is uploaded without verifying
`129280 x 4096`.

**Fix:** add a validating accessor and route every bind through it.

```cpp
// aeon_loader.hpp
const Tensor& require_tensor(const std::string& name,
                             DType expected,
                             std::initializer_list<int64_t> expected_shape) const;
// throws with name, expected vs actual dtype and shape
```

```cpp
template<typename LoaderT, typename T>
void upload_tensor(const LoaderT& loader, const std::string& name, T** dptr,
                   DType expected, std::initializer_list<int64_t> shape) {
    const auto& t = loader.require_tensor(name, expected, shape);
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(dptr), t.byte_size));
    CHECK_HIP(hipMemcpy(*dptr, t.data, t.byte_size, hipMemcpyHostToDevice));
}
```

**Ten-minute pre-test before writing any code:** dump
`layers.0.attn_norm.weight` from the checkpoint and the first 32 values of
`d_attn_norm`. RMSNorm gains should sit near 1.0. If they are structured noise,
the dtype is wrong and every other finding in this document is moot.

### 0.2 P0-4 - hard-fail the W4A16 dispatchers

**Files:** `aeon_w4a16_swizzled_gemv.hpp`, `aeon_moe_fused_w13.hpp`, `aeon_moe_fused_w2.hpp`

Present in `dispatch_aeon_w4a16_swizzled_gemv`,
`dispatch_aeon_w4a16_swizzled_dual_gemv`, `dispatch_aeon_moe_fused_w13_swiglu`,
`dispatch_aeon_moe_fused_w2_accum`:

```cpp
if (K != ITERS * LPR * 32 || N % RPW != 0) { return; }   // no launch, no error
```

One wrong dimension makes the entire MoE block a no-op, leaving the output buffer
holding the **previous layer's or previous token's** data - a perfect generator of
grammatical-but-wrong text, with zero diagnostics. Current call sites do check
out, so this is not today's bug; it is the reason the engine cannot tell you when
it becomes one.

```cpp
if (K != ITERS * LPR * 32 || N % RPW != 0) {
    fprintf(stderr, "FATAL w4a16 dispatch shape mismatch: N=%d K=%d "
                    "expected K=%d, N%%%d==0\n", N, K, ITERS*LPR*32, RPW);
    std::abort();
}
```

### 0.3 P0-5 - config is parsed and then ignored

**Files:** `config.hpp` vs `v4_pipeline.hpp`

`DeepSeekV4Config` parses ~40 fields. The pipeline hardcodes these instead:

| Config field | Hardcoded as |
|---|---|
| `rms_norm_eps` | `1e-6f` at every `v4_rmsnorm_wave32_kernel` launch |
| `hc_eps` | `1e-6f` in `hc_project*`, `hc_sinkhorn_normalize_kernel` |
| `hc_sinkhorn_iters` | `20` |
| `hc_mult` | `constexpr int HC = 4` |
| `routed_scaling_factor` | `1.5f` |
| `n_routed_experts` | `256` / `ROUTER_EXPERTS`, and `s_scores[256]` |
| `num_experts_per_tok` | `6` / `ROUTED_EXPERTS` |
| `swiglu_limit` | `10.0f` |
| `moe_intermediate_size` | `INTER_DIM = 2048` |
| `vocab_size` | `129280` |
| Sinkhorn alpha | `2.0f` - **not present in config.json at all** |

All agree with today's defaults, so nothing is broken right now - but the engine
cannot detect a checkpoint that disagrees and will silently emit garbage. Add
validation even if you do not plumb the values through:

```cpp
AEON_REQUIRE(cfg.hidden_size           == kernel::DSV4_HIDDEN_SIZE);
AEON_REQUIRE(cfg.moe_intermediate_size == INTER_DIM);
AEON_REQUIRE(cfg.n_routed_experts      == 256);
AEON_REQUIRE(cfg.num_experts_per_tok   == 6);
AEON_REQUIRE(cfg.hc_mult               == 4);
AEON_REQUIRE(cfg.hc_sinkhorn_iters     == 20);
AEON_REQUIRE(cfg.vocab_size            == 129280);
AEON_REQUIRE(cfg.num_hidden_layers     == 43);
AEON_REQUIRE(std::abs(cfg.swiglu_limit          - 10.0f) < 1e-6f);
AEON_REQUIRE(std::abs(cfg.routed_scaling_factor -  1.5f) < 1e-6f);
AEON_REQUIRE(std::abs(cfg.rms_norm_eps          -  1e-6f) < 1e-9f);
AEON_REQUIRE(cfg.scoring_func == "sqrtsoftplus");
AEON_REQUIRE(cfg.topk_method  == "noaux_tc");
AEON_REQUIRE(cfg.norm_topk_prob);
AEON_REQUIRE(cfg.quant.num_bits == 4 && cfg.quant.symmetric && cfg.quant.group_size == 32);
```

### 0.4 P1-9 - assert chat special tokens are real added tokens

**File:** `dsv4_chat_formatter.cpp`

`format()` builds the prompt as a **string** via `token_text(id) == decode({id})`
for BOS, `<|User|>`, `<|Assistant|>`, `<think>`, `</think>`, EOS, then re-encodes
the whole thing. Lossless **only if all six IDs are in `added_tokens_`**, because
`encode()` does longest-match added-token scanning before BPE and `decode()`
emits `added->content` verbatim. If any resolves to a base-vocab entry, its
byte-level text is re-split into ordinary BPE pieces, **silently destroying the
chat structure** - which alone causes incoherent generation.

Stage 0: assert all six IDs are present in `added_content_to_id_` with
`kSpecialFlag`. Stage 1 proper fix in 3.8.

### 0.5 P3-12 - error checking after every kernel launch

No `hipGetLastError()` exists anywhere in the pipeline. Add
`AEON_KERNEL_CHECK()` after every dispatch in debug builds.

### 0.6 Build once with `AMD_SERIALIZE_KERNEL=3` and once with `-fsanitize=address`.

---

## 3. STAGE 1 - Unambiguous defects (~1 day)

### 3.1 P0-1 - divergent barrier in `moe_router_kernel`

**File:** `moe_router.hpp`

```cpp
if (tid < top_k) {              // only 6 of 64 threads enter
    float val = s_topk_val[tid];
    out_indices[tid] = s_topk_idx[tid];
    __shared__ float s_sum;
    if (tid == 0) { /* ... */ s_sum = sum; }
    __syncthreads();            // 58 threads never arrive -> UB
    if (renormalize) val = val / (s_sum + 1e-20f);
    val *= routed_scaling_factor;
    out_weights[tid] = val;
}
```

Threads 1-5 may read `s_sum` before thread 0's write is visible: **uninitialised
routing weights**. Behaviour differs between `-O0`/`-O3` and across ROCm versions -
directly consistent with the team's "fixes break unrelated things" experience.

```cpp
__shared__ float s_sum;                 // hoisted to block scope
if (tid == 0) {
    float sum = 0.0f;
    for (int k = 0; k < top_k; ++k) sum += s_topk_val[k];
    s_sum = sum;
}
__syncthreads();                        // all 64 threads reach this
if (tid < top_k) {
    float val = s_topk_val[tid];
    if (renormalize) val /= (s_sum + 1e-20f);
    out_weights[tid] = val * routed_scaling_factor;
    out_indices[tid] = s_topk_idx[tid];
}
```

### 3.2 P0-7 - RoPE table construction ignores config

**File:** `v4_model_resources.hpp`

```cpp
rope_table.init(max_seq_len, config.rope_theta, 1.0f);   // factor hardcoded
compressed_rope_table.init(max_seq_len, config.compress_rope_theta,
                           config.rope_scaling.factor,
                           config.rope_scaling.beta_fast,
                           config.rope_scaling.beta_slow,
                           config.rope_scaling.original_max_position_embeddings);
```

The main table gets `factor = 1.0f` unconditionally plus default betas; the
compressed table gets `rope_scaling.factor`, which **defaults to `1.0f`** when
`config.json` has no `rope_scaling` block - while `cfg.rope_factor` defaults to
`16`. The two tables are therefore built under different scaling regimes, and
YaRN may be inert everywhere despite `rope_factor = 16`.

Also delete this overload - it discards the parsed checkpoint config entirely:

```cpp
void initialize(const AeonModelLoader& loader, uint32_t max_seq_len) {
    initialize(loader, max_seq_len, DeepSeekV4Config{});   // DELETE, grep call sites
}
```

**Resolve which table carries YaRN before touching 3.3 or 3.4 - P0-7, P1-1 and
P1-2 interact.**

### 3.3 P1-2 - YaRN `mscale` is missing

**File:** `v4_attention.hpp`

DeepSeek YaRN multiplies attention logits by `mscale = 0.1*ln(factor) + 1`; at
`factor = 16` that is **1.27726**. Nothing applies it. `DSV4_ATTN_SCALE` is a bare
`1/sqrt(512)` for both main and compressed paths, so **every attention
distribution in the model is ~28% too flat** - across 43 layers this reads as
vagueness and topic drift.

```cpp
const float mscale = 0.1f * std::log(static_cast<float>(rope_factor)) + 1.0f;
const float attn_scale = (1.0f / std::sqrt((float)HEAD_DIM)) * mscale * mscale;
```

Note the **square**: `mscale` is applied to q and k separately, so the logit scale
carries `mscale^2`. Confirm against the reference (P2-3).

### 3.4 P1-1 - YaRN ramp unit mismatch

**File:** `v4_attention.hpp`, `RopeTable::init`

`correction_dim()` returns an index in **full rotary-dim units** (0..64) but is
compared against `k`, a **pair** index (0..31). The ramp window is off by 2x,
partially masked by the clamp `high = half_rope - 1 = 31`.

```cpp
const float low  = std::floor(correction_dim(beta_fast, rope_dim, theta, orig_max)) * 0.5f;
const float high = std::ceil (correction_dim(beta_slow, rope_dim, theta, orig_max)) * 0.5f;
```

Config: `beta_fast=32`, `beta_slow=1`, `factor=16`,
`original_max_position_embeddings=65536`, `rope_theta=10000`.

### 3.5 P1-6 - FP32 logits and FP32 argmax

**Files:** `v4_pipeline.hpp`, `v4_attention.hpp`

`scratch.d_logits` is `__half`; `v4_gemv_fp16_vec8_kernel` accumulates in float
then narrows. Over 129,280 candidates FP16's ~3 significant digits create many
exact ties in the top band, and both `v4_argmax_fp16_partial_kernel` and
`v4_argmax_partial_reduce_kernel` break ties toward the **smallest index** - a
systematic bias toward low token IDs, which in a BPE vocab are byte-fallbacks and
short fragments.

Keep the LM head output in FP32 and argmax over floats. **Do this before Stage 2** -
it removes a confound from every subsequent measurement.

### 3.6 P1-3 - `return` before `__syncthreads()` in HC pre-combine

**File:** `hc_sinkhorn.hpp` - `hc_pre_combine_kernel`, `hc_pre_combine_batched_kernel`

```cpp
if (idx >= hidden_size) return;   // early exit
...
__syncthreads();                  // barrier some threads may skip
```

With `H = 4096` and 1024 threads/block no thread exits, so it works today. Still
UB; breaks the moment `hidden_size` or block size changes. Move the guard below
the barrier or predicate the write.

### 3.7 P1-7 - explicit zeroing of the MoE accumulator

**Files:** `aeon_moe_fused_w13.hpp`, `aeon_moe_fused_w2.hpp`, `v4_pipeline.hpp`

`aeon_moe_fused_w13_swiglu_kernel` zeroes `output_f32[0..output_dim)` from its
`blockIdx.y == 0` blocks; `aeon_moe_fused_w2_accum_kernel` then `atomicAdd`s into
it. Coverage is adequate today (64 x 256 = 16384 >= 4096), but if W13 ever
early-returns (P0-4) or is passed `output_f32 = nullptr` while W2 is not, W2
accumulates on top of the **previous token's** MoE output. The two failure modes
compose.

Zero the buffer at the call site with `hipMemsetAsync` (as already done for
`d_swizzled_counters`) and drop the zeroing from W13.

### 3.8 P1-9 - stop round-tripping control tokens through text

**File:** `dsv4_chat_formatter.cpp`

Build the prompt as `std::vector<uint32_t>` and append special IDs directly,
BPE-encoding only the free-text spans. Never decode-then-re-encode a control
token.

### 3.9 P1-4 - staging release before the final sync

**File:** `v4_pipeline.hpp`, `step()`

```cpp
v4_argmax_partial_reduce_kernel<<<...>>>(...);                  // enqueued
for (uint32_t s : releasable_staging_slots)
    prefetch_staging_->release_after_gpu_transfer(s);           // BEFORE sync  <-- BUG
hipMemcpyAsync(&h_argmax, ..., compute_stream);
hipStreamSynchronize(compute_stream);
for (uint32_t gid : leased_experts) release_lease(gid);         // after sync - OK
```

The staging->VRAM copy has been waited on via `hipStreamWaitEvent`, so the window
is narrow, but under NVMe cold-load pressure a refill can begin while the last
layer's W2 kernel is still resident. Move the staging loop below
`hipStreamSynchronize`.

### 3.10 P1-5 - make lease release explicitly ordered

**File:** `v4_pipeline.hpp`, `step()`

Layer *l*'s leases are released at the top of layer *l+1*'s routed section, which
is preceded by `hipStreamSynchronize(compute_stream)` for the router readback.
**Correct today** - but correctness depends entirely on a sync that exists to copy
6 floats D2H and is an obvious performance target. The moment that readback goes
async this becomes silent VRAM corruption.

```cpp
hipEvent_t layer_moe_done;                       // per-layer, created once
hipEventRecord(layer_moe_done, compute_stream);  // after the W2 dispatch
...
hipEventSynchronize(layer_moe_done);
for (uint32_t gid : leased_experts) release_lease(gid);
```

Add a comment at the router readback stating that removing the sync requires this
change first.

### 3.11 P1-10 - FP32 accumulation in the deterministic MoE path

**Files:** `v4_pipeline.hpp`, `v4_pipeline_ops.hpp`

Fused path: `atomicAdd` into FP32, narrowed once. Deterministic path:
`v4_pipeline_accumulate_expert_kernel` does an **FP16 read-modify-write six times**.
The "deterministic" path is the *less* accurate one and the two will not agree -
a poor debugging baseline. Accumulate into FP32 and narrow once.

### 3.12 P1-11 - config rope-scaling defaults

**File:** `config.hpp`

- `RopeScalingConfig::factor` defaults to `1.0f` while `DeepSeekV4Config::rope_factor`
  defaults to `16`: a `config.json` with no `rope_scaling` block yields
  `rope_factor = 16`, i.e. full YaRN on a model that requested none.
- Top-level `original_max_position_embeddings` is **never read** - only the nested
  one. A checkpoint declaring it at top level silently gets 65536.
- `cfg.rope_factor = static_cast<int32_t>(cfg.rope_scaling.factor)` truncates;
  2.5 becomes 2. Make `rope_factor` a `float`.

### 3.13 P1-12 - two independent sources of `max_seq_len`

**Files:** `v4_model_resources.hpp`, `v4_layer.hpp`

`allocate_rope_cache` sizes the cos/sin caches `max_seq_len * half_rope * 4B`,
while `V4Layer::init_with_loader` takes its own `max_seq` defaulting to **4096**.
If the two ever differ, the RoPE kernels index `cos[pos*half_rope + k]` past the
end of an undersized table - a silent OOB read producing position-dependent noise.
Derive both from one value; assert equality at startup.

---

## 4. STAGE 2 - Numerical bisection (the decisive step)

**Do not apply any Section 5 item before this trace exists.** Two or more
convention errors can partially cancel; only a tensor-level reference diff
separates them.

Freeze the configuration:

```
- all 6 experts pinned hot, no eviction
- deterministic_expert_accumulation_ = true
- batched prefill disabled; prefill() -> step() only
- greedy argmax, single prompt, single token
```

Dump `attention_trace` at **layer 0, position 0** and diff against a PyTorch
reference forward pass for the same token, in this order:

```
 1. token embedding
 2. HC pre-mix output      (d_res_in after hc_pre_combine)   <- P2-4
 3. attention RMSNorm output
 4. q / k before RoPE
 5. rotated_query                                            <- P2-1
 6. attention logits (pre-softmax)                           <- P2-3
 7. attention_output                                         <- P0-2
 8. wo projection
 9. HC post-mix (attn)                                       <- P2-5
10. FFN RMSNorm output
11. router_logits
12. routed_expert_indices / routed_expert_weights            <- P0-1, P2-9
13. shared_expert_output
14. expert hidden (post-SwiGLU)                              <- P2-6
15. moe_output
16. post_ffn_residual
```

**The first tensor that diverges by more than FP16 epsilon names the bug.**

Layer 0 is a `Sliding` layer: it exercises RoPE, sliding attention, HC, hash
routing and the MoE without touching any compressor state - the smallest possible
failing unit. Only once layer 0 / position 0 matches, move to layer 2 (CSA) with
>= 8 tokens so a compressed entry actually materialises.

---

## 5. STAGE 3 - P2 convention questions (require the reference implementation)

In every case the code is internally consistent, so no test inside Aeon can detect
the error. Work them in the order the Stage 2 trace fails, not in table order.

| # | Question | Files | Why it matters |
|---|---|---|---|
| P2-1 | **RoPE pairing: interleaved or split-half?** All kernels use GPT-J interleaved pairs `(nope+2k, nope+2k+1)` with `freq_k = theta^(-2k/rope_dim)`. HF DeepSeek uses `rotate_half` (split-half: `(i, i+rope_dim/2)`). Unless the converter permutes `wq_b`/`wkv` rows into interleaved order, every rotary pair is mismatched. | `v4_attention.hpp`, converter | The most common cause of "loads fine, runs fine, output is grammatical garbage". Cheap test: switch the kernel to split-half for one run. |
| P2-2 | **Does the reference apply inverse RoPE to the attention output at all, and are cached values rotated?** | `v4_pipeline.hpp` | Decides whether the P0-2 fix is "delete the kernel" or "rotate the values". |
| P2-3 | **YaRN `mscale`:** is it `0.1*ln(f)+1`, applied once to logits or twice (q and k)? | `v4_attention.hpp` | 28% error in every attention temperature. |
| P2-4 | **HC stream initialisation.** `step()` replicates the same embedding into all 4 HC streams; after the layer-0 pre-mix `sum_j sigmoid(...)*residual[j]` the layer-0 input is ~2-4x the intended magnitude. Does the reference `HyperConnection.expand` replicate, expand-with-zeros, or use a learned per-stream init? | `v4_pipeline.hpp`, `hc_sinkhorn.hpp` | One-line difference, catastrophic depth-compounding effect. |
| P2-5 | **Sinkhorn orientation.** `hc_sinkhorn_normalize_kernel` row-softmaxes then column-normalises; `hc_post_kernel` consumes it transposed (`cm[hci*4 + hco]`). Verify `alpha = 2.0` and both `eps` placements. | `hc_sinkhorn.hpp` | Row-vs-column is a coin flip that looks plausible either way. `alpha = 2.0` is not in `config.json` at all - where did it come from? |
| P2-6 | **Quant nibble order / zero-point.** Kernel does `w = (nibble - 8) * scale` via the `0x64006400` / `-1032.0f` trick, group 32, nibble j = element j. `config.json` confirms `num_bits=4, symmetric=true, group_size=32`, but compressed-tensors `pack-quantized` stores signed int4 in `[-8,7]`. If the converter omitted the +8 offset, every weight is off by 8 LSBs. | `aeon_w4a16_swizzle.hpp`, converter | Noise-shaped expert output while everything else looks healthy. |
| P2-7 | **APE table shapes.** `v4_save_compressor_state_kernel` computes `ape_offset = (position % ratio) * width`, implying `[4, 1024]` (CSA) and `[128, 512]` (HCA). Tensor *names* are confirmed correct in `v4_dense_weight_binding.hpp`; shapes are unvalidated - subsumed by P0-6. | `v4_pipeline.hpp`, `v4_dense_weight_binding.hpp` | Silent garbage in all non-sliding layers. |
| P2-8 | **Compressor segment ordering.** `window = coefficient * ratio` (8 CSA, 128 HCA), `segment = offset / ratio`. Is segment 0 the older or newer half? Is `rope_position = (boundary/ratio)*ratio` the intended anchor? | `v4_attention.hpp` | Reverses the temporal order of compressed memory. |
| P2-9 | **Hash-routing renormalisation.** In hash mode (layers 0-2) the router still renormalises the 6 raw scores and multiplies by 1.5. Does the reference? | `moe_router.hpp` | If not, layers 0-2 are mis-scaled by a per-token-varying factor. |
| P2-10 | **Chat template.** Verify the `>` vs `>=` asymmetry in the thinking-mode branches. With `drop_thinking = false` in Thinking mode, an assistant turn with empty `reasoning_content` emits a bare `</think>` with nothing before it. | `dsv4_chat_formatter.cpp` | A stray `</think>` derails a reasoning model badly. |

### P0-2 detail - inverse RoPE on unrotated values

**Files:** `v4_pipeline.hpp` (`step()`, `prefill_batched_chunk()`), `v4_attention.hpp`

The pipeline stores the **pre-RoPE** `d_kv_norm_act` into `d_local_value_cache` and
the **post-RoPE** vector into `d_local_key_cache`, so attention output is
`sum_j a_j * v_j` over *unrotated* values. It then launches:

```cpp
v4_inverse_rope_at_pos_wave32_kernel(attn_out, cos, sin, /*pos=*/position, ...);
```

A single inverse rotation by `pos` can only undo a rotation actually applied by
`pos`; here nothing was applied. RoPE dims `[448, 512)` of every head get a
spurious position-dependent rotation on **all 43 layers**, accumulating with depth
and position - the exact signature of "locally fluent, globally incoherent".

- *Option A (absolute)*: delete the inverse-RoPE launch. Keys rotated, values not,
  output in the absolute frame. **Almost certainly what the reference does for
  MLA-style decoupled RoPE.**
- *Option B (relative)*: cache the rotated vector as the value too, making the
  inverse rotation meaningful.

Settle via P2-2 before changing.

---

## 6. STAGE 4 - Tokenizer golden test (~0.5 day)

Encode >= 1000 strings (English, German with umlauts, Cyrillic, Arabic, CJK, emoji,
code, whitespace-heavy) and diff against HuggingFace `tokenizers`. This eliminates
or confirms an entire class of causes (P3-8 .. P3-11) in half a day.

---

## 7. P3 - Robustness, diagnostics, performance

| # | Finding | File |
|---|---|---|
| P3-1 | `s_scores[256]` / `s_choice[256]` hardcoded while `n_routed_experts` is a runtime parameter - silent shared-memory overflow if ever != 256. Template or guard it. | `moe_router.hpp` |
| P3-2 | Top-k selection is serial on thread 0: 6x256 iterations, 63 threads idle, x43 layers x N tokens. Use a wave-level reduction. | `moe_router.hpp` |
| P3-3 | `v4_gemv_fp16_kernel` writes `y[token * gridDim.x + out_col]` - row stride implicitly the launch geometry. Correct at all current call sites; pass `out_dim` explicitly. | `v4_attention.hpp` |
| P3-4 | `prefill_batched_chunk` does `std::swap` on the `d_res_in`/`d_res_out` **members**; with 43 layers (odd) they stay exchanged after the call. Safe only while both buffers are identically sized and unaliased. Use a local pointer pair. | `v4_pipeline.hpp` |
| P3-5 | Indexer materialisation aliases key and value (`compressed_key == compressed_value == d_indexer_key_cache`): two writes to the same address from one thread. Benign, hides intent. | `v4_pipeline.hpp` |
| P3-6 | `encode` re-runs `find` over the entire remaining text for every added token on every iteration - near-quadratic. Use Aho-Corasick or cache the next match position. | `dsv4_tokenizer.cpp` |
| P3-7 | `decode` does a linear `find_if` over `added_tokens_` per token. Build a `std::vector<int32_t> id_to_added_`. | `dsv4_tokenizer.cpp` |
| P3-8 | `is_letter_or_mark` returns true for essentially every code point >= 0x00C0 that is not space/CJK/general-punctuation - sweeping in Cyrillic, Arabic, Hebrew, Hangul, emoji. The real class is `\p{L}\p{M}`. German umlauts are fine; Cyrillic/Arabic/emoji prompts are suspect. | `dsv4_tokenizer.cpp` |
| P3-9 | The `" " + letters` pre-token rule handles only U+0020, not other space classes. | `dsv4_tokenizer.cpp` |
| P3-10 | A code point matching no class falls through as a single-char pre-token, silently. Add a counter/log. | `dsv4_tokenizer.cpp` |
| P3-11 | `is_letter_or_mark` and `is_punctuation_or_symbol` are mutually exclusive by construction, so the pre-tokenizer can never fail - and never report a mismatch. Only a golden-file test can detect divergence. | `dsv4_tokenizer.cpp` |
| P3-12 | No `hipGetLastError()` after any kernel launch anywhere in the pipeline. | `v4_pipeline.hpp` |
| P3-13 | `aeon_moe_fused_w2_accum_kernel` resets `counters[blockIdx.x] = 0` at the end. Correct; add a one-time assert that the buffer was memset before the first call. | `aeon_moe_fused_w2.hpp` |
| P3-14 | If a token ever routes to zero experts, the shared-expert contribution and the FP32->FP16 finalisation both vanish and the row keeps its stale value. Assert `expert_count >= 1`. | `aeon_moe_fused_w2.hpp` |
| P3-15 | `generate_token_ids` appends the EOS token to `result.token_ids` before returning; callers must strip it or it will be detokenised into the visible output. | `text_generation.cpp` |
| P3-16 | `dispatch_aeon_w4a16_swizzled_dual_gemv` is indented ~20 spaces, breaking file formatting. Cosmetic. | `aeon_w4a16_swizzled_gemv.hpp` |

---

## 8. Investigated and CLEARED - do not re-open

- **Router gating function.** `sqrt(softplus(x))` looked wrong (V3 uses sigmoid),
  but `config.json` declares `scoring_func = "sqrtsoftplus"`. **Correct.**
- **Router bias handling.** Bias enters `scores_for_choice` only; weights come from
  unbiased `scores`; renormalise-then-scale order is right. Matches
  `topk_method = "noaux_tc"`, `norm_topk_prob = true`. **Correct.**
- **SwiGLU clamp.** `swiglu_limit = 10.0` is declared in `config.json`; the
  asymmetric clamp (gate from above, up from both sides) is applied consistently in
  `aeon_swiglu_clamped`, `v4_pipeline_swiglu_clamp_kernel` and
  `v4_pipeline_swiglu_clamp_batched_kernel`. **Correct** (value should still be read
  from config - P0-5).
- **W4A16 swizzle round trip.** `permute_w4a16_word` places source nibble `2p` at
  destination `p` and `2p+1` at `p+4`; `unpack2(word, pair)` recovers
  `half2{elem 2p, elem 2p+1}`, dotted against
  `activation[group*32 + word*8 + 2p .. +1]`. **Self-consistent and correct.**
  `swizzle_w4a16` / `unswizzle_w4a16` are exact inverses.
- **Expert memory layout.** `swizzled_expert_format.hpp` offsets and sizes are
  internally consistent: W1 4,194,304 + 524,288 -> 4,718,592; W2 +4,194,304 ->
  8,912,896, +524,288 -> 9,437,184; W3 +4,194,304 -> 13,631,488; total 14,155,776 B.
  Group size 4096/128 = 32 and 2048/64 = 32 both match the kernel. **Correct.**
- **Template parameters.** W13 `<WAVES=8, RPW=4, LPR=8, ITERS=16>` -> K = 4096 = hidden,
  N = 2048 = `moe_intermediate_size`, N % RPW = 0. W2 `<8,8,4,16>` -> K = 2048,
  N = 4096. **Correct.** (W13 `N` == W2 `K` is structural; add a `static_assert`
  anyway, but there is no bug.)
- **Expert-slot <-> top-k weight ordering** (previously suspected major bug).
  `V4ExpertSupplyCoordinator::dispatch_layer_prefetch` builds `requests` in
  `topk_indices` order; `sync_state` copies `transfers[index]` -> `vram_slots[index]`;
  the pipeline builds `fused_w2.w2[k]` from `pending_transfers[k]` and the kernel
  reads `topk_weights[expert = blockIdx.y]` from the same rank-ordered buffer.
  **Correct - indices align.**
- **`aeon_moe_fused_w2_accum_kernel` last-block election.** The `atomicAdd` counter is
  incremented only after each block's `atomicAdd` into `output_f32`, with a
  preceding `__threadfence()`, so the elected last block observes all six
  contributions. `initial_output` and `output_f16` aliasing `d_moe_accum` is safe for
  the same reason. **Correct, though fragile - document the invariant.**
- **`d_swizzled_counters` sizing.** 64 `int32_t`, grid.x = (4096/8)/8 = 64. **Exact.**
- **W13 `output_f32` zeroing coverage.** 64 blocks x 256 threads = 16,384 >= 4096.
  **Adequate** (still move it to the call site - P1-7).
- **`M_PAD` activation replication.** Row 0 replicated to 16 rows for WMMA; the W13
  kernel reads only row 0. Wasteful but **correct**.
- **Tokenizer byte-level alphabet.** The `33-126 / 161-172 / 174-255` direct set plus
  `256 + extra` overflow mapping matches the standard GPT-2 byte-to-unicode table
  exactly; `symbol_to_bytes` is its exact inverse. **Correct.**
- **Tokenizer BPE merge loop.** Lowest-rank-first pair merging with linear rescan is
  the standard algorithm. Slow but **correct.**
- **Digit pre-token grouping.** Runs of up to 3 digits matches `\p{N}{1,3}`. **Correct.**
- **Compressed vs standard RoPE table selection.** The pipeline correctly picks
  `d_compressed_cos_cache` / `d_compressed_sin_cache` for non-`Sliding` layers,
  consistent with `compress_rope_theta = 160000` vs `rope_theta = 10000`. **Correct.**
- **P1-8 `positions` reset sentinel - CLEARED.** `V4Layer::reset_generation_state()`
  clears all four position arrays with `clear_state_buffer(..., 0xFF)`, i.e.
  `int64_t = -1`, and `allocate_state()` calls it at the end of every path. No
  cross-conversation contamination.
- **P0-3 compressed-cache bound - DOWNGRADED to P3.**
  `compressed_capacity = ceil(max_seq / ratio)`; the kernel's
  `compressed_index = (pos+1)/ratio - 1` maxes at `max_seq/ratio - 1`;
  `record_position()` throws if `position >= max_seq_len_`. The write is in bounds.
  Keep a device-side guard as cheap insurance. **Caveat:** the bound holds only if
  `V4Layer::max_seq_len_` equals the context the pipeline actually drives - see P1-12.
- **Generation loop - CLEARED.** `generate_token_ids` prefills sequentially via
  `step(token, index, true)`, takes the first generated token from the last prompt
  token, and indexes decode positions as `prompt.size() + generated - 1`. Position
  arithmetic and EOS handling are correct. Only nit: P3-15.

---

## 9. Coverage and residual risk

**Reviewed (20 files):** `v4_pipeline.hpp`, `v4_attention.hpp`, `hc_sinkhorn.hpp`,
`moe_router.hpp`, `v4_pipeline_ops.hpp`, `aeon_w4a16_swizzle.hpp`,
`aeon_w4a16_swizzled_gemv.hpp`, `aeon_moe_fused_w13.hpp`, `aeon_moe_fused_w2.hpp`,
`swizzled_expert_format.hpp`, `vram_expert_pool.hpp`, `v4_expert_supply.hpp`,
`config.hpp`, `dsv4_tokenizer.cpp`, `dsv4_chat_formatter.cpp`, `v4_layer.hpp`,
`v4_layer_state.hpp`, `v4_dense_weight_binding.hpp`, `v4_model_resources.hpp`,
`text_generation.cpp`, plus the dossier.

**Not reviewed:** `v4_model_spec.hpp`, `v4_model_contract.hpp`,
`v4_pipeline_scratch.hpp`, `device.hpp`, `aeon_chat.cpp`, `dsv4_tokenizer.hpp`,
`dsv4_chat_formatter.hpp`, `text_generation.hpp`, `CMakeLists.txt`,
`project-description.md`.

**The weight-conversion script was never supplied and is the single largest blind
spot.** It governs P0-6 (dtype), P2-1 (RoPE row permutation) and P2-6 (int4
zero-point) - three of the highest-ranked open items. Review it next.

Remaining residual risk in unreviewed code:

1. `v4_model_spec.hpp` - the per-layer `head_dim` / `sliding_window` / `index_*`
   values that all of `V4LayerStateLayout`'s validation depends on.
2. `v4_pipeline_scratch.hpp` - scratch buffer sizing vs the kernels' assumed extents.
3. `aeon_chat.cpp` - whether it uses the 2-arg `V4ModelResources::initialize`
   overload (P0-7) and which prefill path it drives (P3-4).
