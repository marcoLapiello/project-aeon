# Expert Kernel Review Answers

**Scope:** Project Aeon source and the converted DeepSeek-V4 Flash model in this workspace. Values marked **not implemented** or **not measured** should not be treated as kernel-design constraints.

**Live measurement refresh:** 2026-09-10, repository-root runs on GPU 0. Results are workload-specific; historical ledger values remain identified as such.

## 1. Model Geometry

| Question | Answer |
|---|---|
| Hidden and intermediate sizes | `hidden_size=4096`. `intermediate_size` is absent; there is no dense-only FFN in the current model/runtime. `moe_intermediate_size=2048` for both shared and routed experts. |
| Layers and dense prefix | `num_hidden_layers=43`; `0` dense-only prefix layers. All 43 layers run shared plus routed MoE. Layers `0-2` use hash routing; layers `3-42` use score-based routing. |
| Experts and routing | `n_routed_experts=256`, `n_shared_experts=1`, `num_experts_per_tok=6`. `scoring_func=sqrtsoftplus`, `routed_scaling_factor=1.5`, `norm_topk_prob=true`, `topk_method=noaux_tc`. Layers `3-42` have a learned FP32 gate bias; no sigmoid router is used. No explicit `n_group`/`topk_group` or other grouped-routing field is present. |
| Attention | MLA-style implementation: `num_attention_heads=64`, `num_key_value_heads=1`, `q_lora_rank=1024`, `qk_rope_head_dim=64`, `qk_nope_head_dim=448` (derived), `v_head_dim=512` in the implementation, `o_lora_rank=1024`, `o_groups=8`. `kv_lora_rank` is not a config key; the current `wkv` projection and KV cache width are both `512`, which is the implementation contract. |
| MTP | The config declares `num_nextn_predict_layers=1`. The current loader/pipeline has no MTP-head execution path; inference uses one LM head and greedy single-token generation. |
| RoPE | Model config: YaRN, `theta=10000`, `factor=16`, `original_max_position_embeddings=65536`, `beta_fast=32`, `beta_slow=1`, model maximum `1048576`. Current pipeline initialization passes `factor=1.0` to `RopeTable::init`, so the configured YaRN factor is not currently applied. |
| Vocabulary and embeddings | `vocab_size=129280`; `tie_word_embeddings=false`. `embed.weight` and `head.weight` are separate tensors. |
| NSA/sparse/indexed attention | Config metadata contains indexed/compressed-attention fields (`index_head_dim=128`, `index_n_heads=64`, `index_topk=512`, `compress_ratios`, and `dspark_*`). The current runtime does not implement that path; it uses the 512-wide cached sliding-window attention kernel with `window=128`. Compressed/indexed attention remains an open correctness gate. |

## 2. Exact Quantization Layout

| Question | Answer |
|---|---|
| Quantization | 4-bit integer, symmetric, group-wise, `group_size=32`, static (`dynamic=false`). The source config labels the expert dtype `fp4`, but the compressed-tensors metadata and Aeon runtime treat it as symmetric INT4. |
| Zero point and scales | No stored zero-point. Dequantization uses `(nibble - 8) * scale`. Source metadata leaves `scale_dtype`/`zp_dtype` null; the serialized layout and all current consumers use FP16 scales. |
| Matrix orientation | Row-major `[out, in]`; the operation is `A @ W^T`. `W1/W3` are `[2048,4096]`; `W2` is `[4096,2048]`. |
| Packed shape | `W1/W3`: `[2048,512]` `uint32` words and `[2048,128]` FP16 scales. `W2`: `[4096,256]` `uint32` words and `[4096,64]` FP16 scales. |
| Nibble order | Eight sequential 4-bit nibbles per `uint32`; weight `k` uses `(word >> (4 * (k % 8))) & 0xf`. This is low-nibble-first sequential packing. No AWQ permutation or other cross-group permutation is assumed by the runtime. |
| Activation and accumulation dtypes | Activations are FP16. The decode GEMV dequantizes inline and accumulates in FP32, then writes FP16. The WMMA path stages dequantized FP16 weights and uses FP32 WMMA accumulators, then writes FP16. |
| On-disk expert payload | One expert is `14,155,776` bytes (`3,456` 4096-byte sectors): `W1_packed`, `W1_scale`, `W2_packed`, `W2_scale`, `W3_packed`, `W3_scale`. The payload is contiguous and byte-identical between NVMe, host staging, and the unified VRAM slot. |
| Repacking freedom | Yes, offline repacking is possible. The current converter only concatenates the source packed tensors and verifies bit identity; it does not create a kernel-oriented swizzle. A new swizzle would require a new converter/index contract plus updates to the loader, host pool, VRAM pool, and kernel. |

## 3. Current Bottleneck Evidence

### Direct answer

There is **no committed rocprof per-kernel decode/prefill breakdown** separating expert GEMV/GEMM, dense GEMM, attention, router/top-k, and transfer wait time. The repository has isolated benchmarks and end-to-end ledger entries, but not a complete per-step trace.

### Available measurements

| Area | Evidence |
|---|---|
| Routed expert W4A16 | Dedicated live test: `13.3261 us` for `W1/W3` (`N=2048,K=4096`) and `13.9223 us` for `W2` (`N=4096,K=2048`), with max error `0` versus the CPU reference. The `M=16,N=2048,K=4096` WMMA path measured `137.457 us`; the historical decode path was `138.8-140.34 us`. Full-model routed-GEMM time fell from about `108 ms` to `11 ms` per token in the recorded A/B. |
| Attention | Cached sliding-window test: `41.91 us` for 16 tokens, or `2.62 us/token`. This is an isolated kernel test, not a full decode/prefill split. |
| Router/top-k | CPU/GPU assignment parity is tested, but no isolated router latency is recorded. |
| Dense projections | Vectorized FP16 GEMV is implemented for router, shared experts, and LM head. No isolated dense-kernel timing is recorded; the LM head reads about `1.06 GB/token`. The separate generic FP16 `2048^3` WMMA benchmark measured `0.868 ms` and `19.782 TFLOP/s`, with max error `0.000253677`; this is not the W4A16 decode GEMV. |
| Storage and transfer | Current live direct-I/O shape run: forced async 4 MiB subreads reached `3.31 GiB/s` for both contiguous and scattered six-expert sets; whole-expert requests reached `1.72 GiB/s` contiguous and `1.61 GiB/s` scattered. Historical measurements remain `3.17 GiB/s` for six model-backed reads and about `5.88 GiB/s` for a sequential model read. The current pinned PCIe copy measured `24.5951 GB/s`; concurrent WMMA reported `-9.954%` compute overhead versus its isolated run. |
| End to end | Current live runs: the 2-layer hot dynamic pool reached `239.70 tok/s` short and `237.96 tok/s` medium with `100%` hot hits; the 43-layer benchmark with 35 GiB warm host reached `4.93 tok/s`, TTFT `1156.68 ms`, `62.7%` hot hits, `208` warm hits, and `850` cold misses. The 12-slot direct-I/O versus mmap A/B measured `36.36` versus `13.09 tok/s` with identical tokens; this result is page-cache/order-sensitive. The latest ledger native text turn remains TTFT `4084.13 ms`, decode `3.04 tok/s`. Earlier controlled full-model A/Bs identified the cold just-in-time path as dominant. |

### Bottleneck conclusion

For the last instrumented full-model runs, the critical path was cold NVMe/direct-I/O dispatch plus host-to-device transfer wait, approximately `3.5 ms/layer` with about `2.4` novel experts/layer. The W4A16 rewrite reduced routed GEMM time by roughly `97 ms/token`, but end-to-end latency remained transfer/dispatch-bound. The current path executes six routed experts sequentially, with three projection dispatches per expert; there is no gather-into-batched-GEMM path.

## 4. Regime and Batching

| Question | Answer |
|---|---|
| Batch/concurrency target | Single GPU, one sequence, logical batch size 1. Decode is one token per `V4Pipeline::step`; no multi-sequence serving or production batching API is present. |
| Prefill | `generate()` calls the same single-token `step()` once per prompt token. Scratch is padded to `M_PAD=16` for WMMA compatibility, but the current routed decode launch uses `M=1` GEMV. There is no true batched prefill path in the current pipeline. |
| Measured prompts and context | Recorded benchmarks commonly use 4 prompt tokens -> 8 generated tokens; component tests use 3- or 4-token prompts. Runtime default context is `4096`; full-model text checks used `512` and `1024`, and the model advertises `1048576`. Attention uses a `128`-token sliding window. |
| KV cache | FP16, contiguous per layer, shape `[max_seq_len,512]`, one allocation in `V4Layer::d_kv_cache`. It is not paged. The current attention kernel consumes `q[64,512]`, the cached `[position,512]` latent, and treats `V=K` in the weighted-sum path. |

## 5. Environment

| Question | Answer |
|---|---|
| ROCm/compiler | ROCm `7.2.2`, native `/usr/bin/hipcc`, C++20, Linux testbed. CMake uses `/opt/rocm`. |
| GPU target | AMD Radeon RX 7900 XTX / Navi 31, `gfx1100`; the host has four cards, but the current measurements are single-GPU runs. |
| WMMA API | Yes, RDNA3 WMMA is already used and silicon-validated through `rocwmma::fragment`/`mma_sync` in `w4a16_gemm.hpp` and the WMMA tests. The code does not use `__builtin_amdgcn_wmma_*` directly. The current `M=1` decode path is GEMV; the WMMA kernel remains for `M>1`. |
| Wave mode | Wave32 is enforced by `-mno-wavefrontsize64`; kernels use 32-thread blocks/wave operations. |
| Occupancy/LDS data | No rocprof/occupancy report or LDS bank-conflict measurement is committed. |
| Harness availability | Yes. `test_w4a16_wmma` provides CPU-reference correctness checks, including representative expert shapes and a real-checkpoint check; `bench_wmma_gemm` and `bench_async_overlap` provide timing/overlap harnesses. There is no automated tile/layout autotune sweep yet. |

## 6. Current Interfaces

### Expert weights

There is no standalone expert descriptor struct in production. The byte layout is the descriptor contract:

```cpp
struct ExpertLocation {
    uint64_t file_offset;
    size_t byte_length;
};

const uint8_t* AeonModelLoader::get_expert_data(
    uint32_t layer_id, uint32_t expert_id) const;

uint32_t* UnifiedVRAMExpertPool::get_w1_packed(uint32_t slot);
half*     UnifiedVRAMExpertPool::get_w1_scale(uint32_t slot);
uint32_t* UnifiedVRAMExpertPool::get_w2_packed(uint32_t slot);
half*     UnifiedVRAMExpertPool::get_w2_scale(uint32_t slot);
uint32_t* UnifiedVRAMExpertPool::get_w3_packed(uint32_t slot);
half*     UnifiedVRAMExpertPool::get_w3_scale(uint32_t slot);
```

`upload_from_host_expert(slot, payload, stream)` uploads one complete `14,155,776`-byte slot with one asynchronous copy. Host warm slots expose the same six subviews and contiguous payload.

### KV cache and attention

```cpp
half* d_kv_cache;       // [max_seq_len, 512], one allocation per layer
uint32_t max_seq_len_;

__global__ void v4_cached_sliding_window_attn_wave32_kernel(
    const half* q,             // [64, 512]
    const half* kv_cache,      // [max_seq_len, 512]
    const float* attn_sink,    // [64]
    half* out,                 // [64, 512]
    int current_pos,
    int window_size,
    float scale);
```

### MoE FFN launch path

```cpp
inline void dispatch_w4a16_gemm(
    const half* d_a,
    const uint32_t* d_w_packed,
    const half* d_w_scale,
    half* d_out,
    uint32_t M, uint32_t N, uint32_t K,
    hipStream_t stream = 0);
```

For each selected expert, the pipeline currently performs:

1. Router output: `float logits[256]` -> `float topk_weights[6]` and `int32_t topk_indices[6]`.
2. `W1`: `M=1,N=2048,K=4096`; `W3`: `M=1,N=2048,K=4096`.
3. Clamped SwiGLU with limit `10.0`.
4. `W2`: `M=1,N=4096,K=2048`.
5. Weighted accumulation into `[1,4096]`.

The active streams are `compute_stream`, `sdma_stream` for warm/safetensors uploads, and `sdma_cold_stream` for cold I/O uploads. The prefetch arena has 12 staging slots (two six-expert horizons).

## 7. Kernel File Organization

Production kernels are header-defined. There are no standalone production kernel implementation files under `src/kernel` or separate kernel library; the five headers under `src/kernel` are compiled through the HIP C++ targets that include them. The production tree contains 26 `__global__` kernels.

- [src/kernel/w4a16_gemm.hpp](../../../src/kernel/w4a16_gemm.hpp): `wmma_fused_int4_gemm_kernel` for `M>1` W4A16 GEMM, `w4a16_gemv_kernel` for single-token decode, and the inline `dispatch_w4a16_gemm()` selector.
- [src/kernel/v4_attention.hpp](../../../src/kernel/v4_attention.hpp): Wave32 RMSNorm, batched and single-position RoPE, cached and non-cached sliding-window attention, grouped `W_o_a`, scalar and vectorized FP16 GEMV, FP16-to-FP32 conversion, GPU argmax reduction, and HC head reduction.
- [src/kernel/moe_router.hpp](../../../src/kernel/moe_router.hpp): `moe_router_kernel` for hash routing and SqrtSoftplus top-6 routing, plus the `softplus_sqrt()` device helper and CPU reference router.
- [src/kernel/hc_sinkhorn.hpp](../../../src/kernel/hc_sinkhorn.hpp): `hc_project_kernel`, `hc_pre_combine_kernel`, `hc_sinkhorn_normalize_kernel`, and `hc_post_kernel` for Hyper-Connections.
- [src/kernel/v4_pipeline_ops.hpp](../../../src/kernel/v4_pipeline_ops.hpp): clamped SwiGLU, weighted expert accumulation, and FP16/FP32 conversion kernels.

### Launch and ownership layers

- [src/core/v4_pipeline.hpp](../../../src/core/v4_pipeline.hpp) contains the host-side `V4Pipeline::step()` orchestration and all production launches; it does not define device kernels.
- [src/core/v4_layer.hpp](../../../src/core/v4_layer.hpp) owns per-layer weights and the persistent `[max_seq_len,512]` KV cache.
- [src/core/v4_pipeline_scratch.hpp](../../../src/core/v4_pipeline_scratch.hpp) owns reusable activation, router, expert, logits, and argmax buffers.
- [src/core/vram_expert_pool.hpp](../../../src/core/vram_expert_pool.hpp) owns contiguous resident expert slots; [src/core/prefetch_staging.hpp](../../../src/core/prefetch_staging.hpp) owns the 12-slot pinned transfer arena.

The runtime sequence in `V4Pipeline::step()` is:

1. HC attention pre-mix, Sinkhorn, and pre-combine.
2. Attention normalization and MLA projections.
3. RoPE application, KV-cache insertion, cached sliding-window attention, and output projection.
4. HC attention post-expansion.
5. HC FFN pre-mix and normalization.
6. Router GEMV, FP16-to-FP32 conversion, and top-6 selection.
7. Shared FP16 expert execution.
8. Expert prefetch followed by six routed W4A16 executions: `W1`, `W3`, SwiGLU, `W2`, and weighted accumulation.
9. HC FFN post-expansion.
10. HC head reduction, final RMSNorm, LM-head GEMV, and GPU argmax.

### Test-only kernel definitions

These files define local validation or benchmark kernels; they are not production pipeline implementations:

- [tests/test_w4a16_wmma.cpp](../../../tests/test_w4a16_wmma.cpp): `wmma_fused_int4_tile_kernel`.
- [tests/test_wmma_tile.cpp](../../../tests/test_wmma_tile.cpp): `wmma_single_tile_kernel`.
- [tests/bench_wmma_gemm.cpp](../../../tests/bench_wmma_gemm.cpp) and [tests/bench_async_overlap.cpp](../../../tests/bench_async_overlap.cpp): benchmark-local FP16 WMMA kernels.
- [tests/test_swiglu_clamp.cpp](../../../tests/test_swiglu_clamp.cpp): local RMSNorm and SwiGLU kernels.
- [tests/test_toy_moe_layer.cpp](../../../tests/test_toy_moe_layer.cpp): local toy expert GEMM kernel.
- [tests/test_v4_moe_layer.cpp](../../../tests/test_v4_moe_layer.cpp): local SwiGLU and expert-accumulation kernels.

## Known Design Caveats

- `kv_lora_rank`, dense `intermediate_size`, scale dtype, and grouped-routing counts are not independently represented in the source config; the values above use the current tensor/runtime contract where available.
- MTP and compressed/indexed attention are model-config declarations, not current execution features.
- The configured YaRN factor is `16`, but current pipeline RoPE initialization uses `1.0`.
- A new kernel design should be validated against the existing FP32-reference tests and a new decode/prefill rocprof capture before replacing the current path.

## Sources

- [Model config](../../../models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/config.json)
- [Aeon loader and expert layout](../../../src/core/aeon_loader.hpp)
- [Converter and serialization contract](../../../scripts/convert_safetensors_to_aeon.py)
- [W4A16 kernels](../../../src/kernel/w4a16_gemm.hpp)
- [Pipeline and MoE launch path](../../../src/core/v4_pipeline.hpp)
- [Attention kernels and KV layout](../../../src/kernel/v4_attention.hpp)
- [Performance ledger](../../status/PERFORMANCE_LEDGER.md)
- [Build target and Wave32 flags](../../../CMakeLists.txt)
