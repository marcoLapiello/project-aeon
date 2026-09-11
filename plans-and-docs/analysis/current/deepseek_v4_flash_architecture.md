# DeepSeek-V4-Flash: Model Architecture and Inference Logic

## Scope

This document consolidates the model-architecture information identified for the publicly described DeepSeek-V4-Flash family. It focuses exclusively on model geometry, computation, attention, caching, routing, precision, and inference semantics. It intentionally excludes hardware-specific deployment details, execution recommendations, and testbed-specific considerations.

> **Checkpoint note:** DeepSeek-V4-Flash preview and later releases such as DeepSeek-V4-Flash-0731 should not be assumed to be tensor- or serving-compatible. The selected checkpoint's `config.json`, model implementation, tokenizer, and tensor metadata are authoritative where they differ from this summary.

## 1. Model overview

DeepSeek-V4-Flash is a causal decoder-only sparse mixture-of-experts language model.

| Property | Publicly described value |
|---|---:|
| Total parameters | Approximately 284B |
| Activated parameters per token | Approximately 13B |
| Decoder layers | 43 |
| Hidden size | 4096 |
| Vocabulary size | 129,280 |
| Maximum position length | 1,048,576 tokens |
| Attention heads | 64 |
| Key/value heads | 1 |
| Attention head dimension | 512 |
| Routed experts per MoE layer | 256 |
| Shared experts per MoE layer | 1 |
| Routed experts selected per token | 6 |
| Expert intermediate size | 2048 |
| Hyper-connection streams | 4 |
| MTP/next-token prediction layers in the preview configuration | 1 |

The model combines four major architectural ideas:

1. **Hybrid attention:** sliding-window attention, Compressed Sparse Attention (CSA), and Heavily Compressed Attention (HCA).
2. **Shared K/V attention:** 64 query heads use one shared key/value head.
3. **Manifold-constrained hyper-connections (mHC):** four residual streams are mixed using constrained stream-mixing matrices.
4. **Sparse MoE computation:** one shared expert plus six routed experts selected from 256 routed experts.

## 2. Decoder-layer schedule

The public `compress_ratios` schedule is:

```text
[0, 0,
 4, 128, 4, 128, 4, 128, 4, 128,
 4, 128, 4, 128, 4, 128, 4, 128,
 4, 128, 4, 128, 4, 128, 4, 128,
 4, 128, 4, 128, 4, 128, 4, 128,
 4, 128, 4, 128, 4, 128,
 0]
```

Interpreted by layer index:

| Layer indices | Count | Layer type | Compression ratio |
|---|---:|---|---:|
| 0–1 | 2 | Sliding-window attention | Not applicable |
| 2, 4, ..., 40 | 20 | Compressed Sparse Attention | 4 |
| 3, 5, ..., 41 | 20 | Heavily Compressed Attention | 128 |
| 42 | 1 | Sliding-window attention | Not applicable |

The alternating CSA/HCA pattern is part of the model architecture. Attention-layer state is therefore not homogeneous across all 43 decoder blocks.

## 3. Hidden-state and residual geometry

### 3.1 Base hidden state

The model width is 4096:

```text
hidden state: [batch, sequence, 4096]
```

### 3.2 Four-stream hyper-connections

The mHC representation maintains four residual streams:

```text
hyper-connected state: [batch, sequence, 4, 4096]
```

Equivalently, a token is represented by four vectors of width 4096. Attention and MoE sublayers operate within this hyper-connected residual structure, with learned pre-mixing and post/combine operations.

### 3.3 Stream mixing and Sinkhorn normalization

The hyper-connection matrices are constrained using Sinkhorn normalization. The public configuration specifies:

```text
hc_mult = 4
hc_sinkhorn_iters = 20
hc_eps = 1e-6
```

Sinkhorn normalization is used to produce constrained stream-mixing matrices with approximately normalized row and column behavior. The normalization is applied over the small stream dimension rather than over the model's 4096-wide feature dimension.

A conceptual layer flow is:

```text
four residual streams
        ↓
pre-mixing
        ↓
attention or MoE sublayer
        ↓
post/combine mixing
        ↓
four residual streams
```

The final hyper-head combines the four streams into the ordinary model-width representation used for final normalization and vocabulary projection.

## 4. Attention geometry

### 4.1 Multi-query K/V structure

The model has 64 query heads but one key/value head:

```text
Q: [batch, sequence, 64, 512]
K: [batch, sequence, 1, 512]
V: [batch, sequence, 1, 512]
```

The single K/V head is shared across all 64 query heads. The attention output still has 64 heads, each with 512 channels.

The total expanded query width is:

$$
64 \times 512 = 32{,}768
$$

### 4.2 Partial RoPE split

The 512-dimensional query/key head is divided into:

| Component | Width |
|---|---:|
| Non-RoPE component | 448 |
| RoPE component | 64 |
| Total head dimension | 512 |

Thus, for every query and key head:

```text
Q/K head = [non-RoPE 448 channels | RoPE 64 channels]
```

Rotary position encoding is applied only to the 64-dimensional RoPE component, not to all 512 channels.

The attention implementation also includes position-dependent handling of the attention output's RoPE-related component. A conventional implementation that applies RoPE only to ordinary full-width Q/K tensors is not equivalent to the described V4 attention path.

### 4.3 RoPE configuration

Publicly described configuration values include:

```text
rope_theta = 10000
compress_rope_theta = 160000
original_max_position_embeddings = 65536
max_position_embeddings = 1048576
rope_scaling.factor = 16
rope_scaling.beta_fast = 32
rope_scaling.beta_slow = 1
rope_scaling.type = yarn
```

The compressed-attention position encoding uses its own base value. Exact position transforms and frequency construction should follow the selected checkpoint's reference implementation.

## 5. Query and output projections

### 5.1 Low-rank query projection

The query path uses a low-rank intermediate dimension:

```text
q_lora_rank = 1024
```

Conceptually:

```text
hidden width 4096
    ↓
query down projection
    ↓
low-rank query representation [1024]
    ↓
query up projection
    ↓
64 × 512 query channels
```

### 5.2 Grouped low-rank output projection

The output path uses:

```text
o_lora_rank = 1024
o_groups = 8
```

The 64 heads are divided into eight groups:

```text
64 heads / 8 groups = 8 heads per group
```

A conceptual output path is:

```text
attention output [64, 512]
        ↓
8 groups × [8 heads, 512]
        ↓
grouped low-rank projection
        ↓
8 group representations of width 1024
        ↓
final output projection
        ↓
model width 4096
```

## 6. Attention mechanisms

Each decoder layer includes a local sliding-window component. The long-range component depends on the layer type.

### 6.1 Sliding-window attention

The public window size is:

```text
sliding_window = 128
```

At position `t`, the local causal window covers the current position and up to the preceding 127 positions:

```text
[max(0, t - 127), ..., t]
```

The exact inclusive boundary must follow the reference masking implementation.

Sliding-window layers are layers 0, 1, and 42.

### 6.2 Compressed Sparse Attention (CSA)

CSA layers use compression ratio 4 and a Lightning Indexer.

Publicly described CSA parameters:

```text
compression ratio = 4
index_n_heads = 64
index_head_dim = 128
index_topk = 512
```

Conceptual flow:

```text
hidden sequence
      ↓
overlapping compression windows of 4 tokens
      ↓
compressed K/V pool
      ↓
Lightning Indexer
      ↓
top-512 compressed entries per query
      ↓
sparse long-range attention
```

CSA combines two logical attention sources:

1. the most recent 128-token local window;
2. up to 512 selected entries from the compressed long-range pool.

The compressed windows are overlapping rather than simply non-overlapping blocks. Consequently, the exact number and causal availability of compressed entries depend on the reference compressor's boundary rules.

The indexer is a separate scoring and selection path. It is not equivalent to computing a full ordinary attention matrix and truncating it afterward. A query must be scored against eligible compressed entries, and only the selected top-k entries are gathered for sparse attention.

### 6.3 Heavily Compressed Attention (HCA)

HCA layers use compression ratio 128.

Conceptual flow:

```text
hidden sequence
      ↓
non-overlapping windows of 128 tokens
      ↓
one compressed entry per completed window
      ↓
causal attention over valid compressed entries
```

HCA does not use the CSA Lightning Indexer. Its long-range sequence is represented by one compressed entry per completed 128-token window.

For a context of 1,048,576 tokens, the approximate number of HCA entries is:

$$
\left\lceil \frac{1{,}048{,}576}{128} \right\rceil = 8192
$$

The exact valid count at an intermediate position depends on whether the current compression window is complete and on the reference implementation's causal convention.

## 7. KV-cache model

The cache is layer-specific rather than a single uniform full-resolution K/V cache.

| Layer type | Local 128-token K/V | Compressed pool | Lightning Indexer state |
|---|---|---|---|
| Sliding-window | Yes | No | No |
| CSA | Yes | Compression ratio 4 | Yes |
| HCA | Yes | Compression ratio 128 | No |

### 7.1 Local K/V cache

For one layer and one sequence, the local cache contains:

```text
K: [128, 1, 512]
V: [128, 1, 512]
```

The cache is logically a ring or sliding buffer. Absolute token positions must remain available independently of the ring-slot index because position encoding and causal validity depend on absolute positions.

### 7.2 CSA cache

A CSA layer maintains:

1. the local 128-token K/V state;
2. compressed K/V entries produced from overlapping ratio-4 windows;
3. compressor boundary/state information;
4. indexer-related representations or projections as required by the reference implementation;
5. valid-entry metadata;
6. query-dependent top-512 indices and associated selection data.

The compressed pool grows approximately in proportion to `sequence_length / 4`, subject to the overlapping-window and warm-up rules.

### 7.3 HCA cache

An HCA layer maintains:

1. the local 128-token K/V state;
2. one compressed K/V entry per completed 128-token window;
3. the number of valid compressed entries;
4. position or window metadata needed for causal attention.

The compressed pool grows approximately in proportion to `sequence_length / 128`.

### 7.4 Prefill cache semantics

During prefill:

- local K/V entries are created for every input token;
- CSA compressed entries are created according to the overlapping compressor;
- HCA entries are created when 128-token windows become complete;
- incomplete compressor windows must not be exposed as completed causal entries unless the reference implementation explicitly defines such behavior;
- cache state must persist across chunk boundaries.

A chunked prefill operation cannot independently process each chunk if doing so would discard compressor state spanning adjacent chunks.

### 7.5 Decode cache semantics

For one new token:

- the local K/V ring is updated;
- a CSA compressed entry may be created when the relevant overlapping compressor state becomes complete;
- an HCA compressed entry may be created only when a 128-token window closes;
- the CSA indexer selects up to 512 eligible compressed entries for the new query;
- HCA attends over all causally valid compressed entries;
- all attention branches observe the appropriate absolute-position and causal rules.

## 8. Sparse MoE architecture

### 8.1 Expert topology

Each MoE layer contains:

```text
256 routed experts
1 shared expert
6 routed experts selected per token
```

A token therefore receives the shared-expert contribution plus the weighted contributions of six selected routed experts.

### 8.2 Expert geometry

The model width is 4096 and the expert intermediate width is 2048. A SwiGLU-style expert has three principal projections:

```text
gate: 4096 → 2048
up:   4096 → 2048
down: 2048 → 4096
```

Conceptually:

$$
\operatorname{Expert}(x) = W_{down}\left(\operatorname{SiLU}(W_{gate}x) \odot W_{up}x\right)
$$

The shared expert follows the same broad MLP structure.

### 8.3 SwiGLU clamp

The public configuration specifies:

```text
swiglu_limit = 10.0
```

The reference implementation's exact clamp placement must be preserved. The described behavior clamps the relevant SwiGLU intermediates to limit extreme values before the gated product and/or subsequent projection.

### 8.4 Router configuration

Publicly described routing values include:

```text
scoring_func = sqrtsoftplus
routed_scaling_factor = 1.5
topk_method = noaux_tc
```

The router selects six routed experts per token. `sqrtsoftplus` is the router scoring function described for this architecture. `noaux_tc` indicates the top-k routing method with correction/bias behavior rather than a conventional auxiliary-load-balancing-loss path during inference.

### 8.5 Hash routing in initial layers

The first three layers use hash-based routing:

```text
num_hash_layers = 3
```

The hash route is based on a checkpoint-provided token-ID-to-expert mapping. The initial layers therefore have a token-identity-dependent expert assignment mechanism in addition to their router scoring and weighting behavior.

The exact hash tables and associated tensor names are checkpoint data and must be read from the selected model release rather than reconstructed from the parameter count.

## 9. Weight formats and quantization metadata

The publicly described model uses mixed precision. The broad description is:

```text
most non-expert weights: FP8 or related higher-precision forms
routed expert weights: FP4 in relevant releases
```

Public quantization metadata includes:

```text
FP8 format: E4M3
scale format: UE8M0
weight block size: [128, 128]
activation scheme: dynamic
```

The checkpoint may also contain tensors represented using formats or storage types such as:

```text
F8_E4M3
I8
BF16
F32
I64
```

The term `FP4` alone does not uniquely identify the binary encoding. The exact expert-weight representation requires the selected checkpoint's tensor metadata and reference dequantization logic, including:

- nibble or sub-byte packing order;
- scale granularity;
- scale storage type;
- zero-point behavior, if any;
- block layout;
- alignment and transposition conventions;
- accumulator precision.

## 10. Normalization, initialization, and numerical constants

Publicly described numerical configuration values include:

| Parameter | Value |
|---|---:|
| RMSNorm epsilon | 1e-6 |
| mHC epsilon | 1e-6 |
| Sinkhorn iterations | 20 |
| SwiGLU clamp | 10.0 |
| Routed scaling factor | 1.5 |
| Attention dropout | 0 |
| Attention bias | Disabled |
| Initializer range | 0.02 |
| RoPE base | 10,000 |
| Compressed RoPE base | 160,000 |

Inference implementations should preserve the reference ordering of normalization, projection, activation, position encoding, residual mixing, routing, and output combination. Small changes in operation order can produce differences even when individual tensor shapes are correct.

## 11. Prefill computation

For a sequence of length `S`, the main logical steps are:

1. tokenize and encode the input according to the selected DeepSeek-V4 tokenizer and message format;
2. produce embeddings;
3. initialize the four-stream hyper-connected state;
4. execute 43 decoder blocks;
5. in each block, apply mHC stream mixing;
6. compute the layer-specific attention path;
7. update local and compressed cache state;
8. apply the grouped attention output projection;
9. execute the shared expert and six selected routed experts;
10. combine expert outputs according to router weights;
11. apply the second hyper-connection operation;
12. produce the final vocabulary logits or next-token representation.

The attention differences during prefill are:

- sliding layers process causal local windows;
- CSA layers create overlapping ratio-4 entries and perform query-dependent top-512 selection;
- HCA layers create completed ratio-128 entries and attend over the valid compressed sequence.

## 12. Decode computation

For one new token at position `t`, the logical sequence is:

1. embed the new token;
2. update the four-stream state;
3. compute query projections;
4. update each layer's local cache;
5. update CSA/HCA compressor state where a new compressed entry becomes valid;
6. execute the appropriate attention branch;
7. perform position encoding and attention output processing;
8. route the token through the shared expert and six routed experts;
9. apply mHC stream combination;
10. produce the next-token logits.

The decode path is not equivalent to running a dense full-context attention operation with a conventional K/V cache. Its historical context is represented differently in sliding, CSA, and HCA layers.

## 13. Vocabulary, tokenizer, and message encoding

The public vocabulary size is:

```text
129,280 tokens
```

The model repository provides DeepSeek-V4-specific encoding and output-parsing support rather than relying solely on a generic chat-template assumption. The model-level protocol can include:

- message serialization;
- reasoning/thinking mode markers;
- output parsing;
- tool-call formatting;
- checkpoint-release-specific control conventions.

The exact tokenizer files, special-token IDs, message encoding rules, and parser behavior must be taken from the selected checkpoint release.

## 14. Multi-token prediction

The preview configuration includes:

```text
num_nextn_predict_layers = 1
```

This indicates one next-token prediction component for multi-token prediction support. MTP modules should be treated as checkpoint-specific components with their own tensor names and execution semantics; their presence should not be inferred solely from the base decoder-layer count.

Later model releases may expose different serving or speculative-decoding components. Compatibility should be determined from the selected release's configuration and model code.

## 15. Architecture-level tensor-shape summary

The following shapes summarize the principal model geometry for a token or sequence:

```text
Embedding output:              [B, S, 4096]
mHC state:                      [B, S, 4, 4096]
Low-rank query:                 [B, S, 1024]
Expanded query:                 [B, S, 64, 512]
Shared key:                     [B, S, 1, 512]
Shared value:                   [B, S, 1, 512]
Per-head non-RoPE Q/K:          [B, S, 64, 448]
Per-head RoPE Q/K:              [B, S, 64, 64]
Attention output:               [B, S, 64, 512]
Grouped output:                 [B, S, 8, 1024]   # conceptual grouped form
Final attention output:         [B, S, 4096]
Router scores:                  [B, S, 256]
Selected routed experts:        [B, S, 6]
Expert input:                   [tokens, 4096]
Expert intermediate:            [tokens, 2048]
Local K/V cache per layer:      [B, 1, 128, 512] each for K and V
CSA indexer heads:              64
CSA indexer head width:         128
CSA selected entries/query:     up to 512
```

## 16. Important architectural distinctions

- V4-Flash is not a conventional dense decoder-only transformer.
- V4-Flash is not equivalent to a conventional DeepSeek-V2/V3 MLA cache.
- The model has one shared K/V head but 64 query heads.
- RoPE applies to only 64 channels of each 512-dimensional Q/K head.
- The attention schedule alternates CSA and HCA layers between initial and final sliding-window layers.
- CSA uses overlapping compression and a Lightning Indexer with top-512 selection.
- HCA uses non-overlapping ratio-128 compression without the CSA indexer.
- The residual pathway has four mHC streams rather than one ordinary residual stream.
- Every MoE layer combines one shared expert with six routed experts selected from 256 routed experts.
- The first three layers use hash-based routing behavior.
- Active parameter count is not the same as total model parameter count.
- FP4 and FP8 tensor names do not by themselves completely specify binary packing or scaling behavior.
- The selected checkpoint's source implementation and tensor metadata are authoritative for details not exposed in the high-level configuration.

## 17. Primary public references

- [DeepSeek-V4-Flash model repository](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [DeepSeek-V4-Flash public configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json)
- [DeepSeek-V4 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
- [DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348)
- [DeepSeek-V4-Flash serving recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
