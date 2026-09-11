1. Model geometry (I won't guess these)
I don't have reliable architecture details for "DeepSeek V4 Flash" in my knowledge, and I'm not going to invent numbers that silently poison every tile-size choice. Paste from your config.json / repack metadata:

hidden_size, intermediate_size (dense), moe_intermediate_size (per-expert)
num_hidden_layers, number of dense-only prefix layers vs MoE layers
n_routed_experts, n_shared_experts, num_experts_per_tok, whether there's a grouped/limited routing (top-k groups) and any sigmoid+bias routing
attention type: MLA (with q_lora_rank, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim) or plain GQA (num_key_value_heads, head_dim)
MTP / multi-token-prediction heads present? RoPE variant + scaling
vocab size, tied embeddings, any NSA/sparse-attention block

2. Exact quant layout
"int4 w4a16" is underdetermined for kernel writing:

group size (32/64/128), symmetric or zero-point, scale dtype (fp16/bf16)
packing order: 8 nibbles per uint32, interleaved or sequential? Is it AWQ-style permuted (0,2,4,6,1,3,5,7) or GPTQ order?
weight matrix orientation as stored: [out, in] or [in, out], row-major?
activations fp16 or bf16 internally? Accumulate in fp32 presumably.
Are you free to re-repack weights offline? This matters enormously — the fastest gfx1100 W4A16 kernel wants a bespoke swizzle (per-wave K-major blocks aligned so each lane's global_load_dwordx4 feeds v_dot2c_f32_f16 / MFMA-less MAI path without cross-lane shuffles). If I can dictate the on-disk layout, I can give you ~1.5–2× over a generic dequant-GEMM.

3. Which kernels are actually the bottleneck
Do you have per-kernel timings (rocprof) for a decode step and a prefill step? Specifically the split between: expert GEMV/GEMM, dense GEMM, attention, routing/top-k, and PCIe stall waiting on expert streaming. If you're PCIe-bound at decode, a perfect GEMM kernel buys you nothing and I should spend effort on the DMA/overlap machinery and a fused "gather-experts-into-a-batched-GEMM" path instead.

4. Regime and batching
Batch size / concurrency target: single-stream bs=1 decode, or serving multiple sequences?
Typical prompt length and target context (affects whether attention needs a proper flash-decoding split-K over KV, paged KV, etc.)
KV cache dtype (fp16? int8? MLA-compressed latent?) and paged vs contiguous.

5. Environment
ROCm version (6.x vs 7.x changes available intrinsics and __builtin_amdgcn_* availability), hipcc target gfx1100
Are you already using __builtin_amdgcn_wmma_f16_16x16x16_f16? RDNA3 does have WMMA (16×16×16 f16/bf16/iu8/iu4 matrix cores) — this is the single biggest thing most people miss on 7900 XTX, and it's the foundation of any competitive GEMM. Also relevant: is your build using wave32 or wave64?
Do you have LDS-bank-conflict/occupancy data, or should I design conservatively?
Can you run a microbenchmark harness I give you (I can't test, so I'd deliver kernels with a self-checking correctness harness + autotune sweep you run locally and report back)?

6. Interfaces I must not break
Paste (or summarize) the current signatures/structs for: the expert-weight descriptor, the KV cache layout, and however you currently launch the MoE FFN. I'll match them rather than hand you something that requires rewriting your runtime.