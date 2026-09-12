# Phase 1: Single-GPU Core Runtime Micro-Execution Plan
*Architecture Target: DeepSeek-V4-Flash-0731 (INT4-W4A16 Safetensors on AMD RDNA3 / gfx1100)*

---

## 1. Executive Summary & Architectural Decisions

Following empirical verification of the official Hugging Face repositories (`deepseek-ai/DeepSeek-V4-Flash-0731` and `yiminyuan/DeepSeek-V4-Flash-0731-INT4-W4A16`), Phase 1 implements a production-grade single-GPU inference runtime engineered strictly for the **DeepSeek-V4 architecture**:

1. **Target Architecture & Nuances:**
   - **4-Stream Hyper-Connections (HC):** Multi-stream residual state with Sinkhorn normalization (`hc_pre` reduction $\to$ compute $\to$ `hc_post` expansion).
   - **Dual-Mode MoE Gating:** Hash-based deterministic routing for the first 3 layers (`n_hash_layers = 3`) and $\sqrt{\text{softplus}(\text{logits}) + \text{bias}}$ with `topk_method = "noaux_tc"` for remaining layers.
   - **Fine-Grained SwiGLU Experts:** 256 routed experts + 1 permanently active shared expert, with activation clamping (`swiglu_limit = 10.0`).
   - **Hierarchical Attention:** Sliding window ($W=128$) with multi-ratio compressed KV cache and learned top-K indexer.

2. **Weight Format & Quantization (W4A16):**
   - Canonical format: **Safetensors** (`yiminyuan/DeepSeek-V4-Flash-0731-INT4-W4A16`).
   - Layout: Symmetric INT4 with `group_size = 32`, perfectly mapping to RDNA3 Wave32 lanes (1 scale per 32 elements).
   - Execution: **Fused INT4 $\to$ FP16 Dequant-GEMM**. Weights are loaded as 4-bit nibbles into LDS/registers, unpacked in registers via fast bit manipulation, and computed using native Wave32 FP16 WMMA (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`).

3. **Storage & Memory Integration:**
   - Offline packer (`prepare_rdna.py`) ingests Safetensors shards and outputs 4096-byte sector-aligned `.aeon` files with Wave32 pre-swizzled layouts for zero-copy Linux `io_uring` direct I/O.
   - Dynamic Tier 1 (VRAM LRU pool) and Tier 2 (Pinned Host DDR) management.

---

## 2. Micro-Spike Breakdown & Verification Gates

### Spike 1: Model Config, Real Safetensors Parser & 4KB `.aeon` Packer
- **Micro-Step 1.1: DeepSeek-V4 C++20 Configuration & Metadata Parser**
  - Implement `src/architecture/deepseek_v4/core/config.hpp` parsing `config.json` (dimensions, HC multipliers, router parameters, compression ratios, SwiGLU limits).
  - Add unit test verifying parsing of the official `DeepSeek-V4-Flash-0731` config.
  - *Verification:* Bit-accurate match against JSON schema.
- **Micro-Step 1.2: INT4 Safetensors Header Parser & Stream Slicer**
  - Implement clean, zero-dependency C++20 / Python parser for Safetensors metadata headers.
  - Upgrade `scripts/prepare_rdna.py` to parse `INT4-W4A16` Safetensors tensors, align discrete experts to 4096-byte boundaries, and generate binary `.aeon` files.
  - *Verification:* Convert a sample layer/shard; verify 4KB offsets and MD5 checksum of dequantized values.

---

### Spike 2: Dense DeepSeek-V4 Primitives on Silicon (Wave32)
- **Micro-Step 2.1: Wave32 RMSNorm & SwiGLU Kernel with Clamping**
  - Implement RDNA3 Wave32 native RMSNorm kernel.
  - Implement fused SwiGLU kernel with `swiglu_limit = 10.0` clamping ($\text{clamp}(\text{gate}, \text{max}=10) \times \text{clamp}(\text{up}, \min=-10, \max=10)$).
  - *Verification:* `tests/test_swiglu_clamp.cpp` matching CPU reference output with $\epsilon < 10^{-4}$.
- **Micro-Step 2.2: Hyper-Connections (HC) Sinkhorn Kernel**
  - Implement `hc_pre` Sinkhorn normalization and `hc_post` expansion kernels operating on 4 parallel hidden-state copies (`hc_mult = 4`).
  - *Verification:* `tests/test_hc_sinkhorn.cpp` comparing device output with PyTorch reference math from `inference/model.py`.

---

### Spike 3: Fused W4A16 Wave32 Dequantization-GEMM Kernel
- **Micro-Step 3.1: Wave32 INT4 Unpack & WMMA Micro-Kernel**
  - Hand-tune register unpacking from packed INT4 pairs into FP16 half2 vectors.
  - Scale by per-group FP16 scale factors directly inside Wave32 registers.
  - Dispatch to `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`.
  - *Verification:* `tests/test_w4a16_wmma.cpp` comparing GPU output against unquantized FP16 CPU GEMM within quantization tolerance.
- **Micro-Step 3.2: Fused Block GEMM Benchmark for DeepSeek Expert Shapes**
  - Benchmark INT4-W4A16 GEMM for DeepSeek-V4 expert dimensions ($M=1\text{ to }16, N=2048, K=4096$).
  - Measure TFLOP/s and latency on the RX 7900 XTX (`gfx1100`).

---

### Spike 4: Dual-Mode MoE Gating & Expert Dispatch
- **Micro-Step 4.1: Hash-Routing Kernel (Layers 0–2)**
  - Implement token-ID hash lookup (`tid2eid`) for pre-routed expert assignment.
- **Micro-Step 4.2: SqrtSoftplus Router Kernel (Layers 3–42)**
  - Implement router gate computing logits $\to \sqrt{\text{softplus}(\text{logits}) + \text{bias}}$, selecting top-6 experts across groups.
- **Micro-Step 4.3: End-to-End MoE Layer Execution with Tier 1/2 Offloading**
  - Integrate Tier 1 (VRAM LRU cache) and Tier 2 (Host DDR SDMA stream) with real INT4-W4A16 expert weights.
  - Compute active experts and accumulate weighted outputs with the permanent shared expert.
  - *Verification:* `tests/test_v4_moe_layer.cpp` against Python reference layer pass.

---

### Spike 5: Attention Engine & Full Transformer Block
- **Micro-Step 5.1: Sliding-Window Attention & Rotary Embeddings (YaRN)**
  - Implement decoupled RoPE with YaRN scaling.
  - Implement causal sliding-window attention ($W=128$) targeting Wave32 WMMA.
- **Micro-Step 5.2: Full Block Forward Pass (HC -> Attn -> HC -> MoE -> Output)**
  - Assemble complete `DeepSeekV4Block` executing on a single RX 7900 XTX.
  - *Verification:* Golden check against PyTorch forward pass of one complete layer.

---

### Spike 6: Multi-Layer Autoregressive Pipeline & Generation Benchmark
- **Micro-Step 6.1: Multi-Layer Loop & KV Cache Management**
  - Chain consecutive layers in single-GPU execution with dynamic weight streaming.
  - The original feasibility implementation used a zero-copy Safetensors loader; the runtime now uses the native `.aeon` loader.
  - Persistent sliding-window ($W=128$) KV cache on device.
  - *Verification:* `tests/test_aeon_pipeline.cpp` validates chained execution of layers 0 and 1 from native `.aeon` weights on silicon.
- **Micro-Step 6.2: Generation Benchmark on Real Checkpoint Slice**
  - Benchmark TTFT (Time to First Token) and token generation speed (tok/s) under real memory offloading.
  - The original Safetensors benchmark was a feasibility measurement; current performance runs use the native `.aeon` benchmarks.
