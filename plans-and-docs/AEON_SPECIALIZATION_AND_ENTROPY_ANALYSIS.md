# Project Aeon: Architecture Specialization Strategy & Activation Entropy Analysis

## Executive Summary
This document records the architectural decision-making regarding model scope (single-model specialization vs. broad multi-architecture support) and details the empirical methodology for analyzing Mixture-of-Experts (MoE) activation entropy. 

To avoid the performance degradation and architectural bloat characteristic of general-purpose engines (such as [llama.cpp](../../aeon-references/llama.cpp)), Project Aeon will adopt a **"Hyper-Specialized V1, Modular Subsystem"** strategy. Phase 1 will target the **DeepSeek fine-grained MoE architecture** exclusively, optimizing low-level RDNA3 compute and memory streaming against its exact operational characteristics before generalizing to other architectures.

---

## 1. Analysis of MoE Activation Dynamics: Myth vs. Reality

### 1.1 The "Domain Expert" Fallacy vs. Syntactic Clustering
* **The Misconception:** MoE models function as federations of macro-level "domain experts" (e.g., dedicated sub-models for Python programming, medical reasoning, or creative prose).
* **The Mechanistic Reality:** Routing mechanisms operate primarily on **low-level syntactic, structural, and grammatical features**:
  * Specific experts specialize in indentation whitespace, brackets, punctuation, and variable tokens.
  * Other experts trigger consistently on mathematical operators, numeric strings, and logical delimiters.
  * Other experts activate on grammatical structures (passive voice, conjunctions, clause transitions).
* **Predictability Implication:** 
  Even though experts do not correspond to broad semantic domains, **coding, mathematical, and conversational tasks produce highly non-random, stable expert activation patterns**. In a programming session, the high frequency of punctuation, indentation, and syntax tokens concentrates 70–90% of routing volume into a persistent, predictable subset of experts.

### 1.2 Cross-Architecture Variance
Activation entropy varies significantly across different MoE designs:
1. **Coarse-Grained MoEs (e.g., Mixtral 8x7B / 8x22B):**
   * Small expert count ($N = 8$), low active count ($K = 2$).
   * Expert weights are large (~3–7 GB each).
   * Low routing entropy per layer, but swapping incur massive PCIe/storage transfer penalties.
2. **Fine-Grained MoEs with Shared Experts (e.g., DeepSeek-V4):**
   * High expert count ($N = 64, 160, 256$), small active count ($K = 6, 8$).
   * Individual expert size is small (~80–150 MB at 4-bit).
   * Features **permanently active Shared Experts** that absorb baseline representational continuity.
   * Exhibits sharp power-law activation clustering.
3. **Hybrid Attention MoEs (e.g., Minimax M2.5 / M2.7):**
   * Integrates linear/lightning attention variants with custom state updates.
   * Modifies hidden state dynamics entering the router, requiring distinct kernel dispatch formulations.

---

## 2. Quantitative Methodology for Measuring Activation Entropy

Project Aeon will incorporate an offline profiling utility (`aeon profile`) to measure model predictability across standardized prompt suites before deploying caching allocations.

### 2.1 Router Logging Harness
For sequence length $T$ and layer $l$, log selected expert indices from the router gate:
$$E_{l, t} = \{e_1, e_2, \dots, e_k\}, \quad t \in [1, T]$$

### 2.2 Shannon Activation Entropy
Compute the marginal probability $P_l(e)$ of expert $e$ being selected in layer $l$:
$$P_l(e) = \frac{1}{k \cdot T} \sum_{t=1}^{T} \mathbb{I}(e \in E_{l, t})$$

The layer-wise activation entropy $H_l$ across $N$ total experts is defined as:
$$H_l = -\sum_{e=1}^{N} P_l(e) \log_2 P_l(e)$$

* **Maximum Entropy ($H_{\text{max}} = \log_2 N$):** Uniform distribution across all experts. Indicates a worst-case scenario where caching provides minimal benefit.
* **Low Entropy ($H_l \ll \log_2 N$):** Sharp power-law concentration. Guarantees that a fixed fast-tier memory allocation will achieve high hit rates.

### 2.3 Gini Inequality Coefficient
Quantifies the skew of expert utilization:
$$G = \frac{\sum_{i=1}^N \sum_{j=1}^N |P_l(i) - P_l(j)|}{2 N \sum_{i=1}^N P_l(i)}$$
A high Gini coefficient ($G > 0.6$) confirms that pinning the top 20% of experts will capture the majority of token compute passes.

### 2.4 Temporal Hysteresis (Transition Matrix)
Measures the likelihood of consecutive expert reuse across decode steps:
$$T_l(i, j) = P(e_{t+1} = j \mid e_t = i)$$
Significant diagonal mass ($T_l(i, i) \gg \frac{1}{N}$) demonstrates strong temporal locality, proving that loading an expert into VRAM amortizes the I/O transfer cost over multiple subsequent tokens.

### 2.5 DwarfStar Hotlist as a Practical Comparator

The local [DwarfStar (ds4) checkout](../../aeon-references/ds4) contains a generated DeepSeek expert hotlist described as sorted by `hits/weight`. This is useful evidence for comparing frequency-informed expert placement and cache policy against Aeon's current round-robin initialization. The hotlist is a placement prior, not a formal Shannon entropy, Gini coefficient, or transition-matrix measurement; Aeon still needs to collect raw routing decisions over a representative corpus before treating it as an entropy result.

---

## 3. Strategic Decision: Single-Model Focus vs. Broad Support

### 3.1 The Failure Mode of Premature Generalization
Prior art demonstrates that engines attempting to support all architectures from inception (e.g., early multi-backend frameworks) must design around lowest-common-denominator abstractions. This introduces:
* Generalized memory allocators that cannot guarantee zero-copy pinned direct I/O.
* Suboptimal matrix multiplication kernels that accommodate arbitrary tensor layouts rather than specific hardware register swizzling.
* Excessive abstraction layers that obscure low-level driver latency bottlenecks.

Hyper-optimized engines (such as *Flash-MoE* on Apple Silicon or *hipfire* on RDNA3) achieve state-of-the-art throughput by tailoring their dispatch loops and memory layouts to specific hardware and tensor shapes.

### 3.2 Target Selection for Phase 1: DeepSeek Fine-Grained MoE (V4)
Project Aeon will focus Phase 1 exclusively on the **DeepSeek-V4 MoE architecture** based on four structural advantages:
1. **Granular Expert Sizing:** Individual 4-bit experts (~100–150 MB) can be staged across PCIe Gen4 in ~2–3 ms, perfectly fitting speculative asynchronous transfer windows.
2. **Permanent Shared Experts:** Baseline attention and dense representations remain anchored in VRAM at all times, ensuring stability even during routed expert cache misses.
3. **Multi-Head Latent Attention (MLA):** Compressed KV cache dimensions reduce memory footprint by 4×–6×, freeing up to ~80 GB of physical VRAM across the 4x RX 7900 XTX array solely for the resident expert cache pool.
4. **Demand & Impact:** Demonstrating frontier-grade DeepSeek inference on consumer multi-GPU AMD hardware addresses an immediate, high-value real-world workstation use case.

### 3.3 Traps Presented by Secondary Candidates in Phase 1
* **Minimax M2.7:** Features non-standard linear/lightning attention primitives. Implementing custom RDNA3 kernels for hybrid attention diverts resources away from optimizing the 3-tier memory streaming engine.
* **Qwen MoE:** Employs standard Grouped-Query Attention (GQA) and lacks the permanently pinned shared-expert structure, shifting routing risk onto the dynamic pool.

---

## 4. Architectural Decoupling: "Narrow Core, Clean Interface"

To avoid technical lock-in while maintaining maximum single-model performance, the codebase will be split into two strict layers:

```
+-----------------------------------------------------------------+
|                       Project Aeon Runtime                      |
+-----------------------------------------------------------------+
|   [Tiered Memory Manager]    |    [io_uring Direct I/O Engine]  |
|    - Hot / Warm / Cold LRU   |    - 4096-Byte Sector Alignment  |
|    - Pinned Buffer Pools     - Ring Buffer Async Queues         |
+-----------------------------------------------------------------+
                                |
             Abstract Model Interface (C++20 Concepts)
             - ExecuteLayer(hidden_state, routing_mask)
             - RouteTokens(router_logits, top_k)
                                |
    +---------------------------+---------------------------+
    |                                                       |
[Phase 1 Implementation]                          [Phase 2 Extensions]
- DeepSeek MLA Kernels (Wave32 WMMA)             - Qwen MoE Support
- DeepSeek Shared + Routed Dispatch              - Minimax Lightning Kernels
- DeepSeek Swizzled Quant Format                 - Coarse-Grained MoE Adapters
```

### 4.1 Phase 1 Concrete Implementations (Hardcoded for Maximum Speed)
* RDNA3 WMMA Wave32 GEMM kernels tuned specifically for DeepSeek's FFN hidden dimensions.
* DeepSeek MLA low-rank key-value decompressor.
* Pre-swizzled weight converter (`prepare_rdna.py`) tailored to DeepSeek tensor naming and quantization layouts.

### 4.2 Phase 1 Reusable Subsystems (Agnostic to Model Architecture)
* Linux `io_uring` direct NVMe streaming runtime with registered memory buffers.
* Hardware SDMA memory staging queues across PCIe Gen4 buses.
* 1F1B pipeline-parallel scheduler across 4x RX 7900 XTX GPUs.
* Host CPU AVX-512 fallback executor ($q^*$) for cache-miss resolution.

---

## 5. Development Phasing Roadmap

* **Phase 1 (Proof of Feasibility & Speed):**
  * Target: DeepSeek fine-grained MoE on 4x RX 7900 XTX + Threadripper Pro.
  * Deliverable: Working end-to-end engine achieving 20–35+ tok/s decode throughput using 3-tier memory streaming.
* **Phase 2 (Architectural Generalization):**
  * Target: Expand model loaders to support standard GQA MoEs (Qwen MoE family).
  * Deliverable: Parameterized tensor shapes and generalized router dispatch tables.
* **Phase 3 (Hybrid Architectures):**
  * Target: Minimax and non-transformer hybrid recurrent/linear attention architectures.
