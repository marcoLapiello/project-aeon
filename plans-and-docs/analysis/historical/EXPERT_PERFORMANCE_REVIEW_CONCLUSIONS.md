# Project Aeon: Expert Performance Review Conclusions

## Initial Concern

Our initial theory assumed an efficient scheduling and caching system that would provide experts fast enough for the GPU to process them at relatively high performance. I am disappointed because the optimization steps did not produce a significant improvement, and the difference between warm and cold experts is close to zero. I cannot understand why.

Please look at the foundational documents where the vision and intentions are explained.

## Review Conclusion

The numbers do not match the vision, and that is a meaningful signal. The gap is more fundamental than the optimizations we have implemented so far.

### Vision vs. implementation

The [Project Aeon vision](../../reference/strategy/PROJECT_AEON_VISION.md), Section 4A and Blocker 1, is unambiguous about the intended data flow:

> Data flows uni-directionally from Cold to Warm to Hot, ensuring that the GPU compute engine interacts solely with high-speed memory spaces while transfer operations run fully in the background.

The critical mitigation for the [Colibri Trap](../../../../aeon-references/colibri) is equally explicit:

> Never stream synchronously from SSD for the immediate next layer: Tier 3 data transfers must be strictly speculative and asynchronous. If an expert is completely absent from both VRAM and DDR during the active layer pass, the system must trigger host CPU computation fallback rather than stalling the GPU pipeline.

What we actually built does the opposite. Every cold miss is a synchronous, just-in-time, GPU-stalling read on the critical path. The three-tier hierarchy exists as storage topology, but not yet as a latency-hiding scheduling system. That is why warm and cold performance are nearly equal: both are fetched on demand after the router reveals them. The only difference is NVMe latency, approximately $3.5\text{ ms}$, versus pinned-host latency, approximately $0.6\text{ ms}$, and both are dwarfed by the fact that the GPU sits idle while waiting.

### The numbers

For one decode token in M20 with the 35 GiB warm profile and a measured latency of $140.8\text{ ms/token}$:

- $43\text{ layers} \times 6\text{ experts} = 258\text{ expert requests/token}$.
- $1{,}077$ cold misses over $8$ tokens gives approximately $135$ cold misses per token, or a miss rate of approximately $135 / 258 \approx 52\%$.
- At approximately $3.5\text{ ms}$ of exposed `io_uring` latency per miss, this represents roughly $135 \times 3.5\text{ ms} \approx 470\text{ ms}$ of exposed I/O if the misses were fully serialized. The token takes $141\text{ ms}$ because some overlap exists, but the GPU is still waiting on storage for much of the step.

The vision's own analysis in Blocker 1 predicted this: reactive streaming at these miss rates leads to sub-token-per-second performance. We are at approximately $7\text{ tok/s}$ only because the GEMV rewrite in M18 made compute fast enough to expose the I/O bottleneck.

### Why the hit rate is stuck near 57%

This is the deepest issue. It is a capability we never built, not necessarily a bug. The caching thesis in the [specialization and entropy analysis](../../reference/strategy/AEON_SPECIALIZATION_AND_ENTROPY_ANALYSIS.md), Sections 2.3-2.4, rests on these assumptions:

> Aeon will incorporate an offline profiling utility (`aeon profile`) to measure model predictability. A Gini coefficient $G > 0.6$ confirms that pinning the top 20% of experts will capture the majority of token compute passes.

> Significant diagonal mass in the transition matrix $T_l(i,i)$ demonstrates strong temporal locality, proving that loading an expert into VRAM amortizes the I/O cost over multiple subsequent tokens.

We never ran this profiling. The 664-slot hot pool is populated round-robin by `expert_registry.hpp::populate_round_robin`, not by measured activation frequency. We are caching arbitrary experts rather than the empirically hottest experts. The vision was designed around power-law clustering, with 70-90% of routing volume concentrated in a predictable subset, but we have not measured whether DeepSeek-V4 actually exhibits that behavior or exploited it if it does.

The local [DwarfStar (ds4) reference](../../../../aeon-references/ds4) provides a useful comparator: its generated DeepSeek hotlist is sorted by profile-derived `hits/weight`. That supports testing frequency-ordered placement, but it does not provide the raw corpus or formal entropy statistics needed to validate Aeon's $H$, $G$, or transition-matrix hypotheses.

### The three structural gaps

#### 1. No frequency-informed placement

Round-robin filling means the hot pool holds a random 6% of experts per layer instead of the empirically hottest experts. If the power-law thesis holds, frequency-ordered placement alone could push the hit rate from 57% to 80% or more. This is the largest remaining lever.

#### 2. No speculative cross-layer prefetch

The vision's Blocker 2 mitigation, predicting layers $L+1$ through $L+3$ from co-activation history, is unimplemented. M17 proved that history-of-this-sequence speculation adds at most 1%, but that is because the LRU already captures short-term reuse. What is missing is forward-looking prediction: expert co-activation across layers, rather than across tokens, may be the predictable signal.

#### 3. No CPU co-processing fallback

The vision's answer to an expert absent from VRAM and DDR is to compute it on the host CPU via AVX while the GPU proceeds, represented by the fallback quantity $q^*$. We have no such path; every cold miss stalls.

## Recommendation

Before writing more code, run the profiling utility prescribed by the vision, `aeon profile`. Measure per-layer activation entropy, the Gini coefficient, and the transition matrix over a real prompt corpus. This will determine whether DeepSeek-V4 routing has the exploitable locality on which the architecture depends.

If $G > 0.6$, frequency-ordered placement and transition-matrix prefetching are plausible paths toward the 15-35 tok/s target. If the entropy $H$ is near-uniform, the honest conclusion is that single-GPU streaming is capacity-bound and the path forward is Phase 3, multi-GPU pipeline parallelism, which quadruples effective cache depth.

The profiling harness should come first because it provides the measurement that the vision assumed already existed. Frequency-ordered placement should follow only if the power-law thesis is supported by data.

## Follow-up Question

The second structural gap is unclear. If we proved that the previous prediction was useless, what is the actual predictable signal, and how would we predict it?

## Clarification

This distinction is the crux: two different predictions were conflated.

### What M17 measured and ruled out

M17 asked:

> Can we predict token $t+1$'s experts at layer $L$ from token $t$'s experts at layer $L$?

That is a same-layer, previous-token prediction.

The answer was no, for a substantive reason. The union of the last two tokens' top-6 experts covered only 60% of the current token's needs, while the LRU cache already achieved 59.3% by itself. The remaining approximately 40% were genuinely novel experts: the router sent each new token to a mostly fresh set because consecutive tokens in a sequence were semantically and syntactically different enough to route differently. There was no short-term temporal signal to exploit beyond what cache recency already captured. That hypothesis was correctly rejected.

### What Blocker 2 actually proposes

The vision's prefetch horizon is a different prediction entirely: while executing layer $L$, predict which experts will be needed at layers $L+1$, $L+2$, and $L+3$ for the same token, before those layers' routers have run.

The token flows through the layers sequentially:

```text
Layer 3: router runs -> experts {12, 45, 88, ...} -> GPU computes
Layer 4: router runs -> experts {7, 201, 33, ...} -> GPU computes  <- only known now
Layer 5: router runs -> experts {91, 14, 150, ...} -> GPU computes <- only known now
```

Layer 4's experts are physically unknowable until layer 3 finishes because layer 4's router reads layer 3's output hidden state. By the time we know that layer 4 needs expert 201, it is too late to prefetch it, so the GPU stalls for approximately $3.5\text{ ms}$ while `io_uring` fetches it.

### The candidate signal: cross-layer co-activation

The signal proposed by the vision is cross-layer co-activation. Hidden states flowing through the network during a forward pass are not random; they are structured by syntax, grammar, and task. If the token `def`, in a code context, tends to produce hidden states at layer 3 that make the layer 4 router select expert 201, that correlation may be stable across sequences. It is a property of the trained weights and routing functions, rather than a short-term property of adjacent tokens.

The two predictions are therefore different:

| Property | M17: rejected prediction | Blocker 2: untested prediction |
| --- | --- | --- |
| Target | Token $t+1$, layer $L$ | Token $t$, layers $L+1$ through $L+3$ |
| Source | Token $t$'s experts at layer $L$ | Token $t$'s experts at layer $L$ |
| Signal | Temporal locality across tokens | Co-activation across layers for the same token |
| Why it might work | It did not provide useful signal | Hidden-state structure may be determined by the weights, not token adjacency |
| Status | Measured and rejected | Not measured; never instrumented |

### How to construct the prediction

Learn a per-layer-pair co-activation matrix from offline profiling runs. For adjacent layers, define:

$$
C_l[i,j] = P\left(\text{expert }j\text{ at layer }l+1 \,\middle|\, \text{expert }i\text{ at layer }l\right)
$$

At runtime, when layer $l$ selects experts $\{e_1, \ldots, e_6\}$, immediately issue speculative `io_uring` reads for the top-$\operatorname*{arg\,max}_{j}\sum_{k} C_l[e_k,j]$ at layer $l+1$ while the GPU is still computing layer $l$'s experts, leaving approximately 1-3 ms of possible overlap.

Immediately issue speculative `io_uring` reads for $\widehat{E}_{l+1}$ while the GPU is still computing layer $l$'s experts, leaving approximately 1-3 ms of possible overlap. The same procedure can be extended to layers $l+2$ and $l+3$, using the appropriate multi-step conditional model.

### The honest caveat

This is a hypothesis, not a certainty, and there is a real reason to be skeptical. M17 showed high routing novelty across tokens. If routing is similarly high-entropy across layers, with each layer's router acting as a fresh function of a heavily transformed hidden state, the co-activation matrix may be too diffuse to predict with useful precision.

The counterargument is that the entropy analysis document, Section 1.1, claims 70-90% of routing volume concentrates in stable syntactic clusters. If that is true, those clusters should produce measurable cross-layer structure.

### Measurement before implementation

This is why the vision prescribed the `aeon profile` harness before building the prefetch machinery. The measurement is straightforward:

1. Log routing decisions over a representative prompt corpus.
2. Compute the co-activation matrices $C_l$.
3. Check whether the top-8 predictions at layer $L+1$ cover at least 60% of the actual selections.

If the signal is present, build the speculative prefetcher. If it is absent, the honest conclusion is that single-GPU streaming is capacity-bound and the path to 15-35 tok/s is Phase 3's fourfold cache depth from multi-GPU execution.

The co-activation measurement is the missing piece of evidence on which the proposed prefetch design depends.