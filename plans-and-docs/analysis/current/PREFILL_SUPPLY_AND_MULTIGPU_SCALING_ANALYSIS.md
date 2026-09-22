# Prefill Supply and Multi-GPU Scaling — a critical review

*Status: open analysis. Written 2026-09-22. Records a brainstorming session on whether the Step 6 prefill sweep and its supply strategy will scale to a multi-GPU rig, and where the sweep's claims are weaker than they read. No code. The actionable sequence stays in [EXPERT_STREAMING_EXECUTION_PLAN.md](../../execution/active/EXPERT_STREAMING_EXECUTION_PLAN.md); this document supplies the reasoning and the experiments that plan should absorb.*

**Subject.** The routed-expert supply on the assembled graph, examined in two directions it has not been examined in: (1) whether the swept prefill actually beats a cached (decode-style) supply in the same batched setting, and (2) what happens to both when the hardware scales beyond one GPU. Reference comparison is [Colibri](file:///home/marcolap/aeon-references/colibri), the closest sibling implementation, whose prefill supply differs from ours in one decisive respect.

**Scope.** The supply strategy and its scaling. This is not a numerics review (certified elsewhere), not a tuning plan, and not a storage-layer target. Reference commit SHAs are not recorded here because no decision below yet depends on one; the code cited is from a default-branch checkout read on 2026-09-22.

---

## 1. The two questions, and why they are one

The sweep was built to make prefill fast by streaming whole expert sets in layer order (`v4_prefill_sweep.hpp`). The brainstorming raised two worries that turn out to be the same worry seen at two scales:

- **Single GPU:** the sweep's own measured baseline compares *batched compute + sweep supply* against *serial compute + cached supply*. Two variables moved at once, so the result cannot tell us whether the **sweep supply** earns its keep.
- **Multi GPU:** the sweep relies on a blind whole-layer load and a per-layer release, so it carries no cross-layer residency — and the open question is whether the hardware scale-up helps that at all, given the drive does not scale with compute.

Both reduce to: **is the supply strategy the right one, or is it the batching that is doing the work?** That is answerable with one controlled experiment (§9.1), and the multi-GPU answer falls out of the same measurement.

---

## 2. The framing that had to be corrected first

An earlier pass in this session compared the plan's measured `2.48 tok/s` (serial) against `5.71 tok/s` (swept) at `N = 256` and read it as evidence the sweep supply is strong. **That is an apple-to-banana comparison and the reading is wrong.** The two arms differ in *compute granularity* as well as *supply strategy*:

| Arm | Compute | Supply |
| :--- | :--- | :--- |
| "serial" (baseline) | token-by-token body loop | decode-style cached on-demand supply |
| "swept" | chunked/batched body | blind whole-layer sweep |

The measured `2.30×–2.88×` is the **product** of the batching gain and whatever the supply change contributes. It is entirely possible that batching supplies all of the gain and the sweep supply is net-neutral or negative on bytes. Until the missing arm exists —

| Arm | Compute | Supply |
| :--- | :--- | :--- |
| **missing** | batched | decode-style cached supply |

— the statement *"the sweep beats serial"* is true but silent on the actual question. This correction is the reason §9.1 is the first proposed experiment.

**Second correction, same session.** The swept prefill was described here as "releasing per layer while Colibri persists across a segment." Reading `V4Graph::forward_window` (`v4_graph.hpp:325-352`) shows the chunk loop completes for a layer **before** `prefill_after_layer(layer)` runs:

```cpp
for (uint32_t layer = 0; layer < layers; ++layer) {
    host_.prefill_before_layer(layer);
    for (uint32_t offset = 0; offset < count; offset += chunk) { ...run_layer_body_chunk... }
    host_.prefill_after_layer(layer);   // released once, after all chunks
}
```

So a layer set is loaded once, all its chunks run, then it is released — which *is* persistence across chunks within a segment. Our "window" and Colibri's "segment" are the same quantity. The claimed divergence does not exist and is retracted; the real divergences are only **blind vs route-aware** and **gated vs ungated** (§5).

---

## 3. Supply versus compute: where the sweep is and is not the bottleneck

From the plan's verified model composition: routed experts `145.12 GiB` (`13.5 MiB/expert × 11,008`), one layer's set `256 × 13.5 MiB = 3.44 GiB`. On the single drive (`6.33 GB/s`):

- **Supply per layer** = `3.44 GiB ÷ 6.33 GB/s ≈ 0.55 s` (this matches the plan's own figure).
- **Body per layer** for a window of `N` tokens = `120 ms × N ÷ 43` (the plan's `≈120 ms/prompt-token`, still an open figure).

| `N` | body/layer | supply/layer | supply as share of body |
| ---: | ---: | ---: | ---: |
| 64 | 0.18 s | 0.55 s | **3.1×** (supply-bound) |
| 128 | 0.36 s | 0.55 s | 1.5× |
| **205** | **0.55 s** | **0.55 s** | **1.0× (theoretical balance)** |
| 256 | 0.71 s | 0.55 s | 0.77× |
| 4096 | 11.4 s | 0.55 s | 0.05× |

Two conclusions follow, and they are the honest core of the whole review:

1. **"Supply-bound from layer 2" is refuted for long prompts.** For `N ≳ 205` the body dominates and a correctly overlapped supply is a minority of the wall. If supply were the wall at `N = 256`, double-buffering could not have bought `2.30×` over serial — the measured speedup is itself evidence that supply was *not* binding.
2. **But "compute-bound" is a statement about *our* body, not about the workload.** If our body really costs `≈120 ms/prompt-token` where a better implementation costs a fraction of that, then we reach the compute-bound regime only because the body is slow — and a slow body *masks* supply cost at large `N` while *exposing* it at small `N`. The supply question and the body question are independent. §6 records the reference figures that suggest the body is the larger open item.

**Sanity check against the measured runs.** Swept `256 → 5.71 tok/s` ⇒ `256 ÷ 5.71 ≈ 44.8 s` ⇒ `≈1.04 s/layer`, against a body estimate of `0.71 s/layer`: a `≈1.46×` overhead above pure body. Swept `512 → 6.96 tok/s` ⇒ `≈1.71 s/layer` against `1.43 s/layer`: `≈1.20×`. So the swept path is above the body estimate at both sizes, and the gap narrows as `N` grows — consistent with a fixed per-layer overhead (the drain, the release, the registry audit) plus imperfect load/compute overlap, not with disk starvation. **That overhead is unmeasured and should be attributed before any supply redesign.**

---

## 4. The small-`N` crossover, quantified

This is the strongest version of the original worry, and it reduces to a number.

The plan's measured byte rates: swept bytes are **constant at `145.1 GiB`** (one model read, since the whole layer set is loaded regardless of `N`); serial grows at **`1.54 GiB/token`**. They cross at

$$145.1 \ \text{GiB} = 1.54 \ \text{GiB/token} \times N \quad\Longrightarrow\quad N \approx 94 \ \text{tokens.}$$

- **`N > ~94`:** the sweep reads *fewer bytes* — it wins on both bytes and throughput.
- **`N < ~94`:** a cached supply reads *fewer bytes*, and the sweep's wall is pinned at `145 GiB ≈ 23 s` however short the prompt.

Why the sweep's whole-layer load is usually not wasteful follows from the expected distinct expert set of `C` tokens drawn for top-6 of 256:

$$D(C) = 256\left(1 - e^{-6C/256}\right)$$

| `C` | expected distinct/layer | sweep over-read (`256/D`) |
| ---: | ---: | ---: |
| 1 | ~6 | 43× |
| 16 | ~80 | 3.2× |
| 64 | ~199 | 1.29× |
| 128 | ~243 | 1.05× |
| 256 | ~255 | 1.00× |

So "load all 256" is nearly free above `C ≈ 128`, mildly wasteful around `C ≈ 64`, and badly wasteful below `C ≈ 32`. At `N = 11` the sweep reads `145 GiB` where the union is ~`34 GiB` (≈58 experts/layer): a **~4.3× over-read**, ~`18 s` of drive time on a prompt that should take seconds. The waste is confined to a narrow band — but it is exactly the band where a **route-aware** load beats a blind one.

---

## 5. Colibri's supply chain: two strategies, one of them gated

Colibri builds **two different supply strategies**, and the difference between them is the whole point of this document.

### 5.1 Decode — a persistent, route-aware cache

`tier.h` (LFRU with 25% hysteresis), `hybrid_split.h` (bandwidth-balanced GPU/CPU split: upload `q* = m·B_P/B_H` of the missing experts, compute the rest on host), and the store's GPU mirror cache. A classic on-demand decode cache with eviction.

### 5.2 Prefill — a persistent, route-aware **per-layer bank**

A distinct object: a transient 256-expert VRAM bank (`coli_v4_gpu_moe_batch_union` in `c/deepseek_v4.c`). Its refill is explicitly route-aware:

> *"Each chunk routes first, then uploads only the routed experts that are not already resident for this layer, so every expert is read from the store AT MOST once per layer — the CPU union's LRU cache re-reads the same experts several times per layer under prefill's expert-major sweeps (measured ~4-6x the layer's expert bytes)."*

So Colibri's prefill supply is **the decode cache scoped to one layer**: route → upload only the holes → keep a per-expert valid map across the layer's chunks. No LRU, no demotion, no blindness. The whole-layer double buffer (`deepseek_v4_bank_pair.h`, `COLI_CUDA_MOE_DOUBLE=1`) prefetches layer `L+1`'s complete set on the aux stream — structurally the same idea as our sweep — but its swap carries the **valid map**, so a partial prefetch is topped up by the route-aware refill rather than discarded.

### 5.3 The gate we removed

Colibri **refuses the prefill bank below a prompt-length threshold**:

```c
const char *setting = getenv("COLI_CUDA_MOE_BATCH_MIN");
minimum = setting ? atoi(setting) : 256;
if (v4_gpu_moe_batch_hint_tokens < minimum) return -1;   // fall back to the cached path
```

Our plan deliberately removed the equivalent gate (`v4_prefill_sweep.hpp:102-108`): *"There is deliberately no window-size condition here… A policy that is enabled by a threshold nobody can see… is the wrong shape."* The plan's objection was to a **hidden** threshold — that objection is sound — but the correct response is a **visible setting**, not the deletion of the policy. As it stands, **every short prompt now pays a full `145 GiB` sweep**, which is the concrete regression the small-`N` analysis predicts.

### 5.4 Divergence summary (the only real differences)

| | Aeon sweep | Colibri prefill bank |
| :--- | :--- | :--- |
| Load granularity | whole layer, blind | routed holes, route-aware |
| Across-chunk persistence | yes (layer set, per window) | yes (per layer, valid map) |
| Small-`N` behaviour | always sweeps | refuses below 256 |
| Double buffer | one layer deep, engaged whenever feasible | opt-in, partial-tolerant |
| VRAM ownership | reuses the Hot pool (D-d) | separate ~2.2 GiB bank |
| Lifetime | released per layer within a window | released at end of prefill |

The sweep is therefore best understood as a **simplification** of Colibri's route-aware per-layer cache — justified for `C ≳ 128`, where the union `≈ 243/256` makes route-awareness moot, and unjustified below `C ≈ 32`. It is not a strictly better strategy.

---

## 6. Reference-figure caveats (do not build a target on these)

Two Colibri-derived numbers circulate as targets. Both need re-sourcing before they anchor anything.

**`30+ tok/s` on DSV4 Flash at 16 GB — not reproducible.** Colibri's own measurement for this checkpoint on a 16 GB card is:

> *"Performance (RTX 5080 16 GB, 2× NVMe mirror, 32 GB RAM, i9-class CPU): decode at 3.3k context — 0.6 tok/s → 1.5–1.6 tok/s."* — `docs/deepseek-v4.md`

The `30+ tok/s` figures in our own `plans-and-docs/` belong to **other models** — Qwen3.6-35B-A3B int4 (`1.44 → 10.05 tok/s` on two 8 GB cards), GLM-5.2 (`~4 tok/s`), and our own Phase-2 small-model run (`32.8 tok/s`). None is DSV4 Flash. The disk arithmetic also forbids it: DSV4 Flash streams ~230 experts × 12.6 MB ≈ 2.9 GiB/token even at a ~95% VRAM hit rate, so on a ~6 GB/s drive the ceiling is ~2 tok/s. **Treat the `30+ tok/s` DSV4-on-16 GB claim as unsupported until a source is produced.**

**`≈31 ms/token` as the body reference — verify what it measures.** The plan cites it as Colibri's body cost, but `0.35 s/layer × 43 layers ÷ 512 ≈ 29 ms/token` is also exactly Colibri's **disk-bound supply cost per prompt token** at a ~512-token window (`deepseek_v4.c` states the bank refill at `~0.35 s/layer`, disk-bound). If the `31 ms` figure is supply rather than body, then comparing our `120 ms` *body* against it is the same apples-to-bananas error as §2, in the opposite direction. **Split the reference into its supply and body components before using either as a target.**

The practical consequence for §3: if our body really is several times slower than a tuned reference, **kernel and side-op optimisation is the larger lever than supply-strategy redesign**, and the "compute-bound" regime we observe is partly self-inflicted.

---

## 7. Multi-GPU: what actually scales, and what does not

### 7.1 The capacity play — the real multi-GPU win

Colibri's headline multi-GPU result is **not a faster sweep**; it is **residency**:

> 6× RTX 5090: *"a 176.7 GB VRAM tier + 191.3 GB RAM tier (all 19,456 experts resident) … 5.8–6.8 tok/s decode … no disk traffic"* (`docs/cuda.md`).

And its own DSV4 engine is explicit that more GPUs are only a lever through **capacity**:

> *"More RAM/VRAM (higher expert hit rate) is the only lever left below the disk limit; a multi-GPU design exists on paper only."* — `docs/deepseek-v4.md`

with the DSV4 multi-GPU row marked *"single device today; expert-parallel design drafted, not implemented."* (Multi-GPU exists in Colibri's **generic** engine — `colibri.c`, GLM/Qwen — not in DSV4.)

**Implication for Aeon.** The sweep is a strategy for `24 GiB ≪ 145 GiB`. Once aggregate VRAM ≥ `145 GiB`, the premise disappears: everything is resident, supply → 0, and the decode cache becomes the *only* supply strategy. On a 6×24 GiB rig the sweep is unnecessary; on a 2–4×24 GiB rig it still is not enough to be unnecessary. **So "the sweep is single-GPU-shaped" is correct — not because it lacks a cache, but because it is the capacity-wall strategy, and enough GPUs delete the wall.**

### 7.2 The ratio argument — sharding moves nothing

For any shard (layer-parallel or expert-parallel) covering fraction `f` of a layer:

$$\text{supply} = f \cdot 0.55\ \text{s}, \qquad \text{compute} = N \cdot f \cdot \frac{0.120}{43}\ \text{s}$$

Both scale by `f`, so **sharding preserves the supply:compute ratio exactly**. Multi-GPU can neither create nor rescue supply-boundness. It changes throughput only by changing the *resources behind each term* — capacity (7.1) or per-shard storage (7.3).

### 7.3 The storage play — the constraint is topology, not GPU count

Because sharding preserves the ratio, the throughput win requires that each shard's supply term shrink with its compute term. That happens only if **aggregate drive bandwidth scales**:

- **Per-GPU NVMe:** aggregate bandwidth ≈ `k × 6.33 GB/s`, both terms shrink, throughput scales ≈ `k` at constant efficiency. The sweep shards cleanly because each shard streams whole layers in order.
- **One shared NVMe:** the drive serializes the sweep's reads; the serial section dominates and GPU count buys nothing on throughput. **This is the sharpest form of the original worry and it is correct.**

### 7.4 P2P — accepted, and it weakens the cross-device caveat

RDNA3 (`gfx1100`) **does** support peer-to-peer PCIe DMA. This was stated and accepted; it corrects an earlier caveat in this session that treated P2P as unavailable on consumer RDNA3.

The consequence is concrete: in a layer-sharded pipeline the only cross-device traffic is the **residual**, bounded by `W × 64 KB` per layer — negligible even at PCIe Gen4 rates. So cross-device hop cost is **not** the blocker it is in Colibri's `COLI_CUDA_PIPE=2` case, where *"on multi-GPU hosts the per-layer P2P hops cancel the gain"* (`docs/cuda.md`) — and that Colibri penalty is about the *decode-time* residual stream, not the prefill sweep. **A layer-sharded multi-GPU prefill is therefore viable on this hardware.** The binding constraint remains storage topology (7.3).

---

## 8. The VRAM-ownership trade (our D-d versus Colibri's dedicated bank)

In response to "is Colibri's bank a separate prefill-only hot pool?": **yes, it is a distinct allocation, not a carve-out.**

- Created lazily on the first large prefill: `dsv4_cuda_expert_bank_create(256, 4096, 2048, device, …)`, ~2.2 GiB.
- Used only during prefill: *"the bank exists only while a large prefill runs: it is ~2.2 GiB of VRAM that decode never touches."*
- Released at end of prefill: `coli_v4_gpu_moe_batch_release()` — *"free its ~2.2 GiB so the decode expert mirrors get the VRAM instead."*

So Colibri **time-multiplexes VRAM between two disjoint pools**. Our plan's **D-d** does the opposite: the sweep **reuses the Hot pool** and `begin()` **drains it outright** (Warm is frozen; Hot is not). Both are valid; the trade should be recorded as a decision, not left implicit:

| | Shared pool (Aeon D-d) | Dedicated bank (Colibri) |
| :--- | :--- | :--- |
| Decode residency across a prefill | destroyed — `begin()` drains Hot | preserved — decode's pool untouched |
| Cost | decode must re-warm after prefill | ~one layer's VRAM unavailable to decode while the bank exists |
| Fits when | VRAM is binding (`12.71 GiB` dense + KV on `24 GiB`) | spare VRAM exists, or the pool is large (~800 slots) |

**Proposed shape (the brainstorm's idea, sharpened).** With a Hot pool of ~800 slots and a layer set of 256, reserve a **prefill-only bank of one layer (256 slots)**, engage it only above the length gate, and return it to decode when the window ends. This turns D-d's "drain and re-warm" into Colibri's "borrow and return," at the cost of 256 slots of decode capacity during prefill. **Whether that VRAM is spare is a measurement, not an assumption** — it is the same `12.71 GiB + KV` budget question that led to D-d, and the host-memory-pressure investigation is adjacent to it.

---

## 9. Experiments this document proposes

These are the measurements that would settle §2–§4 and §8. They belong in the plan's open work; they are stated here with their predictions so a disconfirming result is informative.

### 9.1 The missing arm — batched compute + cached supply (the decisive test)

Run batched prefill with the **decode-style route-aware supply** (the same tiering, dedup and leases the decode path already uses) and compare against the sweep, at fixed `C`, `W`, and configuration, across `N ∈ {32, 64, 128, 256, 512}`.

- **Predicts:** a byte crossover near `N ≈ 94` (§4) and a throughput crossover below it; the sweep wins only above the crossover and only modestly.
- **If the sweep loses at all sizes:** the batching, not the supply, was the gain, and the sweep should become an optimisation *within* the cached path rather than a replacement for it.
- **Anti-circularity:** both arms must be graded against the same logits/residual oracle, and the cached arm must not share a code path with the sweep's own helpers.

### 9.2 Attach the swept path's overhead (§3)

Instrument the per-layer drain, `release_expert_leases`, the registry audit, and the `prefill_before_layer` wait, so the `1.2×–1.46×` gap above the body estimate is split into compute, supply wait, and per-layer overhead. **Attribute before redesigning.**

### 9.3 Split the reference body/supply figures (§6)

Re-source the `≈31 ms/token` and `0.35 s/layer` numbers as *body* and *supply* separately, and re-verify the `30+ tok/s` DSV4 claim (or retire it). No tuning target should be derived from an unsplit figure.

### 9.4 Visible length gate (§5.3)

Restore the policy Colibri keeps — do not sweep below a threshold — as an **explicit setting** (`--prefill-sweep-min-tokens`), defaulting at the measured crossover rather than a hidden constant. This directly addresses the plan's own misgiving while keeping the throttle visible.

### 9.5 The multi-GPU pre-questions (before any design)

Two facts decide the multi-GPU shape, and both are cheap to obtain:

1. **Aggregate VRAM ≥ `145 GiB`?** If yes, go residency-first and let the sweep retire; if no, the sweep stays and §7.3 decides the rest.
2. **Storage topology:** per-GPU NVMe (bandwidth scales, §7.3) or a shared drive (the drive is the serial section)?

P2P is available (§7.4), so a layer-sharded pipeline is viable *if* topology (2) permits.

### 9.6 Bank-versus-pool A/B (§8)

If spare VRAM exists, A/B D-d (shared pool, drain on entry) against a dedicated one-layer prefill bank (borrow and return), measuring both prefill throughput and the **decode re-warm cost** D-d currently hides.

---

## 10. What this review does not claim

- **It does not claim the sweep is wrong.** For a long prompt it reads one model pass (`145.1 GiB`, constant) where a cached supply reads `1.54 GiB/token`, and it is correctly ordered and double-buffered. The claim is narrower: its advantages are real **above** the crossover and absent **below** it, and its comparison against "serial" was never the right test.
- **It does not claim multi-GPU requires a new cache design.** Multi-GPU's job is to *delete* the capacity wall (residency) or *partition* it (per-shard storage). Concurrency alone helps neither, because the total bytes are fixed (§7.2).
- **It does not claim our body is unoptimised — it flags that the figure suggests so and asks for attribution.** §3 and §6 make that a measurement, not a conclusion.
- **It does not re-open numerics.** Everything here concerns which tier answers which read and when, never the value returned.

---

## 11. Disposition

| Item | Where it goes |
| :--- | :--- |
| Missing arm (§9.1) | New experiment in the plan's open work; **sequence first**. |
| Overhead attribution (§9.2) | Feeds the plan's open `≈120 ms/prompt-token` investigation. |
| Reference re-sourcing (§9.3) | Correct the plan's §6 and any target derived from it. |
| Visible gate (§9.4) | New setting; resolves the plan's removed-threshold misgiving. |
| Multi-GPU pre-questions (§9.5), bank A/B (§9.6) | New open work under the multi-GPU future direction. |
| Bank-versus-pool decision record (§8) | Append to the plan's D-d rationale so the choice is explicit. |
