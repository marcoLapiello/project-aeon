# Prefill chunk/window scaling — the throughput ceiling and the caps that move it

Date `2026-10-03`. Question: every measured prefill run so far used chunk `C = 256` on a `~677`-token prompt, where the GPU computes `~6 s` inside a `~20 s` NVMe read. If the engine is read-bound and the GPU is idle `~70%` of that window, how much longer a prompt and how much bigger a chunk/window can it take before compute becomes the bound — and what scratch/residual VRAM does that cost? Method: read of the chunk body, the expert executor, the prefill workspace and the memory budget; no code changed, no run made.

**Nothing here is measured beyond the plan's existing baseline.** The extrapolations to `2048` tokens / `C = 1024` are reasoned from those measured figures and the code's scaling laws; the section that says so is marked. The measurements live in [KERNELS_IMPROVEMENT.md](../../execution/active/KERNELS_IMPROVEMENT.md) and the [Performance & Accuracy Ledger](../../status/PERFORMANCE_LEDGER.md).

---

## 1. The correction that frames everything: TTFT and tok/s are decoupled here

The plan's Baseline table still reads **TTFT `25,303 ms`**. That predates Area 7, which removed the end-of-prefill expert restore and measured **TTFT `25.29 → 20.83 s`** for exactly this reason — a blocking re-read that bought nothing decode could not rebuild. The baseline table was never refolded after Area 7, so the **live figure for the `677`-token, `C = 256` run is `~20.8 s`, not `25.3 s`**.

That matters because it changes the mental model. In an engine that is not NVMe-bound, `tok/s ≈ 1/TTFT` and the two move together. Here they do not:

- **TTFT has a floor** ≈ one window's bytes ÷ drive bandwidth. At `C = 256` the sweep already touches `8,896` of `11,008` experts (`117.3 GiB`) in `~20.2 s` at `~7 GiB/s` — the drive's ceiling. A longer prompt inside one window barely moves that byte count (it saturates toward a full model read, `~145 GiB ≈ 20.7 s`); it stays **~20 s**.
- **Throughput** `= N / TTFT`. With TTFT pinned, throughput grows **≈ linearly with `N`** while the prompt fits one window.

So "maximise throughput" here means **push `N` up until compute reaches the read**, not "reduce TTFT". A longer prompt at flat TTFT is the win, and it is exactly the regime the current `677`-token run is `~3–4×` short of.

| | measured now (`677` tok, `C=256`) | reasoned (`2048` tok, `C=1024`, `W=4096`) |
| :--- | ---: | ---: |
| NVMe read | `117 GiB` / `~20 s` | `~117–145 GiB` / `~20–21 s` |
| GPU compute | `6.0 s` (`~30%` util) | `~13–16 s` (still under the read) |
| **TTFT** | **`~20.8 s`** | **`~21–22 s`** |
| **throughput** | **`~33 tok/s`** | **`~95 tok/s`** |

The two reasonable constraints on how far `N` can go: the **compute curve** (`~8.9 ms/token` today, improving with `C` as the grouped WMMA and dense GEMM reach their one-read floors) and the **window boundary** (`W` must hold the prompt, or the model read is paid once per window).

**The ceiling.** With the read at `~20.7 s` and compute at a post-reuse `~7 ms/token` (at `C = 1024`), the two lines cross at `N* ≈ 20.7 / 0.007 ≈ ~2,900` tokens. Below it the GPU hides inside the read and extra tokens are nearly free; above it compute becomes the bound and TTFT grows linearly again. **~2,500–3,000 tokens in one `W ≈ 4096` window at `~20 s` is the prefill ceiling as the code stands — `~120–140 tok/s` TTFT throughput, `~4×` today.**

## 2. `kAeonSwizzledMaxExperts = 8` — what it is, and whether it is live

**What.** Not a compute limit and not a chunk cap. It is the length of a **fixed, by-value weight-pointer table** passed as a kernel argument:

```cpp
// src/backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp
constexpr int kAeonSwizzledMaxExperts = 8;
struct SwizzledW13ExpertPtrs {           // 4 pointers × 8 = 32 pointers = 256 B
    const uint4* w1[8]; const half* s1[8];
    const uint4* w3[8]; const half* s3[8];
};
```

**Per launch of what.** One **expert-batch launch pair** (`w13` + `w2`). The grouped path splits a chunk's distinct-expert union into groups of at most `8` and issues one launch pair per group:

```cpp
// src/architecture/deepseek_v4/moe/moe_grouped_batch.hpp
const int per_launch = kernel::kAeonSwizzledMaxExperts;   // 8
for (int base = 0; base < expert_count; base += per_launch) { ... }
```

Because the permutation's offsets ascend and an absent expert is zero-width, a group is just a contiguous expert range rebased by `expert_offsets[base]` — no index copy between groups.

**Live after the kernel plan — twice.**
- The **GEMV** per-token path (`dispatch_dsv4_moe_gemv_w13_swiglu`) still consumes the table; it is the **decode** path (`accumulate_routed`) and the `moe_grouped_batch_enabled() == false` fallback.
- The **grouped WMMA** path's batch width *is* this constant.

The `8` itself is a GEMV-era artefact: one token's MoE touches `6` experts, so `8 ≥ 6` was enough. It is **not** a gfx1100 limit and **not** a property of the int4 format. Lifting it means replacing the by-value table with **device-side pointer indirection** (a resident table indexed by an id) — a real change, not a constant bump — and it interacts with the format-neutrality gap in [MODULARITY_COUPLING_AUDIT.md](MODULARITY_COUPLING_AUDIT.md) item 1.

## 3. Chunk caps and where they live

Two hard caps, both `256`, both validated with a throw:

| Constant | File | Value | Bounds |
| :--- | :--- | :--- | :--- |
| `V4LayerBodyBatchScratch::kMaxTokens` | `layer/v4_layer_body_batch.hpp` | `256` | the body's row cap; the chunk scratch and the composed row-set |
| `V4RoutedExpertScratch::kMaxChunk` | `moe/v4_expert_executor.hpp` | `256` | the grouped-path permutation and `d_chunk_hidden` / `d_chunk_contrib` |

A `C = 1024` run needs **both** raised. Two supporting facts:

- **Not a state-contract bound.** The chunk body explicitly documents that the local ring and the compressor's partial ring do **not** bound the chunk (a chunk's keys go to a per-chunk buffer, committed only after every query ran; the compressor materializes each boundary in position order inside phase 1). A 16-token chunk through a 10-slot local ring and an 8-slot compressor ring was measured bit-identical to serial on `gfx1100`. So **raising `kMaxTokens` is a memory decision, not a correctness one.**
- **The `M` window already scales.** The grouped path's window (`kMoeGroupedMTiles = 4`, 64 tokens) and the dense GEMM's `kDenseGemmMinTokens = 32` are already dispatched on `M`; a wider chunk picks the wide-tile paths automatically. Raising the cap does not require new kernels.

The `--help` text for `--prefill-chunk` still says `1..64`; it is stale — `kMaxTokens = 256` is authoritative.

## 4. Scratch and residual: allocated at load, never returned — and that is what couples the two levers

**They are locked for the whole run.** `V4PrefillWorkspace` (the batch scratch + the residual carry) is allocated **once, at load** (`V4ModelHost::allocate_prefill_workspace`), grown only, and freed **only in `V4ModelHost::free()`** at process teardown. The budget subtracts all three permanently:

```cpp
// memory_budget_engine.hpp
report.vram_scratch_bytes = decode_scratch + batch_scratch + prefill_carry;
// Hot pool = usable − dense − KV − headroom − scratch
```

So **decode pays the full prefill scratch + carry in lost Hot slots all run long**, even though decode's `min(2·experts_per_token, …)` working set is a fraction of prefill's. The scratch scales with **chunk** (measured `~0.69 MiB/row`; the budget's allowance is `1 MiB/row`); the carry scales with **window** (`96 KiB/token`, `hc_mult × hidden × (2 + 4)`).

**Which term moves with which knob:**

| Term | Scales with | `C=256, W=4096` | `C=1024, W=4096` | `C=1024, W=2048` |
| :--- | :--- | ---: | ---: | ---: |
| `V4LayerBodyBatchScratch` | chunk | `~176 MiB` | `~706 MiB` | `~706 MiB` |
| `V4RoutedExpertScratch` (chunk part) | chunk | `~33 MiB` | `~131 MiB` | `~131 MiB` |
| residual carry | window | `384 MiB` | `384 MiB` | `192 MiB` |
| **VRAM total** | | **`~593 MiB`** | **`~1,221 MiB`** | **`~1,029 MiB`** |

At the measured `~6.7 MiB/expert`, the `C = 1024` bump is `~+630 MiB ≈ ~94 fewer Hot slots`; the ledger's own anchor is `C 128→256` = `+87 MiB` = `13` slots (`802 → 789`), so the slope is consistent.

**Can it be dynamic? The pattern exists, and the payoff is exactly the coupling.** The staging corridor already does this: `V4ModelHost::resize_staging_slots(slots)` re-sizes the arena at runtime, guarded by two exact preconditions (every slot free, no outstanding lease), keeping `transient_staging_bytes` equal to the allocation. The same shape applied to the VRAM scratch — free the batch scratch at `prefill_end`, reallocate at `prefill_begin`, keep the carry — would return `~1.2 GiB` to the Hot pool **for decode** while prefill keeps a wide chunk and window.

It is **harder than the staging resize**, and for one structural reason: the Hot pool is sized at load and **every Hot slot is owned from the first token on** (`populate_round_robin`), so the registry's ownership model assumes a fixed capacity. Returning scratch mid-run means the registry re-derives its slot set — a residency-accounting change, not a buffer swap. It should be gated as hard as the staging resize was (all slots free, no lease outstanding, the `Hot pool = usable − …` identity re-checked, `invariants_hold()` at the boundary).

**This is the coupling the whole analysis turns on:** a big window/chunk *is* the high-throughput prefill of §1, and the scratch/carry a big window/chunk costs *is* the decode residency of §4. Returning the scratch on decode unlocks the large-window prefill without taxing decode.

## 5. The staging corridor is independent of both knobs (correction)

A framing error to record so it is not repeated: the linked staging corridor is **not** a function of the window, and **not** of the chunk.

```cpp
// runtime_config.hpp — staging_slot_counts
counts.prefill = cfg.prefill_sweep
    ? std::max(counts.batch, blocks * experts_per_layer)   // blocks × E
    : counts.batch;
```

With the sweep on it is `blocks × E` (`2 × 256 = 512` slots, `~3.4–6.75 GiB` pinned) — a function of **experts-per-layer and the `--staging-blocks` knob only**. The `min(6C, E)` term (`counts.batch`) never exceeds `E` and is dominated by `blocks × E`; it is only a phase cut when the sweep is off. So a bigger chunk or window leaves the pinned corridor unchanged.

## 6. The levers, ranked

1. **Bigger window `W`** (so the whole prompt is one pass and the model read is paid once). This is what buys throughput at flat TTFT; it costs carry (`96 KiB/token`) and, past the context, forces `W < N`.
2. **Bigger chunk `C`** (raise `kMaxTokens` and `kMaxChunk`). Keeps the GPU hidden under the read and improves its `ms/token`; costs chunk scratch (`~0.69 MiB/row`) and, past `8`-expert grouping, no kernel change.
3. **Return the scratch/carry on decode** (§4). The enabler that makes 1 and 2 free of the decode-residency tax.
4. **Lift `kAeonSwizzledMaxExperts`** via device-side pointer indirection (§2) — only matters if the `8`-expert launch grouping becomes a cost, which at `C = 1024` (fewer, larger groups) it may not.

## Open — the experiments that would settle it

1. **Measure the `2048`-token / `C = 1024` / `W = 4096` arm.** Raise both caps, run the pinned config with the prompt repeated to `~2048` tokens, read the `[Prefill timeline]` (load-wait vs issue+GPU) and the phase profile. The projection to beat: TTFT `~21–22 s`, `~95 tok/s`, GPU still under the read. This is the single test that confirms §1.
2. **Sweep `W` at fixed `N = 2048`.** `W = 0` (whole prompt, one pass) against `W = 2048` (one pass) against `W = 1024` (two passes, two model reads). Confirms that the read is per-window, not per-token — the core claim.
3. **A/B the scratch return.** Staging-resize-style: free the batch scratch and shrink the carry at `prefill_end`, and measure the decode Hot-slot gain and tok/s against the run that holds them. Gate with the staging resize's preconditions plus a registry slot-set re-derivation.
4. **Attribute the GPU `6 s` at `C = 1024`.** Confirm the grouped WMMA and dense GEMM actually reach their one-read floors at the wider `M`, so the `~7 ms/token` projection holds.
