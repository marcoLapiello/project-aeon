# Kernel Improvement Plan — throughput beyond GEMV

At the time this plan was written, every compute kernel in prefill was a GEMV (one token at a time): the chunk path looped over tokens on the host for every stage, so each chunk re-read the weights T times, and nothing used WMMA — the 16-row padding in scratch was there, but no kernel read it. So the swept prefill ran at GEMV rate (bandwidth-bound, around 1 flop per byte) when it could run at matrix-multiply rate.

**Scope: throughput in general, not one arm.** The swept prefill is profiled first because it has no supply constraint, so its host/GPU split reads cleanly — a *measurement* choice, not a claim about where the work belongs. Routed prefill and decode run the same per-token body (the same attention kernel, the same GEMVs, the same HC/norm stages), so a fix to those helps every arm; each record states its measured effect.

**Areas, not steps.** An area is a piece of work, not a position in a sequence: the numbers are write-order, and the status table is the work order, so an area's number ranks nothing. The name is deliberately not "step", because the work is done by convenience rather than top-down — Area 8 was found by measurement after Areas 1–4 were closed, and the largest single term when this began — a host loop in `compose_local_rows` — was never an area at all.

## Run configuration

The numbers below are only comparable if the invocation is fixed, and it drifted between sessions. This is the pinned command; every table in this document is a run of it unless a row says otherwise. Change one flag and the comparison is void, so record any deviation in the row that made it.

```
build/bin/aeon_chat \
  --model-dir models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon \
  --prompt "$(cat profiling-prompts/prefill-corpus.txt)" \
  --warm-gib 35 \
  --context-size 32768 \
  --prefill-window 4096 \
  --prefill-chunk 256 \
  --staging-blocks 3 \
  --max-new-tokens 32 \
  --diagnostic --verbose \
  --supply-telemetry <path.jsonl> --run-id <id> \
  --phase-profile
```

- **Prompt** — `profiling-prompts/prefill-corpus.txt`, trimmed to its first four-and-a-half paragraphs so the rendered prompt is ≈`700` tokens (the exact count is whatever `--diagnostic` prints; `profiling-prompts/first-prompt.txt` is the unrelated single-sentence prompt). A per-prompt-token figure is only comparable within one row, because the corpus was re-trimmed more than once and the pre-`4.1` rows were run on a `666`-token prompt or a `701`-token prefix of the longer corpus. The prompt-token count belongs beside every per-token number.
- **`--prefill-chunk 256`** — valid: `kMaxTokens = 256` (`v4_layer_body_batch.hpp`); the `--help` text still says `1..64`, so trust the constant.
- **`--prefill-window 4096`** — `W = 4096`, `C = 256`: `16` chunks fill a window, and the `~677`-token prompt is `ceil(677/256) = 3` chunk bodies per layer × `43` = `129`.
- **`--staging-blocks 3`** — the swept staging arena, carved **inside** the single pinned `--warm-gib` allocation (not added to it). It moves the prefetch depth, not throughput. Some rows below use `--warm-gib 24` to stay off the host ceiling.

### Host memory discipline

The host has `62.62 GiB` and the pinned config plus the runtime's own allocation peaks near `52 GiB` — about `98%` of the allowance (M47) — so the way a run fails here is memory pressure, not the engine. The pressure seen while this plan was written was **not** the engine's; it was the build beside it:

- **Never build next to a measurement.** `cmake --build … -j$(nproc)` spawns `64` `clang++` on heavyweight HIP template headers, each transiently holding `1–3 GiB`, so the parallel maximum is tens of GiB and the compiler's pages do not return instantly. Build and measure **serialised**, with a small job count (`-j8`).
- **VS Code is `~2.6 GiB`.** The `buffer/cache` column is reclaimable file cache, not pressure — read `available`, not `used`.
- **Do not shrink `--warm-gib` to dodge pressure.** The Warm tier is part of the pinned config and its size *is* a performance variable, so lowering it voids the comparison. Free host memory instead of changing the config.
- **One `aeon_chat` process at a time.** Two concurrent runs do not fit.

## Status

| Area | What | State |
| :--- | :--- | :--- |
| 1 | Grouped WMMA expert GEMM (W4A16) | **Done** — enabled in production |
| 2 | Batch the chunk loop in the layer body | **Partly** — phase split and router done; `M`-keyed dispatcher and batched KV/position writes open |
| 3 | WMMA dense GEMM for the dense projections | **Partly** — pre-attention projections and the shared expert batched; `v4_grouped_wo`, HC and dispatcher-on-`M` open |
| 4 | Batched causal attention over the chunk | **Done** — head-group WMMA kernel (heads as the M axis) on by default: attention+norm `8,730 → 226 ms` GPU |
| 4a | Indexer top-k on device | **Done** — the top-k only (`0.13 s`); the scores it feeds were Area 8 |
| — | `compose_local_rows` gather | **Done** |
| 5 | Fuse elementwise/norm over `[T, dim]` | **Partly** — E2, A, C, E1 and all of F/G (incl. the FFN norm and the dead replication) batched; E3/E3b/E4 (compressor/indexer feed) remain per-row |
| 6 | Tune for gfx1100 | **Partly** — attention closed by the head-group kernel; expert-shape/kernel sweeps open |
| 7 | Re-measure and retune supply | **Open** — prefill is NVMe-bound (timeline below); next: the restore off the TTFT path |
| 8 | Batch the indexer scores over the chunk | **Done** — `4.47 → 0.07 s`; pre-attention GPU `−59%`, TTFT `29.1 → 25.1 s` |

## Phase profile

### Baseline — 2026-10-02

The current measured state, on `main` after Areas 1–4 (incl. the head-group attention kernel), 4a, 8 and Area 5's E2, A, C, E1, F/G and the F/G residual, and Area 3's shared expert. Same pinned invocation as **Run configuration** (`677`-token prompt, `W=4096`, `C=256`, `Warm 35 GiB`, `3` staging blocks); a clean run after a contaminated one is discarded. Each area record below carries its own implementation-time before/after; this is the one cross-area reference, and it is **replaced, never accumulated** — a superseded baseline is deleted, not kept beside the new one. Ledger entries are taken only at the end of the plan.

| Phase | host ms | gpu ms | gpu share |
| :--- | ---: | ---: | ---: |
| **routed experts (batched)** | `1,061` | `2,457` | `40.9%` |
| routing dispatch | `1,590` | `1,601` | `26.7%` |
| **pre-attention (per token)** | `201` | `663` | `11.0%` |
| **attention+norm (per token)** | `8` | `226` | `3.8%` |
| &nbsp;&nbsp;· attention kernel (batched) | `1` | `21` | `0.3%` |
| router (batched) | `2,176` | `71` | `1.2%` |
| **total** | **`5,735`** | **`6,003`** | |

- **TTFT `25.3 s`**, prompt `677` tokens — flat against the `25.2 s` before the attention kernel, although prefill GPU fell `13.5 → 6.0 s`. The wall timeline (`[Prefill timeline]`, printed under `--phase-profile`) explains it:

| Wall segment | ms | What |
| :--- | ---: | :--- |
| layer loop: load wait | `12,427` | waiting for the layer's experts from NVMe |
| layer loop: issue + GPU wait | `8,254` | the bodies, overlapped with the next layer's load |
| end (`prefill_end`) | `4,468` | Warm refill of the borrowed staging slots + Hot restore |
| begin + head | `118` | |
| **total** | **`25,303`** | |

- **Prefill is NVMe-bound, and was before this plan's last areas.** The layer loop streams `142.7 GiB` in `~20.2 s` — `~7 GiB/s`, the drive's ceiling. The old `13.5 s` GPU already fit inside it; the earlier "GPU-bound" read came from the router's barrier line and was wrong. Compute savings now only grow `load wait`; TTFT moves only with supply (Area 7).

#### Pre-attention attribution

Pre-attention was one number (`~7 s`); the breakdown is what found Area 8 and Area 5's E2. Sub-regions fire once per `(row, stage)` and are **sampled at stride `16` and scaled back** — at stride `1` the two event records per region cost more than the region measures. The batched regions (E2, E5) are one launch per chunk and so are exact.

| Sub-region | gpu ms |
| :--- | ---: |
| **A hc mix + norm (batched)** | **`18`** |
| B x-projections | `35` |
| **C lora norm (batched)** | **`4`** |
| D q-projections | `108` |
| E pre-attn tail | `541` |
| &nbsp;&nbsp;· **E1 q-norm (batched)** | **`7`** |
| &nbsp;&nbsp;· E3 compressor state | `365` |
| &nbsp;&nbsp;· E3b indexer feed | `291` |
| &nbsp;&nbsp;· E4 materialize | `127` |
| **E2 rope + kv write (batched)** | **`3`** |
| **E5 indexer (batched)** | **`84`** |
| **pre-attention total** | **`711`** |

The remaining pre-attention terms — E3 (`365`), E3b (`291`), E4 (`127`) — are the compressor and indexer feed, which advance position-ordered state and fire on ratio boundaries; they are per-row deliberately, so batching them is the harder half of Area 5 that is left. E2, A, C and E1 each measured `~0` TTFT because the removed work overlapped the attention kernel.

### Decode phase profile — 2026-10-01

Decode carries the product metric (`3.6 tok/s`), so its phase split is measured the same way as prefill's. The two are separate tables — the engine resets the profiler after the prefill report — and the decode regions are named `(decode)` so they never mix. Run: the `first-prompt.txt` instruction, `64` tokens, the same Warm/window/chunk config.

| Phase | host ms | gpu ms | gpu share | per token |
| :--- | ---: | ---: | ---: | ---: |
| **moe (decode)** | `6,640` | `12,511` | **`79.1%`** | `~195 ms` |
| **pre-attention (decode)** | `240` | `1,590` | `10.1%` | `~25 ms` |
| **attention+norm (decode)** | `177` | `1,400` | `8.9%` | `~22 ms` |
| &nbsp;&nbsp;· E attention kernel (nested) | `11` | `382` | `2.4%` | `~6 ms` |
| &nbsp;&nbsp;· F/G hc + norm (nested) | `110` | `277` | `1.8%` | `~4 ms` |
| router (decode) | `8,637` | `315` | `2.0%` | `~5 ms` |
| **total** | **`15,693`** | **`15,816`** | | `~247 ms` |

**The `moe` region is mostly *wait*, not compute, and the supply counters show it.** Over the same `64` tokens the telemetry records `41.58 GiB` from NVMe and `60.58 GiB` host→VRAM: `nvme_wait ≈ 6.5 s`, `h2d_enqueue_to_ready ≈ 12.9 s` — `~202 ms`/token of H2D and `~101 ms` of NVMe wait inside a `~247 ms` token. Compute across all four regions is `~3.3 s` of the `15.8 s` (`~21%`). The `router`'s `8.6 s` host line is the same barrier artifact as in prefill (shown for completeness, not issuance).

**Read: decode's compute share is `~21%`; the rest is expert supply.** That matches the project status's "decode NVMe wait (`36–63%`) redirects the work from the transfer path to residency/placement", now attributable per region. A decode effort would therefore start from the supply/placement documents (`EXPERT_STREAMING_EXECUTION_PLAN`, the host-memory and placement investigations); it does **not** close the prefill compute areas above.

#### Correctness is intact — the run-config prompt is a timing fixture, not a chat

A first read of the baseline's reply looks alarming: the prompt is English but the answer is Chinese. It is **not** a regression, recorded so it is not re-raised:

- `prefill-corpus.txt` is a **raw English text blob with no instruction** — a prompt token-count fixture selected for the profile, not a question. Given that alone the model emits Chinese *commentary on the passage*, this model family's default for instruction-less input.
- Any instruction restores English: appending `"Question: In one sentence, what is this passage about?"` to the same corpus answers in English and on topic; the unrelated `first-prompt.txt` (`"Explain in simple terms how a hot and warm expert cache can reduce inference latency."`) answers in fluent English; `"What is the capital of France?"` answers `"The capital of France is Paris."`.
- The two default-on new paths do **not** move the output: with `attention_tile_enabled` and `moe_grouped_batch_enabled` toggled in all four combinations the corpus reply is byte-identical.
- The gates agree: `test_v4_layer_body_chunk_oracle`, `test_v4_prefill_window`, `test_v4_routed_prefill` and `test_v4_engine` (end-to-end) all pass at this commit.

**Conclusion:** correctness is sound; the Chinese output is an artifact of the fixture. Correctness must be measured with an instructed prompt (M47 used the corpus *plus an instruction*), which is why the run-config invocation alone is not the correctness signal.

## Findings

- The chunk "batch" was serial — pre-attention and attention+norm ran per row; fixed by Areas 3–4.
- The router made one host sync per row — fixed in Area 2.
- Small per-row launches (rmsnorm, rope, HC sinkhorn, KV copies, a position H2D per row) scale with T — Area 5; E2 batched (`0.73 s → 3 ms`), A/C/E1/E3/E3b remain.
- The **indexer scores** kernel, not its top-k, was the pre-attention cost: `4.3 s` against `0.13 s` — a per-token, single-block launch, the same defect Areas 2–3 fixed for the router and the projections. Batched in Area 8 (`4.3 s → 0.07 s`).
- **Attention: query tiling was the wrong axis.** Tiling and warp tuning of the query-tiled kernels were neutral (Area 6); putting the **heads** on the WMMA M axis (one shared KV head, so every key row serves 16 heads) cut the kernel `~8.5 s → 21 ms`.
- **TTFT is a supply metric for this prompt.** It includes the end-of-prefill Warm/Hot restore (`4.5 s`) and the NVMe stream (`~20 s`); read the `[Prefill timeline]`, not TTFT, for compute changes.

## Execution plan

Area 1, Area 4, Area 4a and Area 8 are done; the compose gather with them; Areas 2–3 are
partly done. Each record below is kept for the decisions in it — the status table is
authoritative for state.

### Area 1: W4A16 grouped WMMA GEMM for experts — **done**

Grouped W13+activation and W2 over a token→expert permutation, as one **fused** kernel split into a G2 WMMA loop, a G3 weight feed and a G4 epilogue. Three of the plan's original decisions were corrected by measurement and are superseded: `K` splits **`64`**-wide (not `128`), the kernel lives under `platform/rdna3/` with format and model injected as compile-time policies (not one combined G3 file), and **LDS double-buffering was a measured regression and was reverted**. The `M` window is a dispatcher choice, not a constant. The record below is the authority.

#### Implementation record

_Built `2026-09-29`. Synthetic microbenchmarks, recorded here rather than in the [Performance & Accuracy Ledger](../status/PERFORMANCE_LEDGER.md), which holds end-to-end runs only._

Built, smallest verified piece first:

| Artefact | Role | Gate |
| :--- | :--- | :--- |
| `src/platform/rdna3/wmma.hpp` | G2 primitive: the Wave32 `v_wmma_f32_16x16x16_f16_w32` and its operand lane map | `tests/test_rdna3_wmma_oracle.cpp` |
| `src/platform/rdna3/moe_grouped_ffn.hpp` (G2) + `src/backend/swizzled_w4a16/kernels/swizzled_w4a16_feed.hpp` (G3) + `src/architecture/deepseek_v4/kernels/moe_grouped_dispatch.hpp` (G4) | grouped W13+activation and W2 over a token→expert permutation, split into arch kernel / weight feed / model epilogue | `tests/test_v4_grouped_wmma_oracle.cpp` |
| `tests/bench_expert_pair_ab.cpp` | GEMV pair vs grouped pair: time, weight traffic, M window, crossover | `build/bin/bench_expert_pair_ab` |

The lane map was pinned on silicon before anything was built on it. It is not documented in one place, and a wrong reading still runs and still returns finite numbers — the common failure computes `A · Bᵀ`, which an identity-A test cannot see.

**Measured** (`bench_expert_pair_ab`, Device 0, synthetic experts in the real swizzled format, resident pool `128` experts (`1.69 GiB`, past the `96 MiB` Infinity Cache); draws sampled from the measured layer-0 prefill expert distribution of `routing-profile/first-real-prompt/counts.csv`; `T = 16 / 64 / 256 / 1024`; M windows `w1 / w2 / w4 / w8` token tiles (16–128 tokens sharing one dequantized slab); `n = 3`, best-of; every window compared elementwise against the GEMV arm and agreeing, `max |Δ| ≤ 1e-3 · scale`):

| T | tok/expt | GEMV ms | w1 | w2 | w4 | w8 | traffic `w4` / `w8` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1.9 | `2.56` | `3.20` | **`3.19`** | `3.50` | `4.69` | `1.9x` / `1.9x` |
| 64 | 5.1 | `10.17` | `4.98` | **`4.76`** | `5.13` | `6.80` | `5.1x` / `5.1x` |
| 256 | 20.2 | `39.65` | `7.67` | `6.12` | **`5.53`** | `7.25` | `20.2x` / `20.2x` |
| 1024 | 80.8 | `159.06` | `15.36` | `11.60` | `10.07` | **`9.99`** | `45.5x` / `70.6x` |

Best-window speedup over GEMV: `0.80x` (T=16) / `2.14x` (64) / `7.17x` (256) / `15.93x` (1024). At `T=1024` the traffic is GEMV `81.00 GiB`, `w4` `1.78 GiB`, `w8` `1.15 GiB` against a one-read-per-distinct-expert floor of `1.00 GiB`. The **window is a dispatcher choice, not a constant**: `w2` is fastest at `T ≤ 64`, `w4` at `256`, `w8` at `1024`, and all reach the one-read floor once the window holds the expert (default `w4`). Holding the dequantized slab across every token tile (K-outer) is what took `T=1024` from `15.18 ms` / `10.36x` to `9.99 ms` / `15.93x`, traffic `5.51 → 1.15 GiB`.

Correctness: all four windows agree with the GEMV arm at every chunk size; `test_rdna3_wmma_oracle` and `test_v4_grouped_wmma_oracle` green, the latter bit-identical (`max_rel` `3.969e-04` / `2.390e-04` / `8.427e-06`) across the loop-nest change and the double-buffer revert.

**Attribution — where the gate half's time goes** (`bench_expert_pair_ab`, `T = 1024`, each half run alone with the same grid and trip counts):

| Arm | ms | Regs | LDS | Wave slots |
| :--- | ---: | ---: | ---: | ---: |
| staging (dequant + LDS stores) | `3.24` | `39` | `16 KiB` | `16/32` |
| matrix multiply | `2.42` | `65` | `128 B` | `32/32` |
| production (gate half) | `6.24` — `33.1 TFLOP/s` | `186` | `16 KiB` | `16/32` |

The halves compose almost additively (`production/(staging+mma) = 1.10`), so there is no serialisation to recover; and **the staging is the larger half** (`52%`). Occupancy is LDS-capped, not register-capped — the `4 KiB`/wave slab permits `16` of `32` waves, and double-buffering it halves that and is a measured regression (`6.2 → 8.8 ms`), so it is out. Next lever: cut LDS per wave (K block `32` instead of `64`, which needs reworking the two-groups-per-thread staging) and cut the staging itself (it writes one half per element, so its store count is `4x` the global read it came from).

**Corrections to the plan, from the measurements:**

1. Area 3's `T ≥ 16` threshold is too thin: the crossover is between `T = 16` (`0.80x`, grouped loses) and `T = 64` (`2.14x`). Wire `T ≥ 64`.
2. K is split `64`-wide, not `128`: two 32-wide quantization groups map to one int4 load per thread, which is what makes the staging branch-free.
3. Chunk size and window are **user settings**, not tuning constants; the deliverable is the curve above, not a baked-in value. This also supersedes Area 6's "reach ≥ 16 tokens per expert" rule — `16` is reached at `T = 256` yet the reward keeps rising to `15.93x`.

**Remaining:** the routed-prefill / short-prompt path (same grouped kernel, with a per-expert GEMV fallback for 1–3-token experts), and the `M`-keyed dispatcher that picks grouped above the crossover (see Area 2).

### Area 2: Batch the chunk loop in the layer body — **partly done**

The phase split, the batched router and the layer-wide supply dispatch landed. The router's read-back is **one** synchronisation per chunk; the plan's "copy the ids to the host asynchronously, with no stall" is not reachable, because the host must have the ids before it can issue the layer's union — what was removable is the `C − 1` extra drains, and that is done. **Still open: the `M`-keyed dispatcher and the batched KV/position writes.**

#### Implementation record

_Started `2026-09-30`._

**Seam first (structural prerequisite).** The grouped pair was one G2+G3+G4 file under the G3 path (`aeon_moe_grouped_wmma.hpp`), calling the RDNA3 WMMA primitives, decoding the swizzled format and applying the DSV4 clamp, with the tile/window/batch choices as raw template parameters and no dispatcher. Wiring the executor to it would have built that coupling into production, so the seam landed first. The kernel stays one **fused** kernel — the slab reuse only pays while staging and MMA stay fused — but as compile-time policies, so its G2 file includes neither the format nor the model, and the executor reaches it through one G4 binding:

| G2 (kernel) | G3 (feed) | G4 (epilogue + binding) |
| :--- | :--- | :--- |
| `platform/rdna3/moe_grouped_ffn.hpp` — the WMMA loop, templated on the two policies | `backend/swizzled_w4a16/kernels/swizzled_w4a16_feed.hpp` — dequantize a K-tile into the LDS slab, and state the per-launch expert capacity | `kernels/moe_grouped_epilogue.hpp` (the clamp) + `kernels/moe_grouped_dispatch.hpp` (the binding and the tuning defaults) |

The **GEMV pair carried the same defect** and got the same fix: the SwiGLU baked into `aeon_moe_fused_w13.hpp` (G3) is now an injected `Epilogue` supplied by `kernels/moe_gemv_dispatch.hpp`, and the gfx11 `fdot2` moved out of the G3 decoder into the G2 primitive `platform/rdna3/dot2.hpp`. Architecture is resolved by neutral selectors (`platform/dot2.hpp`, `platform/moe_grouped_ffn.hpp`) keyed on a **build-visible** `AEON_ARCH_*` macro — not the device-only `__gfx1100__`, which would differ between the host and device passes — so adding an architecture is a new `platform/<arch>/` plus a selector branch, and no G3 or G4 file changes. Behaviour-preserving: `test_v4_grouped_wmma_oracle` bit-identical (`8.427e-06`), `test_v4_expert_oracle` unchanged, `test_v4_expert_executor` `moe_out` `0` differing elements. The superseded atomic control path deliberately keeps its coupling; its only consumer is a gate.

**Permutation primitive.** The grouped pair is driven by a token→expert permutation; nothing produced one before.

| Artefact | Role | Gate |
| :--- | :--- | :--- |
| `src/platform/ops/expert_permutation.hpp` | G2 primitive: device counting sort of the router's top-k ids into the expert-contiguous `(expert_offsets, token_indices, draw_indices)` the grouped pair consumes | `tests/test_expert_permutation.cpp` |

One property comes from the grouped pair rather than the sort, and one is a correctness requirement of the engine:

- **A single launch today holds at most `8` experts** (`kAeonSwizzledMaxExperts`). This is **not** a gfx1100 limit and **not** a property of the int4 format: it is the size of the kernels' weight-pointer table (`SwizzledW13ExpertPtrs`/`SwizzledW2ExpertPtrs`, fixed arrays passed by value as a kernel argument), a constant inherited from the GEMV kernels where one token's MoE touched `6` experts. A chunk's `6C` draws cover many more experts than that, so the executor splits the union into expert batches. Because the offsets ascend and an absent expert is zero-width rather than a missing slot, a batch is just a **contiguous expert range rebased by `expert_offsets[first]`** — no index copy between batches, since an empty expert cannot break the range apart. Lifting the cap would mean handing the weights by device-side pointer indirection instead of a by-value table; batching is the pragmatic alternative and also bounds the per-dispatch intermediate (`d_expert_hidden` is sized for one dispatch, not the whole union).
- **Placement is deterministic** (byte-reproducible restore): the scatter visits draws in index order, one thread per expert, instead of racing for a position.

Gate — five shapes (`1` token, `7`, `300`, a small `8`-expert count, and `draws = 0`) against the permutation's definition, not a second copy of the algorithm: monotone offsets anchored at `0` and the draw count, every draw under its own expert with `token = draw / slots`, `draw_indices` a bijection onto `0..draws-1`, ascending draw order within each expert, and widths equal to an independently counted histogram. Passes on `gfx1100`; the `300`-token case leaves `227` of `256` experts present with a widest expert of `94` draws (past one 16-row M tile), and the small-count case has all `8` present.

**Numerical effect — resolved.** The chunk body drives `accumulate_routed_batch`; the executor runs the per-token sequence unless `moe_grouped_batch_enabled()` is set, so one process compares both. The grouped path is a correct reorder, not a defect: `test_moe_grouped_batch_parity` is bit-identical on the synthetic fixture and one fp16 ULP in the real executor (`9.8e-4 … 2.0e-3`), which through `43` layers is a final-logit delta of `2.73e-1` against a top-2 margin of `0.820` — enough that a bit-exact chunk comparison fails by construction. It does not move the answer: `test_v4_routed_prefill` agrees on the greedy token and the whole 8-token continuation, so the bar became greedy agreement — the same move Area 4 later made for attention.

**Per-token router removed.** The router ran once per row, ending in a `hipStreamSynchronize` to read that row's top-k, so phase 2b drained the queue `C` times per layer. Its device work was already batched (`moe_router_kernel` is one block per token); only the read-back was per token. It is now `dispatch_router(count)` — one gate GEMV, one logit widening and one top-k launch for the whole chunk, buffers addressed by row stride (`ffn_norm_act` is the row-0 prefix of each token's padded tile) — read back with **one** synchronisation; decode calls it with `count = 1`. The GEMV primitive gained an activation row pitch (`x_stride`) so a batched caller reads rows in place.

Measured (`bench_prefill_ab routed 128`, bank arm, chunk `128`, window `1024`, warm `0`): `24.181 s` → `23.426 s` (`5.29` → `5.46 tok/s`). The run reads `66.6 GiB` over NVMe, which does not move, so this isolates the ~`0.75 s` of compute the `43 × 127` removed drains cost. Gates green: the layer-body oracles (`0` differing), `test_v4_prefill_window`, `test_v4_routed_prefill`, `test_v4_expert_executor`, `test_v4_mla_oracle`, `test_v4_shared_expert_oracle`, `test_v4_grouped_wo_oracle`, `test_v4_graph_head`, `test_v4_engine`.

**Remaining:** the `M`-keyed dispatcher (grouped above the crossover) and the batched KV/position writes. (The grouped path's chunk comparisons were moved to the agreement bar in Area 4.)

### Area 3: WMMA dense GEMM for attention and shared-expert projections — **partly done**

The same WMMA core as Area 1 (a single group). Remaining consumers: `v4_grouped_wo` and the HC projections. The row threshold is chosen by the dispatcher on `M`, GEMV below it (short-prompt routed prefill and decode).

#### Implementation record

_Started `2026-09-30`._

| Artefact | Role | Gate |
| :--- | :--- | :--- |
| `src/platform/rdna3/dense_gemm.hpp` + selector `platform/dense_gemm.hpp` | G2: `Y = X·Wᵀ`, fp32 accumulate, one wave = 16 output columns × `1/2/4` token tiles; both operands read from global (a weight row *is* the B fragment), no LDS | `tests/test_dense_gemm_wmma_oracle.cpp` |
| `layer/v4_dense_projection.hpp` | G4: `project_dense` — WMMA at `T ≥ 32`, per-token GEMV below (decode unchanged) | chunk / serial / compressed oracles |

Pre-attention is split into stages (`run_pre_attention_mix`, `_x_projections`, `_lora_norm`, `_q_projections`, `_tail`); decode runs them with `count = 1` and the chunk hoists the two projection rounds out of the row loop, so there is still one body. Row pitches are passed (`y_stride`), because the compressor/indexer buffers are strided for the ratio-4 width.

Measured (gate, vs `gemv_fp16_vec8_kernel` on a `(N, T)` grid): `T = 256`, `4096→1024` `0.81 → 0.053 ms` (`15x`); `1024→16384` `2.74 → 0.43 ms` (`6.4x`); `T = 37` `1.7x`; `T = 16`, `N = 64` `0.3x` (hence `T ≥ 32`). All within `1e-3` of a double reference. The crossover is lower than the expert GEMM's because the weight is fp16 (no dequant to amortize).

End to end (`aeon_chat`, swept, `W=4096 C=256`, `n = 1`, `701`-token prefix of `prefill-corpus.txt`; not the baseline prompt): pre-attention GPU `10.3 ms/token` against `21.9` before (`14.6 s / 666`), TTFT `43.5 s` against `50.0 s`. The attention kernel is now `24.5 s` of `67 s` GPU — Area 4. What remains in pre-attention is the per-token tail (RoPE, key write, compressor state, indexer scores/top-k: ~`30` launches per token).

Gates: the layer-body oracles and `test_v4_engine` green. `test_v4_prefill_window` failed `A`/`B` bit-exact (`4/10`) at the time — the grouped-experts reorder, later moved to the agreement bar (Area 4).

#### Implementation record — shared expert batched

The shared expert is a dense FFN that fires on every token, so it was four per-token launches (`w1` GEMV, `w3` GEMV, SwiGLU, `w2` GEMV) plus an `M_PAD·H` memset, re-reading `48 MiB` of dense weights **per token**. That is the same defect Area 3 fixed for the pre-attention projections, and unlike the routed experts its byte saving is **true** `C×` headroom, because every token reads the same dense weight (`294 MiB` per CSA token-layer against `81 MiB` for the routed path is the same argument at larger scale).

One `run_layer_body_moe_shared_expert_batch` now serves the chunk: `project_dense` for `w1`/`w3`/`w2` (WMMA above `kDenseGemmMinTokens`), one SwiGLU over the contiguous `count × INTER_DIM`, one memset for every token's accumulator tile. Below the threshold it keeps the exact per-token sequence — the crossover `project_dense` was measured at — which is also what keeps `test_v4_layer_body_chunk_oracle` bit-identical to serial, since its schedules are all `≤ 16` tokens.

Measured (`aeon_chat`, pinned config, `n = 1`, `677`-token prompt): the `shared expert` phase falls from `1,197 ms → 118 ms` GPU (`−90%`) and `245 ms → 4.3 ms` host (`−98%`).

**Coverage.** No gate *certified* the `count ≥ 32` branch when it landed: the window gate runs `kWindow = 16` and its chunked pass `5`, and the routed gate's window (`48`) was driven at chunk `16`; both are below `kDenseGemmMinTokens = 32`, so they exercised only the per-token fallback. The branch was *executed* by `test_v4_staging_depth` (it runs `forward_window(…, min(64, 256))`), but that gate checks timing-flatness and work-identity, not numerics. The routed gate now drives the same window again at `kBigChunk = 32` and compares it against the serial reference on the same bar (`21` checks, `0` failures; greedy `320` agrees, rel `1.2%`), so both branches of the dispatcher are certified in one pass (`kBigChunk` does not divide `kWindow`, so the window runs a `32` chunk and a `16` chunk).

### Area 4: Batched causal attention over the chunk — **done**

4.1 removed a per-query redundancy; 4.2 replaced the per-token launch with a batched tile plus a split-keys kernel covering all three classes, now **on by default**. WMMA and warp tuning remain (Area 6). Each primitive was pinned in fp64 against an independent reference before anything was built on it.

#### 4.1 — the per-query redundancy

The cached kernels (`v4_cached_sliding_window_attn_wave32_kernel`, `v4_cached_compressed_attention_wave32_kernel`, in `kernels/v4_attention_kernels.hpp`) re-read the loop-invariant query per key slot and ran the max and `expf` on all 32 lanes in lockstep (~`640 × 32` `expf` where `640` suffice on a CSA layer). The query is now read once into registers, and the max (an exact tree) and `expf` are lane-strided.

The denominator stays a **sequential ascending walk**: `scores[]` is indexed by the caller's row count, so a tree sum would regroup terms between the decode and chunk paths and break their bit-identity — the chunk oracle caught exactly that (`3` differing of `1.5M`). Only the max and the transcendental are parallelised. Gates: `test_v4_attention_sink_oracle`, `test_v4_mla_oracle`, `test_v4_layer_body_serial_oracle` / `_compressed_oracle` / `_oracle` / `_chunk_oracle` (bit-identical) and `test_v4_indexer_oracle` green.

_Pre-4.2 snapshot, not an A/B_ (untrimmed `1231`-token corpus): TTFT `86.1 s`, attention kernel `53,679 ms` GPU — ~`68%` of the GPU work and the largest single term. It established the shape 4.2 attacks: one query per block, one key at a time, a per-key wave reduction.

#### 4.2 — the batched launch

The per-token kernel is one block per **head** (`64` blocks of `32` threads) launched once per token — a single-wave grid that cannot fill `96` CUs. Two G2 primitives replace it, both model-agnostic: the DSV4 **sink is passed as a per-head `bias`** (a zero-value softmax candidate), and the second key block is an index array, so neither names a model concept.

- **Tile** (`causal_attention_fp16`): a block serves a `(head, query-tile)`, its queries sharing one key **union** masked to each query's own window. The union is `≤ W + tile − 1` rows; the tile's query rows are read **by stride** out of `d_q` (the `kMPad = 16` tiles are exactly this shape).
- **Split-keys** (`causal_attention_split_fp16`): `kCausalAttentionWarps` warps scan strided key slices and combine a partial `(max, sum, weighted value)` in shared memory; the second block is taken **per query** (an index array), which serves `CSA`'s indexer top-k without a second kernel.

Both take **two key blocks, each with its own window** (block 0 the recent window, block 1 the older compressed rows with `window = 0`), because every CSA/HCA layer attends `local + compressed` under **one softmax**. That covers all three classes — Sliding (one block), HCA (two shared blocks), CSA (shared local + a per-query index block) — and settles 4.2b: the fork was *gather each query's selection into a tile* or *keep the per-query selection and split the keys*, and the **key-split was built**, so no gather and no tiled CSA are needed.

The independent-oracle rule held: the formulation (masked-union ≡ per-query window, bit-for-bit) was pinned in fp64 in `tests/test_v4_tiled_attention_oracle.cpp` before any kernel; the primitives are gated in `tests/test_tiled_causal_attention.cpp` against the same reference. G4 binding: `layer/v4_attention_tile.hpp` supplies the scale, sink and blocks; the kernel splits out of `run_layer_body_attention_and_norm` into `run_layer_body_attention_kernel` (batched launch) + `run_layer_body_attention_tail` (per row); a union compose kernel (`v4_compose_union_rows_kernel` + `compose_tile_rows`) builds the position-labelled row-set.

Two bugs the independent gate caught, not self-consistency: a warp-stride-only softmax made every lane sum the **same** keys (the wave reduction multiplied the denominator by `32`), and phase 3 formed a value pointer for **masked** keys, so a CSA `-1` index was an out-of-bounds load that only the production run hit.

**Measured** (`n = 1`, `~677`-token prompt, `--warm-gib 24`):

| | tile off | + tile (Sliding+HCA) | + split-keys (all) |
| :--- | ---: | ---: | ---: |
| TTFT | `37.2 s` | `34.4 s` | **`29.5 s`** |
| `attention+norm`, GPU | `20,117 ms` | `17,676 ms` | **`11,895 ms`** (`1.69x`) |
| total, GPU | `43,260 ms` | `38,357 ms` | **`26,054 ms`** |

**Enabled by default.** The tiled split-keys path is the production path for all three classes (`attention_tile_enabled()` defaults true; setting it false replays the scalar kernel for a gate). Because the tile reorders, the gates that asserted chunk-vs-serial **bit-exactness** moved their chunk comparison to the **agreement bar** (greedy token agrees, values within a fraction of peak) while the scalar path keeps its bit-exact regression:

| Gate | Before | After |
| :--- | :--- | :--- |
| `test_v4_layer_body_chunk_oracle` | bit-exact C/C2 | C/C2 with the tile **off**; C3 parity, all classes |
| `test_v4_prefill_window` | `4/10` (pre-existing red) | **`10/0`** — greedy `320`, rel `2.6%` |
| `test_v4_routed_prefill` | `20/1` (pre-existing red) | **`20/0`** — greedy `320`, rel `0.1%` |

#### 4.3 — heads as the WMMA M axis, **done**

The query-tiled kernels read each key row once per head. DSV4 has one shared KV head, so `head_group_attention.hpp` (G2) gives a block `(query, 16 heads)`: `S = Q[16 heads]·Kᵀ` and `O = P·V` are both WMMA tiles, the valid keys (window or index selection) are compacted first, and an online softmax consumes them in blocks of `WARPS × 16` keys. Same contract as the split kernel, which stays as the fallback for shapes it does not instantiate.

| | split kernel | head-group |
| :--- | ---: | ---: |
| attention kernel, GPU | `~8,500 ms` | **`21 ms`** |
| attention+norm, GPU | `8,730 ms` | **`226 ms`** |
| prefill total, GPU | `13,548 ms` | **`6,003 ms`** |
| TTFT | `25.2 s` | `25.3 s` (NVMe-bound, see the baseline) |

Gates green: `test_tiled_causal_attention`, `test_v4_layer_body_chunk_oracle`; the corpus reply is unchanged in kind.

#### WMMA QKᵀ — built, gated, **not shipped**

The split kernel is scalar per element. A WMMA variant (`causal_attention_wmma_qk_fp16`, `dispatch_causal_attention_wmma_qk_fp16`) computes the `QKᵀ` tile on the matrix cores, gated in `tests/test_tiled_causal_attention.cpp` case G. It is **correct but not faster**, and both facts are worth keeping:

- **Only QKᵀ fits the matrix cores.** A `16 × 512` PV output tile is `8192` fp32 — `256` registers per lane at one wave, past gfx11's limit — so PV stays scalar with a `16`-register/lane accumulator. A WMMA PV would need the output tile staged in LDS, which tightens LDS toward its cap.
- **One warp per block cancels the gain.** Measured `attention+norm` GPU `11,888 ms` against the split kernel's `11,895 ms` (TTFT `29.8 s` vs `29.5 s`) — i.e. nothing. The WMMA block holds one wave; the split kernel holds `kCausalAttentionWarps = 4`. The matrix-core throughput and the occupancy trade exactly.

So production stays on the split kernel; the primitive remains available for a multi-warp WMMA attempt, which is the only version that could pay. A first-cut bug worth noting: the kernel read the query without the per-head offset, so head `0` matched and every other head did not — an index error the fp64 gate caught immediately.

### Area 4a: Move the indexer top-k on-device — **done**

`select_indexer_topk` selected the CSA layer's `index_topk = 512` compressed rows **on the host** — D2H the scores, `hipStreamSynchronize`, CPU `stable_sort`, H2D the indices, `hipStreamSynchronize` again — once per token per CSA layer (~`38k` drains per window), the mechanism behind the router's `13.8 s` host line at `0.1%` the frequency.

`v4_indexer_topk_kernel` (G4, `v4_attention_kernels.hpp`) does it in one launch: one block per row, iterative max-extraction over the unselected candidates, higher score wins and on an exact tie the **lower index** — a total order, so it matches the host's descending `stable_sort` exactly and no score leaves the device. The mask is a `ceil(candidates/8)`-byte shared-memory bitmap (no VRAM); `select_indexer_topk` keeps its signature and only launches the kernel.

Measured (`aeon_chat --phase-profile`, 666-token prompt, `W=4096 C=256`, swept, `n = 1`):

| Phase | host before → after | gpu before → after |
| :--- | :--- | :--- |
| pre-attention (per token) | `13,057` → **`944`** | `15,718` → `14,657` |
| attention+norm (per token) | `37,271` → `48,343` | `46,954` → `46,944` |
| **total** | `66,238` → **`65,145`** | `66,430` → `65,339` |
| **TTFT** | `71.9 s` → **`70.8 s`** | |

**The honest reading: this removed the stall but not the bottleneck.** Pre-attention's *issuing* cost fell `13.0 s → 0.9 s` — the two drains were real and are gone. But the freed CPU immediately ran ahead into the attention loop, whose own per-token issuance then became the exposed critical path (`37.3 → 48.3 s`); the window is bounded by total host time, so prefill moved only ~`1.5%`. This is the profile's third point restated: the pre-attention sync was expensive *in isolation* and never on the critical path while the attention loop's ~`860k` launches (`666 × 43 × ~30`) were.

Gates: `test_v4_real_scale_state` (the real `index_topk = 512` as a strict selection, `0` differing), `test_v4_indexer_oracle`, `test_v4_layer_body_serial_oracle` / `_compressed_oracle`, `test_v4_engine` (`38/0`) green.

### Area 5: Fuse elementwise and norm stages over [T, dim] — **partly done**

- Make rmsnorm, rope, HC sinkhorn and the residual/HC mixing multi-row (one launch per stage per chunk, one row per wave). Fuse the rmsnorm scaling into the next GEMM's A-load where it's cheap.
- Capture the per-layer chunk sequence in a HIP graph for the swept path, where shapes are fixed per chunk size.

#### Implementation record — E2 (rope + kv write), **done**

The tail ran two `v4_forward_rope_at_pos_wave32_kernel` launches (query heads, then the single key head), two identical key/value `hipMemcpyAsync` and a per-row position H2D, per row: `0.73 s` of GPU, the largest per-row term. The chunk's per-row buffers are contiguous (`d_q` at pitch `num_heads*head_dim`, `d_kv_norm_act` at `head_dim`), and the G2 `v4_forward_rope_wave32_kernel` already indexes the table by `blockIdx.y`, so the whole chunk is one `(heads, rows)` launch with the table **offset by the chunk's first position** — row `r` reads position `start + r` with no new kernel. The chunk writes key and value to the *same* buffer, so the two per-row copies collapse to one bulk copy. The per-row position write is dropped: `chunk_positions_` is never read (the composed row-sets carry kernel-computed positions and the commit recomputes the ring slot). The tail takes a `defer_rope_kv_write` flag, mirroring `defer_indexer_select`; decode and single-row callers keep the inline path, so the scalar and compressed oracles are unchanged.

Measured (`n = 3`, stable runs):

| | before | after |
| :--- | ---: | ---: |
| E2 rope + kv write, GPU | `732 ms` | **`3 ms`** |
| pre-attention, GPU | `2,955 ms` | **`2,429 ms`** |
| total, GPU | `18,956 ms` | **`18,433 ms`** (`−2.8%`) |
| TTFT | `25.1 s` | `25.0 s` (flat) |

**Read:** the GPU saving is real (`−0.53 s`) but TTFT is **flat** — the removed work overlapped the attention kernel, which is `48%` of GPU. Same shape as Area 4a's result: it removes the cost but not the bottleneck. The remaining Area 5 stages (A, C, E1, E3, E3b) are untried; whether they move TTFT depends on the attention term they overlap.

Gates green: `test_v4_rope_oracle`, `test_v4_layer_body_chunk_oracle` (bit-exact with the
tile off), `test_v4_layer_body_serial_oracle`, `test_v4_layer_body_compressed_oracle`.

#### Implementation record — C (lora norm) and E1 (q-norm), **done**

Two more per-row stages, batched the same way E2 was.

- **C (`run_pre_attention_lora_norm` → `_batch`).** The two norms between the projection rounds (`qa → qa_norm`, `kv → kv_norm_act`) ran once per row. `rmsnorm_wave32_kernel` already takes its row on `blockIdx.x`, and the chunk's `qa`/`qa_norm` (pitch `Q_LORA`) and `kv`/`kv_norm_act` (pitch `HEAD_DIM`) are contiguous per token, so the batch is just the row count: `dim3(1) → dim3(count)`.
- **E1 (`run_pre_attention_tail` → `run_chunk_q_norm_batch`).** The per-head weightless query norm ran once per row (`dim3(NUM_HEADS)`); the chunk's `d_q` is one contiguous `count × num_heads × head_dim` block, so flattening it to `count × num_heads` rows of `head_dim` serves every head of every row in one launch. It sits in the tail (it precedes the query RoPE that also writes `d_q`), so it is hoisted to the batched region with a `defer_q_norm` flag — mirroring `defer_rope_kv_write` — and the chunk calls `run_chunk_q_norm_batch` **before** `run_chunk_rope_kv_write_batch`, preserving the tail's order. Decode and single-row callers keep the inline path.

Both are the **same kernels** with a different grid — no arithmetic changed — which is why the chunk oracle stays bit-exact.

Measured (`n = 3`, `<1%` across runs):

| | before | after |
| :--- | ---: | ---: |
| C lora norm, GPU | `447 ms` | **`3.3 ms`** |
| E1 q-norm, GPU | `396 ms` | **`6.8 ms`** |
| pre-attention, GPU | `2,429 ms` | **`1,761 ms`** (`−27%`) |
| prefill total, GPU | `18,433 ms` | **`17,931 ms`** (`−2.7%`) |
| TTFT | `25.0 s` | `25.5 s` (flat, within noise) |

**Read:** `≈843 ms` of GPU removed for two launches, and pre-attention drops `667 ms` — the same shape the earlier batches showed, with the rest of the saving absorbed by the attention kernel's overlap. TTFT stays flat.

Gates green: `test_v4_layer_body_chunk_oracle` (bit-exact: every token, ring, compressed entries, partial ring, final state — across all classes and four schedules), `test_v4_layer_body_serial_oracle`, `test_v4_layer_body_compressed_oracle`.

#### Implementation record — A (hc mix + norm), **done**

The HC pre-mix stage ran four launches per row (`hc_project`, `hc_sinkhorn_normalize`, `hc_pre_combine`, the attention `rmsnorm`) and was, after E1, the largest remaining pre-attention term. The two HC kernels that lacked a token axis got one — `hc_project_kernel` and `hc_pre_combine_kernel` now take the token on `blockIdx.y`; `hc_sinkhorn_normalize_kernel` (`blockIdx.x`) and `rmsnorm_wave32_kernel` (`blockIdx.x`) already had it — so `run_pre_attention_mix_batch` is four launches for the whole chunk. The per-token buffers are contiguous at their vector pitches (`res_in` at `hc_mult*hidden`, `mixes` at `24`, `pre`/`post` at `4`, `comb` at `16`, `x_pre`/`x_norm` at `hidden`), so no pitch is passed and a single-token launch (`blockIdx.y == 0`) is the old behaviour exactly. The `hc_project` redundancy — every one of its 24 mix blocks recomputing the full residual RMS — is untouched; only the per-row launch is gone.

Measured (clean run on a quiet host; a first run was contaminated and discarded):

| | before | after |
| :--- | ---: | ---: |
| A hc mix + norm, GPU | `1,114 ms` | **`18 ms`** |
| pre-attention, GPU | `1,761 ms` | **`711 ms`** (`−60%`) |
| prefill total, GPU | `17,931 ms` | **`16,767 ms`** (`−6.5%`) |
| TTFT | `25.5 s` | `25.1 s` (flat, within noise) |

**Read:** the largest single per-row stage in pre-attention, removed for four launches. The `~1.1 s` of GPU is real and, unlike the smaller stages, shows up in the total (`−1.16 s`) because it was large enough to exceed the attention kernel's overlap. TTFT still reads flat — `~0.4 s` inside a `25 s` run of noise.

Gates green: `test_v4_layer_body_chunk_oracle` (bit-exact), `test_v4_layer_body_serial_oracle`, `test_v4_layer_body_compressed_oracle`, `test_v4_hc_oracle`, `test_v4_hc_head_oracle`.

#### Implementation record — F/G (hc post + HC FFN pre-mix), **done**

The FFN-side twin of A: HC post expansion (`float_to_half`, `hc_post`, `half_to_float`), the HC FFN pre-mix (`hc_project`, `hc_sinkhorn_normalize`, `hc_pre_combine`) and the FFN RMSNorm, all per row — the largest per-row term in the body after A. It now rides the same machinery: the two casts are flat over the chunk's contiguous buffers, `hc_post`/`hc_project`/`hc_pre_combine` take the token on `blockIdx.y` (the first two were readied by A's change), the Sinkhorn on `blockIdx.x`. The tail takes a `defer_hc_ffn` flag and the chunk runs one `run_hc_ffn_batch` after the per-row tails, for **both** attention paths (tile on and off) so the tile-off path the chunk oracle compares bit-exact exercises the batched code.

**Two constraints shaped it, and both are worth recording:**

1. **The final norm and the padded-row replication stay per-row *in this step*.** `ffn_pre` is compact (pitch `H`) but `ffn_norm_act` is a `M_PAD`-row tile (pitch `M_PAD*H`), and `rmsnorm_wave32_kernel` writes at its input pitch — a single batched norm would have token `r` write over token `r+1`'s tile. A first attempt did exactly that and the chunk oracle caught it (`100 of 130` tokens differing). Both were removed in the follow-up residual step below.
2. **The replication is dead but kept *in this step*.** The grouped W13 kernel reads `activation + activation_row * stride` for `K` elements — row 0 only — and the router and shared expert read `in_dim = H` from each tile base, so rows 1..15 are never read. Removed in the follow-up residual step, with the chunk oracle as the proof it was dead.

Measured (clean run; a prior run was contaminated and discarded):

| | before | after |
| :--- | ---: | ---: |
| attention+norm, GPU (now includes the batched F/G) | `11,891 ms` | **`10,638 ms`** (`−1,253`) |
| prefill total, GPU | `16,767 ms` | **`15,317 ms`** (`−1,450`, `−8.6%`) |
| TTFT | `25.1 s` | `25.4 s` (flat, within noise) |

**Read:** the largest remaining per-row term in the body, batched; unlike the smaller stages it clears the attention kernel's overlap and shows in the total. What is left of F/G is the per-row norm and replication above — the next lever inside this stage.

Gates green: `test_v4_layer_body_chunk_oracle` (bit-exact, all classes and four schedules), `test_v4_layer_body_serial_oracle`, `test_v4_layer_body_compressed_oracle`, `test_v4_hc_oracle`, `test_v4_hc_head_oracle`.

#### Implementation record — the F/G residual (FFN norm + dead replication), **done**

The two constraints above, removed. `rmsnorm_wave32_kernel` gained an `out_stride` parameter (default `dim`, i.e. the previous behaviour at every existing call site), so the FFN norm writes row 0 of each token's `M_PAD`-row tile in **one** batched launch instead of one per row; and the padded-row replication is dropped from the batched path, because nothing reads it — the grouped W13 kernel, the router and the shared expert all read `in_dim = H` from a tile base. Decode keeps both its inline norm and its replication: it is the product path, and this change is a prefill one.

Measured (two consecutive clean runs):

| | before | after |
| :--- | ---: | ---: |
| attention+norm, GPU | `10,638 ms` | **`8,730 ms`** (`−1,908`) |
| prefill total, GPU | `15,317 ms` | **`13,548 ms`** (`−1,769`, `−11.5%`) |
| TTFT | `25.4 s` | `25.2 s` (flat, within noise) |

**Read:** `16` launches and copies per row (`1` norm + `15` replications) collapse to one batched norm, and the dead replications disappear. The chunk oracle confirms it: it stays bit-exact with the padded rows **never written**, which is the proof that the replication was dead. This is the second F/G-sized term the residual contained, and it is why the norm pitch mattered enough to change the primitive.

Gates green: `test_v4_layer_body_chunk_oracle` (bit-exact), `test_v4_layer_body_serial_oracle`, `test_v4_layer_body_compressed_oracle`, `test_v4_graph_body`, `test_v4_engine` (end to end).

### Area 6: Tune for gfx1100 — **partly done**

- Sweep the WMMA tile config for the dominant expert shapes: N×K tile, waves per workgroup (4–8), LDS ≤ 64 KiB for 2 workgroups per CU.
- Give the routed-prefill (short-prompt) path the same grouped kernel. Experts with only 1–3 tokens fall back to the GEMV kernel, picked per expert inside one launch.

#### Implementation record — attention variants, measured

Attention is `48%` of prefill GPU, so Area 6's first target was its kernel. The one-warp WMMA-QKᵀ build was recorded **neutral**; the question was whether **query tiling** (the WMMA tile reads the key union once per 16 queries, the split kernel once per query) or a **warp-count** change could break the tie. Both were measured on `gfx1100`, at the pinned config, one process at a time:

| arm | attention+norm GPU | notes |
| :--- | ---: | :--- |
| **split kernel, warps=4, tile=16 (production)** | **`11,855 ms`** | the incumbent |
| WMMA-QKᵀ, tile=16 | `11,935 ms` | the variant the plan called neutral |
| WMMA-QKᵀ, tile=256 | `12,049 ms` | wider union, one launch per 256 queries |
| split kernel, warps=1 | `12,394 ms` | fewer warps, less per-block overhead |

**No tested variant beat the incumbent.** The WMMA-QKᵀ path is neutral-to-worse at every tile width, and the split kernel's warp count is already best at `4` (`1` is worse). The tile is why: the WMMA kernel's block is always 16 queries (its PV is `16 × head_dim` registers), so "tile=256" only enlarges the composed union (`W + tile − 1` rows) that every query then masks down to `W` — more wasted compute, not less. Occupancy and that masked over-read cancel; a **multi-warp WMMA with the PV tile in LDS** is what would have to beat them, and it is untried.

**A synthetic bench did not transfer, and that is worth recording.** `tests/bench_attention_ab.cpp` (a new tool, kept, no model load) measured WMMA-QKᵀ **2× faster** than split at `rows=297, count=256`, but production runs the same arms and is neutral. The bench's grid and union (`count` queries in one launch, a 297-row union) are not production's (16-query tiles, a 128-row sliding union plus a compressed block). A synthetic attention bench must use production's tile and union shapes, or it measures a different question.

**A latent bug fixed.** `dispatch_causal_attention_wmma_qk_fp16` sized its shared memory as `16 × tpad × 6` bytes with no guard; above `tpad = 683` (union `≈ 672` rows) the launch exceeded the workgroup limit and `<<<>>>` **drops the error**, leaving stale output rather than failing. The dispatcher now refuses an over-large row-set with a named exception. It matters only if the WMMA path is ever wired, but the silent failure was reachable from `bench_attention_ab`.

The split kernel is now a template on its warp count (`WARPS`, default `4`) so the sweep needs no rebuild; production instantiates the default. Gates green: `test_tiled_causal_attention` (both arms, all cases), `test_v4_layer_body_chunk_oracle`.

### Area 7: Move the bottleneck back to supply — **open**

- Measured (baseline timeline): compute is now faster than supply, and the sweep's NVMe reads are already minimal — `8,896` experts (`117.3 GiB`) `= 11,008 − 1,886` Warm-sourced `− 226` preserved Hot, at `~5.8 GiB/s` (the drive's ceiling). No resident is re-streamed; the `nvme_gib` printed beside the sweep was the **whole run's** NVMe (prefill + decode) and is now `run_nvme_gib`. The levers left:
  - the end-of-prefill restore (`512` drained Hot + `756` Warm slots the staging arena borrowed, re-read blocking) sits before the first token: skipping it measured TTFT `25.3 → 20.9 s` with decode `3.56 → 3.49 tok/s` (noise) — move it off the TTFT path rather than drop it;
  - the staging arena's Warm borrow costs its experts twice (read in the sweep, re-read in the restore).

### Area 8: Batch the indexer scores over the chunk — **done**

Found by the `2026-10-01` pre-attention attribution, not by the plan: `v4_indexer_scores_kernel` (`E5a`) was `4.3 s` GPU — `60%` of pre-attention and `18%` of all prefill GPU — against `0.13 s` for the top-k it feeds. The defect is the one Areas 2–3 fixed twice: `dim3(ceil(committed / 256))` = **one block**, launched once per token per CSA layer (`21 × 677 ≈ 14k` times at this context).

The batch is one grid with the row on `blockIdx.y` and a row pitch on every buffer (`v4_indexer_scores_batch_kernel`), the shape `dispatch_router` and `project_dense` already take. The ordering is the whole subtlety: a row's compressed entries are materialized **during its own tail** (at ratio boundaries), so the scores run only once the whole chunk's tails have run — and a row ranks only the `committed` entries that predate it, so the later rows' entries are never read. The tail therefore takes `defer_indexer_select`; decode and every single-row caller keep the default and select inline, which is why the layer-body compressed oracle is unaffected. The top-k stays per row: cheap, and its candidate count differs per row.

Measured (`n = 3`, `<1%`):

| | before | after |
| :--- | ---: | ---: |
| indexer scores + top-k, GPU | `4,472 ms` | **`70 ms`** |
| pre-attention, GPU | `7,305 ms` | **`2,955 ms`** (`−59%`) |
| total, GPU | `23,791 ms` | **`18,956 ms`** (`−20%`) |
| TTFT | `29.1 s` | **`25.1 s`** (`−14%`) |

Gates green: `test_v4_indexer_oracle`, `test_v4_layer_body_compressed_oracle`,
`test_v4_layer_body_serial_oracle`, `test_v4_layer_body_chunk_oracle` (bit-exact with the
tile off), `test_v4_real_scale_state` and `test_v4_engine` (end-to-end).

## Dependencies

`G2/G3` (kernels) → `G4` (batched layer body and attention) → `G1` (chunk sizing and
lookahead). Each area is verified alone with an independent-oracle test before the next.