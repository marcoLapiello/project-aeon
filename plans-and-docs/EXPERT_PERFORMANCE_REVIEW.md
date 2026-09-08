# Project Aeon — Expert Performance Review (Post-M15)

* **Date**: 2026-09-08
* **Scope**: Full source-level review of the decode path after Milestone M15 (asynchronous Hot-to-Warm demotion, controlled Warm A/B). Covers `v4_pipeline.hpp`, `expert_registry.hpp`, `vram_expert_pool.hpp`, `host_expert_pool.hpp`, `prefetch_staging.hpp`, `direct_io_reader.hpp`, `aeon_loader.hpp`, `w4a16_gemm.hpp`, and the GEMV kernels in `v4_attention.hpp`.
* **Triggering observation**: Hot+Cold throughput (~4.37 tok/s) is nearly identical to Hot+Warm+Cold (~5.15 tok/s), even though Cold NVMe reads run at ~3 GiB/s while measured PCIe bandwidth is ~25 GB/s.

---

## 0. The Smoking Gun: Why Hot+Cold ≈ Hot+Warm+Cold

Byte accounting for one non-Hot expert request in the current implementation:

### Warm hit (pre-fix)
1. `stage_payload`: blocking CPU `memcpy` of 14.15 MB, Warm slot → staging arena.
2. `allocate_vram_slot` evicts the LRU Hot expert → `download_to_host_expert`: **14.15 MB D2H demotion** on `sdma_stream`, enqueued *ahead of* the critical H2D uploads.
3. `upload_from_host_expert`: 14.15 MB H2D in 6 fragments.

**Total: ~42 MB of host+PCIe traffic per warm hit.**

### Cold miss (pre-fix)
io_uring read 13.5 MB into staging + same D2H demotion 14.15 MB + same H2D 14.15 MB.

**Total: ~41.5 MB.**

The warm tier's only saving was replacing a 13.5 MB NVMe read with a 13.5 MB CPU memcpy. Everything else was identical. Bandwidth was never the constraint — **serialization and self-inflicted traffic were**.

### Defect A — Hot→Warm demotion is pure waste
Expert weights are **immutable**; the payload always exists in `model_experts.aeon`. Copying an evicted Hot expert back to host RAM costs 14.15 MB D2H on the same `sdma_stream` ahead of the critical upload. Every miss paid ~28 MB of DMA when 14 MB was needed.

**Fix**: on eviction, mark the expert `COLD_NVME` and free its slot. Never copy VRAM→host for demotion. (Optional later: lazily re-fill warm slots from disk at idle priority.)

### Defect B — Warm-hit staging memcpy is redundant and runs while the GPU is idle
Warm segments allocated with `hipHostMalloc` are pinned: H2D can go **directly from the warm slot**, no staging copy. Worse, `dispatch_layer_prefetch` ran *after* the router `hipStreamSynchronize` and *before* the shared-expert kernels were queued, so the GPU sat completely idle during the 14 MB memcpy.

**Fix**: (1) skip staging for pinned warm slots; (2) enqueue shared-expert kernels *before* any CPU staging work so staging overlaps compute.

### Defect C — One stream serializes everything
Demotions (D2H), warm uploads (H2D), and cold uploads (H2D) all shared `sdma_stream`. Defect A removes demotions from the request path; a later step should still split critical-path H2D uploads from background traffic.

---

## 1. Root Architectural Flaw: No Prefetch Horizon for 40 of 43 Layers

The "prefetch pipeline" only prefetches hash-routed layers 0–2, where expert identity derives from `token_id` alone. For every routed layer L ∈ [3, 42], the expert set is known only after:

layer L FFN-norm → router GEMV → **D2H sync** → CPU reads top-6 → *then* I/O dispatch.

That is two full `hipStreamSynchronize` drains per layer (86 per token). For 93% of layers, "prefetch" is synchronous just-in-time loading; the only overlap window is the shared expert (~0.3–0.5 ms of GEMV) against a 2.2–4.5 ms storage read. Storage latency sits on the critical path of every routed layer, every token. No tier configuration can fix this — it is a *scheduling* problem.

### The transformative fix: speculative cross-token prefetch
MoE routing has strong temporal locality — token t+1 at layer L typically reuses most of token t's experts. The `activation_count` / `moving_frequency` infrastructure in `ExpertRegistry` is currently written but never read.

Plan:
- After computing layer L's top-6 for token t, immediately dispatch layer L's loads for token t+1 using token t's top-6 as prediction (or the union of the last n tokens' choices, capped by staging capacity).
- This gives a **full-token prefetch horizon (~200 ms)** instead of ~0.4 ms. Even at 50–60% speculation accuracy, most cold/warm requests leave the critical path; on mispredict, fall back to the just-in-time path.
- First measure: log top-6 Jaccard overlap between consecutive tokens per layer. If ≥50%, this is the single biggest unlock.
- Enlarge the staging arena beyond 12 slots (12 × 13.5 MB = 162 MB): a 48–96 slot arena (0.6–1.2 GB pinned) enables 1+ token-deep speculation for several layers concurrently.

---

## 2. Compute Findings (the post-storage floor)

### F6 — W4A16 GEMM runs at 1.9 TFLOP/s (~1.5% of WMMA peak)
Ledger M2: 140 µs for M=16, N=2048, K=4096. Problems in `wmma_fused_int4_gemm_kernel`:
- **Occupancy**: `GEMM_BLOCK_N = 64` → 2048/64 = 32 blocks on a 96-CU GPU. Two-thirds of the silicon idles.
- **K-loop**: 4096/16 = 256 iterations with **two `__syncthreads()` per iteration**, no double buffering.
- **Dequant**: scalar nibble extraction per thread into LDS; no `uint4` vectorized global loads; no register-level dequant.
- **Memory floor check**: each expert GEMM reads ~4.7 MB of weights; at 960 GB/s VRAM bandwidth that is a **~5 µs floor**. Current: 140 µs — 28× above the *memory* floor. (M=16 with 15 replicated rows is effectively a GEMV; memory-bound is the right target.)

Routed experts cost ~2.5 ms/layer × 43 = **~108 ms/token (~43% of the 252 ms step)**. A tuned kernel (K-tile 64–128, double-buffered LDS or register pipeline, N=128–256 per block, vectorized packed loads, dequant to registers, fused SwiGLU-accumulate epilogue) should land within 2–3× of the memory floor: **~20–30 µs per expert triple** → routed compute drops to ~10–15 ms/token. Largest single compute win in the project.

### F7 — All dense GEMVs use scalar 2-byte loads, one warp per row
`v4_gemv_fp16_kernel`: one block per output row, 32 threads, scalar half loads — a warp moves 64 B per transaction. Fixes: `uint4` loads (8 halves/access), 4–8 warps per block with split-K reduction. The LM head alone reads 1.06 GB/token; at ~30% scalar-load efficiency that is ~3 ms/token — vectorized ~1.2 ms. Per-layer dense projections (~260 MB/layer) improve similarly across 43 layers.

### F8 — CPU argmax + router round-trips
258 KB D2H + sync + 129,280-element CPU scan every token. Replace with two-stage GPU argmax and a 4-byte D2H. Same for the router: half→float logit conversion currently happens on the CPU between two syncs — do it in a kernel (or let the router read half directly), collapsing to **one** D2H + sync per layer.

---

## 3. Secondary Findings (cheap, mechanical)

1. **SoA pool layout fragments transfers** — `upload_from_host_expert` issues 6 `hipMemcpyAsync` (alternating 4 MB / 512 KB) per expert. The `.aeon` on-disk payload is already one contiguous 13.5 MB blob; make the VRAM pool per-slot contiguous (mirror disk layout) → single 13.5 MB SDMA packet per upload.
2. **Embedding fetch per token** — 4 unpinned H2D copies from an mmap'd 1 GB table every step (page faults + runtime bounce buffering). Upload the embed table to VRAM once (1.06 GB fits the budget) and gather on-device.
3. **M_PAD row replication** — 15 tiny D2D memcpys per layer; replace with one broadcast kernel.
4. **Physical file layout** — M11 note: experts file has 1,552 extents; 6 scattered experts read at 3.17 GiB/s vs 6.4 GB/s for a fresh sequential file. At repack time, `fallocate` the container and write it sequentially in one pass; order experts layer-major (or by measured access frequency). Raises the cold-tier ceiling toward the ≥6 GB/s target.
5. **Pinning fallback is silent** — if any 845 MB warm segment falls back to `posix_memalign`, every H2D from it goes through the runtime bounce buffer (slow, semi-synchronous). M14's +0.7 GiB swap suggests being at the edge. Track per-segment pinned status; only direct-upload from pinned segments.
6. **Per-step heap churn** — `std::vector<half> h_logits(129280)`, `h_rlogits`, etc. allocated every step/layer; hoist to persistent buffers.

---

## 4. Prioritized Roadmap

| Step | Change | Effort | Expected effect |
| :--- | :--- | :--- | :--- |
| 1 | Delete Hot→Warm demotion from request path (Defect A); direct warm-slot upload without staging memcpy (Defect B); enqueue shared expert before CPU staging | ½–1 day | Warm hit: ~42 MB → 14 MB; GPU never idles for memcpy. Cold miss loses demotion tax. Expect ~5.15 → ~7–9 tok/s and a visible warm/cold A/B gap |
| 2 | Contiguous per-slot VRAM layout → single-copy uploads; split DMA streams | ½ day | ~10–20% on transfer-bound steps; simplifies Step 1 |
| 3 | Measure token-to-token top-6 Jaccard overlap; if ≥50%, speculative cross-token prefetch with 48+ slot staging arena | 2–3 days | Storage leaves the critical path; throughput → compute floor (~140 ms/token ≈ 7 tok/s) |
| 4 | Rewrite W4A16 GEMM toward the memory-bound floor | 3–5 days | Routed compute ~108 → ~15 ms/token |
| 5 | Vectorize GEMVs; GPU argmax; single-sync router; embed table on VRAM | 1–2 days | Dense path ~2× |
| 6 | Repack `.aeon` contiguously (`fallocate`, sequential write, frequency-ordered) | ½ day + repack | Cold tier 3.2 → ~6 GB/s; matters for mispredicts and prefill |

**Post-Step 1–5 realistic single-GPU target: 25–40 ms/token (25–40 tok/s) at 43 layers**, versus 252 ms today — before Phase 3 multi-GPU.

---

## 5. What Is Already Solid

The io_uring batched submission with strict 4KB validation, the staging-slot state machine with per-slot events, the feasibility-gated memory budget, and the bit-exact test discipline are good engineering. The flaws are concentrated in exactly two places — *when* bytes move (scheduling) and *how many* bytes move per request (redundant copies) — plus one under-tuned kernel.

---

## 6. Execution Log

- **2026-09-08 — Step 1 implemented**: demotion removed from `ExpertRegistry::allocate_vram_slot` (evicted Hot experts return directly to Cold NVMe; no D2H copy); warm hits upload H2D directly from pinned warm segments (new `PrefetchStagingArena::begin_direct_transfer` borrows the slot event without staging a payload; `HostExpertPool::is_slot_pinned` gates the fast path); shared-expert kernels are enqueued *before* `dispatch_layer_prefetch` so CPU staging/I/O submission overlaps GPU compute; dead pending-warm-event machinery removed.
- **2026-09-08 — Step 1 silicon results (M16)**: all six regression tests passed with identical golden tokens. Controlled A/B: Warm disabled `4.36 tok/s` (unchanged), Warm 35 GiB `5.11 tok/s` (vs M15 `5.15`). **Throughput unchanged despite 3× less traffic per warm hit — conclusive evidence the loop is latency-bound by just-in-time dispatch, not bandwidth-bound.** The original 7–9 tok/s Step 1 estimate was therefore wrong; the estimate for Step 3 (speculative cross-token prefetch) is unchanged and now empirically prioritized. Warm hits dropped 684 → 157 without refill-on-eviction; acceptable, since warm coverage demonstrably does not bind throughput at this stage. See [Performance Ledger](PERFORMANCE_LEDGER.md) M16.
