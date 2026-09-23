# Supply-Chain Hot-Path Analysis — is the feed a river or a bucket?

*Status: open analysis. Written 2026-09-22; deepened 2026-09-23 in a second source pass. A two-round audit of the expert supply's transfer path and its per-request bookkeeping, in answer to two questions: (1) is the NVMe→VRAM feed a continuous stream or an interrupted one; (2) which substeps on that road are expensive enough to be worth removing. Source read: `direct_io_reader.hpp`, `tiered_expert_supply.hpp`, `prefetch_staging.hpp`, `expert_registry.hpp`, `v4_expert_supply.hpp`, `v4_prefill_sweep.hpp`, `v4_expert_executor.hpp`, `v4_model_host.hpp`, `memory_budget.hpp`, `bench_prefill_ab.cpp`. No code changed.*

**Subject.** The mechanics of moving a routed expert from NVMe into VRAM — the read submission, the staging corridor, the H2D, and the registry bookkeeping around all three. This is the *how fast can the bytes arrive* question, not the *which bytes should arrive* strategy question, which the [prefill supply review](PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md) and the [prefill supply strategy plan](PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md) own.

**Scope.** The paths as written, with the cost of each substep reasoned from the code. **Nothing below is measured.** Every estimate is flagged as such, and §5 names the counters that would confirm or refute it — and reports that one of the two splits it asks for needs no new instrumentation at all, because both counters already exist.

**What the second pass added.** The first pass found the feed's shape (reads a river, swept-layer H2D a bucket) and the two `O(catalog)` scans. The second pass read every call site and found that the bucket is worse than described (§3): `banks = 1` also **idles the disk** across the same fence, and the H2D is enqueued a full body *after* its bytes are ready. It also promoted several first-pass footnotes into named, mechanically-provable cost sites (§4.4) and added the structural options the first pass did not consider (§5.2).

---

## 1. The verdict, split into its two halves

The feed is not one thing, and the two halves have opposite answers:

| Half | Answer | Evidence |
| :--- | :--- | :--- |
| **NVMe reads** | **a river** | one `submit_pending_reads()` per batch, queue depth sized to hold a whole layer, `IOSQE_ASYNC` set |
| **H2D upload of a swept layer** | **a bucket** | enqueued at `before_layer`, then host-synchronized one slot at a time before the body runs |

So the drive is kept busy across layers (the double buffer works), but the **PCIe leg of a swept layer is fully exposed in front of that layer's compute**. That is the concrete interruption, and it is the finding that answers question 1.

The second pass sharpens the bucket in two ways that matter for what to fix:

- **The same fence also idles the disk.** `before_layer(L)` is ordered `materialize_layer()` → `dispatch_ahead(L+1)`. Because staging is one bank and `L+1`'s reads need the slots `L`'s H2D is still holding, `L+1`'s reads cannot be issued until `L`'s upload has fully drained. For that ~`0.14–0.17 s` the drive has **no reads outstanding**. So `banks = 1` costs both the exposed PCIe leg *and* ~`6–7 s/window` of disk idle that the "river" framing hides. The reads are a river *within* a layer; the river is dammed at every layer boundary.
- **The boundary is a serial three-leg critical path, not one leg.** Per layer, before the body can start, the host serially does: (1) wait the layer's reads, (2) enqueue and host-wait the layer's H2D, (3) release staging, (4) *then* issue the next layer's reads, (5) *then* the `O(catalog)` bookkeeping of §4. Every one of these is exposed. The two fixes that matter — a second bank and a batch sync — attack different legs of this path (§5).

---

## 2. What is already a river

`DirectIOReader` (`direct_io_reader.hpp`) is a correct `io_uring` implementation and the dispatch path uses it the right way:

- `submit_read` only writes an SQE and bumps the tail; nothing enters the kernel per request.
- `TieredExpertSupply::dispatch` queues **every** cold request of a batch and then calls `submit_pending_reads()` **once**, which loops `io_uring_enter` until the whole batch is submitted.
- The queue is sized to hold a whole layer: `V4ModelHost` sets `io_queue_depth = max(64, requests_per_expert × experts_per_layer)` when the sweep is on, and an expert is read in `4 MiB` chunks (`DEFAULT_CHUNK_BYTES`), so ~`1024` SQEs are in flight at once for a 256-expert layer.
- `IOSQE_ASYNC` is set (`force_async = true`), so reads do not block the submitter.

No batching inefficiency here. A layer's cold read is one submission, ~`1024` requests deep, and it lands in the staging corridor as fast as the drive can serve it.

---

## 3. The bucket: the swept layer's H2D is exposed

Trace of the sweep's steady state, from `V4PrefillSweep` and `V4Graph::forward_window`:

```
before_layer(L):
    materialize_layer()
        → materialize()          wait every io_uring completion of L
        → release_streamed_staging()
              for each of ≤256 slots: hipEventSynchronize(events[slot])   ← HOST BLOCKS
    dispatch_ahead(L+1)          issue L+1's reads
  body(L)                        ~1.4 s at N≈512
after_layer(L):  release_layer(L)
```

The **reads** for `L+1` are issued before `body(L)` runs, and that is the double buffer working as designed — the drive is busy through the body. But the **H2D of layer `L`** is enqueued inside `materialize()` (`upload_from_host_expert(..., sdma_cold_stream_)`, one call per expert) and then **waited for on the host, slot by slot**, by `release_streamed_staging` *before the body begins*. So:

- the drive runs during the body (good),
- the PCIe copy of a swept layer runs **in front of that layer's compute**, serially, with 256 sequential `hipEventSynchronize` calls on top of it.

**Why it cannot overlap:** the staging arena is sized to **exactly one layer** (256 slots) when `prefill_sweep` is on (`V4ModelHost`: `staging_slots = max(staging_slots, experts_per_layer)`). Layer `L+1`'s reads cannot start until layer `L`'s H2D has drained all 256 staging slots. With one bank there is no second bank to read into, so the read of `L+1` can only start after `L`'s *upload* completes — the reads and the uploads of consecutive layers are forced onto opposite sides of the same fence.

Estimated cost (from figures, **not measured**): `3.44 GiB` per layer at a PCIe rate of `~20–25 GB/s` is `~0.14–0.17 s`, plus 256 driver round-trips for the event synchronize, per layer, times 43 layers. On a window whose body is ~`60 s` (`N = 512`), that is on the order of `6–7 s` of exposed, non-overlapped transfer — consistent with the `≈1.20×–1.46×` gap between the measured swept per-layer time and the plan's pure-body estimate, though that gap also contains the registry scans of §4.

**This is not a mystery; it is the plan's open item.** The plan already lists *"the staging arena's `banks × depth` target — the smaller waved ring is unbuilt."* This analysis gives that item a mechanism: **with `banks = 1`, the read leg and the upload leg of consecutive layers cannot overlap.** Raised to `banks = 2` (the decode path's shape), `L+1`'s reads could fill bank B while `L`'s H2D drains bank A, and the upload would overlap the read instead of the body.

**Second, smaller exposure:** `begin()` drains Hot and immediately `dispatch_ahead(0)`, so layer `0` has no lookahead at its own `before_layer(0)` and is always fully exposed. Once per window; negligible at large `N`, material at small `N`.

**Third — the H2D is issued too late to hide behind compute.** The upload of `L` is enqueued inside `materialize()` at `before_layer(L)` — that is, *after* `body(L−1)` has finished. But `L`'s reads completed *during* `body(L−1)`. So the bytes sit in staging for a full body (~`1.4 s`) before the upload is even submitted. Even a perfect host-wait removal does not fix this: the copy is submitted on the wrong side of the body. Moving it into `body(L−1)`'s shadow is the real prize, and it is why a *completion-driven* or *rolling* corridor (§5.2) may beat a plain second bank.

**The `banks = 2` fix has two costs the first pass did not state.** Before anyone builds it:

1. **Pinned host memory doubles.** Staging is sized to `experts_per_layer = 256` when the sweep is on (`V4ModelHost`), i.e. ~`3.44 GiB` pinned. Two banks ≈ **`6.9 GiB` pinned**, and pinned memory is exactly what cannot be reclaimed — the subject of the open [host-memory pressure investigation](HOST_MEMORY_PRESSURE_INVESTIGATION.md). This is a coupled decision, not a free one.
2. **The submission ring must be re-checked.** `io_queue_depth = 1024` is sized for one layer. If a second bank ever puts *two layers' reads* in flight, `submit_read`'s SQ-full guard fires and the default CQ (`2 × SQ`) is exactly full. Note the current `dispatch_ahead` only ever has one layer's reads outstanding (single `pending_state_`), so "banks = 2" as literally described — overlap `L`'s H2D with `L+1`'s reads — needs a second **staging** bank but not necessarily a deeper ring. Be explicit about which resource is doubled, or the change silently overruns the ring.

---

## 4. Expensive substeps on the road

Two `O(catalog)` scans run **per request**, unconditionally — including when telemetry is off — on the dispatching thread. `catalog` is `num_layers × experts_per_layer = 43 × 256 = 11,008` entries.

### 4.1 `observe_supply_occupancy` → `pending_transfer_count()` (the largest)

`TieredExpertSupply::dispatch` calls `observe_supply_occupancy(request.source_tier)` for **every** request. That function passes `expert_registry_->pending_transfer_count()` as an argument — and `pending_transfer_count()` walks the **entire `11,008`-entry catalog** and inserts each live `operation_id` into a `std::unordered_set<uint64_t>` (a heap allocation every call):

```cpp
uint64_t pending_transfer_count() const {
    std::unordered_set<uint64_t> operations;
    for (const auto& entry : catalog) { ... operations.insert(...); }
    return operations.size();
}
```

Because the walk happens while **evaluating the argument**, `observe_occupancy`'s early `if (!enabled_) return;` does **not** avoid it. The same is true of `published_hot_slots()` (O(1), `hot_vram_lru.size()`) and `published_warm_slots()` (O(1)) — those are fine — but `pending_transfer_count()` is not.

Per layer: `256 × 11,008 ≈ 2.8 M` catalog visits **and** 256 `unordered_set` allocations. Per 43-layer window: `~121 M` visits and `~11 k` allocations, purely to compute seven occupancy numbers that are dropped when telemetry is off.

### 4.2 `schedule_demotion`'s catalog fallback

When a request needs a transfer and `request.demotion` is empty, `schedule_demotion` scans the whole catalog looking for a demotion-pending entry of the same operation:

```cpp
for (const auto& entry : expert_registry_->catalog) {
    if (entry.operation_id == request.operation_id &&
        entry.operation == ExpertOperation::DEMOTION_PENDING) { ... break; }
}
```

In a **swept prefill there is never a demotion** (allocation is free-list only, release is by layer), so this scan runs for all 256 requests, walks all 11,008 entries each, and **finds nothing every time**: another `~2.8 M` visits per layer, `~121 M` per window.

### 4.3 Rough total, and the exact `O(1)` fix

`~240 M` catalog visits per window on the dispatch thread, plus allocation churn. Order-of-magnitude estimate (**not measured**): `~5–15 ms/layer`, i.e. the same order as the exposed H2D of §3.

Both scans are removable for very little, and the second pass pins the recipe rather than gesturing at it.

**`pending_transfer_count()` (§4.1), exact.** The quantity is the number of *distinct* live operation ids. A demotion victim carries the **same** `operation_id` as its incoming transfer (`reserve_vram_destination` stamps the victim with the incoming `operation_id`), so:

$$\text{distinct\_live\_ops} = \text{entries\_with\_a\_live\_operation} - \text{pending\_demotion\_count}$$

Maintain one `uint32_t operations_in_flight_` counter, `++` where an entry transitions `operation NONE → {IO_PENDING, PROMOTION_PENDING, DEMOTION_PENDING}` and `--` where it returns to `NONE` (in `complete_request`, `fail_request`, `complete_demotion`, `drop_demotion`, `fail_demotion`, and the two `warm_shadow` branches). Then `pending_transfer_count()` returns `operations_in_flight_ - pending_demotion_count` in `O(1)` — no hash set, no allocation. ~8 edit sites, all inside `expert_registry.hpp`.

**`schedule_demotion`'s fallback (§4.2), exact.** The scan can only find a `DEMOTION_PENDING` entry sharing the request's `operation_id` when `reserve_vram_destination` *marked a victim and then dropped it* (host capacity 0, queue pressure, or Warm destination unavailable). Every such drop increments `pending_demotion_count` at reservation and decrements it in `drop_demotion`/`fail_demotion`. So a guard `if (expert_registry_->pending_demotion_count == 0) return;` before the loop is correct: when it is zero there is by construction no such entry to find. In the sweep this eliminates the scan outright; in decode it does so whenever no demotion is in flight.

### 4.4 Mechanically-provable cost sites (promoted from the first pass's footnotes)

These four are not estimates of magnitude — they are properties of the code that can be read off it, and each has a small, well-defined fix. They are listed smallest-effort first.

**(a) 256 host syncs where one would do — `release_streamed_staging`.** In the sweep **all** uploads are enqueued on the single `sdma_cold_stream_` (`materialize`), and HIP streams execute in order, so completion of the **last** recorded event implies completion of every preceding one. The 255 earlier `hipEventSynchronize` calls in `release_streamed_staging` are therefore provably redundant driver round-trips. Record one batched event on `sdma_cold_stream_` after the upload loop, sync it once, and release every `GPU_TRANSFER_PENDING` slot. (The mixed-stream executor path groups remaining events by stream and syncs once per stream.) This is fixable **without** touching the arena, and it is the cheapest of all the items here.

**(b) Two HIP events per expert per layer, at the same point on the same stream.** In `materialize`, per expert, `hipEventRecord(prefetch_staging_->events[staging_idx], sdma_cold_stream_)` is immediately followed by `record_h2d_event(...)`, which does `hipEventCreateWithFlags` + `hipEventRecord(transfer->h2d_event, stream)`. The arena's per-slot event and the registry's `h2d_event` are the **same signal at the same location**: 256 event creates + 256 destroys + 256 queries per layer (~`22 k` each per window), and the create/destroy are synchronous driver calls. In the sweep they can be one: because `finish_streamed_batch` already host-synchronizes, the registry transfer can be completed inline at that sync point and never needs an event of its own. A "streamed completion" entry point on the supply that syncs the batch once (item a) and completes each registry request inline bypasses `hipEventCreate`/`Query`/`Destroy` entirely on the sweep path, while the generic async executor path keeps its per-transfer event.

**(c) `reap_registry_transfers` is `O(n²)` and destroys events one at a time.** It erases from `std::vector<PendingTransfer>` at `index` and then `continue`s **without** incrementing `index`, so a steady sweep repeatedly erases from the front. `PendingTransfer` is ~200 B and 256 entries are live, so that is ~`32 k` element moves (~`6.5 MB` memmove) per layer, ~`280 MB` memmove per window, on top of 256 `hipEventDestroy`. The registry is order-independent — `ensure`/`find` are by `operation_id` — so swap-and-pop makes each erase `O(1)`.

**(d) `find_registry_transfer` linear scans.** Per sweep request there are ~5–7 `O(n)` scans: `ensure_registry_transfer` from `schedule_demotion` and again from `dispatch`, then `bind_staging`, `record_h2d_event`, `wait_for_demotion_dependency`, and the post-submit fix-up loop. With 256 in flight that is ~`256 × 6 × 128 ≈ 200 k` comparisons/layer. Small beside the catalog walks, but the same class of fix and the same map removes it: an `unordered_map<uint64_t, size_t>` index maintained beside the vector.

### 4.5 The first pass's table, sharpened

| Site | Cost | Note |
| :--- | :--- | :--- |
| `ExpertRegistry::validate_invariants` on every `release_layer` | allocates `hot_seen(vram_capacity)`, `warm_seen(host_capacity)`, `shadow_seen(total_experts = 11,008)` and two more `seen(catalog.size())` vectors in `validate_lru`; several full-catalog passes | 43×/window. In the sweep the *quadratic* orphaned-reservation rescans are skipped (no live reservations at `release_layer`), so this is `O(catalog)` with ~5 small heap allocations — real but second-order; touch only if a profile points at it. |
| `LayerPrefetchState::sync_state` | re-materializes the batch into 9 parallel vectors, twice per layer (`materialize_layer_prefetch` + `finish_streamed_batch`) | After the first call the capacity is warm, so ~`4.6 k` stores/layer — µs scale, **over-stated** in the first pass. The real smell is that `LayerPrefetchState` duplicates the whole `PayloadBatch` field-by-field into 9 parallel arrays; that is redundancy to remove for clarity, not cost to remove for speed. |
| `direct_io_completions_` rendezvous | shared `unordered_map<uint64_t, DirectIOCompletion>`, one `find` per chunk (~1024/layer) | Request ids are contiguous per layer (`next_direct_io_id_` advances by `request_count`), so a flat ring indexed by `id − layer_base` removes the hashing and the map churn, and cleans up the "leftover completions on throw" edge. Minor. |
| `dispatch_layer_prefetch_batch` dedup | `unordered_map` rebuilt per layer | inherent to dedup (D2); noted for completeness. |
| `dispatch` request ordering | `stable_sort` over requests | `O(256 log 256)` per dispatch. |
| `release_streamed_staging` | `hipEventSynchronize` × 256 | the host-blocking half of §3; item (a) above removes the redundancy. |

### 4.6 The per-request audit is *not* the problem

`checked_validate()` is a no-op unless `validate_each_request_` is set, and it is off by default. So the `O(catalog)` cost of `validate_invariants` does **not** land on the per-request path in production — only on the boundaries of §4.5. Worth stating explicitly so it is not mistaken for the hot-path cost.

---

## 5. What would confirm or refute this

Everything above is reasoned, not measured. The counters already exist for most of it; the fix is mostly to *attribute* rather than to instrument anew.

### 5.1 Measurement — one of the two splits needs no new instrumentation

1. **Separate the read wait from the H2D issuance in the sweep (the one genuine gap).** `V4PrefillSweep::load_ns_` currently lumps `materialize`'s io wait and its H2D enqueue+host-wait together, so the exposed-H2D claim (§3) cannot be read off it. Splitting `materialize_layer` into `io_wait_ns_` and `h2d_wait_ns_` is the only new instrumentation the document requires, and it settles §3 directly. It also gives items 4.4(a) and 4.4(b) their proof: both live entirely inside the `h2d_wait` half.
2. **The `O(catalog)` cost of §4 is already isolable — no new counter needed.** `V4PrefillSweep::io_ns_` wraps the whole `dispatch_layer` (reap + missing-scan + `dispatch_layer_stream`), while `TieredExpertSupply::direct_io_submit_ns_` wraps only `io_uring_enter`. Both are already exposed on `V4ModelHost`. So

   $$(\text{sweep\_io\_ns} - \text{direct\_io\_submit\_ns}) = \text{non-I/O dispatch cost}$$

   which is exactly where §4.1–4.3 live. `bench_prefill_ab.cpp` already prints `io_s`; it needs only to also print `direct_io_submit_ns` (exposed, not emitted) and the difference. One line, and if the residual is small §4 is overestimated and should be dropped.
3. **A/B the two scans.** With the `O(1)` recipe of §4.3 in place, compare swept per-layer time. `n = 1` will not resolve it; the deployable test is whether it moves the non-I/O dispatch residual of item 2 at all.
4. **Prove the H2D overlap is real** by constructing the arena so the next layer's reads can begin while this layer's uploads drain, and re-running a window while checking `load_ns`, `io_ns`, and frontier depth.

### 5.2 Structural options, cheapest-first

The findings that recover *throughput* (as opposed to removing waste) are these, and they are not mutually exclusive:

- **(C2) A rolling staging corridor** — the option the first pass missed. The only reason `L+1`'s reads wait on `L`'s *whole-layer* H2D is that release is whole-layer, but reads **and** H2D are both per-expert (13.5 MiB, 4 chunks). Instead of `wait-all-reads → enqueue-all-H2D → wait-all-H2D → release-all`, do it per expert: as expert *i*'s chunks land, issue its H2D; as its upload event fires, free slot *i*; a concurrent path refills slot *i* with `L+1`'s data. This recovers the ~`6–7 s/window` of disk idle (§1) with only a **handful** of surplus slots rather than a second full bank, and it does not double pinned memory. It does not by itself remove the H2D-before-body exposure (§3 "Third") — that needs the upload moved into the previous body's shadow — but it is a strictly cheaper way to overlap the *reads* with the *uploads* of adjacent layers, and it is compatible with the [host-memory pressure](HOST_MEMORY_PRESSURE_INVESTIGATION.md) constraint that `banks = 2` fights.
- **(C1) `banks = 2`** — removes the PCIe exposure directly, at the two costs §3 names (pinned memory doubled; ring depth to re-check). Build it *after* measuring the `io_wait_ns_ / h2d_wait_ns_` split, and only if the corridor does not already capture most of the overlap.
- **(C5) Collapse 256 H2D copies into one** — all experts in a layer share `payload_bytes`, and if the free-list allocation yields contiguous VRAM runs the 256 `hipMemcpyAsync` calls become one 3.44 GiB copy. Per-call driver overhead is ~few µs × 256 ≈ `1–3 ms/layer`. Only worth it if contiguity can be arranged; otherwise nice-to-have.
- **(C4) Is the H2D avoidable at all on gfx1100?** A bounded spike, flagged so it is not missed: on RDNA3 with Resizable BAR, VRAM is CPU-addressable through the BAR window. *If* the platform routes NVMe→BAR-VRAM DMA as PCIe peer-to-peer (by no means guaranteed on consumer boards — it often hairpins through the root complex into DRAM and back), an `O_DIRECT` read could target a VRAM address directly and delete **both** the pinned staging and the H2D copy, crossing PCIe once under the storage path. Transformative if real; half a day to falsify. Flag, do not plan on.

The recommended experiment order is: measure (5.1 items 1–2) → cheap provable removals (4.4 a–d) → rolling corridor vs `banks = 2` (5.2) → the BAR spike (5.2 C4) if it is worth a half-day.

---

## 6. What this analysis does not claim

- **It does not claim the drive is underused within a layer.** The reads are a properly pipelined submission; the double buffer keeps the drive working through the body. What the second pass *does* claim is narrower and correct: the drive is **idle across layer boundaries** (§1, §3), because the one-bank staging fence sits between `L`'s upload and `L+1`'s reads. The interruption is on the **PCIe** leg and, as a consequence, at the disk boundary — not within a layer.
- **It does not claim the `O(catalog)` scans are the dominant cost.** They are *unconditional waste on the hot path* — cheap to prove and cheap to remove — but they are bookkeeping, not supply speed. §3 is the finding with throughput consequence.
- **It does not claim any number here is measured.** All figures are reasoned from the code and the plan's own constants and are marked so. §5.1 exists precisely to turn them into measurements before any of them is acted on — and notes that one such measurement needs no new counter.
- **It does not claim the four §4.4 sites are costly, only that they are provably wasteful.** Their effect should be read off the same `h2d_wait_ns_` split that §5.1.1 introduces, not assumed.
- **It does not re-open strategy.** River-versus-bucket is about *how fast bytes arrive*, not *which bytes*. The route-aware-versus-blind question is the [prefill supply review](PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md)'s.

---

## 7. Disposition

Ordered by confidence-to-effort, not by size. Every "fix" here is unstarted and gated on the measurement above it.

| # | Item | Mechanism / location | Est. cost | Fix | Effort | Risk |
| :-- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Batch sync** (§4.4a) | `release_streamed_staging` | 55–215 ms/window | one event on `sdma_cold_stream_`, sync once; release all pending slots | S | low |
| 2 | **No-demotion guard** (§4.2) | `schedule_demotion` fallback | ~5–15 ms/layer | `if (pending_demotion_count == 0) return;` | XS | low |
| 3 | **`O(1)` pending count** (§4.3) | `pending_transfer_count()` | ~5–15 ms/layer + alloc | `operations_in_flight_` counter (§4.3 recipe) | S | low |
| 4 | **`O(1)` reap** (§4.4c) | `reap_registry_transfers` | ~280 MB memmove/window | swap-and-pop | XS | low |
| 5 | **Drop duplicate event** (§4.4b) | `record_h2d_event` | 22 k create/destroy/window | complete inline at the batch sync | M | med |
| 6 | **Measure first** (§5.1) | `bench_prefill_ab.cpp` + `materialize_layer` | — | print `direct_io_submit_ns`; split `load_ns` into `io_wait`/`h2d_wait` | S | none |
| 7 | **Rolling corridor** (§5.2 C2) | `materialize` + sweep | recovers ~6–7 s/window disk idle | per-expert read → H2D → release | L | med |
| 8 | **`banks = 2`** (§5.2 C1) | `V4ModelHost` staging sizing | ~0.15 s/layer PCIe | second staging bank | M | pinned-mem coupling, SQ/CQ |
| 9 | **id→index map** (§4.4d) | `find`/`ensure_registry_transfer` | ~200 k cmp/layer | `unordered_map` index | S | low |
| 10 | **Single H2D copy** (§5.2 C5) | `materialize` | ~1–3 ms/layer | contiguous copy | M | low value |
| 11 | **Large-BAR NVMe→VRAM** (§5.2 C4) | spike only | potentially transformative | measure, do not plan | ? | platform-dependent |

| Cross-cutting | Where it goes |
| :--- | :--- |
| Exposed swept-layer H2D (§3) | Directly informs the plan's open **"staging arena's `banks × depth` target"**: `banks = 1` is *why* read and upload cannot overlap across layers — and why the disk also idles there. |
| Pinned-memory coupling of `banks = 2` (§3) | The `banks × depth` target must be decided together with the open [host-memory pressure investigation](HOST_MEMORY_PRESSURE_INVESTIGATION.md); a second bank is ~`3.44 GiB` more non-reclaimable pinned memory. |
| Kernel-side cost | Out of scope here; the compute-side audit is a separate document. |
