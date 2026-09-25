# Supply-Chain Hot-Path Analysis — is the feed a river or a bucket?

*Status: open analysis — **largely measured and settled**. Written 2026-09-22; deepened 2026-09-23 in a second source pass; **measured 2026-09-24** (Step 1 §8, Step 2 §9, Phase 1 §10). A two-round audit of the expert supply's transfer path and its per-request bookkeeping, in answer to two questions: (1) is the NVMe→VRAM feed a continuous stream or an interrupted one; (2) which substeps on that road are expensive enough to be worth removing. Source read: `direct_io_reader.hpp`, `tiered_expert_supply.hpp`, `prefetch_staging.hpp`, `expert_registry.hpp`, `v4_expert_supply.hpp`, `v4_prefill_sweep.hpp`, `v4_expert_executor.hpp`, `v4_model_host.hpp`, `memory_budget.hpp`, `bench_prefill_ab.cpp`. Instrumentation added: `TieredExpertSupply` transfer counters + `tests/bench_supply_split.cpp` + `scripts/supply_split.sh`; probe added: `tools/aeon_c4_probe.cpp`. **Headline: the transfer path is near its ceiling; the exposed swept-layer H2D `4.1–5.4 s/window` was then removed in Phase 1 (§10) by a deferred drain — `3.2–3.5 s/window` recovered, at both Warm 0 and Warm 30 — but only with a second staging bank that costs `+3.44 GiB` pinned, which does not fit the production Warm shape; the shipping form is C2's per-expert recycling (Phase 2). The largest measured supply cost — decode's NVMe wait (§8.5) — has a remedy that is out of scope here.***

**Subject.** The mechanics of moving a routed expert from NVMe into VRAM — the read submission, the staging corridor, the H2D, and the registry bookkeeping around all three. This is the *how fast can the bytes arrive* question, not the *which bytes should arrive* strategy question, which the [prefill supply review](PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md) and the [prefill supply strategy plan](PREFILL_SUPPLY_STRATEGY_EXECUTION_PLAN.md) own.

**Scope.** The paths as written, with the cost of each substep reasoned from the code. **Nothing below is measured.** Every estimate is flagged as such, and §5 names the counters that would confirm or refute it — and reports that one of the two splits it asks for needs no new instrumentation at all, because both counters already exist.

**What the second pass added.** The first pass found the feed's shape (reads a river, swept-layer H2D a bucket) and the two `O(catalog)` scans. The second pass read every call site and found that the bucket is worse than described (§3): `banks = 1` also *appeared to* **idle the disk** across the same fence (**refuted in §8** — only the upload is exposed), and the H2D is enqueued a full body *after* its bytes are ready. It also promoted several first-pass footnotes into named, mechanically-provable cost sites (§4.4) and added the structural options the first pass did not consider (§5.2).

**What the measurement (§8) changed.** Step 1 was built and run, and it **decides** the C1/C2/C3 question — partly against this document. Read §1 and §3's corrections with §8, and see §8's "what the measurement refuted" list: the disk-idle half of the `banks = 1` claim is **refuted** (the sweep's `io_wait` is ≈0.3 s; the disk is fully hidden), the per-slot-sync overhead (§4.4a) is **≈0** (the 5.4 s is PCIe copy bandwidth, not driver round-trips), and the `O(catalog)` scans (§4) are **already negligible** (≈25 ms/window). The one claim that survived, and that the data sharpens, is §3's: the swept-layer **H2D copy is exposed** — `4.1–5.4 s/window`, at the PCIe Gen4 x16 ceiling. The measurement also **bounds** what is left: the largest measured supply cost is **decode's** NVMe wait (36–62% of every token), which the prefill-focused first pass did not look at — but its remedy is *which bytes are resident*, out of scope here, so §8.5 records it and hands it off.

---

## 1. The verdict, split into its two halves

The feed is not one thing, and the two halves have opposite answers:

| Half | Answer | Evidence |
| :--- | :--- | :--- |
| **NVMe reads** | **a river** | one `submit_pending_reads()` per batch, queue depth sized to hold a whole layer, `IOSQE_ASYNC` set |
| **H2D upload of a swept layer** | **a bucket** | enqueued at `before_layer`, then host-synchronized one slot at a time before the body runs |

So the drive is kept busy across layers (the double buffer works), but the **PCIe leg of a swept layer is fully exposed in front of that layer's compute**. That is the concrete interruption, and it is the finding that answers question 1.

The second pass sharpens the bucket in two ways that matter for what to fix:

- **The same fence does *not* idle the disk — the H2D copy is what it exposes.** This was the second pass's sharpest *prediction* and the measurement (§8) **refuted the disk half of it**: the sweep's `io_wait` is ≈`0.3 s` per window, so the reads are essentially fully hidden by the double buffer and the drive is never starved. What the one-bank fence does expose is the **H2D copy itself**: `4.1–5.4 s` per swept window, at the PCIe Gen4 x16 ceiling (`~26 GiB/s`), serialized in front of the body because the next layer's reads cannot start until these slots drain. So the lever is the **upload**, not the reads — and a second bank (C1) does not by itself fix it (§5.2). **Phase 1 (§10) then showed it *is* the fix once coupled with a deferred drain**: the second bank is the headroom that lets the copies stay in flight through the body, and it recovered `3.2–3.5 s/window` — but it costs a whole extra pinned bank, which does not fit the host-memory budget, so C2 is the shipping form.
- **The boundary is a serial three-leg critical path, not one leg.** Per layer, before the body can start, the host serially does: (1) wait the layer's reads, (2) enqueue and host-wait the layer's H2D, (3) release staging, (4) *then* issue the next layer's reads, (5) *then* the `O(catalog)` bookkeeping of §4. With the reads hidden (`io_wait`≈0) and the bookkeeping negligible (§4), the exposed leg is **(2)** — the H2D drain. That is the measured target.

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

Estimated cost (from figures, **not measured**): `3.44 GiB` per layer at a PCIe rate of `~20–25 GB/s` is `~0.14–0.17 s`, plus 256 driver round-trips for the event synchronize, per layer, times 43 layers. On a window whose body is ~`60 s` (`N = 512`), that is on the order of `6–7 s` of exposed, non-overlapped transfer — consistent with the `≈1.20×–1.46×` gap between the measured swept per-layer time and the plan's pure-body estimate, though that gap also contains the registry scans of §4. **Measured (§8): the total is `4.1–5.4 s`, at the Gen4 x16 PCIe ceiling — the driver round-trips are ≈`0`, so the whole figure is the copy itself. The estimate was close on the total and wrong on its composition.**

**This is not a mystery; it is the plan's open item.** The plan already lists *"the staging arena's `banks × depth` target — the smaller waved ring is unbuilt."* This analysis gives that item a mechanism: **with `banks = 1`, the read leg and the upload leg of consecutive layers cannot overlap.** Raised to `banks = 2` (the decode path's shape), `L+1`'s reads could fill bank B while `L`'s H2D drains bank A, and the upload would overlap the read instead of the body.

**Measured (§8), with one correction.** The exposed upload is real and is `4.1–5.4 s` per swept window — but the mechanism is narrower than "reads and uploads cannot overlap": the window's `io_wait` is ≈`0.3 s`, so the reads *are* fully hidden and the drive is never starved. The one bank exposes the **H2D copy** (the next layer's reads need these slots, so they wait), not the disk. `banks = 2` *alone* does not fix that either — it lets the *next reads* start sooner, not the *upload* move off the critical path. **Phase 1 (§10) resolved this:** the upload does move off the critical path, but only when the second bank is combined with a **deferred drain** — the sweep leaves layer `L`'s copies in flight, orders nothing on the compute stream itself, and lets the body's MoE dispatch join each pending transfer and wait on its per-expert event before the MoE reads the weights. Ordering the *whole body* behind the copies instead recovers nothing (the copy simply moves from in front of the compute to in front of attention), which is the subtlest part of the result. The remaining work is to get that overlap without a whole extra bank — C2's per-expert recycling.

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

**(a) 256 host syncs where one would do — `release_streamed_staging`.** In the sweep **all** uploads are enqueued on the single `sdma_cold_stream_` (`materialize`), and HIP streams execute in order, so completion of the **last** recorded event implies completion of every preceding one. The 255 earlier `hipEventSynchronize` calls in `release_streamed_staging` are therefore provably redundant driver round-trips. Record one batched event on `sdma_cold_stream_` after the upload loop, sync it once, and release every `GPU_TRANSFER_PENDING` slot. (The mixed-stream executor path groups remaining events by stream and syncs once per stream.) This is fixable **without** touching the arena.

  **But the measurement (§8) refutes the payoff.** The window holds `10 723` syncs (`~249`/layer — the count is as predicted), yet the *time* in `release_streamed_staging` is `4.1–5.4 s` and the copied bytes are ≈`145 GiB`, i.e. `≈26 GiB/s` — exactly the PCIe Gen4 x16 ceiling. So the sync calls are **not** the cost: they are absorbing copy latency that would be paid anyway. Collapsing 256 syncs into one saves the *driver round-trip* share, which measures ≈`0`. **Deprioritise this item**: it is correct but worthless without also overlapping the copies (§5.2 C2/C3). The first pass's `55–215 ms/window` estimate for the sync overhead is refuted.

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

> **Done 2026-09-24 — see §8.** The three supply-level counters (`io_wait`, `h2d_enqueue`, `h2d_drain` + `h2d_drain_calls`) and a reset were added to `TieredExpertSupply`, exposed through `V4ModelHost`, and driven by a new `bench_supply_split` (one model load, gate-selected strategy per length, greedy decode). §8 carries the results; the items below are kept as the specification that was implemented.

1. **Separate the read wait from the H2D issuance in the sweep (the one genuine gap).** `V4PrefillSweep::load_ns_` currently lumps `materialize`'s io wait and its H2D enqueue+host-wait together, so the exposed-H2D claim (§3) cannot be read off it. This is what the three new counters split — done at the supply rather than the sweep, so decode reports the same split.
2. **The `O(catalog)` cost of §4 is already isolable — no new counter needed.** `V4PrefillSweep::io_ns_` wraps the whole `dispatch_layer` (reap + missing-scan + `dispatch_layer_stream`), while `TieredExpertSupply::direct_io_submit_ns_` wraps only `io_uring_enter`. Both are already exposed on `V4ModelHost`. So

   $$(\text{sweep\_io\_ns} - \text{direct\_io\_submit\_ns}) = \text{non-I/O dispatch cost}$$

   which is exactly where §4.1–4.3 live. `bench_prefill_ab.cpp` already prints `io_s`; it needs only to also print `direct_io_submit_ns` (exposed, not emitted) and the difference. One line, and if the residual is small §4 is overestimated and should be dropped.
3. **A/B the two scans.** With the `O(1)` recipe of §4.3 in place, compare swept per-layer time. `n = 1` will not resolve it; the deployable test is whether it moves the non-I/O dispatch residual of item 2 at all.
4. **Prove the H2D overlap is real** by constructing the arena so the next layer's reads can begin while this layer's uploads drain, and re-running a window while checking `load_ns`, `io_ns`, and frontier depth. **Done in part (§8):** the split shows the reads are already hidden, so the overlap that matters is the *upload's*, not the reads'. The remaining proof is a C2/C3 prototype.

### 5.2 Structural options, cheapest-first

The findings that recover *throughput* (as opposed to removing waste) are these, and they are not mutually exclusive:

- **(C2) A rolling staging corridor** — the option the first pass missed. Reads **and** H2D are both per-expert (13.5 MiB, 4 chunks), so instead of `wait-all-reads → enqueue-all-H2D → wait-all-H2D → release-all`, do it per expert: as expert *i*'s chunks land, issue its H2D; as its upload event fires, free slot *i*; a concurrent path refills slot *i* with `L+1`'s data. This uses only a **handful** of surplus slots rather than a second full bank, and it does not double pinned memory. **§8 corrects the reason it matters:** not disk overlap (the disk is already hidden) but **moving each expert's upload into the previous body's shadow**, which is what actually removes the `4.1–5.4 s` exposed H2D. It is compatible with the [host-memory pressure](HOST_MEMORY_PRESSURE_INVESTIGATION.md) constraint that `banks = 2` fights.
- **(C1) `banks = 2`** — **refuted as a standalone fix (§8), confirmed as half of the coupled fix (§10).** Alone it only lets the next layer's reads begin sooner, and the reads are already `98%` hidden (`io_wait ≤ 2.1 s`), so there is nothing for it to recover. But the second bank is exactly the headroom a **deferred drain** needs: with it, the sweep leaves layer `L`'s copies in flight through `body(L)` instead of host-blocking on them first, and the body's own MoE dispatch supplies the ordering. Measured (§10): `h2d_drain 4.25 → 0.006 s`, `wall −3.2/−3.5 s` at Warm 0 **and** Warm 30, with `io_wait` and decode unmoved. Its cost is real and is why the default stays `1`: `+3.44 GiB` of non-reclaimable pinned memory, which at the production Warm shape (~35 GiB, also pinned) pushes the 62 GiB reference box into swap even though the budget check passes. **Phase 2's per-expert slot recycling is what makes the win shippable** without the whole extra bank.
- **(C5) Collapse 256 H2D copies into one** — all experts in a layer share `payload_bytes`, and if the free-list allocation yields contiguous VRAM runs the 256 `hipMemcpyAsync` calls become one 3.44 GiB copy. Per-call driver overhead is ~few µs × 256 ≈ `1–3 ms/layer`. Only worth it if contiguity can be arranged; otherwise nice-to-have.
- **(C4) Is the H2D avoidable at all on gfx1100?** **Closed — §9 measured it: `NOT_SUPPORTED`.** The preconditions were green (32 GiB BAR, `CONFIG_PCI_P2PDMA=y`, `pcie_p2p=Y`), VRAM exports as a dma-buf and maps correctly, and the mapping verifiably aliases VRAM — but the kernel refuses it as a direct-I/O destination: `pread(O_DIRECT)` → `EFAULT`, and the production `io_uring` read → `cqe.res = -14`. `get_user_pages` cannot pin a BAR/dma-buf VMA, so no bus address reaches the NVMe controller. The accepted buffered path bounces through the page cache at `13 GiB/s` CPU writes and is *slower* than the current two-hop DMA path. **No userspace workaround; the pinned staging and the second hop stay.**

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
| 1 | **Batch sync** (§4.4a) | `release_streamed_staging` | **refuted — ≈0** (§8) | one event on `sdma_cold_stream_`, sync once; release all pending slots | S | low — *not worth doing alone* |
| 2 | **No-demotion guard** (§4.2) | `schedule_demotion` fallback | ~5–15 ms/layer | `if (pending_demotion_count == 0) return;` | XS | low |
| 3 | **`O(1)` pending count** (§4.3) | `pending_transfer_count()` | ~5–15 ms/layer + alloc | `operations_in_flight_` counter (§4.3 recipe) | S | low |
| 4 | **`O(1)` reap** (§4.4c) | `reap_registry_transfers` | ~280 MB memmove/window | swap-and-pop | XS | low |
| 5 | **Drop duplicate event** (§4.4b) | `record_h2d_event` | 22 k create/destroy/window | complete inline at the batch sync | M | med |
| 6 | **Measure first** (§5.1) | `bench_supply_split` + supply counters | — | **done (§8)** | S | none |
| 7 | **Rolling corridor** (§5.2 C2) | `materialize` + sweep | **the shipping form of §10's win** — same overlap from a bounded surplus instead of a whole extra bank | per-expert read → H2D during the previous body → release | L | med |
| 8 | **`banks = 2`** (§5.2 C1) | `V4ModelHost` staging sizing | **coupled with the deferred drain it recovers the whole exposed H2D (§10); standalone it does not** (§8). **§12 measured the mechanism directly**: at `1E` the arena is one room (`0/43` layer-bodies overlapped, drain `4.9 s`); at `2E` it is two (`42/43`, drain `0.006 s`). | **now the unconditional default**; the config knob is removed (R6), depth moves only via `resize_staging_slots` | M | pinned-mem (`+3.44 GiB`) — P2.2/P2.3 must make it cheaper |
| 9 | **id→index map** (§4.4d) | `find`/`ensure_registry_transfer` | ~200 k cmp/layer, but total dispatch CPU ≈0.15% (§8) | `unordered_map` index | S | low value |
| 10 | **Single H2D copy** (§5.2 C5) | `materialize` | ~1–3 ms/layer | contiguous copy | M | low value |
| 11 | **Large-BAR NVMe→VRAM** (§5.2 C4) | spike — **done (§9)** | **`NOT_SUPPORTED`, measured** — kernel refuses VRAM as an O_DIRECT target | closed, no workaround | — | settled |
| 12 | **Decode disk wait** (§8.5) | residency, not transfer | measured `36–63%`/token | **out of scope** — which bytes are resident; hand-off to the routing/placement study | — | — |

| Cross-cutting | Where it goes |
| :--- | :--- |
| Exposed swept-layer H2D (§3) | Directly informs the plan's open **"staging arena's `banks × depth` target"**: `banks = 1` is *why* the upload cannot overlap across layers. **Measured at `4.1–5.4 s/window` (§8)** — the disk is *not* also idle (that half of the claim is refuted). |
| Pinned-memory coupling of `banks = 2` (§3) | The `banks × depth` target must be decided together with the open [host-memory pressure investigation](HOST_MEMORY_PRESSURE_INVESTIGATION.md); a second bank is ~`3.44 GiB` more non-reclaimable pinned memory. |
| Kernel-side cost | Out of scope here; the compute-side audit is a separate document. |

---

## 8. Measured — the exposed-load split (Step 1, 2026-09-24)

*Status: **measured.** This section is the warrant §5.1 asked for. It was built as described there and run on the reference machine; the numbers below are real, not reasoned.*

### 8.1 What was built

Two changes, both kept:

- **Supply-level transfer counters** in `TieredExpertSupply` (`io_wait_ns`, `h2d_enqueue_ns`, `h2d_drain_ns`, `h2d_drain_calls`, plus the pre-existing `direct_io_submit_ns`) with a `reset_transfer_counters()`, exposed on `V4ExpertSupplyCoordinator` and `V4ModelHost`. Deliberately placed at the **supply**, not the sweep, so the swept prefill and decode both report the split from the same counters — decode drives the same `dispatch`/`materialize` and, unlike the sweep, never calls `release_streamed_staging` (it orders copies with `hipStreamWaitEvent` and frees slots in `on_routed_consumed`), which is why decode's `h2d_drain` is 0 and the prefill's is not.
- **`tests/bench_supply_split.cpp` + `scripts/supply_split.sh`** — one model load, a fresh session per length (`host.reset_generation_state()`, which zeroes every layer's KV state without a reload), the **gate-selected** strategy (not forced), and a short **greedy** decode. Not `bench_prefill_ab`: that bench reloads the artifact per arm and is a strategy A/B; this one is an attribution run.

The three counters were placed around exactly the regions §3 named: `io_wait` around `materialize`'s completion wait, `h2d_enqueue` around the upload submission, `h2d_drain` around `release_streamed_staging`'s per-slot sync.

### 8.2 Results

Reference machine: 4× RX 7900 XTX (`gfx1100`), single NVMe (`~6.33 GB/s` reference rate), context 2048, `C = 128`, `W = 1024`, gate `3E/4 = 192`, 797 Hot slots, no Warm tier unless noted. `nvme` is cold bytes only.

**Prefill** (one window; `io_wait`/`h2d_drain` are host-blocked, `h2denq`/`submit` are CPU submit cost):

| Warm | N | strategy | wall_s | tok/s | nvme_GiB | io_wait_s | h2denq_ms | h2ddrn_s | submit_ms | drains |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 64 | routed | 15.000 | 4.27 | 49.64 | **8.355** | 36.4 | 0.000 | 17.1 | 0 |
| 0 | 256 | swept | 34.588 | 7.40 | 141.37 | **2.065** | 71.7 | **4.566** | 31.1 | 10 723 |
| 0 | 512 | swept | 63.589 | 8.05 | 141.37 | **0.531** | 65.8 | **5.357** | 29.3 | 10 723 |
| 35 | 64 | routed | 12.294 | 5.21 | 36.95 | **6.150** | 37.4 | 0.000 | 9.5 | 0 |
| 35 | 256 | swept | 32.436 | 7.89 | 106.42 | **0.277** | 48.7 | **4.061** | 24.1 | 10 723 |
| 35 | 512 | swept | 62.311 | 8.22 | 106.40 | **0.265** | 47.3 | **4.067** | 23.0 | 10 723 |

**Decode** (64 greedy tokens per length; per-token microseconds):

| Warm | N | steps | ms/tok | io_wait_us | h2denq_us | h2ddrn_us | submit_us | nvme_MiB/tok | warm_MiB/tok |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 64 | 64 | 402.01 | **251 966** | 1 603 | 0 | 460 | 1 703 | 0 |
| 0 | 256 | 64 | 415.03 | **249 616** | 1 326 | 0 | 438 | 1 676 | 0 |
| 0 | 512 | 64 | 393.89 | **221 756** | 1 168 | 0 | 397 | 1 483 | 0 |
| 35 | 64 | 64 | 274.07 | **99 236** | 647 | 0 | 208 | 659 | 1 044 |
| 35 | 256 | 64 | 277.79 | **87 166** | 688 | 0 | 210 | 568 | 1 107 |
| 35 | 512 | 64 | 262.47 | **68 500** | 495 | 0 | 161 | 440 | 1 043 |

### 8.3 What the measurement confirms

1. **The exposed swept-layer H2D is real, and it is pure PCIe bandwidth.** `h2d_drain` is `5.36 s` (Warm 0) / `4.07 s` (Warm 35) at `N = 512`. Against `≈141 GiB` (Warm 0) of copied bytes that is `≈26 GiB/s` — the PCIe Gen4 x16 ceiling, exactly. The upload is not *slow*; it is **serialized** in front of the body, so `6.5–8.4%` of a swept window is unrecoverable PCIe time unless it is overlapped with compute. §3 is confirmed.
2. **`h2d_enqueue` and `submit` are negligible — always.** The whole 43-layer window's H2D *submission* is `47–72 ms` and its `io_uring` submission is `23–31 ms`. Whatever the CPU costs on this path, it is ≈`0.1%` of the window, in every configuration.
3. **The `h2d_drain` counter is sweep-only, as predicted.** Decode's `h2d_drain` is `0` in every row, because decode never calls `release_streamed_staging`. The counter's presence/absence is itself the confirmation that the two phases use different copy-ordering mechanisms.

### 8.4 What the measurement refuted

These are corrections to this document, not to the code:

1. **"`banks = 1` also idles the disk ~6–7 s/window" (§1, §3) — refuted.** The swept window's `io_wait` is `0.27–2.07 s`, and `2.07 s` is only at `N = 256` (where the body is short). At `N = 512` it is `0.53 s` (Warm 0) / `0.27 s` (Warm 35). The disk's service time is `≈24 s` of a `63.6 s` window (38% busy), and it is `98%` hidden — the double buffer works exactly as designed. **The drive is never starved; the reads are a river within the layer too.** Only the *upload* is exposed. (`io_wait` is high — `6–8 s`, ~50% — only on the **routed** path at `N = 64`, which is a different strategy with per-expert scattered reads.)
2. **"256 host syncs cost 55–215 ms/window" (§4.4a) — refuted.** The window does make `10 723` syncs (`~249`/layer), but the `4.1–5.4 s` they spend is the PCIe copy time they absorb, not round-trip overhead. Collapsing them to one saves ≈`0`. The item is correct in principle and worthless in practice **unless the copies themselves are overlapped**.
3. **"The `O(catalog)` scans are ~5–15 ms/layer → 215–645 ms/window" (§4) — refuted as a magnitude.** Total non-I/O dispatch CPU across a whole window is bounded by `h2d_enqueue + submit ≈ 70–100 ms`. The scans may well exist, but they are ≈`0.15%` of the window. §4 should be **deprioritised**, as §4.3 already hedged (M43's audit fix was the real win here, and it is already in).
4. **`banks = 2` alone does not fix the exposed upload.** The measurement forces this distinction: a second bank lets the *next layer's reads* start during this layer's drain, but the reads are already hidden. Alone it does **not** move *this layer's upload* off the pre-body critical path. **Partly superseded by §10:** coupled with a deferred drain, the second bank *does* move the upload off the critical path — the copies stay in flight through the body, and the body's MoE dispatch orders their weights per expert. The correction is the **coupling**, not the bank; and because the bank is unaffordable at the production Warm shape, C2's per-expert recycling is the form that ships.

### 8.5 What the measurement bounds — and what it hands off

The biggest measured supply cost is **not in the prefill** — it is **decode's disk wait**, which the prefill-focused first pass did not examine:

- **Decode is disk-bound: `36–63%` of every token is `io_wait`.** Warm 35: `87–99 ms` of a `~275 ms` token. Warm 0: `222–252 ms` of a `~400 ms` token.
- Decode reads `440–1 700 MiB/token` from NVMe (6 experts × 43 layers × 13.5 MiB = `3.48 GiB` if all cold, so `12–49%` of the draws miss). Warm serves a comparable volume (`~1 045 MiB/token`), so Warm roughly halves the cold traffic.
- Decode's `h2d_enqueue` (`0.5–1.6 ms`) and `submit` (`0.16–0.46 ms`) are negligible; `h2d_drain` is `0`.

This is a **characterization of the transfer path**, and it is in scope: decode uses the same channels, so its split is the honest measure of what those channels cost in the other phase. What is **not** in scope is the remedy its size suggests. Making decode read fewer bytes is the *which bytes should arrive* question — explicitly excluded by this document's §Subject, and owned by the [routing profile and placement study](../execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md). So the finding is recorded and **handed off**, not acted on: within this document's scope the remaining lever is C2/C3 (§8.6), and decode's residual is bounded by how fast the path can serve whatever demand exists — which the numbers above say is already the drive, hidden behind nothing left to remove on the copy side.

### 8.6 Revised disposition of the structural items

| Item | Before §8 | After §8 |
| :--- | :--- | :--- |
| **C1 `banks = 2`** | the fix for the exposed load | standalone: only helps reads, already hidden (§8). **Coupled with a deferred drain: recovers the whole exposed upload (§10), but unaffordable at the production Warm shape.** |
| **C2 rolling corridor** | cheaper alternative to C1 | **now the shipping form**: §10's win needs a whole second bank (`+3.44 GiB` pinned, which swaps the box); C2's per-expert recycling gets the same overlap from a bounded surplus. |
| **C3 completion-driven upload** | "the real prize" | **confirmed and partly delivered** — §10 recovered `3.2–3.5 s/window` (`5–11%`) by deferring the drain; C2 completes it at a shippable memory cost. |
| **C4 large-BAR NVMe→VRAM** | possibly transformative | **closed — `NOT_SUPPORTED` (§9).** Measured on the production `io_uring` path: `-EFAULT`. The pinned staging and the second hop stay. |
| **§4 `O(catalog)` scans** | new open-work item | **deprioritised — ~0.15% of the window.** |
| **Decode disk wait** | not considered | measured at `36–63%` of every token, but its **remedy is residency — out of scope** (the "which bytes" question). Recorded as a hand-off to the routing/placement study; **not** an item this analysis acts on. |

### 8.7 Consequence for the document

§3's *mechanism* and §8's *magnitude* together say: the prefill transfer path is **already close to its ceiling** — the drive is hidden and the only exposed leg is the irreducible PCIe copy, worth single-digit percent. The document's prefill framing was correct about the shape and wrong about the disk. The work that remains worth doing on this path is small and specific (C3/C2 for `4–5.4 s`). The largest measured supply cost, decode's NVMe wait, is **outside this document's subject** — its remedy is which bytes are resident, not how fast they arrive — and is handed to the routing/placement study rather than acted on here.

---

## 9. Measured — the C4 probe (Step 2, 2026-09-24)

*Status: **measured and closed.** C4 is dead on this platform.* Tool: `tools/aeon_c4_probe.cpp` (`aeon_c4_probe`, built under `AEON_BUILD_BENCHMARKS`).

### 9.1 The question, and the preconditions

C4 asked whether the NVMe could DMA straight into VRAM, deleting the pinned staging and the second PCIe hop. Three of the four preconditions measured green beforehand:

| Precondition | State |
| :--- | :--- |
| Large BAR aperture | ✅ **32 GiB** (`resource0` = `0x17800000000`–`0x17FFFFFFFFF`) |
| Kernel P2P DMA | ✅ `CONFIG_PCI_P2PDMA=y` |
| Driver P2P | ✅ `amdgpu.pcie_p2p=Y`; GPU↔GPU matrix all-YES |
| **NVMe accepts a VRAM buffer as a read destination** | ❓ — what the probe tested |

The GPU↔GPU matrix being all-YES does **not** answer the fourth: that path uses the GPU's own copy engine, whereas C4 needs the *NVMe controller* to write into the GPU's BAR — a different kernel path (`get_user_pages` → `dma_map_sgtable` on a dma-buf page).

### 9.2 Result

256 MiB span, 5 reps at distinct file offsets, RX 7900 XTX, single NVMe:

```
[1] dma-buf export               : ok  (via hipMemGetHandleForAddressRange)
[1] mmap the dma-buf             : ok  (ptr 4096-aligned)
[2] GPU write -> CPU read        : ok (mapping aliases VRAM)
[2] CPU write -> GPU read        : ok
[3] BAR bandwidth (CPU-visible)  : write 13.05 GiB/s, read 0.01 GiB/s
[6] pread -> pinned host (ref)   : 36.79 ms  6.79 GiB/s  (disk ceiling)
[5] pread -> host -> BAR copy    : 55.15 ms  4.53 GiB/s  (bounce control)
[4] pread -> VRAM (the target)   : FAILED (Bad address)          <-- EFAULT
[4a] control: pread -> anon host : ok (O_DIRECT itself works here)
[4b]     buffered pread -> VRAM  : ok (buffered path accepts VRAM)
[4c] io_uring O_DIRECT -> VRAM   : REJECTED (cqe.res=-14)         <-- EFAULT
```

**Verdict: `NOT_SUPPORTED`.** Two independent rejections with the same cause:

- `pread(O_DIRECT)` into the VRAM mapping → `EFAULT` (`Bad address`).
- The **production mechanism**, `io_uring` `IORING_OP_READ` with `O_DIRECT` into the same mapping → `cqe.res = -14` (`-EFAULT`).

### 9.3 What the controls establish

The controls are what make this a conclusion rather than a failed experiment:

- **[4a] `O_DIRECT` into an anonymous host mapping succeeds.** So `O_DIRECT` itself, the file, the alignment, and the span size are all fine — the failure is **specific to the VRAM mapping**.
- **[4b] A *buffered* read into the same VRAM mapping succeeds.** So the mapping is a valid, writable user address; what the kernel refuses is using it as a **direct-I/O destination**. The buffered path "works" only because it copies through the page cache first and then writes to the BAR from the CPU.
- **The alias checks ([2]) pass in both directions**, so the mapping really is VRAM — the rejection is not an artifact of a broken mapping.

### 9.4 The mechanism, and why it cannot be worked around from userspace

`O_DIRECT` must translate the destination into a **bus address** for the NVMe controller. For ordinary host memory that is `get_user_pages` → `dma_map_sgtable`. For the BAR/dma-buf mapping, the VMA has no ordinary `struct page` that the block layer will pin, so `get_user_pages` fails and the request is rejected with `EFAULT` before any DMA is programmed. Making it work needs a kernel-side P2P-aware direct-I/O path (the role `nvidia-fs` plays for GPUDirect Storage) — **which does not exist here**, and cannot be added from userspace. This is a platform/driver limitation, not a tuning problem.

### 9.5 Even the fallback that *does* work is worse than what we have

Suppose one used the accepted buffered path ([4b]). It costs `≥ 55.15 ms` per 256 MiB (the measured bounce control: disk read, then CPU write across the BAR at `13 GiB/s`). The **current** leg costs `36.79 ms` (read) `+ ~9.6 ms` (DMA copy at `26 GiB/s`) `≈ 46 ms`. So the CPU-mediated "direct" path is **~20% slower** than the two-hop DMA path we already run — and that is before counting its page-cache traffic and its extra non-reclaimable footprint.

Two further facts fall out of the probe and are worth keeping:

- **CPU reads from VRAM are unusable: `0.01 GiB/s`** (uncached, non-posted). Any design that has the CPU touch VRAM contents is off the table for bulk data.
- **CPU writes to VRAM are decent: `13 GiB/s`** (posted/write-combined). This is the only CPU-mediated direction that is within sight of the DMA path, and it is still half of it.

### 9.6 Consequence

**C4 is closed as `NOT_SUPPORTED`, measured on the production mechanism, with the cause identified and the workaround ruled out.** The pinned staging arena and the two-hop path stay. Concretely:

- The `3.44 GiB` pinned staging **cannot be removed** by this route — the host-memory pressure investigation must solve that differently.
- The `4.1–5.4 s/window` exposed H2D **cannot be removed** by this route either; issuing the upload during the previous body remains the only lever, and it keeps the second hop by construction. **§10 then recovered `3.2–3.5 s` of it.**
- The spike was **worth running**: it converts a plausible, preconditions-green idea into a settled negative in about half a day, and it retires the largest "maybe" in this document.

---

## 10. Measured — Phase 1, the deferred drain (2026-09-24)

*Status: **measured.** The first fix of the execution plan was built and A/B'd. The overlap is real; the memory cost is the blocker. Plan: [SUPPLY_CHAIN_HOT_PATH_EXECUTION_PLAN.md](../../execution/active/SUPPLY_CHAIN_HOT_PATH_EXECUTION_PLAN.md).*

### 10.1 What was built

Three changes, all in the sweep's supply path:

- **A second layer-sized staging bank.** With `prefill_sweep_staging_banks = 2`, the arena is `2E` slots instead of `E` (`3.44 → 6.75 GiB` pinned at `E = 256`), and layer `L`'s reads land in bank `L % 2` — one bank in flight, one bank the lookahead reads into. Both the budget report and the arena call the same `staging_slot_count` helper, so the printed figure is the allocated figure.
- **A deferred drain.** `V4PrefillSweep::materialize_layer` no longer calls `release_streamed_staging`; it settles the reads, keeps the copies in flight, **orders nothing on the compute stream itself**, and holds the batch as a `resident_state_` until `after_layer` returns the bank. `after_layer` then resumes the resident bank, reaps (which promotes the completed uploads out of `PROMOTION_PENDING`) and releases the layer.
- **Ordering left to the consumer.** This is the load-bearing detail. The body's MoE dispatch (`on_routing_ready_batch`) joins a transfer that is still `PROMOTION_PENDING` by its `operation_id`, and `V4TieredExpertExecutor::accumulate_routed` waits on that transfer's per-expert event before the MoE reads the weights; a transfer the registry has already reaped is complete by definition. **Ordering the whole body** behind the copies instead (a single `hipStreamWaitEvent` on the compute stream) recovers **nothing** — measured — because it puts the copy back in front of the attention and router it is meant to hide behind.

### 10.2 Results

`scripts/supply_split.sh` (one model load, gate-selected strategy, 64-token greedy decode), single runs, one process at a time. Warm 30 GiB added because the production shape is where the memory question bites.

**Prefill, swept arm** (`h2d_drain`/`io_wait` host-blocked; `nvme` cold bytes):

| Warm | N | banks | wall_s | tok/s | io_wait_s | **h2d_drain_s** | disp_ms | submit_ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 256 | 1 | 33.811 | 7.57 | 0.678 | **4.566** | 567.4 | 31.5 |
| 0 | 256 | 2 | **30.574** | 8.37 | 1.741 | **0.006** | 536.6 | 31.6 |
| 0 | 512 | 1 | 63.593 | 8.05 | 0.526 | **5.357** | 791.6 | 30.0 |
| 0 | 512 | 2 | **59.087** | 8.67 | 0.525 | **0.006** | 790.2 | 31.6 |
| 30 | 256 | 1 | 32.913 | 7.78 | 0.421 | **4.230** | 481.8 | 25.1 |
| 30 | 256 | 2 | **29.400** | 8.71 | 0.416 | **0.006** | 625.1 | 22.8 |
| 30 | 512 | 1 | 62.280 | 8.22 | 0.311 | **4.247** | 697.5 | 25.8 |
| 30 | 512 | 2 | **59.055** | 8.67 | 0.313 | **0.006** | 855.7 | 22.4 |

**Decode** (64 greedy tokens) is unchanged in every pairing (`ms/tok` within noise; `h2d_drain` was already `0`). `io_wait` is unchanged, which is the proof the reads were not the problem.

### 10.3 What the measurement establishes

1. **The exposed upload is removable, and this removes it.** `h2d_drain 4.2–5.4 s → 0.006 s`, and `wall_s` falls by a comparable amount: `−3.24 s` at `N = 256`, `−4.51 s` at `N = 512` (Warm 0); `−3.51 s` / `−3.23 s` at Warm 30. That is `5–11%` of the swept window, matching §8.6's predicted ceiling.
2. **The win holds with an active Warm tier.** User prediction confirmed: Warm 30 shows the same absolute saving as Warm 0. The overlap is a function of the copy overlapping the body, which does not depend on where the bytes came from.
3. **`io_wait` does not rise.** The reads were hidden before and stay hidden; nothing was merely moved from the copy to the disk. `io_wait` at `N = 512` is `0.53 s` (Warm 0) / `0.31 s` (Warm 30) in both arms.
4. **The ordering must be the consumer's, not the driver's.** The first attempt ordered the whole body behind the copies and recovered **zero**. The fix is not "wait somewhere else" but "don't wait at all on the driver side; let the per-expert consumer wait" — which is what the decode path already does.
5. **`disp_ms` rises slightly** (`698 → 856 ms` at Warm 30, `N = 512`) — more transfers in flight at once makes the `O(catalog)` scans of §4 costlier. Still ≈`0.4%` of the window, and it now has a *reason* to exist: Phase 3's `O(1)` count becomes worth doing if the in-flight population grows.

### 10.4 The blocker, and the consequence

**`banks = 2` costs `+3.44 GiB` of non-reclaimable pinned host memory.** At Warm 0 that is fine. At the production Warm shape it is not: Warm 35 GiB is *also* pinned, so `35 + 6.75 = 41.75 GiB` pinned plus the ~13 GiB of process/driver/dense footprint the budget does not model pushed the 62 GiB machine to ~50 GiB used **and 5 GiB of swap** during the measurement — the sweep did not complete.

Two findings fall out, one about this document and one about the budget:

- **The default must stay `banks = 1`.** The overlap is proven but not yet affordable. **Superseded by §12:** the default is now `2E` because `1E` is a measurably worse algorithm (`0/43` overlapped), and the pinned cost is addressed by making the second block cheaper (P2.2/P2.3), not by keeping the parking lot.
- **The host-reserve model is under-sized** (`HOST_RAM_RESERVED_BYTES = 10 GiB`). The budget check *passed* `35 + 6.75 = 41.75 ≤ 52` and the machine still swapped, so the unmodelled overhead is `> 10 GiB`. This belongs to the open [host-memory pressure investigation](HOST_MEMORY_PRESSURE_INVESTIGATION.md) — it is a data point for it, not a defect in this plan.

**Consequence for the plan:** Phase 1's win is **conditional on Phase 2**. Per-expert slot recycling (C2) recovers the same overlap from a bounded surplus (a handful of slots) instead of a whole `E`-slot bank, which is what makes it fit both the memory budget and the host-memory investigation's constraint. Phase 2 is therefore **mandatory, not optional** — it is how Phase 1's result ships.

---

## 11. Measured — the portability sensitivity gates (2026-09-24)

*Status: **measured.** Gate B passes, Gate A fails. The instruments exist and are reusable; the failure is the work item.*

### 11.1 Why gates instead of a profile

The obvious next measurement is the pipeline's slot-occupancy profile over a window, which would say how much corridor room exists. It was rejected: that figure is **fitted to this box** and decides nothing on a machine with a faster SSD, a PCIe-5 link, or a slower GPU. The requirement is a supply chain that is fast *because it is demand-driven*, not because a constant happens to suit the reference machine, so the thing to test is a **property**, not a rate. Both gates below are property tests, and neither compares against a bandwidth, a latency, or any device figure — the verdict is the *shape* of the curve, so it holds on hardware faster or slower than this one.

### 11.2 What was built

- **`tests/test_v4_staging_depth.cpp`** (new gate): one model load, then one identical swept window at each staging depth in `64 128 192 256 384 512`, each in a fresh session with the counters reset. It asserts the two properties below and prints the sweep.
- **`V4ModelHost::resize_staging_slots(slots)`** (new): re-sizes the arena at runtime, guarded by the two exact preconditions (every slot free, no lease outstanding — the resize recreates the per-slot events) and keeping `transient_staging_bytes` equal to the allocation. This is the capability the portability story needs: the depth is a **budget the host can set**, not a figure fixed at load.
- **The sweep's bank count is now derived from the arena it actually has** — `slots / experts_per_layer` — instead of a configured constant. The `2` that used to select "one layer reading, one layer copying" is gone from the control flow; it is now an emergent consequence of the depth. This is the one fitted constant that lived in the behaviour, and it no longer does.

### 11.3 Results

`./build/bin/test_v4_staging_depth`, one swept window, `N = 256`, 43 layers × 256 experts, Warm 0, one process:

| depth | slots | banks | wall_s | io_wait_s | **drain_s** | layers | experts | token | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 64 | — | — | — | — | — | — | — | — | below the swept dispatch's floor |
| 128 | — | — | — | — | — | — | — | — | below the swept dispatch's floor |
| 192 | — | — | — | — | — | — | — | — | below the swept dispatch's floor |
| 256 | 256 | 1 | 34.755 | 2.122 | **4.212** | 43 | 10718 | 86 | swept |
| 384 | 384 | 1 | 35.032 | 2.113 | **4.647** | 43 | 10718 | 86 | swept |
| 512 | 512 | 2 | 30.880 | 1.060 | **0.006** | 43 | 10718 | 86 | swept |

### 11.4 What the gates establish

1. **Gate B — behaviour symmetry — PASSES.** The token, the layer loads, the experts streamed and the layers released are **identical at every depth that ran**. The pipeline's *work* is resource-invariant: it does the same thing whatever the depth, so the logic is not reading a resource as a policy. This is the good half of portability and it already holds.
2. **Gate A — depth insensitivity — FAILS, at exactly one place, and it is a behaviour threshold.** The spread is `1.134×`. `256 → 384` (a `+50%` arena) changes **nothing** — same regime, same time — and then `512` drops `13%` because at `2E` the drain flips from host-blocking to deferred. So the throughput does not depend on the depth continuously; it depends on which **algorithm** the depth selects. A design whose speed is set by a stage change hidden behind an arena size is the definition of fitted: on a machine where the copy is cheap relative to the reads, `2E` is the wrong threshold and there is nothing to notice it with.
3. **The floor is the deeper problem.** Three of six depths cannot run at all: a swept dispatch binds a whole layer's distinct set at once, so the arena must hold `E` slots before the window can start. **Depth is therefore a floor, not a budget** — the smallest corridor the design will accept is one whole layer's worth of pinned memory, which is precisely the `3.44 GiB` that does not fit the production Warm shape (§10.4). This is the same fact as Phase 1's memory blocker, seen from the other side.

> **The premise that per-expert recycling alone would fix it was wrong, and this is why.** The obvious reading of "a bounded surplus instead of a whole bank" is that recycling slots at expert granularity lets `E + S` replace `2E`. It does not, in this structure: `V4Graph::forward_window` synchronizes the compute stream at every layer boundary (it must — `release_layer` frees the VRAM slots the body just read from), so the **host cannot run ahead of the compute**. The reads for `L+1` are dispatched at one instant, before `body(L)`, and they need `E` destinations while `L`'s copies still hold `E`. The `2E` is that instant's demand, not a slack choice. Making it `E + S` requires the host side to advance as **events** complete, which is the work restated in Phase 2.

### 11.5 Consequence

Phase 2 is restated by these gates rather than by a profile, and it now has pass/fail criteria instead of a fitted constant:

- **Gate A must pass over a wide range**, down to depths well below `E`. That is the operational meaning of "the corridor is a budget".
- **Gate B must keep passing** — it is the regression guard that the fix does not make the work depend on the resource.
- Neither gate mentions this hardware, so the same two commands decide the question on any machine.

The instruments are cheap (one model load, three windows) and already committed, so every later step is measured against a property rather than a number.

---

## 12. Measured — the staging arena is one room or two (2026-09-25)

*Status: **measured.** The "corridor" and the "parking lot" are now a number, not a description. Plan step P2.1.*

### 12.1 What was added

- **`PrefetchStagingArena::StateCounts`** — occupancy **by pipeline stage**: `free`, `reading` (a read landing in the slot), `copying` (a copy-into-VRAM draining the slot). `in_use_slots` said how many were busy; this says *what for*, which is the distinction the corridor claim rests on.
- **`V4PrefillSweep::occupancy_samples()`** — one sample per layer, taken as each body begins (after the lookahead is issued), so it is the corridor's fill for that layer. Cleared per window.
- **Gate C** in `test_v4_staging_depth` — a layer counts as *overlapped* when a read and a copy are in flight at the same instant. That is the pipeline working; a single block pinned at `E` with the other empty is the parking lot.
- **`staging_slot_count` returns `2E` unconditionally** when the sweep is on, and the `prefill_sweep_staging_banks` knob is **removed** (a settable depth lets a resource select the algorithm — plan R6). The A/B now moves depth through `resize_staging_slots`.

### 12.2 Results

One swept window, `N = 256`, 43 layers × 256 experts, Warm 0, one process:

| depth | slots | banks | wall_s | drain_s | overlapped layers |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 / 128 / 192 | — | — | — | — | refused (below the dispatch's floor) |
| 256 | 256 | 1 | 34.959 | **4.856** | **0/43** |
| 384 | 384 | 1 | 35.065 | **4.883** | **0/43** |
| 512 | 512 | 2 | **30.641** | **0.006** | **42/43** |

### 12.3 What it establishes

1. **The two models are exactly as claimed, and now measured.** At `1E` the arena is **one room**: `0/43` layer-bodies ever had a read and a copy in flight together, and the drain is the full `4.9 s`. At `2E` it is **two rooms**: `42/43` bodies overlapped, and the drain collapses to `0.006 s`. The `+14%` wall-time gain is that overlap and nothing else.
2. **`384` is not a third shape.** It grew the arena by `50%` and changed nothing (`0/43`, same drain) — because `banks = 384 / 256 = 1`, so it is still the one-room algorithm with `128` idle slots. This is the clearest single piece of evidence that the **depth is selecting the algorithm** rather than being a budget: below `2E`, slots are pure waste.
3. **The floor is the read wave, confirmed behaviourally.** `64/128/192` cannot run at all: the dispatch binds a whole layer's set, so the arena must hold `E` before the window starts. Note that `256 → 384` gaining nothing is the *same fact* seen from the other side — the extra `128` slots buy no overlap because they cannot form a second block.
4. **`vram_reserved_ahead` peaks near `E`** (`250` of `256`) in every configuration, which is the reserved-but-empty figure the pipeline model predicted: the lookahead layer's VRAM block is held while its bytes are still in staging.

### 12.4 Consequence

P2.1's part is done: the arena is `2E` by default in both scenarios, the knob is gone, and the pipeline's fill is observable. The remaining Phase 2 steps are unchanged and now have a measurement to move:

- **P2.2** (release each staging slot on its own copy event) is what could bring the floor down from `E`, since the `256 → 384` result shows that **extra slots below `2E` are worthless** — the win comes from forming a second block, not from having spare slots.
- **Gate A** still fails (`1.144×`). Its target is that the `1E → 2E` step disappears because the algorithm no longer depends on the size.
