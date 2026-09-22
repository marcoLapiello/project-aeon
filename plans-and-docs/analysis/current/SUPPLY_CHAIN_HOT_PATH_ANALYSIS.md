# Supply-Chain Hot-Path Analysis — is the feed a river or a bucket?

*Status: open analysis. Written 2026-09-22. A first-pass audit of the expert supply's transfer path and its per-request bookkeeping, in answer to two questions: (1) is the NVMe→VRAM feed a continuous stream or an interrupted one; (2) which substeps on that road are expensive enough to be worth removing. Source read: `direct_io_reader.hpp`, `tiered_expert_supply.hpp`, `prefetch_staging.hpp`, `expert_registry.hpp`, `v4_expert_supply.hpp`, `v4_prefill_sweep.hpp`, `v4_expert_executor.hpp`, `v4_model_host.hpp`. No code changed.*

**Subject.** The mechanics of moving a routed expert from NVMe into VRAM — the read submission, the staging corridor, the H2D, and the registry bookkeeping around all three. This is the *how fast can the bytes arrive* question, not the *which bytes should arrive* strategy question, which the [prefill supply review](PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md) owns.

**Scope.** The paths as written, with the cost of each substep reasoned from the code. **Nothing below is measured.** Every estimate is flagged as such, and §5 names the counters that would confirm or refute it.

---

## 1. The verdict, split into its two halves

The feed is not one thing, and the two halves have opposite answers:

| Half | Answer | Evidence |
| :--- | :--- | :--- |
| **NVMe reads** | **a river** | one `submit_pending_reads()` per batch, queue depth sized to hold a whole layer, `IOSQE_ASYNC` set |
| **H2D upload of a swept layer** | **a bucket** | enqueued at `before_layer`, then host-synchronized one slot at a time before the body runs |

So the drive is kept busy across layers (the double buffer works), but the **PCIe leg of a swept layer is fully exposed in front of that layer's compute**. That is the concrete interruption, and it is the finding that answers question 1.

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

### 4.3 Rough total

`~240 M` catalog visits per window on the dispatch thread, plus allocation churn. Order-of-magnitude estimate (**not measured**): `~5–15 ms/layer`, i.e. the same order as the exposed H2D of §3. Both are fixable for very little: an `O(1)` pending-transfer counter maintained in `reserve_request`/`complete_request`/`fail_request`, and a reverse `operation_id → demotion` index (or simply skipping the scan when no demotion is pending).

### 4.4 Smaller substeps worth noting

| Site | Cost | Note |
| :--- | :--- | :--- |
| `record_h2d_event` / `reap_registry_transfers` | `hipEventCreateWithFlags` + `hipEventDestroy` **per expert** | ~256 creates + 256 destroys per layer (~22 k/window). The staging arena **already owns** a per-slot event recorded on the same stream (`events[staging_idx]`); the registry's extra event may be redundant. Worth checking whether the two can be one. |
| `find_registry_transfer` / `ensure_registry_transfer` | linear scan of `registry_transfers_` | called several times per transfer (dispatch → `bind_staging` → `record_h2d_event` → reap). With 256 in flight this is `O(256)` per call, repeated. A map keyed by `operation_id` removes it. |
| `ExpertRegistry::validate_invariants` on every `release_layer` | allocates two `uint8_t` vectors (`total_experts`, `vram_capacity`), multiple full-catalog scans, `validate_lru` | 43× per window. Likely hidden under in-flight reads at large `N`; exposed at small `N`. |
| `LayerPrefetchState::sync_state` | re-materializes the batch into 9 parallel vectors (`assign` + loop), twice per layer | snapshot bookkeeping; `9 × 256` writes × 2. |
| `dispatch_layer_prefetch_batch` dedup | `unordered_map` rebuilt per layer | inherent to dedup (D2); noted for completeness. |
| `dispatch` request ordering | `stable_sort` over requests | `O(256 log 256)` per dispatch. |
| `release_streamed_staging` | `hipEventSynchronize` × 256 | the host-blocking half of §3. |

### 4.5 The per-request audit is *not* the problem

`checked_validate()` is a no-op unless `validate_each_request_` is set, and it is off by default. So the `O(catalog)` cost of `validate_invariants` does **not** land on the per-request path in production — only on the boundaries of §4.4. Worth stating explicitly so it is not mistaken for the hot-path cost.

---

## 5. What would confirm or refute this

Everything above is reasoned, not measured. The counters already exist for most of it; the fix is mostly to *attribute* rather than to instrument anew:

1. **Separate the read wait from the H2D issuance in the sweep.** `V4PrefillSweep::load_ns_` currently lumps `materialize`'s io wait and its H2D enqueue+host-wait together, so the exposed-H2D claim (§3) cannot be read off it. Splitting `materialize_layer` into `io_wait_ns` and `h2d_wait_ns` would settle §3 directly.
2. **Time `dispatch()` against `submit_pending_reads()`.** `direct_io_submit_ns_` already isolates the `io_uring_enter` cost; the residual `dispatch()` time is where §4's `O(catalog)` scans live. If the residual is small, §4 is overestimated.
3. **A/B the two scans.** Replace `pending_transfer_count()` with an `O(1)` counter and skip the no-demotion catalog scan, then compare swept per-layer time. `n = 1` will not resolve it; the deployable test is whether it moves `direct_io_submit_ns`'s sibling — the non-io dispatch time — at all.
4. **Prove the H2D overlap is real** by constructing the arena with `banks = 2` and re-running a window, checking `load_ns` and frontier depth. This is the plan's open staging item; this analysis explains why it should matter.

---

## 6. What this analysis does not claim

- **It does not claim the drive is underused.** The reads are a properly pipelined submission; the double buffer keeps the drive working through the body. The interruption is on the **PCIe** leg, not the disk leg.
- **It does not claim the `O(catalog)` scans are the dominant cost.** They are *unconditional waste on the hot path* — cheap to prove and cheap to remove — but they are bookkeeping, not supply speed. §3 is the finding with throughput consequence.
- **It does not claim any number here is measured.** All figures are reasoned from the code and the plan's own constants and are marked so. §5 exists precisely to turn them into measurements before any of them is acted on.
- **It does not re-open strategy.** River-versus-bucket is about *how fast bytes arrive*, not *which bytes*. The route-aware-versus-blind question is the [prefill supply review](PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md)'s.

---

## 7. Disposition

| Item | Where it goes |
| :--- | :--- |
| Exposed swept-layer H2D (§3) | Directly informs the plan's open **"staging arena's `banks × depth` target"**: `banks = 1` is *why* read and upload cannot overlap across layers. |
| `O(catalog)` scans (§4.1–4.2) | New open-work item: replace `pending_transfer_count()` with an `O(1)` counter; skip the no-demotion catalog scan. |
| Redundant per-transfer HIP event (§4.4) | Investigate aliasing the registry event to the arena's per-slot event. |
| `load_ns` split (§5.1) | Small instrumentation change that makes §3 measurable. |
| Kernel-side cost | Out of scope here; the compute-side audit is a separate document. |
