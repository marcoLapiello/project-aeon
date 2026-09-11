# Warm-Tier Repair and Supply Telemetry Execution Plan

**Date:** 2026-09-11
**Status:** Active; next implementation priority
**Priority:** P0 foundational runtime infrastructure
**Parent track:** Phase 2 is paused while this plan is executed
**Scope:** Persistent Hot/Warm residency, asynchronous refill, transfer ownership, and supply-chain telemetry

## 1. Decision Record

The broad Phase 2 optimization sequence is paused. The next implementation is this focused plan. The existing [Phase 2 execution plan](PHASE_2_EXECUTION_PLAN.md) remains the historical record of the completed storage and kernel work and is not being rewritten as part of this step.

The following decisions are accepted for implementation:

1. **Persistent Hot-to-Warm refill is the default whenever Warm capacity is configured.** It is not an opt-in production feature. A no-refill path remains only as a diagnostic A/B control and for `WARM=0` compatibility.
2. **Demotion is asynchronous and droppable, not synchronous and mandatory at any cost.** A Hot victim is demoted whenever a safe Warm destination and transfer capacity are available. The request path must continue without waiting for background D2H work when the destination is leased, the queue is pressured, or the host budget is exhausted.
3. **Transient host transport and persistent Warm ownership are separate concepts.** Every Cold read still travels through an aligned host buffer because the current hardware path has no direct SSD-to-GPU transfer. That buffer may be a transient staging slot; the expert does not become a persistent Warm resident unless the admission path explicitly installs it there.
4. **The first Cold path remains `Cold NVMe -> transient host staging -> Hot VRAM`.** An eligible Hot victim is independently placed in Warm. A future background admission path may read a Cold expert directly into an unused Warm slot, but a request must not pay an unnecessary `Cold -> Warm -> Hot` logical transition.
5. **The single-tier catalog representation may be extended, not discarded for its own sake.** The implementation must expose explicit transfer and lease states while preserving compatibility with existing tier and slot lookups until callers are migrated.
6. **Routing profiles and placement policy are deferred.** No profile-derived placement, rolling layer residency, policy simulator, or MTP work is accepted by this plan. Those decisions require the model-correctness and Prefill gates described in the DeepSeek comparison.

The external expert's `10-15 tok/s` estimate is a useful hypothesis, not an acceptance target. The acceptance target for this plan is correct persistent Warm behavior with lower measured Cold supply cost and no new critical-path synchronization.

## 2. Why This Is the Next Step

The current registry has the exact state transition that causes Warm to drain:

```text
Warm request:
    remove the expert from Warm
    upload it to Hot
    leave a free Warm slot

Hot eviction:
    remove the victim from Hot
    mark it Cold
    reuse the VRAM slot
```

The implementation in `src/core/expert_registry.hpp` has one logical `tier` and one `slot_idx` per expert. `allocate_vram_slot()` explicitly returns Hot victims to Cold, while promotion frees the incoming expert's Warm slot. The runtime path in `src/core/v4_pipeline.hpp` can upload a pinned Warm expert directly to VRAM, but it has no normal operation that publishes the evicted Hot victim into a persistent Warm slot.

The existing measurements establish that Warm is valuable when it remains populated:

- M15: `4.37` tok/s without Warm versus `5.15` tok/s with a 35 GiB Warm profile.
- M20: `6.32` tok/s without Warm versus `7.10` tok/s with a 35 GiB Warm profile.
- M16: removing request-path demotion reduced redundant traffic but allowed Warm coverage to drain, with `157` Warm hits versus `684` in the previous refill behavior.

These numbers do not prove a particular final throughput. They do prove that the next useful experiment is a correct persistent Warm state machine, measured with source-tier bytes and exposed wait rather than one aggregate Hot-hit rate.

## 3. Terminology and Transfer Model

### 3.1 Persistent versus transient host memory

**Persistent Warm ownership** means the registry has assigned an expert to a valid host-pool slot and the slot remains available for a later Warm hit.

**Transient staging** means a host buffer is reserved for one I/O or H2D operation and is released after its consumer GPU event completes. `PrefetchStagingArena` is transient by default; its bytes must not be counted as Warm residency.

The physical and logical paths are therefore:

```text
Warm hit:
    persistent Warm slot -> H2D -> Hot VRAM

Cold miss:
    Cold NVMe -> transient aligned host staging -> H2D -> Hot VRAM

Hot refill:
    Hot VRAM -> asynchronous D2H -> persistent Warm slot
```

A Cold payload may later be admitted to persistent Warm during idle background work, but that is not required to satisfy the immediate request.

### 3.2 Required lifecycle

Every resident or in-flight expert must be representable by a state equivalent to:

```text
COLD
WARM_RESIDENT
HOT_RESIDENT
PROMOTION_PENDING
DEMOTION_PENDING
IO_PENDING
GPU_TRANSFER_PENDING
LEASED
RECLAIMABLE
```

The exact C++ enum and ownership layout may follow existing conventions, but the implementation must distinguish logical residency from an operation that has not completed.

### 3.3 Core invariants

1. An expert has at most one persistent owner: Hot, Warm, or Cold. Transient staging is not a second owner.
2. Every occupied VRAM slot maps to exactly one catalog entry, and every Hot catalog entry maps back to that slot.
3. Every valid Warm slot maps to exactly one catalog entry, and every Warm catalog entry maps back to that slot.
4. A published Warm entry is not visible to request lookup until its D2H payload is complete.
5. A published Hot entry is not consumable by a routed kernel until its H2D payload is complete.
6. A VRAM slot is not overwritten until any D2H operation that reads its previous payload has completed.
7. A host slot is not reused until any H2D operation that reads its previous payload has completed.
8. A leased current-layer expert cannot be evicted or overwritten until its consumer event has completed.
9. A second request for an expert with a pending transfer joins or waits for the existing operation; it must not submit a duplicate read or publish a conflicting owner.
10. With `host_capacity == 0`, no Warm state or D2H refill is attempted and the current Cold behavior remains available as the control path.

## 4. Intended Runtime Transitions

### 4.1 Hot hit

The registry touches the Hot LRU entry, records a Hot service event, and returns the resident VRAM slot. No transfer is issued.

### 4.2 Warm promotion and Hot refill

When a requested expert is persistently Warm:

1. Reserve a Hot slot and a lease for the current layer.
2. Mark the Warm source slot and the selected Hot victim as unavailable for conflicting operations.
3. If Warm is full, select a different Warm LRU victim and mark it Cold to free a host destination. Never select the incoming Warm expert as that victim.
4. Schedule the Hot victim D2H into the freed Warm destination on the demotion stream before the old VRAM slot can be overwritten.
5. Schedule the incoming Warm-to-Hot H2D only after the source Warm payload is protected and any required destination ordering is established.
6. Publish the incoming Hot entry after H2D completion and publish the demoted Warm entry after D2H completion.
7. Release the leases and make the staging or host slots reclaimable only after the relevant HIP events have completed.

If the operation needs a temporary swap buffer because no independent host destination is available, that buffer must be explicitly reserved and included in the capacity budget. The implementation must never overwrite the incoming Warm source slot or the old Hot source slot prematurely.

### 4.3 Cold request

When a requested expert is Cold:

1. Reserve a Hot destination and a current-layer lease.
2. Attempt to demote an eligible Hot victim into a valid Warm destination in the background.
3. Submit the aligned, chunked Cold read into a transient staging slot.
4. After I/O completion, enqueue H2D on the cold-DMA stream and publish the Hot entry only after its transfer event.
5. Release the transient staging slot after the routed GPU consumer has completed.

The Cold request does not need to occupy persistent Warm before reaching Hot. This avoids a needless second host ownership transition and prevents a persistent Warm slot from being treated as both a source and destination during a swap.

If no safe Warm destination exists, a demotion may be dropped or deferred. If no safe Hot destination exists, the existing request must preserve its current correctness behavior and wait for a reclaimable slot rather than reusing a leased slot.

### 4.4 Background Warm fill

Initial Warm preload and idle Cold-to-Warm admission are separate from request-path promotion:

- `warm_capacity`: configured maximum number of logical Warm slots;
- `warm_storage_capacity`: host bytes actually allocated;
- `warm_initial_fill`: whether startup populates selected slots;
- `warm_valid_slots`: slots whose payload and catalog ownership are complete.

The first implementation must permit an empty or partially populated Warm pool to become valid through normal demotion without requiring a synchronous full-model preload. Direct Cold-to-Warm background fills may be added only after the request-path state machine is correct and observable.

## 5. Scope

### In scope

- Explicit Hot/Warm ownership and transfer-pending state.
- Persistent Warm refill on eligible Hot eviction.
- Correct Warm-to-Hot promotion ordering.
- Warm LRU victim selection and host-slot reuse.
- Separate transient staging accounting.
- Configured Warm capacity independent from startup preload.
- Demotion stream and HIP event dependencies.
- Source-tier, byte, transfer, wait, occupancy, queue, and host-pressure telemetry.
- Synthetic registry tests and controlled full-model A/B validation.
- Preservation of the `WARM=0` behavior and current default output contracts.

### Explicitly out of scope

- CSA/HCA/indexer attention implementation.
- Correct compressed-cache Prefill and true batched Prefill.
- Routing-profile collection for placement decisions.
- Frequency-informed or quota-based placement.
- Rolling layer-window residency and speculative future-layer candidates.
- The external policy simulator as an acceptance tool.
- MTP, speculative decoding, CPU expert fallback, or another expert-kernel rewrite.
- Physical `.aeon` repacking and the `>= 6.0 GB/s` scattered-file target.
- Phase 3 multi-GPU execution.

The routing profiler may still be compiled and smoke-tested as infrastructure, but its data remains invalid for placement decisions until the model-correctness gates close.

## 6. Implementation Stages

### Stage 0: Freeze the control measurement

Record a reproducible baseline before changing the runtime state machine:

- same model artifact, device, context, raw-ID prompt, and generation limits;
- Warm disabled and the current full Warm configuration where host pressure permits;
- cold-start and post-warmup labels;
- Prefill and Decode labels;
- generated IDs and stop condition;
- current Hot, Warm, and Cold service counts;
- process RSS and swap before and after the run.

The baseline is a supply-chain control measurement, not model-quality evidence. It must not be mixed with M22 native-text measurements or with the isolated M23/M24 kernel benchmarks.

### Stage 1: Add residency invariants and telemetry without changing policy

Add a focused registry validation surface and a disabled-by-default telemetry collector.

Required observations:

- request count by phase and source tier;
- logical bytes served from Warm and Cold;
- physical H2D and D2H bytes;
- NVMe bytes, read service time, and completion wait;
- H2D enqueue-to-ready time;
- GPU wait for expert readiness;
- demotion attempts, completions, drops, and queue-delay reasons;
- transient staging-slot waits and reuse delays;
- Hot and valid Warm occupancy over time, by layer;
- pending transfer count and queue depth;
- process RSS, configured host budget, and `VmSwap` delta.

The collector must be resettable between warmup and measured phases and must emit machine-readable fields in addition to concise terminal output. Existing default inference must not collect or print detailed telemetry unless enabled.

Acceptance gate:

```text
Telemetry is phase-labelled and source-tier specific.
A long synthetic registry trace can assert all bidirectional slot/catalog invariants.
The default output and existing tests remain unchanged when telemetry is disabled.
```

### Stage 2: Implement the persistent Warm state machine

Extend `ExpertRegistry` and the host-pool ownership boundary with explicit reservation and publication operations. Keep the registry responsible for logical ownership and LRU policy; keep `HostExpertPool` responsible for segmented storage, allocation, pinning status, and matching cleanup.

Required behavior:

- promotion removes Warm ownership only after the incoming payload is protected for H2D;
- a Hot victim is selected only if it is not leased or transfer-protected;
- a Warm destination is freed or allocated before a demotion is published;
- a full Warm pool evicts its own LRU victim to Cold before accepting a new demotion;
- a pending D2H operation is not exposed as a Warm hit;
- a pending H2D operation is not exposed as a Hot hit;
- `WARM=0` bypasses all Warm operations;
- `preload_warm_host=false` can start with valid capacity zero without disabling later refill;
- pinned and unpinned host segments are reported separately.

Add a host-only or lightweight unit test for deterministic transitions, including:

- Hot hit ordering;
- Warm promotion with a free host slot;
- Warm promotion with a full host pool;
- Cold request with and without a Warm destination;
- dropped demotion under a simulated queue-pressure condition;
- pending transfer protection;
- repeated promotion/eviction over more transitions than total slots;
- zero Hot or zero Warm capacity rejection/behavior as appropriate.

### Stage 3: Integrate asynchronous refill into the pipeline

Replace the current demotion-free ownership transition only after Stage 2 passes.

- Create a dedicated demotion stream with the lowest practical priority supported by the HIP runtime. If stream priority is unavailable or ineffective, retain a separate stream and report that fact.
- Enqueue D2H before any H2D overwrites the source VRAM slot.
- Use HIP event dependencies to publish Warm only after D2H and to publish Hot only after H2D.
- Keep Cold NVMe reads on the existing direct-I/O and cold-DMA path unless a measured background Warm-fill operation is being tested.
- Do not synchronously wait for D2H in the request path. A request may wait for a source payload it actually consumes, but not for an optional victim refill.
- Prevent host-slot reuse until the H2D event that consumes the previous host payload has completed.
- Add single-flight handling for an expert that is already promotion- or demotion-pending.
- Record all dropped or deferred demotions so a low refill rate is distinguishable from a low request rate.

### Stage 4: Validate on silicon

Run the same controlled workload through:

1. `WARM=0` control;
2. current demotion-free Warm behavior;
3. repaired default persistent Warm behavior.

Use multiple repeated runs after one identical initialization procedure. Report medians and run-to-run spread rather than a single favorable result.

Required comparisons:

- exact generated token IDs and stop behavior;
- Hot, Warm, and Cold request counts;
- `bytes_from_nvme` per token;
- `bytes_from_host` per token;
- D2H refill bytes per token;
- GPU expert-readiness wait per token;
- total decode step latency and TTFT;
- valid Warm occupancy after warmup;
- demotion completion and drop rates;
- RSS and swap deltas;
- pinned versus unpinned Warm service;
- staging and queue waits.

### Stage 5: Close or redirect the plan

The plan closes only when the structural and silicon gates below pass. On closure:

- record the exact A/B in `PERFORMANCE_LEDGER.md`;
- update `AGENTS.md` and `DOCUMENTATION_STATUS.md`;
- decide whether Phase 2 can resume at cold-layout work or must remain paused for host-pressure or scheduling work;
- do not automatically begin placement profiling or rolling residency.

If the repaired path preserves Warm occupancy but increases exposed GPU wait, keep the correctness state machine and adjust admission, stream priority, or host capacity in a follow-up plan. Do not silently restore demotion-free behavior as the default.

## 7. Acceptance Gates

### Gate A: Registry and ownership correctness

- No duplicate persistent Hot/Warm ownership exists after a long transition trace.
- Every catalog entry, physical slot map, free-slot list, and LRU list agrees.
- No stale Warm bytes are reported as a valid hit.
- No leased or transfer-protected payload is overwritten.
- A pending operation cannot be published twice.

### Gate B: Runtime transition correctness

- Warm capacity does not drain during a steady-state run when refill destinations and transfer capacity are available.
- Repeated promotions produce measurable Warm service after the initial population is consumed.
- Cold requests remain correct when demotions are dropped.
- `WARM=0` behavior remains equivalent to the current control within normal measurement variance.
- Existing focused Phase 2 tests and the golden-token pipeline tests pass.

### Gate C: Supply measurement correctness

- Every measured request is labelled `Prefill` or `Decode` and identifies its source tier.
- Host, NVMe, H2D, D2H, staging, and GPU-readiness quantities are reported separately.
- Warm occupancy and free capacity can be reconstructed from telemetry.
- Warmup counters can be separated from the measured interval.
- Host RSS and swap are reported against the configured budget.

### Gate D: End-to-end benefit

Under an identical post-warmup workload, the repaired path must show:

- lower Cold NVMe bytes per token than the demotion-free control;
- nonzero and stable Warm service after repeated promotions;
- no new synchronous wait for optional demotion;
- no generated-ID divergence;
- no unbounded staging, RSS, or swap growth;
- a decode result that is no worse than the demotion-free control after accounting for run variance, unless the ledger documents a deliberate host-pressure tradeoff.

No fixed throughput number is required at this stage. The first performance decision is whether persistent Warm refill lowers the measured supply cost without moving the wait to another queue.

## 8. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| D2H refill contends with critical H2D on the same PCIe link | Separate stream, event ordering, droppable admission, and explicit D2H bytes/wait telemetry. |
| A Warm source slot is overwritten during promotion | Protect the source until H2D completion; use a freed Warm victim or an explicit swap buffer. |
| A VRAM slot is overwritten before its victim is copied | Enqueue D2H before H2D and publish the new Hot owner only after the dependency chain is complete. |
| Unpinned host segments cause hidden runtime bounce buffering | Record pinning per segment and route unpinned transfers through a bounded fallback without claiming pinned performance. |
| Full startup preload causes swap pressure | Separate maximum capacity from initial fill and permit lazy segment allocation/refill. |
| More Warm capacity hides, rather than fixes, a scheduling stall | Measure exposed GPU readiness wait and queue delay; do not use Hot hit rate as the sole success metric. |
| Current attention approximation contaminates placement conclusions | Keep all routing data in infrastructure-only status until CSA/HCA, compressed state, and reference parity gates pass. |

## 9. Deferred Decision Gates

After this plan closes, and only after the model-correctness work is complete:

1. Collect a verified, phase-labelled routing trace.
2. Compute finite-capacity Hot plus Warm oracle curves.
3. Compare current LRU with profile-seeded or other policies in a corrected finite-capacity replay.
4. Implement rolling layer candidates only if held-out replay lowers measured supply cost.

The external C++ simulator is not adopted unchanged. Its trace and reporting concepts may be reused later, but its finite Warm accounting, pending ownership, prefetch credit, and Belady semantics must be corrected before its results can gate runtime work.

## 10. Definition of Done

This plan is complete when:

1. The Warm registry has explicit, test-covered ownership and transfer transitions.
2. Configured Warm capacity can refill dynamically without a synchronous full-model preload.
3. Eligible Hot evictions attempt asynchronous Warm demotion by default.
4. Cold requests can still use transient staging without being forced through persistent Warm.
5. No slot or catalog invariant is violated under long synthetic and silicon traces.
6. Telemetry reports source-tier bytes, transfer service, exposed GPU wait, occupancy, queue pressure, and host pressure by phase.
7. Controlled A/B measurements show stable Warm service and lower Cold supply cost with identical output IDs.
8. The exact results are recorded in the ledger and progress trackers.
9. The next Phase 2 continuation is chosen from measurements rather than resumed automatically.
