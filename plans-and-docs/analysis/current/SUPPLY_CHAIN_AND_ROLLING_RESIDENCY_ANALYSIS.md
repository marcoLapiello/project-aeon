# Supply Chain and Rolling Residency Analysis

**Date:** 2026-09-11
**Status:** Current analysis and Stage 2 direction
**Scope:** Hot/Warm/Cold expert residency, registry behavior, dynamic allocation, and a future layer-aware rolling supply scheduler. This document records the brainstorming conclusions and recommended stages; it is not a claim that the proposed policy has been implemented or measured.

**Implementation handoff (2026-09-11):** The broad Phase 2 continuation remains paused. The focused [Warm-Tier Repair and Supply Telemetry Plan](../../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_PLAN.md) is complete; its [closure report](../../execution/completed/WARM_TIER_REPAIR_AND_SUPPLY_TELEMETRY_AB_REPORT.md) records the structural and silicon evidence. This document remains the technical rationale for later placement and rolling-residency decisions, not their active checklist.

## Executive conclusion

The current expert supply chain is functional as a residency mechanism, but it does not yet implement the caching strategy assumed by the project vision:

```text
round-robin initial placement
        +
global Hot LRU
        +
exclusive, draining Warm placement
        +
reactive Cold NVMe fallback
```

The runtime does record activation observations, but the recorded frequency fields do not control placement or eviction. The actual Hot policy is recency only. The Warm pool is initially populated, but a Warm-to-Hot promotion removes the expert from the logical Warm set and no normal refill path consumes the resulting free slot. Hot eviction sends the victim directly back to Cold.

The first supply-chain goal should therefore be correctness and scheduling of residency, not another expert-kernel rewrite:

1. make Hot and Warm ownership explicit and non-duplicated;
2. keep both pools populated through an intentional admission/refill policy;
3. measure per-layer routing probability on a representative corpus;
4. use those rankings to stage candidates for future layers;
5. gradually recycle capacity from completed layers toward a calibrated future-layer window;
6. measure exposed GPU wait, transfer service, and prefetch waste rather than relying on one aggregate hit rate.

The proposed future policy is called **deadline-aware rolling layer residency**.

## 1. What the current registry actually does

### 1.1 Runtime activation observations exist, but frequency is not a policy

Every routed expert request passes through `ExpertRegistry::touch_hot_expert()` from the layer prefetch path in [v4_pipeline.hpp](../../../src/architecture/deepseek_v4/core/v4_pipeline.hpp). The registry updates:

- `activation_count`;
- `last_step_used`;
- `moving_frequency`;
- Hot, Warm, and Cold service counters;
- the appropriate LRU list.

However, `activation_count`, `last_step_used`, and `moving_frequency` are never read by the admission or eviction code. The implementation in [expert_registry.hpp](../../../src/infrastructure/core/expert_registry.hpp) uses the LRU list for actual decisions.

The current frequency update is:

$$
f \leftarrow 0.9f + 0.1
$$

on every activation. It has no decay when an expert is not used, is not normalized by elapsed tokens, and is not separated into Prefill and Decode. Many frequently used experts will therefore approach `1.0`, while the value remains irrelevant to the current policy because no placement code consumes it.

The precise description is:

```text
runtime activation counters: yes
meaningful frequency estimator: no
working recency policy: yes
frequency-informed placement: no
```

The separate [routing counter](../../../src/infrastructure/core/routing_counter.hpp) collects phase-specific per-layer selection counts, but it is a profiling facility. It does not currently feed the runtime registry or cache policy.

### 1.2 LRU is real and independent of the frequency fields

The LRU behavior is not random:

1. if an expert is Hot, `touch_hot_expert()` moves its global ID to the front of `hot_vram_lru`;
2. when a Hot slot is needed and no free slot exists, `allocate_vram_slot()` removes the ID at the back of that list;
3. the incoming expert takes the evicted physical slot.

Therefore the current Hot victim is the least recently touched Hot expert. The fact that the weak frequency estimator saturates does not break the LRU, because LRU never reads that estimator.

This also means the registry does not need thousands of tokens before it starts adapting. Recency changes on the first request. What does require a representative run is a reliable frequency ranking, and that ranking does not currently exist in the runtime policy.

### 1.3 Short benchmarks can hide the residency history

`bench_full_model` runs a warmup generation, then resets the hit/miss counters without resetting the registry or repopulating either pool. The subsequent hit rate is therefore a post-warmup residency measurement, not a cold-start measurement.

A generation reset clears KV state but does not clear expert residency. Repeated generations on one pipeline inherit the previous Hot LRU and any remaining Warm contents.

Future measurements must label at least:

- cold-start state;
- post-warmup state;
- token or prompt position;
- Prefill versus Decode;
- Hot, Warm, and Cold service separately.

## 2. How allocation works today

### 2.1 Global capacity and active request size

The model has:

$$
43 \times 256 = 11{,}008
$$

routed experts, and each token selects six experts per layer:

$$
43 \times 6 = 258
$$

expert requests per full-model token before deduplication across layers.

At the usual 4096-context memory budget, the current report gives approximately `664` Hot VRAM slots. A Hot slot holds one complete expert payload, so the device pool does contain one current payload per occupied physical slot.

Because `664 > 258`, a normal single-token forward pass does not inherently require evicting every active expert from the previous token. The current system is not flushing the entire pool on every token. That is the important mitigating fact.

It is not a guarantee of good residency, however. The global pool has no per-layer reservation, deadline protection, or admission filter. A layer can lose all representation in the Hot pool if other layers' requests repeatedly become more recent and consume the shared slots. This risk increases for batched Prefill, where the number of distinct expert requests can greatly exceed the single-token working set.

### 2.2 Round-robin initialization is balanced by layer, arbitrary by expert usefulness

`populate_round_robin()` loops over expert ID first and layer ID second:

```text
for expert_id = 0 .. 255:
    for layer_id = 0 .. 42:
        assign the next available Hot slot
        then the next available Warm slot
        then stop when both pools are full
```

For `664` Hot slots:

$$
664 = 15 \times 43 + 19
$$

So initialization places expert IDs `0..14` in every layer, plus expert ID `15` in 19 layers. Each layer initially receives 15 or 16 Hot experts. It does **not** fill entire early layers before later layers.

For a larger Warm pool, assignment continues from the next unassigned `(expert_id, layer_id)` pair. The initial layout is therefore also approximately balanced by layer, but it is arbitrary with respect to actual activation probability. Expert ID is not a usefulness ranking.

This distinction matters:

- equal layer coverage protects against immediate layer starvation;
- low expert-ID selection does not maximize routing coverage;
- the current initialization does not use the observed or profiled distribution.

### 2.3 The global pool trades fairness for sharing, without enforcing either

A global VRAM pool means all layers share one LRU and one set of physical slots. It is not a fixed quota such as 15 slots permanently assigned to every layer.

Global sharing can be better than rigid quotas when workload demand is highly uneven. It can also be worse when a hot layer monopolizes capacity or when future-layer demand is ignored. The current implementation has no explicit fairness mechanism, so it provides neither a per-layer minimum nor a profile-aware global optimization.

A useful future policy must make the tradeoff explicit:

```text
minimum representation for every layer
+
frequency- or probability-weighted extra capacity
+
short-lived future-layer reservations
```

The current LRU does not provide those guarantees.

## 3. The Warm pool is currently a draining pool

### 3.1 Current transition model

The logical transition graph is:

```text
Warm -> Hot -> Cold
```

When a Warm expert is requested:

1. its Warm LRU entry is erased;
2. its `host_slots` entry is set to `-1`;
3. the host slot is appended to `free_host_slots`;
4. the expert receives a Hot slot;
5. its host bytes are no longer logically owned by the registry.

When a Hot expert is evicted:

1. it is removed from the Hot LRU;
2. its catalog entry becomes `COLD_NVME`;
3. its Hot slot is reused;
4. no Warm copy is created.

There is no normal runtime path that consumes `free_host_slots` to refill Warm. The [HostExpertPool](../../../src/infrastructure/core/host_expert_pool.hpp) allocates and exposes byte storage, but it does not decide which experts belong there or rotate entries.

The result is a logically shrinking Warm set. The allocation may still contain stale bytes from a promoted expert, but those bytes do not count as a Warm hit and must not be treated as valid ownership.

### 3.2 No logical Hot/Warm duplication is currently maintained

The current registry uses one `tier` and one `slot_idx` per expert. An expert is logically in exactly one of Hot, Warm, or Cold. That satisfies non-duplication, but only because promotion destroys its Warm residency.

This is not the useful hierarchy we want. A practical three-tier cache should keep the tiers disjoint while also keeping the Warm pool populated:

```text
one canonical immutable copy on NVMe
one logical resident owner in Hot, Warm, or Cold
no duplicate logical ownership
no empty Warm slots after steady state
```

### 3.3 Recommended replacement transition

The first correct dynamic hierarchy should use a disjoint swap:

```text
Warm request:
    requested Warm expert -> Hot
    selected Hot victim   -> freed Warm slot

Cold request:
    requested Cold expert -> Hot
    selected Hot victim   -> free Warm slot, if admitted
    selected Warm victim  -> Cold, if Warm is full
```

The Hot victim to Warm operation requires an asynchronous D2H copy. That traffic is a real cost and must be compared against the NVMe read avoided by preserving a useful Warm resident. The current demotion-free policy is cheaper per request but drains Warm; it is not a complete Hot/Warm hierarchy.

A later profile-informed policy may decide that some Hot victims should go directly to Cold rather than pay D2H. That is an admission decision, not an accidental side effect of promotion.

## 4. Proposed supply-chain direction: deadline-aware rolling layer residency

### 4.1 Core idea

The proposed strategy is to use the fact that a token visits the layers in a known order. While the GPU is processing the current layer `L`, the runtime can gradually recycle capacity from completed layers and stage likely candidates for future layers `L+H`.

The scheduler does not need to predict the exact six experts for the future layer. It can use a measured per-layer ranking and stage a candidate prefix:

```text
current layer: exact routed experts, highest priority
near future:   ranked candidate experts, asynchronous supply
past layers:   release or demote after their deadline has passed
```

This is the supply-chain form of the user's `L-10 ... L+10` idea. The direction is correct; the offset should be calibrated rather than hardcoded.

### 4.2 Why the offset is a timing parameter

Let:

- $T_{io}$ be the measured service time for the candidate's source tier, including queueing;
- $T_{pcie}$ be the host-to-device transfer time;
- $T_{margin}$ be a safety margin for variability;
- $T_{layer}$ be the compute time available per intervening layer.

A first horizon estimate is:

$$
H \approx \left\lceil
\frac{T_{io} + T_{pcie} + T_{margin}}{T_{layer}}
\right\rceil
$$

This is only a starting estimate. Multiple reads can run concurrently, so the real scheduler must also account for queue depth, staging capacity, total bytes, and the number of candidate experts per future layer.

Warm candidates need less lead time than Cold candidates. The scheduler should therefore reason in deadlines and transfer state, with the layer offset as a derived diagnostic:

```text
candidate deadline = time before the future layer consumes the slot
candidate state    = absent / read pending / host ready / H2D pending / ready
```

### 4.3 Candidate construction

For each layer and phase, the validated routing profile should produce an ordered list:

```text
layer L, phase P:
    expert e0, probability p0
    expert e1, probability p1
    ...
```

At runtime, a future-layer candidate set contains the highest-ranked experts that fit the available rolling budget. The actual router remains authoritative:

- a candidate hit avoids or shortens the request-time transfer;
- a candidate miss follows the normal fallback path;
- an unused candidate is counted as prefetch waste and released or demoted;
- candidate data never replaces the actual routed IDs.

This is safer than speculative router output because it does not change model decisions. It only changes which immutable payloads are supplied early.

### 4.4 Layer fairness is an explicit invariant

The rolling scheduler must not allow the global pool to erase an entire future or current layer. It should maintain:

1. a minimum active reservation for every layer that can be reached within the scheduling window;
2. exact residency for the current layer once routing resolves;
3. a bounded candidate budget for future layers;
4. reclaimable capacity from layers whose compute deadline has passed;
5. observable per-layer Hot and Warm counts.

A static profile baseline can allocate the Hot budget by measured value while preserving a minimum:

$$
\max_{K_0,\ldots,K_{42}}
\sum_l \sum_{e \in \operatorname{TopK}_l} p_l(e)
\quad\text{subject to}\quad
\sum_l K_l \le S_{Hot},\quad K_l \ge K_{min}
$$

The rolling policy then temporarily shifts the flexible portion of the budget toward upcoming layers.

## 5. Registry and data-model changes required

### 5.1 Separate Hot and Warm ownership

The current single `tier` plus `slot_idx` representation should be replaced or extended with independent ownership fields:

```text
hot_slot  = physical VRAM slot or -1
warm_slot = physical host slot or -1
```

The canonical NVMe copy always exists, but logical residency must remain disjoint:

```text
hot_slot >= 0  => warm_slot == -1
warm_slot >= 0 => hot_slot == -1
both negative  => Cold
```

The state machine should reject duplicate ownership and reject an occupied slot being assigned to a second expert.

### 5.2 Replace the weak online frequency field

Use one of these evidence-backed sources:

- offline per-layer, per-phase probability rankings from the validated routing profiler;
- a time-decayed online count with explicit token-window decay;
- a hybrid policy with an offline prior and bounded online adaptation.

A meaningful online score should be based on a known window or elapsed-token decay, for example:

$$
score_e(t) = score_e(t_0)\exp\left(-\frac{t-t_0}{\tau}\right) + w_e
$$

where $w_e$ is the current activation weight. The exact estimator is less important than making its time basis, layer scope, phase scope, and use in admission explicit.

### 5.3 Add residency leases and transfer states

A registry decision must not evict a payload that a queued GPU kernel will still consume. Each Hot and Warm entry should expose a short-lived lease or in-flight state covering:

```text
requested -> transfer pending -> resident -> consumed -> reclaimable
```

The scheduler must distinguish:

- logically resident;
- transfer pending;
- reserved for the current layer;
- speculative candidate;
- reclaimable after deadline.

### 5.4 Add supply-chain telemetry

Every request and candidate should contribute to counters for:

- Hot hits;
- Warm hits;
- Cold misses;
- candidate hits and misses;
- prefetch waste;
- NVMe bytes and read service time;
- H2D bytes and service time;
- D2H bytes caused by Warm refill;
- staging-slot wait;
- GPU compute wait for expert readiness;
- per-layer Hot/Warm resident counts;
- queue depth and deadline misses.

Hot-hit rate alone cannot distinguish a useful Warm hit from a cold miss that happened to overlap another kernel.

## 6. Recommended stages

### Stage 2.0 - Residency instrumentation and invariants

Keep the current policy unchanged and add a trustworthy observation surface:

- dump initial Hot/Warm assignment by layer;
- count logical occupied and free slots after every transition;
- record every promotion, eviction, and Warm hole;
- report per-layer resident counts;
- separate cold-start, warmup, Prefill, and Decode statistics;
- add unit tests for LRU ordering and slot ownership.

Acceptance gate:

```text
registry state is internally consistent after a long synthetic transition trace;
no duplicate logical Hot/Warm ownership;
all physical slot maps agree with catalog entries.
```

### Stage 2.1 - Repair Hot/Warm residency

The active implementation decision is persistent asynchronous Hot-to-Warm refill by default whenever Warm capacity is configured. The no-refill behavior is retained only as a diagnostic control and for `WARM=0`; it is not the intended production policy.

Implement the disjoint ownership model and an explicit Warm refill policy:

- promote Warm to Hot by swapping with a selected Hot victim;
- attempt to demote every eligible Hot victim into a valid Warm destination asynchronously;
- evict Warm victims to Cold without copying because the NVMe copy is canonical;
- keep Warm logically full whenever capacity and transfer policy allow;
- protect current-layer leases from eviction.

The request path must distinguish persistent Warm ownership from transient host staging. A Cold request may use `Cold NVMe -> transient staging -> Hot VRAM`; it must not be forced through persistent Warm merely because host memory is physically involved. Demotion may be dropped or deferred when no safe destination or transfer capacity exists, but it must never block the request path.

Acceptance gate:

```text
Warm capacity does not drain during a steady-state run;
no duplicate Hot/Warm experts exist;
Warm service counts remain measurable after repeated promotions.
```

Use demotion-free eviction only as the control comparison. The production decision is based on whether asynchronous refill maintains Warm occupancy and reduces Cold bytes and exposed wait without introducing a critical-path D2H dependency.

### Stage 2.2 - Build the evidence-backed placement input

Complete the routing correctness gate, then collect a representative profile and held-out corpus using the existing [routing profile study](../../execution/active/ROUTING_PROFILE_AND_PLACEMENT_STUDY.md).

Produce, separately for Prefill and Decode:

- per-layer expert probabilities;
- top-$M$ coverage curves;
- cumulative probability for candidate prefixes;
- cold-start and steady-state routing distributions;
- optional cross-layer co-activation statistics.

The existing short or repetitive pilot must not be treated as a placement prior.

Acceptance gate:

```text
profile and held-out traces are reproducible, phase-labeled, and tied to a verified model/text contract.
```

### Stage 2.3 - Offline policy replay

Before modifying the GPU runtime, replay the recorded traces against a simulator for:

- current global LRU;
- fixed per-layer quotas;
- static frequency-ranked placement;
- protected frequency set plus LRU victim slots;
- rolling layer-window candidates;
- disjoint Hot/Warm swap versus demotion-free Warm behavior.

The simulator should report resident state, Hot/Warm/Cold service, bytes moved, and deadline misses. This determines whether measured probability concentration is strong enough to justify rolling candidate staging.

Acceptance gate:

```text
an experimental policy must reduce modeled exposed transfer time or Cold bytes on held-out traces before runtime implementation.
```

### Stage 2.4 - Deadline-aware rolling supply scheduler

Add a reusable operation conceptually equivalent to:

```text
ensure_experts_resident(layer, expert_ids_or_candidates, phase, deadline)
```

It should:

- deduplicate requests;
- reserve current-layer payloads;
- submit batched Cold reads before waiting;
- use a staging arena sized for the selected horizon;
- enqueue H2D transfers asynchronously;
- stage ranked candidates for future layers;
- release completed-layer leases progressively;
- fall back safely when candidates miss;
- record candidate recall, waste, and deadline misses.

The initial horizon should be a measured sweep, not a fixed `L+10` assumption. Candidate width and horizon must be varied independently.

Acceptance gate:

```text
GPU wait for expert readiness decreases without output divergence,
unbounded staging growth, or pathological prefetch waste.
```

### Stage 2.5 - End-to-end supply-chain validation

Compare cold-start and steady-state runs using the same full-model inputs:

- the version-2 swizzled production path;
- Warm disabled, bounded Warm, and full Warm profiles;
- Prefill and Decode separately;
- short and diverse prompts;
- exact generated IDs and routing parity;
- NVMe, PCIe, D2H, and GPU-wait breakdowns.

The success criterion is not a target Hot-hit percentage. It is lower exposed supply latency per token with stable correctness and bounded resource use.

## 7. Relationship to Prefill and deferred work

The current `generate()` path calls single-token `step()` once per prompt token. `RoutingPhase::Prefill` labels the call but does not create a batched Prefill execution shape. The scratch buffers are padded for WMMA compatibility, while the active routed path remains `M=1`.

A future chunked Prefill path should reuse the same supply API, because a prompt chunk will request a larger union of experts and will otherwise pollute the Decode residency. The supply-chain work should therefore expose phase-aware admission and leases now, even if true batched Prefill remains a separate implementation stage.

MTP, speculative router execution, and another kernel rewrite are deferred until the supply measurements show that the runtime can keep the GPU supplied.

## 8. Final decision

The user's rolling layer-window idea is the most promising next supply-chain direction, with two constraints:

1. it must be driven by measured per-layer probabilities rather than an assumed exact expert prediction;
2. its offset must be calibrated from transfer deadlines, queue capacity, and layer compute time rather than fixed in advance.

The immediate implementation target is not `L+10`. It is a correct, observable residency state machine that can answer:

```text
which experts are resident,
which are being transferred,
which future candidates are worth staging,
which layer deadlines will be missed,
and how much GPU time was actually spent waiting.
```

Only after those answers are available should the rolling scheduler replace the current global LRU behavior.
