#pragma once

// -----------------------------------------------------------------------------
// The runtime configuration knobs and the VRAM/host allowances derived from them.
//
// This is the *input* half of the memory budget: the user-facing settings
// (`AeonRuntimeConfig`) plus the fixed safety margins and the scratch allowances
// the budget engine and the model host both read. It is deliberately free of any
// model-config type, so it can be included by the budget report and the engine
// without a cycle.
//
// The staging slot counts live here because they are a pure function of the knob
// set: the corridor's three dispatch requirements are derived from
// `prefill_chunk`, `prefill_sweep`, and `prefill_sweep_staging_blocks`, and both
// the budget report and the arena's own construction read the same derivation so
// the reported figures are exactly what is allocated.
// -----------------------------------------------------------------------------

#include "infrastructure/core/prefetch_staging.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace aeon::core {

// Constant safety margins & architectural parameters
constexpr size_t VRAM_HEADROOM_SAFETY_BYTES = 300ULL * 1024ULL * 1024ULL; // 300 MB
// Host RAM the engine will never plan to use, held back for the OS and for this
// process's own non-expert footprint (the graph, the tokenizer, the pinned
// transport staging arena, and whatever the dense container still holds in page
// cache).
//
// It replaces a percentage cap (`0.70 * total`, i.e. 43.84 GiB here), which had
// two defects: it never accounted for memory *already in use*, so a 43.84 GiB
// Warm allocation on a machine with 30 GiB resident would swap; and it moved with
// the machine's RAM size rather than with what the process needs. A fixed reserve
// is the honest statement of "the engine may have everything else".
//
// A previous revision of this comment justified the reserve with "the ~13 GiB of
// resident dense weights that the embedding lookup and every oracle read touch
// through the mmap". That was wrong twice over. Only `embed.weight` (0.99 GiB) is
// touched per token — every other dense tensor is uploaded once and never read
// again — and those pages are now released outright after the uploads
// (`AeonModelLoader::release_dense_pages_except`, driven by
// `release_dense_pages_after_upload`). The reserve is not sized around dense
// residency, and it never was.
constexpr size_t HOST_RAM_RESERVED_BYTES = 10ULL * 1024ULL * 1024ULL * 1024ULL; // 10 GiB

// VRAM allowance for the body's batch scratch (`V4LayerBodyBatchScratch`), derived
// from the configured prefill chunk `C`.
//
// It is an **allowance**, not a computed size: the exact figure depends on each
// layer's state layout (the indexer's candidate scores and top-k, the composed
// row-set), so the host allocates the real buffer at load and checks it against this
// number, failing with a named message rather than silently eating into the Hot
// pool. The per-row figure is `1 MiB`, against a measured `~0.69 MiB/row` (a `C = 64`
// scratch is `44 MiB`), which leaves the allowance ~45% above the real cost and is
// what makes it safe to state without walking the layer layouts here.
constexpr size_t BATCH_SCRATCH_BYTES_PER_ROW = 1ULL * 1024ULL * 1024ULL;

inline size_t batch_scratch_allowance_bytes(uint32_t chunk_tokens) {
    return static_cast<size_t>(chunk_tokens) * BATCH_SCRATCH_BYTES_PER_ROW;
}

// VRAM allowance for the **decode** path's fixed workspace: `V4ActivationScratch`
// (the per-op temporaries, sized by the kernel shapes) plus `V4RoutedExpertScratch`
// (six experts' accumulators). Neither is a function of a knob — both are fixed by the
// model — so this is an allowance in the same sense as the batch scratch: the host
// allocates the real buffers at load and checks them against it, so the figure is
// verified rather than trusted. Measured, the two together are ≈`5 MiB`; the margin
// covers a kernel whose padded row count grows.
constexpr size_t DECODE_SCRATCH_ALLOWANCE_BYTES = 16ULL * 1024ULL * 1024ULL;

inline size_t decode_scratch_allowance_bytes() {
    return DECODE_SCRATCH_ALLOWANCE_BYTES;
}


struct AeonRuntimeConfig {
    // User-configurable: target context sequence length (tokens)
    uint32_t context_size{4096};

    // The **total** pinned host RAM for expert payloads: Warm residency *plus* the
    // transport corridor, which share one region cut by a boundary that moves per
    // phase (`ExpertHostRegion`). Zero disables Warm ownership and the corridor
    // becomes the only tenant. Because this is the total, it is also the guard: the
    // host RAM a run holds is exactly this figure, not this figure plus a staging
    // allocation the user had to remember to add.
    //
    // The partition inside it: **decode** gives the corridor `2 x 6` slots (the
    // double buffer) and keeps everything else for Warm; a **routed** prefill needs
    // the layer's deduplicated distinct set, `min(6C, E)`; a **swept** prefill grows
    // the corridor to `prefill_sweep_staging_blocks x E` and hands the difference back
    // when the window ends. So Warm residency is largest during decode — where its
    // cache value is — and smallest during a sweep.
    size_t warm_host_bytes{0};

    // Allocate the configured Warm capacity without requiring a synchronous
    // startup fill. This remains enabled by default for compatibility.
    bool preload_warm_host{true};

    // After the dense uploads finish, release this process's residency of the
    // container's pages, keeping only `embed.weight`. The pages are clean and
    // file-backed, so the kernel may reclaim them anyway; releasing them
    // deterministically keeps the host footprint from depending on when the
    // reclaim happens to run, which matters because the Warm pool is pinned and
    // cannot be reclaimed at all. Zero VRAM cost. Set false to keep every dense
    // page resident, which is what a caller reading the container host-side after
    // initialization wants.
    bool release_dense_pages_after_upload{true};

    // Diagnostic A/B control. The production default keeps asynchronous refill enabled.
    bool enable_warm_refill{true};

    // Supply telemetry sink. Empty path disables recording entirely (the default);
    // a non-empty path opens a JSONL stream that `TieredExpertSupply` writes its
    // request, timing, occupancy, and demotion records into. `run_id` is stamped on
    // every row so several runs can share or be told apart in one directory.
    //
    // Only supply-mediated traffic is captured: the Hot and Warm startup preloads
    // read through `read_experts_direct_blocking`, not through the supply, so the
    // Warmup phase stays empty by construction.
    std::string supply_telemetry_path;
    std::string run_id{"unnamed"};

    // Diagnostic pressure knob. Zero (the default) leaves the Hot pool at the
    // derived size. A non-zero value caps it at `min(derived, value)`, floored at
    // 6 (one layer's routed experts) so the graph stays runnable. It exists so a
    // gate can force the starved-pool path — the emergency drain in
    // `ensure_pool_headroom` and eviction-under-lease-pressure — which the derived
    // size never reaches on this GPU (779 slots against an ≈264-slot arm point).
    uint32_t max_hot_vram_slots{0};

    // Demotion-queue capacity override. Zero (the default) derives it from
    // `enable_warm_refill` — the default capacity when refill is on, 0 when off.
    // A positive value sets it directly, which is how the Step 5 A/B varies the
    // number of evictions allowed in flight before the rest are dropped with
    // `queue_pressure`.
    uint64_t demotion_queue_capacity{0};

    // Phase 1 of the routing study: profile decode routing reuse distances. Off by
    // default; when on, the host enables the profiler and `aeon_chat` prints an
    // ideal-LRU hit-rate curve next to the measured Hot hit rate.
    bool profile_routing_reuse{false};

    // Run the registry's full invariant audit after **every** expert reservation and
    // completion, not only at the phase boundaries. Off by default: the audit is a
    // whole-registry scan, so per-request it dominates a batch (measured 10.2 s of a
    // 338-token swept prefill, ledger M43). On, it localises a bookkeeping defect to
    // the individual operation that caused it. The boundaries and `invariants_hold()`
    // always audit regardless.
    bool validate_registry_each_request{false};

    // The body chunk `C` the layer-major prefill window runs with — the rows in
    // flight per body invocation (Step 6 §6b). User-configurable; validated against
    // the body's own row cap (`V4LayerBodyBatchScratch::kMaxTokens`) and against the
    // window. It bounds the batch scratch (allocated at load) and the staging arena
    // (`min(6C, experts_per_layer)`). Larger = fewer body invocations; measured
    // nearly flat in throughput, so the default is on the wide side.
    uint32_t prefill_chunk{64};

    // The layer-major prefill **window** `W` (Step 6 §6b): how many prompt tokens
    // one layer-major pass carries in its residual, and therefore how many sweeps a
    // prompt of `N` tokens pays (`⌈N/W⌉`). User-configurable. `0` means the whole
    // prompt, bounded by the context — one pass, but a carry allocated for the whole
    // context. The default matches the reference implementation's segment size
    // (colibri `V4_PREFILL_SEGMENT = 4096`), which bounds the carry at ~393 MiB
    // regardless of context and keeps a long prompt at `⌈N/4096⌉` passes.
    uint32_t prefill_window{4096};

    // Step 6 D-b (policy A): freeze the Warm tier during prefill. A prefill touches
    // every expert, so letting the sweep promote from Warm would **move** each
    // Warm-resident expert into VRAM and empty the tier — destroying exactly the
    // "natural selection" decode's Warm hits depend on. With this on, a prefill
    // copies a Warm expert into VRAM **without** transferring ownership (a shadow
    // residency) and evicts by **release** rather than demotion, so Warm's resident
    // set is identical before and after the prefill. On by default, per D-b; a gate
    // A/Bs it against `false`.
    bool freeze_warm_during_prefill{true};

    // Step 6 item 6: drive the layer-major prefill window with the **expert sweep**
    // instead of the per-token dispatch. The sweep frees only what it needs on
    // entry, holds a sliding window of whole layer sets (`L, L+1, L+2, …` up to
    // capacity), releases each layer's prefill-admitted set as it retires, and
    // leaves the residents preserved at entry resident on exit — so Warm and its LRU
    // ranking are untouched across the whole prefill and decode resumes on them. On
    // by default; it is the prefill strategy the plan decided (D-a). Falls back to
    // the per-token path when the Hot pool cannot hold a whole layer.
    bool prefill_sweep{true};

    // The **prompt-length gate** for the sweep (Prefill Supply Strategy plan, Step
    // 3). Below this window length the layer-major window runs the route-aware
    // cached supply instead of the sweep: a whole-layer load over-reads a short
    // prompt's small distinct set, so the sweep is the right strategy only for long
    // ones. Zero (the default) derives the gate from the layer width (`3 E / 4`),
    // placed at the measured crossover, so it is expressed in the model's own terms
    // rather than as a constant tuned for one GPU. This is the **only** condition on
    // the switch besides feasibility — a hidden threshold is exactly what the plan
    // forbids.
    uint32_t prefill_sweep_min_tokens{0};

    // The layer-sized staging **banks** the sweep's arena holds when `prefill_sweep`
    // is on, in units of one layer's payloads (`E`). **A memory budget, not a tuning
    // figure**: the prefetch depth is *derived* from the arena (`blanks - 1` blocks of
    // free staging, bounded to one read wave at a time), so the count decides how much
    // pinned host memory the corridor spends and nothing else. Measured across `2E…
    // 6E`: `29.83–30.39 s`, Gate A spread `1.009x`, `42/43` layer-bodies overlapped at
    // every size (SUPPLY_CHAIN_HOT_PATH_ANALYSIS §17.4).
    //
    // This is why the knob is legitimate where an earlier revision removed it: the
    // objection then was that depth selected the *algorithm* (Gate A failed at
    // `1.38x`, `1E` ran a serial read/copy pipeline). It does not any more, so the
    // arena can be sized from the host's remaining RAM, which is exactly what a
    // portability knob should express.
    //
    // `1` is the minimum and is supported (a single bank, the blocking drain); below
    // one layer the swept dispatch cannot bind a whole layer's set, so smaller values
    // are refused at load. Each block is `E x payload_bytes` of **non-reclaimable
    // pinned** host memory — `3.44 GiB` per block at `E = 256`.
    uint32_t prefill_sweep_staging_blocks{2};
};

// The staging arena's slot count, shared by the budget report and the arena's own
// construction so the reported figures are exactly what is allocated.
//
// There are **three** distinct corridor requirements, because there are three
// dispatch paths and each binds a different number of staging indices at once:
//
//   * **decode** — `dispatch_layer_prefetch` stages one token's `experts_per_token`
//     routed experts into two banks, so it needs `decode_slot_count(experts_per_token)`
//     (the double buffer) and nothing more. This is the certified decode shape.
//   * **routed batched prefill** — `dispatch_layer_prefetch_batch` stages the layer's
//     deduplicated distinct set into `0 .. D-1` with `D <= min(experts_per_token * C, E)`,
//     so it needs `min(experts_per_token * C, E)` slots — one layer's worth at any
//     useful chunk.
//   * **swept batched prefill** — `dispatch_layer_stream` stages a whole layer into
//     `(layer % banks) * E` and keeps `banks` of them live, so it needs
//     `blocks x E` slots; and it needs **at least one whole layer** regardless of the
//     block count, because a swept dispatch binds the layer's entire missing set.
//
// The corridor is sized to its largest requirement, and the Warm/staging boundary is
// cut to the requirement of the phase actually running (`ExpertHostRegion`), so the
// two smaller phases hand the difference back to Warm residency — which is where
// decode's NVMe hits are decided, and where a mistyped `min(experts_per_token * C, E)`
// was costing a full layer's worth of residency for no reason.
struct StagingSlotCounts {
    // Decode: one token's routed experts, double-buffered.
    uint32_t decode{0};
    // A chunked (routed) prefill: the layer's deduplicated distinct set.
    uint32_t batch{0};
    // A swept prefill: `blocks` whole layers.
    uint32_t prefill{0};
    // The largest of the three — the corridor's allocation, and its size when no host
    // region is shared.
    uint32_t peak{0};
};

inline StagingSlotCounts staging_slot_counts(
    const AeonRuntimeConfig& cfg,
    uint32_t experts_per_layer,
    uint32_t experts_per_token
) {
    StagingSlotCounts counts;
    counts.decode = PrefetchStagingArena::decode_slot_count(experts_per_token);
    const uint32_t chunk_ceiling = std::min<uint32_t>(
        experts_per_token * std::max<uint32_t>(1, cfg.prefill_chunk), experts_per_layer);
    counts.batch = std::max<uint32_t>(counts.decode, chunk_ceiling);
    // A swept dispatch binds a whole layer, so the prefill size is at least one layer
    // even at `blocks = 0`; at least two by default, one to read into and one to copy
    // out of.
    const uint32_t blocks = cfg.prefill_sweep
        ? std::max<uint32_t>(1, cfg.prefill_sweep_staging_blocks)
        : 1;
    counts.prefill = cfg.prefill_sweep
        ? std::max<uint32_t>(counts.batch, blocks * experts_per_layer)
        : counts.batch;
    counts.peak = std::max({counts.decode, counts.batch, counts.prefill});
    return counts;
}

// The corridor's allocation size — its largest requirement, shared by the budget
// report and the artifact's own arena construction.
inline uint32_t staging_slot_count(const AeonRuntimeConfig& cfg, uint32_t experts_per_layer,
                                   uint32_t experts_per_token) {
    return staging_slot_counts(cfg, experts_per_layer, experts_per_token).peak;
}

} // namespace aeon::core
