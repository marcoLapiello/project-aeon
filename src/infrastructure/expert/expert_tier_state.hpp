#pragma once

// -----------------------------------------------------------------------------
// The expert tier: the engine-owned state the tiered supply operates on.
//
// This owns the neutral machinery every architecture needs for tiered experts —
// the residency registry, the Hot/Warm payload pools, the staging corridor and its
// pinned region, the Warm/staging partition, the supply telemetry, the routing
// reuse profiler and the direct (`O_DIRECT`) readers — and the `initialize` that
// builds them in load order, so a model composes a whole expert tier with one call
// instead of reproducing steps 10–12 of the assembly.
//
// It owns no model types and no format types. `initialize` is parameterised by the
// few model scalars it needs (`num_layers`, `experts_per_layer`,
// `experts_per_token`) plus the artifact's format descriptor, the memory budget and
// the artifact source. The one thing it deliberately does **not** own is the
// concrete payload pool: how a payload is laid out in a VRAM slot is the weight
// *format*'s business (`UnifiedVRAMExpertPool` in the swizzled backend), so the
// composer constructs the pool and sets it before `initialize`, keeping this header
// free of a backend include and the dependency one-way (backend → infrastructure).
//
// The behaviour over this state stays where it already lives: `HostPartition` owns
// the boundary arithmetic, `PrefillController` and `TieredExpertSupply` the
// lifecycle, and `ExpertTierLoader` the bulk reads. `initialize` only builds the
// objects they are bound to.
// -----------------------------------------------------------------------------

#include "infrastructure/artifact/aeon_loader.hpp"
#include "infrastructure/expert/transport/expert_direct_io.hpp"
#include "infrastructure/expert/storage/expert_format.hpp"
#include "infrastructure/expert/storage/expert_host_region.hpp"
#include "infrastructure/expert/storage/expert_payload_pool.hpp"
#include "infrastructure/expert/residency/expert_registry.hpp"
#include "infrastructure/expert/expert_tier_loader.hpp"
#include "infrastructure/expert/storage/host_expert_pool.hpp"
#include "infrastructure/expert/storage/host_partition.hpp"
#include "infrastructure/expert/layer_batch_supply.hpp"
#include "infrastructure/memory/memory_budget_report.hpp"
#include "infrastructure/expert/transport/prefetch_staging.hpp"
#include "infrastructure/prefill/prefill_sweep.hpp"
#include "infrastructure/routing/routing_reuse.hpp"
#include "infrastructure/memory/runtime_config.hpp"
#include "infrastructure/expert/transport/supply_telemetry.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace aeon::core {

struct ExpertTierState {
    // The Hot VRAM pool. Held through the neutral `ExpertPayloadPool`, because the
    // hot pool is an engine concept — it owns the slot allocation and every neutral
    // consumer takes this interface. The concrete pool a backend builds
    // (`UnifiedVRAMExpertPool`) only adds that format's *view* of a slot's bytes, so
    // it is constructed by whoever composes the tier and moved in here; naming it in
    // this header would make `infrastructure/` include a backend header.
    std::unique_ptr<ExpertPayloadPool> payload_pool;
    ExpertRegistry registry;
    HostExpertPool host_pool;
    std::unique_ptr<PrefetchStagingArena> staging;
    // The pinned region the Warm pool and the corridor share, when a host budget is
    // configured. Empty otherwise, and the corridor allocates its own memory.
    ExpertHostRegion host_region;
    // The Warm/staging partition. It owns the partition numbers and reaches the
    // region, arena, registry and pools as bound services (see `host_partition.hpp`).
    HostPartition host_partition;
    SupplyTelemetry telemetry;
    RoutingReuseProfiler reuse_profiler;
    // The direct (`O_DIRECT`) expert-fragment I/O: the reader, the in-flight
    // completion map and the request-id counter. See `expert_direct_io.hpp`.
    ExpertDirectIO direct_io;
    // The tier's bulk load and restore. See `expert_tier_loader.hpp`.
    ExpertTierLoader bulk_loader;
    // Demotions allowed in flight before the registry drops candidates. The tier
    // does not decide it — the model's routing width does (a layer dispatches
    // `experts_per_token` experts and each may evict a victim) — so the composer sets
    // it after configuring the supply. `bulk_loader` reads it at restore time, which
    // is why it is a member rather than a constructor argument.
    uint64_t demotion_queue_capacity{0};

    // Everything `initialize` builds from. The pointers outlive the call; the scalars
    // are the model's own expert geometry.
    struct Params {
        const ExpertFormatDescriptor* format{nullptr};
        const AeonRuntimeConfig* runtime{nullptr};
        const MemoryBudgetReport* budget{nullptr};
        const AeonModelLoader* source{nullptr};
        uint32_t num_layers{0};
        uint32_t experts_per_layer{0};
        uint32_t experts_per_token{0};
        // The global id of this tier's first layer. A stage over a layer range keeps
        // the registry local-indexed (its layer 0 is the stage's own first layer) and
        // adds this only when it addresses the artifact.
        uint32_t first_layer{0};
        hipStream_t compute{nullptr};
        // The partition refreshes the sweep's bank count from the arena it just built;
        // both are neutral types, supplied by the composer (created model-side, but
        // engine components).
        PrefillSweep* sweep{nullptr};
        LayerBatchSupply* supply{nullptr};
    };

    // The Hot pool, the pinned region and the staging
    // corridor, the Warm/staging partition, the residency registry, the direct reader
    // and the batched Hot/Warm preload — in the one order they must happen.
    //
    // The pool must already be set (the composer builds the backend's concrete pool);
    // this sizes it.
    void initialize(const Params& params) {
        if (payload_pool == nullptr) {
            throw std::invalid_argument(
                "ExpertTierState::initialize: the payload pool must be built first");
        }
        const ExpertFormatDescriptor& format = *params.format;
        const AeonRuntimeConfig& runtime = *params.runtime;
        const MemoryBudgetReport& budget = *params.budget;
        const uint32_t warm_slots = budget.warm_host_slots;

        // The Hot VRAM pool, the staging arena and the direct reader. The arena and the
        // reader are built for the *artifact's* format, not the backend's default, so a
        // payload's staging slot is the artifact's own `payload_bytes` wide.
        //
        // The arena's slot count is the decode shape unless a prefill chunk is
        // configured: a chunk issues up to `experts_per_token * C` deduplicated
        // transfers as one set, and each distinct expert needs its own slot in
        // transit. A chunk's deduplicated distinct set can never exceed the
        // **layer's** expert count, so the term is capped by the layer here — the same
        // ceiling the graph's guard checks against, which is why a legal wide chunk is
        // not refused.
        payload_pool->allocate(budget.hot_vram_slots, format);
        const uint32_t dedup_ceiling = std::min<uint32_t>(
            params.experts_per_token * std::max<uint32_t>(1, runtime.prefill_chunk),
            params.experts_per_layer);
        const StagingSlotCounts staging_counts = staging_slot_counts(
            runtime, params.experts_per_layer, params.experts_per_token);
        const uint32_t staging_slots = staging_counts.peak;

        // The host region: **one** pinned allocation shared by the Warm tier and the
        // corridor, cut by a boundary the phases move (`apply_host_partition`). Its
        // total is the configured host budget, so the RAM a run holds is the number
        // the user set. With no host budget there is no region and the corridor
        // allocates its own memory at its peak, the decode-only shape.
        const uint32_t region_slots = budget.host_region_slots;
        const bool region_active = region_slots > 0;
        if (region_active) {
            if (region_slots < staging_counts.peak) {
                throw std::runtime_error(
                    "ExpertTierState: the host region (" + std::to_string(region_slots) +
                    " slots) cannot hold the corridor's largest requirement (" +
                    std::to_string(staging_counts.peak) + " slots, a swept prefill)");
            }
            host_region.allocate(region_slots, format);
            // The corridor's **resting** size, and therefore the partition every phase
            // cuts back to, is decided by `HostPartition::configure`: with the sweep
            // enabled a window is a cleanly delimited phase, so the resting cut is
            // decode's banks and the boundary moves out for a window. With the sweep
            // **off** decode and a chunked window interleave with no phase boundary to
            // cut at, so every partition is the same and nothing ever moves.
            host_partition.configure(true, staging_counts, runtime.prefill_sweep, region_slots);
            this->staging = std::make_unique<PrefetchStagingArena>(
                host_region.slot_ptr(host_partition.current_warm_slots()),
                host_partition.staging_decode_slots(), format);
        } else {
            host_partition.configure(false, staging_counts, false, 0);
            this->staging = std::make_unique<PrefetchStagingArena>(format, staging_slots);
        }
        // Bind the partition's collaborators now that the arena and the region exist,
        // then derive the sweep's bank count from the arena it actually has — a depth
        // change moves the arena and this derivation together, so nothing can disagree.
        host_partition.bind(HostPartition::Services{
            &registry, &host_pool, this->staging.get(), &host_region,
            params.sweep, params.supply, const_cast<MemoryBudgetReport*>(&budget)});
        host_partition.refresh_sweep_banks(params.experts_per_layer);

        // The direct reader's ring must hold **every read that can be outstanding at
        // once**, not one layer's worth: with the decoupled corridor more than one
        // layer's reads can be in flight, bounded by the staging arena. Sizing it to a
        // single layer while two are outstanding over-subscribes the completion queue,
        // and `wait_for_completion` then stalls. So the ring follows the staging depth.
        size_t batch_requests = ExpertDirectIO::requests_per_fragment(format) * dedup_ceiling;
        if (runtime.prefill_sweep) {
            batch_requests = std::max<size_t>(
                batch_requests,
                ExpertDirectIO::requests_per_fragment(format) * static_cast<size_t>(staging_slots));
        }
        const uint32_t io_queue_depth = static_cast<uint32_t>(
            std::max<size_t>(64, batch_requests));
        direct_io.initialize(io_queue_depth, format.sector_size);

        // 11 — the registry. It decides residency for every request and saturates VRAM
        // at construction: every Hot slot is owned from the first token on, so the
        // production steady state (a cold miss must evict a resident) is the only state
        // that exists.
        registry.init(params.num_layers, params.experts_per_layer, budget.hot_vram_slots,
                      warm_slots, runtime.preload_warm_host);
        registry.set_validate_each_request(runtime.validate_registry_each_request);

        // The bulk I/O (load and restore) is neutral; bind it now that the registry,
        // pools and reader exist. `preload_*` runs below; the restore runs at a window
        // boundary, reading the demotion capacity through the pointer.
        bulk_loader.bind(ExpertTierLoader::Services{
            &registry, payload_pool.get(), &host_pool, &direct_io, params.source, &budget,
            params.compute, &demotion_queue_capacity, params.first_layer});

        // 12 — Hot, then Warm, filled by batched `O_DIRECT` reads. This is the only
        // step with real mass.
        bulk_loader.preload_hot(format);
        if (warm_slots > 0) {
            // The Warm pool is a **view** over the region's head when one is shared,
            // and an allocation of its own otherwise. Either way it is `warm_slots`
            // wide, the capacity the registry was initialised with.
            if (host_partition.active()) {
                host_pool.bind(host_region.base(), warm_slots, host_region.format(),
                               host_region.is_pinned());
            } else {
                host_pool.allocate(warm_slots, format);
            }
            if (runtime.preload_warm_host) {
                bulk_loader.preload_warm(format);
            }
        }
    }

    void free() noexcept {
        payload_pool.reset();
        host_pool.free();
        staging.reset();
        // The partition's bound staging pointer is now stale; clear it so nothing
        // reaches a freed arena before the next `initialize` re-binds it.
        host_partition.bind(HostPartition::Services{});
        direct_io.free();
        demotion_queue_capacity = 0;
    }
};

} // namespace aeon::core
