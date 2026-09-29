#pragma once

// -----------------------------------------------------------------------------
// The expert tier: the engine-owned state the tiered supply operates on.
//
// This is a pure state bundle, not a behaviour class. It groups the neutral
// machinery every architecture needs for tiered experts — the residency registry,
// the Warm host pool, the staging corridor and its pinned region, the Warm/staging
// partition, the supply telemetry, the routing-reuse profiler, and the direct
// (`O_DIRECT`) readers — so the storage half of the engine is one thing a model
// composes with rather than ten loose members inside a model host.
//
// It owns no model types and no format types. The one thing it deliberately does
// **not** own is the payload pool: how a payload is laid out in a VRAM slot is the
// weight *format*'s business (`UnifiedVRAMExpertPool` in the swizzled backend), so
// the pool stays with whoever composes the tier and is passed to the parts that
// need it (`ExpertTierLoader`, `HostPartition`). That keeps this header free of a
// backend include and the dependency one-way (backend → infrastructure).
//
// The behaviour over this state stays where it already lives and was already
// moved: `HostPartition` owns the boundary arithmetic, `PrefillController` and
// `TieredExpertSupply` the lifecycle, and `ExpertTierLoader` the bulk reads. This
// struct only owns the objects they are bound to.
// -----------------------------------------------------------------------------

#include "infrastructure/core/expert_direct_io.hpp"
#include "infrastructure/core/expert_host_region.hpp"
#include "infrastructure/core/expert_payload_pool.hpp"
#include "infrastructure/core/expert_registry.hpp"
#include "infrastructure/core/expert_tier_loader.hpp"
#include "infrastructure/core/host_expert_pool.hpp"
#include "infrastructure/core/host_partition.hpp"
#include "infrastructure/core/prefetch_staging.hpp"
#include "infrastructure/core/routing_reuse.hpp"
#include "infrastructure/core/supply_telemetry.hpp"

#include <memory>

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
};

} // namespace aeon::core
