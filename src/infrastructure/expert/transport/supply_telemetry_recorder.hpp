#pragma once

// -----------------------------------------------------------------------------
// The supply's telemetry *record-building*: the two derivations that decide what
// to record, as opposed to the pass-through calls that forward an already-computed
// value.
//
// The transfer lifecycle also calls `SupplyTelemetry` directly in ~18 places for
// demotion/transfer/timing events — those pass values the lifecycle already has,
// so wrapping them would be pure indirection. These two are different: they
// *derive* their arguments (a request's logical/physical byte classification, and
// the pinned/unpinned Warm occupancy), so they are the part worth naming and
// testing on their own.
//
// The recorder holds no state beyond the sink and the artifact's payload width, so
// the supply can drop its copy of that knowledge here.
// -----------------------------------------------------------------------------

#include "infrastructure/expert/residency/expert_registry.hpp"
#include "infrastructure/expert/storage/host_expert_pool.hpp"
#include "infrastructure/expert/transport/supply_telemetry.hpp"

#include <cstddef>
#include <cstdint>

namespace aeon::core {

class SupplyTelemetryRecorder {
public:
    void bind(SupplyTelemetry* telemetry, size_t expert_payload_bytes) {
        telemetry_ = telemetry;
        expert_payload_bytes_ = expert_payload_bytes;
    }

    // Classify a reservation into the record's logical/physical byte terms. A Hot
    // hit and a still-pending request move nothing, so their physical bytes are
    // zero; the logical bytes are the payload width unless the answer was Hot.
    void record_request(const ExpertRequestReservation& request) {
        const bool physical_transfer = request.kind != ExpertRequestKind::HOT_HIT &&
                                       request.kind != ExpertRequestKind::PENDING;
        const uint64_t logical_bytes = request.source_tier == ExpertTier::HOT_VRAM
            ? 0
            : expert_payload_bytes_;
        telemetry_->record_request(
            telemetry_->current_phase(),
            request.source_tier,
            logical_bytes,
            physical_transfer ? expert_payload_bytes_ : 0,
            request.source_tier == ExpertTier::COLD_NVME && physical_transfer
                ? expert_payload_bytes_ : 0,
            request.source_tier == ExpertTier::WARM_HOST && physical_transfer
                ? expert_payload_bytes_ : 0,
            request.source_tier == ExpertTier::COLD_NVME && physical_transfer
                ? expert_payload_bytes_ : 0
        );
    }

    // Sample the residency the request was served against: the two tiers' published
    // slots, the in-flight transfer/demotion counts, and the Warm pool's pinned and
    // unpinned byte split (a pinned slot is not reclaimable, so the split is what
    // says how much of the tier is exposed to host pressure).
    void observe_occupancy(ExpertTier source_tier,
                           const ExpertRegistry& registry,
                           const HostExpertPool* host_pool) {
        const uint64_t warm_pinned_bytes = host_pool
            ? static_cast<uint64_t>(host_pool->pinned_slot_count()) * host_pool->payload_bytes()
            : 0;
        const uint64_t warm_unpinned_bytes = host_pool
            ? static_cast<uint64_t>(host_pool->unpinned_slot_count()) * host_pool->payload_bytes()
            : 0;
        telemetry_->observe_occupancy(
            registry.published_hot_slots(),
            registry.published_warm_slots(),
            registry.pending_transfer_count(),
            registry.pending_demotion_count,
            warm_pinned_bytes,
            warm_unpinned_bytes,
            source_tier
        );
    }

private:
    SupplyTelemetry* telemetry_{nullptr};
    size_t expert_payload_bytes_{0};
};

} // namespace aeon::core
