#pragma once

// -----------------------------------------------------------------------------
// The payload vocabulary `TieredExpertSupply` moves between tiers.
//
// These six structs were nested in the supply class; they are the shape of a
// request, its location in the artifact, and the transfer state that tracks it.
// They are defined at namespace scope here so the types can be named without
// pulling in the whole transfer lifecycle — the supply re-exports them under
// their established `TieredExpertSupply::X` spellings for existing callers.
//
// Nothing here moves bytes: the structs are data, and the lifecycle that fills
// them stays in `tiered_expert_supply.hpp`.
// -----------------------------------------------------------------------------

#include "infrastructure/core/expert_registry.hpp"
#include "infrastructure/core/supply_telemetry.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace aeon::core {

struct PayloadLocation {
    uint64_t file_offset{0};
    size_t byte_length{0};
};

struct PayloadSource {
    int direct_fd{-1};
    std::function<PayloadLocation(uint32_t)> locate;
    std::function<const uint8_t*(uint32_t)> host_payload;

    bool supports_direct_io() const noexcept {
        return direct_fd >= 0 && static_cast<bool>(locate);
    }
};

struct PayloadRequest {
    uint32_t global_expert_id{0};
    uint32_t staging_idx{0};
};

struct PayloadTransfer {
    uint32_t global_expert_id{0};
    uint64_t operation_id{0};
    int32_t vram_slot{-1};
    bool is_prefetched{false};
    uint32_t staging_idx{0};
    bool io_pending{false};
    // The reads have all landed in staging but the copy has not been enqueued.
    // Two reasons: the blocking path has not run yet, or the operation is
    // staged-only and no VRAM slot was free. Either way the bytes are safe in
    // staging and the copy is retried later, which is what makes the read leg and
    // the copy leg independently bounded.
    bool io_complete{false};
    uint64_t io_user_data{0};
    uint32_t io_request_count{0};
    // A deferred Warm hand-off: its bytes are already in host memory, so its staging
    // slot is `IO_COMPLETE` from the outset and the copy runs later, when
    // `attach_vram_destination` supplies a VRAM slot. `staging_ready` tells the copy
    // path not to run the read-completion transition, and `warm_host_slot` (when
    // pinned) is the upload source, so no payload is copied through the arena.
    bool staging_ready{false};
    int32_t warm_host_slot{-1};
    // The tier that answered this request. Carried on the transfer so a
    // caller can classify a whole layer's outcome (all-Hot / Warm / any-Cold)
    // without reconstructing it from the telemetry records.
    ExpertTier source_tier{ExpertTier::COLD_NVME};
};

struct PayloadBatch {
    std::vector<PayloadTransfer> transfers;
};

struct PendingTransfer {
    uint64_t operation_id{0};
    uint32_t global_expert_id{0};
    uint32_t demoted_expert_id{0};
    ExpertTier source_tier{ExpertTier::COLD_NVME};
    SupplyTelemetryPhase phase{SupplyTelemetryPhase::Decode};
    std::chrono::steady_clock::time_point h2d_enqueued_at{};
    hipEvent_t demotion_event{nullptr};
    hipEvent_t h2d_event{nullptr};
    uint32_t staging_idx{0};
    bool has_staging{false};
    bool h2d_submitted{false};
    int32_t demotion_source_slot{-1};
    int32_t demotion_destination_slot{-1};
    int32_t demotion_staging_idx{-1};
    bool demotion_uses_staging{false};
    bool demotion_submitted{false};
    int32_t h2d_source_slot{-1};
    int32_t h2d_destination_slot{-1};
    std::chrono::steady_clock::time_point io_submitted_at{};
    uint64_t nvme_read_service_ns{0};
    uint64_t nvme_completion_wait_ns{0};
    uint64_t staging_wait_ns{0};
    uint64_t staging_reuse_wait_ns{0};
    std::chrono::steady_clock::time_point staging_acquired_at{};
    std::chrono::steady_clock::time_point gpu_wait_started_at{};
    bool request_failed{false};
    std::string failure_reason;
};

} // namespace aeon::core
