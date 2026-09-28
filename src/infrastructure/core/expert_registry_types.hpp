#pragma once

// -----------------------------------------------------------------------------
// The expert registry's data vocabulary: the tier / operation / transfer enums,
// the demotion drop-reason name helper, and the catalog and reservation PODs.
//
// Split out of `expert_registry.hpp` so the vocabulary can be read and included
// on its own, without pulling in the registry class body. Nothing here depends
// on the registry; the registry depends on it.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <list>
#include <optional>

namespace aeon::core {

enum class ExpertTier : uint8_t {
    HOT_VRAM = 0,
    WARM_HOST = 1,
    COLD_NVME = 2
};

enum class ExpertOperation : uint8_t {
    NONE = 0,
    IO_PENDING = 1,
    PROMOTION_PENDING = 2,
    DEMOTION_PENDING = 3
};

enum class ExpertGpuTransfer : uint8_t {
    NONE = 0,
    H2D_PENDING = 1,
    D2H_PENDING = 2
};

enum class ExpertDemotionDropReason : uint8_t {
    NONE = 0,
    QUEUE_PRESSURE = 1,
    WARM_DESTINATION_UNAVAILABLE = 2
};

inline const char* expert_demotion_drop_reason_name(ExpertDemotionDropReason reason) {
    switch (reason) {
    case ExpertDemotionDropReason::QUEUE_PRESSURE: return "queue_pressure";
    case ExpertDemotionDropReason::WARM_DESTINATION_UNAVAILABLE:
        return "warm_destination_unavailable";
    case ExpertDemotionDropReason::NONE: return "unspecified";
    }
    return "unspecified";
}

enum class ExpertPublication : uint8_t {
    PUBLISHED = 0,
    UNPUBLISHED = 1
};

enum class ExpertSlotState : uint8_t {
    UNALLOCATED = 0,
    ACTIVE = 1,
    RECLAIMABLE = 2
};

enum class ExpertRequestKind : uint8_t {
    HOT_HIT = 0,
    WARM_PROMOTION = 1,
    COLD_MISS = 2,
    PENDING = 3,
    // A cold read reserved **without** a VRAM destination: the bytes land in the
    // staging arena and wait there until `attach_vram_destination` gives them a slot.
    // This is what lets the supply's read leg and copy leg be bounded separately
    // (a read is limited by staging, a copy by VRAM) instead of both by VRAM.
    COLD_STAGED = 4
};

struct ExpertCatalogEntry {
    uint32_t global_expert_id{0};
    uint16_t layer_id{0};
    uint16_t expert_id{0};

    ExpertTier owner{ExpertTier::COLD_NVME};
    int32_t slot_idx{-1};
    int32_t pending_slot_idx{-1};

    ExpertOperation operation{ExpertOperation::NONE};
    ExpertGpuTransfer gpu_transfer{ExpertGpuTransfer::NONE};
    ExpertPublication publication{ExpertPublication::PUBLISHED};
    ExpertSlotState slot_state{ExpertSlotState::UNALLOCATED};
    ExpertDemotionDropReason demotion_drop_reason{ExpertDemotionDropReason::NONE};
    uint64_t operation_id{0};
    uint32_t lease_count{0};
    bool in_lru{false};

    // Frozen prefill (Step 6 D-b): a Warm-owned expert can additionally hold a
    // **shadow** VRAM residency — a copy taken without transferring ownership, so
    // Warm's resident set survives a prefill. `warm_shadow` marks a copy in
    // flight; `shadow_vram_slot` is the extra VRAM slot once it lands. Both are
    // false/-1 outside frozen prefill, so decode's ownership model is untouched.
    int32_t shadow_vram_slot{-1};
    bool warm_shadow{false};

    // Prefill restore (the plan's §3): set on every Hot resident when a prefill
    // opens, cleared when that resident is drained (so a marked entry is exactly a
    // **preserved** resident) or when the prefill ends. A marked resident is spared
    // by the per-layer release, so only prefill-admitted residents are released and
    // the pre-prefill set survives the pass. Only a Hot resident can carry it: a Warm
    // shadow exists only during frozen prefill and every one is cleared at entry, so
    // there is no shadow side to mark.
    bool resident_at_prefill_begin{false};

    uint64_t activation_count{0};
    uint64_t last_step_used{0};
    float moving_frequency{0.0f};

    std::list<uint32_t>::iterator lru_it;
};

struct ExpertDemotionReservation {
    uint64_t operation_id{0};
    uint32_t victim_gid{0};
    uint32_t source_vram_slot{0};
    uint32_t destination_host_slot{0};
};

struct ExpertRequestReservation {
    ExpertRequestKind kind{ExpertRequestKind::COLD_MISS};
    ExpertTier source_tier{ExpertTier::COLD_NVME};
    uint32_t global_expert_id{0};
    uint64_t operation_id{0};
    int32_t vram_slot{-1};
    int32_t source_host_slot{-1};
    bool lease_acquired{false};
    std::optional<ExpertDemotionReservation> demotion;
};

} // namespace aeon::core
