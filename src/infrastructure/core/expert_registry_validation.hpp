#pragma once

// -----------------------------------------------------------------------------
// Expert registry validation — the full audit of the catalog and slot maps.
//
// This header holds the out-of-line definitions of the members that verify the
// registry's invariants (`invariants_hold`, `checked_validate`, the audit itself,
// and the per-tier LRU check); the class declares them in `expert_registry.hpp`.
//
// It is a pure relocation of those definitions: same bodies, same private access.
// -----------------------------------------------------------------------------

#include "infrastructure/core/expert_registry.hpp"

#include <cstddef>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <vector>

namespace aeon::core {

inline bool ExpertRegistry::invariants_hold() const noexcept {
        try {
            validate_invariants();
            return true;
        } catch (...) {
            return false;
        }
    }

// The per-request path's audit: a no-op unless `set_validate_each_request(true)`.
// It exists as a named function so the guarded sites read as a decision rather
// than as a missing call — every one of them *could* audit here, and does when
// the flag is on.
inline void ExpertRegistry::checked_validate() const {
        if (validate_each_request_) validate_invariants();
    }

// The audit itself. Called unconditionally at the boundaries and by
// `invariants_hold()`, and through `checked_validate()` on the per-request path.
inline void ExpertRegistry::validate_invariants() const {
        if (vram_slots.size() != vram_capacity || host_slots.size() != host_capacity ||
            vram_slot_reservations.size() != vram_capacity ||
            host_slot_reservations.size() != host_capacity) {
            throw std::logic_error("ExpertRegistry: physical slot vectors have inconsistent sizes");
        }

        std::vector<uint8_t> hot_seen(vram_capacity, 0);
        std::vector<uint8_t> warm_seen(host_capacity, 0);
        for (const auto& entry : catalog) {
            if (entry.owner == ExpertTier::HOT_VRAM) {
                if (entry.slot_idx < 0 || static_cast<size_t>(entry.slot_idx) >= vram_capacity ||
                    vram_slots[static_cast<size_t>(entry.slot_idx)] !=
                        static_cast<int32_t>(entry.global_expert_id)) {
                    throw std::logic_error("ExpertRegistry: Hot entry does not map to its VRAM slot");
                }
            } else if (entry.owner == ExpertTier::WARM_HOST) {
                if (entry.slot_idx < 0 ||
                    static_cast<size_t>(entry.slot_idx) >= host_capacity_usable ||
                    host_slots[static_cast<size_t>(entry.slot_idx)] !=
                        static_cast<int32_t>(entry.global_expert_id)) {
                    throw std::logic_error("ExpertRegistry: Warm entry does not map to its host slot");
                }
            } else if (entry.slot_idx != -1) {
                throw std::logic_error("ExpertRegistry: Cold entry owns a physical slot");
            }

            if (entry.operation != ExpertOperation::NONE && entry.operation_id == 0) {
                throw std::logic_error("ExpertRegistry: pending entry has no operation ID");
            }
            if (entry.operation == ExpertOperation::NONE &&
                entry.gpu_transfer != ExpertGpuTransfer::NONE) {
                throw std::logic_error("ExpertRegistry: idle entry has a pending GPU transfer");
            }
            if (entry.operation != ExpertOperation::NONE && entry.pending_slot_idx >= 0 &&
                (static_cast<size_t>(entry.pending_slot_idx) >= vram_capacity ||
                 vram_slot_reservations[static_cast<size_t>(entry.pending_slot_idx)] != entry.operation_id)) {
                throw std::logic_error("ExpertRegistry: pending entry does not own its VRAM reservation");
            }
        }

        for (uint32_t slot = 0; slot < vram_capacity; ++slot) {
            const int32_t gid = vram_slots[slot];
            if (gid >= 0) {
                if (static_cast<size_t>(gid) >= catalog.size()) {
                    throw std::logic_error("ExpertRegistry: VRAM slot names an unknown expert");
                }
                // A VRAM slot has two admissible owners: an ordinary Hot expert, or
                // a Warm expert's **shadow** copy taken during frozen prefill (Step 6
                // D-b). The map is still bijective under that disjunction — it is the
                // ownership model that is now explicit rather than the map that is
                // loosened.
                const auto& owner_entry = catalog[static_cast<size_t>(gid)];
                const bool is_hot_owner =
                    owner_entry.owner == ExpertTier::HOT_VRAM &&
                    owner_entry.slot_idx == static_cast<int32_t>(slot);
                const bool is_shadow =
                    owner_entry.owner == ExpertTier::WARM_HOST &&
                    owner_entry.shadow_vram_slot == static_cast<int32_t>(slot);
                if ((!is_hot_owner && !is_shadow) || hot_seen[slot]) {
                    throw std::logic_error("ExpertRegistry: VRAM slot map is not bijective");
                }
                hot_seen[slot] = 1;
            }
            if (vram_slot_reservations[slot] != 0 && gid < 0) {
                bool reservation_found = false;
                for (const auto& entry : catalog) {
                    if (entry.operation_id == vram_slot_reservations[slot] &&
                        entry.pending_slot_idx == static_cast<int32_t>(slot)) {
                        reservation_found = true;
                        break;
                    }
                }
                if (!reservation_found) {
                    throw std::logic_error("ExpertRegistry: orphaned VRAM reservation");
                }
            }
        }

        for (uint32_t slot = 0; slot < host_capacity; ++slot) {
            const int32_t gid = host_slots[slot];
            if (gid >= 0) {
                if (static_cast<size_t>(gid) >= catalog.size() ||
                    catalog[static_cast<size_t>(gid)].owner != ExpertTier::WARM_HOST ||
                    catalog[static_cast<size_t>(gid)].slot_idx != static_cast<int32_t>(slot) ||
                    warm_seen[slot]) {
                    throw std::logic_error("ExpertRegistry: host slot map is not bijective");
                }
                warm_seen[slot] = 1;
            }
            if (host_slot_reservations[slot] != 0) {
                bool reservation_found = false;
                for (const auto& entry : catalog) {
                    if (entry.operation_id == host_slot_reservations[slot] &&
                        (entry.slot_idx == static_cast<int32_t>(slot) ||
                         entry.operation == ExpertOperation::DEMOTION_PENDING)) {
                        reservation_found = true;
                        break;
                    }
                }
                if (!reservation_found) {
                    throw std::logic_error("ExpertRegistry: orphaned host reservation");
                }
            }
        }

        // Shadow residencies: a Warm expert's VRAM copy. The LRU and the per-expert
        // map must agree with the catalog and with the physical slot map, or a copy
        // would be unreclaimable (a leak) or reclaimed twice.
        {
            std::vector<uint8_t> shadow_seen(total_experts, 0);
            for (uint32_t gid : shadow_lru_) {
                if (gid >= total_experts || shadow_seen[gid]) {
                    throw std::logic_error("ExpertRegistry: shadow LRU contains an invalid expert");
                }
                const auto& entry = catalog[gid];
                if (entry.owner != ExpertTier::WARM_HOST || entry.shadow_vram_slot < 0 ||
                    static_cast<size_t>(entry.shadow_vram_slot) >= vram_capacity ||
                    vram_slots[static_cast<size_t>(entry.shadow_vram_slot)] !=
                        static_cast<int32_t>(gid)) {
                    throw std::logic_error("ExpertRegistry: shadow LRU disagrees with the catalog");
                }
                shadow_seen[gid] = 1;
            }
            for (const auto& entry : catalog) {
                if (entry.shadow_vram_slot >= 0 &&
                    shadow_seen[entry.global_expert_id] == 0) {
                    throw std::logic_error(
                        "ExpertRegistry: shadow residency is not in the shadow LRU");
                }
                if (entry.warm_shadow && entry.operation == ExpertOperation::NONE) {
                    throw std::logic_error(
                        "ExpertRegistry: an idle entry is flagged as a shadow copy in flight");
                }
                if (shadow_slot_of_expert_.size() == total_experts &&
                    shadow_slot_of_expert_[entry.global_expert_id] != entry.shadow_vram_slot) {
                    throw std::logic_error(
                        "ExpertRegistry: shadow slot map disagrees with the catalog");
                }
            }
        }

        // Single ownership is decode's rule, and the shadow residency is the one
        // exception to it — legal **only** while the Warm tier is frozen, which is to
        // say only during prefill. Making that an invariant rather than a convention
        // is what lets the prefill modes move slots around without either being able
        // to leak a second ownership into decode.
        if (!warm_frozen_ && !shadow_lru_.empty()) {
            throw std::logic_error(
                "ExpertRegistry: a shadow residency exists outside prefill streaming — "
                "decode admits single ownership only");
        }

        // The prefill-begin mark is scoped to a prefill: it is set only at
        // `begin_prefill_stream` and cleared only at `end_prefill_stream`, so it can
        // never be observed outside one, and while one is open it marks exactly the
        // Hot residents present at entry.
        if (!prefill_stream_) {
            for (const auto& entry : catalog) {
                if (entry.resident_at_prefill_begin) {
                    throw std::logic_error(
                        "ExpertRegistry: a prefill-begin residency mark exists outside "
                        "prefill streaming");
                }
            }
        }
        // A marked entry is a preserved **Hot** resident: the drain clears the mark,
        // so the mark can never outlive its residency.
        for (const auto& entry : catalog) {
            if (entry.resident_at_prefill_begin && entry.owner != ExpertTier::HOT_VRAM) {
                throw std::logic_error(
                    "ExpertRegistry: a prefill-begin residency mark on a non-Hot entry");
            }
        }

        validate_lru(hot_vram_lru, ExpertTier::HOT_VRAM);
        validate_lru(warm_host_lru, ExpertTier::WARM_HOST);
    }

inline void ExpertRegistry::validate_lru(
        const std::list<uint32_t>& lru,
        ExpertTier tier
    ) const {
        std::vector<uint8_t> seen(catalog.size(), 0);
        for (uint32_t gid : lru) {
            if (gid >= catalog.size() || seen[gid] || !catalog[gid].in_lru ||
                catalog[gid].owner != tier ||
                catalog[gid].publication != ExpertPublication::PUBLISHED ||
                catalog[gid].operation != ExpertOperation::NONE) {
                throw std::logic_error("ExpertRegistry: LRU contains an invalid catalog entry");
            }
            seen[gid] = 1;
        }
        for (const auto& entry : catalog) {
            if (entry.owner == tier && entry.in_lru != (seen[entry.global_expert_id] != 0)) {
                throw std::logic_error("ExpertRegistry: LRU membership disagrees with catalog entry");
            }
        }
    }

} // namespace aeon::core
