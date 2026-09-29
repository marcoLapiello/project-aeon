#pragma once

// -----------------------------------------------------------------------------
// The Warm/staging partition — the movable boundary of `ExpertHostRegion`.
//
// Warm and the transport corridor share one pinned region; the boundary between
// them moves at phase boundaries (`grow_host_capacity` / `shrink_host_capacity` /
// `release_host_tail`) rather than being fixed at load. This header holds the
// out-of-line definitions of that concern's `ExpertRegistry` members; the class
// declares them in `expert_registry.hpp`.
//
// It is a pure relocation of those definitions: same bodies, same private access,
// reached through the class declaration. See `expert_registry.hpp`.
// -----------------------------------------------------------------------------

#include "infrastructure/core/expert_registry.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::core {

// ---- the Warm/staging partition ------------------------------------------
//
// Warm and the transport corridor are one pinned region cut by a movable
// boundary (`ExpertHostRegion`): Warm takes the head, the arena the tail, and
// `set_host_capacity` is the cut. The two consumers want opposite shapes — decode
// wants `12` staging slots and every other byte as Warm residency, a swept prefill
// wants `blocks x E` staging and gives the rest back — so the boundary is moved at
// the phase boundaries rather than fixed at load.
//
// Neither move relocates storage: the region is allocated once, so a boundary
// move edits the free list and the arena's base pointer.

// How many region slots the Warm tier may use right now.
inline uint32_t ExpertRegistry::usable_host_capacity() const noexcept { return host_capacity_usable; }

// The Warm residents a boundary move surrendered to the corridor, in the order
// they were released. The caller re-admits them at `prefill_end` so a window
// leaves the Warm tier **as it found it** — the frozen-prefill guarantee, and the
// reason a move is a borrow rather than a loss. Cleared by `clear_host_restore_set`
// once they have been restored.
inline const std::vector<uint32_t>& ExpertRegistry::host_restore_set() const noexcept { return host_restore_set_; }

inline void ExpertRegistry::clear_host_restore_set() noexcept { host_restore_set_.clear(); }

// Take a free region slot for a Warm re-admission, or -1 when none is free.
inline int32_t ExpertRegistry::take_free_host_slot() {
    if (free_host_slots.empty()) return -1;
    const int32_t slot = static_cast<int32_t>(free_host_slots.back());
    free_host_slots.pop_back();
    return slot;
}

// Give `gid` Warm ownership of `slot`. The caller has already filled the slot's
// payload, so this is the bookkeeping half of a re-admission: owner, slot map, and
// LRU. Rejects an expert that is not Cold — an entry that still holds a residency
// would end up with two, which is the one thing the tier model forbids.
inline void ExpertRegistry::admit_warm(uint32_t gid, uint32_t slot) {
        if (gid >= catalog.size()) {
            throw std::out_of_range("ExpertRegistry: re-admitted expert is out of range");
        }
        if (slot >= host_capacity_usable) {
            throw std::out_of_range("ExpertRegistry: re-admission slot is outside Warm");
        }
        auto& entry = catalog[gid];
        if (entry.owner != ExpertTier::COLD_NVME || entry.slot_idx != -1) {
            throw std::logic_error(
                "ExpertRegistry: re-admitting expert " + std::to_string(gid) +
                " that still holds a residency (owner=" +
                std::to_string(static_cast<int>(entry.owner)) +
                ", slot=" + std::to_string(entry.slot_idx) + ")");
        }
        if (host_slots[slot] >= 0 || host_slot_reservations[slot] != 0) {
            throw std::logic_error("ExpertRegistry: re-admission slot is already in use");
        }
        entry.owner = ExpertTier::WARM_HOST;
        entry.slot_idx = static_cast<int32_t>(slot);
        entry.slot_state = ExpertSlotState::ACTIVE;
        host_slots[slot] = static_cast<int32_t>(gid);
        push_lru_front(entry, warm_host_lru);
        checked_validate();
    }

// Move the boundary **upward**: return slots to the Warm free list. Callers do
// this when a prefill window ends. Slots that were never surrendered are already
// in the list, so only the newly recovered range is added.
inline void ExpertRegistry::grow_host_capacity(uint32_t usable) {
        if (usable <= host_capacity_usable) return;
        if (usable > host_capacity) {
            throw std::out_of_range(
                "ExpertRegistry: host capacity exceeds the allocated region");
        }
        for (uint32_t slot = host_capacity_usable; slot < usable; ++slot) {
            if (host_slots[slot] >= 0 || host_slot_reservations[slot] != 0) {
                throw std::logic_error(
                    "ExpertRegistry: cannot grow Warm capacity over a slot still in use");
            }
            free_host_slots.push_back(slot);
        }
        host_capacity_usable = usable;
        checked_validate();
    }

// Move the boundary **downward**: `[usable, host_capacity_usable)` is surrendered.
// The caller must have drained it first — `release_host_tail` is the drain — so an
// occupied slot here is a caller defect, refused rather than silently orphaned.
// The surrendered range leaves the free list, so nothing can be admitted into it.
inline void ExpertRegistry::shrink_host_capacity(uint32_t usable) {
        if (usable >= host_capacity_usable) return;
        for (uint32_t slot = usable; slot < host_capacity_usable; ++slot) {
            if (host_slots[slot] >= 0 || host_slot_reservations[slot] != 0) {
                throw std::logic_error(
                    "ExpertRegistry: cannot shrink Warm capacity under a slot still in use");
            }
        }
        host_capacity_usable = usable;
        rebuild_free_host_slots();
        checked_validate();
    }

// Demote every Warm resident in `[keep, host_capacity_usable)` to Cold and free its
// slot: ownership is dropped and the slot returned to service as staging. **No data
// moves** — the payload is still in the container, so the expert is re-read on
// demand. This is the drain half of a boundary move, and it is what lets decode
// reclaim the corridor's share of the region.
//
// Requires a boundary state (no live lease, no in-flight transfer), which the
// caller reaches exactly as `begin_prefill_stream` does. Returns how many experts
// were demoted, so a caller can report the cost of the trade it just made.
inline uint32_t ExpertRegistry::release_host_tail(uint32_t keep) {
        if (keep > host_capacity_usable) {
            throw std::out_of_range("ExpertRegistry: host tail keep-point is out of range");
        }
        uint32_t released = 0;
        for (uint32_t slot = keep; slot < host_capacity_usable; ++slot) {
            const int32_t gid = host_slots[slot];
            if (gid < 0) continue;
            auto& entry = catalog[static_cast<size_t>(gid)];
            if (entry.owner != ExpertTier::WARM_HOST) {
                throw std::logic_error("ExpertRegistry: host tail slot is not Warm-owned");
            }
            if (entry.lease_count != 0 || entry.operation != ExpertOperation::NONE) {
                throw std::logic_error(
                    "ExpertRegistry: cannot surrender a Warm slot with a live lease or "
                    "transfer — the caller must reach a boundary first");
            }
            if (entry.in_lru) {
                remove_from_lru(entry, warm_host_lru, static_cast<uint32_t>(gid));
            }
            host_slots[slot] = -1;
            entry.owner = ExpertTier::COLD_NVME;
            entry.slot_idx = -1;
            entry.slot_state = ExpertSlotState::UNALLOCATED;
            // Remembered so the caller can put them back: the corridor's borrow must
            // not cost Warm its cache, only the re-read.
            host_restore_set_.push_back(static_cast<uint32_t>(gid));
            ++released;
        }
        checked_validate();
        return released;
    }

// A free region slot reserved for a demotion's Warm destination, or -1. The first
// pass takes an unused slot; the second accepts a slot that is reserved but whose
// expert has not landed yet. Also here because it is host-slot bookkeeping that
// belongs with the partition's other slot arithmetic.
inline int32_t ExpertRegistry::find_reserved_host_slot(uint64_t operation_id) const {
        for (uint32_t slot = 0; slot < host_capacity; ++slot) {
            if (host_slot_reservations[slot] == operation_id && host_slots[slot] < 0) {
                return static_cast<int32_t>(slot);
            }
        }
        for (uint32_t slot = 0; slot < host_capacity; ++slot) {
            if (host_slot_reservations[slot] == operation_id) {
                return static_cast<int32_t>(slot);
            }
        }
        return -1;
    }

inline void ExpertRegistry::release_reserved_host_slot(uint32_t slot) {
        host_slots[slot] = -1;
        host_slot_reservations[slot] = 0;
        free_host_slots.push_back(slot);
    }

// Rebuild the free list to exactly the usable prefix, preserving every slot's
// occupancy. Used after a boundary move, where the set of admissible slots
// changes wholesale and an incremental edit would be easy to get wrong.
inline void ExpertRegistry::rebuild_free_host_slots() {
        free_host_slots.clear();
        for (uint32_t slot = host_capacity_usable; slot-- > 0;) {
            if (host_slots[slot] < 0 && host_slot_reservations[slot] == 0) {
                free_host_slots.push_back(slot);
            }
        }
    }

} // namespace aeon::core
