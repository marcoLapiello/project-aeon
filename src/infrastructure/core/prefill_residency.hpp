#pragma once

// -----------------------------------------------------------------------------
// Prefill residency — the streaming mode and the frozen-prefill shadow copies.
//
// Prefill and decode are two different allocation strategies, not one strategy
// with a parameter. This header holds the out-of-line definitions of the members
// that switch between them and that manage the non-destructive Warm "shadow"
// residency taken during frozen prefill; the class declares them in
// `expert_registry.hpp`.
//
// It is a pure relocation of those definitions: same bodies, same private access.
// -----------------------------------------------------------------------------

#include "infrastructure/core/expert_registry.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::core {

// --- Prefill streaming (Step 6 item 6) ------------------------------------
//
// Prefill and decode are two **different allocation strategies**, not one
// strategy with a parameter. Decode keeps a persistent, LRU-ranked resident set
// and demotes its evictions into Warm so that "natural selection" accumulates.
// Prefill streams the whole model through VRAM once per window in layer order:
// at any moment it holds a sliding window of whole layer sets, it allocates
// only from the free list, and it retires each layer exactly when that layer is
// done. Crossing between them is therefore a **hard switch**:
//
//   begin_prefill_stream()  marks every Hot resident present at entry, then
//                           drains only the worst-LRU residents the pass needs
//                           (up to `drain_slots`) into a restore set; the marked
//                           remainder is **preserved**. Warm and its LRU ranking
//                           are untouched for the whole pass;
//   release_layer(L)        returns one computed layer's prefill-admitted set
//                           (its Hot copies *and* its Warm shadows) to the free
//                           list, **sparing** any resident marked at entry;
//   end_prefill_stream()    requires that no prefill-admitted resident is left,
//                           which the per-layer release guarantees by
//                           construction, and clears the mode so decode resumes
//                           on the preserved set plus the caller's reload of
//                           `restore_set()`.
//
// Both modes are still expressed over the same catalog, so `invariants_hold()`
// checks the switch as tightly as it checks a decode step.

// Drain Hot and enter streaming. The caller must first have reached a
// compute-stream boundary and reaped the registry — a live lease or an in-flight
// transfer is a caller defect, not something to drain around.
// `drain_slots` bounds how much of the Hot pool is freed for the pass: the
// **worst-LRU** residents are drained first (the most-recently-used are the ones
// worth keeping) and the rest are **preserved**. The default covers the whole
// pool, which is the original full drain; a strategy that needs only a bounded
// working set passes its own figure and keeps the remainder resident. Whatever
// is drained is recorded in `restore_set()` for the caller to reload at the end.
//
// `alloc` chooses how a load with no free slot is answered: the sweep's
// free-list-only policy, or the routed bank's release-based eviction.
inline void ExpertRegistry::begin_prefill_stream(uint32_t drain_slots, PrefillAlloc alloc) {
        if (prefill_stream_) return;
        for (const auto& entry : catalog) {
            if (entry.lease_count != 0) {
                throw std::logic_error(
                    "ExpertRegistry: begin_prefill_stream with a live lease — the caller "
                    "must reach a compute-stream boundary first");
            }
            if (entry.operation != ExpertOperation::NONE) {
                throw std::logic_error(
                    "ExpertRegistry: begin_prefill_stream with a transfer still in flight — "
                    "the caller must reap the registry first");
            }
        }

        // Mark every resident present at entry, and clear the shadow state that can
        // only exist inside a frozen prefill. Warm itself is deliberately untouched —
        // its resident set and its LRU ranking are exactly what decode resumes on.
        // A drained entry has its mark cleared below, so "marked" means exactly
        // "present at entry and still resident", which is what the spare test reads.
        for (auto& entry : catalog) {
            entry.resident_at_prefill_begin = entry.owner == ExpertTier::HOT_VRAM;
            entry.pending_slot_idx = -1;
            entry.warm_shadow = false;
            entry.shadow_vram_slot = -1;
        }
        shadow_lru_.clear();
        shadow_slot_of_expert_.assign(total_experts, -1);
        std::fill(vram_slot_reservations.begin(), vram_slot_reservations.end(), 0);

        // Drain the worst-LRU residents, up to `drain_slots`, into the restore set.
        restore_set_.clear();
        uint32_t remaining = std::min<uint32_t>(
            drain_slots, static_cast<uint32_t>(hot_vram_lru.size()));
        while (remaining > 0) {
            const uint32_t gid = hot_vram_lru.back();
            hot_vram_lru.pop_back();
            auto& entry = catalog[gid];
            entry.in_lru = false;
            entry.resident_at_prefill_begin = false;
            const int32_t slot = entry.slot_idx;
            if (slot >= 0) {
                vram_slots[static_cast<size_t>(slot)] = -1;
                free_vram_slots.push_back(static_cast<uint32_t>(slot));
            }
            entry.owner = ExpertTier::COLD_NVME;
            entry.slot_idx = -1;
            entry.slot_state = ExpertSlotState::UNALLOCATED;
            restore_set_.push_back(gid);
            --remaining;
        }

        prefill_stream_ = true;
        prefill_alloc_ = alloc;
        warm_frozen_ = true;
        validate_invariants();
    }

// Leave streaming. The Hot pool must be empty — every layer was released as it
// computed — and no shadow may remain, because decode admits single ownership.
// A leftover is a driver defect and is refused, not silently cleaned up.
// The Hot pool may still hold residents **present at entry** (preserved by the
// sparing release); what it must not hold is a prefill-admitted resident, which
// is one the per-layer release should already have returned. A leftover of that
// kind is a driver defect, refused rather than cleaned up.
inline void ExpertRegistry::end_prefill_stream() {
        if (!prefill_stream_) return;
        for (const auto& entry : catalog) {
            if (entry.owner == ExpertTier::HOT_VRAM && !entry.resident_at_prefill_begin) {
                throw std::logic_error(
                    "ExpertRegistry: end_prefill_stream with a prefill-admitted Hot "
                    "resident — every layer must be released as it is computed");
            }
            if (entry.shadow_vram_slot >= 0) {
                throw std::logic_error(
                    "ExpertRegistry: end_prefill_stream with a resident Warm shadow — "
                    "every layer's shadows must be released as it is computed");
            }
            if (entry.lease_count != 0) {
                throw std::logic_error("ExpertRegistry: end_prefill_stream with a live lease");
            }
            if (entry.operation != ExpertOperation::NONE) {
                throw std::logic_error(
                    "ExpertRegistry: end_prefill_stream with a pending transfer");
            }
        }
        prefill_stream_ = false;
        warm_frozen_ = false;
        for (auto& entry : catalog) {
            entry.resident_at_prefill_begin = false;
        }
        // `restore_set_` is deliberately left intact: the caller reloads the drained
        // experts through the normal cold path before decode resumes, and the next
        // `begin_prefill_stream` overwrites it.
        validate_invariants();
    }

inline bool ExpertRegistry::prefill_streaming() const noexcept { return prefill_stream_; }

// Bulk release of one layer's residency, after it has computed. The layer's Hot
// copies and Warm shadows both return to the free list; Warm's own host slots
// are untouched. Scanning the layer's `experts_per_layer` catalog entries is
// nothing against the ~0.5 s read it follows, and it needs no parallel
// bookkeeping that could fall out of step with the catalog.
inline void ExpertRegistry::release_layer(uint32_t layer_id) {
        if (!prefill_stream_) {
            throw std::logic_error("ExpertRegistry: release_layer outside prefill streaming");
        }
        if (layer_id >= num_layers) {
            throw std::out_of_range("ExpertRegistry: release_layer layer is out of range");
        }
        const uint32_t first = layer_id * experts_per_layer;
        for (uint32_t index = 0; index < experts_per_layer; ++index) {
            auto& entry = catalog[first + index];
            if (entry.lease_count != 0) {
                throw std::logic_error(
                    "ExpertRegistry: release_layer with a live lease — the caller must "
                    "release leases at the layer boundary first (Step 0 D3)");
            }
            if (entry.operation != ExpertOperation::NONE) {
                throw std::logic_error(
                    "ExpertRegistry: release_layer with a transfer still in flight "
                    "(expert=" + std::to_string(entry.global_expert_id) +
                    ", layer=" + std::to_string(layer_id) +
                    ", operation=" + std::to_string(static_cast<int>(entry.operation)) +
                    ", gpu_transfer=" + std::to_string(static_cast<int>(entry.gpu_transfer)) +
                    ")");
            }
            // A resident **present at prefill entry** is preserved: it was not the
            // pass's to spend, so the per-layer release leaves it Hot. Only a
            // prefill-admitted resident is returned here. A preserved resident is
            // Hot-owned, so it never also carries a shadow.
            if (entry.owner == ExpertTier::HOT_VRAM && !entry.resident_at_prefill_begin) {
                if (entry.in_lru) {
                    remove_from_lru(entry, hot_vram_lru, entry.global_expert_id);
                }
                const int32_t slot = entry.slot_idx;
                if (slot >= 0) {
                    vram_slots[static_cast<size_t>(slot)] = -1;
                    free_vram_slots.push_back(static_cast<uint32_t>(slot));
                }
                entry.owner = ExpertTier::COLD_NVME;
                entry.slot_idx = -1;
                entry.slot_state = ExpertSlotState::UNALLOCATED;
            }
            if (entry.shadow_vram_slot >= 0) {
                const int32_t slot = entry.shadow_vram_slot;
                release_shadow_residency(entry);
                vram_slots[static_cast<size_t>(slot)] = -1;
                free_vram_slots.push_back(static_cast<uint32_t>(slot));
            }
        }
        validate_invariants();
    }

// Legacy per-token prefill (the engine's `forward_token` path) freezes Warm
// without streaming. The swept prefill (`begin_prefill_stream`) is the stronger
// mode and owns the flag while it is active.
inline void ExpertRegistry::set_warm_frozen(bool frozen) {
        if (prefill_stream_) return;
        if (frozen == warm_frozen_) return;
        warm_frozen_ = frozen;
        if (!frozen) {
            release_shadow_residencies();
        }
        validate_invariants();
    }

inline bool ExpertRegistry::warm_frozen() const noexcept { return warm_frozen_; }

// Warm-owned experts currently holding an extra VRAM copy. A gate reads this
// before and after a prefill: it is the shadow half of "Warm preserved", and
// it must return to 0 when the frozen phase ends.
inline uint32_t ExpertRegistry::shadow_resident_count() const noexcept {
        return static_cast<uint32_t>(shadow_lru_.size());
    }

// The VRAM copy of a Warm-owned expert, or -1. Used by a gate to observe the
// frozen prefill without reaching into the catalog.
inline int32_t ExpertRegistry::shadow_slot_of(uint32_t gid) const {
        if (gid >= shadow_slot_of_expert_.size()) return -1;
        return shadow_slot_of_expert_[gid];
    }

// Releases every *idle* shadow residency, returning its VRAM slot to the free
// list. An entry with a copy in flight or a live lease is skipped: it will be
// reclaimed by `reserve_vram_destination` once it settles.
inline void ExpertRegistry::release_shadow_residencies() {
        for (auto it = shadow_lru_.begin(); it != shadow_lru_.end();) {
            auto& entry = catalog[*it];
            if (entry.operation != ExpertOperation::NONE || entry.lease_count != 0) {
                ++it;
                continue;
            }
            const int32_t slot = entry.shadow_vram_slot;
            entry.shadow_vram_slot = -1;
            shadow_slot_of_expert_[entry.global_expert_id] = -1;
            if (slot >= 0) {
                vram_slots[static_cast<size_t>(slot)] = -1;
                free_vram_slots.push_back(static_cast<uint32_t>(slot));
            }
            it = shadow_lru_.erase(it);
        }
    }

// Drops one shadow residency from the catalog and the LRU. The caller owns the
// physical slot bookkeeping, because the two callers (a release into the free
// list, and an eviction that hands the slot to an incoming expert) need
// different things done with it.
inline void ExpertRegistry::release_shadow_residency(ExpertCatalogEntry& entry) {
        if (entry.shadow_vram_slot < 0) return;
        shadow_lru_.remove(entry.global_expert_id);
        shadow_slot_of_expert_[entry.global_expert_id] = -1;
        entry.shadow_vram_slot = -1;
    }

inline void ExpertRegistry::shadow_touch(uint32_t gid) {
        shadow_lru_.remove(gid);
        shadow_lru_.push_front(gid);
    }

} // namespace aeon::core
