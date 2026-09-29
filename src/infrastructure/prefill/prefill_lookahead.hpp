#pragma once

// -----------------------------------------------------------------------------
// The prefill sweep's read-ahead policy: how deep the frontier may go and how the
// read waves are issued.
//
// This header holds the out-of-line definitions of the `PrefillSweep` members
// that decide the lookahead depth from the **free blocks** (never a constant) and
// that issue one read wave at a time as the previous lands; the class declares
// them in `prefill_sweep.hpp`.
//
// The depth is derived from the staging arena alone, bounded to one wave in
// flight, and re-derived at each boundary.
// -----------------------------------------------------------------------------

#include "infrastructure/prefill/prefill_sweep.hpp"

#include <algorithm>
#include <cstdint>

namespace aeon::core {

// The lookahead length the free blocks allow at this instant: the biggest number
// of layers whose reads may be in flight. Derived from the runtime free blocks,
// never a constant — see `read_lookahead_capacity`.
inline uint32_t PrefillSweep::derived_lookahead_capacity() const noexcept {
    return read_lookahead_capacity();
}

// How many layers may have their reads in flight at once. Two bounds, both from
// the **cartridge box** — the staging arena — and from nothing else:
//
//   * **staging** — a layer's reads need `E` free slots, so the read leg is
//     bounded by the corridor and not by VRAM;
//   * **banks - 1** — a read must land in a bank the *resident* layer is not still
//     copying out of.
//
// VRAM is deliberately absent: the staging arena and the VRAM pool are two
// independent resources, and how many loaded magazines the rifle holds must not
// decide how many are prepared in the box. A copy that finds no free VRAM waits in
// staging; it does not stop the reads after it. A host that wants a deeper corridor
// pins more staging and the depth follows.
inline uint32_t PrefillSweep::read_lookahead_capacity() const noexcept {
        if (registry_ == nullptr || supply_ == nullptr) return 0;
        const uint32_t per_layer = registry_->experts_per_layer;
        if (per_layer == 0) return 0;
        // **At most `banks - 1`** when there are two or more banks, because a read must
        // land in a bank no other queued layer holds, and the resident layer's bank is
        // not a candidate: its bytes are still the source of its copies.
        //
        // The single-bank case keeps a depth of 1: there the drain is **blocking**
        // (`deferred_drain_` is false), so the bank is returned before the next
        // dispatch and reusing it is the legacy, correct behaviour. A bound of 0 would
        // degenerate the sweep to fully serial loading.
        const uint32_t bank_bound = staging_banks_ > 1 ? staging_banks_ - 1 : 1;
        // **Nothing else.** In particular the depth is deliberately **not** bounded by
        // free VRAM: the staging arena and the VRAM pool are two independent resources
        // — a cartridge box and a rifle's magazine wells — and the number of magazines
        // in the box must not be a function of how many the rifle has loaded. A read
        // needs a staging slot and nothing more; its copy takes a VRAM slot later, at
        // copy time, and a copy that finds no free well simply waits in staging without
        // stopping the reads that follow it. Bounding the read queue by VRAM is exactly
        // the coupling this decoupling exists to remove.
        const uint32_t staging_layers = supply_->staging_free_slots() / per_layer;
        const uint32_t bound = std::min(staging_layers, bank_bound);
        return read_ahead_max_ == 0 ? bound : std::min(bound, read_ahead_max_);
    }

// Cap the read lookahead (0 = only the staging arena bounds it). A **test
// instrument**: it lets a gate separate "how deep the reads run" from "what the
// pump costs", which are otherwise confounded.
inline void PrefillSweep::set_read_ahead_max(uint32_t max_depth) noexcept { read_ahead_max_ = max_depth; }

// Queue the next layers' reads, as many as the **derived** capacity allows. The
// capacity is recomputed each call rather than held as a constant, so the queue
// length is a function of the free blocks at this boundary and nothing else.
//
// `from` is where the caller believes the lookahead should start; the queue may
// already hold layers beyond it, in which case dispatching resumes after the
// queue's tail so nothing is dispatched twice.
inline void PrefillSweep::dispatch_ahead(uint32_t from) {
        if (registry_ == nullptr || supply_ == nullptr) return;
        if (from >= registry_->num_layers) return;
        // **One read wave in flight at a time.** A second layer's reads must not
        // overlap the first: the drive interleaves whatever is outstanding, so a
        // second wave dilutes the first and delays the layer that is actually needed
        // next. Measured on this drive: two layers outstanding cost `io_wait 1.6 ->
        // 12.9 s` and `+11.5 s` of window at every arena size that admitted them
        // (`3E` at depth 2), while one wave at a time holds `io_wait ~1.7 s`. This is a
        // structural rule, not a hardware constant: the drive's order is not ours to
        // choose, so we do not put two competing waves in front of it.
        //
        // The next wave is issued by `advance_reads` from the per-token hook, the
        // instant the previous one has landed — so the drive stays continuously busy
        // (no gap at the layer boundary) without ever holding two waves. Staging depth
        // is unchanged by this: it still decides how many layers may be **staged**
        // (read and waiting for a copy), which is the cartridge box's size.
        if (reads_in_flight()) return;

        uint32_t budget = read_lookahead_capacity();
        if (budget <= ahead_.size()) return;
        budget -= static_cast<uint32_t>(ahead_.size());

        uint32_t layer = ahead_.empty() ? from : std::max(from, ahead_.back().layer + 1);
        while (budget > 0 && layer < registry_->num_layers) {
            const uint32_t per_layer = registry_->experts_per_layer;
            if (registry_->layer_resident_count(layer) == per_layer) {
                // Already resident (a preserved resident): it costs nothing and must
                // not consume the budget, but it is not a lookahead entry either.
                ++layer;
                continue;
            }
            LookaheadEntry entry;
            entry.layer = layer;
            if (dispatch_layer_into(layer, entry.state)) {
                ahead_.push_back(std::move(entry));
                --budget;
                // The gate re-checked per planned layer, so **one call never puts two
                // waves in front of the drive**. Later layers are planned by
                // `advance_reads` as this wave lands.
                if (reads_in_flight()) break;
            }
            ++layer;
        }
        lookahead_depth_ = static_cast<uint32_t>(ahead_.size());
    }

// Issue the next read wave as soon as the previous one has landed. Called from the
// per-token hook, which is the only host activity inside a body, so the corridor
// keeps advancing while the layer computes instead of waiting for the next
// boundary. A no-op while a wave is still in flight, or when the queue is full.
inline void PrefillSweep::advance_reads() {
        if (!active_) return;
        const uint32_t from = ahead_.empty() ? active_layer_ + 1 : ahead_.back().layer + 1;
        dispatch_ahead(from);
    }

// Whether any queued layer still has reads in the drive. `io_pending` is cleared
// by the supply as each expert's read lands (the pump's `materialize_available`),
// so this is fresh within one token of the layer body.
inline bool PrefillSweep::reads_in_flight() const noexcept {
        for (const auto& entry : ahead_) {
            for (const uint8_t pending : entry.state.io_pending) {
                if (pending != 0) return true;
            }
        }
        return false;
    }

inline void PrefillSweep::update_frontier() {
        const uint32_t layers = registry_->num_layers;
        uint32_t resident_layers = 0;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            if (registry_->layer_resident_count(layer) != 0) ++resident_layers;
        }
        // The queued layers are resident by reservation even before their bytes land.
        if (!ahead_.empty()) ++resident_layers;
        frontier_depth_ = std::max(frontier_depth_, resident_layers);
    }

} // namespace aeon::core
