#pragma once

// -----------------------------------------------------------------------------
// The Warm/staging partition: one pinned host region cut by a boundary the
// phases move.
//
// Warm and the transport corridor share a single `ExpertHostRegion`; this type
// owns the slot arithmetic that describes how that region is divided, and the two
// operations that move the boundary (`apply`) or re-depth the arena (`resize`).
// It holds the collaborators it edits as a small bound `Services` set rather than
// a back-reference to the host, so it can be read and reasoned about without the
// assembly around it.
//
// The boundary is cut to **the requirement of the phase actually running**,
// because there are three dispatch paths and each binds a different number of
// staging slots at once:
//
//   * **decode** (`dispatch_layer_prefetch`) stages one token's six experts into
//     `(layer % 2) * 6`, so it needs `2 x 6` and nothing more;
//   * a **routed** prefill (`dispatch_layer_prefetch_batch`) stages the layer's
//     deduplicated distinct set, `min(6C, E)` — one layer at any useful chunk;
//   * a **swept** prefill (`dispatch_layer_stream`) stages whole layers, `E` each,
//     and keeps `banks` live, so it needs `blocks x E`.
//
// A single "prefill" size for all three would hand decode a whole layer's worth of
// slots it never binds, and every one of those is a Warm residency given away in
// the phase whose NVMe hits they decide. So the phases are cut separately: decode
// cuts the corridor smallest and Warm largest, a routed window takes the middle,
// and a swept window takes the most and hands it back at `prefill_end`.
//
// Moving the boundary is a pointer and a free-list edit: the storage is allocated
// once, so no page is committed or pinned by the move. The real cost is that a
// slot surrendered by Warm must be drained, and a Warm expert that is dropped is
// re-read from NVMe on demand — which is why decode gets the most residency and
// the swept window takes the least.
//
// The corridor is always the region's tail, so its size alone places it. The Warm
// lane starts at `lane_base` and holds `warm_slots`: one stage's own lane when the
// region is shared by a pipeline, the whole head otherwise. Every step is
// idempotent, so a caller may apply the same partition repeatedly.
// -----------------------------------------------------------------------------

#include "infrastructure/memory/runtime_config.hpp"
#include "infrastructure/memory/memory_budget_report.hpp"
#include "infrastructure/expert/layer_batch_supply.hpp"
#include "infrastructure/prefill/prefill_sweep.hpp"
#include "infrastructure/expert/storage/expert_host_region.hpp"
#include "infrastructure/expert/residency/expert_registry.hpp"
#include "infrastructure/expert/storage/host_expert_pool.hpp"
#include "infrastructure/expert/transport/prefetch_staging.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace aeon::core {

class HostPartition {
public:
    // The collaborators a boundary move edits. Bound once from the host once the
    // region, arena, registry, pool, sweep and budget exist; the owner is
    // responsible for re-binding after the arena is (re)created or cleared.
    struct Services {
        ExpertRegistry* registry{nullptr};
        HostExpertPool* host_pool{nullptr};
        PrefetchStagingArena* staging{nullptr};
        ExpertHostRegion* region{nullptr};
        PrefillSweep* sweep{nullptr};
        LayerBatchSupply* supply{nullptr};
        MemoryBudgetReport* budget{nullptr};
    };

    void bind(const Services& services) { services_ = services; }

    bool active() const noexcept { return active_; }

    uint32_t staging_decode_slots() const noexcept { return staging_decode_slots_; }
    uint32_t staging_batch_slots() const noexcept { return staging_batch_slots_; }
    uint32_t staging_prefill_slots() const noexcept { return staging_prefill_slots_; }
    uint32_t warm_decode_slots() const noexcept { return warm_slots_decode_; }
    uint32_t warm_routed_slots() const noexcept { return warm_slots_routed_; }
    uint32_t warm_prefill_slots() const noexcept { return warm_slots_prefill_; }
    uint32_t current_warm_slots() const noexcept { return current_warm_slots_; }

    // The corridor capacity a **chunked** window will have, which is what a caller's
    // chunk has to fit — not the live arena, which is cut to decode's smaller shape
    // between windows.
    uint32_t batch_staging_capacity() const noexcept { return staging_batch_slots_; }

    // Layer-sized staging banks the sweep's arena holds, derived from the arena's
    // actual depth (`slots / experts_per_layer`). The default is `2`; the derivation
    // exists so a runtime resize moves the arena and the sweep's bank indexing
    // together, with no second place for the two to disagree.
    uint32_t sweep_staging_banks() const noexcept { return sweep_staging_banks_; }

    // Sets the partition shape at load. `phase_cuts` is whether the sweep is on and
    // therefore whether the resting cut is decode's `2 x 6` (a window is a cleanly
    // delimited phase) or the chunked size (sweep off: decode and a chunked window
    // interleave with no phase boundary, so every partition is the same).
    //
    // The region may be shared with other stages: this stage's Warm lane is
    // `[lane_base, lane_base + lane_slots)` and the corridor is anchored at the region's
    // tail, so a larger corridor takes the lanes at the end first and a stage never needs
    // to know how many others there are.
    void configure(bool active, const StagingSlotCounts& staging, bool phase_cuts,
                   uint32_t region_slots, uint32_t lane_base, uint32_t lane_slots) {
        active_ = active;
        region_slots_ = region_slots;
        lane_base_ = lane_base;
        lane_slots_ = lane_slots;
        // With the sweep on, a window is a cleanly delimited phase and the resting cut
        // is decode's `2 x 6`; with it off, decode and a chunked window interleave with
        // no phase boundary to cut at, so the resting cut is the chunked size instead.
        // With no shared region the value is unused (it is only read through `apply`,
        // which is guarded by `active_`), so it keeps decode's shape.
        staging_decode_slots_ = (active && !phase_cuts) ? staging.batch : staging.decode;
        staging_batch_slots_ = staging.batch;
        staging_prefill_slots_ = staging.prefill;
        // The smallest corridor that can still run the certified decode path: one
        // token's routed experts, double-buffered. `resize` refuses to go below it.
        staging_floor_slots_ = staging.decode;
        if (active) {
            warm_slots_decode_ = warm_for(staging_decode_slots_);
            warm_slots_routed_ = warm_for(staging.batch);
            warm_slots_prefill_ = warm_for(staging.prefill);
            current_warm_slots_ = warm_slots_decode_;
        } else {
            warm_slots_decode_ = 0;
            warm_slots_routed_ = 0;
            warm_slots_prefill_ = 0;
            current_warm_slots_ = 0;
        }
    }

    // Recompute the sweep's bank count from the arena it actually has, then point
    // the sweep at the right bank. Called wherever the arena's depth changes.
    void refresh_sweep_banks(uint32_t experts_per_layer) {
        if (services_.staging == nullptr || services_.sweep == nullptr) return;
        sweep_staging_banks_ = experts_per_layer == 0
            ? 1u
            : std::max<uint32_t>(1u, services_.staging->slot_count() / experts_per_layer);
        services_.sweep->configure(services_.supply, services_.registry, sweep_staging_banks_);
    }

    // Re-size the staging arena at runtime: a **depth** change, not a format change.
    // The depth is a resource budget the transfer corridor consumes, not a behaviour
    // constant — above the minimum the design needs, more slots only give the
    // lookahead more room to run ahead, and the throughput must not change. This is
    // what lets a host size the corridor from its remaining RAM instead of from a
    // figure tuned on one machine, and it is what the depth-sensitivity gate varies.
    //
    // Preconditions, refused rather than assumed: every staging slot free and no
    // expert lease outstanding. The resize destroys and recreates the per-slot
    // events, so nothing may still reference one — which is exactly the state a
    // window boundary leaves behind. `prefill_active_` is deliberately **not** the
    // guard: it records the strategy the last window chose and stays set afterwards
    // (gates read it), so it does not mean "a window is running now". The two state
    // conditions below are the exact ones, and an in-progress window cannot satisfy
    // both while a caller outside the engine is on the stack.
    bool resize(uint32_t slots, uint32_t experts_per_layer, size_t outstanding_leases) {
        if (services_.staging == nullptr) return false;
        if (services_.staging->in_use_slots() != 0) return false;
        if (outstanding_leases != 0) return false;
        if (slots < staging_floor_slots_) return false;
        // A shared region has a **fixed** total: its corridor can only be resized by
        // moving the boundary, which is what the phase transitions do. Letting a
        // caller resize the arena alone would silently over-commit the region, so the
        // two directions are different operations and this one refuses.
        if (active_) return false;

        services_.staging->resize(slots);
        refresh_sweep_banks(experts_per_layer);
        // The budget report figure; keep it equal to the allocation so the two cannot
        // drift after a resize.
        services_.budget->transient_staging_bytes =
            static_cast<size_t>(slots) * services_.staging->payload_bytes();
        return true;
    }

    // Move the boundary. `warm_slots` is the Warm head; the corridor is the rest.
    // Every step is idempotent, so re-requesting the current partition is a no-op.
    void apply(uint32_t warm_slots, uint32_t staging_slots,
               uint32_t experts_per_layer, size_t outstanding_leases) {
        if (!active_ || services_.staging == nullptr) return;
        // **Idempotent by construction.** `prefill_begin` is not the only caller that
        // enters a window — the driver enters one itself when it begins a window, so
        // the same partition is requested twice for the same window. Re-applying it
        // would re-base the arena and destroy its per-slot events while the sweep's
        // own lookahead is using them, so an already-current partition is a no-op.
        if (warm_slots == current_warm_slots_ &&
            staging_slots == services_.staging->slot_count()) {
            return;
        }
        // A boundary move re-bases the arena and re-creates its per-slot events, so
        // nothing may still reference one. This is the same precondition `resize`
        // enforces, and it is what a phase boundary leaves.
        if (services_.staging->in_use_slots() != 0 || outstanding_leases != 0) {
            throw std::logic_error(
                "HostPartition: cannot move the host partition with a transfer or lease live "
                "(staging_in_use=" + std::to_string(services_.staging->in_use_slots()) +
                " of " + std::to_string(services_.staging->slot_count()) +
                ", leases=" + std::to_string(outstanding_leases) + ")");
        }
        const uint32_t region_slots = services_.region->slot_count();
        // A lane the corridor has swallowed entirely holds nothing, so only a lane that
        // keeps slots can overlap it.
        if (staging_slots > region_slots ||
            (warm_slots > 0 && lane_base_ + warm_slots > region_slots - staging_slots)) {
            throw std::logic_error(
                "HostPartition: the Warm lane and the corridor overlap in the region");
        }

        // Shrink: demote the Warm tail **before** the capacity moves under it, so the
        // free-list edit finds every surrendered slot empty.
        if (warm_slots < services_.registry->usable_host_capacity()) {
            services_.registry->release_host_tail(warm_slots);
            services_.registry->shrink_host_capacity(warm_slots);
        }
        services_.host_pool->bind(services_.region->slot_ptr(lane_base_), warm_slots,
                                  services_.region->format(), services_.region->is_pinned());
        if (warm_slots > services_.registry->usable_host_capacity()) {
            services_.registry->grow_host_capacity(warm_slots);
        }
        services_.staging->bind(services_.region->slot_ptr(region_slots - staging_slots),
                                staging_slots, services_.region->format());

        refresh_sweep_banks(experts_per_layer);
        services_.budget->transient_staging_bytes =
            static_cast<size_t>(staging_slots) * services_.staging->payload_bytes();
        current_warm_slots_ = warm_slots;
    }

private:
    // Warm's share when the corridor holds `staging_slots` of the region's tail: what is
    // left of the region ahead of the corridor after this lane's start, up to the lane.
    uint32_t warm_for(uint32_t staging_slots) const noexcept {
        const uint32_t room = region_slots_ > staging_slots ? region_slots_ - staging_slots : 0u;
        const uint32_t usable = room > lane_base_ ? room - lane_base_ : 0u;
        return std::min(usable, lane_slots_);
    }

    Services services_{};
    bool active_{false};
    uint32_t region_slots_{0};
    uint32_t lane_base_{0};
    uint32_t lane_slots_{0};
    uint32_t staging_decode_slots_{0};
    uint32_t staging_batch_slots_{0};
    uint32_t staging_prefill_slots_{0};
    // The decode shape, the smallest corridor `resize` will leave in place.
    uint32_t staging_floor_slots_{0};
    uint32_t warm_slots_decode_{0};
    uint32_t warm_slots_routed_{0};
    uint32_t warm_slots_prefill_{0};
    uint32_t current_warm_slots_{0};
    // Layer-sized staging banks the sweep's arena holds, derived from the arena's
    // depth (`slots / experts_per_layer`). The default is `2` — two layer-blocks, one
    // the read destination and one the copy source — and it is deliberately **not** a
    // config knob, because a settable depth would let a resource select the algorithm.
    uint32_t sweep_staging_banks_{2};
};

} // namespace aeon::core
