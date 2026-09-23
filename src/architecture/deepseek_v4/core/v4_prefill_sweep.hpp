#pragma once

// -----------------------------------------------------------------------------
// Step 6 item 6 — the prefill sweep driver.
//
// Prefill and decode are two allocation strategies, not one strategy with a
// parameter, so this is the component that owns the **switch** between them and the
// **order** prefill streams in. It holds no bytes: it drives `TieredExpertSupply` to
// load and `ExpertRegistry` to release, and it never runs on the decode path.
//
// ## The model
//
// Prefill is layer-major within a window (Step 6 D-a): the window visits layers
// 0…42 once. So at any instant the useful question is not "which expert is coldest"
// — nothing is reused inside a window — but "which layer is next". The sweep answers
// it directly:
//
//   * `begin()` drains Hot outright. No decode resident survives into prefill,
//     because not one of them is in the plan the sweep follows. Warm and its LRU
//     ranking are untouched for the whole sweep;
//   * `before_layer(L)` guarantees layer `L`'s whole set is resident. It is normally
//     already there, because the lookahead loaded it one or more layers ago;
//   * `after_layer(L)` bulk-releases layer `L` — its Hot copies and its Warm shadows
//     alike — and then refills the freed slots with the next unvisited layers in
//     layer order, until one no longer fits;
//   * `end()` requires Hot to be empty again. The per-layer release guarantees it by
//     construction, so a leftover is a driver defect and is refused, not cleaned up.
//
// Because the lookahead loads **whole layers** in order, the Hot pool is a sliding
// window over layer sets: `L, L+1, L+2, …` up to capacity, a partial layer at the
// frontier when the next full one does not fit. When `L` retires, its slots are the
// room the frontier advances into.
//
// ## The double buffer
//
// The slip above — "the lookahead loaded it one or more layers ago" — is the reason
// this class exists rather than a loop in the driver, and it is worth being exact
// about because it is where the speed is.
//
// A layer's reads are **issued before the previous layer's body runs**
// (`before_layer(L)` dispatches `L + 1`, then the driver computes `L`). So while the
// body spends ~1.4 s of compute, the drive is already reading the next layer's set;
// `before_layer(L + 1)` then finds the transfer complete and its wait is near zero.
// Measured before this split (ledger M42): a `512`-token swept prefill spent `34.6 s`
// of `96.8 s` **blocked inside `materialize`**, at `4.2 GB/s` against a `6.33 GB/s`
// drive — i.e. a third of the prefill waiting on a disk that was idle between layers.
//
// The buffer is one layer deep, and that is enough: a layer set is ~0.55 s of drive
// against ~1.4 s of compute, so one layer in flight already keeps the drive busy.
// Keeping only the dispatched state and not a queue is deliberate — the driver visits
// layers strictly in order, so at most one layer is ever ahead.
//
// When the pool cannot hold two layers at once, `dispatch_ahead` is a no-op and
// `ensure_layer` loads each layer at its own boundary: the strategy degrades to the
// serial form rather than failing.
//
// ## What this is not
//
// The lookahead loads a layer **whole**, not a routing prediction. It can: the
// router lives inside the layer body, after attention, so layer `L+1`'s *selection*
// is unknown while `L` computes — but its *set* is the whole layer, which is known,
// which is exactly why the sweep is strong in prefill and candidate staging is weak
// (§2). The loads are issued and materialized in layer order; overlapping them with
// compute is a separate concern, bounded by the staging depth (D4).
//
// ## Why not LRU here
//
// Eviction by recency ranks candidates that will be reused. Inside a window nothing
// is reused — each layer is visited once and its set then dies all at once — so the
// only correct release is the whole layer, and the only correct admission order is
// layer order. LRU is not a slower way to do this; it is the wrong instrument.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/v4_expert_supply.hpp"
#include "infrastructure/core/expert_registry.hpp"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::core {

class V4PrefillSweep {
public:
    V4PrefillSweep() = default;

    V4PrefillSweep(const V4PrefillSweep&) = delete;
    V4PrefillSweep& operator=(const V4PrefillSweep&) = delete;

    void configure(V4ExpertSupplyCoordinator* supply, ExpertRegistry* registry) {
        supply_ = supply;
        registry_ = registry;
    }
    // True when a swept prefill is possible at all: a layer's whole set must fit in
    // VRAM, since the sweep has no victim to evict and no demotion to fall back on.
    // A pool too small for one layer falls back to the per-token path rather than
    // throwing mid-prompt.
    //
    // There is deliberately **no** window-size condition here. An earlier revision
    // gated the sweep on `6W >= 2 x experts_per_layer` to stop a narrow window paying
    // a whole-layer load; the measured cost of that rule was that it silently turned
    // the sweep off on the production path (a 103-token prompt swept, an 11-token one
    // did not) while a gate's window was moved down until it stopped being slow. A
    // policy that is enabled by a threshold nobody can see, on the one path the plan
    // exists to make fast, is the wrong shape. If the narrow-window case needs a
    // different dispatch it belongs in the measurement, not in a hidden switch.
    bool is_feasible() const noexcept {
        return supply_ != nullptr && registry_ != nullptr &&
               registry_->vram_capacity >= registry_->experts_per_layer;
    }

    // Drain Hot and establish the frontier. The caller must already have reached a
    // compute-stream boundary and reaped the registry.
    //
    // The drain is **bounded**, not a full flush: the sweep needs room for one
    // layer's set (`E`), or for two when the pool can hold the lookahead as well
    // (`2E`). So it frees `2E` when `H >= 2E`, else `E` — the worst-LRU residents
    // first — and leaves the remainder **preserved** (the registry marks it). An
    // `H < E` pool is infeasible and never reaches here. Freeing only what the pass
    // needs is what lets decode's set survive the prefill; whatever is genuinely
    // freed is recorded as the registry's restore set and reloaded at the end.
    //
    // Layer 0's reads are **issued here, not waited for**: `before_layer(0)` — which
    // the driver calls immediately after — materializes them. Nothing is gained in
    // this particular gap, and it is what makes the loop below uniform.
    void begin() {
        if (active_) return;
        if (!is_feasible()) {
            throw std::logic_error(
                "V4PrefillSweep: the Hot pool holds " +
                std::to_string(registry_ ? registry_->vram_capacity : 0) +
                " slots, fewer than one layer's " +
                std::to_string(registry_ ? registry_->experts_per_layer : 0) +
                "; a swept prefill is infeasible");
        }
        const uint32_t per_layer = registry_->experts_per_layer;
        const uint32_t drain = registry_->vram_capacity >= 2u * per_layer
            ? 2u * per_layer
            : per_layer;
        registry_->begin_prefill_stream(drain);
        // Recorded at the exact switch point, before the frontier refills: this is
        // the observable proof that the drain freed only what the pass needs, and
        // that the preserved residents are the ones decode resumes on.
        hot_after_drain_ = registry_->published_hot_slots();
        active_ = true;
        dispatch_ahead(0);
    }

    // Guarantee layer `layer`'s whole set is resident before it computes, and leave
    // the **next** layer's reads in flight behind it (the double buffer).
    //
    // The order matters and is the whole point of this class's shape. A layer's reads
    // are issued *before* the previous layer's body runs, so the disk works through
    // the ~1.4 s of compute the body needs instead of sitting idle until the next
    // layer is asked for. Materializing then finds the transfer already complete and
    // costs a wait of near zero rather than the full read.
    void before_layer(uint32_t layer) {
        if (!active_) return;
        if (pending_valid_) {
            if (pending_layer_ == layer) {
                materialize_layer();
            } else {
                // A *later* layer is in flight (only reachable if the driver skipped
                // ahead). Flush it so the free-slot accounting stays simple, then load
                // this one the synchronous way.
                materialize_layer();
                ensure_layer(layer);
            }
        } else {
            ensure_layer(layer);
        }
        dispatch_ahead(layer + 1);
    }

    // Retire layer `layer`. The room it frees is what the layer after `layer + 1`
    // was waiting for; if the pool was too small to hold two layers the next
    // dispatch was skipped at `before_layer` and `ensure_layer` picks it up then.
    void after_layer(uint32_t layer) {
        if (!active_) return;
        registry_->release_layer(layer);
        ++layers_released_;
        update_frontier();
    }

    // Leave the mode. Hot must be empty, which the per-layer release guarantees.
    void end() {
        if (!active_) return;
        // Defensive: a pending dispatch would leave a transfer in flight, and
        // `end_prefill_stream` refuses that. The driver never leaves one (it only ever
        // dispatches `layer + 1`, and there is no layer 43), so this is a guard rather
        // than a path — but completing it beats throwing out of a teardown.
        if (pending_valid_) {
            const uint32_t layer = pending_layer_;
            materialize_layer();
            registry_->release_layer(layer);
        }
        active_ = false;
        leases_.clear();
        registry_->end_prefill_stream();
    }

    // ---- observability, for the gate ----------------------------------------

    uint64_t layer_loads() const noexcept { return layer_loads_; }
    uint64_t experts_streamed() const noexcept { return experts_streamed_; }
    uint64_t layers_released() const noexcept { return layers_released_; }
    // Nanoseconds **blocked** in a layer's loads: `materialize` (the completion waits
    // and the H2D issuance) plus the arena release. This is the share that a
    // load/compute overlap removes, and the number the prefill A/B reports.
    uint64_t load_ns() const noexcept { return load_ns_; }
    // Nanoseconds spent *submitting* a layer's reads (the io_uring queue plus the
    // registry bookkeeping). Not blocking on the drive, but not free either — it is
    // what the dispatcher pays to keep the disk busy.
    uint64_t io_ns() const noexcept { return io_ns_; }
    // Layers whose reads were in flight at the moment a body began — 1 when the
    // double buffer is engaged, 0 when the pool is too small to hold two layers and
    // the loads fall back to synchronous.
    uint32_t lookahead_depth() const noexcept { return lookahead_depth_; }
    // Hot residents remaining at the instant of the drain, before the frontier
    // refilled. Must be 0: the switch carries nothing over.
    uint32_t hot_after_drain() const noexcept { return hot_after_drain_; }
    // Resident layers at the moment of the deepest lookahead seen — the width of the
    // sliding window, measured rather than assumed.
    uint32_t frontier_depth() const noexcept { return frontier_depth_; }

private:
    // Loads the missing experts of one layer and waits for them: the caller's next
    // step is to compute this layer.
    void ensure_layer(uint32_t layer) {
        const uint32_t per_layer = registry_->experts_per_layer;
        const uint32_t missing = per_layer - registry_->layer_resident_count(layer);
        if (missing == 0) return;
        if (registry_->free_vram_slot_count() < missing) {
            throw std::runtime_error(
                "V4PrefillSweep: layer " + std::to_string(layer) + " needs " +
                std::to_string(missing) + " more slots but only " +
                std::to_string(registry_->free_vram_slot_count()) +
                " are free — the lookahead must not outrun the release order");
        }
        dispatch_layer(layer);
        materialize_layer();
    }

    // Issue one layer's reads and return without waiting. The bytes are in flight
    // when this returns; nothing is resident yet.
    void dispatch_layer(uint32_t layer) {
        const auto started = std::chrono::steady_clock::now();
        supply_->reap_registry_transfers();
        std::vector<uint32_t> missing;
        missing.reserve(registry_->experts_per_layer);
        for (uint32_t expert = 0; expert < registry_->experts_per_layer; ++expert) {
            if (!registry_->expert_resident(layer, expert)) {
                missing.push_back(expert);
            }
        }
        if (missing.empty()) {
            pending_valid_ = false;
            return;
        }

        pending_state_ = supply_->dispatch_layer_stream(layer, missing, leases_);
        pending_layer_ = layer;
        pending_valid_ = true;
        ++layer_loads_;
        experts_streamed_ += missing.size();
        io_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
    }

    // Wait for the dispatched layer's reads and hand its staging slots back. With the
    // double buffer engaged the wait is short by construction: the reads were issued
    // one layer's compute ago.
    void materialize_layer() {
        if (!pending_valid_) return;
        const auto started = std::chrono::steady_clock::now();
        supply_->materialize_layer_prefetch(pending_state_);
        // The sweep has no `on_routed_consumed` hook, so it hands the arena's slots
        // back itself — its loads are the only staging traffic at this point.
        supply_->finish_streamed_batch(pending_state_);
        load_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());

        // The leases the dispatch took are meaningless here: nothing evicts during a
        // swept prefill (allocation is free-list only and release is by layer), so
        // they are dropped immediately rather than carried to a boundary.
        for (const uint32_t gid : leases_) {
            registry_->release_lease(gid);
        }
        leases_.clear();
        pending_valid_ = false;
        update_frontier();
    }

    // Start the next layer's reads if the pool can hold them alongside the current
    // one. When it cannot, this is a no-op and `ensure_layer` loads that layer
    // synchronously at its own boundary — the strategy degrades to the serial form
    // rather than failing.
    void dispatch_ahead(uint32_t layer) {
        if (pending_valid_) return;
        if (layer >= registry_->num_layers) return;
        const uint32_t per_layer = registry_->experts_per_layer;
        const uint32_t resident = registry_->layer_resident_count(layer);
        if (resident == per_layer) return;
        if (registry_->free_vram_slot_count() < per_layer - resident) return;
        dispatch_layer(layer);
        lookahead_depth_ = pending_valid_ ? 1u : 0u;
    }

    void update_frontier() {
        const uint32_t layers = registry_->num_layers;
        uint32_t resident_layers = 0;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            if (registry_->layer_resident_count(layer) != 0) ++resident_layers;
        }
        if (pending_valid_) ++resident_layers;
        frontier_depth_ = std::max(frontier_depth_, resident_layers);
    }

    V4ExpertSupplyCoordinator* supply_{nullptr};
    ExpertRegistry* registry_{nullptr};
    bool active_{false};
    // The one layer whose reads are in flight, if any. A single slot and not a queue:
    // the driver visits layers strictly in order, so at most one layer is ever ahead.
    V4ExpertSupplyCoordinator::LayerPrefetchState pending_state_{};
    uint32_t pending_layer_{0};
    bool pending_valid_{false};
    std::vector<uint32_t> leases_;
    uint64_t layer_loads_{0};
    uint64_t experts_streamed_{0};
    uint64_t layers_released_{0};
    uint64_t load_ns_{0};
    uint64_t io_ns_{0};
    uint32_t lookahead_depth_{0};
    uint32_t hot_after_drain_{0};
    uint32_t frontier_depth_{0};
};

} // namespace aeon::core
