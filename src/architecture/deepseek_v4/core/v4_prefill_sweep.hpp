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
//     ranking are untouched for the whole sweep.");
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
    bool is_feasible() const noexcept {
        return supply_ != nullptr && registry_ != nullptr &&
               registry_->vram_capacity >= registry_->experts_per_layer;
    }

    // Drain Hot and establish the frontier. The caller must already have reached a
    // compute-stream boundary and reaped the registry.
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
        registry_->begin_prefill_stream();
        // Recorded at the exact switch point, before the frontier refills: this is
        // the observable proof that no decode resident crossed into prefill.
        hot_after_drain_ = registry_->published_hot_slots();
        active_ = true;
        fill_ahead(0);
    }

    // Guarantee layer `layer`'s whole set is resident before it computes. Normally a
    // no-op, because the lookahead reached this layer several releases ago.
    void before_layer(uint32_t layer) {
        if (!active_) return;
        ensure_layer(layer);
    }

    // Retire layer `layer` and advance the frontier into the room it frees.
    void after_layer(uint32_t layer) {
        if (!active_) return;
        registry_->release_layer(layer);
        fill_ahead(layer + 1);
    }

    // Leave the mode. Hot must be empty, which the per-layer release guarantees.
    void end() {
        if (!active_) return;
        active_ = false;
        pending_.clear();
        leases_.clear();
        registry_->end_prefill_stream();
    }

    // ---- observability, for the gate ----------------------------------------

    uint64_t layer_loads() const noexcept { return layer_loads_; }
    uint64_t experts_streamed() const noexcept { return experts_streamed_; }
    uint64_t layers_released() const noexcept { return layers_released_; }
    // Hot residents remaining at the instant of the drain, before the frontier
    // refilled. Must be 0: the switch carries nothing over.
    uint32_t hot_after_drain() const noexcept { return hot_after_drain_; }
    // Resident layers at the moment of the deepest lookahead seen — the width of the
    // sliding window, measured rather than assumed.
    uint32_t frontier_depth() const noexcept { return frontier_depth_; }

private:
    // Loads the missing experts of one layer, synchronously: the caller's next step
    // is to compute this layer, so the bytes must be resident.
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
        load_layer(layer);
    }

    // Fills the free slots with the next unvisited layers in layer order, topping up
    // a partially resident frontier layer before moving on, and stopping when the
    // next layer no longer fits. A partial last layer is deliberate: the spec fills
    // the room it has, in computation order.
    void fill_ahead(uint32_t from) {
        const uint32_t per_layer = registry_->experts_per_layer;
        const uint32_t layers = registry_->num_layers;
        while (from < layers) {
            const uint32_t resident = registry_->layer_resident_count(from);
            if (resident == per_layer) {
                ++from;
                continue;
            }
            if (registry_->free_vram_slot_count() < per_layer - resident) {
                break;
            }
            load_layer(from);
            ++from;
        }
        uint32_t resident_layers = 0;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            if (registry_->layer_resident_count(layer) != 0) ++resident_layers;
        }
        frontier_depth_ = std::max(frontier_depth_, resident_layers);
    }

    void load_layer(uint32_t layer) {
        supply_->reap_registry_transfers();
        std::vector<uint32_t> missing;
        missing.reserve(registry_->experts_per_layer);
        for (uint32_t expert = 0; expert < registry_->experts_per_layer; ++expert) {
            if (!registry_->expert_resident(layer, expert)) {
                missing.push_back(expert);
            }
        }
        if (missing.empty()) return;

        V4ExpertSupplyCoordinator::LayerPrefetchState state =
            supply_->dispatch_layer_stream(layer, missing, leases_);
        supply_->materialize_layer_prefetch(state);
        // The sweep has no `on_routed_consumed` hook, so it hands the arena's slots
        // back itself — its loads are the only staging traffic at this point.
        supply_->finish_streamed_batch(state);

        ++layer_loads_;
        experts_streamed_ += missing.size();
        // The leases the dispatch took are meaningless here: nothing evicts during a
        // swept prefill (allocation is free-list only and release is by layer), so
        // they are dropped immediately rather than carried to a boundary.
        for (const uint32_t gid : leases_) {
            registry_->release_lease(gid);
        }
        leases_.clear();
    }

    V4ExpertSupplyCoordinator* supply_{nullptr};
    ExpertRegistry* registry_{nullptr};
    bool active_{false};
    std::vector<V4ExpertSupplyCoordinator::LayerPrefetchState> pending_;
    std::vector<uint32_t> leases_;
    uint64_t layer_loads_{0};
    uint64_t experts_streamed_{0};
    uint64_t layers_released_{0};
    uint32_t hot_after_drain_{0};
    uint32_t frontier_depth_{0};
};

} // namespace aeon::core
