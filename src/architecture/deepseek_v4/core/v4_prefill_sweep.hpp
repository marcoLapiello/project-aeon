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
// ## The deferred drain
//
// A layer's H2D copies are **not waited for on the host**, and their staging bank is
// returned at `after_layer` rather than at the layer's own boundary. The copy therefore
// overlaps the layer's attention and router — the work the layer body does before its
// MoE — instead of fronting it: the sweep used to block ~0.12 s per layer on it.
// Ordering is the **consumer's**, not the driver's: the body's MoE dispatch joins each
// still-pending transfer and the executor waits on its per-expert event before the MoE
// reads the weights, while a transfer the registry has already reaped is complete by
// definition. Ordering the whole body here instead would recover nothing, because it
// would put the copy back in front of the very work it hides behind.
//
// The bank cannot be reused before it is returned, which is why the arena must hold
// **two** layer-sized banks when this is on: the bank in flight and the bank the
// lookahead reads into. With one bank the sweep degrades to the blocking drain, exactly
// as before.
//
// ## What this is not
//
// The lookahead loads a layer **whole**, not a routing prediction. It can: the
// router lives inside the layer body, after attention, so layer `L+1`'s *selection*
// is unknown while `L` computes — but its *set* is the whole layer, which is known,
// which is exactly why the sweep is strong in prefill and candidate staging is weak
// (§2). The loads are issued and materialized in layer order; overlapping the upload
// with compute is the deferred drain above, bounded by the staging banks.
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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::core {

class V4PrefillSweep {
public:
    V4PrefillSweep() = default;

    V4PrefillSweep(const V4PrefillSweep&) = delete;
    V4PrefillSweep& operator=(const V4PrefillSweep&) = delete;

    // `staging_banks` is how many layer-sized banks the arena holds (`1` = the
    // pre-Phase-1 shape, `2` = the deferred drain's headroom). It only decides which
    // bank a layer's reads land in: layer `L` uses bank `L % staging_banks`, so with
    // two banks the layer whose copies are still in flight and the layer being read
    // ahead never share a slot. The bank count must match the arena the host built.
    void configure(V4ExpertSupplyCoordinator* supply, ExpertRegistry* registry,
                   uint32_t staging_banks = 1) {
        supply_ = supply;
        registry_ = registry;
        staging_banks_ = staging_banks == 0 ? 1u : staging_banks;
        // Whether the drain can be deferred: it needs a second bank to read the
        // lookahead into while this layer's uploads are still in flight. With one bank
        // the sweep keeps the blocking drain — the pre-Phase-1 shape.
        deferred_drain_ = staging_banks_ > 1;
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
        // The corridor's fill is sampled per **window**, not accumulated across
        // windows: a gate reads the readout for the pass it just ran.
        occupancy_samples_.clear();
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
        if (!ahead_.empty() && ahead_.front().layer == layer) {
            materialize_front();
        } else {
            // Nothing queued for this layer: either the resource-derived depth did not
            // reach it, or it is already resident (a preserved resident is skipped by
            // the lookahead rather than queued). `ensure_layer` is idempotent.
            ensure_layer(layer);
        }
        dispatch_ahead(layer + 1);
        // Sampled **after** the lookahead is issued, at the instant the body starts:
        // this is the corridor's fill for this layer, which is the thing a gate reads
        // (a `reading` block and a `copying` block both non-zero means the pipeline is
        // overlapped; one pinned at `E` with the other at zero is a parking lot).
        record_occupancy(layer);
    }

    // The mid-body pump (plan P2.3): enqueue each queued layer's copies as its own
    // reads land, instead of in one block at the next boundary. Called from the
    // executor's per-token hook — the only host activity inside a body — so the
    // lookahead's VRAM blocks fill during this layer's body rather than after it.
    // Non-blocking, and over **every** queued layer, so a deeper lookahead (P2.4) is
    // pumped too.
    size_t pump() {
        if (!active_) return 0;
        size_t pumped = 0;
        for (auto& entry : ahead_) {
            pumped += supply_->pump_layer_prefetch(entry.state);
        }
        return pumped;
    }

    // Retire layer `layer`. The room it frees is what the layers after it were waiting
    // for; the lookahead is re-derived at the next `before_layer`.
    //
    // The staging bank is returned **here**, not at the layer's own `before_layer`.
    // The driver synchronizes the compute stream at this boundary (Step 0 D3), so the
    // layer's H2D copies have landed by now — the reclaim's event sync is instant, and
    // the copy spent the layer's body in flight instead of fronting it. The extra
    // `reap_registry_transfers` is what promotes the layer's completed uploads out of
    // `PROMOTION_PENDING` before `release_layer` refuses a live transfer.
    void after_layer(uint32_t layer) {
        if (!active_) return;
        // The resident bank belongs to the layer being retired: the driver calls
        // `before_layer(L)` then `after_layer(L)`, so a mismatch is a driver defect and
        // is refused rather than silently reclaiming the wrong bank.
        if (resident_valid_ && resident_layer_ != layer) {
            throw std::logic_error(
                "V4PrefillSweep: layer " + std::to_string(layer) +
                " retired while layer " + std::to_string(resident_layer_) +
                "'s staging bank is still resident");
        }
        // **Reap first, reclaim second** (plan R3): the copies landed during the body,
        // so the reaper releases their staging slots **by completion** and the block
        // reclaim below then finds the bank already drained. The reclaim stays as the
        // fallback for any slot whose event had not fired at this instant.
        supply_->reap_registry_transfers();
        reclaim_resident_staging();
        registry_->release_layer(layer);
        ++layers_released_;
        update_frontier();
    }

    // Leave the mode. Hot must be empty, which the per-layer release guarantees.
    void end() {
        if (!active_) return;
        // Settle and retire every queued layer. The driver has synchronized, so the
        // reads and copies are done and this is quick; a leftover would leave a
        // transfer in flight and `end_prefill_stream` refuses one.
        while (!ahead_.empty()) {
            LookaheadEntry entry = std::move(ahead_.front());
            ahead_.pop_front();
            materialize_entry(entry);
            reclaim_resident_staging();
            supply_->reap_registry_transfers();
            registry_->release_layer(entry.layer);
        }
        reclaim_resident_staging();
        active_ = false;
        leases_.clear();
        registry_->end_prefill_stream();
    }

    // ---- observability, for the gate ----------------------------------------

    // One layer's corridor fill, sampled as its body begins.
    struct BlockOccupancy {
        uint32_t layer{0};
        // Staging arena, by pipeline stage. `reading` is a read landing in a slot;
        // `copying` is a copy-into-VRAM draining a slot. Both non-zero = overlapped.
        uint32_t staging_free{0};
        uint32_t staging_reading{0};
        uint32_t staging_copying{0};
        // Experts whose VRAM slots are **reserved for the lookahead layers** at this
        // instant. These are outstanding: the block is held but its bytes have not
        // all arrived, which is the "reserved-but-empty" figure.
        uint32_t vram_reserved_ahead{0};
        // The lookahead length the free blocks allowed at this sample — derived, not
        // fixed (R5). Zero means a resource was exhausted.
        uint32_t derived_capacity{0};
        // The components of `derived_capacity`, so the gate can show **what limits it**
        // and **what it is relative to**: the computing layer, or the staging queue.
        uint32_t vram_free_slots{0};
        uint32_t staging_free_slots{0};
        uint32_t vram_free_layers{0};
        uint32_t staging_free_layers{0};
        // Layers whose reads are dispatched and not yet materialized — the queue depth.
        uint32_t ahead_count{0};
        // Hot residents held **preserved** (not the sweep's to spend); they are what
        // shrinks the pool the lookahead is derived from.
        uint32_t hot_preserved{0};
    };

    const std::vector<BlockOccupancy>& occupancy_samples() const noexcept {
        return occupancy_samples_;
    }

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
    // How many layer-sets the **staging** arena can hold in flight right now — the
    // read leg's bound and the lookahead depth (plan R5/P2.5). Deliberately **not**
    // derived from VRAM: a read needs only a staging slot, a copy needs a VRAM slot,
    // so a VRAM-derived read queue would stop reads whenever VRAM filled even though
    // staging was free. That coupling is what the old `min(free VRAM, free staging)`
    // encoded; removing it is what lets reads run a layer ahead of copies.
    uint32_t derived_lookahead_capacity() const noexcept { return read_lookahead_capacity(); }

    // How many layers the staging arena can hold in flight — the read budget. The copy
    // leg is bounded separately (in the supply, by free VRAM slots), so this is the
    // only bound on reads.
    uint32_t read_lookahead_capacity() const noexcept {
        if (registry_ == nullptr || supply_ == nullptr) return 0;
        const uint32_t per_layer = registry_->experts_per_layer;
        if (per_layer == 0) return 0;
        // **At most `banks - 1`** when there are two or more banks, because a read must
        // land in a bank no other queued layer holds, and the resident layer's bank is
        // not a candidate: its bytes are still the source of its copies. With the
        // staging arena at `2E` that is exactly one layer, which is why a depth of 2
        // measured as a regression — it let the layer two ahead land in the resident's
        // bank (`reading 498`, `io_wait 1.7 -> 12.8 s`). A deeper lookahead is therefore
        // a **larger arena** question, not a scheduling one: `3E` staging admits 2.
        //
        // The single-bank case keeps a depth of 1: there the drain is **blocking**
        // (`deferred_drain_` is false), so the bank is returned before the next
        // dispatch and reusing it is the legacy, correct behaviour. A bound of 0 would
        // degenerate the sweep to fully serial loading.
        const uint32_t bank_bound = staging_banks_ > 1 ? staging_banks_ - 1 : 1;
        const uint32_t staging_layers = supply_->staging_free_slots() / per_layer;
        const uint32_t bound = std::min(staging_layers, bank_bound);
        return read_ahead_max_ == 0 ? bound : std::min(bound, read_ahead_max_);
    }

    // Cap the read lookahead (0 = only the staging arena bounds it). A **test
    // instrument**: it lets a gate separate "how deep the reads run" from "what the
    // pump costs", which are otherwise confounded.
    void set_read_ahead_max(uint32_t max_depth) noexcept { read_ahead_max_ = max_depth; }

private:
    // One dispatched-ahead layer: its reads are in flight (or landed), in layer order.
    struct LookaheadEntry {
        V4ExpertSupplyCoordinator::LayerPrefetchState state{};
        uint32_t layer{0};
    };

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
        LookaheadEntry entry;
        entry.layer = layer;
        if (!dispatch_layer_into(layer, entry.state)) return;
        materialize_entry(entry);
    }

    // Issue one layer's reads and return without waiting. The bytes are in flight
    // when this returns; nothing is resident yet. Returns `false` when the layer has
    // nothing missing (so the caller does not queue an empty entry).
    bool dispatch_layer_into(uint32_t layer, V4ExpertSupplyCoordinator::LayerPrefetchState& out) {
        const auto started = std::chrono::steady_clock::now();
        supply_->reap_registry_transfers();
        std::vector<uint32_t> missing;
        missing.reserve(registry_->experts_per_layer);
        for (uint32_t expert = 0; expert < registry_->experts_per_layer; ++expert) {
            if (!registry_->expert_resident(layer, expert)) {
                missing.push_back(expert);
            }
        }
        if (missing.empty()) return false;

        out = supply_->dispatch_layer_stream(
            layer, missing, leases_, staging_base_for(layer), /*stage_only=*/true);
        ++layer_loads_;
        experts_streamed_ += missing.size();
        io_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        return true;
    }

    // Settle the queue's front: wait its reads, hold its bank (or drain it when there
    // is only one bank), and make it the resident layer.
    void materialize_front() {
        if (ahead_.empty()) return;
        LookaheadEntry entry = std::move(ahead_.front());
        ahead_.pop_front();
        materialize_entry(entry);
    }

    // Wait for `entry`'s reads, leave its uploads in flight, and hold the batch
    // resident. With the double buffer engaged the wait is short by construction: the
    // reads were issued one or more layers' compute ago.
    //
    // With a deferred drain the host does **not** order the compute stream here, so the
    // body may launch while the copies are still in flight and the slots are held until
    // `after_layer` returns them. Blocking (or ordering the whole body) here would put
    // the copy in front of the attention and router it is meant to hide behind, which
    // recovers nothing at all. With one bank there is no room to hold the copies
    // through the body, so the drain is the blocking one, as before.
    void materialize_entry(LookaheadEntry& entry) {
        // Reap before the defensive reclaim, for the same reason as `after_layer`: a
        // slot whose copy has already landed is freed by its own completion event.
        supply_->reap_registry_transfers();
        // Defensive: the normal flow returns the previous bank in `after_layer`, but
        // returning it here beats leaking one.
        reclaim_resident_staging();
        const auto started = std::chrono::steady_clock::now();
        supply_->materialize_layer_prefetch(entry.state);
        if (!deferred_drain_) {
            // One bank: no room to hold the copies through the body, so the drain is
            // the blocking one and the slots are returned before the next dispatch —
            // the pre-Phase-1 shape.
            supply_->finish_streamed_batch(entry.state);
        }
        // With a deferred drain nothing more is done here: the copies stay in flight
        // and **nothing** is ordered on the compute stream. Ordering is the consumer's:
        // the body's MoE dispatch joins each still-pending transfer and the executor
        // waits on its per-expert event before the MoE reads the weights
        // (`V4TieredExpertExecutor::accumulate_routed`), while a transfer the registry
        // has already reaped is complete by definition. Ordering the whole body here
        // instead would put the copy in front of attention and the router — the work
        // it is meant to hide behind — and recover nothing at all.
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
        if (deferred_drain_) {
            resident_state_ = std::move(entry.state);
            resident_layer_ = entry.layer;
            resident_valid_ = true;
        }
        update_frontier();
    }

    // Hand a materialized layer's staging bank back to the arena. Its uploads have
    // landed by the time this runs (the driver synchronized the compute stream at the
    // boundary), so the per-slot event sync inside is instant and charges ~nothing to
    // `h2d_drain_ns_`.
    void reclaim_resident_staging() {
        if (!resident_valid_) return;
        const auto started = std::chrono::steady_clock::now();
        supply_->finish_streamed_batch(resident_state_);
        load_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        resident_valid_ = false;
    }

    // The staging bank this layer's reads land in. With one bank it is always 0 (the
    // pre-Phase-1 shape, where each layer's slots are freed before the next dispatch);
    // with two the layers alternate, so the layer in flight and the layer being read
    // ahead never share a slot.
    uint32_t staging_base_for(uint32_t layer) const noexcept {
        if (staging_banks_ <= 1 || registry_ == nullptr) return 0u;
        return (layer % staging_banks_) * registry_->experts_per_layer;
    }

    // Queue the next layers' reads, as many as the **derived** capacity allows. The
    // capacity is recomputed each call rather than held as a constant, so the queue
    // length is a function of the free blocks at this boundary and nothing else.
    //
    // `from` is where the caller believes the lookahead should start; the queue may
    // already hold layers beyond it, in which case dispatching resumes after the
    // queue's tail so nothing is dispatched twice.
    void dispatch_ahead(uint32_t from) {
        if (registry_ == nullptr || supply_ == nullptr) return;
        if (from >= registry_->num_layers) return;

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
            }
            ++layer;
        }
        lookahead_depth_ = static_cast<uint32_t>(ahead_.size());
    }

    void update_frontier() {
        const uint32_t layers = registry_->num_layers;
        uint32_t resident_layers = 0;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            if (registry_->layer_resident_count(layer) != 0) ++resident_layers;
        }
        // The queued layers are resident by reservation even before their bytes land.
        if (!ahead_.empty()) ++resident_layers;
        frontier_depth_ = std::max(frontier_depth_, resident_layers);
    }

    // Sample the corridor's fill for one layer. Cheap (an arena scan) and bounded by
    // the layer count, so it is on unconditionally for the swept path.
    void record_occupancy(uint32_t layer) {
        const auto counts = supply_->staging_state_counts();
        BlockOccupancy sample;
        sample.layer = layer;
        sample.staging_free = counts.free;
        sample.staging_reading = counts.reading;
        sample.staging_copying = counts.copying;
        sample.vram_reserved_ahead = 0;
        for (const auto& entry : ahead_) {
            sample.vram_reserved_ahead += static_cast<uint32_t>(entry.state.expert_count());
        }
        sample.derived_capacity = derived_lookahead_capacity();
        sample.ahead_count = static_cast<uint32_t>(ahead_.size());
        sample.hot_preserved = registry_->published_hot_slots();
        const uint32_t per_layer = registry_->experts_per_layer;
        sample.vram_free_slots = static_cast<uint32_t>(registry_->free_vram_slot_count());
        sample.staging_free_slots = counts.free;
        if (per_layer > 0) {
            sample.vram_free_layers = sample.vram_free_slots / per_layer;
            sample.staging_free_layers = sample.staging_free_slots / per_layer;
        }
        occupancy_samples_.push_back(sample);
    }

    V4ExpertSupplyCoordinator* supply_{nullptr};
    ExpertRegistry* registry_{nullptr};
    bool active_{false};
    // The lookahead queue: layers whose reads have been dispatched and whose bytes are
    // on their way, in layer order. Its length is **derived from the free blocks** at
    // each boundary (plan R5), so it deepens on a larger pool and empties when either
    // VRAM or staging runs out — it is not a fixed depth.
    std::deque<LookaheadEntry> ahead_;
    // Test instrument: upper bound on the read lookahead (0 = staging only).
    uint32_t read_ahead_max_{0};
    // The layer whose reads are settled and whose uploads are ordered behind the
    // compute stream, but whose staging bank is still held: the deferred-drain state.
    // `after_layer` returns it, so the upload overlaps the body instead of fronting it.
    V4ExpertSupplyCoordinator::LayerPrefetchState resident_state_{};
    uint32_t resident_layer_{0};
    bool resident_valid_{false};
    // Layer-sized staging banks the arena holds. `1` restores the pre-Phase-1 shape;
    // `2` is the headroom the deferred drain reads into. Must match the host's arena.
    uint32_t staging_banks_{1};
    // `staging_banks_ > 1`: whether the drain is deferred to `after_layer`. Derived so
    // the two uses of the bank count cannot disagree.
    bool deferred_drain_{false};
    std::vector<uint32_t> leases_;
    uint64_t layer_loads_{0};
    uint64_t experts_streamed_{0};
    uint64_t layers_released_{0};
    uint64_t load_ns_{0};
    uint64_t io_ns_{0};
    uint32_t lookahead_depth_{0};
    uint32_t hot_after_drain_{0};
    uint32_t frontier_depth_{0};
    // One corridor-fill sample per layer, for the gate.
    std::vector<BlockOccupancy> occupancy_samples_;
};

} // namespace aeon::core
