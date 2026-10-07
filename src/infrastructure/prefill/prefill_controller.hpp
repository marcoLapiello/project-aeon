#pragma once

// -----------------------------------------------------------------------------
// The layer-major prefill controller: the per-window lifecycle and the sweep.
//
// `forward_window` drives `begin` / `before_layer` / `after_layer` / `end`. The
// whole switch is local to a window: the sweep frees only what the pass needs on
// entry, streams the layers in order, and leaves the residents that were present
// at entry (plus the restored drain set) on exit, so a window is self-contained
// and decode can resume the moment it ends on the set it had before.
//
// There are two prefill strategies, and the window length picks between them:
//
//   * at or above the prompt-length gate, the **sweep** (whole layers, in order);
//   * below it, the route-aware **routed bank**, which keeps the residents and
//     admits the layer's deduplicated set.
//
// `AeonRuntimeConfig::prefill_sweep` enables them; the length picks which. This
// controller owns the choice and the flags a gate reads, plus the strategy
// dispatch. It reaches its collaborators as a small bound `Services` set rather
// than a host back-reference, and the two host-side "restore after a boundary
// move" steps (Warm re-admission and Hot re-admission) are injected as callbacks,
// because they are the host's blocking-read machinery and not prefill policy.
// -----------------------------------------------------------------------------

#include "infrastructure/device_streams.hpp"
#include "infrastructure/expert/expert_lease_holder.hpp"
#include "infrastructure/expert/layer_batch_supply.hpp"
#include "infrastructure/expert/storage/host_partition.hpp"
#include "infrastructure/prefill/prefill_sweep.hpp"
#include "infrastructure/expert/residency/expert_registry.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

namespace aeon::core {

class PrefillController {
public:
    // The collaborators the lifecycle edits.
    // The executor is nullable (a budget with no Hot slot leaves the graph
    // unbuildable but constructible, and the lifecycle still runs).
    struct Services {
        const DeviceStreams* streams{nullptr};
        LayerBatchSupply* supply{nullptr};
        ExpertLeaseHolder* executor{nullptr};
        ExpertRegistry* registry{nullptr};
        HostPartition* partition{nullptr};
    };

    void bind(const Services& services) { services_ = services; }

    // Whether `bind` has run. A budget with no Hot VRAM slot leaves the expert tier
    // unbuilt (and therefore the controller unbound), so the lifecycle below must be
    // inert rather than dereference a null service.
    bool bound() const noexcept { return services_.partition != nullptr; }

    // Resolve the switch: whether the sweep is requested at all, and the
    // prompt-length gate it must clear for a window to use it.
    void configure(bool requested, uint32_t min_tokens) {
        sweep_requested_ = requested;
        min_tokens_ = min_tokens;
    }

    // Enabled only when `prefill_sweep` is set **and** the Hot pool can hold a whole
    // layer (the sweep has no victim to evict). This is the configuration-level
    // feasibility; whether a given window *uses* the sweep also depends on the
    // prompt-length gate, which `engaged_for` applies.
    bool enabled() const noexcept { return sweep_requested_ && sweep_.is_feasible(); }

    // The prompt-length gate, resolved at load (`E / 4` unless configured).
    uint32_t min_tokens() const noexcept { return min_tokens_; }

    // Whether a window of `window_tokens` runs the sweep: enabled and feasible, and
    // at least the gate long. This is the **only** condition on the switch.
    bool engaged_for(uint32_t window_tokens) const noexcept {
        return enabled() && window_tokens >= min_tokens_;
    }

    // The strategy the last window began with, chosen from its length. Recorded
    // rather than inferred so a gate can read it.
    bool engaged() const noexcept { return sweep_active_; }

    // `window_tokens` is the window length `W`, which the driver is the only one to
    // know at this point — the gate is read here and nowhere else.
    //
    // The lifecycle has two levels. The **window** level (`open_window` / `close_window`)
    // moves the corridor to the window's shape; the **strategy** level
    // (`begin_strategy` / `end_strategy`) is what reads and releases experts. A single
    // stage runs them back to back (`begin` / `end`). A pipeline opens the window on
    // every stage first — the corridor is one region cut for all of them — but runs each
    // stage's strategy only while that stage's layers run, because all stages read
    // through the same corridor memory and two sweeps in it at once would overwrite each
    // other.
    void begin(uint32_t window_tokens) {
        open_window(window_tokens);
        begin_strategy();
    }

    void open_window(uint32_t window_tokens) {
        if (!bound()) {
            prefill_active_ = false;
            sweep_active_ = false;
            return;
        }
        prefill_active_ = enabled();
        sweep_active_ = prefill_active_ && window_tokens >= min_tokens_;
        // A window opens at a compute-stream boundary, so the corridor is quiescent
        // and the partition can be moved. This runs for **every** window, including
        // one that runs the per-token path with `prefill_sweep` off: a chunk still
        // binds its deduplicated set (`min(6C, E)`), so the corridor needs the batch
        // size whether or not the sweep is driving. Moving it only for a swept window
        // leaves the arena at decode's `2 x 6` while a chunk indexes past it.
        if (prefill_active_ || services_.partition->active()) {
            drain();
            services_.supply->reap_registry_transfers();
            if (services_.executor != nullptr) services_.executor->release_leases();
        }
        // The window's shape decides the partition: a **swept** window takes `blocks`
        // whole layers of corridor and gives Warm the least; any other window takes
        // the layer's deduplicated set, `min(6C, E)`, and gives Warm the rest.
        if (services_.partition->active()) {
            services_.partition->apply(
                sweep_active_ ? services_.partition->warm_prefill_slots()
                              : services_.partition->warm_routed_slots(),
                sweep_active_ ? services_.partition->staging_prefill_slots()
                              : services_.partition->staging_batch_slots(),
                services_.registry->experts_per_layer, outstanding_leases());
        }
    }

    void begin_strategy() {
        if (!bound() || !prefill_active_) return;
        if (sweep_active_) {
            sweep_.begin();
        } else {
            // The routed bank: drain one layer's worth of the worst-LRU residents,
            // keep the rest resident, and admit route-aware. The layer's union grows
            // into the freed `E` slots and is leased until the layer retires, so it
            // persists across the layer's chunks without a whole-layer pre-load.
            services_.registry->begin_prefill_stream(
                services_.registry->experts_per_layer,
                ExpertRegistry::PrefillAlloc::BoundedEvict);
        }
    }

    void before_layer(uint32_t layer) {
        if (!prefill_active_) return;
        if (sweep_active_) sweep_.before_layer(layer);
        // The routed path needs no pre-load: the body's router drives admission, and
        // the union is held resident by its leases until the layer retires.
    }

    void after_layer(uint32_t layer) {
        if (!prefill_active_) return;
        if (sweep_active_) {
            sweep_.after_layer(layer);
            return;
        }
        // The layer is dead the moment it retires, so its prefill-admitted set is
        // released while the residents present at entry are spared. Unlike the sweep
        // — which loads a layer in one batch and settles it before the body — the
        // routed path's **last chunk** may have left an upload in flight, and the
        // release refuses a pending transfer, so settle and reap it first.
        drain();
        services_.supply->reap_registry_transfers();
        services_.registry->release_layer(layer);
    }

    void end() {
        end_strategy();
        close_window();
    }

    void end_strategy() {
        if (!bound()) return;
        // Reach a quiescent boundary first: both the partition move and the mode
        // change below require no transfer in flight and no lease held.
        if (services_.partition->active() || prefill_active_) {
            drain();
            services_.supply->reap_registry_transfers();
            if (services_.executor != nullptr) services_.executor->release_leases();
        }
        if (prefill_active_) {
            if (sweep_active_) {
                sweep_.end();
            } else {
                services_.registry->end_prefill_stream();
            }
        }
    }

    void close_window() {
        if (!bound()) return;
        // Return the corridor to the **decode** partition: the window is over, so the
        // surrendered slots rejoin the Warm free list and decode resumes with the
        // largest residency — the whole reason the boundary moves rather than the
        // corridor being sized for the worst case. This runs for **every** window,
        // including one that ran the per-token path with the sweep off, since that
        // window moved the boundary too.
        if (services_.partition->active()) {
            services_.partition->apply(services_.partition->warm_decode_slots(),
                                       services_.partition->staging_decode_slots(),
                                       services_.registry->experts_per_layer,
                                       outstanding_leases());
        }
        // The drained Hot slots and the borrowed Warm slots return empty: decode fills
        // free slots before it evicts, so re-reading them here only delayed the first token.
        // `prefill_active_`/`sweep_active_` deliberately stay set: they record the
        // strategy the window chose, for the gates that read the choice afterwards.
        // `begin` re-decides both for the next window.
    }

    // Reaches a compute-stream boundary on every stream that can carry expert
    // traffic. Used on entering and leaving a window, and by the host's supply-phase
    // transition (frozen-Warm leave), since entering a sweep frees the whole Hot pool
    // and no upload or demotion may still be reading or writing a slot.
    void drain() const {
        const DeviceStreams& streams = *services_.streams;
        CHECK_HIP(hipStreamSynchronize(streams.compute));
        if (streams.sdma != nullptr) { CHECK_HIP(hipStreamSynchronize(streams.sdma)); }
        if (streams.sdma_cold != nullptr) { CHECK_HIP(hipStreamSynchronize(streams.sdma_cold)); }
        if (streams.demotion != nullptr) { CHECK_HIP(hipStreamSynchronize(streams.demotion)); }
    }

    // The mid-body pump hook: let the per-token pump drive the swept lookahead's
    // copies. The sweep ignores the call when it is not driving a window.
    void pump() { (void)sweep_.pump(); }

    PrefillSweep& sweep() noexcept { return sweep_; }
    const PrefillSweep& sweep() const noexcept { return sweep_; }

    // The corridor's **fill** per layer — staging slots reading, staging slots
    // copying, and the lookahead layer's reserved-but-not-yet-arrived VRAM experts.
    const std::vector<PrefillSweep::BlockOccupancy>& occupancy() const noexcept {
        return sweep_.occupancy_samples();
    }
    uint64_t load_ns() const noexcept { return sweep_.load_ns(); }
    uint64_t io_ns() const noexcept { return sweep_.io_ns(); }
    // Layers whose reads were in flight when a body started (1 = the double buffer
    // is engaged; 0 = the pool is too small for two layers and loads are serial).
    uint32_t lookahead_depth() const noexcept { return sweep_.lookahead_depth(); }
    // The lookahead length the **free blocks** allow at this instant — the smaller of
    // the free VRAM blocks and the free staging blocks. Derived, not configured.
    uint32_t derived_ahead_capacity() const noexcept { return sweep_.derived_lookahead_capacity(); }
    // Test instrument: cap the sweep's read lookahead (0 = staging-bounded only).
    void set_read_ahead_max(uint32_t max_depth) noexcept { sweep_.set_read_ahead_max(max_depth); }

private:
    size_t outstanding_leases() const noexcept {
        return services_.executor != nullptr ? services_.executor->outstanding_leases() : 0;
    }

    Services services_{};
    PrefillSweep sweep_;
    bool sweep_requested_{false};
    uint32_t min_tokens_{0};
    // Whether the layer-major prefill supply is active at all this window (either
    // strategy). The length picks the strategy; this says one was chosen.
    bool prefill_active_{false};
    // Set by `begin` for the window it opens, so the per-layer hooks and `end` act on
    // the strategy chosen for *this* window rather than re-deciding it.
    bool sweep_active_{false};
};

} // namespace aeon::core
