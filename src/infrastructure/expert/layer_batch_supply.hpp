#pragma once

// -----------------------------------------------------------------------------
// The layer-batch supply port: what a layer-major prefill drives.
//
// A swept prefill loads a whole layer's expert set, one layer at a time, and
// needs exactly seven things from the supply: issue a layer's reads, settle them,
// return its staging bank, pump its copies as the reads land, reap completed
// transfers, and read the staging arena's free/occupied counts. Those are
// *movement* operations over the tiered supply and they know nothing of any
// model, so they are declared here, neutrally, and implemented by the
// architecture's supply adapter (`V4ExpertSupplyCoordinator`).
//
// `LayerBatchState` is the per-batch vocabulary: the generic slot/index/flag
// arrays a dispatch fills, plus the neutral `PayloadBatch` the tiered supply
// returns. The routed path's richer state (with its model-shaped token map) is an
// architecture concern and derives from this one.
// -----------------------------------------------------------------------------

#include "infrastructure/expert/transport/prefetch_staging.hpp"
#include "infrastructure/expert/transport/tiered_expert_supply_types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aeon::core {

// One batch's dispatch state: the generic arrays the supply fills, one entry per
// dispatched expert, plus the transfer batch itself. This is the vocabulary the
// layer-major driver reads; it holds no model shape.
struct LayerBatchState {
    std::vector<int32_t> vram_slots;
    std::vector<uint32_t> global_expert_ids;
    std::vector<uint64_t> operation_ids;
    std::vector<uint8_t> is_prefetched;
    std::vector<uint32_t> staging_indices;
    std::vector<uint8_t> io_pending;
    std::vector<uint64_t> io_user_data;
    std::vector<uint32_t> io_request_counts;
    PayloadBatch supply_batch;

    // The position (token index for a routed dispatch, layer for a streamed one)
    // the batch starts at, and how many tokens it covers.
    uint32_t first_position{0};
    uint32_t token_count{0};

    size_t expert_count() const noexcept { return supply_batch.transfers.size(); }
};

// The port a layer-major prefill drives. Implemented by the architecture's supply
// adapter; the driver holds only this.
class LayerBatchSupply {
public:
    virtual ~LayerBatchSupply() = default;

    // Issue one layer's reads and return without waiting: the bytes are in flight
    // when this returns and nothing is resident yet. `stage_only` leaves the copy
    // for a later pump.
    virtual LayerBatchState dispatch_layer_stream(
        uint32_t layer,
        const std::vector<uint32_t>& local_expert_ids,
        std::vector<uint32_t>& leased_experts,
        uint32_t staging_base,
        bool stage_only) = 0;

    // Wait for `state`'s reads and settle them.
    virtual void materialize_layer_prefetch(LayerBatchState& state) = 0;

    // Return the staging slots a streamed batch borrowed.
    virtual void finish_streamed_batch(LayerBatchState& state) = 0;

    // Enqueue a copy for each expert whose reads have landed; non-blocking.
    virtual size_t pump_layer_prefetch(LayerBatchState& state) = 0;

    virtual void reap_registry_transfers() = 0;
    virtual uint32_t staging_free_slots() const = 0;
    virtual PrefetchStagingArena::StateCounts staging_state_counts() const = 0;
};

} // namespace aeon::core
