#pragma once

// -----------------------------------------------------------------------------
// The table of expert transfers in flight, keyed by operation id.
//
// These are the operations the supply has reserved but not yet retired: one entry
// per transfer, carrying its demotion and H2D events, its staging binding, and its
// per-leg timestamps. The registry owns the table and, because it owns the entries,
// the destruction of the HIP events they hold — so no caller has to remember to
// destroy an event it did not create.
//
// It deliberately holds no policy: it does not decide when a demotion is scheduled
// or a copy enqueued, only where the record lives and how it is retired. The
// lifecycle that reads these fields (and the reaper that retires them) stays in
// `tiered_expert_supply.hpp`.
// -----------------------------------------------------------------------------

#include "infrastructure/core/tiered_expert_supply_types.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aeon::core {

class PendingTransferRegistry {
public:
    // The record for `operation_id`, created on first use. The caller supplies the
    // tier and phase the surrounding supply already resolved, so the registry never
    // reaches into the catalog or the telemetry sink.
    PendingTransfer& ensure(uint64_t operation_id, uint32_t gid,
                            ExpertTier source_tier, SupplyTelemetryPhase phase) {
        for (auto& transfer : transfers_) {
            if (transfer.operation_id == operation_id) {
                return transfer;
            }
        }
        transfers_.push_back(PendingTransfer{});
        auto& transfer = transfers_.back();
        transfer.operation_id = operation_id;
        transfer.global_expert_id = gid;
        transfer.source_tier = source_tier;
        transfer.phase = phase;
        return transfer;
    }

    PendingTransfer* find(uint64_t operation_id) {
        for (auto& transfer : transfers_) {
            if (transfer.operation_id == operation_id) {
                return &transfer;
            }
        }
        return nullptr;
    }

    const PendingTransfer* find(uint64_t operation_id) const {
        for (const auto& transfer : transfers_) {
            if (transfer.operation_id == operation_id) {
                return &transfer;
            }
        }
        return nullptr;
    }

    // The reaper iterates and retires entries; it needs the table itself, and the
    // mutable accessor is what keeps the retirement logic next to the transfer
    // semantics rather than here.
    std::vector<PendingTransfer>& entries() noexcept { return transfers_; }
    const std::vector<PendingTransfer>& entries() const noexcept { return transfers_; }

    void erase_at(size_t index) {
        transfers_.erase(transfers_.begin() + static_cast<std::ptrdiff_t>(index));
    }

    bool empty() const noexcept { return transfers_.empty(); }
    size_t size() const noexcept { return transfers_.size(); }

    // Destroy every held event and drop the table. Safe to call on an empty
    // registry, and safe to call twice.
    void clear() noexcept {
        for (auto& transfer : transfers_) {
            if (transfer.demotion_event != nullptr) {
                (void)hipEventDestroy(transfer.demotion_event);
                transfer.demotion_event = nullptr;
            }
            if (transfer.h2d_event != nullptr) {
                (void)hipEventDestroy(transfer.h2d_event);
                transfer.h2d_event = nullptr;
            }
        }
        transfers_.clear();
    }

private:
    std::vector<PendingTransfer> transfers_;
};

} // namespace aeon::core
