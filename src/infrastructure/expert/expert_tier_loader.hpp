#pragma once

// -----------------------------------------------------------------------------
// The expert tier's bulk load and restore.
//
// Three routines that read whole expert payloads from the artifact with batched
// `O_DIRECT` and place them in the tier's pools:
//
//   * `preload_hot`  — every Hot VRAM resident, once, at load;
//   * `preload_warm` — every Warm host resident, once, at load;
//   * `restore_residents` — the Hot residents a prefill drained, at the window
//     boundary, through the normal cold path (reserve → read → publish → release).
//
// None of it is model-specific: it moves fixed-size payloads into slots the
// `ExpertRegistry` names, and it knows only the container's format and the pools.
// It lives in `infrastructure/` so a second architecture reuses the load and
// restore path instead of re-deriving it.
//
// It holds no bytes. Its collaborators are bound as a small pointer set (the same
// idiom `HostPartition` and `PrefillController` use) and it is bound by whoever
// composes a tier, so the model keeps ownership of the pools, the registry and the
// reader. `demotion_queue_capacity` is read through a pointer because the tier sets
// it after the supply is configured, later than the first bind.
// -----------------------------------------------------------------------------

#include "infrastructure/artifact/aeon_loader.hpp"
#include "infrastructure/expert/transport/expert_direct_io.hpp"
#include "infrastructure/expert/storage/expert_payload_pool.hpp"
#include "infrastructure/expert/residency/expert_registry.hpp"
#include "infrastructure/expert/storage/host_expert_pool.hpp"
#include "infrastructure/memory/memory_budget_report.hpp"
#include "infrastructure/hip_check.hpp"
#include "infrastructure/io/aligned_allocator.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace aeon::core {

class ExpertTierLoader {
public:
    struct Services {
        ExpertRegistry* registry{nullptr};
        ExpertPayloadPool* payload_pool{nullptr};
        HostExpertPool* host_pool{nullptr};       // null when there is no Warm tier
        ExpertDirectIO* direct_io{nullptr};
        const AeonModelLoader* source{nullptr};
        const MemoryBudgetReport* budget{nullptr};
        hipStream_t compute{nullptr};
        uint64_t* demotion_queue_capacity{nullptr};  // read at restore time
    };

    void bind(const Services& services) { services_ = services; }

    // Every Hot resident, read straight from the artifact and uploaded once. The
    // physical slots come from the registry's own `vram_slots` rather than being
    // re-derived from the round-robin order.
    void preload_hot(const ExpertFormatDescriptor& format) {
        const size_t batch = std::max<size_t>(
            1, services_.direct_io->submission_capacity() /
                   ExpertDirectIO::requests_per_fragment(format));
        for (uint32_t start = 0; start < services_.budget->hot_vram_slots;
             start += static_cast<uint32_t>(batch)) {
            const uint32_t end = std::min<uint32_t>(
                services_.budget->hot_vram_slots, start + static_cast<uint32_t>(batch));

            std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
            std::vector<aeon::io::AlignedBuffer> buffers;
            for (uint32_t slot = start; slot < end; ++slot) {
                const int32_t gid = services_.registry->vram_slots[slot];
                if (gid < 0) continue;
                const auto& entry = services_.registry->catalog[static_cast<size_t>(gid)];
                expert_ids.emplace_back(entry.layer_id, entry.expert_id);
                buffers.emplace_back(format.payload_bytes, format.sector_size);
            }

            std::vector<uint8_t*> destinations;
            destinations.reserve(buffers.size());
            for (auto& buffer : buffers) {
                destinations.push_back(static_cast<uint8_t*>(buffer.data()));
            }
            services_.direct_io->read_blocking(*services_.source, expert_ids, destinations);

            size_t index = 0;
            for (uint32_t slot = start; slot < end; ++slot) {
                if (services_.registry->vram_slots[slot] < 0) continue;
                services_.payload_pool->upload_from_host_expert(
                    slot, destinations[index++], services_.compute);
            }
            CHECK_HIP(hipStreamSynchronize(services_.compute));
        }
    }

    // The Warm pool is filled in place — the registry chose the owning experts at
    // construction, so this writes their payloads into the pinned slots it
    // reserved rather than deciding residency again.
    void preload_warm(const ExpertFormatDescriptor& format) {
        const size_t batch = std::max<size_t>(
            1, services_.direct_io->submission_capacity() /
                   ExpertDirectIO::requests_per_fragment(format));
        for (uint32_t start = 0; start < services_.budget->warm_host_slots;
             start += static_cast<uint32_t>(batch)) {
            const uint32_t end = std::min<uint32_t>(
                services_.budget->warm_host_slots, start + static_cast<uint32_t>(batch));

            std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
            std::vector<uint8_t*> destinations;
            for (uint32_t slot = start; slot < end; ++slot) {
                const int32_t gid = services_.registry->host_slots[slot];
                if (gid < 0) continue;
                const auto& entry = services_.registry->catalog[static_cast<size_t>(gid)];
                expert_ids.emplace_back(entry.layer_id, entry.expert_id);
                destinations.push_back(services_.host_pool->get_expert_slot_ptr(slot));
            }
            services_.direct_io->read_blocking(*services_.source, expert_ids, destinations);
        }
    }

    // Reloads the Hot residents a prefill drained, so the pool returns to the set it
    // held before the pass. The gids come from the registry's `restore_set()`; each is
    // admitted through the normal cold path — reserve a slot, read the payload, publish
    // it — which is the same machinery decode uses, so nothing here is a second code
    // path. It runs at a boundary (the prefill has already ended), so its blocking
    // reads are off the hot path, and it is batched exactly like `preload_hot`.
    void restore_residents(const ExpertFormatDescriptor& format) {
        const std::vector<uint32_t> restore = services_.registry->restore_set();
        if (restore.empty()) return;
        const uint32_t per_layer = services_.registry->experts_per_layer;
        const size_t batch = std::max<size_t>(
            1, services_.direct_io->submission_capacity() /
                   ExpertDirectIO::requests_per_fragment(format));
        for (size_t start = 0; start < restore.size(); start += batch) {
            const size_t end = std::min(restore.size(), start + batch);
            std::vector<uint32_t> operation_ids;
            std::vector<int32_t> slots;
            std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
            std::vector<aeon::io::AlignedBuffer> buffers;
            operation_ids.reserve(end - start);
            for (size_t i = start; i < end; ++i) {
                const uint32_t gid = restore[i];
                const auto request = services_.registry->reserve_request_by_gid(
                    gid, 0, *services_.demotion_queue_capacity);
                if (request.kind != ExpertRequestKind::COLD_MISS || request.vram_slot < 0) {
                    throw std::runtime_error(
                        "ExpertTierLoader: a drained prefill resident was not cold at restore");
                }
                operation_ids.push_back(request.operation_id);
                slots.push_back(request.vram_slot);
                expert_ids.emplace_back(gid / per_layer, gid % per_layer);
                buffers.emplace_back(format.payload_bytes, format.sector_size);
            }
            std::vector<uint8_t*> destinations;
            destinations.reserve(buffers.size());
            for (auto& buffer : buffers) {
                destinations.push_back(static_cast<uint8_t*>(buffer.data()));
            }
            services_.direct_io->read_blocking(*services_.source, expert_ids, destinations);
            for (size_t i = 0; i < slots.size(); ++i) {
                services_.payload_pool->upload_from_host_expert(
                    static_cast<uint32_t>(slots[i]), destinations[i], services_.compute);
            }
            CHECK_HIP(hipStreamSynchronize(services_.compute));
            for (size_t i = 0; i < operation_ids.size(); ++i) {
                services_.registry->complete_request(operation_ids[i]);
                services_.registry->release_lease(restore[start + i]);
            }
        }
    }

private:
    Services services_{};
};

} // namespace aeon::core
