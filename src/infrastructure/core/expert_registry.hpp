#pragma once

#include <cstdint>
#include <list>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace aeon::core {

enum class ExpertTier : uint8_t {
    HOT_VRAM = 0,
    WARM_HOST = 1,
    COLD_NVME = 2
};

enum class ExpertOperation : uint8_t {
    NONE = 0,
    IO_PENDING = 1,
    PROMOTION_PENDING = 2,
    DEMOTION_PENDING = 3
};

enum class ExpertGpuTransfer : uint8_t {
    NONE = 0,
    H2D_PENDING = 1,
    D2H_PENDING = 2
};

enum class ExpertDemotionDropReason : uint8_t {
    NONE = 0,
    QUEUE_PRESSURE = 1,
    WARM_DESTINATION_UNAVAILABLE = 2
};

inline const char* expert_demotion_drop_reason_name(ExpertDemotionDropReason reason) {
    switch (reason) {
    case ExpertDemotionDropReason::QUEUE_PRESSURE: return "queue_pressure";
    case ExpertDemotionDropReason::WARM_DESTINATION_UNAVAILABLE:
        return "warm_destination_unavailable";
    case ExpertDemotionDropReason::NONE: return "unspecified";
    }
    return "unspecified";
}

enum class ExpertPublication : uint8_t {
    PUBLISHED = 0,
    UNPUBLISHED = 1
};

enum class ExpertSlotState : uint8_t {
    UNALLOCATED = 0,
    ACTIVE = 1,
    RECLAIMABLE = 2
};

enum class ExpertRequestKind : uint8_t {
    HOT_HIT = 0,
    WARM_PROMOTION = 1,
    COLD_MISS = 2,
    PENDING = 3
};

struct ExpertCatalogEntry {
    uint32_t global_expert_id{0};
    uint16_t layer_id{0};
    uint16_t expert_id{0};

    ExpertTier owner{ExpertTier::COLD_NVME};
    int32_t slot_idx{-1};
    int32_t pending_slot_idx{-1};

    ExpertOperation operation{ExpertOperation::NONE};
    ExpertGpuTransfer gpu_transfer{ExpertGpuTransfer::NONE};
    ExpertPublication publication{ExpertPublication::PUBLISHED};
    ExpertSlotState slot_state{ExpertSlotState::UNALLOCATED};
    ExpertDemotionDropReason demotion_drop_reason{ExpertDemotionDropReason::NONE};
    uint64_t operation_id{0};
    uint32_t lease_count{0};
    bool in_lru{false};

    uint64_t activation_count{0};
    uint64_t last_step_used{0};
    float moving_frequency{0.0f};

    std::list<uint32_t>::iterator lru_it;
};

struct ExpertDemotionReservation {
    uint64_t operation_id{0};
    uint32_t victim_gid{0};
    uint32_t source_vram_slot{0};
    uint32_t destination_host_slot{0};
};

struct ExpertRequestReservation {
    ExpertRequestKind kind{ExpertRequestKind::COLD_MISS};
    ExpertTier source_tier{ExpertTier::COLD_NVME};
    uint32_t global_expert_id{0};
    uint64_t operation_id{0};
    int32_t vram_slot{-1};
    int32_t source_host_slot{-1};
    bool lease_acquired{false};
    std::optional<ExpertDemotionReservation> demotion;
};

class ExpertRegistry {
public:
    uint32_t num_layers{43};
    uint32_t experts_per_layer{256};
    uint32_t total_experts{11008};
    uint32_t vram_capacity{0};
    uint32_t host_capacity{0};

    std::vector<ExpertCatalogEntry> catalog;
    std::vector<int32_t> vram_slots;
    std::vector<uint32_t> free_vram_slots;
    std::vector<int32_t> host_slots;
    std::vector<uint32_t> free_host_slots;
    std::vector<uint64_t> vram_slot_reservations;
    std::vector<uint64_t> host_slot_reservations;

    std::list<uint32_t> hot_vram_lru;
    std::list<uint32_t> warm_host_lru;

    uint64_t hits_hot{0};
    uint64_t hits_warm{0};
    uint64_t misses_cold{0};
    uint64_t demotion_attempts{0};
    uint64_t demotion_completions{0};
    uint64_t demotion_drops{0};
    uint64_t next_operation_id{1};
    uint64_t pending_demotion_count{0};

    ExpertRegistry() = default;

    ExpertRegistry(
        uint32_t layers,
        uint32_t experts_layer,
        uint32_t vram_cap,
        uint32_t host_cap,
        bool preload_warm_host = true
    ) {
        init(layers, experts_layer, vram_cap, host_cap, preload_warm_host);
    }

    void init(
        uint32_t layers,
        uint32_t experts_layer,
        uint32_t vram_cap,
        uint32_t host_cap,
        bool preload_warm_host = true
    ) {
        if (vram_cap == 0) {
            throw std::invalid_argument("ExpertRegistry: hot_capacity must be greater than zero");
        }

        num_layers = layers;
        experts_per_layer = experts_layer;
        total_experts = layers * experts_layer;
        vram_capacity = vram_cap;
        host_capacity = host_cap;

        catalog.assign(total_experts, ExpertCatalogEntry{});
        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            for (uint32_t expert = 0; expert < experts_per_layer; ++expert) {
                const uint32_t gid = get_global_id(layer, expert);
                auto& entry = catalog[gid];
                entry.global_expert_id = gid;
                entry.layer_id = static_cast<uint16_t>(layer);
                entry.expert_id = static_cast<uint16_t>(expert);
            }
        }

        vram_slots.assign(vram_capacity, -1);
        vram_slot_reservations.assign(vram_capacity, 0);
        free_vram_slots.clear();
        for (uint32_t slot = vram_capacity; slot-- > 0;) {
            free_vram_slots.push_back(slot);
        }
        hot_vram_lru.clear();

        host_slots.assign(host_capacity, -1);
        host_slot_reservations.assign(host_capacity, 0);
        free_host_slots.clear();
        for (uint32_t slot = host_capacity; slot-- > 0;) {
            free_host_slots.push_back(slot);
        }
        warm_host_lru.clear();

        hits_hot = 0;
        hits_warm = 0;
        misses_cold = 0;
        demotion_attempts = 0;
        demotion_completions = 0;
        demotion_drops = 0;
        next_operation_id = 1;
        pending_demotion_count = 0;

        populate_round_robin(preload_warm_host);
        validate_invariants();
    }

    uint32_t get_global_id(uint32_t layer_id, uint32_t expert_id) const {
        if (layer_id >= num_layers || expert_id >= experts_per_layer) {
            throw std::out_of_range("ExpertRegistry: expert coordinates are out of range");
        }
        return layer_id * experts_per_layer + expert_id;
    }

    ExpertRequestReservation reserve_request(
        uint32_t layer_id,
        uint32_t expert_id,
        uint64_t current_step,
        uint64_t demotion_queue_capacity
    ) {
        return reserve_request(get_global_id(layer_id, expert_id), current_step, demotion_queue_capacity);
    }

    ExpertRequestReservation reserve_request(
        uint32_t gid,
        uint64_t current_step,
        uint64_t demotion_queue_capacity
    ) {
        if (gid >= catalog.size()) {
            throw std::out_of_range("ExpertRegistry: global expert ID is out of range");
        }

        auto& entry = catalog[gid];
        record_activation(entry, current_step);

        if (entry.operation != ExpertOperation::NONE ||
            entry.publication == ExpertPublication::UNPUBLISHED) {
            if (entry.operation == ExpertOperation::NONE || entry.operation_id == 0) {
                throw std::logic_error("ExpertRegistry: unpublished entry has no pending operation");
            }
            ++entry.lease_count;
            return ExpertRequestReservation{
                ExpertRequestKind::PENDING,
                entry.owner,
                gid,
                entry.operation_id,
                entry.pending_slot_idx,
                entry.owner == ExpertTier::WARM_HOST ? entry.slot_idx : -1,
                true,
                std::nullopt
            };
        }

        if (entry.owner == ExpertTier::HOT_VRAM) {
            ++hits_hot;
            touch_lru(entry, hot_vram_lru, gid);
            ++entry.lease_count;
            return ExpertRequestReservation{
                ExpertRequestKind::HOT_HIT,
                ExpertTier::HOT_VRAM,
                gid,
                0,
                entry.slot_idx,
                -1,
                true,
                std::nullopt
            };
        }

        const uint64_t operation_id = next_operation_id++;
        const ExpertTier source_tier = entry.owner;
        const auto destination = reserve_vram_destination(
            gid, operation_id, demotion_queue_capacity);

        int32_t source_host_slot = -1;
        if (source_tier == ExpertTier::WARM_HOST) {
            ++hits_warm;
            source_host_slot = entry.slot_idx;
            remove_from_lru(entry, warm_host_lru, gid);
            host_slot_reservations[static_cast<size_t>(source_host_slot)] = operation_id;
            entry.operation = ExpertOperation::PROMOTION_PENDING;
        } else {
            ++misses_cold;
            entry.operation = ExpertOperation::IO_PENDING;
        }

        entry.operation_id = operation_id;
        entry.gpu_transfer = ExpertGpuTransfer::H2D_PENDING;
        entry.publication = ExpertPublication::UNPUBLISHED;
        entry.slot_state = ExpertSlotState::ACTIVE;
        entry.lease_count++;
        entry.pending_slot_idx = static_cast<int32_t>(destination.vram_slot);

        ExpertRequestReservation request{
            source_tier == ExpertTier::WARM_HOST
                ? ExpertRequestKind::WARM_PROMOTION
                : ExpertRequestKind::COLD_MISS,
            source_tier,
            gid,
            operation_id,
            static_cast<int32_t>(destination.vram_slot),
            source_host_slot,
            true,
            destination.demotion
        };

        validate_invariants();
        return request;
    }

    void complete_demotion(uint64_t operation_id) {
        auto* victim = find_operation_victim(operation_id);
        if (victim == nullptr || victim->gpu_transfer != ExpertGpuTransfer::D2H_PENDING) {
            throw std::logic_error("ExpertRegistry: demotion completion does not match a pending D2H");
        }

        const int32_t source_slot = victim->slot_idx;
        const int32_t destination_slot = find_reserved_host_slot(operation_id);
        if (source_slot < 0 || destination_slot < 0) {
            throw std::logic_error("ExpertRegistry: demotion completion lost a physical slot");
        }

        host_slots[static_cast<size_t>(destination_slot)] = static_cast<int32_t>(victim->global_expert_id);
        host_slot_reservations[static_cast<size_t>(destination_slot)] = 0;
        victim->owner = ExpertTier::WARM_HOST;
        victim->slot_idx = destination_slot;
        victim->pending_slot_idx = -1;
        victim->operation = ExpertOperation::NONE;
        victim->gpu_transfer = ExpertGpuTransfer::NONE;
        victim->publication = ExpertPublication::PUBLISHED;
        victim->slot_state = ExpertSlotState::ACTIVE;
        victim->demotion_drop_reason = ExpertDemotionDropReason::NONE;
        push_lru_front(*victim, warm_host_lru);

        vram_slots[static_cast<size_t>(source_slot)] = -1;
        --pending_demotion_count;
        ++demotion_completions;
        validate_invariants();
    }

    void drop_demotion(uint64_t operation_id) {
        auto* victim = find_operation_victim(operation_id);
        if (victim == nullptr || victim->operation != ExpertOperation::DEMOTION_PENDING) {
            throw std::logic_error("ExpertRegistry: demotion drop does not match a pending victim");
        }
        if (victim->gpu_transfer == ExpertGpuTransfer::D2H_PENDING) {
            throw std::logic_error("ExpertRegistry: submitted D2H cannot be dropped");
        }
        const int32_t destination_slot = find_reserved_host_slot(operation_id);
        if (destination_slot >= 0) {
            release_reserved_host_slot(static_cast<uint32_t>(destination_slot));
        }
        victim->gpu_transfer = ExpertGpuTransfer::NONE;
        victim->demotion_drop_reason = ExpertDemotionDropReason::NONE;
        ++demotion_drops;
        validate_invariants();
    }

    void fail_demotion(uint64_t operation_id) {
        auto* victim = find_operation_victim(operation_id);
        if (victim == nullptr || victim->operation != ExpertOperation::DEMOTION_PENDING) {
            throw std::logic_error("ExpertRegistry: demotion failure does not match a pending victim");
        }
        if (victim->gpu_transfer != ExpertGpuTransfer::D2H_PENDING) {
            throw std::logic_error("ExpertRegistry: unsent demotion is not a transfer failure");
        }

        const int32_t destination_slot = find_reserved_host_slot(operation_id);
        if (destination_slot >= 0) {
            release_reserved_host_slot(static_cast<uint32_t>(destination_slot));
        }
        const int32_t source_slot = victim->slot_idx;
        victim->owner = ExpertTier::COLD_NVME;
        victim->slot_idx = -1;
        victim->pending_slot_idx = -1;
        victim->operation = ExpertOperation::NONE;
        victim->gpu_transfer = ExpertGpuTransfer::NONE;
        victim->demotion_drop_reason = ExpertDemotionDropReason::NONE;
        victim->publication = ExpertPublication::PUBLISHED;
        victim->slot_state = ExpertSlotState::UNALLOCATED;
        if (source_slot >= 0) {
            vram_slots[static_cast<size_t>(source_slot)] = -1;
        }
        --pending_demotion_count;
        ++demotion_drops;
        validate_invariants();
    }

    void complete_request(uint64_t operation_id) {
        auto* incoming = find_incoming_operation(operation_id);
        if (incoming == nullptr || incoming->gpu_transfer != ExpertGpuTransfer::H2D_PENDING) {
            throw std::logic_error("ExpertRegistry: H2D completion does not match a pending request");
        }

        const int32_t destination_slot = incoming->pending_slot_idx;
        if (destination_slot < 0 ||
            vram_slot_reservations[static_cast<size_t>(destination_slot)] != operation_id) {
            throw std::logic_error("ExpertRegistry: request completion lost its VRAM reservation");
        }

        auto* victim = find_operation_victim(operation_id);
        if (victim != nullptr) {
            if (victim->gpu_transfer == ExpertGpuTransfer::D2H_PENDING) {
                throw std::logic_error("ExpertRegistry: H2D completed before its D2H dependency");
            }
            const int32_t victim_slot = victim->slot_idx;
            if (victim_slot >= 0 &&
                vram_slots[static_cast<size_t>(victim_slot)] ==
                    static_cast<int32_t>(victim->global_expert_id)) {
                vram_slots[static_cast<size_t>(victim_slot)] = -1;
            }
            victim->owner = ExpertTier::COLD_NVME;
            victim->slot_idx = -1;
            victim->pending_slot_idx = -1;
            victim->operation = ExpertOperation::NONE;
            victim->gpu_transfer = ExpertGpuTransfer::NONE;
            victim->publication = ExpertPublication::PUBLISHED;
            victim->slot_state = ExpertSlotState::UNALLOCATED;
            victim->demotion_drop_reason = ExpertDemotionDropReason::NONE;
        }

        if (incoming->owner == ExpertTier::WARM_HOST) {
            const int32_t source_slot = incoming->slot_idx;
            if (source_slot < 0 ||
                host_slots[static_cast<size_t>(source_slot)] !=
                    static_cast<int32_t>(incoming->global_expert_id)) {
                throw std::logic_error("ExpertRegistry: promotion completion lost its Warm source");
            }
            host_slots[static_cast<size_t>(source_slot)] = -1;
            host_slot_reservations[static_cast<size_t>(source_slot)] = 0;
            free_host_slots.push_back(static_cast<uint32_t>(source_slot));
        }

        incoming->owner = ExpertTier::HOT_VRAM;
        incoming->slot_idx = destination_slot;
        incoming->pending_slot_idx = -1;
        incoming->operation = ExpertOperation::NONE;
        incoming->gpu_transfer = ExpertGpuTransfer::NONE;
        incoming->publication = ExpertPublication::PUBLISHED;
        incoming->slot_state = ExpertSlotState::ACTIVE;
        vram_slots[static_cast<size_t>(destination_slot)] =
            static_cast<int32_t>(incoming->global_expert_id);
        vram_slot_reservations[static_cast<size_t>(destination_slot)] = 0;
        push_lru_front(*incoming, hot_vram_lru);
        validate_invariants();
    }

    void fail_request(uint64_t operation_id) {
        auto* incoming = find_incoming_operation(operation_id);
        if (incoming == nullptr) {
            throw std::logic_error("ExpertRegistry: request failure does not match a pending request");
        }

        const int32_t destination_slot = incoming->pending_slot_idx;
        auto* victim = find_operation_victim(operation_id);
        if (victim != nullptr && victim->gpu_transfer == ExpertGpuTransfer::D2H_PENDING) {
            throw std::logic_error("ExpertRegistry: request failed before submitted D2H completed");
        }

        if (destination_slot >= 0) {
            vram_slot_reservations[static_cast<size_t>(destination_slot)] = 0;
            if (vram_slots[static_cast<size_t>(destination_slot)] == -1) {
                free_vram_slots.push_back(static_cast<uint32_t>(destination_slot));
            }
        }

        if (victim != nullptr) {
            const int32_t victim_slot = victim->slot_idx;
            if (victim_slot >= 0 &&
                vram_slots[static_cast<size_t>(victim_slot)] ==
                    static_cast<int32_t>(victim->global_expert_id)) {
                vram_slots[static_cast<size_t>(victim_slot)] = -1;
                free_vram_slots.push_back(static_cast<uint32_t>(victim_slot));
            }
            victim->owner = ExpertTier::COLD_NVME;
            victim->slot_idx = -1;
            victim->pending_slot_idx = -1;
            victim->operation = ExpertOperation::NONE;
            victim->gpu_transfer = ExpertGpuTransfer::NONE;
            victim->publication = ExpertPublication::PUBLISHED;
            victim->slot_state = ExpertSlotState::UNALLOCATED;
            victim->demotion_drop_reason = ExpertDemotionDropReason::NONE;
        }

        if (incoming->owner == ExpertTier::WARM_HOST) {
            const int32_t source_slot = incoming->slot_idx;
            host_slot_reservations[static_cast<size_t>(source_slot)] = 0;
            push_lru_front(*incoming, warm_host_lru);
        }
        incoming->pending_slot_idx = -1;
        incoming->operation = ExpertOperation::NONE;
        incoming->gpu_transfer = ExpertGpuTransfer::NONE;
        incoming->publication = ExpertPublication::PUBLISHED;
        validate_invariants();
    }

    void release_lease(uint32_t gid) {
        if (gid >= catalog.size() || catalog[gid].lease_count == 0) {
            throw std::logic_error("ExpertRegistry: releasing an unleased expert");
        }
        --catalog[gid].lease_count;
    }

    void reset_counters() {
        hits_hot = 0;
        hits_warm = 0;
        misses_cold = 0;
        demotion_attempts = 0;
        demotion_completions = 0;
        demotion_drops = 0;
    }

    uint64_t pending_transfer_count() const {
        std::unordered_set<uint64_t> operations;
        for (const auto& entry : catalog) {
            if (entry.operation != ExpertOperation::NONE && entry.operation_id != 0) {
                operations.insert(entry.operation_id);
            }
        }
        return operations.size();
    }

    uint32_t published_hot_slots() const {
        return static_cast<uint32_t>(hot_vram_lru.size());
    }

    uint32_t published_warm_slots() const {
        return static_cast<uint32_t>(warm_host_lru.size());
    }

    bool invariants_hold() const noexcept {
        try {
            validate_invariants();
            return true;
        } catch (...) {
            return false;
        }
    }

    void validate_invariants() const {
        if (vram_slots.size() != vram_capacity || host_slots.size() != host_capacity ||
            vram_slot_reservations.size() != vram_capacity ||
            host_slot_reservations.size() != host_capacity) {
            throw std::logic_error("ExpertRegistry: physical slot vectors have inconsistent sizes");
        }

        std::vector<uint8_t> hot_seen(vram_capacity, 0);
        std::vector<uint8_t> warm_seen(host_capacity, 0);
        for (const auto& entry : catalog) {
            if (entry.owner == ExpertTier::HOT_VRAM) {
                if (entry.slot_idx < 0 || static_cast<size_t>(entry.slot_idx) >= vram_capacity ||
                    vram_slots[static_cast<size_t>(entry.slot_idx)] !=
                        static_cast<int32_t>(entry.global_expert_id)) {
                    throw std::logic_error("ExpertRegistry: Hot entry does not map to its VRAM slot");
                }
            } else if (entry.owner == ExpertTier::WARM_HOST) {
                if (entry.slot_idx < 0 || static_cast<size_t>(entry.slot_idx) >= host_capacity ||
                    host_slots[static_cast<size_t>(entry.slot_idx)] !=
                        static_cast<int32_t>(entry.global_expert_id)) {
                    throw std::logic_error("ExpertRegistry: Warm entry does not map to its host slot");
                }
            } else if (entry.slot_idx != -1) {
                throw std::logic_error("ExpertRegistry: Cold entry owns a physical slot");
            }

            if (entry.operation != ExpertOperation::NONE && entry.operation_id == 0) {
                throw std::logic_error("ExpertRegistry: pending entry has no operation ID");
            }
            if (entry.operation == ExpertOperation::NONE &&
                entry.gpu_transfer != ExpertGpuTransfer::NONE) {
                throw std::logic_error("ExpertRegistry: idle entry has a pending GPU transfer");
            }
            if (entry.operation != ExpertOperation::NONE && entry.pending_slot_idx >= 0 &&
                (static_cast<size_t>(entry.pending_slot_idx) >= vram_capacity ||
                 vram_slot_reservations[static_cast<size_t>(entry.pending_slot_idx)] != entry.operation_id)) {
                throw std::logic_error("ExpertRegistry: pending entry does not own its VRAM reservation");
            }
        }

        for (uint32_t slot = 0; slot < vram_capacity; ++slot) {
            const int32_t gid = vram_slots[slot];
            if (gid >= 0) {
                if (static_cast<size_t>(gid) >= catalog.size() ||
                    catalog[static_cast<size_t>(gid)].owner != ExpertTier::HOT_VRAM ||
                    catalog[static_cast<size_t>(gid)].slot_idx != static_cast<int32_t>(slot) ||
                    hot_seen[slot]) {
                    throw std::logic_error("ExpertRegistry: VRAM slot map is not bijective");
                }
                hot_seen[slot] = 1;
            }
            if (vram_slot_reservations[slot] != 0 && gid < 0) {
                bool reservation_found = false;
                for (const auto& entry : catalog) {
                    if (entry.operation_id == vram_slot_reservations[slot] &&
                        entry.pending_slot_idx == static_cast<int32_t>(slot)) {
                        reservation_found = true;
                        break;
                    }
                }
                if (!reservation_found) {
                    throw std::logic_error("ExpertRegistry: orphaned VRAM reservation");
                }
            }
        }

        for (uint32_t slot = 0; slot < host_capacity; ++slot) {
            const int32_t gid = host_slots[slot];
            if (gid >= 0) {
                if (static_cast<size_t>(gid) >= catalog.size() ||
                    catalog[static_cast<size_t>(gid)].owner != ExpertTier::WARM_HOST ||
                    catalog[static_cast<size_t>(gid)].slot_idx != static_cast<int32_t>(slot) ||
                    warm_seen[slot]) {
                    throw std::logic_error("ExpertRegistry: host slot map is not bijective");
                }
                warm_seen[slot] = 1;
            }
            if (host_slot_reservations[slot] != 0) {
                bool reservation_found = false;
                for (const auto& entry : catalog) {
                    if (entry.operation_id == host_slot_reservations[slot] &&
                        (entry.slot_idx == static_cast<int32_t>(slot) ||
                         entry.operation == ExpertOperation::DEMOTION_PENDING)) {
                        reservation_found = true;
                        break;
                    }
                }
                if (!reservation_found) {
                    throw std::logic_error("ExpertRegistry: orphaned host reservation");
                }
            }
        }

        validate_lru(hot_vram_lru, ExpertTier::HOT_VRAM);
        validate_lru(warm_host_lru, ExpertTier::WARM_HOST);
    }

private:
    struct VramDestination {
        uint32_t vram_slot{0};
        std::optional<ExpertDemotionReservation> demotion;
    };

    void record_activation(ExpertCatalogEntry& entry, uint64_t current_step) {
        ++entry.activation_count;
        entry.last_step_used = current_step;
        constexpr float alpha = 0.1f;
        entry.moving_frequency = (1.0f - alpha) * entry.moving_frequency + alpha;
    }

    static void push_lru_front(ExpertCatalogEntry& entry, std::list<uint32_t>& lru) {
        if (entry.in_lru) {
            return;
        }
        lru.push_front(entry.global_expert_id);
        entry.lru_it = lru.begin();
        entry.in_lru = true;
    }

    static void remove_from_lru(
        ExpertCatalogEntry& entry,
        std::list<uint32_t>& lru,
        uint32_t gid
    ) {
        if (!entry.in_lru) {
            return;
        }
        if (*entry.lru_it != gid) {
            throw std::logic_error("ExpertRegistry: LRU iterator does not match catalog entry");
        }
        lru.erase(entry.lru_it);
        entry.in_lru = false;
    }

    static void touch_lru(
        ExpertCatalogEntry& entry,
        std::list<uint32_t>& lru,
        uint32_t gid
    ) {
        remove_from_lru(entry, lru, gid);
        push_lru_front(entry, lru);
    }

    VramDestination reserve_vram_destination(
        uint32_t incoming_gid,
        uint64_t operation_id,
        uint64_t demotion_queue_capacity
    ) {
        uint32_t vram_slot = 0;
        ExpertCatalogEntry* victim = nullptr;

        if (!free_vram_slots.empty()) {
            vram_slot = free_vram_slots.back();
            free_vram_slots.pop_back();
        } else {
            for (auto it = hot_vram_lru.rbegin(); it != hot_vram_lru.rend(); ++it) {
                auto& candidate = catalog[*it];
                if (candidate.global_expert_id == incoming_gid ||
                    candidate.operation != ExpertOperation::NONE ||
                    candidate.gpu_transfer != ExpertGpuTransfer::NONE ||
                    candidate.publication != ExpertPublication::PUBLISHED ||
                    candidate.lease_count != 0) {
                    continue;
                }
                victim = &candidate;
                break;
            }
            if (victim == nullptr) {
                uint32_t leased_hot = 0;
                uint32_t pending_hot = 0;
                for (const auto& entry : catalog) {
                    if (entry.owner != ExpertTier::HOT_VRAM) continue;
                    if (entry.lease_count != 0) ++leased_hot;
                    if (entry.operation != ExpertOperation::NONE ||
                        entry.gpu_transfer != ExpertGpuTransfer::NONE) {
                        ++pending_hot;
                    }
                }
                throw std::runtime_error(
                    "ExpertRegistry: no reclaimable Hot VRAM slot is available (hot=" +
                    std::to_string(hot_vram_lru.size()) + ", leased=" +
                    std::to_string(leased_hot) + ", pending=" +
                    std::to_string(pending_hot) + ")");
            }
            vram_slot = static_cast<uint32_t>(victim->slot_idx);
            remove_from_lru(*victim, hot_vram_lru, victim->global_expert_id);
        }

        vram_slot_reservations[vram_slot] = operation_id;
        if (victim == nullptr) {
            return VramDestination{vram_slot, std::nullopt};
        }

        victim->operation = ExpertOperation::DEMOTION_PENDING;
        victim->operation_id = operation_id;
        victim->publication = ExpertPublication::UNPUBLISHED;
        victim->slot_state = ExpertSlotState::ACTIVE;
        victim->pending_slot_idx = -1;

        if (host_capacity == 0) {
            victim->gpu_transfer = ExpertGpuTransfer::NONE;
            return VramDestination{vram_slot, std::nullopt};
        }

        ++demotion_attempts;
        if (pending_demotion_count >= demotion_queue_capacity) {
            victim->gpu_transfer = ExpertGpuTransfer::NONE;
            victim->demotion_drop_reason = ExpertDemotionDropReason::QUEUE_PRESSURE;
            ++demotion_drops;
            return VramDestination{vram_slot, std::nullopt};
        }

        const int32_t destination_host_slot = reserve_warm_destination(incoming_gid, operation_id);
        if (destination_host_slot < 0) {
            victim->gpu_transfer = ExpertGpuTransfer::NONE;
            victim->demotion_drop_reason = ExpertDemotionDropReason::WARM_DESTINATION_UNAVAILABLE;
            ++demotion_drops;
            return VramDestination{vram_slot, std::nullopt};
        }

        victim->gpu_transfer = ExpertGpuTransfer::D2H_PENDING;
        victim->demotion_drop_reason = ExpertDemotionDropReason::NONE;
        ++pending_demotion_count;
        return VramDestination{
            vram_slot,
            ExpertDemotionReservation{
                operation_id,
                victim->global_expert_id,
                vram_slot,
                static_cast<uint32_t>(destination_host_slot)
            }
        };
    }

    int32_t reserve_warm_destination(uint32_t excluded_gid, uint64_t operation_id) {
        if (!free_host_slots.empty()) {
            const uint32_t slot = free_host_slots.back();
            free_host_slots.pop_back();
            host_slot_reservations[slot] = operation_id;
            return static_cast<int32_t>(slot);
        }

        for (auto it = warm_host_lru.rbegin(); it != warm_host_lru.rend(); ++it) {
            auto& candidate = catalog[*it];
            if (candidate.global_expert_id == excluded_gid ||
                candidate.operation != ExpertOperation::NONE ||
                candidate.gpu_transfer != ExpertGpuTransfer::NONE ||
                candidate.publication != ExpertPublication::PUBLISHED ||
                candidate.lease_count != 0) {
                continue;
            }
            const int32_t slot = candidate.slot_idx;
            remove_from_lru(candidate, warm_host_lru, candidate.global_expert_id);
            candidate.owner = ExpertTier::COLD_NVME;
            candidate.slot_idx = -1;
            candidate.pending_slot_idx = -1;
            candidate.slot_state = ExpertSlotState::UNALLOCATED;
            host_slots[static_cast<size_t>(slot)] = -1;
            host_slot_reservations[static_cast<size_t>(slot)] = operation_id;
            return slot;
        }
        return -1;
    }

    ExpertCatalogEntry* find_incoming_operation(uint64_t operation_id) {
        for (auto& entry : catalog) {
            if (entry.operation_id == operation_id &&
                (entry.operation == ExpertOperation::IO_PENDING ||
                 entry.operation == ExpertOperation::PROMOTION_PENDING)) {
                return &entry;
            }
        }
        return nullptr;
    }

    ExpertCatalogEntry* find_operation_victim(uint64_t operation_id) {
        for (auto& entry : catalog) {
            if (entry.operation_id == operation_id &&
                entry.operation == ExpertOperation::DEMOTION_PENDING) {
                return &entry;
            }
        }
        return nullptr;
    }

    int32_t find_reserved_host_slot(uint64_t operation_id) const {
        for (uint32_t slot = 0; slot < host_capacity; ++slot) {
            if (host_slot_reservations[slot] == operation_id && host_slots[slot] < 0) {
                return static_cast<int32_t>(slot);
            }
        }
        for (uint32_t slot = 0; slot < host_capacity; ++slot) {
            if (host_slot_reservations[slot] == operation_id) {
                return static_cast<int32_t>(slot);
            }
        }
        return -1;
    }

    void release_reserved_host_slot(uint32_t slot) {
        host_slots[slot] = -1;
        host_slot_reservations[slot] = 0;
        free_host_slots.push_back(slot);
    }

    void validate_lru(const std::list<uint32_t>& lru, ExpertTier tier) const {
        std::vector<uint8_t> seen(catalog.size(), 0);
        for (uint32_t gid : lru) {
            if (gid >= catalog.size() || seen[gid] || !catalog[gid].in_lru ||
                catalog[gid].owner != tier ||
                catalog[gid].publication != ExpertPublication::PUBLISHED ||
                catalog[gid].operation != ExpertOperation::NONE) {
                throw std::logic_error("ExpertRegistry: LRU contains an invalid catalog entry");
            }
            seen[gid] = 1;
        }
        for (const auto& entry : catalog) {
            if (entry.owner == tier && entry.in_lru != (seen[entry.global_expert_id] != 0)) {
                throw std::logic_error("ExpertRegistry: LRU membership disagrees with catalog entry");
            }
        }
    }

    void populate_round_robin(bool preload_warm_host) {
        uint32_t vram_assigned = 0;
        uint32_t host_assigned = 0;
        for (uint32_t expert = 0; expert < experts_per_layer; ++expert) {
            for (uint32_t layer = 0; layer < num_layers; ++layer) {
                const uint32_t gid = get_global_id(layer, expert);
                auto& entry = catalog[gid];
                if (vram_assigned < vram_capacity) {
                    const uint32_t slot = free_vram_slots.back();
                    free_vram_slots.pop_back();
                    entry.owner = ExpertTier::HOT_VRAM;
                    entry.slot_idx = static_cast<int32_t>(slot);
                    entry.slot_state = ExpertSlotState::ACTIVE;
                    vram_slots[slot] = static_cast<int32_t>(gid);
                    push_lru_front(entry, hot_vram_lru);
                    ++vram_assigned;
                } else if (preload_warm_host && host_assigned < host_capacity) {
                    const uint32_t slot = free_host_slots.back();
                    free_host_slots.pop_back();
                    entry.owner = ExpertTier::WARM_HOST;
                    entry.slot_idx = static_cast<int32_t>(slot);
                    entry.slot_state = ExpertSlotState::ACTIVE;
                    host_slots[slot] = static_cast<int32_t>(gid);
                    push_lru_front(entry, warm_host_lru);
                    ++host_assigned;
                } else if (preload_warm_host) {
                    return;
                }
            }
        }
    }
};

} // namespace aeon::core