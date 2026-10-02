#pragma once

// -----------------------------------------------------------------------------
// The expert registry: shared expert residency and usage tracking across the
// VRAM / host / NVMe tiers, plus the prefill streaming switch.
//
// This header owns the class declaration. The out-of-line definitions of the
// Warm/staging partition, the prefill residency and the invariant audit live in
// `warm_partition.hpp`, `prefill_residency.hpp` and
// `expert_registry_validation.hpp`, which are included at the foot of this file;
// the data vocabulary is in `expert_registry_types.hpp`.
// -----------------------------------------------------------------------------

#include "infrastructure/expert/residency/expert_registry_types.hpp"

#include <algorithm>
#include <cstdint>
#include <list>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace aeon::core {

class ExpertRegistry {
public:
    // How a prefill obtains VRAM for its loads. Both policies share the same marks
    // and sparing release; only the no-free-slot case differs.
    enum class PrefillAlloc : uint8_t {
        // The sweep: whole-layer loads in layer order. Allocation is free-list only,
        // because the sweep's order decides what is dead and a preserved resident is
        // never the pass's to spend — a request with no free slot is a driver defect.
        FreeListOnly = 0,
        // The routed bank: route-aware, cached admission. A full free list is met by
        // releasing the worst-ranked **non-preserved** resident, as decode would evict,
        // so a layer's working set can be built up to its bank without stalling.
        BoundedEvict = 1
    };

    uint32_t num_layers{43};
    uint32_t experts_per_layer{256};
    uint32_t total_experts{11008};
    uint32_t vram_capacity{0};
    uint32_t host_capacity{0};
    // How many host slots the current partition exposes to Warm. The arrays above are
    // sized to `host_capacity`, which is the **maximum** (the region minus the base
    // staging), because the Warm/staging boundary moves: a swept prefill hands the
    // corridor more of the shared region and takes it back when the window ends.
    // Keeping the arrays at their maximum is what makes a boundary move a free-list
    // edit rather than a reallocation or a catalog re-map. Equals `host_capacity`
    // until the first move.
    uint32_t host_capacity_usable{0};

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
    // Frozen prefill: how many non-destructive Warm copies were issued. A gate
    // reads it to prove the frozen prefill really read Warm instead of bypassing
    // it to NVMe, and that the run is therefore non-vacuous.
    uint64_t shadow_copies{0};
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
        host_capacity_usable = host_capacity;
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
        shadow_copies = 0;
        next_operation_id = 1;
        pending_demotion_count = 0;
        shadow_slot_of_expert_.assign(total_experts, -1);
        shadow_lru_.clear();
        warm_frozen_ = false;
        prefill_stream_ = false;

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
        uint64_t demotion_queue_capacity,
        bool stage_only = false
    ) {
        return reserve_request_by_gid(
            get_global_id(layer_id, expert_id), current_step, demotion_queue_capacity,
            stage_only);
    }

    // The same request addressed by **global** expert id, at an arity the pair form
    // cannot be called with — three arguments, while the pair form needs four. The
    // distinct arity is the point: the two were overloads of one name and **both** take
    // four integers, so once `stage_only` gave the pair form a default a four-argument
    // call became ambiguous (and a target that happened not to be rebuilt carried the
    // breakage unnoticed). Keeping this form at three arguments removes the overlap
    // without renaming anything a caller already uses.
    ExpertRequestReservation reserve_request(
        uint32_t gid,
        uint64_t current_step,
        uint64_t demotion_queue_capacity
    ) {
        return reserve_request_by_gid(gid, current_step, demotion_queue_capacity, false);
    }

    // The gid form with the deferral, named so it cannot take part in overload
    // resolution against the pair form.
    //
    // `stage_only` reserves the operation with no VRAM destination at reservation:
    // the bytes land in the staging arena and wait there until
    // `attach_vram_destination` gives them a slot, at the moment the copy can run.
    // This is what lets the supply's read leg and copy leg be bounded separately (a
    // read is limited by staging, a copy by VRAM) instead of both by VRAM.
    //
    // It applies to a **cold read** and to a **Warm shadow** alike. It is ignored for
    // any other source: a demotion needs its destination decided at reservation, and
    // a non-frozen Warm promotion has no staging leg at all.
    ExpertRequestReservation reserve_request_by_gid(
        uint32_t gid,
        uint64_t current_step,
        uint64_t demotion_queue_capacity,
        bool stage_only = false
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

        // Frozen prefill. Warm's resident set must survive a prefill, so a
        // Warm-resident expert is **copied** into VRAM rather than promoted: the
        // catalog entry keeps `owner == WARM_HOST` and its host slot, and the VRAM
        // copy is recorded as a **shadow residency** on that same entry. Leaving the
        // mode — `end_prefill_stream()`, or `set_warm_frozen(false)` on the legacy
        // path — releases every idle shadow. A shadow is legal **only** while frozen,
        // which `validate_invariants` enforces so decode never sees a second owner.
        if (warm_frozen_ && entry.owner == ExpertTier::WARM_HOST) {
            if (entry.shadow_vram_slot >= 0) {
                // Already shadow-resident in VRAM: a plain hit, no transfer.
                ++hits_hot;
                shadow_touch(gid);
                ++entry.lease_count;
                return ExpertRequestReservation{
                    ExpertRequestKind::HOT_HIT,
                    ExpertTier::HOT_VRAM,
                    gid,
                    0,
                    entry.shadow_vram_slot,
                    -1,
                    true,
                    std::nullopt
                };
            }

            const uint64_t shadow_operation = next_operation_id++;
            // A shadow promotion may **also** be reserved staged-only. Under a swept
            // prefill the lookahead depth is derived from the **staging** arena, so a
            // Warm expert that took its VRAM destination here would commit a whole
            // layer's slots at *reservation* time — invisible to the depth bound, so
            // the pool over-commits and the next load finds no free slot. Deferring the
            // destination to copy time keeps **one** admission rule for both sources:
            // a read (or a Warm hand-off) takes a staging slot immediately and a VRAM
            // slot only when its copy can run. `attach_vram_destination` supplies it.
            std::optional<VramDestination> destination;
            if (!stage_only) {
                destination = reserve_vram_destination(
                    gid, shadow_operation, demotion_queue_capacity);
            }
            const int32_t source_host_slot = entry.slot_idx;

            ++hits_warm;
            ++shadow_copies;
            // Out of the Warm LRU while the copy is in flight; back in on
            // completion. The host slot itself is never reserved or freed, which is
            // what makes the read non-destructive.
            remove_from_lru(entry, warm_host_lru, gid);
            entry.operation = ExpertOperation::PROMOTION_PENDING;
            entry.operation_id = shadow_operation;
            entry.gpu_transfer = ExpertGpuTransfer::H2D_PENDING;
            entry.publication = ExpertPublication::UNPUBLISHED;
            entry.slot_state = ExpertSlotState::ACTIVE;
            entry.lease_count++;
            entry.pending_slot_idx = destination.has_value()
                ? static_cast<int32_t>(destination->vram_slot)
                : -1;
            entry.warm_shadow = true;
            checked_validate();
            return ExpertRequestReservation{
                ExpertRequestKind::WARM_PROMOTION,
                ExpertTier::WARM_HOST,
                gid,
                shadow_operation,
                destination.has_value() ? static_cast<int32_t>(destination->vram_slot) : -1,
                source_host_slot,
                true,
                std::nullopt
            };
        }

        const uint64_t operation_id = next_operation_id++;
        const ExpertTier source_tier = entry.owner;
        // A cold read may be reserved **staged-only**: no VRAM slot is taken now, and
        // `attach_vram_destination` supplies one when the copy can run. Every other
        // source decides its destination here, because a Warm shadow or a demotion
        // depends on the victim/ownership state at reservation time.
        const bool defer_vram = stage_only && source_tier == ExpertTier::COLD_NVME;
        std::optional<VramDestination> destination;
        if (!defer_vram) {
            destination = reserve_vram_destination(gid, operation_id, demotion_queue_capacity);
        }

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
        entry.pending_slot_idx = destination.has_value()
            ? static_cast<int32_t>(destination->vram_slot)
            : -1;

        ExpertRequestReservation request{
            defer_vram
                ? ExpertRequestKind::COLD_STAGED
                : (source_tier == ExpertTier::WARM_HOST
                       ? ExpertRequestKind::WARM_PROMOTION
                       : ExpertRequestKind::COLD_MISS),
            source_tier,
            gid,
            operation_id,
            destination.has_value() ? static_cast<int32_t>(destination->vram_slot) : -1,
            source_host_slot,
            true,
            destination.has_value() ? destination->demotion
                                    : std::optional<ExpertDemotionReservation>{}
        };

        checked_validate();
        return request;
    }

    // Give a staged-only operation its VRAM destination, at the moment its copy can
    // actually run. Returns the slot. Idempotent: an operation that already has a
    // destination returns it unchanged, so a caller may attach and then copy without
    // tracking whether an earlier call already did. The destination is reserved exactly
    // as `reserve_request` would have done at reservation time, so the free-slot and
    // demotion rules are unchanged — only **when** they run is.
    int32_t attach_vram_destination(uint64_t operation_id, uint64_t demotion_queue_capacity) {
        auto* incoming = find_incoming_operation(operation_id);
        if (incoming == nullptr) {
            throw std::logic_error(
                "ExpertRegistry: attach_vram_destination has no pending request");
        }
        if (incoming->pending_slot_idx >= 0) {
            return incoming->pending_slot_idx;
        }
        const auto destination = reserve_vram_destination(
            incoming->global_expert_id, operation_id, demotion_queue_capacity);
        incoming->pending_slot_idx = static_cast<int32_t>(destination.vram_slot);
        return incoming->pending_slot_idx;
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
        checked_validate();
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
        checked_validate();
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
        checked_validate();
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

        // Frozen prefill's non-destructive copy: the VRAM slot becomes a shadow
        // residency of a Warm-owned expert. Ownership stays Warm — the host slot is
        // NOT freed and `owner`/`slot_idx` are untouched — so Warm's resident set is
        // exactly what it was before the copy.
        if (incoming->warm_shadow) {
            incoming->operation = ExpertOperation::NONE;
            incoming->operation_id = 0;
            incoming->gpu_transfer = ExpertGpuTransfer::NONE;
            incoming->publication = ExpertPublication::PUBLISHED;
            incoming->slot_state = ExpertSlotState::ACTIVE;
            incoming->pending_slot_idx = -1;
            incoming->warm_shadow = false;
            incoming->shadow_vram_slot = destination_slot;
            shadow_slot_of_expert_[incoming->global_expert_id] = destination_slot;
            vram_slots[static_cast<size_t>(destination_slot)] =
                static_cast<int32_t>(incoming->global_expert_id);
            vram_slot_reservations[static_cast<size_t>(destination_slot)] = 0;
            push_lru_front(*incoming, warm_host_lru);
            shadow_touch(incoming->global_expert_id);
            checked_validate();
            return;
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
        checked_validate();
    }

    void fail_request(uint64_t operation_id) {
        auto* incoming = find_incoming_operation(operation_id);
        if (incoming == nullptr) {
            throw std::logic_error("ExpertRegistry: request failure does not match a pending request");
        }

        // A failed frozen-prefill copy leaves the Warm entry exactly as it was: the
        // host slot was never reserved, so only the VRAM reservation is undone.
        if (incoming->warm_shadow) {
            const int32_t shadow_destination = incoming->pending_slot_idx;
            if (shadow_destination >= 0) {
                vram_slot_reservations[static_cast<size_t>(shadow_destination)] = 0;
                if (vram_slots[static_cast<size_t>(shadow_destination)] == -1) {
                    free_vram_slots.push_back(static_cast<uint32_t>(shadow_destination));
                }
            }
            incoming->operation = ExpertOperation::NONE;
            incoming->operation_id = 0;
            incoming->gpu_transfer = ExpertGpuTransfer::NONE;
            incoming->publication = ExpertPublication::PUBLISHED;
            incoming->slot_state = ExpertSlotState::ACTIVE;
            incoming->pending_slot_idx = -1;
            incoming->warm_shadow = false;
            push_lru_front(*incoming, warm_host_lru);
            checked_validate();
            return;
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
        checked_validate();
    }

    void release_lease(uint32_t gid) {
        if (gid >= catalog.size() || catalog[gid].lease_count == 0) {
            throw std::logic_error("ExpertRegistry: releasing an unleased expert");
        }
        --catalog[gid].lease_count;
    }

    // --- Prefill streaming ----------------------------------------------------
    //
    // Definitions live in `prefill_residency.hpp`; the streaming model they
    // implement (hard switch, preservation, per-layer release) is documented there.
    void begin_prefill_stream(
        uint32_t drain_slots = UINT32_MAX,
        PrefillAlloc alloc = PrefillAlloc::FreeListOnly
    );
    void end_prefill_stream();
    bool prefill_streaming() const noexcept;
    void release_layer(uint32_t layer_id);

    // How many of a layer's experts currently hold a VRAM copy (Hot or shadow). The
    // lookahead reads it to know how much of a layer is already resident before it
    // fills the rest.
    uint32_t layer_resident_count(uint32_t layer_id) const {
        if (layer_id >= num_layers) return 0;
        const uint32_t first = layer_id * experts_per_layer;
        uint32_t resident = 0;
        for (uint32_t index = 0; index < experts_per_layer; ++index) {
            const auto& entry = catalog[first + index];
            if (entry.owner == ExpertTier::HOT_VRAM || entry.shadow_vram_slot >= 0) {
                ++resident;
            }
        }
        return resident;
    }

    // Whether one expert currently holds a VRAM copy. The sweep's top-up reads it to
    // build the list of a layer's still-missing experts.
    bool expert_resident(uint32_t layer_id, uint32_t expert_id) const {
        const auto& entry = catalog[get_global_id(layer_id, expert_id)];
        return entry.owner == ExpertTier::HOT_VRAM || entry.shadow_vram_slot >= 0;
    }

    // Free VRAM slots. The lookahead stops when a whole layer no longer fits.
    size_t free_vram_slot_count() const noexcept { return free_vram_slots.size(); }

    // ---- the Warm/staging partition ------------------------------------------
    //
    // Definitions live in `warm_partition.hpp`; the movable boundary is documented there.
    uint32_t usable_host_capacity() const noexcept;
    void grow_host_capacity(uint32_t usable);
    void shrink_host_capacity(uint32_t usable);
    uint32_t release_host_tail(uint32_t keep);

    uint32_t preserved_resident_count() const noexcept {
        uint32_t count = 0;
        for (const auto& entry : catalog) {
            if (entry.resident_at_prefill_begin) ++count;
        }
        return count;
    }

    // Legacy per-token prefill (the engine's `forward_token` path) freezes Warm
    // without streaming. The swept prefill (`begin_prefill_stream`) is the stronger
    // mode and owns the flag while it is active. Definitions in `prefill_residency.hpp`.
    void set_warm_frozen(bool frozen);
    bool warm_frozen() const noexcept;

    // Per-request invariant validation — **off by default**.
    //
    // `validate_invariants()` is a full audit of the whole registry (every catalog
    // entry, every VRAM and host slot, and — for each slot currently reserved — a
    // rescan of the catalog). It is `O(total_experts + slots)`, and it is called once
    // per reserved expert, so the cost is quadratic in the batch and linear in the
    // model: a layer-wide prefill dispatch is 256 reservations and a decode pass is
    // 258, which measured **10.2 s of a 338-token swept prefill**.
    //
    // So it is a debugging instrument, not a production cost. With it off the audit
    // still runs at every **boundary** — `init`, `begin_prefill_stream`,
    // `end_prefill_stream`, `release_layer`, `set_warm_frozen` — which is where the
    // registry actually changes shape, and `invariants_hold()` always runs the full
    // audit regardless, so a gate that asks the question always gets a real answer.
    // Turn it on to localise a defect to the individual reservation that caused it.
    void set_validate_each_request(bool enabled) noexcept {
        validate_each_request_ = enabled;
    }
    bool validate_each_request() const noexcept { return validate_each_request_; }

    // Warm-owned experts currently holding an extra VRAM copy, and the VRAM copy of
    // one such expert. Definitions in `prefill_residency.hpp`.
    uint32_t shadow_resident_count() const noexcept;
    int32_t shadow_slot_of(uint32_t gid) const;

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

    // The invariant audit. Definitions live in `expert_registry_validation.hpp`.
    bool invariants_hold() const noexcept;
    void checked_validate() const;
    void validate_invariants() const;

private:
    struct VramDestination {
        uint32_t vram_slot{0};
        std::optional<ExpertDemotionReservation> demotion;
    };

    // Shadow-residency helpers. Definitions in `prefill_residency.hpp`.
    void release_shadow_residencies();
    void release_shadow_residency(ExpertCatalogEntry& entry);
    void shadow_touch(uint32_t gid);

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
        } else if (prefill_stream_ && prefill_alloc_ == PrefillAlloc::FreeListOnly) {
            // The sweep allocates from the free list only and releases each layer
            // deterministically, so there is no victim to choose (its own order
            // decides what is dead) and no demotion to run (Warm is frozen). A request
            // that finds no free slot means the lookahead over-committed — a driver
            // defect, refused rather than evicted around.
            throw std::runtime_error(
                "ExpertRegistry: prefill stream has no free VRAM slot for a load "
                "(the layer lookahead must not exceed capacity)");
        } else if (prefill_stream_ || warm_frozen_) {
            // A prefill eviction is a **release**, never a demotion: the victim is a
            // shadow residency (a Warm expert's prefill copy) when one is available,
            // else an ordinary Hot resident. Either way no ownership moves and Warm is
            // not written to. This is the routed bank's admission path, and also the
            // legacy per-token frozen path.
            //
            // A resident **present at prefill entry** is never a victim: it is not the
            // pass's to spend, and the sparing release would restore it anyway.
            for (auto it = shadow_lru_.rbegin(); it != shadow_lru_.rend(); ++it) {
                auto& candidate = catalog[*it];
                if (candidate.shadow_vram_slot < 0 || candidate.lease_count != 0 ||
                    candidate.operation != ExpertOperation::NONE ||
                    candidate.publication != ExpertPublication::PUBLISHED) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>(candidate.shadow_vram_slot);
                release_shadow_residency(candidate);
                vram_slots[static_cast<size_t>(slot)] = -1;
                vram_slot_reservations[static_cast<size_t>(slot)] = operation_id;
                return VramDestination{slot, std::nullopt};
            }
            for (auto it = hot_vram_lru.rbegin(); it != hot_vram_lru.rend(); ++it) {
                auto& candidate = catalog[*it];
                if (candidate.global_expert_id == incoming_gid ||
                    candidate.resident_at_prefill_begin ||
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
                throw std::runtime_error(
                    "ExpertRegistry: no reclaimable Hot VRAM slot is available in "
                    "prefill (every resident is preserved, leased, or in flight)");
            }
            const uint32_t slot = static_cast<uint32_t>(victim->slot_idx);
            remove_from_lru(*victim, hot_vram_lru, victim->global_expert_id);
            vram_slots[static_cast<size_t>(slot)] = -1;
            victim->owner = ExpertTier::COLD_NVME;
            victim->slot_idx = -1;
            victim->pending_slot_idx = -1;
            victim->publication = ExpertPublication::PUBLISHED;
            victim->slot_state = ExpertSlotState::UNALLOCATED;
            victim->operation = ExpertOperation::NONE;
            victim->gpu_transfer = ExpertGpuTransfer::NONE;
            vram_slot_reservations[static_cast<size_t>(slot)] = operation_id;
            return VramDestination{slot, std::nullopt};
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

    // Host-slot reservations and the free-list rebuild. Definitions in
    // `warm_partition.hpp`.
    int32_t find_reserved_host_slot(uint64_t operation_id) const;
    void release_reserved_host_slot(uint32_t slot);
    void rebuild_free_host_slots();

    // Per-tier LRU membership audit. Definition in `expert_registry_validation.hpp`.
    void validate_lru(const std::list<uint32_t>& lru, ExpertTier tier) const;

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
                } else if (preload_warm_host && host_assigned < host_capacity_usable) {
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

    // Frozen-prefill shadow state. `shadow_slot_of_expert_` mirrors
    // `ExpertCatalogEntry::shadow_vram_slot` so a caller can resolve an expert's
    // VRAM copy without scanning the catalog.
    bool warm_frozen_{false};
    // See `set_validate_each_request`. Off unless a debug run or a gate turns it on.
    bool validate_each_request_{false};
    // The prefill sweep's streaming mode. Drains Hot on entry, holds a sliding window
    // of whole layer sets in layer order, and leaves only the residents preserved at
    // entry on exit. Implies `warm_frozen_`; decode never sees either flag set.
    bool prefill_stream_{false};
    // How a prefill load with no free slot is answered. See `PrefillAlloc`.
    PrefillAlloc prefill_alloc_{PrefillAlloc::FreeListOnly};
    std::vector<int32_t> shadow_slot_of_expert_;
    std::list<uint32_t> shadow_lru_;
};

} // namespace aeon::core

// The out-of-line member definitions, grouped by concern. Included after the
// class so each concern header sees a complete `ExpertRegistry`; each includes
// this file, so a concern header is also includable on its own.
#include "infrastructure/expert/residency/warm_partition.hpp"
#include "infrastructure/expert/residency/prefill_residency.hpp"
#include "infrastructure/expert/residency/expert_registry_validation.hpp"
