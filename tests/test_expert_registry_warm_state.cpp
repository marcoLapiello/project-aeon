#include "infrastructure/core/expert_registry.hpp"
#include "infrastructure/core/host_expert_pool.hpp"
#include "infrastructure/core/prefetch_staging.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using aeon::core::ExpertRequestKind;
using aeon::core::ExpertRegistry;
using aeon::core::ExpertDemotionDropReason;

void finish(ExpertRegistry& registry, const aeon::core::ExpertRequestReservation& request) {
    if (request.demotion) {
        registry.complete_demotion(request.operation_id);
    }
    if (request.kind != ExpertRequestKind::HOT_HIT) {
        registry.complete_request(request.operation_id);
    }
    registry.release_lease(request.global_expert_id);
}

} // namespace

int main() {
    ExpertRegistry registry(1, 8, 2, 2);
    // This gate's subject *is* the registry's bookkeeping, so the full audit runs
    // after every operation here — the fine-grained assertion is the point, and it is
    // what the production path leaves off for speed (ledger M43).
    registry.set_validate_each_request(true);
    assert(registry.invariants_hold());

    ExpertRegistry layer_order(2, 4, 2, 2);
    layer_order.set_validate_each_request(true);
    const auto ordered_request = layer_order.reserve_request(0, 2, 1, 2);
    assert(ordered_request.demotion.has_value());
    assert(layer_order.catalog[ordered_request.demotion->victim_gid].layer_id == 0);
    finish(layer_order, ordered_request);
    assert(layer_order.invariants_hold());

    ExpertRegistry future_layer_victim(2, 4, 2, 2);
    const auto touched_layer_zero = future_layer_victim.reserve_request(0, 0, 0, 2);
    assert(touched_layer_zero.kind == ExpertRequestKind::HOT_HIT);
    future_layer_victim.release_lease(touched_layer_zero.global_expert_id);
    const auto admitted_future_layer = future_layer_victim.reserve_request(0, 2, 1, 2);
    assert(admitted_future_layer.demotion.has_value());
    assert(future_layer_victim.catalog[admitted_future_layer.demotion->victim_gid].layer_id > 0);
    finish(future_layer_victim, admitted_future_layer);
    assert(future_layer_victim.invariants_hold());

    const auto hot = registry.reserve_request(0, 1, 0, 2);
    assert(hot.kind == ExpertRequestKind::HOT_HIT);
    assert(hot.vram_slot >= 0);
    registry.release_lease(hot.global_expert_id);

    const auto full_warm_promotion = registry.reserve_request(2, 2, 2);
    assert(full_warm_promotion.kind == ExpertRequestKind::WARM_PROMOTION);
    assert(full_warm_promotion.demotion.has_value());
    const uint32_t first_victim = full_warm_promotion.demotion->victim_gid;
    finish(registry, full_warm_promotion);
    assert(registry.catalog[first_victim].owner == aeon::core::ExpertTier::WARM_HOST);

    const auto free_warm_promotion = registry.reserve_request(0, 0, 3, 2);
    assert(free_warm_promotion.kind == ExpertRequestKind::WARM_PROMOTION);
    assert(free_warm_promotion.demotion.has_value());
    finish(registry, free_warm_promotion);

    const auto cold_with_demotion = registry.reserve_request(4, 4, 2);
    assert(cold_with_demotion.kind == ExpertRequestKind::COLD_MISS);
    assert(cold_with_demotion.demotion.has_value());
    finish(registry, cold_with_demotion);

    const auto cold_with_drop = registry.reserve_request(5, 5, 0);
    assert(cold_with_drop.kind == ExpertRequestKind::COLD_MISS);
    assert(!cold_with_drop.demotion.has_value());
    bool found_queue_pressure_drop = false;
    for (const auto& entry : registry.catalog) {
        if (entry.operation_id == cold_with_drop.operation_id &&
            entry.operation == aeon::core::ExpertOperation::DEMOTION_PENDING) {
            found_queue_pressure_drop = true;
            assert(entry.demotion_drop_reason == ExpertDemotionDropReason::QUEUE_PRESSURE);
        }
    }
    assert(found_queue_pressure_drop);
    finish(registry, cold_with_drop);

    const auto pending_source = registry.reserve_request(2, 6, 2);
    assert(pending_source.kind == ExpertRequestKind::WARM_PROMOTION);
    const auto joined = registry.reserve_request(2, 7, 2);
    assert(joined.kind == ExpertRequestKind::PENDING);
    assert(joined.operation_id == pending_source.operation_id);
    assert(registry.catalog[pending_source.global_expert_id].lease_count == 2);
    assert(registry.host_slot_reservations[static_cast<size_t>(pending_source.source_host_slot)] ==
           pending_source.operation_id);
    assert(registry.pending_transfer_count() == 1);
    assert(registry.invariants_hold());
    if (pending_source.demotion) {
        registry.complete_demotion(pending_source.operation_id);
    }
    registry.complete_request(pending_source.operation_id);
    registry.release_lease(pending_source.global_expert_id);
    registry.release_lease(joined.global_expert_id);

    assert(registry.hits_hot == 1);
    assert(registry.hits_warm == 3);
    assert(registry.misses_cold == 2);
    assert(registry.demotion_attempts == 5);
    assert(registry.demotion_completions == 4);
    assert(registry.demotion_drops == 1);
    assert(registry.pending_transfer_count() == 0);
    assert(registry.invariants_hold());

    std::mt19937 rng(0xAE0F);
    std::uniform_int_distribution<uint32_t> expert_distribution(0, 7);
    for (uint32_t transition = 0; transition < 10000; ++transition) {
        const uint32_t expert_id = expert_distribution(rng);
        const auto request = registry.reserve_request(0, expert_id, transition + 8, 2);
        assert(request.kind != ExpertRequestKind::PENDING);
        finish(registry, request);
        assert(registry.invariants_hold());
    }

    ExpertRegistry no_warm(1, 8, 2, 0);
    const auto cold_without_warm = no_warm.reserve_request(4, 1, 2);
    assert(!cold_without_warm.demotion.has_value());
    finish(no_warm, cold_without_warm);
    assert(no_warm.demotion_attempts == 0);
    assert(no_warm.demotion_completions == 0);
    assert(no_warm.demotion_drops == 0);
    assert(no_warm.invariants_hold());

    aeon::core::HostExpertPool host_pool(2);
    assert(host_pool.pinned_slot_count() + host_pool.unpinned_slot_count() == host_pool.num_slots);
    uint32_t observed_pinned_slots = 0;
    for (uint32_t slot = 0; slot < host_pool.num_slots; ++slot) {
        observed_pinned_slots += host_pool.is_slot_pinned(slot) ? 1 : 0;
    }
    assert(observed_pinned_slots == host_pool.pinned_slot_count());

    aeon::core::PrefetchStagingArena staging;
    std::vector<uint8_t> payload(aeon::core::AEON_EXPERT_BYTES, 0xA5);
    staging.stage_payload(0, payload.data());
    staging.begin_gpu_transfer(0);

    hipStream_t staging_stream = nullptr;
    hipStream_t consumer_stream = nullptr;
    assert(hipStreamCreateWithFlags(&staging_stream, hipStreamNonBlocking) == hipSuccess);
    assert(hipStreamCreateWithFlags(&consumer_stream, hipStreamNonBlocking) == hipSuccess);
    uint8_t* device_payload = nullptr;
    assert(hipMalloc(reinterpret_cast<void**>(&device_payload), aeon::core::AEON_EXPERT_BYTES) == hipSuccess);
    assert(hipMemcpyAsync(
        device_payload,
        staging.get_slot_ptr(0),
        aeon::core::AEON_EXPERT_BYTES,
        hipMemcpyHostToDevice,
        staging_stream) == hipSuccess);
    assert(hipEventRecord(staging.events[0], staging_stream) == hipSuccess);
    assert(hipStreamWaitEvent(consumer_stream, staging.events[0], 0) == hipSuccess);
    assert(hipMemsetAsync(device_payload, 0, aeon::core::AEON_EXPERT_BYTES, consumer_stream) == hipSuccess);
    assert(hipStreamSynchronize(consumer_stream) == hipSuccess);
    assert(hipEventQuery(staging.events[0]) == hipSuccess);
    staging.release_after_gpu_transfer(0);
    assert(staging.slot_state(0) == aeon::core::PrefetchStagingArena::SlotState::AVAILABLE);

    staging.begin_direct_transfer(1);
    assert(hipEventRecord(staging.events[1], staging_stream) == hipSuccess);
    assert(hipStreamSynchronize(staging_stream) == hipSuccess);
    staging.release_after_failure(1);
    assert(staging.slot_state(1) == aeon::core::PrefetchStagingArena::SlotState::AVAILABLE);
    assert(hipFree(device_payload) == hipSuccess);
    assert(hipStreamDestroy(consumer_stream) == hipSuccess);
    assert(hipStreamDestroy(staging_stream) == hipSuccess);

    ExpertRegistry lazy_warm(1, 8, 2, 2, false);
    assert(lazy_warm.published_warm_slots() == 0);
    assert(lazy_warm.free_host_slots.size() == 2);
    const auto lazy_cold_request = lazy_warm.reserve_request(2, 1, 2);
    assert(lazy_cold_request.kind == ExpertRequestKind::COLD_MISS);
    assert(lazy_cold_request.demotion.has_value());
    finish(lazy_warm, lazy_cold_request);
    assert(lazy_warm.published_warm_slots() == 1);
    assert(lazy_warm.invariants_hold());

    ExpertRegistry failed_demotion(1, 8, 2, 2);
    const auto failed_request = failed_demotion.reserve_request(2, 1, 2);
    assert(failed_request.demotion.has_value());
    const uint32_t failed_victim = failed_request.demotion->victim_gid;
    const size_t host_slots_before_failure = failed_demotion.free_host_slots.size();
    failed_demotion.fail_demotion(failed_request.operation_id);
    failed_demotion.fail_request(failed_request.operation_id);
    failed_demotion.release_lease(failed_request.global_expert_id);
    assert(failed_demotion.catalog[failed_victim].owner == aeon::core::ExpertTier::COLD_NVME);
    assert(failed_demotion.free_host_slots.size() == host_slots_before_failure + 1);
    assert(failed_demotion.pending_transfer_count() == 0);
    assert(failed_demotion.invariants_hold());

    ExpertRegistry failed_cold_request(1, 8, 2, 0);
    const auto failed_cold = failed_cold_request.reserve_request(2, 1, 2);
    failed_cold_request.fail_request(failed_cold.operation_id);
    failed_cold_request.release_lease(failed_cold.global_expert_id);
    assert(failed_cold_request.pending_transfer_count() == 0);
    assert(failed_cold_request.invariants_hold());

    bool rejected_zero_hot = false;
    try {
        ExpertRegistry invalid(1, 8, 0, 2);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected_zero_hot = true;
    }
    assert(rejected_zero_hot);

    // --- Prefill restore infrastructure (Prefill Supply Strategy plan, Step 1) ---
    //
    // The pool holds 8 Hot residents out of 16 experts across 4 layers. A prefill
    // that drains only 4 preserves the other 4; the per-layer release must spare the
    // preserved ones and return the prefill-admitted ones, and ending must leave the
    // marks cleared and the drained set exposed for the caller to reload.
    {
        std::vector<uint32_t> pre_fill_hot;
        {
            ExpertRegistry prefill(4, 4, 8, 4);
            prefill.set_validate_each_request(true);
            for (const auto& entry : prefill.catalog) {
                if (entry.owner == aeon::core::ExpertTier::HOT_VRAM) {
                    pre_fill_hot.push_back(entry.global_expert_id);
                }
            }
            assert(pre_fill_hot.size() == 8);

            // Bounded drain: the worst-LRU half goes, the remainder is preserved.
            prefill.begin_prefill_stream(4);
            assert(prefill.invariants_hold());
            assert(prefill.published_hot_slots() == 4);
            assert(prefill.free_vram_slot_count() == 4);
            assert(prefill.restore_set().size() == 4);
            assert(prefill.preserved_resident_count() == 4);
            for (const auto& entry : prefill.catalog) {
                if (entry.resident_at_prefill_begin) {
                    assert(entry.owner == aeon::core::ExpertTier::HOT_VRAM);
                    const bool drained =
                        std::find(prefill.restore_set().begin(), prefill.restore_set().end(),
                                  entry.global_expert_id) != prefill.restore_set().end();
                    assert(!drained);
                }
            }

            // Admit a drained expert of layer 0 as a prefill-admitted resident.
            const auto admit = [&](uint32_t gid) {
                const auto request = prefill.reserve_request(gid, 0, 4);
                assert(request.kind == ExpertRequestKind::COLD_MISS);
                prefill.complete_request(request.operation_id);
                prefill.release_lease(gid);
            };
            assert(prefill.catalog[0].owner == aeon::core::ExpertTier::COLD_NVME);
            admit(0);
            assert(prefill.catalog[0].owner == aeon::core::ExpertTier::HOT_VRAM);
            assert(!prefill.catalog[0].resident_at_prefill_begin);

            // Retire layer 0: the admitted expert goes, the preserved one stays.
            const uint32_t preserved_layer0 = 1;
            assert(prefill.catalog[preserved_layer0].resident_at_prefill_begin);
            prefill.release_layer(0);
            assert(prefill.invariants_hold());
            assert(prefill.catalog[0].owner == aeon::core::ExpertTier::COLD_NVME);
            assert(prefill.catalog[preserved_layer0].owner == aeon::core::ExpertTier::HOT_VRAM);
            assert(prefill.catalog[preserved_layer0].resident_at_prefill_begin);

            for (uint32_t layer = 1; layer < 4; ++layer) {
                prefill.release_layer(layer);
            }
            prefill.end_prefill_stream();
            assert(prefill.invariants_hold());
            assert(prefill.preserved_resident_count() == 0);

            // Exactly the preserved residents remain, and the drained set is exposed.
            std::vector<uint32_t> hot_after;
            for (const auto& entry : prefill.catalog) {
                if (entry.owner == aeon::core::ExpertTier::HOT_VRAM) {
                    hot_after.push_back(entry.global_expert_id);
                }
            }
            assert(hot_after.size() == 4);
            for (const uint32_t gid : hot_after) {
                assert(std::find(pre_fill_hot.begin(), pre_fill_hot.end(), gid) !=
                       pre_fill_hot.end());
            }

            // The caller reloads the drained set through the normal cold path; the
            // pre-prefill set is then resident again, byte-for-byte the same set.
            for (const uint32_t gid : prefill.restore_set()) {
                admit(gid);
            }
            std::vector<uint32_t> restored;
            for (const auto& entry : prefill.catalog) {
                if (entry.owner == aeon::core::ExpertTier::HOT_VRAM) {
                    restored.push_back(entry.global_expert_id);
                }
            }
            std::sort(restored.begin(), restored.end());
            std::vector<uint32_t> expected = pre_fill_hot;
            std::sort(expected.begin(), expected.end());
            assert(restored == expected);
            assert(prefill.invariants_hold());
        }

        // A full drain (the default) preserves nothing and exposes every resident.
        {
            ExpertRegistry full(4, 4, 8, 4);
            full.set_validate_each_request(true);
            full.begin_prefill_stream();
            assert(full.invariants_hold());
            assert(full.published_hot_slots() == 0);
            assert(full.preserved_resident_count() == 0);
            assert(full.restore_set().size() == 8);
            for (uint32_t layer = 0; layer < 4; ++layer) {
                full.release_layer(layer);
            }
            full.end_prefill_stream();
            assert(full.invariants_hold());
            assert(full.preserved_resident_count() == 0);
        }
    }

    std::cout << "ExpertRegistry Warm state trace passed: 10000 transitions, seed=0xAE0F\n";
    return 0;
}