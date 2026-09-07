#pragma once

#include "core/aeon_loader.hpp"
#include <cstdint>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace aeon::core {

enum class ExpertTier : uint8_t {
    HOT_VRAM  = 0,
    WARM_HOST = 1,
    COLD_NVME = 2
};

struct ExpertCatalogEntry {
    uint32_t global_expert_id{0}; // layer_id * 256 + expert_id
    uint16_t layer_id{0};
    uint16_t expert_id{0};

    ExpertTier tier{ExpertTier::COLD_NVME};
    int32_t slot_idx{-1}; // Physical index in Hot VRAM or Warm Host pool

    // Dynamic activation & entropy tracking
    uint64_t activation_count{0};
    uint64_t last_step_used{0};
    float moving_frequency{0.0f}; // Exponential Moving Average (EMA)

    // Iterator into the respective tier's LRU list for O(1) recency updates
    std::list<uint32_t>::iterator lru_it;
};

class ExpertRegistry {
public:
    uint32_t num_layers{43};
    uint32_t experts_per_layer{256};
    uint32_t total_experts{11008};

    uint32_t vram_capacity{0};
    uint32_t host_capacity{0};

    // Global catalog indexed by global_id = layer_id * 256 + expert_id
    std::vector<ExpertCatalogEntry> catalog;

    // VRAM physical slot mapping: vram_slots[slot_idx] = global_expert_id
    std::vector<int32_t> vram_slots;
    std::vector<uint32_t> free_vram_slots;

    // Host physical slot mapping: host_slots[slot_idx] = global_expert_id
    std::vector<int32_t> host_slots;
    std::vector<uint32_t> free_host_slots;

    // LRU double-linked lists (front = MRU, back = LRU)
    std::list<uint32_t> hot_vram_lru;
    std::list<uint32_t> warm_host_lru;

    // Statistics
    uint64_t hits_hot{0};
    uint64_t hits_warm{0};
    uint64_t misses_cold{0};

    ExpertRegistry() = default;

    ExpertRegistry(
        uint32_t layers,
        uint32_t experts_layer,
        uint32_t vram_cap,
        uint32_t host_cap
    ) {
        init(layers, experts_layer, vram_cap, host_cap);
    }

    void init(
        uint32_t layers,
        uint32_t experts_layer,
        uint32_t vram_cap,
        uint32_t host_cap
    ) {
        num_layers = layers;
        experts_per_layer = experts_layer;
        total_experts = layers * experts_layer;
        vram_capacity = vram_cap;
        host_capacity = host_cap;

        catalog.resize(total_experts);
        for (uint32_t l = 0; l < num_layers; ++l) {
            for (uint32_t e = 0; e < experts_per_layer; ++e) {
                uint32_t gid = l * experts_per_layer + e;
                catalog[gid].global_expert_id = gid;
                catalog[gid].layer_id = static_cast<uint16_t>(l);
                catalog[gid].expert_id = static_cast<uint16_t>(e);
                catalog[gid].tier = ExpertTier::COLD_NVME;
                catalog[gid].slot_idx = -1;
                catalog[gid].activation_count = 0;
                catalog[gid].last_step_used = 0;
                catalog[gid].moving_frequency = 0.0f;
            }
        }

        // Initialize VRAM slots
        vram_slots.assign(vram_capacity, -1);
        free_vram_slots.clear();
        for (int32_t s = static_cast<int32_t>(vram_capacity) - 1; s >= 0; --s) {
            free_vram_slots.push_back(static_cast<uint32_t>(s));
        }
        hot_vram_lru.clear();

        // Initialize Host slots
        host_slots.assign(host_capacity, -1);
        free_host_slots.clear();
        for (int32_t s = static_cast<int32_t>(host_capacity) - 1; s >= 0; --s) {
            free_host_slots.push_back(static_cast<uint32_t>(s));
        }
        warm_host_lru.clear();

        hits_hot = 0;
        hits_warm = 0;
        misses_cold = 0;

        // Populate initial placement: round-robin interleaving across layers
        populate_round_robin();
    }

    inline uint32_t get_global_id(uint32_t layer_id, uint32_t expert_id) const {
        return layer_id * experts_per_layer + expert_id;
    }

    // Lookup an expert; if in Hot VRAM, update recency and return its physical slot immediately
    // Returns slot_idx if in Hot VRAM, or -1 if not in Hot VRAM (miss)
    int32_t touch_hot_expert(uint32_t layer_id, uint32_t expert_id, uint64_t current_step) {
        uint32_t gid = get_global_id(layer_id, expert_id);
        auto& entry = catalog[gid];

        entry.activation_count++;
        entry.last_step_used = current_step;
        // Exponential moving average for activation frequency
        constexpr float alpha = 0.1f;
        entry.moving_frequency = (1.0f - alpha) * entry.moving_frequency + alpha * 1.0f;

        if (entry.tier == ExpertTier::HOT_VRAM) {
            hits_hot++;
            // Move to front of Hot LRU list
            hot_vram_lru.erase(entry.lru_it);
            hot_vram_lru.push_front(gid);
            entry.lru_it = hot_vram_lru.begin();
            return entry.slot_idx;
        }

        if (entry.tier == ExpertTier::WARM_HOST) {
            hits_warm++;
        } else {
            misses_cold++;
        }

        return -1;
    }

    // Allocate or evict a Hot VRAM slot for the incoming expert.
    // Returns {vram_slot_idx, evicted_gid_or_minus_1}
    std::pair<uint32_t, int32_t> allocate_vram_slot(uint32_t gid) {
        uint32_t slot = 0;
        int32_t evicted_gid = -1;

        if (!free_vram_slots.empty()) {
            slot = free_vram_slots.back();
            free_vram_slots.pop_back();
        } else {
            // Evict LRU victim from hot VRAM
            evicted_gid = hot_vram_lru.back();
            hot_vram_lru.pop_back();

            slot = static_cast<uint32_t>(catalog[evicted_gid].slot_idx);

            // Demote evicted expert to Warm Host DDR if space exists or evict LRU warm
            if (host_capacity > 0) {
                catalog[evicted_gid].tier = ExpertTier::WARM_HOST;
                // If free host slots exist, assign one
                if (!free_host_slots.empty()) {
                    uint32_t hslot = free_host_slots.back();
                    free_host_slots.pop_back();
                    catalog[evicted_gid].slot_idx = static_cast<int32_t>(hslot);
                    warm_host_lru.push_front(evicted_gid);
                    catalog[evicted_gid].lru_it = warm_host_lru.begin();
                    host_slots[hslot] = evicted_gid;
                } else if (!warm_host_lru.empty()) {
                    // Evict LRU from warm host pool to cold NVMe
                    uint32_t cold_gid = warm_host_lru.back();
                    warm_host_lru.pop_back();
                    uint32_t hslot = static_cast<uint32_t>(catalog[cold_gid].slot_idx);
                    catalog[cold_gid].tier = ExpertTier::COLD_NVME;
                    catalog[cold_gid].slot_idx = -1;

                    catalog[evicted_gid].slot_idx = static_cast<int32_t>(hslot);
                    warm_host_lru.push_front(evicted_gid);
                    catalog[evicted_gid].lru_it = warm_host_lru.begin();
                    host_slots[hslot] = evicted_gid;
                } else {
                    catalog[evicted_gid].tier = ExpertTier::COLD_NVME;
                    catalog[evicted_gid].slot_idx = -1;
                }
            } else {
                catalog[evicted_gid].tier = ExpertTier::COLD_NVME;
                catalog[evicted_gid].slot_idx = -1;
            }
        }

        // Install new expert in Hot VRAM
        auto& entry = catalog[gid];
        entry.tier = ExpertTier::HOT_VRAM;
        entry.slot_idx = static_cast<int32_t>(slot);
        hot_vram_lru.push_front(gid);
        entry.lru_it = hot_vram_lru.begin();
        vram_slots[slot] = static_cast<int32_t>(gid);

        return {slot, evicted_gid};
    }

private:
    void populate_round_robin() {
        // Round-robin placement across all layers into Hot VRAM first, then Warm Host DDR
        uint32_t vram_assigned = 0;
        uint32_t host_assigned = 0;

        for (uint32_t e = 0; e < experts_per_layer; ++e) {
            for (uint32_t l = 0; l < num_layers; ++l) {
                uint32_t gid = l * experts_per_layer + e;
                if (vram_assigned < vram_capacity) {
                    uint32_t slot = free_vram_slots.back();
                    free_vram_slots.pop_back();

                    catalog[gid].tier = ExpertTier::HOT_VRAM;
                    catalog[gid].slot_idx = static_cast<int32_t>(slot);
                    hot_vram_lru.push_front(gid);
                    catalog[gid].lru_it = hot_vram_lru.begin();
                    vram_slots[slot] = static_cast<int32_t>(gid);
                    vram_assigned++;
                } else if (host_assigned < host_capacity) {
                    uint32_t slot = free_host_slots.back();
                    free_host_slots.pop_back();

                    catalog[gid].tier = ExpertTier::WARM_HOST;
                    catalog[gid].slot_idx = static_cast<int32_t>(slot);
                    warm_host_lru.push_front(gid);
                    catalog[gid].lru_it = warm_host_lru.begin();
                    host_slots[slot] = static_cast<int32_t>(gid);
                    host_assigned++;
                } else {
                    return; // All VRAM and Host pools filled
                }
            }
        }
    }
};

} // namespace aeon::core
