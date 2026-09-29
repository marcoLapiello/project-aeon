#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aeon::core {

enum class RoutingPhase : uint8_t {
    Prefill = 0,
    Decode = 1,
};

class RoutingCounter {
public:
    static constexpr uint32_t kPhaseCount = 2;
    static constexpr uint32_t kExpertCount = 256;
    static constexpr uint32_t kTopK = 6;

    explicit RoutingCounter(uint32_t num_layers)
        : num_layers_(num_layers),
          counts_(static_cast<size_t>(kPhaseCount) * num_layers * kExpertCount, 0) {}

    void reset() {
        std::fill(counts_.begin(), counts_.end(), uint64_t{0});
    }

    void record(
        RoutingPhase phase,
        uint32_t layer_id,
        uint32_t position,
        const int32_t* expert_ids
    ) {
        (void)position;
        if (static_cast<uint32_t>(phase) >= kPhaseCount || layer_id >= num_layers_ || expert_ids == nullptr) {
            return;
        }

        for (uint32_t k = 0; k < kTopK; ++k) {
            int32_t expert_id = expert_ids[k];
            if (expert_id >= 0 && static_cast<uint32_t>(expert_id) < kExpertCount) {
                ++counts_[index(phase, layer_id, static_cast<uint32_t>(expert_id))];
            }
        }
    }

    uint64_t selection_count(RoutingPhase phase, uint32_t layer_id, uint32_t expert_id) const {
        if (static_cast<uint32_t>(phase) >= kPhaseCount || layer_id >= num_layers_ || expert_id >= kExpertCount) {
            return 0;
        }
        return counts_[index(phase, layer_id, expert_id)];
    }

    uint64_t total_selections(RoutingPhase phase, uint32_t layer_id) const {
        uint64_t total = 0;
        for (uint32_t expert_id = 0; expert_id < kExpertCount; ++expert_id) {
            total += selection_count(phase, layer_id, expert_id);
        }
        return total;
    }

    uint32_t num_layers() const {
        return num_layers_;
    }

private:
    size_t index(RoutingPhase phase, uint32_t layer_id, uint32_t expert_id) const {
        return (static_cast<size_t>(phase) * num_layers_ + layer_id) * kExpertCount + expert_id;
    }

    uint32_t num_layers_;
    std::vector<uint64_t> counts_;
};

} // namespace aeon::core