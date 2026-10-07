#pragma once

// -----------------------------------------------------------------------------
// The parallel topology: which physical devices the model is spread across, and
// how the layers divide between them.
//
// Pure and model-agnostic (G1): it knows device ids, the two degrees of freedom,
// and the layer count, and nothing else. It does not touch HIP — the caller
// supplies the visible-device count — so a host test can exercise every
// acceptance and every refusal without a GPU.
//
// The two axes are **composed**, not alternatives: `tp` ranks share a stage's
// work by slicing every tensor, and `pp` stages each own a contiguous layer
// range and hand the residual to the next. A stage's device set is its own `tp`
// ranks, laid out stage-major — `device(stage, rank) = device_ids[stage * tp +
// rank]` — so a caller orders the ids to put each TP group on one PCIe switch.
//
// A stage's layer ranges are contiguous and cover `[0, num_layers)` exactly. The
// remainder of an uneven divide goes to the **earliest** stages, so a stage is
// never empty and the head-bearing last stage is the one that can be smallest
// (43 layers is prime, so some split is uneven whatever the degree).
// -----------------------------------------------------------------------------

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace aeon::core {

// The knobs, as they arrive from the CLI / runtime config.
struct ParallelTopologyConfig {
    // Explicit device ids in stage-major order. Empty means the default
    // `0 .. tp * pp - 1`, which a caller resolves against the visible devices.
    std::vector<int> device_ids;
    uint32_t tensor_parallel{1};
    uint32_t pipeline_parallel{1};
};

// A contiguous, half-open run of layers: `[first, first + count)`.
struct LayerRange {
    uint32_t first{0};
    uint32_t count{0};

    uint32_t end() const noexcept { return first + count; }
    bool contains(uint32_t layer) const noexcept {
        return layer >= first && layer < end();
    }
};

// Parse a `"0,1,3"` id list. An empty string yields an empty vector, which the
// topology reads as "use the default". Whitespace around an entry is ignored; an
// empty entry or a non-numeric entry is refused with the offending text named.
inline std::vector<int> parse_device_ids(std::string_view text) {
    std::vector<int> ids;
    size_t start = 0;
    while (start <= text.size()) {
        size_t comma = text.find(',', start);
        if (comma == std::string_view::npos) comma = text.size();
        std::string_view entry = text.substr(start, comma - start);
        // Trim ASCII whitespace.
        while (!entry.empty() && (entry.front() == ' ' || entry.front() == '\t')) {
            entry.remove_prefix(1);
        }
        while (!entry.empty() && (entry.back() == ' ' || entry.back() == '\t')) {
            entry.remove_suffix(1);
        }
        if (entry.empty()) {
            // A trailing or doubled comma is a malformed list, not a default.
            if (text.empty()) return ids;
            throw std::invalid_argument(
                "device id list has an empty entry: \"" + std::string(text) + "\"");
        }
        size_t consumed = 0;
        long parsed = 0;
        try {
            parsed = std::stol(std::string(entry), &consumed, 10);
        } catch (const std::exception&) {
            throw std::invalid_argument("invalid device id: \"" + std::string(entry) + "\"");
        }
        if (consumed != entry.size()) {
            throw std::invalid_argument("invalid device id: \"" + std::string(entry) + "\"");
        }
        if (parsed < 0) {
            throw std::invalid_argument(
                "device id must be non-negative: \"" + std::string(entry) + "\"");
        }
        ids.push_back(static_cast<int>(parsed));
        if (comma == text.size()) break;
        start = comma + 1;
    }
    return ids;
}

// A resolved topology: the device id per (stage, rank) and the layer ranges. All
// the validation happens in `resolve`, so the accessors are total.
class ParallelTopology {
public:
    // `visible_devices` is the caller's `hipGetDeviceCount`; `artifact_max_tp` is
    // the maximum tensor-parallel degree the artifact declares (1 for a dense-only
    // or v1 artifact), `num_layers` the model's layer count.
    static ParallelTopology resolve(
        const ParallelTopologyConfig& config,
        int visible_devices,
        uint32_t artifact_max_tp,
        uint32_t num_layers
    ) {
        ParallelTopology topo;
        const uint32_t tp = config.tensor_parallel;
        const uint32_t pp = config.pipeline_parallel;

        if (tp == 0 || pp == 0) {
            throw std::invalid_argument(
                "parallel topology: tensor and pipeline degrees must be at least 1 "
                "(tensor_parallel=" + std::to_string(tp) +
                ", pipeline_parallel=" + std::to_string(pp) + ")");
        }
        if (num_layers == 0) {
            throw std::invalid_argument("parallel topology: the model has no layers");
        }
        if (artifact_max_tp == 0) {
            throw std::invalid_argument(
                "parallel topology: the artifact's max tensor-parallel degree is 0");
        }
        if (artifact_max_tp % tp != 0) {
            throw std::invalid_argument(
                "parallel topology: tensor_parallel=" + std::to_string(tp) +
                " does not divide the artifact's max tensor-parallel degree " +
                std::to_string(artifact_max_tp));
        }
        if (pp > num_layers) {
            throw std::invalid_argument(
                "parallel topology: pipeline_parallel=" + std::to_string(pp) +
                " exceeds the " + std::to_string(num_layers) + " model layers");
        }

        const size_t rank_count = static_cast<size_t>(tp) * static_cast<size_t>(pp);
        std::vector<int> ids = config.device_ids;
        if (ids.empty()) {
            ids.reserve(rank_count);
            for (size_t i = 0; i < rank_count; ++i) ids.push_back(static_cast<int>(i));
        }
        if (ids.size() != rank_count) {
            throw std::invalid_argument(
                "parallel topology: device_ids has " + std::to_string(ids.size()) +
                " entries but tp * pp = " + std::to_string(rank_count));
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || ids[i] >= visible_devices) {
                throw std::invalid_argument(
                    "parallel topology: device id " + std::to_string(ids[i]) +
                    " is out of range for " + std::to_string(visible_devices) +
                    " visible devices");
            }
            for (size_t j = 0; j < i; ++j) {
                if (ids[i] == ids[j]) {
                    throw std::invalid_argument(
                        "parallel topology: device id " + std::to_string(ids[i]) +
                        " is listed more than once");
                }
            }
        }

        topo.tp_ = tp;
        topo.pp_ = pp;
        topo.num_layers_ = num_layers;
        topo.device_ids_ = std::move(ids);

        // Contiguous ranges, remainder to the earliest stages (so no stage is
        // empty and the head stage can be the smallest).
        const uint32_t base = num_layers / pp;
        const uint32_t remainder = num_layers % pp;
        topo.stages_.reserve(pp);
        uint32_t first = 0;
        for (uint32_t stage = 0; stage < pp; ++stage) {
            const uint32_t count = base + (stage < remainder ? 1u : 0u);
            topo.stages_.push_back(LayerRange{first, count});
            first += count;
        }
        return topo;
    }

    uint32_t tp() const noexcept { return tp_; }
    uint32_t pp() const noexcept { return pp_; }

    // Stage-major: `device_ids[stage * tp + rank]`.
    int device(uint32_t stage, uint32_t rank) const noexcept {
        return device_ids_[static_cast<size_t>(stage) * tp_ + rank];
    }

    LayerRange stage_layers(uint32_t stage) const noexcept { return stages_[stage]; }

    uint32_t stage_of(uint32_t layer) const noexcept {
        uint32_t stage = 0;
        while (stage + 1 < stages_.size() && stages_[stage].end() <= layer) ++stage;
        return stage;
    }

    bool is_single_device() const noexcept { return tp_ == 1 && pp_ == 1; }

    const std::vector<int>& device_ids() const noexcept { return device_ids_; }

private:
    uint32_t tp_{1};
    uint32_t pp_{1};
    uint32_t num_layers_{0};
    std::vector<int> device_ids_;
    std::vector<LayerRange> stages_;
};

} // namespace aeon::core
