#pragma once

// -----------------------------------------------------------------------------
// Decode routing reuse-distance profiler (Phase 1 of the routing study).
//
// The question this answers is not "does routing concentrate in aggregate" —
// that is what `RoutingCounter`/`RoutingProfile` already measure — but **"is
// there reuse that the current recency policy is failing to capture?"** A layer
// can look concentrated in aggregate and still have no temporal locality (a
// different popular expert each token), in which case no placement policy helps.
//
// The instrument is the **reuse distance** of every request, measured as a
// *stack distance*: the number of distinct experts selected since that expert was
// last selected. A fully-associative cache of `N` expert slots answers a request
// from memory iff its stack distance is `< N` — the stack property of LRU — so the
// cumulative distance histogram is exactly the hit-rate curve of an ideal LRU of
// capacity `N`. Comparing that curve against the artifact's *measured* Hot hit
// rate at the same capacity says how much locality the current implementation
// leaves on the table:
//
//   * measured << ideal-LRU(N)  -> implementation headroom (the LRU-thrash
//     thesis); a placement change is worth pursuing.
//   * measured ~= ideal-LRU(N)  -> the recency policy is already at its ceiling
//     and only a *different strategy* (frequency/OPT) could beat it.
//
// Routing differs by layer, but the Hot/Warm pools are global, so the stream is
// global (global expert ids) and the pools compete across layers.
//
// Layers 0–2 use a hash router keyed on token id: their routing is flat and
// non-repeating by construction. Their requests still *occupy* the cache (the
// clock advances through them), but their outcomes are excluded from the reported
// aggregate — including them would bias every number toward "no locality".
// -----------------------------------------------------------------------------

#include <algorithm>
#include <cstdint>
#include <vector>

namespace aeon::core {

class RoutingReuseProfiler {
public:
    // Requests below this layer index are excluded from the reported aggregate.
    static constexpr uint32_t kFirstLearnedLayer = 3;

    struct Observation {
        uint32_t global_expert_id{0};
        bool answered_from_hot{false};
    };

    // Capacity points, in expert slots, chosen to map onto real hardware
    // quantities: one layer's routed set, the measured Hot pool, the measured Warm
    // pool, and the whole model.
    static const std::vector<uint64_t>& default_capacities() {
        static const std::vector<uint64_t> caps{6, 64, 128, 258, 779, 1558, 3022, 11008};
        return caps;
    }

    struct Curve {
        std::vector<uint64_t> capacities;
        std::vector<double> ideal_lru_hit_rate;   // per capacity, 0..1
        double measured_hit_rate{0.0};            // Hot answers / observed requests
        uint64_t observed{0};                     // learned-layer requests
        uint64_t compulsory{0};                   // first-ever selections
    };

    void reset(uint32_t total_experts) {
        total_experts_ = total_experts;
        last_pos_.assign(total_experts, -1);
        // Exact distance histogram; distances above `total_experts_` collapse into
        // one overflow bucket (they are misses at every capacity we report anyway).
        distance_hist_.assign(static_cast<size_t>(total_experts_) + 1, 0);
        overflow_ = 0;
        // 1-based Fenwick over request positions; index 0 is unused. Start at a
        // capacity that covers a short run and grow (by rebuild) when exceeded.
        fenwick_capacity_ = std::max<size_t>(total_experts_, 1024);
        fenwick_.assign(fenwick_capacity_ + 1, 0);
        clock_ = 0;
        observed_ = 0;
        measured_hot_ = 0;
        compulsory_ = 0;
        enabled_ = true;
    }

    bool enabled() const noexcept { return enabled_; }
    void set_enabled(bool value) noexcept { enabled_ = value; }

    uint64_t observed() const noexcept { return observed_; }

    // Raw distance histogram for tests and diagnostics: `histogram[d]` counts
    // learned-layer requests whose stack distance was exactly `d`.
    const std::vector<uint64_t>& histogram() const noexcept { return distance_hist_; }

    // Feed one layer's requests, in slot order. `layer_id` selects whether the
    // outcome is aggregated; the clock always advances so the cache model stays
    // correct across every layer.
    void observe_layer(uint32_t layer_id, const Observation* requests, size_t count) {
        if (!enabled_ || total_experts_ == 0) return;
        const bool learned = layer_id >= kFirstLearnedLayer;
        for (size_t i = 0; i < count; ++i) {
            const uint32_t gid = requests[i].global_expert_id;
            if (gid >= total_experts_) continue;

            const int64_t last = last_pos_[gid];
            if (last < 0) {
                if (learned) {
                    ++observed_;
                    ++compulsory_;
                    if (requests[i].answered_from_hot) ++measured_hot_;
                }
                add_mark(clock_, +1);
                last_pos_[gid] = static_cast<int64_t>(clock_);
                ++clock_;
                continue;
            }

            // Distinct experts selected since this gid's last selection: the count
            // of marks in (last, clock-1].
            const int64_t distance =
                fenwick_sum(clock_ - 1) - fenwick_sum(static_cast<size_t>(last));
            if (learned) {
                ++observed_;
                if (requests[i].answered_from_hot) ++measured_hot_;
                record_distance(distance);
            }
            add_mark(static_cast<size_t>(last), -1);
            add_mark(clock_, +1);
            last_pos_[gid] = static_cast<int64_t>(clock_);
            ++clock_;
        }
    }

    Curve curve() const {
        Curve result;
        result.capacities = default_capacities();
        result.observed = observed_;
        result.compulsory = compulsory_;
        result.measured_hit_rate = observed_ == 0
            ? 0.0
            : static_cast<double>(measured_hot_) / static_cast<double>(observed_);

        for (const uint64_t capacity : result.capacities) {
            uint64_t hits = 0;
            const size_t limit = std::min<size_t>(
                distance_hist_.size(), static_cast<size_t>(capacity));
            for (size_t distance = 0; distance < limit; ++distance) {
                hits += distance_hist_[distance];
            }
            result.ideal_lru_hit_rate.push_back(
                observed_ == 0
                    ? 0.0
                    : static_cast<double>(hits) / static_cast<double>(observed_));
        }
        return result;
    }

private:
    void record_distance(int64_t distance) {
        if (distance < 0) distance = 0;
        const size_t index = static_cast<size_t>(distance);
        if (index < distance_hist_.size()) {
            ++distance_hist_[index];
        } else {
            ++overflow_;
        }
    }

    // Fenwick tree over request positions, marking where each expert was last
    // selected. The number of distinct experts selected in a range is the count of
    // marks in it, which is why the tree is needed rather than a plain counter.
    //
    // Growth is by rebuild, not by appending zeros: a Fenwick node's covered range
    // is fixed by its low bit, so a node created late must still receive the marks
    // of the earlier positions it covers. Appending zeros silently drops them.
    void add_mark(size_t index, int64_t delta) {
        if (index + 1 > fenwick_capacity_) {
            grow(index + 1);
        }
        for (size_t i = index + 1; i <= fenwick_capacity_; i += i & (~i + 1)) {
            fenwick_[i] += delta;
        }
    }

    int64_t fenwick_sum(size_t index) const {
        int64_t total = 0;
        for (size_t i = std::min(index + 1, fenwick_capacity_); i > 0; i -= i & (~i + 1)) {
            total += fenwick_[i];
        }
        return total;
    }

    void grow(size_t needed) {
        size_t capacity = fenwick_capacity_;
        while (capacity < needed) capacity *= 2;
        std::vector<int64_t> fresh(capacity + 1, 0);
        for (uint32_t gid = 0; gid < total_experts_; ++gid) {
            if (last_pos_[gid] < 0) continue;
            for (size_t i = static_cast<size_t>(last_pos_[gid]) + 1; i <= capacity;
                 i += i & (~i + 1)) {
                fresh[i] += 1;
            }
        }
        fenwick_ = std::move(fresh);
        fenwick_capacity_ = capacity;
    }

    bool enabled_{false};
    uint32_t total_experts_{0};
    std::vector<int64_t> last_pos_;   // per global expert; -1 = never selected
    std::vector<int64_t> fenwick_;    // 1-based; index 0 unused
    size_t fenwick_capacity_{0};
    size_t clock_{0};                 // request index across the whole stream
    uint64_t observed_{0};
    uint64_t measured_hot_{0};
    uint64_t compulsory_{0};
    uint64_t overflow_{0};
    std::vector<uint64_t> distance_hist_{};
};

} // namespace aeon::core
