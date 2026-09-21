#include "infrastructure/core/routing_reuse.hpp"

#include <cassert>
#include <iostream>
#include <vector>

// Hand-computed stack distances (anti-circularity: the expected values are derived
// by hand, not from the profiler's own helpers).
//
// Trace over global expert ids, all in layer 3 (a learned layer):
//   1, 2, 3, 4, 5, 6, 7, 1
//   - 1..7: first ever                -> compulsory
//   - 1: last at pos 0; distinct experts selected in (0, 6] = {2..7} -> distance 6
// So: observed 8, compulsory 7, one distance-6 observation. But the default
// capacity list has no 6... it does: 6 is the first entry. At capacity 6 that
// request misses (6 < 6 is false); at capacity 64 it hits.

using aeon::core::RoutingReuseProfiler;

static std::vector<RoutingReuseProfiler::Observation> obs(const std::vector<uint32_t>& ids,
                                                          bool hot = false) {
    std::vector<RoutingReuseProfiler::Observation> out;
    for (uint32_t id : ids) out.push_back({id, hot});
    return out;
}

static double rate_at(const RoutingReuseProfiler::Curve& c, uint64_t capacity) {
    for (size_t i = 0; i < c.capacities.size(); ++i) {
        if (c.capacities[i] == capacity) return c.ideal_lru_hit_rate[i];
    }
    return -1.0;
}

static double opt_at(const RoutingReuseProfiler::Curve& c, uint64_t capacity) {
    for (size_t i = 0; i < c.capacities.size(); ++i) {
        if (c.capacities[i] == capacity) return c.opt_hit_rate[i];
    }
    return -1.0;
}

int main() {
    std::cout << "=== routing reuse-distance profiler ===\n";

    // 1. Distances and the capacity boundary: distance 6 misses at capacity 6 and
    //    hits at capacity 64.
    {
        RoutingReuseProfiler p;
        p.reset(16);
        auto layer = obs({1, 2, 3, 4, 5, 6, 7, 1});
        p.observe_layer(3, layer.data(), layer.size());

        const auto c = p.curve();
        assert(c.observed == 8 && "observed count");
        assert(c.compulsory == 7 && "compulsory count");
        assert(rate_at(c, 6) == 0.0 && "distance 6 misses at capacity 6");
        assert(rate_at(c, 64) == 0.125 && "distance 6 hits at capacity 64");
        std::cout << "  [PASSED] distance-6 boundary honored (miss@6, hit@64)\n";
    }

    // 2. A repeated expert at distance 0 hits at any capacity >= 1.
    {
        RoutingReuseProfiler p;
        p.reset(16);
        auto layer = obs({5, 5, 5});
        p.observe_layer(3, layer.data(), layer.size());
        const auto c = p.curve();
        assert(c.observed == 3 && c.compulsory == 1);
        assert(rate_at(c, 6) == 2.0 / 3.0 && "two distance-0 hits of three");
        std::cout << "  [PASSED] immediate reuse gives distance 0\n";
    }

    // 3. Hash-router layers (0-2) are excluded from the aggregate but still
    //    advance the cache clock.
    {
        RoutingReuseProfiler p;
        p.reset(16);
        auto hash_layer = obs({7});
        p.observe_layer(0, hash_layer.data(), hash_layer.size());   // not counted
        auto learned = obs({7});
        p.observe_layer(3, learned.data(), learned.size());

        const auto c = p.curve();
        assert(c.observed == 1 && "only the learned-layer request is counted");
        assert(c.compulsory == 0 && "the hash-layer selection made it non-compulsory");
        assert(rate_at(c, 6) == 1.0 && "distance 0 -> hit");
        std::cout << "  [PASSED] layers 0-2 excluded but advance the clock\n";
    }

    // 4. Measured Hot hit rate is tracked alongside the ideal curve.
    {
        RoutingReuseProfiler p;
        p.reset(16);
        std::vector<RoutingReuseProfiler::Observation> layer{{1, true}, {2, false}, {3, true}};
        p.observe_layer(3, layer.data(), layer.size());
        const auto c = p.curve();
        assert(c.observed == 3);
        assert(c.measured_hit_rate > 0.66 && c.measured_hit_rate < 0.67);
        std::cout << "  [PASSED] measured Hot hit rate = 2/3\n";
    }

    // 5. Belady-OPT vs ideal-LRU on an LRU-pessimal trace.
    //    Trace 1,2,3,4,5,6,7,1,2,3,4,5,6,7 at capacity 6.
    //    LRU: after the first fill it evicts exactly the item the next round needs,
    //      so it scores 0 hits of 14.
    //    OPT: on the id7 miss it evicts the farthest next use (id6), keeping 1..5;
    //      hits 1..5 on the next five requests, evicts id1 for id6, then hits id7.
    //      That is 6 hits of 14.
    {
        RoutingReuseProfiler p;
        p.reset(16);
        auto layer = obs({1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7});
        p.observe_layer(3, layer.data(), layer.size());
        const auto c = p.curve();
        assert(rate_at(c, 6) == 0.0 && "LRU scores zero on the cycling trace");
        assert(opt_at(c, 6) > 6.0 / 14.0 - 1e-9 && "OPT hits six of fourteen");
        // OPT is optimal: it can never be worse than ideal-LRU at any capacity.
        for (size_t i = 0; i < c.capacities.size(); ++i) {
            assert(c.opt_hit_rate[i] + 1e-12 >= c.ideal_lru_hit_rate[i]);
        }
        std::cout << "  [PASSED] OPT 6/14 vs LRU 0/14 on the cycling trace; OPT >= LRU\n";
    }

    std::cout << "=== all routing-reuse tests passed ===\n";
    return 0;
}