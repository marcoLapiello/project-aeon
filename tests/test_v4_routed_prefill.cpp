// -----------------------------------------------------------------------------
// Step 4 — the routed bank: route-aware, cached prefill below the sweep's gate.
//
// A window shorter than the gate runs the same layer-major loop as the sweep, but
// the expert supply is the **routed bank** instead of a blind whole-layer load:
//
//   A. ROUTED, NOT SWEPT. With the sweep enabled and a window below the gate, the
//      window selects the routed strategy — `prefill_sweep_engaged()` is false while
//      a prefill supply is active.
//   B. BOUNDED DRAIN, SHARED RESTORE. It frees one layer's worth (`E`) of the
//      worst-LRU residents and preserves the rest, then restores the freed set on
//      exit, exactly as the sweep does (the shared registry machinery of Steps 1–2).
//   C. THE BANK'S CEILING. Each layer reads at most one layer's set from NVMe — the
//      union is fetched once and held to the layer boundary — and, below the gate,
//      strictly less than a whole layer, which is what makes the routed strategy the
//      right one for a short prompt.
//   D. CORRECT AND NON-VACUOUS. The window is byte-identical to the certified serial
//      path, dedup collapsed its `6C` draws, Warm was frozen (unchanged across the
//      window), and nothing leaked.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_graph.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace {

using aeon::core::ExpertTier;
using aeon::core::V4Graph;
using aeon::core::V4ModelHost;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kContext = 256;
// `W` is below the derived gate (`E / 4`, i.e. 64 for the V4 layer width), so the
// window selects the routed strategy. `C` divides `W` so a layer runs several chunks
// — which is what exercises the cross-chunk residency the bank exists to give.
constexpr uint32_t kWindow = 48;
constexpr uint32_t kChunk = 16;
// The host budget is the **total** pinned region: Warm *and* the transport corridor
// share it (`ExpertHostRegion`), so a figure that only covers the Warm residency the
// routed bank wants is now rejected at load. `8 GiB` covers the corridor at this
// chunk's shape and leaves the routed bank the residency it is here to exercise.
constexpr size_t kWarmBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr double kPayloadBytes = 14'155'776.0;

uint32_t g_checks = 0;
uint32_t g_failures = 0;

bool assert_that(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-56s %-34s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    ++g_checks;
    if (!ok) ++g_failures;
    return ok;
}

std::set<uint32_t> warm_set(const V4ModelHost& host) {
    std::set<uint32_t> experts;
    const auto& catalog = host.registry().catalog;
    for (uint32_t gid = 0; gid < catalog.size(); ++gid) {
        if (catalog[gid].owner == ExpertTier::WARM_HOST) {
            experts.insert(gid);
        }
    }
    return experts;
}

std::set<uint32_t> hot_set(const V4ModelHost& host) {
    std::set<uint32_t> experts;
    const auto& catalog = host.registry().catalog;
    for (uint32_t gid = 0; gid < catalog.size(); ++gid) {
        if (catalog[gid].owner == ExpertTier::HOT_VRAM) {
            experts.insert(gid);
        }
    }
    return experts;
}

size_t set_difference_size(const std::set<uint32_t>& a, const std::set<uint32_t>& b) {
    size_t differing = 0;
    for (uint32_t gid : a) {
        if (b.find(gid) == b.end()) ++differing;
    }
    for (uint32_t gid : b) {
        if (a.find(gid) == a.end()) ++differing;
    }
    return differing;
}

std::vector<uint8_t> read_bytes(const void* source, size_t bytes) {
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<uint8_t> host(bytes);
    CHECK_HIP(hipMemcpy(host.data(), source, bytes, hipMemcpyDeviceToHost));
    return host;
}

size_t differing_bytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) return a.size() + b.size();
    size_t differing = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++differing;
    }
    return differing;
}

} // namespace

int main(int argc, char** argv) {
    std::printf("================================================================================\n");
    std::printf("  Step 4 — the routed bank: cached prefill below the sweep's gate\n");
    std::printf("================================================================================\n");
    aeon::core::select_compute_device(true);

    // Optional cap on the Hot pool, so the floor configuration (`H == E`, where the
    // drain is the whole pool and nothing can be preserved) is exercised too. Zero
    // keeps the derived size.
    const uint32_t max_hot_slots =
        argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 0;

    aeon::core::AeonRuntimeConfig runtime;
    runtime.context_size = kContext;
    runtime.warm_host_bytes = kWarmBytes;
    runtime.prefill_chunk = kChunk;
    runtime.prefill_sweep = true;
    runtime.max_hot_vram_slots = max_hot_slots;

    V4ModelHost host;
    host.initialize(kModelDir, runtime, /*verbose=*/true);
    V4Graph graph(host);

    // At or above one layer the routed bank is feasible and is selected below the
    // gate; below one layer it is not, and the window falls back to the per-token
    // supply — which must still produce the same numbers.
    const bool routed_expected =
        host.prefill_sweep_enabled() && kWindow < host.prefill_sweep_min_tokens();

    std::vector<uint32_t> ids(kWindow);
    for (uint32_t i = 0; i < kWindow; ++i) {
        ids[i] = 1000 + i * 37;
    }

    const uint32_t vocab = static_cast<uint32_t>(host.config().vocab_size);
    const uint32_t hc_dim = static_cast<uint32_t>(host.config().hc_mult) *
                            static_cast<uint32_t>(host.config().hidden_size);
    const uint32_t per_layer = host.registry().experts_per_layer;
    const uint32_t layers = host.num_layers();
    const uint32_t gate = host.prefill_sweep_min_tokens();

    std::printf("\n  Hot %u slots, Warm %u experts, layer set %u, layers %u, gate %u\n",
                host.registry().vram_capacity,
                static_cast<uint32_t>(warm_set(host).size()), per_layer, layers, gate);

    // ---- the certified serial reference (decode-shaped, per token) -----------
    std::printf("\n[A] The serial reference (forward_token, one token at a time)\n");
    host.reset_generation_state();
    for (uint32_t position = 0; position < kWindow; ++position) {
        (void)graph.forward_token(ids[position], position, host.streams().compute);
    }
    const std::vector<uint8_t> serial_logits =
        read_bytes(graph.logits(), static_cast<size_t>(vocab) * sizeof(uint16_t));

    // ---- enter the routed prefill -------------------------------------------
    std::printf("\n[B] Prefill begin (bounded drain, routed bank)\n");
    const std::set<uint32_t> warm_before_switch = warm_set(host);
    host.prefill_begin(kWindow);
    const bool prefill_active = host.registry().prefill_streaming();
    const std::vector<uint32_t> drained = host.registry().restore_set();    const std::set<uint32_t> preserved = hot_set(host);
    std::set<uint32_t> switch_hot = preserved;
    switch_hot.insert(drained.begin(), drained.end());
    bool preserved_disjoint_from_drained = true;
    for (uint32_t gid : drained) {
        if (preserved.find(gid) != preserved.end()) {
            preserved_disjoint_from_drained = false;
        }
    }
    const std::set<uint32_t> warm_before = warm_set(host);
    const size_t switch_drift = set_difference_size(warm_before_switch, warm_before);

    // ---- run the window (the bank is already active) -------------------------
    std::printf("\n[C] The routed window (layer-major, per-chunk unions)\n");
    const uint64_t draws_before = host.expert_batch_draws();
    const uint64_t distinct_before = host.expert_batch_distinct();
    const uint64_t nvme_before = host.supply_bytes_from_nvme();
    host.reset_generation_state();
    (void)graph.forward_window(ids.data(), 0, kWindow, kChunk, host.streams().compute);
    const std::vector<uint8_t> routed_logits =
        read_bytes(graph.logits(), static_cast<size_t>(vocab) * sizeof(uint16_t));

    const uint64_t draws = host.expert_batch_draws() - draws_before;
    const uint64_t distinct = host.expert_batch_distinct() - distinct_before;
    const uint64_t nvme_bytes = host.supply_bytes_from_nvme() - nvme_before;

    const std::set<uint32_t> warm_after = warm_set(host);
    const size_t warm_drift = set_difference_size(warm_before, warm_after);
    const std::set<uint32_t> hot_after = hot_set(host);

    std::printf("\n--- results ---\n");

    const bool bank_engaged = prefill_active;
    assert_that("A: the default gate is three quarters of the layer width",
                gate == (per_layer * 3) / 4,
                "gate " + std::to_string(gate) + " = 3E/4 of " + std::to_string(per_layer));
    assert_that("A: a window at the gate would engage the sweep",
                host.prefill_sweep_engaged_for(gate),
                "window " + std::to_string(gate));
    assert_that("A: a window below the gate does not",
                !host.prefill_sweep_engaged_for(gate - 1),
                "window " + std::to_string(gate - 1));
    assert_that("A: the routed bank is selected exactly when the pool can hold a layer",
                bank_engaged == routed_expected,
                std::string("active=") + (bank_engaged ? "true" : "false") +
                    ", expected=" + (routed_expected ? "true" : "false") +
                    ", engaged=" + (host.prefill_sweep_engaged() ? "true" : "false"));
    assert_that("A: the window below the gate never chose the sweep",
                !host.prefill_sweep_engaged(),
                std::string("engaged=") + (host.prefill_sweep_engaged() ? "true" : "false"));
    if (bank_engaged) {
        const uint32_t expected_drain =
            per_layer < host.registry().vram_capacity ? per_layer
                                                      : host.registry().vram_capacity;
        assert_that("B: the drain freed one layer's worth (or the whole floor pool)",
                    drained.size() == expected_drain,
                    std::to_string(drained.size()) + " drained of " +
                        std::to_string(host.registry().vram_capacity));
        assert_that("B: preserved and drained partition the switch-point pool",
                    preserved_disjoint_from_drained &&
                        preserved.size() + drained.size() == host.registry().vram_capacity,
                    std::to_string(preserved.size()) + " preserved + " +
                        std::to_string(drained.size()) + " drained = " +
                        std::to_string(host.registry().vram_capacity));
    }

    const size_t logits_diff = differing_bytes(serial_logits, routed_logits);
    assert_that("D: routed logits == serial, bit-exact", logits_diff == 0,
                std::to_string(logits_diff) + " differing of " +
                    std::to_string(serial_logits.size()));

    assert_that("D: dedup collapsed the chunk draws", distinct < draws,
                std::to_string(distinct) + " distinct of " + std::to_string(draws) + " draws");

    if (bank_engaged) {
        const double nvme_experts = static_cast<double>(nvme_bytes) / kPayloadBytes;
        const double layer_sets = static_cast<double>(layers) * static_cast<double>(per_layer);
        assert_that("C: the bank read at most one layer's set per layer",
                    nvme_experts <= layer_sets,
                    std::to_string(nvme_experts) + " experts vs ceiling " +
                        std::to_string(layer_sets));
        assert_that("C: below the gate the bank reads a fraction of a sweep",
                    nvme_experts < 0.8 * layer_sets,
                    std::to_string(nvme_experts) + " experts, a sweep reads " +
                        std::to_string(layer_sets));
        assert_that("B: the Hot set is restored to its switch-point set",
                    hot_after == switch_hot,
                    std::to_string(hot_after.size()) + " Hot, expected " +
                        std::to_string(switch_hot.size()));
    }
    assert_that("D: Warm is unchanged across the whole window", warm_drift == 0,
                std::to_string(warm_drift) + " experts differ of " +
                    std::to_string(warm_before.size()) +
                    " (switch settled " + std::to_string(switch_drift) + ")");
    assert_that("D: no shadow survived the window",
                host.registry().shadow_resident_count() == 0,
                std::to_string(host.registry().shadow_resident_count()) + " shadows");
    assert_that("D: streaming mode was left", !host.registry().prefill_streaming(),
                "unfrozen");
    assert_that("D: no lease leaked", host.outstanding_expert_leases() == 0,
                std::to_string(host.outstanding_expert_leases()) + " outstanding");
    assert_that("D: the staging arena drained", host.staging_in_use_slots() == 0,
                std::to_string(host.staging_in_use_slots()) + " slots in use");
    assert_that("D: registry invariants hold", host.registry().invariants_hold(),
                "invariants_hold()");

    std::printf("\n  window %u tokens (chunk %u) over %u layers, %u experts each, "
                "NVMe %.3f GiB\n",
                kWindow, kChunk, layers, per_layer, nvme_bytes / 1073741824.0);
    (void)hc_dim;

    std::printf("\n================================================================================\n");
    std::printf("  %u checks, %u failures\n", g_checks, g_failures);
    std::printf("================================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
