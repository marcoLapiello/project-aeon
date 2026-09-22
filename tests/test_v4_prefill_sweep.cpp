// -----------------------------------------------------------------------------
// Step 6 item 6 — the prefill sweep: drain, layer-ordered streaming, empty on exit.
//
// Prefill and decode are two allocation strategies, not one with a parameter, so
// the switch between them is asserted as a switch. Through the real host (43
// layers, the real supply, a real Warm tier):
//
//   A. DRAIN. Entering a swept prefill empties the Hot pool outright — no decode
//      resident survives, because none of them is in the plan the sweep follows.
//   B. WARM PRESERVED. Warm's resident set is identical before and after the whole
//      swept window, because the sweep allocates Hot from the free list only and
//      releases by layer; it never promotes from Warm and never demotes into it.
//   C. EMPTY ON EXIT. The last layer's release leaves Hot empty, so the window
//      hands decode a clean pool. No shadow survives (decode admits single
//      ownership), no lease leaks, and the registry invariants hold throughout.
//   D. NON-VACUOUS AND CORRECT. The window streams every layer exactly once
//      (`experts_streamed == layers x experts_per_layer`), the lookahead holds more
//      than one layer's worth (so it is a sliding window, not depth 1), and the
//      result is still **byte-identical** to the certified serial path — a residency
//      policy that changed a number would not be a residency policy.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_graph.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

using aeon::core::ExpertTier;
using aeon::core::V4Graph;
using aeon::core::V4ModelHost;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kContext = 256;
constexpr uint32_t kWindow = 16;
constexpr size_t kWarmBytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;

uint32_t g_checks = 0;
uint32_t g_failures = 0;

bool assert_that(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-54s %-34s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
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

int main() {
    std::printf("================================================================================\n");
    std::printf("  Step 6 item 6 — the prefill sweep: drain, layer order, empty on exit\n");
    std::printf("================================================================================\n");
    aeon::core::select_compute_device(true);

    aeon::core::AeonRuntimeConfig runtime;
    runtime.context_size = kContext;
    runtime.warm_host_bytes = kWarmBytes;
    runtime.prefill_chunk = kWindow;
    runtime.prefill_sweep = true;

    V4ModelHost host;
    host.initialize(kModelDir, runtime, /*verbose=*/true);
    V4Graph graph(host);

    if (!host.prefill_sweep_enabled()) {
        std::printf("\n  the sweep is not feasible in this configuration — aborting\n");
        return 1;
    }

    std::vector<uint32_t> ids(kWindow);
    for (uint32_t i = 0; i < kWindow; ++i) {
        ids[i] = 1000 + i * 37;
    }

    const uint32_t vocab = static_cast<uint32_t>(host.config().vocab_size);
    const uint32_t hc_dim = static_cast<uint32_t>(host.config().hc_mult) *
                            static_cast<uint32_t>(host.config().hidden_size);
    const uint32_t per_layer = host.registry().experts_per_layer;
    const uint32_t layers = host.num_layers();

    std::printf("\n  Hot %u slots, Warm %u experts, layer set %u, layers %u\n",
                host.registry().vram_capacity,
                static_cast<uint32_t>(warm_set(host).size()), per_layer, layers);

    // ---- the certified serial reference (decode-shaped, per token) -----------
    std::printf("\n[A] The serial reference (forward_token, one token at a time)\n");
    host.reset_generation_state();
    for (uint32_t position = 0; position < kWindow; ++position) {
        (void)graph.forward_token(ids[position], position, host.streams().compute);
    }
    const std::vector<uint8_t> serial_logits =
        read_bytes(graph.logits(), static_cast<size_t>(vocab) * sizeof(uint16_t));

    // ---- enter the swept prefill --------------------------------------------
    std::printf("\n[B] Prefill begin (drain Hot, freeze Warm)\n");
    const std::set<uint32_t> warm_before_switch = warm_set(host);
    host.prefill_begin();
    const uint32_t shadows_after_drain = host.registry().shadow_resident_count();
    const bool streaming = host.registry().prefill_streaming();
    // The switch deliberately settles and reaps what the previous phase left in
    // flight, so a demotion the serial reference had already committed can land
    // here. Warm's baseline for the *sweep* is therefore taken after the switch,
    // not before it.
    const std::set<uint32_t> warm_before = warm_set(host);
    const size_t switch_drift = set_difference_size(warm_before_switch, warm_before);

    // ---- run the window (the sweep is already active) ------------------------
    std::printf("\n[C] The swept window (layer-major, whole layer sets)\n");
    host.reset_generation_state();
    (void)graph.forward_window(ids.data(), 0, kWindow, kWindow, host.streams().compute);
    const std::vector<uint8_t> swept_logits =
        read_bytes(graph.logits(), static_cast<size_t>(vocab) * sizeof(uint16_t));

    const std::set<uint32_t> warm_after = warm_set(host);
    const size_t warm_drift = set_difference_size(warm_before, warm_after);

    std::printf("\n--- results ---\n");

    assert_that("B: prefill entered streaming mode", streaming, "streaming");
    assert_that("B: Hot was drained on entry",
                host.prefill_sweep().hot_after_drain() == 0,
                std::to_string(host.prefill_sweep().hot_after_drain()) + " Hot residents");
    assert_that("B: no shadow survived the drain", shadows_after_drain == 0,
                std::to_string(shadows_after_drain) + " shadows");

    const size_t logits_diff = differing_bytes(serial_logits, swept_logits);
    assert_that("D: swept logits == serial, bit-exact", logits_diff == 0,
                std::to_string(logits_diff) + " differing of " +
                    std::to_string(serial_logits.size()));

    assert_that("D: every layer streamed exactly once",
                host.prefill_sweep().experts_streamed() ==
                    static_cast<uint64_t>(layers) * per_layer,
                std::to_string(host.prefill_sweep().experts_streamed()) + " experts in " +
                    std::to_string(host.prefill_sweep().layer_loads()) + " loads");
    assert_that("D: the lookahead holds more than one layer", 
                host.prefill_sweep().frontier_depth() > 1,
                std::to_string(host.prefill_sweep().frontier_depth()) + " layers deep");

    assert_that("C: Hot is empty again after the window",
                host.registry().published_hot_slots() == 0,
                std::to_string(host.registry().published_hot_slots()) + " Hot residents");
    assert_that("C: no shadow survives the window",
                host.registry().shadow_resident_count() == 0,
                std::to_string(host.registry().shadow_resident_count()) + " shadows");
    assert_that("C: streaming mode was left", !host.registry().prefill_streaming(),
                "unfrozen");
    assert_that("B: Warm is unchanged across the whole sweep", warm_drift == 0,
                std::to_string(warm_drift) + " experts differ of " +
                    std::to_string(warm_before.size()) +
                    " (switch settled " + std::to_string(switch_drift) + ")");
    if (warm_drift != 0) {
        for (uint32_t gid : warm_before) {
            if (warm_after.find(gid) == warm_after.end()) {
                std::printf("      Warm expert %u was resident before only\n", gid);
            }
        }
        for (uint32_t gid : warm_after) {
            if (warm_before.find(gid) == warm_before.end()) {
                std::printf("      Warm expert %u is resident after only\n", gid);
            }
        }
    }
    assert_that("C: no lease leaked", host.outstanding_expert_leases() == 0,
                std::to_string(host.outstanding_expert_leases()) + " outstanding");
    assert_that("C: the staging arena drained", host.staging_in_use_slots() == 0,
                std::to_string(host.staging_in_use_slots()) + " slots in use");
    assert_that("C: registry invariants hold", host.registry().invariants_hold(),
                "invariants_hold()");

    std::printf("\n  window %u tokens over %u layers, %u experts each\n",
                kWindow, layers, per_layer);
    (void)hc_dim;

    std::printf("\n================================================================================\n");
    std::printf("  %u checks, %u failures\n", g_checks, g_failures);
    std::printf("================================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
