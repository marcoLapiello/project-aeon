// -----------------------------------------------------------------------------
// Step 6 item 6 — the prefill sweep: bounded drain, layer-ordered streaming, restore.
//
// Prefill and decode are two allocation strategies, not one with a parameter, so
// the switch between them is asserted as a switch. Through the real host (43
// layers, the real supply, a real Warm tier):
//
//   A. BOUNDED DRAIN. Entering a swept prefill frees only what the pass needs —
//      `2E` when the pool can hold two layer sets, else `E` — worst-LRU first. The
//      remaining residents are **preserved** and survive the whole pass.
//   B. WARM PRESERVED. Warm's resident set is identical before and after the whole
//      swept window, because the sweep allocates Hot from the free list only and
//      releases by layer; it never promotes from Warm and never demotes into it.
//   C. RESTORED ON EXIT. The window hands decode back the set it had before the
//      pass: the preserved residents never left, and the drained set is reloaded
//      through the normal cold path. No shadow survives (decode admits single
//      ownership), no lease leaks, and the registry invariants hold throughout.
//   D. NON-VACUOUS AND CORRECT. Every layer is visited exactly once and its missing
//      experts streamed once (`streamed + preserved == layers x experts_per_layer`),
//      the lookahead holds more than one layer's worth (so it is a sliding window,
//      not depth 1), and the result is still **byte-identical** to the certified
//      serial path — a residency policy that changed a number would not be a
//      residency policy.
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
// The window `W` and the body chunk `C`. `C` is the body's own row cap (raised to
// `64` in Step 6 item 7; colibri's equivalent is 128) and is a batch size *inside* a
// layer, not a partition of the window. `W` is a real layer-major window rather than
// the chunk-major degenerate `W = C`: this gate ran at `W = 16` for a while, which is
// exactly chunk-major and therefore certified nothing about the layer-major order.
constexpr uint32_t kWindow = 96;
constexpr uint32_t kChunk = 16;
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

int main() {
    std::printf("================================================================================\n");
    std::printf("  Step 6 item 6 — the prefill sweep: bounded drain, layer order, restore\n");
    std::printf("================================================================================\n");
    aeon::core::select_compute_device(true);

    aeon::core::AeonRuntimeConfig runtime;
    runtime.context_size = kContext;
    runtime.warm_host_bytes = kWarmBytes;
    runtime.prefill_chunk = kChunk;
    runtime.prefill_sweep = true;
    // Force the gate open: this gate is about the swept supply at any window length.
    runtime.prefill_sweep_min_tokens = 1;

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
    std::printf("\n[B] Prefill begin (bounded drain, freeze Warm)\n");
    const std::set<uint32_t> warm_before_switch = warm_set(host);
    // No shadow can exist before the switch: a shadow residency is legal only while
    // Warm is frozen (`validate_invariants` refuses it otherwise), and decode never
    // freezes. So this is the observable form of "no second ownership crossed over"
    // — asserted *before* the switch, because the switch's own lookahead creates
    // shadows legitimately, by design (a Warm expert in the frontier is copied, not
    // promoted).
    const uint32_t shadows_before_switch = host.registry().shadow_resident_count();
    host.prefill_begin(kWindow);
    const bool streaming = host.registry().prefill_streaming();
    const uint32_t hot_after_drain = host.prefill_sweep().hot_after_drain();
    const std::vector<uint32_t> drained = host.registry().restore_set();
    const std::set<uint32_t> preserved = hot_set(host);
    // The set the drain actually saw at the switch point: what it preserved, plus
    // what it freed (which the window must hand back). Referenced from the drain
    // itself rather than a pre-switch capture, because the switch reaps in-flight
    // transfers before it drains, so a pre-switch capture would differ by whatever
    // the reap settled.
    std::set<uint32_t> switch_hot = preserved;
    switch_hot.insert(drained.begin(), drained.end());
    bool preserved_disjoint_from_drained = true;
    for (uint32_t gid : drained) {
        if (preserved.find(gid) != preserved.end()) {
            preserved_disjoint_from_drained = false;
        }
    }
    // Warm's baseline for the *sweep* is taken after the switch, not before it, for
    // the same settling reason.
    const std::set<uint32_t> warm_before = warm_set(host);
    const size_t switch_drift = set_difference_size(warm_before_switch, warm_before);

    // ---- run the window (the sweep is already active) ------------------------
    std::printf("\n[C] The swept window (layer-major, whole layer sets)\n");
    host.reset_generation_state();
    (void)graph.forward_window(ids.data(), 0, kWindow, kChunk, host.streams().compute);
    const std::vector<uint8_t> swept_logits =
        read_bytes(graph.logits(), static_cast<size_t>(vocab) * sizeof(uint16_t));

    const std::set<uint32_t> warm_after = warm_set(host);
    const size_t warm_drift = set_difference_size(warm_before, warm_after);
    const std::set<uint32_t> hot_after = hot_set(host);

    std::printf("\n--- results ---\n");

    assert_that("B: prefill entered streaming mode", streaming, "streaming");
    const uint32_t vram_capacity = host.registry().vram_capacity;
    const uint32_t expected_drain = vram_capacity >= 2u * per_layer
        ? 2u * per_layer
        : per_layer;
    assert_that("B: the drain was bounded to the pass's need",
                drained.size() == expected_drain && hot_after_drain > 0,
                std::to_string(drained.size()) + " drained, " +
                    std::to_string(hot_after_drain) + " preserved");
    assert_that("B: preserved and drained partition the switch-point pool",
                preserved_disjoint_from_drained &&
                    static_cast<size_t>(hot_after_drain) + drained.size() ==
                        vram_capacity,
                std::to_string(hot_after_drain) + " preserved + " +
                    std::to_string(drained.size()) + " drained = " +
                    std::to_string(vram_capacity));
    assert_that("B: no shadow existed to survive the switch", shadows_before_switch == 0,
                std::to_string(shadows_before_switch) + " decode-era shadows");

    const size_t logits_diff = differing_bytes(serial_logits, swept_logits);
    assert_that("D: swept logits == serial, bit-exact", logits_diff == 0,
                std::to_string(logits_diff) + " differing of " +
                    std::to_string(serial_logits.size()));

    assert_that("D: every layer's set streamed once (minus what was preserved)",
                host.prefill_sweep().experts_streamed() + hot_after_drain ==
                    static_cast<uint64_t>(layers) * per_layer,
                std::to_string(host.prefill_sweep().experts_streamed()) + " experts in " +
                    std::to_string(host.prefill_sweep().layer_loads()) + " loads");
    assert_that("D: the lookahead holds more than one layer", 
                host.prefill_sweep().frontier_depth() > 1,
                std::to_string(host.prefill_sweep().frontier_depth()) + " layers deep");

    assert_that("C: the Hot set is restored to its switch-point set",
                hot_after == switch_hot,
                std::to_string(hot_after.size()) + " Hot, expected " +
                    std::to_string(switch_hot.size()));
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

    // ---- the prompt-length gate (Step 3) -------------------------------------
    //
    // **Forced open here** (`prefill_sweep_min_tokens = 1`) so this gate is about the
    // sweep at any window length; the *default* gate and the routed path below it are
    // `test_v4_routed_prefill`'s subject. What is asserted here is that the switch is
    // a function of the window length and of the configured gate, and nothing else.
    std::printf("\n[E] The prompt-length gate (forced open)\n");
    const uint32_t gate = host.prefill_sweep_min_tokens();
    assert_that("E: the gate is forced open for this gate", gate == 1,
                "gate " + std::to_string(gate));
    assert_that("E: any non-empty window engages the sweep",
                host.prefill_sweep_engaged_for(kWindow),
                "window " + std::to_string(kWindow));

    std::printf("\n  window %u tokens (chunk %u) over %u layers, %u experts each\n",
                kWindow, kChunk, layers, per_layer);
    (void)hc_dim;

    std::printf("\n================================================================================\n");
    std::printf("  %u checks, %u failures\n", g_checks, g_failures);
    std::printf("================================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
