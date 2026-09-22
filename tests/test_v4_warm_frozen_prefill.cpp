// -----------------------------------------------------------------------------
// Step 6 outcome 3 — the Warm tier survives a prefill intact (D-b, policy A).
//
// The plan's D-b freezes Warm during prefill: the sweep must not **promote** from
// Warm (a promotion is a *move*, so each Warm-resident expert it touches would
// leave the tier) and must not **demote** into it. The decision's whole point is
// that decode's "natural selection" in Warm is still there when the prompt is
// done, so the first generated tokens start warm instead of cold.
//
// What is asserted, through the real host (43 layers, the real expert supply, a
// real Warm tier on the host):
//
//   A. FROZEN PREFILL PRESERVES WARM. The set of Warm-resident experts is
//      **identical** before and after a layer-major prefill window.
//   B. IT REALLY READ WARM. `shadow_copies > 0` and the prefill served
//      `logical_bytes_from_warm > 0` — the copy is non-destructive, so the sweep
//      still gets Warm's bandwidth instead of paying an NVMe read to bypass it.
//   C. LEAVING THE PHASE RELEASES THE COPIES. Every shadow residency is returned
//      when decode resumes (`shadow_resident_count() == 0`), so the frozen
//      prefill does not strand VRAM, and Warm is *still* unchanged.
//   D. THE TEST IS NOT VACUOUS. A second prefill window run with the freeze
//      **off** (no phase set) does change Warm — so assertion A is measuring the
//      policy, not a workload that could never touch Warm anyway.
//
// The reference for A is the registry's own residency map, which is the thing the
// policy is defined over; the tier answer itself (which tier served a request) is
// not re-derived here — that is Step 3's tier-invariance gate.
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
constexpr const char* kTelemetryPath = "/tmp/aeon-warm-frozen.jsonl";
constexpr uint32_t kContext = 256;
constexpr uint32_t kWindow = 16;
// Enough Warm to be a real tier (≈300 slots) without a slow preload.
constexpr size_t kWarmBytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;

uint32_t g_checks = 0;
uint32_t g_failures = 0;

bool assert_that(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-54s %-34s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    ++g_checks;
    if (!ok) ++g_failures;
    return ok;
}

// The experts the registry currently holds in Warm. The policy under test is a
// statement about this set, so this is the observable the gate reads.
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

} // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  Step 6 outcome 3 — Warm is preserved across a prefill (D-b)\n");
    std::printf("================================================================================\n");
    aeon::core::select_compute_device(true);

    aeon::core::AeonRuntimeConfig runtime;
    runtime.context_size = kContext;
    runtime.warm_host_bytes = kWarmBytes;
    runtime.prefill_chunk = kWindow;
    runtime.supply_telemetry_path = kTelemetryPath;
    runtime.run_id = "warm-frozen";

    V4ModelHost host;
    host.initialize(kModelDir, runtime, /*verbose=*/true);
    V4Graph graph(host);

    std::vector<uint32_t> ids(kWindow);
    for (uint32_t i = 0; i < kWindow; ++i) {
        ids[i] = 1000 + i * 37;
    }

    const uint32_t warm_slots = static_cast<uint32_t>(warm_set(host).size());
    std::printf("\n  Warm tier: %u resident experts, staging %u slots\n",
                warm_slots, host.staging_slot_count());

    if (warm_slots == 0) {
        std::printf("\n  no Warm tier to test — aborting\n");
        return 1;
    }

    // ---- A/B: a frozen prefill window ----------------------------------------
    std::printf("\n[A] Frozen prefill window (phase = prefill, freeze on)\n");
    host.set_supply_phase(/*prefill=*/true);
    const bool frozen_entered = host.warm_frozen();
    const std::set<uint32_t> warm_before = warm_set(host);

    (void)graph.forward_window(ids.data(), 0, kWindow, kWindow, host.streams().compute);

    const std::set<uint32_t> warm_after_prefill = warm_set(host);
    const size_t prefill_drift = set_difference_size(warm_before, warm_after_prefill);

    // ---- C: leaving the phase releases the copies ----------------------------
    std::printf("\n[B] Decode resumes (phase = decode)\n");
    host.set_supply_phase(/*prefill=*/false);
    const uint32_t shadows_after_unfreeze = host.shadow_resident_count();
    const std::set<uint32_t> warm_after_unfreeze = warm_set(host);
    const size_t unfreeze_drift = set_difference_size(warm_before, warm_after_unfreeze);
    const bool frozen_exited = !host.warm_frozen();

    // ---- D: the control — the same window with the freeze off ---------------
    std::printf("\n[C] Control window with the freeze off (no phase set)\n");
    // No `set_supply_phase` call, so the registry is not in frozen mode: this is
    // the pre-D-b behaviour, where prefill promotes Warm experts into VRAM and
    // empties the tier.
    (void)graph.forward_window(ids.data(), 0, kWindow, kWindow, host.streams().compute);
    const std::set<uint32_t> warm_after_control = warm_set(host);
    const size_t control_drift = set_difference_size(warm_before, warm_after_control);

    std::printf("\n--- results ---\n");

    assert_that("A: prefill entered frozen mode", frozen_entered, "frozen");
    assert_that("A: Warm is unchanged by the frozen prefill", prefill_drift == 0,
                std::to_string(prefill_drift) + " experts differ of " +
                    std::to_string(warm_before.size()));
    assert_that("B: the frozen prefill read Warm (non-vacuous)",
                host.shadow_copies() > 0,
                std::to_string(host.shadow_copies()) + " shadow copies");
    assert_that("B: prefill served logical bytes from Warm",
                host.supply_logical_bytes_from_warm(/*prefill=*/true) > 0,
                std::to_string(host.supply_logical_bytes_from_warm(true)) + " B");
    assert_that("C: decode left frozen mode", frozen_exited, "unfrozen");
    assert_that("C: every shadow residency was released",
                shadows_after_unfreeze == 0,
                std::to_string(shadows_after_unfreeze) + " shadows held");
    assert_that("C: Warm is still unchanged after unfreezing", unfreeze_drift == 0,
                std::to_string(unfreeze_drift) + " experts differ");
    assert_that("D: the unfrozen control does change Warm", control_drift > 0,
                std::to_string(control_drift) + " experts differ");
    assert_that("D: no lease leaked", host.outstanding_expert_leases() == 0,
                std::to_string(host.outstanding_expert_leases()) + " outstanding");
    assert_that("D: the staging arena drained", host.staging_in_use_slots() == 0,
                std::to_string(host.staging_in_use_slots()) + " slots in use");
    assert_that("D: registry invariants hold", host.registry().invariants_hold(),
                "invariants_hold()");

    std::printf("\n  Warm before %u, after frozen prefill %u, after control %u\n",
                static_cast<uint32_t>(warm_before.size()),
                static_cast<uint32_t>(warm_after_prefill.size()),
                static_cast<uint32_t>(warm_after_control.size()));

    std::printf("\n================================================================================\n");
    std::printf("  %u checks, %u failures\n", g_checks, g_failures);
    std::printf("================================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
