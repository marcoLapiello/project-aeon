// -----------------------------------------------------------------------------
// Supply-chain portability — the two sensitivity gates.
//
// The transfer path must be fast because it is **demand-driven**, not because it was
// tuned to one machine's disk, PCIe link, or GPU. A frozen constant that happens to
// suit this box is invisible in a benchmark run here and wrong on hardware that is
// faster or slower — a faster SSD, PCIe 5, a slower GPU all move the balance the
// constant silently encodes. The plan's answer is not a new constant; it is to make
// the corridor a resource the pipeline consumes and then prove the property. Two
// gates test it without fitting anything to this machine:
//
//   A. DEPTH INSENSITIVITY. Run the **same** swept window at several staging-arena
//      depths. Throughput must be flat above the minimum the design genuinely needs:
//      the depth is a budget the corridor consumes, not a knob that sets the speed.
//      The verdict is the *shape* of the curve, not a rate, so it holds on any
//      hardware: a design that only runs fast at one depth fails, and so does one
//      that cannot run at all below one layer's worth of slots.
//   B. BEHAVIOUR SYMMETRY. Across the same depths the **work** must be identical —
//      layers loaded, experts streamed, layers released, and the produced token.
//      Only the timing may differ. If the work changes with the depth, the logic is
//      reading a *resource* as a *policy*, which is exactly the hardware coupling
//      these gates exist to catch.
//
// Neither gate compares against a bandwidth, a latency, or any device figure, so a
// run on a faster SSD changes the absolute milliseconds in the table and nothing
// about the verdict.
//
// Usage: test_v4_staging_depth [depth ...]     (default: 64 128 192 256 384 512)
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_graph.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "architecture/deepseek_v4/core/v4_sampler.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using aeon::core::V4Graph;
using aeon::core::V4ModelHost;
using aeon::core::V4PrefillSweep;
using aeon::core::V4Sampler;
using aeon::core::V4SamplerConfig;
using Clock = std::chrono::steady_clock;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kContext = 2048;
constexpr uint32_t kChunk = 64;
constexpr uint32_t kWindow = 1024;
// The swept window every depth is measured over. Above the prompt-length gate
// (`3E/4 = 192`), so the sweep is what runs; short enough that the whole matrix
// fits one process and one model load.
constexpr uint32_t kLength = 256;

uint32_t g_checks = 0;
uint32_t g_failures = 0;

bool assert_that(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-54s %-34s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    ++g_checks;
    if (!ok) ++g_failures;
    return ok;
}

double since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// `MemAvailable` in bytes, or 0 when `/proc/meminfo` cannot be read. The depth
// sweep pins a large arena on **top of** the Warm pool, and Warm 30 plus a deep
// arena can push this box into swap — where a run does not fail, it stops
// progressing, which is the worst possible failure mode for a gate. So the arena
// change is refused when the projection says it would not fit, rather than being
// attempted and hanging.
uint64_t mem_available_bytes() {
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (file == nullptr) return 0;
    char line[256];
    unsigned long long kb = 0;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    }
    std::fclose(file);
    return static_cast<uint64_t>(kb) * 1024ULL;
}

struct Row {
    uint32_t depth{0};
    bool ran{false};
    std::string note;
    uint32_t staging_slots{0};
    uint32_t banks{0};
    double wall_s{0.0};
    uint32_t token{0};
    uint64_t layer_loads{0};
    uint64_t experts_streamed{0};
    uint64_t layers_released{0};
    double io_wait_s{0.0};
    double drain_s{0.0};
    // Corridor fill, from the sweep's per-layer samples: how many layers had a
    // staging block reading **and** a staging block copying at the same instant —
    // the signature of an overlapped pipeline rather than a parking lot.
    uint32_t layers_sampled{0};
    uint32_t layers_overlapped{0};
    uint32_t peak_reserved_ahead{0};
    // Staging slots freed by their own copy's completion event (P2.2) rather than as
    // a boundary block.
    uint64_t released_on_completion{0};
    // Copies the mid-body pump issued (P2.3) — the lookahead layer's uploads that
    // moved off the boundary into the previous layer's body.
    uint64_t copies_pumped{0};
    // The lookahead length the free blocks allowed at the sample — derived, not fixed (R5).
    uint32_t derived_depth{0};
    // The same, at the mid-window steady state (the max is seeded at the start, when
    // the whole pool is briefly free, and is not representative).
    uint32_t derived_depth_mid{0};
    // Components of that derivation, taken from one representative steady-state layer
    // (the middle of the window), so a reader can see **what limits the depth** and
    // **what it is relative to**.
    uint32_t vram_free_slots{0};
    uint32_t staging_free_slots{0};
    uint32_t vram_free_layers{0};
    uint32_t staging_free_layers{0};
    uint32_t ahead_count{0};
    uint32_t hot_preserved{0};
    uint32_t sample_layer{0};
    // The full per-layer sample set, kept for the optional corridor trace.
    std::vector<V4PrefillSweep::BlockOccupancy> samples;
};

} // namespace

int main(int argc, char** argv) {
    const std::string model_dir = argc > 1 && argv[1][0] != '-' ? argv[1] : kModelDir;
    std::vector<uint32_t> depths;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') continue;
        depths.push_back(static_cast<uint32_t>(std::atoi(argv[i])));
    }
    if (depths.empty()) {
        depths = {64, 128, 192, 256, 384, 512};
    }

    aeon::core::select_compute_device(true);

    aeon::core::AeonRuntimeConfig runtime;
    runtime.context_size = kContext;
    runtime.prefill_sweep = true;
    runtime.prefill_chunk = kChunk;
    runtime.prefill_window = kWindow;
    // Warm is a **local** read of the environment, because nothing in `src/` reads
    // `AEON_WARM_GIB`: it is a shell variable the scripts translate into a binary
    // *argument*. Prefixing this binary with it was therefore a silent no-op, and this
    // line used to hardcode 0 besides — so a documented `AEON_WARM_GIB=30` run was
    // really a Warm-0 run. Both shapes are required here (plan guardrail #7): Warm 0
    // exercises the corridor on its own, Warm 30 is the production shape where Warm
    // residency removes part of the cold stream.
    const char* warm_env = std::getenv("AEON_WARM_GIB");
    const uint64_t warm_gib = warm_env != nullptr
        ? std::strtoull(warm_env, nullptr, 10)
        : 0ULL;
    runtime.warm_host_bytes = warm_gib * 1024ULL * 1024ULL * 1024ULL;

    V4ModelHost host;
    host.initialize(model_dir, runtime, /*verbose=*/false);
    if (const char* cap = std::getenv("AEON_READ_AHEAD_MAX")) {
        host.set_sweep_read_ahead_max(static_cast<uint32_t>(std::atoi(cap)));
    }
    V4Graph graph(host);

    const uint32_t per_layer = host.registry().experts_per_layer;
    // The floor the current design imposes: a swept dispatch binds a whole layer's
    // distinct set at once, so the arena must hold it. A pipelined corridor would
    // not need this; that it does is the depth-dependence gate B is meant to expose.
    const uint32_t required = std::min<uint32_t>(6u * kChunk, per_layer);

    // The pinned bytes one staging slot costs, from the host's own accounting, so the
    // projection below cannot drift from what the arena really allocates.
    const uint32_t base_slots = host.staging_slot_count();
    const uint64_t slot_bytes = base_slots == 0
        ? 0
        : host.budget().transient_staging_bytes / base_slots;
    // Keep a reserve: reaching exactly MemAvailable still leaves the box thrashing on
    // a 8 GiB swap, so the guard wants real headroom rather than a zero margin.
    constexpr uint64_t kReserveBytes = 8ULL * 1024 * 1024 * 1024;

    std::vector<uint32_t> ids(kLength);
    for (uint32_t i = 0; i < kLength; ++i) {
        ids[i] = (i * 7u + 1u) % 1000u;
    }

    V4Sampler sampler(static_cast<uint32_t>(host.config().vocab_size));
    sampler.set_config(V4SamplerConfig{});  // defaults: temperature 0 => greedy

    std::printf(
        "[staging-depth] layers=%u E=%u window=%u chunk=%u floor=%u slots "
        "(initial: %u slots, %u banks), warm=%llu GiB\n",
        host.num_layers(), per_layer, kLength, kChunk, required,
        host.staging_slot_count(), host.sweep_staging_banks(),
        static_cast<unsigned long long>(warm_gib));

    std::vector<Row> rows;
    uint32_t current_slots = base_slots;
    for (const uint32_t depth : depths) {
        Row row;
        row.depth = depth;

        if (depth < required) {
            row.note = "below the swept dispatch's floor";
            rows.push_back(row);
            continue;
        }
        // Memory guard (derived, not a hardware constant): the arena is pinned on top
        // of the Warm pool and the loaded model, so a depth that does not fit must be
        // refused, not attempted. Attempting it enters swap, where the run does not
        // fail — it stops progressing.
        const uint64_t delta_bytes = depth > current_slots
            ? static_cast<uint64_t>(depth - current_slots) * slot_bytes
            : 0;
        const uint64_t available = mem_available_bytes();
        if (delta_bytes + kReserveBytes > available) {
            char note[160];
            std::snprintf(note, sizeof(note),
                          "refused: needs +%.1f GiB, only %.1f GiB available "
                          "(reserve %.1f GiB)",
                          static_cast<double>(delta_bytes) / (1024.0 * 1024.0 * 1024.0),
                          static_cast<double>(available) / (1024.0 * 1024.0 * 1024.0),
                          static_cast<double>(kReserveBytes) / (1024.0 * 1024.0 * 1024.0));
            row.note = note;
            rows.push_back(row);
            continue;
        }
        if (!host.resize_staging_slots(depth)) {
            row.note = "refused (host not in a resizable state)";
            rows.push_back(row);
            continue;
        }
        row.staging_slots = host.staging_slot_count();
        row.banks = host.sweep_staging_banks();
        if (row.staging_slots != depth) {
            row.note = "arena did not take the requested depth";
            rows.push_back(row);
            continue;
        }
        current_slots = row.staging_slots;

        // A fresh session and a fresh phase: the KV/compressor/indexer state is
        // zeroed so every depth runs the identical window, and the counters start at
        // zero so each row's split is its own.
        host.reset_generation_state();
        host.set_supply_phase(true);
        host.reset_supply_transfer_counters();
        const uint64_t loads_before = host.prefill_sweep().layer_loads();
        const uint64_t experts_before = host.prefill_sweep().experts_streamed();
        const uint64_t released_before = host.prefill_sweep().layers_released();

        const auto started = Clock::now();
        const half* logits = graph.forward_window(
            ids.data(), 0, kLength, std::min(kChunk, kLength), host.streams().compute);
        CHECK_HIP(hipStreamSynchronize(host.streams().compute));
        row.wall_s = since(started);

        row.token = sampler.select(logits, host.streams().compute);
        row.layer_loads = host.prefill_sweep().layer_loads() - loads_before;
        row.experts_streamed = host.prefill_sweep().experts_streamed() - experts_before;
        row.layers_released = host.prefill_sweep().layers_released() - released_before;
        row.io_wait_s = static_cast<double>(host.supply_io_wait_ns()) / 1e9;
        row.drain_s = static_cast<double>(host.supply_h2d_drain_ns()) / 1e9;
        row.released_on_completion = host.supply_staging_released_on_completion();
        row.copies_pumped = host.supply_copies_pumped();

        // Corridor fill: a sample is "overlapped" when a read is landing in one
        // staging block while a copy drains another — the pipeline working. One block
        // pinned at `E` with the other at zero is the parking lot.
        const uint32_t mid_layer = host.num_layers() / 2;
        row.samples = host.sweep_occupancy();
        for (const auto& sample : host.sweep_occupancy()) {
            ++row.layers_sampled;
            if (sample.staging_reading > 0 && sample.staging_copying > 0) {
                ++row.layers_overlapped;
            }
            row.peak_reserved_ahead = std::max(row.peak_reserved_ahead,
                                               sample.vram_reserved_ahead);
            row.derived_depth = std::max(row.derived_depth, sample.derived_capacity);
            // Keep the middle of the window as the representative steady-state sample.
            if (sample.layer == mid_layer) {
                row.sample_layer = sample.layer;
                row.vram_free_slots = sample.vram_free_slots;
                row.staging_free_slots = sample.staging_free_slots;
                row.vram_free_layers = sample.vram_free_layers;
                row.staging_free_layers = sample.staging_free_layers;
                row.ahead_count = sample.ahead_count;
                row.hot_preserved = sample.hot_preserved;
                row.derived_depth_mid = sample.derived_capacity;
            }
        }

        row.ran = true;
        row.note = host.prefill_sweep_engaged() ? "swept" : "NOT swept";
        rows.push_back(row);
    }

    std::printf("%s\n", std::string(104, '=').c_str());
    std::printf(
        "  DEPTH SWEEP — one swept window, N=%u, %u layers x %u experts, one process\n",
        kLength, host.num_layers(), per_layer);
    std::printf("%s\n", std::string(104, '=').c_str());
    std::printf("%-6s %-6s %-5s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %s\n",
                "depth", "slots", "banks", "wall_s", "io_wait", "drain_s",
                "samples", "overlap", "pumped", "derived", "note");
    for (const Row& r : rows) {
        if (!r.ran) {
            std::printf("%-6u %-6s %-5s %-8s %-9s %-8s %-8s %-8s %-8s %-8s %s\n",
                        r.depth, "-", "-", "-", "-", "-", "-", "-", "-", "-",
                        r.note.c_str());
            continue;
        }
        std::printf("%-6u %-6u %-5u %-8.3f %-9.3f %-8.3f %-8u %-8s %-8llu %-8u %s\n",
                    r.depth, r.staging_slots, r.banks, r.wall_s, r.io_wait_s, r.drain_s,
                    r.layers_sampled,
                    (std::to_string(r.layers_overlapped) + "/" +
                     std::to_string(r.layers_sampled)).c_str(),
                    static_cast<unsigned long long>(r.copies_pumped),
                    r.derived_depth, r.note.c_str());
    }
    std::printf("\n");

    std::vector<const Row*> ran;
    for (const Row& r : rows) {
        if (r.ran) ran.push_back(&r);
    }

    // ---- Gate A — throughput is insensitive to the depth ---------------------
    std::printf("%s\n", std::string(104, '=').c_str());
    std::printf("  A. DEPTH INSENSITIVITY — throughput must be flat above the floor\n");
    std::printf("%s\n", std::string(104, '=').c_str());
    if (ran.size() < 2) {
        assert_that("A: at least two depths ran", false,
                    std::to_string(ran.size()) + " ran");
    } else {
        double fastest = ran[0]->wall_s;
        double slowest = ran[0]->wall_s;
        for (const Row* r : ran) {
            fastest = std::min(fastest, r->wall_s);
            slowest = std::max(slowest, r->wall_s);
        }
        const double spread = slowest / fastest;
        char detail[128];
        std::snprintf(detail, sizeof(detail),
                      "fastest %.3f s, slowest %.3f s, spread %.3fx",
                      fastest, slowest, spread);
        // 5% is deliberately loose: it must not fail on run-to-run noise, and the
        // current design's step at `2E` is ~10%, so the gate still separates them.
        assert_that("A: wall time flat within 5% across depths", spread <= 1.05, detail);
    }

    // ---- Gate B — the work is identical; only the timing differs -------------
    std::printf("\n%s\n", std::string(104, '=').c_str());
    std::printf("  B. BEHAVIOUR SYMMETRY — same work, only the timing may differ\n");
    std::printf("%s\n", std::string(104, '=').c_str());
    if (ran.size() < 2) {
        assert_that("B: at least two depths ran", false,
                    std::to_string(ran.size()) + " ran");
    } else {
        const Row& ref = *ran[0];
        bool same_token = true;
        bool same_layers = true;
        bool same_experts = true;
        bool same_released = true;
        for (const Row* r : ran) {
            same_token = same_token && r->token == ref.token;
            same_layers = same_layers && r->layer_loads == ref.layer_loads;
            same_experts = same_experts && r->experts_streamed == ref.experts_streamed;
            same_released = same_released && r->layers_released == ref.layers_released;
        }
        assert_that("B: the produced token is identical at every depth", same_token,
                    "token " + std::to_string(ref.token));
        assert_that("B: layers loaded is identical at every depth", same_layers,
                    std::to_string(ref.layer_loads) + " layers");
        assert_that("B: experts streamed is identical at every depth", same_experts,
                    std::to_string(ref.experts_streamed) + " experts");
        assert_that("B: layers released is identical at every depth", same_released,
                    std::to_string(ref.layers_released) + " layers");
    }

    // ---- Gate C — the corridor is filled, not just sized ---------------------
    std::printf("\n%s\n", std::string(104, '=').c_str());
    std::printf("  C. CORRIDOR FILL — a read landing while a copy drains, concurrently\n");
    std::printf("%s\n", std::string(104, '=').c_str());
    if (ran.empty()) {
        assert_that("C: a swept window ran", false, "none ran");
    } else {
        for (const Row* r : ran) {
            char detail[128];
            std::snprintf(detail, sizeof(detail),
                          "%u/%u layers overlapped, peak reserved-ahead %u experts",
                          r->layers_overlapped, r->layers_sampled, r->peak_reserved_ahead);
            // The **default** shape (2E) must overlap; a deliberately reduced depth
            // may not, and that is the point of the gate rather than a failure.
            const bool is_default = r->banks >= 2;
            assert_that(
                ("C: depth " + std::to_string(r->depth) +
                 (is_default ? " (default 2E) overlaps" : " (reduced depth)")).c_str(),
                !is_default || r->layers_overlapped > 0, detail);
        }
    }

    // ---- Per-layer trace of the corridor, for the depth that matters --------
    // The summary hides the question the user asks: are **both** staging halves in
    // use, or only one? A per-half magnitude trace answers it directly.
    if (const char* trace = std::getenv("AEON_TRACE_LAYERS")) {
        const uint32_t want = static_cast<uint32_t>(std::atoi(trace));
        std::printf("%s\n", std::string(104, '=').c_str());
        std::printf("  PER-LAYER CORRIDOR TRACE (depth %u)\n", want);
        std::printf("%s\n", std::string(104, '=').c_str());
        std::printf("%-6s %-6s %-9s %-9s %-9s %-10s\n",
                    "layer", "ahead", "reading", "copying", "stg_free", "vram_free");
        for (const Row& r : rows) {
            if (!r.ran || r.depth != want) continue;
            for (const auto& s : r.samples) {
                std::printf("%-6u %-6u %-9u %-9u %-9u %-10u\n",
                            s.layer, s.ahead_count, s.staging_reading,
                            s.staging_copying, s.staging_free, s.vram_free_slots);
            }
        }
        std::printf("\n");
    }

    // ---- The derived depth, broken into its components -----------------------
    // "Lookahead" is ambiguous without saying **relative to what**. The columns
    // answer it: mid-window, while one layer computes, how many whole layer-blocks of
    // VRAM and of staging are free, how many layers are queued ahead, and how many
    // Hot residents are preserved (and so shrink the pool the depth is derived from).
    std::printf("%s\n", std::string(104, '=').c_str());
    std::printf("  DERIVED DEPTH — mid-window, one layer computing\n");
    std::printf("%s\n", std::string(104, '=').c_str());
    std::printf("%-6s %-9s %-9s %-11s %-11s %-11s %-11s %-9s %s\n",
                "depth", "derived", "mid", "vram_free", "stg_free", "vram_lyrs",
                "stg_lyrs", "queued", "preserved");
    for (const Row& r : rows) {
        if (!r.ran) continue;
        std::printf("%-6u %-9u %-9u %-11s %-11s %-11u %-11u %-9u %u\n",
                    r.depth, r.derived_depth, r.derived_depth_mid,
                    (std::to_string(r.vram_free_slots) + " sl").c_str(),
                    (std::to_string(r.staging_free_slots) + " sl").c_str(),
                    r.vram_free_layers, r.staging_free_layers,
                    r.ahead_count, r.hot_preserved);
    }
    std::printf("\n");

    // ---- Reported, not asserted: the floor the design imposes ----------------
    // A pipelined corridor would accept a depth well below one layer's set. That it
    // is refused today is the finding, and printing it here keeps it visible in the
    // gate's own output rather than only in a document.
    size_t unsupported = 0;
    for (const Row& r : rows) {
        if (!r.ran) ++unsupported;
    }
    std::printf("\n%s\n", std::string(104, '=').c_str());
    std::printf("  floor: %u slots required (%zu of %zu depths unsupported); "
                "depth is currently a floor, not a budget\n",
                required, unsupported, rows.size());
    std::printf("%s\n\n", std::string(104, '=').c_str());

    std::printf("%s\n", std::string(64, '=').c_str());
    std::printf("  %u checks, %u failures\n", g_checks, g_failures);
    std::printf("%s\n", std::string(64, '=').c_str());
    return g_failures == 0 ? 0 : 1;
}
