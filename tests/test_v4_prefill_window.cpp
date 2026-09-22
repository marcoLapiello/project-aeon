// -----------------------------------------------------------------------------
// Step 6 gate — the layer-major prefill window is `window == serial`.
//
// The plan's Step 6 opens with a decision (D-a): prefill is **layer-major within
// a bounded window**, because that fetches each layer's routed-expert set once
// per window instead of once per body chunk. This gate certifies the *equality*
// half of that decision before any of its speed is claimed, for the reason the
// plan gives: correctness and speed are two gates, and correctness comes first.
//
// What is asserted:
//
//   A. WINDOW == SERIAL, BIT-EXACT. `forward_window(ids, 0, N, N)` — one
//      layer-major pass over `N` tokens — produces byte-identical final logits
//      and a byte-identical final residual to driving `forward_token` once per
//      token, in position order. An ordering that changed a number would not be
//      an ordering.
//
//   B. SCHEDULE-INDEPENDENT. The same window run with a smaller body chunk
//      (`C < N`, several body invocations per layer) is byte-identical to both
//      of the above. The chunk is a batch size *inside* a layer, not a partition
//      of the prompt, so it must not be observable in the result.
//
//   C. NON-VACUOUS AND CLEAN. The comparison has content (the logits are not a
//      constant), every lease is handed back, and the staging arena is empty — so
//      a pass cannot mean "nothing ran".
//
// ## Why the equality can be exact
//
// The reference is the *certified* serial path (`forward_token`, the composition
// plan's acceptance-criterion path), not a second copy of the new code, which is
// what keeps this from proving self-consistency. The batched body's equality to
// the decode body is Tier-3 item 19's result; this gate sits above it and adds
// the two things item 19 deliberately did not cover: the **host** (43 layers, the
// real expert supply) and the **layer stack**.
//
// ## The read-ordering trap this gate walked into first
//
// The forward paths enqueue on the **compute** stream, which is non-default and
// non-blocking. A plain `hipMemcpy` is not guaranteed to order against it, so a
// read taken right after a forward pass can return the *previous* run's buffer.
// That is exactly what happened: an earlier version of this gate read without
// synchronizing and reported ~76% of the logits "differing" — a fabricated
// divergence that cost a long investigation and pointed at the wrong component
// (staging, then the residual chain). `read_bytes` therefore synchronizes the
// device first, and that is load-bearing, not hygiene. The body's own
// `hipStreamSynchronize` sits *before* its final stages, so it does not cover
// them.
//
// Deliberately NOT covered here:
//   * throughput. Speed is Step 6's separate outcome, measured with the ledger.
//   * the chunk-wide expert dispatch and dedup (build items 4–5). This gate runs
//     the existing per-token dispatch through the windowed driver, so it
//     certifies the driver, not the dispatch.
//   * the double-buffered sweep (item 6): the window serializes sweep against
//     compute here. That is slower, and identical.
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
#include <string>
#include <vector>

namespace {

using aeon::core::V4Graph;
using aeon::core::V4ModelHost;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kContext = 256;
constexpr uint32_t kWindow = 16;
constexpr uint32_t kSmallChunk = 5;

uint32_t g_checks = 0;
uint32_t g_failures = 0;

bool assert_that(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-52s %-34s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    ++g_checks;
    if (!ok) ++g_failures;
    return ok;
}

// Raw bytes, so the comparison is on fp16 bit patterns and never goes through a
// float conversion that could hide a one-ulp difference.
//
// The device is synchronized first: the forward paths enqueue on a non-default
// compute stream and a plain `hipMemcpy` does not order against it, so without
// this the read can return the previous run's buffer and invent a divergence.
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

bool all_zero(const std::vector<uint8_t>& bytes) {
    for (const uint8_t byte : bytes) {
        if (byte != 0) return false;
    }
    return true;
}

struct Snapshot {
    std::vector<uint8_t> logits;
    std::vector<uint8_t> residual;
};

struct HostShape {
    uint32_t vocab;
    uint32_t hc_dim;
};

HostShape shape_of(V4ModelHost& host) {
    return HostShape{
        static_cast<uint32_t>(host.config().vocab_size),
        static_cast<uint32_t>(host.config().hc_mult) *
            static_cast<uint32_t>(host.config().hidden_size)
    };
}

Snapshot capture_window(V4ModelHost& host, V4Graph& graph, const std::vector<uint32_t>& ids,
                        uint32_t chunk) {
    const HostShape shape = shape_of(host);
    graph.forward_window(ids.data(), 0, static_cast<uint32_t>(ids.size()), chunk,
                         host.streams().compute);
    return Snapshot{
        read_bytes(graph.logits(), static_cast<size_t>(shape.vocab) * sizeof(uint16_t)),
        read_bytes(graph.residual(), static_cast<size_t>(shape.hc_dim) * sizeof(float))
    };
}

Snapshot capture_serial(V4ModelHost& host, V4Graph& graph, const std::vector<uint32_t>& ids) {
    const HostShape shape = shape_of(host);
    for (uint32_t position = 0; position < ids.size(); ++position) {
        (void)graph.forward_token(ids[position], position, host.streams().compute);
    }
    return Snapshot{
        read_bytes(graph.logits(), static_cast<size_t>(shape.vocab) * sizeof(uint16_t)),
        read_bytes(graph.residual(), static_cast<size_t>(shape.hc_dim) * sizeof(float))
    };
}

} // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  Step 6 — layer-major prefill: window == serial, bit-exact\n");
    std::printf("================================================================================\n");
    aeon::core::select_compute_device(true);

    aeon::core::AeonRuntimeConfig runtime;
    runtime.context_size = kContext;
    // The layer-major window dispatches a chunk's `6C` routed requests as one
    // deduplicated set (Step 6 item 4), each distinct expert held in a staging slot
    // while in transit. Configure a prefill chunk at least as large as the window
    // so the host sizes the arena for it (Step 6 D4); a smaller configuration would
    // be refused by `forward_window` rather than silently colliding two transfers.
    runtime.prefill_chunk = kWindow;
    // This gate certifies the **driver** — that a layer-major pass equals serial and
    // that the chunk size is not observable — not the swept residency policy. The
    // sweep has its own gate (`test_v4_prefill_sweep`), so it is off here: with it on
    // the window would drain Hot and stream all 43 layers, which is the subject of
    // that other gate and would triple this one's runtime without adding a claim.
    runtime.prefill_sweep = false;

    V4ModelHost host;
    host.initialize(kModelDir, runtime, /*verbose=*/true);
    V4Graph graph(host);

    std::printf("\n  staging arena: %u slots (chunk %u -> 6C = %u)\n",
                host.staging_slot_count(), kWindow, 6u * kWindow);

    const HostShape shape = shape_of(host);

    // A window of distinct, in-vocabulary ids. Distinct so a mis-indexed gather
    // cannot accidentally match, and fixed so the run is reproducible.
    std::vector<uint32_t> ids(kWindow);
    for (uint32_t i = 0; i < kWindow; ++i) {
        ids[i] = 1000 + i * 37;
    }

    std::printf("\n[A] The serial reference (the certified path, one token at a time)\n");
    host.reset_generation_state();
    const Snapshot serial = capture_serial(host, graph, ids);
    const uint64_t serial_draws = host.expert_batch_draws();
    const uint64_t serial_distinct = host.expert_batch_distinct();

    std::printf("\n[B] The layer-major window (one body invocation per layer)\n");
    host.reset_generation_state();
    const Snapshot window = capture_window(host, graph, ids, kWindow);
    const uint32_t window_batch_tokens = host.last_expert_dispatch_tokens();
    const uint64_t window_draws = host.expert_batch_draws() - serial_draws;
    const uint64_t window_distinct = host.expert_batch_distinct() - serial_distinct;

    std::printf("\n[C] The same window at a smaller body chunk (several per layer)\n");
    host.reset_generation_state();
    const Snapshot chunked = capture_window(host, graph, ids, kSmallChunk);

    std::printf("\n--- results ---\n");

    assert_that("A: the run is non-vacuous",
                !all_zero(serial.logits) &&
                    serial.logits.size() == static_cast<size_t>(shape.vocab) * sizeof(uint16_t),
                std::to_string(serial.logits.size()) + " logit bytes");

    const size_t logits_diff = differing_bytes(serial.logits, window.logits);
    assert_that("A: window logits == serial, bit-exact", logits_diff == 0,
                std::to_string(logits_diff) + " differing of " +
                    std::to_string(serial.logits.size()));

    const size_t residual_diff = differing_bytes(serial.residual, window.residual);
    assert_that("A: window residual == serial, bit-exact", residual_diff == 0,
                std::to_string(residual_diff) + " differing of " +
                    std::to_string(serial.residual.size()));

    const size_t chunk_logits_diff = differing_bytes(serial.logits, chunked.logits);
    assert_that("B: chunked logits == serial, bit-exact", chunk_logits_diff == 0,
                std::to_string(chunk_logits_diff) + " differing");

    const size_t chunk_residual_diff = differing_bytes(serial.residual, chunked.residual);
    assert_that("B: chunked residual == serial, bit-exact", chunk_residual_diff == 0,
                std::to_string(chunk_residual_diff) + " differing");

    assert_that("B: the chunk size is not observable",
                differing_bytes(window.logits, chunked.logits) == 0,
                "window (" + std::to_string(kWindow) + ") vs chunk (" +
                    std::to_string(kSmallChunk) + ")");

    assert_that("C: every lease handed back", host.outstanding_expert_leases() == 0,
                std::to_string(host.outstanding_expert_leases()) + " outstanding");

    assert_that("C: the staging arena drained", host.staging_in_use_slots() == 0,
                std::to_string(host.staging_in_use_slots()) + " slots in use");

    // D — the dispatch really is the layer-wide one, and it really deduplicates.
    // Without these a byte-exact pass could be the per-token dispatch under the
    // batch's name, and `window == serial` would prove nothing about item 4.
    assert_that("D: the window issued one layer-wide batch",
                window_batch_tokens == kWindow,
                std::to_string(window_batch_tokens) + " tokens in the last dispatch");
    assert_that("D: dedup collapsed the chunk's 6C draws",
                window_distinct > 0 && window_distinct < window_draws,
                std::to_string(window_distinct) + " distinct of " +
                    std::to_string(window_draws) + " draws");

    std::printf("\n  window %u tokens, residual compared over %u fp32 values (%u per token)\n",
                kWindow, shape.hc_dim, shape.hc_dim);

    std::printf("\n================================================================================\n");
    std::printf("  %u checks, %u failures\n", g_checks, g_failures);
    std::printf("================================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
