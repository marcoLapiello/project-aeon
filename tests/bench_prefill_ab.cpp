// -----------------------------------------------------------------------------
// Prefill A/B — serial (per-token) versus the layer-major swept window.
//
// ## Why this exists
//
// Step 6's outcome 5 is a *speed* claim, and a speed claim without its baseline is
// not a measurement. An earlier version of this work reported "2.1 tok/s" for the
// batched window with nothing beside it — which, against the ~3 tok/s serial
// prefill we already had, reads as a regression. It was worse than that: the run
// was at a 103-token prompt, and the two arms do not scale the same way.
//
// **The shape that matters.** The sweep reads each layer's expert set **once per
// window**, so its byte cost is very nearly **constant in the prompt length** (one
// model read, ~148 GiB). The serial path re-fetches as it walks: token by token it
// traverses all 43 layers, and by the time it returns to a layer the Hot pool has
// evicted that layer's set, so every token re-reads. Serial cost therefore grows
// **linearly in N**. There is a crossover, and a single point below it says nothing
// about the strategy:
//
//     serial_bytes(N)  ~  N x 6 draws x 43 layers      (grows with N)
//     sweep_bytes      ~  43 x experts_per_layer       (constant in N)
//
// So this program measures **one arm at a time**, each in its own process with a
// pristine host, and a driver script tabulates the matrix. That isolation matters:
// both pools are stateful, so running serial first would leave Warm populated (or
// Hot drained) for the swept arm, and the comparison would be between a warm run and
// a cold one.
//
// ## What it does not do
//
// It does not assert. It prints, because the numbers are the deliverable and a
// threshold would turn a measurement into a test that passes. Correctness under the
// sweep is `test_v4_prefill_sweep`'s subject (byte-identical to serial, ledger
// M40); this file never re-checks it.
//
// ## The reference, for scale
//
// Colibri (`aeon-references/colibri/c/deepseek_v4.c`) runs the same strategy with
// `V4_PREFILL_CHUNK = 128` inside `V4_PREFILL_SEGMENT = 4096`, and records
// **3324 tokens in 103.9 s = 32 tok/s** on a 2x NVMe mirror, at ~0.35 s/layer for
// the sweep (~0.7 s/layer on one drive). Those are the numbers to compare against.
//
// Usage: bench_prefill_ab <arm:serial|swept> <n_tokens> [model_dir] [warm_gib]
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_graph.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using aeon::core::V4Graph;
using aeon::core::V4LayerBodyBatchScratch;
using aeon::core::V4ModelHost;
using Clock = std::chrono::steady_clock;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kContext = 2048;
// The body's own row cap. Colibri's `V4_PREFILL_CHUNK` is 128; ours is 16, which is
// a *launch-count* difference, not a byte one — the sweep loads whole layers either
// way. It is called out in the report because the chunk is the knob Step 7 sweeps.
constexpr uint32_t kChunk = V4LayerBodyBatchScratch::kMaxTokens;

double since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// A deterministic, in-vocabulary prompt. Pseudo-random so the router sees a mix of
// routings rather than one token's repeated, and drawn from the middle of the
// vocabulary so it never lands on a special id.
std::vector<uint32_t> make_prompt(uint32_t count, uint32_t vocab) {
    std::vector<uint32_t> ids(count);
    uint32_t state = 12345u;
    for (uint32_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        ids[i] = 100u + (state % (vocab - 200u));
    }
    return ids;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: %s <serial|swept> <n_tokens> [model_dir] [warm_gib]\n", argv[0]);
        return 2;
    }
    const std::string arm = argv[1];
    const uint32_t n = static_cast<uint32_t>(std::atoi(argv[2]));
    const std::string model_dir = argc > 3 ? argv[3] : kModelDir;
    const uint64_t warm_gib = argc > 4 ? static_cast<uint64_t>(std::atoi(argv[4])) : 0;

    aeon::core::select_compute_device(true);

    aeon::core::AeonRuntimeConfig runtime;
    runtime.context_size = kContext;
    runtime.prefill_sweep = true;
    runtime.prefill_chunk = kChunk;
    runtime.warm_host_bytes = warm_gib * 1024ULL * 1024ULL * 1024ULL;

    V4ModelHost host;
    host.initialize(model_dir, runtime, /*verbose=*/false);
    V4Graph graph(host);

    const uint32_t vocab = static_cast<uint32_t>(host.config().vocab_size);
    const uint32_t layers = host.num_layers();
    const uint32_t per_layer = host.registry().experts_per_layer;
    const double model_bytes = static_cast<double>(layers) * per_layer * 14'155'776.0;

    const std::vector<uint32_t> ids = make_prompt(n, vocab);

    const uint64_t b0n = host.supply_bytes_from_nvme();
    const uint64_t b0h = host.supply_bytes_from_host();
    const uint64_t b0d = host.supply_h2d_bytes();
    const uint64_t b0r = host.supply_requests();
    const uint64_t l0 = host.sweep_load_ns();

    host.reset_generation_state();
    const auto start = Clock::now();
    if (arm == "serial") {
        for (uint32_t position = 0; position < n; ++position) {
            (void)graph.forward_token(ids[position], position, host.streams().compute);
        }
    } else if (arm == "swept") {
        (void)graph.forward_window(ids.data(), 0, n, kChunk, host.streams().compute);
    } else {
        std::printf("unknown arm: %s\n", arm.c_str());
        return 2;
    }
    CHECK_HIP(hipStreamSynchronize(host.streams().compute));
    const double seconds = since(start);

    const uint64_t nvme = host.supply_bytes_from_nvme() - b0n;
    const uint64_t warmb = host.supply_bytes_from_host() - b0h;
    const uint64_t h2d = host.supply_h2d_bytes() - b0d;
    const uint64_t requests = host.supply_requests() - b0r;
    const double load_s = static_cast<double>(host.sweep_load_ns() - l0) / 1e9;
    const double io_s = static_cast<double>(host.sweep_io_ns()) / 1e9;

    // One machine-readable line so the driver script can tabulate without parsing
    // prose, plus the parts a reader needs to attribute the seconds.
    std::printf("RESULT arm=%s n=%u seconds=%.3f tok_per_s=%.3f nvme_gib=%.3f warm_gib=%.3f "
                "h2d_gib=%.3f requests=%llu load_s=%.3f io_s=%.3f lookahead=%u swept=%d "
                "model_gib=%.3f\n",
                arm.c_str(), n, seconds, static_cast<double>(n) / seconds,
                nvme / 1073741824.0, warmb / 1073741824.0, h2d / 1073741824.0,
                static_cast<unsigned long long>(requests), load_s, io_s,
                host.sweep_lookahead_depth(), host.prefill_sweep_engaged() ? 1 : 0,
                model_bytes / 1073741824.0);
    return 0;
}
