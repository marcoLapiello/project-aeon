// -----------------------------------------------------------------------------
// Supply split — where does a batched window's exposed load actually go?
//
// ## Why this exists
//
// The supply-chain hot-path analysis found that a swept prefill's layer load is
// **exposed in front of that layer's compute** — the reads are a river, but the
// H2D upload is a bucket, and the layer boundary is a serial critical path. It then
// proposed several fixes (a second staging bank, a rolling corridor, moving the
// upload into the previous body's shadow), all of which *assume* which leg of that
// boundary is expensive. This bench exists to stop assuming.
//
// `sweep_load_ns` lumps the whole load together, so it cannot say whether the
// exposed time is the drive, the PCIe copy, or the bookkeeping around them. The
// counters this bench reads split it three ways, at the supply (so both phases
// report them):
//
//   io_wait      host time blocked waiting for NVMe completions   (`materialize`)
//   h2d_enqueue  CPU time to submit the H2D copies                 (`materialize`)
//   h2d_drain    host time blocked for the copies to land          (`release_streamed_staging`)
//   submit       time inside `io_uring_enter` alone                (the drive's queue)
//
// `h2d_drain` is sweep-only today (decode orders its copies asynchronously with
// `hipStreamWaitEvent`), which is itself worth seeing: it is the 256 per-slot
// `hipEventSynchronize` calls the analysis flagged. `h2d_drain_calls` records how
// many those were, so the driver-round-trip share is separable from the copy.
//
// ## How it differs from `bench_prefill_ab`
//
// That bench is a **strategy A/B**: three arms, each in its own process, over many
// lengths, to find the crossover. This one is an **attribution run**:
//
//   * the model is loaded **once** and every length is measured in one process —
//     the startup (a 160 GiB artifact, ~24 GiB of VRAM preload) dominates a fresh
//     process, which is most of what makes the A/B slow;
//   * a fresh session between lengths is `host.reset_generation_state()` — it zeroes
//     every layer's KV/compressor/indexer state, so each prompt is independent
//     without a reload (this is what `aeon_chat` does per conversation);
//   * the strategy is **not forced**: the production gate picks it, so a length
//     below the crossover runs the routed bank and one above runs the sweep. The
//     table prints which engaged — read it, because a "routed" row and a "swept" row
//     answer different questions;
//   * there is **no serial arm** — this is not a strategy comparison;
//   * it also measures **decode**, which drives the same supply at six experts per
//     layer per token, 64 greedy tokens. Prefill and decode are reset apart so each
//     reports its own split.
//
// **Greedy, not sampling.** Sampling would change which experts are touched (so two
// runs would not be comparable) and could truncate decode early on EOS. Greedy is
// deterministic, so the token sequence — and therefore the routing — is identical
// run to run. Decode stops early on EOS and the real step count is printed.
//
// **Expert residency intentionally persists across lengths**, exactly as it does
// between requests in production; only the KV/generation state is reset. The Hot-Hot
// / cold-bytes columns are therefore also a check for drift: if a later length looks
// anomalous, its residency carried over from the one before it.
//
// ## What it does not do
//
// It does not assert. The numbers are the deliverable; correctness under both
// supplies is `test_v4_prefill_sweep`'s and `test_v4_routed_prefill`'s subject.
//
// Usage: bench_supply_split [model_dir] [warm_gib] [prompt_file] [len ...]
//   Defaults: model dir, warm 0, the prefill corpus, lengths 64 256 512.
//   AEON_SPLIT_DECODE  number of decode tokens per length (default 64).
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_graph.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "architecture/deepseek_v4/core/v4_sampler.hpp"
#include "architecture/deepseek_v4/text/dsv4_prompt_encoder.hpp"
#include "architecture/deepseek_v4/text/dsv4_tokenizer.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using aeon::core::V4Graph;
using aeon::core::V4LayerBodyBatchScratch;
using aeon::core::V4ModelHost;
using aeon::core::V4Sampler;
using aeon::core::V4SamplerConfig;
using Clock = std::chrono::steady_clock;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr const char* kPromptFile = "profiling-prompts/prefill-corpus.txt";
constexpr uint32_t kContext = 2048;
constexpr uint32_t kChunk = 128;
constexpr uint32_t kWindow = 1024;
constexpr uint32_t kDefaultDecode = 64;
static_assert(kChunk <= V4LayerBodyBatchScratch::kMaxTokens,
              "the body chunk must fit the batch scratch");

double since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("bench_supply_split: cannot open prompt file " + path);
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// One row's worth of transfer accounting, read off the host after a phase.
struct SupplySplit {
    uint64_t io_wait_ns{0};
    uint64_t h2d_enqueue_ns{0};
    uint64_t h2d_drain_ns{0};
    uint64_t h2d_drain_calls{0};
    uint64_t dispatch_cpu_ns{0};
    uint64_t submit_ns{0};
    uint64_t submit_calls{0};
    uint64_t requests{0};
    // Lifetime byte counters, differenced across the phase.
    uint64_t nvme_bytes{0};
    uint64_t warm_bytes{0};
    uint64_t h2d_bytes{0};
};

SupplySplit read_split(V4ModelHost& host) {
    SupplySplit s;
    s.io_wait_ns = host.supply_io_wait_ns();
    s.h2d_enqueue_ns = host.supply_h2d_enqueue_ns();
    s.h2d_drain_ns = host.supply_h2d_drain_ns();
    s.h2d_drain_calls = host.supply_h2d_drain_calls();
    s.dispatch_cpu_ns = host.supply_dispatch_cpu_ns();
    s.submit_ns = host.direct_io_submit_ns();
    s.submit_calls = host.direct_io_submit_calls();
    s.requests = host.supply_requests();
    s.nvme_bytes = host.supply_bytes_from_nvme();
    s.warm_bytes = host.supply_bytes_from_host();
    s.h2d_bytes = host.supply_h2d_bytes();
    return s;
}

SupplySplit diff(const SupplySplit& after, const SupplySplit& before) {
    SupplySplit d;
    d.io_wait_ns = after.io_wait_ns - before.io_wait_ns;
    d.h2d_enqueue_ns = after.h2d_enqueue_ns - before.h2d_enqueue_ns;
    d.h2d_drain_ns = after.h2d_drain_ns - before.h2d_drain_ns;
    d.h2d_drain_calls = after.h2d_drain_calls - before.h2d_drain_calls;
    d.dispatch_cpu_ns = after.dispatch_cpu_ns - before.dispatch_cpu_ns;
    d.submit_ns = after.submit_ns - before.submit_ns;
    d.submit_calls = after.submit_calls - before.submit_calls;
    d.requests = after.requests - before.requests;
    d.nvme_bytes = after.nvme_bytes - before.nvme_bytes;
    d.warm_bytes = after.warm_bytes - before.warm_bytes;
    d.h2d_bytes = after.h2d_bytes - before.h2d_bytes;
    return d;
}

struct PrefillRow {
    uint32_t n{0};
    bool swept{false};
    double seconds{0.0};
    SupplySplit split;
    uint32_t lookahead{0};
    uint32_t hot_slots{0};
};

struct DecodeRow {
    uint32_t n{0};
    uint32_t steps{0};
    bool stopped_early{false};
    double seconds{0.0};
    SupplySplit split;
};

} // namespace

int main(int argc, char** argv) {
    // Positional: [model_dir] [warm_gib] [prompt_file] [len ...]. A leading `-` is
    // not a length, so the defaults still work when the markers are omitted.
    const std::string model_dir = argc > 1 ? argv[1] : kModelDir;
    const uint64_t warm_gib = argc > 2 ? static_cast<uint64_t>(std::atoi(argv[2])) : 0;
    // An empty third argument is the "use the default corpus" signal the driver
    // script passes, so it is not mistaken for a path.
    const std::string prompt_file =
        (argc > 3 && argv[3][0] != '\0') ? argv[3] : kPromptFile;
    std::vector<uint32_t> lengths;
    for (int i = 4; i < argc; ++i) {
        lengths.push_back(static_cast<uint32_t>(std::atoi(argv[i])));
    }
    if (lengths.empty()) {
        lengths = {64, 256, 512};
    }
    const uint32_t decode_tokens = [] {
        const char* env = std::getenv("AEON_SPLIT_DECODE");
        return env != nullptr ? static_cast<uint32_t>(std::atoi(env)) : kDefaultDecode;
    }();

    aeon::core::select_compute_device(true);

    aeon::core::AeonRuntimeConfig runtime;
    runtime.context_size = kContext;
    runtime.prefill_sweep = true;          // production gate decides per window
    runtime.prefill_chunk = kChunk;
    runtime.prefill_window = kWindow;
    runtime.warm_host_bytes = warm_gib * 1024ULL * 1024ULL * 1024ULL;
    // The sweep's arena is `2E` (two layer-blocks) by default. The A/B arm resizes it
    // to **one** layer-block at runtime — the pre-Phase-2 shape — through the depth
    // API rather than a config flag, because the depth must not be a behaviour switch
    // (plan R6).
    const uint32_t sweep_banks_env = [] {
        const char* env = std::getenv("AEON_SWEEP_BANKS");
        return env != nullptr ? static_cast<uint32_t>(std::atoi(env)) : 0u;
    }();

    V4ModelHost host;
    host.initialize(model_dir, runtime, /*verbose=*/false);
    if (sweep_banks_env == 1) {
        const uint32_t per_layer = host.registry().experts_per_layer;
        if (!host.resize_staging_slots(per_layer)) {
            throw std::runtime_error("bench_supply_split: could not resize staging to one bank");
        }
    } else if (sweep_banks_env != 0 && sweep_banks_env != 2) {
        throw std::runtime_error("bench_supply_split: AEON_SWEEP_BANKS must be 1 or 2");
    }
    V4Graph graph(host);

    // One tokenizer/encoder for the whole run, and the ids encoded once so every
    // length reads a prefix of the same prompt.
    aeon::text::Dsv4Tokenizer tokenizer;
    tokenizer.load(model_dir + "/tokenizer.aeon");
    aeon::text::Dsv4PromptEncoder encoder(tokenizer);
    aeon::text::Dsv4PromptMessage message;
    message.role = aeon::text::Dsv4Role::User;
    message.content = read_text_file(prompt_file);
    aeon::text::Dsv4PromptOptions options;
    std::vector<uint32_t> ids = encoder.encode_tokens({message}, options);
    const uint32_t eos = tokenizer.eos_token_id();

    uint32_t max_len = 0;
    for (uint32_t n : lengths) max_len = std::max(max_len, n);
    if (ids.size() < max_len) {
        throw std::runtime_error(
            "bench_supply_split: the prompt encodes to " + std::to_string(ids.size()) +
            " tokens, fewer than the requested " + std::to_string(max_len));
    }

    V4Sampler sampler(static_cast<uint32_t>(host.config().vocab_size));
    sampler.set_config(V4SamplerConfig{});  // defaults: temperature 0 => greedy

    const uint32_t layers = host.num_layers();
    const uint32_t per_layer = host.registry().experts_per_layer;
    const double model_gib =
        static_cast<double>(layers) * per_layer * 14'155'776.0 / 1073741824.0;
    const uint32_t gate = host.prefill_sweep_min_tokens();

    // P1.2 requirement: the pinned staging figure a gate reads is the figure
    // allocated. The budget report and the arena share `staging_slot_count`, so this
    // is the cross-check that they have not drifted.
    const size_t staging_slot_bytes = host.staging_slot_count() *
        host.loader().expert_format().payload_bytes;
    if (staging_slot_bytes != host.budget().transient_staging_bytes) {
        throw std::runtime_error(
            "bench_supply_split: staging budget " +
            std::to_string(host.budget().transient_staging_bytes) +
            " bytes does not match the arena's " + std::to_string(staging_slot_bytes));
    }

    std::printf(
        "[supply-split] model=%.1f GiB layers=%u E=%u warm=%llu GiB decode=%u tokens "
        "gate=%u sweep_engaged_for_window=%d banks=%u staging_slots=%u staging=%.2f GiB\n",
        model_gib, layers, per_layer, static_cast<unsigned long long>(warm_gib),
        decode_tokens, gate, host.prefill_sweep_enabled() ? 1 : 0,
        host.sweep_staging_banks(), host.staging_slot_count(),
        static_cast<double>(staging_slot_bytes) / 1073741824.0);

    std::vector<PrefillRow> prefill;
    std::vector<DecodeRow> decode;

    for (const uint32_t n : lengths) {
        // Fresh session: the KV/compressor/indexer state is zeroed for every layer,
        // so this prompt cannot read the previous one's context. Residency is not
        // touched — that is deliberate (see the header).
        host.reset_generation_state();

        // ---- prefill ----
        host.set_supply_phase(true);
        host.reset_supply_transfer_counters();
        const SupplySplit pre_before = read_split(host);

        const auto pstart = Clock::now();
        const half* logits = graph.forward_window(
            ids.data(), 0, n, std::min(kChunk, n), host.streams().compute);
        CHECK_HIP(hipStreamSynchronize(host.streams().compute));
        const double pre_seconds = since(pstart);

        PrefillRow prow;
        prow.n = n;
        prow.swept = host.prefill_sweep_engaged();
        prow.seconds = pre_seconds;
        prow.split = diff(read_split(host), pre_before);
        prow.lookahead = host.sweep_lookahead_depth();
        prow.hot_slots = host.registry().published_hot_slots();
        prefill.push_back(prow);

        // ---- decode ----
        host.set_supply_phase(false);
        host.reset_supply_transfer_counters();
        const SupplySplit dec_before = read_split(host);

        uint32_t next = sampler.select(logits, host.streams().compute);
        uint32_t position = n;
        uint32_t steps = 0;
        bool stopped_early = false;
        const auto dstart = Clock::now();
        for (uint32_t step = 0; step < decode_tokens; ++step) {
            if (next == eos) {
                stopped_early = true;
                break;
            }
            const half* dlogits = graph.forward_token(next, position, host.streams().compute);
            next = sampler.select(dlogits, host.streams().compute);
            ++position;
            ++steps;
        }
        CHECK_HIP(hipStreamSynchronize(host.streams().compute));

        DecodeRow drow;
        drow.n = n;
        drow.steps = steps;
        drow.stopped_early = stopped_early;
        drow.seconds = since(dstart);
        drow.split = diff(read_split(host), dec_before);
        decode.push_back(drow);
    }

    // ---- tables ----
    auto ms = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
    auto s = [](uint64_t ns) { return static_cast<double>(ns) / 1e9; };
    auto gib = [](uint64_t b) { return static_cast<double>(b) / 1073741824.0; };

    std::printf("\n");
    std::printf(
        "=============================================================="
        "==========================================================\n");
    std::printf("  PREFILL — exposed load split (one process, gate-selected strategy)\n");
    std::printf(
        "=============================================================="
        "==========================================================\n");
    std::printf(
        "%-5s %-8s %8s %7s %9s %9s %9s %9s %9s %9s %6s %6s\n",
        "N", "strategy", "wall_s", "tok/s", "nvme_GiB", "io_wait_s", "h2denq_ms",
        "h2ddrn_s", "disp_ms", "submit_ms", "drains", "hot");
    for (const auto& r : prefill) {
        std::printf(
            "%-5u %-8s %8.3f %7.2f %9.2f %9.3f %9.1f %9.3f %9.1f %9.1f %6llu %6u\n",
            r.n, r.swept ? "swept" : "routed", r.seconds,
            static_cast<double>(r.n) / r.seconds, gib(r.split.nvme_bytes),
            s(r.split.io_wait_ns), ms(r.split.h2d_enqueue_ns), s(r.split.h2d_drain_ns),
            ms(r.split.dispatch_cpu_ns), ms(r.split.submit_ns),
            static_cast<unsigned long long>(r.split.h2d_drain_calls), r.hot_slots);
    }
    std::printf("\n  note: io_wait/h2d_drain are host-blocked; disp/h2denq/submit are CPU "
                "cost; nvme = cold bytes only.\n");

    std::printf("\n");
    std::printf(
        "=============================================================="
        "==========================================================\n");
    std::printf("  DECODE — same channels, 6 experts/layer/token (greedy)\n");
    std::printf(
        "=============================================================="
        "==========================================================\n");
    std::printf(
        "%-5s %6s %9s %8s %12s %11s %11s %11s %11s %7s\n",
        "N", "steps", "ms/tok", "nvme_MiB", "io_wait_us", "h2denq_us", "h2ddrn_us",
        "disp_us", "submit_us", "warm_MiB");
    for (const auto& r : decode) {
        const double per_tok = r.steps > 0 ? 1000.0 * r.seconds / r.steps : 0.0;
        const double steps = r.steps > 0 ? r.steps : 1.0;
        std::printf(
            "%-5u %6u %9.2f %8.2f %12.1f %11.1f %11.1f %11.1f %11.1f %7.2f%s\n",
            r.n, r.steps, per_tok,
            static_cast<double>(r.split.nvme_bytes) / 1048576.0 / steps,
            static_cast<double>(r.split.io_wait_ns) / 1000.0 / steps,
            static_cast<double>(r.split.h2d_enqueue_ns) / 1000.0 / steps,
            static_cast<double>(r.split.h2d_drain_ns) / 1000.0 / steps,
            static_cast<double>(r.split.dispatch_cpu_ns) / 1000.0 / steps,
            static_cast<double>(r.split.submit_ns) / 1000.0 / steps,
            static_cast<double>(r.split.warm_bytes) / 1048576.0 / steps,
            r.stopped_early ? "  (eos)" : "");
    }
    std::printf("\n  note: h2ddrn_us is expected ~0 in decode (async stream order, no "
                "per-slot sync).\n");

    return 0;
}
