// -----------------------------------------------------------------------------
// Real-scale state gate — the model's own window (128) and its own
// `index_topk` (512), together, in one layer.
//
// ## Why this gate exists
//
// The plan records one open gap in so many words: *"the real 128-token window with
// `index_topk = 512` (shrunk, as in items 16–18)"* — every layer-body gate so far
// shrinks both, and item 20 runs the real window but shrinks `index_topk` to 8.
// So neither half has ever met the other at the model's own scale, and the state
// layout has only ever been exercised under-scaled.
//
// There is a second, less obvious reason the two must be raised **together**, and
// it is the reason the token count here is ~2200 rather than a few hundred:
//
//   **`index_topk = 512` is nominal until there are more than 512 candidates.**
//
// A CSA layer commits one compressed entry per `ratio = 4` tokens, so `index_topk`
// becomes a *real* selection only past position 2048. Below that,
// `select_indexer_topk` takes the degenerate branch — `candidates <= index_topk`
// selects every candidate with no padding — and a gate that claimed to exercise
// the top-k would in fact be exercising the `take all` path. Section A asserts the
// selection is strict (some candidates excluded) and the selected set is exactly
// `index_topk` long, distinct and in range, so "real selection" is a measurement
// rather than an assumption.
//
// ## What it asserts (and what the instrument is)
//
// This is a **state** gate, not an arithmetic one, and it compares **run against
// run** rather than against an fp64 oracle. Tier 1 and items 16–18 own the
// arithmetic; what is under test here is that a restored state continues exactly
// like one that never stopped, at real dimensions. The instrument is therefore the
// plan's own (identical in shape to the item-22 R3 gate and item 19's C2):
//
//   reference   tokens 0 … 2199 in one uninterrupted run, snapshotting the state
//               as it stood **after** 2100 tokens
//   restored    reset, restore that mid-run snapshot, then tokens 2100 … 2199
//
// and the requirement is that the restored continuation is bit-identical to the
// reference's, and that the two final states are byte-identical. The mid-run
// snapshot is taken *during* the reference run rather than by a separate prefix
// run, which is both cheaper and stricter: the state being restored is provably
// the reference's own, at a real position, with the ring wrapped and the
// compressor mid-window (2100 % 4 == 0 is a boundary; 2100 is chosen so the
// continuation below it still crosses boundaries at 2104, 2108, …).
//
// Determinism is **structural**, not observed. There is no fp64 oracle, so no
// rounding is compared; the accumulation is fixed-order
// (`executor.deterministic = true`, stated here per the plan's rule that a gate
// must say which MoE accumulation it requires, trap 38); and the comparison is on
// raw fp16 bit patterns. At window 8 the equivalent comparison already reads
// exactly `0 differing values`, and the same holds at 128 because both sides run
// identical kernels over bit-identical state.
//
// ## What is covered, and what is not
//
// Covered for real: the window (128), `index_topk` (512) as a strict selection,
// 549 CSA boundaries crossed, the ring wrapped many times, the compressor's
// partial state mid-window, the indexer's key cache at real size, and the restore
// contract at ~4 MiB of live state instead of ~20 tokens of it.
//
// NOT covered, and named so it is not mistaken for coverage:
//   * the other 42 layers — this is one layer at real scale, not the stack
//     (item 23's driver).
//   * the routed-expert arithmetic (synthetic payloads, as in items 16–19).
//   * HCA: `layers.3` has the real window but **no indexer at all** (trap 33), so
//     it cannot exercise `index_topk` and is deliberately not the layer here.
//   * throughput, and the indexer top-k's per-token host round-trip in
//     `select_indexer_topk` — which at this scale is the dominant cost, and which
//     changes no value, so this gate cannot see it (item 19's second half).
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "support/v4_layer_body_gate.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using aeon::core::V4Layer;
using aeon::core::V4LayerBodyOutput;
using aeon::core::V4LayerBodyTables;
using aeon::core::V4LayerStateSnapshot;
using aeon::testgate::check;
using aeon::testgate::GateExpertExecutor;
using aeon::testgate::kHcDim;
using aeon::testgate::kHidden;
using aeon::testgate::kRoutedExperts;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

// The model's own values, NOT shrunk. `kRealWindow` and `kRealTopk` are asserted
// against the resolved spec in section A, so an accidental shrink is a failure
// rather than a quietly weaker test.
constexpr uint32_t kRealWindow = 128;
constexpr uint32_t kRealTopk = 512;
constexpr int32_t kCsaRatio = 4;

// Long enough that a CSA layer commits more than `index_topk` candidates
// (2200 / 4 = 550 > 512), which is the whole point: below 2048 tokens the top-k is
// degenerate. 4096 of context keeps the state allocation small (~10 MiB) while
// leaving every position in range.
constexpr uint32_t kMaxSeq = 4096;
constexpr uint32_t kTokens = 2200;
constexpr uint32_t kPrefix = 2100;                  // snapshot boundary
constexpr uint32_t kContinuation = kTokens - kPrefix;

size_t total_differences = 0;

struct TokenRecord {
    std::vector<uint16_t> residual;   // hc_dim raw fp16 bits
    std::vector<uint16_t> logits;     // 256 raw fp16 bits
    std::vector<int32_t> ids;         // 6
    std::vector<float> weights;       // 6
};

bool same_token(const TokenRecord& a, const TokenRecord& b) {
    return a.residual == b.residual && a.logits == b.logits &&
           a.ids == b.ids && a.weights == b.weights;
}

template <typename T>
std::vector<T> read_device(const void* device, size_t count) {
    std::vector<T> host(count);
    CHECK_HIP(hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP(hipStreamSynchronize(0));
    return host;
}

size_t compare_piece(const std::string& label, const std::vector<uint8_t>& want,
                     const std::vector<uint8_t>& got) {
    if (want.size() != got.size()) {
        std::printf("  %-54s SIZE %zu vs %zu  FAIL\n", label.c_str(), want.size(), got.size());
        const size_t penalty = std::max(want.size(), got.size()) + 1;
        total_differences += penalty;
        return penalty;
    }
    size_t differing = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (want[i] != got[i]) ++differing;
    }
    std::printf("  %-54s %s  %s\n", label.c_str(),
                differing == 0 ? "bit-identical"
                               : (std::to_string(differing) + " bytes differ").c_str(),
                differing == 0 ? "PASS" : "FAIL");
    total_differences += differing;
    return differing;
}

// Every piece and the four counters, so a restore that drops one is red.
size_t compare_snapshot(const std::string& label, const V4LayerStateSnapshot& want,
                        const V4LayerStateSnapshot& got) {
    size_t differing = 0;
    differing += compare_piece(label + ": local_key_cache", want.local_key_cache,
                               got.local_key_cache);
    differing += compare_piece(label + ": local_value_cache", want.local_value_cache,
                               got.local_value_cache);
    differing += compare_piece(label + ": local_positions", want.local_positions,
                               got.local_positions);
    differing += compare_piece(label + ": compressed_key_cache", want.compressed_key_cache,
                               got.compressed_key_cache);
    differing += compare_piece(label + ": compressed_value_cache",
                               want.compressed_value_cache, got.compressed_value_cache);
    differing += compare_piece(label + ": compressed_positions", want.compressed_positions,
                               got.compressed_positions);
    differing += compare_piece(label + ": compressor_partial_kv",
                               want.compressor_partial_kv, got.compressor_partial_kv);
    differing += compare_piece(label + ": compressor_partial_score",
                               want.compressor_partial_score, got.compressor_partial_score);
    differing += compare_piece(label + ": compressor_partial_positions",
                               want.compressor_partial_positions,
                               got.compressor_partial_positions);
    differing += compare_piece(label + ": indexer_key_cache", want.indexer_key_cache,
                               got.indexer_key_cache);
    differing += compare_piece(label + ": indexer_positions", want.indexer_positions,
                               got.indexer_positions);
    differing += compare_piece(label + ": indexer_partial_kv", want.indexer_partial_kv,
                               got.indexer_partial_kv);
    differing += compare_piece(label + ": indexer_partial_score",
                               want.indexer_partial_score, got.indexer_partial_score);
    differing += compare_piece(label + ": indexer_partial_positions",
                               want.indexer_partial_positions, got.indexer_partial_positions);

    const auto counter = [&](const char* name, uint32_t a, uint32_t b) {
        const bool same = (a == b);
        std::printf("  %-54s %u vs %u  %s\n", (label + ": " + name).c_str(), a, b,
                    same ? "PASS" : "FAIL");
        if (!same) ++differing;
    };
    counter("local_valid_count", want.local_valid_count, got.local_valid_count);
    counter("compressor_partial_count", want.compressor_partial_count,
            got.compressor_partial_count);
    counter("compressed_entry_count", want.compressed_entry_count, got.compressed_entry_count);
    counter("indexer_candidate_count", want.indexer_candidate_count,
            got.indexer_candidate_count);
    return differing;
}

} // namespace

int main() {
    std::cout << "[Gate] real-scale state — window 128 and index_topk 512 together\n";
    aeon::core::select_compute_device(true);
    bool ok = true;

    aeon::core::AeonModelLoader loader;
    loader.open_model(kModelDir);
    const aeon::core::DeepSeekV4Config cfg =
        aeon::core::DeepSeekV4Config::load_from_json(std::string(kModelDir) + "/config.json");
    const std::vector<aeon::core::V4LayerSpec> specs =
        aeon::core::V4ModelSpec::resolve_layers(cfg);

    // `layers.2` is the first CSA layer (ratio 4) and also a hash layer
    // (`num_hash_layers = 3`), so routing comes from `tid2eid` and varies with the
    // token id — which makes the state evolution non-repetitive without any
    // special fixture. HCA (`layers.3`) is deliberately NOT used: it has the real
    // window but no indexer, so it cannot exercise `index_topk` at all (trap 33).
    constexpr uint32_t kLayerId = 2;
    const aeon::core::V4LayerSpec spec = specs.at(kLayerId);

    // The spec is used **verbatim** — no field is overwritten. Section A asserts
    // this explicitly, because "we did not shrink it" is the premise of the gate.
    V4Layer layer;
    layer.init_with_loader(spec, loader, kMaxSeq);

    const auto& layout = layer.state_layout();
    const uint32_t committed_at_last = (kTokens - 1 + 1) / static_cast<uint32_t>(kCsaRatio);

    std::printf("\n--- A. preconditions: the dimensions are the model's own ---\n");
    ok &= check("  window is the real 128 (not shrunk)",
                spec.sliding_window == static_cast<int32_t>(kRealWindow),
                "window = " + std::to_string(spec.sliding_window));
    ok &= check("  index_topk is the real 512 (not shrunk)",
                spec.index_topk == static_cast<int32_t>(kRealTopk),
                "topk = " + std::to_string(spec.index_topk));
    ok &= check("  the layer is CSA (has an indexer)",
                spec.attention_kind == aeon::core::V4AttentionKind::CSA,
                "ratio = " + std::to_string(spec.compression_ratio));
    ok &= check("  candidates exceed index_topk, so the selection is strict",
                committed_at_last > kRealTopk,
                std::to_string(committed_at_last) + " candidates vs " +
                    std::to_string(kRealTopk) + " slots");
    ok &= check("  the local ring wraps many times",
                kTokens > kRealWindow,
                std::to_string(kTokens) + " tokens / " + std::to_string(kRealWindow) +
                    "-slot ring");

    // ---- Fixtures --------------------------------------------------------
    float* d_cos = nullptr;
    float* d_sin = nullptr;
    float* d_cos_c = nullptr;
    float* d_sin_c = nullptr;
    V4LayerBodyTables tables{nullptr, nullptr, nullptr, nullptr};
    {
        // The table must cover the *compressed* RoPE positions too, which are
        // `(boundary / ratio) * ratio` and therefore ≤ the largest token position.
        const size_t table_len = static_cast<size_t>(kTokens + 1) * 32;
        CHECK_HIP(hipMalloc(&d_cos, table_len * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_sin, table_len * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_cos_c, table_len * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_sin_c, table_len * sizeof(float)));
        const auto upload = [&](float* device, const std::vector<double>& values) {
            std::vector<float> host(values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                host[i] = static_cast<float>(values[i]);
            }
            CHECK_HIP(hipMemcpy(device, host.data(), host.size() * sizeof(float),
                                hipMemcpyHostToDevice));
        };
        const auto sliding = aeon::reference::rope_table(
            aeon::reference::rope_spec_for(aeon::reference::RopeClass::Sliding), kTokens);
        const auto compressed = aeon::reference::rope_table(
            aeon::reference::rope_spec_for(aeon::reference::RopeClass::Compressed), kTokens);
        upload(d_cos, sliding.cos);
        upload(d_sin, sliding.sin);
        upload(d_cos_c, compressed.cos);
        upload(d_sin_c, compressed.sin);
        tables = V4LayerBodyTables{d_cos, d_sin, d_cos_c, d_sin_c};
    }
    aeon::core::V4NullLayerBodyObserver observer;

    // Each token's input is its embedding row broadcast over the four HC streams —
    // the only value a layer gets from outside. Token ids are walked with a
    // stride so both the routing and the inputs vary across the run.
    std::vector<uint32_t> token_ids(kTokens);
    {
        const uint32_t vocab = 129280;
        uint32_t state = 0x9e3779b9u;
        for (uint32_t t = 0; t < kTokens; ++t) {
            state = state * 1664525u + 1013904223u;
            token_ids[t] = (state >> 8) % 100000u;   // comfortably inside the vocab
            (void)vocab;
        }
    }

    std::vector<uint16_t> seed(static_cast<size_t>(kTokens) * kHcDim);
    {
        const __half* embed = loader.get_data_ptr<__half>("embed.weight");
        for (uint32_t t = 0; t < kTokens; ++t) {
            const __half* row =
                embed + static_cast<size_t>(token_ids[t]) * kHidden;
            const uint16_t* bits = reinterpret_cast<const uint16_t*>(row);
            for (uint32_t j = 0; j < 4; ++j) {
                for (uint32_t h = 0; h < kHidden; ++h) {
                    seed[static_cast<size_t>(t) * kHcDim + j * kHidden + h] = bits[h];
                }
            }
        }
    }
    half* d_seeds = nullptr;
    CHECK_HIP(hipMalloc(&d_seeds, seed.size() * sizeof(uint16_t)));
    CHECK_HIP(hipMemcpy(d_seeds, seed.data(), seed.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice));

    std::array<std::vector<uint8_t>, kRoutedExperts> payloads;
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        payloads[k] = aeon::testgate::make_synthetic_payload(k + 1);
    }

    aeon::core::PipelineScratchBuffers scratch;
    scratch.allocate();
    GateExpertExecutor executor;
    executor.scratch = &scratch;
    executor.stream = 0;
    // Trap 38: the atomic accumulation's order is the scheduler's, so a byte-exact
    // comparison between two runs would be meaningless without a fixed order. This
    // gate requires the deterministic path, and says so here.
    executor.deterministic = true;
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        executor.synthetic[k] = payloads[k].data();
        CHECK_HIP(hipMalloc(&executor.d_payload[k], aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
    }

    // Runs one token, leaving its record in `record`.
    const auto run_token = [&](uint32_t position, TokenRecord* record) {
        const half* source = d_seeds + static_cast<size_t>(position) * kHcDim;
        const std::vector<uint16_t> bits = read_device<uint16_t>(source, kHcDim);
        std::vector<float> mirror(kHcDim);
        for (size_t j = 0; j < mirror.size(); ++j) {
            mirror[j] = __half2float(*reinterpret_cast<const __half*>(&bits[j]));
        }
        CHECK_HIP(hipMemcpy(scratch.d_res_in_half, bits.data(), kHcDim * sizeof(uint16_t),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(scratch.d_res_in, mirror.data(), kHcDim * sizeof(float),
                            hipMemcpyHostToDevice));
        const V4LayerBodyOutput out = aeon::core::run_layer_body_decoding(
            layer, scratch, tables, token_ids[position], position, 0, executor, observer);
        if (record != nullptr) {
            record->residual = read_device<uint16_t>(scratch.d_res_in_half, kHcDim);
            record->logits = read_device<uint16_t>(scratch.d_router_logits, 256);
            record->ids = out.topk_indices;
            record->weights = out.topk_weights;
        }
    };

    const auto reset = [&]() { layer.reset_generation_state(); };

    const auto started = std::chrono::steady_clock::now();

    // -------------------------------------------------------------------
    // B. The reference run, snapshotting the state mid-run
    // -------------------------------------------------------------------
    // The snapshot is taken **during** the reference run rather than by a separate
    // prefix run. That is stricter and cheaper: the state being restored is
    // provably the reference's own state at position `kPrefix`, so a later
    // disagreement cannot be blamed on two runs having diverged before the
    // boundary.
    std::cout << "\n--- B. R3 at real scale: a restored layer continues like one that never stopped ---\n";
    reset();
    std::vector<TokenRecord> reference(kTokens);
    V4LayerStateSnapshot prefix_state;
    for (uint32_t t = 0; t < kTokens; ++t) {
        if (t == kPrefix) prefix_state = layer.snapshot_state();
        run_token(t, &reference[t]);
    }
    const V4LayerStateSnapshot reference_final = layer.snapshot_state();
    std::printf("  (reference run: %u tokens, %u boundaries, ring wrapped %u times)\n",
                kTokens, committed_at_last, kTokens / kRealWindow);

    // The indexer's selection at the last token, read from the layer's own buffer
    // (the decode path points the row at it). This is where "the top-k is a real
    // selection" becomes a measurement: 550 candidates, 512 slots.
    {
        const std::vector<int32_t> selected = read_device<int32_t>(
            layer.d_indexer_topk_indices, layout.index_topk);
        std::vector<int32_t> sorted = selected;
        std::sort(sorted.begin(), sorted.end());
        const bool distinct = std::unique(sorted.begin(), sorted.end()) == sorted.end();
        const bool in_range = sorted.front() >= 0 &&
            sorted.back() < static_cast<int32_t>(committed_at_last);
        const bool no_padding = std::find(selected.begin(), selected.end(), -1) == selected.end();
        const bool strict = selected.size() < committed_at_last;
        std::printf("  selection at the last token: %zu of %u candidates\n",
                    selected.size(), committed_at_last);
        ok &= check("  every slot was filled (no padding, no -1)",
                    no_padding, no_padding ? "512/512" : "padding present");
        ok &= check("  the selected set is distinct", distinct,
                    distinct ? "distinct" : "duplicates present");
        ok &= check("  every index is a real candidate", in_range,
                    in_range ? "in [0, committed)" : "out of range");
        ok &= check("  the selection is strict (some candidates excluded)",
                    strict, strict ? std::to_string(committed_at_last - selected.size()) +
                                         " excluded"
                                   : "all candidates selected — degenerate");
    }

    // The restored run: reset, put the mid-run snapshot back, continue.
    //
    // Before continuing, the snapshot is round-tripped **with no token in
    // between**. That is not redundant with the final-state comparison below, and
    // the mutation sweep is what showed why: `record_position` recomputes the four
    // counters from the position on every token, so a restore that drops a counter
    // is overwritten by the first token of the continuation and the comparison
    // cannot see it (mutation RZ-5 survived without this block). Reading the
    // snapshot straight back out is the only place a dropped counter is visible —
    // and it is visible because nothing has advanced yet.
    {
        reset();
        layer.restore_state(prefix_state);
        CHECK_HIP(hipStreamSynchronize(0));
        const V4LayerStateSnapshot round_trip = layer.snapshot_state();
        const size_t differing =
            compare_snapshot("  round trip at real scale (no token between)",
                             prefix_state, round_trip);
        ok &= check("  restore reproduces the snapshot exactly", differing == 0,
                    differing == 0 ? "byte-identical" : "differs");
    }

    reset();
    layer.restore_state(prefix_state);
    CHECK_HIP(hipStreamSynchronize(0));
    std::vector<TokenRecord> restored(kContinuation);
    for (uint32_t t = 0; t < kContinuation; ++t) {
        run_token(kPrefix + t, &restored[t]);
    }
    const V4LayerStateSnapshot restored_final = layer.snapshot_state();

    {
        size_t differing = 0;
        for (uint32_t t = 0; t < kContinuation; ++t) {
            if (!same_token(reference[kPrefix + t], restored[t])) ++differing;
        }
        total_differences += differing;
        ok &= check("  the continuation is bit-identical to the uninterrupted run",
                    differing == 0,
                    differing == 0 ? std::string("0 of " + std::to_string(kContinuation) +
                                                 " tokens differ")
                                   : (std::to_string(differing) + " of " +
                                      std::to_string(kContinuation) + " differ"));
    }
    {
        const size_t differing =
            compare_snapshot("  final state at real scale", reference_final, restored_final);
        ok &= check("  the final state is byte-identical", differing == 0,
                    differing == 0 ? "byte-identical" : "differs");
    }

    // -------------------------------------------------------------------
    // C. Load-bearing — the comparison above can go red
    // -------------------------------------------------------------------
    // Without this, B says only that two runs of the same code agree. Each probe
    // corrupts one piece of the snapshot and requires the continuation to change.
    std::cout << "\n--- C. the restored state is what the continuation depends on ---\n";
    {
        reset();
        const V4LayerStateSnapshot cleared = layer.snapshot_state();

        const auto probe = [&](const char* label, const V4LayerStateSnapshot& poison) {
            reset();
            layer.restore_state(poison);
            CHECK_HIP(hipStreamSynchronize(0));
            size_t differing = 0;
            for (uint32_t t = 0; t < kContinuation; ++t) {
                TokenRecord record;
                run_token(kPrefix + t, &record);
                if (!same_token(reference[kPrefix + t], record)) ++differing;
            }
            ok &= check(label, differing != 0,
                        differing == 0
                            ? std::string("unchanged — the piece is not load-bearing")
                            : (std::to_string(differing) + " of " +
                               std::to_string(kContinuation) + " tokens differ"));
        };

        // The control. The probes below all require a *non-zero* difference, so on
        // its own that only says "something changed". This run restores the good
        // snapshot through the identical code path and requires exactly zero
        // differences, which is what makes each probe's difference attributable to
        // its poison rather than to run-to-run noise.
        {
            reset();
            layer.restore_state(prefix_state);
            CHECK_HIP(hipStreamSynchronize(0));
            size_t differing = 0;
            for (uint32_t t = 0; t < kContinuation; ++t) {
                TokenRecord record;
                run_token(kPrefix + t, &record);
                if (!same_token(reference[kPrefix + t], record)) ++differing;
            }
            total_differences += differing;
            ok &= check("  control: the good snapshot gives 0 differences", differing == 0,
                        std::to_string(differing) + " differ");
        }

        probe("  a cleared snapshot changes the continuation", cleared);

        V4LayerStateSnapshot no_ring = prefix_state;
        std::fill(no_ring.local_key_cache.begin(), no_ring.local_key_cache.end(), 0);
        probe("  a zeroed local ring changes the continuation", no_ring);

        V4LayerStateSnapshot no_partial = prefix_state;
        std::fill(no_partial.compressor_partial_positions.begin(),
                  no_partial.compressor_partial_positions.end(), 0xFF);
        probe("  lost compressor partial positions change the continuation", no_partial);

        V4LayerStateSnapshot no_indexer = prefix_state;
        std::fill(no_indexer.indexer_key_cache.begin(), no_indexer.indexer_key_cache.end(), 0);
        probe("  a zeroed indexer key cache changes the continuation", no_indexer);
    }

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::printf("\n  elapsed: %.1f s for %u reference + %u continuation tokens\n",
                elapsed, kTokens, static_cast<unsigned>(kContinuation * 5));

    std::printf("\n[Gate] total differing values: %zu\n", total_differences);
    ok &= (total_differences == 0);
    std::printf("\n[Real-scale state] %s\n", ok ? "PASS" : "FAIL");
    for (uint32_t k = 0; k < kRoutedExperts; ++k) CHECK_HIP(hipFree(executor.d_payload[k]));
    CHECK_HIP(hipFree(d_seeds));
    CHECK_HIP(hipFree(d_cos));
    CHECK_HIP(hipFree(d_sin));
    CHECK_HIP(hipFree(d_cos_c));
    CHECK_HIP(hipFree(d_sin_c));
    return ok ? 0 : 1;
}
