// -----------------------------------------------------------------------------
// Tier-4, item 22 — the state contract: snapshot / restore, and R3.
//
// The plan's gate for the prefix cache manager is one sentence: "restore is
// **byte-exact** with respect to never having evicted, and a candidate boundary
// outside the local window is detected rather than served stale." This gate covers
// the first half (R3) and the layout it rests on (R1/R2).
//
// ## Why a restore gate is not the same as a copy round-trip
//
// `V4Layer::snapshot_state()` already existed, and a test that only checked
// `restore(snapshot()) == snapshot()` would be satisfied by a restore that copies
// the bytes and forgets a counter, or that silently skips a piece the layer's own
// byte accessor reports as zero. Neither would show up until a *continuation* went
// wrong, which is exactly the failure a prefix cache produces in the field: a
// reused prefix whose next token is subtly not what the unreused run would give.
//
// So the instrument here is the plan's own — a run that never stopped:
//
//   reference   tokens 0 … N+K−1, one layer, never interrupted
//   restored    tokens 0 … N−1, **snapshot**, reset, **restore**, tokens N … N+K−1
//
// and the requirement is that the restored run's tokens N … N+K−1 are bit-identical
// to the reference's, and that the two final states are byte-identical. Run length
// is chosen so the boundary is genuinely mid-ratio-window (so the compressor's
// in-progress partial state must survive the round trip) and so the local ring has
// wrapped (so the ring's slot assignment must survive it too).
//
// ## Anti-circularity, and how it is avoided here
//
// The comparison is not against an fp64 oracle — Tier 1 and items 16–18 own the
// arithmetic — it is against **the same certified body driven without the
// interruption**. `run_layer_body_decoding` is the Tier-2 certified path, and both
// sides go through it; the only difference between them is the snapshot/restore in
// the middle. That is what makes this a test of the state contract rather than of
// the graph. It is the same shape as item 19's C2, from the other direction.
//
// ## What is asserted
//
//   A. ROUND TRIP — snapshot, reset, restore, snapshot again is byte-identical, and
//      the state was non-empty to begin with (so A is not "nothing round-trips to
//      nothing").
//   B. R3 — a restored layer continues exactly like one that never stopped, at a
//      mid-ratio-window boundary and at a boundary, for Sliding / CSA / HCA.
//   C. LOAD-BEARING — the comparison in B can fail. A cleared snapshot, a zeroed
//      local ring, and a zeroed compressor-partial *position* each break the
//      continuation, so B is a statement about the state and not about a run that
//      would have agreed anyway.
//
// Deliberately NOT covered, and named so it is not mistaken for coverage:
//   * R4 — detecting a candidate boundary *outside* the local window and rebuilding
//     the ring rather than serving a stale one. That is the prefix *matcher*'s half
//     of item 22 and needs the cache key / block table, which do not exist yet.
//   * tier placement (R5): the pieces are moved VRAM→VRAM here, not to warm/cold.
//   * the cache key's non-token inputs (thinking mode, active tool set) — trap 23.
//   * the routed-expert arithmetic and the real 128-token window (shrunk here, as in
//     items 16–19); the compressed paths run for real, since HCA commits an entry.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "support/v4_layer_body_gate.hpp"

#include <algorithm>
#include <array>
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
using aeon::testgate::kHeadDim;
using aeon::testgate::kHcDim;
using aeon::testgate::kHidden;
using aeon::testgate::kRoutedExperts;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kMaxSeq = 512;
// The window is shrunk for the reason items 16–19 give: the ring must wrap inside a
// short run. It does not affect the compressor (whose dynamics depend on `ratio`),
// so the compressed half of the test is at the model's own ratio for its class.
constexpr uint32_t kLocalWindow = 8;
constexpr uint32_t kIndexTopk = 3;
constexpr uint32_t kTokens = 200;
constexpr uint32_t kStackSize = 3;

const std::array<uint32_t, 8> kTokenIds = {1000, 42, 7777, 1780, 90125, 130, 55, 4096};

size_t total_differences = 0;

struct StackLayer {
    const char* label{nullptr};
    uint32_t layer_id{0};
    V4Layer device;
    int32_t ratio{0};
    bool compressed{false};
    uint32_t coefficient{1};
};

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

// Counts differing bytes of one state piece and prints one line.
size_t compare_piece(const std::string& label, const std::vector<uint8_t>& want,
                     const std::vector<uint8_t>& got) {
    if (want.size() != got.size()) {
        std::printf("  %-52s SIZE %zu vs %zu  FAIL\n", label.c_str(), want.size(), got.size());
        const size_t penalty = std::max(want.size(), got.size()) + 1;
        total_differences += penalty;
        return penalty;
    }
    size_t differing = 0;
    size_t first = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (want[i] != got[i]) {
            if (differing == 0) first = i;
            ++differing;
        }
    }
    const std::string note = differing == 0
        ? std::string("bit-identical")
        : (std::to_string(differing) + "/" + std::to_string(want.size()) +
           " differ, first at " + std::to_string(first));
    std::printf("  %-52s %s  %s\n", label.c_str(), note.c_str(),
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
    differing += compare_piece(label + ": compressed_value_cache", want.compressed_value_cache,
                               got.compressed_value_cache);
    differing += compare_piece(label + ": compressed_positions", want.compressed_positions,
                               got.compressed_positions);
    differing += compare_piece(label + ": compressor_partial_kv", want.compressor_partial_kv,
                               got.compressor_partial_kv);
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
    differing += compare_piece(label + ": indexer_partial_score", want.indexer_partial_score,
                               got.indexer_partial_score);
    differing += compare_piece(label + ": indexer_partial_positions",
                               want.indexer_partial_positions, got.indexer_partial_positions);

    const auto counter = [&](const char* name, uint32_t a, uint32_t b) {
        const bool same = (a == b);
        std::printf("  %-52s %u vs %u  %s\n", (label + ": " + name).c_str(), a, b,
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

// Runs `count` tokens from logical position `start`, recording every one. The input
// residual for each token is the (read-only) seed row — a single layer's input is the
// embedding broadcast over the four HC streams, as in item 19's gate, so the tokens
// are comparable across runs regardless of what the layer wrote back.
std::vector<TokenRecord> run_tokens(
    StackLayer& layer,
    aeon::core::PipelineScratchBuffers& scratch,
    const V4LayerBodyTables& tables,
    const half* d_seeds,
    uint32_t start,
    uint32_t count,
    GateExpertExecutor& executor,
    aeon::core::V4LayerBodyObserver& observer) {
    std::vector<TokenRecord> records(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t position = start + i;
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
            layer.device, scratch, tables, kTokenIds[position % kTokenIds.size()],
            position, 0, executor, observer);

        TokenRecord& record = records[i];
        record.residual = read_device<uint16_t>(scratch.d_res_in_half, kHcDim);
        record.logits = read_device<uint16_t>(scratch.d_router_logits, 256);
        record.ids = out.topk_indices;
        record.weights = out.topk_weights;
    }
    return records;
}

// Compares the continuation (tokens `start …`) of two runs and prints one line.
bool same_continuation(const std::string& label, const std::vector<TokenRecord>& reference,
                       uint32_t start, const std::vector<TokenRecord>& got) {
    size_t differing = 0;
    for (uint32_t i = 0; i < got.size(); ++i) {
        if (!same_token(reference[start + i], got[i])) ++differing;
    }
    total_differences += differing;
    return check(label.c_str(), differing == 0,
                 differing == 0 ? std::string("bit-identical")
                                : (std::to_string(differing) + "/" + std::to_string(got.size()) +
                                   " tokens differ"));
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-4 item 22: state restore is byte-exact (R3)\n";
    aeon::core::select_compute_device(true);
    bool ok = true;

    aeon::core::AeonModelLoader loader;
    loader.open_model(kModelDir);
    const aeon::core::DeepSeekV4Config cfg =
        aeon::core::DeepSeekV4Config::load_from_json(std::string(kModelDir) + "/config.json");
    const std::vector<aeon::core::V4LayerSpec> specs =
        aeon::core::V4ModelSpec::resolve_layers(cfg);

    const std::array<std::pair<const char*, uint32_t>, kStackSize> plans = {
        std::pair{"Sliding", 0u}, std::pair{"CSA", 2u}, std::pair{"HCA", 3u}};

    std::vector<StackLayer> stack;
    stack.reserve(kStackSize);
    for (const auto& plan : plans) {
        stack.emplace_back();
        StackLayer& s = stack.back();
        s.label = plan.first;
        s.layer_id = plan.second;
        aeon::core::V4LayerSpec spec = specs.at(plan.second);
        spec.sliding_window = static_cast<int32_t>(kLocalWindow);
        if (spec.attention_kind == aeon::core::V4AttentionKind::CSA) {
            spec.index_topk = static_cast<int32_t>(kIndexTopk);
        }
        s.device.init_with_loader(spec, loader, kMaxSeq);
        s.ratio = spec.compression_ratio;
        s.compressed = s.ratio != 0;
        s.coefficient = (s.ratio == 4) ? 2u : 1u;
    }

    // ---- Device fixtures -------------------------------------------------
    float* d_cos = nullptr;
    float* d_sin = nullptr;
    float* d_cos_c = nullptr;
    float* d_sin_c = nullptr;
    V4LayerBodyTables tables{nullptr, nullptr, nullptr, nullptr};
    {
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

    std::vector<uint16_t> seed(kTokens * kHcDim);
    {
        const __half* embed = loader.get_data_ptr<__half>("embed.weight");
        for (uint32_t t = 0; t < kTokens; ++t) {
            const __half* row =
                embed + static_cast<size_t>(kTokenIds[t % kTokenIds.size()]) * kHidden;
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
    // gate requires the deterministic path and states so here.
    executor.deterministic = true;
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        executor.synthetic[k] = payloads[k].data();
        CHECK_HIP(hipMalloc(&executor.d_payload[k], aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
    }

    // -------------------------------------------------------------------
    // A. The round trip, and that there was something to round-trip
    // -------------------------------------------------------------------
    std::cout << "\n--- A. snapshot -> reset -> restore -> snapshot ---\n";
    for (StackLayer& layer : stack) {
        layer.device.reset_generation_state();
        // Long enough to wrap the 8-slot local ring and, for the compressed classes,
        // to commit entries (CSA 3, HCA 0 at 20 tokens — the HCA entry is reached in
        // section B).
        (void)run_tokens(layer, scratch, tables, d_seeds, 0, 20, executor, observer);
        CHECK_HIP(hipStreamSynchronize(0));

        const V4LayerStateSnapshot want = layer.device.snapshot_state();
        layer.device.reset_generation_state();
        const V4LayerStateSnapshot cleared = layer.device.snapshot_state();

        // Non-vacuity: the state the round trip carries must not be the cleared state,
        // or A would pass for a restore that writes nothing.
        bool differs_from_cleared =
            !(want.local_key_cache == cleared.local_key_cache &&
              want.local_positions == cleared.local_positions &&
              want.compressor_partial_positions == cleared.compressor_partial_positions &&
              want.local_valid_count == cleared.local_valid_count);
        ok &= check((std::string("  [") + layer.label +
                     "] the snapshot is not the cleared state").c_str(),
                    differs_from_cleared,
                    differs_from_cleared ? "non-empty" : "empty (A would be vacuous)");

        layer.device.restore_state(want);
        CHECK_HIP(hipStreamSynchronize(0));
        const V4LayerStateSnapshot got = layer.device.snapshot_state();
        const size_t differing = compare_snapshot(std::string("  [") + layer.label +
                                                      "] round trip", want, got);
        ok &= check((std::string("  [") + layer.label +
                     "] restore reproduces the snapshot exactly").c_str(),
                    differing == 0, differing == 0 ? "byte-identical" : "differs");
    }

    // -------------------------------------------------------------------
    // B. R3 — a restored layer continues like one that never stopped
    // -------------------------------------------------------------------
    std::cout << "\n--- B. restore is byte-exact w.r.t. never having evicted ---\n";
    struct Case {
        const char* label;
        size_t layer_index;
        uint32_t prefix;
        uint32_t continuation;
    };
    // Sliding: any boundary. CSA (ratio 4): prefix 10 is mid-window (10 % 4 == 2) and
    // prefix 12 lands exactly on a boundary. HCA (ratio 128): prefix 140 is past its
    // first committed entry (position 127) and mid-window (140 % 128 == 12).
    const std::array<Case, 4> cases = {{
        {"    [Sliding] mid-window", 0, 20, 20},
        {"    [CSA] mid-window", 1, 10, 30},
        {"    [CSA] on a boundary", 1, 12, 20},
        {"    [HCA] mid-window", 2, 140, 8},
    }};
    for (const Case& c : cases) {
        StackLayer& layer = stack[c.layer_index];
        const uint32_t total = c.prefix + c.continuation;
        if (total > kTokens) throw std::logic_error("case exceeds the seed buffer");

        // State the boundary honestly: mid-window is the case that needs the partial
        // state to survive, and it must be a real one.
        if (layer.ratio != 0) {
            const bool mid_window = (c.prefix % static_cast<uint32_t>(layer.ratio)) != 0;
            const bool expect_mid = std::string(c.label).find("mid-window") != std::string::npos;
            if (mid_window != expect_mid) {
                throw std::logic_error("case label disagrees with the boundary arithmetic");
            }
        }

        layer.device.reset_generation_state();
        const std::vector<TokenRecord> reference =
            run_tokens(layer, scratch, tables, d_seeds, 0, total, executor, observer);
        const V4LayerStateSnapshot reference_state = layer.device.snapshot_state();

        layer.device.reset_generation_state();
        (void)run_tokens(layer, scratch, tables, d_seeds, 0, c.prefix, executor, observer);
        const V4LayerStateSnapshot prefix_state = layer.device.snapshot_state();
        // Reset *before* restoring, so the restore has to rebuild the state rather
        // than merely leave the prefix's own bytes in place.
        layer.device.reset_generation_state();
        layer.device.restore_state(prefix_state);
        CHECK_HIP(hipStreamSynchronize(0));
        const std::vector<TokenRecord> restored =
            run_tokens(layer, scratch, tables, d_seeds, c.prefix, c.continuation, executor,
                       observer);
        const V4LayerStateSnapshot restored_state = layer.device.snapshot_state();

        ok &= same_continuation(std::string(c.label) + ": continuation", reference, c.prefix,
                                restored);
        const size_t differing = compare_snapshot(std::string(c.label) + ": final state",
                                                  reference_state, restored_state);
        ok &= check((std::string(c.label) + ": final state").c_str(), differing == 0,
                    differing == 0 ? "byte-identical" : "differs");
    }

    // -------------------------------------------------------------------
    // C. Load-bearing — the comparison in B can go red
    // -------------------------------------------------------------------
    // Without this, B says only that two runs of the same code agree. Each probe
    // corrupts one piece of the snapshot before restoring it and requires the
    // continuation to change, so B is a statement about the state.
    std::cout << "\n--- C. the restored state is what the continuation depends on ---\n";
    {
        StackLayer& layer = stack[1];   // CSA: has both a ring and a compressor
        const uint32_t prefix = 10;
        const uint32_t continuation = 30;
        const uint32_t total = prefix + continuation;

        layer.device.reset_generation_state();
        const std::vector<TokenRecord> reference =
            run_tokens(layer, scratch, tables, d_seeds, 0, total, executor, observer);

        layer.device.reset_generation_state();
        (void)run_tokens(layer, scratch, tables, d_seeds, 0, prefix, executor, observer);
        const V4LayerStateSnapshot good = layer.device.snapshot_state();
        layer.device.reset_generation_state();
        const V4LayerStateSnapshot cleared = layer.device.snapshot_state();

        const auto probe = [&](const char* label, const V4LayerStateSnapshot& poison) {
            layer.device.reset_generation_state();
            layer.device.restore_state(poison);
            CHECK_HIP(hipStreamSynchronize(0));
            const std::vector<TokenRecord> got =
                run_tokens(layer, scratch, tables, d_seeds, prefix, continuation, executor,
                           observer);
            size_t differing = 0;
            for (uint32_t i = 0; i < got.size(); ++i) {
                if (!same_token(reference[prefix + i], got[i])) ++differing;
            }
            ok &= check(label, differing != 0,
                        differing == 0 ? std::string("unchanged — the piece is not load-bearing")
                                       : (std::to_string(differing) + " tokens differ"));
        };

        // The untouched snapshot must reproduce the reference (this is B, restated as
        // the control for the three probes).
        layer.device.reset_generation_state();
        layer.device.restore_state(good);
        CHECK_HIP(hipStreamSynchronize(0));
        const std::vector<TokenRecord> control =
            run_tokens(layer, scratch, tables, d_seeds, prefix, continuation, executor,
                       observer);
        ok &= same_continuation("    [CSA] control: the good snapshot is exact", reference,
                                prefix, control);

        probe("    [CSA] cleared snapshot changes the continuation", cleared);

        V4LayerStateSnapshot no_ring = good;
        std::fill(no_ring.local_key_cache.begin(), no_ring.local_key_cache.end(), 0);
        probe("    [CSA] zeroed local ring changes the continuation", no_ring);

        V4LayerStateSnapshot no_partial = good;
        std::fill(no_partial.compressor_partial_positions.begin(),
                  no_partial.compressor_partial_positions.end(), 0xFF);
        probe("    [CSA] lost partial positions change the continuation", no_partial);

        // The size check is load-bearing in its own right: a snapshot from a
        // differently shaped layer must be refused, not written into buffers of the
        // wrong length.
        V4LayerStateSnapshot truncated = good;
        truncated.local_key_cache.clear();
        bool refused = false;
        try {
            layer.device.restore_state(truncated);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        ok &= check("    [CSA] a wrong-sized snapshot is refused", refused,
                    refused ? std::string("refused") : std::string("written"));
    }

    std::printf("\n[Gate] total differing values: %zu\n", total_differences);
    ok &= (total_differences == 0);
    std::printf("\n[Tier-4 item 22 restore] %s\n", ok ? "PASS" : "FAIL");
    for (uint32_t k = 0; k < kRoutedExperts; ++k) CHECK_HIP(hipFree(executor.d_payload[k]));
    CHECK_HIP(hipFree(d_seeds));
    CHECK_HIP(hipFree(d_cos));
    CHECK_HIP(hipFree(d_sin));
    CHECK_HIP(hipFree(d_cos_c));
    CHECK_HIP(hipFree(d_sin_c));
    return ok ? 0 : 1;
}
