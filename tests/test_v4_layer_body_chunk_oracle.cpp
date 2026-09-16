// -----------------------------------------------------------------------------
// Tier-3 gate, item 19 — chunked batched prefill, `chunk ≡ serial`.
//
// The plan's gate is one sentence: "prefill(chunk) must produce byte-equivalent
// state, logits, and token id to running the same tokens serially". This gate
// asserts exactly that, for the three attention classes, and it asserts it as
// **equality** rather than as a tolerance, because with the composition this
// repository now uses there is no numerical reason for a difference to exist.
//
// There is no fp64 oracle in this file, deliberately. `chunk ≡ serial` is not a
// claim about the graph's arithmetic — Tier 1 and Tier 2 own that — it is a claim
// about *batching*: that processing `N` tokens together gives what the same `N`
// tokens give one at a time. So the reference is the same path driven with
// `count = 1`, which is serial by definition, and the gate asks whether batching
// changed anything. Three comparisons, each of which fails on its own kind of
// defect:
//
//   C   chunk {5,5,3} == chunk {1,…}            batching changes nothing
//   C   chunk {6,6,1} == chunk {4,4,4,1} == {1,…}    schedule-independent
//   C2  chunk {1,…}   == run_layer_body_decoding     the reference is the graph
//
// C2 is what keeps C from being circular: it ties the one-at-a-time chunk run to
// the Tier-2 certified decode body, so the equality in C is between the batch path
// and the certified path rather than between two copies of the same new code.
//
// ## The structural property, and the false version of it
//
// The obvious chunking — write every key of the chunk into the local ring, then run
// every query — is **not equivalent to serial at any chunk length above one**, and
// the gate asserts that the implementation does not do it. The proof is in the
// header of `core/v4_layer_body_batch.hpp` and is recorded as trap 39: for a ring
// of `C` slots the write for position `p` lands in slot `p mod C`, which held
// position `p − C`, and `p − C` is the *oldest* key of the window of the chunk's
// own first query. The chunk's last write therefore evicts a key its first query
// needs.
//
// The implementation keeps the chunk's keys in a per-chunk buffer, gives each query
// a *composed row-set* (pre-chunk ring rows in its window + the chunk's own rows up
// to itself), and commits the keys to the ring afterwards. The gate checks that
// ordering directly: **after every token's pre-attention half the ring has not
// moved** and the chunk's keys are in the chunk buffer. Mutation M19-2 (write the
// ring as the chunk goes) is the false version and fails section B.
//
// ## Why equality rather than a tolerance
//
// The composed rows are ordered by *ring slot*, which is the order the decode
// path's attention kernel already iterates the ring in. Both paths therefore sum
// the same `exp` terms in the same sequence over bit-identical keys, so the only
// thing that can separate them is a defect. A tolerance would hide exactly what
// this gate exists to find, so the comparison is on raw fp16 bit patterns.
//
// ## What is asserted
//
//   A. CLOSED FORMS — the composed row-set's size and position multiset against
//      `min(pos+1, capacity)` and the window, with no device work.
//   B. THE RING DOES NOT MOVE DURING A CHUNK — after every token's pre-attention
//      half the ring is exactly what it was before the chunk.
//   C. chunk ≡ serial, BIT-EXACT — per token: residual, router logits, ids, weights;
//      and at the end: the whole ring, every committed compressed entry with its
//      position, and the compressor's partial ring positions. Four schedules, over
//      a prompt long enough to commit HCA's compressed entry and to cross many CSA
//      boundaries and ring wraps.
//   D. NON-VACUITY — the refusals fire, and the gate reports the total number of
//      differing values it saw rather than only a pass bit.
//
// Deliberately NOT covered here, and named so it is not mistaken for coverage:
//   * throughput. The plan says structural equivalence and speed are two different
//     gates. The composition is currently a per-row loop of device-to-device copies:
//     correct, and not fast. Turning it into one gather kernel is a separate step
//     with its own measurement.
//   * the indexer's per-token host round-trip in `select_indexer_topk`, which the
//     plan requires to become on-device for prefill. It changes no value, so this
//     gate cannot see it — recorded in the plan's open list instead.
//   * the routed-expert arithmetic and any tiering (synthetic payloads, as in items
//     17/18), and the real 128-token window with `index_topk = 512` (shrunk, as in
//     items 16–18). The *compressed* paths are exercised for real: the prompt is
//     long enough for HCA to commit an entry and for CSA to commit many.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body_batch.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "support/v4_layer_body_gate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using aeon::core::PipelineScratchBuffers;
using aeon::core::V4LayerBodyBatchScratch;
using aeon::core::V4LayerBodyOutput;
using aeon::core::V4LayerBodyRow;
using aeon::core::V4LayerBodyTables;
using aeon::testgate::check;
using aeon::testgate::GateExpertExecutor;
using aeon::testgate::kHeadDim;
using aeon::testgate::kHcDim;
using aeon::testgate::kHidden;
using aeon::testgate::kRoutedExperts;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kMaxSeq = 512;

// The window is 10 rather than the model's 128, for the reason items 16–18 give:
// the ring must wrap inside a short run. 10 also makes the two refusal paths
// *different* — CSA's compressor partial ring is `2 × ratio = 8` wide, so a chunk
// of 9 is legal for the local ring and illegal for the compressor, which is what
// lets the gate isolate the second limit.
constexpr uint32_t kLocalWindow = 10;
constexpr uint32_t kIndexTopk = 3;

// Long enough for HCA (ratio 128) to commit its first compressed entry at position
// 127, and for CSA (ratio 4) to commit 32. 13 cycles of 10 tokens, so the four
// schedules below — the serial one and three chunkings — all cover it exactly.
constexpr uint32_t kTokens = 130;
constexpr uint32_t kStackSize = 3;
constexpr uint32_t kWorkspaceTokens = 16;

const std::array<uint32_t, 8> kTokenIds = {1000, 42, 7777, 1780, 90125, 130, 55, 4096};

struct StackLayer {
    const char* label{nullptr};
    uint32_t layer_id{0};
    aeon::core::V4Layer device;
    uint32_t local_capacity{0};
    uint32_t compressor_capacity{0};
    int32_t ratio{0};
    bool compressed{false};
};

struct TokenRecord {
    std::vector<uint16_t> residual;   // hc_dim raw fp16 bits
    std::vector<uint16_t> logits;     // 256 raw fp16 bits
    std::vector<int32_t> ids;         // 6
    std::vector<float> weights;       // 6
};

struct StateRecord {
    std::vector<uint16_t> ring_keys;
    std::vector<int64_t> ring_positions;
    std::vector<uint16_t> compressed_keys;
    std::vector<int64_t> compressed_positions;
    std::vector<int64_t> compressor_partial_positions;
};

size_t total_differences = 0;

uint32_t committed_for(int32_t ratio, uint32_t tokens) {
    if (ratio <= 0) return 0;
    return static_cast<uint32_t>(static_cast<int64_t>(tokens) / static_cast<int64_t>(ratio));
}

template <typename T>
std::vector<T> read_device(const void* device, size_t count) {
    std::vector<T> host(count);
    CHECK_HIP(hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP(hipStreamSynchronize(0));
    return host;
}

StateRecord snapshot_state(const StackLayer& layer, uint32_t committed) {
    StateRecord record;
    const size_t capacity = layer.local_capacity;
    record.ring_keys = read_device<uint16_t>(layer.device.d_local_key_cache,
                                             capacity * kHeadDim);
    record.ring_positions = read_device<int64_t>(layer.device.d_local_positions, capacity);
    if (committed != 0) {
        record.compressed_keys = read_device<uint16_t>(layer.device.d_compressed_key_cache,
                                                       static_cast<size_t>(committed) * kHeadDim);
        record.compressed_positions = read_device<int64_t>(layer.device.d_compressed_positions,
                                                           committed);
    }
    if (layer.compressed) {
        record.compressor_partial_positions = read_device<int64_t>(
            layer.device.d_compressor_partial_positions, layer.compressor_capacity);
    }
    return record;
}

// Counts differing elements, prints one line, accumulates the total.
template <typename T>
size_t compare_bits(const std::string& label, const std::vector<T>& want,
                    const std::vector<T>& got) {
    if (want.size() != got.size()) {
        std::printf("  %-58s SIZE %zu vs %zu  FAIL\n", label.c_str(), want.size(), got.size());
        total_differences += std::max(want.size(), got.size()) + 1;
        return std::max(want.size(), got.size()) + 1;
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
    std::printf("  %-58s %s  %s\n", label.c_str(), note.c_str(),
                differing == 0 ? "PASS" : "FAIL");
    total_differences += differing;
    return differing;
}

// Runs one schedule over one layer and returns every token's record.
//
// `limit` is how many tokens to process: `kTokens` for a full run, less for the
// warm-up in section B, which only needs a few tokens in the ring.
//
// `use_decode_body` runs the certified Tier-2 decode body one token at a time and
// is only ever used with an all-ones schedule; otherwise the chunk path is driven,
// which at `count = 1` is the serial reference.
std::vector<TokenRecord> run_layer_schedule(
    StackLayer& layer,
    V4LayerBodyBatchScratch& workspace,
    const V4LayerBodyTables& tables,
    const std::vector<uint32_t>& schedule,
    bool use_decode_body,
    half* d_residuals,
    PipelineScratchBuffers& scratch,
    GateExpertExecutor& executor,
    aeon::core::V4LayerBodyObserver& observer,
    uint32_t limit = kTokens) {
    std::vector<TokenRecord> records(kTokens);
    uint32_t offset = 0;
    for (uint32_t size : schedule) {
        if (offset >= limit) break;
        const uint32_t count = std::min(size, limit - offset);
        if (use_decode_body) {
            for (uint32_t r = 0; r < count; ++r) {
                const uint32_t position = offset + r;
                const half* source = d_residuals + static_cast<size_t>(position) * kHcDim;
                const std::vector<uint16_t> bits = read_device<uint16_t>(source, kHcDim);
                std::vector<float> mirror(kHcDim);
                for (size_t i = 0; i < mirror.size(); ++i) {
                    mirror[i] = __half2float(*reinterpret_cast<const __half*>(&bits[i]));
                }
                CHECK_HIP(hipMemcpy(scratch.d_res_in_half, bits.data(), kHcDim * sizeof(uint16_t),
                                    hipMemcpyHostToDevice));
                CHECK_HIP(hipMemcpy(scratch.d_res_in, mirror.data(), kHcDim * sizeof(float),
                                    hipMemcpyHostToDevice));

                const V4LayerBodyOutput out = aeon::core::run_layer_body_decoding(
                    layer.device, scratch, tables, kTokenIds[position % kTokenIds.size()],
                    position, 0, executor, observer);

                TokenRecord record;
                record.residual = read_device<uint16_t>(scratch.d_res_in_half, kHcDim);
                record.logits = read_device<uint16_t>(scratch.d_router_logits, 256);
                record.ids = out.topk_indices;
                record.weights = out.topk_weights;
                // Copy back *before* the move: `std::move` leaves `record` empty,
                // and a zero-length host pointer is an invalid copy argument.
                CHECK_HIP(hipMemcpy(d_residuals + static_cast<size_t>(position) * kHcDim,
                                    record.residual.data(), kHcDim * sizeof(uint16_t),
                                    hipMemcpyHostToDevice));
                records[position] = std::move(record);
            }
        } else {
            for (uint32_t r = 0; r < count; ++r) {
                const uint32_t position = offset + r;
                const half* source = d_residuals + static_cast<size_t>(position) * kHcDim;
                const std::vector<uint16_t> bits = read_device<uint16_t>(source, kHcDim);
                std::vector<float> mirror(kHcDim);
                for (size_t i = 0; i < mirror.size(); ++i) {
                    mirror[i] = __half2float(*reinterpret_cast<const __half*>(&bits[i]));
                }
                V4LayerBodyRow view = workspace.row(r);
                CHECK_HIP(hipMemcpy(view.d_res_in_half, bits.data(), kHcDim * sizeof(uint16_t),
                                    hipMemcpyHostToDevice));
                CHECK_HIP(hipMemcpy(view.d_res_in, mirror.data(), kHcDim * sizeof(float),
                                    hipMemcpyHostToDevice));
            }

            std::vector<uint32_t> ids(count);
            for (uint32_t r = 0; r < count; ++r) {
                ids[r] = kTokenIds[(offset + r) % kTokenIds.size()];
            }
            const std::vector<V4LayerBodyOutput> outputs = aeon::core::run_layer_body_chunk(
                layer.device, workspace, tables, ids.data(), offset, count, 0, executor,
                observer);

            for (uint32_t r = 0; r < count; ++r) {
                const uint32_t position = offset + r;
                V4LayerBodyRow view = workspace.row(r);
                TokenRecord record;
                record.residual = read_device<uint16_t>(view.d_res_in_half, kHcDim);
                record.logits = read_device<uint16_t>(view.d_router_logits, 256);
                record.ids = outputs[r].topk_indices;
                record.weights = outputs[r].topk_weights;
                CHECK_HIP(hipMemcpy(d_residuals + static_cast<size_t>(position) * kHcDim,
                                    record.residual.data(), kHcDim * sizeof(uint16_t),
                                    hipMemcpyHostToDevice));
                records[position] = std::move(record);
            }
        }
        offset += count;
    }
    if (offset != limit) {
        throw std::logic_error("run_layer_schedule: the schedule did not cover every token");
    }
    return records;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-3 item 19: chunked batched prefill, chunk == serial\n";
    aeon::core::select_compute_device(true);
    bool ok = true;

    aeon::core::AeonModelLoader loader;
    loader.open_model(kModelDir);
    const aeon::core::DeepSeekV4Config cfg =
        aeon::core::DeepSeekV4Config::load_from_json(std::string(kModelDir) + "/config.json");
    const std::vector<aeon::core::V4LayerSpec> specs =
        aeon::core::V4ModelSpec::resolve_layers(cfg);

    struct LayerPlan {
        const char* label;
        uint32_t layer_id;
    };
    const LayerPlan plans[kStackSize] = {{"Sliding", 0}, {"CSA", 2}, {"HCA", 3}};

    std::vector<StackLayer> stack;
    stack.reserve(kStackSize);
    for (const LayerPlan& plan : plans) {
        stack.emplace_back();
        StackLayer& s = stack.back();
        s.label = plan.label;
        s.layer_id = plan.layer_id;
        aeon::core::V4LayerSpec spec = specs.at(plan.layer_id);
        spec.sliding_window = static_cast<int32_t>(kLocalWindow);
        if (spec.attention_kind == aeon::core::V4AttentionKind::CSA) {
            spec.index_topk = static_cast<int32_t>(kIndexTopk);
        }
        s.device.init_with_loader(spec, loader, kMaxSeq);
        s.local_capacity = s.device.local_cache_capacity();
        s.compressor_capacity = static_cast<uint32_t>(
            s.device.state_layout().compressor_partial_capacity);
        s.ratio = spec.compression_ratio;
        s.compressed = s.ratio != 0;
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

    // Each prompt token's embedding row, fp16, broadcast over the four HC streams —
    // the only value a prefill is handed from outside.
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
    half* d_residuals = nullptr;
    CHECK_HIP(hipMalloc(&d_residuals, seed.size() * sizeof(uint16_t)));

    std::array<std::vector<uint8_t>, kRoutedExperts> payloads;
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        payloads[k] = aeon::testgate::make_synthetic_payload(k + 1);
    }

    PipelineScratchBuffers scratch;
    scratch.allocate();
    GateExpertExecutor executor;
    executor.scratch = &scratch;
    executor.stream = 0;
    // Trap 38: the atomic accumulation's order is the scheduler's, so an equality
    // between two runs of the same schedule would be meaningless without a fixed
    // order. The plan requires a gate to state which accumulation it needs; this
    // one requires the deterministic path, and says so here.
    executor.deterministic = true;
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        executor.synthetic[k] = payloads[k].data();
        CHECK_HIP(hipMalloc(&executor.d_payload[k], aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
    }

    std::vector<V4LayerBodyBatchScratch> workspaces(kStackSize);
    for (size_t i = 0; i < stack.size(); ++i) {
        workspaces[i].allocate(stack[i].device, kWorkspaceTokens);
    }

    const auto reset_all = [&]() {
        for (StackLayer& s : stack) s.device.reset_generation_state();
        CHECK_HIP(hipMemcpy(d_residuals, seed.data(), seed.size() * sizeof(uint16_t),
                            hipMemcpyHostToDevice));
    };

    // -------------------------------------------------------------------
    // A. Closed forms for the composed row-set
    // -------------------------------------------------------------------
    std::cout << "\n--- A. the composed row-set, closed form ---\n";
    {
        const StackLayer& sliding = stack.front();
        V4LayerBodyBatchScratch& workspace = workspaces.front();
        bool all_ok = true;
        std::string detail = "every query checked";
        // Every (start, row) pair of short chunks, plus the wrap.
        for (uint32_t start = 0; start < 40 && all_ok; ++start) {
            for (uint32_t row = 0; row < 3 && all_ok; ++row) {
                const uint32_t query = start + row;
                const uint32_t rows = aeon::core::compose_local_rows(
                    sliding.device, workspace, start, row, query, 0);
                CHECK_HIP(hipStreamSynchronize(0));
                const uint32_t expect = std::min<uint32_t>(query + 1, sliding.local_capacity);
                const int64_t first = std::max<int64_t>(
                    0, static_cast<int64_t>(query) -
                           static_cast<int64_t>(sliding.local_capacity) + 1);
                int64_t expect_sum = 0;
                for (int64_t p = first; p <= static_cast<int64_t>(query); ++p) expect_sum += p;
                const std::vector<int64_t> positions =
                    read_device<int64_t>(workspace.composed_positions(), rows);
                int64_t sum = 0;
                bool in_window = true;
                for (uint32_t i = 0; i < rows; ++i) {
                    sum += positions[i];
                    in_window = in_window && positions[i] >= first &&
                                positions[i] <= static_cast<int64_t>(query);
                }
                if (rows != expect || sum != expect_sum || !in_window) {
                    all_ok = false;
                    detail = "query " + std::to_string(query) + ": " +
                             std::to_string(rows) + " rows (expected " +
                             std::to_string(expect) + "), position sum " +
                             std::to_string(sum) + " (expected " +
                             std::to_string(expect_sum) + "), in window " +
                             (in_window ? "yes" : "no");
                }
            }
        }
        ok &= check("row-set size = min(pos+1, window), positions are the window",
                    all_ok, detail);
    }

    // -------------------------------------------------------------------
    // B. The ring does not move while a chunk is being processed
    // -------------------------------------------------------------------
    std::cout << "\n--- B. the ring is not written during a chunk ---\n";
    for (size_t li = 0; li < stack.size(); ++li) {
        StackLayer& layer = stack[li];
        reset_all();

        // Warm the ring with three tokens, one at a time, so a chunk has pre-chunk
        // rows it must preserve.
        (void)run_layer_schedule(layer, workspaces[li], tables, {1, 1, 1}, false,
                                 d_residuals, scratch, executor, observer, 3);
        const StateRecord before = snapshot_state(layer, 0);

        // Phase 1 only: every token's pre-attention half — no query, no commit.
        const uint32_t count = 4;
        for (uint32_t r = 0; r < count; ++r) {
            (void)aeon::core::run_chunk_pre_attention(
                layer.device, workspaces[li], tables,
                kTokenIds[(3 + r) % kTokenIds.size()], 3 + r, r, 0, observer);
        }
        CHECK_HIP(hipStreamSynchronize(0));

        const StateRecord after = snapshot_state(layer, 0);
        size_t moved = compare_bits(std::string("    [") + layer.label +
                                        "] ring keys after phase 1",
                                    before.ring_keys, after.ring_keys);
        moved += compare_bits(std::string("    [") + layer.label +
                                  "] ring positions after phase 1",
                              before.ring_positions, after.ring_positions);
        ok &= check((std::string("    [") + layer.label +
                     "] the ring is untouched by phase 1").c_str(),
                    moved == 0, moved == 0 ? "unchanged" : "the chunk wrote the ring");

        // And the chunk's own keys really are in the chunk buffer, so "untouched"
        // is not just "nothing was written anywhere".
        std::vector<uint16_t> chunk_bits(count * kHeadDim);
        for (uint32_t r = 0; r < count; ++r) {
            const std::vector<uint16_t> bits =
                read_device<uint16_t>(workspaces[li].chunk_key(r), kHeadDim);
            std::copy(bits.begin(), bits.end(),
                      chunk_bits.begin() + static_cast<size_t>(r) * kHeadDim);
        }
        bool nonzero = false;
        for (uint16_t value : chunk_bits) {
            if (value != 0u && value != 0x8000u) {
                nonzero = true;
                break;
            }
        }
        ok &= check((std::string("    [") + layer.label + "] chunk keys were written").c_str(),
                    nonzero, nonzero ? "populated" : "all zero");
    }

    // -------------------------------------------------------------------
    // C. chunk ≡ serial
    // -------------------------------------------------------------------
    std::cout << "\n--- C. chunk == serial, bit-exact ---\n";
    // Six schedules, each covering the 130 tokens exactly: the serial reference, and
    // five chunkings. Three of them have boundaries inside ratio windows (CSA's ratio
    // is 4) and inside the local window's wrap (10 tokens); the last two are the
    // measured counterexample to the cap this repository used to enforce — a chunk of
    // 10 exceeds CSA's 8-wide compressor ring, and a chunk of 16 exceeds that *and*
    // the 10-wide local ring. Both are bit-identical to serial, which is why neither
    // ring guards the chunk size any more (see `v4_layer_body_batch.hpp`).
    std::vector<std::vector<uint32_t>> plan;
    plan.push_back(std::vector<uint32_t>(kTokens, 1));
    {
        std::vector<uint32_t> s;
        for (size_t cycle = 0; cycle < 13; ++cycle) s.insert(s.end(), {5, 5, 3});
        plan.push_back(s);
        s.clear();
        for (size_t cycle = 0; cycle < 13; ++cycle) s.insert(s.end(), {6, 6, 1});
        plan.push_back(s);
        s.clear();
        for (size_t cycle = 0; cycle < 13; ++cycle) s.insert(s.end(), {4, 4, 4, 1});
        plan.push_back(s);
        s.clear();
        for (size_t cycle = 0; cycle < 13; ++cycle) s.insert(s.end(), {10});
        plan.push_back(s);
        s.clear();
        for (size_t cycle = 0; cycle < 8; ++cycle) s.insert(s.end(), {16});
        s.push_back(2);
        plan.push_back(s);
    }

    std::vector<std::vector<std::vector<TokenRecord>>> layer_runs(
        kStackSize, std::vector<std::vector<TokenRecord>>(plan.size()));
    std::vector<std::vector<StateRecord>> layer_states(
        kStackSize, std::vector<StateRecord>(plan.size()));
    for (size_t li = 0; li < stack.size(); ++li) {
        for (size_t si = 0; si < plan.size(); ++si) {
            reset_all();
            layer_runs[li][si] = run_layer_schedule(stack[li], workspaces[li], tables,
                                                   plan[si], false, d_residuals, scratch,
                                                   executor, observer);
            layer_states[li][si] = snapshot_state(
                stack[li], committed_for(stack[li].ratio, kTokens));
        }
    }

    for (size_t li = 0; li < stack.size(); ++li) {
        std::printf("  [%s] layer %u: window %u, compressor ring %u, ratio %d\n",
                    stack[li].label, stack[li].layer_id, stack[li].local_capacity,
                    stack[li].compressor_capacity, stack[li].ratio);
        for (size_t si = 1; si < plan.size(); ++si) {
            std::string name = "    chunks of";
            for (size_t j = 0; j < 3 && j < plan[si].size(); ++j) {
                name += " " + std::to_string(plan[si][j]);
            }
            name += " x...";

            size_t differing = 0;
            size_t first_token = 0;
            for (uint32_t t = 0; t < kTokens; ++t) {
                const TokenRecord& want = layer_runs[li][0][t];
                const TokenRecord& got = layer_runs[li][si][t];
                const bool same = (want.residual == got.residual) &&
                                  (want.logits == got.logits) &&
                                  (want.ids == got.ids) &&
                                  (want.weights == got.weights);
                if (!same) {
                    if (differing == 0) first_token = t;
                    ++differing;
                }
            }
            total_differences += differing;
            ok &= check((name + ": every token").c_str(), differing == 0,
                        differing == 0
                            ? std::string("bit-identical")
                            : (std::to_string(differing) + " of " + std::to_string(kTokens) +
                               " differ, first at " + std::to_string(first_token)));

            const StateRecord& want = layer_states[li][0];
            const StateRecord& got = layer_states[li][si];
            size_t state_diff = 0;
            state_diff += compare_bits(name + ": ring keys", want.ring_keys, got.ring_keys);
            state_diff += compare_bits(name + ": ring positions",
                                       want.ring_positions, got.ring_positions);
            state_diff += compare_bits(name + ": compressed entries",
                                       want.compressed_keys, got.compressed_keys);
            state_diff += compare_bits(name + ": compressed positions",
                                       want.compressed_positions, got.compressed_positions);
            state_diff += compare_bits(name + ": partial ring positions",
                                       want.compressor_partial_positions,
                                       got.compressor_partial_positions);
            ok &= check((name + ": final state").c_str(), state_diff == 0,
                        state_diff == 0 ? std::string("bit-identical")
                                        : std::string("state differs"));
        }
    }

    // -------------------------------------------------------------------
    // C2. The serial reference is the certified decode body
    // -------------------------------------------------------------------
    //
    // This is the anchor for section C. Without it, section C only ever compares
    // the chunk path against itself at a different chunk length, so a mistake
    // applied to *both* sides — a wrong ring position at commit, say — would be
    // invisible. That is not hypothetical: mutation M19-4 (commit the position one
    // below the token's) survived the whole of section C, because the reference run
    // went through the same commit loop. The comparison below is therefore both the
    // tokens *and* the final state against `run_layer_body_decoding`, which is the
    // Tier-2 certified path and does not share the chunk driver at all.
    std::cout << "\n--- C2. the serial reference is the Tier-2 decode body ---\n";
    for (size_t li = 0; li < stack.size(); ++li) {
        const std::vector<uint32_t> serial(kTokens, 1);
        std::vector<std::vector<TokenRecord>> results(2);
        std::vector<StateRecord> state(2);
        for (size_t mode = 0; mode < 2; ++mode) {
            reset_all();
            results[mode] = run_layer_schedule(stack[li], workspaces[li], tables, serial,
                                               mode == 1, d_residuals, scratch, executor,
                                               observer);
            state[mode] = snapshot_state(stack[li],
                                         committed_for(stack[li].ratio, kTokens));
        }
        size_t differing = 0;
        size_t first_token = 0;
        for (uint32_t t = 0; t < kTokens; ++t) {
            const TokenRecord& want = results[0][t];
            const TokenRecord& got = results[1][t];
            if (!(want.residual == got.residual && want.logits == got.logits &&
                  want.ids == got.ids && want.weights == got.weights)) {
                if (differing == 0) first_token = t;
                ++differing;
            }
        }
        total_differences += differing;
        ok &= check((std::string("    [") + stack[li].label +
                     "] decode body == chunk path at count 1").c_str(),
                    differing == 0,
                    differing == 0
                        ? std::string("bit-identical")
                        : (std::to_string(differing) + " tokens differ, first at " +
                           std::to_string(first_token)));

        size_t state_diff = 0;
        state_diff += compare_bits(std::string("    [") + stack[li].label +
                                       "] decode body state: ring keys",
                                   state[0].ring_keys, state[1].ring_keys);
        state_diff += compare_bits(std::string("    [") + stack[li].label +
                                       "] decode body state: ring positions",
                                   state[0].ring_positions, state[1].ring_positions);
        state_diff += compare_bits(std::string("    [") + stack[li].label +
                                       "] decode body state: compressed entries",
                                   state[0].compressed_keys, state[1].compressed_keys);
        state_diff += compare_bits(std::string("    [") + stack[li].label +
                                       "] decode body state: compressed positions",
                                   state[0].compressed_positions,
                                   state[1].compressed_positions);
        state_diff += compare_bits(std::string("    [") + stack[li].label +
                                       "] decode body state: partial ring",
                                   state[0].compressor_partial_positions,
                                   state[1].compressor_partial_positions);
        ok &= check((std::string("    [") + stack[li].label +
                     "] decode body == chunk path final state").c_str(),
                    state_diff == 0,
                    state_diff == 0 ? std::string("bit-identical")
                                    : std::string("state differs"));
    }

    // -------------------------------------------------------------------
    // D. The refusals, and the total
    // -------------------------------------------------------------------
    std::cout << "\n--- D. non-vacuity ---\n";
    {
        StackLayer& sliding = stack.front();
        StackLayer& csa = stack[1];
        std::vector<uint32_t> ids(kWorkspaceTokens, 1);

        const auto refuses = [&](StackLayer& layer, V4LayerBodyBatchScratch& workspace,
                                 uint32_t count, const char* what) {
            try {
                (void)aeon::core::run_layer_body_chunk(layer.device, workspace, tables,
                                                       ids.data(), 0, count, 0, executor,
                                                       observer);
            } catch (const std::exception&) {
                return check(what, true, "refused");
            }
            return check(what, false, "accepted");
        };
        const auto accepts = [&](StackLayer& layer, V4LayerBodyBatchScratch& workspace,
                                 uint32_t count, const char* what) {
            try {
                (void)aeon::core::run_layer_body_chunk(layer.device, workspace, tables,
                                                       ids.data(), 0, count, 0, executor,
                                                       observer);
            } catch (const std::exception&) {
                return check(what, false, "refused");
            }
            return check(what, true, "accepted");
        };
        // The workspace is the real bound, and it is the *only* one. `workspaces[0]`
        // was allocated for `kWorkspaceTokens` = 16 tokens, so 17 exceeds it.
        ok &= refuses(sliding, workspaces[0], kWorkspaceTokens + 1,
                      "    a chunk larger than the workspace");
        // The rings are NOT a bound. CSA's local ring is 10 and its compressor ring
        // is 8; a chunk of 16 exceeds both and is accepted. Section C proved it
        // bit-identical to serial, so this acceptance is not merely paper: the old
        // guard refused exactly this chunk, and its stated reason — a boundary
        // reading a row a later token had overwritten — does not occur, because each
        // boundary materializes as its token is processed, before any later write.
        ok &= accepts(csa, workspaces[1], 16, "    a chunk larger than both rings (16)");

        std::printf("  %-58s %zu\n",
                    "D: total differing values across every comparison",
                    total_differences);
        ok &= (total_differences == 0);
    }

    std::printf("\n[Tier-3 chunk == serial] %s\n", ok ? "PASS" : "FAIL");
    for (uint32_t k = 0; k < kRoutedExperts; ++k) CHECK_HIP(hipFree(executor.d_payload[k]));
    CHECK_HIP(hipFree(d_residuals));
    CHECK_HIP(hipFree(d_cos));
    CHECK_HIP(hipFree(d_sin));
    CHECK_HIP(hipFree(d_cos_c));
    CHECK_HIP(hipFree(d_sin_c));
    return ok ? 0 : 1;
}
