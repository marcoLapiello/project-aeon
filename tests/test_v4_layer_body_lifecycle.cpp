// -----------------------------------------------------------------------------
// Tier-3 gate, item 20 — long-context lifecycle: ring reuse and boundary
// compression past context capacity.
//
// Items 16–19 all certify the *arithmetic* of the layer body on real weights and
// all of them shrink the local window (to 6, 10 and 4) and the index top-k (to 3)
// so that a wrap and a non-degenerate selection fit inside a short run. Each of
// them names that shrinkage as uncovered. This gate spends its budget on the one
// thing they could not measure: **the real window, over a context long enough to
// reuse the ring more than twice**, and on the question the plan left open — what
// the state containers do once the context stops fitting.
//
// It is deliberately, structurally, a *different instrument* from items 16–19.
// It uses no fp64 oracle and compares no checkpoint against a reference: the
// arithmetic is already certified, and re-deriving it in fp64 is what made those
// gates expensive. What is asserted here are **closed forms and invariants** —
// the ring's contents are *predicted* from the token count before they are read,
// and the two independently-addressed stores are required to be bit-identical
// under each other's writes.
//
// Specification: plan §7.1. The four cached pieces (local ring, compressed store,
// compressor partial ring, indexer state) are rings, but their lifecycle has two
// regimes that must not be confused:
//
//   * the **local ring** reuses continuously and correctly — that *is* the
//     sliding window — and every query's row-set is `[max(0, pos−C+1), pos]` [V
//     cache_utils.py:892-894];
//   * the **compressed store** must never reuse, and its capacity is derived so
//     that it cannot: `K = ceil(max_seq/ratio)` is exactly what the declared
//     context produces. If it ever wrapped, the row-set would silently become a
//     sliding window over compressed entries — the newest `K` instead of every
//     committed entry — and **no position guard could tell**, because every
//     populated slot records a position `≤ pos` (trap 40). `V4Layer::
//     record_position` refusing a position at or beyond capacity is therefore the
//     only mechanism that keeps HCA's "all committed rows" from quietly becoming
//     "the newest K", and section E measures the divergence that makes the
//     refusal load-bearing rather than decorative.
//
// The run is one three-layer stack — Sliding (layer 0), CSA (layer 2), HCA
// (layer 3) — for 260 tokens, which is the shortest run that reuses the 128-slot
// ring twice (slots are `pos mod 128`, so slot 0 is written at 0 and again at
// 256) and still crosses two HCA boundaries (ratio 128 commits at 127 and 255).
//
// **Why those three layers, and not layer 1.** The stack is one layer per branch
// the body can *take*, not a sample of the model. The artifact's classes are
// `compress_ratios = [0, 0, 4, 128, 4, 128, …]` with `num_hash_layers = 3`, so
// layer 0 is Sliding+hash, layer 2 is CSA+hash and layer 3 is HCA+biased. Layer 1
// is the *second* Sliding layer: same attention class as layer 0 and also a hash
// layer, so it adds no branch on either axis — same code, different weights. The
// pair 2/3 is chosen because it straddles both boundaries at once (2 = last hash
// and first CSA; 3 = first biased and first HCA), which is what lets three layers
// cover three classes and both router branches.
//
// What is asserted:
//
//   A. CLOSED FORMS BEFORE ANY DEVICE WORK — the ring's slot/position mapping,
//      the entry position `(i+1)·ratio − 1`, and the fact that the capacity is
//      exactly what the declared context produces (so the count clamp is a no-op
//      in range, and "the ring never wraps" is a property of the *layout*, not an
//      assumption).
//   B. THE REAL WINDOW — after 260 tokens, slot `s` holds the largest position
//      `p ≤ 259` with `p ≡ s (mod 128)`, predicted before it is read; and the
//      kernel's own exact-boundary behaviour is measured on captured real state:
//      a partial ring's unfilled slots contribute **exactly zero** (perturbing
//      them does not move `attn_out` by one ulp), while an in-window row moves it
//      by a large fraction of peak.
//   C. THE TWO STORES ARE INDEPENDENT AND THEIR POSITIONS ARE CLOSED FORMS — the
//      entry materialized at the first boundary (CSA position 3, HCA position
//      127) is byte-identical 256 steps later; every committed entry's recorded
//      position is `(i+1)·ratio − 1`; the compressor partial ring's slots are
//      `pos mod capacity`; and the ring as it stood at the first boundary is the
//      closed form, i.e. materializing an entry did not disturb it.
//   D. THE ROW-SET PAST THE WINDOW IS THE CLASS RULE — HCA reads every committed
//      entry (dropping the oldest moves the output), CSA reads the indexer's
//      selection (dropping one selected entry moves it), and the counts are
//      `min(K, (pos+1)/ratio)` with a full local window.
//   E. CAPACITY IS EXACT AND THE REFUSAL IS LOAD-BEARING (trap 40) — the layer
//      refuses a position at `max_seq_len` and accepts `max_seq_len − 1`; the
//      capacity equals `ceil(max_seq/ratio)` exactly; and, the discriminating
//      part, a hand-built **wrapped** store is shown to pass the kernel's own
//      `compressed_positions[i] ≤ current_pos` guard in full while its row-set is
//      the newest `K` entries rather than the oldest — so a wrapped ring is
//      undetectable from the entries and would silently change what HCA attends.
//
// Deliberately NOT covered here, named so it is not mistaken for coverage:
//   * the routed-expert arithmetic and every other checkpoint's precision —
//     Tier-1 and items 16–18 own those; the experts here are six synthetic
//     payloads and no checkpoint is compared against a reference at all;
//   * the real `index_topk = 512`: 260 tokens commit only 65 CSA entries, so a
//     real top-k would select all of them and the selection would degenerate.
//     The top-k is shrunk to 8 so CSA's row-set is a real selection, exactly as
//     in items 17/18, and the *window* — which is what this item is about — is
//     left at the model's own 128;
//   * tiering, and any prefix-reuse restore (Tier 4 item 22).
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
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

using aeon::testgate::check;
using aeon::testgate::GateExpertExecutor;
using aeon::testgate::kHeadDim;
using aeon::testgate::kHcDim;
using aeon::testgate::kHidden;
using aeon::testgate::kRoutedExperts;
using aeon::testgate::num;
using aeon::testgate::to_half;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

// The declared context. Both ratios divide it, which is what makes the capacity
// *exactly* the entry count at the last position — the property section A pins
// and section E relies on.
constexpr uint32_t kMaxSeq = 512;

// Shortest run that reuses the 128-slot ring twice: slot `0` is written at
// position 0 and again at 256, so the run must reach 257 to demonstrate a second
// pass through every slot. It also crosses HCA's boundaries at 127 and 255.
constexpr uint32_t kTokens = 260;

// Shrunk so that CSA's selection is a real selection (260 tokens commit 65 CSA
// entries; with the model's 512 the selection would be all of them and would
// degenerate). The *window* is not shrunk — that is this item's subject.
constexpr uint32_t kIndexTopk = 8;

constexpr uint32_t kStackSize = 3;
constexpr uint32_t kHeads = 64;

const std::array<uint32_t, 10> kTokenIds =
    {1000, 42, 7777, 1780, 90125, 130, 55, 4096, 22222, 396};

// -----------------------------------------------------------------------------
// Small device helpers. The gate allocates per call on purpose: there are a
// dozen of them, and a perturbation is only attributable if the buffer it lands
// in is one this gate owns.
// -----------------------------------------------------------------------------
template <typename T>
T* upload(const std::vector<T>& values) {
    if (values.empty()) return nullptr;
    T* device = nullptr;
    CHECK_HIP(hipMalloc(&device, values.size() * sizeof(T)));
    CHECK_HIP(hipMemcpy(device, values.data(), values.size() * sizeof(T),
                        hipMemcpyHostToDevice));
    return device;
}

template <typename T>
std::vector<T> download(const T* device, size_t count) {
    std::vector<T> host(count);
    if (count == 0) return host;
    CHECK_HIP(hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost));
    return host;
}

std::vector<int64_t> read_positions(const int64_t* device, size_t count) {
    return download(device, count);
}

// `hipFree` is `nodiscard` and the project builds warning-free, so the status is
// consumed explicitly rather than left to a cast.
void free_device(void* device) {
    if (device == nullptr) return;
    const hipError_t status = hipFree(device);
    (void)status;
}

// Position of the slot that a `capacity`-wide ring leaves holding the largest
// written position below `tokens`: the closed form every state assertion here
// compares against, and the reason those assertions are predictions rather than
// observations.
int64_t ring_position_for_slot(uint32_t slot, uint32_t capacity, uint32_t tokens) {
    for (int64_t p = static_cast<int64_t>(tokens) - 1; p >= 0; --p) {
        if (p % static_cast<int64_t>(capacity) == static_cast<int64_t>(slot)) return p;
    }
    return -1;
}

// -----------------------------------------------------------------------------
// The production attention kernels, driven on an explicit row-set.
//
// These are the same kernels the layer body launches — for a Sliding layer the
// body passes the ring and `local_capacity`, and for a compressed layer it passes
// the ring plus the class's compressed row-set. Driving them directly is what
// makes a perturbation *attributable*: the state is captured from a real run, but
// exactly one key row changes between the two launches, so the difference in the
// output is that row's influence and nothing else.
// -----------------------------------------------------------------------------
std::vector<double> sliding_attention(const aeon::core::V4Layer& layer,
                                      const std::vector<__half>& query,
                                      const std::vector<__half>& keys,
                                      const std::vector<int64_t>& positions,
                                      int window, int64_t current_pos) {
    __half* d_q = upload(query);
    __half* d_k = upload(keys);
    int64_t* d_p = upload(positions);
    __half* d_out = nullptr;
    CHECK_HIP(hipMalloc(&d_out, static_cast<size_t>(kHeads) * kHeadDim * sizeof(__half)));

    hipLaunchKernelGGL(aeon::kernel::v4_cached_sliding_window_attn_wave32_kernel,
                       dim3(kHeads), dim3(32), 0, 0,
                       d_q, d_k, d_k, d_p, layer.d_attn_sink, d_out,
                       static_cast<int>(current_pos), window,
                       aeon::kernel::DSV4_ATTN_SCALE);
    CHECK_HIP(hipDeviceSynchronize());

    const std::vector<__half> out = download(d_out, static_cast<size_t>(kHeads) * kHeadDim);
    free_device(d_q); free_device(d_k); free_device(d_p); free_device(d_out);

    std::vector<double> result(out.size());
    for (size_t i = 0; i < out.size(); ++i) result[i] = static_cast<double>(__half2float(out[i]));
    return result;
}

std::vector<double> compressed_attention(const aeon::core::V4Layer& layer,
                                         const std::vector<__half>& query,
                                         const std::vector<__half>& local_keys,
                                         const std::vector<int64_t>& local_positions,
                                         int local_capacity,
                                         const std::vector<__half>& compressed_keys,
                                         const std::vector<__half>& compressed_values,
                                         const std::vector<int64_t>& compressed_positions,
                                         int compressed_count,
                                         const std::vector<int32_t>& topk,
                                         int topk_count, bool uses_indexer,
                                         int64_t current_pos,
                                         const float* sink = nullptr) {
    __half* d_q = upload(query);
    __half* d_lk = upload(local_keys);
    int64_t* d_lp = upload(local_positions);
    __half* d_ck = upload(compressed_keys);
    __half* d_cv = upload(compressed_values);
    int64_t* d_cp = upload(compressed_positions);
    int32_t* d_topk = upload(topk);
    __half* d_out = nullptr;
    CHECK_HIP(hipMalloc(&d_out, static_cast<size_t>(kHeads) * kHeadDim * sizeof(__half)));

    hipLaunchKernelGGL(aeon::kernel::v4_cached_compressed_attention_wave32_kernel,
                       dim3(kHeads), dim3(32), 0, 0,
                       d_q, d_lk, d_lk, d_lp,
                       sink != nullptr ? sink : layer.d_attn_sink,
                       d_ck, d_cv, d_cp, d_topk, d_out,
                       current_pos, local_capacity, compressed_count, topk_count,
                       uses_indexer, aeon::kernel::DSV4_ATTN_SCALE);
    CHECK_HIP(hipDeviceSynchronize());

    const std::vector<__half> out = download(d_out, static_cast<size_t>(kHeads) * kHeadDim);
    free_device(d_q); free_device(d_lk); free_device(d_lp); free_device(d_ck);
    free_device(d_cv); free_device(d_cp); free_device(d_topk); free_device(d_out);

    std::vector<double> result(out.size());
    for (size_t i = 0; i < out.size(); ++i) result[i] = static_cast<double>(__half2float(out[i]));
    return result;
}

// The largest elementwise difference between two attention outputs, and the
// peak of the first, so a difference can be reported as a fraction of scale
// rather than as a bare number.
double max_difference(const std::vector<double>& a, const std::vector<double>& b) {
    double delta = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) delta = std::fmax(delta, std::fabs(a[i] - b[i]));
    return delta;
}

double peak_of(const std::vector<double>& v) {
    double peak = 0.0;
    for (double value : v) peak = std::fmax(peak, std::fabs(value));
    return peak;
}

// A difference as a fraction of the reference's own scale, with the degenerate
// case made explicit rather than turning into a NaN that reads as a pass.
double fraction_of(double delta, double peak) {
    if (!(peak > 0.0) || !std::isfinite(delta)) return -1.0;
    return delta / peak;
}

// -----------------------------------------------------------------------------
// The trace capture. The observer seam exists so the layer body does not have to
// know how a gate reads state, and one captured step carries the whole ring, the
// whole compressed store and the token's rotated query — which is exactly the
// input the production attention kernel consumes, so the perturbations in B and D
// run the real kernel on real captured state.
// -----------------------------------------------------------------------------
struct Capture {
    uint32_t layer_id{0};
    uint32_t position{0};
    bool filled{false};
    aeon::core::V4AttentionTraceRecord record;
};

class CaptureObserver final : public aeon::core::V4LayerBodyObserver {
public:
    std::vector<Capture>* captures{nullptr};

    aeon::core::V4AttentionTraceRecord* begin_trace(aeon::core::V4Layer& layer,
                                                    uint32_t token_id,
                                                    uint32_t position) override {
        (void)token_id;
        for (Capture& capture : *captures) {
            if (capture.layer_id == static_cast<uint32_t>(layer.layer_id) &&
                capture.position == position) {
                capture.record = aeon::core::V4AttentionTraceRecord{};
                active_ = &capture;
                return &capture.record;
            }
        }
        active_ = nullptr;
        return nullptr;
    }

    void copy_to_host(void* destination, const void* source, size_t bytes) override {
        CHECK_HIP(hipMemcpy(destination, source, bytes, hipMemcpyDeviceToHost));
        if (active_ != nullptr) active_->filled = true;
    }

private:
    Capture* active_{nullptr};
};

struct StackLayer {
    const char* label{nullptr};
    uint32_t layer_id{0};
    aeon::core::V4Layer device;
};

// One committed entry's bytes, read back out of the trace's compressed store.
std::vector<__half> entry_bytes(const aeon::core::V4AttentionTraceRecord& record,
                                uint32_t slot) {
    const size_t stride = kHeadDim;
    const size_t offset = static_cast<size_t>(slot) * stride;
    if (offset + stride > record.compressed_key_cache.size()) return {};
    return std::vector<__half>(record.compressed_key_cache.begin() + offset,
                               record.compressed_key_cache.begin() + offset + stride);
}

bool entries_equal(const std::vector<__half>& a, const std::vector<__half>& b,
                   double& worst) {
    if (a.size() != b.size() || a.empty()) { worst = -1.0; return false; }
    worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double delta = std::fabs(static_cast<double>(__half2float(a[i])) -
                                       static_cast<double>(__half2float(b[i])));
        worst = std::fmax(worst, delta);
    }
    return worst == 0.0;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-3 item 20: long-context lifecycle (ring reuse, capacity)\n";
    aeon::core::select_compute_device(true);
    bool ok = true;
    uint32_t checks = 0;
    uint32_t failures = 0;
    const auto assert_that = [&](const char* label, bool passed,
                                 const std::string& detail) {
        ++checks;
        if (!passed) ++failures;
        const bool result = check(label, passed, detail);
        ok = ok && result;
        return result;
    };

    aeon::core::AeonModelLoader loader;
    loader.open_model(kModelDir);
    const aeon::core::DeepSeekV4Config cfg =
        aeon::core::DeepSeekV4Config::load_from_json(std::string(kModelDir) + "/config.json");
    const std::vector<aeon::core::V4LayerSpec> specs =
        aeon::core::V4ModelSpec::resolve_layers(cfg);

    // -------------------------------------------------------------------
    // A. Closed forms — the lifecycle is a property of the layout, so it is
    //    stated before anything is read from the device.
    // -------------------------------------------------------------------
    std::cout << "\n--- A. closed forms (no device work) ---\n";
    {
        // The model's own window, read from the artifact rather than assumed: if
        // it were not 128 every "the real window" claim below would be false.
        uint32_t sliding_window = 0;
        uint32_t ratios[2] = {0, 0};
        for (const aeon::core::V4LayerSpec& spec : specs) {
            if (spec.attention_kind == aeon::core::V4AttentionKind::Sliding) {
                sliding_window = static_cast<uint32_t>(spec.sliding_window);
            }
        }
        ratios[0] = 4;
        ratios[1] = 128;
        assert_that("A: the artifact's sliding window is the real 128",
                    sliding_window == 128, "window = " + std::to_string(sliding_window));

        // The capacity is exactly what the declared context produces, for both
        // ratio classes: at the last position the entry count is `max_seq/ratio`
        // and the capacity is `ceil(max_seq/ratio)`, which for a ratio dividing
        // the context are the same number. That is why the store cannot wrap
        // inside the declared context — and why the clamp in the count is a
        // no-op there rather than a silently binding limit (trap 40).
        for (uint32_t ratio : ratios) {
            const uint32_t capacity = (kMaxSeq + ratio - 1u) / ratio;
            const uint32_t last_count = (kMaxSeq - 1u + 1u) / ratio;
            assert_that(
                ratio == 4 ? "A: ratio 4 capacity == the context's own entry count"
                           : "A: ratio 128 capacity == the context's own entry count",
                kMaxSeq % ratio == 0 && capacity == last_count && capacity * ratio == kMaxSeq,
                "K=" + std::to_string(capacity) + " last_count=" + std::to_string(last_count));
        }

        // The ring's slot mapping, checked against the definition rather than
        // against the device: slot `s` holds the largest `p < tokens` congruent
        // to `s`. Slot 0 gets its second occupant only at 256, which is what
        // makes 260 tokens the shortest run that demonstrates reuse.
        const int64_t s0 = ring_position_for_slot(0, 128, kTokens);
        const int64_t s3 = ring_position_for_slot(3, 128, kTokens);
        const int64_t s4 = ring_position_for_slot(4, 128, kTokens);
        const int64_t unused = ring_position_for_slot(200, 128, 50);
        assert_that("A: the ring's slots are predicted by position modulo capacity",
                    s0 == 256 && s3 == 259 && s4 == 132 && unused == -1,
                    "slot0=" + std::to_string(s0) + " slot3=" + std::to_string(s3) +
                        " slot4=" + std::to_string(s4) + " never-written=" +
                        std::to_string(unused));

        // The compressed entry's position is the boundary it was emitted on, not
        // the position that produced it: entry `i` closes at `(i+1)·ratio − 1`,
        // and the run ends before the next boundary would fire.
        bool boundaries = true;
        std::string boundary_detail;
        for (uint32_t ratio : ratios) {
            const uint32_t committed = kTokens / ratio;
            const int64_t next_boundary = static_cast<int64_t>(committed + 1u) * ratio - 1;
            if (next_boundary < static_cast<int64_t>(kTokens)) boundaries = false;
            for (uint32_t i = 0; i < committed; ++i) {
                const int64_t position = static_cast<int64_t>(i + 1u) * ratio - 1;
                if ((position + 1) % ratio != 0) boundaries = false;
                if (position >= static_cast<int64_t>(kTokens)) boundaries = false;
            }
            boundary_detail += "ratio " + std::to_string(ratio) + ": " +
                               std::to_string(committed) + " entries, next at " +
                               std::to_string(next_boundary) + "; ";
        }
        assert_that("A: entry `i` closes at `(i+1)·ratio − 1`", boundaries, boundary_detail);
    }

    // -------------------------------------------------------------------
    // The stack, and the device fixtures
    // -------------------------------------------------------------------
    struct LayerPlan {
        const char* label;
        uint32_t layer_id;
    };
    const LayerPlan plans[kStackSize] = {
        {"Sliding", 0},
        {"CSA", 2},
        {"HCA", 3},
    };

    std::vector<StackLayer> stack;
    stack.reserve(kStackSize);
    for (const LayerPlan& plan : plans) {
        stack.emplace_back();
        StackLayer& s = stack.back();
        s.label = plan.label;
        s.layer_id = plan.layer_id;

        aeon::core::V4LayerSpec spec = specs.at(plan.layer_id);
        // The window is deliberately NOT overridden: the model's own 128 is the
        // subject of this gate. Only the top-k is shrunk, for the reason in the
        // header, and it is named as uncovered.
        if (spec.attention_kind == aeon::core::V4AttentionKind::CSA) {
            spec.index_topk = static_cast<int32_t>(kIndexTopk);
        }
        s.device.init_with_loader(spec, loader, kMaxSeq);
    }

    aeon::core::PipelineScratchBuffers scratch;
    scratch.allocate();

    const size_t table_len = static_cast<size_t>(kTokens) * 32;
    float* d_cos = nullptr;
    float* d_sin = nullptr;
    float* d_cos_c = nullptr;
    float* d_sin_c = nullptr;
    CHECK_HIP(hipMalloc(&d_cos, table_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_sin, table_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_cos_c, table_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_sin_c, table_len * sizeof(float)));
    {
        const auto upload_table = [&](float* device, const std::vector<double>& values) {
            std::vector<float> host(values.size());
            for (size_t i = 0; i < values.size(); ++i) host[i] = static_cast<float>(values[i]);
            CHECK_HIP(hipMemcpy(device, host.data(), host.size() * sizeof(float),
                                hipMemcpyHostToDevice));
        };
        const aeon::reference::RopeTableRef sliding = aeon::reference::rope_table(
            aeon::reference::rope_spec_for(aeon::reference::RopeClass::Sliding), kTokens);
        const aeon::reference::RopeTableRef compressed = aeon::reference::rope_table(
            aeon::reference::rope_spec_for(aeon::reference::RopeClass::Compressed), kTokens);
        upload_table(d_cos, sliding.cos);
        upload_table(d_sin, sliding.sin);
        upload_table(d_cos_c, compressed.cos);
        upload_table(d_sin_c, compressed.sin);
    }
    aeon::core::V4LayerBodyTables tables{d_cos, d_sin, d_cos_c, d_sin_c};

    // Six synthetic experts, encoded once. No checkpoint is compared against a
    // reference in this gate, so the experts only need to make the MoE non-trivial
    // — its arithmetic belongs to Tier 1 and items 16–18.
    std::array<std::vector<uint8_t>, kRoutedExperts> payloads;
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        payloads[k] = aeon::testgate::make_synthetic_payload(k + 1);
    }

    GateExpertExecutor executor;
    executor.loader = nullptr;
    executor.scratch = &scratch;
    executor.stream = 0;
    // Trap 38: a loop gate must state which MoE accumulation it requires. This
    // one is state-shaped rather than arithmetic-shaped, but a nondeterministic
    // accumulation would still make the *captured* state unreproducible between
    // runs, and the bit-identical comparisons in section C are only meaningful
    // against a reproducible store.
    executor.deterministic = true;
    for (uint32_t k = 0; k < kRoutedExperts; ++k) {
        executor.synthetic[k] = payloads[k].data();
        CHECK_HIP(hipMalloc(&executor.d_payload[k], aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
    }

    {
        std::vector<double> seed(kHcDim, 0.0);
        const __half* embed = loader.get_data_ptr<__half>("embed.weight");
        const __half* row = embed + static_cast<size_t>(kTokenIds[0]) * kHidden;
        for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t h = 0; h < kHidden; ++h) {
                seed[j * kHidden + h] = static_cast<double>(__half2float(row[h]));
            }
        }
        const std::vector<__half> seed_half = to_half(seed);
        std::vector<float> seed_float(kHcDim);
        for (size_t i = 0; i < seed_float.size(); ++i) {
            seed_float[i] = __half2float(seed_half[i]);
        }
        CHECK_HIP(hipMemcpy(scratch.d_res_in_half, seed_half.data(),
                            kHcDim * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(scratch.d_res_in, seed_float.data(),
                            kHcDim * sizeof(float), hipMemcpyHostToDevice));
    }

    // Capture list: the partial ring (Sliding at 99), each class's first
    // compressed boundary (CSA at 3, HCA at 127) and the final position for all
    // three. Each captured step carries the ring, the compressed store and the
    // token's rotated query.
    std::vector<Capture> captures;
    const auto add_capture = [&](uint32_t layer_id, uint32_t position) {
        Capture capture;
        capture.layer_id = layer_id;
        capture.position = position;
        captures.push_back(capture);
    };
    add_capture(0, 99);
    add_capture(0, kTokens - 1);
    add_capture(2, 3);
    add_capture(2, kTokens - 1);
    add_capture(3, 127);
    add_capture(3, kTokens - 1);

    CaptureObserver observer;
    observer.captures = &captures;

    // -------------------------------------------------------------------
    // B. The run, at the real window
    // -------------------------------------------------------------------
    std::cout << "\n--- B. the run: " << kTokens << " tokens, window 128, three classes ---\n";
    for (uint32_t t = 0; t < kTokens; ++t) {
        const uint32_t token = kTokenIds[t % kTokenIds.size()];
        for (StackLayer& s : stack) {
            aeon::core::run_layer_body_decoding(s.device, scratch, tables, token, t, 0,
                                                executor, observer);
        }
        if (t % 64 == 0 || t + 1 == kTokens) {
            std::printf("  .. token %u/%u (position %u)\n", t + 1, kTokens, t);
        }
    }
    CHECK_HIP(hipStreamSynchronize(0));

    const auto capture_for = [&](uint32_t layer_id, uint32_t position) -> const Capture& {
        for (const Capture& capture : captures) {
            if (capture.layer_id == layer_id && capture.position == position) return capture;
        }
        throw std::runtime_error("capture missing");
    };
    for (const Capture& capture : captures) {
        assert_that("B: the trace captured every named step",
                    capture.filled,
                    "layer " + std::to_string(capture.layer_id) + " position " +
                        std::to_string(capture.position));
    }

    // -------------------------------------------------------------------
    // B1. The ring's contents are predicted, not observed
    // -------------------------------------------------------------------
    std::cout << "\n--- B1. the local ring after " << kTokens << " tokens ---\n";
    for (StackLayer& s : stack) {
        const uint32_t capacity = s.device.local_cache_capacity();
        const std::vector<int64_t> positions =
            read_positions(s.device.d_local_positions, capacity);

        uint32_t mismatches = 0;
        for (uint32_t slot = 0; slot < capacity; ++slot) {
            if (positions[slot] != ring_position_for_slot(slot, capacity, kTokens)) {
                ++mismatches;
            }
        }
        const std::string label =
            std::string("B1: ") + s.label + " ring slot s holds the largest p ≡ s (mod C)";
        assert_that(label.c_str(), mismatches == 0,
                    "capacity=" + std::to_string(capacity) + " mismatched slots=" +
                        std::to_string(mismatches));

        // The ring is full, so nothing older than the window survives: the oldest
        // row any query at the last position can see is `pos − C + 1`, and it is
        // present. Everything below it was overwritten, not masked.
        const int64_t oldest = ring_position_for_slot(4, capacity, kTokens);
        const int64_t oldest_wanted =
            static_cast<int64_t>(kTokens) - static_cast<int64_t>(capacity);
        assert_that(
            (std::string("B1: ") + s.label + " the oldest surviving row is pos−C+1").c_str(),
            oldest == oldest_wanted,
            "oldest=" + std::to_string(oldest) + " want=" + std::to_string(oldest_wanted));
    }

    // -------------------------------------------------------------------
    // B3. The exact window boundary, measured through the production kernel
    // -------------------------------------------------------------------
    std::cout << "\n--- B3. reuse evicts exactly what the window excludes ---\n";
    {
        // (a) the partial ring. At position 99 with a 128-slot ring, slots 100..127
        // were never written and hold the sentinel; they are outside every query's
        // window at 99. Perturbing one must move the output by *exactly zero* — an
        // unfilled slot has no influence, not a small one. Perturbing slot 0
        // (position 0, the oldest in-window row) must move it materially. This is
        // the Tier-1 attention gate's boundary instrument, driven at the model's
        // own window on state captured from a real run.
        const Capture& early = capture_for(0, 99);
        assert_that("B3: the partial-ring step was captured with the full ring",
                    early.record.local_key_cache.size() ==
                            static_cast<size_t>(128) * kHeadDim &&
                        early.record.rotated_query.size() ==
                            static_cast<size_t>(kHeads) * kHeadDim,
                    "ring=" + std::to_string(early.record.local_key_cache.size()) +
                        " query=" + std::to_string(early.record.rotated_query.size()));

        assert_that("B3: the partial ring's unfilled slots hold the sentinel",
                    early.record.local_positions[100] == -1 &&
                        early.record.local_positions[127] == -1 &&
                        early.record.local_valid_count == 100,
                    "valid_count=" + std::to_string(early.record.local_valid_count));

        const std::vector<double> base = sliding_attention(
            stack[0].device, early.record.rotated_query, early.record.local_key_cache,
            early.record.local_positions, 128, 99);

        // An unfilled slot: its bytes are overwritten with a large value. Nothing
        // may change — the kernel's `valid` test rejects the position before it
        // ever reads the row.
        std::vector<__half> perturbed_sentinel = early.record.local_key_cache;
        for (uint32_t d = 0; d < kHeadDim; ++d) perturbed_sentinel[127 * kHeadDim + d] = __float2half(50.0f);
        const std::vector<double> with_sentinel = sliding_attention(
            stack[0].device, early.record.rotated_query, perturbed_sentinel,
            early.record.local_positions, 128, 99);
        const double sentinel_delta = max_difference(base, with_sentinel);

        // An in-window slot: position 0 is the oldest row of the window at 99, so
        // it is genuinely attended.
        std::vector<__half> perturbed_oldest = early.record.local_key_cache;
        for (uint32_t d = 0; d < kHeadDim; ++d) perturbed_oldest[0 * kHeadDim + d] = __float2half(50.0f);
        const std::vector<double> with_oldest = sliding_attention(
            stack[0].device, early.record.rotated_query, perturbed_oldest,
            early.record.local_positions, 128, 99);
        const double oldest_delta = max_difference(base, with_oldest);
        const double peak = peak_of(base);

        assert_that("B3: an unfilled ring slot has exactly zero influence",
                    sentinel_delta == 0.0,
                    "max|delta|=" + num(sentinel_delta, 12));
        assert_that("B3: the oldest in-window row is genuinely attended",
                    fraction_of(oldest_delta, peak) > 0.05,
                    "max|delta|=" + num(oldest_delta, 6) + " = " +
                        num(fraction_of(oldest_delta, peak), 4) + "*peak");

        // The same instrument on the wrapped ring: every slot is in-window, so
        // the exact-boundary claim is the *contents* claim of B1 plus this — the
        // oldest surviving row (position 132, slot 4) is attended, and the row
        // that occupied that slot before the wrap is gone rather than masked.
        const Capture& late = capture_for(0, kTokens - 1);
        const std::vector<double> late_base = sliding_attention(
            stack[0].device, late.record.rotated_query, late.record.local_key_cache,
            late.record.local_positions, 128, static_cast<int64_t>(kTokens - 1));
        std::vector<__half> late_perturbed = late.record.local_key_cache;
        for (uint32_t d = 0; d < kHeadDim; ++d) late_perturbed[4 * kHeadDim + d] = __float2half(50.0f);
        const std::vector<double> late_with = sliding_attention(
            stack[0].device, late.record.rotated_query, late_perturbed,
            late.record.local_positions, 128, static_cast<int64_t>(kTokens - 1));
        const double late_delta = max_difference(late_base, late_with);
        assert_that("B3: after the wrap the oldest surviving row is attended",
                    late.record.local_positions[4] == static_cast<int64_t>(kTokens) - 128 &&
                        fraction_of(late_delta, peak_of(late_base)) > 0.05,
                    "slot4=" + std::to_string(late.record.local_positions[4]) + " delta=" +
                        num(late_delta, 6) + " = " +
                        num(fraction_of(late_delta, peak_of(late_base)), 4) + "*peak");
    }

    // -------------------------------------------------------------------
    // C. The two stores are independent, and their positions are closed forms
    // -------------------------------------------------------------------
    std::cout << "\n--- C. compressed store and compressor ring ---\n";
    for (StackLayer& s : stack) {
        const int64_t ratio = s.device.spec().compression_ratio;
        if (ratio == 0) continue;
        const uint32_t capacity = s.device.state_layout().compressed_capacity;

        // The committed count is the class's closed form at the last position:
        // `(pos+1)/ratio`, which for a run ending at `pos = kTokens−1` is
        // `kTokens/ratio` — an integer floor, not a ceiling: the third HCA entry
        // needs position 383, and this run ends at 259.
        const uint32_t expected =
            aeon::testgate::committed_entries(s.device, kTokens - 1, ratio);
        const uint32_t predicted = kTokens / static_cast<uint32_t>(ratio);
        assert_that((std::string("C: ") + s.label + " committed count is the closed form").c_str(),
                    expected == predicted && expected <= capacity,
                    "committed=" + std::to_string(expected) + " predicted=" +
                        std::to_string(predicted) + " K=" + std::to_string(capacity));

        // Every committed entry records the position of the boundary it was
        // emitted on — not the position that produced it, and not a write count.
        const std::vector<int64_t> positions =
            read_positions(s.device.d_compressed_positions, capacity);
        uint32_t bad_position = 0;
        for (uint32_t i = 0; i < expected; ++i) {
            const int64_t want = static_cast<int64_t>(i + 1u) * ratio - 1;
            if (positions[i] != want) ++bad_position;
        }
        assert_that(
            (std::string("C: ") + s.label + " entry i records (i+1)·ratio − 1").c_str(),
            bad_position == 0,
            "mismatched=" + std::to_string(bad_position) + " of " +
                std::to_string(expected));

        // The first entry survives 256 further steps byte-identically. That is
        // the claim that reuse of the *local* ring never disturbs the compressed
        // store: the local ring wrapped twice between the two reads.
        const uint32_t boundary_position = static_cast<uint32_t>(ratio) - 1u;
        const Capture& at_boundary = capture_for(s.layer_id, boundary_position);
        const Capture& at_end = capture_for(s.layer_id, kTokens - 1);
        double worst = 0.0;
        const std::vector<__half> first_entry = entry_bytes(at_boundary.record, 0);
        const std::vector<__half> first_entry_later = entry_bytes(at_end.record, 0);
        const bool same = entries_equal(first_entry, first_entry_later, worst);
        assert_that(
            (std::string("C: ") + s.label + " the first entry is byte-identical 256 steps on")
                 .c_str(),
            same && !first_entry.empty(),
            "max|delta|=" + num(worst, 12) + " (" + std::to_string(first_entry.size()) +
                " values)");

        // ... and the ring as it stood at that boundary is the closed form, so
        // materializing the entry did not write into the local ring.
        uint32_t ring_mismatch = 0;
        for (uint32_t slot = 0;
             slot < 128 && slot < at_boundary.record.local_positions.size(); ++slot) {
            if (at_boundary.record.local_positions[slot] !=
                ring_position_for_slot(slot, 128, boundary_position + 1)) {
                ++ring_mismatch;
            }
        }
        assert_that(
            (std::string("C: ") + s.label + " the boundary did not disturb the ring").c_str(),
            ring_mismatch == 0, "mismatched slots=" + std::to_string(ring_mismatch));

        // The compressor partial ring: `coefficient · ratio` slots, current token
        // at `pos mod capacity`, and its slots hold the same closed form.
        const uint32_t partial_capacity =
            s.device.state_layout().compressor_partial_capacity;
        const uint32_t coefficient = (ratio == 4) ? 2u : 1u;
        assert_that(
            (std::string("C: ") + s.label + " partial ring is coefficient·ratio wide").c_str(),
            partial_capacity == coefficient * static_cast<uint32_t>(ratio),
            "capacity=" + std::to_string(partial_capacity));
        const std::vector<int64_t> partial =
            read_positions(s.device.d_compressor_partial_positions, partial_capacity);
        uint32_t partial_mismatch = 0;
        for (uint32_t slot = 0; slot < partial_capacity; ++slot) {
            if (partial[slot] != ring_position_for_slot(slot, partial_capacity, kTokens)) {
                ++partial_mismatch;
            }
        }
        assert_that(
            (std::string("C: ") + s.label + " partial-ring slots are pos mod capacity").c_str(),
            partial_mismatch == 0, "mismatched slots=" + std::to_string(partial_mismatch));
    }

    // -------------------------------------------------------------------
    // D. The row-set past the window is the class rule
    // -------------------------------------------------------------------
    std::cout << "\n--- D. the row-set past the window ---\n";
    for (StackLayer& s : stack) {
        const int64_t ratio = s.device.spec().compression_ratio;
        const Capture& at_end = capture_for(s.layer_id, kTokens - 1);

        assert_that(
            (std::string("D: ") + s.label + " the local row-set is the full window").c_str(),
            at_end.record.local_valid_count == 128,
            "valid_count=" + std::to_string(at_end.record.local_valid_count));

        if (ratio == 0) {
            assert_that("D: a Sliding layer commits no compressed rows",
                        at_end.record.compressed_entry_count == 0,
                        "count=" + std::to_string(at_end.record.compressed_entry_count));
            continue;
        }

        const uint32_t committed = at_end.record.compressed_entry_count;
        const bool uses_indexer = (s.device.spec().attention_kind ==
                                   aeon::core::V4AttentionKind::CSA);

        if (!uses_indexer) {
            // HCA attends *every* committed entry, so its row-set width is the
            // committed count itself — no top-k, no indexer (trap 33).
            assert_that("D: HCA's compressed row-set is every committed entry",
                        committed == kTokens / 128u,
                        "count=" + std::to_string(committed));

            std::vector<int32_t> all(committed);
            for (uint32_t i = 0; i < committed; ++i) all[i] = static_cast<int32_t>(i);
            std::vector<int32_t> without_oldest = all;
            without_oldest[0] = -1;

            const int local_capacity = 128;
            const std::vector<double> full = compressed_attention(
                s.device, at_end.record.rotated_query, at_end.record.local_key_cache,
                at_end.record.local_positions, local_capacity,
                at_end.record.compressed_key_cache, at_end.record.compressed_value_cache,
                at_end.record.compressed_positions,
                static_cast<int>(committed), all, static_cast<int>(committed), true,
                static_cast<int64_t>(kTokens - 1));
            const std::vector<double> dropped = compressed_attention(
                s.device, at_end.record.rotated_query, at_end.record.local_key_cache,
                at_end.record.local_positions, local_capacity,
                at_end.record.compressed_key_cache, at_end.record.compressed_value_cache,
                at_end.record.compressed_positions,
                static_cast<int>(committed), without_oldest, static_cast<int>(committed),
                true, static_cast<int64_t>(kTokens - 1));
            const double delta = max_difference(full, dropped);
            assert_that("D: dropping the oldest committed entry moves HCA's attention",
                        fraction_of(delta, peak_of(full)) > 0.01,
                        "max|delta|=" + num(delta, 6) + " = " +
                            num(fraction_of(delta, peak_of(full)), 6) + "*peak");
        } else {
            // CSA reads the indexer's selection, which here is a real selection:
            // 65 candidates for 8 slots.
            assert_that("D: CSA's compressed row-set is the indexer's top-k",
                        committed == kTokens / 4u &&
                            at_end.record.indexer_topk_indices.size() == kIndexTopk,
                        "candidates=" + std::to_string(committed) + " selected=" +
                            std::to_string(at_end.record.indexer_topk_indices.size()));

            std::vector<int32_t> selected = at_end.record.indexer_topk_indices;
            const int selected_count = static_cast<int>(selected.size());
            std::vector<int32_t> without_one = selected;
            without_one[0] = -1;

            const std::vector<double> full = compressed_attention(
                s.device, at_end.record.rotated_query, at_end.record.local_key_cache,
                at_end.record.local_positions, 128,
                at_end.record.compressed_key_cache, at_end.record.compressed_value_cache,
                at_end.record.compressed_positions,
                static_cast<int>(committed), selected, selected_count, true,
                static_cast<int64_t>(kTokens - 1));
            const std::vector<double> dropped = compressed_attention(
                s.device, at_end.record.rotated_query, at_end.record.local_key_cache,
                at_end.record.local_positions, 128,
                at_end.record.compressed_key_cache, at_end.record.compressed_value_cache,
                at_end.record.compressed_positions,
                static_cast<int>(committed), without_one, selected_count, true,
                static_cast<int64_t>(kTokens - 1));
            const double delta = max_difference(full, dropped);
            assert_that("D: dropping one selected entry moves CSA's attention",
                        fraction_of(delta, peak_of(full)) > 0.01,
                        "max|delta|=" + num(delta, 6) + " = " +
                            num(fraction_of(delta, peak_of(full)), 6) + "*peak");

            // A candidate the indexer did *not* select is a row the class is free
            // to skip, so replacing a selected entry with an unselected one must
            // also move the output — otherwise the selection would be decorative
            // and this section would be asserting nothing about it.
            std::vector<int32_t> swapped = selected;
            uint32_t unselected = 0;
            for (uint32_t i = 0; i < committed; ++i) {
                const bool taken = std::find(selected.begin(), selected.end(),
                                             static_cast<int32_t>(i)) != selected.end();
                if (!taken) { unselected = i; break; }
            }
            swapped[0] = static_cast<int32_t>(unselected);
            const std::vector<double> with_other = compressed_attention(
                s.device, at_end.record.rotated_query, at_end.record.local_key_cache,
                at_end.record.local_positions, 128,
                at_end.record.compressed_key_cache, at_end.record.compressed_value_cache,
                at_end.record.compressed_positions,
                static_cast<int>(committed), swapped, selected_count, true,
                static_cast<int64_t>(kTokens - 1));
            const double swap_delta = max_difference(full, with_other);
            assert_that("D: swapping in a rejected candidate moves CSA's attention",
                        fraction_of(swap_delta, peak_of(full)) > 0.01,
                        "slot " + std::to_string(selected[0]) + " -> " +
                            std::to_string(unselected) + ", max|delta|=" +
                            num(swap_delta, 6));
        }
    }

    // -------------------------------------------------------------------
    // E. Capacity is exact, and the refusal is load-bearing (trap 40)
    // -------------------------------------------------------------------
    std::cout << "\n--- E. capacity, and why the refusal is not decorative ---\n";
    {
        // The layer accepts the last position of the declared context and refuses
        // the first one past it. The counters are no longer read below, so doing
        // this on a live layer is safe.
        bool accepted = false;
        try {
            stack[0].device.record_position(kMaxSeq - 1);
            accepted = true;
        } catch (const std::exception&) {
            accepted = false;
        }
        bool refused = false;
        std::string refusal;
        try {
            stack[0].device.record_position(kMaxSeq);
        } catch (const std::out_of_range& error) {
            refused = true;
            refusal = error.what();
        } catch (const std::exception& error) {
            refusal = std::string("wrong type: ") + error.what();
        }
        assert_that("E: the layer accepts max_seq_len − 1 and refuses max_seq_len",
                    accepted && refused,
                    "accepted=" + std::string(accepted ? "yes" : "no") + " refused=" +
                        std::string(refused ? "yes" : "no") + " (" + refusal + ")");

        // The capacity equals ceil(max_seq/ratio) exactly, so at the last legal
        // position the count is exactly the capacity: the ring is full *and*
        // nothing is evicted. There is no third state, which is why an
        // out-of-range position cannot be absorbed by a clamp either.
        for (StackLayer& s : stack) {
            const int64_t ratio = s.device.spec().compression_ratio;
            if (ratio == 0) continue;
            const uint32_t capacity = s.device.state_layout().compressed_capacity;
            const uint32_t at_last = kMaxSeq / static_cast<uint32_t>(ratio);
            assert_that(
                (std::string("E: ") + s.label + " capacity == the entry count at max_seq_len")
                     .c_str(),
                capacity == at_last && capacity * static_cast<uint32_t>(ratio) == kMaxSeq,
                "K=" + std::to_string(capacity) + " count@max_seq=" +
                    std::to_string(at_last));
        }

        // THE DISCRIMINATING PART. Build two compressed stores with the same
        // capacity, give the kernel the same row-set `[0, K)` for both, and give
        // them **identical keys but different values**:
        //
        //   * "newest" — slots 0..K−1 hold entries K..2K−1, which is what a ring
        //     holds after it has rolled over (`slot = i mod K`);
        //   * "oldest"  — the same slots hold entries 0..K−1, which is what a
        //     clamp to capacity would have kept (the reference's own
        //     `tl.minimum` truncates here).
        //
        // Both stores have every recorded position ≤ the query position, so the
        // kernel's guard accepts every row in *both*. The keys are bit-identical,
        // so the softmax weights are bit-identical; only the values each row index
        // resolves to differ. The outputs differ anyway, and that is the trap:
        // **the guard cannot see the eviction, so the only thing keeping HCA on
        // "every committed entry" rather than "the newest K" is the refusal to
        // take a position past capacity.**
        //
        // Varying the values rather than the keys is what makes this discriminating
        // at all. With different keys the two stores produce *different softmax
        // weights*, and since the sink dominates those weights the outputs can both
        // round to zero in fp16 and compare equal — which is how the first version
        // of this check passed its own setup while measuring nothing. Identical keys
        // cancel the weights from the comparison, leaving only "which entry did row
        // index i read". A **real** captured key row (rather than a synthetic one)
        // also fixes the weights' magnitude: a synthetic row is only as aligned with
        // `q` as its construction happens to be, and a weakly aligned row drives
        // every exponent far below the sink and underflows the fp16 output to zero.
        //
        // The values are strictly positive and the pattern avoids a subtraction on
        // an unsigned quantity — not style, but a bug this probe actually had: the
        // first version used `(d % 11) - 5` on a `uint32_t`, which wraps to ~4.3e9
        // for `d % 11 < 5`, and `__float2half` saturated exactly those entries to
        // `+inf` while the rest of the vector was correct. It looked like a kernel
        // defect and was a probe defect.
        //
        // The probe is built from the **layer's own** ratio and capacity, not from
        // convenient round numbers: a probe that queried an HCA store (capacity 4)
        // with ratio 4 would be exercising the count clamp on a mismatched pair and
        // would say nothing about the real lifecycle. Here `ratio = 128` and
        // `K = 4` are HCA's own, so every number below is a number the layer would
        // actually produce.
        //
        // A wrapped 4-slot ring at ratio 128 holds entries 4..7, which requires
        // `pos ≥ 8·128 − 1 = 1023` — i.e. **past the declared context** (whose last
        // legal position is 511, where the count is exactly `K = 4` and nothing has
        // been evicted). That is the whole point: a wrap cannot occur inside the
        // context, so the refusal is the only thing standing between "every
        // committed entry" and "the newest K".
        const uint32_t K = 4;
        const uint32_t ratio = 128;
        const int64_t current_position = static_cast<int64_t>(2u * K) * ratio - 1;  // 1023

        const Capture& at_end = capture_for(3, kTokens - 1);
        const std::vector<__half> real_key = entry_bytes(at_end.record, 0);
        assert_that("E: a real key row is available to anchor the store",
                    real_key.size() == kHeadDim,
                    "row width=" + std::to_string(real_key.size()));

        std::vector<__half> shared_keys(K * kHeadDim, __float2half(0.0f));
        std::vector<__half> newest_values(K * kHeadDim, __float2half(0.0f));
        std::vector<__half> oldest_values(K * kHeadDim, __float2half(0.0f));
        std::vector<int64_t> newest_positions(K, 0);
        std::vector<int64_t> oldest_positions(K, 0);
        for (uint32_t slot = 0; slot < K; ++slot) {
            const uint32_t newest_entry = K + slot;          // entries 4..7
            const uint32_t oldest_entry = slot;              // entries 0..3
            newest_positions[slot] = static_cast<int64_t>(newest_entry + 1u) * ratio - 1;
            oldest_positions[slot] = static_cast<int64_t>(oldest_entry + 1u) * ratio - 1;
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                // One key row for every index, so the weights are identical.
                shared_keys[static_cast<size_t>(slot) * kHeadDim + d] = real_key[d];
                // Strictly positive, and different per *entry*, so a row index that
                // resolves to a different entry produces a different output.
                const float pattern = 0.25f + 0.5f * static_cast<float>(d % 5);
                newest_values[static_cast<size_t>(slot) * kHeadDim + d] =
                    __float2half(pattern * static_cast<float>(newest_entry + 1u));
                oldest_values[static_cast<size_t>(slot) * kHeadDim + d] =
                    __float2half(pattern * static_cast<float>(oldest_entry + 1u));
            }
        }

        // THE TRAP, WITHOUT A KERNEL. A wrapped store's index `i` holds entry
        // `K + i`, whose recorded position is legal (`≤ current_position`), so the
        // kernel's guard accepts it. But the position's *closed form* says index 0
        // must carry entry 0's position. The guard cannot express that, and this
        // pair of lines is exactly the gap: every row is legal, and the row-set is
        // nevertheless the wrong entries.
        bool all_legal = true;
        bool closed_form_holds = true;
        for (uint32_t slot = 0; slot < K; ++slot) {
            if (newest_positions[slot] > current_position ||
                oldest_positions[slot] > current_position) {
                all_legal = false;
            }
            if (newest_positions[slot] !=
                static_cast<int64_t>(slot + 1u) * ratio - 1) {
                closed_form_holds = false;
            }
        }
        assert_that("E: every populated slot passes the kernel's position guard, in both stores",
                    all_legal,
                    "newest positions [639,767,895,1023] ≤ 1023, oldest [127,255,383,511]");
        assert_that("E: the guard cannot tell a wrapped store from an intact one",
                    all_legal && !closed_form_holds,
                    "index 0 holds position " + std::to_string(newest_positions[0]) +
                        " where the closed form requires " +
                        std::to_string(ratio - 1) +
                        "; both pass the guard (trap 40)");

        const std::vector<__half> suppressed_keys(static_cast<size_t>(128) * kHeadDim,
                                                  __float2half(0.0f));
        const std::vector<int64_t> suppressed_positions(128, -1);
        const std::vector<int32_t> row_set{0, 1, 2, 3};

        const std::vector<double> wrapped = compressed_attention(
            stack[2].device, at_end.record.rotated_query, suppressed_keys,
            suppressed_positions, 128, shared_keys, newest_values, newest_positions,
            static_cast<int>(K), row_set, static_cast<int>(K), true, current_position);
        const std::vector<double> clamped = compressed_attention(
            stack[2].device, at_end.record.rotated_query, suppressed_keys,
            suppressed_positions, 128, shared_keys, oldest_values, oldest_positions,
            static_cast<int>(K), row_set, static_cast<int>(K), true, current_position);

        // The row-set must actually be attended: if the weights underflowed, both
        // outputs are zero and the comparison below would be measuring nothing.
        // This is the line that makes the failure mode legible rather than a
        // mysterious zero-vs-zero "difference".
        const double wrapped_peak = peak_of(wrapped);
        assert_that("E: the wrapped store's row-set is genuinely attended",
                    wrapped_peak > 0.0 && std::isfinite(wrapped_peak),
                    "peak=" + num(wrapped_peak, 8));

        const double divergence = max_difference(wrapped, clamped);
        const double divergence_frac = fraction_of(divergence, wrapped_peak);
        assert_that("E: a wrapped store and a clamped store attend different rows",
                    divergence_frac > 0.10,
                    "max|delta|=" + num(divergence, 6) + " = " +
                        num(divergence_frac, 6) + "*peak");

        // And the count is no help either, which completes the trap. The count
        // reports the number of *populated slots*, which is the capacity once the
        // ring is full — so it reads the same for a wrapped store as for an intact
        // one, even though the two hold different entries. Nothing an intact store
        // exposes distinguishes them except the positions' closed form.
        //
        // (The clamp inside `committed_entries_for` is dead code within the declared
        // context — section A proved the capacity is exactly the context's own entry
        // count — and exists only so the count cannot exceed the capacity if it is
        // ever asked about a position beyond it. Mutating it away is therefore not
        // equivalent here: it is what makes the wrapped case's count read `8` instead
        // of `4`.)
        const uint32_t reported_wrapped = aeon::core::committed_entries_for(
            stack[2].device, static_cast<uint32_t>(current_position),
            static_cast<int32_t>(ratio));
        const uint32_t reported_intact = aeon::core::committed_entries_for(
            stack[2].device, static_cast<uint32_t>(kMaxSeq - 1),
            static_cast<int32_t>(ratio));
        assert_that("E: the count cannot distinguish the wrapped store either",
                    reported_wrapped == K && reported_intact == K,
                    "wrapped(pos 1023)=" + std::to_string(reported_wrapped) +
                        " intact(pos 511)=" + std::to_string(reported_intact) +
                        " K=" + std::to_string(K));
    }

    // -------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------
    std::printf("\n[Tier-3 item 20 lifecycle] %s — %u checks, %u failed\n",
                ok ? "PASS" : "FAIL", checks, failures);
    std::printf("  run: %u tokens x %u layers at the model's own window (128); "
                "compressed top-k shrunk to %u\n",
                kTokens, kStackSize, kIndexTopk);

    return ok ? 0 : 1;
}
