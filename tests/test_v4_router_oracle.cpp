// -----------------------------------------------------------------------------
// Tier-1 gate: the MoE router (Step 2.9) — versus an independent fp64 oracle.
//
// The router has two mutually exclusive branches and four things the plan flags
// as easy to get wrong. This gate measures each of them rather than asserting
// the output, because every one of the four produces a plausible-looking result:
//
//   1. the bias is added to the **post-softplus score**, not the logit. A loader
//      that reads `ffn.gate.bias` as a linear bias still routes, just wrongly.
//   2. selection is **flat** — a global top-6 over 256 experts, with no
//      `n_group`/`topk_group` pre-filter (they are absent from the config). A
//      grouped implementation also routes, just differently.
//   3. ties resolve to the **lowest** expert index.
//   4. the stored weight is the **unbiased** score; the bias participates in
//      selection only, never in the weight.
//
// Items 1, 2 and 4 are checked by computing the wrong variant deliberately and
// requiring both that it differs materially and that the kernel follows the
// right one. Item 3 is checked by construction, since random logits never tie.
//
// The hash branch is then replayed against the **artifact's own** `tid2eid`
// table for real token ids, and the table's shape, dtype, and hash/biased layer
// split are verified on disk — the plan's requirement that `tid2eid` be checked
// against our artifact and not against a synthetic table.
//
// The existing `test_moe_router` is not a substitute and is not relied on here:
// its reference (`cpu_moe_router`) lives in the same header as the kernel and
// shares its assumptions, and it never opens the artifact. This gate compares
// against an oracle written from the reference semantics and sharing no code
// with `moe_router.hpp`.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/moe_router.hpp"
#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "infrastructure/core/aeon_loader.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_HIP(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

namespace {

using aeon::reference::ErrorStats;
using aeon::reference::RouterSelection;

constexpr size_t kExperts  = 256;
constexpr size_t kTopK     = 6;
constexpr double kScaling  = 1.5;
constexpr size_t kHashLayers = 3;

// The router kernel computes in fp32 against an fp64 oracle; the only rounding
// is one fp32 divide and one fp32 multiply per weight. This is not an fp16
// tolerance.
constexpr double kTol = 1e-5;

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got, double tol = kTol) {
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-52s SIZE MISMATCH                 FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(s.max_rel) && s.max_rel <= tol;
    std::printf("  %-52s max_abs=%.3e max_rel=%.3e  %s\n",
                label, s.max_abs, s.max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-52s %-24s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

// Positions at which two id vectors differ. Both id *set* and id *position* are
// reported, because the hash branch preserves table column order while the top-k
// branch is score-ordered — a test that only sorted would miss that.
size_t first_difference(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return i;
    }
    return a.size();
}

bool same_set(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    std::vector<int32_t> sa = a, sb = b;
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    return sa == sb;
}

// Groups of 32 experts, top-2 groups: the DeepSeek-V3 `noaux_tc` variant that
// this config does **not** use. Implemented here, labelled, purely so the gate
// can show that the flat selection is distinguishable from it.
RouterSelection grouped_variant(const std::vector<double>& logits,
                                const std::vector<double>& bias,
                                size_t top_k, double scaling,
                                size_t group_size, size_t top_groups) {
    const size_t experts = logits.size();
    const size_t groups = experts / group_size;

    std::vector<double> scores(experts);
    for (size_t e = 0; e < experts; ++e) {
        scores[e] = aeon::reference::router_score(logits[e]) + bias[e];
    }

    // Group score = sum of its top-2 members, as in DeepSeek-V3.
    std::vector<std::pair<double, size_t>> group_scores(groups);
    for (size_t g = 0; g < groups; ++g) {
        const std::vector<double> members(
            scores.begin() + static_cast<long>(g * group_size),
            scores.begin() + static_cast<long>((g + 1) * group_size));
        const std::vector<int32_t> top2 = aeon::reference::topk_indices(members, 2);
        group_scores[g] = {members[static_cast<size_t>(top2[0])] +
                               members[static_cast<size_t>(top2[1])],
                           g};
    }
    std::stable_sort(group_scores.begin(), group_scores.end(),
                     [](const auto& a, const auto& b) {
                         if (a.first != b.first) return a.first > b.first;
                         return a.second < b.second;
                     });

    std::vector<double> masked(experts, -1e30);
    for (size_t i = 0; i < top_groups; ++i) {
        const size_t g = group_scores[i].second;
        for (size_t j = 0; j < group_size; ++j) masked[g * group_size + j] = scores[g * group_size + j];
    }

    RouterSelection out;
    out.ids = aeon::reference::topk_indices(masked, top_k);
    out.weights.resize(top_k);
    double sum = 0.0;
    for (size_t k = 0; k < top_k; ++k) {
        out.weights[k] = aeon::reference::router_score(logits[static_cast<size_t>(out.ids[k])]);
        sum += out.weights[k];
    }
    for (size_t k = 0; k < top_k; ++k) out.weights[k] = out.weights[k] / sum * scaling;
    return out;
}

// Real token ids from `profiling-prompts/first-prompt.jsonl` — a genuine
// formatted DeepSeek-V4 prompt, so the hash branch is exercised at ids that
// actually occur rather than at synthetic ones.
const int32_t kRealTokenIds[] = {
    65106, 295, 4654, 3999, 1192, 260, 6025, 305, 7722, 8739,
    23809, 588, 6252, 34788, 52636, 16
};

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: MoE router (top-k + hash)\n";
    aeon::core::select_compute_device(true);

    bool ok = true;
    aeon::reference::Rng gen(0x1207E8C0ull);

    const size_t tokens = 8;
    // Every section below shares these buffers, so they are sized for the largest
    // token count any section uses (the artifact replay uses 16).
    constexpr size_t kMaxTokens = 32;

    // Device buffers, shared by every section.
    float* d_logits = nullptr;
    float* d_bias = nullptr;
    int64_t* d_table = nullptr;
    int32_t* d_token_ids = nullptr;
    float* d_weights = nullptr;
    int32_t* d_indices = nullptr;
    CHECK_HIP(hipMalloc(&d_logits, kMaxTokens * kExperts * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_bias, kExperts * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_weights, kMaxTokens * kTopK * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_indices, kMaxTokens * kTopK * sizeof(int32_t)));

    auto run = [&](int num_tokens, const int64_t* table, const int32_t* token_ids,
                   const std::vector<float>& logits, const std::vector<float>& bias) {
        CHECK_HIP(hipMemcpy(d_logits, logits.data(), logits.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_bias, bias.data(), bias.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        aeon::kernel::moe_router_kernel<<<dim3(static_cast<unsigned>(num_tokens)),
                                          dim3(64)>>>(
            d_logits, bias.empty() ? nullptr : d_bias, table, token_ids,
            d_weights, d_indices, static_cast<int>(kExperts), static_cast<int>(kTopK),
            static_cast<float>(kScaling), /*renormalize=*/true);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<float> weights(num_tokens * kTopK);
        std::vector<int32_t> indices(num_tokens * kTopK);
        CHECK_HIP(hipMemcpy(weights.data(), d_weights, weights.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(indices.data(), d_indices, indices.size() * sizeof(int32_t),
                            hipMemcpyDeviceToHost));
        return std::make_pair(indices, weights);
    };

    // =======================================================================
    // A. Top-k layer (index >= num_hash_layers): bias, flatness, tie-break
    // =======================================================================
    std::cout << "\n  --- A. top-k layer (layers >= 3) ---\n";
    std::vector<float> logits(tokens * kExperts);
    for (float& x : logits) x = static_cast<float>(gen.symmetric(4.0));
    std::vector<float> bias(kExperts);
    for (float& b : bias) b = static_cast<float>(gen.symmetric(0.5));

    std::vector<double> logits_d(logits.begin(), logits.end());
    std::vector<double> bias_d(bias.begin(), bias.end());

    auto [got_idx, got_w] = run(static_cast<int>(tokens), nullptr, nullptr, logits, bias);

    // (1) The reference selection, and the position-exact id comparison.
    {
        size_t positional = 0;
        size_t set_mismatch = 0;
        std::vector<double> want_w;
        std::vector<int32_t> want_idx_all;

        for (size_t t = 0; t < tokens; ++t) {
            const std::vector<double> lg(logits_d.begin() + static_cast<long>(t * kExperts),
                                         logits_d.begin() + static_cast<long>((t + 1) * kExperts));
            const RouterSelection want =
                aeon::reference::router_topk(lg, bias_d, kTopK, kScaling);

            const std::vector<int32_t> got_t(got_idx.begin() + static_cast<long>(t * kTopK),
                                             got_idx.begin() + static_cast<long>((t + 1) * kTopK));
            if (first_difference(want.ids, got_t) != kTopK) ++positional;
            if (!same_set(want.ids, got_t)) ++set_mismatch;

            want_idx_all.insert(want_idx_all.end(), want.ids.begin(), want.ids.end());
            want_w.insert(want_w.end(), want.weights.begin(), want.weights.end());
        }

        ok &= check("top-k ids match the oracle positionally", positional == 0,
                    std::to_string(positional) + " of " + std::to_string(tokens) +
                        " tokens differ positionally");
        ok &= check("top-k ids match the oracle as a set", set_mismatch == 0,
                    std::to_string(set_mismatch) + " of " + std::to_string(tokens) +
                        " tokens differ as a set");

        std::vector<double> got_wf(got_w.begin(), got_w.end());
        ok &= report("top-k weights vs oracle (unbiased scores)", want_w, got_wf);

        // Positional agreement must be *earned*: the id vector is score-ordered,
        // so verify descending selection order explicitly rather than trusting
        // that "positional match" implied it.
        bool descending = true;
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t k = 0; k + 1 < kTopK; ++k) {
                const size_t a = static_cast<size_t>(got_idx[t * kTopK + k]);
                const size_t b = static_cast<size_t>(got_idx[t * kTopK + k + 1]);
                const double sa = aeon::reference::router_score(logits_d[t * kExperts + a]) +
                                  bias_d[a];
                const double sb = aeon::reference::router_score(logits_d[t * kExperts + b]) +
                                  bias_d[b];
                if (sa < sb) descending = false;
            }
        }
        ok &= check("selection is ordered by descending selection score", descending,
                    "non-increasing selection score");
    }

    // (2) The bias must be added AFTER softplus. The wrong variant is a plausible
    //     loader bug (reading `ffn.gate.bias` as a linear logit bias) and it also
    //     routes — so the check is that it differs, and that the kernel follows
    //     the post-softplus rule.
    {
        size_t changed_tokens = 0;
        double worst_score_delta = 0.0;
        for (size_t t = 0; t < tokens; ++t) {
            const std::vector<double> lg(logits_d.begin() + static_cast<long>(t * kExperts),
                                         logits_d.begin() + static_cast<long>((t + 1) * kExperts));
            const RouterSelection right =
                aeon::reference::router_topk(lg, bias_d, kTopK, kScaling);
            const RouterSelection wrong =
                aeon::reference::router_topk(lg, bias_d, kTopK, kScaling,
                                             /*bias_before_softplus=*/true);
            if (!same_set(right.ids, wrong.ids)) ++changed_tokens;
            const ErrorStats s = aeon::reference::compare(right.weights, wrong.weights, 1.0);
            worst_score_delta = std::fmax(worst_score_delta, s.max_rel);
        }
        ok &= check("bias-before-softplus is a different router", changed_tokens > 0,
                    std::to_string(changed_tokens) + " of " + std::to_string(tokens) +
                        " tokens route differently (max weight rel " +
                        std::to_string(worst_score_delta) + ")");

        // And the kernel is on the correct side of that fork.
        std::vector<double> want_w;
        for (size_t t = 0; t < tokens; ++t) {
            const std::vector<double> lg(logits_d.begin() + static_cast<long>(t * kExperts),
                                         logits_d.begin() + static_cast<long>((t + 1) * kExperts));
            const RouterSelection right =
                aeon::reference::router_topk(lg, bias_d, kTopK, kScaling);
            want_w.insert(want_w.end(), right.weights.begin(), right.weights.end());
        }
        std::vector<double> got_wf(got_w.begin(), got_w.end());
        const double to_wrong = aeon::reference::compare(
            [&] {
                std::vector<double> v;
                for (size_t t = 0; t < tokens; ++t) {
                    const std::vector<double> lg(
                        logits_d.begin() + static_cast<long>(t * kExperts),
                        logits_d.begin() + static_cast<long>((t + 1) * kExperts));
                    const RouterSelection w = aeon::reference::router_topk(
                        lg, bias_d, kTopK, kScaling, true);
                    v.insert(v.end(), w.weights.begin(), w.weights.end());
                }
                return v;
            }(),
            got_wf, 1.0).max_rel;
        ok &= check("kernel matches post-softplus bias, not pre", to_wrong > 1e-2,
                    "distance from the wrong variant = " + std::to_string(to_wrong));
    }

    // (3) Omitting the bias changes the routing, so the bias cannot be a no-op
    //     that the pass above ignores.
    {
        size_t changed = 0;
        for (size_t t = 0; t < tokens; ++t) {
            const std::vector<double> lg(logits_d.begin() + static_cast<long>(t * kExperts),
                                         logits_d.begin() + static_cast<long>((t + 1) * kExperts));
            const RouterSelection with = aeon::reference::router_topk(lg, bias_d, kTopK, kScaling);
            const RouterSelection without = aeon::reference::router_topk(lg, {}, kTopK, kScaling);
            if (!same_set(with.ids, without.ids)) ++changed;
        }
        ok &= check("bias participates in selection", changed > 0,
                    std::to_string(changed) + " of " + std::to_string(tokens) +
                        " tokens change without bias");
    }

    // (4) Selection is FLAT. Construct logits whose six largest land in six
    //     different 32-expert groups, and show that group-limited routing
    //     (top-2 groups), which this config does not use, picks differently —
    //     while the kernel agrees with the flat rule.
    {
        std::vector<double> spread(kExperts, -10.0);
        for (size_t g = 0; g < 6; ++g) spread[g * 32 + 3] = 5.0 - 0.1 * static_cast<double>(g);

        const RouterSelection flat = aeon::reference::router_topk(spread, {}, kTopK, kScaling);
        const RouterSelection grouped = grouped_variant(spread, std::vector<double>(kExperts, 0.0),
                                                        kTopK, kScaling, 32, 2);
        ok &= check("group-limited routing is distinguishable (and differs)",
                    !same_set(flat.ids, grouped.ids),
                    std::to_string(flat.ids.size()) + " flat vs " +
                        std::to_string(grouped.ids.size()) + " grouped ids");

        std::vector<float> spread_f(spread.begin(), spread.end());
        std::vector<float> spread_bias(kExperts, 0.0f);
        auto [s_idx, s_w] = run(1, nullptr, nullptr, spread_f, spread_bias);

        std::vector<int32_t> got_first(s_idx.begin(), s_idx.begin() + static_cast<long>(kTopK));
        ok &= check("kernel selects the flat top-6, not the grouped one",
                    same_set(flat.ids, got_first) &&
                        first_difference(flat.ids, got_first) == kTopK,
                    "positional agreement with the flat rule");

        const double to_grouped = aeon::reference::compare(
            grouped.weights, std::vector<double>(s_w.begin(), s_w.end()), 1.0).max_rel;
        ok &= check("kernel weights are far from the grouped variant", to_grouped > 1e-2,
                    "distance = " + std::to_string(to_grouped));
    }

    // (5) Tie-break: lowest index wins. Random logits never tie, so this is
    //     constructed directly.
    {
        std::vector<double> tied(kExperts, 0.0);
        tied[7] = tied[70] = tied[180] = tied[255] = 3.0; // four-way tie at the top
        std::vector<float> tied_f(tied.begin(), tied.end());
        std::vector<float> zero_bias(kExperts, 0.0f);
        auto [t_idx, t_w] = run(1, nullptr, nullptr, tied_f, zero_bias);

        std::vector<int32_t> got_t(t_idx.begin(), t_idx.begin() + static_cast<long>(kTopK));
        const RouterSelection want = aeon::reference::router_topk(tied, {}, kTopK, kScaling);
        ok &= check("ties resolve to the lowest expert index",
                    first_difference(want.ids, got_t) == kTopK,
                    "winner sequence starts " + std::to_string(got_t[0]) + ", " +
                        std::to_string(got_t[1]) + ", " + std::to_string(got_t[2]));
    }

    // =======================================================================
    // B. Hash layer (0, 1, 2): ids straight from the table, bias ignored
    // =======================================================================
    std::cout << "\n--- B. hash layer (layers 0..2, synthetic table) ---\n";
    {
        // A table row per token id, deliberately UNSORTED so that a score-sorting
        // implementation is distinguishable from a faithful one. The kernel
        // indexes the table by **token id**, so it must have a row for every id
        // this section passes (11, 22, 33) — a short table would be read out of
        // bounds rather than fail loudly.
        const size_t n_tokens = 3;
        constexpr size_t kSyntheticRows = 64;
        const int32_t token_ids[] = {11, 22, 33};

        std::vector<int64_t> table(kSyntheticRows * kTopK, 0);
        for (size_t t = 0; t < kSyntheticRows; ++t) {
            for (size_t k = 0; k < kTopK; ++k) {
                // Reversed score order per row, so column order is not score order.
                table[t * kTopK + k] =
                    static_cast<int64_t>((t * 7 + 5 * (kTopK - 1 - k)) % kExperts);
            }
        }

        CHECK_HIP(hipMalloc(&d_table, table.size() * sizeof(int64_t)));
        CHECK_HIP(hipMalloc(&d_token_ids, sizeof(token_ids)));
        CHECK_HIP(hipMemcpy(d_table, table.data(), table.size() * sizeof(int64_t),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_token_ids, token_ids, sizeof(token_ids),
                            hipMemcpyHostToDevice));

        std::vector<float> h_logits(n_tokens * kExperts);
        for (float& x : h_logits) x = static_cast<float>(gen.symmetric(4.0));
        // A bias is passed on purpose: the hash branch must ignore it entirely.
        std::vector<float> h_bias(kExperts);
        for (float& b : h_bias) b = static_cast<float>(gen.symmetric(5.0));

        auto [h_idx, h_w] = run(static_cast<int>(n_tokens), d_table, d_token_ids,
                                h_logits, h_bias);

        std::vector<double> want_w;
        std::vector<int32_t> want_idx;
        for (size_t t = 0; t < n_tokens; ++t) {
            const std::vector<double> lg(h_logits.begin() + static_cast<long>(t * kExperts),
                                         h_logits.begin() + static_cast<long>((t + 1) * kExperts));
            const std::vector<int64_t> row(table.begin() + static_cast<long>(token_ids[t] * kTopK),
                                           table.begin() + static_cast<long>((token_ids[t] + 1) * kTopK));
            const RouterSelection want = aeon::reference::router_hash(lg, row, kScaling);
            want_idx.insert(want_idx.end(), want.ids.begin(), want.ids.end());
            want_w.insert(want_w.end(), want.weights.begin(), want.weights.end());
        }

        std::vector<int32_t> got_idx(h_idx.begin(), h_idx.end());
        ok &= check("hash ids are the table row, in column order",
                    first_difference(want_idx, got_idx) == want_idx.size(),
                    "positional agreement with the unsorted table");

        std::vector<double> got_w(h_w.begin(), h_w.end());
        ok &= report("hash weights vs oracle (unbiased scores)", want_w, got_w);

        // The bias must be ignored: with an extreme bias the selection must not move.
        std::vector<float> huge_bias(kExperts, 100.0f);
        auto [h_idx2, h_w2] = run(static_cast<int>(n_tokens), d_table, d_token_ids,
                                  h_logits, huge_bias);
        std::vector<int32_t> got_idx2(h_idx2.begin(), h_idx2.end());
        ok &= check("hash branch ignores the bias entirely",
                    first_difference(got_idx, got_idx2) == got_idx.size(),
                    "ids unchanged under a +100 bias");
        (void)h_w2;

        CHECK_HIP(hipFree(d_table));
        CHECK_HIP(hipFree(d_token_ids));
        d_table = nullptr;
        d_token_ids = nullptr;
    }

    // =======================================================================
    // C. The artifact's own tid2eid table
    // =======================================================================
    std::cout << "\n--- C. artifact: tid2eid shape, dtype and layer split ---\n";
    {
        const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
        aeon::core::AeonModelLoader loader;
        loader.open_model(model_dir);

        bool shape_ok = true;
        bool split_ok = true;
        std::string detail;

        for (int layer = 0; layer < 43; ++layer) {
            const std::string prefix = "layers." + std::to_string(layer) + ".ffn.gate.";
            const bool has_table = loader.has_tensor(prefix + "tid2eid");
            const bool has_bias = loader.has_tensor(prefix + "bias");

            if (layer < static_cast<int>(kHashLayers)) {
                if (!has_table || has_bias) split_ok = false;
                if (layer == 0) {
                    if (!has_table) {
                        shape_ok = false;
                        detail = "layers.0.ffn.gate.tid2eid is absent";
                    } else {
                        const auto& t = loader.get_tensor(prefix + "tid2eid");
                        const bool ok_dtype = (t.dtype == "I64");
                        const bool ok_shape = t.shape.size() == 2 &&
                            t.shape[0] == 129280 && t.shape[1] == 6;
                        shape_ok = ok_dtype && ok_shape;
                        detail = t.dtype + " [" +
                            std::to_string(t.shape.size() > 0 ? t.shape[0] : -1) + ", " +
                            std::to_string(t.shape.size() > 1 ? t.shape[1] : -1) + "]";
                    }
                }
            } else {
                if (has_table || !has_bias) split_ok = false;
            }
        }

        ok &= check("tid2eid is I64 [129280, 6]", shape_ok, detail);
        ok &= check("tid2eid only on layers 0-2; bias only on 3+", split_ok,
                    "43 layers checked");

        // The real end-to-end check: replay the hash branch at real token ids
        // with the artifact's own table.
        if (loader.has_tensor("layers.0.ffn.gate.tid2eid")) {
            const auto& t = loader.get_tensor("layers.0.ffn.gate.tid2eid");
            const int64_t* table_host = reinterpret_cast<const int64_t*>(t.data);

            constexpr size_t n = sizeof(kRealTokenIds) / sizeof(kRealTokenIds[0]);
            int64_t* d_real_table = nullptr;
            int32_t* d_real_ids = nullptr;
            CHECK_HIP(hipMalloc(&d_real_table, static_cast<size_t>(t.byte_size)));
            CHECK_HIP(hipMalloc(&d_real_ids, n * sizeof(int32_t)));
            CHECK_HIP(hipMemcpy(d_real_table, table_host, static_cast<size_t>(t.byte_size),
                                hipMemcpyHostToDevice));
            CHECK_HIP(hipMemcpy(d_real_ids, kRealTokenIds, n * sizeof(int32_t),
                                hipMemcpyHostToDevice));

            std::vector<float> real_logits(n * kExperts);
            for (float& x : real_logits) x = static_cast<float>(gen.symmetric(4.0));

            auto [r_idx, r_w] = run(static_cast<int>(n), d_real_table, d_real_ids,
                                    real_logits, std::vector<float>(kExperts, 0.0f));

            size_t wrong = 0;
            int64_t sampled_min = 1 << 30;
            int64_t sampled_max = -(1 << 30);
            for (size_t i = 0; i < n; ++i) {
                for (size_t k = 0; k < kTopK; ++k) {
                    const int64_t want = table_host[static_cast<int64_t>(kRealTokenIds[i]) * 6 +
                                                    static_cast<int64_t>(k)];
                    if (static_cast<int64_t>(r_idx[i * kTopK + k]) != want) ++wrong;
                    sampled_min = std::min(sampled_min, want);
                    sampled_max = std::max(sampled_max, want);
                }
            }
            ok &= check("real token ids reproduce the artifact table row", wrong == 0,
                        std::to_string(wrong) + " of " + std::to_string(n * kTopK) +
                            " ids wrong");
            ok &= check("table entries are valid expert ids",
                        sampled_min >= 0 && sampled_max < static_cast<int64_t>(kExperts),
                        "range [" + std::to_string(sampled_min) + ", " +
                            std::to_string(sampled_max) + "] over real tokens");
            (void)r_w;

            CHECK_HIP(hipFree(d_real_table));
            CHECK_HIP(hipFree(d_real_ids));
        }
    }

    // =======================================================================
    // D. Config: the flat-routing claim, at the artifact
    // =======================================================================
    std::cout << "\n--- D. config: flat top-6, no group routing ---\n";
    {
        const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
        const auto config = aeon::core::DeepSeekV4Config::load_from_json(
            model_dir + "/config.json");

        ok &= check("n_routed_experts == 256", config.n_routed_experts == 256,
                    std::to_string(config.n_routed_experts));
        ok &= check("num_experts_per_tok == 6", config.num_experts_per_tok == 6,
                    std::to_string(config.num_experts_per_tok));
        ok &= check("num_hash_layers == 3", config.num_hash_layers == 3,
                    std::to_string(config.num_hash_layers));
        ok &= check("routed_scaling_factor == 1.5",
                    std::fabs(config.routed_scaling_factor - 1.5f) < 1e-6f,
                    std::to_string(config.routed_scaling_factor));
        ok &= check("scoring_func == sqrtsoftplus",
                    config.scoring_func == "sqrtsoftplus", config.scoring_func);

        // The plan's structural claim is that `n_group`/`topk_group` are *absent*
        // from the config, which is what makes this a flat top-6 rather than the
        // DeepSeek-V3 `noaux_tc` grouped variant. A typed parser cannot show
        // absence, so read the file itself.
        std::ifstream in(model_dir + "/config.json");
        const std::string raw((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
        const bool no_group = raw.find("\"n_group\"") == std::string::npos;
        const bool no_topk_group = raw.find("\"topk_group\"") == std::string::npos;
        ok &= check("n_group and topk_group are absent from config.json",
                    no_group && no_topk_group,
                    no_group && no_topk_group ? "both absent"
                                              : "a group field is present");
    }

    CHECK_HIP(hipFree(d_logits));
    CHECK_HIP(hipFree(d_bias));
    CHECK_HIP(hipFree(d_weights));
    CHECK_HIP(hipFree(d_indices));

    std::cout << (ok ? "\n[SUCCESS] MoE router gate passed.\n"
                     : "\n[FAILURE] MoE router gate failed.\n");
    return ok ? 0 : 1;
}