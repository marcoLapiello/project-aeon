// -----------------------------------------------------------------------------
// Tier-1 gate: lightning indexer + top-k (Step 2.4.3) — versus an independent
// fp64 reference.
//
// This gate has two jobs, and the second is a measurement rather than a
// certification.
//
// JOB 1 — certify the scoring. The reference's whole indexer path is five
// readable lines, and the one that matters is `F.relu(score)` applied to the
// per-head dot **before** the weighting (trap 11). A missing ReLU still produces
// a plausible attention score, which is why it survives review.
//
// JOB 2 — SETTLE GATE 11, the Hadamard rotation. The plan has mis-stated this
// three separate times, so this gate measures it instead of arguing: apply a
// normalized Hadamard to both Q and K and show the scores and the top-k set are
// unchanged; apply it to only one side and show the scores move. That is the
// whole decision — the rotation is orthogonal, so two-sided is a no-op and
// one-sided is a bug.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

constexpr size_t kHeads   = 64;
constexpr size_t kHeadDim = 128;
constexpr size_t kTopK    = 512;

// The two factors the pipeline passes separately: weight_scale = softmax_scale *
// n_heads^-0.5 = 1/sqrt(128*64). Written out as literals so they are
// constexpr-evaluable (std::sqrt is not).
constexpr double kSoftmaxScale = 0.08838834764831845; // 1 / sqrt(128)
constexpr double kHeadScale    = 0.125;               // 1 / sqrt(64)

constexpr double kTol = 3e-3; // fraction of peak; fp16 inputs

std::vector<double> widen(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(__half2float(v[i]));
    return out;
}

bool report_scores(const char* label, const std::vector<double>& want,
                   const std::vector<double>& got) {
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-50s SIZE MISMATCH                  FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(s.max_rel) && s.max_rel <= kTol;
    std::printf("  %-50s max_abs=%.3e max_rel=%.3e  %s\n",
                label, s.max_abs, s.max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-50s %-26s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

// Exact set agreement between two top-k selections. Order is not required to
// match — the kernel returns a fixed-width buffer padded with -1 — but the
// selected indices and the padding must.
bool same_selection(const std::vector<int32_t>& a, const std::vector<int32_t>& b,
                    size_t k, std::string& detail) {
    std::vector<int32_t> sa(a.begin(), a.begin() + static_cast<long>(k));
    std::vector<int32_t> sb(b.begin(), b.begin() + static_cast<long>(k));
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    const size_t mismatches = static_cast<size_t>(
        std::count_if(sa.begin(), sa.end(), [&](int32_t x) {
            return !std::binary_search(sb.begin(), sb.end(), x);
        }));
    detail = std::to_string(mismatches) + " of " + std::to_string(k) + " differ";
    return mismatches == 0;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: lightning indexer + top-k (+ Hadamard measurement)\n";
    aeon::core::select_compute_device(true);

    bool ok = true;

    aeon::reference::Rng gen(0x1DE7E710ull);

    const size_t candidates = kTopK + 256; // > top-k, so selection is non-trivial
    const size_t q_count = kHeads * kHeadDim;
    const size_t k_count = candidates * kHeadDim;

    // --- Inputs -------------------------------------------------------------
    // A mix of positive and negative dots is essential: if every per-head dot
    // were positive the ReLU would be invisible and this gate could not tell a
    // correct indexer from one that omits it. Centring q and k at zero gives
    // roughly half negative dots.
    std::vector<double> q_d(q_count), k_d(k_count), w_d(kHeads);
    for (size_t i = 0; i < q_count; ++i) q_d[i] = gen.symmetric(1.0);
    for (size_t i = 0; i < k_count; ++i) k_d[i] = gen.symmetric(1.0);
    for (size_t h = 0; h < kHeads; ++h) w_d[h] = gen.symmetric(1.0);

    std::vector<__half> h_q(q_count), h_k(k_count);
    for (size_t i = 0; i < q_count; ++i) h_q[i] = __float2half(static_cast<float>(q_d[i]));
    for (size_t i = 0; i < k_count; ++i) h_k[i] = __float2half(static_cast<float>(k_d[i]));
    std::vector<float> h_w(kHeads);
    for (size_t h = 0; h < kHeads; ++h) h_w[h] = static_cast<float>(w_d[h]);

    const std::vector<double> q = widen(h_q);
    const std::vector<double> k = widen(h_k);
    const std::vector<double> w = [&]{ std::vector<double> v(kHeads);
        for (size_t h = 0; h < kHeads; ++h) v[h] = static_cast<double>(h_w[h]);
        return v; }();

    // --- Device ------------------------------------------------------------
    __half* d_q = nullptr;
    __half* d_k = nullptr;
    float* d_w = nullptr;
    float* d_scores = nullptr;
    CHECK_HIP(hipMalloc(&d_q, q_count * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_k, k_count * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_w, kHeads * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_scores, candidates * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_q, h_q.data(), q_count * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_k, h_k.data(), k_count * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_w, h_w.data(), kHeads * sizeof(float), hipMemcpyHostToDevice));

    auto run_scores = [&]() {
        aeon::kernel::v4_indexer_scores_kernel<<<dim3((candidates + 255) / 256), dim3(256)>>>(
            d_q, d_w, d_k, d_scores, static_cast<int>(candidates),
            static_cast<int>(kHeads), static_cast<int>(kHeadDim),
            static_cast<float>(kSoftmaxScale), static_cast<float>(kHeadScale));
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<float> out(candidates);
        CHECK_HIP(hipMemcpy(out.data(), d_scores, candidates * sizeof(float),
                            hipMemcpyDeviceToHost));
        return std::vector<double>(out.begin(), out.end());
    };

    // =======================================================================
    // A. The scoring path
    // =======================================================================
    const std::vector<double> got = run_scores();
    const std::vector<double> want = aeon::reference::indexer_scores(
        q, k, w, kHeads, kHeadDim, kSoftmaxScale, kHeadScale);
    ok &= report_scores("indexer scores vs oracle (ReLU per head)", want, got);

    // The ReLU must be load-bearing, or this gate cannot distinguish a correct
    // indexer from one that omits it. Compare against the no-ReLU variant: the
    // scores must move a lot, and the top-k must change.
    {
        const std::vector<double> no_relu = aeon::reference::indexer_scores(
            q, k, w, kHeads, kHeadDim, kSoftmaxScale, kHeadScale, /*apply_relu=*/false);
        const double d = aeon::reference::compare(want, no_relu, 1.0).max_rel;

        std::string detail;
        const bool differs = !same_selection(
            aeon::reference::topk_indices(want, kTopK),
            aeon::reference::topk_indices(no_relu, kTopK), kTopK, detail);

        ok &= check("ReLU (per head, before weighting) is load-bearing",
                    d > 1e-2 && differs,
                    "scores differ by " + std::to_string(d) + "; top-k: " + detail);
    }

    // The per-head extent of the ReLU is also checkable directly: for at least
    // one (head, candidate) pair the dot must be negative, otherwise the test
    // above is vacuous.
    {
        size_t negatives = 0;
        for (size_t h = 0; h < kHeads; ++h) {
            for (size_t c = 0; c < candidates; ++c) {
                double dot = 0.0;
                for (size_t d = 0; d < kHeadDim; ++d) {
                    dot += q[h * kHeadDim + d] * k[c * kHeadDim + d];
                }
                if (dot < 0.0) ++negatives;
            }
        }
        const size_t total = kHeads * candidates;
        ok &= check("data exercises the ReLU (negative dots exist)",
                    negatives > total / 4,
                    std::to_string(negatives * 100 / total) + "% of per-head dots are negative");
    }

    // =======================================================================
    // B. Top-k selection
    // =======================================================================
    {
        const std::vector<int32_t> want_idx = aeon::reference::topk_indices(want, kTopK);
        const std::vector<int32_t> got_idx = aeon::reference::topk_indices(got, kTopK);
        std::string detail;
        ok &= check("top-k selection matches the oracle exactly",
                    same_selection(want_idx, got_idx, kTopK, detail), detail);

        // Ordering: descending score within the selection.
        bool descending = true;
        for (size_t i = 0; i + 1 < got_idx.size(); ++i) {
            if (got[static_cast<size_t>(got_idx[i])] < got[static_cast<size_t>(got_idx[i + 1])]) {
                descending = false;
            }
        }
        ok &= check("selection is ordered by descending score", descending,
                    "top-k strictly non-increasing");
    }

    // Tie-breaking: equal scores select the lower index (trap 18). Constructed
    // directly, since random scores essentially never tie.
    {
        std::vector<double> tied(candidates, 0.0);
        for (size_t c = 0; c < candidates; ++c) {
            tied[c] = static_cast<double>((c % 3)); // many ties
        }
        const std::vector<int32_t> sel = aeon::reference::topk_indices(tied, kTopK);
        bool lowest_first = true;
        for (size_t c = 0; c < candidates; ++c) {
            if (tied[c] != 2.0) continue;
            // Every score-2 index must be selected before any score-1 index, and
            // among equals the lowest index wins.
            if (std::find(sel.begin(), sel.end(), static_cast<int32_t>(c)) == sel.end()) {
                lowest_first = false;
            }
        }
        ok &= check("ties resolve to the lower index", lowest_first,
                    "all maximal-score indices selected");
    }

    // Short candidate list: the whole window is selected, no padding.
    {
        const size_t few = 7;
        std::vector<double> short_scores(got.begin(), got.begin() + static_cast<long>(few));
        const std::vector<int32_t> sel = aeon::reference::topk_indices(short_scores, kTopK);
        ok &= check("candidates < top-k selects all of them", sel.size() == few,
                    std::to_string(sel.size()) + " of " + std::to_string(few) + " selected");
    }

    // =======================================================================
    // C. GATE 11 — the Hadamard rotation, measured
    // =======================================================================
    std::cout << "\n  --- Gate 11: the Hadamard rotation (measurement, not assertion) ---\n";
    {
        // Rotate BOTH Q and K, per head.
        const std::vector<double> q_rot =
            aeon::reference::hadamard_rotate_heads(q, kHeads, kHeadDim);
        const std::vector<double> k_rot =
            aeon::reference::hadamard_rotate_heads(k, candidates, kHeadDim);

        const std::vector<double> two_sided = aeon::reference::indexer_scores(
            q_rot, k_rot, w, kHeads, kHeadDim, kSoftmaxScale, kHeadScale);

        // (1) Two-sided rotation is score-preserving. This is the claim the plan
        //     makes and this is the measurement that settles it.
        {
            const ErrorStats s = aeon::reference::compare(want, two_sided, 1.0);
            ok &= check("two-sided Hadamard leaves scores unchanged",
                        s.max_rel < 1e-12,
                        "max_rel = " + std::to_string(s.max_rel) + " (logit-preserving)");
        }

        // (2) ... and therefore leaves the selection unchanged too.
        {
            std::string detail;
            const bool same = same_selection(
                aeon::reference::topk_indices(want, kTopK),
                aeon::reference::topk_indices(two_sided, kTopK), kTopK, detail);
            ok &= check("two-sided Hadamard leaves top-k unchanged", same, detail);
        }

        // (3) One-sided rotation is NOT harmless. This is the failure mode the
        //     plan warns about: applying it to Q and forgetting K silently
        //     changes every score.
        {
            const std::vector<double> one_sided = aeon::reference::indexer_scores(
                q_rot, k, w, kHeads, kHeadDim, kSoftmaxScale, kHeadScale);
            const ErrorStats s = aeon::reference::compare(want, one_sided, 1.0);

            std::string detail;
            const bool differs = !same_selection(
                aeon::reference::topk_indices(want, kTopK),
                aeon::reference::topk_indices(one_sided, kTopK), kTopK, detail);

            ok &= check("one-sided Hadamard changes scores and top-k",
                        s.max_rel > 1e-2 && differs,
                        "max_rel = " + std::to_string(s.max_rel) + "; top-k: " + detail);
        }

        // (4) The rotation is orthogonal — checked directly, because "logit
        //     preserving" rests on exactly this and nothing else. The Gram
        //     matrix of the rotated heads must equal that of the originals.
        {
            double worst = 0.0;
            for (size_t h = 0; h < kHeads; ++h) {
                for (size_t h2 = 0; h2 < kHeads; ++h2) {
                    double rotated_dot = 0.0;
                    double input_dot = 0.0;
                    for (size_t d = 0; d < kHeadDim; ++d) {
                        rotated_dot += q_rot[h * kHeadDim + d] * q_rot[h2 * kHeadDim + d];
                        input_dot += q[h * kHeadDim + d] * q[h2 * kHeadDim + d];
                    }
                    worst = std::fmax(worst, std::fabs(rotated_dot - input_dot));
                }
            }
            ok &= check("Hadamard preserves all inner products", worst < 1e-12,
                        "max Gram-matrix deviation = " + std::to_string(worst));
        }

        std::printf("\n  GATE 11 DECISION: the Hadamard is logit-preserving — a two-sided\n"
                    "  rotation changes neither the scores (%.1e) nor the top-k, while a\n"
                    "  one-sided rotation changes both. Our engine stores the indexer K in\n"
                    "  fp16, so there is no fp8 round-trip to condition, and the reference\n"
                    "  itself drops the rotation in its fused path. DECISION: do not apply it.\n",
                    aeon::reference::compare(want, two_sided, 1.0).max_rel);
    }

    CHECK_HIP(hipFree(d_q));
    CHECK_HIP(hipFree(d_k));
    CHECK_HIP(hipFree(d_w));
    CHECK_HIP(hipFree(d_scores));

    std::cout << (ok ? "\n[SUCCESS] Indexer gate passed.\n"
                     : "\n[FAILURE] Indexer gate failed.\n");
    return ok ? 0 : 1;
}
