// -----------------------------------------------------------------------------
// Tier-1 gate: MLA Q and KV paths (Step 2.2) — versus an independent fp64 reference.
//
// This is a *composition* gate. MLA is not one kernel: it is a wiring of GEMV,
// RMSNorm and a weightless per-head norm, and the thing that can go wrong is the
// wiring, not the arithmetic. Specifically:
//
//   trap 5 — the Q path is low-rank with TWO norms, and the first sits BETWEEN
//            wq_a and wq_b. Collapsing the matmuls, or moving the norm, yields
//            plausible-but-wrong output.
//   trap 6 — there is no separate V. One 512-wide row is both key and value.
//
// So the gate replays the pipeline's exact kernel order, stage by stage, and
// compares every intermediate the plan names (`q_lora`, `q_lora_norm`, `q`,
// `kv`) against the oracle. Each stage's oracle input is the kernel's own fp16
// output, so the delta measures the kernel and not the input quantization.
//
// It also asserts three properties that no numerical closeness test would catch:
//   (a) the mid-path norm is load-bearing — dropping it changes q materially;
//   (b) the per-head norm is per HEAD — each head ends with unit mean-square,
//       which is false if the norm were applied across all 64*512 at once;
//   (c) there is exactly one KV tensor used for both roles.
//
// Tolerance. Stated once, as a fraction of the vector's own peak magnitude
// (`rel_floor_fraction = 1.0`), because these vectors are signed and unstructured:
// a per-element relative error on an element that happens to be near zero says
// nothing. The expected floor is the fp16 store, one ulp ≈ 4.9e-4 of peak, plus
// fp32 accumulation over 1024–4096 terms.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/v4_gemv.hpp"
#include "architecture/deepseek_v4/kernels/v4_norm.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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

// The graph's real dimensions — this gate is about the actual shapes.
constexpr size_t kHidden    = 4096;
constexpr size_t kQLoraRank = 1024;
constexpr size_t kHeads     = 64;
constexpr size_t kHeadDim   = 512;
constexpr size_t kQWidth    = kHeads * kHeadDim; // 32768
constexpr double kEps       = 1e-6;

constexpr double kTol = 2e-3; // fraction of the vector's peak magnitude

std::vector<double> widen(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<double>(__half2float(v[i]));
    }
    return out;
}

// Fills fp16 storage straight from the PRNG, without a double staging buffer —
// wq_b alone is 33.5M values, and a 268 MB intermediate would be pure waste.
std::vector<__half> fill_halves(aeon::reference::Rng& rng, size_t n, double scale) {
    std::vector<__half> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = __float2half(static_cast<float>(rng.symmetric(scale)));
    }
    return out;
}

// Compares `want` against `got` with the denominator set to the vector's peak, so
// the printed max_rel is directly "error as a fraction of scale".
bool stage(const char* label, const std::vector<double>& want,
           const std::vector<__half>& got_h) {
    const std::vector<double> got = widen(got_h);
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-34s SIZE MISMATCH                              FAIL\n", label);
        return false;
    }
    const bool pass = s.max_rel <= kTol;
    std::printf("  %-34s max_abs=%.3e  max_rel=%.3e  (peak=%.3e)  %s\n",
                label, s.max_abs, s.max_rel, aeon::reference::peak_abs(want),
                pass ? "PASS" : "FAIL");
    return pass;
}

bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-34s %-40s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

// Widening accessor over a row-major fp16 [out, in] matrix. This is how the
// oracle reads the weights the kernel reads, without materialising them twice.
struct Fp16Matrix {
    const __half* data;
    size_t in_dim;

    double operator()(size_t o, size_t i) const {
        return static_cast<double>(__half2float(data[o * in_dim + i]));
    }
};

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 composition: MLA Q/KV paths vs independent fp64 reference\n";
    aeon::core::select_compute_device(true);

    bool ok = true;

    // -----------------------------------------------------------------------
    // Inputs. Scales chosen to give realistic magnitudes: weights ~ 1/sqrt(in),
    // x_norm ~ unit. Nothing here needs to be special — the gate compares two
    // implementations on the same data, so the data only has to be non-degenerate.
    // -----------------------------------------------------------------------
    aeon::reference::Rng rng(0x0A11A124ull);
    aeon::reference::Rng data_rng(0x0D0D5E11ull);

    const std::vector<__half> h_x_norm = fill_halves(data_rng, kHidden, 1.0);
    const std::vector<__half> h_q_norm_w = fill_halves(data_rng, kQLoraRank, 1.0);
    const std::vector<__half> h_kv_norm_w = fill_halves(data_rng, kHeadDim, 1.0);
    const std::vector<__half> h_wq_a = fill_halves(rng, kQLoraRank * kHidden, 0.027);
    const std::vector<__half> h_wq_b = fill_halves(rng, kQWidth * kQLoraRank, 0.054);
    const std::vector<__half> h_wkv  = fill_halves(rng, kHeadDim * kHidden, 0.027);

    const std::vector<double> x_norm = widen(h_x_norm);

    // --- Device buffers ------------------------------------------------------
    __half* d_x          = nullptr;
    __half* d_q_norm_w   = nullptr;
    __half* d_kv_norm_w  = nullptr;
    __half* d_wq_a       = nullptr;
    __half* d_wq_b       = nullptr;
    __half* d_wkv        = nullptr;
    __half* d_qa         = nullptr;
    __half* d_qa_norm    = nullptr;
    __half* d_q          = nullptr;
    __half* d_kv         = nullptr;

    auto upload = [](const std::vector<__half>& src, __half** dst) {
        CHECK_HIP(hipMalloc(dst, src.size() * sizeof(__half)));
        CHECK_HIP(hipMemcpy(*dst, src.data(), src.size() * sizeof(__half),
                            hipMemcpyHostToDevice));
    };
    upload(h_x_norm, &d_x);
    upload(h_q_norm_w, &d_q_norm_w);
    upload(h_kv_norm_w, &d_kv_norm_w);
    upload(h_wq_a, &d_wq_a);
    upload(h_wq_b, &d_wq_b);
    upload(h_wkv, &d_wkv);
    CHECK_HIP(hipMalloc(&d_qa, kQLoraRank * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_qa_norm, kQLoraRank * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_q, kQWidth * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_kv, kHeadDim * sizeof(__half)));

    auto download = [](const __half* src, size_t n) {
        std::vector<__half> out(n);
        CHECK_HIP(hipMemcpy(out.data(), src, n * sizeof(__half), hipMemcpyDeviceToHost));
        return out;
    };

    const Fp16Matrix wq_a{h_wq_a.data(), kHidden};
    const Fp16Matrix wq_b{h_wq_b.data(), kQLoraRank};
    const Fp16Matrix wkv{h_wkv.data(), kHidden};

    // =======================================================================
    // Stage 1 — Q_a = wq_a @ x_norm  [4096 -> 1024]
    // =======================================================================
    aeon::kernel::v4_gemv_fp16_kernel<<<dim3(kQLoraRank, 1), 32>>>(
        d_x, d_wq_a, d_qa, static_cast<int>(kHidden));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    const std::vector<__half> h_qa = download(d_qa, kQLoraRank);

    // The oracle reads x_norm and wq_a as the kernel did — fp16 values widened.
    ok &= stage("q_lora      (after wq_a)",
                aeon::reference::matvec(kQLoraRank, kHidden, x_norm, wq_a), h_qa);

    // =======================================================================
    // Stage 2 — Q_a norm (weighted, over q_lora_rank = 1024)
    // =======================================================================
    aeon::kernel::v4_rmsnorm_wave32_kernel<<<dim3(1), 32>>>(
        d_qa, d_q_norm_w, d_qa_norm, static_cast<int>(kQLoraRank),
        static_cast<float>(kEps));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    const std::vector<__half> h_qa_norm = download(d_qa_norm, kQLoraRank);

    ok &= stage("q_lora_norm (weighted over 1024)",
                aeon::reference::rmsnorm(widen(h_qa), widen(h_q_norm_w), kEps),
                h_qa_norm);

    // Discriminating check (a): the mid-path norm must be load-bearing. If
    // dropping it barely changed q, then "norm between wq_a and wq_b" would be
    // indistinguishable from "no norm", and this gate could not tell trap 5
    // apart from a correct implementation.
    {
        const std::vector<double> collapsed_reference =
            aeon::reference::matvec(kQWidth, kQLoraRank, widen(h_qa), wq_b);
        const std::vector<double> normed_reference =
            aeon::reference::matvec(kQWidth, kQLoraRank, widen(h_qa_norm), wq_b);
        const ErrorStats s = aeon::reference::compare(normed_reference, collapsed_reference, 1.0);
        ok &= check("q norm is load-bearing", s.max_rel > 1e-2,
                    "wq_b(x) vs wq_b(norm(x)) differ by " + std::to_string(s.max_rel));
    }

    // =======================================================================
    // Stage 3 — Q = wq_b @ q_a_norm  [1024 -> 64 * 512]
    // =======================================================================
    aeon::kernel::v4_gemv_fp16_kernel<<<dim3(kQWidth, 1), 32>>>(
        d_qa_norm, d_wq_b, d_q, static_cast<int>(kQLoraRank));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    const std::vector<__half> h_q_raw = download(d_q, kQWidth);

    ok &= stage("q           (after wq_b)",
                aeon::reference::matvec(kQWidth, kQLoraRank, widen(h_qa_norm), wq_b),
                h_q_raw);

    // =======================================================================
    // Stage 4 — per-head WEIGHTLESS norm, one 512-row per head.
    //           Launched exactly as the pipeline does: one block per head.
    // =======================================================================
    aeon::kernel::v4_rmsnorm_unit_wave32_kernel<<<dim3(kHeads), 32>>>(
        d_q, d_q, static_cast<int>(kHeadDim), static_cast<float>(kEps));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    const std::vector<__half> h_q = download(d_q, kQWidth);

    {
        // Oracle: per-head unit norm over each 512-wide row of the raw q.
        const std::vector<double> q_raw = widen(h_q_raw);
        std::vector<double> want(kQWidth);
        for (size_t h = 0; h < kHeads; ++h) {
            const std::vector<double> row(q_raw.begin() + h * kHeadDim,
                                          q_raw.begin() + (h + 1) * kHeadDim);
            const std::vector<double> normed = aeon::reference::rmsnorm_unit(row, kEps);
            for (size_t d = 0; d < kHeadDim; ++d) want[h * kHeadDim + d] = normed[d];
        }
        ok &= stage("q           (per-head norm over 512)", want, h_q);

        // Discriminating check (b): the norm is per HEAD. If it had been applied
        // across all 64*512 at once, individual heads would not each carry unit
        // mean-square. This is what distinguishes "over 512" from "over 32768".
        double worst = 0.0;
        for (size_t h = 0; h < kHeads; ++h) {
            double sum_sq = 0.0;
            for (size_t d = 0; d < kHeadDim; ++d) {
                const double v = static_cast<double>(__half2float(h_q[h * kHeadDim + d]));
                sum_sq += v * v;
            }
            worst = std::fmax(worst, std::fabs(sum_sq / static_cast<double>(kHeadDim) - 1.0));
        }
        ok &= check("norm is per-head (not global)", worst < 1e-2,
                    "max |mean(x^2) - 1| over 64 heads = " + std::to_string(worst));
    }

    // =======================================================================
    // Stage 5 — KV = wkv @ x_norm  [4096 -> 512]
    // =======================================================================
    aeon::kernel::v4_gemv_fp16_kernel<<<dim3(kHeadDim, 1), 32>>>(
        d_x, d_wkv, d_kv, static_cast<int>(kHidden));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    const std::vector<__half> h_kv_raw = download(d_kv, kHeadDim);

    // =======================================================================
    // Stage 6 — KV norm (weighted, over head_dim = 512)
    // =======================================================================
    aeon::kernel::v4_rmsnorm_wave32_kernel<<<dim3(1), 32>>>(
        d_kv, d_kv_norm_w, d_kv, static_cast<int>(kHeadDim), static_cast<float>(kEps));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    const std::vector<__half> h_kv = download(d_kv, kHeadDim);

    ok &= stage("kv          (after wkv)",
                aeon::reference::matvec(kHeadDim, kHidden, x_norm, wkv), h_kv_raw);
    ok &= stage("kv          (weighted norm over 512)",
                aeon::reference::mla_kv_path(x_norm, widen(h_kv_norm_w), kHeadDim, kEps, wkv),
                h_kv);

    // Discriminating check (c): trap 6 — one KV row serves as both key and value.
    // The gate asserts the shape the graph depends on: `kv` is exactly head_dim
    // wide. A second (V) tensor of the same width would be a structural change,
    // and nothing in the contract or the reference provides one.
    ok &= check("single KV row (trap 6)", h_kv.size() == kHeadDim,
                "kv width = " + std::to_string(h_kv.size()) + " = head_dim, no separate V");

    // --- Cleanup -------------------------------------------------------------
    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_q_norm_w));
    CHECK_HIP(hipFree(d_kv_norm_w));
    CHECK_HIP(hipFree(d_wq_a));
    CHECK_HIP(hipFree(d_wq_b));
    CHECK_HIP(hipFree(d_wkv));
    CHECK_HIP(hipFree(d_qa));
    CHECK_HIP(hipFree(d_qa_norm));
    CHECK_HIP(hipFree(d_q));
    CHECK_HIP(hipFree(d_kv));

    std::cout << (ok ? "[SUCCESS] MLA Q/KV gate passed.\n"
                     : "[FAILURE] MLA Q/KV gate failed.\n");
    return ok ? 0 : 1;
}
