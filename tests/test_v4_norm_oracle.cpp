// -----------------------------------------------------------------------------
// Tier-1 gate: Wave32 RMSNorm versus an independent fp64 reference.
//
// This is the first certified graph primitive of the rewrite, and it fixes the
// pattern the rest of Tier 1 follows:
//
//   1. THE REFERENCE IS INDEPENDENT. It lives in `reference/dsv4_oracle.hpp` and
//      shares no code with the kernel under test. The plan's anti-circularity
//      rule is explicit: an oracle derived from the kernel proves only
//      self-consistency, and that defect is why the previous implementation's
//      tests passed on wrong logic.
//   2. THE KERNEL IS THE KEPT ONE. `kernels/v4_norm.hpp`, not a local copy.
//   3. THE GATE ASSERTS A NUMBER. "It produced output" is not a gate; a stated
//      tolerance is.
//
// The oracle is fed the *same fp16 values* the kernel reads. Feeding it the
// un-rounded doubles would fold input quantization into the measured delta and
// make the gate measure the wrong quantity.
//
// Tolerance rationale. The output is fp16, whose relative resolution is 2^-11
// (~4.9e-4); the reference rounds nothing, so the expected delta is bounded by
// about one fp16 ulp of the largest output, plus the kernel's fp32 accumulation
// error over `dim` O(1) terms (negligible at dim = 4096). The gate therefore
// allows 2e-3 relative, measured against a floor of `1e-3 * max|output|`.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/v4_norm.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
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

constexpr double kTolAbs = 2e-3;
constexpr double kTolRel = 2e-3;

// Converts an fp16 buffer to double. Kept out of the oracle header so that the
// oracle stays free of device includes.
std::vector<double> widen(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<double>(__half2float(v[i]));
    }
    return out;
}

// Prints one gate line and returns whether it passed.
bool report(const char* label, const ErrorStats& s) {
    const bool pass = aeon::reference::within(s, kTolAbs, kTolRel);
    if (s.size_mismatch) {
        std::printf("  %-24s SIZE MISMATCH                        FAIL\n", label);
        return false;
    }
    std::printf("  %-24s max_abs=%.3e @%-5zu max_rel=%.3e @%-5zu rms=%.3e  %s\n",
                label,
                s.max_abs, s.max_abs_index,
                s.max_rel, s.max_rel_index,
                s.rms_abs,
                pass ? "PASS" : "FAIL");
    return pass;
}

// Validates the oracle itself, without involving any kernel. A constant input
// `c` reduces to `out = sign(c) * weight`, because mean(c^2) = c^2 and the eps is
// negligible against a unit-scale constant. If this fails, the oracle is wrong
// and no kernel result measured against it means anything.
bool oracle_self_check() {
    constexpr size_t kDim = 4096;
    constexpr double kEps = 1e-6;

    aeon::reference::Rng rng(0x0C0FFEEull);
    const std::vector<double> w = rng.vector_filled(kDim, 1.0);
    const std::vector<double> x(kDim, 2.0); // constant, positive

    const std::vector<double> got = aeon::reference::rmsnorm(x, w, kEps);
    std::vector<double> want(kDim);
    for (size_t i = 0; i < kDim; ++i) want[i] = w[i]; // c/|c| = 1

    const ErrorStats s = aeon::reference::compare(want, got);
    return report("oracle: constant input", s);
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: Wave32 RMSNorm vs independent fp64 reference\n";
    aeon::core::select_compute_device(true);

    constexpr int kRows = 4;
    constexpr int kDim = 4096;
    constexpr float kEps = 1e-6f;
    constexpr size_t kCount = static_cast<size_t>(kRows) * kDim;

    bool ok = true;

    // --- The oracle must be trustworthy before it is used as a yardstick ------
    ok &= oracle_self_check();

    // --- Deterministic inputs -------------------------------------------------
    aeon::reference::Rng rng(0xAE04D15Eull);
    const std::vector<double> x_src = rng.vector_filled(kCount, 1.0);
    const std::vector<double> w_src = rng.vector_filled(kDim, 1.0);

    std::vector<__half> h_x(kCount), h_w(kDim);
    for (size_t i = 0; i < kCount; ++i) h_x[i] = __float2half(static_cast<float>(x_src[i]));
    for (size_t i = 0; i < kDim; ++i)  h_w[i] = __float2half(static_cast<float>(w_src[i]));

    // The oracle sees the post-quantization values, exactly as the kernel does.
    const std::vector<double> x = widen(h_x);
    const std::vector<double> w = widen(h_w);

    // --- Device buffers -------------------------------------------------------
    __half* d_x = nullptr;
    __half* d_w = nullptr;
    __half* d_y = nullptr;
    CHECK_HIP(hipMalloc(&d_x, kCount * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_w, kDim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_y, kCount * sizeof(__half)));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), kCount * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_w, h_w.data(), kDim * sizeof(__half), hipMemcpyHostToDevice));

    std::vector<__half> h_y(kCount);

    // --- Weighted form (Step 2.1 attention RMSNorm) ---------------------------
    CHECK_HIP(hipMemset(d_y, 0, kCount * sizeof(__half)));
    aeon::kernel::v4_rmsnorm_wave32_kernel<<<kRows, 32>>>(d_x, d_w, d_y, kDim, kEps);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_y.data(), d_y, kCount * sizeof(__half), hipMemcpyDeviceToHost));

    std::vector<double> ref(kCount);
    for (int r = 0; r < kRows; ++r) {
        const std::vector<double> row_in(x.begin() + static_cast<size_t>(r) * kDim,
                                         x.begin() + static_cast<size_t>(r + 1) * kDim);
        const std::vector<double> row_out = aeon::reference::rmsnorm(row_in, w, kEps);
        for (int i = 0; i < kDim; ++i) ref[static_cast<size_t>(r) * kDim + i] = row_out[i];
    }
    ok &= report("rmsnorm (weighted)", aeon::reference::compare(ref, widen(h_y)));

    // --- Unit form ------------------------------------------------------------
    CHECK_HIP(hipMemset(d_y, 0, kCount * sizeof(__half)));
    aeon::kernel::v4_rmsnorm_unit_wave32_kernel<<<kRows, 32>>>(d_x, d_y, kDim, kEps);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_y.data(), d_y, kCount * sizeof(__half), hipMemcpyDeviceToHost));

    for (int r = 0; r < kRows; ++r) {
        const std::vector<double> row_in(x.begin() + static_cast<size_t>(r) * kDim,
                                         x.begin() + static_cast<size_t>(r + 1) * kDim);
        const std::vector<double> row_out = aeon::reference::rmsnorm_unit(row_in, kEps);
        for (int i = 0; i < kDim; ++i) ref[static_cast<size_t>(r) * kDim + i] = row_out[i];
    }
    ok &= report("rmsnorm (unit)", aeon::reference::compare(ref, widen(h_y)));

    // --- Low-magnitude input: where `eps` is load-bearing ---------------------
    //
    // Found by mutation testing, not by review. With unit-scale input the eps
    // shifts `inv_rms` by ~5e-7 relative, which is *below* one fp16 ulp, so a
    // kernel with the eps deleted is bit-identical to a correct one and this gate
    // passed it. The eps exists for exactly the case the first version did not
    // test: an input whose RMS is comparable to `eps` itself.
    //
    // Scaling by 2^-10 makes mean(x^2) ~ 1e-6 = eps, so `rsqrt(rms^2)` and
    // `rsqrt(rms^2 + eps)` differ by a factor of ~sqrt(2). That is not a
    // tolerance question; it is a 40% error, and it is caught outright.
    {
        const double kShrink = 1.0 / 1024.0;
        std::vector<__half> h_small(kCount);
        for (size_t i = 0; i < kCount; ++i) {
            h_small[i] = __float2half(static_cast<float>(x_src[i] * kShrink));
        }
        const std::vector<double> x_small = widen(h_small);

        CHECK_HIP(hipMemcpy(d_x, h_small.data(), kCount * sizeof(__half),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(d_y, 0, kCount * sizeof(__half)));
        aeon::kernel::v4_rmsnorm_wave32_kernel<<<kRows, 32>>>(d_x, d_w, d_y, kDim, kEps);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_y.data(), d_y, kCount * sizeof(__half),
                            hipMemcpyDeviceToHost));

        for (int r = 0; r < kRows; ++r) {
            const std::vector<double> row_in(
                x_small.begin() + static_cast<size_t>(r) * kDim,
                x_small.begin() + static_cast<size_t>(r + 1) * kDim);
            const std::vector<double> row_out = aeon::reference::rmsnorm(row_in, w, kEps);
            for (int i = 0; i < kDim; ++i) ref[static_cast<size_t>(r) * kDim + i] = row_out[i];
        }
        ok &= report("rmsnorm (weighted, low RMS: eps is load-bearing)",
                     aeon::reference::compare(ref, widen(h_y)));

        // And confirm the regime really is discriminating: the eps-free oracle
        // must differ from the eps-full one by far more than a rounding step.
        std::vector<double> no_eps(kCount);
        for (int r = 0; r < kRows; ++r) {
            const std::vector<double> row_in(
                x_small.begin() + static_cast<size_t>(r) * kDim,
                x_small.begin() + static_cast<size_t>(r + 1) * kDim);
            const std::vector<double> row_out = aeon::reference::rmsnorm(row_in, w, 0.0);
            for (int i = 0; i < kDim; ++i) {
                no_eps[static_cast<size_t>(r) * kDim + i] = row_out[i];
            }
        }
        const ErrorStats s = aeon::reference::compare(ref, no_eps);
        {
            // A LARGE difference is the pass condition here: this line exists to
            // prove the regime above can see the eps at all.
            const bool discriminating = s.max_abs > 1e-3;
            std::printf("  %-58s %-24s %s\n",
                        "   (eps-free reading is distinguishable here)",
                        ("max_abs = " + std::to_string(s.max_abs)).c_str(),
                        discriminating ? "PASS" : "FAIL");
            ok &= discriminating;
        }
    }

    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_w));
    CHECK_HIP(hipFree(d_y));

    std::cout << (ok ? "[SUCCESS] RMSNorm gate passed.\n"
                     : "[FAILURE] RMSNorm gate failed.\n");
    return ok ? 0 : 1;
}
