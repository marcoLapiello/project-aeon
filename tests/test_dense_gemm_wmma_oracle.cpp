// -----------------------------------------------------------------------------
// Gate: the RDNA3 dense fp16 WMMA GEMM versus a plain triple-loop reference in double.
//
// The reference is the definition `Y[t][n] = sum_k X[t][k] * W[n][k]`, independent of
// the fragment mapping. Shapes cover the chunk's real projections (K = 4096 / 1024),
// a ragged token count, a ragged output width, and padded row pitches on both sides.
// Timing against the one-token-per-block GEMV the chunk loop uses today is printed for
// information; it is not a pass condition.
// -----------------------------------------------------------------------------

#include "platform/dense_gemm.hpp"
#include "platform/ops/gemv.hpp"
#include "platform/device.hpp"
#include "test_device.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
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

struct Shape {
    const char* name;
    int tokens, in_dim, out_dim, x_stride, y_stride;
};

std::vector<__half> random_half(size_t count, float sigma, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, sigma);
    std::vector<__half> out(count);
    for (auto& v : out) v = __float2half(dist(rng));
    return out;
}

bool run_case(const Shape& s) {
    const auto x = random_half(static_cast<size_t>(s.tokens) * s.x_stride, 1.0f, 1);
    const auto w = random_half(static_cast<size_t>(s.out_dim) * s.in_dim, 0.02f, 2);

    std::vector<double> want(static_cast<size_t>(s.tokens) * s.out_dim);
    double sum_sq = 0.0;
    for (int t = 0; t < s.tokens; ++t) {
        for (int n = 0; n < s.out_dim; ++n) {
            double acc = 0.0;
            for (int k = 0; k < s.in_dim; ++k) {
                acc += static_cast<double>(__half2float(x[static_cast<size_t>(t) * s.x_stride + k])) *
                       static_cast<double>(__half2float(w[static_cast<size_t>(n) * s.in_dim + k]));
            }
            want[static_cast<size_t>(t) * s.out_dim + n] = acc;
            sum_sq += acc * acc;
        }
    }
    const double rms = std::sqrt(sum_sq / static_cast<double>(want.size()));

    __half *d_x = nullptr, *d_w = nullptr, *d_y = nullptr, *d_g = nullptr;
    const size_t y_elems = static_cast<size_t>(s.tokens) * s.y_stride;
    CHECK_HIP(hipMalloc(&d_x, x.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_w, w.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_y, y_elems * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_g, static_cast<size_t>(s.tokens) * s.out_dim * sizeof(__half)));
    CHECK_HIP(hipMemcpy(d_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_w, w.data(), w.size() * sizeof(__half), hipMemcpyHostToDevice));
    // A sentinel proves the store mask leaves the pitch padding untouched.
    CHECK_HIP(hipMemset(d_y, 0x7B, y_elems * sizeof(__half)));

    aeon::dispatch_dense_gemm_fp16(d_x, d_w, d_y, s.tokens, s.in_dim, s.out_dim,
                                   s.x_stride, s.y_stride, nullptr);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<__half> got(y_elems);
    CHECK_HIP(hipMemcpy(got.data(), d_y, y_elems * sizeof(__half), hipMemcpyDeviceToHost));

    int bad = 0;
    double worst = 0.0;
    for (int t = 0; t < s.tokens; ++t) {
        for (int n = 0; n < s.y_stride; ++n) {
            const __half g = got[static_cast<size_t>(t) * s.y_stride + n];
            if (n >= s.out_dim) {
                uint16_t bits = 0;
                std::memcpy(&bits, &g, sizeof(bits));
                if (bits != 0x7B7B) ++bad;
                continue;
            }
            const double ref = want[static_cast<size_t>(t) * s.out_dim + n];
            const double err = std::fabs(static_cast<double>(__half2float(g)) - ref);
            const double tol = 1e-3 * std::fabs(ref) + 1e-3 * rms;
            worst = std::max(worst, err / (std::fabs(ref) + rms));
            if (!(err <= tol)) ++bad;
        }
    }

    float gemv_ms = 0.0f, wmma_ms = 0.0f;
    if (s.tokens >= 16 && s.x_stride == s.in_dim && s.in_dim % 8 == 0) {
        hipEvent_t a, b;
        CHECK_HIP(hipEventCreate(&a));
        CHECK_HIP(hipEventCreate(&b));
        constexpr int kIters = 10;
        auto time = [&](auto&& launch) {
            launch();
            CHECK_HIP(hipDeviceSynchronize());
            CHECK_HIP(hipEventRecord(a));
            for (int i = 0; i < kIters; ++i) launch();
            CHECK_HIP(hipEventRecord(b));
            CHECK_HIP(hipEventSynchronize(b));
            float ms = 0.0f;
            CHECK_HIP(hipEventElapsedTime(&ms, a, b));
            return ms / kIters;
        };
        gemv_ms = time([&] {
            aeon::kernel::gemv_fp16_vec8_kernel<<<dim3(s.out_dim, s.tokens), 32>>>(
                d_x, d_w, d_g, s.in_dim, s.x_stride);
        });
        wmma_ms = time([&] {
            aeon::dispatch_dense_gemm_fp16(d_x, d_w, d_y, s.tokens, s.in_dim, s.out_dim,
                                           s.x_stride, s.y_stride, nullptr);
        });
        CHECK_HIP(hipEventDestroy(a));
        CHECK_HIP(hipEventDestroy(b));
    }

    std::printf("  %-28s T=%-4d K=%-5d N=%-6d  bad=%d  worst_rel=%.2e", s.name, s.tokens,
                s.in_dim, s.out_dim, bad, worst);
    if (wmma_ms > 0.0f) {
        std::printf("  gemv=%.3f ms  wmma=%.3f ms  (%.1fx)", gemv_ms, wmma_ms, gemv_ms / wmma_ms);
    }
    std::printf("\n");

    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_w));
    CHECK_HIP(hipFree(d_y));
    CHECK_HIP(hipFree(d_g));
    return bad == 0;
}

} // namespace

int main() {
    std::cout << "[Gate] dense fp16 WMMA GEMM vs definition (double)\n";
    aeon::test::select_test_device(true);

    const Shape shapes[] = {
        {"single token", 1, 4096, 1024, 4096, 1024},
        {"one tile, narrow out", 16, 4096, 64, 4096, 64},
        {"ragged tokens", 37, 4096, 1024, 4096, 1024},
        {"ragged out + padded pitch", 21, 48, 40, 56, 48},
        {"chunk, hidden -> q_lora", 256, 4096, 1024, 4096, 1024},
        {"chunk, q_lora -> wide", 256, 1024, 16384, 1024, 16384},
    };
    bool ok = true;
    for (const Shape& s : shapes) ok = run_case(s) && ok;

    std::cout << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
