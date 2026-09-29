// -----------------------------------------------------------------------------
// Primitive gate: the RDNA3 Wave32 FP16 WMMA instruction and its operand lane
// map, versus a plain triple-loop matrix multiply.
//
// The value of this gate is that the reference has nothing to do with the
// fragment layout: it computes `D[m][n] = sum_k A[m][k] * B[k][n]` from the
// definition of a matrix product. Every reading of the lane map that is wrong
// still runs — the common failure computes `A x B^T`, which an identity-A test
// cannot see — so the operands here are dense and asymmetric, and the check is
// elementwise.
//
// Both directions of the map are exercised: the fragments are *built* from the
// documented lane assignment and the accumulator is *stored* through the
// documented (lane = N, slot = M parity-interleaved) one. A single mistake in
// either half moves elements and the comparison fails.
//
// K is split into two 16-wide tiles for one of the cases so the accumulator
// fragment is also checked across successive MMA accumulates, not just one.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "platform/rdna3/wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

constexpr int kM = aeon::rdna3::kWmmaTileM;
constexpr int kN = aeon::rdna3::kWmmaTileN;
constexpr int kK = aeon::rdna3::kWmmaTileK;

// A: [M, k_tiles*16] row-major. B: [k_tiles*16, N] row-major. C: [M, N] fp32.
__global__ void wmma_probe_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    int k_tiles
) {
    const int lane = threadIdx.x;
    const int m = aeon::rdna3::wmma_lane_axis(lane);
    const int n = aeon::rdna3::wmma_lane_axis(lane);
    const int k_total = k_tiles * kK;

    // One `[K][N]` tile in LDS at a time, in the row-major order the fragment
    // helper expects, so the helper itself is under test and not bypassed.
    __shared__ __half b_lds[kK * kN];

    aeon::rdna3::f32_vec8 accumulator = aeon::rdna3::wmma_zero_accumulator();
    for (int tile = 0; tile < k_tiles; ++tile) {
        for (int i = lane; i < kK * kN; i += aeon::rdna3::kWmmaLaneCount) {
            b_lds[i] = b[static_cast<size_t>(tile) * kK * kN + i];
        }
        __syncthreads();

        const aeon::rdna3::f16_vec16 a_fragment =
            aeon::rdna3::wmma_load_a_row(a + static_cast<size_t>(m) * k_total + tile * kK);
        const aeon::rdna3::f16_vec16 b_fragment =
            aeon::rdna3::wmma_load_b_from_lds(b_lds, kN, n);
        accumulator = aeon::rdna3::wmma_mma(a_fragment, b_fragment, accumulator);
        __syncthreads();
    }

    const int parity = aeon::rdna3::wmma_lane_parity(lane);
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        c[static_cast<size_t>(2 * slot + parity) * kN + n] = accumulator[slot];
    }
}

std::vector<float> reference_matrix_product(const std::vector<__half>& a,
                                            const std::vector<__half>& b,
                                            int k_total) {
    std::vector<float> out(static_cast<size_t>(kM) * kN, 0.0f);
    for (int m = 0; m < kM; ++m) {
        for (int n = 0; n < kN; ++n) {
            double sum = 0.0;
            for (int k = 0; k < k_total; ++k) {
                sum += static_cast<double>(__half2float(a[static_cast<size_t>(m) * k_total + k])) *
                       static_cast<double>(__half2float(b[static_cast<size_t>(k) * kN + n]));
            }
            out[static_cast<size_t>(m) * kN + n] = static_cast<float>(sum);
        }
    }
    return out;
}

bool run_case(const char* name, int k_tiles) {
    const int k_total = k_tiles * kK;

    // Dense, asymmetric, exactly representable operands: values are small
    // integers so the fp32 accumulate is exact and any mismatch is a mapping
    // error rather than rounding.
    std::vector<__half> a(static_cast<size_t>(kM) * k_total);
    std::vector<__half> b(static_cast<size_t>(k_total) * kN);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = __float2half(static_cast<float>(static_cast<int>(i % 7) - 3));
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = __float2half(static_cast<float>(static_cast<int>((i * 5) % 9) - 4) * 0.5f);
    }

    const std::vector<float> want = reference_matrix_product(a, b, k_total);

    __half* d_a = nullptr;
    __half* d_b = nullptr;
    float* d_c = nullptr;
    CHECK_HIP(hipMalloc(&d_a, a.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_b, b.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_c, static_cast<size_t>(kM) * kN * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_a, a.data(), a.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, b.data(), b.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_c, 0, static_cast<size_t>(kM) * kN * sizeof(float)));

    wmma_probe_kernel<<<1, aeon::rdna3::kWmmaLaneCount>>>(d_a, d_b, d_c, k_tiles);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<float> got(static_cast<size_t>(kM) * kN);
    CHECK_HIP(hipMemcpy(got.data(), d_c, got.size() * sizeof(float), hipMemcpyDeviceToHost));

    // The reference sums in double and the instruction accumulates in fp32, so a
    // small delta is expected. A *mapping* error instead moves whole entries, a
    // difference of the same order as the data itself, so an absolute bound scaled
    // to the data separates the two by several orders of magnitude.
    double scale = 0.0;
    for (float v : want) scale = std::max(scale, std::abs(static_cast<double>(v)));
    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(static_cast<double>(got[i]) - want[i]));
    }
    const double bound = 1e-5 * scale;
    const bool pass = max_abs <= bound;
    std::printf("  %-40s max_abs=%.3e (scaled bound %.3e)  %s\n",
                name, max_abs, bound, pass ? "PASS" : "FAIL");

    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_c));
    return pass;
}

} // namespace

int main() {
    std::cout << "[Gate] RDNA3 primitive: Wave32 FP16 WMMA fragment lane map\n";
    aeon::core::select_compute_device(true);

    bool ok = true;
    ok &= run_case("16x16x16, single K tile", 1);
    ok &= run_case("16x16x32, two accumulating K tiles", 2);
    ok &= run_case("16x16x256, sixteen accumulating K tiles", 16);

    if (!ok) {
        std::cout << "[FAIL] WMMA fragment lane map does not match the matrix product\n";
        return 1;
    }
    std::cout << "[PASS] WMMA fragment lane map matches the matrix product\n";
    return 0;
}
