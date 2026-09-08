#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_HIP(call)                                                    \
    do {                                                                   \
        hipError_t err = (call);                                           \
        if (err != hipSuccess) {                                           \
            std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__     \
                      << " -> " << hipGetErrorString(err) << std::endl;   \
            return 1;                                                      \
        }                                                                  \
    } while (0)

using namespace rocwmma;

constexpr uint32_t WMMA_M = 16;
constexpr uint32_t WMMA_N = 16;
constexpr uint32_t WMMA_K = 16;

// Single-wavefront (32 threads) WMMA kernel computing D = A * B + C
// A: 16x16 row-major, B: 16x16 col-major, C: 16x16 row-major, D: 16x16 row-major
__global__ void wmma_single_tile_kernel(const half* __restrict__ a,
                                       const half* __restrict__ b,
                                       const float* __restrict__ c,
                                       float* __restrict__ d) {
    // Exactly 32 threads execute as a cooperative wavefront
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_b;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;

    // Load matrices from global memory into register fragments
    load_matrix_sync(frag_a, a, WMMA_K);
    load_matrix_sync(frag_b, b, WMMA_K);
    load_matrix_sync(frag_c, c, WMMA_N, mem_row_major);

    // Hardware-accelerated Wave32 WMMA instruction: D = A * B + C
    mma_sync(frag_c, frag_a, frag_b, frag_c);

    // Store result matrix D from accumulator fragment back to global memory
    store_matrix_sync(d, frag_c, WMMA_N, mem_row_major);
}

// CPU reference matrix multiplication: D = A * B + C
void cpu_reference_gemm(const std::vector<float>& h_a,
                        const std::vector<float>& h_b,
                        const std::vector<float>& h_c,
                        std::vector<float>& h_d_ref) {
    for (uint32_t i = 0; i < WMMA_M; ++i) {
        for (uint32_t j = 0; j < WMMA_N; ++j) {
            float sum = h_c[i * WMMA_N + j];
            for (uint32_t k = 0; k < WMMA_K; ++k) {
                // A is row-major (i, k), B is col-major stored as (k, j) -> B_col[j * K + k]
                sum += h_a[i * WMMA_K + k] * h_b[j * WMMA_K + k];
            }
            h_d_ref[i * WMMA_N + j] = sum;
        }
    }
}

int main() {
    std::cout << "====================================================================" << std::endl;
    std::cout << "  Project Aeon — Single-Tile Wave32 WMMA Hardware Validation" << std::endl;
    std::cout << "====================================================================" << std::endl;

    // Set GPU 0
    CHECK_HIP(hipSetDevice(0));

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "[Device] Running on: " << prop.name << " (" << prop.gcnArchName << ")" << std::endl;
    std::cout << "[Device] Wavefront size: " << prop.warpSize << " threads" << std::endl;

    constexpr size_t a_elements = WMMA_M * WMMA_K;
    constexpr size_t b_elements = WMMA_K * WMMA_N;
    constexpr size_t c_elements = WMMA_M * WMMA_N;

    // Host data allocation
    std::vector<float> h_a_f32(a_elements);
    std::vector<float> h_b_f32(b_elements);
    std::vector<half>  h_a(a_elements);
    std::vector<half>  h_b(b_elements);
    std::vector<float> h_c(c_elements);
    std::vector<float> h_d(c_elements, 0.0f);
    std::vector<float> h_d_ref(c_elements, 0.0f);

    // Initialize with bounded pseudo-random values to avoid FP16 overflow
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    for (size_t i = 0; i < a_elements; ++i) {
        float val = dist(rng);
        h_a_f32[i] = val;
        h_a[i] = __float2half(val);
        // Synchronize FP32 reference with the actual half precision representation
        h_a_f32[i] = __half2float(h_a[i]);
    }
    for (size_t i = 0; i < b_elements; ++i) {
        float val = dist(rng);
        h_b_f32[i] = val;
        h_b[i] = __float2half(val);
        h_b_f32[i] = __half2float(h_b[i]);
    }
    for (size_t i = 0; i < c_elements; ++i) {
        h_c[i] = dist(rng);
    }

    // Compute CPU golden reference
    cpu_reference_gemm(h_a_f32, h_b_f32, h_c, h_d_ref);

    // Device memory allocation
    half* d_a = nullptr;
    half* d_b = nullptr;
    float* d_c = nullptr;
    float* d_d = nullptr;

    CHECK_HIP(hipMalloc(&d_a, a_elements * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_b, b_elements * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_c, c_elements * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_d, c_elements * sizeof(float)));

    // Copy to device
    CHECK_HIP(hipMemcpy(d_a, h_a.data(), a_elements * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.data(), b_elements * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_c, h_c.data(), c_elements * sizeof(float), hipMemcpyHostToDevice));

    // Launch exactly 1 wavefront of 32 threads (1 wave tile)
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);

    std::cout << "[Kernel] Launching wmma_single_tile_kernel<<<grid(1,1,1), block(32,1,1)>>>..." << std::endl;
    hipLaunchKernelGGL(wmma_single_tile_kernel, grid, block, 0, 0, d_a, d_b, d_c, d_d);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    // Copy result back
    CHECK_HIP(hipMemcpy(h_d.data(), d_d, c_elements * sizeof(float), hipMemcpyDeviceToHost));

    // Cleanup GPU buffers
    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_c));
    CHECK_HIP(hipFree(d_d));

    // Verification against CPU reference
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    for (size_t i = 0; i < c_elements; ++i) {
        float diff = std::abs(h_d[i] - h_d_ref[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        float ref_abs = std::abs(h_d_ref[i]);
        if (ref_abs > 1e-6f) {
            float rel = diff / ref_abs;
            if (rel > max_rel_diff) max_rel_diff = rel;
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "[Result] Max Absolute Error: " << max_abs_diff << std::endl;
    std::cout << "[Result] Max Relative Error: " << max_rel_diff << std::endl;

    // Display first 4x4 matrix corner preview
    std::cout << "\n[Matrix Output Preview: First 4x4 Sub-block]" << std::endl;
    std::cout << "Row | GPU Result           | CPU Reference" << std::endl;
    for (uint32_t r = 0; r < 4; ++r) {
        for (uint32_t c = 0; c < 4; ++c) {
            uint32_t idx = r * WMMA_N + c;
            std::cout << "(" << r << "," << c << "): "
                      << std::setw(10) << h_d[idx] << "  vs  "
                      << std::setw(10) << h_d_ref[idx] << std::endl;
        }
    }

    // Strict numerical tolerance check (epsilon < 1e-4)
    constexpr float TOLERANCE = 1e-4f;
    if (max_abs_diff < TOLERANCE) {
        std::cout << "--------------------------------------------------------------------" << std::endl;
        std::cout << ">>> VERIFICATION PASSED: RDNA3 Wave32 WMMA matches reference math! <<<" << std::endl;
        std::cout << "====================================================================" << std::endl;
        return 0;
    } else {
        std::cerr << "--------------------------------------------------------------------" << std::endl;
        std::cerr << ">>> VERIFICATION FAILED: Max diff " << max_abs_diff
                  << " exceeds tolerance " << TOLERANCE << "! <<<" << std::endl;
        std::cerr << "====================================================================" << std::endl;
        return 1;
    }
}
