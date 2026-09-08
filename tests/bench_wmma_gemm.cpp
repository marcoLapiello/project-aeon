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

// Tile configuration
constexpr uint32_t WMMA_M = 16;
constexpr uint32_t WMMA_N = 16;
constexpr uint32_t WMMA_K = 16;

// Block tile: 64x64 output per thread block, stepping by 16 along K
// Each block has (64/16) * (64/16) = 4 * 4 = 16 warps = 16 * 32 = 512 threads
constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = 64;
constexpr uint32_t BLOCK_K = 16;

constexpr uint32_t WARPS_M = BLOCK_M / WMMA_M; // 4 warps
constexpr uint32_t WARPS_N = BLOCK_N / WMMA_N; // 4 warps
constexpr uint32_t THREADS_PER_BLOCK = WARPS_M * WARPS_N * 32; // 512 threads

// 2D Block-Tiled Wave32 WMMA GEMM: D = alpha * (A * B) + beta * C
// A: M x K (row-major)
// B: K x N (col-major, i.e. leading dimension K, column index selects column of N)
// D: M x N (row-major)
__global__ void wmma_gemm_block_tiled_kernel(const half* __restrict__ a,
                                             const half* __restrict__ b,
                                             float* __restrict__ d,
                                             uint32_t M,
                                             uint32_t N,
                                             uint32_t K) {
    // 2D block coordinate
    const uint32_t block_row = blockIdx.y * BLOCK_M;
    const uint32_t block_col = blockIdx.x * BLOCK_N;

    // Warp coordinates inside this thread block
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t warp_row = (warp_id / WARPS_N) * WMMA_M;
    const uint32_t warp_col = (warp_id % WARPS_N) * WMMA_N;

    // Global matrix row & col for this warp
    const uint32_t global_row = block_row + warp_row;
    const uint32_t global_col = block_col + warp_col;

    // Shared memory staging for LDS-assisted tile loading
    __shared__ half lds_a[BLOCK_M][BLOCK_K];
    __shared__ half lds_b[BLOCK_N][BLOCK_K]; // col-major in global -> stored transposed for coalesced access

    // Accumulator fragment for this warp's 16x16 tile
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    fill_fragment(frag_c, 0.0f);

    // Number of elements to load per thread
    // Total elements in lds_a = BLOCK_M * BLOCK_K = 64 * 16 = 1024
    // With 512 threads, each thread loads 1024 / 512 = 2 elements
    const uint32_t tid = threadIdx.x;

    // Main K-loop over tiles of size BLOCK_K (16)
    for (uint32_t k_step = 0; k_step < K; k_step += BLOCK_K) {
        // Cooperative load A into LDS: A is (M x K), row-major
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            uint32_t elem_idx = tid * 2 + i;
            uint32_t r = elem_idx / BLOCK_K;
            uint32_t c = elem_idx % BLOCK_K;
            uint32_t gr = block_row + r;
            uint32_t gc = k_step + c;
            if (gr < M && gc < K) {
                lds_a[r][c] = a[gr * K + gc];
            } else {
                lds_a[r][c] = __float2half(0.0f);
            }
        }

        // Cooperative load B into LDS: B is (K x N), stored col-major with stride K
        // b[col * K + k_elem]
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            uint32_t elem_idx = tid * 2 + i;
            uint32_t r = elem_idx / BLOCK_K; // col in B (0..63)
            uint32_t c = elem_idx % BLOCK_K; // k in B (0..15)
            uint32_t gc = block_col + r;
            uint32_t gk = k_step + c;
            if (gc < N && gk < K) {
                lds_b[r][c] = b[gc * K + gk];
            } else {
                lds_b[r][c] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Each warp loads its 16x16 slice from LDS and performs WMMA
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_b;

        load_matrix_sync(frag_a, &lds_a[warp_row][0], BLOCK_K);
        load_matrix_sync(frag_b, &lds_b[warp_col][0], BLOCK_K);

        // Hardware Wave32 WMMA instruction
        mma_sync(frag_c, frag_a, frag_b, frag_c);

        __syncthreads();
    }

    // Store accumulator fragment to global memory
    if (global_row < M && global_col < N) {
        store_matrix_sync(&d[global_row * N + global_col], frag_c, N, mem_row_major);
    }
}

// CPU verification on a subset of the output
bool verify_output_subset(const std::vector<float>& h_a_f32,
                          const std::vector<float>& h_b_f32,
                          const std::vector<float>& h_d,
                          uint32_t M, uint32_t N, uint32_t K,
                          uint32_t samples = 64) {
    float max_diff = 0.0f;
    std::mt19937 rng(1337);
    std::uniform_int_distribution<uint32_t> dist_m(0, M - 1);
    std::uniform_int_distribution<uint32_t> dist_n(0, N - 1);

    for (uint32_t s = 0; s < samples; ++s) {
        uint32_t r = dist_m(rng);
        uint32_t c = dist_n(rng);

        float expected = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            expected += h_a_f32[r * K + k] * h_b_f32[c * K + k];
        }

        float actual = h_d[r * N + c];
        float diff = std::abs(actual - expected);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "[Verification] Checked " << samples << " random elements. Max Abs Diff: "
              << max_diff << (max_diff < 1e-2f ? " (PASS)" : " (WARN)") << "\n";
    return max_diff < 1e-2f;
}

int main(int argc, char** argv) {
    std::cout << "====================================================================" << std::endl;
    std::cout << "  Project Aeon — Tiled Wave32 WMMA GEMM Benchmark" << std::endl;
    std::cout << "====================================================================" << std::endl;

    CHECK_HIP(hipSetDevice(0));
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "[Device] " << prop.name << " (" << prop.gcnArchName << "), "
              << prop.multiProcessorCount << " CUs, "
              << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB VRAM\n";

    // Matrix dimensions (M, N, K)
    // Default to representative MoE expert GEMM size (e.g. 2048 x 2048 x 2048)
    uint32_t M = 2048;
    uint32_t N = 2048;
    uint32_t K = 2048;

    if (argc >= 4) {
        M = std::stoul(argv[1]);
        N = std::stoul(argv[2]);
        K = std::stoul(argv[3]);
    }

    std::cout << "[Problem] M = " << M << ", N = " << N << ", K = " << K << "\n";

    const size_t a_size = (size_t)M * K;
    const size_t b_size = (size_t)K * N;
    const size_t d_size = (size_t)M * N;

    // Allocate host memory
    std::vector<half> h_a(a_size);
    std::vector<half> h_b(b_size);
    std::vector<float> h_d(d_size, 0.0f);
    std::vector<float> h_a_f32(a_size);
    std::vector<float> h_b_f32(b_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::cout << "[Host] Initializing input matrices with pseudo-random FP16 values..." << std::endl;
    for (size_t i = 0; i < a_size; ++i) {
        float v = dist(rng);
        half h = __float2half(v);
        h_a[i] = h;
        h_a_f32[i] = __half2float(h);
    }
    for (size_t i = 0; i < b_size; ++i) {
        float v = dist(rng);
        half h = __float2half(v);
        h_b[i] = h;
        h_b_f32[i] = __half2float(h);
    }

    // Allocate device memory
    half* d_a = nullptr;
    half* d_b = nullptr;
    float* d_d = nullptr;

    CHECK_HIP(hipMalloc(&d_a, a_size * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_b, b_size * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_d, d_size * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_a, h_a.data(), a_size * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.data(), b_size * sizeof(half), hipMemcpyHostToDevice));

    // Execution grid
    dim3 block(THREADS_PER_BLOCK, 1, 1); // 512 threads (16 warps)
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M, 1);

    std::cout << "[Grid] Grid(" << grid.x << ", " << grid.y << "), Block(" << block.x << ") -> "
              << (grid.x * grid.y) << " thread blocks ("
              << (grid.x * grid.y * 16) << " Wave32 warps)\n";

    // Warm-up pass
    std::cout << "[Kernel] Running warmup iteration..." << std::endl;
    hipLaunchKernelGGL(wmma_gemm_block_tiled_kernel, grid, block, 0, 0, d_a, d_b, d_d, M, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    // Copy back and verify accuracy on subset
    CHECK_HIP(hipMemcpy(h_d.data(), d_d, d_size * sizeof(float), hipMemcpyDeviceToHost));
    verify_output_subset(h_a_f32, h_b_f32, h_d, M, N, K, 128);

    // Timing benchmark: 20 iterations
    constexpr int ITERS = 20;
    std::cout << "[Kernel] Benchmarking " << ITERS << " iterations..." << std::endl;

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start, 0));
    for (int it = 0; it < ITERS; ++it) {
        hipLaunchKernelGGL(wmma_gemm_block_tiled_kernel, grid, block, 0, 0, d_a, d_b, d_d, M, N, K);
    }
    CHECK_HIP(hipEventRecord(stop, 0));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;

    // FLOPs calculation: 2 * M * N * K floating point operations
    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (total_flops / (avg_ms / 1000.0)) / 1e12;

    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[Result] Average Latency : " << avg_ms << " ms per GEMM ("
              << (avg_ms * 1000.0) << " us)\n";
    std::cout << "[Result] Sustained TFLOPs: " << tflops << " TFLOP/s\n";
    std::cout << "====================================================================" << std::endl;

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_d));

    return 0;
}
