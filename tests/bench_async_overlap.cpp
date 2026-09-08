#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
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

constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = 64;
constexpr uint32_t BLOCK_K = 16;
constexpr uint32_t WARPS_M = BLOCK_M / WMMA_M;
constexpr uint32_t WARPS_N = BLOCK_N / WMMA_N;
constexpr uint32_t THREADS_PER_BLOCK = WARPS_M * WARPS_N * 32;

// Reused tiled GEMM kernel for the transfer-overlap benchmark.
__global__ void wmma_gemm_block_tiled_kernel(const half* __restrict__ a,
                                             const half* __restrict__ b,
                                             float* __restrict__ d,
                                             uint32_t M,
                                             uint32_t N,
                                             uint32_t K) {
    const uint32_t block_row = blockIdx.y * BLOCK_M;
    const uint32_t block_col = blockIdx.x * BLOCK_N;

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t warp_row = (warp_id / WARPS_N) * WMMA_M;
    const uint32_t warp_col = (warp_id % WARPS_N) * WMMA_N;

    const uint32_t global_row = block_row + warp_row;
    const uint32_t global_col = block_col + warp_col;

    __shared__ half lds_a[BLOCK_M][BLOCK_K];
    __shared__ half lds_b[BLOCK_N][BLOCK_K];

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    fill_fragment(frag_c, 0.0f);

    const uint32_t tid = threadIdx.x;

    for (uint32_t k_step = 0; k_step < K; k_step += BLOCK_K) {
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

        #pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            uint32_t elem_idx = tid * 2 + i;
            uint32_t r = elem_idx / BLOCK_K;
            uint32_t c = elem_idx % BLOCK_K;
            uint32_t gc = block_col + r;
            uint32_t gk = k_step + c;
            if (gc < N && gk < K) {
                lds_b[r][c] = b[gc * K + gk];
            } else {
                lds_b[r][c] = __float2half(0.0f);
            }
        }

        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_b;

        load_matrix_sync(frag_a, &lds_a[warp_row][0], BLOCK_K);
        load_matrix_sync(frag_b, &lds_b[warp_col][0], BLOCK_K);

        mma_sync(frag_c, frag_a, frag_b, frag_c);

        __syncthreads();
    }

    if (global_row < M && global_col < N) {
        store_matrix_sync(&d[global_row * N + global_col], frag_c, N, mem_row_major);
    }
}

int main() {
    std::cout << "====================================================================" << std::endl;
    std::cout << "  Project Aeon — Concurrent Compute + SDMA Overlap Benchmark" << std::endl;
    std::cout << "====================================================================" << std::endl;

    CHECK_HIP(hipSetDevice(0));
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "[Device] " << prop.name << " (Async Engine Count: " << prop.asyncEngineCount << ")\n";

    // 1. Setup GEMM Compute Resources on stream_compute
    constexpr uint32_t M = 2048, N = 2048, K = 2048;
    const size_t a_size = (size_t)M * K;
    const size_t b_size = (size_t)K * N;
    const size_t d_size = (size_t)M * N;

    half* d_a = nullptr;
    half* d_b = nullptr;
    float* d_d = nullptr;

    CHECK_HIP(hipMalloc(&d_a, a_size * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_b, b_size * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_d, d_size * sizeof(float)));

    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M, 1);

    // 2. Setup SDMA Transfer Resources on stream_transfer
    // Typical MoE expert size at 4-bit / 8-bit: 100 MB payload
    constexpr size_t TRANSFER_BYTES = 100 * 1024 * 1024; // 100 MB
    void* h_pinned_staging = nullptr;
    void* d_transfer_dest = nullptr;

    // Use page-locked pinned host memory (required for hardware SDMA async transfers)
    CHECK_HIP(hipHostMalloc(&h_pinned_staging, TRANSFER_BYTES, hipHostMallocDefault));
    CHECK_HIP(hipMalloc(&d_transfer_dest, TRANSFER_BYTES));

    // Fill pinned buffer with mock data
    std::memset(h_pinned_staging, 0x5C, TRANSFER_BYTES);

    // 3. Create two independent, non-blocking HIP streams
    hipStream_t stream_compute, stream_transfer;
    CHECK_HIP(hipStreamCreateWithFlags(&stream_compute, hipStreamNonBlocking));
    CHECK_HIP(hipStreamCreateWithFlags(&stream_transfer, hipStreamNonBlocking));

    // Warm-up both streams
    hipLaunchKernelGGL(wmma_gemm_block_tiled_kernel, grid, block, 0, stream_compute, d_a, d_b, d_d, M, N, K);
    CHECK_HIP(hipMemcpyAsync(d_transfer_dest, h_pinned_staging, TRANSFER_BYTES, hipMemcpyHostToDevice, stream_transfer));
    CHECK_HIP(hipStreamSynchronize(stream_compute));
    CHECK_HIP(hipStreamSynchronize(stream_transfer));

    constexpr int COMPUTE_REPEATS = 30;

    // -------------------------------------------------------------
    // Phase A: Measure Baseline Compute Latency (Isolated)
    // -------------------------------------------------------------
    std::cout << "\n[Test A] Measuring Baseline Compute Latency (Isolated, " << COMPUTE_REPEATS << " GEMMs)..." << std::endl;
    hipEvent_t start_comp, stop_comp;
    CHECK_HIP(hipEventCreate(&start_comp));
    CHECK_HIP(hipEventCreate(&stop_comp));

    CHECK_HIP(hipEventRecord(start_comp, stream_compute));
    for (int i = 0; i < COMPUTE_REPEATS; ++i) {
        hipLaunchKernelGGL(wmma_gemm_block_tiled_kernel, grid, block, 0, stream_compute, d_a, d_b, d_d, M, N, K);
    }
    CHECK_HIP(hipEventRecord(stop_comp, stream_compute));
    CHECK_HIP(hipStreamSynchronize(stream_compute));

    float baseline_compute_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&baseline_compute_ms, start_comp, stop_comp));
    float avg_baseline_gemm_ms = baseline_compute_ms / COMPUTE_REPEATS;
    std::cout << "  - Total Compute Time: " << baseline_compute_ms << " ms\n";
    std::cout << "  - Average per GEMM  : " << avg_baseline_gemm_ms << " ms ("
              << (avg_baseline_gemm_ms * 1000.0) << " us)\n";

    // -------------------------------------------------------------
    // Phase B: Measure Baseline SDMA Transfer Latency (Isolated)
    // -------------------------------------------------------------
    std::cout << "\n[Test B] Measuring Isolated SDMA PCIe Transfer (100 MB H2D)..." << std::endl;
    hipEvent_t start_xfer, stop_xfer;
    CHECK_HIP(hipEventCreate(&start_xfer));
    CHECK_HIP(hipEventCreate(&stop_xfer));

    CHECK_HIP(hipEventRecord(start_xfer, stream_transfer));
    CHECK_HIP(hipMemcpyAsync(d_transfer_dest, h_pinned_staging, TRANSFER_BYTES, hipMemcpyHostToDevice, stream_transfer));
    CHECK_HIP(hipEventRecord(stop_xfer, stream_transfer));
    CHECK_HIP(hipStreamSynchronize(stream_transfer));

    float transfer_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&transfer_ms, start_xfer, stop_xfer));
    double transfer_gb = (double)TRANSFER_BYTES / (1024.0 * 1024.0 * 1024.0);
    double pcie_bandwidth_gb_s = transfer_gb / (transfer_ms / 1000.0);
    std::cout << "  - Transfer Duration : " << transfer_ms << " ms\n";
    std::cout << "  - PCIe SDMA Speed   : " << pcie_bandwidth_gb_s << " GB/s\n";

    // -------------------------------------------------------------
    // Phase C: Concurrent Compute + Background SDMA Transfer
    // -------------------------------------------------------------
    std::cout << "\n[Test C] Running Concurrent Compute + Background SDMA Streaming..." << std::endl;
    // Launch background transfers continuously while compute is running
    constexpr int XFER_REPEATS = 5; // 5 x 100 MB = 500 MB streamed during compute

    hipEvent_t start_concurrent_comp, stop_concurrent_comp;
    CHECK_HIP(hipEventCreate(&start_concurrent_comp));
    CHECK_HIP(hipEventCreate(&stop_concurrent_comp));

    // Concurrently record compute duration while transfers flood the PCIe SDMA engine
    CHECK_HIP(hipEventRecord(start_concurrent_comp, stream_compute));

    // Fire compute kernels on stream_compute
    for (int i = 0; i < COMPUTE_REPEATS; ++i) {
        hipLaunchKernelGGL(wmma_gemm_block_tiled_kernel, grid, block, 0, stream_compute, d_a, d_b, d_d, M, N, K);
    }
    CHECK_HIP(hipEventRecord(stop_concurrent_comp, stream_compute));

    // Concurrently fire DMA transfers on stream_transfer
    for (int i = 0; i < XFER_REPEATS; ++i) {
        CHECK_HIP(hipMemcpyAsync(d_transfer_dest, h_pinned_staging, TRANSFER_BYTES, hipMemcpyHostToDevice, stream_transfer));
    }

    // Wait for both streams to complete
    CHECK_HIP(hipStreamSynchronize(stream_compute));
    CHECK_HIP(hipStreamSynchronize(stream_transfer));

    float concurrent_compute_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&concurrent_compute_ms, start_concurrent_comp, stop_concurrent_comp));
    float avg_concurrent_gemm_ms = concurrent_compute_ms / COMPUTE_REPEATS;

    float slowdown_percent = ((avg_concurrent_gemm_ms - avg_baseline_gemm_ms) / avg_baseline_gemm_ms) * 100.0f;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  - Total Concurrent Compute Time : " << concurrent_compute_ms << " ms\n";
    std::cout << "  - Average per Concurrent GEMM   : " << avg_concurrent_gemm_ms << " ms ("
              << (avg_concurrent_gemm_ms * 1000.0) << " us)\n";
    std::cout << "  - Jitter / Overhead             : " << (slowdown_percent > 0 ? "+" : "")
              << slowdown_percent << " %\n";

    std::cout << "--------------------------------------------------------------------" << std::endl;
    // If slowdown is less than 10%, SDMA and compute are truly orthogonal without driver serialization
    if (slowdown_percent < 10.0f) {
        std::cout << ">>> VERIFICATION PASSED: Hardware SDMA & WMMA compute are non-blocking! <<<" << std::endl;
        std::cout << "====================================================================" << std::endl;
    } else {
        std::cout << ">>> NOTICE: Slowdown of " << slowdown_percent << "% observed under saturated PCIe traffic. <<<" << std::endl;
        std::cout << "====================================================================" << std::endl;
    }

    // Cleanup
    CHECK_HIP(hipEventDestroy(start_comp));
    CHECK_HIP(hipEventDestroy(stop_comp));
    CHECK_HIP(hipEventDestroy(start_xfer));
    CHECK_HIP(hipEventDestroy(stop_xfer));
    CHECK_HIP(hipEventDestroy(start_concurrent_comp));
    CHECK_HIP(hipEventDestroy(stop_concurrent_comp));

    CHECK_HIP(hipStreamDestroy(stream_compute));
    CHECK_HIP(hipStreamDestroy(stream_transfer));

    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_d));
    CHECK_HIP(hipFree(d_transfer_dest));
    CHECK_HIP(hipHostFree(h_pinned_staging));

    return 0;
}
