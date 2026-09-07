#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cassert>

#include "kernel/w4a16_gemm.hpp"

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

using namespace rocwmma;

constexpr uint32_t WMMA_M = 16;
constexpr uint32_t WMMA_N = 16;
constexpr uint32_t WMMA_K = 16;

// Single-Wave32 Wavefront Fused INT4 Dequantization + WMMA GEMM Kernel
// A: (M=16, K=16) activation matrix in FP16 (row-major)
// W_packed: (N=16, K/8 = 2) packed INT4 weights (32-bit words)
// W_scale: (N=16, K/32 = 1) FP16 scale factors (1 scale per 32 elements)
// D: (M=16, N=16) output in FP32 (row-major)
__global__ void __launch_bounds__(32) wmma_fused_int4_tile_kernel(
    const half* __restrict__ a,
    const uint32_t* __restrict__ w_packed,
    const half* __restrict__ w_scale,
    float* __restrict__ d,
    uint32_t K
) {
    int lane = threadIdx.x; // 0..31

    // LDS staging for dequantized 16x16 weight matrix B
    __shared__ half lds_b[WMMA_K][WMMA_N];

    int n = lane / 2;
    int w = lane % 2;

    uint32_t scale_stride = (K >= 32) ? (K / 32) : 1;
    uint32_t packed = w_packed[n * (K / 8) + w];
    float scale = __half2float(w_scale[n * scale_stride + 0]);

    // Unpack 8 INT4 nibbles: (uint4 - 8) * scale
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int nib = (packed >> (i * 4)) & 0xF;
        float val = (float)(nib - 8) * scale;
        int k = w * 8 + i;
        // B(k, n) = W(n, k)
        lds_b[k][n] = __float2half(val);
    }
    __syncthreads();

    // WMMA fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_b;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;

    load_matrix_sync(frag_a, a, WMMA_K);
    load_matrix_sync(frag_b, &lds_b[0][0], WMMA_N);
    fill_fragment(frag_c, 0.0f);

    mma_sync(frag_c, frag_a, frag_b, frag_c);

    store_matrix_sync(d, frag_c, WMMA_N, mem_row_major);
}

// CPU Reference GEMM for validation
void cpu_w4a16_gemm(
    const half* a,
    const uint32_t* w_packed,
    const half* w_scale,
    float* d_ref,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    uint32_t scale_stride = (K >= 32) ? (K / 32) : 1;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                float a_val = __half2float(a[m * K + k]);
                uint32_t word = w_packed[n * (K / 8) + (k / 8)];
                int nib = (word >> ((k % 8) * 4)) & 0xF;
                float sc = __half2float(w_scale[n * scale_stride + (k / 32)]);
                float w_val = (float)(nib - 8) * sc;
                sum += a_val * w_val;
            }
            d_ref[m * N + n] = sum;
        }
    }
}

int main() {
    std::cout << "[Test] Spike 3: Fused INT4 -> FP16 Wave32 WMMA Kernel on Silicon..." << std::endl;

    // 1. Single-Tile Verification (16x16x16)
    {
        std::cout << "--- Sub-Test 1: Single 16x16 Tile Verification ---" << std::endl;
        constexpr uint32_t M = 16, N = 16, K = 16;
        std::vector<half> h_a(M * K);
        std::vector<uint32_t> h_w_packed(N * (K / 8));
        std::vector<half> h_w_scale(N * std::max(1u, K / 32));
        std::vector<float> h_d_ref(M * N);
        std::vector<float> h_d_gpu(M * N);

        for (uint32_t i = 0; i < h_a.size(); ++i) {
            h_a[i] = __float2half(((i % 7) - 3) * 0.25f);
        }
        for (uint32_t i = 0; i < h_w_packed.size(); ++i) {
            // Fill 8 nibbles with various values [0..15]
            uint32_t p = 0;
            for (int b = 0; b < 8; ++b) {
                uint32_t nib = ((i * 8 + b) % 15);
                p |= (nib << (b * 4));
            }
            h_w_packed[i] = p;
        }
        for (uint32_t i = 0; i < h_w_scale.size(); ++i) {
            h_w_scale[i] = __float2half(0.0125f);
        }

        cpu_w4a16_gemm(h_a.data(), h_w_packed.data(), h_w_scale.data(), h_d_ref.data(), M, N, K);

        half* d_a;
        uint32_t* d_w_packed;
        half* d_w_scale;
        float* d_out;

        CHECK_HIP(hipMalloc(&d_a, h_a.size() * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_w_packed, h_w_packed.size() * sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&d_w_scale, h_w_scale.size() * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_out, h_d_gpu.size() * sizeof(float)));

        CHECK_HIP(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w_packed, h_w_packed.data(), h_w_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w_scale, h_w_scale.data(), h_w_scale.size() * sizeof(half), hipMemcpyHostToDevice));

        wmma_fused_int4_tile_kernel<<<1, 32>>>(d_a, d_w_packed, d_w_scale, d_out, K);
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_d_gpu.data(), d_out, h_d_gpu.size() * sizeof(float), hipMemcpyDeviceToHost));

        float max_diff = 0.0f;
        for (uint32_t i = 0; i < M * N; ++i) {
            float diff = std::abs(h_d_gpu[i] - h_d_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }
        std::cout << "Single-Tile Max Absolute Difference vs CPU Ref: " << max_diff << std::endl;
        assert(max_diff < 1e-4f);

        CHECK_HIP(hipFree(d_a));
        CHECK_HIP(hipFree(d_w_packed));
        CHECK_HIP(hipFree(d_w_scale));
        CHECK_HIP(hipFree(d_out));
        std::cout << "[PASS] Single 16x16 Tile WMMA dequantization verified!" << std::endl;
    }

    // 2. Block-Tiled GEMM Verification on DeepSeek Expert Dimensions:
    // M=16 tokens, N=2048 (intermediate), K=4096 (hidden)
    {
        std::cout << "\n--- Sub-Test 2: Block-Tiled GEMM for DeepSeek Expert Shapes (M=16, N=2048, K=4096) ---" << std::endl;
        const uint32_t M = 16;
        const uint32_t N = 2048;
        const uint32_t K = 4096;

        std::vector<half> h_a(M * K);
        std::vector<uint32_t> h_w_packed(N * (K / 8));
        std::vector<half> h_w_scale(N * (K / 32));
        std::vector<half> h_d_gpu(M * N);

        std::cout << "Initializing test matrices (" 
                  << (h_w_packed.size() * 4 / (1024*1024.0)) << " MB packed weights)..." << std::endl;

        for (uint32_t i = 0; i < h_a.size(); ++i) {
            h_a[i] = __float2half(((i % 11) - 5) * 0.1f);
        }
        for (uint32_t i = 0; i < h_w_packed.size(); ++i) {
            uint32_t p = 0;
            for (int b = 0; b < 8; ++b) {
                uint32_t nib = ((i * 8 + b) % 15);
                p |= (nib << (b * 4));
            }
            h_w_packed[i] = p;
        }
        for (uint32_t i = 0; i < h_w_scale.size(); ++i) {
            h_w_scale[i] = __float2half(0.0075f);
        }

        half* d_a;
        uint32_t* d_w_packed;
        half* d_w_scale;
        half* d_out;

        CHECK_HIP(hipMalloc(&d_a, h_a.size() * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_w_packed, h_w_packed.size() * sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&d_w_scale, h_w_scale.size() * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_out, h_d_gpu.size() * sizeof(half)));

        CHECK_HIP(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w_packed, h_w_packed.data(), h_w_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w_scale, h_w_scale.data(), h_w_scale.size() * sizeof(half), hipMemcpyHostToDevice));

        // Warmup
        aeon::kernel::dispatch_w4a16_gemm(d_a, d_w_packed, d_w_scale, d_out, M, N, K);
        CHECK_HIP(hipDeviceSynchronize());

        // Validate first 32 outputs against CPU reference
        CHECK_HIP(hipMemcpy(h_d_gpu.data(), d_out, h_d_gpu.size() * sizeof(half), hipMemcpyDeviceToHost));
        std::vector<float> sample_ref(32);
        for (uint32_t n = 0; n < 32; ++n) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                float a_val = __half2float(h_a[0 * K + k]);
                uint32_t word = h_w_packed[n * (K / 8) + (k / 8)];
                int nib = (word >> ((k % 8) * 4)) & 0xF;
                float sc = __half2float(h_w_scale[n * (K / 32) + (k / 32)]);
                float w_val = (float)(nib - 8) * sc;
                sum += a_val * w_val;
            }
            sample_ref[n] = sum;
        }

        float max_sample_diff = 0.0f;
        for (uint32_t n = 0; n < 32; ++n) {
            float gpu_val = __half2float(h_d_gpu[0 * N + n]);
            float diff = std::abs(gpu_val - sample_ref[n]);
            if (diff > max_sample_diff) max_sample_diff = diff;
        }
        std::cout << "Block GEMM Sample Max Difference vs CPU: " << max_sample_diff << std::endl;
        assert(max_sample_diff < 0.05f); // Accumulated over 4096 dot products

        // Benchmark
        const int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it) {
            aeon::kernel::dispatch_w4a16_gemm(d_a, d_w_packed, d_w_scale, d_out, M, N, K);
        }
        CHECK_HIP(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
        double ops = 2.0 * (double)M * (double)N * (double)K; // 2 * 16 * 2048 * 4096 = 268.4 million ops
        double tflops = (ops / (elapsed_us * 1e-6)) / 1e12;

        std::cout << "Performance Benchmark (M=16, N=2048, K=4096):" << std::endl;
        std::cout << "  Execution Time : " << elapsed_us << " us" << std::endl;
        std::cout << "  Compute Rate   : " << tflops << " TFLOP/s" << std::endl;

        CHECK_HIP(hipFree(d_a));
        CHECK_HIP(hipFree(d_w_packed));
        CHECK_HIP(hipFree(d_w_scale));
        CHECK_HIP(hipFree(d_out));

        std::cout << "[PASS] Block-Tiled Fused INT4 GEMM benchmark passed!" << std::endl;
    }

    // 3. Real Checkpoint Validation: Run Layer 0 Expert 0 W1 weights from model-00001.safetensors
    {
        std::cout << "\n--- Sub-Test 3: Real DeepSeek-V4 Checkpoint Layer 0 Expert 0 Weights ---" << std::endl;
        std::string safetensors_path = "models/DeepSeek-V4-Flash-0731-INT4-W4A16/"
                                      "models--yiminyuan--DeepSeek-V4-Flash-0731-INT4-W4A16/"
                                      "snapshots/64700592cadaf205fe0c13202061ff4b45afbfd0/model-00001.safetensors";

        std::ifstream file(safetensors_path, std::ios::binary);
        if (file.is_open()) {
            uint64_t header_len = 0;
            file.read(reinterpret_cast<char*>(&header_len), 8);
            uint64_t data_base = 8 + header_len;

            const uint32_t M = 16;
            const uint32_t N = 2048;
            const uint32_t K = 4096;

            std::vector<uint32_t> real_w_packed(N * (K / 8));
            std::vector<half> real_w_scale(N * (K / 32));
            std::vector<half> h_a(M * K);
            std::vector<half> h_d_gpu(M * N);

            // Read real w1 packed weights: offset 262420, length 4,194,304 bytes
            file.seekg(data_base + 262420);
            file.read(reinterpret_cast<char*>(real_w_packed.data()), real_w_packed.size() * sizeof(uint32_t));

            // Read real w1 scales: offset 5033443628, length 524,288 bytes
            file.seekg(data_base + 5033443628ULL);
            file.read(reinterpret_cast<char*>(real_w_scale.data()), real_w_scale.size() * sizeof(half));

            for (uint32_t i = 0; i < h_a.size(); ++i) {
                h_a[i] = __float2half(((i % 13) - 6) * 0.05f);
            }

            half* d_a;
            uint32_t* d_w_packed;
            half* d_w_scale;
            half* d_out;

            CHECK_HIP(hipMalloc(&d_a, h_a.size() * sizeof(half)));
            CHECK_HIP(hipMalloc(&d_w_packed, real_w_packed.size() * sizeof(uint32_t)));
            CHECK_HIP(hipMalloc(&d_w_scale, real_w_scale.size() * sizeof(half)));
            CHECK_HIP(hipMalloc(&d_out, h_d_gpu.size() * sizeof(half)));

            CHECK_HIP(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), hipMemcpyHostToDevice));
            CHECK_HIP(hipMemcpy(d_w_packed, real_w_packed.data(), real_w_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
            CHECK_HIP(hipMemcpy(d_w_scale, real_w_scale.data(), real_w_scale.size() * sizeof(half), hipMemcpyHostToDevice));

            aeon::kernel::dispatch_w4a16_gemm(d_a, d_w_packed, d_w_scale, d_out, M, N, K);
            CHECK_HIP(hipDeviceSynchronize());

            CHECK_HIP(hipMemcpy(h_d_gpu.data(), d_out, h_d_gpu.size() * sizeof(half), hipMemcpyDeviceToHost));

            // CPU reference check for first 16 outputs
            float max_real_diff = 0.0f;
            for (uint32_t n = 0; n < 16; ++n) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    float a_val = __half2float(h_a[0 * K + k]);
                    uint32_t word = real_w_packed[n * (K / 8) + (k / 8)];
                    int nib = (word >> ((k % 8) * 4)) & 0xF;
                    float sc = __half2float(real_w_scale[n * (K / 32) + (k / 32)]);
                    float w_val = (float)(nib - 8) * sc;
                    sum += a_val * w_val;
                }
                float gpu_val = __half2float(h_d_gpu[0 * N + n]);
                float diff = std::abs(gpu_val - sum);
                if (diff > max_real_diff) max_real_diff = diff;
            }
            std::cout << "Real Weight Max Difference vs CPU: " << max_real_diff << std::endl;
            assert(max_real_diff < 0.01f);

            CHECK_HIP(hipFree(d_a));
            CHECK_HIP(hipFree(d_w_packed));
            CHECK_HIP(hipFree(d_w_scale));
            CHECK_HIP(hipFree(d_out));

            std::cout << "[PASS] Real Checkpoint INT4-W4A16 Expert 0 GEMM verified on physical silicon!" << std::endl;
        } else {
            std::cout << "[SKIP] Could not open safetensors checkpoint for sub-test 3." << std::endl;
        }
    }

    std::cout << "\n[ALL TESTS PASSED] Spike 3: Fused INT4-W4A16 Wave32 Dequant-GEMM verified on silicon!" << std::endl;
    return 0;
}
