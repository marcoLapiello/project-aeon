#include "core/config.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// Wave32 RMSNorm kernel: 1 warp per token row, assuming hidden_dim = 4096.
// Each of the 32 threads in the Wave32 warp processes 4096 / 32 = 128 elements.
__global__ void __launch_bounds__(32) rms_norm_wave32_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int hidden_dim,
    float eps
) {
    int lane_id = threadIdx.x; // 0..31
    int row = blockIdx.x;

    const __half* row_in = input + row * hidden_dim;
    __half* row_out = output + row * hidden_dim;

    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = lane_id; i < hidden_dim; i += 32) {
        float val = __half2float(row_in[i]);
        sum_sq += val * val;
    }

    // Warp reduction across 32 threads
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor(sum_sq, offset, 32);
    }

    float inv_rms = rsqrtf((sum_sq / (float)hidden_dim) + eps);

    #pragma unroll 4
    for (int i = lane_id; i < hidden_dim; i += 32) {
        float val = __half2float(row_in[i]);
        float w = __half2float(weight[i]);
        row_out[i] = __float2half(val * inv_rms * w);
    }
}

// Wave32 Fused SwiGLU with Clamping kernel:
// out = clamp(gate, -limit, limit) / (1 + exp(-clamp(gate, -limit, limit))) * clamp(up, -limit, limit)
// Or standard SwiGLU with activation limit: silu(clamp(gate, max=limit)) * clamp(up, min=-limit, max=limit)
// DeepSeek-V4 spec: swiglu_limit = 10.0
__global__ void swiglu_clamp_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    int total_elements,
    float limit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);

        // Clamping according to DeepSeek-V4 spec
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);

        // SiLU(g) = g / (1.0f + expf(-g))
        float silu_g = g / (1.0f + expf(-g));
        float res = silu_g * u;

        out[idx] = __float2half(res);
    }
}

// CPU reference implementations
void cpu_rms_norm(const float* in, const float* w, float* out, int rows, int dim, float eps) {
    for (int r = 0; r < rows; ++r) {
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum_sq += in[r * dim + i] * in[r * dim + i];
        }
        float inv_rms = 1.0f / std::sqrt((sum_sq / (float)dim) + eps);
        for (int i = 0; i < dim; ++i) {
            out[r * dim + i] = in[r * dim + i] * inv_rms * w[i];
        }
    }
}

void cpu_swiglu_clamp(const __half* gate, const __half* up, __half* out, int n, float limit) {
    for (int i = 0; i < n; ++i) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        g = std::min(g, limit);
        u = std::max(-limit, std::min(u, limit));
        float silu_g = g / (1.0f + std::exp(-g));
        out[i] = __float2half(silu_g * u);
    }
}

int main() {
    std::cout << "[Test] Wave32 RMSNorm and Clamped SwiGLU Kernel Validation..." << std::endl;

    const int num_tokens = 4;
    const int hidden_dim = 4096;
    const int intermediate_dim = 2048;
    const float eps = 1e-06f;
    const float swiglu_limit = 10.0f;

    // 1. RMSNorm Verification
    std::vector<float> h_in(num_tokens * hidden_dim);
    std::vector<float> h_weight(hidden_dim);
    std::vector<float> h_out_ref(num_tokens * hidden_dim);
    std::vector<__half> h_in_fp16(num_tokens * hidden_dim);
    std::vector<__half> h_weight_fp16(hidden_dim);
    std::vector<__half> h_out_gpu(num_tokens * hidden_dim);

    for (int i = 0; i < (int)h_in.size(); ++i) {
        h_in[i] = ((i % 17) - 8) * 0.125f;
        h_in_fp16[i] = __float2half(h_in[i]);
    }
    for (int i = 0; i < hidden_dim; ++i) {
        h_weight[i] = 1.0f + ((i % 5) - 2) * 0.05f;
        h_weight_fp16[i] = __float2half(h_weight[i]);
    }

    cpu_rms_norm(h_in.data(), h_weight.data(), h_out_ref.data(), num_tokens, hidden_dim, eps);

    __half *d_in, *d_weight, *d_out;
    CHECK_HIP(hipMalloc(&d_in, h_in_fp16.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_weight, h_weight_fp16.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_out, h_out_gpu.size() * sizeof(__half)));

    CHECK_HIP(hipMemcpy(d_in, h_in_fp16.data(), h_in_fp16.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_weight, h_weight_fp16.data(), h_weight_fp16.size() * sizeof(__half), hipMemcpyHostToDevice));

    // Launch RMSNorm Wave32 kernel: 1 wave32 block per token
    rms_norm_wave32_kernel<<<num_tokens, 32>>>(d_in, d_weight, d_out, hidden_dim, eps);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_out_gpu.data(), d_out, h_out_gpu.size() * sizeof(__half), hipMemcpyDeviceToHost));

    float max_err_norm = 0.0f;
    for (int i = 0; i < (int)h_out_ref.size(); ++i) {
        float gpu_val = __half2float(h_out_gpu[i]);
        float diff = std::abs(gpu_val - h_out_ref[i]);
        if (diff > max_err_norm) max_err_norm = diff;
    }
    std::cout << "RMSNorm Max Absolute Error: " << max_err_norm << std::endl;
    assert(max_err_norm < 1e-3f);

    // 2. SwiGLU with Clamping Verification
    const int swiglu_elements = num_tokens * intermediate_dim;
    std::vector<__half> h_gate_fp16(swiglu_elements);
    std::vector<__half> h_up_fp16(swiglu_elements);
    std::vector<__half> h_swiglu_ref(swiglu_elements);
    std::vector<__half> h_swiglu_gpu(swiglu_elements);

    // Fill with values that exceed limits to test clamping
    for (int i = 0; i < swiglu_elements; ++i) {
        float g = ((i % 50) - 20) * 0.8f; // Values up to 23.2 and down to -16.0
        float u = ((i % 40) - 20) * 0.9f;   // Values up to 17.1 and down to -18.0
        h_gate_fp16[i] = __float2half(g);
        h_up_fp16[i] = __float2half(u);
    }

    cpu_swiglu_clamp(h_gate_fp16.data(), h_up_fp16.data(), h_swiglu_ref.data(), swiglu_elements, swiglu_limit);

    __half *d_gate, *d_up, *d_swiglu_out;
    CHECK_HIP(hipMalloc(&d_gate, swiglu_elements * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_up, swiglu_elements * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_swiglu_out, swiglu_elements * sizeof(__half)));

    CHECK_HIP(hipMemcpy(d_gate, h_gate_fp16.data(), swiglu_elements * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_up, h_up_fp16.data(), swiglu_elements * sizeof(__half), hipMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (swiglu_elements + block_size - 1) / block_size;
    swiglu_clamp_kernel<<<grid_size, block_size>>>(d_gate, d_up, d_swiglu_out, swiglu_elements, swiglu_limit);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_swiglu_gpu.data(), d_swiglu_out, swiglu_elements * sizeof(__half), hipMemcpyDeviceToHost));

    float max_err_swiglu = 0.0f;
    for (int i = 0; i < swiglu_elements; ++i) {
        float gpu_val = __half2float(h_swiglu_gpu[i]);
        float ref_val = __half2float(h_swiglu_ref[i]);
        float diff = std::abs(gpu_val - ref_val);
        if (diff > max_err_swiglu) max_err_swiglu = diff;
    }
    std::cout << "SwiGLU Clamp Max Absolute Error: " << max_err_swiglu << std::endl;
    assert(max_err_swiglu < 1e-4f);

    CHECK_HIP(hipFree(d_in));
    CHECK_HIP(hipFree(d_weight));
    CHECK_HIP(hipFree(d_out));
    CHECK_HIP(hipFree(d_gate));
    CHECK_HIP(hipFree(d_up));
    CHECK_HIP(hipFree(d_swiglu_out));

    std::cout << "[Test PASS] Wave32 RMSNorm and Clamped SwiGLU successfully verified on silicon!" << std::endl;
    return 0;
}
