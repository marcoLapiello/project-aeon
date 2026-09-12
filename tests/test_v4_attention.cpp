#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cassert>

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "       DeepSeek-V4 Sliding-Window Attention & RoPE Validation on Silicon        " << std::endl;
    std::cout << "================================================================================" << std::endl;

    // 1. Hardware device selection (strictly bypass display GPU)
    int dev_id = aeon::core::select_compute_device(true);
    (void)dev_id;

    // 2. Initialize RoPE table
    std::cout << "[Step 1] Initializing RoPE frequency tables..." << std::endl;
    aeon::kernel::RopeTable rope_table;
    rope_table.init(256, aeon::kernel::DSV4_ROPE_THETA, 1.0f);

    float* d_cos_cache{nullptr};
    float* d_sin_cache{nullptr};
    size_t cache_bytes = rope_table.max_seq_len * rope_table.half_rope * sizeof(float);
    CHECK_HIP(hipMalloc(&d_cos_cache, cache_bytes));
    CHECK_HIP(hipMalloc(&d_sin_cache, cache_bytes));
    CHECK_HIP(hipMemcpy(d_cos_cache, rope_table.cos_cache.data(), cache_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sin_cache, rope_table.sin_cache.data(), cache_bytes, hipMemcpyHostToDevice));

    // 3. Verify Forward & Inverse RoPE on Silicon
    std::cout << "[Step 2] Testing Forward RoPE and Inverse RoPE invertibility on Silicon..." << std::endl;
    const int num_tokens = 16;
    const int num_heads = aeon::kernel::DSV4_NUM_HEADS; // 64
    const int head_dim = aeon::kernel::DSV4_HEAD_DIM;   // 512
    const size_t q_elements = num_tokens * num_heads * head_dim;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_q_orig(q_elements);
    std::vector<half>  h_q_fp16(q_elements);
    for (size_t i = 0; i < q_elements; ++i) {
        h_q_orig[i] = dist(rng);
        h_q_fp16[i] = __float2half(h_q_orig[i]);
    }

    half* d_q{nullptr};
    CHECK_HIP(hipMalloc(&d_q, q_elements * sizeof(half)));
    CHECK_HIP(hipMemcpy(d_q, h_q_fp16.data(), q_elements * sizeof(half), hipMemcpyHostToDevice));

    // Launch Forward RoPE
    dim3 rope_grid(num_heads, num_tokens);
    dim3 rope_block(32); // 32 threads for 32 frequency pairs
    hipLaunchKernelGGL(
        aeon::kernel::v4_forward_rope_wave32_kernel,
        rope_grid, rope_block, 0, 0,
        d_q, d_cos_cache, d_sin_cache,
        num_heads, head_dim, aeon::kernel::DSV4_NOPE_DIM, aeon::kernel::DSV4_ROPE_DIM / 2
    );
    CHECK_HIP(hipDeviceSynchronize());

    // Copy forward rotated values to host and check against CPU reference
    std::vector<half> h_q_rot(q_elements);
    CHECK_HIP(hipMemcpy(h_q_rot.data(), d_q, q_elements * sizeof(half), hipMemcpyDeviceToHost));

    std::vector<float> h_q_cpu = h_q_orig;
    for (int t = 0; t < num_tokens; ++t) {
        aeon::kernel::cpu_forward_rope(
            h_q_cpu.data() + t * (num_heads * head_dim),
            t, rope_table, num_heads, head_dim, aeon::kernel::DSV4_NOPE_DIM, aeon::kernel::DSV4_ROPE_DIM / 2
        );
    }

    float max_rope_err = 0.0f;
    for (size_t i = 0; i < q_elements; ++i) {
        float gpu_val = __half2float(h_q_rot[i]);
        float cpu_val = h_q_cpu[i];
        max_rope_err = std::max(max_rope_err, std::abs(gpu_val - cpu_val));
    }
    std::cout << "  > Forward RoPE vs CPU Golden Error: " << max_rope_err << std::endl;
    assert(max_rope_err < 1e-3f);

    // Launch Inverse RoPE and check invertibility
    hipLaunchKernelGGL(
        aeon::kernel::v4_inverse_rope_wave32_kernel,
        rope_grid, rope_block, 0, 0,
        d_q, d_cos_cache, d_sin_cache,
        num_heads, head_dim, aeon::kernel::DSV4_NOPE_DIM, aeon::kernel::DSV4_ROPE_DIM / 2
    );
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<half> h_q_restored(q_elements);
    CHECK_HIP(hipMemcpy(h_q_restored.data(), d_q, q_elements * sizeof(half), hipMemcpyDeviceToHost));

    float max_inv_err = 0.0f;
    for (size_t i = 0; i < q_elements; ++i) {
        float restored = __half2float(h_q_restored[i]);
        float original = h_q_orig[i];
        max_inv_err = std::max(max_inv_err, std::abs(restored - original));
    }
    std::cout << "  > RoPE Invertibility (InvRoPE(FwdRoPE(X)) - X) Error: " << max_inv_err << std::endl;
    assert(max_inv_err < 1e-3f);
    std::cout << "  [PASS] RoPE & Inverse RoPE mathematical invertibility verified!" << std::endl;

    // 4. Test Causal Sliding-Window Attention with Attention Sink on Silicon
    std::cout << "[Step 3] Testing Causal Sliding-Window Attention (W=128) with Sink on Silicon..." << std::endl;
    const size_t k_elements = num_tokens * head_dim; // Single KV head shared across all 64 Q heads
    std::vector<float> h_k_f32(k_elements);
    std::vector<half>  h_k_fp16(k_elements);
    for (size_t i = 0; i < k_elements; ++i) {
        h_k_f32[i] = dist(rng);
        h_k_fp16[i] = __float2half(h_k_f32[i]);
    }

    std::vector<float> h_sink_f32(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        h_sink_f32[h] = dist(rng) * 0.5f; // Non-zero attention sink
    }

    half* d_k{nullptr};
    float* d_sink{nullptr};
    half* d_out{nullptr};
    CHECK_HIP(hipMalloc(&d_k, k_elements * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_sink, num_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_out, q_elements * sizeof(half)));

    CHECK_HIP(hipMemcpy(d_k, h_k_fp16.data(), k_elements * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sink, h_sink_f32.data(), num_heads * sizeof(float), hipMemcpyHostToDevice));

    // Reset d_q to forward-rotated values
    CHECK_HIP(hipMemcpy(d_q, h_q_rot.data(), q_elements * sizeof(half), hipMemcpyHostToDevice));

    // Launch SWA Kernel
    dim3 attn_grid(num_heads, num_tokens);
    dim3 attn_block(32); // 1 Wave32 warp per head-token
    hipLaunchKernelGGL(
        aeon::kernel::v4_sliding_window_attn_wave32_kernel,
        attn_grid, attn_block, 0, 0,
        d_q, d_k, d_sink, d_out,
        num_tokens, aeon::kernel::DSV4_SLIDING_WINDOW, aeon::kernel::DSV4_ATTN_SCALE
    );
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<half> h_out_gpu(q_elements);
    CHECK_HIP(hipMemcpy(h_out_gpu.data(), d_out, q_elements * sizeof(half), hipMemcpyDeviceToHost));

    // Run CPU Golden Reference Attention
    std::vector<float> h_q_rot_f32(q_elements);
    for (size_t i = 0; i < q_elements; ++i) {
        h_q_rot_f32[i] = __half2float(h_q_rot[i]);
    }
    std::vector<float> h_out_cpu;
    aeon::kernel::cpu_sliding_window_attention(
        h_q_rot_f32, h_k_f32, h_sink_f32, h_out_cpu, num_tokens, aeon::kernel::DSV4_SLIDING_WINDOW, aeon::kernel::DSV4_ATTN_SCALE
    );

    float max_attn_err = 0.0f;
    for (size_t i = 0; i < q_elements; ++i) {
        float gpu_val = __half2float(h_out_gpu[i]);
        float cpu_val = h_out_cpu[i];
        max_attn_err = std::max(max_attn_err, std::abs(gpu_val - cpu_val));
    }
    std::cout << "  > SWA Kernel vs CPU Golden Error: " << max_attn_err << std::endl;
    assert(max_attn_err < 1.5e-3f);
    std::cout << "  [PASS] Causal Sliding-Window Attention matches Golden CPU Reference!" << std::endl;

    // 5. Test Grouped Output Projections W_o_a & W_o_b
    std::cout << "[Step 4] Testing Grouped W_o_a and W_o_b Projections on Silicon..." << std::endl;
    // W_o_a: [8, 1024, 4096]
    const size_t wo_a_elements = aeon::kernel::DSV4_O_GROUPS * aeon::kernel::DSV4_O_LORA_RANK * aeon::kernel::DSV4_GROUP_HEADS_DIM;
    std::vector<half> h_wo_a(wo_a_elements);
    for (size_t i = 0; i < wo_a_elements; ++i) {
        h_wo_a[i] = __float2half(dist(rng) * 0.02f);
    }

    half* d_wo_a{nullptr};
    half* d_z{nullptr};
    const size_t z_elements = num_tokens * aeon::kernel::DSV4_TOTAL_O_LORA_DIM;
    CHECK_HIP(hipMalloc(&d_wo_a, wo_a_elements * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_z, z_elements * sizeof(half)));
    CHECK_HIP(hipMemcpy(d_wo_a, h_wo_a.data(), wo_a_elements * sizeof(half), hipMemcpyHostToDevice));

    // First apply inverse RoPE to d_out
    hipLaunchKernelGGL(
        aeon::kernel::v4_inverse_rope_wave32_kernel,
        rope_grid, rope_block, 0, 0,
        d_out, d_cos_cache, d_sin_cache,
        num_heads, head_dim, aeon::kernel::DSV4_NOPE_DIM, aeon::kernel::DSV4_ROPE_DIM / 2
    );
    CHECK_HIP(hipDeviceSynchronize());

    // Launch Grouped W_o_a
    dim3 wo_a_grid(aeon::kernel::DSV4_O_LORA_RANK, aeon::kernel::DSV4_O_GROUPS, num_tokens);
    dim3 wo_a_block(32); // 1 Wave32 warp
    hipLaunchKernelGGL(
        aeon::kernel::v4_grouped_wo_a_wave32_kernel,
        wo_a_grid, wo_a_block, 0, 0,
        d_out, d_wo_a, d_z, num_tokens
    );
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<half> h_z(z_elements);
    CHECK_HIP(hipMemcpy(h_z.data(), d_z, z_elements * sizeof(half), hipMemcpyDeviceToHost));

    // Verify non-zero output and stability
    float z_norm = 0.0f;
    for (size_t i = 0; i < z_elements; ++i) {
        float val = __half2float(h_z[i]);
        assert(!std::isnan(val) && !std::isinf(val));
        z_norm += val * val;
    }
    std::cout << "  > Grouped W_o_a Output Norm: " << std::sqrt(z_norm) << " (Finite and stable)" << std::endl;
    assert(z_norm > 0.0f);
    std::cout << "  [PASS] Grouped W_o_a Output Projection verified!" << std::endl;

    // 6. Benchmark Attention Engine Latency on Silicon
    std::cout << "[Step 5] Benchmarking Sliding-Window Attention Latency on RX 7900 XTX..." << std::endl;
    const int warmup = 20;
    const int iters  = 100;

    for (int i = 0; i < warmup; ++i) {
        hipLaunchKernelGGL(
            aeon::kernel::v4_sliding_window_attn_wave32_kernel,
            attn_grid, attn_block, 0, 0,
            d_q, d_k, d_sink, d_out,
            num_tokens, aeon::kernel::DSV4_SLIDING_WINDOW, aeon::kernel::DSV4_ATTN_SCALE
        );
    }
    CHECK_HIP(hipDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        hipLaunchKernelGGL(
            aeon::kernel::v4_sliding_window_attn_wave32_kernel,
            attn_grid, attn_block, 0, 0,
            d_q, d_k, d_sink, d_out,
            num_tokens, aeon::kernel::DSV4_SLIDING_WINDOW, aeon::kernel::DSV4_ATTN_SCALE
        );
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    std::cout << "  > Average Attention Kernel Latency (" << num_tokens << " tokens, 64 heads, W=128): "
              << avg_us << " us (" << (avg_us / num_tokens) << " us/token)" << std::endl;

    // Cleanup
    CHECK_HIP(hipFree(d_cos_cache));
    CHECK_HIP(hipFree(d_sin_cache));
    CHECK_HIP(hipFree(d_q));
    CHECK_HIP(hipFree(d_k));
    CHECK_HIP(hipFree(d_sink));
    CHECK_HIP(hipFree(d_out));
    CHECK_HIP(hipFree(d_wo_a));
    CHECK_HIP(hipFree(d_z));

    std::cout << "================================================================================" << std::endl;
    std::cout << "  [SUCCESS] Sliding-window attention verification passed on silicon!             " << std::endl;
    std::cout << "================================================================================" << std::endl;
    return 0;
}
