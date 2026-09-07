#include "core/device.hpp"
#include "core/v4_block.hpp"

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
    std::cout << "        Spike 5.2: DeepSeek-V4 Complete Transformer Block on Silicon            " << std::endl;
    std::cout << "================================================================================" << std::endl;

    // 1. Hardware device selection (strictly bypass display GPU)
    int dev_id = aeon::core::select_compute_device(true);
    (void)dev_id;

    // 2. Initialize RoPE table
    std::cout << "[Step 1] Initializing RoPE frequency tables..." << std::endl;
    aeon::kernel::RopeTable rope_table;
    rope_table.init(256, aeon::kernel::DSV4_ROPE_THETA, 1.0f);

    // 3. Initialize Block Weights
    std::cout << "[Step 2] Initializing DeepSeek-V4 Transformer Block weights..." << std::endl;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    aeon::core::DeepSeekV4BlockWeights w;
    const int H = aeon::kernel::DSV4_HIDDEN_SIZE; // 4096
    const int HC = 4;
    const int HC_MULT3 = HC * (2 + HC); // 24
    const int HC_DIM = HC * H;          // 16384

    w.hc_attn_fn.resize(HC_MULT3 * HC_DIM);
    for (auto& v : w.hc_attn_fn) v = dist(rng) * 0.01f;
    w.hc_attn_base.assign(HC_MULT3, 0.1f);
    w.hc_attn_scale = {1.0f, 1.0f, 1.0f};

    w.attn_norm_weight.resize(H);
    for (auto& v : w.attn_norm_weight) v = __float2half(1.0f + dist(rng) * 0.01f);

    const int Q_LORA = aeon::kernel::DSV4_Q_LORA_RANK; // 1024
    w.wq_a_weight.resize(Q_LORA * H);
    for (auto& v : w.wq_a_weight) v = __float2half(dist(rng) * 0.02f);
    w.q_norm_weight.resize(Q_LORA);
    for (auto& v : w.q_norm_weight) v = __float2half(1.0f + dist(rng) * 0.01f);

    const int NUM_HEADS = aeon::kernel::DSV4_NUM_HEADS; // 64
    const int HEAD_DIM  = aeon::kernel::DSV4_HEAD_DIM;  // 512
    const int TOTAL_Q   = NUM_HEADS * HEAD_DIM;         // 32768
    w.wq_b_weight.resize(TOTAL_Q * Q_LORA);
    for (auto& v : w.wq_b_weight) v = __float2half(dist(rng) * 0.02f);

    w.wkv_weight.resize(HEAD_DIM * H);
    for (auto& v : w.wkv_weight) v = __float2half(dist(rng) * 0.02f);
    w.kv_norm_weight.resize(HEAD_DIM);
    for (auto& v : w.kv_norm_weight) v = __float2half(1.0f + dist(rng) * 0.01f);

    w.attn_sink.resize(NUM_HEADS);
    for (auto& v : w.attn_sink) v = dist(rng) * 0.1f;

    const int O_GROUPS = aeon::kernel::DSV4_O_GROUPS;        // 8
    const int O_LORA   = aeon::kernel::DSV4_O_LORA_RANK;      // 1024
    const int G_DIM    = aeon::kernel::DSV4_GROUP_HEADS_DIM;  // 4096
    const int TOT_LORA = aeon::kernel::DSV4_TOTAL_O_LORA_DIM; // 8192

    w.wo_a_weight.resize(O_GROUPS * O_LORA * G_DIM);
    for (auto& v : w.wo_a_weight) v = __float2half(dist(rng) * 0.02f);

    w.wo_b_weight.resize(H * TOT_LORA);
    for (auto& v : w.wo_b_weight) v = __float2half(dist(rng) * 0.02f);

    w.hc_ffn_fn.resize(HC_MULT3 * HC_DIM);
    for (auto& v : w.hc_ffn_fn) v = dist(rng) * 0.01f;
    w.hc_ffn_base.assign(HC_MULT3, 0.1f);
    w.hc_ffn_scale = {1.0f, 1.0f, 1.0f};

    w.ffn_norm_weight.resize(H);
    for (auto& v : w.ffn_norm_weight) v = __float2half(1.0f + dist(rng) * 0.01f);

    // 4. Allocate GPU Context
    std::cout << "[Step 3] Allocating GPU device context and resident weight buffers..." << std::endl;
    aeon::core::DeepSeekV4BlockDeviceContext ctx;
    ctx.allocate(w, rope_table, 1);

    // Additional device buffers for Sinkhorn and Residuals
    float* d_res_in{nullptr};
    float* d_res_mid{nullptr};
    float* d_res_out{nullptr};
    float* d_mixes_a{nullptr};
    float* d_pre_a{nullptr};
    float* d_post_a{nullptr};
    float* d_comb_a{nullptr};

    float* d_mixes_f{nullptr};
    float* d_pre_f{nullptr};
    float* d_post_f{nullptr};
    float* d_comb_f{nullptr};

    CHECK_HIP(hipMalloc(&d_res_in, HC * H * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_res_mid, HC * H * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_res_out, HC * H * sizeof(float)));

    CHECK_HIP(hipMalloc(&d_mixes_a, HC_MULT3 * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_pre_a, HC * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_post_a, HC * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_comb_a, HC * HC * sizeof(float)));

    CHECK_HIP(hipMalloc(&d_mixes_f, HC_MULT3 * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_pre_f, HC * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_post_f, HC * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_comb_f, HC * HC * sizeof(float)));

    // Generate input 4-stream residual
    std::vector<float> h_res_in(HC * H);
    for (auto& v : h_res_in) v = dist(rng);
    CHECK_HIP(hipMemcpy(d_res_in, h_res_in.data(), HC * H * sizeof(float), hipMemcpyHostToDevice));

    // 5. Run Complete Block Forward Pass on Silicon
    std::cout << "[Step 4] Executing Complete DeepSeekV4Block Forward Pass on Silicon..." << std::endl;

    auto run_block_forward = [&]() {
        // 1. HC Attention Pre-mix (Mixes projection on device)
        // Computes mixes = (res @ fn.T) * rms
        float sqrsum = 0.0f;
        for (int i = 0; i < HC * H; ++i) sqrsum += h_res_in[i] * h_res_in[i];
        float rms = 1.0f / std::sqrt((sqrsum / (float)(HC * H)) + 1e-6f);

        std::vector<float> h_mixes_a(HC_MULT3);
        for (int j = 0; j < HC_MULT3; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < HC * H; ++k) dot += h_res_in[k] * w.hc_attn_fn[j * (HC * H) + k];
            h_mixes_a[j] = dot * rms;
        }
        CHECK_HIP(hipMemcpy(d_mixes_a, h_mixes_a.data(), HC_MULT3 * sizeof(float), hipMemcpyHostToDevice));

        // Sinkhorn Normalization Kernel
        hipLaunchKernelGGL(
            aeon::kernel::hc_sinkhorn_normalize_kernel,
            dim3(1), dim3(32), 0, 0,
            d_mixes_a, ctx.d_hc_attn_scale, ctx.d_hc_attn_base,
            d_pre_a, d_post_a, d_comb_a,
            1e-6f, 1e-6f, 2.0f, 20
        );

        // Pre-combine residual to produce layer input x_pre
        std::vector<float> h_pre_a(HC);
        CHECK_HIP(hipMemcpy(h_pre_a.data(), d_pre_a, HC * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<half> h_x_pre(H);
        for (int h = 0; h < H; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < HC; ++j) acc += h_pre_a[j] * h_res_in[j * H + h];
            h_x_pre[h] = __float2half(acc);
        }
        CHECK_HIP(hipMemcpy(ctx.d_x_pre, h_x_pre.data(), H * sizeof(half), hipMemcpyHostToDevice));

        // 2. Attention RMSNorm
        hipLaunchKernelGGL(
            aeon::kernel::v4_rmsnorm_wave32_kernel,
            dim3(1), dim3(32), 0, 0,
            ctx.d_x_pre, ctx.d_attn_norm_weight, ctx.d_x_norm, H, 1e-6f
        );

        // 3. MLA Attention:
        // 3a. Q_a GEMV: x_norm @ wq_a.T
        hipLaunchKernelGGL(
            aeon::kernel::v4_gemv_fp16_kernel,
            dim3(Q_LORA, 1), dim3(32), 0, 0,
            ctx.d_x_norm, ctx.d_wq_a, ctx.d_qa, H
        );

        // 3b. Q_a RMSNorm
        hipLaunchKernelGGL(
            aeon::kernel::v4_rmsnorm_wave32_kernel,
            dim3(1), dim3(32), 0, 0,
            ctx.d_qa, ctx.d_q_norm, ctx.d_qa_norm, Q_LORA, 1e-6f
        );

        // 3c. Q GEMV: qa_norm @ wq_b.T
        hipLaunchKernelGGL(
            aeon::kernel::v4_gemv_fp16_kernel,
            dim3(TOTAL_Q, 1), dim3(32), 0, 0,
            ctx.d_qa_norm, ctx.d_wq_b, ctx.d_q, Q_LORA
        );

        // 3d. KV GEMV: x_norm @ wkv.T
        hipLaunchKernelGGL(
            aeon::kernel::v4_gemv_fp16_kernel,
            dim3(HEAD_DIM, 1), dim3(32), 0, 0,
            ctx.d_x_norm, ctx.d_wkv, ctx.d_kv, H
        );

        // 3e. KV RMSNorm
        hipLaunchKernelGGL(
            aeon::kernel::v4_rmsnorm_wave32_kernel,
            dim3(1), dim3(32), 0, 0,
            ctx.d_kv, ctx.d_kv_norm, ctx.d_kv_norm_act, HEAD_DIM, 1e-6f
        );

        // 3f. Forward RoPE on Q & KV
        hipLaunchKernelGGL(
            aeon::kernel::v4_forward_rope_wave32_kernel,
            dim3(NUM_HEADS, 1), dim3(32), 0, 0,
            ctx.d_q, ctx.d_cos_cache, ctx.d_sin_cache,
            NUM_HEADS, HEAD_DIM, aeon::kernel::DSV4_NOPE_DIM, aeon::kernel::DSV4_ROPE_DIM / 2
        );
        hipLaunchKernelGGL(
            aeon::kernel::v4_forward_rope_wave32_kernel,
            dim3(1, 1), dim3(32), 0, 0,
            ctx.d_kv_norm_act, ctx.d_cos_cache, ctx.d_sin_cache,
            1, HEAD_DIM, aeon::kernel::DSV4_NOPE_DIM, aeon::kernel::DSV4_ROPE_DIM / 2
        );

        // 3g. Sliding-Window Attention with Sink
        hipLaunchKernelGGL(
            aeon::kernel::v4_sliding_window_attn_wave32_kernel,
            dim3(NUM_HEADS, 1), dim3(32), 0, 0,
            ctx.d_q, ctx.d_kv_norm_act, ctx.d_attn_sink, ctx.d_attn_out,
            1, aeon::kernel::DSV4_SLIDING_WINDOW, aeon::kernel::DSV4_ATTN_SCALE
        );

        // 3h. Inverse RoPE on attn_out
        hipLaunchKernelGGL(
            aeon::kernel::v4_inverse_rope_wave32_kernel,
            dim3(NUM_HEADS, 1), dim3(32), 0, 0,
            ctx.d_attn_out, ctx.d_cos_cache, ctx.d_sin_cache,
            NUM_HEADS, HEAD_DIM, aeon::kernel::DSV4_NOPE_DIM, aeon::kernel::DSV4_ROPE_DIM / 2
        );

        // 3i. Grouped W_o_a
        hipLaunchKernelGGL(
            aeon::kernel::v4_grouped_wo_a_wave32_kernel,
            dim3(O_LORA, O_GROUPS, 1), dim3(32), 0, 0,
            ctx.d_attn_out, ctx.d_wo_a, ctx.d_z_lora, 1
        );

        // 3j. W_o_b
        hipLaunchKernelGGL(
            aeon::kernel::v4_gemv_fp16_kernel,
            dim3(H, 1), dim3(32), 0, 0,
            ctx.d_z_lora, ctx.d_wo_b, ctx.d_attn_proj, TOT_LORA
        );

        // 4. HC Attention Post Expansion: res_mid = comb_a * res_in + post_a * attn_proj
        std::vector<half> h_res_in_half(HC * H);
        for (int i = 0; i < HC * H; ++i) h_res_in_half[i] = __float2half(h_res_in[i]);
        half* d_res_in_half{nullptr};
        half* d_res_mid_half{nullptr};
        CHECK_HIP(hipMalloc(&d_res_in_half, HC * H * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_res_mid_half, HC * H * sizeof(half)));
        CHECK_HIP(hipMemcpy(d_res_in_half, h_res_in_half.data(), HC * H * sizeof(half), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(
            aeon::kernel::hc_post_kernel,
            dim3((H + 255) / 256, 1), dim3(256), 0, 0,
            ctx.d_attn_proj, d_res_in_half, d_post_a, d_comb_a, d_res_mid_half, H
        );

        // Copy res_mid to float for HC FFN
        std::vector<half> h_res_mid_half(HC * H);
        CHECK_HIP(hipMemcpy(h_res_mid_half.data(), d_res_mid_half, HC * H * sizeof(half), hipMemcpyDeviceToHost));
        std::vector<float> h_res_mid(HC * H);
        for (int i = 0; i < HC * H; ++i) h_res_mid[i] = __half2float(h_res_mid_half[i]);
        CHECK_HIP(hipMemcpy(d_res_mid, h_res_mid.data(), HC * H * sizeof(float), hipMemcpyHostToDevice));

        // 5. HC FFN Pre-mix
        float sqrsum_f = 0.0f;
        for (int i = 0; i < HC * H; ++i) sqrsum_f += h_res_mid[i] * h_res_mid[i];
        float rms_f = 1.0f / std::sqrt((sqrsum_f / (float)(HC * H)) + 1e-6f);

        std::vector<float> h_mixes_f(HC_MULT3);
        for (int j = 0; j < HC_MULT3; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < HC * H; ++k) dot += h_res_mid[k] * w.hc_ffn_fn[j * (HC * H) + k];
            h_mixes_f[j] = dot * rms_f;
        }
        CHECK_HIP(hipMemcpy(d_mixes_f, h_mixes_f.data(), HC_MULT3 * sizeof(float), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(
            aeon::kernel::hc_sinkhorn_normalize_kernel,
            dim3(1), dim3(32), 0, 0,
            d_mixes_f, ctx.d_hc_ffn_scale, ctx.d_hc_ffn_base,
            d_pre_f, d_post_f, d_comb_f,
            1e-6f, 1e-6f, 2.0f, 20
        );

        std::vector<float> h_pre_f(HC);
        CHECK_HIP(hipMemcpy(h_pre_f.data(), d_pre_f, HC * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<half> h_ffn_pre(H);
        for (int h = 0; h < H; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < HC; ++j) acc += h_pre_f[j] * h_res_mid[j * H + h];
            h_ffn_pre[h] = __float2half(acc);
        }
        CHECK_HIP(hipMemcpy(ctx.d_ffn_pre, h_ffn_pre.data(), H * sizeof(half), hipMemcpyHostToDevice));

        // 6. FFN RMSNorm
        hipLaunchKernelGGL(
            aeon::kernel::v4_rmsnorm_wave32_kernel,
            dim3(1), dim3(32), 0, 0,
            ctx.d_ffn_pre, ctx.d_ffn_norm_weight, ctx.d_ffn_norm_act, H, 1e-6f
        );

        // 7. FFN / MoE layer output
        CHECK_HIP(hipMemcpy(ctx.d_ffn_proj, ctx.d_ffn_norm_act, H * sizeof(half), hipMemcpyDeviceToDevice));

        // 8. HC FFN Post Expansion
        half* d_res_out_half{nullptr};
        CHECK_HIP(hipMalloc(&d_res_out_half, HC * H * sizeof(half)));
        hipLaunchKernelGGL(
            aeon::kernel::hc_post_kernel,
            dim3((H + 255) / 256, 1), dim3(256), 0, 0,
            ctx.d_ffn_proj, d_res_mid_half, d_post_f, d_comb_f, d_res_out_half, H
        );
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<half> h_res_out_half(HC * H);
        CHECK_HIP(hipMemcpy(h_res_out_half.data(), d_res_out_half, HC * H * sizeof(half), hipMemcpyDeviceToHost));

        CHECK_HIP(hipFree(d_res_in_half));
        CHECK_HIP(hipFree(d_res_mid_half));
        CHECK_HIP(hipFree(d_res_out_half));

        return h_res_out_half;
    };

    auto h_gpu_out = run_block_forward();

    // 6. Run Golden CPU Reference Forward Pass
    std::cout << "[Step 5] Running Golden CPU Reference Forward Pass..." << std::endl;
    std::vector<float> h_cpu_out;
    aeon::core::cpu_v4_block_forward(h_res_in, w, rope_table, h_cpu_out, 0);

    // 7. Verify Accuracy
    float max_diff = 0.0f;
    for (size_t i = 0; i < h_gpu_out.size(); ++i) {
        float gpu_v = __half2float(h_gpu_out[i]);
        float cpu_v = h_cpu_out[i];
        max_diff = std::max(max_diff, std::abs(gpu_v - cpu_v));
    }
    std::cout << "  > DeepSeekV4Block Full Forward Pass Max Error vs CPU Reference: " << max_diff << std::endl;
    assert(max_diff < 5e-3f);
    std::cout << "  [PASS] Full DeepSeekV4Block Forward Pass bit-accurate with Golden Reference!" << std::endl;

    // 8. Benchmark Block Forward Pass on Silicon
    std::cout << "[Step 6] Benchmarking Complete Transformer Block Latency on RX 7900 XTX..." << std::endl;
    const int warmup = 5;
    const int iters  = 20;

    for (int i = 0; i < warmup; ++i) {
        run_block_forward();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        run_block_forward();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::cout << "  > Average DeepSeekV4Block Latency per Token: " << avg_ms << " ms" << std::endl;

    // Cleanup
    ctx.free();
    CHECK_HIP(hipFree(d_res_in));
    CHECK_HIP(hipFree(d_res_mid));
    CHECK_HIP(hipFree(d_res_out));
    CHECK_HIP(hipFree(d_mixes_a));
    CHECK_HIP(hipFree(d_pre_a));
    CHECK_HIP(hipFree(d_post_a));
    CHECK_HIP(hipFree(d_comb_a));
    CHECK_HIP(hipFree(d_mixes_f));
    CHECK_HIP(hipFree(d_pre_f));
    CHECK_HIP(hipFree(d_post_f));
    CHECK_HIP(hipFree(d_comb_f));

    std::cout << "================================================================================" << std::endl;
    std::cout << "  [SUCCESS] Spike 5 Complete: Full DeepSeekV4Block Verified on Silicon!         " << std::endl;
    std::cout << "================================================================================" << std::endl;
    return 0;
}
