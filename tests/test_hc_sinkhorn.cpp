#include "kernel/hc_sinkhorn.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    std::cout << "[Test] Hyper-Connections Sinkhorn & Post-Expansion on Silicon..." << std::endl;

    const int num_tokens = 4;
    const int hidden_size = 4096;
    const int hc_mult = 4;
    const int mix_hc = 24;
    const float hc_pre_eps = 1e-6f;
    const float hc_sinkhorn_eps = 1e-6f;
    const float hc_post_alpha = 2.0f;
    const int sinkhorn_iters = 20;

    // Allocate host test data
    std::vector<float> h_mixes(num_tokens * mix_hc);
    std::vector<float> h_scale = {2.076959f, 0.019465f, 0.238313f};
    std::vector<float> h_base(mix_hc);

    for (int i = 0; i < (int)h_mixes.size(); ++i) {
        h_mixes[i] = ((i % 13) - 6) * 0.5f;
    }
    for (int i = 0; i < mix_hc; ++i) {
        h_base[i] = ((i % 7) - 3) * 0.25f;
    }

    // CPU Reference calculations for Sinkhorn
    std::vector<float> ref_pre_mix(num_tokens * hc_mult);
    std::vector<float> ref_post_mix(num_tokens * hc_mult);
    std::vector<float> ref_comb_mix(num_tokens * hc_mult * hc_mult);

    for (int t = 0; t < num_tokens; ++t) {
        // Pre-mix
        for (int j = 0; j < hc_mult; ++j) {
            float val = h_mixes[t * mix_hc + j] * h_scale[0] + h_base[j];
            ref_pre_mix[t * hc_mult + j] = (1.0f / (1.0f + std::exp(-val))) + hc_pre_eps;
        }
        // Post-mix
        for (int j = 0; j < hc_mult; ++j) {
            float val = h_mixes[t * mix_hc + j + hc_mult] * h_scale[1] + h_base[j + hc_mult];
            ref_post_mix[t * hc_mult + j] = (1.0f / (1.0f + std::exp(-val))) * hc_post_alpha;
        }
        // Comb-mix logits
        std::vector<float> cm(16);
        for (int j = 0; j < 16; ++j) {
            cm[j] = h_mixes[t * mix_hc + j + 8] * h_scale[2] + h_base[j + 8];
        }
        // Softmax rows
        for (int r = 0; r < 4; ++r) {
            float max_v = cm[r * 4];
            for (int c = 1; c < 4; ++c) if (cm[r * 4 + c] > max_v) max_v = cm[r * 4 + c];
            float sum_e = 0.0f;
            for (int c = 0; c < 4; ++c) {
                cm[r * 4 + c] = std::exp(cm[r * 4 + c] - max_v);
                sum_e += cm[r * 4 + c];
            }
            for (int c = 0; c < 4; ++c) {
                cm[r * 4 + c] = (cm[r * 4 + c] / sum_e) + hc_sinkhorn_eps;
            }
        }
        // Col norm
        for (int c = 0; c < 4; ++c) {
            float sum_c = cm[0 * 4 + c] + cm[1 * 4 + c] + cm[2 * 4 + c] + cm[3 * 4 + c];
            for (int r = 0; r < 4; ++r) cm[r * 4 + c] /= (sum_c + hc_sinkhorn_eps);
        }
        // Sinkhorn repeats
        for (int it = 0; it < sinkhorn_iters - 1; ++it) {
            for (int r = 0; r < 4; ++r) {
                float sum_r = cm[r * 4 + 0] + cm[r * 4 + 1] + cm[r * 4 + 2] + cm[r * 4 + 3];
                for (int c = 0; c < 4; ++c) cm[r * 4 + c] /= (sum_r + hc_sinkhorn_eps);
            }
            for (int c = 0; c < 4; ++c) {
                float sum_c = cm[0 * 4 + c] + cm[1 * 4 + c] + cm[2 * 4 + c] + cm[3 * 4 + c];
                for (int r = 0; r < 4; ++r) cm[r * 4 + c] /= (sum_c + hc_sinkhorn_eps);
            }
        }
        for (int i = 0; i < 16; ++i) {
            ref_comb_mix[t * 16 + i] = cm[i];
        }
    }

    // Allocate GPU buffers
    float *d_mixes, *d_scale, *d_base, *d_pre, *d_post, *d_comb;
    CHECK_HIP(hipMalloc(&d_mixes, h_mixes.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_scale, h_scale.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_base, h_base.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_pre, ref_pre_mix.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_post, ref_post_mix.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_comb, ref_comb_mix.size() * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_mixes, h_mixes.data(), h_mixes.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_scale, h_scale.data(), h_scale.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_base, h_base.data(), h_base.size() * sizeof(float), hipMemcpyHostToDevice));

    // Launch Sinkhorn kernel (1 Wave32 warp per token)
    aeon::kernel::hc_sinkhorn_normalize_kernel<<<num_tokens, 32>>>(
        d_mixes, d_scale, d_base, d_pre, d_post, d_comb,
        hc_pre_eps, hc_sinkhorn_eps, hc_post_alpha, sinkhorn_iters
    );
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<float> gpu_pre(ref_pre_mix.size());
    std::vector<float> gpu_post(ref_post_mix.size());
    std::vector<float> gpu_comb(ref_comb_mix.size());

    CHECK_HIP(hipMemcpy(gpu_pre.data(), d_pre, gpu_pre.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(gpu_post.data(), d_post, gpu_post.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(gpu_comb.data(), d_comb, gpu_comb.size() * sizeof(float), hipMemcpyDeviceToHost));

    float max_err_pre = 0.0f, max_err_post = 0.0f, max_err_comb = 0.0f;
    for (size_t i = 0; i < gpu_pre.size(); ++i) max_err_pre = std::max(max_err_pre, std::abs(gpu_pre[i] - ref_pre_mix[i]));
    for (size_t i = 0; i < gpu_post.size(); ++i) max_err_post = std::max(max_err_post, std::abs(gpu_post[i] - ref_post_mix[i]));
    for (size_t i = 0; i < gpu_comb.size(); ++i) max_err_comb = std::max(max_err_comb, std::abs(gpu_comb[i] - ref_comb_mix[i]));

    std::cout << "HC Pre Max Error: " << max_err_pre << std::endl;
    std::cout << "HC Post Max Error: " << max_err_post << std::endl;
    std::cout << "HC Comb (Sinkhorn) Max Error: " << max_err_comb << std::endl;

    assert(max_err_pre < 1e-5f);
    assert(max_err_post < 1e-5f);
    assert(max_err_comb < 1e-5f);

    // 2. Validate HC Post Kernel
    std::vector<__half> h_layer_out(num_tokens * hidden_size);
    std::vector<__half> h_residual_in(num_tokens * 4 * hidden_size);
    std::vector<__half> h_residual_out_ref(num_tokens * 4 * hidden_size);
    std::vector<__half> h_residual_out_gpu(num_tokens * 4 * hidden_size);

    for (size_t i = 0; i < h_layer_out.size(); ++i) {
        h_layer_out[i] = __float2half(((i % 19) - 9) * 0.1f);
    }
    for (size_t i = 0; i < h_residual_in.size(); ++i) {
        h_residual_in[i] = __float2half(((i % 23) - 11) * 0.1f);
    }

    // Reference Post Expansion
    for (int t = 0; t < num_tokens; ++t) {
        for (int hco = 0; hco < 4; ++hco) {
            for (int h = 0; h < hidden_size; ++h) {
                float val = ref_post_mix[t * 4 + hco] * __half2float(h_layer_out[t * hidden_size + h]);
                for (int hci = 0; hci < 4; ++hci) {
                    val += ref_comb_mix[t * 16 + hci * 4 + hco] * __half2float(h_residual_in[t * 4 * hidden_size + hci * hidden_size + h]);
                }
                h_residual_out_ref[t * 4 * hidden_size + hco * hidden_size + h] = __float2half(val);
            }
        }
    }

    __half *d_layer_out, *d_res_in, *d_res_out;
    CHECK_HIP(hipMalloc(&d_layer_out, h_layer_out.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_res_in, h_residual_in.size() * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_res_out, h_residual_out_gpu.size() * sizeof(__half)));

    CHECK_HIP(hipMemcpy(d_layer_out, h_layer_out.data(), h_layer_out.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_res_in, h_residual_in.data(), h_residual_in.size() * sizeof(__half), hipMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((hidden_size + block.x - 1) / block.x, num_tokens);
    aeon::kernel::hc_post_kernel<<<grid, block>>>(d_layer_out, d_res_in, d_post, d_comb, d_res_out, hidden_size);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_residual_out_gpu.data(), d_res_out, h_residual_out_gpu.size() * sizeof(__half), hipMemcpyDeviceToHost));

    float max_err_post_exp = 0.0f;
    for (size_t i = 0; i < h_residual_out_gpu.size(); ++i) {
        float g_val = __half2float(h_residual_out_gpu[i]);
        float r_val = __half2float(h_residual_out_ref[i]);
        float diff = std::abs(g_val - r_val);
        if (diff > max_err_post_exp) max_err_post_exp = diff;
    }
    std::cout << "HC Post-Expansion Max Error: " << max_err_post_exp << std::endl;
    assert(max_err_post_exp < 1e-4f);

    CHECK_HIP(hipFree(d_mixes));
    CHECK_HIP(hipFree(d_scale));
    CHECK_HIP(hipFree(d_base));
    CHECK_HIP(hipFree(d_pre));
    CHECK_HIP(hipFree(d_post));
    CHECK_HIP(hipFree(d_comb));
    CHECK_HIP(hipFree(d_layer_out));
    CHECK_HIP(hipFree(d_res_in));
    CHECK_HIP(hipFree(d_res_out));

    std::cout << "[Test PASS] Hyper-Connections Sinkhorn & Post-Expansion validated on silicon!" << std::endl;
    return 0;
}
