#include "kernel/moe_router.hpp"
#include <iostream>
#include <fstream>
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
    std::cout << "[Test] Spike 4: Dual-Mode MoE Router Validation on Silicon..." << std::endl;

    const int num_tokens = 4;
    const int n_routed_experts = 256;
    const int top_k = 6;
    const float routed_scaling_factor = 1.5f;

    // 1. SqrtSoftplus Router with Bias (Layers 3..42)
    {
        std::cout << "--- Sub-Test 1: SqrtSoftplus Router with Bias ---" << std::endl;
        std::vector<float> h_logits(num_tokens * n_routed_experts);
        std::vector<float> h_bias(n_routed_experts);

        for (int i = 0; i < (int)h_logits.size(); ++i) {
            h_logits[i] = ((i % 31) - 15) * 0.2f;
        }
        for (int i = 0; i < n_routed_experts; ++i) {
            h_bias[i] = ((i % 17) - 8) * 0.05f;
        }

        std::vector<float> ref_weights(num_tokens * top_k);
        std::vector<int32_t> ref_indices(num_tokens * top_k);

        aeon::kernel::cpu_moe_router(
            h_logits.data(), h_bias.data(), nullptr, nullptr,
            ref_weights.data(), ref_indices.data(),
            num_tokens, n_routed_experts, top_k, routed_scaling_factor, true
        );

        float *d_logits, *d_bias, *d_weights;
        int32_t *d_indices;

        CHECK_HIP(hipMalloc(&d_logits, h_logits.size() * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_bias, h_bias.size() * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_weights, ref_weights.size() * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_indices, ref_indices.size() * sizeof(int32_t)));

        CHECK_HIP(hipMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), hipMemcpyHostToDevice));

        aeon::kernel::moe_router_kernel<<<num_tokens, 64>>>(
            d_logits, d_bias, nullptr, nullptr,
            d_weights, d_indices,
            n_routed_experts, top_k, routed_scaling_factor, true
        );
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<float> gpu_weights(ref_weights.size());
        std::vector<int32_t> gpu_indices(ref_indices.size());

        CHECK_HIP(hipMemcpy(gpu_weights.data(), d_weights, gpu_weights.size() * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(gpu_indices.data(), d_indices, gpu_indices.size() * sizeof(int32_t), hipMemcpyDeviceToHost));

        for (int t = 0; t < num_tokens; ++t) {
            std::cout << "Token " << t << " Selected Experts: [";
            for (int k = 0; k < top_k; ++k) {
                std::cout << gpu_indices[t * top_k + k] << (k == top_k - 1 ? "" : ", ");
                assert(gpu_indices[t * top_k + k] == ref_indices[t * top_k + k]);
                float diff = std::abs(gpu_weights[t * top_k + k] - ref_weights[t * top_k + k]);
                assert(diff < 1e-4f);
            }
            std::cout << "]" << std::endl;
        }

        CHECK_HIP(hipFree(d_logits));
        CHECK_HIP(hipFree(d_bias));
        CHECK_HIP(hipFree(d_weights));
        CHECK_HIP(hipFree(d_indices));
        std::cout << "[PASS] SqrtSoftplus router successfully matched CPU reference!" << std::endl;
    }

    // 2. Hash Router (Layers 0..2 using real tid2eid table)
    {
        std::cout << "\n--- Sub-Test 2: Hash Router (Layers 0..2) ---" << std::endl;
        const int vocab_size = 129280;
        std::vector<int64_t> h_hash_table(vocab_size * top_k);
        for (int v = 0; v < vocab_size; ++v) {
            for (int k = 0; k < top_k; ++k) {
                h_hash_table[v * top_k + k] = (v * 7 + k * 13) % n_routed_experts;
            }
        }

        std::vector<int32_t> h_tokens = {101, 2054, 42, 9999};
        std::vector<float> h_logits(num_tokens * n_routed_experts);
        for (int i = 0; i < (int)h_logits.size(); ++i) {
            h_logits[i] = ((i % 23) - 11) * 0.1f;
        }

        std::vector<float> ref_weights(num_tokens * top_k);
        std::vector<int32_t> ref_indices(num_tokens * top_k);

        aeon::kernel::cpu_moe_router(
            h_logits.data(), nullptr, h_hash_table.data(), h_tokens.data(),
            ref_weights.data(), ref_indices.data(),
            num_tokens, n_routed_experts, top_k, routed_scaling_factor, true
        );

        float *d_logits, *d_weights;
        int64_t *d_hash;
        int32_t *d_tokens, *d_indices;

        CHECK_HIP(hipMalloc(&d_logits, h_logits.size() * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_hash, h_hash_table.size() * sizeof(int64_t)));
        CHECK_HIP(hipMalloc(&d_tokens, h_tokens.size() * sizeof(int32_t)));
        CHECK_HIP(hipMalloc(&d_weights, ref_weights.size() * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_indices, ref_indices.size() * sizeof(int32_t)));

        CHECK_HIP(hipMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_hash, h_hash_table.data(), h_hash_table.size() * sizeof(int64_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_tokens, h_tokens.data(), h_tokens.size() * sizeof(int32_t), hipMemcpyHostToDevice));

        aeon::kernel::moe_router_kernel<<<num_tokens, 64>>>(
            d_logits, nullptr, d_hash, d_tokens,
            d_weights, d_indices,
            n_routed_experts, top_k, routed_scaling_factor, true
        );
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<float> gpu_weights(ref_weights.size());
        std::vector<int32_t> gpu_indices(ref_indices.size());

        CHECK_HIP(hipMemcpy(gpu_weights.data(), d_weights, gpu_weights.size() * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(gpu_indices.data(), d_indices, gpu_indices.size() * sizeof(int32_t), hipMemcpyDeviceToHost));

        for (int t = 0; t < num_tokens; ++t) {
            std::cout << "Token ID " << h_tokens[t] << " Pre-Routed Experts: [";
            for (int k = 0; k < top_k; ++k) {
                std::cout << gpu_indices[t * top_k + k] << (k == top_k - 1 ? "" : ", ");
                assert(gpu_indices[t * top_k + k] == ref_indices[t * top_k + k]);
                float diff = std::abs(gpu_weights[t * top_k + k] - ref_weights[t * top_k + k]);
                assert(diff < 1e-4f);
            }
            std::cout << "]" << std::endl;
        }

        CHECK_HIP(hipFree(d_logits));
        CHECK_HIP(hipFree(d_hash));
        CHECK_HIP(hipFree(d_tokens));
        CHECK_HIP(hipFree(d_weights));
        CHECK_HIP(hipFree(d_indices));
        std::cout << "[PASS] Hash router successfully matched CPU reference!" << std::endl;
    }

    std::cout << "\n[ALL TESTS PASSED] Spike 4: Dual-Mode MoE Router validated on silicon!" << std::endl;
    return 0;
}
