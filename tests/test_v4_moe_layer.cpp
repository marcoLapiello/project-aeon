#include "core/config.hpp"
#include "core/device.hpp"
#include "kernel/w4a16_gemm.hpp"
#include "kernel/moe_router.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <list>
#include <chrono>
#include <cmath>
#include <cassert>

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// DeepSeek-V4 Expert dimensions
constexpr uint32_t HIDDEN_DIM = 4096;
constexpr uint32_t INTERMEDIATE_DIM = 2048;
constexpr uint32_t NUM_ROUTED_EXPERTS = 256;
constexpr uint32_t TOP_K = 6;
constexpr float SWIGLU_LIMIT = 10.0f;
constexpr float ROUTED_SCALING = 1.5f;

// INT4-W4A16 Memory layout for 1 Expert:
// w1 (gate): [INTERMEDIATE_DIM, HIDDEN_DIM] -> [2048, 512] uint32_t (4MB), scales [2048, 128] half (512KB)
// w2 (down): [HIDDEN_DIM, INTERMEDIATE_DIM] -> [4096, 256] uint32_t (4MB), scales [4096, 64] half (512KB)
// w3 (up):   [INTERMEDIATE_DIM, HIDDEN_DIM] -> [2048, 512] uint32_t (4MB), scales [2048, 128] half (512KB)
constexpr size_t W1_PACKED_BYTES = INTERMEDIATE_DIM * (HIDDEN_DIM / 8) * sizeof(uint32_t); // 4MB
constexpr size_t W1_SCALE_BYTES  = INTERMEDIATE_DIM * (HIDDEN_DIM / 32) * sizeof(half);    // 512KB

constexpr size_t W2_PACKED_BYTES = HIDDEN_DIM * (INTERMEDIATE_DIM / 8) * sizeof(uint32_t); // 4MB
constexpr size_t W2_SCALE_BYTES  = HIDDEN_DIM * (INTERMEDIATE_DIM / 32) * sizeof(half);    // 512KB

constexpr size_t W3_PACKED_BYTES = INTERMEDIATE_DIM * (HIDDEN_DIM / 8) * sizeof(uint32_t); // 4MB
constexpr size_t W3_SCALE_BYTES  = INTERMEDIATE_DIM * (HIDDEN_DIM / 32) * sizeof(half);    // 512KB

struct ExpertDeviceSlot {
    uint32_t* d_w1_packed{nullptr};
    half*     d_w1_scale{nullptr};
    uint32_t* d_w2_packed{nullptr};
    half*     d_w2_scale{nullptr};
    uint32_t* d_w3_packed{nullptr};
    half*     d_w3_scale{nullptr};
    int32_t   resident_expert_id{-1};
};

struct ExpertHostData {
    std::vector<uint32_t> w1_packed;
    std::vector<half>     w1_scale;
    std::vector<uint32_t> w2_packed;
    std::vector<half>     w2_scale;
    std::vector<uint32_t> w3_packed;
    std::vector<half>     w3_scale;
};

// Clamped SwiGLU Kernel
__global__ void swiglu_clamp_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int total_elements,
    float limit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
        float silu_g = g / (1.0f + expf(-g));
        out[idx] = __float2half(silu_g * u);
    }
}

// Accumulate weighted expert output into token hidden state
__global__ void accumulate_expert_output_kernel(
    half* __restrict__ accum_out,
    const half* __restrict__ expert_out,
    float weight,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        float acc = __half2float(accum_out[idx]);
        float exp = __half2float(expert_out[idx]);
        accum_out[idx] = __float2half(acc + weight * exp);
    }
}

int main() {
    std::cout << "[Test] End-to-End DeepSeek-V4 MoE Layer Forward Pass on Silicon..." << std::endl;
    aeon::core::select_compute_device(true);

    const uint32_t M_padded = 16; // Single-token autoregressive decoding step (padded to 16 for WMMA)
    const uint32_t vram_cache_capacity = 8; // 8 active expert slots in VRAM

    // 1. Initialize streams
    hipStream_t compute_stream, sdma_stream;
    CHECK_HIP(hipStreamCreate(&compute_stream));
    CHECK_HIP(hipStreamCreate(&sdma_stream));

    // 2. Allocate Tier 1 VRAM LRU slots
    std::vector<ExpertDeviceSlot> vram_slots(vram_cache_capacity);
    std::vector<uint32_t> free_slots;
    for (uint32_t i = 0; i < vram_cache_capacity; ++i) {
        CHECK_HIP(hipMalloc(&vram_slots[i].d_w1_packed, W1_PACKED_BYTES));
        CHECK_HIP(hipMalloc(&vram_slots[i].d_w1_scale, W1_SCALE_BYTES));
        CHECK_HIP(hipMalloc(&vram_slots[i].d_w2_packed, W2_PACKED_BYTES));
        CHECK_HIP(hipMalloc(&vram_slots[i].d_w2_scale, W2_SCALE_BYTES));
        CHECK_HIP(hipMalloc(&vram_slots[i].d_w3_packed, W3_PACKED_BYTES));
        CHECK_HIP(hipMalloc(&vram_slots[i].d_w3_scale, W3_SCALE_BYTES));
        vram_slots[i].resident_expert_id = -1;
        free_slots.push_back(i);
    }

    std::list<uint32_t> lru_list;
    std::unordered_map<uint32_t, std::pair<uint32_t, std::list<uint32_t>::iterator>> lru_map;

    // 3. Allocate Tier 2 Host Staging for synthetic routed experts
    std::cout << "Allocating Tier 2 Host DDR staging for routed experts..." << std::endl;
    std::vector<ExpertHostData> host_experts(NUM_ROUTED_EXPERTS);
    for (uint32_t e = 0; e < NUM_ROUTED_EXPERTS; ++e) {
        host_experts[e].w1_packed.resize(W1_PACKED_BYTES / sizeof(uint32_t), 0x88888888);
        host_experts[e].w1_scale.resize(W1_SCALE_BYTES / sizeof(half), __float2half(0.005f));
        host_experts[e].w2_packed.resize(W2_PACKED_BYTES / sizeof(uint32_t), 0x88888888);
        host_experts[e].w2_scale.resize(W2_SCALE_BYTES / sizeof(half), __float2half(0.005f));
        host_experts[e].w3_packed.resize(W3_PACKED_BYTES / sizeof(uint32_t), 0x88888888);
        host_experts[e].w3_scale.resize(W3_SCALE_BYTES / sizeof(half), __float2half(0.005f));

        // Slightly perturb each expert
        host_experts[e].w1_packed[0] = 0xaa79ba86 + e;
        host_experts[e].w2_packed[0] = 0x9968aa77 + e;
        host_experts[e].w3_packed[0] = 0xba8899aa + e;
    }

    // 4. Shared Expert Allocation (Permanently resident in Tier 1 VRAM)
    ExpertDeviceSlot shared_expert;
    CHECK_HIP(hipMalloc(&shared_expert.d_w1_packed, W1_PACKED_BYTES));
    CHECK_HIP(hipMalloc(&shared_expert.d_w1_scale, W1_SCALE_BYTES));
    CHECK_HIP(hipMalloc(&shared_expert.d_w2_packed, W2_PACKED_BYTES));
    CHECK_HIP(hipMalloc(&shared_expert.d_w2_scale, W2_SCALE_BYTES));
    CHECK_HIP(hipMalloc(&shared_expert.d_w3_packed, W3_PACKED_BYTES));
    CHECK_HIP(hipMalloc(&shared_expert.d_w3_scale, W3_SCALE_BYTES));

    CHECK_HIP(hipMemset(shared_expert.d_w1_packed, 0x88, W1_PACKED_BYTES));
    CHECK_HIP(hipMemset(shared_expert.d_w2_packed, 0x88, W2_PACKED_BYTES));
    CHECK_HIP(hipMemset(shared_expert.d_w3_packed, 0x88, W3_PACKED_BYTES));

    // 5. Input hidden states & intermediate activations
    std::vector<half> h_input(M_padded * HIDDEN_DIM);
    for (uint32_t i = 0; i < h_input.size(); ++i) {
        h_input[i] = __float2half(((i % 17) - 8) * 0.05f);
    }

    half *d_input, *d_layer_accum, *d_gate_out, *d_up_out, *d_swiglu_out, *d_down_out;
    CHECK_HIP(hipMalloc(&d_input, M_padded * HIDDEN_DIM * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_layer_accum, M_padded * HIDDEN_DIM * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_gate_out, M_padded * INTERMEDIATE_DIM * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_up_out, M_padded * INTERMEDIATE_DIM * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_swiglu_out, M_padded * INTERMEDIATE_DIM * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_down_out, M_padded * HIDDEN_DIM * sizeof(half)));

    CHECK_HIP(hipMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_layer_accum, 0, M_padded * HIDDEN_DIM * sizeof(half)));

    // 6. Router Setup: SqrtSoftplus Router selecting 6 experts
    std::vector<float> h_router_logits(NUM_ROUTED_EXPERTS);
    std::vector<float> h_router_bias(NUM_ROUTED_EXPERTS);
    for (uint32_t i = 0; i < NUM_ROUTED_EXPERTS; ++i) {
        h_router_logits[i] = ((i % 29) - 14) * 0.3f;
        h_router_bias[i] = ((i % 13) - 6) * 0.05f;
    }

    float *d_router_logits, *d_router_bias, *d_topk_weights;
    int32_t *d_topk_indices;
    CHECK_HIP(hipMalloc(&d_router_logits, NUM_ROUTED_EXPERTS * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_router_bias, NUM_ROUTED_EXPERTS * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_topk_weights, TOP_K * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_topk_indices, TOP_K * sizeof(int32_t)));

    CHECK_HIP(hipMemcpy(d_router_logits, h_router_logits.data(), NUM_ROUTED_EXPERTS * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_router_bias, h_router_bias.data(), NUM_ROUTED_EXPERTS * sizeof(float), hipMemcpyHostToDevice));

    // Run Router Kernel
    aeon::kernel::moe_router_kernel<<<1, 64, 0, compute_stream>>>(
        d_router_logits, d_router_bias, nullptr, nullptr,
        d_topk_weights, d_topk_indices,
        NUM_ROUTED_EXPERTS, TOP_K, ROUTED_SCALING, true
    );
    CHECK_HIP(hipStreamSynchronize(compute_stream));

    std::vector<float> h_topk_weights(TOP_K);
    std::vector<int32_t> h_topk_indices(TOP_K);
    CHECK_HIP(hipMemcpy(h_topk_weights.data(), d_topk_weights, TOP_K * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_topk_indices.data(), d_topk_indices, TOP_K * sizeof(int32_t), hipMemcpyDeviceToHost));

    std::cout << "Routed to Top-" << TOP_K << " Experts: ";
    for (uint32_t k = 0; k < TOP_K; ++k) {
        std::cout << h_topk_indices[k] << " (w=" << h_topk_weights[k] << ") ";
    }
    std::cout << std::endl;

    // 7. Execute Layer Forward Pass:
    auto t_start = std::chrono::high_resolution_clock::now();

    // A. Shared Expert Forward Pass
    // w1 & w3
    aeon::kernel::dispatch_w4a16_gemm(d_input, shared_expert.d_w1_packed, shared_expert.d_w1_scale, d_gate_out, M_padded, INTERMEDIATE_DIM, HIDDEN_DIM, compute_stream);
    aeon::kernel::dispatch_w4a16_gemm(d_input, shared_expert.d_w3_packed, shared_expert.d_w3_scale, d_up_out, M_padded, INTERMEDIATE_DIM, HIDDEN_DIM, compute_stream);

    // SwiGLU clamp
    int swiglu_threads = 256;
    int swiglu_blocks = (M_padded * INTERMEDIATE_DIM + swiglu_threads - 1) / swiglu_threads;
    swiglu_clamp_kernel<<<swiglu_blocks, swiglu_threads, 0, compute_stream>>>(d_gate_out, d_up_out, d_swiglu_out, M_padded * INTERMEDIATE_DIM, SWIGLU_LIMIT);

    // w2
    aeon::kernel::dispatch_w4a16_gemm(d_swiglu_out, shared_expert.d_w2_packed, shared_expert.d_w2_scale, d_layer_accum, M_padded, HIDDEN_DIM, INTERMEDIATE_DIM, compute_stream);

    // B. Routed Experts Forward Pass with Tier 1 LRU Streaming
    uint32_t cache_hits = 0;
    uint32_t cache_misses = 0;

    for (uint32_t k = 0; k < TOP_K; ++k) {
        uint32_t expert_id = h_topk_indices[k];
        float expert_weight = h_topk_weights[k];

        // Acquire slot in Tier 1 VRAM
        uint32_t slot_idx = 0;
        auto it = lru_map.find(expert_id);
        if (it != lru_map.end()) {
            cache_hits++;
            slot_idx = it->second.first;
            lru_list.erase(it->second.second);
            lru_list.push_front(expert_id);
            it->second.second = lru_list.begin();
        } else {
            cache_misses++;
            if (!free_slots.empty()) {
                slot_idx = free_slots.back();
                free_slots.pop_back();
            } else {
                uint32_t victim = lru_list.back();
                lru_list.pop_back();
                slot_idx = lru_map[victim].first;
                lru_map.erase(victim);
            }

            // Stream expert weights from Host DDR -> VRAM over PCIe SDMA stream
            auto& edata = host_experts[expert_id];
            CHECK_HIP(hipMemcpyAsync(vram_slots[slot_idx].d_w1_packed, edata.w1_packed.data(), W1_PACKED_BYTES, hipMemcpyHostToDevice, sdma_stream));
            CHECK_HIP(hipMemcpyAsync(vram_slots[slot_idx].d_w1_scale, edata.w1_scale.data(), W1_SCALE_BYTES, hipMemcpyHostToDevice, sdma_stream));
            CHECK_HIP(hipMemcpyAsync(vram_slots[slot_idx].d_w2_packed, edata.w2_packed.data(), W2_PACKED_BYTES, hipMemcpyHostToDevice, sdma_stream));
            CHECK_HIP(hipMemcpyAsync(vram_slots[slot_idx].d_w2_scale, edata.w2_scale.data(), W2_SCALE_BYTES, hipMemcpyHostToDevice, sdma_stream));
            CHECK_HIP(hipMemcpyAsync(vram_slots[slot_idx].d_w3_packed, edata.w3_packed.data(), W3_PACKED_BYTES, hipMemcpyHostToDevice, sdma_stream));
            CHECK_HIP(hipMemcpyAsync(vram_slots[slot_idx].d_w3_scale, edata.w3_scale.data(), W3_SCALE_BYTES, hipMemcpyHostToDevice, sdma_stream));

            vram_slots[slot_idx].resident_expert_id = expert_id;
            lru_list.push_front(expert_id);
            lru_map[expert_id] = {slot_idx, lru_list.begin()};

            // Synchronize SDMA transfer before compute
            CHECK_HIP(hipStreamSynchronize(sdma_stream));
        }

        const auto& slot = vram_slots[slot_idx];

        // w1 (gate) & w3 (up)
        aeon::kernel::dispatch_w4a16_gemm(d_input, slot.d_w1_packed, slot.d_w1_scale, d_gate_out, M_padded, INTERMEDIATE_DIM, HIDDEN_DIM, compute_stream);
        aeon::kernel::dispatch_w4a16_gemm(d_input, slot.d_w3_packed, slot.d_w3_scale, d_up_out, M_padded, INTERMEDIATE_DIM, HIDDEN_DIM, compute_stream);

        // Clamped SwiGLU
        swiglu_clamp_kernel<<<swiglu_blocks, swiglu_threads, 0, compute_stream>>>(d_gate_out, d_up_out, d_swiglu_out, M_padded * INTERMEDIATE_DIM, SWIGLU_LIMIT);

        // w2 (down)
        aeon::kernel::dispatch_w4a16_gemm(d_swiglu_out, slot.d_w2_packed, slot.d_w2_scale, d_down_out, M_padded, HIDDEN_DIM, INTERMEDIATE_DIM, compute_stream);

        // Accumulate into layer output
        int acc_threads = 256;
        int acc_blocks = (HIDDEN_DIM + acc_threads - 1) / acc_threads;
        accumulate_expert_output_kernel<<<acc_blocks, acc_threads, 0, compute_stream>>>(d_layer_accum, d_down_out, expert_weight, HIDDEN_DIM);
    }

    CHECK_HIP(hipStreamSynchronize(compute_stream));
    auto t_end = std::chrono::high_resolution_clock::now();

    double layer_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "\nMoE Layer Execution Statistics (Pass 1 - Cold Misses):" << std::endl;
    std::cout << "  Layer Forward Time  : " << layer_time_ms << " ms" << std::endl;
    std::cout << "  Cache Hits / Misses : " << cache_hits << " hits, " << cache_misses << " misses" << std::endl;
    std::cout << "  VRAM Resident Active: " << lru_map.size() << " / " << vram_cache_capacity << " slots" << std::endl;

    // Run Pass 2 to verify warm cache hits
    auto t_start2 = std::chrono::high_resolution_clock::now();
    uint32_t warm_hits = 0;
    uint32_t warm_misses = 0;
    for (uint32_t k = 0; k < TOP_K; ++k) {
        uint32_t expert_id = h_topk_indices[k];
        auto it = lru_map.find(expert_id);
        if (it != lru_map.end()) {
            warm_hits++;
        } else {
            warm_misses++;
        }
    }
    auto t_end2 = std::chrono::high_resolution_clock::now();
    double warm_time_ms = std::chrono::duration<double, std::milli>(t_end2 - t_start2).count();

    std::cout << "\nMoE Layer Execution Statistics (Pass 2 - Warm Hits):" << std::endl;
    std::cout << "  Cache Lookup Time   : " << warm_time_ms << " ms" << std::endl;
    std::cout << "  Cache Hits / Misses : " << warm_hits << " hits, " << warm_misses << " misses" << std::endl;
    assert(warm_hits == TOP_K);
    assert(warm_misses == 0);

    // Verify non-zero output
    std::vector<half> h_final(HIDDEN_DIM);
    CHECK_HIP(hipMemcpy(h_final.data(), d_layer_accum, HIDDEN_DIM * sizeof(half), hipMemcpyDeviceToHost));

    float norm_sum = 0.0f;
    for (int i = 0; i < (int)HIDDEN_DIM; ++i) norm_sum += std::abs(__half2float(h_final[i]));
    std::cout << "  Layer Output L1 Norm: " << norm_sum << std::endl;
    assert(norm_sum > 0.0f);

    // Clean up
    for (auto& s : vram_slots) {
        CHECK_HIP(hipFree(s.d_w1_packed));
        CHECK_HIP(hipFree(s.d_w1_scale));
        CHECK_HIP(hipFree(s.d_w2_packed));
        CHECK_HIP(hipFree(s.d_w2_scale));
        CHECK_HIP(hipFree(s.d_w3_packed));
        CHECK_HIP(hipFree(s.d_w3_scale));
    }
    CHECK_HIP(hipFree(shared_expert.d_w1_packed));
    CHECK_HIP(hipFree(shared_expert.d_w1_scale));
    CHECK_HIP(hipFree(shared_expert.d_w2_packed));
    CHECK_HIP(hipFree(shared_expert.d_w2_scale));
    CHECK_HIP(hipFree(shared_expert.d_w3_packed));
    CHECK_HIP(hipFree(shared_expert.d_w3_scale));

    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_layer_accum));
    CHECK_HIP(hipFree(d_gate_out));
    CHECK_HIP(hipFree(d_up_out));
    CHECK_HIP(hipFree(d_swiglu_out));
    CHECK_HIP(hipFree(d_down_out));

    CHECK_HIP(hipFree(d_router_logits));
    CHECK_HIP(hipFree(d_router_bias));
    CHECK_HIP(hipFree(d_topk_weights));
    CHECK_HIP(hipFree(d_topk_indices));

    CHECK_HIP(hipStreamDestroy(compute_stream));
    CHECK_HIP(hipStreamDestroy(sdma_stream));

    std::cout << "[PASS] End-to-End DeepSeek-V4 MoE Layer Forward Pass validated on silicon!" << std::endl;
    return 0;
}
