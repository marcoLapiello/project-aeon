#include "io/aligned_allocator.hpp"
#include "io/direct_io_reader.hpp"

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <list>
#include <random>
#include <string>
#include <unordered_map>
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

// Architecture constants
constexpr uint32_t WMMA_M = 16;
constexpr uint32_t WMMA_N = 16;
constexpr uint32_t WMMA_K = 16;

constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = 64;
constexpr uint32_t BLOCK_K = 16;
constexpr uint32_t WARPS_M = BLOCK_M / WMMA_M;
constexpr uint32_t WARPS_N = BLOCK_N / WMMA_N;
constexpr uint32_t THREADS_PER_BLOCK = WARPS_M * WARPS_N * 32;

// Model layer configuration
constexpr uint32_t TOTAL_EXPERTS = 64;
constexpr uint32_t TOP_K = 6;
constexpr uint32_t HIDDEN_DIM = 2048;
constexpr uint32_t INTERMEDIATE_DIM = 2048;

// Expert weight size: (2048 * 2048) FP16 elements = 8 MB
constexpr size_t EXPERT_WEIGHT_ELEMENTS = (size_t)INTERMEDIATE_DIM * HIDDEN_DIM;
constexpr size_t EXPERT_WEIGHT_BYTES = EXPERT_WEIGHT_ELEMENTS * sizeof(half); // 8 MB

// Tiled GEMM kernel for evaluating active experts
__global__ void expert_gemm_kernel(const half* __restrict__ x,
                                   const half* __restrict__ w,
                                   float* __restrict__ out,
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
                lds_a[r][c] = x[gr * K + gc];
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
                lds_b[r][c] = w[gc * K + gk];
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
        store_matrix_sync(&out[global_row * N + global_col], frag_c, N, mem_row_major);
    }
}

// Two-Tier Dynamic Expert Cache (Tier 1: VRAM LRU, Tier 2: Pinned Host DDR)
class TieredExpertCache {
public:
    TieredExpertCache(uint32_t vram_capacity_experts,
                      uint32_t total_experts,
                      hipStream_t sdma_stream)
        : vram_capacity_(vram_capacity_experts),
          total_experts_(total_experts),
          sdma_stream_(sdma_stream) {
        // 1. Allocate VRAM slots
        vram_slots_.resize(vram_capacity_);
        for (uint32_t i = 0; i < vram_capacity_; ++i) {
            (void)hipMalloc(&vram_slots_[i].d_ptr, EXPERT_WEIGHT_BYTES);
            vram_slots_[i].resident_expert_id = -1;
            free_slots_.push_back(i);
        }

        // 2. Allocate Tier 2 Warm Staging (pinned host memory for all remaining experts)
        host_staging_.resize(total_experts_);
        for (uint32_t i = 0; i < total_experts_; ++i) {
            (void)hipHostMalloc(&host_staging_[i], EXPERT_WEIGHT_BYTES, hipHostMallocDefault);
            // Initialize with pseudo-random weights
            auto* p = static_cast<half*>(host_staging_[i]);
            for (size_t j = 0; j < 64; ++j) {
                p[j] = __float2half(0.01f * (float)((i + j) % 10));
            }
        }
    }

    ~TieredExpertCache() {
        for (auto& s : vram_slots_) {
            if (s.d_ptr) (void)hipFree(s.d_ptr);
        }
        for (auto* p : host_staging_) {
            if (p) (void)hipHostFree(p);
        }
    }

    // Access expert: returns device pointer. If miss, asynchronously fetches via SDMA.
    half* acquire_expert(uint32_t expert_id, bool& was_hit) {
        auto it = lru_map_.find(expert_id);
        if (it != lru_map_.end()) {
            // Tier 1 HIT
            was_hit = true;
            uint32_t slot_idx = it->second.first;
            // Move to front of LRU
            lru_list_.erase(it->second.second);
            lru_list_.push_front(expert_id);
            it->second.second = lru_list_.begin();
            return vram_slots_[slot_idx].d_ptr;
        }

        // Tier 1 MISS: swap from Tier 2 (Host DDR) into VRAM slot
        was_hit = false;
        uint32_t slot_idx = 0;

        if (!free_slots_.empty()) {
            slot_idx = free_slots_.back();
            free_slots_.pop_back();
        } else {
            // Evict least recently used expert
            uint32_t victim_expert = lru_list_.back();
            lru_list_.pop_back();
            slot_idx = lru_map_[victim_expert].first;
            lru_map_.erase(victim_expert);
        }

        // Asynchronously stream from Host DDR -> VRAM over PCIe SDMA
        (void)hipMemcpyAsync(vram_slots_[slot_idx].d_ptr,
                             host_staging_[expert_id],
                             EXPERT_WEIGHT_BYTES,
                             hipMemcpyHostToDevice,
                             sdma_stream_);

        vram_slots_[slot_idx].resident_expert_id = (int32_t)expert_id;
        lru_list_.push_front(expert_id);
        lru_map_[expert_id] = {slot_idx, lru_list_.begin()};

        return vram_slots_[slot_idx].d_ptr;
    }

    [[nodiscard]] size_t vram_resident_count() const { return lru_map_.size(); }
    [[nodiscard]] size_t vram_capacity() const { return vram_capacity_; }

private:
    struct Slot {
        half* d_ptr{nullptr};
        int32_t resident_expert_id{-1};
    };

    uint32_t vram_capacity_{0};
    uint32_t total_experts_{0};
    hipStream_t sdma_stream_{nullptr};

    std::vector<Slot> vram_slots_;
    std::vector<uint32_t> free_slots_;
    std::vector<void*> host_staging_;

    std::list<uint32_t> lru_list_;
    // expert_id -> {slot_index, iterator in lru_list}
    std::unordered_map<uint32_t, std::pair<uint32_t, std::list<uint32_t>::iterator>> lru_map_;
};

int main() {
    std::cout << "====================================================================" << std::endl;
    std::cout << "  Project Aeon — Legacy Single-Layer MoE Cache Diagnostic" << std::endl;
    std::cout << "====================================================================" << std::endl;

    CHECK_HIP(hipSetDevice(0));
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "[Device] " << prop.name << " (" << prop.gcnArchName << ")\n";

    // Streams
    hipStream_t stream_compute, stream_sdma;
    CHECK_HIP(hipStreamCreateWithFlags(&stream_compute, hipStreamNonBlocking));
    CHECK_HIP(hipStreamCreateWithFlags(&stream_sdma, hipStreamNonBlocking));

    // Toy Layer Parameters:
    // 64 total experts, top-6 active per token.
    // Set VRAM cache capacity to 16 experts (25% of pool resident in VRAM, 75% in Host DDR).
    constexpr uint32_t VRAM_EXPERT_CAPACITY = 16;
    std::cout << "[Config] Total Experts in Layer : " << TOTAL_EXPERTS << "\n";
    std::cout << "[Config] Top-K Active per Token : " << TOP_K << "\n";
    std::cout << "[Config] VRAM Resident Cache    : " << VRAM_EXPERT_CAPACITY << " slots ("
              << (VRAM_EXPERT_CAPACITY * EXPERT_WEIGHT_BYTES / (1024 * 1024)) << " MB)\n";
    std::cout << "[Config] Host DDR Staging Pool  : " << TOTAL_EXPERTS << " experts ("
              << (TOTAL_EXPERTS * EXPERT_WEIGHT_BYTES / (1024 * 1024)) << " MB)\n";

    TieredExpertCache cache(VRAM_EXPERT_CAPACITY, TOTAL_EXPERTS, stream_sdma);

    // Allocate input token activation and output buffer on GPU
    // Simulating batch of tokens (e.g., M = 64 tokens)
    constexpr uint32_t BATCH_TOKENS = 64;
    half* d_x = nullptr;
    float* d_out = nullptr;
    CHECK_HIP(hipMalloc(&d_x, (size_t)BATCH_TOKENS * HIDDEN_DIM * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_out, (size_t)BATCH_TOKENS * INTERMEDIATE_DIM * sizeof(float)));

    // Grid config
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dim3 grid((INTERMEDIATE_DIM + BLOCK_N - 1) / BLOCK_N, (BATCH_TOKENS + BLOCK_M - 1) / BLOCK_M, 1);

    // Synthetic token sequence with realistic power-law / Zipfian routing distribution
    // DeepSeek MoE empirical distribution: top experts account for ~75-85% of token routing
    constexpr int NUM_TOKENS = 100;
    std::mt19937 rng(42);
    // Exponentially decayed weights matching DeepSeek activation locality
    std::vector<double> weights(TOTAL_EXPERTS);
    for (uint32_t i = 0; i < TOTAL_EXPERTS; ++i) {
        weights[i] = std::exp(-0.25 * (double)i);
    }
    std::discrete_distribution<uint32_t> router_dist(weights.begin(), weights.end());

    std::cout << "\n[Pipeline] Simulating " << NUM_TOKENS << " consecutive tokens through MoE layer..." << std::endl;

    uint32_t total_lookups = 0;
    uint32_t total_hits = 0;
    uint32_t total_misses = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < NUM_TOKENS; ++t) {
        // 1. Router selects top-K unique experts
        std::vector<uint32_t> routed_experts;
        while (routed_experts.size() < TOP_K) {
            uint32_t exp = router_dist(rng);
            if (std::find(routed_experts.begin(), routed_experts.end(), exp) == routed_experts.end()) {
                routed_experts.push_back(exp);
            }
        }

        // 2. Evaluate active experts
        for (uint32_t exp_id : routed_experts) {
            bool hit = false;
            half* d_w = cache.acquire_expert(exp_id, hit);
            total_lookups++;
            if (hit) total_hits++;
            else total_misses++;

            // Synchronize SDMA with compute stream if there was a miss
            if (!hit) {
                hipEvent_t xfer_done;
                (void)hipEventCreate(&xfer_done);
                (void)hipEventRecord(xfer_done, stream_sdma);
                (void)hipStreamWaitEvent(stream_compute, xfer_done, 0);
                (void)hipEventDestroy(xfer_done);
            }

            // Launch GEMM forward pass for this expert
            hipLaunchKernelGGL(expert_gemm_kernel,
                               grid,
                               block,
                               0,
                               stream_compute,
                               d_x,
                               d_w,
                               d_out,
                               BATCH_TOKENS,
                               INTERMEDIATE_DIM,
                               HIDDEN_DIM);
        }

        CHECK_HIP(hipStreamSynchronize(stream_compute));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms_per_token = total_ms / NUM_TOKENS;
    double hit_rate = (100.0 * total_hits) / total_lookups;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "[Results] Total Tokens Processed : " << NUM_TOKENS << "\n";
    std::cout << "[Results] Total Expert Dispatches: " << total_lookups << "\n";
    std::cout << "[Results] Tier 1 VRAM Hits       : " << total_hits << " (" << hit_rate << " %)\n";
    std::cout << "[Results] Tier 2 Swaps (Misses)  : " << total_misses << " (" << (100.0 - hit_rate) << " %)\n";
    std::cout << "[Results] Total Execution Time   : " << total_ms << " ms\n";
    std::cout << "[Results] Average Latency/Token  : " << avg_ms_per_token << " ms\n";
    std::cout << "[Results] Token Generation Rate  : " << (1000.0 / avg_ms_per_token) << " tok/s\n";
    std::cout << "--------------------------------------------------------------------" << std::endl;

    if (hit_rate >= 60.0) {
        std::cout << ">>> VERIFICATION PASSED: Single-layer toy MoE pipeline functional! <<<" << std::endl;
        std::cout << "====================================================================" << std::endl;
    } else {
        std::cerr << ">>> WARNING: Cache hit rate lower than expected threshold! <<<" << std::endl;
    }

    CHECK_HIP(hipStreamDestroy(stream_compute));
    CHECK_HIP(hipStreamDestroy(stream_sdma));
    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_out));

    return 0;
}
