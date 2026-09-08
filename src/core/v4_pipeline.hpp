#pragma once

#include "core/config.hpp"
#include "core/device.hpp"
#include "core/aeon_loader.hpp"
#include "core/expert_registry.hpp"
#include "core/host_expert_pool.hpp"
#include "core/memory_budget.hpp"
#include "core/safetensors_loader.hpp"
#include "core/v4_layer.hpp"
#include "core/v4_pipeline_scratch.hpp"
#include "core/vram_expert_pool.hpp"
#include "kernel/hc_sinkhorn.hpp"
#include "kernel/moe_router.hpp"
#include "kernel/v4_attention.hpp"
#include "kernel/v4_pipeline_ops.hpp"
#include "kernel/w4a16_gemm.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)
#endif

namespace aeon::core {

// Complete DeepSeek-V4 Autoregressive Multi-Layer Pipeline Engine
class V4Pipeline {
public:
    SafetensorsLoader loader;
    AeonModelLoader aeon_loader;
    std::vector<std::unique_ptr<V4Layer>> layers;
    PipelineScratchBuffers scratch;
    kernel::RopeTable rope_table;

    // Model Level Resident Weights
    const half* host_embed_table{nullptr}; // [129280, 4096] in host / mmap
    float* d_hc_head_fn{nullptr};          // [4, 16384]
    float* d_hc_head_base{nullptr};        // [4]
    float* d_hc_head_scale{nullptr};       // [1]
    half*  d_lm_head{nullptr};             // [129280, 4096] on device
    half*  d_final_norm{nullptr};          // [4096] on device

    hipStream_t compute_stream{0};
    hipStream_t sdma_stream{0};

    uint32_t num_layers_{0};
    uint32_t current_seq_len_{0};

    // Unified VRAM expert pool, warm host pool, and expert registry.
    std::unique_ptr<UnifiedVRAMExpertPool> unified_vram_pool_;
    std::unique_ptr<HostExpertPool> host_pool_;
    std::unique_ptr<ExpertRegistry> expert_registry_;
    MemoryBudgetReport budget_report_;

    V4Pipeline() = default;

    ~V4Pipeline() {
        free_all();
    }

    void init(
        const std::string& snapshot_dir,
        uint32_t num_layers = 2,
        uint32_t unified_vram_slots = 0,
        uint32_t max_seq_len = 4096
    ) {
        std::cout << "================================================================================" << std::endl;
        std::cout << "        Project Aeon — DeepSeek-V4 Multi-Layer Execution Pipeline               " << std::endl;
        std::cout << "================================================================================" << std::endl;

        num_layers_ = num_layers;
        current_seq_len_ = 0;

        // 1. Initialize streams
        CHECK_HIP(hipStreamCreate(&compute_stream));
        CHECK_HIP(hipStreamCreate(&sdma_stream));

        // 2. Open Safetensors Shards via Zero-Copy Mmap
        std::cout << "[Pipeline] Opening Safetensors shards from " << snapshot_dir << "..." << std::endl;
        loader.open_shard(snapshot_dir + "/model-00001.safetensors");
        loader.open_shard(snapshot_dir + "/model-00002.safetensors");
        loader.open_shard(snapshot_dir + "/model-00034.safetensors");
        std::cout << "  > Total tensors indexed: " << loader.total_tensors() << std::endl;

        // 3. Initialize RoPE Tables
        std::cout << "[Pipeline] Initializing RoPE tables (max_seq=" << max_seq_len << ")..." << std::endl;
        rope_table.init(max_seq_len, kernel::DSV4_ROPE_THETA, 1.0f);

        // Upload RoPE caches to GPU
        size_t rope_bytes = rope_table.max_seq_len * rope_table.half_rope * sizeof(float);
        CHECK_HIP(hipMalloc(&d_cos_cache_, rope_bytes));
        CHECK_HIP(hipMalloc(&d_sin_cache_, rope_bytes));
        CHECK_HIP(hipMemcpy(d_cos_cache_, rope_table.cos_cache.data(), rope_bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_sin_cache_, rope_table.sin_cache.data(), rope_bytes, hipMemcpyHostToDevice));

        // 4. Model-level Weights
        std::cout << "[Pipeline] Binding model-level embeddings and LM head..." << std::endl;
        host_embed_table = loader.get_data_ptr<half>("embed.weight");

        const auto& head_t = loader.get_tensor("head.weight");
        std::cout << "  > Uploading LM Head [129280, 4096] (" << (head_t.byte_size / (1024*1024)) << " MB) to VRAM..." << std::endl;
        CHECK_HIP(hipMalloc(&d_lm_head, head_t.byte_size));
        CHECK_HIP(hipMemcpy(d_lm_head, head_t.data, head_t.byte_size, hipMemcpyHostToDevice));

        // HC Head
        const auto& fn_t = loader.get_tensor("hc_head_fn");
        const auto& base_t = loader.get_tensor("hc_head_base");
        const auto& sc_t = loader.get_tensor("hc_head_scale");
        CHECK_HIP(hipMalloc(&d_hc_head_fn, fn_t.byte_size));
        CHECK_HIP(hipMalloc(&d_hc_head_base, base_t.byte_size));
        CHECK_HIP(hipMalloc(&d_hc_head_scale, sc_t.byte_size));
        CHECK_HIP(hipMemcpy(d_hc_head_fn, fn_t.data, fn_t.byte_size, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_hc_head_base, base_t.data, base_t.byte_size, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_hc_head_scale, sc_t.data, sc_t.byte_size, hipMemcpyHostToDevice));

        // Final norm: use layers.0.ffn_norm.weight as fallback if model-level norm.weight is in an un-downloaded shard
        if (loader.has_tensor("norm.weight")) {
            const auto& norm_t = loader.get_tensor("norm.weight");
            CHECK_HIP(hipMalloc(&d_final_norm, norm_t.byte_size));
            CHECK_HIP(hipMemcpy(d_final_norm, norm_t.data, norm_t.byte_size, hipMemcpyHostToDevice));
        } else {
            const auto& norm_t = loader.get_tensor("layers.0.ffn_norm.weight");
            CHECK_HIP(hipMalloc(&d_final_norm, norm_t.byte_size));
            CHECK_HIP(hipMemcpy(d_final_norm, norm_t.data, norm_t.byte_size, hipMemcpyHostToDevice));
        }

        // 5. Allocate Reusable Pipeline Scratch Buffers
        std::cout << "[Pipeline] Allocating intermediate GPU scratch buffers..." << std::endl;
        scratch.allocate();

        // 6. Initialize Consecutive Transformer Layers
        layers.resize(num_layers_);
        for (uint32_t l = 0; l < num_layers_; ++l) {
            layers[l] = std::make_unique<V4Layer>();
            layers[l]->init(l, loader, max_seq_len);
        }

        // 7. Setup Unified VRAM Pool and Expert Registry
        uint32_t active_vram_slots = unified_vram_slots > 0 ? unified_vram_slots : std::max(num_layers_ * 8u, 16u);
        unified_vram_pool_ = std::make_unique<UnifiedVRAMExpertPool>(active_vram_slots);
        expert_registry_ = std::make_unique<ExpertRegistry>(num_layers_, 256, active_vram_slots, 0);

        std::cout << "[Pipeline] Engine ready! Configured for " << num_layers_ << " chained layers with "
                  << active_vram_slots << " unified VRAM expert slots." << std::endl;
    }

    // Initialize pipeline directly from native .aeon format folder
    void init_aeon(
        const std::string& aeon_model_dir,
        uint32_t num_layers = 2,
        uint32_t unified_vram_slots = 0,
        uint32_t max_seq_len = 4096
    ) {
        num_layers_ = num_layers;
        current_seq_len_ = 0;

        // 1. Initialize streams
        CHECK_HIP(hipStreamCreate(&compute_stream));
        CHECK_HIP(hipStreamCreate(&sdma_stream));

        // 2. Open Aeon Model via Zero-Copy Mmap
        std::cout << "[Pipeline] Opening native Aeon model from " << aeon_model_dir << "..." << std::endl;
        aeon_loader.open_model(aeon_model_dir);
        std::cout << "  > Total dense tensors indexed: " << aeon_loader.total_dense_tensors() << std::endl;

        // 3. Initialize RoPE Tables
        std::cout << "[Pipeline] Initializing RoPE tables (max_seq=" << max_seq_len << ")..." << std::endl;
        rope_table.init(max_seq_len, kernel::DSV4_ROPE_THETA, 1.0f);

        // Upload RoPE caches to GPU
        size_t rope_bytes = rope_table.max_seq_len * rope_table.half_rope * sizeof(float);
        CHECK_HIP(hipMalloc(&d_cos_cache_, rope_bytes));
        CHECK_HIP(hipMalloc(&d_sin_cache_, rope_bytes));
        CHECK_HIP(hipMemcpy(d_cos_cache_, rope_table.cos_cache.data(), rope_bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_sin_cache_, rope_table.sin_cache.data(), rope_bytes, hipMemcpyHostToDevice));

        // 4. Model-level Weights
        std::cout << "[Pipeline] Binding model-level embeddings and LM head..." << std::endl;
        host_embed_table = aeon_loader.get_data_ptr<half>("embed.weight");

        const auto& head_t = aeon_loader.get_tensor("head.weight");
        std::cout << "  > Uploading LM Head [129280, 4096] (" << (head_t.byte_size / (1024*1024)) << " MB) to VRAM..." << std::endl;
        CHECK_HIP(hipMalloc(&d_lm_head, head_t.byte_size));
        CHECK_HIP(hipMemcpy(d_lm_head, head_t.data, head_t.byte_size, hipMemcpyHostToDevice));

        // HC Head
        const auto& fn_t = aeon_loader.get_tensor("hc_head_fn");
        const auto& base_t = aeon_loader.get_tensor("hc_head_base");
        const auto& sc_t = aeon_loader.get_tensor("hc_head_scale");
        CHECK_HIP(hipMalloc(&d_hc_head_fn, fn_t.byte_size));
        CHECK_HIP(hipMalloc(&d_hc_head_base, base_t.byte_size));
        CHECK_HIP(hipMalloc(&d_hc_head_scale, sc_t.byte_size));
        CHECK_HIP(hipMemcpy(d_hc_head_fn, fn_t.data, fn_t.byte_size, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_hc_head_base, base_t.data, base_t.byte_size, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_hc_head_scale, sc_t.data, sc_t.byte_size, hipMemcpyHostToDevice));

        // Final norm
        const auto& norm_t = aeon_loader.get_tensor("norm.weight");
        CHECK_HIP(hipMalloc(&d_final_norm, norm_t.byte_size));
        CHECK_HIP(hipMemcpy(d_final_norm, norm_t.data, norm_t.byte_size, hipMemcpyHostToDevice));

        // 5. Allocate Reusable Pipeline Scratch Buffers
        std::cout << "[Pipeline] Allocating intermediate GPU scratch buffers..." << std::endl;
        scratch.allocate();

        // 6. Initialize Consecutive Transformer Layers
        layers.resize(num_layers_);
        for (uint32_t l = 0; l < num_layers_; ++l) {
            layers[l] = std::make_unique<V4Layer>();
            layers[l]->init(l, aeon_loader, max_seq_len);
        }

        // 7. Setup Unified VRAM Pool and Expert Registry
        uint32_t active_vram_slots = unified_vram_slots > 0 ? unified_vram_slots : std::max(num_layers_ * 8u, 16u);
        unified_vram_pool_ = std::make_unique<UnifiedVRAMExpertPool>(active_vram_slots);
        expert_registry_ = std::make_unique<ExpertRegistry>(num_layers_, 256, active_vram_slots, 0);

        // Preload initial hot experts
        for (uint32_t slot = 0; slot < active_vram_slots; ++slot) {
            int32_t gid = expert_registry_->vram_slots[slot];
            if (gid >= 0) {
                uint32_t lay = expert_registry_->catalog[gid].layer_id;
                uint32_t exp = expert_registry_->catalog[gid].expert_id;
                const uint8_t* p = aeon_loader.get_expert_data(lay, exp);
                unified_vram_pool_->upload_from_host_expert(slot, p, compute_stream);
            }
        }
        CHECK_HIP(hipStreamSynchronize(compute_stream));

        std::cout << "[Pipeline] Engine ready! Configured for " << num_layers_ << " chained layers with "
                  << active_vram_slots << " unified VRAM expert slots in native .aeon format." << std::endl;
    }

    // Initialize dynamic memory budgeting and the unified VRAM expert pool.
    void init_dynamic_global(
        const std::string& aeon_model_dir,
        const AeonRuntimeConfig& runtime_cfg,
        uint32_t num_layers = 2
    ) {
        std::cout << "================================================================================" << std::endl;
        std::cout << "      Project Aeon — Dynamic VRAM Budget & Global Expert Pool Pipeline          " << std::endl;
        std::cout << "================================================================================" << std::endl;

        num_layers_ = num_layers;
        current_seq_len_ = 0;

        // 1. Initialize streams
        CHECK_HIP(hipStreamCreate(&compute_stream));
        CHECK_HIP(hipStreamCreate(&sdma_stream));

        // 2. Open Model Containers
        std::cout << "[Pipeline] Opening native .aeon model from " << aeon_model_dir << "..." << std::endl;
        aeon_loader.open_model(aeon_model_dir);
        std::cout << "  > Dense tensors indexed: " << aeon_loader.total_dense_tensors() << std::endl;

        // Load config
        auto model_cfg = DeepSeekV4Config::load_from_json(aeon_model_dir + "/config.json");

        // 3. Evaluate Memory Budget & Feasibility Gate
        size_t dense_bytes = 15745100992ULL; // model_dense.aeon size
        budget_report_ = MemoryBudgetEngine::evaluate(runtime_cfg, model_cfg, dense_bytes);
        std::cout << budget_report_.to_string() << std::endl;

        if (!budget_report_.is_feasible) {
            throw std::runtime_error("V4Pipeline: Feasibility gate REJECTED startup: " + budget_report_.rejection_reason);
        }

        // 4. Initialize RoPE Tables
        std::cout << "[Pipeline] Initializing RoPE tables (max_seq=" << runtime_cfg.context_size << ")..." << std::endl;
        rope_table.init(runtime_cfg.context_size, kernel::DSV4_ROPE_THETA, 1.0f);

        size_t rope_bytes = rope_table.max_seq_len * rope_table.half_rope * sizeof(float);
        CHECK_HIP(hipMalloc(&d_cos_cache_, rope_bytes));
        CHECK_HIP(hipMalloc(&d_sin_cache_, rope_bytes));
        CHECK_HIP(hipMemcpy(d_cos_cache_, rope_table.cos_cache.data(), rope_bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_sin_cache_, rope_table.sin_cache.data(), rope_bytes, hipMemcpyHostToDevice));

        // 5. Model-level Weights (LM Head & Norms)
        std::cout << "[Pipeline] Binding model-level embeddings and LM head..." << std::endl;
        host_embed_table = aeon_loader.get_data_ptr<half>("embed.weight");

        const auto& head_t = aeon_loader.get_tensor("head.weight");
        std::cout << "  > Uploading LM Head [129280, 4096] (" << (head_t.byte_size / (1024*1024)) << " MB) to VRAM..." << std::endl;
        CHECK_HIP(hipMalloc(&d_lm_head, head_t.byte_size));
        CHECK_HIP(hipMemcpy(d_lm_head, head_t.data, head_t.byte_size, hipMemcpyHostToDevice));

        // HC Head
        const auto& fn_t = aeon_loader.get_tensor("hc_head_fn");
        const auto& base_t = aeon_loader.get_tensor("hc_head_base");
        const auto& sc_t = aeon_loader.get_tensor("hc_head_scale");
        CHECK_HIP(hipMalloc(&d_hc_head_fn, fn_t.byte_size));
        CHECK_HIP(hipMalloc(&d_hc_head_base, base_t.byte_size));
        CHECK_HIP(hipMalloc(&d_hc_head_scale, sc_t.byte_size));
        CHECK_HIP(hipMemcpy(d_hc_head_fn, fn_t.data, fn_t.byte_size, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_hc_head_base, base_t.data, base_t.byte_size, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_hc_head_scale, sc_t.data, sc_t.byte_size, hipMemcpyHostToDevice));

        // Final norm
        const auto& norm_t = aeon_loader.get_tensor("norm.weight");
        CHECK_HIP(hipMalloc(&d_final_norm, norm_t.byte_size));
        CHECK_HIP(hipMemcpy(d_final_norm, norm_t.data, norm_t.byte_size, hipMemcpyHostToDevice));

        // 6. Allocate Intermediate GPU Scratch Buffers
        std::cout << "[Pipeline] Allocating intermediate GPU scratch buffers..." << std::endl;
        scratch.allocate();

        // 7. Initialize Consecutive Transformer Layers in Global Mode (Dense weights + KV Cache)
        std::cout << "[Pipeline] Initializing " << num_layers_ << " Transformer Layers (Dense weights + KV Cache in VRAM)..." << std::endl;
        layers.resize(num_layers_);
        for (uint32_t l = 0; l < num_layers_; ++l) {
            layers[l] = std::make_unique<V4Layer>();
            layers[l]->init_global(l, aeon_loader, runtime_cfg.context_size);
        }
        std::cout << "  > Dense weights and KV cache for all " << num_layers_ << " layers uploaded to VRAM." << std::endl;

        // 8. Allocate Unified VRAM Expert Pool (Hot Pool)
        std::cout << "[Pipeline] Allocating Unified VRAM Expert Pool (" << budget_report_.hot_vram_slots << " slots, "
                  << (budget_report_.hot_vram_bytes / (1024*1024*1024.0)) << " GB)..." << std::endl;
        unified_vram_pool_ = std::make_unique<UnifiedVRAMExpertPool>(budget_report_.hot_vram_slots);

        // 9. Initialize Expert Registry Catalog
        std::cout << "[Pipeline] Initializing Expert Registry (VRAM=" << budget_report_.hot_vram_slots
                  << ", Host=" << budget_report_.warm_host_slots << ")..." << std::endl;
        expert_registry_ = std::make_unique<ExpertRegistry>(
            num_layers_, model_cfg.n_routed_experts,
            budget_report_.hot_vram_slots, budget_report_.warm_host_slots
        );

        // 10. Preload Hot VRAM slots into Unified Pool
        std::cout << "[Pipeline] Pre-populating Hot VRAM slots into Unified Pool..." << std::endl;
        for (uint32_t slot = 0; slot < budget_report_.hot_vram_slots; ++slot) {
            int32_t gid = expert_registry_->vram_slots[slot];
            if (gid >= 0) {
                uint32_t lay = expert_registry_->catalog[gid].layer_id;
                uint32_t exp = expert_registry_->catalog[gid].expert_id;
                const uint8_t* p = aeon_loader.get_expert_data(lay, exp);
                unified_vram_pool_->upload_from_host_expert(slot, p, compute_stream);
            }
        }
        CHECK_HIP(hipStreamSynchronize(compute_stream));
        std::cout << "  > GPU complete: Dense backbone, KV cache, and Hot Expert Pool resident in VRAM!" << std::endl;

        // 11. Allocate and Preload Tier 2 Warm Host DDR Pool
        if (budget_report_.warm_host_slots > 0) {
            std::cout << "[Pipeline] Allocating Tier 2 Warm Host DDR Pool (" << budget_report_.warm_host_slots << " slots, "
                      << (budget_report_.warm_host_bytes / (1024*1024*1024.0)) << " GB)..." << std::endl;
            host_pool_ = std::make_unique<HostExpertPool>(budget_report_.warm_host_slots);

            std::cout << "[Pipeline] Pre-populating Warm Host DDR Pool ("
                      << budget_report_.warm_host_slots << " experts)..." << std::endl;
            for (uint32_t hslot = 0; hslot < budget_report_.warm_host_slots; ++hslot) {
                int32_t gid = expert_registry_->host_slots[hslot];
                if (gid >= 0) {
                    uint32_t lay = expert_registry_->catalog[gid].layer_id;
                    uint32_t exp = expert_registry_->catalog[gid].expert_id;
                    const uint8_t* p = aeon_loader.get_expert_data(lay, exp);
                    host_pool_->copy_from(hslot, p);
                }
            }
            std::cout << "  > Tier 2 Warm Host DDR Pool populated with " << budget_report_.warm_host_slots << " experts." << std::endl;
        }

        std::cout << "[Pipeline] Engine ready! Configured for " << num_layers_
                  << " layers with " << budget_report_.hot_vram_slots << " hot VRAM slots and "
                  << budget_report_.warm_host_slots << " warm host slots." << std::endl;
    }

    // Run Single Autoregressive Step for token_id at sequence position `pos`
    // Returns next token ID via greedy argmax
    uint32_t step(uint32_t token_id, uint32_t pos) {
        constexpr int H = kernel::DSV4_HIDDEN_SIZE; // 4096
        constexpr int HC = 4;
        constexpr int HC_DIM = HC * H;             // 16384
        constexpr int M_PAD = 16;
        constexpr int Q_LORA = kernel::DSV4_Q_LORA_RANK;
        constexpr int HEAD_DIM = kernel::DSV4_HEAD_DIM;
        constexpr int NUM_HEADS = kernel::DSV4_NUM_HEADS;
        constexpr int TOTAL_Q = NUM_HEADS * HEAD_DIM;
        constexpr int O_LORA = kernel::DSV4_O_LORA_RANK;
        constexpr int O_GROUPS = kernel::DSV4_O_GROUPS;
        constexpr int TOT_LORA = kernel::DSV4_TOTAL_O_LORA_DIM;
        constexpr int INTER_DIM = 2048;

        // 1. Embed Token & Replicate to 4 HC streams
        const half* token_emb = host_embed_table + token_id * H;

        // Replicate embedding across 4 streams into d_res_in_half on GPU
        // Shape [4, 4096]
        for (int s = 0; s < HC; ++s) {
            CHECK_HIP(hipMemcpyAsync(scratch.d_res_in_half + s * H, token_emb, H * sizeof(half), hipMemcpyHostToDevice, compute_stream));
        }
        // Convert to float for HC pre-mix
        kernel::v4_half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, compute_stream>>>(scratch.d_res_in_half, scratch.d_res_in, HC_DIM);

        // Upload single token ID for hash routing
        int32_t h_token = static_cast<int32_t>(token_id);
        CHECK_HIP(hipMemcpyAsync(scratch.d_token_id, &h_token, sizeof(int32_t), hipMemcpyHostToDevice, compute_stream));

        // 2. Execute Consecutive Transformer Layers
        for (uint32_t l = 0; l < num_layers_; ++l) {
            auto& layer = *layers[l];

            // -----------------------------------------------------------------
            // A. Hyper-Connections Attention Pre-Mix & Sinkhorn
            // -----------------------------------------------------------------
            // Host mixes projection or device mixes projection
            // mixes = (res @ fn.T) * rms
            // Mixes projection (24 outputs from 16384 float inputs)
            // We launch Wave32 GEMV for mixes [24, 16384]
            // We use cpu/device helper or kernel
            // Since HC_DIM = 16384, we do dot products on compute_stream:
            // First compute mean square
            std::vector<float> h_res(HC_DIM);
            CHECK_HIP(hipMemcpyAsync(h_res.data(), scratch.d_res_in, HC_DIM * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            float sqrsum = 0.0f;
            for (int i = 0; i < HC_DIM; ++i) sqrsum += h_res[i] * h_res[i];
            float rms = 1.0f / std::sqrt((sqrsum / (float)HC_DIM) + 1e-6f);

            // Fetch fn from layer weights
            const float* hc_fn_ptr = layer.h_hc_attn_fn;
            std::vector<float> h_mixes_a(24);
            for (int j = 0; j < 24; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < HC_DIM; ++k) dot += h_res[k] * hc_fn_ptr[j * HC_DIM + k];
                h_mixes_a[j] = dot * rms;
            }
            CHECK_HIP(hipMemcpyAsync(scratch.d_mixes_a, h_mixes_a.data(), 24 * sizeof(float), hipMemcpyHostToDevice, compute_stream));

            // Sinkhorn Normalize Kernel
            hipLaunchKernelGGL(
                kernel::hc_sinkhorn_normalize_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_mixes_a, layer.d_hc_attn_scale, layer.d_hc_attn_base,
                scratch.d_pre_a, scratch.d_post_a, scratch.d_comb_a,
                1e-6f, 1e-6f, 2.0f, 20
            );

            // Pre-combine streams: x_pre = sum_{s} pre_a[s] * res_in[s]
            std::vector<float> h_pre_a(HC);
            CHECK_HIP(hipMemcpyAsync(h_pre_a.data(), scratch.d_pre_a, HC * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            std::vector<half> h_x_pre(H);
            for (int h = 0; h < H; ++h) {
                float acc = 0.0f;
                for (int s = 0; s < HC; ++s) acc += h_pre_a[s] * h_res[s * H + h];
                h_x_pre[h] = __float2half(acc);
            }
            // Copy row 0 into d_x_pre (M_PAD rows)
            CHECK_HIP(hipMemcpyAsync(scratch.d_x_pre, h_x_pre.data(), H * sizeof(half), hipMemcpyHostToDevice, compute_stream));

            // -----------------------------------------------------------------
            // B. Attention RMSNorm
            // -----------------------------------------------------------------
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_x_pre, layer.d_attn_norm, scratch.d_x_norm, H, 1e-6f
            );

            // -----------------------------------------------------------------
            // C. MLA Projections
            // -----------------------------------------------------------------
            // Q_a = x_norm @ wq_a.T [1024]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(Q_LORA, 1), dim3(32), 0, compute_stream,
                scratch.d_x_norm, layer.d_wq_a, scratch.d_qa, H
            );

            // Q_a RMSNorm
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_qa, layer.d_q_norm, scratch.d_qa_norm, Q_LORA, 1e-6f
            );

            // Q = qa_norm @ wq_b.T [64 * 512 = 32768]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(TOTAL_Q, 1), dim3(32), 0, compute_stream,
                scratch.d_qa_norm, layer.d_wq_b, scratch.d_q, Q_LORA
            );

            // KV = x_norm @ wkv.T [512]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(HEAD_DIM, 1), dim3(32), 0, compute_stream,
                scratch.d_x_norm, layer.d_wkv, scratch.d_kv, H
            );

            // KV RMSNorm
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_kv, layer.d_kv_norm, scratch.d_kv_norm_act, HEAD_DIM, 1e-6f
            );

            // -----------------------------------------------------------------
            // D. RoPE & KV Cache Persistence
            // -----------------------------------------------------------------
            hipLaunchKernelGGL(
                kernel::v4_forward_rope_at_pos_wave32_kernel,
                dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                scratch.d_q, d_cos_cache_, d_sin_cache_, pos,
                NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2
            );

            hipLaunchKernelGGL(
                kernel::v4_forward_rope_at_pos_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_kv_norm_act, d_cos_cache_, d_sin_cache_, pos,
                1, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2
            );

            // Insert new KV vector into layer's persistent KV Cache at `pos`
            CHECK_HIP(hipMemcpyAsync(
                layer.d_kv_cache + pos * HEAD_DIM,
                scratch.d_kv_norm_act,
                HEAD_DIM * sizeof(half),
                hipMemcpyDeviceToDevice,
                compute_stream
            ));

            // -----------------------------------------------------------------
            // E. Autoregressive Sliding-Window Attention over Cached States
            // -----------------------------------------------------------------
            hipLaunchKernelGGL(
                kernel::v4_cached_sliding_window_attn_wave32_kernel,
                dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                scratch.d_q, layer.d_kv_cache, layer.d_attn_sink, scratch.d_attn_out,
                pos, kernel::DSV4_SLIDING_WINDOW, kernel::DSV4_ATTN_SCALE
            );

            // Inverse RoPE on attention output
            hipLaunchKernelGGL(
                kernel::v4_inverse_rope_at_pos_wave32_kernel,
                dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                scratch.d_attn_out, d_cos_cache_, d_sin_cache_, pos,
                NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2
            );

            // Grouped W_o_a: 8 groups x [1024, 4096] -> [8192]
            hipLaunchKernelGGL(
                kernel::v4_grouped_wo_a_wave32_kernel,
                dim3(O_LORA, O_GROUPS, 1), dim3(32), 0, compute_stream,
                scratch.d_attn_out, layer.d_wo_a, scratch.d_z_lora, 1
            );

            // W_o_b: [4096, 8192] -> [4096]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(H, 1), dim3(32), 0, compute_stream,
                scratch.d_z_lora, layer.d_wo_b, scratch.d_attn_proj, TOT_LORA
            );

            // -----------------------------------------------------------------
            // F. HC Attention Post Expansion: res_mid = comb_a * res_in + post_a * attn_proj
            // -----------------------------------------------------------------
            kernel::v4_float_to_half_kernel<<<(HC_DIM + 255) / 256, 256, 0, compute_stream>>>(scratch.d_res_in, scratch.d_res_in_half, HC_DIM);

            hipLaunchKernelGGL(
                kernel::hc_post_kernel,
                dim3((H + 255) / 256, 1), dim3(256), 0, compute_stream,
                scratch.d_attn_proj, scratch.d_res_in_half, scratch.d_post_a, scratch.d_comb_a, scratch.d_res_mid_half, H
            );

            kernel::v4_half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, compute_stream>>>(scratch.d_res_mid_half, scratch.d_res_mid, HC_DIM);

            // -----------------------------------------------------------------
            // G. HC FFN Pre-Mix & Sinkhorn
            // -----------------------------------------------------------------
            std::vector<float> h_res_mid(HC_DIM);
            CHECK_HIP(hipMemcpyAsync(h_res_mid.data(), scratch.d_res_mid, HC_DIM * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            float sqrsum_f = 0.0f;
            for (int i = 0; i < HC_DIM; ++i) sqrsum_f += h_res_mid[i] * h_res_mid[i];
            float rms_f = 1.0f / std::sqrt((sqrsum_f / (float)HC_DIM) + 1e-6f);

            // Fetch fn from layer weights
            const float* hc_ffn_fn_ptr = layer.h_hc_ffn_fn;
            std::vector<float> h_mixes_f(24);
            for (int j = 0; j < 24; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < HC_DIM; ++k) dot += h_res_mid[k] * hc_ffn_fn_ptr[j * HC_DIM + k];
                h_mixes_f[j] = dot * rms_f;
            }
            CHECK_HIP(hipMemcpyAsync(scratch.d_mixes_f, h_mixes_f.data(), 24 * sizeof(float), hipMemcpyHostToDevice, compute_stream));

            hipLaunchKernelGGL(
                kernel::hc_sinkhorn_normalize_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_mixes_f, layer.d_hc_ffn_scale, layer.d_hc_ffn_base,
                scratch.d_pre_f, scratch.d_post_f, scratch.d_comb_f,
                1e-6f, 1e-6f, 2.0f, 20
            );

            std::vector<float> h_pre_f(HC);
            CHECK_HIP(hipMemcpyAsync(h_pre_f.data(), scratch.d_pre_f, HC * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            std::vector<half> h_ffn_pre(H);
            for (int h = 0; h < H; ++h) {
                float acc = 0.0f;
                for (int s = 0; s < HC; ++s) acc += h_pre_f[s] * h_res_mid[s * H + h];
                h_ffn_pre[h] = __float2half(acc);
            }
            CHECK_HIP(hipMemcpyAsync(scratch.d_ffn_pre, h_ffn_pre.data(), H * sizeof(half), hipMemcpyHostToDevice, compute_stream));

            // FFN RMSNorm
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_ffn_pre, layer.d_ffn_norm, scratch.d_ffn_norm_act, H, 1e-6f
            );

            // Replicate row 0 to M_PAD rows of d_ffn_norm_act for WMMA compatibility
            for (int r = 1; r < M_PAD; ++r) {
                CHECK_HIP(hipMemcpyAsync(scratch.d_ffn_norm_act + r * H, scratch.d_ffn_norm_act, H * sizeof(half), hipMemcpyDeviceToDevice, compute_stream));
            }

            // -----------------------------------------------------------------
            // H. MoE Routing & Expert Execution
            // -----------------------------------------------------------------
            // 1. Router Logits: gate_weight @ ffn_norm_act [256]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(256, 1), dim3(32), 0, compute_stream,
                scratch.d_ffn_norm_act, layer.d_gate_weight, reinterpret_cast<half*>(scratch.d_router_logits), H
            );
            // Convert logits from half to float in-place
            // Or router_logits directly
            std::vector<half> h_rlogits_half(256);
            CHECK_HIP(hipMemcpyAsync(h_rlogits_half.data(), scratch.d_router_logits, 256 * sizeof(half), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));
            std::vector<float> h_rlogits(256);
            for (int i = 0; i < 256; ++i) h_rlogits[i] = __half2float(h_rlogits_half[i]);
            CHECK_HIP(hipMemcpyAsync(scratch.d_router_logits, h_rlogits.data(), 256 * sizeof(float), hipMemcpyHostToDevice, compute_stream));

            // Launch Router Kernel (Hash mode for layers 0..2)
            hipLaunchKernelGGL(
                kernel::moe_router_kernel,
                dim3(1), dim3(64), 0, compute_stream,
                scratch.d_router_logits, nullptr, layer.d_tid2eid, scratch.d_token_id,
                scratch.d_topk_weights, scratch.d_topk_indices,
                256, 6, 1.5f, true
            );

            std::vector<float> h_topk_weights(6);
            std::vector<int32_t> h_topk_indices(6);
            CHECK_HIP(hipMemcpyAsync(h_topk_weights.data(), scratch.d_topk_weights, 6 * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipMemcpyAsync(h_topk_indices.data(), scratch.d_topk_indices, 6 * sizeof(int32_t), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            // Clear MoE accumulation buffer
            CHECK_HIP(hipMemsetAsync(scratch.d_moe_accum, 0, M_PAD * H * sizeof(half), compute_stream));

            // 2. Shared Expert (FP16 unquantized)
            // w1 [2048, 4096] & w3 [2048, 4096]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(INTER_DIM, 1), dim3(32), 0, compute_stream,
                scratch.d_ffn_norm_act, layer.d_shared_w1, scratch.d_shared_gate, H
            );
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(INTER_DIM, 1), dim3(32), 0, compute_stream,
                scratch.d_ffn_norm_act, layer.d_shared_w3, scratch.d_shared_up, H
            );

            // Clamped SwiGLU
            int swiglu_threads = 256;
            int swiglu_blocks = (INTER_DIM + swiglu_threads - 1) / swiglu_threads;
            hipLaunchKernelGGL(
                kernel::v4_pipeline_swiglu_clamp_kernel,
                dim3(swiglu_blocks), dim3(swiglu_threads), 0, compute_stream,
                scratch.d_shared_gate, scratch.d_shared_up, scratch.d_shared_swiglu, INTER_DIM, 10.0f
            );

            // w2 [4096, 2048]
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(H, 1), dim3(32), 0, compute_stream,
                scratch.d_shared_swiglu, layer.d_shared_w2, scratch.d_moe_accum, INTER_DIM
            );

            // 3. 6 Routed Experts (INT4-W4A16 WMMA GEMM)
            for (int k = 0; k < 6; ++k) {
                uint32_t expert_id = h_topk_indices[k];
                float expert_weight = h_topk_weights[k];

                const uint32_t* d_w1_p = nullptr;
                const half*     d_w1_s = nullptr;
                const uint32_t* d_w2_p = nullptr;
                const half*     d_w2_s = nullptr;
                const uint32_t* d_w3_p = nullptr;
                const half*     d_w3_s = nullptr;

                // Unified Cross-Layer VRAM Expert Pool lookup
                int32_t slot = expert_registry_->touch_hot_expert(l, expert_id, pos);
                if (slot < 0) {
                    // Cache miss in Hot VRAM: check if resident in Tier 2 Warm Host DDR
                    uint32_t gid = expert_registry_->get_global_id(l, expert_id);
                    const auto& entry = expert_registry_->catalog[gid];

                    auto [allocated_slot, evicted_gid] = expert_registry_->allocate_vram_slot(gid);

                    // If an expert was evicted from VRAM to Host DDR, copy its weights to host pool if space exists
                    if (evicted_gid >= 0 && host_pool_) {
                        const auto& ev_entry = expert_registry_->catalog[evicted_gid];
                        if (ev_entry.tier == ExpertTier::WARM_HOST && ev_entry.slot_idx >= 0) {
                            uint32_t ev_hslot = static_cast<uint32_t>(ev_entry.slot_idx);
                            if (aeon_loader.total_dense_tensors() > 0) {
                                const uint8_t* ev_raw = aeon_loader.get_expert_data(
                                    expert_registry_->catalog[evicted_gid].layer_id,
                                    expert_registry_->catalog[evicted_gid].expert_id
                                );
                                host_pool_->copy_from(ev_hslot, ev_raw);
                            }
                        }
                    }

                    if (entry.tier == ExpertTier::WARM_HOST && host_pool_ && entry.slot_idx >= 0) {
                        // Hit in Tier 2 Warm Host DDR! DMA directly from physical host memory
                        const uint8_t* p = host_pool_->get_expert_slot_ptr(static_cast<uint32_t>(entry.slot_idx));
                        unified_vram_pool_->upload_from_host_expert(allocated_slot, p, compute_stream);
                    } else if (aeon_loader.total_dense_tensors() > 0) {
                        // Cold NVMe: stream from .aeon disk container via AeonModelLoader
                        const uint8_t* p = aeon_loader.get_expert_data(l, expert_id);
                        unified_vram_pool_->upload_from_host_expert(allocated_slot, p, compute_stream);
                    } else {
                        // Stream from Safetensors host source
                        std::string exp_pfx = "layers." + std::to_string(l) + ".ffn.experts." + std::to_string(expert_id) + ".";
                        const auto* w1_p = loader.get_data_ptr<uint32_t>(exp_pfx + "w1.weight_packed");
                        const auto* w1_s = loader.get_data_ptr<half>(exp_pfx + "w1.weight_scale");
                        const auto* w2_p = loader.get_data_ptr<uint32_t>(exp_pfx + "w2.weight_packed");
                        const auto* w2_s = loader.get_data_ptr<half>(exp_pfx + "w2.weight_scale");
                        const auto* w3_p = loader.get_data_ptr<uint32_t>(exp_pfx + "w3.weight_packed");
                        const auto* w3_s = loader.get_data_ptr<half>(exp_pfx + "w3.weight_scale");

                        unified_vram_pool_->upload_from_pointers(
                            allocated_slot,
                            w1_p, w1_s,
                            w2_p, w2_s,
                            w3_p, w3_s,
                            compute_stream
                        );
                    }

                    slot = static_cast<int32_t>(allocated_slot);
                }

                d_w1_p = unified_vram_pool_->get_w1_packed(slot);
                d_w1_s = unified_vram_pool_->get_w1_scale(slot);
                d_w2_p = unified_vram_pool_->get_w2_packed(slot);
                d_w2_s = unified_vram_pool_->get_w2_scale(slot);
                d_w3_p = unified_vram_pool_->get_w3_packed(slot);
                d_w3_s = unified_vram_pool_->get_w3_scale(slot);

                // Update layer statistics from central expert registry
                layer.cache_hits = expert_registry_->hits_hot;
                layer.cache_misses = expert_registry_->hits_warm + expert_registry_->misses_cold;

                // w1 & w3 via fused W4A16 WMMA
                kernel::dispatch_w4a16_gemm(
                    scratch.d_ffn_norm_act, d_w1_p, d_w1_s,
                    scratch.d_expert_gate, M_PAD, INTER_DIM, H, compute_stream
                );
                kernel::dispatch_w4a16_gemm(
                    scratch.d_ffn_norm_act, d_w3_p, d_w3_s,
                    scratch.d_expert_up, M_PAD, INTER_DIM, H, compute_stream
                );

                // SwiGLU clamp
                int m_swiglu_blocks = (M_PAD * INTER_DIM + swiglu_threads - 1) / swiglu_threads;
                hipLaunchKernelGGL(
                    kernel::v4_pipeline_swiglu_clamp_kernel,
                    dim3(m_swiglu_blocks), dim3(swiglu_threads), 0, compute_stream,
                    scratch.d_expert_gate, scratch.d_expert_up, scratch.d_expert_swiglu, M_PAD * INTER_DIM, 10.0f
                );

                // w2
                kernel::dispatch_w4a16_gemm(
                    scratch.d_expert_swiglu, d_w2_p, d_w2_s,
                    scratch.d_expert_down, M_PAD, H, INTER_DIM, compute_stream
                );

                // Accumulate row 0
                hipLaunchKernelGGL(
                    kernel::v4_pipeline_accumulate_expert_kernel,
                    dim3((H + 255) / 256), dim3(256), 0, compute_stream,
                    scratch.d_moe_accum, scratch.d_expert_down, expert_weight, H
                );
            }

            // -----------------------------------------------------------------
            // I. HC FFN Post Expansion: res_out = comb_f * res_mid + post_f * moe_accum
            // -----------------------------------------------------------------
            hipLaunchKernelGGL(
                kernel::hc_post_kernel,
                dim3((H + 255) / 256, 1), dim3(256), 0, compute_stream,
                scratch.d_moe_accum, scratch.d_res_mid_half, scratch.d_post_f, scratch.d_comb_f, scratch.d_res_out_half, H
            );

            // Copy res_out into res_in for next layer
            CHECK_HIP(hipMemcpyAsync(
                scratch.d_res_in_half, scratch.d_res_out_half, HC_DIM * sizeof(half),
                hipMemcpyDeviceToDevice, compute_stream
            ));
            kernel::v4_half_to_float_kernel<<<(HC_DIM + 255) / 256, 256, 0, compute_stream>>>(scratch.d_res_in_half, scratch.d_res_in, HC_DIM);
        }

        // 3. HC Head Reduction on Final Residual
        hipLaunchKernelGGL(
            kernel::hc_head_wave32_kernel,
            dim3(1), dim3(32), 0, compute_stream,
            scratch.d_res_in, d_hc_head_fn, d_hc_head_base, d_hc_head_scale,
            scratch.d_hc_head_out, H, HC, 1e-6f, 1e-6f
        );

        // 4. Final RMSNorm
        hipLaunchKernelGGL(
            kernel::v4_rmsnorm_wave32_kernel,
            dim3(1), dim3(32), 0, compute_stream,
            scratch.d_hc_head_out, d_final_norm, scratch.d_head_norm, H, 1e-6f
        );

        // 5. LM Head Projection: logits = head_norm @ lm_head.T [129280]
        hipLaunchKernelGGL(
            kernel::v4_gemv_fp16_kernel,
            dim3(129280, 1), dim3(32), 0, compute_stream,
            scratch.d_head_norm, d_lm_head, scratch.d_logits, H
        );

        // 6. Argmax Sampling
        std::vector<half> h_logits(129280);
        CHECK_HIP(hipMemcpyAsync(h_logits.data(), scratch.d_logits, 129280 * sizeof(half), hipMemcpyDeviceToHost, compute_stream));
        CHECK_HIP(hipStreamSynchronize(compute_stream));

        uint32_t best_tok = 0;
        float best_val = -1e30f;
        for (uint32_t v = 0; v < 129280; ++v) {
            float val = __half2float(h_logits[v]);
            if (val > best_val) {
                best_val = val;
                best_tok = v;
            }
        }

        current_seq_len_ = pos + 1;
        return best_tok;
    }

    // Prefill Prompt and Generate Next Tokens
    std::vector<uint32_t> generate(
        const std::vector<uint32_t>& prompt,
        uint32_t max_new_tokens = 16,
        double* out_ttft_ms = nullptr,
        double* out_tok_per_sec = nullptr
    ) {
        if (prompt.empty()) return {};

        std::vector<uint32_t> generated;
        current_seq_len_ = 0;

        // Reset layer KV caches
        for (auto& l : layers) {
            CHECK_HIP(hipMemset(l->d_kv_cache, 0, l->max_seq_len_ * kernel::DSV4_HEAD_DIM * sizeof(half)));
        }

        // Prefill Phase
        auto t_prefill_start = std::chrono::high_resolution_clock::now();
        uint32_t next_tok = 0;
        for (size_t i = 0; i < prompt.size(); ++i) {
            next_tok = step(prompt[i], i);
        }
        auto t_prefill_end = std::chrono::high_resolution_clock::now();

        double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
        if (out_ttft_ms) *out_ttft_ms = ttft_ms;

        generated.push_back(next_tok);

        // Autoregressive Decoding Phase
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        for (uint32_t step_idx = 1; step_idx < max_new_tokens; ++step_idx) {
            uint32_t pos = prompt.size() + step_idx - 1;
            next_tok = step(next_tok, pos);
            generated.push_back(next_tok);
        }
        auto t_decode_end = std::chrono::high_resolution_clock::now();

        double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
        double tok_sec = (max_new_tokens > 1) ? ((max_new_tokens - 1) / (decode_ms / 1000.0)) : 0.0;
        if (out_tok_per_sec) *out_tok_per_sec = tok_sec;

        return generated;
    }

    void free_all() {
        if (compute_stream) { (void)hipStreamDestroy(compute_stream); compute_stream = 0; }
        if (sdma_stream) { (void)hipStreamDestroy(sdma_stream); sdma_stream = 0; }
        if (d_cos_cache_) { (void)hipFree(d_cos_cache_); d_cos_cache_ = nullptr; }
        if (d_sin_cache_) { (void)hipFree(d_sin_cache_); d_sin_cache_ = nullptr; }
        if (d_lm_head) { (void)hipFree(d_lm_head); d_lm_head = nullptr; }
        if (d_hc_head_fn) { (void)hipFree(d_hc_head_fn); d_hc_head_fn = nullptr; }
        if (d_hc_head_base) { (void)hipFree(d_hc_head_base); d_hc_head_base = nullptr; }
        if (d_hc_head_scale) { (void)hipFree(d_hc_head_scale); d_hc_head_scale = nullptr; }
        if (d_final_norm) { (void)hipFree(d_final_norm); d_final_norm = nullptr; }

        scratch.free();
        for (auto& l : layers) {
            if (l) l->free();
        }
        layers.clear();
        unified_vram_pool_.reset();
        host_pool_.reset();
        expert_registry_.reset();
        loader.close_all();
        aeon_loader.close_all();
    }

private:
    float* d_cos_cache_{nullptr};
    float* d_sin_cache_{nullptr};
};

} // namespace aeon::core
