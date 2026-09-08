#pragma once

#include "core/config.hpp"
#include "core/device.hpp"
#include "core/aeon_loader.hpp"
#include "io/direct_io_reader.hpp"
#include "core/expert_registry.hpp"
#include "core/host_expert_pool.hpp"
#include "core/memory_budget.hpp"
#include "core/prefetch_staging.hpp"
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
#include <utility>
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
    std::unique_ptr<PrefetchStagingArena> prefetch_staging_;
    std::unique_ptr<aeon::io::DirectIOReader> direct_io_reader_;
    std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> direct_io_completions_;
    uint64_t next_direct_io_id_{1};
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
        prefetch_staging_ = std::make_unique<PrefetchStagingArena>();

        std::cout << "[Pipeline] Engine ready! Configured for " << num_layers_ << " chained layers with "
                  << active_vram_slots << " unified VRAM expert slots." << std::endl;
    }

    // Initialize pipeline directly from native .aeon format folder
    void init_aeon(
        const std::string& aeon_model_dir,
        uint32_t num_layers = 2,
        uint32_t unified_vram_slots = 0,
        uint32_t max_seq_len = 4096,
        bool enable_direct_io = true
    ) {
        num_layers_ = num_layers;
        current_seq_len_ = 0;

        // 1. Initialize streams
        CHECK_HIP(hipStreamCreate(&compute_stream));
        CHECK_HIP(hipStreamCreate(&sdma_stream));

        // 2. Open Aeon Model via Zero-Copy Mmap
        std::cout << "[Pipeline] Opening native Aeon model from " << aeon_model_dir << "..." << std::endl;
        aeon_loader.open_model(aeon_model_dir);
        if (enable_direct_io) {
            direct_io_reader_ = std::make_unique<aeon::io::DirectIOReader>(64);
        }
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
        prefetch_staging_ = std::make_unique<PrefetchStagingArena>();

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
        direct_io_reader_ = std::make_unique<aeon::io::DirectIOReader>(64);
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
        prefetch_staging_ = std::make_unique<PrefetchStagingArena>();

        // 9. Initialize Expert Registry Catalog
        const uint32_t active_total_experts = num_layers_ * model_cfg.n_routed_experts;
        const uint32_t active_remaining_after_vram =
            active_total_experts > budget_report_.hot_vram_slots
                ? active_total_experts - budget_report_.hot_vram_slots
                : 0;
        uint32_t active_warm_host_slots = runtime_cfg.preload_warm_host
            ? std::min(budget_report_.warm_host_slots, active_remaining_after_vram)
            : 0;
        std::cout << "[Pipeline] Initializing Expert Registry (VRAM=" << budget_report_.hot_vram_slots
              << ", Host=" << active_warm_host_slots << ")..." << std::endl;
        expert_registry_ = std::make_unique<ExpertRegistry>(
            num_layers_, model_cfg.n_routed_experts,
            budget_report_.hot_vram_slots, active_warm_host_slots
        );

        // 10. Preload Hot VRAM slots into Unified Pool
        std::cout << "[Pipeline] Pre-populating Hot VRAM slots into Unified Pool..." << std::endl;
        const size_t direct_batch_slots = std::max<size_t>(
            1, direct_io_reader_->submission_capacity() /
               ((AEON_EXPERT_BYTES + aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES - 1) /
                aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES));
        for (uint32_t batch_start = 0; batch_start < budget_report_.hot_vram_slots; batch_start += direct_batch_slots) {
            const uint32_t batch_end = std::min<uint32_t>(
                budget_report_.hot_vram_slots,
                batch_start + static_cast<uint32_t>(direct_batch_slots));
            std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
            std::vector<aeon::io::AlignedBuffer> buffers;
            expert_ids.reserve(batch_end - batch_start);
            buffers.reserve(batch_end - batch_start);

            for (uint32_t slot = batch_start; slot < batch_end; ++slot) {
                int32_t gid = expert_registry_->vram_slots[slot];
                if (gid < 0) continue;
                const auto& entry = expert_registry_->catalog[gid];
                expert_ids.emplace_back(entry.layer_id, entry.expert_id);
                buffers.emplace_back(AEON_EXPERT_BYTES);
            }

            std::vector<uint8_t*> destinations;
            destinations.reserve(buffers.size());
            for (auto& buffer : buffers) {
                destinations.push_back(static_cast<uint8_t*>(buffer.data()));
            }
            read_experts_direct_blocking(expert_ids, destinations);

            size_t buffer_idx = 0;
            for (uint32_t slot = batch_start; slot < batch_end; ++slot) {
                if (expert_registry_->vram_slots[slot] < 0) continue;
                unified_vram_pool_->upload_from_host_expert(
                    slot, destinations[buffer_idx++], compute_stream);
            }
            CHECK_HIP(hipStreamSynchronize(compute_stream));
        }
        std::cout << "  > GPU complete: Dense backbone, KV cache, and Hot Expert Pool resident in VRAM!" << std::endl;

        // 11. Allocate and Preload Tier 2 Warm Host DDR Pool
        if (runtime_cfg.preload_warm_host && active_warm_host_slots > 0) {
            const size_t active_warm_host_bytes = static_cast<size_t>(active_warm_host_slots) * AEON_EXPERT_BYTES;
            std::cout << "[Pipeline] Allocating Tier 2 Warm Host DDR Pool (" << active_warm_host_slots << " slots, "
                      << (active_warm_host_bytes / (1024*1024*1024.0)) << " GB)..." << std::endl;
            host_pool_ = std::make_unique<HostExpertPool>(active_warm_host_slots);

            std::cout << "[Pipeline] Pre-populating Warm Host DDR Pool ("
                      << active_warm_host_slots << " experts)..." << std::endl;
            for (uint32_t batch_start = 0; batch_start < active_warm_host_slots; batch_start += direct_batch_slots) {
                const uint32_t batch_end = std::min<uint32_t>(
                    active_warm_host_slots,
                    batch_start + static_cast<uint32_t>(direct_batch_slots));
                std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
                std::vector<uint8_t*> destinations;
                expert_ids.reserve(batch_end - batch_start);
                destinations.reserve(batch_end - batch_start);

                for (uint32_t hslot = batch_start; hslot < batch_end; ++hslot) {
                    int32_t gid = expert_registry_->host_slots[hslot];
                    if (gid < 0) continue;
                    const auto& entry = expert_registry_->catalog[gid];
                    expert_ids.emplace_back(entry.layer_id, entry.expert_id);
                    destinations.push_back(host_pool_->get_expert_slot_ptr(hslot));
                }
                read_experts_direct_blocking(expert_ids, destinations);
            }
            std::cout << "  > Tier 2 Warm Host DDR Pool populated with " << active_warm_host_slots << " experts." << std::endl;
        } else if (!runtime_cfg.preload_warm_host) {
            std::cout << "[Pipeline] Warm Host DDR preload disabled; non-hot experts will stream from the cold .aeon tier on demand." << std::endl;
        }

        std::cout << "[Pipeline] Engine ready! Configured for " << num_layers_
                  << " layers with " << budget_report_.hot_vram_slots << " hot VRAM slots and "
                  << active_warm_host_slots << " active warm host slots." << std::endl;
    }

    // Run Single Autoregressive Step for token_id at sequence position `pos`
    // Returns next token ID via greedy argmax
    uint32_t step(uint32_t token_id, uint32_t pos) {
        constexpr int H = kernel::DSV4_HIDDEN_SIZE; // 4096
        constexpr int HC = 4;
        constexpr int HC_DIM = HC * H;             // 16384
        constexpr int HC_MULT3 = HC * (2 + HC);    // 24
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

        // State tracking for inter-layer prefetching across pipeline stages
        struct LayerPrefetchState {
            bool is_active{false};
            uint32_t layer_idx{0};
            std::array<int32_t, 6> vram_slots{-1, -1, -1, -1, -1, -1};
            std::array<bool, 6> is_prefetched{false, false, false, false, false, false};
            std::array<uint32_t, 6> staging_indices{0, 0, 0, 0, 0, 0};
            std::array<bool, 6> io_pending{false, false, false, false, false, false};
            std::array<uint64_t, 6> io_user_data{0, 0, 0, 0, 0, 0};
            std::array<uint32_t, 6> io_request_counts{0, 0, 0, 0, 0, 0};
        };
        LayerPrefetchState lookahead_prefetch;
        std::vector<uint32_t> releasable_staging_slots;

        auto dispatch_layer_prefetch = [&](uint32_t target_l, const std::vector<int32_t>& topk_experts) -> LayerPrefetchState {
            LayerPrefetchState state;
            state.is_active = true;
            state.layer_idx = target_l;
            uint32_t buf_offset = (target_l % 2) * 6;
            bool submitted_direct_io = false;

            for (int k = 0; k < 6; ++k) {
                uint32_t expert_id = static_cast<uint32_t>(topk_experts[k]);
                int32_t slot = expert_registry_->touch_hot_expert(target_l, expert_id, pos);
                if (slot >= 0) {
                    state.vram_slots[k] = slot;
                    state.is_prefetched[k] = false;
                } else {
                    uint32_t gid = expert_registry_->get_global_id(target_l, expert_id);
                    const auto& entry = expert_registry_->catalog[gid];
                    const ExpertTier source_tier = entry.tier;
                    const int32_t source_slot = entry.slot_idx;
                    const bool source_is_warm = source_tier == ExpertTier::WARM_HOST &&
                                                host_pool_ && source_slot >= 0;

                    // Evicted Hot experts return directly to Cold NVMe: expert weights
                    // are immutable and permanently available on disk, so eviction
                    // never copies payloads back to host memory on the request path.
                    auto [allocated_slot, evicted_gid] = expert_registry_->allocate_vram_slot(gid);
                    (void)evicted_gid;

                    const bool can_direct_read = source_tier == ExpertTier::COLD_NVME &&
                                                 direct_io_reader_ &&
                                                 aeon_loader.total_dense_tensors() > 0 &&
                                                 prefetch_staging_;
                    if (can_direct_read) {
                        uint32_t staging_idx = buf_offset + k;
                        const auto location = aeon_loader.get_expert_location(target_l, expert_id);
                        const uint64_t request_id = next_direct_io_id_;

                        prefetch_staging_->begin_io(staging_idx);
                        const size_t request_count = direct_io_reader_->submit_read_chunks(
                            aeon_loader.expert_direct_fd(),
                            prefetch_staging_->get_slot_ptr(staging_idx),
                            location.byte_length,
                            location.file_offset,
                            request_id
                        );
                        next_direct_io_id_ += request_count;

                        state.vram_slots[k] = static_cast<int32_t>(allocated_slot);
                        state.staging_indices[k] = staging_idx;
                        state.io_pending[k] = true;
                        state.io_user_data[k] = request_id;
                        state.io_request_counts[k] = static_cast<uint32_t>(request_count);
                        submitted_direct_io = true;
                    } else if (source_is_warm &&
                               host_pool_->is_slot_pinned(static_cast<uint32_t>(source_slot))) {
                        // Fast path: page-locked warm slots upload straight to VRAM via
                        // SDMA, skipping the redundant host-to-host staging memcpy.
                        const uint32_t staging_idx = buf_offset + k;
                        prefetch_staging_->begin_direct_transfer(staging_idx);
                        unified_vram_pool_->upload_from_host_expert(
                            allocated_slot,
                            host_pool_->get_expert_slot_ptr(static_cast<uint32_t>(source_slot)),
                            sdma_stream
                        );
                        CHECK_HIP(hipEventRecord(prefetch_staging_->events[staging_idx], sdma_stream));

                        state.vram_slots[k] = static_cast<int32_t>(allocated_slot);
                        state.is_prefetched[k] = true;
                        state.staging_indices[k] = staging_idx;
                    } else {
                        if (source_is_warm) {
                            // Unpinned warm-segment fallback: stage through the arena.
                            const uint32_t source_staging_idx = buf_offset + k;
                            prefetch_staging_->stage_payload(
                                source_staging_idx,
                                host_pool_->get_expert_slot_ptr(static_cast<uint32_t>(source_slot))
                            );
                            const uint8_t* pinned_payload = prefetch_staging_->get_slot_ptr(source_staging_idx);
                            prefetch_staging_->begin_gpu_transfer(source_staging_idx);
                            unified_vram_pool_->upload_from_host_expert(allocated_slot, pinned_payload, sdma_stream);
                            CHECK_HIP(hipEventRecord(prefetch_staging_->events[source_staging_idx], sdma_stream));

                            state.vram_slots[k] = static_cast<int32_t>(allocated_slot);
                            state.is_prefetched[k] = true;
                            state.staging_indices[k] = source_staging_idx;
                        } else {
                            const uint8_t* src_ptr = nullptr;
                            if (aeon_loader.total_dense_tensors() > 0) {
                                src_ptr = aeon_loader.get_expert_data(target_l, expert_id);
                            }

                            if (src_ptr && prefetch_staging_) {
                                const uint32_t staging_idx = buf_offset + k;
                                prefetch_staging_->stage_payload(staging_idx, src_ptr);
                                const uint8_t* pinned_payload = prefetch_staging_->get_slot_ptr(staging_idx);
                                prefetch_staging_->begin_gpu_transfer(staging_idx);
                                unified_vram_pool_->upload_from_host_expert(allocated_slot, pinned_payload, sdma_stream);
                                CHECK_HIP(hipEventRecord(prefetch_staging_->events[staging_idx], sdma_stream));

                                state.vram_slots[k] = static_cast<int32_t>(allocated_slot);
                                state.is_prefetched[k] = true;
                                state.staging_indices[k] = staging_idx;
                            } else if (src_ptr) {
                                unified_vram_pool_->upload_from_host_expert(allocated_slot, src_ptr, sdma_stream);
                                state.vram_slots[k] = static_cast<int32_t>(allocated_slot);
                                state.is_prefetched[k] = false;
                            } else {
                                std::string exp_pfx = "layers." + std::to_string(target_l) + ".ffn.experts." + std::to_string(expert_id) + ".";
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
                                state.vram_slots[k] = static_cast<int32_t>(allocated_slot);
                                state.is_prefetched[k] = false;
                            }
                        }
                    }
                }
            }
            if (submitted_direct_io) {
                direct_io_reader_->submit_pending_reads();
            }
            return state;
        };

        auto materialize_layer_prefetch = [&](LayerPrefetchState& state) {
            for (int k = 0; k < 6; ++k) {
                if (!state.io_pending[k]) {
                    continue;
                }

                const size_t request_count = state.io_request_counts[k];
                for (size_t chunk = 0; chunk < request_count; ++chunk) {
                    const uint64_t request_id = state.io_user_data[k] + chunk;
                    auto completion_it = direct_io_completions_.find(request_id);
                    while (completion_it == direct_io_completions_.end()) {
                        if (!direct_io_reader_) {
                            throw std::runtime_error("V4Pipeline: direct I/O request has no reader");
                        }
                        const auto completion = direct_io_reader_->wait_for_completion();
                        direct_io_completions_[completion.user_data] = completion;
                        completion_it = direct_io_completions_.find(request_id);
                    }

                    const auto completion = completion_it->second;
                    direct_io_completions_.erase(completion_it);
                    if (completion.result < 0) {
                        throw std::runtime_error("V4Pipeline: direct expert read failed: " +
                                                 std::string(strerror(-completion.result)));
                    }
                    const size_t chunk_offset = chunk * aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
                    const size_t expected_bytes = std::min(
                        aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES,
                        static_cast<size_t>(AEON_EXPERT_BYTES) - chunk_offset
                    );
                    if (completion.result != static_cast<int32_t>(expected_bytes)) {
                        throw std::runtime_error("V4Pipeline: direct expert read returned a short payload");
                    }
                }

                const uint32_t staging_idx = state.staging_indices[k];
                prefetch_staging_->complete_io(staging_idx);
                prefetch_staging_->begin_gpu_transfer(staging_idx);
                unified_vram_pool_->upload_from_host_expert(
                    static_cast<uint32_t>(state.vram_slots[k]),
                    prefetch_staging_->get_slot_ptr(staging_idx),
                    sdma_stream
                );
                CHECK_HIP(hipEventRecord(prefetch_staging_->events[staging_idx], sdma_stream));

                state.is_prefetched[k] = true;
                state.io_pending[k] = false;
            }
        };

        // Bootstrap: If Layer 0 is a Hash Layer, look ahead and dispatch its SDMA prefetch before Layer 0 Attention!
        if (num_layers_ > 0 && layers[0]->is_hash_layer && layers[0]->host_tid2eid) {
            std::vector<int32_t> l0_topk(6);
            for (int k = 0; k < 6; ++k) {
                l0_topk[k] = static_cast<int32_t>(layers[0]->host_tid2eid[h_token * 6 + k]);
            }
            lookahead_prefetch = dispatch_layer_prefetch(0, l0_topk);
        }

        // 2. Execute Consecutive Transformer Layers
        for (uint32_t l = 0; l < num_layers_; ++l) {
            auto& layer = *layers[l];

            // -----------------------------------------------------------------
            // A. Hyper-Connections Attention Pre-Mix & Sinkhorn
            // -----------------------------------------------------------------
            hipLaunchKernelGGL(
                kernel::hc_project_kernel,
                dim3(HC_MULT3), dim3(256), 0, compute_stream,
                scratch.d_res_in, layer.d_hc_attn_fn, scratch.d_mixes_a,
                H, HC, 1e-6f
            );

            // Sinkhorn Normalize Kernel
            hipLaunchKernelGGL(
                kernel::hc_sinkhorn_normalize_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_mixes_a, layer.d_hc_attn_scale, layer.d_hc_attn_base,
                scratch.d_pre_a, scratch.d_post_a, scratch.d_comb_a,
                1e-6f, 1e-6f, 2.0f, 20
            );

            hipLaunchKernelGGL(
                kernel::hc_pre_combine_kernel,
                dim3((H / 4 + 255) / 256), dim3(256), 0, compute_stream,
                scratch.d_res_in, scratch.d_pre_a, scratch.d_x_pre, H, HC
            );

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
            hipLaunchKernelGGL(
                kernel::hc_project_kernel,
                dim3(HC_MULT3), dim3(256), 0, compute_stream,
                scratch.d_res_mid, layer.d_hc_ffn_fn, scratch.d_mixes_f,
                H, HC, 1e-6f
            );

            hipLaunchKernelGGL(
                kernel::hc_sinkhorn_normalize_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_mixes_f, layer.d_hc_ffn_scale, layer.d_hc_ffn_base,
                scratch.d_pre_f, scratch.d_post_f, scratch.d_comb_f,
                1e-6f, 1e-6f, 2.0f, 20
            );

            hipLaunchKernelGGL(
                kernel::hc_pre_combine_kernel,
                dim3((H / 4 + 255) / 256), dim3(256), 0, compute_stream,
                scratch.d_res_mid, scratch.d_pre_f, scratch.d_ffn_pre, H, HC
            );

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

            for (uint32_t staging_idx : releasable_staging_slots) {
                prefetch_staging_->release_after_gpu_transfer(staging_idx);
            }
            releasable_staging_slots.clear();

            // -----------------------------------------------------------------
            // Shared Expert FIRST: enqueue all GPU work before any CPU-side staging
            // or I/O dispatch so the device stays busy while the CPU submits
            // io_uring reads and performs staging copies.
            // -----------------------------------------------------------------
            // Clear MoE accumulation buffer
            CHECK_HIP(hipMemsetAsync(scratch.d_moe_accum, 0, M_PAD * H * sizeof(half), compute_stream));

            // 2. Shared Expert (FP16 unquantized) executes on compute_stream while
            // the CPU dispatches expert transfers and SDMA moves them across PCIe.
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

            // -----------------------------------------------------------------
            // Dual-Stream Asynchronous SDMA Prefetching Pipeline
            // If this layer was prefetched in advance by lookahead routing, reuse its slots and events!
            // Otherwise, perform intra-layer prefetch concurrently with Shared Expert execution.
            // -----------------------------------------------------------------
            struct PendingPrefetch {
                int32_t vram_slot{-1};
                bool is_prefetched{false};
                uint32_t staging_idx{0};
            };
            std::array<PendingPrefetch, 6> pending_transfers;
            const bool layer_already_prefetched = (lookahead_prefetch.is_active && lookahead_prefetch.layer_idx == l);
            LayerPrefetchState active_prefetch;
            if (layer_already_prefetched) {
                active_prefetch = lookahead_prefetch;
                lookahead_prefetch.is_active = false;
            } else {
                active_prefetch = dispatch_layer_prefetch(l, h_topk_indices);
            }

            // Lookahead prefetch trigger for Layer L+1:
            // If the next layer is a Hash Layer, we already know its top-6 experts from token_id!
            // Dispatch Layer L+1 transfers immediately on sdma_stream while Layer L executes Shared & Routed experts.
            if (l + 1 < num_layers_ && layers[l + 1]->is_hash_layer && layers[l + 1]->host_tid2eid) {
                std::vector<int32_t> next_topk(6);
                for (int k = 0; k < 6; ++k) {
                    next_topk[k] = static_cast<int32_t>(layers[l + 1]->host_tid2eid[h_token * 6 + k]);
                }
                lookahead_prefetch = dispatch_layer_prefetch(l + 1, next_topk);
            }

            // Let cold NVMe reads overlap the shared expert pass; materialization
            // still completes before the first routed expert consumes each slot.
            materialize_layer_prefetch(active_prefetch);

            for (int k = 0; k < 6; ++k) {
                pending_transfers[k].vram_slot = active_prefetch.vram_slots[k];
                pending_transfers[k].is_prefetched = active_prefetch.is_prefetched[k];
                pending_transfers[k].staging_idx = active_prefetch.staging_indices[k];
            }

            // 3. 6 Routed Experts (INT4-W4A16 WMMA GEMM)
            for (int k = 0; k < 6; ++k) {
                float expert_weight = h_topk_weights[k];
                int32_t slot = pending_transfers[k].vram_slot;

                // If this expert was transferred asynchronously on sdma_stream,
                // wait for transfer to finish before executing expert GEMM
                if (pending_transfers[k].is_prefetched && prefetch_staging_) {
                    CHECK_HIP(hipStreamWaitEvent(compute_stream, prefetch_staging_->events[pending_transfers[k].staging_idx], 0));
                }

                const uint32_t* d_w1_p = unified_vram_pool_->get_w1_packed(slot);
                const half*     d_w1_s = unified_vram_pool_->get_w1_scale(slot);
                const uint32_t* d_w2_p = unified_vram_pool_->get_w2_packed(slot);
                const half*     d_w2_s = unified_vram_pool_->get_w2_scale(slot);
                const uint32_t* d_w3_p = unified_vram_pool_->get_w3_packed(slot);
                const half*     d_w3_s = unified_vram_pool_->get_w3_scale(slot);

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

            for (int k = 0; k < 6; ++k) {
                if (pending_transfers[k].is_prefetched) {
                    releasable_staging_slots.push_back(pending_transfers[k].staging_idx);
                }
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
        for (uint32_t staging_idx : releasable_staging_slots) {
            prefetch_staging_->release_after_gpu_transfer(staging_idx);
        }

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
        if (compute_stream) { (void)hipStreamSynchronize(compute_stream); }
        if (sdma_stream) { (void)hipStreamSynchronize(sdma_stream); }
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
        prefetch_staging_.reset();
        direct_io_completions_.clear();
        direct_io_reader_.reset();
        loader.close_all();
        aeon_loader.close_all();
    }

private:
    void read_experts_direct_blocking(
        const std::vector<std::pair<uint32_t, uint32_t>>& expert_ids,
        const std::vector<uint8_t*>& destinations
    ) {
        if (expert_ids.size() != destinations.size()) {
            throw std::invalid_argument("V4Pipeline: direct expert read batch has mismatched inputs");
        }
        if (expert_ids.empty()) return;
        if (!direct_io_reader_) {
            throw std::runtime_error("V4Pipeline: direct expert reader is not initialized");
        }

        const size_t requests_per_expert =
            (AEON_EXPERT_BYTES + aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES - 1) /
            aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
        const size_t max_batch_experts = std::max<size_t>(
            1, direct_io_reader_->submission_capacity() / requests_per_expert);

        struct ReadJob {
            uint64_t first_user_data{0};
            size_t request_count{0};
        };

        for (size_t batch_start = 0; batch_start < expert_ids.size(); batch_start += max_batch_experts) {
            const size_t batch_end = std::min(expert_ids.size(), batch_start + max_batch_experts);
            std::vector<ReadJob> jobs;
            jobs.reserve(batch_end - batch_start);
            size_t total_requests = 0;

            for (size_t i = batch_start; i < batch_end; ++i) {
                const auto location = aeon_loader.get_expert_location(
                    expert_ids[i].first, expert_ids[i].second);
                const uint64_t first_user_data = next_direct_io_id_;
                const size_t request_count = direct_io_reader_->submit_read_chunks(
                    aeon_loader.expert_direct_fd(),
                    destinations[i],
                    location.byte_length,
                    location.file_offset,
                    first_user_data
                );
                next_direct_io_id_ += request_count;
                total_requests += request_count;
                jobs.push_back(ReadJob{first_user_data, request_count});
            }

            const size_t submitted = direct_io_reader_->submit_pending_reads();
            if (submitted != total_requests) {
                throw std::runtime_error("V4Pipeline: direct expert batch submitted an unexpected request count");
            }

            std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> completions;
            completions.reserve(total_requests);
            for (size_t i = 0; i < total_requests; ++i) {
                const auto completion = direct_io_reader_->wait_for_completion();
                completions.emplace(completion.user_data, completion);
            }

            for (const auto& job : jobs) {
                for (size_t chunk = 0; chunk < job.request_count; ++chunk) {
                    const uint64_t request_id = job.first_user_data + chunk;
                    const auto completion_it = completions.find(request_id);
                    if (completion_it == completions.end()) {
                        throw std::runtime_error("V4Pipeline: missing direct expert completion");
                    }

                    const auto completion = completion_it->second;
                    if (completion.result < 0) {
                        throw std::runtime_error("V4Pipeline: direct expert read failed: " +
                                                 std::string(strerror(-completion.result)));
                    }
                    const size_t chunk_offset = chunk * aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
                    const size_t expected_bytes = std::min(
                        aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES,
                        static_cast<size_t>(AEON_EXPERT_BYTES) - chunk_offset
                    );
                    if (completion.result != static_cast<int32_t>(expected_bytes)) {
                        throw std::runtime_error("V4Pipeline: direct expert read returned a short payload");
                    }
                }
            }
        }
    }

    float* d_cos_cache_{nullptr};
    float* d_sin_cache_{nullptr};
};

} // namespace aeon::core
