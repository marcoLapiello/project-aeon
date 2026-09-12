#pragma once

#include "core/config.hpp"
#include "core/device.hpp"
#include "core/aeon_loader.hpp"
#include "io/direct_io_reader.hpp"
#include "core/expert_registry.hpp"
#include "core/host_expert_pool.hpp"
#include "core/memory_budget.hpp"
#include "core/prefetch_staging.hpp"
#include "core/routing_counter.hpp"
#include "core/supply_telemetry.hpp"
#include "core/v4_layer.hpp"
#include "core/v4_pipeline_scratch.hpp"
#include "core/vram_expert_pool.hpp"
#include "kernel/hc_sinkhorn.hpp"
#include "kernel/aeon_moe_fused_w13.hpp"
#include "kernel/aeon_moe_fused_w2.hpp"
#include "kernel/moe_router.hpp"
#include "kernel/v4_attention.hpp"
#include "kernel/v4_pipeline_ops.hpp"
#include "text/text_generation.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)
#endif

namespace aeon::core {

// Complete DeepSeek-V4 Autoregressive Multi-Layer Pipeline Engine
class V4Pipeline {
public:
    struct ExpertTimingPhase {
        uint64_t routed_layer_count{0};
        double routed_section_ms{0.0};
        double expert_compute_ms{0.0};
    };

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
    hipStream_t sdma_stream{0};       // Warm Host H2D uploads
    hipStream_t sdma_cold_stream{0};  // io_uring staging -> VRAM cold uploads
    hipStream_t demotion_stream{0};   // lowest-priority Hot -> Warm D2H refills

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
    std::optional<RoutingCounter> routing_counter_;
    SupplyTelemetry supply_telemetry_;
    bool expert_timing_enabled_{false};
    std::array<ExpertTimingPhase, 2> expert_timing_{};
    std::vector<hipEvent_t> routed_section_start_events_;
    std::vector<hipEvent_t> routed_section_stop_events_;
    std::vector<hipEvent_t> expert_compute_start_events_;
    std::vector<hipEvent_t> expert_compute_stop_events_;

    struct PendingRegistryTransfer {
        uint64_t operation_id{0};
        uint32_t global_expert_id{0};
        uint32_t demoted_expert_id{0};
        ExpertTier source_tier{ExpertTier::COLD_NVME};
        SupplyTelemetryPhase phase{SupplyTelemetryPhase::Decode};
        std::chrono::steady_clock::time_point h2d_enqueued_at{};
        hipEvent_t demotion_event{nullptr};
        hipEvent_t h2d_event{nullptr};
        uint32_t staging_idx{0};
        bool has_staging{false};
        bool h2d_submitted{false};
        int32_t demotion_source_slot{-1};
        int32_t demotion_destination_slot{-1};
        int32_t demotion_staging_idx{-1};
        bool demotion_uses_staging{false};
        bool demotion_submitted{false};
        int32_t h2d_source_slot{-1};
        int32_t h2d_destination_slot{-1};
        std::chrono::steady_clock::time_point io_submitted_at{};
        uint64_t nvme_read_service_ns{0};
        uint64_t nvme_completion_wait_ns{0};
        uint64_t staging_wait_ns{0};
        uint64_t staging_reuse_wait_ns{0};
        std::chrono::steady_clock::time_point staging_acquired_at{};
        std::chrono::steady_clock::time_point gpu_wait_started_at{};
        bool request_failed{false};
        std::string failure_reason;
    };
    std::vector<PendingRegistryTransfer> registry_transfers_;
    static constexpr uint64_t DEMOTION_QUEUE_CAPACITY = 2;
    uint64_t demotion_queue_capacity_{DEMOTION_QUEUE_CAPACITY};

    V4Pipeline() = default;

    ~V4Pipeline() {
        free_all();
    }

    void enable_routing_counter() {
        routing_counter_.emplace(num_layers_);
    }

    void reset_routing_counter() {
        if (routing_counter_) {
            routing_counter_->reset();
        }
    }

    const RoutingCounter* routing_counter() const {
        return routing_counter_ ? &*routing_counter_ : nullptr;
    }

    void enable_supply_telemetry(const std::string& jsonl_path, const std::string& run_id) {
        supply_telemetry_.enable_jsonl(jsonl_path, run_id);
    }

    void reset_supply_telemetry() {
        supply_telemetry_.reset();
    }

    void flush_supply_telemetry() {
        supply_telemetry_.flush();
    }

    void enable_expert_timing() {
        free_expert_timing_events();
        routed_section_start_events_.resize(num_layers_, nullptr);
        routed_section_stop_events_.resize(num_layers_, nullptr);
        const size_t expert_event_count = num_layers_;
        expert_compute_start_events_.resize(expert_event_count, nullptr);
        expert_compute_stop_events_.resize(expert_event_count, nullptr);
        try {
            for (uint32_t layer = 0; layer < num_layers_; ++layer) {
                CHECK_HIP(hipEventCreate(&routed_section_start_events_[layer]));
                CHECK_HIP(hipEventCreate(&routed_section_stop_events_[layer]));
            }
            for (size_t index = 0; index < expert_event_count; ++index) {
                CHECK_HIP(hipEventCreate(&expert_compute_start_events_[index]));
                CHECK_HIP(hipEventCreate(&expert_compute_stop_events_[index]));
            }
        } catch (...) {
            free_expert_timing_events();
            throw;
        }
        expert_timing_enabled_ = true;
        reset_expert_timing();
    }

    void reset_expert_timing() {
        expert_timing_ = {};
    }

    const ExpertTimingPhase& expert_timing(RoutingPhase phase) const {
        return expert_timing_[static_cast<size_t>(phase)];
    }

    uint32_t context_capacity() const {
        return layers.empty() ? 0 : layers.front()->max_seq_len_;
    }

    // Initialize pipeline directly from native .aeon format folder
    void init_aeon(
        const std::string& aeon_model_dir,
        uint32_t num_layers = 2,
        uint32_t unified_vram_slots = 0,
        uint32_t max_seq_len = 4096,
        bool enable_direct_io = true,
        const AeonArtifactSpec& artifact = make_current_swizzled_artifact_spec()
    ) {
        num_layers_ = num_layers;
        current_seq_len_ = 0;

        // 1. Initialize streams
        initialize_streams();

        // 2. Open Aeon Model via Zero-Copy Mmap
        std::cout << "[Pipeline] Opening native Aeon model from " << aeon_model_dir << "..." << std::endl;
        aeon_loader.open_model(aeon_model_dir, artifact);
        const auto& expert_format = aeon_loader.expert_format();
        if (expert_format.kind != ExpertFormatKind::SWIZZLED_W4A16) {
            throw std::runtime_error(
                "V4Pipeline: selected artifact requires a different weight backend");
        }
        if (enable_direct_io) {
            direct_io_reader_ = std::make_unique<aeon::io::DirectIOReader>(
                64, true, expert_format.sector_size);
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
        unified_vram_pool_ = std::make_unique<UnifiedVRAMExpertPool>(
            active_vram_slots, expert_format);
        expert_registry_ = std::make_unique<ExpertRegistry>(
            num_layers_, expert_format.experts_per_layer, active_vram_slots, 0);
        prefetch_staging_ = std::make_unique<PrefetchStagingArena>(expert_format);

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
        uint32_t num_layers = 2,
        const AeonArtifactSpec& artifact = make_current_swizzled_artifact_spec()
    ) {
        std::cout << "================================================================================" << std::endl;
        std::cout << "      Project Aeon — Dynamic VRAM Budget & Global Expert Pool Pipeline          " << std::endl;
        std::cout << "================================================================================" << std::endl;

        num_layers_ = num_layers;
        current_seq_len_ = 0;
        demotion_queue_capacity_ = runtime_cfg.enable_warm_refill
            ? DEMOTION_QUEUE_CAPACITY
            : 0;

        // 1. Initialize streams
        initialize_streams();

        // 2. Open Model Containers
        std::cout << "[Pipeline] Opening native .aeon model from " << aeon_model_dir << "..." << std::endl;
        aeon_loader.open_model(aeon_model_dir, artifact);
        const auto& expert_format = aeon_loader.expert_format();
        if (expert_format.kind != ExpertFormatKind::SWIZZLED_W4A16) {
            throw std::runtime_error(
                "V4Pipeline: selected artifact requires a different weight backend");
        }
        direct_io_reader_ = std::make_unique<aeon::io::DirectIOReader>(
            64, true, expert_format.sector_size);
        std::cout << "  > Dense tensors indexed: " << aeon_loader.total_dense_tensors() << std::endl;

        // Load config
        auto model_cfg = DeepSeekV4Config::load_from_json(aeon_model_dir + "/config.json");

        // 3. Evaluate Memory Budget & Feasibility Gate
        const size_t dense_bytes = aeon_loader.dense_file_size();
        budget_report_ = MemoryBudgetEngine::evaluate(
            runtime_cfg, model_cfg, dense_bytes, expert_format);
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
        unified_vram_pool_ = std::make_unique<UnifiedVRAMExpertPool>(
            budget_report_.hot_vram_slots, expert_format);
        prefetch_staging_ = std::make_unique<PrefetchStagingArena>(expert_format);

        // 9. Initialize Expert Registry Catalog
        const uint32_t active_warm_host_slots = budget_report_.warm_host_slots;
        std::cout << "[Pipeline] Initializing Expert Registry (VRAM=" << budget_report_.hot_vram_slots
              << ", Host=" << active_warm_host_slots << ")..." << std::endl;
        expert_registry_ = std::make_unique<ExpertRegistry>(
            num_layers_, expert_format.experts_per_layer,
            budget_report_.hot_vram_slots, active_warm_host_slots,
            runtime_cfg.preload_warm_host
        );

        // 10. Preload Hot VRAM slots into Unified Pool
        std::cout << "[Pipeline] Pre-populating Hot VRAM slots into Unified Pool..." << std::endl;
        const size_t direct_batch_slots = std::max<size_t>(
            1, direct_io_reader_->submission_capacity() /
               ((expert_format.payload_bytes + aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES - 1) /
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
                buffers.emplace_back(expert_format.payload_bytes, expert_format.sector_size);
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

        // 11. Allocate and populate the persistent Warm Host DDR pool.
        if (active_warm_host_slots > 0) {
            const size_t active_warm_host_bytes = static_cast<size_t>(active_warm_host_slots) *
                                                  expert_format.payload_bytes;
            std::cout << "[Pipeline] Allocating Tier 2 Warm Host DDR Pool (" << active_warm_host_slots << " slots, "
                      << (active_warm_host_bytes / (1024*1024*1024.0)) << " GB)..." << std::endl;
            host_pool_ = std::make_unique<HostExpertPool>(
                active_warm_host_slots, expert_format);

            if (runtime_cfg.preload_warm_host) {
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
            } else {
                std::cout << "  > Tier 2 Warm Host DDR Pool allocated empty; refill is lazy and asynchronous."
                          << std::endl;
            }
        }

        std::cout << "[Pipeline] Engine ready! Configured for " << num_layers_
                  << " layers with " << budget_report_.hot_vram_slots << " hot VRAM slots and "
                  << active_warm_host_slots << " active warm host slots." << std::endl;
    }

    // Run Single Autoregressive Step for token_id at sequence position `pos`
    // Returns next token ID via greedy argmax
    uint32_t step(uint32_t token_id, uint32_t pos, RoutingPhase phase = RoutingPhase::Decode) {
        supply_telemetry_.set_phase(phase);
        reap_registry_transfers();
        const size_t expert_payload_bytes = this->expert_payload_bytes();

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
            std::array<int32_t, 6> vram_slots{-1, -1, -1, -1, -1, -1};
            std::array<uint32_t, 6> global_expert_ids{0, 0, 0, 0, 0, 0};
            std::array<uint64_t, 6> operation_ids{0, 0, 0, 0, 0, 0};
            std::array<bool, 6> is_prefetched{false, false, false, false, false, false};
            std::array<uint32_t, 6> staging_indices{0, 0, 0, 0, 0, 0};
            std::array<bool, 6> io_pending{false, false, false, false, false, false};
            std::array<uint64_t, 6> io_user_data{0, 0, 0, 0, 0, 0};
            std::array<uint32_t, 6> io_request_counts{0, 0, 0, 0, 0, 0};
        };
        std::vector<uint32_t> releasable_staging_slots;
        std::vector<uint32_t> leased_experts;

        auto dispatch_layer_prefetch = [&](uint32_t target_l, const std::vector<int32_t>& topk_experts) -> LayerPrefetchState {
            LayerPrefetchState state;
            uint32_t buf_offset = (target_l % 2) * 6;
            bool submitted_direct_io = false;

            std::array<int, 6> request_order{0, 1, 2, 3, 4, 5};
            auto reservation_priority = [&](int request_index) {
                const uint32_t gid = expert_registry_->get_global_id(
                    target_l, static_cast<uint32_t>(topk_experts[request_index]));
                const auto& entry = expert_registry_->catalog[gid];
                if (entry.owner == ExpertTier::HOT_VRAM &&
                    entry.operation == ExpertOperation::NONE &&
                    entry.publication == ExpertPublication::PUBLISHED) {
                    return 0;
                }
                if (entry.owner == ExpertTier::HOT_VRAM) {
                    return 1;
                }
                return 2;
            };
            std::stable_sort(request_order.begin(), request_order.end(), [&](int left, int right) {
                return reservation_priority(left) < reservation_priority(right);
            });

            for (int request_order_index = 0; request_order_index < 6; ++request_order_index) {
                const int k = request_order[request_order_index];
                uint32_t expert_id = static_cast<uint32_t>(topk_experts[k]);
                ExpertRequestReservation request;
                for (;;) {
                    request = expert_registry_->reserve_request(
                        target_l, expert_id, pos, demotion_queue_capacity_);
                    if (request.kind != ExpertRequestKind::PENDING) {
                        break;
                    }

                    const auto* pending = find_registry_transfer(request.operation_id);
                    const auto& pending_entry = expert_registry_->catalog[request.global_expert_id];
                    if (pending_entry.operation != ExpertOperation::DEMOTION_PENDING) {
                        break;
                    }
                    const hipError_t h2d_status = pending != nullptr && pending->h2d_event != nullptr
                        ? hipEventQuery(pending->h2d_event)
                        : hipErrorNotReady;
                    if (h2d_status == hipSuccess) {
                        expert_registry_->release_lease(request.global_expert_id);
                        reap_registry_transfers();
                        continue;
                    }
                    if (h2d_status != hipErrorNotReady) {
                        throw std::runtime_error(
                            "V4Pipeline: failed to query an optional demotion dependency "
                            "(layer=" + std::to_string(target_l) +
                            ", expert=" + std::to_string(expert_id) +
                            ", operation=" + std::to_string(request.operation_id) + ")"
                        );
                    }
                    expert_registry_->release_lease(request.global_expert_id);
                    throw std::runtime_error(
                        "V4Pipeline: request encountered an in-flight optional demotion; "
                        "request-path CPU synchronization is forbidden "
                        "(layer=" + std::to_string(target_l) +
                        ", expert=" + std::to_string(expert_id) +
                        ", operation=" + std::to_string(request.operation_id) + ")"
                    );
                }
                state.vram_slots[k] = request.vram_slot;
                state.global_expert_ids[k] = request.global_expert_id;
                state.operation_ids[k] = request.operation_id;

                record_supply_request(request);
                observe_supply_occupancy(request.source_tier);
                leased_experts.push_back(request.global_expert_id);

                if (request.kind == ExpertRequestKind::HOT_HIT) {
                    continue;
                }

                if (request.kind == ExpertRequestKind::PENDING) {
                    const auto* pending = find_registry_transfer(request.operation_id);
                    if (pending == nullptr || !pending->h2d_submitted) {
                        throw std::runtime_error(
                            "V4Pipeline: duplicate request joined before its transfer was submitted "
                            "(layer=" + std::to_string(target_l) +
                            ", k=" + std::to_string(k) +
                            ", expert=" + std::to_string(expert_id) +
                            ", gid=" + std::to_string(request.global_expert_id) +
                            ", operation=" + std::to_string(request.operation_id) +
                            ", transfer=" + (pending == nullptr ? "missing" : "present") +
                            ", h2d=" + (pending != nullptr && pending->h2d_submitted ? "submitted" : "pending") + ")"
                        );
                    }
                    state.is_prefetched[k] = pending->has_staging;
                    state.staging_indices[k] = pending->staging_idx;
                    continue;
                }

                try {
                    schedule_demotion(request);
                    const bool source_is_warm = request.source_tier == ExpertTier::WARM_HOST;
                    const int32_t source_slot = request.source_host_slot;
                    const bool can_direct_read = request.source_tier == ExpertTier::COLD_NVME &&
                                                 direct_io_reader_ &&
                                                 aeon_loader.total_dense_tensors() > 0 &&
                                                 prefetch_staging_;
                    ensure_registry_transfer(request.operation_id, request.global_expert_id);

                    if (can_direct_read) {
                        const uint32_t staging_idx = buf_offset + k;
                        bind_staging(request.operation_id, staging_idx);
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

                        state.staging_indices[k] = staging_idx;
                        state.io_pending[k] = true;
                        state.io_user_data[k] = request_id;
                        state.io_request_counts[k] = static_cast<uint32_t>(request_count);
                        submitted_direct_io = true;
                    } else if (source_is_warm && host_pool_ && source_slot >= 0 &&
                               host_pool_->is_slot_pinned(static_cast<uint32_t>(source_slot))) {
                        const uint32_t staging_idx = buf_offset + k;
                        bind_staging(request.operation_id, staging_idx);
                        prefetch_staging_->begin_direct_transfer(staging_idx);
                        wait_for_demotion_dependency(request.operation_id, sdma_stream);
                        unified_vram_pool_->upload_from_host_expert(
                            static_cast<uint32_t>(request.vram_slot),
                            host_pool_->get_expert_slot_ptr(static_cast<uint32_t>(source_slot)),
                            sdma_stream
                        );
                        CHECK_HIP(hipEventRecord(prefetch_staging_->events[staging_idx], sdma_stream));
                        record_h2d_event(
                            request.operation_id, sdma_stream, staging_idx, true,
                            source_slot, request.vram_slot);

                        state.is_prefetched[k] = true;
                        state.staging_indices[k] = staging_idx;
                    } else if (source_is_warm) {
                        const uint32_t staging_idx = buf_offset + k;
                        auto* transfer = find_registry_transfer(request.operation_id);
                        transfer->staging_idx = staging_idx;
                        transfer->has_staging = true;
                        prefetch_staging_->stage_payload(
                            staging_idx,
                            host_pool_->get_expert_slot_ptr(static_cast<uint32_t>(source_slot))
                        );
                        const uint8_t* pinned_payload = prefetch_staging_->get_slot_ptr(staging_idx);
                        prefetch_staging_->begin_gpu_transfer(staging_idx);
                        wait_for_demotion_dependency(request.operation_id, sdma_stream);
                        unified_vram_pool_->upload_from_host_expert(
                            static_cast<uint32_t>(request.vram_slot), pinned_payload, sdma_stream);
                        CHECK_HIP(hipEventRecord(prefetch_staging_->events[staging_idx], sdma_stream));
                        record_h2d_event(
                            request.operation_id, sdma_stream, staging_idx, true,
                            static_cast<int32_t>(staging_idx), request.vram_slot);

                        state.is_prefetched[k] = true;
                        state.staging_indices[k] = staging_idx;
                    } else {
                        const uint8_t* src_ptr = nullptr;
                        if (aeon_loader.total_dense_tensors() > 0) {
                            src_ptr = aeon_loader.get_expert_data(target_l, expert_id);
                        }

                        if (src_ptr && prefetch_staging_) {
                            const uint32_t staging_idx = buf_offset + k;
                            bind_staging(request.operation_id, staging_idx);
                            prefetch_staging_->stage_payload(staging_idx, src_ptr);
                            const uint8_t* pinned_payload = prefetch_staging_->get_slot_ptr(staging_idx);
                            prefetch_staging_->begin_gpu_transfer(staging_idx);
                            wait_for_demotion_dependency(request.operation_id, sdma_stream);
                            unified_vram_pool_->upload_from_host_expert(
                                static_cast<uint32_t>(request.vram_slot), pinned_payload, sdma_stream);
                            CHECK_HIP(hipEventRecord(prefetch_staging_->events[staging_idx], sdma_stream));
                            record_h2d_event(
                                request.operation_id, sdma_stream, staging_idx, true,
                                static_cast<int32_t>(staging_idx), request.vram_slot);

                            state.is_prefetched[k] = true;
                            state.staging_indices[k] = staging_idx;
                        } else if (src_ptr) {
                            wait_for_demotion_dependency(request.operation_id, compute_stream);
                            unified_vram_pool_->upload_from_host_expert(
                                static_cast<uint32_t>(request.vram_slot), src_ptr, compute_stream);
                            record_h2d_event(
                                request.operation_id, compute_stream, 0, false,
                                -1, request.vram_slot);
                        } else {
                            throw std::runtime_error(
                                "V4Pipeline: native Aeon expert payload unavailable for requested expert"
                            );
                        }
                    }
                } catch (...) {
                    mark_registry_request_failed(request.operation_id, "transfer_submission_failure");
                    reap_registry_transfers();
                    throw;
                }
            }
            if (submitted_direct_io) {
                try {
                    direct_io_reader_->submit_pending_reads();
                } catch (...) {
                    for (int k = 0; k < 6; ++k) {
                        if (state.io_pending[k]) {
                            mark_registry_request_failed(
                                state.operation_ids[k], "nvme_submit_failure");
                        }
                    }
                    throw;
                }
                const auto submitted_at = std::chrono::steady_clock::now();
                for (int k = 0; k < 6; ++k) {
                    if (!state.io_pending[k]) continue;
                    auto* transfer = find_registry_transfer(state.operation_ids[k]);
                    if (transfer != nullptr) {
                        transfer->io_submitted_at = submitted_at;
                    }
                }
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
                    const bool waited_for_completion = completion_it == direct_io_completions_.end();
                    const auto wait_started_at = std::chrono::steady_clock::now();
                    while (completion_it == direct_io_completions_.end()) {
                        if (!direct_io_reader_) {
                            mark_registry_request_failed(
                                state.operation_ids[k], "nvme_reader_unavailable");
                            throw std::runtime_error("V4Pipeline: direct I/O request has no reader");
                        }
                        aeon::io::DirectIOCompletion completion;
                        try {
                            completion = direct_io_reader_->wait_for_completion();
                        } catch (...) {
                            mark_registry_request_failed(
                                state.operation_ids[k], "nvme_completion_wait_failure");
                            throw;
                        }
                        direct_io_completions_[completion.user_data] = completion;
                        completion_it = direct_io_completions_.find(request_id);
                    }

                    const auto completion = completion_it->second;
                    direct_io_completions_.erase(completion_it);
                    const auto completed_at = std::chrono::steady_clock::now();
                    if (auto* transfer = find_registry_transfer(state.operation_ids[k])) {
                        if (transfer->io_submitted_at.time_since_epoch().count() != 0) {
                            transfer->nvme_read_service_ns += static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    completed_at - transfer->io_submitted_at).count());
                        }
                        if (waited_for_completion) {
                            transfer->nvme_completion_wait_ns += static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    completed_at - wait_started_at).count());
                        }
                    }
                    if (completion.result < 0) {
                        mark_registry_request_failed(
                            state.operation_ids[k], "nvme_read_failure");
                        throw std::runtime_error("V4Pipeline: direct expert read failed: " +
                                                 std::string(strerror(-completion.result)));
                    }
                    const size_t chunk_offset = chunk * aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
                    const size_t expected_bytes = std::min(
                        aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES,
                        expert_payload_bytes - chunk_offset
                    );
                    if (completion.result != static_cast<int32_t>(expected_bytes)) {
                        mark_registry_request_failed(
                            state.operation_ids[k], "nvme_short_read");
                        throw std::runtime_error("V4Pipeline: direct expert read returned a short payload");
                    }
                }

                const uint32_t staging_idx = state.staging_indices[k];
                prefetch_staging_->complete_io(staging_idx);
                prefetch_staging_->begin_gpu_transfer(staging_idx);
                // Cold NVMe payloads upload on the dedicated cold-DMA stream so a
                // burst of io_uring completions never head-of-line blocks warm-hit
                // or Warm Host H2D transfers on sdma_stream.
                wait_for_demotion_dependency(state.operation_ids[k], sdma_cold_stream);
                unified_vram_pool_->upload_from_host_expert(
                    static_cast<uint32_t>(state.vram_slots[k]),
                    prefetch_staging_->get_slot_ptr(staging_idx),
                    sdma_cold_stream
                );
                CHECK_HIP(hipEventRecord(prefetch_staging_->events[staging_idx], sdma_cold_stream));
                record_h2d_event(
                    state.operation_ids[k], sdma_cold_stream, staging_idx, true,
                    static_cast<int32_t>(staging_idx), state.vram_slots[k]);

                state.is_prefetched[k] = true;
                state.io_pending[k] = false;
            }
        };

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

            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_unit_wave32_kernel,
                dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                scratch.d_q, scratch.d_q, HEAD_DIM, 1e-6f
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
            // Keep the half GEMV output separate from the float router input:
            // widening in-place would overwrite unread half logits.
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_vec8_kernel,
                dim3(256, 1), dim3(32), 0, compute_stream,
                scratch.d_ffn_norm_act, layer.d_gate_weight, scratch.d_router_logits_half, H
            );
            kernel::v4_half_to_float_n_kernel<<<(256 + 255) / 256, 256, 0, compute_stream>>>(
                scratch.d_router_logits_half, scratch.d_router_logits, 256
            );

            // Launch Router Kernel (Hash mode for layers 0..2)
            hipLaunchKernelGGL(
                kernel::moe_router_kernel,
                dim3(1), dim3(64), 0, compute_stream,
                scratch.d_router_logits,
                layer.is_hash_layer ? nullptr : layer.d_gate_bias,
                layer.d_tid2eid, scratch.d_token_id,
                scratch.d_topk_weights, scratch.d_topk_indices,
                256, 6, 1.5f, true
            );

            std::vector<float> h_topk_weights(6);
            std::vector<int32_t> h_topk_indices(6);
            CHECK_HIP(hipMemcpyAsync(h_topk_weights.data(), scratch.d_topk_weights, 6 * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipMemcpyAsync(h_topk_indices.data(), scratch.d_topk_indices, 6 * sizeof(int32_t), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            for (uint32_t gid : leased_experts) {
                expert_registry_->release_lease(gid);
            }
            leased_experts.clear();
            reap_registry_transfers();

            if (routing_counter_) {
                routing_counter_->record(phase, l, pos, h_topk_indices.data());
            }

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
                kernel::v4_gemv_fp16_vec8_kernel,
                dim3(INTER_DIM, 1), dim3(32), 0, compute_stream,
                scratch.d_ffn_norm_act, layer.d_shared_w1, scratch.d_shared_gate, H
            );
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_vec8_kernel,
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
                kernel::v4_gemv_fp16_vec8_kernel,
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
            LayerPrefetchState active_prefetch = dispatch_layer_prefetch(l, h_topk_indices);

            // Let cold NVMe reads overlap the shared expert pass; materialization
            // still completes before the first routed expert consumes each slot.
            materialize_layer_prefetch(active_prefetch);

            for (int k = 0; k < 6; ++k) {
                pending_transfers[k].vram_slot = active_prefetch.vram_slots[k];
                pending_transfers[k].is_prefetched = active_prefetch.is_prefetched[k];
                pending_transfers[k].staging_idx = active_prefetch.staging_indices[k];
            }

            if (expert_timing_enabled_) {
                CHECK_HIP(hipEventRecord(
                    routed_section_start_events_[l], compute_stream));
            }

            // All routed experts use the wave-oriented layout and fused dispatches.
            kernel::SwizzledW13ExpertPtrs fused_w13{};
            kernel::SwizzledW2ExpertPtrs fused_w2{};
            for (int k = 0; k < 6; ++k) {
                const int32_t slot = pending_transfers[k].vram_slot;
                if (pending_transfers[k].is_prefetched && prefetch_staging_) {
                    mark_gpu_readiness_wait_start(active_prefetch.operation_ids[k]);
                    CHECK_HIP(hipStreamWaitEvent(
                        compute_stream,
                        prefetch_staging_->events[pending_transfers[k].staging_idx], 0));
                }

                fused_w13.w1[k] = reinterpret_cast<const uint4*>(
                    unified_vram_pool_->get_w1_packed(slot));
                fused_w13.s1[k] = unified_vram_pool_->get_w1_scale(slot);
                fused_w13.w3[k] = reinterpret_cast<const uint4*>(
                    unified_vram_pool_->get_w3_packed(slot));
                fused_w13.s3[k] = unified_vram_pool_->get_w3_scale(slot);
                fused_w2.w2[k] = reinterpret_cast<const uint4*>(
                    unified_vram_pool_->get_w2_packed(slot));
                fused_w2.s2[k] = unified_vram_pool_->get_w2_scale(slot);
            }

            if (expert_timing_enabled_) {
                const size_t event_index = static_cast<size_t>(l);
                CHECK_HIP(hipEventRecord(
                    expert_compute_start_events_[event_index], compute_stream));
            }

            layer.cache_hits = expert_registry_->hits_hot;
            layer.cache_misses = expert_registry_->hits_warm + expert_registry_->misses_cold;

            kernel::dispatch_aeon_moe_fused_w13_swiglu<8, 4, 8, 16>(
                scratch.d_ffn_norm_act,
                fused_w13,
                scratch.d_swizzled_expert_hidden,
                scratch.d_swizzled_moe_accum_f32,
                H,
                6, INTER_DIM, H, 10.0f, compute_stream);
            kernel::dispatch_aeon_moe_fused_w2_accum<8, 8, 4, 16>(
                scratch.d_swizzled_expert_hidden,
                fused_w2,
                scratch.d_topk_weights,
                scratch.d_moe_accum,
                scratch.d_swizzled_moe_accum_f32,
                scratch.d_moe_accum,
                scratch.d_swizzled_counters,
                6, H, INTER_DIM, compute_stream);

            if (expert_timing_enabled_) {
                const size_t event_index = static_cast<size_t>(l);
                CHECK_HIP(hipEventRecord(
                    expert_compute_stop_events_[event_index], compute_stream));
            }

            if (expert_timing_enabled_) {
                CHECK_HIP(hipEventRecord(
                    routed_section_stop_events_[l], compute_stream));
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
            kernel::v4_gemv_fp16_vec8_kernel,
            dim3(129280, 1), dim3(32), 0, compute_stream,
            scratch.d_head_norm, d_lm_head, scratch.d_logits, H
        );

        // 6. GPU Argmax Sampling over the [129280] logit head: replaces the
        // 258 KB D2H + sync + 129,280-element CPU scan with one device kernel,
        // leaving only a 4-byte result readback on the critical path.
        kernel::v4_argmax_fp16_partial_kernel<<<kernel::V4_ARGMAX_BLOCKS, 256, 0, compute_stream>>>(
            scratch.d_logits, 129280,
            scratch.d_argmax_partial_vals, scratch.d_argmax_partial_idx
        );
        kernel::v4_argmax_partial_reduce_kernel<<<1, 256, 0, compute_stream>>>(
            scratch.d_argmax_partial_vals,
            scratch.d_argmax_partial_idx,
            kernel::V4_ARGMAX_BLOCKS,
            scratch.d_argmax_result
        );
        for (uint32_t staging_idx : releasable_staging_slots) {
            prefetch_staging_->release_after_gpu_transfer(staging_idx);
        }

        int32_t h_argmax = 0;
        CHECK_HIP(hipMemcpyAsync(&h_argmax, scratch.d_argmax_result, sizeof(int32_t), hipMemcpyDeviceToHost, compute_stream));
        CHECK_HIP(hipStreamSynchronize(compute_stream));
        for (uint32_t gid : leased_experts) {
            expert_registry_->release_lease(gid);
        }
        if (phase == RoutingPhase::Decode) {
            supply_telemetry_.record_decode_token();
        }
        reap_registry_transfers();
        if (expert_timing_enabled_) {
            collect_expert_timing(phase);
        }
        uint32_t best_tok = static_cast<uint32_t>(h_argmax);

        current_seq_len_ = pos + 1;
        return best_tok;
    }

    void reset_generation_state() {
        current_seq_len_ = 0;
        for (auto& l : layers) {
            CHECK_HIP(hipMemset(l->d_kv_cache, 0, l->max_seq_len_ * kernel::DSV4_HEAD_DIM * sizeof(half)));
        }
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
        reset_generation_state();

        // Prefill Phase
        auto t_prefill_start = std::chrono::high_resolution_clock::now();
        uint32_t next_tok = 0;
        for (size_t i = 0; i < prompt.size(); ++i) {
            next_tok = step(prompt[i], static_cast<uint32_t>(i), RoutingPhase::Prefill);
        }
        auto t_prefill_end = std::chrono::high_resolution_clock::now();

        double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
        if (out_ttft_ms) *out_ttft_ms = ttft_ms;

        generated.push_back(next_tok);

        // Autoregressive Decoding Phase
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        for (uint32_t step_idx = 1; step_idx < max_new_tokens; ++step_idx) {
            uint32_t pos = prompt.size() + step_idx - 1;
            next_tok = step(next_tok, pos, RoutingPhase::Decode);
            generated.push_back(next_tok);
        }
        auto t_decode_end = std::chrono::high_resolution_clock::now();

        double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
        double tok_sec = (max_new_tokens > 1) ? ((max_new_tokens - 1) / (decode_ms / 1000.0)) : 0.0;
        if (out_tok_per_sec) *out_tok_per_sec = tok_sec;

        return generated;
    }

    aeon::text::GenerationResult generate_until_stop(
        const std::vector<uint32_t>& prompt,
        const aeon::text::GenerationOptions& options,
        double* out_ttft_ms = nullptr,
        double* out_tok_per_sec = nullptr
    ) {
        aeon::text::GenerationOptions effective_options = options;
        if (effective_options.context_limit == 0) {
            effective_options.context_limit = context_capacity();
        }

        reset_generation_state();
        const auto prefill_start = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point prefill_end{};
        std::chrono::high_resolution_clock::time_point decode_start{};
        bool decode_started = false;

        const auto result = aeon::text::generate_token_ids(
            prompt,
            effective_options,
            [&](uint32_t token_id, uint32_t position, bool prefill) {
                if (!prefill && !decode_started) {
                    prefill_end = std::chrono::high_resolution_clock::now();
                    decode_start = prefill_end;
                    decode_started = true;
                }
                const auto next = step(
                    token_id,
                    position,
                    prefill ? RoutingPhase::Prefill : RoutingPhase::Decode
                );
                if (prefill && position + 1 == prompt.size()) {
                    prefill_end = std::chrono::high_resolution_clock::now();
                }
                return next;
            }
        );

        if (out_ttft_ms) {
            const auto end = prefill_end.time_since_epoch().count() == 0
                ? std::chrono::high_resolution_clock::now()
                : prefill_end;
            *out_ttft_ms = std::chrono::duration<double, std::milli>(end - prefill_start).count();
        }
        if (out_tok_per_sec) {
            if (!decode_started || result.token_ids.size() <= 1) {
                *out_tok_per_sec = 0.0;
            } else {
                const auto decode_end = std::chrono::high_resolution_clock::now();
                const double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
                *out_tok_per_sec = (result.token_ids.size() - 1) / (decode_ms / 1000.0);
            }
        }
        return result;
    }

    void free_all() {
        if (compute_stream) { (void)hipStreamSynchronize(compute_stream); }
        if (sdma_stream) { (void)hipStreamSynchronize(sdma_stream); }
        if (sdma_cold_stream) { (void)hipStreamSynchronize(sdma_cold_stream); }
        if (demotion_stream) { (void)hipStreamSynchronize(demotion_stream); }
        reap_registry_transfers();
        supply_telemetry_.flush();
        if (compute_stream) { (void)hipStreamDestroy(compute_stream); compute_stream = 0; }
        if (sdma_stream) { (void)hipStreamDestroy(sdma_stream); sdma_stream = 0; }
        if (sdma_cold_stream) { (void)hipStreamDestroy(sdma_cold_stream); sdma_cold_stream = 0; }
        if (demotion_stream) { (void)hipStreamDestroy(demotion_stream); demotion_stream = 0; }
        if (d_cos_cache_) { (void)hipFree(d_cos_cache_); d_cos_cache_ = nullptr; }
        if (d_sin_cache_) { (void)hipFree(d_sin_cache_); d_sin_cache_ = nullptr; }
        if (d_lm_head) { (void)hipFree(d_lm_head); d_lm_head = nullptr; }
        if (d_hc_head_fn) { (void)hipFree(d_hc_head_fn); d_hc_head_fn = nullptr; }
        if (d_hc_head_base) { (void)hipFree(d_hc_head_base); d_hc_head_base = nullptr; }
        if (d_hc_head_scale) { (void)hipFree(d_hc_head_scale); d_hc_head_scale = nullptr; }
        if (d_final_norm) { (void)hipFree(d_final_norm); d_final_norm = nullptr; }

        scratch.free();
        free_expert_timing_events();
        routing_counter_.reset();
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
        registry_transfers_.clear();
        supply_telemetry_.disable();
        aeon_loader.close_all();
    }

private:
    size_t expert_payload_bytes() const noexcept {
        return aeon_loader.expert_format().payload_bytes;
    }

    void free_expert_timing_events() {
        for (auto event : routed_section_start_events_) {
            if (event != nullptr) (void)hipEventDestroy(event);
        }
        for (auto event : routed_section_stop_events_) {
            if (event != nullptr) (void)hipEventDestroy(event);
        }
        for (auto event : expert_compute_start_events_) {
            if (event != nullptr) (void)hipEventDestroy(event);
        }
        for (auto event : expert_compute_stop_events_) {
            if (event != nullptr) (void)hipEventDestroy(event);
        }
        routed_section_start_events_.clear();
        routed_section_stop_events_.clear();
        expert_compute_start_events_.clear();
        expert_compute_stop_events_.clear();
        expert_timing_enabled_ = false;
    }

    void collect_expert_timing(RoutingPhase phase) {
        auto& stats = expert_timing_[static_cast<size_t>(phase)];
        for (uint32_t layer = 0; layer < num_layers_; ++layer) {
            float routed_section_ms = 0.0f;
            CHECK_HIP(hipEventElapsedTime(
                &routed_section_ms,
                routed_section_start_events_[layer],
                routed_section_stop_events_[layer]));

            float expert_compute_ms = 0.0f;
            const size_t event_index = static_cast<size_t>(layer);
            CHECK_HIP(hipEventElapsedTime(
                &expert_compute_ms,
                expert_compute_start_events_[event_index],
                expert_compute_stop_events_[event_index]));

            ++stats.routed_layer_count;
            stats.routed_section_ms += routed_section_ms;
            stats.expert_compute_ms += expert_compute_ms;
        }
    }

    void initialize_streams() {
        CHECK_HIP(hipStreamCreate(&compute_stream));
        CHECK_HIP(hipStreamCreate(&sdma_stream));
        CHECK_HIP(hipStreamCreate(&sdma_cold_stream));

        int least_priority = 0;
        int greatest_priority = 0;
        const hipError_t priority_result = hipDeviceGetStreamPriorityRange(
            &least_priority, &greatest_priority);
        if (priority_result == hipSuccess) {
            CHECK_HIP(hipStreamCreateWithPriority(
                &demotion_stream, hipStreamNonBlocking, least_priority));
        } else {
            CHECK_HIP(hipStreamCreateWithFlags(&demotion_stream, hipStreamNonBlocking));
            std::cerr << "[Pipeline] HIP stream priorities unavailable; using a separate demotion stream."
                      << std::endl;
        }
    }

    PendingRegistryTransfer& ensure_registry_transfer(uint64_t operation_id, uint32_t gid) {
        for (auto& transfer : registry_transfers_) {
            if (transfer.operation_id == operation_id) {
                return transfer;
            }
        }
        registry_transfers_.push_back(PendingRegistryTransfer{});
        auto& transfer = registry_transfers_.back();
        transfer.operation_id = operation_id;
        transfer.global_expert_id = gid;
        transfer.source_tier = expert_registry_->catalog[gid].owner;
        transfer.phase = supply_telemetry_.current_phase();
        return transfer;
    }

    PendingRegistryTransfer* find_registry_transfer(uint64_t operation_id) {
        for (auto& transfer : registry_transfers_) {
            if (transfer.operation_id == operation_id) {
                return &transfer;
            }
        }
        return nullptr;
    }

    const PendingRegistryTransfer* find_registry_transfer(uint64_t operation_id) const {
        for (const auto& transfer : registry_transfers_) {
            if (transfer.operation_id == operation_id) {
                return &transfer;
            }
        }
        return nullptr;
    }

    void schedule_demotion(const ExpertRequestReservation& request) {
        if (request.demotion) {
            auto& transfer = ensure_registry_transfer(
                request.operation_id, request.global_expert_id);
            transfer.demoted_expert_id = request.demotion->victim_gid;
            transfer.demotion_source_slot = static_cast<int32_t>(
                request.demotion->source_vram_slot);
            transfer.demotion_destination_slot = static_cast<int32_t>(
                request.demotion->destination_host_slot);
            supply_telemetry_.record_demotion_attempt(transfer.source_tier);
            uint8_t* destination = host_pool_->get_expert_slot_ptr(
                request.demotion->destination_host_slot);
            if (!host_pool_->is_slot_pinned(request.demotion->destination_host_slot)) {
                uint32_t staging_idx = 0;
                if (prefetch_staging_ == nullptr ||
                    !prefetch_staging_->try_begin_direct_transfer(staging_idx)) {
                    expert_registry_->drop_demotion(request.operation_id);
                    supply_telemetry_.record_demotion_drop(
                        transfer.source_tier, "unpinned_fallback_unavailable");
                    supply_telemetry_.record_transfer_event(
                        request.operation_id,
                        transfer.demoted_expert_id,
                        "d2h",
                        transfer.demotion_source_slot,
                        transfer.demotion_destination_slot,
                        "drop",
                        "unpinned_fallback_unavailable");
                    return;
                }
                transfer.demotion_staging_idx = static_cast<int32_t>(staging_idx);
                transfer.demotion_uses_staging = true;
                destination = prefetch_staging_->get_slot_ptr(staging_idx);
            }
            try {
                CHECK_HIP(hipEventCreateWithFlags(&transfer.demotion_event, hipEventDisableTiming));
                unified_vram_pool_->download_to_host_expert(
                    request.demotion->source_vram_slot,
                    destination,
                    demotion_stream
                );
                transfer.demotion_submitted = true;
                CHECK_HIP(hipEventRecord(transfer.demotion_event, demotion_stream));
            } catch (...) {
                if (!transfer.demotion_submitted) {
                    expert_registry_->drop_demotion(request.operation_id);
                    if (transfer.demotion_staging_idx >= 0 && prefetch_staging_) {
                        prefetch_staging_->release_after_failure(
                            static_cast<uint32_t>(transfer.demotion_staging_idx));
                    }
                    if (transfer.demotion_event != nullptr) {
                        (void)hipEventDestroy(transfer.demotion_event);
                        transfer.demotion_event = nullptr;
                    }
                    supply_telemetry_.record_demotion_drop(
                        transfer.source_tier, "d2h_submission_failure");
                    supply_telemetry_.record_transfer_event(
                        request.operation_id,
                        transfer.demoted_expert_id,
                        "d2h",
                        transfer.demotion_source_slot,
                        transfer.demotion_destination_slot,
                        "failed",
                        "d2h_submission_failure");
                }
                throw;
            }
            supply_telemetry_.record_transfer_event(
                request.operation_id,
                transfer.demoted_expert_id,
                "d2h",
                transfer.demotion_source_slot,
                transfer.demotion_destination_slot,
                "submitted");
            return;
        }

        if (expert_registry_->host_capacity == 0) {
            return;
        }

        auto& transfer = ensure_registry_transfer(
            request.operation_id, request.global_expert_id);
        for (const auto& entry : expert_registry_->catalog) {
            if (entry.operation_id == request.operation_id &&
                entry.operation == ExpertOperation::DEMOTION_PENDING) {
                const char* drop_reason = expert_demotion_drop_reason_name(
                    entry.demotion_drop_reason);
                transfer.demoted_expert_id = entry.global_expert_id;
                transfer.demotion_source_slot = entry.slot_idx;
                transfer.demotion_destination_slot = -1;
                supply_telemetry_.record_demotion_attempt(request.source_tier);
                supply_telemetry_.record_demotion_drop(
                    request.source_tier, drop_reason);
                supply_telemetry_.record_transfer_event(
                    request.operation_id,
                    entry.global_expert_id,
                    "d2h",
                    entry.slot_idx,
                    -1,
                    "drop",
                    drop_reason);
                break;
            }
        }
    }

    void wait_for_demotion_dependency(uint64_t operation_id, hipStream_t stream) {
        const auto* transfer = find_registry_transfer(operation_id);
        if (transfer != nullptr && transfer->demotion_event != nullptr) {
            CHECK_HIP(hipStreamWaitEvent(stream, transfer->demotion_event, 0));
        }
    }

    void record_h2d_event(
        uint64_t operation_id,
        hipStream_t stream,
        uint32_t staging_idx,
        bool has_staging,
        int32_t source_slot,
        int32_t destination_slot
    ) {
        auto* transfer = find_registry_transfer(operation_id);
        if (transfer == nullptr) {
            throw std::logic_error("V4Pipeline: H2D completion has no registry transfer");
        }
        if (transfer->h2d_event != nullptr) {
            throw std::logic_error("V4Pipeline: duplicate H2D submission for one expert operation");
        }
        CHECK_HIP(hipEventCreateWithFlags(&transfer->h2d_event, hipEventDisableTiming));
        CHECK_HIP(hipEventRecord(transfer->h2d_event, stream));
        transfer->staging_idx = staging_idx;
        transfer->has_staging = has_staging;
        transfer->h2d_source_slot = source_slot;
        transfer->h2d_destination_slot = destination_slot;
        transfer->h2d_enqueued_at = std::chrono::steady_clock::now();
        if (has_staging && transfer->staging_acquired_at.time_since_epoch().count() != 0) {
            transfer->staging_wait_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    transfer->h2d_enqueued_at - transfer->staging_acquired_at).count());
        }
        transfer->h2d_submitted = true;
    }

    void bind_staging(uint64_t operation_id, uint32_t staging_idx) {
        auto* transfer = find_registry_transfer(operation_id);
        if (transfer == nullptr) {
            throw std::logic_error("V4Pipeline: staging binding has no registry transfer");
        }
        transfer->staging_idx = staging_idx;
        transfer->has_staging = true;
        transfer->staging_acquired_at = std::chrono::steady_clock::now();
        transfer->staging_reuse_wait_ns = prefetch_staging_->take_reuse_delay_ns(staging_idx);
    }

    void mark_gpu_readiness_wait_start(uint64_t operation_id) {
        auto* transfer = find_registry_transfer(operation_id);
        if (transfer != nullptr && transfer->gpu_wait_started_at.time_since_epoch().count() == 0) {
            transfer->gpu_wait_started_at = std::chrono::steady_clock::now();
        }
    }

    void mark_registry_request_failed(uint64_t operation_id, const std::string& reason) {
        auto* transfer = find_registry_transfer(operation_id);
        if (transfer == nullptr) return;
        transfer->request_failed = true;
        transfer->failure_reason = reason;
        if (transfer->has_staging && !transfer->h2d_submitted && prefetch_staging_) {
            prefetch_staging_->release_after_failure(transfer->staging_idx);
        }
    }

    void reap_registry_transfers() {
        for (size_t index = 0; index < registry_transfers_.size();) {
            auto& transfer = registry_transfers_[index];
            bool demotion_ready = transfer.demotion_event == nullptr;
            if (transfer.demotion_event != nullptr) {
                const hipError_t result = hipEventQuery(transfer.demotion_event);
                if (result == hipSuccess) {
                    if (transfer.demotion_uses_staging && prefetch_staging_) {
                        std::memcpy(
                            host_pool_->get_expert_slot_ptr(
                                static_cast<uint32_t>(transfer.demotion_destination_slot)),
                            prefetch_staging_->get_slot_ptr(
                                static_cast<uint32_t>(transfer.demotion_staging_idx)),
                            expert_payload_bytes());
                        prefetch_staging_->release_after_gpu_transfer(
                            static_cast<uint32_t>(transfer.demotion_staging_idx));
                    }
                    expert_registry_->complete_demotion(transfer.operation_id);
                    supply_telemetry_.record_demotion_completion(
                        transfer.source_tier, expert_payload_bytes());
                    supply_telemetry_.record_transfer_event(
                        transfer.operation_id,
                        transfer.demoted_expert_id,
                        "d2h",
                        transfer.demotion_source_slot,
                        transfer.demotion_destination_slot,
                        "complete");
                    (void)hipEventDestroy(transfer.demotion_event);
                    transfer.demotion_event = nullptr;
                    demotion_ready = true;
                } else if (result != hipErrorNotReady) {
                    if (transfer.demotion_uses_staging && prefetch_staging_) {
                        prefetch_staging_->release_after_failure(
                            static_cast<uint32_t>(transfer.demotion_staging_idx));
                    }
                    expert_registry_->fail_demotion(transfer.operation_id);
                    supply_telemetry_.record_demotion_drop(
                        transfer.source_tier, "d2h_failure");
                    supply_telemetry_.record_transfer_event(
                        transfer.operation_id,
                        transfer.demoted_expert_id,
                        "d2h",
                        transfer.demotion_source_slot,
                        transfer.demotion_destination_slot,
                        "failed",
                        "d2h_failure");
                    (void)hipEventDestroy(transfer.demotion_event);
                    transfer.demotion_event = nullptr;
                    demotion_ready = true;
                }
            }

            if (transfer.request_failed && !transfer.h2d_submitted) {
                if (!demotion_ready) {
                    ++index;
                    continue;
                }
                expert_registry_->fail_request(transfer.operation_id);
                supply_telemetry_.record_transfer_event(
                    transfer.operation_id,
                    transfer.global_expert_id,
                    "h2d",
                    transfer.h2d_source_slot,
                    transfer.h2d_destination_slot,
                    "failed",
                    transfer.failure_reason.c_str());
                registry_transfers_.erase(
                    registry_transfers_.begin() + static_cast<std::ptrdiff_t>(index));
                continue;
            }

            if (!transfer.h2d_submitted || transfer.h2d_event == nullptr || !demotion_ready) {
                ++index;
                continue;
            }

            const hipError_t result = hipEventQuery(transfer.h2d_event);
            if (result == hipErrorNotReady) {
                ++index;
                continue;
            }
            if (result != hipSuccess) {
                if (transfer.has_staging && prefetch_staging_) {
                    prefetch_staging_->release_after_failure(transfer.staging_idx);
                }
                expert_registry_->fail_request(transfer.operation_id);
                supply_telemetry_.record_transfer_event(
                    transfer.operation_id,
                    transfer.global_expert_id,
                    "h2d",
                    transfer.h2d_source_slot,
                    transfer.h2d_destination_slot,
                    "failed",
                    "h2d_failure");
                (void)hipEventDestroy(transfer.h2d_event);
                registry_transfers_.erase(registry_transfers_.begin() + static_cast<std::ptrdiff_t>(index));
                continue;
            }

            expert_registry_->complete_request(transfer.operation_id);
            const auto ready_at = std::chrono::steady_clock::now();
            const auto h2d_ns = transfer.h2d_enqueued_at.time_since_epoch().count() == 0
                ? uint64_t{0}
                : static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    ready_at - transfer.h2d_enqueued_at).count());
            const auto gpu_wait_ns = transfer.gpu_wait_started_at.time_since_epoch().count() == 0
                ? uint64_t{0}
                : static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    ready_at - transfer.gpu_wait_started_at).count());
            supply_telemetry_.record_timing(
                transfer.phase,
                transfer.source_tier,
                transfer.nvme_read_service_ns,
                transfer.nvme_completion_wait_ns,
                h2d_ns,
                gpu_wait_ns,
                transfer.staging_wait_ns,
                transfer.staging_reuse_wait_ns);
            supply_telemetry_.record_transfer_event(
                transfer.operation_id,
                transfer.global_expert_id,
                "h2d",
                transfer.h2d_source_slot,
                transfer.h2d_destination_slot,
                "complete");
            (void)hipEventDestroy(transfer.h2d_event);
            registry_transfers_.erase(registry_transfers_.begin() + static_cast<std::ptrdiff_t>(index));
        }
    }

    void record_supply_request(const ExpertRequestReservation& request) {
        const bool physical_transfer = request.kind != ExpertRequestKind::HOT_HIT &&
                                       request.kind != ExpertRequestKind::PENDING;
        const uint64_t logical_bytes = request.source_tier == ExpertTier::HOT_VRAM
            ? 0
            : expert_payload_bytes();
        supply_telemetry_.record_request(
            supply_telemetry_.current_phase(),
            request.source_tier,
            logical_bytes,
            physical_transfer ? expert_payload_bytes() : 0,
            request.source_tier == ExpertTier::COLD_NVME && physical_transfer
                ? expert_payload_bytes() : 0,
            request.source_tier == ExpertTier::WARM_HOST && physical_transfer
                ? expert_payload_bytes() : 0,
            request.source_tier == ExpertTier::COLD_NVME && physical_transfer
                ? expert_payload_bytes() : 0
        );
    }

    void observe_supply_occupancy(ExpertTier source_tier) {
        const uint64_t warm_pinned_bytes = host_pool_
            ? static_cast<uint64_t>(host_pool_->pinned_slot_count()) * host_pool_->payload_bytes()
            : 0;
        const uint64_t warm_unpinned_bytes = host_pool_
            ? static_cast<uint64_t>(host_pool_->unpinned_slot_count()) * host_pool_->payload_bytes()
            : 0;
        supply_telemetry_.observe_occupancy(
            expert_registry_->published_hot_slots(),
            expert_registry_->published_warm_slots(),
            expert_registry_->pending_transfer_count(),
            expert_registry_->pending_demotion_count,
            warm_pinned_bytes,
            warm_unpinned_bytes,
            source_tier
        );
    }

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
            (expert_payload_bytes() + aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES - 1) /
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
                        expert_payload_bytes() - chunk_offset
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
