#pragma once

#include "architecture/deepseek_v4/core/config.hpp"
#include "platform/rdna3/device.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "infrastructure/backend_registry/expert_backend.hpp"
#include "infrastructure/io/direct_io_reader.hpp"
#include "infrastructure/core/expert_registry.hpp"
#include "infrastructure/core/host_expert_pool.hpp"
#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_model_contract.hpp"
#include "architecture/deepseek_v4/core/v4_expert_supply.hpp"
#include "architecture/deepseek_v4/core/v4_attention_trace.hpp"
#include "architecture/deepseek_v4/core/v4_model_resources.hpp"
#include "infrastructure/core/prefetch_staging.hpp"
#include "infrastructure/core/routing_counter.hpp"
#include "infrastructure/core/supply_telemetry.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline_scratch.hpp"
#include "backend/swizzled_w4a16/core/vram_expert_pool.hpp"
#include "architecture/deepseek_v4/kernels/hc_sinkhorn.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "architecture/deepseek_v4/kernels/moe_router.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp"
#include "infrastructure/text/text_generation.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
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

struct V4PipelineStateSnapshot {
    uint32_t current_seq_len{0};
    std::vector<V4LayerStateSnapshot> layers;
};

enum class V4PrefillExecutionPath : uint8_t {
    Batched,
    SerializedFallback,
};

struct V4PrefillResult {
    uint32_t next_token{0};
    size_t token_count{0};
    V4PrefillExecutionPath execution_path{V4PrefillExecutionPath::SerializedFallback};
};

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
    PipelineBatchScratchBuffers batch_scratch;

    hipStream_t compute_stream{0};
    hipStream_t sdma_stream{0};       // Warm Host H2D uploads
    hipStream_t sdma_cold_stream{0};  // io_uring staging -> VRAM cold uploads
    hipStream_t demotion_stream{0};   // lowest-priority Hot -> Warm D2H refills

    uint32_t num_layers_{0};
    uint32_t current_seq_len_{0};
    std::vector<V4LayerSpec> layer_specs_;

    // Unified VRAM expert pool, warm host pool, and expert registry.
    std::unique_ptr<UnifiedVRAMExpertPool> unified_vram_pool_;
    std::unique_ptr<HostExpertPool> host_pool_;
    std::unique_ptr<ExpertRegistry> expert_registry_;
    std::unique_ptr<PrefetchStagingArena> prefetch_staging_;
    std::unique_ptr<aeon::io::DirectIOReader> direct_io_reader_;
    std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> direct_io_completions_;
    uint64_t next_direct_io_id_{1};
    MemoryBudgetReport budget_report_;
    V4ExpertSupplyCoordinator expert_supply_;
    std::optional<RoutingCounter> routing_counter_;
    SupplyTelemetry supply_telemetry_;
    bool expert_timing_enabled_{false};
    std::array<ExpertTimingPhase, 2> expert_timing_{};
    std::vector<hipEvent_t> routed_section_start_events_;
    std::vector<hipEvent_t> routed_section_stop_events_;
    std::vector<hipEvent_t> expert_compute_start_events_;
    std::vector<hipEvent_t> expert_compute_stop_events_;
    bool deterministic_expert_accumulation_{false};

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

    void enable_attention_trace(uint32_t layer_id, size_t max_records) {
        if (layers.empty() || layer_id >= layers.size()) {
            throw std::out_of_range("V4Pipeline::enable_attention_trace: invalid layer");
        }
        if (max_records == 0) {
            throw std::invalid_argument("V4Pipeline::enable_attention_trace: record limit must be positive");
        }
        attention_trace_layer_ = layer_id;
        attention_trace_limit_ = max_records;
        attention_trace_records_.clear();
        attention_trace_records_.reserve(max_records);
        attention_trace_enabled_ = true;
    }

    void disable_attention_trace() noexcept {
        attention_trace_enabled_ = false;
        attention_trace_records_.clear();
    }

    const std::vector<V4AttentionTraceRecord>& attention_trace() const noexcept {
        return attention_trace_records_;
    }

    V4PipelineStateSnapshot snapshot_generation_state() const {
        if (compute_stream) CHECK_HIP(hipStreamSynchronize(compute_stream));
        V4PipelineStateSnapshot snapshot;
        snapshot.current_seq_len = current_seq_len_;
        snapshot.layers.reserve(layers.size());
        for (const auto& layer : layers) {
            snapshot.layers.push_back(layer->snapshot_state());
        }
        return snapshot;
    }

    uint32_t context_capacity() const {
        return layers.empty() ? 0 : layers.front()->max_seq_len_;
    }

    // Initialize the production runtime from model metadata and runtime policy.
    void initialize(
        const std::string& aeon_model_dir,
        const AeonRuntimeConfig& runtime_cfg
    ) {
        std::cout << "================================================================================" << std::endl;
        std::cout << "      Project Aeon — Dynamic VRAM Budget & Global Expert Pool Pipeline          " << std::endl;
        std::cout << "================================================================================" << std::endl;

        current_seq_len_ = 0;
        deterministic_expert_accumulation_ = runtime_cfg.deterministic_expert_accumulation;
        // 1. Initialize streams
        initialize_streams();

        // 2. Open Model Containers
        std::cout << "[Pipeline] Opening native .aeon model from " << aeon_model_dir << "..." << std::endl;
        aeon_loader.open_model(aeon_model_dir);
        const auto& expert_format = aeon_loader.expert_format();
        const auto& backend = ExpertBackendRegistry::resolve(expert_format);
        if (!backend.supports_v4_pipeline) {
            throw std::runtime_error(
                "V4Pipeline: selected artifact requires a different weight backend");
        }
        direct_io_reader_ = std::make_unique<aeon::io::DirectIOReader>(
            64, true, expert_format.sector_size);
        std::cout << "  > Dense tensors indexed: " << aeon_loader.total_dense_tensors() << std::endl;

        // Load the architecture contract from the model package. The runtime
        // policy controls resources; it does not redefine model dimensions.
        const auto model_cfg = DeepSeekV4Config::load_from_json(aeon_model_dir + "/config.json");
        if (model_cfg.num_hidden_layers <= 0) {
            throw std::runtime_error("V4Pipeline: model configuration must declare at least one layer");
        }
        validate_supported_model_config(model_cfg);
        layer_specs_ = V4ModelSpec::resolve_layers(model_cfg);
        V4ModelContract::validate(model_cfg, aeon_loader);
        num_layers_ = static_cast<uint32_t>(model_cfg.num_hidden_layers);

        std::array<uint32_t, 3> attention_kind_counts{};
        for (const auto& layer_spec : layer_specs_) {
            ++attention_kind_counts[static_cast<size_t>(layer_spec.attention_kind)];
        }
        std::cout << "[Pipeline] Layer contract: "
                  << attention_kind_counts[static_cast<size_t>(V4AttentionKind::Sliding)] << " Sliding, "
                  << attention_kind_counts[static_cast<size_t>(V4AttentionKind::CSA)] << " CSA, "
                  << attention_kind_counts[static_cast<size_t>(V4AttentionKind::HCA)] << " HCA" << std::endl;
        for (const uint32_t layer_id : {0u, 2u, 3u, 41u, 42u}) {
            const auto& layer_spec = layer_specs_.at(layer_id);
            std::cout << "  > Layer " << layer_id << ": "
                      << v4_attention_kind_name(layer_spec.attention_kind)
                      << " (ratio=" << layer_spec.compression_ratio << ')' << std::endl;
        }

        // 3. Evaluate Memory Budget & Feasibility Gate
        const size_t dense_bytes = aeon_loader.dense_file_size();
        budget_report_ = MemoryBudgetEngine::evaluate(
            runtime_cfg, model_cfg, dense_bytes, expert_format);
        std::cout << budget_report_.to_string() << std::endl;

        if (!budget_report_.is_feasible) {
            throw std::runtime_error("V4Pipeline: Feasibility gate REJECTED startup: " + budget_report_.rejection_reason);
        }

        // 4. Initialize model-level device resources.
        std::cout << "[Pipeline] Initializing RoPE tables and model-level weights..." << std::endl;
        model_resources_.initialize(aeon_loader, runtime_cfg.context_size, model_cfg);

        // 6. Allocate Intermediate GPU Scratch Buffers
        std::cout << "[Pipeline] Allocating intermediate GPU scratch buffers..." << std::endl;
        scratch.allocate();
        batch_scratch.allocate();

        // 7. Initialize Consecutive Transformer Layers in Global Mode (Dense weights + KV Cache)
        std::cout << "[Pipeline] Initializing " << num_layers_ << " Transformer Layers (Dense weights + KV Cache in VRAM)..." << std::endl;
        layers.resize(num_layers_);
        for (uint32_t l = 0; l < num_layers_; ++l) {
            layers[l] = std::make_unique<V4Layer>();
            layers[l]->init_with_loader(layer_specs_.at(l), aeon_loader, runtime_cfg.context_size);
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

        expert_supply_.configure(
            &aeon_loader,
            unified_vram_pool_.get(),
            host_pool_.get(),
            expert_registry_.get(),
            prefetch_staging_.get(),
            &supply_telemetry_,
            direct_io_reader_.get(),
            &direct_io_completions_,
            &next_direct_io_id_,
            compute_stream,
            sdma_stream,
            sdma_cold_stream,
            demotion_stream,
            expert_format.payload_bytes,
            runtime_cfg.enable_warm_refill
                ? V4ExpertSupplyCoordinator::DEFAULT_DEMOTION_QUEUE_CAPACITY
                : 0
        );

        std::cout << "[Pipeline] Engine ready! Configured for " << num_layers_
                  << " layers with " << budget_report_.hot_vram_slots << " hot VRAM slots and "
                  << active_warm_host_slots << " active warm host slots." << std::endl;
    }

    // Run Single Autoregressive Step for token_id at sequence position `pos`
    // Returns next token ID via greedy argmax
    uint32_t step(uint32_t token_id, uint32_t pos, RoutingPhase phase = RoutingPhase::Decode) {
        if (layers.empty() || pos >= context_capacity()) {
            throw std::out_of_range("V4Pipeline::step: position exceeds configured context capacity");
        }
        supply_telemetry_.set_phase(phase);
        reap_registry_transfers();

        constexpr int H = kernel::DSV4_HIDDEN_SIZE; // 4096
        constexpr int HC = 4;
        constexpr int HC_DIM = HC * H;             // 16384
        constexpr int HC_MULT3 = HC * (2 + HC);    // 24
        constexpr int M_PAD = 16;
        constexpr int Q_LORA = kernel::DSV4_Q_LORA_RANK;
        constexpr int HEAD_DIM = kernel::DSV4_HEAD_DIM;
        constexpr int NUM_HEADS = kernel::DSV4_NUM_HEADS;
        constexpr int TOTAL_Q = NUM_HEADS * HEAD_DIM;
        constexpr int INDEXER_Q = kernel::DSV4_INDEX_N_HEADS * kernel::DSV4_INDEX_HEAD_DIM;
        constexpr int O_LORA = kernel::DSV4_O_LORA_RANK;
        constexpr int O_GROUPS = kernel::DSV4_O_GROUPS;
        constexpr int TOT_LORA = kernel::DSV4_TOTAL_O_LORA_DIM;
        constexpr int INTER_DIM = 2048;

        // 1. Embed Token & Replicate to 4 HC streams
        const half* token_emb = model_resources_.host_embed_table + token_id * H;

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

        std::vector<uint32_t> releasable_staging_slots;
        std::vector<uint32_t> leased_experts;
        // 2. Execute Consecutive Transformer Layers
        for (uint32_t l = 0; l < num_layers_; ++l) {
            auto& layer = *layers[l];
            const bool uses_compressed_rope = layer.spec().attention_kind != V4AttentionKind::Sliding;
            const float* layer_cos_cache = uses_compressed_rope
                ? model_resources_.d_compressed_cos_cache
                : model_resources_.d_cos_cache;
            const float* layer_sin_cache = uses_compressed_rope
                ? model_resources_.d_compressed_sin_cache
                : model_resources_.d_sin_cache;

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

            if (uses_compressed_rope) {
                const int ratio = layer.spec().compression_ratio;
                const int coefficient = ratio == 4 ? 2 : 1;
                const int compressor_width = coefficient * HEAD_DIM;
                hipLaunchKernelGGL(
                    kernel::v4_gemv_fp16_kernel,
                    dim3(compressor_width, 1), dim3(32), 0, compute_stream,
                    scratch.d_x_norm, layer.d_compressor_wkv, scratch.d_compressor_kv, H
                );
                hipLaunchKernelGGL(
                    kernel::v4_gemv_fp16_kernel,
                    dim3(compressor_width, 1), dim3(32), 0, compute_stream,
                    scratch.d_x_norm, layer.d_compressor_wgate, scratch.d_compressor_score, H
                );

                if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                    hipLaunchKernelGGL(
                        kernel::v4_gemv_fp16_kernel,
                        dim3(INDEXER_Q, 1), dim3(32), 0, compute_stream,
                        scratch.d_qa_norm, layer.d_indexer_wq_b, scratch.d_indexer_query, Q_LORA
                    );
                    hipLaunchKernelGGL(
                        kernel::v4_gemv_fp16_kernel,
                        dim3(kernel::DSV4_INDEX_N_HEADS, 1), dim3(32), 0, compute_stream,
                        scratch.d_x_norm, layer.d_indexer_weights_proj, scratch.d_indexer_weights, H
                    );
                    hipLaunchKernelGGL(
                        kernel::v4_gemv_fp16_kernel,
                        dim3(coefficient * kernel::DSV4_INDEX_HEAD_DIM, 1), dim3(32), 0, compute_stream,
                        scratch.d_x_norm, layer.d_indexer_compressor_wkv,
                        scratch.d_indexer_compressor_kv, H
                    );
                    hipLaunchKernelGGL(
                        kernel::v4_gemv_fp16_kernel,
                        dim3(coefficient * kernel::DSV4_INDEX_HEAD_DIM, 1), dim3(32), 0, compute_stream,
                        scratch.d_x_norm, layer.d_indexer_compressor_wgate,
                        scratch.d_indexer_compressor_score, H
                    );
                }
            }

            V4AttentionTraceRecord* attention_trace = begin_attention_trace(
                layer, pos, scratch, HEAD_DIM, TOTAL_Q);

            // -----------------------------------------------------------------
            // D. RoPE & KV Cache Persistence
            // -----------------------------------------------------------------
            const uint32_t local_slot = pos % layer.local_cache_capacity();
            const size_t local_offset = static_cast<size_t>(local_slot) * HEAD_DIM;
            CHECK_HIP(hipMemcpyAsync(
                layer.d_local_value_cache + local_offset,
                scratch.d_kv_norm_act,
                HEAD_DIM * sizeof(half),
                hipMemcpyDeviceToDevice,
                compute_stream
            ));

            hipLaunchKernelGGL(
                kernel::v4_forward_rope_at_pos_wave32_kernel,
                dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                scratch.d_q, layer_cos_cache, layer_sin_cache, pos,
                NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2
            );

            hipLaunchKernelGGL(
                kernel::v4_forward_rope_at_pos_wave32_kernel,
                dim3(1), dim3(32), 0, compute_stream,
                scratch.d_kv_norm_act, layer_cos_cache, layer_sin_cache, pos,
                1, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2
            );

            CHECK_HIP(hipMemcpyAsync(
                layer.d_local_key_cache + local_offset,
                scratch.d_kv_norm_act,
                HEAD_DIM * sizeof(half),
                hipMemcpyDeviceToDevice,
                compute_stream
            ));
            const int64_t absolute_position = static_cast<int64_t>(pos);
            CHECK_HIP(hipMemcpyAsync(
                layer.d_local_positions + local_slot,
                &absolute_position,
                sizeof(absolute_position),
                hipMemcpyHostToDevice,
                compute_stream
            ));
            layer.record_position(pos);

            if (attention_trace != nullptr) {
                queue_trace_copy(attention_trace->rotated_query, scratch.d_q, TOTAL_Q);
                queue_trace_copy(attention_trace->rotated_local_key, scratch.d_kv_norm_act, HEAD_DIM);
                queue_trace_copy(
                    attention_trace->local_key_cache,
                    layer.d_local_key_cache,
                    static_cast<size_t>(layer.state_layout().local_capacity) * HEAD_DIM);
                queue_trace_copy(
                    attention_trace->local_value_cache,
                    layer.d_local_value_cache,
                    static_cast<size_t>(layer.state_layout().local_capacity) * HEAD_DIM);
                queue_trace_copy(
                    attention_trace->local_positions,
                    layer.d_local_positions,
                    layer.state_layout().local_capacity);
                attention_trace->local_valid_count = layer.local_valid_count_;
            }

            if (uses_compressed_rope) {
                const int ratio = layer.spec().compression_ratio;
                const int coefficient = ratio == 4 ? 2 : 1;
                const int compressor_width = coefficient * HEAD_DIM;
                const int partial_capacity = static_cast<int>(layer.state_layout().compressor_partial_capacity);
                hipLaunchKernelGGL(
                    kernel::v4_save_compressor_state_kernel,
                    dim3(1), dim3(256), 0, compute_stream,
                    scratch.d_compressor_kv, scratch.d_compressor_score,
                    layer.d_compressor_partial_kv, layer.d_compressor_partial_score,
                    layer.d_compressor_partial_positions, layer.d_compressor_ape,
                    absolute_position, ratio, partial_capacity, compressor_width
                );

                if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                    hipLaunchKernelGGL(
                        kernel::v4_forward_rope_at_pos_wave32_kernel,
                        dim3(kernel::DSV4_INDEX_N_HEADS), dim3(32), 0, compute_stream,
                        scratch.d_indexer_query, layer_cos_cache, layer_sin_cache, pos,
                        kernel::DSV4_INDEX_N_HEADS, kernel::DSV4_INDEX_HEAD_DIM,
                        kernel::DSV4_INDEX_HEAD_DIM - kernel::DSV4_ROPE_DIM,
                        kernel::DSV4_ROPE_DIM / 2
                    );
                    CHECK_HIP(hipMemcpyAsync(
                        layer.d_indexer_query,
                        scratch.d_indexer_query,
                        INDEXER_Q * sizeof(half),
                        hipMemcpyDeviceToDevice,
                        compute_stream
                    ));
                    kernel::v4_half_to_float_n_kernel<<<1, 128, 0, compute_stream>>>(
                        scratch.d_indexer_weights, layer.d_indexer_weights,
                        kernel::DSV4_INDEX_N_HEADS
                    );

                    hipLaunchKernelGGL(
                        kernel::v4_save_compressor_state_kernel,
                        dim3(1), dim3(256), 0, compute_stream,
                        scratch.d_indexer_compressor_kv, scratch.d_indexer_compressor_score,
                        layer.d_indexer_partial_kv, layer.d_indexer_partial_score,
                        layer.d_indexer_partial_positions, layer.d_indexer_compressor_ape,
                        absolute_position, ratio, partial_capacity,
                        coefficient * kernel::DSV4_INDEX_HEAD_DIM
                    );
                }

                if ((pos + 1u) % static_cast<uint32_t>(ratio) == 0) {
                    const int compressed_index = static_cast<int>(
                        (pos + 1u) / static_cast<uint32_t>(ratio) - 1u);
                    hipLaunchKernelGGL(
                        kernel::v4_materialize_compressed_entry_kernel,
                        dim3(1), dim3(512), 0, compute_stream,
                        layer.d_compressor_partial_kv, layer.d_compressor_partial_score,
                        layer.d_compressor_partial_positions, layer.d_compressor_norm,
                        layer.d_compressed_key_cache, layer.d_compressed_value_cache,
                        layer.d_compressed_positions,
                        model_resources_.d_compressed_cos_cache,
                        model_resources_.d_compressed_sin_cache,
                        absolute_position, ratio, partial_capacity, HEAD_DIM,
                        compressor_width, compressed_index,
                        kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM, 1e-6f
                    );

                    if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                        hipLaunchKernelGGL(
                            kernel::v4_materialize_compressed_entry_kernel,
                            dim3(1), dim3(512), 0, compute_stream,
                            layer.d_indexer_partial_kv, layer.d_indexer_partial_score,
                            layer.d_indexer_partial_positions, layer.d_indexer_compressor_norm,
                            layer.d_indexer_key_cache, layer.d_indexer_key_cache,
                            layer.d_indexer_positions,
                            model_resources_.d_compressed_cos_cache,
                            model_resources_.d_compressed_sin_cache,
                            absolute_position, ratio,
                            static_cast<int>(layer.state_layout().indexer_partial_capacity),
                            kernel::DSV4_INDEX_HEAD_DIM,
                            coefficient * kernel::DSV4_INDEX_HEAD_DIM,
                            compressed_index,
                            kernel::DSV4_INDEX_HEAD_DIM - kernel::DSV4_ROPE_DIM,
                            kernel::DSV4_ROPE_DIM, 1e-6f
                        );
                    }
                }

                if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                    if (layer.indexer_candidate_count_ != 0) {
                        hipLaunchKernelGGL(
                            kernel::v4_indexer_scores_kernel,
                            dim3((layer.indexer_candidate_count_ + 255u) / 256u), dim3(256), 0, compute_stream,
                            layer.d_indexer_query, layer.d_indexer_weights, layer.d_indexer_key_cache,
                            layer.d_indexer_scores, static_cast<int>(layer.indexer_candidate_count_),
                            kernel::DSV4_INDEX_N_HEADS, kernel::DSV4_INDEX_HEAD_DIM,
                            1.0f / std::sqrt(static_cast<float>(kernel::DSV4_INDEX_HEAD_DIM)),
                            1.0f / std::sqrt(static_cast<float>(kernel::DSV4_INDEX_N_HEADS))
                        );
                    }
                    select_indexer_topk(layer);
                }

                if (attention_trace != nullptr) {
                    queue_trace_copy(
                        attention_trace->compressor_partial_kv,
                        layer.d_compressor_partial_kv,
                        layer.state_layout().compressor_partial_vector_bytes() / sizeof(float));
                    queue_trace_copy(
                        attention_trace->compressor_partial_score,
                        layer.d_compressor_partial_score,
                        layer.state_layout().compressor_partial_vector_bytes() / sizeof(float));
                    queue_trace_copy(
                        attention_trace->compressor_partial_positions,
                        layer.d_compressor_partial_positions,
                        layer.state_layout().compressor_partial_capacity);
                    queue_trace_copy(
                        attention_trace->compressed_key_cache,
                        layer.d_compressed_key_cache,
                        static_cast<size_t>(layer.state_layout().compressed_capacity) * HEAD_DIM);
                    queue_trace_copy(
                        attention_trace->compressed_value_cache,
                        layer.d_compressed_value_cache,
                        static_cast<size_t>(layer.state_layout().compressed_capacity) * HEAD_DIM);
                    queue_trace_copy(
                        attention_trace->compressed_positions,
                        layer.d_compressed_positions,
                        layer.state_layout().compressed_capacity);
                    attention_trace->compressor_partial_count = layer.compressor_partial_count_;
                    attention_trace->compressed_entry_count = layer.compressed_entry_count_;
                    attention_trace->indexer_candidate_count = layer.indexer_candidate_count_;

                    if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                        queue_trace_copy(
                            attention_trace->indexer_partial_kv,
                            layer.d_indexer_partial_kv,
                            layer.state_layout().indexer_partial_vector_bytes() / sizeof(float));
                        queue_trace_copy(
                            attention_trace->indexer_partial_score,
                            layer.d_indexer_partial_score,
                            layer.state_layout().indexer_partial_vector_bytes() / sizeof(float));
                        queue_trace_copy(
                            attention_trace->indexer_partial_positions,
                            layer.d_indexer_partial_positions,
                            layer.state_layout().indexer_partial_capacity);
                        queue_trace_copy(
                            attention_trace->indexer_key_cache,
                            layer.d_indexer_key_cache,
                            static_cast<size_t>(layer.state_layout().compressed_capacity) *
                                layer.state_layout().index_head_dim);
                        queue_trace_copy(
                            attention_trace->indexer_positions,
                            layer.d_indexer_positions,
                            layer.state_layout().compressed_capacity);
                        queue_trace_copy(
                            attention_trace->indexer_scores,
                            layer.d_indexer_scores,
                            layer.indexer_candidate_count_);
                        queue_trace_copy(
                            attention_trace->indexer_topk_indices,
                            layer.d_indexer_topk_indices,
                            layer.state_layout().index_topk);
                    }
                }
            }

            // -----------------------------------------------------------------
            // E. Class-specific serial attention over cached states
            // -----------------------------------------------------------------
            if (layer.spec().attention_kind == V4AttentionKind::Sliding) {
                hipLaunchKernelGGL(
                    kernel::v4_cached_sliding_window_attn_wave32_kernel,
                    dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                    scratch.d_q, layer.d_local_key_cache, layer.d_local_value_cache,
                    layer.d_local_positions, layer.d_attn_sink, scratch.d_attn_out,
                    pos, static_cast<int>(layer.local_cache_capacity()), kernel::DSV4_ATTN_SCALE
                );
            } else {
                const bool uses_indexer = layer.spec().attention_kind == V4AttentionKind::CSA;
                const int compressed_count = static_cast<int>(layer.compressed_entry_count_);
                const int topk_count = uses_indexer
                    ? std::min<int>(compressed_count, static_cast<int>(layer.state_layout().index_topk))
                    : 0;
                hipLaunchKernelGGL(
                    kernel::v4_cached_compressed_attention_wave32_kernel,
                    dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                    scratch.d_q, layer.d_local_key_cache, layer.d_local_value_cache,
                    layer.d_local_positions, layer.d_attn_sink,
                    layer.d_compressed_key_cache, layer.d_compressed_value_cache,
                    layer.d_compressed_positions,
                    uses_indexer ? layer.d_indexer_topk_indices : nullptr,
                    scratch.d_attn_out, absolute_position,
                    static_cast<int>(layer.local_cache_capacity()), compressed_count,
                    topk_count, uses_indexer, kernel::DSV4_ATTN_SCALE
                );
            }

            if (attention_trace != nullptr) {
                queue_trace_copy(attention_trace->attention_output, scratch.d_attn_out, TOTAL_Q);
            }

            // Inverse RoPE on attention output
            hipLaunchKernelGGL(
                kernel::v4_inverse_rope_at_pos_wave32_kernel,
                dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                scratch.d_attn_out, layer_cos_cache, layer_sin_cache, pos,
                NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2
            );

            if (attention_trace != nullptr) {
                queue_trace_copy(attention_trace->inverse_rope_output, scratch.d_attn_out, TOTAL_Q);
            }

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

            if (attention_trace != nullptr) {
                queue_trace_copy(attention_trace->grouped_output, scratch.d_attn_proj, H);
            }

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
            auto active_prefetch = expert_supply_.dispatch_layer_prefetch(
                l, pos, h_topk_indices, leased_experts);

            // Let cold NVMe reads overlap the shared expert pass; materialization
            // still completes before the first routed expert consumes each slot.
            expert_supply_.materialize_layer_prefetch(active_prefetch);

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
            CHECK_HIP(hipMemsetAsync(
                scratch.d_swizzled_counters,
                0,
                64 * sizeof(int32_t),
                compute_stream));
            if (deterministic_expert_accumulation_) {
                for (int expert = 0; expert < 6; ++expert) {
                    kernel::dispatch_aeon_w4a16_swizzled_gemv<8, 8, 4, 16>(
                        scratch.d_swizzled_expert_hidden + static_cast<size_t>(expert) * INTER_DIM,
                        reinterpret_cast<const uint32_t*>(fused_w2.w2[expert]),
                        fused_w2.s2[expert],
                        scratch.d_expert_down,
                        H,
                        INTER_DIM,
                        compute_stream);
                    hipLaunchKernelGGL(
                        kernel::v4_pipeline_accumulate_expert_kernel,
                        dim3((H + 255) / 256), dim3(256), 0, compute_stream,
                        scratch.d_moe_accum,
                        scratch.d_expert_down,
                        h_topk_weights[expert],
                        H
                    );
                }
            } else {
                kernel::dispatch_aeon_moe_fused_w2_accum<8, 8, 4, 16>(
                    scratch.d_swizzled_expert_hidden,
                    fused_w2,
                    scratch.d_topk_weights,
                    scratch.d_moe_accum,
                    scratch.d_swizzled_moe_accum_f32,
                    scratch.d_moe_accum,
                    scratch.d_swizzled_counters,
                    6, H, INTER_DIM, compute_stream);
            }

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
            scratch.d_res_in,
            model_resources_.d_hc_head_fn,
            model_resources_.d_hc_head_base,
            model_resources_.d_hc_head_scale,
            scratch.d_hc_head_out, H, HC, 1e-6f, 1e-6f
        );

        // 4. Final RMSNorm
        hipLaunchKernelGGL(
            kernel::v4_rmsnorm_wave32_kernel,
            dim3(1), dim3(32), 0, compute_stream,
            scratch.d_hc_head_out, model_resources_.d_final_norm, scratch.d_head_norm, H, 1e-6f
        );

        // 5. LM Head Projection: logits = head_norm @ lm_head.T [129280]
        hipLaunchKernelGGL(
            kernel::v4_gemv_fp16_vec8_kernel,
            dim3(129280, 1), dim3(32), 0, compute_stream,
            scratch.d_head_norm, model_resources_.d_lm_head, scratch.d_logits, H
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
            l->reset_generation_state();
        }
    }

    uint32_t prefill(std::span<const uint32_t> token_ids, uint32_t start_position = 0) {
        validate_prefill_span(token_ids, start_position);

        uint32_t next_token = 0;
        for (size_t index = 0; index < token_ids.size(); ++index) {
            next_token = step(
                token_ids[index],
                start_position + static_cast<uint32_t>(index),
                RoutingPhase::Prefill);
        }
        return next_token;
    }

    V4PrefillResult prefill_batched(
        std::span<const uint32_t> token_ids,
        uint32_t start_position = 0,
        size_t requested_batch_size = 16,
        bool allow_serialized_fallback = true
    ) {
        validate_prefill_span(token_ids, start_position);
        if (requested_batch_size == 0) {
            throw std::invalid_argument(
                "V4Pipeline::prefill_batched: requested batch size must be greater than zero");
        }
        if (requested_batch_size > 1 && token_ids.size() > 1) {
            const size_t batch_size = std::min(
                requested_batch_size,
                static_cast<size_t>(PipelineBatchScratchBuffers::kMaxTokens));
            uint32_t next_token = 0;
            for (size_t offset = 0; offset < token_ids.size(); offset += batch_size) {
                const size_t count = std::min(batch_size, token_ids.size() - offset);
                next_token = prefill_batched_chunk(
                    token_ids.subspan(offset, count),
                    start_position + static_cast<uint32_t>(offset));
            }
            return V4PrefillResult{
                next_token,
                token_ids.size(),
                V4PrefillExecutionPath::Batched
            };
        }
        if (!allow_serialized_fallback) {
            throw std::runtime_error(
                "V4Pipeline::prefill_batched: true batched execution is not available for the current V4 state path");
        }

        uint32_t next_token = 0;
        for (size_t offset = 0; offset < token_ids.size(); offset += requested_batch_size) {
            const size_t count = std::min(requested_batch_size, token_ids.size() - offset);
            next_token = prefill(
                token_ids.subspan(offset, count),
                start_position + static_cast<uint32_t>(offset));
        }
        return V4PrefillResult{
            next_token,
            token_ids.size(),
            V4PrefillExecutionPath::SerializedFallback
        };
    }

    // Prefill Prompt and Generate Next Tokens
    std::vector<uint32_t> generate(
        const std::vector<uint32_t>& prompt,
        uint32_t max_new_tokens,
        double* out_ttft_ms = nullptr,
        double* out_tok_per_sec = nullptr
    ) {
        if (prompt.empty()) return {};

        std::vector<uint32_t> generated;
        reset_generation_state();

        // Prefill Phase
        auto t_prefill_start = std::chrono::high_resolution_clock::now();
        uint32_t next_tok = prefill(std::span<const uint32_t>(prompt), 0);
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
        model_resources_.free();

        batch_scratch.free();
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
        expert_supply_.clear();
        supply_telemetry_.disable();
        aeon_loader.close_all();
    }

private:
    uint32_t prefill_batched_chunk(
        std::span<const uint32_t> token_ids,
        uint32_t start_position
    ) {
        if (token_ids.empty() || token_ids.size() > PipelineBatchScratchBuffers::kMaxTokens) {
            throw std::invalid_argument("V4Pipeline::prefill_batched_chunk: unsupported token count");
        }

        constexpr int H = kernel::DSV4_HIDDEN_SIZE;
        constexpr int HC = 4;
        constexpr int HC_DIM = HC * H;
        constexpr int HC_MULT3 = HC * (2 + HC);
        constexpr int Q_LORA = kernel::DSV4_Q_LORA_RANK;
        constexpr int HEAD_DIM = kernel::DSV4_HEAD_DIM;
        constexpr int NUM_HEADS = kernel::DSV4_NUM_HEADS;
        constexpr int TOTAL_Q = NUM_HEADS * HEAD_DIM;
        constexpr int INDEXER_Q = kernel::DSV4_INDEX_N_HEADS * kernel::DSV4_INDEX_HEAD_DIM;
        constexpr int O_LORA = kernel::DSV4_O_LORA_RANK;
        constexpr int O_GROUPS = kernel::DSV4_O_GROUPS;
        constexpr int TOT_LORA = kernel::DSV4_TOTAL_O_LORA_DIM;
        constexpr int INTER_DIM = 2048;
        constexpr int ROUTER_EXPERTS = 256;
        constexpr int ROUTED_EXPERTS = 6;
        const int batch_count = static_cast<int>(token_ids.size());

        validate_prefill_span(token_ids, start_position);
        supply_telemetry_.set_phase(RoutingPhase::Prefill);
        reap_registry_transfers();

        for (int token = 0; token < batch_count; ++token) {
            const half* token_embedding = model_resources_.host_embed_table +
                static_cast<size_t>(token_ids[static_cast<size_t>(token)]) * H;
            for (int stream = 0; stream < HC; ++stream) {
                CHECK_HIP(hipMemcpyAsync(
                    batch_scratch.d_res_in_half +
                        static_cast<size_t>(token) * HC_DIM + stream * H,
                    token_embedding,
                    H * sizeof(half),
                    hipMemcpyHostToDevice,
                    compute_stream));
            }
            const int32_t token_id = static_cast<int32_t>(token_ids[static_cast<size_t>(token)]);
            CHECK_HIP(hipMemcpyAsync(
                batch_scratch.d_token_ids + token,
                &token_id,
                sizeof(token_id),
                hipMemcpyHostToDevice,
                compute_stream));
        }
        kernel::v4_half_to_float_kernel<<<
            (batch_count * HC_DIM + 255) / 256, 256, 0, compute_stream>>>(
                batch_scratch.d_res_in_half,
                batch_scratch.d_res_in,
                batch_count * HC_DIM);

        for (uint32_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
            auto& layer = *layers[layer_index];
            const bool uses_compressed_rope = layer.spec().attention_kind != V4AttentionKind::Sliding;
            const float* layer_cos_cache = uses_compressed_rope
                ? model_resources_.d_compressed_cos_cache
                : model_resources_.d_cos_cache;
            const float* layer_sin_cache = uses_compressed_rope
                ? model_resources_.d_compressed_sin_cache
                : model_resources_.d_sin_cache;

            hipLaunchKernelGGL(
                kernel::hc_project_batched_kernel,
                dim3(HC_MULT3, batch_count), dim3(256), 0, compute_stream,
                batch_scratch.d_res_in, layer.d_hc_attn_fn, batch_scratch.d_mixes_a,
                H, HC, 1e-6f);
            hipLaunchKernelGGL(
                kernel::hc_sinkhorn_normalize_kernel,
                dim3(batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_mixes_a, layer.d_hc_attn_scale, layer.d_hc_attn_base,
                batch_scratch.d_pre_a, batch_scratch.d_post_a, batch_scratch.d_comb_a,
                1e-6f, 1e-6f, 2.0f, 20);
            hipLaunchKernelGGL(
                kernel::hc_pre_combine_batched_kernel,
                dim3((H / 4 + 255) / 256, batch_count), dim3(256), 0, compute_stream,
                batch_scratch.d_res_in, batch_scratch.d_pre_a, batch_scratch.d_x_pre,
                H, HC);
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_x_pre, layer.d_attn_norm, batch_scratch.d_x_norm, H, 1e-6f);

            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(Q_LORA, batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_x_norm, layer.d_wq_a, batch_scratch.d_qa, H);
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_qa, layer.d_q_norm, batch_scratch.d_qa_norm, Q_LORA, 1e-6f);
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(TOTAL_Q, batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_qa_norm, layer.d_wq_b, batch_scratch.d_q, Q_LORA);
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_unit_wave32_kernel,
                dim3(batch_count * NUM_HEADS), dim3(32), 0, compute_stream,
                batch_scratch.d_q, batch_scratch.d_q, HEAD_DIM, 1e-6f);
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(HEAD_DIM, batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_x_norm, layer.d_wkv, batch_scratch.d_kv, H);
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_kv, layer.d_kv_norm, batch_scratch.d_kv_norm_act,
                HEAD_DIM, 1e-6f);

            if (uses_compressed_rope) {
                const int ratio = layer.spec().compression_ratio;
                const int coefficient = ratio == 4 ? 2 : 1;
                const int compressor_width = coefficient * HEAD_DIM;
                hipLaunchKernelGGL(
                    kernel::v4_gemv_fp16_kernel,
                    dim3(compressor_width, batch_count), dim3(32), 0, compute_stream,
                    batch_scratch.d_x_norm, layer.d_compressor_wkv,
                    batch_scratch.d_compressor_kv, H);
                hipLaunchKernelGGL(
                    kernel::v4_gemv_fp16_kernel,
                    dim3(compressor_width, batch_count), dim3(32), 0, compute_stream,
                    batch_scratch.d_x_norm, layer.d_compressor_wgate,
                    batch_scratch.d_compressor_score, H);

                if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                    hipLaunchKernelGGL(
                        kernel::v4_gemv_fp16_kernel,
                        dim3(INDEXER_Q, batch_count), dim3(32), 0, compute_stream,
                        batch_scratch.d_qa_norm, layer.d_indexer_wq_b,
                        batch_scratch.d_indexer_query, Q_LORA);
                    hipLaunchKernelGGL(
                        kernel::v4_gemv_fp16_kernel,
                        dim3(kernel::DSV4_INDEX_N_HEADS, batch_count), dim3(32), 0, compute_stream,
                        batch_scratch.d_x_norm, layer.d_indexer_weights_proj,
                        batch_scratch.d_indexer_weights, H);
                    hipLaunchKernelGGL(
                        kernel::v4_gemv_fp16_kernel,
                        dim3(2 * kernel::DSV4_INDEX_HEAD_DIM, batch_count), dim3(32), 0, compute_stream,
                        batch_scratch.d_x_norm, layer.d_indexer_compressor_wkv,
                        batch_scratch.d_indexer_compressor_kv, H);
                    hipLaunchKernelGGL(
                        kernel::v4_gemv_fp16_kernel,
                        dim3(2 * kernel::DSV4_INDEX_HEAD_DIM, batch_count), dim3(32), 0, compute_stream,
                        batch_scratch.d_x_norm, layer.d_indexer_compressor_wgate,
                        batch_scratch.d_indexer_compressor_score, H);
                }
            }

            for (int token = 0; token < batch_count; ++token) {
                const uint32_t position = start_position + static_cast<uint32_t>(token);
                half* query = batch_scratch.d_q + static_cast<size_t>(token) * TOTAL_Q;
                half* value = batch_scratch.d_kv_norm_act + static_cast<size_t>(token) * HEAD_DIM;
                half* rotated_value = batch_scratch.d_kv_rotated + static_cast<size_t>(token) * HEAD_DIM;
                CHECK_HIP(hipMemcpyAsync(
                    rotated_value, value, HEAD_DIM * sizeof(half),
                    hipMemcpyDeviceToDevice, compute_stream));
                hipLaunchKernelGGL(
                    kernel::v4_forward_rope_at_pos_wave32_kernel,
                    dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                    query, layer_cos_cache, layer_sin_cache, position,
                    NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2);
                hipLaunchKernelGGL(
                    kernel::v4_forward_rope_at_pos_wave32_kernel,
                    dim3(1), dim3(32), 0, compute_stream,
                    rotated_value, layer_cos_cache, layer_sin_cache, position,
                    1, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2);

                const uint32_t local_slot = position % layer.local_cache_capacity();
                const size_t local_offset = static_cast<size_t>(local_slot) * HEAD_DIM;
                CHECK_HIP(hipMemcpyAsync(
                    layer.d_local_value_cache + local_offset,
                    value,
                    HEAD_DIM * sizeof(half),
                    hipMemcpyDeviceToDevice,
                    compute_stream));
                CHECK_HIP(hipMemcpyAsync(
                    layer.d_local_key_cache + local_offset,
                    rotated_value,
                    HEAD_DIM * sizeof(half),
                    hipMemcpyDeviceToDevice,
                    compute_stream));
                const int64_t absolute_position = static_cast<int64_t>(position);
                CHECK_HIP(hipMemcpyAsync(
                    layer.d_local_positions + local_slot,
                    &absolute_position,
                    sizeof(absolute_position),
                    hipMemcpyHostToDevice,
                    compute_stream));
                layer.record_position(position);

                if (uses_compressed_rope) {
                    const int ratio = layer.spec().compression_ratio;
                    const int coefficient = ratio == 4 ? 2 : 1;
                    const int compressor_width = coefficient * HEAD_DIM;
                    hipLaunchKernelGGL(
                        kernel::v4_save_compressor_state_kernel,
                        dim3(1), dim3(256), 0, compute_stream,
                        batch_scratch.d_compressor_kv +
                            static_cast<size_t>(token) * compressor_width,
                        batch_scratch.d_compressor_score +
                            static_cast<size_t>(token) * compressor_width,
                        layer.d_compressor_partial_kv, layer.d_compressor_partial_score,
                        layer.d_compressor_partial_positions, layer.d_compressor_ape,
                        static_cast<int64_t>(position), ratio,
                        static_cast<int>(layer.state_layout().compressor_partial_capacity),
                        compressor_width);

                    if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                        hipLaunchKernelGGL(
                            kernel::v4_forward_rope_at_pos_wave32_kernel,
                            dim3(kernel::DSV4_INDEX_N_HEADS), dim3(32), 0, compute_stream,
                            batch_scratch.d_indexer_query +
                                static_cast<size_t>(token) * INDEXER_Q,
                            layer_cos_cache, layer_sin_cache, position,
                            kernel::DSV4_INDEX_N_HEADS, kernel::DSV4_INDEX_HEAD_DIM,
                            kernel::DSV4_INDEX_HEAD_DIM - kernel::DSV4_ROPE_DIM,
                            kernel::DSV4_ROPE_DIM / 2);
                        CHECK_HIP(hipMemcpyAsync(
                            layer.d_indexer_query,
                            batch_scratch.d_indexer_query +
                                static_cast<size_t>(token) * INDEXER_Q,
                            INDEXER_Q * sizeof(half),
                            hipMemcpyDeviceToDevice,
                            compute_stream));
                        kernel::v4_half_to_float_n_kernel<<<1, 128, 0, compute_stream>>>(
                            batch_scratch.d_indexer_weights + static_cast<size_t>(token) * kernel::DSV4_INDEX_N_HEADS,
                            layer.d_indexer_weights,
                            kernel::DSV4_INDEX_N_HEADS);
                        hipLaunchKernelGGL(
                            kernel::v4_save_compressor_state_kernel,
                            dim3(1), dim3(256), 0, compute_stream,
                            batch_scratch.d_indexer_compressor_kv +
                                static_cast<size_t>(token) * coefficient * kernel::DSV4_INDEX_HEAD_DIM,
                            batch_scratch.d_indexer_compressor_score +
                                static_cast<size_t>(token) * coefficient * kernel::DSV4_INDEX_HEAD_DIM,
                            layer.d_indexer_partial_kv, layer.d_indexer_partial_score,
                            layer.d_indexer_partial_positions, layer.d_indexer_compressor_ape,
                            static_cast<int64_t>(position), ratio,
                            static_cast<int>(layer.state_layout().indexer_partial_capacity),
                            coefficient * kernel::DSV4_INDEX_HEAD_DIM);
                    }

                    if ((position + 1u) % static_cast<uint32_t>(ratio) == 0) {
                        const int compressed_index = static_cast<int>(
                            (position + 1u) / static_cast<uint32_t>(ratio) - 1u);
                        hipLaunchKernelGGL(
                            kernel::v4_materialize_compressed_entry_kernel,
                            dim3(1), dim3(512), 0, compute_stream,
                            layer.d_compressor_partial_kv, layer.d_compressor_partial_score,
                            layer.d_compressor_partial_positions, layer.d_compressor_norm,
                            layer.d_compressed_key_cache, layer.d_compressed_value_cache,
                            layer.d_compressed_positions,
                            model_resources_.d_compressed_cos_cache,
                            model_resources_.d_compressed_sin_cache,
                            static_cast<int64_t>(position), ratio,
                            static_cast<int>(layer.state_layout().compressor_partial_capacity),
                            HEAD_DIM, compressor_width, compressed_index,
                            kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM, 1e-6f);
                        if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                            hipLaunchKernelGGL(
                                kernel::v4_materialize_compressed_entry_kernel,
                                dim3(1), dim3(512), 0, compute_stream,
                                layer.d_indexer_partial_kv, layer.d_indexer_partial_score,
                                layer.d_indexer_partial_positions, layer.d_indexer_compressor_norm,
                                layer.d_indexer_key_cache, layer.d_indexer_key_cache,
                                layer.d_indexer_positions,
                                model_resources_.d_compressed_cos_cache,
                                model_resources_.d_compressed_sin_cache,
                                static_cast<int64_t>(position), ratio,
                                static_cast<int>(layer.state_layout().indexer_partial_capacity),
                                kernel::DSV4_INDEX_HEAD_DIM,
                                coefficient * kernel::DSV4_INDEX_HEAD_DIM,
                                compressed_index,
                                kernel::DSV4_INDEX_HEAD_DIM - kernel::DSV4_ROPE_DIM,
                                kernel::DSV4_ROPE_DIM, 1e-6f);
                        }
                    }

                    if (layer.spec().attention_kind == V4AttentionKind::CSA) {
                        if (layer.indexer_candidate_count_ != 0) {
                            hipLaunchKernelGGL(
                                kernel::v4_indexer_scores_kernel,
                                dim3((layer.indexer_candidate_count_ + 255u) / 256u),
                                dim3(256), 0, compute_stream,
                                layer.d_indexer_query, layer.d_indexer_weights,
                                layer.d_indexer_key_cache, layer.d_indexer_scores,
                                static_cast<int>(layer.indexer_candidate_count_),
                                kernel::DSV4_INDEX_N_HEADS, kernel::DSV4_INDEX_HEAD_DIM,
                                1.0f / std::sqrt(static_cast<float>(kernel::DSV4_INDEX_HEAD_DIM)),
                                1.0f / std::sqrt(static_cast<float>(kernel::DSV4_INDEX_N_HEADS)));
                        }
                        select_indexer_topk(layer);
                    }
                }

                if (layer.spec().attention_kind == V4AttentionKind::Sliding) {
                    hipLaunchKernelGGL(
                        kernel::v4_cached_sliding_window_attn_wave32_kernel,
                        dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                        query, layer.d_local_key_cache, layer.d_local_value_cache,
                        layer.d_local_positions, layer.d_attn_sink,
                        batch_scratch.d_attn_out + static_cast<size_t>(token) * TOTAL_Q,
                        position, static_cast<int>(layer.local_cache_capacity()),
                        kernel::DSV4_ATTN_SCALE);
                } else {
                    const bool uses_indexer = layer.spec().attention_kind == V4AttentionKind::CSA;
                    const int compressed_count = static_cast<int>(layer.compressed_entry_count_);
                    const int topk_count = uses_indexer
                        ? std::min<int>(compressed_count, static_cast<int>(layer.state_layout().index_topk))
                        : 0;
                    hipLaunchKernelGGL(
                        kernel::v4_cached_compressed_attention_wave32_kernel,
                        dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                        query, layer.d_local_key_cache, layer.d_local_value_cache,
                        layer.d_local_positions, layer.d_attn_sink,
                        layer.d_compressed_key_cache, layer.d_compressed_value_cache,
                        layer.d_compressed_positions,
                        uses_indexer ? layer.d_indexer_topk_indices : nullptr,
                        batch_scratch.d_attn_out + static_cast<size_t>(token) * TOTAL_Q,
                        static_cast<int64_t>(position),
                        static_cast<int>(layer.local_cache_capacity()), compressed_count,
                        topk_count, uses_indexer, kernel::DSV4_ATTN_SCALE);
                }
                hipLaunchKernelGGL(
                    kernel::v4_inverse_rope_at_pos_wave32_kernel,
                    dim3(NUM_HEADS), dim3(32), 0, compute_stream,
                    batch_scratch.d_attn_out + static_cast<size_t>(token) * TOTAL_Q,
                    layer_cos_cache, layer_sin_cache, position,
                    NUM_HEADS, HEAD_DIM, kernel::DSV4_NOPE_DIM, kernel::DSV4_ROPE_DIM / 2);
            }

            hipLaunchKernelGGL(
                kernel::v4_grouped_wo_a_wave32_kernel,
                dim3(O_LORA, O_GROUPS, batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_attn_out, layer.d_wo_a, batch_scratch.d_z_lora, batch_count);
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_kernel,
                dim3(H, batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_z_lora, layer.d_wo_b, batch_scratch.d_attn_proj, TOT_LORA);

            kernel::v4_float_to_half_kernel<<<
                (batch_count * HC_DIM + 255) / 256, 256, 0, compute_stream>>>(
                    batch_scratch.d_res_in,
                    batch_scratch.d_res_in_half,
                    batch_count * HC_DIM);
            hipLaunchKernelGGL(
                kernel::hc_post_kernel,
                dim3((H + 255) / 256, batch_count), dim3(256), 0, compute_stream,
                batch_scratch.d_attn_proj, batch_scratch.d_res_in_half,
                batch_scratch.d_post_a, batch_scratch.d_comb_a,
                batch_scratch.d_res_mid_half, H);
            kernel::v4_half_to_float_kernel<<<
                (batch_count * HC_DIM + 255) / 256, 256, 0, compute_stream>>>(
                    batch_scratch.d_res_mid_half,
                    batch_scratch.d_res_mid,
                    batch_count * HC_DIM);

            hipLaunchKernelGGL(
                kernel::hc_project_batched_kernel,
                dim3(HC_MULT3, batch_count), dim3(256), 0, compute_stream,
                batch_scratch.d_res_mid, layer.d_hc_ffn_fn, batch_scratch.d_mixes_f,
                H, HC, 1e-6f);
            hipLaunchKernelGGL(
                kernel::hc_sinkhorn_normalize_kernel,
                dim3(batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_mixes_f, layer.d_hc_ffn_scale, layer.d_hc_ffn_base,
                batch_scratch.d_pre_f, batch_scratch.d_post_f, batch_scratch.d_comb_f,
                1e-6f, 1e-6f, 2.0f, 20);
            hipLaunchKernelGGL(
                kernel::hc_pre_combine_batched_kernel,
                dim3((H / 4 + 255) / 256, batch_count), dim3(256), 0, compute_stream,
                batch_scratch.d_res_mid, batch_scratch.d_pre_f, batch_scratch.d_ffn_pre,
                H, HC);
            hipLaunchKernelGGL(
                kernel::v4_rmsnorm_wave32_kernel,
                dim3(batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_ffn_pre, layer.d_ffn_norm,
                batch_scratch.d_ffn_norm_act, H, 1e-6f);

            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_vec8_kernel,
                dim3(INTER_DIM, batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_ffn_norm_act, layer.d_shared_w1,
                batch_scratch.d_shared_gate, H);
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_vec8_kernel,
                dim3(INTER_DIM, batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_ffn_norm_act, layer.d_shared_w3,
                batch_scratch.d_shared_up, H);
            kernel::v4_pipeline_swiglu_clamp_batched_kernel<<<
                (batch_count * INTER_DIM + 255) / 256, 256, 0, compute_stream>>>(
                batch_scratch.d_shared_gate,
                batch_scratch.d_shared_up,
                batch_scratch.d_shared_swiglu,
                batch_count * INTER_DIM,
                10.0f);
            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_vec8_kernel,
                dim3(H, batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_shared_swiglu, layer.d_shared_w2,
                batch_scratch.d_moe_accum, INTER_DIM);

            hipLaunchKernelGGL(
                kernel::v4_gemv_fp16_vec8_kernel,
                dim3(ROUTER_EXPERTS, batch_count), dim3(32), 0, compute_stream,
                batch_scratch.d_ffn_norm_act, layer.d_gate_weight,
                batch_scratch.d_router_logits_half, H);
            kernel::v4_half_to_float_n_kernel<<<
                (batch_count * ROUTER_EXPERTS + 255) / 256, 256, 0, compute_stream>>>(
                batch_scratch.d_router_logits_half,
                batch_scratch.d_router_logits,
                batch_count * ROUTER_EXPERTS);
            hipLaunchKernelGGL(
                kernel::moe_router_kernel,
                dim3(batch_count), dim3(64), 0, compute_stream,
                batch_scratch.d_router_logits,
                layer.is_hash_layer ? nullptr : layer.d_gate_bias,
                layer.d_tid2eid,
                batch_scratch.d_token_ids,
                batch_scratch.d_topk_weights,
                batch_scratch.d_topk_indices,
                ROUTER_EXPERTS, ROUTED_EXPERTS, 1.5f, true);

            std::vector<float> host_topk_weights(static_cast<size_t>(batch_count) * ROUTED_EXPERTS);
            std::vector<int32_t> host_topk_indices(static_cast<size_t>(batch_count) * ROUTED_EXPERTS);
            CHECK_HIP(hipMemcpyAsync(
                host_topk_weights.data(), batch_scratch.d_topk_weights,
                host_topk_weights.size() * sizeof(float), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipMemcpyAsync(
                host_topk_indices.data(), batch_scratch.d_topk_indices,
                host_topk_indices.size() * sizeof(int32_t), hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            for (int token = 0; token < batch_count; ++token) {
                std::vector<int32_t> topk_indices(
                    host_topk_indices.begin() + static_cast<size_t>(token) * ROUTED_EXPERTS,
                    host_topk_indices.begin() + static_cast<size_t>(token + 1) * ROUTED_EXPERTS);
                std::vector<float> topk_weights(
                    host_topk_weights.begin() + static_cast<size_t>(token) * ROUTED_EXPERTS,
                    host_topk_weights.begin() + static_cast<size_t>(token + 1) * ROUTED_EXPERTS);
                std::vector<uint32_t> leased_experts;
                auto active_prefetch = expert_supply_.dispatch_layer_prefetch(
                    layer_index, start_position + static_cast<uint32_t>(token),
                    topk_indices, leased_experts);
                expert_supply_.materialize_layer_prefetch(active_prefetch);

                kernel::SwizzledW13ExpertPtrs fused_w13{};
                kernel::SwizzledW2ExpertPtrs fused_w2{};
                for (int expert = 0; expert < ROUTED_EXPERTS; ++expert) {
                    if (active_prefetch.is_prefetched[expert] && prefetch_staging_) {
                        expert_supply_.mark_gpu_readiness_wait_start(
                            active_prefetch.operation_ids[expert]);
                        CHECK_HIP(hipStreamWaitEvent(
                            compute_stream,
                            prefetch_staging_->events[active_prefetch.staging_indices[expert]],
                            0));
                    }
                    const int32_t slot = active_prefetch.vram_slots[expert];
                    fused_w13.w1[expert] = reinterpret_cast<const uint4*>(
                        unified_vram_pool_->get_w1_packed(slot));
                    fused_w13.s1[expert] = unified_vram_pool_->get_w1_scale(slot);
                    fused_w13.w3[expert] = reinterpret_cast<const uint4*>(
                        unified_vram_pool_->get_w3_packed(slot));
                    fused_w13.s3[expert] = unified_vram_pool_->get_w3_scale(slot);
                    fused_w2.w2[expert] = reinterpret_cast<const uint4*>(
                        unified_vram_pool_->get_w2_packed(slot));
                    fused_w2.s2[expert] = unified_vram_pool_->get_w2_scale(slot);
                }

                layer.cache_hits = expert_registry_->hits_hot;
                layer.cache_misses = expert_registry_->hits_warm + expert_registry_->misses_cold;
                kernel::dispatch_aeon_moe_fused_w13_swiglu<8, 4, 8, 16>(
                    batch_scratch.d_ffn_norm_act + static_cast<size_t>(token) * H,
                    fused_w13,
                    scratch.d_swizzled_expert_hidden,
                    scratch.d_swizzled_moe_accum_f32,
                    H, ROUTED_EXPERTS, INTER_DIM, H, 10.0f, compute_stream);
                CHECK_HIP(hipMemsetAsync(
                    scratch.d_swizzled_counters, 0, 64 * sizeof(int32_t), compute_stream));
                if (deterministic_expert_accumulation_) {
                    for (int expert = 0; expert < ROUTED_EXPERTS; ++expert) {
                        kernel::dispatch_aeon_w4a16_swizzled_gemv<8, 8, 4, 16>(
                            scratch.d_swizzled_expert_hidden + static_cast<size_t>(expert) * INTER_DIM,
                            reinterpret_cast<const uint32_t*>(fused_w2.w2[expert]),
                            fused_w2.s2[expert],
                            scratch.d_expert_down,
                            H, INTER_DIM, compute_stream);
                        hipLaunchKernelGGL(
                            kernel::v4_pipeline_accumulate_expert_kernel,
                            dim3((H + 255) / 256), dim3(256), 0, compute_stream,
                            batch_scratch.d_moe_accum + static_cast<size_t>(token) * H,
                            scratch.d_expert_down,
                            topk_weights[expert], H);
                    }
                } else {
                    kernel::dispatch_aeon_moe_fused_w2_accum<8, 8, 4, 16>(
                        scratch.d_swizzled_expert_hidden,
                        fused_w2,
                        batch_scratch.d_topk_weights + static_cast<size_t>(token) * ROUTED_EXPERTS,
                        batch_scratch.d_moe_accum + static_cast<size_t>(token) * H,
                        scratch.d_swizzled_moe_accum_f32,
                        batch_scratch.d_moe_accum + static_cast<size_t>(token) * H,
                        scratch.d_swizzled_counters,
                        ROUTED_EXPERTS, H, INTER_DIM, compute_stream);
                }

                CHECK_HIP(hipStreamSynchronize(compute_stream));
                for (int expert = 0; expert < ROUTED_EXPERTS; ++expert) {
                    if (active_prefetch.is_prefetched[expert] && prefetch_staging_) {
                        prefetch_staging_->release_after_gpu_transfer(
                            active_prefetch.staging_indices[expert]);
                    }
                }
                for (uint32_t global_expert : leased_experts) {
                    expert_registry_->release_lease(global_expert);
                }
                reap_registry_transfers();
                if (routing_counter_) {
                    routing_counter_->record(
                        RoutingPhase::Prefill,
                        layer_index,
                        start_position + static_cast<uint32_t>(token),
                        topk_indices.data());
                }
            }

            hipLaunchKernelGGL(
                kernel::hc_post_kernel,
                dim3((H + 255) / 256, batch_count), dim3(256), 0, compute_stream,
                batch_scratch.d_moe_accum, batch_scratch.d_res_mid_half,
                batch_scratch.d_post_f, batch_scratch.d_comb_f,
                batch_scratch.d_res_out_half, H);
            kernel::v4_half_to_float_kernel<<<
                (batch_count * HC_DIM + 255) / 256, 256, 0, compute_stream>>>(
                batch_scratch.d_res_out_half,
                batch_scratch.d_res_out,
                batch_count * HC_DIM);
            std::swap(batch_scratch.d_res_in, batch_scratch.d_res_out);
            std::swap(batch_scratch.d_res_in_half, batch_scratch.d_res_out_half);
        }

        const size_t last_residual_offset = static_cast<size_t>(batch_count - 1) * HC_DIM;
        hipLaunchKernelGGL(
            kernel::hc_head_wave32_kernel,
            dim3(1), dim3(32), 0, compute_stream,
            batch_scratch.d_res_in + last_residual_offset,
            model_resources_.d_hc_head_fn,
            model_resources_.d_hc_head_base,
            model_resources_.d_hc_head_scale,
            scratch.d_hc_head_out, H, HC, 1e-6f, 1e-6f);
        hipLaunchKernelGGL(
            kernel::v4_rmsnorm_wave32_kernel,
            dim3(1), dim3(32), 0, compute_stream,
            scratch.d_hc_head_out, model_resources_.d_final_norm,
            scratch.d_head_norm, H, 1e-6f);
        hipLaunchKernelGGL(
            kernel::v4_gemv_fp16_vec8_kernel,
            dim3(129280, 1), dim3(32), 0, compute_stream,
            scratch.d_head_norm, model_resources_.d_lm_head, scratch.d_logits, H);
        kernel::v4_argmax_fp16_partial_kernel<<<
            kernel::V4_ARGMAX_BLOCKS, 256, 0, compute_stream>>>(
            scratch.d_logits, 129280,
            scratch.d_argmax_partial_vals, scratch.d_argmax_partial_idx);
        kernel::v4_argmax_partial_reduce_kernel<<<1, 256, 0, compute_stream>>>(
            scratch.d_argmax_partial_vals, scratch.d_argmax_partial_idx,
            kernel::V4_ARGMAX_BLOCKS, scratch.d_argmax_result);

        int32_t host_argmax = 0;
        CHECK_HIP(hipMemcpyAsync(
            &host_argmax, scratch.d_argmax_result, sizeof(host_argmax),
            hipMemcpyDeviceToHost, compute_stream));
        CHECK_HIP(hipStreamSynchronize(compute_stream));
        if (sdma_stream) CHECK_HIP(hipStreamSynchronize(sdma_stream));
        if (sdma_cold_stream) CHECK_HIP(hipStreamSynchronize(sdma_cold_stream));
        if (demotion_stream) CHECK_HIP(hipStreamSynchronize(demotion_stream));
        reap_registry_transfers();
        current_seq_len_ = start_position + static_cast<uint32_t>(batch_count);
        return static_cast<uint32_t>(host_argmax);
    }

    void validate_prefill_span(
        std::span<const uint32_t> token_ids,
        uint32_t start_position
    ) const {
        if (token_ids.empty()) {
            throw std::invalid_argument("V4Pipeline::prefill: token span must not be empty");
        }
        if (start_position != current_seq_len_) {
            throw std::invalid_argument(
                "V4Pipeline::prefill: start position must continue the current generation state");
        }
        const uint64_t end_position = static_cast<uint64_t>(start_position) + token_ids.size();
        if (end_position > context_capacity()) {
            throw std::out_of_range("V4Pipeline::prefill: token span exceeds configured context capacity");
        }
    }

    template<typename T>
    void queue_trace_copy(std::vector<T>& destination, const T* source, size_t count) {
        destination.resize(count);
        if (count == 0) return;
        CHECK_HIP(hipMemcpyAsync(
            destination.data(), source, count * sizeof(T), hipMemcpyDeviceToHost, compute_stream));
    }

    V4AttentionTraceRecord* begin_attention_trace(
        V4Layer& layer,
        uint32_t position,
        PipelineScratchBuffers& layer_scratch,
        int head_dim,
        int total_query
    ) {
        if (!attention_trace_enabled_ || layer.layer_id != static_cast<int>(attention_trace_layer_) ||
            attention_trace_records_.size() >= attention_trace_limit_) {
            return nullptr;
        }

        attention_trace_records_.emplace_back();
        auto& trace = attention_trace_records_.back();
        trace.position = position;
        trace.attention_kind = layer.spec().attention_kind;
        queue_trace_copy(trace.query, layer_scratch.d_q, total_query);
        queue_trace_copy(trace.local_key, layer_scratch.d_kv_norm_act, head_dim);
        queue_trace_copy(trace.local_value, layer_scratch.d_kv_norm_act, head_dim);

        if (layer.spec().attention_kind == V4AttentionKind::Sliding) return &trace;

        const int coefficient = layer.spec().compression_ratio == 4 ? 2 : 1;
        const size_t compressor_width = static_cast<size_t>(coefficient * head_dim);
        queue_trace_copy(trace.compressor_kv, layer_scratch.d_compressor_kv, compressor_width);
        queue_trace_copy(trace.compressor_score, layer_scratch.d_compressor_score, compressor_width);
        if (layer.spec().attention_kind != V4AttentionKind::CSA) return &trace;

        const size_t indexer_query_width = static_cast<size_t>(
            kernel::DSV4_INDEX_N_HEADS * kernel::DSV4_INDEX_HEAD_DIM);
        const size_t indexer_width = static_cast<size_t>(
            coefficient * kernel::DSV4_INDEX_HEAD_DIM);
        queue_trace_copy(trace.indexer_query, layer_scratch.d_indexer_query, indexer_query_width);
        queue_trace_copy(trace.indexer_weights, layer_scratch.d_indexer_weights, kernel::DSV4_INDEX_N_HEADS);
        queue_trace_copy(trace.indexer_compressor_kv, layer_scratch.d_indexer_compressor_kv, indexer_width);
        queue_trace_copy(trace.indexer_compressor_score, layer_scratch.d_indexer_compressor_score, indexer_width);
        return &trace;
    }

    void select_indexer_topk(V4Layer& layer) {
        const size_t candidate_count = layer.indexer_candidate_count_;
        if (candidate_count == 0) return;

        std::vector<float> scores(candidate_count);
        CHECK_HIP(hipMemcpyAsync(
            scores.data(),
            layer.d_indexer_scores,
            candidate_count * sizeof(float),
            hipMemcpyDeviceToHost,
            compute_stream
        ));
        CHECK_HIP(hipStreamSynchronize(compute_stream));

        std::vector<int32_t> order(candidate_count);
        for (size_t index = 0; index < candidate_count; ++index) {
            order[index] = static_cast<int32_t>(index);
        }
        const size_t topk = std::min(candidate_count, static_cast<size_t>(layer.state_layout().index_topk));
        std::vector<int32_t> selected(static_cast<size_t>(layer.state_layout().index_topk), -1);
        if (candidate_count <= static_cast<size_t>(layer.state_layout().index_topk)) {
            std::copy(order.begin(), order.end(), selected.begin());
        } else {
            std::stable_sort(order.begin(), order.end(), [&scores](int32_t left, int32_t right) {
                const float left_score = scores[static_cast<size_t>(left)];
                const float right_score = scores[static_cast<size_t>(right)];
                if (left_score != right_score) return left_score > right_score;
                return left < right;
            });
            std::copy_n(order.begin(), topk, selected.begin());
        }
        CHECK_HIP(hipMemcpyAsync(
            layer.d_indexer_topk_indices,
            selected.data(),
            selected.size() * sizeof(int32_t),
            hipMemcpyHostToDevice,
            compute_stream
        ));
        CHECK_HIP(hipStreamSynchronize(compute_stream));
    }

    void validate_supported_model_config(const DeepSeekV4Config& model_cfg) const {
        V4ModelSpec::validate_config(model_cfg);
    }

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

    void mark_gpu_readiness_wait_start(uint64_t operation_id) {
        expert_supply_.mark_gpu_readiness_wait_start(operation_id);
    }

    void reap_registry_transfers() {
        expert_supply_.reap_registry_transfers();
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

    V4ModelResources model_resources_;
    bool attention_trace_enabled_{false};
    uint32_t attention_trace_layer_{0};
    size_t attention_trace_limit_{0};
    std::vector<V4AttentionTraceRecord> attention_trace_records_;
};

} // namespace aeon::core
