#include "infrastructure/core/aeon_loader.hpp"
#include "architecture/deepseek_v4/core/config.hpp"
#include "platform/rdna3/device.hpp"
#include "infrastructure/core/expert_registry.hpp"
#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline.hpp"
#include "backend/swizzled_w4a16/core/vram_expert_pool.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "    Dynamic Memory Budget, Feasibility & Global Pool Validation                 " << std::endl;
    std::cout << "================================================================================" << std::endl;

    // 1. Device selection
    aeon::core::select_compute_device(true);

    // 2. Load model configuration
    std::string aeon_model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    std::string config_path = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/config.json";
    auto model_cfg = aeon::core::DeepSeekV4Config::load_from_json(config_path);
    aeon::core::AeonModelLoader loader;
    loader.open_model(aeon_model_dir);
    const size_t dense_weights_bytes = loader.dense_file_size();

    // -------------------------------------------------------------------------
    // Test 1: Hard Feasibility Gate with Over-Budget Context Length
    // -------------------------------------------------------------------------
    std::cout << "\n[Test 1] Testing Hard Feasibility Gate with Excessive Context Size (262,144 tokens)..." << std::endl;
    aeon::core::AeonRuntimeConfig overbudget_cfg;
    overbudget_cfg.context_size = 262144; // MLA KV would require ~11.27 GB, exceeding remaining 24GB VRAM
    overbudget_cfg.warm_host_bytes = 0;   // Warm disabled

    auto overbudget_report = aeon::core::MemoryBudgetEngine::evaluate(
        overbudget_cfg, model_cfg, dense_weights_bytes, loader.expert_format());
    std::cout << overbudget_report.to_string() << std::endl;
    assert(!overbudget_report.is_feasible);
    assert(overbudget_report.max_viable_context_size > 0);
    std::cout << "  > [PASSED] Overbudget configuration correctly rejected! Max viable context: "
              << overbudget_report.max_viable_context_size << " tokens.\n";

    // -------------------------------------------------------------------------
    // Test 2: Hard Feasibility Gate with Valid 4,096 Context Length
    // -------------------------------------------------------------------------
    std::cout << "\n[Test 2] Testing Feasibility Gate with Valid 4k Context (4,096 tokens)..." << std::endl;
    aeon::core::AeonRuntimeConfig valid_cfg;
    valid_cfg.context_size = 4096;
    valid_cfg.warm_host_bytes = 0; // Warm disabled

    auto valid_report = aeon::core::MemoryBudgetEngine::evaluate(
        valid_cfg, model_cfg, dense_weights_bytes, loader.expert_format());
    std::cout << valid_report.to_string() << std::endl;
    assert(valid_report.is_feasible);
    assert(valid_report.hot_vram_slots >= 12); // Must guarantee at least 2*num_experts_per_tok
    std::cout << "  > [PASSED] Valid 4k context approved with " << valid_report.hot_vram_slots
              << " Hot VRAM expert slots and " << valid_report.warm_host_slots << " Warm Host slots.\n";

    // -------------------------------------------------------------------------
    // Test 3: Host-Side Expert Registry Initialization & Tier Tracking
    // -------------------------------------------------------------------------
    std::cout << "\n[Test 3] Testing ExpertRegistry initialization & published ownership maps..." << std::endl;
    // Use smaller slot counts for quick unit check
    uint32_t test_vram_slots = 64;
    uint32_t test_host_slots = 128;
    aeon::core::ExpertRegistry registry(model_cfg.num_hidden_layers, model_cfg.n_routed_experts,
                                        test_vram_slots, test_host_slots);

    assert(registry.total_experts == 43 * 256);
    assert(registry.hot_vram_lru.size() == test_vram_slots);
    assert(registry.warm_host_lru.size() == test_host_slots);

    // Verify a published Hot request acquires a lease without mutating ownership.
    uint32_t sample_gid = registry.hot_vram_lru.front();
    uint32_t sample_layer = registry.catalog[sample_gid].layer_id;
    uint32_t sample_expert = registry.catalog[sample_gid].expert_id;
    auto hot_request = registry.reserve_request(sample_layer, sample_expert, 1, 2);
    assert(hot_request.kind == aeon::core::ExpertRequestKind::HOT_HIT);
    assert(hot_request.vram_slot >= 0);
    assert(registry.hits_hot == 1);
    registry.release_lease(sample_gid);
    assert(registry.invariants_hold());
    std::cout << "  > [PASSED] ExpertRegistry ownership and Hot lease verified (Layer "
              << sample_layer << ", Expert " << sample_expert << " in Slot "
              << hot_request.vram_slot << ").\n";

    // -------------------------------------------------------------------------
    // Test 4: Physical Silicon Global VRAM Expert Pool Allocation & Streaming
    // -------------------------------------------------------------------------
    std::cout << "\n[Test 4] Allocating UnifiedVRAMExpertPool on physical silicon (16 slots)..." << std::endl;
    uint32_t pool_slots = 16;
    aeon::core::UnifiedVRAMExpertPool vram_pool(pool_slots);

    const uint8_t* expert_payload = loader.get_expert_data(0, 0);
    assert(expert_payload != nullptr);

    vram_pool.upload_from_host_expert(0, expert_payload, 0);
    CHECK_HIP(hipStreamSynchronize(0));

    // Read back a small sample of W1 packed weights from device slot 0 to verify data integrity
    std::vector<uint32_t> host_w1_check(16);
    CHECK_HIP(hipMemcpy(host_w1_check.data(), vram_pool.get_w1_packed(0),
                        host_w1_check.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));

    const uint32_t* orig_w1 = reinterpret_cast<const uint32_t*>(expert_payload + aeon::core::AEON_W1_PACKED_OFFSET);
    for (size_t i = 0; i < host_w1_check.size(); ++i) {
        assert(host_w1_check[i] == orig_w1[i]);
    }
    std::cout << "  > [PASSED] UnifiedVRAMExpertPool DMA stream & silicon bit-parity verified!\n";

    // Future backends can use opaque payload slots without exposing swizzled views.
    aeon::core::ExpertFormatDescriptor opaque_format{
        aeon::core::ExpertFormatKind::UNKNOWN,
        7,
        aeon::core::AEON_SECTOR_SIZE,
        1,
        1,
        aeon::core::AEON_SECTOR_SIZE
    };
    aeon::core::ExpertPayloadPool opaque_pool(1, opaque_format);
    std::vector<uint8_t> opaque_source(opaque_format.payload_bytes, 0x5A);
    std::vector<uint8_t> opaque_result(opaque_format.payload_bytes, 0);
    opaque_pool.upload_from_host_expert(0, opaque_source.data(), 0);
    CHECK_HIP(hipStreamSynchronize(0));
    opaque_pool.download_to_host_expert(0, opaque_result.data(), 0);
    CHECK_HIP(hipStreamSynchronize(0));
    assert(std::memcmp(opaque_source.data(), opaque_result.data(), opaque_source.size()) == 0);

    aeon::core::UnifiedVRAMExpertPool incompatible_pool(1, opaque_format);
    bool rejected_swizzled_view = false;
    try {
        (void)incompatible_pool.get_w1_packed(0);
    } catch (const std::logic_error&) {
        rejected_swizzled_view = true;
    }
    assert(rejected_swizzled_view);
    std::cout << "  > [PASSED] Opaque future-format payload storage and view guard verified!\n";

    // -------------------------------------------------------------------------
    // Test 5: End-to-End V4Pipeline with Global Unified Pool on Physical Silicon
    // -------------------------------------------------------------------------
    std::cout << "\n[Test 5] Initializing end-to-end V4Pipeline with Global Unified Pool (2 layers)..." << std::endl;
    aeon::core::V4Pipeline pipeline;
    aeon::core::AeonRuntimeConfig pipeline_cfg;
    pipeline_cfg.context_size = 4096;
    pipeline_cfg.warm_host_bytes = 0; // Warm disabled

    pipeline.init_dynamic_global(aeon_model_dir, pipeline_cfg, 2);

    // Run forward step (token 1, pos 0)
    std::cout << "\n  > Executing forward step on Global Unified Pool..." << std::endl;
    uint32_t next_tok = pipeline.step(1, 0);
    std::cout << "  > Output next token: " << next_tok << std::endl;
    assert(next_tok < 129280);

    // Multi-token generation
    std::cout << "  > Running multi-token autoregressive generation (4 new tokens)..." << std::endl;
    std::vector<uint32_t> prompt = {1, 100, 256};
    double ttft_ms = 0.0;
    double tok_sec = 0.0;
    auto gen = pipeline.generate(prompt, 4, &ttft_ms, &tok_sec);

    std::cout << "  > Generated tokens: [";
    for (size_t i = 0; i < gen.size(); ++i) {
        std::cout << gen[i] << (i + 1 < gen.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;
    assert(gen.size() == 4);
    for (auto t : gen) assert(t < 129280);

    // Check registry hit statistics
    std::cout << "  > Global Pool Registry Stats: Hot Hits=" << pipeline.expert_registry_->hits_hot
              << ", Warm Hits=" << pipeline.expert_registry_->hits_warm
              << ", Cold Misses=" << pipeline.expert_registry_->misses_cold << std::endl;
    assert(pipeline.expert_registry_->hits_hot > 0);

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  [SUCCESS] Dynamic memory budget and global pool validation passed on silicon!" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
