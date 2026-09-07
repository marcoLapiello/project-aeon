#include "core/aeon_loader.hpp"
#include "core/config.hpp"
#include "core/device.hpp"
#include "core/expert_registry.hpp"
#include "core/memory_budget.hpp"
#include "core/vram_expert_pool.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "    Phase 2 Spike 1 Test: Dynamic Memory Budget, Feasibility & Global Pool      " << std::endl;
    std::cout << "================================================================================" << std::endl;

    // 1. Device selection
    aeon::core::select_compute_device(true);

    // 2. Load model configuration
    std::string config_path = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/config.json";
    auto model_cfg = aeon::core::DeepSeekV4Config::load_from_json(config_path);
    size_t dense_weights_bytes = 15745100992ULL; // 14.66 GB as measured on model_dense.aeon

    // -------------------------------------------------------------------------
    // Test 1: Hard Feasibility Gate with Over-Budget Context Length
    // -------------------------------------------------------------------------
    std::cout << "\n[Test 1] Testing Hard Feasibility Gate with Excessive Context Size (262,144 tokens)..." << std::endl;
    aeon::core::AeonRuntimeConfig overbudget_cfg;
    overbudget_cfg.context_size = 262144; // MLA KV would require ~11.27 GB, exceeding remaining 24GB VRAM
    overbudget_cfg.host_ram_bytes = 0;   // Auto 80%

    auto overbudget_report = aeon::core::MemoryBudgetEngine::evaluate(overbudget_cfg, model_cfg, dense_weights_bytes);
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
    valid_cfg.host_ram_bytes = 0; // Auto 80%

    auto valid_report = aeon::core::MemoryBudgetEngine::evaluate(valid_cfg, model_cfg, dense_weights_bytes);
    std::cout << valid_report.to_string() << std::endl;
    assert(valid_report.is_feasible);
    assert(valid_report.hot_vram_slots >= 12); // Must guarantee at least 2*num_experts_per_tok
    std::cout << "  > [PASSED] Valid 4k context approved with " << valid_report.hot_vram_slots
              << " Hot VRAM expert slots and " << valid_report.warm_host_slots << " Warm Host slots.\n";

    // -------------------------------------------------------------------------
    // Test 3: Host-Side Expert Registry Initialization & Tier Tracking
    // -------------------------------------------------------------------------
    std::cout << "\n[Test 3] Testing ExpertRegistry initialization & round-robin tier assignment..." << std::endl;
    // Use smaller slot counts for quick unit check
    uint32_t test_vram_slots = 64;
    uint32_t test_host_slots = 128;
    aeon::core::ExpertRegistry registry(model_cfg.num_hidden_layers, model_cfg.n_routed_experts,
                                        test_vram_slots, test_host_slots);

    assert(registry.total_experts == 43 * 256);
    assert(registry.hot_vram_lru.size() == test_vram_slots);
    assert(registry.warm_host_lru.size() == test_host_slots);

    // Verify touch on hot expert
    uint32_t sample_gid = registry.hot_vram_lru.front();
    uint32_t sample_layer = registry.catalog[sample_gid].layer_id;
    uint32_t sample_expert = registry.catalog[sample_gid].expert_id;
    int32_t slot = registry.touch_hot_expert(sample_layer, sample_expert, 1);
    assert(slot >= 0);
    assert(registry.hits_hot == 1);
    std::cout << "  > [PASSED] ExpertRegistry round-robin & hot touch verified (Layer "
              << sample_layer << ", Expert " << sample_expert << " in Slot " << slot << ").\n";

    // -------------------------------------------------------------------------
    // Test 4: Physical Silicon Global VRAM Expert Pool Allocation & Streaming
    // -------------------------------------------------------------------------
    std::cout << "\n[Test 4] Allocating GlobalVRAMExpertPool on physical silicon (16 slots)..." << std::endl;
    uint32_t pool_slots = 16;
    aeon::core::GlobalVRAMExpertPool vram_pool(pool_slots);

    // Open AeonModelLoader to stream a real expert payload into slot 0
    std::string aeon_model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    aeon::core::AeonModelLoader loader;
    loader.open_model(aeon_model_dir);

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
    std::cout << "  > [PASSED] GlobalVRAMExpertPool DMA stream & silicon bit-parity verified!\n";

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  [SUCCESS] Spike 1 Foundations PASSED on physical AMD silicon!" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
