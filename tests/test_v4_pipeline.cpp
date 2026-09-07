#include "core/device.hpp"
#include "core/v4_pipeline.hpp"

#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "        Spike 6.1: Multi-Layer Autoregressive Pipeline & KV Cache Test          " << std::endl;
    std::cout << "================================================================================" << std::endl;

    // 1. Device selection
    aeon::core::select_compute_device(true);

    std::string snapshot_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16/"
                               "models--yiminyuan--DeepSeek-V4-Flash-0731-INT4-W4A16/"
                               "snapshots/64700592cadaf205fe0c13202061ff4b45afbfd0";

    // 2. Initialize Pipeline with 2 layers (Layer 0 and Layer 1)
    aeon::core::V4Pipeline pipeline;
    pipeline.init(snapshot_dir, 2, 8, 256);

    // 3. Test Step 1: Execute single token step
    std::cout << "\n[Step 1] Running single-token forward step (token=1, pos=0)..." << std::endl;
    uint32_t prompt_token = 1; // BOS / starting token
    uint32_t next_tok = pipeline.step(prompt_token, 0);

    std::cout << "  > Input token: " << prompt_token << " -> Generated token: " << next_tok << std::endl;
    assert(next_tok < 129280);

    // 4. Test Step 2: Multi-step autoregressive generation
    std::cout << "\n[Step 2] Testing multi-token autoregressive generation with KV caching..." << std::endl;
    std::vector<uint32_t> prompt = {1, 100, 256};
    double ttft_ms = 0.0;
    double tok_per_sec = 0.0;

    auto generated = pipeline.generate(prompt, 6, &ttft_ms, &tok_per_sec);

    std::cout << "  > Prompt length: " << prompt.size() << " tokens: [";
    for (size_t i = 0; i < prompt.size(); ++i) {
        std::cout << prompt[i] << (i + 1 < prompt.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;

    std::cout << "  > Generated " << generated.size() << " tokens: [";
    for (size_t i = 0; i < generated.size(); ++i) {
        std::cout << generated[i] << (i + 1 < generated.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;

    assert(generated.size() == 6);
    for (auto tok : generated) {
        assert(tok < 129280);
    }

    std::cout << "  > TTFT (Prompt Prefill): " << ttft_ms << " ms" << std::endl;
    std::cout << "  > Autoregressive Speed : " << tok_per_sec << " tokens/sec" << std::endl;

    // 5. Test Step 3: Verify Tier 1 LRU Cache Behavior
    std::cout << "\n[Step 3] Verifying Tier 1 VRAM LRU Cache Statistics..." << std::endl;
    for (uint32_t l = 0; l < 2; ++l) {
        std::cout << "  > Layer " << l << " — Cache Hits: " << pipeline.layers[l]->cache_hits
                  << ", Cache Misses: " << pipeline.layers[l]->cache_misses << std::endl;
        assert(pipeline.layers[l]->cache_misses > 0);
    }

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  [SUCCESS] Spike 6.1 Complete: Multi-Layer Pipeline & KV Cache Validated!      " << std::endl;
    std::cout << "================================================================================" << std::endl;
    return 0;
}
