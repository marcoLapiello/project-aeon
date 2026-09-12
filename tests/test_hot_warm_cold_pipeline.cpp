#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline.hpp"

#include <cassert>
#include <iostream>

int main() {
    aeon::core::select_compute_device(true);

    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    aeon::core::AeonRuntimeConfig runtime_cfg;
    runtime_cfg.context_size = 256;
    runtime_cfg.warm_host_bytes = 20ULL * aeon::core::AEON_EXPERT_BYTES;

    aeon::core::V4Pipeline pipeline;
    pipeline.init_dynamic_global(model_dir, runtime_cfg, 43);

    assert(pipeline.expert_registry_);
    assert(pipeline.expert_registry_->host_capacity == 8);
    assert(pipeline.expert_registry_->published_warm_slots() > 0);
    assert(pipeline.expert_registry_->invariants_hold());

    const uint32_t next_token = pipeline.step(1, 0);
    assert(next_token < 129280);

    std::cout << "Hot/warm/cold pipeline smoke test passed: token=" << next_token
              << ", hot_hits=" << pipeline.expert_registry_->hits_hot
              << ", warm_hits=" << pipeline.expert_registry_->hits_warm
              << ", warm_slots=" << pipeline.expert_registry_->published_warm_slots()
              << ", cold_misses=" << pipeline.expert_registry_->misses_cold << std::endl;
    return 0;
}