#include "core/device.hpp"
#include "core/v4_pipeline.hpp"

#include <cassert>
#include <iostream>

int main() {
    aeon::core::select_compute_device(true);

    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    aeon::core::AeonRuntimeConfig runtime_cfg;
    runtime_cfg.context_size = 256;
    runtime_cfg.host_ram_bytes = 8ULL * aeon::core::AEON_EXPERT_BYTES;
    runtime_cfg.preload_warm_host = true;

    aeon::core::V4Pipeline pipeline;
    pipeline.init_dynamic_global(model_dir, runtime_cfg, 43);

    assert(pipeline.expert_registry_);
    assert(pipeline.expert_registry_->host_capacity == 8);

    const uint32_t next_token = pipeline.step(1, 0);
    assert(next_token < 129280);

    std::cout << "Hot/warm/cold pipeline smoke test passed: token=" << next_token
              << ", hot_hits=" << pipeline.expert_registry_->hits_hot
              << ", warm_hits=" << pipeline.expert_registry_->hits_warm
              << ", cold_misses=" << pipeline.expert_registry_->misses_cold << std::endl;
    return 0;
}