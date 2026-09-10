#include "core/device.hpp"
#include "core/v4_pipeline.hpp"

#include <cassert>
#include <iostream>

int main() {
    aeon::core::select_compute_device(true);
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

    aeon::core::V4Pipeline pipeline;
    pipeline.init_aeon(model_dir, 2, 8, 256, true, true);
    const uint32_t next_token = pipeline.step(1, 0);
    assert(next_token < 129280);

    std::cout << "[PASS] Opt-in swizzled two-layer Aeon pipeline step: "
              << next_token << std::endl;
    return 0;
}