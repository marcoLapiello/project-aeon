#include "platform/rdna3/device.hpp"
#include "infrastructure/core/routing_counter.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    aeon::core::select_compute_device(true);

    aeon::core::V4Pipeline pipeline;
    pipeline.init_aeon("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon", 2, 8, 256);
    pipeline.enable_routing_counter();

    const std::vector<uint32_t> prompt = {1, 100, 256};
    const auto generated = pipeline.generate(prompt, 4);
    assert(generated.size() == 4);

    const auto* counter = pipeline.routing_counter();
    assert(counter != nullptr);
    assert(counter->num_layers() == 2);

    for (uint32_t layer_id = 0; layer_id < 2; ++layer_id) {
        assert(counter->total_selections(
            aeon::core::RoutingPhase::Prefill, layer_id
        ) == prompt.size() * aeon::core::RoutingCounter::kTopK);
        assert(counter->total_selections(
            aeon::core::RoutingPhase::Decode, layer_id
        ) == 3 * aeon::core::RoutingCounter::kTopK);
    }

    std::cout << "[SUCCESS] Routing counter captured prefill and decode selections.\n";
    return 0;
}