#include "core/aeon_loader.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    aeon::core::AeonModelLoader loader;
    loader.open_model(model_dir);

    assert(loader.expert_format_version() == 2);
    assert(loader.num_layers() == 43);
    assert(loader.experts_per_layer() == 256);
    assert(loader.get_expert_location(0, 0).byte_length == aeon::core::AEON_EXPERT_BYTES);
    assert(loader.get_expert_location(42, 255).file_offset ==
           static_cast<uint64_t>(42 * 256 + 255) * aeon::core::AEON_EXPERT_BYTES);
    assert(loader.get_expert_data(0, 0) != nullptr);

    std::cout << "[PASS] Version-2 swizzled Aeon expert loader" << std::endl;
    return 0;
}