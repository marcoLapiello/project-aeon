#include "core/aeon_loader.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "Testing AeonModelLoader..." << std::endl;
    std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

    aeon::core::AeonModelLoader loader;
    loader.open_model(model_dir);

    std::cout << "Dense tensors loaded: " << loader.total_dense_tensors() << std::endl;
    assert(loader.total_dense_tensors() == 1271);

    // Verify sample dense tensor
    assert(loader.has_tensor("embed.weight"));
    const auto& embed = loader.get_tensor("embed.weight");
    std::cout << "embed.weight size: " << embed.byte_size << " bytes, dtype: " << embed.dtype << std::endl;

    // Verify sample routed expert
    const uint8_t* exp0 = loader.get_expert_data(0, 0);
    assert(exp0 != nullptr);
    const uint8_t* exp_last = loader.get_expert_data(42, 255);
    assert(exp_last != nullptr);
    std::cout << "Expert pointers valid. num_layers: " << loader.num_layers()
              << ", experts_per_layer: " << loader.experts_per_layer() << std::endl;

    std::cout << "AeonModelLoader smoke check PASSED!" << std::endl;
    return 0;
}
