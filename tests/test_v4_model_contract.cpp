#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_model_contract.hpp"

#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect_invalid(const std::function<void()>& operation) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main() {
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    const auto config = aeon::core::DeepSeekV4Config::load_from_json(model_dir + "/config.json");
    assert(config.compress_ratios.size() == 46);
    assert(config.compress_ratios[0] == 0);
    assert(config.compress_ratios[2] == 4);
    assert(config.compress_ratios[3] == 128);
    assert(config.compress_ratios[41] == 128);
    assert(config.compress_ratios[42] == 4);
    assert(config.compress_ratios[43] == 0);
    assert(config.index_head_dim == 128);
    assert(config.index_n_heads == 64);
    assert(config.index_topk == 512);
    assert(config.o_groups == 8);
    assert(config.compress_rope_theta == 160000.0f);
    assert(config.rope_scaling.original_max_position_embeddings == 65536);
    assert(config.dspark_target_layer_ids == std::vector<int32_t>({40, 41, 42}));

    const auto layers = aeon::core::V4ModelSpec::resolve_layers(config);
    assert(layers.size() == 43);
    assert(layers[0].attention_kind == aeon::core::V4AttentionKind::Sliding);
    assert(layers[1].attention_kind == aeon::core::V4AttentionKind::Sliding);
    assert(layers[2].attention_kind == aeon::core::V4AttentionKind::CSA);
    assert(layers[3].attention_kind == aeon::core::V4AttentionKind::HCA);
    assert(layers[41].attention_kind == aeon::core::V4AttentionKind::HCA);
    assert(layers[42].attention_kind == aeon::core::V4AttentionKind::CSA);

    auto invalid_schedule = config;
    invalid_schedule.compress_ratios[2] = 128;
    expect_invalid([&] { aeon::core::V4ModelSpec::validate_config(invalid_schedule); });

    auto invalid_indexer = config;
    invalid_indexer.index_topk = 256;
    expect_invalid([&] { aeon::core::V4ModelSpec::validate_config(invalid_indexer); });

    auto invalid_rope = config;
    invalid_rope.compress_rope_theta = 10000.0f;
    expect_invalid([&] { aeon::core::V4ModelSpec::validate_config(invalid_rope); });

    aeon::core::AeonModelLoader loader;
    loader.open_model(model_dir);
    aeon::core::V4ModelContract::validate(config, loader);

    const auto inventory = loader.dense_tensor_inventory();
    assert(inventory.size() > 0);
    assert(loader.has_tensor("layers.2.attn.indexer.wq_b.weight"));
    assert(loader.get_tensor("layers.2.attn.indexer.wq_b.weight").shape ==
           std::vector<int64_t>({8192, 1024}));
    assert(loader.get_tensor("layers.3.attn.compressor.ape").shape ==
           std::vector<int64_t>({128, 512}));

    auto missing_tensor = aeon::core::V4ModelContract::model_tensor_requirements(config).front();
    missing_tensor.name = "missing.required.tensor";
    expect_invalid([&] {
        aeon::core::V4ModelContract::validate_tensor_requirement(missing_tensor, loader);
    });

    std::cout << "V4 model contract passed: " << layers.size()
              << " layers, " << inventory.size() << " dense tensors,"
              << " classes [3 Sliding, 20 CSA, 20 HCA]" << std::endl;
    return 0;
}