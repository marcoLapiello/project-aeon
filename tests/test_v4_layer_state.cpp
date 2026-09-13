#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer_state.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
    const auto config = aeon::core::DeepSeekV4Config::load_from_json(
        "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/config.json");
    const auto specs = aeon::core::V4ModelSpec::resolve_layers(config);
    assert(specs.size() == 43);

    size_t sliding_count = 0;
    size_t csa_count = 0;
    size_t hca_count = 0;
    for (const auto& spec : specs) {
        const auto layout = aeon::core::V4LayerStateLayout::from_spec(spec, 4096);
        assert(layout.local_capacity == 128);
        if (spec.attention_kind == aeon::core::V4AttentionKind::Sliding) {
            ++sliding_count;
            assert(layout.compressed_capacity == 0);
            assert(layout.compressor_state_bytes() == 0);
            assert(layout.indexer_workspace_bytes() == 0);
        } else if (spec.attention_kind == aeon::core::V4AttentionKind::CSA) {
            ++csa_count;
            assert(layout.compressed_capacity == 1024);
            assert(layout.compressor_partial_capacity == 8);
            assert(layout.compressor_width == 1024);
            assert(layout.indexer_partial_capacity == 8);
            assert(layout.indexer_width == 256);
            assert(layout.indexer_cache_bytes() > 0);
            assert(layout.indexer_workspace_bytes() > 0);
        } else {
            ++hca_count;
            assert(layout.compressed_capacity == 32);
            assert(layout.compressor_partial_capacity == 128);
            assert(layout.compressor_width == 512);
            assert(layout.indexer_cache_bytes() == 0);
            assert(layout.indexer_workspace_bytes() == 0);
        }
        assert(layout.total_device_bytes() >= layout.local_state_bytes());
    }

    assert(sliding_count == 2);
    assert(csa_count == 21);
    assert(hca_count == 20);

    aeon::kernel::RopeTable main_rope;
    aeon::kernel::RopeTable compressed_rope;
    main_rope.init(65537, config.rope_theta, 1.0f);
    compressed_rope.init(
        65537,
        config.compress_rope_theta,
        config.rope_scaling.factor,
        config.rope_scaling.beta_fast,
        config.rope_scaling.beta_slow,
        static_cast<uint32_t>(config.rope_scaling.original_max_position_embeddings));
    const size_t rope_offset = static_cast<size_t>(65536) * main_rope.half_rope;
    assert(main_rope.cos_cache[rope_offset] != compressed_rope.cos_cache[rope_offset]);
    assert(main_rope.sin_cache[rope_offset] != compressed_rope.sin_cache[rope_offset]);

    const auto short_layout = aeon::core::V4LayerStateLayout::from_spec(specs[2], 3);
    assert(short_layout.local_capacity == 3);
    assert(short_layout.compressed_capacity == 1);

    std::cout << "V4 layer state layout passed: classes ["
              << sliding_count << " Sliding, " << csa_count << " CSA, "
              << hca_count << " HCA] and class-aware RoPE" << std::endl;
    return 0;
}