#include "architecture/deepseek_v4/core/v4_pipeline.hpp"
#include "platform/rdna3/device.hpp"

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    aeon::core::select_compute_device(true);

    aeon::core::AeonRuntimeConfig runtime_config;
    runtime_config.context_size = 132;
    runtime_config.warm_host_bytes = 0;

    aeon::core::V4Pipeline pipeline;
    pipeline.initialize(
        "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon",
        runtime_config);

    for (uint32_t position = 0; position < 132; ++position) {
        const uint32_t next_token = pipeline.step(1u + position, position);
        assert(next_token < 129280u);
    }

    assert(pipeline.layers.at(0)->spec().attention_kind == aeon::core::V4AttentionKind::Sliding);
    assert(pipeline.layers.at(2)->spec().attention_kind == aeon::core::V4AttentionKind::CSA);
    assert(pipeline.layers.at(3)->spec().attention_kind == aeon::core::V4AttentionKind::HCA);
    assert(pipeline.layers.at(0)->local_valid_count_ == 128);
    assert(pipeline.layers.at(2)->compressed_entry_count_ == 33);
    assert(pipeline.layers.at(2)->indexer_candidate_count_ == 33);
    assert(pipeline.layers.at(3)->compressed_entry_count_ == 1);

    int64_t compressed_position = -1;
    int64_t indexer_position = -1;
    int64_t hca_position = -1;
    int32_t topk_index = -1;
    auto& csa_layer = *pipeline.layers.at(2);
    auto& hca_layer = *pipeline.layers.at(3);
    if (hipMemcpy(
            &compressed_position,
            csa_layer.d_compressed_positions + 32,
            sizeof(compressed_position),
            hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(
            &indexer_position,
            csa_layer.d_indexer_positions + 32,
            sizeof(indexer_position),
            hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(
            &topk_index,
            csa_layer.d_indexer_topk_indices,
            sizeof(topk_index),
            hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(
            &hca_position,
            hca_layer.d_compressed_positions,
            sizeof(hca_position),
            hipMemcpyDeviceToHost) != hipSuccess) {
        return 1;
    }
    assert(compressed_position == 131);
    assert(indexer_position == 131);
    assert(topk_index >= 0 && topk_index < 33);
    assert(hca_position == 127);

    std::cout << "V4 Stage 4 dispatch passed: C4A entries="
              << csa_layer.compressed_entry_count_
              << ", HCA entries=" << hca_layer.compressed_entry_count_
              << ", local ring=" << pipeline.layers.at(0)->local_valid_count_ << std::endl;
    return 0;
}