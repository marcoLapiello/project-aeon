#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "platform/rdna3/device.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void assert_reset_state(const aeon::core::V4Layer& layer) {
    const auto& layout = layer.state_layout();
    std::vector<int64_t> local_positions(layout.local_capacity);
    CHECK_HIP(hipMemcpy(
        local_positions.data(),
        layer.d_local_positions,
        layout.local_metadata_bytes(),
        hipMemcpyDeviceToHost));
    for (const int64_t position : local_positions) assert(position == -1);

    std::vector<half> local_values(layout.local_capacity * layout.head_dim);
    CHECK_HIP(hipMemcpy(
        local_values.data(),
        layer.d_local_key_cache,
        layout.local_vector_bytes(),
        hipMemcpyDeviceToHost));
    for (const half value : local_values) assert(__half2float(value) == 0.0f);

    if (layout.uses_indexer()) {
        std::vector<int32_t> topk(layout.index_topk);
        CHECK_HIP(hipMemcpy(
            topk.data(),
            layer.d_indexer_topk_indices,
            layout.indexer_topk_bytes(),
            hipMemcpyDeviceToHost));
        for (const int32_t index : topk) assert(index == -1);
    }
}

void assert_ring_slot_reuse(aeon::core::V4Layer& layer) {
    const auto& layout = layer.state_layout();
    const size_t vector_bytes = layout.local_vector_bytes();
    const size_t positions_bytes = layout.local_metadata_bytes();
    half* d_query{nullptr};
    half* d_output{nullptr};
    float* d_sink{nullptr};
    CHECK_HIP(hipMalloc(&d_query, vector_bytes * 64));
    CHECK_HIP(hipMalloc(&d_output, vector_bytes * 64));
    CHECK_HIP(hipMalloc(&d_sink, 64 * sizeof(float)));
    CHECK_HIP(hipMemset(d_query, 0, vector_bytes * 64));
    std::vector<float> sink(64, -20.0f);
    CHECK_HIP(hipMemcpy(d_sink, sink.data(), 64 * sizeof(float), hipMemcpyHostToDevice));

    std::vector<half> values(layout.head_dim, __float2half(2.0f));
    std::vector<int64_t> positions(layout.local_capacity, -1);
    positions[0] = 0;
    CHECK_HIP(hipMemcpy(
        layer.d_local_value_cache,
        values.data(),
        values.size() * sizeof(half),
        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(
        layer.d_local_positions,
        positions.data(),
        positions_bytes,
        hipMemcpyHostToDevice));

    hipLaunchKernelGGL(
        aeon::kernel::v4_cached_sliding_window_attn_wave32_kernel,
        dim3(64), dim3(32), 0, 0,
        d_query,
        layer.d_local_key_cache,
        layer.d_local_value_cache,
        layer.d_local_positions,
        d_sink,
        d_output,
        0,
        128,
        aeon::kernel::DSV4_ATTN_SCALE);
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<half> output(vector_bytes * 64 / sizeof(half));
    CHECK_HIP(hipMemcpy(
        output.data(),
        d_output,
        output.size() * sizeof(half),
        hipMemcpyDeviceToHost));
    assert(std::abs(__half2float(output[0]) - 2.0f) < 1e-2f);

    values.assign(layout.head_dim, __float2half(4.0f));
    positions.assign(layout.local_capacity, -1);
    positions[0] = 128;
    CHECK_HIP(hipMemcpy(
        layer.d_local_value_cache,
        values.data(),
        values.size() * sizeof(half),
        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(
        layer.d_local_positions,
        positions.data(),
        positions_bytes,
        hipMemcpyHostToDevice));
    hipLaunchKernelGGL(
        aeon::kernel::v4_cached_sliding_window_attn_wave32_kernel,
        dim3(64), dim3(32), 0, 0,
        d_query,
        layer.d_local_key_cache,
        layer.d_local_value_cache,
        layer.d_local_positions,
        d_sink,
        d_output,
        128,
        128,
        aeon::kernel::DSV4_ATTN_SCALE);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(
        output.data(),
        d_output,
        output.size() * sizeof(half),
        hipMemcpyDeviceToHost));
    assert(std::abs(__half2float(output[0]) - 4.0f) < 1e-2f);

    CHECK_HIP(hipFree(d_query));
    CHECK_HIP(hipFree(d_output));
    CHECK_HIP(hipFree(d_sink));
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);
    const auto config = aeon::core::DeepSeekV4Config::load_from_json(
        "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/config.json");
    const auto specs = aeon::core::V4ModelSpec::resolve_layers(config);

    aeon::core::V4Layer sliding;
    sliding.initialize_state(specs[0], 131);
    assert(sliding.d_kv_cache == sliding.d_local_key_cache);
    assert(sliding.d_compressed_key_cache == nullptr);
    sliding.record_position(130);
    assert(sliding.local_valid_count_ == 128);
    assert(sliding.compressed_entry_count_ == 0);
    assert_ring_slot_reuse(sliding);
    sliding.reset_generation_state();
    assert(sliding.local_valid_count_ == 0);
    assert_reset_state(sliding);

    aeon::core::V4Layer csa;
    csa.initialize_state(specs[2], 131);
    assert(csa.d_compressed_key_cache != nullptr);
    assert(csa.d_compressor_partial_kv != nullptr);
    assert(csa.d_indexer_key_cache != nullptr);
    assert(csa.d_indexer_topk_indices != nullptr);
    csa.record_position(3);
    assert(csa.compressor_partial_count_ == 4);
    assert(csa.compressed_entry_count_ == 1);
    assert(csa.indexer_candidate_count_ == 1);
    csa.record_position(130);
    assert(csa.compressor_partial_count_ == 8);
    assert(csa.compressed_entry_count_ == 32);
    assert(csa.indexer_candidate_count_ == 32);
    csa.reset_generation_state();
    assert(csa.compressor_partial_count_ == 0);
    assert_reset_state(csa);

    aeon::core::V4Layer hca;
    hca.initialize_state(specs[3], 131);
    assert(hca.d_compressed_key_cache != nullptr);
    assert(hca.d_compressor_partial_kv != nullptr);
    assert(hca.d_indexer_key_cache == nullptr);
    hca.record_position(130);
    assert(hca.compressor_partial_count_ == 128);
    assert(hca.compressed_entry_count_ == 1);
    assert(hca.indexer_candidate_count_ == 0);
    hca.reset_generation_state();
    assert_reset_state(hca);

    std::cout << "V4 layer state device test passed: Sliding, CSA, HCA allocation and reset" << std::endl;
    return 0;
}