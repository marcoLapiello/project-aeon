#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "platform/rdna3/device.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_hip(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(error));
    }
}

template<typename T>
T* allocate_device(size_t count) {
    T* pointer = nullptr;
    check_hip(hipMalloc(&pointer, count * sizeof(T)), "hipMalloc");
    return pointer;
}

void test_compressor_boundary(int ratio, int width, int partial_capacity, int compressed_capacity) {
    constexpr int head_dim = 512;
    constexpr int rope_dim = 64;
    const int coefficient = width / head_dim;

    half* d_input_kv = allocate_device<half>(width);
    half* d_input_score = allocate_device<half>(width);
    float* d_partial_kv = allocate_device<float>(partial_capacity * width);
    float* d_partial_score = allocate_device<float>(partial_capacity * width);
    int64_t* d_partial_positions = allocate_device<int64_t>(partial_capacity);
    float* d_ape = allocate_device<float>(ratio * width);
    half* d_norm = allocate_device<half>(head_dim);
    half* d_compressed_key = allocate_device<half>(compressed_capacity * head_dim);
    half* d_compressed_value = allocate_device<half>(compressed_capacity * head_dim);
    int64_t* d_compressed_positions = allocate_device<int64_t>(compressed_capacity);
    float* d_cos = allocate_device<float>(256 * (rope_dim / 2));
    float* d_sin = allocate_device<float>(256 * (rope_dim / 2));

    std::vector<half> input_kv(width, __float2half(1.0f));
    std::vector<half> input_score(width, __float2half(0.0f));
    std::vector<float> ape(ratio * width, 0.0f);
    std::vector<half> norm(head_dim, __float2half(1.0f));
    std::vector<float> cos_cache(256 * (rope_dim / 2), 1.0f);
    std::vector<float> sin_cache(256 * (rope_dim / 2), 0.0f);
    std::vector<int64_t> empty_positions(partial_capacity, -1);
    check_hip(hipMemcpy(d_ape, ape.data(), ape.size() * sizeof(float), hipMemcpyHostToDevice), "copy APE");
    check_hip(hipMemcpy(d_norm, norm.data(), norm.size() * sizeof(half), hipMemcpyHostToDevice), "copy norm");
    check_hip(hipMemcpy(d_cos, cos_cache.data(), cos_cache.size() * sizeof(float), hipMemcpyHostToDevice), "copy cos");
    check_hip(hipMemcpy(d_sin, sin_cache.data(), sin_cache.size() * sizeof(float), hipMemcpyHostToDevice), "copy sin");
    check_hip(hipMemcpy(d_partial_positions, empty_positions.data(), empty_positions.size() * sizeof(int64_t), hipMemcpyHostToDevice), "clear positions");

    for (int64_t position = 0; position < ratio * coefficient; ++position) {
        check_hip(hipMemcpy(d_input_kv, input_kv.data(), input_kv.size() * sizeof(half), hipMemcpyHostToDevice), "copy kv");
        check_hip(hipMemcpy(d_input_score, input_score.data(), input_score.size() * sizeof(half), hipMemcpyHostToDevice), "copy score");
        hipLaunchKernelGGL(
            aeon::kernel::v4_save_compressor_state_kernel,
            dim3(1), dim3(256), 0, 0,
            d_input_kv, d_input_score, d_partial_kv, d_partial_score,
            d_partial_positions, d_ape, position, ratio, partial_capacity, width);
        check_hip(hipGetLastError(), "launch partial state");
    }

    const int boundary = ratio * coefficient - 1;
    hipLaunchKernelGGL(
        aeon::kernel::v4_materialize_compressed_entry_kernel,
        dim3(1), dim3(512), 0, 0,
        d_partial_kv, d_partial_score, d_partial_positions, d_norm,
        d_compressed_key, d_compressed_value, d_compressed_positions,
        d_cos, d_sin, static_cast<int64_t>(boundary), ratio, partial_capacity,
        head_dim, width, 0, aeon::kernel::DSV4_NOPE_DIM, rope_dim, 1e-6f);
    check_hip(hipGetLastError(), "launch compressed entry");
    check_hip(hipDeviceSynchronize(), "synchronize compressed entry");

    std::vector<half> output(head_dim);
    int64_t output_position = -1;
    check_hip(hipMemcpy(output.data(), d_compressed_key, output.size() * sizeof(half), hipMemcpyDeviceToHost), "read compressed key");
    check_hip(hipMemcpy(&output_position, d_compressed_positions, sizeof(output_position), hipMemcpyDeviceToHost), "read compressed position");
    assert(output_position == boundary);
    for (const half value : output) assert(std::abs(__half2float(value) - 1.0f) < 2e-2f);

    check_hip(hipFree(d_input_kv), "free input kv");
    check_hip(hipFree(d_input_score), "free input score");
    check_hip(hipFree(d_partial_kv), "free partial kv");
    check_hip(hipFree(d_partial_score), "free partial score");
    check_hip(hipFree(d_partial_positions), "free partial positions");
    check_hip(hipFree(d_ape), "free ape");
    check_hip(hipFree(d_norm), "free norm");
    check_hip(hipFree(d_compressed_key), "free compressed key");
    check_hip(hipFree(d_compressed_value), "free compressed value");
    check_hip(hipFree(d_compressed_positions), "free compressed positions");
    check_hip(hipFree(d_cos), "free cos");
    check_hip(hipFree(d_sin), "free sin");
}

void test_indexer_and_mixed_attention() {
    constexpr int head_dim = 512;
    constexpr int index_head_dim = 128;
    constexpr int local_capacity = 4;
    constexpr int compressed_count = 1;

    half* d_query = allocate_device<half>(aeon::kernel::DSV4_INDEX_N_HEADS * index_head_dim);
    float* d_weights = allocate_device<float>(aeon::kernel::DSV4_INDEX_N_HEADS);
    half* d_indexer_key = allocate_device<half>(compressed_count * index_head_dim);
    float* d_scores = allocate_device<float>(compressed_count);
    half* d_attention_query = allocate_device<half>(aeon::kernel::DSV4_NUM_HEADS * head_dim);
    half* d_local_key = allocate_device<half>(local_capacity * head_dim);
    half* d_local_value = allocate_device<half>(local_capacity * head_dim);
    int64_t* d_local_positions = allocate_device<int64_t>(local_capacity);
    float* d_sink = allocate_device<float>(aeon::kernel::DSV4_NUM_HEADS);
    half* d_compressed_key = allocate_device<half>(compressed_count * head_dim);
    half* d_compressed_value = allocate_device<half>(compressed_count * head_dim);
    int64_t* d_compressed_positions = allocate_device<int64_t>(compressed_count);
    half* d_output = allocate_device<half>(aeon::kernel::DSV4_NUM_HEADS * head_dim);

    std::vector<half> ones_query(aeon::kernel::DSV4_INDEX_N_HEADS * index_head_dim, __float2half(1.0f));
    std::vector<float> ones_weights(aeon::kernel::DSV4_INDEX_N_HEADS, 1.0f);
    std::vector<half> ones_indexer_key(compressed_count * index_head_dim, __float2half(1.0f));
    check_hip(hipMemcpy(d_query, ones_query.data(), ones_query.size() * sizeof(half), hipMemcpyHostToDevice), "copy indexer query");
    check_hip(hipMemcpy(d_weights, ones_weights.data(), ones_weights.size() * sizeof(float), hipMemcpyHostToDevice), "copy indexer weights");
    check_hip(hipMemcpy(d_indexer_key, ones_indexer_key.data(), ones_indexer_key.size() * sizeof(half), hipMemcpyHostToDevice), "copy indexer key");

    hipLaunchKernelGGL(
        aeon::kernel::v4_indexer_scores_kernel,
        dim3(1), dim3(32), 0, 0,
        d_query, d_weights, d_indexer_key, d_scores, compressed_count,
        aeon::kernel::DSV4_INDEX_N_HEADS, index_head_dim,
        1.0f / std::sqrt(128.0f), 1.0f / std::sqrt(64.0f));
    check_hip(hipGetLastError(), "launch indexer scores");
    float score = 0.0f;
    check_hip(hipMemcpy(&score, d_scores, sizeof(score), hipMemcpyDeviceToHost), "read indexer score");
    assert(score > 80.0f);

    std::vector<half> zero_query(aeon::kernel::DSV4_NUM_HEADS * head_dim, __float2half(0.0f));
    std::vector<half> zero_key(local_capacity * head_dim, __float2half(0.0f));
    std::vector<half> local_value(local_capacity * head_dim, __float2half(2.0f));
    std::vector<half> compressed_key(head_dim, __float2half(0.0f));
    std::vector<half> compressed_value(head_dim, __float2half(4.0f));
    std::vector<int64_t> local_positions{0, 1, 2, 3};
    std::vector<float> sink(aeon::kernel::DSV4_NUM_HEADS, 0.0f);
    const int64_t compressed_position = 3;
    check_hip(hipMemcpy(d_attention_query, zero_query.data(), zero_query.size() * sizeof(half), hipMemcpyHostToDevice), "copy attention query");
    check_hip(hipMemcpy(d_local_key, zero_key.data(), zero_key.size() * sizeof(half), hipMemcpyHostToDevice), "copy local key");
    check_hip(hipMemcpy(d_local_value, local_value.data(), local_value.size() * sizeof(half), hipMemcpyHostToDevice), "copy local value");
    check_hip(hipMemcpy(d_local_positions, local_positions.data(), local_positions.size() * sizeof(int64_t), hipMemcpyHostToDevice), "copy local positions");
    check_hip(hipMemcpy(d_sink, sink.data(), sink.size() * sizeof(float), hipMemcpyHostToDevice), "copy sink");
    check_hip(hipMemcpy(d_compressed_key, compressed_key.data(), compressed_key.size() * sizeof(half), hipMemcpyHostToDevice), "copy compressed key");
    check_hip(hipMemcpy(d_compressed_value, compressed_value.data(), compressed_value.size() * sizeof(half), hipMemcpyHostToDevice), "copy compressed value");
    check_hip(hipMemcpy(d_compressed_positions, &compressed_position, sizeof(compressed_position), hipMemcpyHostToDevice), "copy compressed position");

    hipLaunchKernelGGL(
        aeon::kernel::v4_cached_compressed_attention_wave32_kernel,
        dim3(aeon::kernel::DSV4_NUM_HEADS), dim3(32), 0, 0,
        d_attention_query, d_local_key, d_local_value, d_local_positions, d_sink,
        d_compressed_key, d_compressed_value, d_compressed_positions, nullptr,
        d_output, 3, local_capacity, compressed_count, 0, false,
        aeon::kernel::DSV4_ATTN_SCALE);
    check_hip(hipGetLastError(), "launch mixed attention");
    check_hip(hipDeviceSynchronize(), "synchronize mixed attention");
    std::vector<half> output(aeon::kernel::DSV4_NUM_HEADS * head_dim);
    check_hip(hipMemcpy(output.data(), d_output, output.size() * sizeof(half), hipMemcpyDeviceToHost), "read mixed attention");
    assert(std::abs(__half2float(output[0]) - 2.0f) < 2e-2f);

    check_hip(hipFree(d_query), "free query");
    check_hip(hipFree(d_weights), "free weights");
    check_hip(hipFree(d_indexer_key), "free indexer key");
    check_hip(hipFree(d_scores), "free scores");
    check_hip(hipFree(d_attention_query), "free attention query");
    check_hip(hipFree(d_local_key), "free local key");
    check_hip(hipFree(d_local_value), "free local value");
    check_hip(hipFree(d_local_positions), "free local positions");
    check_hip(hipFree(d_sink), "free sink");
    check_hip(hipFree(d_compressed_key), "free compressed key");
    check_hip(hipFree(d_compressed_value), "free compressed value");
    check_hip(hipFree(d_compressed_positions), "free compressed positions");
    check_hip(hipFree(d_output), "free output");
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);
    test_compressor_boundary(4, 1024, 8, 2);
    test_compressor_boundary(128, 512, 128, 1);
    test_indexer_and_mixed_attention();
    std::cout << "V4 class attention device primitives passed: C4/C128 boundaries, indexer scoring, and mixed attention" << std::endl;
    return 0;
}