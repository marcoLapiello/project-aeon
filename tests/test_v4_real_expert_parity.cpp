#include "platform/rdna3/device.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_w4a16_swizzled_gemv.hpp"
#include "architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp"
#include "architecture/deepseek_v4/reference/v4_int4_reference.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_HIP(command) do { \
    hipError_t error = command; \
    if (error != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(error) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

namespace {

constexpr int kExpertCount = 6;
constexpr int kW13Rows = 2048;
constexpr int kW13Columns = 4096;
constexpr int kW2Rows = 4096;
constexpr int kW2Columns = 2048;
constexpr int kWaves = 8;
constexpr int kW13RowsPerWave = 4;
constexpr int kW13LanesPerRow = 8;
constexpr int kW2RowsPerWave = 8;
constexpr int kW2LanesPerRow = 4;
constexpr int kIterations = 16;
constexpr float kSwiGluLimit = 10.0f;
constexpr float kProjectionAbsoluteTolerance = 0.005f;
constexpr float kProjectionRelativeTolerance = 0.001f;
constexpr float kFfnAbsoluteTolerance = 0.01f;
constexpr float kFfnRelativeTolerance = 0.002f;

struct RepresentativeLayer {
    int layer_id;
    const char* label;
};

struct DeviceExpertWeights {
    uint32_t* w1{nullptr};
    half* s1{nullptr};
    uint32_t* w2{nullptr};
    half* s2{nullptr};
    uint32_t* w3{nullptr};
    half* s3{nullptr};
};

float max_abs_reference(const std::vector<float>& values) {
    float maximum = 0.0f;
    for (float value : values) maximum = std::max(maximum, std::abs(value));
    return maximum;
}

template <typename ActualT>
void verify_close(
    const std::string& label,
    const std::vector<ActualT>& actual,
    const std::vector<float>& expected,
    float absolute_tolerance,
    float relative_tolerance
) {
    assert(actual.size() == expected.size());
    float max_error = 0.0f;
    float squared_error = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float difference = __half2float(actual[index]) - expected[index];
        max_error = std::max(max_error, std::abs(difference));
        squared_error += difference * difference;
    }
    const float rms_error = std::sqrt(squared_error / static_cast<float>(actual.size()));
    const float allowed_error = absolute_tolerance +
        relative_tolerance * std::max(1.0f, max_abs_reference(expected));
    std::cout << "  " << label << " max_abs=" << std::setprecision(7) << max_error
              << " rms=" << rms_error << " allowed=" << allowed_error << std::endl;
    if (max_error > allowed_error) {
        std::cerr << "Stage 1 parity failure: " << label << std::endl;
        std::exit(1);
    }
}

void copy_expert_parts(
    const uint8_t* payload,
    std::vector<uint32_t>& w1,
    std::vector<half>& s1,
    std::vector<uint32_t>& w2,
    std::vector<half>& s2,
    std::vector<uint32_t>& w3,
    std::vector<half>& s3
) {
    std::memcpy(w1.data(), payload + aeon::core::AEON_W1_PACKED_OFFSET,
                aeon::core::AEON_W1_PACKED_BYTES);
    std::memcpy(s1.data(), payload + aeon::core::AEON_W1_SCALE_OFFSET,
                aeon::core::AEON_W1_SCALE_BYTES);
    std::memcpy(w2.data(), payload + aeon::core::AEON_W2_PACKED_OFFSET,
                aeon::core::AEON_W2_PACKED_BYTES);
    std::memcpy(s2.data(), payload + aeon::core::AEON_W2_SCALE_OFFSET,
                aeon::core::AEON_W2_SCALE_BYTES);
    std::memcpy(w3.data(), payload + aeon::core::AEON_W3_PACKED_OFFSET,
                aeon::core::AEON_W3_PACKED_BYTES);
    std::memcpy(s3.data(), payload + aeon::core::AEON_W3_SCALE_OFFSET,
                aeon::core::AEON_W3_SCALE_BYTES);
}

void free_device_weights(std::array<DeviceExpertWeights, kExpertCount>& weights) {
    for (DeviceExpertWeights& expert : weights) {
        if (expert.w1) CHECK_HIP(hipFree(expert.w1));
        if (expert.s1) CHECK_HIP(hipFree(expert.s1));
        if (expert.w2) CHECK_HIP(hipFree(expert.w2));
        if (expert.s2) CHECK_HIP(hipFree(expert.s2));
        if (expert.w3) CHECK_HIP(hipFree(expert.w3));
        if (expert.s3) CHECK_HIP(hipFree(expert.s3));
        expert = {};
    }
}

void verify_device_payload(
    const std::string& label,
    const void* host_data,
    const void* device_data,
    size_t byte_count
) {
    std::vector<uint8_t> roundtrip(byte_count);
    CHECK_HIP(hipMemcpy(roundtrip.data(), device_data, byte_count,
                        hipMemcpyDeviceToHost));
    if (std::memcmp(roundtrip.data(), host_data, byte_count) != 0) {
        std::cerr << "Stage 1 payload identity failure: " << label << std::endl;
        std::exit(1);
    }
}

void verify_layer(
    const aeon::core::AeonModelLoader& loader,
    const RepresentativeLayer& representative,
    const std::array<int, kExpertCount>& expert_ids,
    const std::vector<half>& activation,
    const std::vector<float>& activation_float,
    const std::vector<half>& w2_activation,
    const std::vector<float>& w2_activation_float
) {
    const size_t w13_packed_words =
        aeon::core::AEON_W1_PACKED_BYTES / sizeof(uint32_t);
    const size_t w13_scale_count =
        aeon::core::AEON_W1_SCALE_BYTES / sizeof(half);
    const size_t w2_packed_words =
        aeon::core::AEON_W2_PACKED_BYTES / sizeof(uint32_t);
    const size_t w2_scale_count =
        aeon::core::AEON_W2_SCALE_BYTES / sizeof(half);

    std::array<std::vector<uint32_t>, kExpertCount> host_w1;
    std::array<std::vector<half>, kExpertCount> host_s1;
    std::array<std::vector<uint32_t>, kExpertCount> host_w2;
    std::array<std::vector<half>, kExpertCount> host_s2;
    std::array<std::vector<uint32_t>, kExpertCount> host_w3;
    std::array<std::vector<half>, kExpertCount> host_s3;
    std::array<std::vector<float>, kExpertCount> reference_w1;
    std::array<std::vector<float>, kExpertCount> reference_w2;
    std::array<std::vector<float>, kExpertCount> reference_w3;
    std::array<std::vector<float>, kExpertCount> reference_hidden;
    std::array<std::vector<float>, kExpertCount> reference_ffn;

    for (int expert_index = 0; expert_index < kExpertCount; ++expert_index) {
        host_w1[expert_index].resize(w13_packed_words);
        host_s1[expert_index].resize(w13_scale_count);
        host_w2[expert_index].resize(w2_packed_words);
        host_s2[expert_index].resize(w2_scale_count);
        host_w3[expert_index].resize(w13_packed_words);
        host_s3[expert_index].resize(w13_scale_count);
        reference_w1[expert_index].resize(kW13Rows);
        reference_w2[expert_index].resize(kW2Rows);
        reference_w3[expert_index].resize(kW13Rows);
        reference_hidden[expert_index].resize(kW13Rows);
        reference_ffn[expert_index].resize(kW2Rows);

        const uint8_t* payload = loader.get_expert_data(
            static_cast<uint32_t>(representative.layer_id),
            static_cast<uint32_t>(expert_ids[expert_index]));
        copy_expert_parts(payload,
                          host_w1[expert_index], host_s1[expert_index],
                          host_w2[expert_index], host_s2[expert_index],
                          host_w3[expert_index], host_s3[expert_index]);
        aeon::reference::decode_gemv(
            payload, aeon::reference::SwizzledMatrixKind::W1,
            activation_float.data(), reference_w1[expert_index].data());
        aeon::reference::decode_gemv(
            payload, aeon::reference::SwizzledMatrixKind::W2,
            w2_activation_float.data(), reference_w2[expert_index].data());
        aeon::reference::decode_gemv(
            payload, aeon::reference::SwizzledMatrixKind::W3,
            activation_float.data(), reference_w3[expert_index].data());
        for (int row = 0; row < kW13Rows; ++row) {
            const float gate = std::min(reference_w1[expert_index][row], kSwiGluLimit);
            const float up = std::min(std::max(reference_w3[expert_index][row], -kSwiGluLimit), kSwiGluLimit);
            reference_hidden[expert_index][row] =
                (gate / (1.0f + std::exp(-gate))) * up;
        }
        aeon::reference::decode_routed_ffn(
            payload, activation_float.data(), reference_ffn[expert_index].data(), kSwiGluLimit);
    }

    half* device_activation = nullptr;
    half* device_w2_activation = nullptr;
    half* device_gate = nullptr;
    half* device_up = nullptr;
    half* device_w2_output = nullptr;
    half* device_expert_hidden = nullptr;
    float* device_output_f32 = nullptr;
    half* device_output_f16 = nullptr;
    float* device_topk_weights = nullptr;
    int* device_counters = nullptr;
    CHECK_HIP(hipMalloc(&device_activation, activation.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_w2_activation, w2_activation.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_gate, kW13Rows * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_up, kW13Rows * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_w2_output, kW2Rows * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_expert_hidden,
                       static_cast<size_t>(kExpertCount) * kW13Rows * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_output_f32, kW2Rows * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_output_f16, kW2Rows * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_topk_weights, kExpertCount * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_counters,
                       (kW2Rows / (kWaves * kW2RowsPerWave)) * sizeof(int)));
    CHECK_HIP(hipMemcpy(device_activation, activation.data(),
                        activation.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_w2_activation, w2_activation.data(),
                        w2_activation.size() * sizeof(half), hipMemcpyHostToDevice));

    std::array<DeviceExpertWeights, kExpertCount> device_storage;
    aeon::kernel::SwizzledW13ExpertPtrs w13_weights{};
    aeon::kernel::SwizzledW2ExpertPtrs w2_weights{};
    for (int expert_index = 0; expert_index < kExpertCount; ++expert_index) {
        DeviceExpertWeights& device_expert = device_storage[expert_index];
        CHECK_HIP(hipMalloc(&device_expert.w1, aeon::core::AEON_W1_PACKED_BYTES));
        CHECK_HIP(hipMalloc(&device_expert.s1, aeon::core::AEON_W1_SCALE_BYTES));
        CHECK_HIP(hipMalloc(&device_expert.w2, aeon::core::AEON_W2_PACKED_BYTES));
        CHECK_HIP(hipMalloc(&device_expert.s2, aeon::core::AEON_W2_SCALE_BYTES));
        CHECK_HIP(hipMalloc(&device_expert.w3, aeon::core::AEON_W3_PACKED_BYTES));
        CHECK_HIP(hipMalloc(&device_expert.s3, aeon::core::AEON_W3_SCALE_BYTES));
        CHECK_HIP(hipMemcpy(device_expert.w1, host_w1[expert_index].data(),
                            aeon::core::AEON_W1_PACKED_BYTES, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_expert.s1, host_s1[expert_index].data(),
                            aeon::core::AEON_W1_SCALE_BYTES, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_expert.w2, host_w2[expert_index].data(),
                            aeon::core::AEON_W2_PACKED_BYTES, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_expert.s2, host_s2[expert_index].data(),
                            aeon::core::AEON_W2_SCALE_BYTES, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_expert.w3, host_w3[expert_index].data(),
                            aeon::core::AEON_W3_PACKED_BYTES, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(device_expert.s3, host_s3[expert_index].data(),
                            aeon::core::AEON_W3_SCALE_BYTES, hipMemcpyHostToDevice));

        w13_weights.w1[expert_index] = reinterpret_cast<const uint4*>(device_expert.w1);
        w13_weights.s1[expert_index] = device_expert.s1;
        w13_weights.w3[expert_index] = reinterpret_cast<const uint4*>(device_expert.w3);
        w13_weights.s3[expert_index] = device_expert.s3;
        w2_weights.w2[expert_index] = reinterpret_cast<const uint4*>(device_expert.w2);
        w2_weights.s2[expert_index] = device_expert.s2;

        verify_device_payload("L" + std::to_string(representative.layer_id) +
                                  " E" + std::to_string(expert_ids[expert_index]) + " W1",
                              host_w1[expert_index].data(), device_expert.w1,
                              aeon::core::AEON_W1_PACKED_BYTES);
        verify_device_payload("L" + std::to_string(representative.layer_id) +
                                  " E" + std::to_string(expert_ids[expert_index]) + " W1 scale",
                              host_s1[expert_index].data(), device_expert.s1,
                              aeon::core::AEON_W1_SCALE_BYTES);
        verify_device_payload("L" + std::to_string(representative.layer_id) +
                                  " E" + std::to_string(expert_ids[expert_index]) + " W2",
                              host_w2[expert_index].data(), device_expert.w2,
                              aeon::core::AEON_W2_PACKED_BYTES);
        verify_device_payload("L" + std::to_string(representative.layer_id) +
                                  " E" + std::to_string(expert_ids[expert_index]) + " W2 scale",
                              host_s2[expert_index].data(), device_expert.s2,
                              aeon::core::AEON_W2_SCALE_BYTES);
        verify_device_payload("L" + std::to_string(representative.layer_id) +
                                  " E" + std::to_string(expert_ids[expert_index]) + " W3",
                              host_w3[expert_index].data(), device_expert.w3,
                              aeon::core::AEON_W3_PACKED_BYTES);
        verify_device_payload("L" + std::to_string(representative.layer_id) +
                                  " E" + std::to_string(expert_ids[expert_index]) + " W3 scale",
                              host_s3[expert_index].data(), device_expert.s3,
                              aeon::core::AEON_W3_SCALE_BYTES);
    }

    std::vector<half> native_projection(kW13Rows);
    std::vector<half> native_w2_projection(kW2Rows);
    for (int expert_index = 0; expert_index < kExpertCount; ++expert_index) {
        aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<
            kWaves, kW13RowsPerWave, kW13LanesPerRow, kIterations>(
            device_activation, device_storage[expert_index].w1,
            device_storage[expert_index].s1, device_gate, kW13Rows, kW13Columns);
        aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<
            kWaves, kW13RowsPerWave, kW13LanesPerRow, kIterations>(
            device_activation, device_storage[expert_index].w3,
            device_storage[expert_index].s3, device_up, kW13Rows, kW13Columns);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(native_projection.data(), device_gate,
                            native_projection.size() * sizeof(half), hipMemcpyDeviceToHost));
        verify_close("L" + std::to_string(representative.layer_id) +
                         " E" + std::to_string(expert_ids[expert_index]) + " W1",
                     native_projection, reference_w1[expert_index],
                     kProjectionAbsoluteTolerance, kProjectionRelativeTolerance);
        CHECK_HIP(hipMemcpy(native_projection.data(), device_up,
                            native_projection.size() * sizeof(half), hipMemcpyDeviceToHost));
        verify_close("L" + std::to_string(representative.layer_id) +
                         " E" + std::to_string(expert_ids[expert_index]) + " W3",
                     native_projection, reference_w3[expert_index],
                     kProjectionAbsoluteTolerance, kProjectionRelativeTolerance);

        aeon::kernel::dispatch_aeon_w4a16_swizzled_gemv<
            kWaves, kW2RowsPerWave, kW2LanesPerRow, kIterations>(
            device_w2_activation, device_storage[expert_index].w2,
            device_storage[expert_index].s2, device_w2_output, kW2Rows, kW2Columns);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(native_w2_projection.data(), device_w2_output,
                            native_w2_projection.size() * sizeof(half), hipMemcpyDeviceToHost));
        verify_close("L" + std::to_string(representative.layer_id) +
                         " E" + std::to_string(expert_ids[expert_index]) + " W2",
                     native_w2_projection, reference_w2[expert_index],
                     kProjectionAbsoluteTolerance, kProjectionRelativeTolerance);
    }

    const std::array<float, kExpertCount> route_weights = {
        0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.25f
    };
    CHECK_HIP(hipMemcpy(device_topk_weights, route_weights.data(),
                        route_weights.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(device_expert_hidden, 0,
                        static_cast<size_t>(kExpertCount) * kW13Rows * sizeof(half)));
    aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<
        kWaves, kW13RowsPerWave, kW13LanesPerRow, kIterations>(
        device_activation, w13_weights, device_expert_hidden,
        nullptr, 0, kExpertCount, kW13Rows, kW13Columns, kSwiGluLimit);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<half> native_hidden(static_cast<size_t>(kExpertCount) * kW13Rows);
    CHECK_HIP(hipMemcpy(native_hidden.data(), device_expert_hidden,
                        native_hidden.size() * sizeof(half), hipMemcpyDeviceToHost));
    for (int expert_index = 0; expert_index < kExpertCount; ++expert_index) {
        std::vector<half> hidden_slice(
            native_hidden.begin() + static_cast<size_t>(expert_index) * kW13Rows,
            native_hidden.begin() + static_cast<size_t>(expert_index + 1) * kW13Rows);
        verify_close("L" + std::to_string(representative.layer_id) +
                         " E" + std::to_string(expert_ids[expert_index]) + " SwiGLU",
                     hidden_slice, reference_hidden[expert_index],
                     kFfnAbsoluteTolerance, kFfnRelativeTolerance);
    }

    CHECK_HIP(hipMemset(device_output_f32, 0, kW2Rows * sizeof(float)));
    CHECK_HIP(hipMemset(device_output_f16, 0, kW2Rows * sizeof(half)));
    CHECK_HIP(hipMemset(device_counters, 0,
                        (kW2Rows / (kWaves * kW2RowsPerWave)) * sizeof(int)));
    aeon::kernel::dispatch_aeon_moe_fused_w2_accum<
        kWaves, kW2RowsPerWave, kW2LanesPerRow, kIterations>(
        device_expert_hidden, w2_weights, device_topk_weights, nullptr,
        device_output_f32, device_output_f16, device_counters,
        kExpertCount, kW2Rows, kW2Columns);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<float> reference_routed(kW2Rows, 0.0f);
    for (int expert_index = 0; expert_index < kExpertCount; ++expert_index) {
        for (int row = 0; row < kW2Rows; ++row) {
            reference_routed[row] += route_weights[expert_index] *
                reference_ffn[expert_index][row];
        }
    }
    std::vector<half> native_routed(kW2Rows);
    CHECK_HIP(hipMemcpy(native_routed.data(), device_output_f16,
                        native_routed.size() * sizeof(half), hipMemcpyDeviceToHost));
    verify_close("L" + std::to_string(representative.layer_id) +
                     " weighted routed FFN",
                 native_routed, reference_routed,
                 kFfnAbsoluteTolerance, kFfnRelativeTolerance);

    free_device_weights(device_storage);
    CHECK_HIP(hipFree(device_activation));
    CHECK_HIP(hipFree(device_w2_activation));
    CHECK_HIP(hipFree(device_gate));
    CHECK_HIP(hipFree(device_up));
    CHECK_HIP(hipFree(device_w2_output));
    CHECK_HIP(hipFree(device_expert_hidden));
    CHECK_HIP(hipFree(device_output_f32));
    CHECK_HIP(hipFree(device_output_f16));
    CHECK_HIP(hipFree(device_topk_weights));
    CHECK_HIP(hipFree(device_counters));
    std::cout << "[PASS] " << representative.label << " real expert parity" << std::endl;
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);

    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    aeon::core::AeonModelLoader loader;
    loader.open_model(model_dir);

    const std::array<RepresentativeLayer, 3> representatives = {{
        {0, "early"},
        {21, "middle"},
        {42, "late"}
    }};
    const std::array<int, kExpertCount> expert_ids = {0, 1, 7, 31, 127, 255};
    std::vector<half> activation(kW13Columns);
    std::vector<float> activation_float(kW13Columns);
    std::vector<half> w2_activation(kW2Columns);
    std::vector<float> w2_activation_float(kW2Columns);
    for (int index = 0; index < kW13Columns; ++index) {
        const float value = static_cast<float>((index * 17) % 31 - 15) * 0.0078125f;
        activation[index] = __float2half(value);
        activation_float[index] = __half2float(activation[index]);
    }
    for (int index = 0; index < kW2Columns; ++index) {
        const float value = static_cast<float>((index * 11) % 23 - 11) * 0.015625f;
        w2_activation[index] = __float2half(value);
        w2_activation_float[index] = __half2float(w2_activation[index]);
    }

    std::cout << "Stage 1 real expert parity: version-2 swizzled W4A16, "
              << "fixed FP16 activations, projection tolerance "
              << kProjectionAbsoluteTolerance << " + "
              << kProjectionRelativeTolerance << " * max(1, |reference|), "
              << "FFN tolerance " << kFfnAbsoluteTolerance << " + "
              << kFfnRelativeTolerance << " * max(1, |reference|)" << std::endl;
    for (const RepresentativeLayer& representative : representatives) {
        verify_layer(loader, representative, expert_ids, activation, activation_float,
                     w2_activation, w2_activation_float);
    }
    std::cout << "[PASS] Stage 1 independent real-expert W1/W2/W3 and routed FFN parity"
              << std::endl;
    return 0;
}