#include "platform/rdna3/device.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "architecture/deepseek_v4/kernels/hc_sinkhorn.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
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

constexpr int kHiddenSize = 4096;
constexpr int kQueryRank = 1024;
constexpr int kHeadDim = 512;
constexpr int kOutputGroups = 8;
constexpr int kOutputRank = 1024;
constexpr int kOutputProjectionDim = kOutputGroups * kOutputRank;
constexpr int kIntermediateSize = 2048;
constexpr int kRouterExperts = 256;
constexpr int kHcStreams = 4;
constexpr int kHcMixes = 24;
constexpr int kHcFlatSize = kHcStreams * kHiddenSize;
constexpr float kEpsilon = 1e-6f;
constexpr float kProjectionAbsoluteTolerance = 0.005f;
constexpr float kProjectionRelativeTolerance = 0.001f;
constexpr float kNormAbsoluteTolerance = 0.002f;
constexpr float kNormRelativeTolerance = 0.001f;
constexpr float kHcAbsoluteTolerance = 0.002f;
constexpr float kHcRelativeTolerance = 0.001f;

struct HalfFixture {
    std::vector<half> values;
    std::vector<uint16_t> bits;
};

HalfFixture make_half_fixture(int size, int seed) {
    HalfFixture fixture;
    fixture.values.resize(size);
    fixture.bits.resize(size);
    for (int index = 0; index < size; ++index) {
        const float value = static_cast<float>((index * (seed + 5)) % 37 - 18) * 0.0078125f;
        fixture.values[index] = __float2half(value);
        std::memcpy(&fixture.bits[index], &fixture.values[index], sizeof(uint16_t));
    }
    return fixture;
}

float max_abs_reference(const std::vector<float>& values) {
    float maximum = 0.0f;
    for (float value : values) maximum = std::max(maximum, std::abs(value));
    return maximum;
}

void verify_float_close(
    const std::string& label,
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float absolute_tolerance,
    float relative_tolerance
) {
    assert(actual.size() == expected.size());
    float max_error = 0.0f;
    float squared_error = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float difference = actual[index] - expected[index];
        max_error = std::max(max_error, std::abs(difference));
        squared_error += difference * difference;
    }
    const float rms_error = std::sqrt(squared_error / static_cast<float>(actual.size()));
    const float allowed_error = absolute_tolerance +
        relative_tolerance * std::max(1.0f, max_abs_reference(expected));
    std::cout << "  " << label << " max_abs=" << std::setprecision(7) << max_error
              << " rms=" << rms_error << " allowed=" << allowed_error << std::endl;
    if (max_error > allowed_error) {
        std::cerr << "Stage 1 dense parity failure: " << label << std::endl;
        std::exit(1);
    }
}

void verify_half_close(
    const std::string& label,
    const std::vector<half>& actual,
    const std::vector<float>& expected,
    float absolute_tolerance,
    float relative_tolerance
) {
    std::vector<float> actual_float(actual.size());
    for (size_t index = 0; index < actual.size(); ++index) {
        actual_float[index] = __half2float(actual[index]);
    }
    verify_float_close(label, actual_float, expected,
                       absolute_tolerance, relative_tolerance);
}

const aeon::core::LoadedTensor& require_tensor(
    const aeon::core::AeonModelLoader& loader,
    const std::string& name,
    const std::vector<int64_t>& shape
) {
    assert(loader.has_tensor(name));
    const auto& tensor = loader.get_tensor(name);
    assert(tensor.dtype == "F16" || tensor.dtype == "F32");
    assert(tensor.shape == shape);
    return tensor;
}

void run_fp16_gemv(
    const aeon::core::AeonModelLoader& loader,
    const std::string& tensor_name,
    const HalfFixture& activation,
    int rows,
    int columns,
    const std::string& label,
    bool vectorized
) {
    const auto& tensor = require_tensor(loader, tensor_name, {rows, columns});
    std::vector<float> expected(rows);
    aeon::reference::decode_fp16_gemv(
        tensor.data, activation.bits.data(), rows, columns, expected.data());

    half* device_activation = nullptr;
    half* device_weight = nullptr;
    half* device_output = nullptr;
    CHECK_HIP(hipMalloc(&device_activation, activation.values.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_weight, tensor.byte_size));
    CHECK_HIP(hipMalloc(&device_output, expected.size() * sizeof(half)));
    CHECK_HIP(hipMemcpy(device_activation, activation.values.data(),
                        activation.values.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_weight, tensor.data, tensor.byte_size,
                        hipMemcpyHostToDevice));
    if (vectorized) {
        hipLaunchKernelGGL(
            aeon::kernel::v4_gemv_fp16_vec8_kernel,
            dim3(rows, 1, 1), dim3(32, 1, 1), 0, 0,
            device_activation, device_weight, device_output, columns);
    } else {
        hipLaunchKernelGGL(
            aeon::kernel::v4_gemv_fp16_kernel,
            dim3(rows, 1, 1), dim3(32, 1, 1), 0, 0,
            device_activation, device_weight, device_output, columns);
    }
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<half> actual(rows);
    CHECK_HIP(hipMemcpy(actual.data(), device_output,
                        actual.size() * sizeof(half), hipMemcpyDeviceToHost));
    verify_half_close(label, actual, expected,
                      kProjectionAbsoluteTolerance, kProjectionRelativeTolerance);
    CHECK_HIP(hipFree(device_activation));
    CHECK_HIP(hipFree(device_weight));
    CHECK_HIP(hipFree(device_output));
}

void run_rmsnorm(
    const aeon::core::AeonModelLoader& loader,
    const std::string& tensor_name,
    const HalfFixture& input,
    int dimension,
    const std::string& label
) {
    const auto& tensor = require_tensor(loader, tensor_name, {dimension});
    const auto* weight_bits = reinterpret_cast<const uint16_t*>(tensor.data);
    std::vector<float> expected(dimension);
    float sum_squared = 0.0f;
    for (int index = 0; index < dimension; ++index) {
        const float value = aeon::reference::fp16_to_float(input.bits[index]);
        sum_squared += value * value;
    }
    const float inverse_rms = 1.0f / std::sqrt(
        sum_squared / static_cast<float>(dimension) + kEpsilon);
    for (int index = 0; index < dimension; ++index) {
        expected[index] = aeon::reference::fp16_to_float(input.bits[index]) *
            inverse_rms * aeon::reference::fp16_to_float(weight_bits[index]);
    }

    half* device_input = nullptr;
    half* device_weight = nullptr;
    half* device_output = nullptr;
    CHECK_HIP(hipMalloc(&device_input, input.values.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_weight, tensor.byte_size));
    CHECK_HIP(hipMalloc(&device_output, expected.size() * sizeof(half)));
    CHECK_HIP(hipMemcpy(device_input, input.values.data(),
                        input.values.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_weight, tensor.data, tensor.byte_size,
                        hipMemcpyHostToDevice));
    hipLaunchKernelGGL(
        aeon::kernel::v4_rmsnorm_wave32_kernel,
        dim3(1, 1, 1), dim3(32, 1, 1), 0, 0,
        device_input, device_weight, device_output, dimension, kEpsilon);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<half> actual(dimension);
    CHECK_HIP(hipMemcpy(actual.data(), device_output,
                        actual.size() * sizeof(half), hipMemcpyDeviceToHost));
    verify_half_close(label, actual, expected,
                      kNormAbsoluteTolerance, kNormRelativeTolerance);
    CHECK_HIP(hipFree(device_input));
    CHECK_HIP(hipFree(device_weight));
    CHECK_HIP(hipFree(device_output));
}

void run_grouped_wo_a(
    const aeon::core::AeonModelLoader& loader,
    int layer_id,
    const HalfFixture& activation
) {
    const std::string tensor_name =
        "layers." + std::to_string(layer_id) + ".attn.wo_a.weight";
    const auto& tensor = require_tensor(
        loader, tensor_name, {kOutputProjectionDim, kHiddenSize});
    std::vector<float> expected(kOutputProjectionDim);
    for (int group = 0; group < kOutputGroups; ++group) {
        for (int row = 0; row < kOutputRank; ++row) {
            float dot = 0.0f;
            for (int column = 0; column < kHiddenSize; ++column) {
                const size_t weight_index =
                    (static_cast<size_t>(group * kOutputRank + row) * kHiddenSize + column) *
                    sizeof(uint16_t);
                dot += aeon::reference::fp16_to_float(activation.bits[
                    static_cast<size_t>(group) * kHiddenSize + column]) *
                    aeon::reference::fp16_to_float(
                        aeon::reference::load_u16(tensor.data + weight_index));
            }
            expected[static_cast<size_t>(group) * kOutputRank + row] = dot;
        }
    }

    half* device_activation = nullptr;
    half* device_weight = nullptr;
    half* device_output = nullptr;
    CHECK_HIP(hipMalloc(&device_activation, activation.values.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&device_weight, tensor.byte_size));
    CHECK_HIP(hipMalloc(&device_output, expected.size() * sizeof(half)));
    CHECK_HIP(hipMemcpy(device_activation, activation.values.data(),
                        activation.values.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_weight, tensor.data, tensor.byte_size,
                        hipMemcpyHostToDevice));
    hipLaunchKernelGGL(
        aeon::kernel::v4_grouped_wo_a_wave32_kernel,
        dim3(kOutputRank, kOutputGroups, 1), dim3(32, 1, 1), 0, 0,
        device_activation, device_weight, device_output, 1);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<half> actual(expected.size());
    CHECK_HIP(hipMemcpy(actual.data(), device_output,
                        actual.size() * sizeof(half), hipMemcpyDeviceToHost));
    verify_half_close("L" + std::to_string(layer_id) + " grouped wo_a",
                      actual, expected,
                      kProjectionAbsoluteTolerance, kProjectionRelativeTolerance);
    CHECK_HIP(hipFree(device_activation));
    CHECK_HIP(hipFree(device_weight));
    CHECK_HIP(hipFree(device_output));
}

void run_hc_projection(
    const aeon::core::AeonModelLoader& loader,
    int layer_id,
    const std::string& family,
    const std::vector<float>& residual
) {
    const std::string prefix = "layers." + std::to_string(layer_id) + ".hc_" + family;
    const auto& fn = require_tensor(loader, prefix + "_fn", {kHcMixes, kHcFlatSize});
    const auto& base = require_tensor(loader, prefix + "_base", {kHcMixes});
    const auto& scale = require_tensor(loader, prefix + "_scale", {3});
    const auto* fn_values = reinterpret_cast<const float*>(fn.data);
    const auto* base_values = reinterpret_cast<const float*>(base.data);
    const auto* scale_values = reinterpret_cast<const float*>(scale.data);

    std::vector<float> expected_mixes(kHcMixes, 0.0f);
    float squared_sum = 0.0f;
    for (float value : residual) squared_sum += value * value;
    const float inverse_rms = 1.0f / std::sqrt(
        squared_sum / static_cast<float>(kHcFlatSize) + kEpsilon);
    for (int mix = 0; mix < kHcMixes; ++mix) {
        float dot = 0.0f;
        for (int index = 0; index < kHcFlatSize; ++index) {
            dot += residual[index] * fn_values[mix * kHcFlatSize + index];
        }
        expected_mixes[mix] = dot * inverse_rms;
    }

    float* device_residual = nullptr;
    float* device_fn = nullptr;
    float* device_mixes = nullptr;
    float* device_base = nullptr;
    float* device_scale = nullptr;
    float* device_pre = nullptr;
    float* device_post = nullptr;
    float* device_comb = nullptr;
    half* device_layer_input = nullptr;
    CHECK_HIP(hipMalloc(&device_residual, residual.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_fn, fn.byte_size));
    CHECK_HIP(hipMalloc(&device_mixes, kHcMixes * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_base, base.byte_size));
    CHECK_HIP(hipMalloc(&device_scale, scale.byte_size));
    CHECK_HIP(hipMalloc(&device_pre, kHcStreams * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_post, kHcStreams * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_comb, kHcStreams * kHcStreams * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_layer_input, kHiddenSize * sizeof(half)));
    CHECK_HIP(hipMemcpy(device_residual, residual.data(),
                        residual.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_fn, fn.data, fn.byte_size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_base, base.data, base.byte_size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_scale, scale.data, scale.byte_size, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(
        aeon::kernel::hc_project_kernel,
        dim3(kHcMixes, 1, 1), dim3(256, 1, 1), 0, 0,
        device_residual, device_fn, device_mixes,
        kHiddenSize, kHcStreams, kEpsilon);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<float> actual_mixes(kHcMixes);
    CHECK_HIP(hipMemcpy(actual_mixes.data(), device_mixes,
                        actual_mixes.size() * sizeof(float), hipMemcpyDeviceToHost));
    verify_float_close("L" + std::to_string(layer_id) + " " + family + " HC projection",
                       actual_mixes, expected_mixes,
                       kHcAbsoluteTolerance, kHcRelativeTolerance);

    std::vector<float> expected_layer_input(kHiddenSize);
    std::vector<float> expected_post(kHcStreams);
    std::vector<float> expected_comb(kHcStreams * kHcStreams);
    aeon::kernel::cpu_sinkhorn_and_mix(
        residual.data(), fn_values, base_values, scale_values,
        expected_layer_input.data(), expected_post.data(), expected_comb.data(),
        kHiddenSize, kHcStreams, kEpsilon, kEpsilon, kEpsilon, 2.0f, 20);
    std::vector<float> expected_pre(kHcStreams);
    for (int index = 0; index < kHcStreams; ++index) {
        const float value = expected_mixes[index] * scale_values[0] + base_values[index];
        expected_pre[index] = 1.0f / (1.0f + std::exp(-value)) + kEpsilon;
    }

    hipLaunchKernelGGL(
        aeon::kernel::hc_sinkhorn_normalize_kernel,
        dim3(1, 1, 1), dim3(32, 1, 1), 0, 0,
        device_mixes, device_scale, device_base, device_pre, device_post, device_comb,
        kEpsilon, kEpsilon, 2.0f, 20);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<float> actual_pre(kHcStreams);
    std::vector<float> actual_post(kHcStreams);
    std::vector<float> actual_comb(kHcStreams * kHcStreams);
    CHECK_HIP(hipMemcpy(actual_pre.data(), device_pre,
                        actual_pre.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(actual_post.data(), device_post,
                        actual_post.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(actual_comb.data(), device_comb,
                        actual_comb.size() * sizeof(float), hipMemcpyDeviceToHost));
    verify_float_close("L" + std::to_string(layer_id) + " " + family + " HC pre",
                       actual_pre, expected_pre, kHcAbsoluteTolerance, kHcRelativeTolerance);
    verify_float_close("L" + std::to_string(layer_id) + " " + family + " HC post",
                       actual_post, expected_post, kHcAbsoluteTolerance, kHcRelativeTolerance);
    verify_float_close("L" + std::to_string(layer_id) + " " + family + " HC comb",
                       actual_comb, expected_comb, kHcAbsoluteTolerance, kHcRelativeTolerance);

    hipLaunchKernelGGL(
        aeon::kernel::hc_pre_combine_kernel,
        dim3((kHiddenSize + 1023) / 1024, 1, 1), dim3(256, 1, 1), 0, 0,
        device_residual, device_pre, device_layer_input, kHiddenSize, kHcStreams);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<half> actual_layer_input(kHiddenSize);
    CHECK_HIP(hipMemcpy(actual_layer_input.data(), device_layer_input,
                        actual_layer_input.size() * sizeof(half), hipMemcpyDeviceToHost));
    verify_half_close("L" + std::to_string(layer_id) + " " + family + " HC input",
                      actual_layer_input, expected_layer_input,
                      kHcAbsoluteTolerance, kHcRelativeTolerance);

    CHECK_HIP(hipFree(device_residual));
    CHECK_HIP(hipFree(device_fn));
    CHECK_HIP(hipFree(device_mixes));
    CHECK_HIP(hipFree(device_base));
    CHECK_HIP(hipFree(device_scale));
    CHECK_HIP(hipFree(device_pre));
    CHECK_HIP(hipFree(device_post));
    CHECK_HIP(hipFree(device_comb));
    CHECK_HIP(hipFree(device_layer_input));
}

void run_layer_dense_parity(
    const aeon::core::AeonModelLoader& loader,
    int layer_id
) {
    const std::string prefix = "layers." + std::to_string(layer_id) + ".";
    const HalfFixture hidden = make_half_fixture(kHiddenSize, layer_id + 1);
    const HalfFixture query = make_half_fixture(kQueryRank, layer_id + 3);
    const HalfFixture key_value = make_half_fixture(kHeadDim, layer_id + 5);
    const HalfFixture output_lora = make_half_fixture(kOutputProjectionDim, layer_id + 7);
    const HalfFixture intermediate = make_half_fixture(kIntermediateSize, layer_id + 9);
    const HalfFixture attention_output = make_half_fixture(
        kOutputGroups * kHiddenSize, layer_id + 11);

    run_rmsnorm(loader, prefix + "attn_norm.weight", hidden,
                kHiddenSize, "L" + std::to_string(layer_id) + " attention norm");
    run_fp16_gemv(loader, prefix + "attn.wq_a.weight", hidden,
                  kQueryRank, kHiddenSize,
                  "L" + std::to_string(layer_id) + " wq_a", false);
    run_rmsnorm(loader, prefix + "attn.q_norm.weight", query,
                kQueryRank, "L" + std::to_string(layer_id) + " q norm");
    run_fp16_gemv(loader, prefix + "attn.wq_b.weight", query,
                  kHiddenSize * 8, kQueryRank,
                  "L" + std::to_string(layer_id) + " wq_b", false);
    run_fp16_gemv(loader, prefix + "attn.wkv.weight", hidden,
                  kHeadDim, kHiddenSize,
                  "L" + std::to_string(layer_id) + " wkv", false);
    run_rmsnorm(loader, prefix + "attn.kv_norm.weight", key_value,
                kHeadDim, "L" + std::to_string(layer_id) + " kv norm");
    run_grouped_wo_a(loader, layer_id, attention_output);
    run_fp16_gemv(loader, prefix + "attn.wo_b.weight", output_lora,
                  kHiddenSize, kOutputProjectionDim,
                  "L" + std::to_string(layer_id) + " wo_b", false);
    run_rmsnorm(loader, prefix + "ffn_norm.weight", hidden,
                kHiddenSize, "L" + std::to_string(layer_id) + " ffn norm");
    run_fp16_gemv(loader, prefix + "ffn.shared_experts.w1.weight", hidden,
                  kIntermediateSize, kHiddenSize,
                  "L" + std::to_string(layer_id) + " shared w1", false);
    run_fp16_gemv(loader, prefix + "ffn.shared_experts.w3.weight", hidden,
                  kIntermediateSize, kHiddenSize,
                  "L" + std::to_string(layer_id) + " shared w3", false);
    run_fp16_gemv(loader, prefix + "ffn.shared_experts.w2.weight", intermediate,
                  kHiddenSize, kIntermediateSize,
                  "L" + std::to_string(layer_id) + " shared w2", false);
    run_fp16_gemv(loader, prefix + "ffn.gate.weight", hidden,
                  kRouterExperts, kHiddenSize,
                  "L" + std::to_string(layer_id) + " router", false);

    std::vector<float> residual(kHcFlatSize);
    for (size_t index = 0; index < residual.size(); ++index) {
        const int signed_pattern = static_cast<int>((index * (layer_id + 13)) % 41) - 20;
        residual[index] = static_cast<float>(signed_pattern) * 0.00390625f;
    }
    run_hc_projection(loader, layer_id, "attn", residual);
    run_hc_projection(loader, layer_id, "ffn", residual);
    std::cout << "[PASS] L" << layer_id << " dense projection parity" << std::endl;
}

void run_final_head_parity(
    const aeon::core::AeonModelLoader& loader,
    const HalfFixture& hidden
) {
    run_rmsnorm(loader, "norm.weight", hidden,
                kHiddenSize, "final norm");
    run_fp16_gemv(loader, "head.weight", hidden,
                  129280, kHiddenSize, "LM head", true);
}

} // namespace

int main() {
    aeon::core::select_compute_device(true);
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    aeon::core::AeonModelLoader loader;
    loader.open_model(model_dir);

    std::cout << "Stage 1 real dense parity: FP16 source tensors, float CPU accumulation, "
              << "projection tolerance " << kProjectionAbsoluteTolerance << " + "
              << kProjectionRelativeTolerance << " * max(1, |reference|), HC tolerance "
              << kHcAbsoluteTolerance << " + " << kHcRelativeTolerance
              << " * max(1, |reference|)" << std::endl;
    for (int layer_id : {0, 2, 3}) {
        run_layer_dense_parity(loader, layer_id);
    }
    run_final_head_parity(loader, make_half_fixture(kHiddenSize, 47));
    std::cout << "[PASS] Stage 1 independent real dense projection and output parity"
              << std::endl;
    return 0;
}