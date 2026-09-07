#include "core/safetensors.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "[Test] Parsing Safetensors header for model-00001.safetensors..." << std::endl;

    std::string path = "models/DeepSeek-V4-Flash-0731-INT4-W4A16/"
                        "models--yiminyuan--DeepSeek-V4-Flash-0731-INT4-W4A16/"
                        "snapshots/64700592cadaf205fe0c13202061ff4b45afbfd0/model-00001.safetensors";

    auto header = aeon::core::SafetensorsHeader::parse_file(path);

    std::cout << "Header bytes length: " << header.header_bytes_len << std::endl;
    std::cout << "Data base offset: " << header.data_base_offset << std::endl;
    std::cout << "Total parsed tensors: " << header.tensors.size() << std::endl;

    // Verify presence of expert 0 weights
    std::string w1_key = "layers.0.ffn.experts.0.w1.weight_packed";
    std::string w1_scale_key = "layers.0.ffn.experts.0.w1.weight_scale";

    assert(header.tensors.find(w1_key) != header.tensors.end());
    assert(header.tensors.find(w1_scale_key) != header.tensors.end());

    const auto& w1 = header.tensors[w1_key];
    std::cout << "Found " << w1_key << ": dtype=" << w1.dtype
              << " shape=[" << w1.shape[0] << ", " << w1.shape[1] << "]"
              << " bytes=" << w1.byte_size() << std::endl;

    assert(w1.dtype == "I32");
    assert(w1.shape.size() == 2);
    assert(w1.shape[0] == 2048);
    assert(w1.shape[1] == 512);
    assert(w1.byte_size() == 2048 * 512 * 4); // 4,194,304 bytes

    const auto& w1_scale = header.tensors[w1_scale_key];
    std::cout << "Found " << w1_scale_key << ": dtype=" << w1_scale.dtype
              << " shape=[" << w1_scale.shape[0] << ", " << w1_scale.shape[1] << "]"
              << " bytes=" << w1_scale.byte_size() << std::endl;

    assert(w1_scale.dtype == "F16");
    assert(w1_scale.shape[0] == 2048);
    assert(w1_scale.shape[1] == 128);
    assert(w1_scale.byte_size() == 2048 * 128 * 2); // 524,288 bytes

    std::cout << "[Test PASS] Safetensors header parsing verified!" << std::endl;
    return 0;
}
