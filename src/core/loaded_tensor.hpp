#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aeon::core {

struct LoadedTensor {
    const uint8_t* data{nullptr};
    int64_t byte_size{0};
    std::string dtype;
    std::vector<int64_t> shape;
};

} // namespace aeon::core