#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>

namespace aeon::core {

struct SafetensorItem {
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t offset_begin{0};
    int64_t offset_end{0};

    int64_t byte_size() const {
        return offset_end - offset_begin;
    }
};

class SafetensorsHeader {
public:
    int64_t header_bytes_len{0};
    int64_t data_base_offset{0};
    std::unordered_map<std::string, SafetensorItem> tensors;

    static SafetensorsHeader parse_file(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("SafetensorsHeader: Could not open " + filepath);
        }

        uint64_t header_len = 0;
        file.read(reinterpret_cast<char*>(&header_len), 8);
        if (!file) {
            throw std::runtime_error("SafetensorsHeader: Failed to read 8-byte header length from " + filepath);
        }

        SafetensorsHeader result;
        result.header_bytes_len = static_cast<int64_t>(header_len);
        result.data_base_offset = 8 + result.header_bytes_len;

        std::string json_str(result.header_bytes_len, '\0');
        file.read(&json_str[0], result.header_bytes_len);
        if (!file) {
            throw std::runtime_error("SafetensorsHeader: Failed to read header payload from " + filepath);
        }

        // Lightweight parser for Safetensors JSON header
        // Header looks like: { "tensor_name": { "dtype": "I32", "shape": [2048, 512], "data_offsets": [0, 4194304] }, ... }
        size_t idx = 0;
        const size_t len = json_str.size();

        while (idx < len) {
            // Find key opening quote
            size_t key_open = json_str.find('"', idx);
            if (key_open == std::string::npos) break;
            size_t key_close = json_str.find('"', key_open + 1);
            if (key_close == std::string::npos) break;
            std::string key = json_str.substr(key_open + 1, key_close - key_open - 1);

            // Find colon after key
            size_t colon = json_str.find(':', key_close + 1);
            if (colon == std::string::npos) break;

            // Check if value is an object or metadata string
            size_t next_non_ws = json_str.find_first_not_of(" \t\r\n", colon + 1);
            if (next_non_ws == std::string::npos) break;

            if (json_str[next_non_ws] != '{') {
                // E.g., "__metadata__": ... Skip until next comma or end
                idx = next_non_ws + 1;
                continue;
            }

            // Find matching closing brace for this tensor object
            size_t obj_start = next_non_ws;
            size_t obj_end = json_str.find('}', obj_start);
            if (obj_end == std::string::npos) break;

            std::string obj_text = json_str.substr(obj_start, obj_end - obj_start + 1);

            if (key != "__metadata__") {
                SafetensorItem item;

                // Extract dtype: "dtype": "..."
                size_t dtype_pos = obj_text.find("\"dtype\"");
                if (dtype_pos != std::string::npos) {
                    size_t d_q1 = obj_text.find('"', obj_text.find(':', dtype_pos) + 1);
                    size_t d_q2 = obj_text.find('"', d_q1 + 1);
                    if (d_q1 != std::string::npos && d_q2 != std::string::npos) {
                        item.dtype = obj_text.substr(d_q1 + 1, d_q2 - d_q1 - 1);
                    }
                }

                // Extract shape: "shape": [...]
                size_t shape_pos = obj_text.find("\"shape\"");
                if (shape_pos != std::string::npos) {
                    size_t b_open = obj_text.find('[', shape_pos);
                    size_t b_close = obj_text.find(']', b_open);
                    if (b_open != std::string::npos && b_close != std::string::npos) {
                        std::string dims = obj_text.substr(b_open + 1, b_close - b_open - 1);
                        std::stringstream ss(dims);
                        std::string dim_token;
                        while (std::getline(ss, dim_token, ',')) {
                            size_t s = dim_token.find_first_of("0123456789-");
                            if (s != std::string::npos) {
                                size_t e = dim_token.find_first_not_of("0123456789-", s);
                                int64_t d = std::stoll(dim_token.substr(s, e == std::string::npos ? std::string::npos : e - s));
                                item.shape.push_back(d);
                            }
                        }
                    }
                }

                // Extract data_offsets: "data_offsets": [begin, end]
                size_t off_pos = obj_text.find("\"data_offsets\"");
                if (off_pos != std::string::npos) {
                    size_t b_open = obj_text.find('[', off_pos);
                    size_t b_close = obj_text.find(']', b_open);
                    if (b_open != std::string::npos && b_close != std::string::npos) {
                        std::string offs = obj_text.substr(b_open + 1, b_close - b_open - 1);
                        size_t comma = offs.find(',');
                        if (comma != std::string::npos) {
                            item.offset_begin = std::stoll(offs.substr(0, comma));
                            item.offset_end = std::stoll(offs.substr(comma + 1));
                        }
                    }
                }

                result.tensors[key] = item;
            }

            idx = obj_end + 1;
        }

        return result;
    }
};

} // namespace aeon::core
