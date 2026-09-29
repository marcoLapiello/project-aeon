#pragma once

// -----------------------------------------------------------------------------
// Routing-profile JSON and atomic-file helpers: the JSONL field parser, the
// JSON escaper, the binary string/trivial writers, the atomic rename/write
// helpers, the FNV-1a hashes, and the JSON-string field extractor. Split out of
// `routing_profile.hpp`, which includes this header for its prompt parsing and
// its store.
//
// All of this lives in `namespace routing_profile_detail`; `routing_profile.hpp`
// uses it with `using namespace routing_profile_detail;`.
// -----------------------------------------------------------------------------

#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace aeon::core {

namespace routing_profile_detail {

inline void skip_whitespace(std::string_view input, size_t& cursor) {
    while (cursor < input.size()) {
        const unsigned char ch = static_cast<unsigned char>(input[cursor]);
        if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
            break;
        }
        ++cursor;
    }
}

inline void expect(std::string_view input, size_t& cursor, char expected) {
    skip_whitespace(input, cursor);
    if (cursor >= input.size() || input[cursor] != expected) {
        throw std::runtime_error("invalid JSONL record: expected '" + std::string(1, expected) + "'");
    }
    ++cursor;
}

inline uint32_t parse_uint32(std::string_view input, size_t& cursor) {
    skip_whitespace(input, cursor);
    const size_t begin = cursor;
    while (cursor < input.size() && input[cursor] >= '0' && input[cursor] <= '9') {
        ++cursor;
    }
    if (begin == cursor) {
        throw std::runtime_error("invalid JSONL record: expected unsigned integer");
    }

    uint32_t value = 0;
    const char* first = input.data() + begin;
    const char* last = input.data() + cursor;
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last) {
        throw std::runtime_error("invalid JSONL record: integer is out of range");
    }
    return value;
}

inline void append_utf8(std::string& output, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

inline std::string parse_string(std::string_view input, size_t& cursor) {
    skip_whitespace(input, cursor);
    if (cursor >= input.size() || input[cursor] != '"') {
        throw std::runtime_error("invalid JSONL record: expected string");
    }
    ++cursor;

    std::string value;
    while (cursor < input.size()) {
        const unsigned char ch = static_cast<unsigned char>(input[cursor++]);
        if (ch == '"') {
            return value;
        }
        if (ch < 0x20) {
            throw std::runtime_error("invalid JSONL record: control character in string");
        }
        if (ch != '\\') {
            value.push_back(static_cast<char>(ch));
            continue;
        }
        if (cursor >= input.size()) {
            throw std::runtime_error("invalid JSONL record: incomplete escape");
        }
        const char escaped = input[cursor++];
        switch (escaped) {
        case '"': value.push_back('"'); break;
        case '\\': value.push_back('\\'); break;
        case '/': value.push_back('/'); break;
        case 'b': value.push_back('\b'); break;
        case 'f': value.push_back('\f'); break;
        case 'n': value.push_back('\n'); break;
        case 'r': value.push_back('\r'); break;
        case 't': value.push_back('\t'); break;
        case 'u': {
            if (cursor + 4 > input.size()) {
                throw std::runtime_error("invalid JSONL record: incomplete unicode escape");
            }
            uint32_t codepoint = 0;
            for (size_t i = 0; i < 4; ++i) {
                const char digit = input[cursor++];
                codepoint <<= 4;
                if (digit >= '0' && digit <= '9') {
                    codepoint |= static_cast<uint32_t>(digit - '0');
                } else if (digit >= 'a' && digit <= 'f') {
                    codepoint |= static_cast<uint32_t>(digit - 'a' + 10);
                } else if (digit >= 'A' && digit <= 'F') {
                    codepoint |= static_cast<uint32_t>(digit - 'A' + 10);
                } else {
                    throw std::runtime_error("invalid JSONL record: malformed unicode escape");
                }
            }
            if (codepoint >= 0xd800 && codepoint <= 0xdfff) {
                throw std::runtime_error("invalid JSONL record: surrogate escape is unsupported");
            }
            append_utf8(value, codepoint);
            break;
        }
        default:
            throw std::runtime_error("invalid JSONL record: unsupported escape");
        }
    }
    throw std::runtime_error("invalid JSONL record: unterminated string");
}

inline std::vector<uint32_t> parse_token_array(std::string_view input, size_t& cursor) {
    expect(input, cursor, '[');
    skip_whitespace(input, cursor);
    std::vector<uint32_t> tokens;
    if (cursor < input.size() && input[cursor] == ']') {
        ++cursor;
        return tokens;
    }

    while (true) {
        tokens.push_back(parse_uint32(input, cursor));
        skip_whitespace(input, cursor);
        if (cursor >= input.size()) {
            throw std::runtime_error("invalid JSONL record: unterminated token array");
        }
        if (input[cursor] == ']') {
            ++cursor;
            return tokens;
        }
        expect(input, cursor, ',');
    }
}

inline std::string json_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (unsigned char ch : value) {
        switch (ch) {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\b': escaped += "\\b"; break;
        case '\f': escaped += "\\f"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (ch < 0x20) {
                std::ostringstream code;
                code << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                     << static_cast<unsigned int>(ch);
                escaped += code.str();
            } else {
                escaped.push_back(static_cast<char>(ch));
            }
            break;
        }
    }
    return escaped;
}

inline void write_string(std::ostream& output, std::string_view value) {
    const uint64_t size = value.size();
    output.write(reinterpret_cast<const char*>(&size), sizeof(size));
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!output) {
        throw std::runtime_error("failed to write routing profile state");
    }
}

inline std::string read_string(std::istream& input) {
    uint64_t size = 0;
    input.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!input || size > 16 * 1024 * 1024) {
        throw std::runtime_error("invalid routing profile state string");
    }
    std::string value(size, '\0');
    input.read(value.data(), static_cast<std::streamsize>(size));
    if (!input) {
        throw std::runtime_error("truncated routing profile state");
    }
    return value;
}

template <typename Value>
inline void write_value(std::ostream& output, const Value& value) {
    static_assert(std::is_trivially_copyable_v<Value>);
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error("failed to write routing profile state");
    }
}

template <typename Value>
inline Value read_value(std::istream& input) {
    static_assert(std::is_trivially_copyable_v<Value>);
    Value value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("truncated routing profile state");
    }
    return value;
}

inline void rename_atomic(const std::filesystem::path& temporary, const std::filesystem::path& target) {
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("failed to publish routing profile file " + target.string() + ": " + error.message());
    }
}

inline void write_text_atomic(const std::filesystem::path& path, std::string_view content) {
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("failed to open routing profile file " + temporary);
        }
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        output.flush();
        if (!output) {
            throw std::runtime_error("failed to write routing profile file " + temporary);
        }
    }
    rename_atomic(temporary, path);
}

template <typename Writer>
inline void write_binary_atomic(const std::filesystem::path& path, Writer&& writer) {
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("failed to open routing profile file " + temporary);
        }
        writer(output);
        output.flush();
        if (!output) {
            throw std::runtime_error("failed to write routing profile file " + temporary);
        }
    }
    rename_atomic(temporary, path);
}

inline uint64_t fnv1a_bytes(const void* data, size_t size, uint64_t hash = 1469598103934665603ULL) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline uint64_t fnv1a_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return 0;
    }
    uint64_t hash = 1469598103934665603ULL;
    std::array<char, 8192> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            hash = fnv1a_bytes(buffer.data(), static_cast<size_t>(count), hash);
        }
    }
    return hash;
}

inline std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open routing profile file " + path.string());
    }
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

inline std::string extract_json_string_field(std::string_view object, std::string_view field) {
    const std::string marker = "\"" + std::string(field) + "\"";
    const size_t field_pos = object.find(marker);
    if (field_pos == std::string_view::npos) {
        throw std::runtime_error("routing profile metadata is missing " + std::string(field));
    }
    size_t cursor = field_pos + marker.size();
    expect(object, cursor, ':');
    return parse_string(object, cursor);
}

} // namespace routing_profile_detail

} // namespace aeon::core
