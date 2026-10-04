#pragma once

// -----------------------------------------------------------------------------
// Raw top-level member extraction.
//
// The shared `infrastructure/json.hpp` sorts object keys. Tool schemas and
// `response_format` must reach the prompt encoder with their key order intact —
// the encoder re-serializes them in reference form — so those two members are
// passed through as their exact source text rather than parsed and rebuilt.
//
// This scans only the top level of the body object and tracks string/escape state
// and `{`/`[` depth, so a nested `"tools"` key, a brace inside a string, or an
// escaped quote can never be mistaken for the member.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace aeon::server {

namespace detail {

inline bool json_space(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

// Decodes a JSON string literal (including its quotes) into its value. Handles the
// standard escapes and \uXXXX (BMP only); used to compare keys, which clients may
// spell with escapes.
inline std::string decode_json_string(std::string_view literal) {
    std::string out;
    if (literal.size() < 2 || literal.front() != '"') return out;
    size_t index = 1;
    const size_t end = literal.size() - 1;
    const auto append_utf8 = [&out](uint32_t code) {
        if (code <= 0x7F) {
            out.push_back(static_cast<char>(code));
        } else if (code <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    };
    const auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };
    while (index < end) {
        const char c = literal[index];
        if (c != '\\') {
            out.push_back(c);
            ++index;
            continue;
        }
        if (index + 1 >= end) break;
        const char escape = literal[index + 1];
        index += 2;
        switch (escape) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (index + 4 > end) break;
                uint32_t code = (hex(literal[index]) << 12) | (hex(literal[index + 1]) << 8) |
                                (hex(literal[index + 2]) << 4) | hex(literal[index + 3]);
                index += 4;
                if (code >= 0xD800 && code <= 0xDBFF && index + 6 <= end &&
                    literal[index] == '\\' && literal[index + 1] == 'u') {
                    const uint32_t low = (hex(literal[index + 2]) << 12) |
                                         (hex(literal[index + 3]) << 8) |
                                         (hex(literal[index + 4]) << 4) | hex(literal[index + 5]);
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        index += 6;
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    }
                }
                append_utf8(code);
                break;
            }
            default: break;
        }
    }
    return out;
}

// The index just past the value starting at `index` (which must point at its first
// non-space byte).
inline size_t skip_value(std::string_view body, size_t index) {
    if (index >= body.size()) return index;
    const char first = body[index];
    if (first == '"') {
        size_t j = index + 1;
        while (j < body.size() && body[j] != '"') {
            if (body[j] == '\\') ++j;
            ++j;
        }
        if (j < body.size()) ++j;
        return j;
    }
    if (first == '{' || first == '[') {
        int depth = 0;
        bool in_string = false;
        size_t j = index;
        for (; j < body.size(); ++j) {
            const char c = body[j];
            if (in_string) {
                if (c == '\\') { ++j; continue; }
                if (c == '"') in_string = false;
                continue;
            }
            if (c == '"') { in_string = true; continue; }
            if (c == '{' || c == '[') ++depth;
            else if (c == '}' || c == ']') {
                --depth;
                if (depth == 0) { ++j; break; }
            }
        }
        return j;
    }
    size_t j = index;
    while (j < body.size()) {
        const char c = body[j];
        if (c == ',' || c == '}' || c == ']' || json_space(c)) break;
        ++j;
    }
    return j;
}

}  // namespace detail

// The exact source text of the top-level member `key` in the JSON object `body`,
// or `std::nullopt` when it is absent (or `body` is not a top-level object).
inline std::optional<std::string_view> raw_member(std::string_view body, std::string_view key) {
    size_t index = 0;
    while (index < body.size() && detail::json_space(body[index])) ++index;
    if (index >= body.size() || body[index] != '{') return std::nullopt;
    ++index;

    while (index < body.size()) {
        while (index < body.size() && detail::json_space(body[index])) ++index;
        if (index >= body.size()) return std::nullopt;
        if (body[index] == '}') return std::nullopt;
        if (body[index] == ',') { ++index; continue; }
        if (body[index] != '"') return std::nullopt;

        const size_t key_start = index;
        ++index;
        while (index < body.size() && body[index] != '"') {
            if (body[index] == '\\') ++index;
            ++index;
        }
        if (index >= body.size()) return std::nullopt;
        ++index;  // closing quote
        const std::string_view key_literal =
            body.substr(key_start, index - key_start);

        while (index < body.size() && detail::json_space(body[index])) ++index;
        if (index >= body.size() || body[index] != ':') return std::nullopt;
        ++index;
        while (index < body.size() && detail::json_space(body[index])) ++index;
        const size_t value_start = index;
        const size_t value_end = detail::skip_value(body, index);
        if (value_end > body.size()) return std::nullopt;

        if (detail::decode_json_string(key_literal) == key) {
            return body.substr(value_start, value_end - value_start);
        }
        index = value_end;
    }
    return std::nullopt;
}

}  // namespace aeon::server
