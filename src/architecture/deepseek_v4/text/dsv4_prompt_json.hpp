#pragma once

// Order-preserving JSON codec for the DeepSeek-V4 prompt encoder.
//
// The encoder must reproduce the reference `json.dumps(value, ensure_ascii=False)`
// byte-for-byte, because tool schemas are embedded in the prompt text. That needs
// two properties `infrastructure/core/json.hpp` does not have:
//
//   1. **Insertion order.** Its object type is `std::map`, which sorts keys.
//      Python dicts preserve insertion order, and a tool schema like
//      `{"type","properties","required"}` re-serialized sorted would not match.
//   2. **Integers are not reals.** Its numbers are all `double`, so `-1` and
//      `-1.0` are indistinguishable; Python emits `-1` for the first and `-1.0`
//      for the second.
//
// So this is a separate, small codec used only for prompt text. It is deliberately
// not a general-purpose JSON facility and does not replace `core/json.hpp`.
//
// Representation note: an object keeps its keys and values in two parallel
// vectors rather than `vector<pair<string, PromptJson>>`. A `std::pair` needs its
// members complete, and `PromptJson` is necessarily incomplete inside itself;
// `std::vector` is the container the standard permits to hold an incomplete type.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace aeon::text {

class PromptJson {
public:
    enum class Kind { Null, Bool, Int, Real, String, Array, Object };

    using Array = std::vector<PromptJson>;

    Kind kind{Kind::Null};
    bool boolean{false};
    int64_t integer{0};
    double real{0.0};
    std::string text;
    Array array;
    std::vector<std::string> keys;   // object keys, insertion order
    Array values;                    // object values, parallel to `keys`

    static PromptJson parse(std::string_view input);
    static PromptJson parse_object(std::string_view input);  // requires an object

    bool is_object() const noexcept { return kind == Kind::Object; }
    bool is_string() const noexcept { return kind == Kind::String; }
    bool is_array() const noexcept { return kind == Kind::Array; }
    bool is_int() const noexcept { return kind == Kind::Int; }

    size_t size() const noexcept { return keys.size(); }
    const std::string& key_at(size_t index) const { return keys[index]; }
    const PromptJson& value_at(size_t index) const { return values[index]; }

    const PromptJson* find(std::string_view key) const noexcept;

    // Serialize the way Python's `json.dumps(value, ensure_ascii=False)` does:
    // separators `", "` and `": "`, insertion order preserved, non-ASCII emitted
    // raw, integers without a decimal point.
    std::string to_python_json() const;

private:
    class Parser {
    public:
        explicit Parser(std::string_view input) : input_(input) {}

        PromptJson parse_document();
        PromptJson parse_value();

    private:
        void skip_whitespace();
        PromptJson parse_object();
        PromptJson parse_array();
        PromptJson parse_string();
        PromptJson parse_number();
        PromptJson parse_literal();
        std::string parse_string_body();
        void expect(char expected);

        [[noreturn]] void fail(const std::string& message) const {
            throw std::runtime_error("PromptJson: " + message + " at offset " +
                                     std::to_string(position_));
        }

        std::string_view input_;
        size_t position_{0};
    };
};

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

namespace detail {

inline void append_python_string(std::string& out, std::string_view value) {
    out.push_back('"');
    for (const char raw : value) {
        const auto byte = static_cast<unsigned char>(raw);
        switch (byte) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (byte < 0x20) {
                    static const char* digits = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(digits[(byte >> 4) & 0xF]);
                    out.push_back(digits[byte & 0xF]);
                } else {
                    out.push_back(raw);
                }
                break;
        }
    }
    out.push_back('"');
}

// Shortest decimal form that round-trips, matching Python's float repr closely
// enough for the values that occur in prompt text.
inline std::string python_real(double value) {
    for (int precision : {15, 16, 17}) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
        if (std::strtod(buffer, nullptr) == value) {
            std::string result(buffer);
            if (result.find('.') == std::string::npos && result.find('e') == std::string::npos &&
                result.find("inf") == std::string::npos &&
                result.find("nan") == std::string::npos) {
                result += ".0";
            }
            return result;
        }
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return buffer;
}

} // namespace detail

inline std::string PromptJson::to_python_json() const {
    std::string out;
    switch (kind) {
        case Kind::Null:
            return "null";
        case Kind::Bool:
            return boolean ? "true" : "false";
        case Kind::Int:
            return std::to_string(integer);
        case Kind::Real:
            return detail::python_real(real);
        case Kind::String:
            detail::append_python_string(out, text);
            return out;
        case Kind::Array:
            out.push_back('[');
            for (size_t i = 0; i < array.size(); ++i) {
                if (i != 0) out += ", ";
                out += array[i].to_python_json();
            }
            out.push_back(']');
            return out;
        case Kind::Object:
            out.push_back('{');
            for (size_t i = 0; i < keys.size(); ++i) {
                if (i != 0) out += ", ";
                detail::append_python_string(out, keys[i]);
                out += ": ";
                out += values[i].to_python_json();
            }
            out.push_back('}');
            return out;
    }
    return out;
}

inline const PromptJson* PromptJson::find(std::string_view key) const noexcept {
    if (kind != Kind::Object) return nullptr;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] == key) return &values[i];
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

inline void PromptJson::Parser::skip_whitespace() {
    while (position_ < input_.size()) {
        const char c = input_[position_];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ++position_;
        } else {
            break;
        }
    }
}

inline void PromptJson::Parser::expect(char expected) {
    if (position_ >= input_.size() || input_[position_] != expected) {
        fail(std::string("expected '") + expected + "'");
    }
    ++position_;
}

inline PromptJson PromptJson::Parser::parse_document() {
    skip_whitespace();
    PromptJson value = parse_value();
    skip_whitespace();
    if (position_ != input_.size()) fail("trailing characters after document");
    return value;
}

inline PromptJson PromptJson::Parser::parse_value() {
    skip_whitespace();
    if (position_ >= input_.size()) fail("unexpected end of input");
    const char c = input_[position_];
    switch (c) {
        case '{': return parse_object();
        case '[': return parse_array();
        case '"': return parse_string();
        case 't':
        case 'f':
        case 'n': return parse_literal();
        default:
            if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
            fail("unexpected character");
    }
}

inline PromptJson PromptJson::Parser::parse_object() {
    expect('{');
    PromptJson value;
    value.kind = Kind::Object;
    skip_whitespace();
    if (position_ < input_.size() && input_[position_] == '}') {
        ++position_;
        return value;
    }
    while (true) {
        skip_whitespace();
        if (position_ >= input_.size() || input_[position_] != '"') {
            fail("expected object key");
        }
        std::string key = parse_string_body();
        skip_whitespace();
        expect(':');
        PromptJson member = parse_value();
        value.keys.push_back(std::move(key));
        value.values.push_back(std::move(member));
        skip_whitespace();
        if (position_ >= input_.size()) fail("unterminated object");
        if (input_[position_] == ',') {
            ++position_;
            continue;
        }
        if (input_[position_] == '}') {
            ++position_;
            return value;
        }
        fail("expected ',' or '}' in object");
    }
}

inline PromptJson PromptJson::Parser::parse_array() {
    expect('[');
    PromptJson value;
    value.kind = Kind::Array;
    skip_whitespace();
    if (position_ < input_.size() && input_[position_] == ']') {
        ++position_;
        return value;
    }
    while (true) {
        value.array.push_back(parse_value());
        skip_whitespace();
        if (position_ >= input_.size()) fail("unterminated array");
        if (input_[position_] == ',') {
            ++position_;
            continue;
        }
        if (input_[position_] == ']') {
            ++position_;
            return value;
        }
        fail("expected ',' or ']' in array");
    }
}

inline std::string PromptJson::Parser::parse_string_body() {
    expect('"');
    std::string out;
    while (true) {
        if (position_ >= input_.size()) fail("unterminated string");
        const char c = input_[position_++];
        if (c == '"') return out;
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (position_ >= input_.size()) fail("unterminated escape");
        const char escape = input_[position_++];
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
                if (position_ + 4 > input_.size()) fail("truncated \\u escape");
                uint32_t code_point = 0;
                for (int i = 0; i < 4; ++i) {
                    const char hex = input_[position_++];
                    code_point <<= 4;
                    if (hex >= '0' && hex <= '9') code_point |= static_cast<uint32_t>(hex - '0');
                    else if (hex >= 'a' && hex <= 'f') code_point |= static_cast<uint32_t>(hex - 'a' + 10);
                    else if (hex >= 'A' && hex <= 'F') code_point |= static_cast<uint32_t>(hex - 'A' + 10);
                    else fail("invalid hex digit in \\u escape");
                }
                if (code_point >= 0xD800 && code_point <= 0xDBFF && position_ + 6 <= input_.size() &&
                    input_[position_] == '\\' && input_[position_ + 1] == 'u') {
                    position_ += 2;
                    uint32_t low = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char hex = input_[position_++];
                        low <<= 4;
                        if (hex >= '0' && hex <= '9') low |= static_cast<uint32_t>(hex - '0');
                        else if (hex >= 'a' && hex <= 'f') low |= static_cast<uint32_t>(hex - 'a' + 10);
                        else if (hex >= 'A' && hex <= 'F') low |= static_cast<uint32_t>(hex - 'A' + 10);
                        else fail("invalid hex digit in \\u escape");
                    }
                    code_point = 0x10000 + ((code_point - 0xD800) << 10) + (low - 0xDC00);
                }
                if (code_point < 0x80) {
                    out.push_back(static_cast<char>(code_point));
                } else if (code_point < 0x800) {
                    out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
                    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
                } else if (code_point < 0x10000) {
                    out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
                    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
                } else {
                    out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
                    out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
                }
                break;
            }
            default:
                fail("unsupported escape");
        }
    }
}

inline PromptJson PromptJson::Parser::parse_string() {
    PromptJson value;
    value.kind = Kind::String;
    value.text = parse_string_body();
    return value;
}

inline PromptJson PromptJson::Parser::parse_number() {
    const size_t start = position_;
    bool is_real = false;
    if (position_ < input_.size() && input_[position_] == '-') ++position_;
    while (position_ < input_.size()) {
        const char c = input_[position_];
        if (c >= '0' && c <= '9') {
            ++position_;
        } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
            is_real = is_real || (c == '.' || c == 'e' || c == 'E');
            ++position_;
        } else {
            break;
        }
    }
    const std::string literal(input_.substr(start, position_ - start));
    PromptJson value;
    if (is_real) {
        value.kind = Kind::Real;
        value.real = std::strtod(literal.c_str(), nullptr);
    } else {
        value.kind = Kind::Int;
        value.integer = std::strtoll(literal.c_str(), nullptr, 10);
    }
    return value;
}

inline PromptJson PromptJson::Parser::parse_literal() {
    PromptJson value;
    if (input_.compare(position_, 4, "true") == 0) {
        value.kind = Kind::Bool;
        value.boolean = true;
        position_ += 4;
        return value;
    }
    if (input_.compare(position_, 5, "false") == 0) {
        value.kind = Kind::Bool;
        value.boolean = false;
        position_ += 5;
        return value;
    }
    if (input_.compare(position_, 4, "null") == 0) {
        value.kind = Kind::Null;
        position_ += 4;
        return value;
    }
    fail("invalid literal");
}

inline PromptJson PromptJson::parse(std::string_view input) {
    Parser parser(input);
    return parser.parse_document();
}

inline PromptJson PromptJson::parse_object(std::string_view input) {
    PromptJson value = parse(input);
    if (!value.is_object()) {
        throw std::runtime_error("PromptJson: expected a JSON object");
    }
    return value;
}

} // namespace aeon::text
