#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace aeon::core {

class JsonValue {
public:
    using array_type = std::vector<JsonValue>;
    using object_type = std::map<std::string, JsonValue>;

    JsonValue() : value_(nullptr) {}
    explicit JsonValue(std::nullptr_t) : value_(nullptr) {}
    explicit JsonValue(bool value) : value_(value) {}
    explicit JsonValue(double value) : value_(value) {}
    explicit JsonValue(std::string value) : value_(std::move(value)) {}
    explicit JsonValue(array_type value) : value_(std::move(value)) {}
    explicit JsonValue(object_type value) : value_(std::move(value)) {}

    static JsonValue parse(const std::string& text) {
        Parser parser(text);
        return parser.parse_document();
    }

    static JsonValue parse_file(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("JsonValue: failed to open file: " + path);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return parse(buffer.str());
    }

    bool is_null() const noexcept { return std::holds_alternative<std::nullptr_t>(value_); }
    bool is_bool() const noexcept { return std::holds_alternative<bool>(value_); }
    bool is_number() const noexcept { return std::holds_alternative<double>(value_); }
    bool is_string() const noexcept { return std::holds_alternative<std::string>(value_); }
    bool is_array() const noexcept { return std::holds_alternative<array_type>(value_); }
    bool is_object() const noexcept { return std::holds_alternative<object_type>(value_); }

    bool as_bool() const {
        if (!is_bool()) throw std::runtime_error("JSON value is not a boolean");
        return std::get<bool>(value_);
    }

    double as_number() const {
        if (!is_number()) throw std::runtime_error("JSON value is not a number");
        return std::get<double>(value_);
    }

    int64_t as_int64() const {
        const double number = as_number();
        if (!std::isfinite(number) || std::trunc(number) != number ||
            number < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
            number > static_cast<double>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("JSON number is not a signed 64-bit integer");
        }
        return static_cast<int64_t>(number);
    }

    const std::string& as_string() const {
        if (!is_string()) throw std::runtime_error("JSON value is not a string");
        return std::get<std::string>(value_);
    }

    const array_type& as_array() const {
        if (!is_array()) throw std::runtime_error("JSON value is not an array");
        return std::get<array_type>(value_);
    }

    const object_type& as_object() const {
        if (!is_object()) throw std::runtime_error("JSON value is not an object");
        return std::get<object_type>(value_);
    }

    const JsonValue* find(const std::string& key) const noexcept {
        if (!is_object()) return nullptr;
        const auto& object = std::get<object_type>(value_);
        const auto it = object.find(key);
        return it == object.end() ? nullptr : &it->second;
    }

    const JsonValue& at(const std::string& key) const {
        const JsonValue* value = find(key);
        if (value == nullptr) {
            throw std::runtime_error("JSON object is missing key: " + key);
        }
        return *value;
    }

private:
    class Parser {
    public:
        explicit Parser(const std::string& text) : text_(text) {}

        JsonValue parse_document() {
            skip_whitespace();
            JsonValue value = parse_value();
            skip_whitespace();
            if (position_ != text_.size()) fail("trailing characters");
            return value;
        }

    private:
        const std::string& text_;
        size_t position_{0};

        [[noreturn]] void fail(const std::string& message) const {
            throw std::runtime_error(
                "JSON parse error at position " + std::to_string(position_) + ": " + message);
        }

        void skip_whitespace() {
            while (position_ < text_.size()) {
                const char value = text_[position_];
                if (value != ' ' && value != '\n' && value != '\r' && value != '\t') break;
                ++position_;
            }
        }

        bool consume(char expected) {
            if (position_ < text_.size() && text_[position_] == expected) {
                ++position_;
                return true;
            }
            return false;
        }

        void expect(char expected) {
            if (!consume(expected)) {
                fail(std::string("expected '") + expected + "'");
            }
        }

        JsonValue parse_value() {
            skip_whitespace();
            if (position_ >= text_.size()) fail("unexpected end of input");
            switch (text_[position_]) {
                case 'n': return parse_literal("null", JsonValue(nullptr));
                case 't': return parse_literal("true", JsonValue(true));
                case 'f': return parse_literal("false", JsonValue(false));
                case '"': return JsonValue(parse_string());
                case '[': return parse_array();
                case '{': return parse_object();
                default: return parse_number();
            }
        }

        JsonValue parse_literal(const char* literal, JsonValue value) {
            const std::string expected(literal);
            if (text_.compare(position_, expected.size(), expected) != 0) {
                fail("invalid literal");
            }
            position_ += expected.size();
            return value;
        }

        JsonValue parse_number() {
            const size_t start = position_;
            if (position_ < text_.size() && text_[position_] == '-') ++position_;
            if (position_ >= text_.size() || text_[position_] < '0' || text_[position_] > '9') {
                fail("expected value");
            }
            if (text_[position_] == '0') {
                ++position_;
            } else {
                while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
                    ++position_;
                }
            }
            if (position_ < text_.size() && text_[position_] == '.') {
                ++position_;
                if (position_ >= text_.size() || text_[position_] < '0' || text_[position_] > '9') {
                    fail("invalid fractional part");
                }
                while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
                    ++position_;
                }
            }
            if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
                ++position_;
                if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
                if (position_ >= text_.size() || text_[position_] < '0' || text_[position_] > '9') {
                    fail("invalid exponent");
                }
                while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
                    ++position_;
                }
            }
            try {
                size_t consumed = 0;
                const double number = std::stod(text_.substr(start, position_ - start), &consumed);
                if (consumed != position_ - start) fail("invalid number");
                return JsonValue(number);
            } catch (const std::exception&) {
                fail("invalid number");
            }
        }

        std::string parse_string() {
            expect('"');
            std::string result;
            while (position_ < text_.size()) {
                const unsigned char value = static_cast<unsigned char>(text_[position_++]);
                if (value == '"') return result;
                if (value < 0x20) fail("control character in string");
                if (value != '\\') {
                    result.push_back(static_cast<char>(value));
                    continue;
                }
                if (position_ >= text_.size()) fail("unterminated escape");
                const char escaped = text_[position_++];
                switch (escaped) {
                    case '"': result.push_back('"'); break;
                    case '\\': result.push_back('\\'); break;
                    case '/': result.push_back('/'); break;
                    case 'b': result.push_back('\b'); break;
                    case 'f': result.push_back('\f'); break;
                    case 'n': result.push_back('\n'); break;
                    case 'r': result.push_back('\r'); break;
                    case 't': result.push_back('\t'); break;
                    case 'u': append_unicode_escape(result); break;
                    default: fail("invalid string escape");
                }
            }
            fail("unterminated string");
        }

        static int hex_value(char value) {
            if (value >= '0' && value <= '9') return value - '0';
            if (value >= 'a' && value <= 'f') return value - 'a' + 10;
            if (value >= 'A' && value <= 'F') return value - 'A' + 10;
            return -1;
        }

        void append_unicode_escape(std::string& result) {
            if (position_ + 4 > text_.size()) fail("truncated unicode escape");
            uint32_t codepoint = 0;
            for (int index = 0; index < 4; ++index) {
                const int digit = hex_value(text_[position_++]);
                if (digit < 0) fail("invalid unicode escape");
                codepoint = (codepoint << 4) | static_cast<uint32_t>(digit);
            }
            if (codepoint <= 0x7f) {
                result.push_back(static_cast<char>(codepoint));
            } else if (codepoint <= 0x7ff) {
                result.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
                result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
            } else {
                result.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
                result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
                result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
            }
        }

        JsonValue parse_array() {
            expect('[');
            array_type values;
            skip_whitespace();
            if (consume(']')) return JsonValue(std::move(values));
            while (true) {
                values.push_back(parse_value());
                skip_whitespace();
                if (consume(']')) return JsonValue(std::move(values));
                expect(',');
            }
        }

        JsonValue parse_object() {
            expect('{');
            object_type values;
            skip_whitespace();
            if (consume('}')) return JsonValue(std::move(values));
            while (true) {
                skip_whitespace();
                if (position_ >= text_.size() || text_[position_] != '"') fail("expected object key");
                const std::string key = parse_string();
                skip_whitespace();
                expect(':');
                JsonValue value = parse_value();
                if (!values.emplace(key, std::move(value)).second) fail("duplicate object key");
                skip_whitespace();
                if (consume('}')) return JsonValue(std::move(values));
                expect(',');
            }
        }
    };

    std::variant<std::nullptr_t, bool, double, std::string, array_type, object_type> value_;
};

} // namespace aeon::core