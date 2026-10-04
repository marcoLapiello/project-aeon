#pragma once

// -----------------------------------------------------------------------------
// Minimal JSON emission.
//
// Responses are *written*, not assembled from `JsonValue`: the writer keeps key
// order and lets a prepared raw value (a `timings` sub-object, a tool call) be
// spliced in without a round-trip through a map. `json_escape` passes valid UTF-8
// through byte-for-byte and escapes only what JSON requires, so multi-byte text
// survives intact.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace aeon::server {

inline std::string json_escape(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char raw : text) {
        const unsigned char byte = static_cast<unsigned char>(raw);
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
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", byte);
                    out += buffer;
                } else {
                    out.push_back(static_cast<char>(byte));
                }
        }
    }
    return out;
}

inline std::string json_number(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return buffer;
}

// Writes one JSON value at a time into a caller-owned buffer. Keys and values are
// emitted in the order the caller writes them.
class JsonWriter {
public:
    explicit JsonWriter(std::string& out) : out_(out) {}

    JsonWriter& begin_object() { open('{'); return *this; }
    JsonWriter& begin_array() { open('['); return *this; }
    JsonWriter& end_object() { close(); return *this; }
    JsonWriter& end_array() { close(); return *this; }

    JsonWriter& key(std::string_view name) {
        separator_for_key();
        out_ += '"';
        out_ += json_escape(name);
        out_ += "\":";
        pending_key_ = true;
        return *this;
    }

    JsonWriter& string(std::string_view value) {
        separator();
        out_ += '"';
        out_ += json_escape(value);
        out_ += '"';
        return *this;
    }

    JsonWriter& raw(std::string_view value) {
        separator();
        out_ += value;
        return *this;
    }

    JsonWriter& number(double value) {
        separator();
        out_ += json_number(value);
        return *this;
    }

    JsonWriter& integer(int64_t value) {
        separator();
        out_ += std::to_string(value);
        return *this;
    }

    JsonWriter& boolean(bool value) {
        separator();
        out_ += value ? "true" : "false";
        return *this;
    }

    JsonWriter& null_value() {
        separator();
        out_ += "null";
        return *this;
    }

private:
    void open(char brace) {
        separator();
        out_ += brace;
        container_.push_back(brace);
        has_child_.push_back(false);
    }

    void close() {
        out_ += (container_.back() == '[') ? ']' : '}';
        container_.pop_back();
        has_child_.pop_back();
        pending_key_ = false;
    }

    // Writes the comma before a key (not before its value; the value follows the
    // colon already written).
    void separator_for_key() {
        if (has_child_.empty()) return;
        if (has_child_.back()) out_ += ',';
        has_child_.back() = true;
    }

    void separator() {
        if (has_child_.empty()) return;
        if (pending_key_) {
            pending_key_ = false;
            has_child_.back() = true;
        } else if (has_child_.back()) {
            out_ += ',';
        } else {
            has_child_.back() = true;
        }
    }

    std::string& out_;
    std::vector<char> container_;
    std::vector<bool> has_child_;
    bool pending_key_{false};
};

}  // namespace aeon::server
