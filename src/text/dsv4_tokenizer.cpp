#include "text/dsv4_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace aeon::text {
namespace {

constexpr uint8_t kSpecialFlag = 1u << 0;

class BinaryReader {
public:
    explicit BinaryReader(const std::string& path) : input_(path, std::ios::binary) {
        if (!input_) {
            throw std::runtime_error("Unable to open tokenizer artifact: " + path);
        }
    }

    template <typename T>
    T read() {
        T value{};
        input_.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!input_) {
            throw std::runtime_error("Unexpected end of tokenizer artifact");
        }
        return value;
    }

    std::string read_string() {
        const uint32_t size = read<uint32_t>();
        std::string value(size, '\0');
        input_.read(value.data(), static_cast<std::streamsize>(size));
        if (!input_) {
            throw std::runtime_error("Unexpected end of tokenizer artifact string");
        }
        return value;
    }

    void read_bytes(char* destination, std::streamsize size) {
        input_.read(destination, size);
        if (!input_) {
            throw std::runtime_error("Unexpected end of tokenizer artifact bytes");
        }
    }

private:
    std::ifstream input_;
};

uint32_t decode_code_point(std::string_view text, size_t* offset) {
    if (*offset >= text.size()) {
        throw std::runtime_error("Invalid UTF-8: missing code point");
    }
    const auto byte = static_cast<uint8_t>(text[*offset]);
    uint32_t value = 0;
    size_t width = 0;
    if (byte < 0x80) {
        value = byte;
        width = 1;
    } else if ((byte & 0xe0) == 0xc0) {
        value = byte & 0x1f;
        width = 2;
    } else if ((byte & 0xf0) == 0xe0) {
        value = byte & 0x0f;
        width = 3;
    } else if ((byte & 0xf8) == 0xf0) {
        value = byte & 0x07;
        width = 4;
    } else {
        throw std::runtime_error("Invalid UTF-8 leading byte");
    }
    if (*offset + width > text.size()) {
        throw std::runtime_error("Invalid UTF-8: truncated code point");
    }
    for (size_t index = 1; index < width; ++index) {
        const auto continuation = static_cast<uint8_t>(text[*offset + index]);
        if ((continuation & 0xc0) != 0x80) {
            throw std::runtime_error("Invalid UTF-8 continuation byte");
        }
        value = (value << 6) | (continuation & 0x3f);
    }
    if ((width == 2 && value < 0x80) ||
        (width == 3 && value < 0x800) ||
        (width == 4 && value < 0x10000) ||
        value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) {
        throw std::runtime_error("Invalid UTF-8 code point");
    }
    *offset += width;
    return value;
}

std::string encode_code_point(uint32_t value) {
    std::string result;
    if (value <= 0x7f) {
        result.push_back(static_cast<char>(value));
    } else if (value <= 0x7ff) {
        result.push_back(static_cast<char>(0xc0 | (value >> 6)));
        result.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else if (value <= 0xffff) {
        result.push_back(static_cast<char>(0xe0 | (value >> 12)));
        result.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        result.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else {
        result.push_back(static_cast<char>(0xf0 | (value >> 18)));
        result.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
        result.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        result.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
    return result;
}

bool is_ascii_digit(uint32_t value) {
    return value >= '0' && value <= '9';
}

bool is_cjk(uint32_t value) {
    return (value >= 0x4e00 && value <= 0x9fff) ||
           (value >= 0x3400 && value <= 0x4dbf) ||
           (value >= 0x3040 && value <= 0x30ff) ||
           (value >= 0x31f0 && value <= 0x31ff);
}

bool is_space(uint32_t value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
           value == '\f' || value == '\v' || value == 0x00a0 || value == 0x1680 ||
           (value >= 0x2000 && value <= 0x200a) || value == 0x2028 || value == 0x2029 ||
           value == 0x202f || value == 0x205f || value == 0x3000;
}

bool is_letter_or_mark(uint32_t value) {
    if ((value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z')) {
        return true;
    }
    if (value >= 0x300 && value <= 0x36f) {
        return true;
    }
    return value >= 0x00c0 && !is_space(value) && !is_cjk(value) &&
           !(value >= 0x2000 && value <= 0x206f);
}

bool is_ascii_punctuation_or_symbol(uint32_t value) {
    return (value >= 0x21 && value <= 0x2f) ||
           (value >= 0x3a && value <= 0x40) ||
           (value >= 0x5b && value <= 0x60) ||
           (value >= 0x7b && value <= 0x7e);
}

bool is_punctuation_or_symbol(uint32_t value) {
    if (is_ascii_punctuation_or_symbol(value)) {
        return true;
    }
    if (is_space(value) || is_letter_or_mark(value) || is_ascii_digit(value) || is_cjk(value)) {
        return false;
    }
    return true;
}

} // namespace

void Dsv4Tokenizer::load(const std::string& artifact_path) {
    BinaryReader reader(artifact_path);
    char magic[8]{};
    reader.read_bytes(magic, sizeof(magic));
    if (std::memcmp(magic, "AEONTOK1", sizeof(magic)) != 0) {
        throw std::runtime_error("Invalid DSV4 tokenizer artifact magic");
    }

    const uint32_t version = reader.read<uint32_t>();
    if (version != 2) {
        throw std::runtime_error("Unsupported DSV4 tokenizer artifact version");
    }
    const uint32_t base_vocab_size = reader.read<uint32_t>();
    id_limit_ = reader.read<uint32_t>();
    const uint32_t merge_count = reader.read<uint32_t>();
    const uint32_t added_count = reader.read<uint32_t>();
    bos_token_id_ = reader.read<uint32_t>();
    eos_token_id_ = reader.read<uint32_t>();
    user_token_id_ = reader.read<uint32_t>();
    assistant_token_id_ = reader.read<uint32_t>();
    thinking_start_token_id_ = reader.read<uint32_t>();
    thinking_end_token_id_ = reader.read<uint32_t>();
    std::array<char, 32> source_hash{};
    reader.read_bytes(source_hash.data(), static_cast<std::streamsize>(source_hash.size()));

    if (base_vocab_size == 0 || id_limit_ == 0 ||
        bos_token_id_ >= id_limit_ || eos_token_id_ >= id_limit_ ||
        user_token_id_ >= id_limit_ || assistant_token_id_ >= id_limit_ ||
        thinking_start_token_id_ >= id_limit_ || thinking_end_token_id_ >= id_limit_) {
        throw std::runtime_error("Invalid DSV4 tokenizer artifact dimensions");
    }

    vocab_symbols_.assign(id_limit_, {});
    decoded_symbols_.assign(id_limit_, {});
    symbol_to_id_.clear();
    for (uint32_t index = 0; index < base_vocab_size; ++index) {
        const uint32_t token_id = reader.read<uint32_t>();
        const std::string symbol = reader.read_string();
        if (token_id >= id_limit_ || !vocab_symbols_[token_id].empty()) {
            throw std::runtime_error("Invalid base vocabulary ID");
        }
        vocab_symbols_[token_id] = symbol;
        decoded_symbols_[token_id] = decode_base_symbol(symbol);
        symbol_to_id_.emplace(symbol, token_id);
    }

    merge_ranks_.clear();
    merge_ranks_.reserve(merge_count * 2);
    for (uint32_t rank = 0; rank < merge_count; ++rank) {
        const std::string left = reader.read_string();
        const std::string right = reader.read_string();
        merge_ranks_.emplace(pair_key(left, right), rank);
    }

    added_tokens_.clear();
    added_content_to_id_.clear();
    added_tokens_.reserve(added_count);
    for (uint32_t index = 0; index < added_count; ++index) {
        AddedToken token;
        token.id = reader.read<uint32_t>();
        token.flags = reader.read<uint8_t>();
        token.content = reader.read_string();
        if (token.id >= id_limit_ || token.content.empty()) {
            throw std::runtime_error("Invalid added token");
        }
        added_content_to_id_[token.content] = token.id;
        added_tokens_.push_back(std::move(token));
    }
    std::sort(added_tokens_.begin(), added_tokens_.end(), [](const AddedToken& left, const AddedToken& right) {
        if (left.content.size() != right.content.size()) {
            return left.content.size() > right.content.size();
        }
        return left.content < right.content;
    });

    loaded_ = true;
}

uint32_t Dsv4Tokenizer::vocab_size() const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    return id_limit_;
}

uint32_t Dsv4Tokenizer::bos_token_id() const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    return bos_token_id_;
}

uint32_t Dsv4Tokenizer::eos_token_id() const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    return eos_token_id_;
}

uint32_t Dsv4Tokenizer::user_token_id() const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    return user_token_id_;
}

uint32_t Dsv4Tokenizer::assistant_token_id() const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    return assistant_token_id_;
}

uint32_t Dsv4Tokenizer::thinking_start_token_id() const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    return thinking_start_token_id_;
}

uint32_t Dsv4Tokenizer::thinking_end_token_id() const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    return thinking_end_token_id_;
}

uint32_t Dsv4Tokenizer::token_id(std::string_view content) const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    const auto found = added_content_to_id_.find(std::string(content));
    if (found == added_content_to_id_.end()) {
        throw std::invalid_argument("Unknown DSV4 token: " + std::string(content));
    }
    return found->second;
}

std::vector<Dsv4Tokenizer::CodePoint> Dsv4Tokenizer::decode_utf8(std::string_view text) const {
    std::vector<CodePoint> code_points;
    size_t offset = 0;
    while (offset < text.size()) {
        const size_t begin = offset;
        const uint32_t value = decode_code_point(text, &offset);
        code_points.push_back({value, begin, offset});
    }
    return code_points;
}

std::vector<std::string_view> Dsv4Tokenizer::split_pretokens(std::string_view text) const {
    const auto code_points = decode_utf8(text);
    std::vector<std::string_view> pieces;
    size_t index = 0;
    while (index < code_points.size()) {
        const auto& current = code_points[index];
        size_t end = index + 1;

        if (is_ascii_digit(current.value)) {
            size_t digits = 1;
            while (end < code_points.size() && digits < 3 && is_ascii_digit(code_points[end].value)) {
                ++end;
                ++digits;
            }
        } else if (is_cjk(current.value)) {
            while (end < code_points.size() && is_cjk(code_points[end].value)) {
                ++end;
            }
        } else if (current.value == '\r' || current.value == '\n') {
            while (end < code_points.size() &&
                   (code_points[end].value == '\r' || code_points[end].value == '\n')) {
                ++end;
            }
        } else if (is_space(current.value)) {
            if (current.value == ' ' && end < code_points.size() && is_letter_or_mark(code_points[end].value)) {
                while (end < code_points.size() && is_letter_or_mark(code_points[end].value)) {
                    ++end;
                }
            } else {
                while (end < code_points.size() && is_space(code_points[end].value) &&
                       code_points[end].value != '\r' && code_points[end].value != '\n') {
                    ++end;
                }
            }
        } else if (is_letter_or_mark(current.value)) {
            while (end < code_points.size() && is_letter_or_mark(code_points[end].value)) {
                ++end;
            }
        } else if (is_punctuation_or_symbol(current.value)) {
            while (end < code_points.size() && is_punctuation_or_symbol(code_points[end].value)) {
                ++end;
            }
            while (end < code_points.size() &&
                   (code_points[end].value == '\r' || code_points[end].value == '\n')) {
                ++end;
            }
        }

        pieces.emplace_back(text.data() + current.begin, code_points[end - 1].end - current.begin);
        index = end;
    }
    return pieces;
}

std::vector<uint32_t> Dsv4Tokenizer::encode_standard(std::string_view text) const {
    std::vector<uint32_t> output;
    for (const auto piece : split_pretokens(text)) {
        const auto bytes = std::string(piece);
        std::string byte_level;
        byte_level.reserve(bytes.size() * 2);
        for (const auto byte : bytes) {
            byte_level += byte_to_symbol(static_cast<uint8_t>(byte));
        }
        std::vector<std::string> symbols;
        size_t offset = 0;
        while (offset < byte_level.size()) {
            const size_t begin = offset;
            (void)decode_code_point(byte_level, &offset);
            symbols.emplace_back(byte_level.substr(begin, offset - begin));
        }

        while (symbols.size() > 1) {
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();
            size_t best_index = symbols.size();
            for (size_t index = 0; index + 1 < symbols.size(); ++index) {
                const auto found = merge_ranks_.find(pair_key(symbols[index], symbols[index + 1]));
                if (found != merge_ranks_.end() && found->second < best_rank) {
                    best_rank = found->second;
                    best_index = index;
                }
            }
            if (best_index == symbols.size()) {
                break;
            }
            symbols[best_index] += symbols[best_index + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
        }

        for (const auto& symbol : symbols) {
            const auto found = symbol_to_id_.find(symbol);
            if (found == symbol_to_id_.end()) {
                throw std::runtime_error("Tokenizer BPE produced an unknown symbol");
            }
            output.push_back(found->second);
        }
    }
    return output;
}

std::vector<uint32_t> Dsv4Tokenizer::encode(std::string_view text) const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    std::vector<uint32_t> output;
    size_t offset = 0;
    while (offset < text.size()) {
        const AddedToken* match = nullptr;
        for (const auto& token : added_tokens_) {
            if (token.content.size() <= text.size() - offset &&
                text.substr(offset, token.content.size()) == token.content) {
                match = &token;
                break;
            }
        }
        if (match != nullptr) {
            output.push_back(match->id);
            offset += match->content.size();
            continue;
        }

        size_t next_added = text.size();
        for (const auto& token : added_tokens_) {
            const size_t found = text.find(token.content, offset);
            if (found != std::string_view::npos && found < next_added) {
                next_added = found;
            }
        }
        const size_t end = next_added;
        if (end <= offset) {
            throw std::runtime_error("Tokenizer failed to advance");
        }
        const auto encoded = encode_standard(text.substr(offset, end - offset));
        output.insert(output.end(), encoded.begin(), encoded.end());
        offset = end;
    }
    return output;
}

std::string Dsv4Tokenizer::decode(const std::vector<uint32_t>& token_ids, bool skip_special_tokens) const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    std::string output;
    for (const uint32_t token_id : token_ids) {
        if (token_id >= id_limit_) {
            throw std::out_of_range("Token ID exceeds tokenizer vocabulary");
        }
        const auto added = std::find_if(added_tokens_.begin(), added_tokens_.end(),
            [token_id](const AddedToken& token) { return token.id == token_id; });
        if (added != added_tokens_.end()) {
            if (!(skip_special_tokens && (added->flags & kSpecialFlag) != 0)) {
                output += added->content;
            }
            continue;
        }
        if (vocab_symbols_[token_id].empty()) {
            throw std::runtime_error("Token ID has no vocabulary entry");
        }
        output += decoded_symbols_[token_id];
    }
    return output;
}

std::string Dsv4Tokenizer::decode_base_symbol(std::string_view symbol) const {
    const auto bytes = symbol_to_bytes(symbol);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::string Dsv4Tokenizer::pair_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left);
    key.push_back('\0');
    key.append(right);
    return key;
}

std::string Dsv4Tokenizer::byte_to_symbol(uint8_t value) {
    static const std::array<std::string, 256> symbols = [] {
        std::array<std::string, 256> result{};
        std::array<bool, 256> direct{};
        for (uint32_t byte = 33; byte <= 126; ++byte) direct[byte] = true;
        for (uint32_t byte = 161; byte <= 172; ++byte) direct[byte] = true;
        for (uint32_t byte = 174; byte <= 255; ++byte) direct[byte] = true;
        uint32_t extra = 0;
        for (uint32_t byte = 0; byte < 256; ++byte) {
            const uint32_t code_point = direct[byte] ? byte : 256 + extra++;
            result[byte] = encode_code_point(code_point);
        }
        return result;
    }();
    return symbols[value];
}

std::vector<uint8_t> Dsv4Tokenizer::symbol_to_bytes(std::string_view symbol) {
    static const std::unordered_map<uint32_t, uint8_t> reverse = [] {
        std::unordered_map<uint32_t, uint8_t> result;
        std::array<bool, 256> direct{};
        for (uint32_t byte = 33; byte <= 126; ++byte) direct[byte] = true;
        for (uint32_t byte = 161; byte <= 172; ++byte) direct[byte] = true;
        for (uint32_t byte = 174; byte <= 255; ++byte) direct[byte] = true;
        uint32_t extra = 0;
        for (uint32_t byte = 0; byte < 256; ++byte) {
            const uint32_t code_point = direct[byte] ? byte : 256 + extra++;
            result.emplace(code_point, static_cast<uint8_t>(byte));
        }
        return result;
    }();

    std::vector<uint8_t> bytes;
    size_t offset = 0;
    while (offset < symbol.size()) {
        const uint32_t code_point = decode_code_point(symbol, &offset);
        const auto found = reverse.find(code_point);
        if (found == reverse.end()) {
            const std::string utf8 = encode_code_point(code_point);
            bytes.insert(bytes.end(), utf8.begin(), utf8.end());
        } else {
            bytes.push_back(found->second);
        }
    }
    return bytes;
}

} // namespace aeon::text