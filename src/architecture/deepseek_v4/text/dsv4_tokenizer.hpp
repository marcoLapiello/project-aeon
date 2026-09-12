#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aeon::text {

class Dsv4Tokenizer {
public:
    void load(const std::string& artifact_path);

    uint32_t vocab_size() const;
    uint32_t bos_token_id() const;
    uint32_t eos_token_id() const;
    uint32_t user_token_id() const;
    uint32_t assistant_token_id() const;
    uint32_t thinking_start_token_id() const;
    uint32_t thinking_end_token_id() const;
    uint32_t token_id(std::string_view content) const;

    std::vector<uint32_t> encode(std::string_view text) const;
    std::string decode(const std::vector<uint32_t>& token_ids, bool skip_special_tokens = false) const;

private:
    struct AddedToken {
        uint32_t id{0};
        uint8_t flags{0};
        std::string content;
    };

    struct CodePoint {
        uint32_t value{0};
        size_t begin{0};
        size_t end{0};
    };

    std::vector<std::string> vocab_symbols_;
    std::vector<std::string> decoded_symbols_;
    std::vector<AddedToken> added_tokens_;
    std::vector<uint32_t> added_ids_by_length_;
    std::unordered_map<std::string, uint32_t> symbol_to_id_;
    std::unordered_map<std::string, uint32_t> added_content_to_id_;
    std::unordered_map<std::string, uint32_t> merge_ranks_;
    uint32_t id_limit_{0};
    uint32_t bos_token_id_{0};
    uint32_t eos_token_id_{0};
    uint32_t user_token_id_{0};
    uint32_t assistant_token_id_{0};
    uint32_t thinking_start_token_id_{0};
    uint32_t thinking_end_token_id_{0};
    bool loaded_{false};

    std::vector<CodePoint> decode_utf8(std::string_view text) const;
    std::vector<std::string_view> split_pretokens(std::string_view text) const;
    std::vector<uint32_t> encode_standard(std::string_view text) const;
    std::vector<uint32_t> encode_bpe(std::string_view text) const;
    std::string decode_base_symbol(std::string_view symbol) const;
    static std::string pair_key(std::string_view left, std::string_view right);
    static std::string byte_to_symbol(uint8_t value);
    static std::vector<uint8_t> symbol_to_bytes(std::string_view symbol);
};

} // namespace aeon::text