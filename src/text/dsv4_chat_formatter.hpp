#pragma once

#include "text/dsv4_tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace aeon::text {

enum class Dsv4MessageRole {
    System,
    User,
    Assistant,
};

enum class Dsv4ThinkingMode {
    Chat,
    Thinking,
};

struct Dsv4Message {
    Dsv4MessageRole role{Dsv4MessageRole::User};
    std::string content;
    std::string reasoning_content;
};

struct Dsv4FormattedPrompt {
    std::string rendered_text;
    std::vector<uint32_t> token_ids;
};

class Dsv4ChatFormatter {
public:
    explicit Dsv4ChatFormatter(const Dsv4Tokenizer& tokenizer) : tokenizer_(tokenizer) {}

    Dsv4FormattedPrompt format(
        const std::vector<Dsv4Message>& messages,
        Dsv4ThinkingMode thinking_mode,
        bool drop_thinking = true,
        bool add_default_bos_token = true
    ) const;

private:
    const Dsv4Tokenizer& tokenizer_;

    std::string token_text(uint32_t token_id) const;
    static bool is_user(const Dsv4Message& message);
};

} // namespace aeon::text