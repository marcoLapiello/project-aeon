#include "text/dsv4_chat_formatter.hpp"

#include <stdexcept>

namespace aeon::text {

std::string Dsv4ChatFormatter::token_text(uint32_t token_id) const {
    return tokenizer_.decode({token_id});
}

bool Dsv4ChatFormatter::is_user(const Dsv4Message& message) {
    return message.role == Dsv4MessageRole::User;
}

Dsv4FormattedPrompt Dsv4ChatFormatter::format(
    const std::vector<Dsv4Message>& messages,
    Dsv4ThinkingMode thinking_mode,
    bool drop_thinking,
    bool add_default_bos_token
) const {
    if (messages.empty()) {
        throw std::invalid_argument("DSV4 conversation must contain at least one message");
    }

    size_t last_user_index = messages.size();
    bool seen_system = false;
    for (size_t index = 0; index < messages.size(); ++index) {
        const auto& message = messages[index];
        if (is_user(message)) {
            last_user_index = index;
        } else if (message.role == Dsv4MessageRole::System) {
            if (seen_system || index != 0) {
                throw std::invalid_argument("DSV4 system message must be the first and only system message");
            }
            seen_system = true;
        } else if (message.role == Dsv4MessageRole::Assistant) {
            if (last_user_index == messages.size()) {
                throw std::invalid_argument("DSV4 assistant message requires a preceding user message");
            }
        }
    }
    if (last_user_index == messages.size()) {
        throw std::invalid_argument("DSV4 conversation must contain a user message");
    }

    std::string rendered;
    if (add_default_bos_token) {
        rendered += token_text(tokenizer_.bos_token_id());
    }

    for (size_t index = 0; index < messages.size(); ++index) {
        const auto& message = messages[index];
        if (message.role == Dsv4MessageRole::System) {
            rendered += message.content;
        } else if (message.role == Dsv4MessageRole::User) {
            rendered += token_text(tokenizer_.user_token_id());
            rendered += message.content;
        } else if (message.role == Dsv4MessageRole::Assistant) {
            if (thinking_mode == Dsv4ThinkingMode::Thinking &&
                (!drop_thinking || index > last_user_index)) {
                rendered += message.reasoning_content;
                rendered += token_text(tokenizer_.thinking_end_token_id());
            }
            rendered += message.content;
            rendered += token_text(tokenizer_.eos_token_id());
        }

        if (message.role == Dsv4MessageRole::User) {
            rendered += token_text(tokenizer_.assistant_token_id());
            if (thinking_mode == Dsv4ThinkingMode::Thinking &&
                (!drop_thinking || index >= last_user_index)) {
                rendered += token_text(tokenizer_.thinking_start_token_id());
            } else {
                rendered += token_text(tokenizer_.thinking_end_token_id());
            }
        }
    }

    return {rendered, tokenizer_.encode(rendered)};
}

} // namespace aeon::text