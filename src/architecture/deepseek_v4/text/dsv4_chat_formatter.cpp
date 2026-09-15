#include "architecture/deepseek_v4/text/dsv4_chat_formatter.hpp"

#include "architecture/deepseek_v4/text/dsv4_prompt_encoder.hpp"

#include <stdexcept>

namespace aeon::text {

// The rendering logic lives in `Dsv4PromptEncoder`. This type is the small,
// validated front door for simple chat turns: it converts its message type and
// keeps the one guard that matters for a chat API (a conversation must not be
// empty), then delegates. Keeping one encoder means the template cannot drift
// between the simple API and the full one.

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

    Dsv4PromptOptions options;
    options.thinking_mode = thinking_mode;
    options.drop_thinking = drop_thinking;
    options.add_default_bos_token = add_default_bos_token;

    std::vector<Dsv4PromptMessage> converted;
    converted.reserve(messages.size());
    for (const Dsv4Message& message : messages) {
        Dsv4PromptMessage out;
        switch (message.role) {
            case Dsv4MessageRole::System: out.role = Dsv4Role::System; break;
            case Dsv4MessageRole::User: out.role = Dsv4Role::User; break;
            case Dsv4MessageRole::Assistant: out.role = Dsv4Role::Assistant; break;
        }
        out.content = message.content;
        out.reasoning_content = message.reasoning_content;
        converted.push_back(std::move(out));
    }

    const Dsv4PromptEncoder encoder(tokenizer_);
    Dsv4FormattedPrompt result;
    result.rendered_text = encoder.encode(converted, options);
    result.token_ids = tokenizer_.encode(result.rendered_text);
    return result;
}

} // namespace aeon::text
