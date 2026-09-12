#include "infrastructure/text/text_generation.hpp"

#include <stdexcept>

namespace aeon::text {

GenerationResult generate_token_ids(
    const std::vector<uint32_t>& prompt,
    const GenerationOptions& options,
    const TokenStep& step
) {
    if (prompt.empty()) {
        throw std::invalid_argument("Generation prompt must not be empty");
    }
    if (options.max_new_tokens == 0) {
        throw std::invalid_argument("max_new_tokens must be greater than zero");
    }
    if (!step) {
        throw std::invalid_argument("Generation step callback is required");
    }
    if (options.context_limit != 0 && prompt.size() >= options.context_limit) {
        throw std::invalid_argument("Prompt does not leave room for generated tokens");
    }

    const uint32_t requested_tokens = options.max_new_tokens;
    uint32_t available_tokens = requested_tokens;
    bool context_limited = false;
    if (options.context_limit != 0) {
        const uint64_t available = options.context_limit - prompt.size();
        if (available < available_tokens) {
            available_tokens = static_cast<uint32_t>(available);
            context_limited = true;
        }
    }

    GenerationResult result;
    result.token_ids.reserve(available_tokens);

    uint32_t next_token = 0;
    for (size_t index = 0; index < prompt.size(); ++index) {
        next_token = step(prompt[index], static_cast<uint32_t>(index), true);
    }
    result.token_ids.push_back(next_token);
    if (options.stop_on_eos && next_token == options.eos_token_id) {
        result.stop_reason = StopReason::Eos;
        return result;
    }

    for (uint32_t generated = 1; generated < available_tokens; ++generated) {
        const uint32_t position = static_cast<uint32_t>(prompt.size()) + generated - 1;
        next_token = step(next_token, position, false);
        result.token_ids.push_back(next_token);
        if (options.stop_on_eos && next_token == options.eos_token_id) {
            result.stop_reason = StopReason::Eos;
            return result;
        }
    }

    result.stop_reason = context_limited ? StopReason::ContextLimit : StopReason::MaxNewTokens;
    return result;
}

const char* stop_reason_name(StopReason reason) {
    switch (reason) {
    case StopReason::Eos:
        return "eos";
    case StopReason::MaxNewTokens:
        return "max_new_tokens";
    case StopReason::ContextLimit:
        return "context_limit";
    case StopReason::Error:
        return "error";
    }
    return "error";
}

} // namespace aeon::text