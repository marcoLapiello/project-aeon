#pragma once

// -----------------------------------------------------------------------------
// The computation key — the non-token inputs a resident state was built under.
//
// Prefix reuse compares the *tokens* of the new turn against the tokens the
// resident state was fed. For DSV4 those tokens already encode most non-token
// inputs — thinking mode changes the `<think>` / `</think>` transition, reasoning
// effort prepends a preamble, tools and response format render into system-message
// text — so a strict token compare invalidates on any of them. That is the primary
// mechanism.
//
// This key is the *cheap, model-agnostic second line*: a short canonical string of
// the encoder options and the declared tool/response-format schemas, so a change
// that a future encoder revision might not surface in the tokens still invalidates.
// It is deliberately independent of message text — text is what the token compare
// already covers.
//
// `drop_thinking` is folded to its **effective** value, false when any message
// declares tools, mirroring `Dsv4PromptEncoder::encode`. Reading the raw option
// instead would report a change on tool turns where the encoder ignores it.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/text/dsv4_prompt_encoder.hpp"

#include <string>
#include <vector>

namespace aeon::text {

inline std::string dsv4_computation_key(const std::vector<Dsv4PromptMessage>& messages,
                                        const Dsv4PromptOptions& options) {
    // Mirrors the encoder: a declared tool forces thinking to be kept.
    bool effective_drop_thinking = options.drop_thinking;
    for (const Dsv4PromptMessage& message : messages) {
        if (!message.tools.empty()) {
            effective_drop_thinking = false;
            break;
        }
    }

    // The tool schemas and the response format are the parts of the system/developer
    // text that are configuration rather than prose, so they are named explicitly.
    // The encoder renders them on system and developer messages only.
    std::string tools;
    std::string response_format;
    for (const Dsv4PromptMessage& message : messages) {
        if (message.role != Dsv4Role::System && message.role != Dsv4Role::Developer) continue;
        for (const Dsv4ToolDefinition& tool : message.tools) {
            tools += tool.function_json;
            tools += '\x1f';
        }
        if (!message.response_format_json.empty()) {
            response_format += message.response_format_json;
            response_format += '\x1f';
        }
    }

    std::string key;
    key += "thinking=";
    key += options.thinking_mode == Dsv4ThinkingMode::Thinking ? "1" : "0";
    key += "|drop=";
    key += effective_drop_thinking ? "1" : "0";
    key += "|effort=";
    key += options.reasoning_effort;
    key += "|bos=";
    key += options.add_default_bos_token ? "1" : "0";
    key += "|tools=";
    key += tools;
    key += "|fmt=";
    key += response_format;
    return key;
}

}  // namespace aeon::text
