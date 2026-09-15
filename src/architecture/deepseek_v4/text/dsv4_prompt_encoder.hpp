#pragma once

// DeepSeek-V4 prompt encoder — a port of the artifact's own encoder.
//
// Authority: `models/…-Aeon`'s sibling `encoding/encoding_dsv4.py` shipped inside the
// checkpoint (760 lines). That copy, not vLLM's, is the reference for this artifact's
// template; the two are sibling revisions and differ.
//
// Step 0 of the inference pipeline plan is not deferrable: an almost-right template
// silently changes every prefix and defeats prefix caching. This type exists so the
// template is expressed once, with an oracle (`test_dsv4_prompt_encoding_oracle`)
// that compares against the artifact's golden vectors byte-for-byte.
//
// Scope note: multi-turn *history* is not part of this type's job beyond rendering.
// `context` (a separately encoded prefix) is intentionally not implemented; the plan
// treats prefix reuse as a state-contract concern, not a formatting one.

#include "architecture/deepseek_v4/text/dsv4_tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace aeon::text {

enum class Dsv4ThinkingMode {
    Chat,
    Thinking,
};

enum class Dsv4Role {
    System,
    Developer,
    User,
    Assistant,
    LatestReminder,
    Tool,  // preprocessed away, never rendered directly
};

// One entry of a user message's content: either literal text or a tool result.
// This is how the encoder represents tool output, since DeepSeek-V4 has no
// standalone `tool` role.
struct Dsv4ContentBlock {
    enum class Kind { Text, ToolResult };
    Kind kind{Kind::Text};
    std::string text;          // Kind::Text
    std::string tool_result;   // Kind::ToolResult payload
    std::string tool_use_id;   // Kind::ToolResult call id, used for result ordering
};

// A tool definition as it appears in a system/developer message. The function
// object is kept as raw JSON text so it re-serializes with the reference's key
// order and formatting.
struct Dsv4ToolDefinition {
    std::string function_json;  // the `function` object: {name, description, parameters}
};

// An assistant tool call. `id` is only used to order tool results.
struct Dsv4ToolCall {
    std::string id;
    std::string name;
    std::string arguments;  // JSON string
};

struct Dsv4PromptMessage {
    Dsv4Role role{Dsv4Role::User};
    std::string content;
    std::string reasoning_content;
    std::vector<Dsv4ContentBlock> content_blocks;   // user messages after tool merging
    std::vector<Dsv4ToolDefinition> tools;          // system / developer
    std::string response_format_json;               // optional, re-serialized
    std::vector<Dsv4ToolCall> tool_calls;           // assistant
    std::string tool_call_id;                       // tool messages: which call this answers
    std::string task;                               // classification task, optional
    bool wo_eos{false};                             // assistant: omit the trailing eos
    bool mask{false};                               // preserved through merging, not rendered
};

struct Dsv4PromptOptions {
    Dsv4ThinkingMode thinking_mode{Dsv4ThinkingMode::Chat};
    bool drop_thinking{true};
    bool add_default_bos_token{true};
    std::string reasoning_effort{"low"};  // "low" (default) | "high" | "max"
};

class Dsv4PromptEncoder {
public:
    explicit Dsv4PromptEncoder(const Dsv4Tokenizer& tokenizer) : tokenizer_(tokenizer) {}

    // Render the conversation to the model's prompt text.
    //
    // The pipeline mirrors the reference:
    //   1. merge `tool` messages into user messages as tool-result blocks
    //   2. order tool results by the preceding assistant's tool-call order
    //   3. if any message declares tools, keep thinking (drop_thinking is forced off)
    //   4. in thinking mode with drop_thinking, strip earlier reasoning
    //   5. render each message, appending transition tokens
    std::string encode(const std::vector<Dsv4PromptMessage>& messages,
                       const Dsv4PromptOptions& options) const;

    // Convenience: encode, then tokenize with the artifact tokenizer.
    std::vector<uint32_t> encode_tokens(const std::vector<Dsv4PromptMessage>& messages,
                                        const Dsv4PromptOptions& options) const;

private:
    const Dsv4Tokenizer& tokenizer_;
};

} // namespace aeon::text
