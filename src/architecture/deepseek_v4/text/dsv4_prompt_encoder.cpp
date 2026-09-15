#include "architecture/deepseek_v4/text/dsv4_prompt_encoder.hpp"

#include "architecture/deepseek_v4/text/dsv4_prompt_json.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aeon::text {
namespace {

// Token *contents* are assembled from safe pieces and then validated against the
// artifact tokenizer. Two reasons for the concatenation:
//   * A `\xNN` escape consumes every following hex digit, so `"\x9CUser"` would be
//     read as one escape (`0x9CUser` is not even valid). Adjacent literals break it.
//   * Resolving through `added_token_text` means a renamed or missing token fails
//     loudly instead of silently embedding a wrong literal.
constexpr const char* kFullwidthBar = "\xEF\xBD\x9C";   // U+FF5C ｜
constexpr const char* kLowerOneEighth = "\xE2\x96\x81"; // U+2581 ▁

struct Tokens {
    std::string bos;
    std::string eos;
    std::string thinking_start;
    std::string thinking_end;
    std::string user;
    std::string assistant;
    std::string latest_reminder;
    std::string dsml;

    std::string wrap(std::string_view inner) const {
        return std::string("<") + kFullwidthBar + std::string(inner) + kFullwidthBar + ">";
    }

    std::string task_token(std::string_view name) const { return wrap(name); }
};

Tokens resolve_tokens(const Dsv4Tokenizer& tokenizer) {
    const std::string bar = kFullwidthBar;
    const std::string one = kLowerOneEighth;
    Tokens tokens;
    tokens.bos = tokenizer.added_token_text("<" + bar + "begin" + one + "of" + one + "sentence" + bar + ">");
    tokens.eos = tokenizer.added_token_text("<" + bar + "end" + one + "of" + one + "sentence" + bar + ">");
    tokens.thinking_start = tokenizer.added_token_text("<think>");
    tokens.thinking_end = tokenizer.added_token_text("</think>");
    tokens.user = tokenizer.added_token_text("<" + bar + "User" + bar + ">");
    tokens.assistant = tokenizer.added_token_text("<" + bar + "Assistant" + bar + ">");
    tokens.latest_reminder =
        tokenizer.added_token_text("<" + bar + "latest_reminder" + bar + ">");
    tokens.dsml = tokenizer.added_token_text(bar + "DSML" + bar);
    return tokens;
}

constexpr const char* kToolCallsBlockName = "tool_calls";
constexpr const char* kToolResultOpen = "<tool_result>";
constexpr const char* kToolResultClose = "</tool_result>";

constexpr const char* kResponseFormatTemplate =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n";

constexpr const char* kReasoningEffortHigh =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to "
    "resolve the root cause, rigorously stress-testing your logic against all potential paths, "
    "edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, "
    "considered alternative, and rejected hypothesis to ensure absolutely no assumption is left "
    "unchecked.\n\n";

constexpr const char* kReasoningEffortMax =
    "Reasoning Effort: Beyond maximum \xE2\x80\x94 exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: "
    "exhaustively decompose the problem into its most fundamental components, trace every causal "
    "chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple "
    "angles and are certain that no assumption remains unchecked and no error remains "
    "undiscovered.\n\n";

std::string replace_all(std::string text, std::string_view needle, std::string_view value) {
    if (needle.empty()) return text;
    size_t position = 0;
    while ((position = text.find(needle, position)) != std::string::npos) {
        text.replace(position, needle.size(), value);
        position += value.size();
    }
    return text;
}

// Python: `to_json(value)` — re-serialize preserving key order and Python's
// separators, so tool schemas embed exactly as the reference emits them.
std::string to_reference_json(std::string_view raw_json) {
    return PromptJson::parse(raw_json).to_python_json();
}

bool is_assistant(const Dsv4PromptMessage& message) {
    return message.role == Dsv4Role::Assistant;
}

bool is_latest_reminder(const Dsv4PromptMessage& message) {
    return message.role == Dsv4Role::LatestReminder;
}

bool is_user_like(const Dsv4PromptMessage& message) {
    return message.role == Dsv4Role::User || message.role == Dsv4Role::Developer;
}

bool has_content_blocks(const Dsv4PromptMessage& message) {
    return !message.content_blocks.empty();
}

int find_last_user_index(const std::vector<Dsv4PromptMessage>& messages) {
    for (int index = static_cast<int>(messages.size()) - 1; index >= 0; --index) {
        const Dsv4Role role = messages[static_cast<size_t>(index)].role;
        if (role == Dsv4Role::User || role == Dsv4Role::Developer) return index;
    }
    return -1;
}

// Step 1: give `tool` messages a home. DeepSeek-V4 has no standalone tool role, so
// their payloads become tool-result blocks inside a user message.
std::vector<Dsv4PromptMessage> merge_tool_messages(const std::vector<Dsv4PromptMessage>& input) {
    std::vector<Dsv4PromptMessage> merged;
    for (const Dsv4PromptMessage& original : input) {
        if (original.role == Dsv4Role::Tool) {
            Dsv4ContentBlock tool_block;
            tool_block.kind = Dsv4ContentBlock::Kind::ToolResult;
            tool_block.tool_result = original.content;
            tool_block.tool_use_id = original.tool_call_id;

            if (!merged.empty() && merged.back().role == Dsv4Role::User &&
                has_content_blocks(merged.back())) {
                merged.back().content_blocks.push_back(std::move(tool_block));
            } else {
                Dsv4PromptMessage message;
                message.role = Dsv4Role::User;
                message.content_blocks.push_back(std::move(tool_block));
                merged.push_back(std::move(message));
            }
            continue;
        }

        if (original.role == Dsv4Role::User) {
            Dsv4ContentBlock text_block;
            text_block.kind = Dsv4ContentBlock::Kind::Text;
            text_block.text = original.content;

            if (!merged.empty() && merged.back().role == Dsv4Role::User &&
                has_content_blocks(merged.back()) && merged.back().task.empty()) {
                merged.back().content_blocks.push_back(std::move(text_block));
                continue;
            }

            Dsv4PromptMessage message = original;
            message.content_blocks.clear();
            message.content_blocks.push_back(std::move(text_block));
            merged.push_back(std::move(message));
            continue;
        }

        merged.push_back(original);
    }
    return merged;
}

// Step 2: order tool results by the preceding assistant's tool-call order, so a
// batch of results renders deterministically.
void sort_tool_results(const std::vector<Dsv4PromptMessage>& messages,
                       std::vector<Dsv4PromptMessage>& mutable_messages) {
    std::map<std::string, size_t> call_order;
    for (size_t index = 0; index < messages.size(); ++index) {
        const Dsv4PromptMessage& message = messages[index];
        if (message.role == Dsv4Role::Assistant && !message.tool_calls.empty()) {
            call_order.clear();
            for (size_t order = 0; order < message.tool_calls.size(); ++order) {
                if (!message.tool_calls[order].id.empty()) {
                    call_order[message.tool_calls[order].id] = order;
                }
            }
            continue;
        }

        if (message.role != Dsv4Role::User || !has_content_blocks(message)) continue;

        std::vector<Dsv4ContentBlock> tool_blocks;
        for (const Dsv4ContentBlock& block : message.content_blocks) {
            if (block.kind == Dsv4ContentBlock::Kind::ToolResult) tool_blocks.push_back(block);
        }
        if (tool_blocks.size() <= 1 || call_order.empty()) continue;

        std::stable_sort(tool_blocks.begin(), tool_blocks.end(),
                         [&](const Dsv4ContentBlock& left, const Dsv4ContentBlock& right) {
                             const auto left_it = call_order.find(left.tool_use_id);
                             const auto right_it = call_order.find(right.tool_use_id);
                             const size_t left_order =
                                 left_it == call_order.end() ? 0 : left_it->second;
                             const size_t right_order =
                                 right_it == call_order.end() ? 0 : right_it->second;
                             return left_order < right_order;
                         });

        std::vector<Dsv4ContentBlock>& blocks = mutable_messages[index].content_blocks;
        size_t next = 0;
        for (Dsv4ContentBlock& block : blocks) {
            if (block.kind == Dsv4ContentBlock::Kind::ToolResult) block = tool_blocks[next++];
        }
    }
}

// Step 4: keep reasoning only from the last user turn onward; drop pre-last-user
// developer messages entirely. Only reached when no message declares tools, since
// a tool-bearing prompt forces thinking to be kept.
std::vector<Dsv4PromptMessage> drop_thinking_messages(
    const std::vector<Dsv4PromptMessage>& messages) {
    const int last_user_index = find_last_user_index(messages);
    std::vector<Dsv4PromptMessage> result;
    for (size_t index = 0; index < messages.size(); ++index) {
        const Dsv4PromptMessage& message = messages[index];
        const int position = static_cast<int>(index);
        const bool keep_role = message.role == Dsv4Role::User || message.role == Dsv4Role::System ||
                               message.role == Dsv4Role::Tool ||
                               message.role == Dsv4Role::LatestReminder;
        if (keep_role || position >= last_user_index) {
            result.push_back(message);
        } else if (message.role == Dsv4Role::Assistant) {
            Dsv4PromptMessage stripped = message;
            stripped.reasoning_content.clear();
            result.push_back(std::move(stripped));
        }
    }
    return result;
}

std::string encode_arguments_to_dsml(const std::string& arguments, const std::string& dsml) {
    PromptJson parsed;
    try {
        parsed = PromptJson::parse_object(arguments);
    } catch (const std::exception&) {
        // Reference behaviour: fall back to a single "arguments" parameter.
        PromptJson fallback;
        fallback.kind = PromptJson::Kind::Object;
        PromptJson value;
        value.kind = PromptJson::Kind::String;
        value.text = arguments;
        fallback.keys.emplace_back("arguments");
        fallback.values.push_back(std::move(value));
        parsed = std::move(fallback);
    }

    std::string out;
    for (size_t index = 0; index < parsed.size(); ++index) {
        if (index != 0) out.push_back('\n');
        const std::string& key = parsed.key_at(index);
        const PromptJson& member = parsed.value_at(index);
        const bool is_string = member.kind == PromptJson::Kind::String;
        const std::string value = is_string ? member.text : member.to_python_json();
        out += "<" + dsml + "parameter name=\"" + key + "\" string=\"" +
               (is_string ? "true" : "false") + "\">" + value + "</" + dsml + "parameter>";
    }
    return out;
}

std::string render_tools(const std::vector<Dsv4ToolDefinition>& tools, const Tokens& tokens) {
    std::string schemas;
    for (size_t index = 0; index < tools.size(); ++index) {
        if (index != 0) schemas.push_back('\n');
        schemas += to_reference_json(tools[index].function_json);
    }

    static const char* kTemplate =
        "## Tools\n"
        "\n"
        "You have access to a set of tools to help answer the user's question. You can invoke "
        "tools by writing a \"<{dsml}tool_calls>\" block like the following:\n"
        "\n"
        "<{dsml}tool_calls>\n"
        "<{dsml}invoke name=\"$TOOL_NAME\">\n"
        "<{dsml}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</"
        "{dsml}parameter>\n"
        "...\n"
        "</{dsml}invoke>\n"
        "<{dsml}invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</{dsml}invoke>\n"
        "</{dsml}tool_calls>\n"
        "\n"
        "String parameters should be specified as is and set `string=\"true\"`. For all other "
        "types (numbers, booleans, arrays, objects), pass the value in JSON format and set "
        "`string=\"false\"`.\n"
        "\n"
        "If thinking_mode is enabled (triggered by {think_start}), you MUST output your complete "
        "reasoning inside {think_start}...{think_end} BEFORE any tool calls or final response.\n"
        "\n"
        "Otherwise, output directly after {think_end} with tool calls or final response.\n"
        "\n"
        "### Available Tool Schemas\n"
        "\n"
        "{schemas}\n"
        "\n"
        "You MUST strictly follow the above defined tool name and parameter schemas to invoke "
        "tool calls.\n";

    std::string rendered = kTemplate;
    rendered = replace_all(std::move(rendered), "{dsml}", tokens.dsml);
    rendered = replace_all(std::move(rendered), "{think_start}", tokens.thinking_start);
    rendered = replace_all(std::move(rendered), "{think_end}", tokens.thinking_end);
    rendered = replace_all(std::move(rendered), "{schemas}", schemas);
    return rendered;
}

} // namespace

std::string Dsv4PromptEncoder::encode(const std::vector<Dsv4PromptMessage>& messages,
                                      const Dsv4PromptOptions& options) const {
    const Tokens tokens = resolve_tokens(tokenizer_);

    if (options.reasoning_effort != "low" && options.reasoning_effort != "high" &&
        options.reasoning_effort != "max") {
        throw std::invalid_argument("Invalid reasoning effort: " + options.reasoning_effort);
    }

    const bool thinking = options.thinking_mode == Dsv4ThinkingMode::Thinking;

    const std::vector<Dsv4PromptMessage> merged = merge_tool_messages(messages);
    std::vector<Dsv4PromptMessage> ordered = merged;
    sort_tool_results(merged, ordered);

    std::string prompt;
    if (options.add_default_bos_token) prompt += tokens.bos;

    // Any declared tools force thinking to be kept: dropping reasoning while a tool
    // schema is present silently changes the prefix.
    bool effective_drop_thinking = options.drop_thinking;
    for (const Dsv4PromptMessage& message : ordered) {
        if (!message.tools.empty()) {
            effective_drop_thinking = false;
            break;
        }
    }

    std::vector<Dsv4PromptMessage> render_list = ordered;
    if (thinking && effective_drop_thinking) render_list = drop_thinking_messages(ordered);

    const int last_user_index = find_last_user_index(render_list);
    const size_t count = render_list.size();

    for (size_t index = 0; index < count; ++index) {
        const Dsv4PromptMessage& message = render_list[index];
        const int position = static_cast<int>(index);
        std::string piece;

        // Reasoning-effort prefix applies at the very start, thinking mode only;
        // "low" (the default) contributes nothing.
        if (index == 0 && thinking) {
            if (options.reasoning_effort == "high") piece += kReasoningEffortHigh;
            else if (options.reasoning_effort == "max") piece += kReasoningEffortMax;
        }

        switch (message.role) {
            case Dsv4Role::System:
                piece += message.content;
                if (!message.tools.empty()) piece += "\n\n" + render_tools(message.tools, tokens);
                if (!message.response_format_json.empty()) {
                    piece += "\n\n";
                    piece += kResponseFormatTemplate;
                    piece += to_reference_json(message.response_format_json);
                }
                break;

            case Dsv4Role::Developer: {
                if (message.content.empty()) {
                    throw std::invalid_argument("Developer message requires content");
                }
                std::string developer = tokens.user + message.content;
                if (!message.tools.empty()) developer += "\n\n" + render_tools(message.tools, tokens);
                if (!message.response_format_json.empty()) {
                    developer += "\n\n";
                    developer += kResponseFormatTemplate;
                    developer += to_reference_json(message.response_format_json);
                }
                piece += developer;
                break;
            }

            case Dsv4Role::User: {
                piece += tokens.user;
                if (has_content_blocks(message)) {
                    for (size_t block_index = 0; block_index < message.content_blocks.size();
                         ++block_index) {
                        if (block_index != 0) piece += "\n\n";
                        const Dsv4ContentBlock& block = message.content_blocks[block_index];
                        if (block.kind == Dsv4ContentBlock::Kind::Text) {
                            piece += block.text;
                        } else {
                            piece += kToolResultOpen + block.tool_result + kToolResultClose;
                        }
                    }
                } else {
                    piece += message.content;
                }
                break;
            }

            case Dsv4Role::LatestReminder:
                piece += tokens.latest_reminder + message.content;
                break;

            case Dsv4Role::Tool:
                throw std::runtime_error(
                    "DSV4 has no standalone tool role; merge tool messages into users");

            case Dsv4Role::Assistant: {
                std::string tool_call_content;
                if (!message.tool_calls.empty()) {
                    std::string calls;
                    for (size_t call_index = 0; call_index < message.tool_calls.size();
                         ++call_index) {
                        if (call_index != 0) calls.push_back('\n');
                        const Dsv4ToolCall& call = message.tool_calls[call_index];
                        calls += "<" + tokens.dsml + "invoke name=\"" + call.name + "\">\n" +
                                 encode_arguments_to_dsml(call.arguments, tokens.dsml) + "\n</" +
                                 tokens.dsml + "invoke>";
                    }
                    tool_call_content = "\n\n<" + tokens.dsml + kToolCallsBlockName + ">\n" + calls +
                                        "\n</" + tokens.dsml + kToolCallsBlockName + ">";
                }

                const bool previous_has_task =
                    position > 0 && !render_list[static_cast<size_t>(position - 1)].task.empty();

                std::string thinking_part;
                if (thinking && !previous_has_task) {
                    if (!effective_drop_thinking || position > last_user_index) {
                        thinking_part = message.reasoning_content + tokens.thinking_end;
                    }
                }

                piece += thinking_part + message.content + tool_call_content;
                if (!message.wo_eos) piece += tokens.eos;
                break;
            }
        }

        prompt += piece;

        // Transition tokens are emitted only when the next message continues the
        // generation-adjacent sequence.
        if (index + 1 < count) {
            const Dsv4PromptMessage& next = render_list[index + 1];
            if (!is_assistant(next) && !is_latest_reminder(next)) continue;
        }

        if (!message.task.empty()) {
            const std::string token = tokenizer_.added_token_text(tokens.task_token(message.task));
            if (message.task != "action") {
                prompt += token;
            } else {
                prompt += tokens.assistant;
                prompt += thinking ? tokens.thinking_start : tokens.thinking_end;
                prompt += token;
            }
        } else if (is_user_like(message)) {
            prompt += tokens.assistant;
            if (!effective_drop_thinking && thinking) {
                prompt += tokens.thinking_start;
            } else if (effective_drop_thinking && thinking && position >= last_user_index) {
                prompt += tokens.thinking_start;
            } else {
                prompt += tokens.thinking_end;
            }
        }
    }

    return prompt;
}

std::vector<uint32_t> Dsv4PromptEncoder::encode_tokens(
    const std::vector<Dsv4PromptMessage>& messages, const Dsv4PromptOptions& options) const {
    return tokenizer_.encode(encode(messages, options));
}

} // namespace aeon::text
