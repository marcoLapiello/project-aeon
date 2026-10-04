#include "server/openai_codec.hpp"

#include "infrastructure/json.hpp"
#include "server/json_span.hpp"
#include "server/json_write.hpp"

#include <atomic>
#include <ctime>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace aeon::server {

namespace {

using aeon::core::JsonValue;

[[noreturn]] void fail(int status, std::string code, std::string message) {
    throw CodecError{status, std::move(code), std::move(message)};
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(400, "invalid_request_error", message);
}

uint64_t draw_seed() {
    static std::atomic<uint64_t> counter{0};
    std::random_device device;
    const uint64_t a = static_cast<uint64_t>(device());
    const uint64_t b = static_cast<uint64_t>(device());
    return (a << 32) ^ b ^ (counter.fetch_add(1, std::memory_order_relaxed) * 0x9E3779B97F4A7C15ULL);
}

session::Role parse_role(const std::string& role) {
    if (role == "system") return session::Role::System;
    if (role == "developer") return session::Role::Developer;
    if (role == "user") return session::Role::User;
    if (role == "assistant") return session::Role::Assistant;
    if (role == "tool") return session::Role::Tool;
    fail(400, "invalid_request_error", "unknown role: " + role);
}

std::string parse_content(const JsonValue& message) {
    const JsonValue* content = message.find("content");
    if (content == nullptr || content->is_null()) return "";
    if (content->is_string()) return content->as_string();
    if (content->is_array()) {
        std::string out;
        for (const JsonValue& part : content->as_array()) {
            require(part.is_object(), "each `content` part must be an object");
            const JsonValue* type = part.find("type");
            require(type != nullptr && type->is_string(),
                    "each `content` part needs a string `type`");
            if (type->as_string() != "text") {
                fail(400, "unsupported_parameter",
                     "content part type `" + type->as_string() + "` is not supported");
            }
            const JsonValue* text = part.find("text");
            require(text != nullptr && text->is_string(),
                    "a text content part needs string `text`");
            out += text->as_string();
        }
        return out;
    }
    fail(400, "invalid_request_error",
         "`content` must be a string, null, or an array of text parts");
}

std::vector<session::ToolCall> parse_tool_calls(const JsonValue& message) {
    std::vector<session::ToolCall> out;
    const JsonValue* calls = message.find("tool_calls");
    if (calls == nullptr || calls->is_null()) return out;
    require(calls->is_array(), "`tool_calls` must be an array");
    for (const JsonValue& call : calls->as_array()) {
        require(call.is_object(), "each tool call must be an object");
        session::ToolCall tool_call;
        if (const JsonValue* id = call.find("id"); id != nullptr && id->is_string()) {
            tool_call.id = id->as_string();
        }
        const JsonValue* function = call.find("function");
        require(function != nullptr && function->is_object(),
                "a tool call needs a `function` object");
        if (const JsonValue* name = function->find("name");
            name != nullptr && name->is_string()) {
            tool_call.name = name->as_string();
        }
        const JsonValue* arguments = function->find("arguments");
        require(arguments != nullptr && arguments->is_string(),
                "a tool call's `function.arguments` must be a JSON string");
        tool_call.arguments_json = arguments->as_string();
        out.push_back(std::move(tool_call));
    }
    return out;
}

bool is_empty_value(const JsonValue* value) {
    if (value == nullptr || value->is_null()) return true;
    if (value->is_string()) return value->as_string().empty();
    if (value->is_array()) return value->as_array().empty();
    if (value->is_object()) return value->as_object().empty();
    return false;
}

}  // namespace

ParsedRequest parse_chat_request(std::string_view body) {
    JsonValue document;
    try {
        document = JsonValue::parse(std::string(body));
    } catch (const std::exception& error) {
        fail(400, "invalid_request_error", std::string("malformed JSON: ") + error.what());
    }
    require(document.is_object(), "the request body must be a JSON object");

    ParsedRequest parsed;
    session::ConversationRequest& request = parsed.request;

    // --- messages -----------------------------------------------------------
    const JsonValue* messages = document.find("messages");
    require(messages != nullptr && messages->is_array() && !messages->as_array().empty(),
            "`messages` must be a non-empty array");
    for (const JsonValue& node : messages->as_array()) {
        require(node.is_object(), "each message must be an object");
        const JsonValue* role = node.find("role");
        require(role != nullptr && role->is_string(), "each message needs a string `role`");

        session::Message message;
        message.role = parse_role(role->as_string());
        message.content = parse_content(node);
        if (const JsonValue* reasoning = node.find("reasoning_content");
            reasoning != nullptr && reasoning->is_string()) {
            message.reasoning_content = reasoning->as_string();
        }
        if (const JsonValue* call_id = node.find("tool_call_id");
            call_id != nullptr && call_id->is_string()) {
            message.tool_call_id = call_id->as_string();
        }
        if (message.role == session::Role::Tool && message.tool_call_id.empty()) {
            fail(400, "invalid_request_error", "a tool message requires `tool_call_id`");
        }
        message.tool_calls = parse_tool_calls(node);
        request.messages.push_back(std::move(message));
    }

    // --- raw passthroughs ---------------------------------------------------
    if (const auto span = raw_member(body, "tools"); span.has_value() && *span != "null") {
        request.tools_json = std::string(*span);
    }
    if (const auto span = raw_member(body, "response_format");
        span.has_value() && *span != "null") {
        request.response_format_json = std::string(*span);
    }

    // --- thinking -----------------------------------------------------------
    // There is no single "OpenAI" field for the thinking toggle: vLLM reads
    // `chat_template_kwargs.enable_thinking`, vLLM/sglang/qwen send a top-level
    // `enable_thinking`, DeepSeek's own API uses `thinking: {"type": ...}`, and
    // OpenAI's `reasoning_effort` implies it. Be liberal and accept all of them —
    // a client must not need one vendor's spelling to get the behaviour it asked
    // for. An explicit enable from any spelling turns thinking on; an explicit
    // disable or `reasoning_effort: "none"` turns it off; `reasoning_effort` is
    // evaluated last so it is authoritative when present.
    bool thinking = false;
    const auto read_toggle = [](const JsonValue* value) -> std::optional<bool> {
        if (value == nullptr || value->is_null()) return std::nullopt;
        if (value->is_bool()) return value->as_bool();
        if (value->is_object()) {  // DeepSeek's `thinking: {"type": "enabled"}`
            if (const JsonValue* type = value->find("type");
                type != nullptr && type->is_string()) {
                if (type->as_string() == "enabled") return true;
                if (type->as_string() == "disabled") return false;
            }
        }
        return std::nullopt;
    };
    if (const JsonValue* kwargs = document.find("chat_template_kwargs");
        kwargs != nullptr && kwargs->is_object()) {
        for (const char* key : {"thinking", "enable_thinking"}) {
            if (auto toggle = read_toggle(kwargs->find(key)); toggle.has_value()) {
                thinking = *toggle;
            }
        }
    }
    for (const char* key : {"enable_thinking", "thinking"}) {
        if (auto toggle = read_toggle(document.find(key)); toggle.has_value()) {
            thinking = *toggle;
        }
    }
    std::string effort;
    if (const JsonValue* value = document.find("reasoning_effort");
        value != nullptr && !value->is_null()) {
        require(value->is_string(), "`reasoning_effort` must be a string");
        effort = value->as_string();
        if (effort == "none") {
            thinking = false;
            effort.clear();
        } else if (effort == "minimal" || effort == "medium") {
            thinking = true;
            effort = "low";
        } else if (effort == "low" || effort == "high" || effort == "max") {
            thinking = true;
        } else {
            fail(400, "invalid_request_error",
                 "`reasoning_effort` must be one of low|high|max");
        }
    }
    request.thinking = thinking;
    request.reasoning_effort = effort;

    // --- token cap ----------------------------------------------------------
    for (const char* key : {"max_tokens", "max_completion_tokens"}) {
        if (const JsonValue* value = document.find(key); value != nullptr && !value->is_null()) {
            require(value->is_number(), std::string("`") + key + "` must be a number");
            const int64_t cap = value->as_int64();
            require(cap > 0, std::string("`") + key + "` must be positive");
            request.max_new_tokens = static_cast<uint32_t>(cap);
        }
    }

    // --- sampling -----------------------------------------------------------
    if (const JsonValue* value = document.find("temperature");
        value != nullptr && !value->is_null()) {
        require(value->is_number(), "`temperature` must be a number");
        request.sampling.temperature = static_cast<float>(value->as_number());
    }
    if (const JsonValue* value = document.find("top_p"); value != nullptr && !value->is_null()) {
        require(value->is_number(), "`top_p` must be a number");
        request.sampling.top_p = static_cast<float>(value->as_number());
    }
    if (const JsonValue* value = document.find("seed"); value != nullptr && !value->is_null()) {
        require(value->is_number(), "`seed` must be a number");
        parsed.seed = static_cast<uint64_t>(value->as_int64());
    } else {
        // A fixed default would make two identical sampled requests return
        // identical replies; draw one per request and echo it in `timings`.
        parsed.seed = draw_seed();
    }
    request.sampling.seed = parsed.seed;

    // --- stream -------------------------------------------------------------
    if (const JsonValue* value = document.find("stream"); value != nullptr && !value->is_null()) {
        require(value->is_bool(), "`stream` must be a boolean");
        parsed.stream = value->as_bool();
    }
    if (const JsonValue* options = document.find("stream_options");
        options != nullptr && options->is_object()) {
        if (const JsonValue* include = options->find("include_usage");
            include != nullptr && include->is_bool()) {
            parsed.include_usage = include->as_bool();
        }
    }

    // --- reject what we do not honour --------------------------------------
    if (const JsonValue* value = document.find("n"); value != nullptr && !value->is_null()) {
        require(value->is_number(), "`n` must be a number");
        require(value->as_int64() == 1, "`n` must be 1");
    }
    for (const char* key : {"stop", "logit_bias"}) {
        if (!is_empty_value(document.find(key))) {
            fail(400, "unsupported_parameter", std::string("`") + key + "` is not supported");
        }
    }
    if (const JsonValue* value = document.find("logprobs");
        value != nullptr && !value->is_null()) {
        const bool active =
            (value->is_bool() && value->as_bool()) ||
            (value->is_number() && value->as_number() != 0.0);
        if (active) fail(400, "unsupported_parameter", "`logprobs` is not supported");
    }
    for (const char* key : {"presence_penalty", "frequency_penalty", "repetition_penalty"}) {
        if (const JsonValue* value = document.find(key);
            value != nullptr && !value->is_null()) {
            require(value->is_number(), std::string("`") + key + "` must be a number");
            if (value->as_number() != 0.0) {
                fail(400, "unsupported_parameter",
                     std::string("`") + key + "` is not supported");
            }
        }
    }

    return parsed;
}

// --- writers -----------------------------------------------------------------

std::string finish_reason_json(text::StopReason stop) {
    switch (stop) {
        case text::StopReason::Eos: return "\"stop\"";
        case text::StopReason::MaxNewTokens: return "\"length\"";
        case text::StopReason::ContextLimit: return "\"length\"";
        case text::StopReason::Cancelled: return "null";
        case text::StopReason::Error: return "null";
    }
    return "null";
}

std::string new_completion_id() {
    static std::atomic<uint64_t> counter{0};
    char buffer[48];
    std::snprintf(buffer, sizeof(buffer), "chatcmpl-%llx%llx",
                  static_cast<unsigned long long>(std::time(nullptr)),
                  static_cast<unsigned long long>(counter.fetch_add(1)));
    return buffer;
}

namespace {
void write_chunk_envelope(JsonWriter& writer, const std::string& id, const std::string& model,
                          std::string_view choices_body) {
    writer.begin_object()
        .key("id").string(id)
        .key("object").string("chat.completion.chunk")
        .key("created").integer(static_cast<int64_t>(std::time(nullptr)))
        .key("model").string(model)
        .key("choices").raw(choices_body)
        .end_object();
}
}  // namespace

std::string chat_completion_chunk_role(const std::string& id, const std::string& model) {
    std::string out;
    JsonWriter writer(out);
    write_chunk_envelope(writer, id, model,
                         "[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
                         "\"finish_reason\":null}]");
    return out;
}

std::string chat_completion_chunk_delta(const std::string& id, const std::string& model,
                                        session::Channel channel, const std::string& text) {
    std::string delta;
    {
        JsonWriter writer(delta);
        writer.begin_object()
            .key(channel == session::Channel::Reasoning ? "reasoning_content" : "content")
            .string(text)
            .end_object();
    }
    std::string choices = "[{\"index\":0,\"delta\":";
    choices += delta;
    choices += ",\"finish_reason\":null}]";

    std::string out;
    JsonWriter writer(out);
    write_chunk_envelope(writer, id, model, choices);
    return out;
}

std::string chat_completion_chunk_finish(const std::string& id, const std::string& model,
                                         text::StopReason stop) {
    std::string choices = "[{\"index\":0,\"delta\":{},\"finish_reason\":";
    choices += finish_reason_json(stop);
    choices += "}]";

    std::string out;
    JsonWriter writer(out);
    write_chunk_envelope(writer, id, model, choices);
    return out;
}

std::string chat_completion_chunk_usage(const std::string& id, const std::string& model,
                                        const Usage& usage) {
    std::string out;
    JsonWriter writer(out);
    writer.begin_object()
        .key("id").string(id)
        .key("object").string("chat.completion.chunk")
        .key("created").integer(static_cast<int64_t>(std::time(nullptr)))
        .key("model").string(model)
        .key("choices").raw("[]")
        .key("usage").begin_object()
            .key("prompt_tokens").integer(usage.prompt_tokens)
            .key("completion_tokens").integer(usage.completion_tokens)
            .key("total_tokens").integer(usage.total_tokens)
        .end_object()
        .end_object();
    return out;
}

std::string chat_completion(const std::string& id, const std::string& model,
                            const std::string& reasoning_content, const std::string& content,
                            text::StopReason stop, const Usage& usage,
                            const Timings& timings) {
    std::string out;
    JsonWriter writer(out);
    writer.begin_object()
        .key("id").string(id)
        .key("object").string("chat.completion")
        .key("created").integer(static_cast<int64_t>(std::time(nullptr)))
        .key("model").string(model)
        .key("choices").begin_array().begin_object()
            .key("index").integer(0)
            .key("message").begin_object()
                .key("role").string("assistant")
                .key("content").string(content);
    if (!reasoning_content.empty()) {
        writer.key("reasoning_content").string(reasoning_content);
    }
    writer.end_object()
            .key("finish_reason").raw(finish_reason_json(stop))
        .end_object().end_array()
        .key("usage").begin_object()
            .key("prompt_tokens").integer(usage.prompt_tokens)
            .key("completion_tokens").integer(usage.completion_tokens)
            .key("total_tokens").integer(usage.total_tokens)
        .end_object()
        .key("timings").begin_object()
            .key("ttft_ms").number(timings.ttft_ms)
            .key("decode_tokens_per_second").number(timings.decode_tokens_per_second)
            .key("reused_tokens").integer(timings.reused_tokens)
            .key("prefilled_tokens").integer(timings.prefilled_tokens)
            .key("reuse_verdict").string(timings.reuse_verdict)
            .key("seed").integer(static_cast<int64_t>(timings.seed))
        .end_object()
        .end_object();
    return out;
}

std::string error_json(const CodecError& error) {
    std::string out;
    JsonWriter writer(out);
    writer.begin_object()
        .key("error").begin_object()
            .key("message").string(error.message)
            .key("type").string(error.code)
            .key("code").string(error.code)
        .end_object()
        .end_object();
    return out;
}

}  // namespace aeon::server
