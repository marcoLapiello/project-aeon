// -----------------------------------------------------------------------------
// Gate — the OpenAI codec.
//
// One table of valid bodies raising the expected neutral request, one of invalid
// bodies raising the expected {status, code}, and a writer pass that re-parses
// every response with the shared JSON parser and round-trips control characters
// and multi-byte text. The `tools` member is checked byte-for-byte, because a
// re-ordered schema would silently miss the encoder's reference form.
// -----------------------------------------------------------------------------

#include "server/openai_codec.hpp"

#include "infrastructure/json.hpp"

#include <cstdio>
#include <string>
#include <vector>

using aeon::core::JsonValue;
using aeon::server::chat_completion;
using aeon::server::chat_completion_chunk_delta;
using aeon::server::chat_completion_chunk_finish;
using aeon::server::chat_completion_chunk_role;
using aeon::server::chat_completion_chunk_usage;
using aeon::server::CodecError;
using aeon::server::error_json;
using aeon::server::finish_reason_json;
using aeon::server::parse_chat_request;
using aeon::server::Timings;
using aeon::server::Usage;
using aeon::session::Channel;
using aeon::session::Role;

namespace {

uint32_t checks = 0;
uint32_t failures = 0;

void expect(bool ok, const std::string& label) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("  FAIL  %s\n", label.c_str());
    }
}

void expect_ok(const std::string& body, const std::string& label) {
    bool ok = true;
    try {
        (void)parse_chat_request(body);
    } catch (const std::exception&) {
        ok = false;
    }
    expect(ok, "valid: " + label);
}

void expect_error(const std::string& body, int status, const std::string& code,
                  const std::string& label) {
    bool matched = false;
    try {
        (void)parse_chat_request(body);
    } catch (const CodecError& error) {
        matched = error.http_status == status && error.code == code;
    } catch (const std::exception&) {
        matched = false;
    }
    expect(matched, "invalid (" + std::to_string(status) + " " + code + "): " + label);
}

const char* kUser = R"({"messages":[{"role":"user","content":"hi"}]})";

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  openai codec\n");
    std::printf("================================================================================\n");

    // --- valid bodies ---------------------------------------------------------
    {
        const auto parsed = parse_chat_request(kUser);
        expect(parsed.request.messages.size() == 1 &&
                   parsed.request.messages[0].role == Role::User &&
                   parsed.request.messages[0].content == "hi",
               "simple user message maps to a neutral Message");
        expect(!parsed.stream && !parsed.include_usage, "stream defaults off");
        expect(parsed.seed != 0, "an absent seed is drawn fresh");
        expect(parsed.request.sampling.seed.has_value() &&
                   *parsed.request.sampling.seed == parsed.seed,
               "the drawn seed is installed on the request");
    }
    expect_ok(R"({"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}})",
              "stream + include_usage");
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}})");
        expect(parsed.stream && parsed.include_usage, "stream and include_usage are read");
    }
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]})");
        expect(parsed.request.messages[0].content == "ab", "text content parts concatenate");
    }
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get","arguments":"{}"}}]}]})");
        expect(parsed.request.messages[0].role == Role::Assistant &&
                   parsed.request.messages[0].tool_calls.size() == 1 &&
                   parsed.request.messages[0].tool_calls[0].name == "get",
               "assistant with null content and tool_calls");
    }
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"tool","content":"result","tool_call_id":"call_1"}]})");
        expect(parsed.request.messages[0].role == Role::Tool &&
                   parsed.request.messages[0].tool_call_id == "call_1",
               "tool message keeps its tool_call_id");
    }
    {
        // A `tools` value with a nested "tools" key, braces and escaped quotes
        // inside strings: the span must be preserved byte-for-byte.
        const std::string body =
            R"({"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"z","description":"say \"tools\" { not a member }","parameters":{"type":"object","properties":{"nested":{"tools":1}}}}}]})";
        const auto parsed = parse_chat_request(body);
        const std::string expected =
            R"([{"type":"function","function":{"name":"z","description":"say \"tools\" { not a member }","parameters":{"type":"object","properties":{"nested":{"tools":1}}}}}] )";
        // Trim the trailing space used to keep the raw literal readable.
        expect(parsed.request.tools_json == expected.substr(0, expected.size() - 1),
               "the tools span is preserved byte-for-byte");
    }
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"enable_thinking":true}})");
        expect(parsed.request.thinking, "thinking via chat_template_kwargs.enable_thinking");
    }
    {
        // The spelling qwen / sglang / vLLM actually send: top-level.
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":"x"}],"enable_thinking":true})");
        expect(parsed.request.thinking, "thinking via top-level enable_thinking (qwen)");
    }
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":"x"}],"thinking":{"type":"enabled"}})");
        expect(parsed.request.thinking, "thinking via top-level thinking:{type:enabled}");
    }
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":"x"}],"enable_thinking":true,"reasoning_effort":"none"})");
        expect(!parsed.request.thinking && parsed.request.reasoning_effort.empty(),
               "reasoning_effort:none overrides a top-level enable_thinking");
    }
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"high"})");
        expect(parsed.request.thinking && parsed.request.reasoning_effort == "high",
               "reasoning_effort:high enables thinking");
    }
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"medium"})");
        expect(parsed.request.reasoning_effort == "low", "medium maps to low");
    }
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"none"})");
        expect(!parsed.request.thinking && parsed.request.reasoning_effort.empty(),
               "none disables thinking");
    }
    {
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":"x"}],"max_tokens":7,"temperature":0.5,"top_p":0.9,"seed":42})");
        expect(parsed.request.max_new_tokens.has_value() && *parsed.request.max_new_tokens == 7,
               "max_tokens is read");
        expect(parsed.request.sampling.temperature.has_value() &&
                   *parsed.request.sampling.temperature == 0.5f,
               "temperature is read");
        expect(parsed.seed == 42, "an explicit seed is honoured");
    }
    expect_ok(R"({"messages":[{"role":"user","content":"x"}],"n":1})", "n=1 accepted");
    expect_ok(R"({"messages":[{"role":"user","content":"x"}],"stop":null,"logit_bias":{},"presence_penalty":0})",
              "null/empty/zero unsupported fields accepted");
    {
        // A surrogate-pair escape must decode to one astral code point, not two lone
        // surrogates (which are not valid UTF-8 and would be rejected downstream).
        const auto parsed = parse_chat_request(
            R"({"messages":[{"role":"user","content":"hi \ud83d\ude00"}]})");
        expect(parsed.request.messages[0].content == "hi \xF0\x9F\x98\x80",
               "a surrogate-pair escape decodes to one astral code point");
    }

    // --- invalid bodies -------------------------------------------------------
    expect_error("{not json}", 400, "invalid_request_error", "malformed JSON");
    expect_error("[]", 400, "invalid_request_error", "body not an object");
    expect_error(R"({"messages":[]})", 400, "invalid_request_error", "empty messages");
    expect_error(R"({"messages":[{"role":"wizard","content":"x"}]})", 400,
                 "invalid_request_error", "unknown role");
    expect_error(R"({"messages":[{"role":"user","content":5}]})", 400,
                 "invalid_request_error", "content wrong type");
    expect_error(R"({"messages":[{"role":"user","content":[{"type":"image","image_url":"x"}]}]})",
                 400, "unsupported_parameter", "non-text content part");
    expect_error(R"({"messages":[{"role":"tool","content":"r"}]})", 400,
                 "invalid_request_error", "tool without tool_call_id");
    expect_error(R"({"messages":[{"role":"user","content":"x"}],"n":2})", 400,
                 "invalid_request_error", "n must be 1");
    expect_error(R"({"messages":[{"role":"user","content":"x"}],"stop":"END"})", 400,
                 "unsupported_parameter", "stop strings");
    expect_error(R"({"messages":[{"role":"user","content":"x"}],"logit_bias":{"1":1}})", 400,
                 "unsupported_parameter", "logit_bias");
    expect_error(R"({"messages":[{"role":"user","content":"x"}],"logprobs":true})", 400,
                 "unsupported_parameter", "logprobs");
    expect_error(R"({"messages":[{"role":"user","content":"x"}],"presence_penalty":1})", 400,
                 "unsupported_parameter", "non-zero penalty");
    expect_error(R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"weird"})", 400,
                 "invalid_request_error", "bad reasoning_effort");

    // --- writers --------------------------------------------------------------
    const std::string text = "line1\n\"quoted\"\\slash\ttab café 😀";
    {
        const std::string json = chat_completion_chunk_delta("id", "m", Channel::Content, text);
        const JsonValue value = JsonValue::parse(json);
        const JsonValue& delta = value.at("choices").as_array()[0].at("delta");
        expect(delta.at("content").as_string() == text,
               "content delta round-trips control characters and multi-byte text");
        expect(value.at("object").as_string() == "chat.completion.chunk", "chunk object name");
    }
    {
        const std::string json =
            chat_completion_chunk_delta("id", "m", Channel::Reasoning, "thinking");
        const JsonValue value = JsonValue::parse(json);
        expect(value.at("choices").as_array()[0].at("delta").at("reasoning_content").as_string() ==
                   "thinking",
               "reasoning delta lands in reasoning_content");
    }
    {
        const JsonValue role = JsonValue::parse(chat_completion_chunk_role("id", "m"));
        expect(role.at("choices").as_array()[0].at("delta").at("role").as_string() ==
                   "assistant",
               "role chunk carries the assistant role");
    }
    expect(finish_reason_json(aeon::text::StopReason::Eos) == "\"stop\"", "Eos -> stop");
    expect(finish_reason_json(aeon::text::StopReason::MaxNewTokens) == "\"length\"",
           "MaxNewTokens -> length");
    expect(finish_reason_json(aeon::text::StopReason::ContextLimit) == "\"length\"",
           "ContextLimit -> length");
    expect(finish_reason_json(aeon::text::StopReason::Cancelled) == "null",
           "Cancelled -> null");
    {
        const JsonValue value =
            JsonValue::parse(chat_completion_chunk_finish("id", "m", aeon::text::StopReason::Eos));
        expect(value.at("choices").as_array()[0].at("finish_reason").as_string() == "stop",
               "finish chunk carries finish_reason");
    }
    {
        Usage usage{3, 5, 8};
        const JsonValue value = JsonValue::parse(chat_completion_chunk_usage("id", "m", usage));
        expect(value.at("usage").at("total_tokens").as_int64() == 8, "usage chunk totals");
        expect(value.at("choices").as_array().empty(), "usage chunk has empty choices");
    }
    {
        Usage usage{3, 5, 8};
        Timings timings;
        timings.ttft_ms = 12.5;
        timings.decode_tokens_per_second = 42.0;
        timings.reused_tokens = 10;
        timings.prefilled_tokens = 2;
        timings.reuse_verdict = "reused";
        timings.seed = 99;
        const std::string json =
            chat_completion("id", "m", "reason", text, aeon::text::StopReason::Eos, usage, timings);
        const JsonValue value = JsonValue::parse(json);
        const JsonValue& message = value.at("choices").as_array()[0].at("message");
        expect(message.at("content").as_string() == text, "completion content round-trips");
        expect(message.at("reasoning_content").as_string() == "reason",
               "completion reasoning round-trips");
        expect(value.at("timings").at("reuse_verdict").as_string() == "reused",
               "timings carries the reuse verdict");
        expect(value.at("timings").at("seed").as_int64() == 99, "timings echoes the seed");
    }
    {
        const JsonValue value =
            JsonValue::parse(error_json(CodecError{400, "context_length_exceeded", "too long"}));
        expect(value.at("error").at("code").as_string() == "context_length_exceeded",
               "error object carries the code");
        expect(value.at("error").at("message").as_string() == "too long",
               "error object carries the message");
    }

    std::printf("================================================================================\n");
    std::printf("  %u/%u checks passed\n", checks - failures, checks);
    std::printf("================================================================================\n");
    return failures == 0 ? 0 : 1;
}
