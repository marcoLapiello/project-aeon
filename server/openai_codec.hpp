#pragma once

// -----------------------------------------------------------------------------
// The OpenAI `/v1/chat/completions` codec.
//
// One direction parses a request body into the neutral `ConversationRequest`; the
// other writes the two response shapes (streamed chunks and the single object) and
// the error object. Everything the model understands but OpenAI does not is
// rejected rather than silently dropped — a `stop` string that was ignored would
// change the reply, so accepting it would be a lie.
//
// `tools` and `response_format` are carried as raw source text (see `json_span`),
// because the prompt encoder re-serializes them in reference form and a round-trip
// through `JsonValue` would reorder their keys.
// -----------------------------------------------------------------------------

#include "infrastructure/session/conversation.hpp"
#include "infrastructure/text/text_generation.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace aeon::server {

struct CodecError {
    int http_status{400};
    std::string code{"invalid_request_error"};
    std::string message;

    [[noreturn]] void raise() const { throw *this; }
};

struct ParsedRequest {
    session::ConversationRequest request;
    bool stream{false};
    bool include_usage{false};
    uint64_t seed{0};  // the effective seed, echoed in `timings`
};

// Parses a chat-completions body. Throws `CodecError` on anything invalid.
ParsedRequest parse_chat_request(std::string_view body);

// --- response writers --------------------------------------------------------

struct Usage {
    uint32_t prompt_tokens{0};
    uint32_t completion_tokens{0};
    uint32_t total_tokens{0};
};

struct Timings {
    double ttft_ms{0.0};
    double decode_tokens_per_second{0.0};
    uint32_t reused_tokens{0};
    uint32_t prefilled_tokens{0};
    std::string reuse_verdict;
    uint64_t seed{0};
};

// The `finish_reason` for a stop reason: `"stop"`, `"length"`, or `null`.
std::string finish_reason_json(text::StopReason stop);

// A fresh `chatcmpl-...` id.
std::string new_completion_id();

std::string chat_completion_chunk_role(const std::string& id, const std::string& model);
std::string chat_completion_chunk_delta(const std::string& id, const std::string& model,
                                        session::Channel channel, const std::string& text);
std::string chat_completion_chunk_finish(const std::string& id, const std::string& model,
                                         text::StopReason stop);
std::string chat_completion_chunk_usage(const std::string& id, const std::string& model,
                                        const Usage& usage);

std::string chat_completion(const std::string& id, const std::string& model,
                            const std::string& reasoning_content, const std::string& content,
                            text::StopReason stop, const Usage& usage,
                            const Timings& timings);

std::string error_json(const CodecError& error);

}  // namespace aeon::server
