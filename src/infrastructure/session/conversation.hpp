#pragma once

// -----------------------------------------------------------------------------
// The conversation seam — the one neutral definition of "one conversation".
//
// Below this header: the engine, which knows the model, the weights and the
// kernels. Above it: the serving layer, which knows an HTTP request and an SSE
// stream. Neither names the other's types. This is the only contract between
// them, and it is deliberately model-agnostic: a conversation is a list of
// messages, tools as raw JSON text, sampling that the artifact may supply, and a
// stream of decoded *text* deltas out.
//
// It is G1 (engine strategy): the serving layer may include it and nothing else
// from the engine, and the CLI and the tests use the same seam, so "one
// conversation" has exactly one definition in the codebase.
//
// The text is decoded here, not at the transport: the server must never tokenize
// or detokenize, so `TextDelta` carries ready-to-send UTF-8. Whether a delta is
// answer text or reasoning text is model knowledge (the thinking split), so it is
// expressed as a channel the adapter fills rather than a field the server
// interprets.
// -----------------------------------------------------------------------------

#include "infrastructure/session/prefix_record.hpp"
#include "infrastructure/text/text_generation.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::session {

enum class Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

struct Message {
    Role role{Role::User};
    std::string content;
    std::string reasoning_content;
    std::string tool_call_id;  // Role::Tool: which call this answers
    std::vector<ToolCall> tool_calls;
};

// Per-request sampling. An unset field means "use the artifact's own value".
struct Sampling {
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<uint64_t> seed;
};

struct ConversationRequest {
    std::vector<Message> messages;
    std::string tools_json;            // raw JSON array exactly as sent; "" = none
    std::string response_format_json;  // raw; "" = none
    bool thinking{false};
    std::string reasoning_effort;      // "" = engine default
    std::optional<uint32_t> max_new_tokens;  // unset = until EOS or context
    Sampling sampling;
};

// Which channel a delta belongs to. The adapter decides (the thinking split);
// the transport only routes it to the matching JSON field.
enum class Channel {
    Content,
    Reasoning,
};

struct TextDelta {
    Channel channel{Channel::Content};
    std::string text;
};

using DeltaSink = std::function<void(const TextDelta&)>;

struct ConversationResult {
    text::StopReason stop{text::StopReason::Error};
    uint32_t prompt_tokens{0};
    uint32_t completion_tokens{0};
    uint32_t reused_tokens{0};
    uint32_t prefilled_tokens{0};
    ReuseVerdict verdict{ReuseVerdict::Cold};
    double ttft_ms{0.0};
    double decode_tokens_per_second{0.0};
};

enum class ErrorKind {
    InvalidRequest,
    ContextOverflow,
    Internal,
};

struct ConversationError : std::runtime_error {
    ConversationError(ErrorKind k, const std::string& message)
        : std::runtime_error(message), kind(k) {}
    ErrorKind kind{ErrorKind::Internal};
};

// What the engine can say about itself without running: enough for `/status` and
// `/v1/models`.
struct ConversationInfo {
    std::string model_id;
    uint32_t context_capacity{0};
    uint32_t resident_tokens{0};
    // Whether the engine's supply invariants hold after the last turn: no broken
    // registry, no leaked expert lease, no staging slot still held.
    bool invariants_ok{true};
};

// The engine as the serving layer sees it. One conversation at a time, driven on
// one thread; `run` throws `ConversationError` for anything the caller can act on.
class ConversationEngine {
public:
    virtual ~ConversationEngine() = default;

    virtual ConversationInfo info() const = 0;

    virtual ConversationResult run(const ConversationRequest& request,
                                   const DeltaSink& sink,
                                   const std::function<bool()>& cancelled) = 0;

    // Forget the resident session after a failure, so the next turn is a cold run
    // rather than an attempt to build on state the failure may have corrupted.
    virtual void reset() noexcept = 0;
};

}  // namespace aeon::session
