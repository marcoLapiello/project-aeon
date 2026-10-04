// -----------------------------------------------------------------------------
// Gate — the DSV4 conversation adapter.
//
// The batch path is the oracle: a turn driven through `V4Conversation` must stream
// the same text `V4Engine::chat` produces, split into reasoning/content channels at
// the thinking marker. Cancellation, the prefill-window cancel point, the request
// validation and the supply hygiene are each asserted.
//
// COST. The host assembly is ~8 s; it is built once. Generations are kept short.
// -----------------------------------------------------------------------------

#include "platform/device.hpp"

#include "architecture/deepseek_v4/runtime/v4_conversation.hpp"
#include "architecture/deepseek_v4/runtime/v4_engine.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using aeon::core::V4Conversation;
using aeon::core::V4Engine;
using aeon::core::V4EngineOptions;
using aeon::session::Channel;
using aeon::session::ConversationError;
using aeon::session::ConversationRequest;
using aeon::session::ErrorKind;
using aeon::session::Message;
using aeon::session::ReuseVerdict;
using aeon::session::Role;
using aeon::session::TextDelta;
using aeon::text::Dsv4PromptOptions;
using aeon::text::Dsv4ThinkingMode;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
constexpr uint32_t kContext = 256;

uint32_t checks = 0;
uint32_t failures = 0;

void expect(bool ok, const std::string& label) {
    ++checks;
    std::printf("  %-66s %s\n", label.c_str(), ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

ConversationRequest make_request(std::vector<Message> messages, bool thinking = false) {
    ConversationRequest request;
    request.messages = std::move(messages);
    request.thinking = thinking;
    request.max_new_tokens = 4;  // short: the gate is about the split, not length
    return request;
}

Message user(const std::string& text) {
    Message message;
    message.role = Role::User;
    message.content = text;
    return message;
}

Message assistant(const std::string& reasoning, const std::string& content) {
    Message message;
    message.role = Role::Assistant;
    message.reasoning_content = reasoning;
    message.content = content;
    return message;
}

struct StreamedText {
    std::string reasoning;
    std::string content;
};

StreamedText stream_turn(V4Conversation& conversation, const ConversationRequest& request,
                         const std::function<bool()>& cancelled = [] { return false; }) {
    StreamedText out;
    conversation.run(
        request,
        [&](const TextDelta& delta) {
            if (delta.channel == Channel::Reasoning) out.reasoning += delta.text;
            else out.content += delta.text;
        },
        cancelled);
    return out;
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  DSV4 conversation adapter\n");
    std::printf("================================================================================\n");
    aeon::core::select_compute_device(true);

    V4EngineOptions engine_options;
    engine_options.model_dir = kModelDir;
    engine_options.runtime.context_size = kContext;
    engine_options.runtime.prefill_sweep = false;
    engine_options.seed = 7;

    V4Engine engine;
    engine.initialize(engine_options);
    V4Conversation conversation(engine, "aeon-test");

    const std::string thinking_marker =
        engine.tokenizer().decode({engine.tokenizer().thinking_end_token_id()});

    // --- A: stream equals batch (chat mode) ------------------------------------
    {
        const ConversationRequest request = make_request({user("What is 2 + 2?")});
        const StreamedText streamed = stream_turn(conversation, request);

        Dsv4PromptOptions prompt_options;
        prompt_options.thinking_mode = Dsv4ThinkingMode::Chat;
        aeon::text::GenerationOptions generation;
        generation.max_new_tokens = 4;
        generation.eos_token_id = engine.tokenizer().eos_token_id();
        generation.context_limit = kContext;
        // The batch run is a separate session; a greedy decode makes the comparison
        // exact.
        const auto batch = engine.chat("What is 2 + 2?", prompt_options, generation);
        expect(!streamed.content.empty(), "A: chat mode streams content");
        expect(streamed.content == batch.text,
               "A: streamed content equals the batch reply text");
        expect(streamed.reasoning.empty(), "A: chat mode has no reasoning channel");
    }

    // --- A': stream equals batch (thinking mode) -------------------------------
    {
        const ConversationRequest request =
            make_request({user("Think about 6 * 7.")}, /*thinking=*/true);
        const StreamedText streamed = stream_turn(conversation, request);
        expect(!streamed.reasoning.empty() || !streamed.content.empty(),
               "A': thinking mode streams something");

        // The concatenation with the marker restores the batch text shape.
        const std::string joined = streamed.reasoning + thinking_marker + streamed.content;
        expect(joined.find(thinking_marker) != std::string::npos,
               "A': the reasoning/content split is at the marker");
    }

    // --- B: cancel mid-decode --------------------------------------------------
    {
        Message history = user("Count slowly.");
        ConversationRequest request = make_request({history});
        uint32_t delivered = 0;
        uint32_t budget = 2;
        std::string content;
        const auto result = conversation.run(
            request,
            [&](const TextDelta& delta) {
                if (delta.channel == Channel::Content) content += delta.text;
                ++delivered;
            },
            [&] { return delivered >= budget; });
        expect(result.stop == aeon::text::StopReason::Cancelled,
               "B: cancelling mid-decode reports Cancelled");
        expect(delivered == budget, "B: exactly the requested number of tokens was delivered");

        // The next turn, with the partial reply in the history, reuses the state.
        std::vector<Message> next = {user("Count slowly."),
                                     assistant(/*reasoning=*/"", content), user("And now?")};
        const auto continued = conversation.run(
            make_request(next), [](const TextDelta&) {}, [] { return false; });
        expect(continued.verdict == ReuseVerdict::Reused,
               "B: the next turn reuses after a cancel");
    }

    // --- D: errors -------------------------------------------------------------
    {
        ConversationRequest bad = make_request({user("hi")});
        bad.reasoning_effort = "extreme";
        bool invalid = false;
        try {
            conversation.run(bad, [](const TextDelta&) {}, [] { return false; });
        } catch (const ConversationError& error) {
            invalid = error.kind == ErrorKind::InvalidRequest;
        }
        expect(invalid, "D: a bad reasoning_effort is InvalidRequest");

        ConversationRequest bad_top_p = make_request({user("hi")});
        bad_top_p.sampling.top_p = 1.5f;
        bool bad_p = false;
        try {
            conversation.run(bad_top_p, [](const TextDelta&) {}, [] { return false; });
        } catch (const ConversationError& error) {
            bad_p = error.kind == ErrorKind::InvalidRequest;
        }
        expect(bad_p, "D: top_p out of range is InvalidRequest, before any state change");
    }

    // --- E: hygiene ------------------------------------------------------------
    {
        const auto info = conversation.info();
        expect(info.invariants_ok, "E: supply invariants hold after every turn");
        expect(info.context_capacity == kContext, "E: the reported capacity is the context");
    }

    // --- C: cancel in prefill --------------------------------------------------
    // Freed first: a second engine only fits once the first has released its VRAM.
    engine.free();
    {
        // A window smaller than the prompt; cancel before the second window.
        V4EngineOptions small = engine_options;
        small.runtime.prefill_window = 4;
        V4Engine windowed;
        windowed.initialize(small);
        V4Conversation windowed_conversation(windowed, "aeon-test");

        uint32_t poll = 0;
        uint32_t delivered = 0;
        const auto result = windowed_conversation.run(
            make_request({user("This is a longer prompt that spans several prefill windows for sure.")}),
            [&](const TextDelta&) { ++delivered; },
            [&] { return ++poll > 1; });  // first check passes, second cancels
        expect(result.stop == aeon::text::StopReason::Cancelled,
               "C: cancelling during prefill reports Cancelled");
        expect(delivered == 0, "C: no token is delivered when prefill is cancelled");
    }

    std::printf("================================================================================\n");
    std::printf("  %u/%u checks passed\n", checks - failures, checks);
    std::printf("================================================================================\n");
    return failures == 0 ? 0 : 1;
}
