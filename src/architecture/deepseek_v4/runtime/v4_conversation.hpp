#pragma once

// -----------------------------------------------------------------------------
// V4Conversation — the DeepSeek-V4 engine as the neutral conversation seam.
//
// This is the one place that knows both the engine (G4) and the serving layer's
// neutral request/response types (G1), which is exactly what the composition root
// exists to avoid having two definitions of. Everything model-specific — the chat
// template, the thinking split, the tool schema, the exact-token history — is
// decided here; the server above sees only decoded text and scalars.
//
// Exact-token history (prefix-reuse's Step 9): a live conversation must not route
// the model's own replies back through text, because re-encoding a decoded reply
// is not the generated ids and the token prefix would break at the reply boundary.
// So each turn's body ids are kept and re-attached to the matching assistant
// message on the next request; a history that cannot be matched is left unset and
// the turn replays (correct, just not reused).
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/runtime/v4_engine.hpp"
#include "architecture/deepseek_v4/text/dsv4_computation_key.hpp"
#include "architecture/deepseek_v4/text/dsv4_prompt_json.hpp"
#include "architecture/deepseek_v4/text/dsv4_stream_decoder.hpp"
#include "infrastructure/session/conversation.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace aeon::core {

class V4Conversation : public session::ConversationEngine {
public:
    V4Conversation(V4Engine& engine, std::string model_id)
        : engine_(engine), model_id_(std::move(model_id)) {}

    session::ConversationInfo info() const override {
        session::ConversationInfo info;
        info.model_id = model_id_;
        info.context_capacity = engine_.host().context_capacity();
        info.resident_tokens = engine_.resident_tokens();
        // The three invariants `aeon_chat::print_invariants` reports.
        info.invariants_ok = engine_.host().registry().invariants_hold() &&
                             engine_.host().outstanding_expert_leases() == 0 &&
                             engine_.host().staging_in_use_slots() == 0;
        return info;
    }

    void reset() noexcept override { engine_.end_session(); }

    session::ConversationResult run(const session::ConversationRequest& request,
                                    const session::DeltaSink& sink,
                                    const std::function<bool()>& cancelled) override {
        using namespace aeon::text;

        const uint32_t capacity = engine_.host().context_capacity();

        // --- sampling: the artifact's policy unless the request overrides ------
        SamplerConfig sampling =
            engine_.policy().to_sampler_config(request.sampling.seed.value_or(engine_.options().seed));
        if (request.sampling.temperature.has_value()) {
            const float temperature = *request.sampling.temperature;
            if (!std::isfinite(temperature) || temperature < 0.0f) {
                throw session::ConversationError(session::ErrorKind::InvalidRequest,
                                                 "temperature must be finite and >= 0");
            }
            sampling.temperature = temperature;
        }
        if (request.sampling.top_p.has_value()) {
            const float top_p = *request.sampling.top_p;
            if (!(top_p > 0.0f && top_p <= 1.0f)) {
                throw session::ConversationError(session::ErrorKind::InvalidRequest,
                                                 "top_p must be in (0, 1]");
            }
            sampling.top_p = top_p;
        }

        // --- prompt options -----------------------------------------------------
        Dsv4PromptOptions options;
        options.thinking_mode =
            request.thinking ? Dsv4ThinkingMode::Thinking : Dsv4ThinkingMode::Chat;
        // A reuse session keeps the previous turn's reasoning verbatim; dropping it
        // would rewrite the earlier body and conflict with the exact ids.
        options.drop_thinking = false;
        if (!request.reasoning_effort.empty()) {
            const std::string& effort = request.reasoning_effort;
            if (effort != "low" && effort != "high" && effort != "max") {
                throw session::ConversationError(session::ErrorKind::InvalidRequest,
                                                 "reasoning_effort must be one of low|high|max");
            }
            options.reasoning_effort = effort;
        }

        // --- map the neutral conversation --------------------------------------
        const std::vector<Dsv4PromptMessage> messages = map_messages(request);

        // --- encode -------------------------------------------------------------
        std::vector<uint32_t> prompt;
        try {
            prompt = engine_.encoder().encode_tokens(messages, options);
        } catch (const std::exception& error) {
            throw session::ConversationError(session::ErrorKind::InvalidRequest, error.what());
        }
        if (prompt.size() >= capacity) {
            throw session::ConversationError(
                session::ErrorKind::ContextOverflow,
                std::to_string(prompt.size()) + " tokens, context " + std::to_string(capacity));
        }

        // --- generation options -------------------------------------------------
        uint32_t max_new = capacity - static_cast<uint32_t>(prompt.size());
        if (request.max_new_tokens.has_value() && *request.max_new_tokens < max_new) {
            max_new = *request.max_new_tokens;
        }

        GenerationOptions generation;
        generation.max_new_tokens = max_new;
        generation.eos_token_id = engine_.tokenizer().eos_token_id();
        generation.context_limit = capacity;
        generation.thinking_mode = request.thinking;
        generation.cancelled = cancelled;

        Dsv4StreamDecoder decoder(engine_.tokenizer(), request.thinking);
        std::vector<session::TextDelta> deltas;
        std::string reasoning_text;
        std::string content_text;
        const auto emit = [&](const session::TextDelta& delta) {
            if (delta.channel == session::Channel::Reasoning) {
                reasoning_text += delta.text;
            } else {
                content_text += delta.text;
            }
            sink(delta);
        };
        generation.on_token = [&](uint32_t token_id) {
            deltas.clear();
            decoder.push(token_id, deltas);
            for (const session::TextDelta& delta : deltas) emit(delta);
        };

        // --- generate -----------------------------------------------------------
        const std::string key = dsv4_computation_key(messages, options);
        V4Reply reply;
        try {
            reply = engine_.generate(prompt, generation, sampling, /*reuse_prefix=*/true, key);
        } catch (const session::ConversationError&) {
            throw;
        } catch (const std::exception& error) {
            engine_.end_session();
            throw session::ConversationError(session::ErrorKind::Internal, error.what());
        }

        deltas.clear();
        decoder.finish(deltas);
        for (const session::TextDelta& delta : deltas) emit(delta);

        // --- exact-token history ------------------------------------------------
        std::vector<uint32_t> body = reply.token_ids;
        if (!body.empty() && body.back() == engine_.tokenizer().eos_token_id()) {
            body.pop_back();
        }
        if (!body.empty()) {
            assistant_ids_[assistant_signature(reasoning_text, content_text)] = std::move(body);
        }

        // --- result -------------------------------------------------------------
        session::ConversationResult result;
        result.stop = reply.stop_reason;
        result.prompt_tokens = reply.prompt_tokens;
        result.completion_tokens = static_cast<uint32_t>(reply.token_ids.size());
        result.reused_tokens = reply.reused_tokens;
        result.prefilled_tokens = reply.prefilled_tokens;
        result.verdict = reply.reuse_verdict;
        result.ttft_ms = reply.ttft_ms;
        result.decode_tokens_per_second = reply.decode_tokens_per_second;
        return result;
    }

private:
    static std::string assistant_signature(const std::string& reasoning,
                                           const std::string& content) {
        return reasoning + '\x1f' + content;
    }

    std::vector<aeon::text::Dsv4PromptMessage> map_messages(
        const session::ConversationRequest& request) const {
        using namespace aeon::text;
        std::vector<Dsv4PromptMessage> messages;
        messages.reserve(request.messages.size() + 1);

        // Tools and response_format ride a new empty leading system message, the
        // shape vLLM's DSV4 `apply_chat_template` produces — even when the client
        // also sent a system message.
        const bool has_tools = !request.tools_json.empty();
        const bool has_response_format = !request.response_format_json.empty();
        if (has_tools || has_response_format) {
            Dsv4PromptMessage system;
            system.role = Dsv4Role::System;
            if (has_tools) {
                PromptJson tools;
                try {
                    tools = PromptJson::parse(request.tools_json);
                } catch (const std::exception& error) {
                    throw session::ConversationError(session::ErrorKind::InvalidRequest,
                                                     std::string("invalid tools: ") + error.what());
                }
                if (tools.is_array()) {
                    for (const PromptJson& tool : tools.array) {
                        const PromptJson* function = tool.find("function");
                        if (function == nullptr) continue;
                        Dsv4ToolDefinition definition;
                        definition.function_json = function->to_python_json();
                        system.tools.push_back(std::move(definition));
                    }
                } else {
                    throw session::ConversationError(session::ErrorKind::InvalidRequest,
                                                     "`tools` must be a JSON array");
                }
            }
            system.response_format_json = request.response_format_json;
            messages.push_back(std::move(system));
        }

        for (const session::Message& message : request.messages) {
            Dsv4PromptMessage converted;
            converted.role = to_dsv4_role(message.role);
            converted.content = message.content;
            converted.reasoning_content = message.reasoning_content;
            converted.tool_call_id = message.tool_call_id;
            for (const session::ToolCall& call : message.tool_calls) {
                Dsv4ToolCall tool_call;
                tool_call.id = call.id;
                tool_call.name = call.name;
                tool_call.arguments = call.arguments_json;
                converted.tool_calls.push_back(std::move(tool_call));
            }
            if (message.role == session::Role::Assistant) {
                const auto it = assistant_ids_.find(
                    assistant_signature(message.reasoning_content, message.content));
                if (it != assistant_ids_.end()) converted.preencoded_ids = it->second;
            }
            messages.push_back(std::move(converted));
        }
        return messages;
    }

    static aeon::text::Dsv4Role to_dsv4_role(session::Role role) {
        using aeon::text::Dsv4Role;
        switch (role) {
            case session::Role::System: return Dsv4Role::System;
            case session::Role::Developer: return Dsv4Role::Developer;
            case session::Role::User: return Dsv4Role::User;
            case session::Role::Assistant: return Dsv4Role::Assistant;
            case session::Role::Tool: return Dsv4Role::Tool;
        }
        return Dsv4Role::User;
    }

    V4Engine& engine_;
    std::string model_id_;

    // The body ids of every assistant turn this session produced, keyed by the
    // turn's (reasoning, content) signature so a re-sent history can be matched.
    std::map<std::string, std::vector<uint32_t>> assistant_ids_;
};

}  // namespace aeon::core
