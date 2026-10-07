#pragma once

// -----------------------------------------------------------------------------
// The engine — where the graph meets text.
//
// A binding, not a pipeline: it renders the conversation with the prompt encoder
// and drives `text::generate_token_ids`, whose steps are exactly
//
//     V4Graph::forward_window(prompt, 0, N, C, compute)    // prefill -> logits
//     V4Graph::forward_token(token_id, position, compute)  // decode  -> logits
//     Sampler::select(logits, compute)                     // -> token
//
// Prefill and decode are two steps rather than one flag because they are two
// *allocation strategies*: prefill streams a whole layer's expert set inside a
// bounded window, while decode asks for a few experts per layer and keeps them by
// recency. A per-token step with a `prefill = true` flag could only express the
// first.
//
// The generation loop is still `text::generate_token_ids`: it owns the EOS stop,
// the `max_new_tokens` cap and the context limit, and the windowed prefill is a
// *step* it is handed rather than a second loop. Sampling policy is read from
// `generation_config.json` rather than assumed, so an artifact that ships
// `do_sample = true` does not silently become argmax. Thinking-mode stripping is a
// pure string operation and stays out of the model.
//
// Position is the engine's and is not derived: `forward_token` takes an absolute
// position, and the engine bounds generation by `host().context_capacity()` rather
// than computing one by wrapping or clamping.
//
// The engine also owns a **session**: the resident layer state produced by the
// preceding turns plus a `session::PrefixRecord` describing it. `generate` /
// `chat` with `reuse_prefix = true` verify the new prompt extends that record and
// prefill only the addition; with it false (the default) each call is a cold run
// and the record is rebuilt. A caller that reaches past the engine to
// `graph().forward_*` or `host().reset_generation_state()` is outside this
// contract — it must call `end_session()` before a reuse, and a reset it performs
// is caught by the host state epoch.
// -----------------------------------------------------------------------------

#include "infrastructure/memory/memory_budget.hpp"
#include "architecture/deepseek_v4/runtime/v4_graph.hpp"
#include "architecture/deepseek_v4/runtime/v4_model_host.hpp"
#include "infrastructure/sampler.hpp"
#include "infrastructure/session/prefix_record.hpp"
#include "architecture/deepseek_v4/text/dsv4_computation_key.hpp"
#include "architecture/deepseek_v4/text/dsv4_prompt_encoder.hpp"
#include "architecture/deepseek_v4/text/dsv4_tokenizer.hpp"
#include "infrastructure/json.hpp"
#include "infrastructure/expert/transport/prefetch_staging.hpp"
#include "infrastructure/profiling/phase_profiler.hpp"
#include "infrastructure/text/text_generation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace aeon::core {

// -----------------------------------------------------------------------------
// The artifact's sampling policy
// -----------------------------------------------------------------------------

// `generation_config.json`, as the artifact declares it. Read rather than
// hardcoded because it is a property of *this checkpoint*.
struct V4GenerationPolicy {
    bool do_sample{true};
    float temperature{1.0f};
    float top_p{1.0f};
    uint32_t eos_token_id{1};
    uint32_t bos_token_id{0};

    static V4GenerationPolicy load_from_json(const std::string& path) {
        const JsonValue document = JsonValue::parse_file(path);
        if (!document.is_object()) {
            throw std::runtime_error(
                "V4GenerationPolicy: " + path + " is not a JSON object");
        }
        V4GenerationPolicy policy;
        if (const JsonValue* value = document.find("do_sample")) {
            if (value->is_bool()) policy.do_sample = value->as_bool();
        }
        if (const JsonValue* value = document.find("temperature")) {
            if (value->is_number()) policy.temperature = static_cast<float>(value->as_number());
        }
        if (const JsonValue* value = document.find("top_p")) {
            if (value->is_number()) policy.top_p = static_cast<float>(value->as_number());
        }
        if (const JsonValue* value = document.find("eos_token_id")) {
            if (value->is_number()) {
                policy.eos_token_id = static_cast<uint32_t>(value->as_int64());
            }
        }
        if (const JsonValue* value = document.find("bos_token_id")) {
            if (value->is_number()) {
                policy.bos_token_id = static_cast<uint32_t>(value->as_int64());
            }
        }
        return policy;
    }

    // The policy as the sampler consumes it. `do_sample = false` is the greedy
    // path — `temperature <= 0` is the sampler's own `T -> 0` limit, not a second
    // mode — so the two conventions meet here rather than in the sampler.
    SamplerConfig to_sampler_config(uint64_t seed) const {
        SamplerConfig config;
        config.temperature = do_sample ? temperature : 0.0f;
        config.top_p = top_p;
        config.top_k = 0;  // the artifact declares no top_k; the nucleus is its truncation
        config.seed = seed;
        return config;
    }
};

// -----------------------------------------------------------------------------
// What one generation produced
// -----------------------------------------------------------------------------

struct V4Reply {
    // The decoded reply, EOS stripped. Thinking content is *not* stripped here —
    // `strip_thinking` is the caller's decision.
    std::string text;

    // The generated ids exactly as the loop produced them, including a trailing
    // EOS when it stopped on one. A gate comparing only the text could not tell an
    // EOS stop from a length stop.
    std::vector<uint32_t> token_ids;

    text::StopReason stop_reason{text::StopReason::Error};

    // The number of tokens the rendered conversation produced.
    uint32_t prompt_tokens{0};

    // Prefix-reuse accounting: how many leading prompt tokens the resident state
    // already covered, how many were prefilled this turn, and the verdict that
    // decided it. `reused_tokens + prefilled_tokens == prompt_tokens`. On a
    // stateless call these are 0 / prompt_tokens / `Cold`.
    uint32_t reused_tokens{0};
    uint32_t prefilled_tokens{0};
    session::ReuseVerdict reuse_verdict{session::ReuseVerdict::Cold};

    // Time to first token, and the mean per-token time over the decode steps.
    double ttft_ms{0.0};
    double decode_tokens_per_second{0.0};
};

struct V4EngineOptions {
    std::string model_dir;
    std::string tokenizer_path;  // empty -> `<model_dir>/tokenizer.aeon`
    AeonRuntimeConfig runtime{};

    // The seed the sampler is reset to before every `chat`, so a generation is a
    // pure function of (conversation, options, seed). Reset per call rather than
    // per engine so turn 2 does not depend on what turn 1 drew.
    uint64_t seed{0};

    // When non-empty, every position's raw fp16 logits are appended to this file,
    // in order, as `[vocab_size]` little-endian halves with no header. A
    // tier-invariance gate compares two runs byte-for-byte (`cmp`): which tier
    // answered must not change the number. One row per prefill window's last
    // position plus one per generated token. Inert when empty.
    std::string dump_logits_path;

    bool verbose{false};
};

// -----------------------------------------------------------------------------
// The engine
// -----------------------------------------------------------------------------

class V4Engine {
public:
    V4Engine() = default;

    ~V4Engine() {
        free();
    }

    V4Engine(const V4Engine&) = delete;
    V4Engine& operator=(const V4Engine&) = delete;
    V4Engine(V4Engine&&) = delete;
    V4Engine& operator=(V4Engine&&) = delete;

    // The whole stack, in the order it has to be built: the host (which owns the
    // device, the weights and the expert tiers), the graph over it, the tokenizer,
    // the encoder over the tokenizer, and the sampler sized from the loaded
    // vocabulary. The order cannot be permuted, so it is one function.
    void initialize(const V4EngineOptions& options) {
        free();
        options_ = options;
        if (options_.tokenizer_path.empty()) {
            options_.tokenizer_path = options_.model_dir + "/tokenizer.aeon";
        }

        host_.initialize(options_.model_dir, options_.runtime, options_.verbose);
        graph_ = std::make_unique<V4Graph>(host_);

        if (!options_.dump_logits_path.empty()) {
            logits_dump_.open(options_.dump_logits_path,
                              std::ios::out | std::ios::binary | std::ios::trunc);
            if (!logits_dump_) {
                throw std::runtime_error(
                    "V4Engine: failed to open logits dump: " + options_.dump_logits_path);
            }
        }

        tokenizer_.load(options_.tokenizer_path);
        encoder_ = std::make_unique<text::Dsv4PromptEncoder>(tokenizer_);

        const uint32_t vocab = static_cast<uint32_t>(host_.config().vocab_size);
        if (tokenizer_.vocab_size() != vocab) {
            throw std::runtime_error(
                "V4Engine: the tokenizer's vocabulary (" +
                std::to_string(tokenizer_.vocab_size()) +
                ") is not the model's (" + std::to_string(vocab) + ")");
        }
        // The sampler's device workspace (the argmax reduction buffers) is allocated
        // on the device that is current here, and the logits it reads live on the head
        // stage's device. Build it there so the reduction runs on the stream that wrote
        // them.
        {
            DeviceScope sampler_scope(host_.head_device());
            sampler_ = std::make_unique<Sampler>(vocab);
        }

        // The artifact's own policy, if it ships one. Absent is not an error — a
        // checkpoint without `generation_config.json` gets the deterministic
        // default, and says so through `policy_loaded_`.
        const std::string policy_path = options_.model_dir + "/generation_config.json";
        try {
            policy_ = V4GenerationPolicy::load_from_json(policy_path);
            policy_loaded_ = true;
        } catch (const std::exception&) {
            policy_ = V4GenerationPolicy{};
            policy_loaded_ = false;
        }
    }

    void free() noexcept {
        session_.clear();
        if (logits_dump_.is_open()) logits_dump_.close();
        sampler_.reset();
        encoder_.reset();
        graph_.reset();
        host_.free();
    }

    // Forget the resident-state description without touching the device state. The
    // next `generate` without reuse resets the state and rebuilds the record; the
    // next with reuse replans against an empty record. For a caller that drives the
    // graph directly and must not leave a stale record behind.
    void end_session() noexcept { session_.clear(); }

    // How many tokens the resident state already covers. Zero when there is no live
    // session. Read by a serving layer's `/status` without touching the device.
    uint32_t resident_tokens() const noexcept { return session_.size(); }

    bool ready() const noexcept { return graph_ != nullptr && sampler_ != nullptr; }
    bool policy_loaded() const noexcept { return policy_loaded_; }
    const V4GenerationPolicy& policy() const noexcept { return policy_; }
    const V4EngineOptions& options() const noexcept { return options_; }

    V4ModelHost& host() noexcept { return host_; }
    const V4ModelHost& host() const noexcept { return host_; }
    V4Graph& graph() { return *graph_; }
    const V4Graph& graph() const { return *graph_; }
    Sampler& sampler() { return *sampler_; }
    const Sampler& sampler() const { return *sampler_; }
    const text::Dsv4Tokenizer& tokenizer() const noexcept { return tokenizer_; }
    const text::Dsv4PromptEncoder& encoder() const noexcept { return *encoder_; }

    // --- the binding ---------------------------------------------------------

    // One token in, one token out, at an absolute position. Feeds the session
    // record at the same site it feeds the graph, so the record cannot describe a
    // token the state never saw.
    uint32_t advance(uint32_t token_id, uint32_t position) {
        const half* logits = graph_->forward_token(token_id, position, host_.streams().compute);
        session_.feed(position, token_id);
        if (logits_dump_.is_open()) dump_logits(logits);
        DeviceScope head_scope(host_.head_device());
        return sampler_->select(logits, host_.head_stream());
    }

    // Appends one position's raw fp16 logits to the dump, if one is open. Ordered
    // on the compute stream that just wrote them.
    void dump_logits(const half* logits) {
        const uint32_t vocab = static_cast<uint32_t>(host_.config().vocab_size);
        logits_host_.resize(vocab);
        DeviceScope head_scope(host_.head_device());
        (void)hipMemcpyAsync(logits_host_.data(), logits,
                             static_cast<size_t>(vocab) * sizeof(half),
                             hipMemcpyDeviceToHost, host_.head_stream());
        (void)hipStreamSynchronize(host_.head_stream());
        logits_dump_.write(reinterpret_cast<const char*>(logits_host_.data()),
                           static_cast<std::streamsize>(vocab) * sizeof(half));
    }

    // --- the prefill window --------------------------------------------------

    // The body chunk `C` — the rows in flight per body invocation. It is the
    // **configured** value (`AeonRuntimeConfig::prefill_chunk`), not a derived one,
    // because the workspace was allocated at load from that same knob. The only
    // adjustment is downward, for a prompt shorter than one chunk.
    uint32_t prefill_chunk_for(uint32_t tokens) const noexcept {
        const uint32_t configured = host_.prefill_chunk_tokens();
        return std::max<uint32_t>(1, std::min(configured, tokens));
    }

    // The window `W` for a prompt of `tokens`. The load-time allocation is for
    // `host_.prefill_window_tokens()`, so a longer prompt runs `⌈N/W⌉` windows and
    // a shorter one is a single window.
    uint32_t prefill_window_for(uint32_t tokens) const noexcept {
        return std::max<uint32_t>(1, std::min(host_.prefill_window_tokens(), tokens));
    }

    // --- the run -------------------------------------------------------------

    // The token-level entry: a rendered prompt in, a reply out. Everything above
    // it — the encoder, the message shape — is the caller's. This is the neutral
    // conversation seam a serving layer drives once it owns a session, and the
    // entry the prefix-reuse gate needs (a prompt built from ids).
    //
    // `generation` supplies the token cap and the stop policy; the EOS id comes
    // from the tokenizer. `sampling` is the artifact's policy unless overridden.
    V4Reply generate(const std::vector<uint32_t>& prompt,
                     const text::GenerationOptions& generation,
                     const SamplerConfig& sampling,
                     bool reuse_prefix = false,
                     const std::string& computation_key = {}) {
        if (!ready()) {
            throw std::logic_error("V4Engine::generate: the engine was not initialized");
        }
        if (prompt.empty()) {
            throw std::runtime_error("V4Engine::generate: the prompt has no tokens");
        }

        const uint32_t capacity = host_.context_capacity();
        if (prompt.size() >= capacity) {
            throw std::runtime_error(
                "V4Engine::generate: the prompt is " +
                std::to_string(prompt.size()) + " tokens, which leaves no room to generate "
                "inside a context of " + std::to_string(capacity));
        }

        // Decide whether the resident state already covers a prefix of this prompt.
        // When reuse is not requested this is a cold run and the record is rebuilt,
        // so the stateless behaviour is exactly preserved. The capacity refusal above
        // runs first, so a refused turn leaves the session untouched.
        session::ReuseDecision decision;
        if (reuse_prefix) {
            decision = session_.plan(prompt, computation_key);
            if (decision.verdict == session::ReuseVerdict::Reused &&
                session_epoch_ != host_.state_epoch()) {
                // The device state was reset out from under the record (a direct
                // `host().reset_generation_state()`), so the record describes a state
                // the engine no longer holds.
                decision.verdict = session::ReuseVerdict::StaleState;
                decision.start = 0;
            }
        }

        const uint32_t start = decision.start;
        if (start == 0) {
            host_.reset_generation_state();
            // The record is rebuilt under this run's key, so the next turn compares
            // against the state actually held — including after a rejected reuse.
            session_.begin(computation_key);
            session_epoch_ = host_.state_epoch();
        }

        // `set_config` validates and seeds the generator, so installing the config
        // is also the reseed. Unconditional: a reused turn is still reproducible
        // from the seed regardless of what was kept.
        sampler_->set_config(sampling);

        text::GenerationOptions loop_options = generation;
        loop_options.eos_token_id = tokenizer_.eos_token_id();
        // The context limit is the host's capacity, not the caller's number: a
        // position at or beyond it is refused by the layer, and a bound stated up
        // front is a better failure than a refusal mid-decode.
        loop_options.context_limit = capacity;

        V4Reply reply;
        reply.prompt_tokens = static_cast<uint32_t>(prompt.size());
        reply.reused_tokens = start;
        reply.prefilled_tokens = static_cast<uint32_t>(prompt.size()) - start;
        reply.reuse_verdict = decision.verdict;

        // Prefill and decode are two strategies, so they are two steps and not one
        // with a flag:
        //
        //   prefill  the whole prompt through `V4Graph::forward_window` — one
        //            layer-major pass (or `⌈N/W⌉` of them), where the swept supply,
        //            the chunk-wide deduplicated dispatch and the Hot drain live;
        //   decode   one token at a time through `advance` → `forward_token`, on
        //            the on-demand + LRU + demotion strategy decode owns.
        //
        // The loop is still `text::generate_token_ids`; this only adds the prefill
        // step it is handed.
        //
        // The tail is what this turn actually prefills; on a cold run it is the whole
        // prompt. Sizing the window and chunk from the tail (not the prompt) keeps a
        // small addition on the cheap path.
        const uint32_t tail = static_cast<uint32_t>(prompt.size()) - start;
        double first_token_ms = 0.0;
        Clock::time_point decode_started{};
        const auto started = Clock::now();

        const uint32_t window = prefill_window_for(tail);
        const uint32_t chunk = prefill_chunk_for(tail);

        const text::PromptPrefill prefill_step =
            [&](const std::vector<uint32_t>& tokens) -> uint32_t {
                // The phase is set before the forward pass, because the expert
                // dispatch happens inside it and `TieredExpertSupply` records
                // against whichever phase is current. A no-op when telemetry is off.
                host_.set_supply_phase(true);
                const uint32_t count = static_cast<uint32_t>(tokens.size());
                const half* logits = nullptr;
                PhaseProfiler& phases = PhaseProfiler::instance();
                const bool profiling = phases.enabled();
                if (profiling) phases.reset();
                // Only the tail from `start` is prefilled; the resident state already
                // holds the first `start` tokens. On a cold run `start` is 0 and this is
                // the whole prompt.
                try {
                    for (uint32_t offset = start; offset < count; offset += window) {
                        // A cancel between windows is honoured here; a cancel inside
                        // one window cannot be, so the worst case is one window.
                        if (generation.cancelled && generation.cancelled()) {
                            throw text::GenerationCancelled{};
                        }
                        const uint32_t span = std::min(window, count - offset);
                        logits = graph_->forward_window(
                            tokens.data() + offset, offset, span, std::min(chunk, span),
                            host_.streams().compute);
                        // The window succeeded, so its tokens are now in the state.
                        for (uint32_t k = 0; k < span; ++k) {
                            session_.feed(offset + k, tokens[offset + k]);
                        }
                    }
                } catch (const text::GenerationCancelled&) {
                    // The windows that ran are fed and recorded, so state and record
                    // still agree; a later turn may reuse them. Do not clear.
                    throw;
                } catch (...) {
                    // The state is partially advanced and no longer matches the record,
                    // so the record must not be trusted by a later turn.
                    session_.clear();
                    throw;
                }
                // The phase breakdown is a per-prompt aggregate, so it is read and
                // printed once here rather than per window. `resolve` synchronizes
                // only the profiler's own events; the prefill already has a stream
                // boundary inside `forward_window`.
                if (profiling) {
                    phases.resolve();
                    phases.report(stdout);
                    // Clear the prefill samples so the decode's phase split is its own
                    // table; the two use distinct region names, so this is only about
                    // not resolving the prefill's events a second time.
                    phases.reset();
                }
                if (logits_dump_.is_open()) dump_logits(logits);
                // TTFT is the moment the *last* prompt token's forward produced a
                // token. The decode clock starts there for the same reason: the
                // first generated token came out of the prefill, so charging it to
                // the decode rate would report a rate the decode never ran.
                first_token_ms = elapsed_ms(started);
                decode_started = Clock::now();
                DeviceScope head_scope(host_.head_device());
                return sampler_->select(logits, host_.head_stream());
            };

        const text::TokenStep decode_step =
            [&](uint32_t token_id, uint32_t position, bool /*prefill*/) -> uint32_t {
                host_.set_supply_phase(false);
                const uint32_t next = advance(token_id, position);
                host_.record_supply_decode_token();
                return next;
            };

        text::GenerationResult generated;
        try {
            generated = text::generate_token_ids(prompt, loop_options, prefill_step, decode_step);
        } catch (const text::GenerationCancelled&) {
            // Prefill was cancelled between windows: nothing was sampled and the
            // state matches the record, so the reply carries no tokens and a later
            // turn may still reuse the state.
            reply.stop_reason = text::StopReason::Cancelled;
            return reply;
        }
        // The decode phase split: one token's regions accumulate over every decoded
        // token, so the figures are per-run totals and divide by the token count. The
        // prefill reset above makes this table decode-only.
        if (PhaseProfiler::instance().enabled()) {
            PhaseProfiler::instance().resolve();
            PhaseProfiler::instance().report(stdout);
        }

        reply.token_ids = generated.token_ids;
        reply.stop_reason = generated.stop_reason;
        reply.ttft_ms = first_token_ms;

        // Measured over the decode steps alone, so the rate is a property of the
        // decode and not of the prefill it follows.
        const uint32_t generated_tokens = static_cast<uint32_t>(generated.token_ids.size());
        if (generated_tokens > 1 && first_token_ms > 0.0) {
            const double decode_seconds = seconds_since(decode_started);
            reply.decode_tokens_per_second = decode_seconds > 0.0
                ? static_cast<double>(generated_tokens - 1) / decode_seconds
                : 0.0;
        }

        std::vector<uint32_t> visible = generated.token_ids;
        if (!visible.empty() && visible.back() == tokenizer_.eos_token_id()) {
            visible.pop_back();
        }
        reply.text = tokenizer_.decode(visible);
        return reply;
    }

    // Render the conversation, then generate. The render is the only thing `chat`
    // adds to `generate`, so a token-level caller and a message-level caller take
    // the exact same path from the prompt onward.
    V4Reply chat(const std::vector<text::Dsv4PromptMessage>& messages,
                 const text::Dsv4PromptOptions& prompt_options,
                 const text::GenerationOptions& generation,
                 const SamplerConfig& sampling,
                 bool reuse_prefix = false) {
        if (!ready()) {
            throw std::logic_error("V4Engine::chat: the engine was not initialized");
        }
        if (messages.empty()) {
            throw std::invalid_argument("V4Engine::chat: the conversation is empty");
        }
        const std::vector<uint32_t> prompt = encoder_->encode_tokens(messages, prompt_options);
        return generate(prompt, generation, sampling, reuse_prefix,
                        dsv4_computation_key(messages, prompt_options));
    }

    // The common case: one user turn, the artifact's own sampling policy.
    V4Reply chat(const std::string& user_text,
                 const text::Dsv4PromptOptions& prompt_options,
                 const text::GenerationOptions& generation) {
        text::Dsv4PromptMessage message;
        message.role = text::Dsv4Role::User;
        message.content = user_text;
        return chat({message}, prompt_options, generation,
                    policy_.to_sampler_config(options_.seed));
    }

    // --- text, and nothing else ----------------------------------------------

    // Everything after the thinking-end marker, or the whole string when it never
    // appears. A pure function of the string, so it is testable with no model.
    static std::string strip_thinking(const std::string& text,
                                      const text::Dsv4Tokenizer& tokenizer) {
        const std::string marker = tokenizer.decode({tokenizer.thinking_end_token_id()});
        if (marker.empty()) return text;
        const size_t position = text.rfind(marker);
        return position == std::string::npos ? text : text.substr(position + marker.size());
    }

private:
    using Clock = std::chrono::steady_clock;

    static double elapsed_ms(const Clock::time_point& start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    static double seconds_since(const Clock::time_point& start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }

    V4EngineOptions options_{};
    V4GenerationPolicy policy_{};
    bool policy_loaded_{false};

    // Construction order is load order: the graph holds a reference to the host,
    // so the host must outlive it, and the encoder holds a reference to the
    // tokenizer for the same reason.
    V4ModelHost host_{};
    std::unique_ptr<V4Graph> graph_{};
    std::ofstream logits_dump_{};
    std::vector<half> logits_host_{};
    text::Dsv4Tokenizer tokenizer_{};
    std::unique_ptr<text::Dsv4PromptEncoder> encoder_{};
    std::unique_ptr<Sampler> sampler_{};

    // The resident-state description and the host state epoch it was built against.
    // `session_epoch_` is the value of `host_.state_epoch()` at the last cold start,
    // so a direct `reset_generation_state()` between turns is detected as stale.
    session::PrefixRecord session_{};
    uint64_t session_epoch_{0};
};

}  // namespace aeon::core
