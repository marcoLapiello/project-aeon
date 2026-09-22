#pragma once

// -----------------------------------------------------------------------------
// The engine — where the graph meets text.
//
// This is the composition plan's **G5** and its phase **P4**, and it is the last
// thing between the rewrite and the plan's **acceptance criterion**: one command
// takes a conversation and returns text (§1). Everything below it already runs —
// the host (`V4ModelHost`), the graph (`V4Graph`), the sampler (`V4Sampler`) —
// and everything it binds to text already runs — the tokenizer, the canonical
// prompt encoder, and the generation loop. What did not exist was the **binding**,
// and that is the whole of this file.
//
// **It is a binding and not a pipeline.** There is no forward pass here, no
// sampling rule, no template and no tokenizer: the engine renders with the
// encoder, drives `generate_token_ids`, and the steps that loop calls are exactly
//
//     V4Graph::forward_window(prompt, 0, N, C, compute)     // prefill -> logits
//     V4Graph::forward_token(token_id, position, compute)   // decode  -> logits
//     V4Sampler::select(logits, compute)                    // -> token
//
// One line each, because every other decision was made by the component that owns
// it. The gate asserts this identity rather than assuming it (§A of the gate), which
// is what keeps "the engine is the binding" from being a claim about the source
// rather than about the behaviour.
//
// **Why prefill and decode are two steps and not one flag.** The expert supply's
// plan (Step 6, §3) established that prefill and decode are two *allocation
// strategies*: prefill visits each layer once inside a bounded window and streams a
// whole layer's set in computation order, while decode asks for six experts per
// layer and keeps them by recency. A single per-token step with a `prefill = true`
// flag could only ever express the first strategy, which is why the prompt is now
// handed to `forward_window` as one unit and the flag is gone.
//
// **Why the generation loop is still `text::generate_token_ids`.** It already owns
// the EOS stop, the `max_new_tokens` cap and the context limit, and
// `test_text_generation` certifies all four; it takes the prefill as a
// `PromptPrefill` step, so the windowed prefill is a *step* it is handed rather
// than a second loop. Writing a second loop here would be the duplication the plan
// forbids — and worse, it would be a decode loop *in the engine*, which is how a
// graph grows a second body.
//
// **The artifact's sampling policy is read, not assumed.** `generation_config.json`
// ships `do_sample = true, temperature = 1.0, top_p = 1.0` (plan Step 5), so the
// faithful default is **not** argmax: `V4GenerationPolicy::load_from_json` reads
// the file and `chat` passes it to the sampler. P3's shipped default is the
// deterministic one and stays that way — the policy belongs to the artifact and is
// the engine's to read, which is the split P3's gate recorded.
//
// **Thinking mode is a rendering concern, and it stays one.** The encoder decides
// what the model *sees*; what the model *returns* may still carry a thinking
// block, and stripping it is a text operation on the decoded string (the reference
// CLI's own behaviour). `strip_thinking` is exposed here as a pure string
// function so it can be tested without a model and so the engine never has to
// decide what "the user-visible reply" means.
//
// **Position is the engine's, and it is not derived.** `forward_token` takes an
// absolute position and `V4Layer::record_position` refuses one at or beyond the
// context (trap 40 — refused, never clamped). The engine therefore bounds its own
// generation by `host().context_capacity()` and never computes a position by
// wrapping or by `min`: the position it hands the graph is the token's index in
// the conversation, which is what the model was trained on.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_graph.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "architecture/deepseek_v4/core/v4_sampler.hpp"
#include "architecture/deepseek_v4/text/dsv4_prompt_encoder.hpp"
#include "architecture/deepseek_v4/text/dsv4_tokenizer.hpp"
#include "infrastructure/core/json.hpp"
#include "infrastructure/core/prefetch_staging.hpp"
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
// hardcoded because it is a property of *this checkpoint*, and because the plan
// records the values as evidence (`do_sample = true, temperature = 1.0,
// top_p = 1.0`) that a reader can check against the file.
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
    // mode — so the two conventions meet here, in one place, rather than in the
    // sampler learning about `do_sample`.
    V4SamplerConfig to_sampler_config(uint64_t seed) const {
        V4SamplerConfig config;
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

    // The generated ids exactly as the loop produced them, **including** a
    // trailing EOS when it stopped on one. The raw sequence is kept because it is
    // what a later turn would have to re-feed, and because a gate that compared
    // only the text could not tell an EOS stop from a length stop.
    std::vector<uint32_t> token_ids;

    text::StopReason stop_reason{text::StopReason::Error};

    // The whole rendered conversation, for diagnostics and for a gate that wants
    // to see what the model was actually asked.
    uint32_t prompt_tokens{0};

    // Time from the call to the first generated token, and the mean per-token time
    // over the decode steps. Measured, not estimated: the loop's own step callback
    // is where both boundaries are observed.
    double ttft_ms{0.0};
    double decode_tokens_per_second{0.0};
};

struct V4EngineOptions {
    std::string model_dir;
    std::string tokenizer_path;  // empty -> `<model_dir>/tokenizer.aeon`
    AeonRuntimeConfig runtime{};

    // The seed the sampler is reset to before every `chat`, so a generation is a
    // pure function of (conversation, options, seed). Reset per call rather than
    // per engine so turn 2 does not depend on what turn 1 happened to draw.
    uint64_t seed{0};

    // When non-empty, every position's raw fp16 logits are appended to this file,
    // in order, as `[vocab_size]` little-endian halves with no header. It exists so
    // a tier-invariance gate can compare two runs byte-for-byte (`cmp`): the claim
    // is that *which tier answered did not change the number*, and a byte compare
    // is the only assertion that states it without a tolerance. It perturbs nothing
    // when empty.
    //
    // Since the prompt became one layer-major window (Step 6 item 7) it holds one
    // row per **window** — that window's last position, which is the only head the
    // window computes — plus one row per generated token, rather than one row per
    // prompt token. Two runs of the same configuration still write the same
    // positions in the same order, which is what the byte compare needs.
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
    // vocabulary. Nothing here has an order that can be permuted, which is why it
    // is one function rather than five the caller must get right.
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
        sampler_ = std::make_unique<V4Sampler>(vocab);

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
        if (logits_dump_.is_open()) logits_dump_.close();
        sampler_.reset();
        encoder_.reset();
        graph_.reset();
        host_.free();
    }

    bool ready() const noexcept { return graph_ != nullptr && sampler_ != nullptr; }
    bool policy_loaded() const noexcept { return policy_loaded_; }
    const V4GenerationPolicy& policy() const noexcept { return policy_; }
    const V4EngineOptions& options() const noexcept { return options_; }

    V4ModelHost& host() noexcept { return host_; }
    const V4ModelHost& host() const noexcept { return host_; }
    V4Graph& graph() { return *graph_; }
    const V4Graph& graph() const { return *graph_; }
    V4Sampler& sampler() { return *sampler_; }
    const V4Sampler& sampler() const { return *sampler_; }
    const text::Dsv4Tokenizer& tokenizer() const noexcept { return tokenizer_; }
    const text::Dsv4PromptEncoder& encoder() const noexcept { return *encoder_; }

    // --- the binding ---------------------------------------------------------

    // One token in, one token out, at an absolute position. This is the whole of
    // what the engine adds to the decode loop, and it is public because the gate's
    // first statement is that `chat` is exactly this composition.
    uint32_t advance(uint32_t token_id, uint32_t position) {
        const half* logits = graph_->forward_token(token_id, position, host_.streams().compute);
        if (logits_dump_.is_open()) dump_logits(logits);
        return sampler_->select(logits, host_.streams().compute);
    }

    // Appends one position's raw fp16 logits to the dump, if one is open. The
    // compute stream just wrote them, so the copy is ordered after them on that
    // stream; the synchronize is the readback boundary the sampler relies on
    // anyway, so the dump adds no ordering of its own.
    void dump_logits(const half* logits) {
        const uint32_t vocab = static_cast<uint32_t>(host_.config().vocab_size);
        logits_host_.resize(vocab);
        (void)hipMemcpyAsync(logits_host_.data(), logits,
                             static_cast<size_t>(vocab) * sizeof(half),
                             hipMemcpyDeviceToHost, host_.streams().compute);
        (void)hipStreamSynchronize(host_.streams().compute);
        logits_dump_.write(reinterpret_cast<const char*>(logits_host_.data()),
                           static_cast<std::streamsize>(vocab) * sizeof(half));
    }

    // --- the prefill window (Step 6 D-a) -------------------------------------

    // The body chunk `C` the window runs with — the rows in flight per body
    // invocation (Step 6 §6b). It is a memory decision and it is **derived**, not
    // assumed: a chunk issues its `6C` routed-expert requests as one deduplicated
    // set, and that set must be staged (the arena) and, whenever the sweep is not
    // running, resident at once (the Hot pool). The body's own row cap bounds it
    // from above. A configuration that leaves only a few slots therefore degrades
    // the chunk instead of failing at the arena or starving on leases.
    uint32_t prefill_chunk_for(uint32_t tokens) const {
        const uint32_t per_horizon = PrefetchStagingArena::EXPERTS_PER_HORIZON;
        const uint32_t by_arena = host_.staging_slot_count() / per_horizon;
        const uint32_t by_pool =
            std::max<uint32_t>(1, host_.registry().vram_capacity / per_horizon);
        const uint32_t limit = std::min(
            std::min(V4LayerBodyBatchScratch::kMaxTokens, by_arena), by_pool);
        return std::max<uint32_t>(1, std::min(limit, tokens));
    }

    // The window `W` for a prompt of `tokens` (Step 6 §6b): the whole prompt unless
    // `AeonRuntimeConfig::prefill_window` bounds it, in which case the prompt is
    // `⌈N/W⌉` layer-major passes.
    uint32_t prefill_window_for(uint32_t tokens) const noexcept {
        const uint32_t configured = options_.runtime.prefill_window;
        if (configured == 0) return tokens;
        return std::min(configured, tokens);
    }

    // --- the run -------------------------------------------------------------

    // Render the conversation, then generate. Stateless across calls: the layer
    // state is reset and the sampler reseeded first, so two calls with the same
    // arguments produce the same reply (which is a property a gate can hold the
    // engine to, and one a caller swapping sessions would otherwise have to know
    // to restore by hand).
    //
    // `generation` supplies the token cap and the stop policy; the EOS id comes
    // from the tokenizer, not from the caller, because the tokenizer is the thing
    // that knows it. `sampling` is the artifact's policy unless the caller
    // overrides it.
    V4Reply chat(const std::vector<text::Dsv4PromptMessage>& messages,
                 const text::Dsv4PromptOptions& prompt_options,
                 const text::GenerationOptions& generation,
                 const V4SamplerConfig& sampling) {
        if (!ready()) {
            throw std::logic_error("V4Engine::chat: the engine was not initialized");
        }
        if (messages.empty()) {
            throw std::invalid_argument("V4Engine::chat: the conversation is empty");
        }

        const std::vector<uint32_t> prompt = encoder_->encode_tokens(messages, prompt_options);
        if (prompt.empty()) {
            throw std::runtime_error("V4Engine::chat: the rendered prompt has no tokens");
        }

        const uint32_t capacity = host_.context_capacity();
        if (prompt.size() >= capacity) {
            throw std::runtime_error(
                "V4Engine::chat: the rendered prompt is " +
                std::to_string(prompt.size()) + " tokens, which leaves no room to generate "
                "inside a context of " + std::to_string(capacity));
        }

        host_.reset_generation_state();
        // `set_config` validates and seeds the generator, so installing the config
        // is also the reseed — two calls would be two places for the seed to be
        // set and one place for them to disagree.
        sampler_->set_config(sampling);

        text::GenerationOptions loop_options = generation;
        loop_options.eos_token_id = tokenizer_.eos_token_id();
        // The context limit is the host's capacity, not the caller's number: a
        // position at or beyond it is refused by the layer, and a refusal three
        // tokens into a decode is a worse failure than a bound stated up front.
        loop_options.context_limit = capacity;

        V4Reply reply;
        reply.prompt_tokens = static_cast<uint32_t>(prompt.size());

        // The two phases are two strategies (expert-streaming plan Step 6, §3), so
        // they are two steps and not one with a flag:
        //
        //   prefill  the whole prompt through `V4Graph::forward_window` — one
        //            layer-major pass (or `⌈N/W⌉` of them), which is where the
        //            swept supply, the chunk-wide deduplicated dispatch and the
        //            hard Hot drain live;
        //   decode   one token at a time through `advance` → `forward_token`, on
        //            the on-demand + LRU + demotion strategy decode owns.
        //
        // The loop itself is still `text::generate_token_ids`: it owns the EOS
        // stop, the `max_new_tokens` cap and the context limit, and all this adds
        // is the prefill *step* it is handed. The `prefill = true` flag on the
        // decode step is therefore dropped — every decode step is a decode step.
        double first_token_ms = 0.0;
        Clock::time_point decode_started{};
        const auto started = Clock::now();

        const uint32_t window = prefill_window_for(static_cast<uint32_t>(prompt.size()));
        const uint32_t chunk = prefill_chunk_for(static_cast<uint32_t>(prompt.size()));

        const text::PromptPrefill prefill_step =
            [&](const std::vector<uint32_t>& tokens) -> uint32_t {
                // The phase is set before the forward pass, because the expert
                // dispatch happens inside it and `TieredExpertSupply` records against
                // whichever phase is current at that moment. Opening the sink already
                // made this a no-op when telemetry is off.
                host_.set_supply_phase(true);
                const uint32_t count = static_cast<uint32_t>(tokens.size());
                const half* logits = nullptr;
                for (uint32_t offset = 0; offset < count; offset += window) {
                    const uint32_t span = std::min(window, count - offset);
                    logits = graph_->forward_window(
                        tokens.data() + offset, offset, span, std::min(chunk, span),
                        host_.streams().compute);
                }
                if (logits_dump_.is_open()) dump_logits(logits);
                // TTFT is the moment the *last* prompt token's forward has produced a
                // token — not the moment the loop was entered, and not the first
                // prompt token's. The decode clock starts there for the same reason:
                // the first generated token came out of the prefill, so charging it
                // to the decode rate would report a rate the decode never ran.
                first_token_ms = elapsed_ms(started);
                decode_started = Clock::now();
                return sampler_->select(logits, host_.streams().compute);
            };

        const text::TokenStep decode_step =
            [&](uint32_t token_id, uint32_t position, bool /*prefill*/) -> uint32_t {
                host_.set_supply_phase(false);
                const uint32_t next = advance(token_id, position);
                host_.record_supply_decode_token();
                return next;
            };

        const text::GenerationResult generated =
            text::generate_token_ids(prompt, loop_options, prefill_step, decode_step);

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
    // appears. A pure function of the string, so it is testable with no model, and
    // so the engine is not the thing that decides what a user may see.
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
    std::unique_ptr<V4Sampler> sampler_{};
};

}  // namespace aeon::core
