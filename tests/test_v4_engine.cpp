// -----------------------------------------------------------------------------
// P4 gate — the text binding: one conversation in, text out.
//
// This is the composition plan's phase P4 and, in its own words, the plan's
// **acceptance criterion**: "one command takes a conversation and returns text"
// (§1). Everything it binds already ran — the tokenizer, the canonical prompt
// encoder and `text::generate_token_ids` were certified by Tiers 0–3; the host,
// the graph and the sampler by P1–P3. What never existed was the binding, and that
// is the only thing this gate is about.
//
// WHAT THE GATE IS ENTITLED TO ASSERT, and what it is not. Coherence — "the reply
// is a reply to *that* prompt" — is the criterion's first clause and it is **not
// machine-checkable**: no threshold on a token sequence says whether prose is
// responsive. So the gate asserts what *is* observable and prints what is not:
//
//   A. THE ENGINE IS THE BINDING. `chat` must produce the same token sequence as
//      a loop written **here**, in the test, that drives `graph.forward_token` and
//      `sampler.select` directly and computes the positions itself. A binding that
//      skipped the state reset, mis-positioned a token, fed the wrong prompt or
//      reseeded wrongly cannot pass it. This is P2's "the driver is the loop it
//      claims to be", moved one layer up.
//   B. THE REPLY IS TEXT. Non-empty, valid UTF-8, decoded by the artifact's own
//      tokenizer — the pipeline's *last* step, which nothing else checks.
//   C. IT IS DETERMINISTIC. Two calls with the same arguments produce the same
//      ids and the same text, because the engine resets state and reseeds before
//      each one. (The graph itself is bit-reproducible — the rewrite has one
//      accumulation, the fixed-order fp32 reduce — which P2 measured as 0
//      differing of 517 120 fp16 logits.)
//   D. THE HISTORY IS IN THE CONTEXT. The same user turn, asked after a prior
//      exchange, produces a **different** first token than it does alone: the
//      conversation is genuinely rendered and fed, not merely accepted. This is
//      the closest machine-checkable stand-in for the criterion's second clause.
//   E. EVERY STOP REASON IS REACHABLE, and the bound is refused rather than
//      clamped: `max_new_tokens` is honoured; a prompt that fills the context is
//      refused; and a position at capacity throws (trap 40, through the graph).
//   F. THE ARTIFACT'S POLICY IS READ, not assumed: `generation_config.json`
//      converts to a sampler config the way the plan says it does.
//   G. THE THINKING STRIP IS A PURE FUNCTION of the decoded string, tested on
//      hand-built text so it needs no model and cannot be carried by the
//      acceptance run.
//
// WHAT IS PRINTED RATHER THAN ASSERTED. The generated text itself. The one claim
// a human has to judge is the one the gate must not pretend to decide, so it is
// shown, and the gate's green line means exactly what section A–F say and no more.
//
// COST. The host assembly is ~8 s and there is no reason to pay it per section, so
// it is built once and reused; generations are kept to four new tokens each, which
// is enough for every statement above and keeps the whole gate near a minute. The
// alternative — one token per generation — would make section D (where the first
// token *is* the observation) still work but would make section A unable to see a
// decode-position error, which is the defect it exists to catch.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_engine.hpp"
#include "architecture/deepseek_v4/core/v4_graph.hpp"
#include "architecture/deepseek_v4/core/v4_model_host.hpp"
#include "architecture/deepseek_v4/core/v4_sampler.hpp"
#include "infrastructure/hip_check.hpp"
#include "infrastructure/text/text_generation.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using aeon::core::V4Engine;
using aeon::core::V4EngineOptions;
using aeon::core::V4GenerationPolicy;
using aeon::core::V4Graph;
using aeon::core::V4ModelHost;
using aeon::core::V4Reply;
using aeon::core::V4Sampler;
using aeon::core::V4SamplerConfig;
using aeon::text::Dsv4PromptMessage;
using aeon::text::Dsv4PromptOptions;
using aeon::text::Dsv4Role;
using aeon::text::Dsv4ThinkingMode;
using aeon::text::GenerationOptions;
using aeon::text::StopReason;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

// The context the whole gate runs at. 256 is the P1/P2 gates' value: the model's
// real window (128) and its real indexer width (512) are exercised for real, and
// the expert tier fits comfortably, which is what keeps the assembly honest
// without making the gate a memory test.
constexpr uint32_t kContext = 256;

// Four new tokens per generation. Enough that a decode-position error is visible
// (one token would not be), small enough that ~ten generations stay inside a
// minute at the measured ~0.4 s per layer-step pass over 43 layers.
constexpr uint32_t kNewTokens = 4;

constexpr const char* kUserTurn = "What is the capital of France?";

// --- the harness -------------------------------------------------------------

struct Harness {
    uint32_t checks{0};
    uint32_t failures{0};

    bool assert_that(const char* label, bool ok, const std::string& detail) {
        std::printf("  %-58s %-40s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
        ++checks;
        if (!ok) ++failures;
        return ok;
    }
};

Harness harness;

using Clock = std::chrono::steady_clock;

double since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void stage(const char* name, const Clock::time_point& start) {
    std::printf("  [time] %-46s %.1f s\n", name, since(start));
}

std::string ids_to_string(const std::vector<uint32_t>& ids, size_t limit = 12) {
    std::string out = "[";
    for (size_t i = 0; i < ids.size() && i < limit; ++i) {
        if (i) out += ", ";
        out += std::to_string(ids[i]);
    }
    if (ids.size() > limit) out += ", ...";
    return out + "]";
}

// Is this a well-formed UTF-8 byte string? The tokenizer's decoder is supposed to
// emit one; a binding that sliced bytes rather than tokens would not. Checked
// here rather than trusted, because "it printed something" is not evidence that
// the detokenizer was used.
bool is_valid_utf8(const std::string& text) {
    size_t i = 0;
    while (i < text.size()) {
        const uint8_t byte = static_cast<uint8_t>(text[i]);
        size_t extra = 0;
        uint32_t code_point = 0;
        if (byte < 0x80) {
            i += 1;
            continue;
        } else if ((byte & 0xE0) == 0xC0) {
            extra = 1; code_point = byte & 0x1F;
        } else if ((byte & 0xF0) == 0xE0) {
            extra = 2; code_point = byte & 0x0F;
        } else if ((byte & 0xF8) == 0xF0) {
            extra = 3; code_point = byte & 0x07;
        } else {
            return false;
        }
        if (i + extra >= text.size()) return false;
        for (size_t k = 1; k <= extra; ++k) {
            const uint8_t continuation = static_cast<uint8_t>(text[i + k]);
            if ((continuation & 0xC0) != 0x80) return false;
            code_point = (code_point << 6) | (continuation & 0x3F);
        }
        // Reject the two shapes UTF-8 forbids outright: over-long encodings and
        // the surrogate range.
        if (extra == 1 && code_point < 0x80) return false;
        if (extra == 2 && code_point < 0x800) return false;
        if (extra == 3 && code_point < 0x10000) return false;
        if (code_point > 0x10FFFF) return false;
        if (code_point >= 0xD800 && code_point <= 0xDFFF) return false;
        i += extra + 1;
    }
    return true;
}

Dsv4PromptMessage user_message(const std::string& text) {
    Dsv4PromptMessage message;
    message.role = Dsv4Role::User;
    message.content = text;
    return message;
}

Dsv4PromptMessage assistant_message(const std::string& text) {
    Dsv4PromptMessage message;
    message.role = Dsv4Role::Assistant;
    message.content = text;
    return message;
}

// The gate's own generation loop. It shares no line with `V4Engine::chat`: it
// drives the graph and the sampler directly and computes every position itself,
// which is what makes section A a statement about the engine rather than about
// the gate agreeing with itself.
//
// It mirrors `text::generate_token_ids`'s positions exactly — prompt token `i` at
// position `i`, then generated token `g` at `prompt.size() + g - 1` — because that
// is the contract the engine is required to satisfy, not a re-derivation of it.
std::vector<uint32_t> independent_replay(V4ModelHost& host, V4Graph& graph, V4Sampler& sampler,
                                         const std::vector<uint32_t>& prompt,
                                         uint32_t new_tokens,
                                         const V4SamplerConfig& sampling) {
    host.reset_generation_state();
    sampler.set_config(sampling);
    sampler.reseed(sampling.seed);

    std::vector<uint32_t> produced;
    uint32_t next = 0;

    for (size_t index = 0; index < prompt.size(); ++index) {
        const half* logits = graph.forward_token(
            prompt[index], static_cast<uint32_t>(index), host.streams().compute);
        next = sampler.select(logits, host.streams().compute);
    }
    produced.push_back(next);

    for (uint32_t generated = 1; generated < new_tokens; ++generated) {
        const uint32_t position = static_cast<uint32_t>(prompt.size()) + generated - 1;
        const half* logits = graph.forward_token(next, position, host.streams().compute);
        next = sampler.select(logits, host.streams().compute);
        produced.push_back(next);
    }
    return produced;
}

GenerationOptions generation_options(uint32_t new_tokens) {
    GenerationOptions options;
    options.max_new_tokens = new_tokens;
    options.context_limit = kContext;
    options.stop_on_eos = true;
    return options;
}

// The logits after feeding a whole prompt, read back at the last prompt position.
// Used to compare two contexts *by their distribution* rather than by the token
// one of them happened to draw: a peaked model can draw the same token from two
// different distributions, so a token comparison is the weaker instrument.
std::vector<double> prompt_logits(V4Engine& engine, const std::vector<uint32_t>& prompt) {
    engine.host().reset_generation_state();
    const half* logits = nullptr;
    for (size_t index = 0; index < prompt.size(); ++index) {
        logits = engine.graph().forward_token(
            prompt[index], static_cast<uint32_t>(index), engine.host().streams().compute);
    }
    const uint32_t vocab = engine.sampler().vocab();
    std::vector<half> host_logits(vocab);
    CHECK_HIP(hipMemcpyAsync(host_logits.data(), logits, vocab * sizeof(half),
                             hipMemcpyDeviceToHost, engine.host().streams().compute));
    CHECK_HIP(hipStreamSynchronize(engine.host().streams().compute));
    std::vector<double> out(vocab);
    for (uint32_t i = 0; i < vocab; ++i) out[i] = static_cast<double>(host_logits[i]);
    return out;
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  P4 — the text binding: conversation in, text out\n");
    std::printf("================================================================================\n");
    aeon::core::select_compute_device(true);

    const auto gate_start = Clock::now();
    const auto build_start = Clock::now();

    V4EngineOptions engine_options;
    engine_options.model_dir = kModelDir;
    engine_options.runtime.context_size = kContext;
    engine_options.seed = 1234;

    V4Engine engine;
    engine.initialize(engine_options);
    stage("A: the engine assembly", build_start);

    // The policy is used by every generation below, so it is fixed once.
    const V4SamplerConfig sampling = engine.policy().to_sampler_config(engine.options().seed);

    Dsv4PromptOptions prompt_options;
    prompt_options.thinking_mode = Dsv4ThinkingMode::Chat;

    // =========================================================================
    // F. The artifact's own sampling policy
    // =========================================================================
    std::printf("\n[F] The artifact's sampling policy\n");
    {
        harness.assert_that("F: generation_config.json was loaded",
                            engine.policy_loaded(), engine.policy_loaded() ? "loaded" : "absent");

        const V4GenerationPolicy& policy = engine.policy();
        char detail[128];
        std::snprintf(detail, sizeof(detail), "do_sample=%s T=%.2f top_p=%.2f",
                      policy.do_sample ? "true" : "false",
                      static_cast<double>(policy.temperature),
                      static_cast<double>(policy.top_p));
        harness.assert_that("F: it says what the plan records (sampling, T=1, top_p=1)",
                            policy.do_sample && policy.temperature == 1.0f &&
                                policy.top_p == 1.0f,
                            detail);

        harness.assert_that("F: the artifact's EOS is the tokenizer's EOS",
                            policy.eos_token_id == engine.tokenizer().eos_token_id(),
                            std::to_string(policy.eos_token_id) + " == " +
                                std::to_string(engine.tokenizer().eos_token_id()));

        // `do_sample = false` must land on the sampler's greedy path, which is the
        // `T -> 0` limit rather than a second mode. This mapping is the one place
        // the two conventions meet, so it is asserted rather than reasoned about.
        V4GenerationPolicy greedy_policy = policy;
        greedy_policy.do_sample = false;
        harness.assert_that("F: do_sample=false maps to the sampler's greedy path",
                            greedy_policy.to_sampler_config(0).temperature == 0.0f,
                            "temperature 0");
    }

    // =========================================================================
    // A. The engine is the binding
    // =========================================================================
    std::printf("\n[A] The engine is the binding\n");
    const auto turn_start = Clock::now();
    std::vector<uint32_t> prompt_ids;
    V4Reply reply;
    {
        const std::vector<Dsv4PromptMessage> conversation = {user_message(kUserTurn)};
        prompt_ids = engine.encoder().encode_tokens(conversation, prompt_options);

        reply = engine.chat(conversation, prompt_options, generation_options(kNewTokens), sampling);

        harness.assert_that("A: the rendered conversation has tokens",
                            !prompt_ids.empty(),
                            std::to_string(prompt_ids.size()) + " prompt tokens");
        harness.assert_that("A: the reply reports the prompt it was given",
                            reply.prompt_tokens == prompt_ids.size(),
                            std::to_string(reply.prompt_tokens) + " == " +
                                std::to_string(prompt_ids.size()));

        const std::vector<uint32_t> replayed = independent_replay(
            engine.host(), engine.graph(), engine.sampler(), prompt_ids, kNewTokens, sampling);

        // Equality is expected outright: with a four-token cap and a prompt this
        // short, four tokens were available and EOS is not expected. If the model
        // did stop early, the engine's sequence must still be the replay's prefix
        // — a shorter run, never a different one.
        const bool exact = reply.token_ids == replayed;
        const bool prefix = reply.token_ids.size() <= replayed.size() &&
            std::equal(reply.token_ids.begin(), reply.token_ids.end(), replayed.begin());

        if (reply.stop_reason != StopReason::Eos && reply.stop_reason != StopReason::ContextLimit) {
            harness.assert_that("A: chat == the gate's own forward_token + select loop",
                                exact, ids_to_string(reply.token_ids) + " vs " +
                                           ids_to_string(replayed));
        } else {
            harness.assert_that("A: chat == a prefix of the gate's own loop (stopped early)",
                                prefix, "stop: " +
                                            std::string(aeon::text::stop_reason_name(
                                                reply.stop_reason)));
        }

        harness.assert_that("A: the graph's logits at two positions differ (RoPE is live)",
                            [&] {
                                engine.host().reset_generation_state();
                                const half* first = engine.graph().forward_token(
                                    prompt_ids.front(), 0, engine.host().streams().compute);
                                std::vector<half> zero_logits(engine.sampler().vocab());
                                CHECK_HIP(hipMemcpyAsync(
                                    zero_logits.data(), first,
                                    zero_logits.size() * sizeof(half), hipMemcpyDeviceToHost,
                                    engine.host().streams().compute));
                                CHECK_HIP(hipStreamSynchronize(engine.host().streams().compute));

                                engine.host().reset_generation_state();
                                const half* second = engine.graph().forward_token(
                                    prompt_ids.front(), 5, engine.host().streams().compute);
                                std::vector<half> five_logits(engine.sampler().vocab());
                                CHECK_HIP(hipMemcpyAsync(
                                    five_logits.data(), second,
                                    five_logits.size() * sizeof(half), hipMemcpyDeviceToHost,
                                    engine.host().streams().compute));
                                CHECK_HIP(hipStreamSynchronize(engine.host().streams().compute));

                                uint32_t differing = 0;
                                for (size_t i = 0; i < zero_logits.size(); ++i) {
                                    if (zero_logits[i] != five_logits[i]) ++differing;
                                }
                                std::printf("  [note] position 0 vs 5: %u of %zu logits differ\n",
                                            differing, zero_logits.size());
                                return differing > zero_logits.size() / 2;
                            }(),
                            "see the note above");
    }
    stage("A: one generation and its replay", turn_start);

    // =========================================================================
    // B. The reply is text
    // =========================================================================
    std::printf("\n[B] The reply is text\n");
    {
        harness.assert_that("B: the reply is non-empty", !reply.text.empty(),
                            std::to_string(reply.text.size()) + " bytes");
        harness.assert_that("B: the reply is valid UTF-8", is_valid_utf8(reply.text),
                            is_valid_utf8(reply.text) ? "well-formed" : "malformed");

        // The text recomputed from the ids, independently of the engine's own
        // path: EOS stripped, then the artifact tokenizer's decode. A binding that
        // returned the raw ids as bytes, or decoded with EOS attached, fails here
        // and nowhere else.
        std::vector<uint32_t> visible = reply.token_ids;
        if (!visible.empty() && visible.back() == engine.tokenizer().eos_token_id()) {
            visible.pop_back();
        }
        harness.assert_that("B: the text is the detokenization of the ids, EOS stripped",
                            reply.text == engine.tokenizer().decode(visible),
                            std::to_string(visible.size()) + " visible token(s)");

        // The tokenizer round-trip, which is what makes "the reply is the model's
        // text" a statement about the detokenizer rather than about a byte copy.
        const std::vector<uint32_t> round_trip = engine.tokenizer().encode(kUserTurn);
        harness.assert_that("B: the prompt round-trips through the tokenizer",
                            engine.tokenizer().decode(round_trip) == kUserTurn,
                            std::to_string(round_trip.size()) + " tokens");
    }

    // =========================================================================
    // C. Determinism
    // =========================================================================
    std::printf("\n[C] Determinism\n");
    {
        const std::vector<Dsv4PromptMessage> conversation = {user_message(kUserTurn)};
        const V4Reply second = engine.chat(conversation, prompt_options,
                                           generation_options(kNewTokens), sampling);
        harness.assert_that("C: the same call twice gives the same token ids",
                            second.token_ids == reply.token_ids,
                            ids_to_string(second.token_ids));
        harness.assert_that("C: and the same text", second.text == reply.text,
                            std::to_string(reply.text.size()) + " bytes");

        // A different seed must be able to differ, or the sampler was never
        // sampling and every determinism check above is vacuous. At the artifact's
        // own `T=1` the distribution is peaked enough that eight seeds can agree on
        // four tokens — measured, not assumed — so the divergence is forced where
        // it is guaranteed: a flattened distribution (`T=5`), where the seed is the
        // only thing deciding.
        V4SamplerConfig flat = sampling;
        flat.temperature = 5.0f;
        const V4Reply flat_reference =
            engine.chat(conversation, prompt_options, generation_options(kNewTokens), flat);
        bool differed = false;
        for (uint64_t offset = 1; offset <= 4 && !differed; ++offset) {
            V4SamplerConfig other_seed = flat;
            other_seed.seed = engine.options().seed + offset;
            const V4Reply other = engine.chat(conversation, prompt_options,
                                              generation_options(kNewTokens), other_seed);
            differed = other.token_ids != flat_reference.token_ids;
        }
        harness.assert_that("C: at T=5 a different seed gives a different sequence",
                            differed, differed ? "differs" : "4 seeds coincide");

        // The negative control: the greedy path never consults the generator, so
        // the seed is *irrelevant* there. This is what makes the check above a
        // statement about sampling rather than about the RNG being called.
        V4SamplerConfig greedy = sampling;
        greedy.temperature = 0.0f;
        greedy.seed = 1;
        const V4Reply greedy_first =
            engine.chat(conversation, prompt_options, generation_options(kNewTokens), greedy);
        uint32_t seed_variants_matching = 0;
        for (uint64_t seed_value : {2ull, 3ull, 99ull}) {
            greedy.seed = seed_value;
            const V4Reply variant =
                engine.chat(conversation, prompt_options, generation_options(kNewTokens), greedy);
            if (variant.token_ids == greedy_first.token_ids) ++seed_variants_matching;
        }
        harness.assert_that("C: the greedy path ignores the seed (3 of 3 seeds agree)",
                            seed_variants_matching == 3,
                            std::to_string(seed_variants_matching) + " of 3");
    }

    // =========================================================================
    // D. The history is in the context
    // =========================================================================
    std::printf("\n[D] The history is in the context\n");
    {
        const std::vector<Dsv4PromptMessage> one_turn = {user_message(kUserTurn)};
        const std::vector<Dsv4PromptMessage> two_turns = {
            user_message(kUserTurn),
            assistant_message("The capital of France is Paris, on the Seine."),
            user_message("And what river runs through it?"),
        };

        const std::vector<uint32_t> short_prompt =
            engine.encoder().encode_tokens(one_turn, prompt_options);
        const std::vector<uint32_t> long_prompt =
            engine.encoder().encode_tokens(two_turns, prompt_options);

        harness.assert_that("D: the longer conversation renders a longer prompt",
                            long_prompt.size() > short_prompt.size(),
                            std::to_string(short_prompt.size()) + " -> " +
                                std::to_string(long_prompt.size()) + " tokens");
        harness.assert_that("D: the shorter prompt is a prefix of the longer one",
                            std::equal(short_prompt.begin(), short_prompt.end(),
                                       long_prompt.begin()),
                            "prefix");

        const V4Reply alone = engine.chat(one_turn, prompt_options,
                                          generation_options(1), sampling);
        const V4Reply after_history = engine.chat(two_turns, prompt_options,
                                                  generation_options(1), sampling);

        // The token the model happens to draw is the *weaker* instrument here, and
        // the gate says so: a peaked model can draw the same token from two
        // different distributions. The distribution itself is the observable, so
        // the history claim is made on the logits — which cannot coincide by
        // chance — and the drawn tokens are reported beside it.
        const std::vector<double> logits_alone = prompt_logits(engine, short_prompt);
        const std::vector<double> logits_after = prompt_logits(engine, long_prompt);
        uint32_t differing = 0;
        for (size_t i = 0; i < logits_alone.size(); ++i) {
            if (logits_alone[i] != logits_after[i]) ++differing;
        }
        const double fraction = static_cast<double>(differing) /
                                static_cast<double>(logits_alone.size());

        harness.assert_that("D: the history changes the model's own distribution (logits)",
                            fraction > 0.5,
                            std::to_string(differing) + " of " +
                                std::to_string(logits_alone.size()) + " differ");
        std::printf("  [note] the same user turn decides %s alone and %s after the history\n",
                    ids_to_string(alone.token_ids).c_str(),
                    ids_to_string(after_history.token_ids).c_str());
    }

    // =========================================================================
    // E. Stop conditions and the refusal
    // =========================================================================
    std::printf("\n[E] Stop conditions\n");
    {
        const std::vector<Dsv4PromptMessage> conversation = {user_message(kUserTurn)};

        const V4Reply one = engine.chat(conversation, prompt_options,
                                        generation_options(1), sampling);
        harness.assert_that("E: max_new_tokens = 1 produces one token",
                            one.token_ids.size() == 1,
                            std::to_string(one.token_ids.size()) + " token(s)");

        const V4Reply many = engine.chat(conversation, prompt_options,
                                         generation_options(6), sampling);
        harness.assert_that("E: max_new_tokens = 6 is honoured (or EOS stopped it early)",
                            many.token_ids.size() == 6 || many.stop_reason == StopReason::Eos,
                            std::to_string(many.token_ids.size()) + " token(s), " +
                                aeon::text::stop_reason_name(many.stop_reason));

        // The stop reason is a property of the count, not a free-floating label.
        const bool reason_consistent =
            (many.stop_reason == StopReason::Eos && many.token_ids.back() ==
             engine.tokenizer().eos_token_id()) ||            (many.stop_reason == StopReason::MaxNewTokens && many.token_ids.size() == 6) ||
            (many.stop_reason == StopReason::ContextLimit);
        harness.assert_that("E: the stop reason matches what the sequence shows",
                            reason_consistent,
                            aeon::text::stop_reason_name(many.stop_reason));

        // THE EOS STRIP, ON A REPLY THAT ACTUALLY REACHED EOS.
        //
        // This is a check the earlier sections structurally cannot make: with a
        // four- or six-token cap the sequence never reaches EOS, so `visible ==
        // token_ids` there and the strip is never exercised. The acceptance prompt
        // does reach EOS (at token 9), and the comparison is against the
        // detokenization recomputed by the gate — both with and without the EOS
        // token — so it fails if the strip is skipped *and* if more than the EOS
        // token is dropped.
        const V4Reply stopped = engine.chat(conversation, prompt_options,
                                            generation_options(16), sampling);
        const bool reached_eos = stopped.stop_reason == StopReason::Eos;
        harness.assert_that("E: the acceptance prompt reaches EOS within 16 tokens",
                            reached_eos && !stopped.token_ids.empty() &&
                                stopped.token_ids.back() == engine.tokenizer().eos_token_id(),
                            std::string(aeon::text::stop_reason_name(stopped.stop_reason)) +
                                ", " + std::to_string(stopped.token_ids.size()) + " tokens");
        if (reached_eos && !stopped.token_ids.empty()) {
            std::vector<uint32_t> without_eos = stopped.token_ids;
            without_eos.pop_back();  // the EOS the loop appended
            const std::string with_eos_text = engine.tokenizer().decode(stopped.token_ids);
            const std::string without_eos_text = engine.tokenizer().decode(without_eos);
            harness.assert_that("E: the text is the detokenization WITHOUT the EOS token",
                                stopped.text == without_eos_text,
                                std::to_string(without_eos.size()) + " visible token(s)");
            harness.assert_that("E: and it differs from the detokenization WITH it",
                                with_eos_text != stopped.text, "EOS token not in the text");
        }

        // A prompt that fills the context is **refused**, not truncated. The
        // engine's own words are what is asserted, since the alternative — a
        // silently shorter prompt — is a different conversation.
        std::string long_text;
        for (int i = 0; i < 4000; ++i) long_text += "word ";
        bool refused = false;
        std::string message;
        try {
            (void)engine.chat({user_message(long_text)}, prompt_options,
                              generation_options(kNewTokens), sampling);
        } catch (const std::exception& error) {
            refused = true;
            message = error.what();
        }
        harness.assert_that("E: a prompt that fills the context is refused",
                            refused, refused ? "refused: " + message.substr(0, 40) : "accepted");

        // *Which* component refused is the observable, and it is the only one.
        // The threshold necessarily coincides with `generate_token_ids`'s own
        // (`prompt.size() >= context_limit`, and `chat` sets `context_limit = `
        // capacity), so a bare "it threw" cannot tell the engine's guard from the
        // loop's — the check has to name the message. The placement is the
        // property: the engine validates before it resets state, installs the
        // sampling config or runs a single layer.
        harness.assert_that("E: the refusal is the engine's own, not the loop's",
                            refused && message.find("leaves no room to generate") !=
                                           std::string::npos,
                            refused ? message.substr(0, 46) : "no refusal");

        // And the position bound is a refusal rather than a wrap (trap 40), seen
        // through the graph this time rather than through the layer.
        bool position_refused = false;
        try {
            (void)engine.graph().forward_token(
                prompt_ids.front(), kContext, engine.host().streams().compute);
        } catch (const std::exception&) {
            position_refused = true;
        }
        harness.assert_that("E: a position at capacity throws rather than wraps",
                            position_refused, position_refused ? "threw" : "accepted");
    }

    // =========================================================================
    // H. The prefill strategy follows the window (Step 6 D-a)
    // =========================================================================
    //
    // The engine's prefill is the layer-major window, and the *sweep* — drain Hot,
    // stream whole layer sets in computation order, release each layer as it
    // retires — is engaged only when the window clears the over-fetch rule
    // (`V4PrefillSweep::worth`): the sweep loads a whole layer, so a window whose
    // routed draws do not reach the layer's size would fetch far more than the
    // layer-major driver needed. Both claims are asserted, in opposite directions,
    // because "the sweep ran" alone cannot distinguish the strategy from a code
    // path that always sweeps or never does.
    //
    // What is *not* re-asserted here: byte-equality of the swept path. That is
    // `test_v4_prefill_sweep`'s subject (ledger M40), at the graph level and
    // without the engine, so re-running it through `chat` would buy nothing and
    // cost a second whole-model read. What is new here is that the *engine* chooses
    // the strategy and that the switch leaves no residue on either side of it.
    std::printf("\n[H] The prefill strategy follows the window\n");
    {
        const uint64_t loads_before = engine.host().prefill_sweep().layer_loads();
        harness.assert_that("H: no swept prefill has run yet (the short prompts)",
                            loads_before == 0,
                            std::to_string(loads_before) + " layer loads");

        // A prompt long enough to clear `6W >= 2 x experts_per_layer` at this
        // context's 256 experts per layer, built by repetition so the token count
        // is measured rather than guessed.
        std::string long_text;
        std::vector<uint32_t> long_prompt;
        do {
            long_text += "The committee reviewed the proposal carefully. ";
            long_prompt = engine.encoder().encode_tokens({user_message(long_text)},
                                                         prompt_options);
        } while (long_prompt.size() < 100);

        harness.assert_that("H: the long prompt clears the sweep's over-fetch rule",
                            long_prompt.size() >= 86 && long_prompt.size() < kContext,
                            std::to_string(long_prompt.size()) + " tokens");

        const auto prefill_start = Clock::now();
        const V4Reply swept = engine.chat({user_message(long_text)}, prompt_options,
                                          generation_options(1), sampling);
        const double prefill_seconds = since(prefill_start);

        const uint64_t loads_after = engine.host().prefill_sweep().layer_loads();
        harness.assert_that("H: the sweep is engaged for a long window",
                            engine.host().prefill_sweep_engaged() &&
                                loads_after - loads_before == engine.host().num_layers(),
                            std::to_string(loads_after - loads_before) + " layer loads, " +
                                std::to_string(engine.host().prefill_sweep().experts_streamed()) +
                                " experts streamed");
        harness.assert_that("H: Hot was drained on entry — nothing crossed over",
                            engine.host().prefill_sweep().hot_after_drain() == 0,
                            std::to_string(engine.host().prefill_sweep().hot_after_drain()) +
                                " Hot residents at the switch");
        harness.assert_that("H: the switch left nothing behind",
                            engine.host().outstanding_expert_leases() == 0 &&
                                engine.host().staging_in_use_slots() == 0 &&
                                engine.host().registry().invariants_hold() &&
                                !engine.host().registry().prefill_streaming(),
                            "leases 0, staging 0, invariants hold, not streaming");
        harness.assert_that("H: the swept prefill produced a token", !swept.token_ids.empty(),
                            ids_to_string(swept.token_ids));

        // Step 6 outcome 5, at the engine: how many bytes a prompt of this length
        // costs and how long it took. Printed rather than asserted — it is a
        // measurement, and the plan's `⌈N/W⌉ x 156 GB` is the comparison it is for.
        std::printf("  [note] swept prefill: %zu tokens in %.1f s (%.1f tok/s), %llu experts "
                    "in %llu loads, frontier %u layers\n",
                    long_prompt.size(), prefill_seconds,
                    prefill_seconds > 0.0
                        ? static_cast<double>(long_prompt.size()) / prefill_seconds : 0.0,
                    static_cast<unsigned long long>(
                        engine.host().prefill_sweep().experts_streamed()),
                    static_cast<unsigned long long>(loads_after - loads_before),
                    engine.host().prefill_sweep().frontier_depth());

        // And the other direction: a short prompt keeps the layer-major window and
        // its chunk-wide dedup but does not pre-load whole layers.
        const V4Reply short_reply = engine.chat({user_message(kUserTurn)}, prompt_options,
                                                generation_options(1), sampling);
        harness.assert_that("H: a short window does not sweep",
                            engine.host().prefill_sweep().layer_loads() == loads_after &&
                                !engine.host().prefill_sweep_engaged(),
                            std::to_string(engine.host().prefill_sweep().layer_loads()) +
                                " layer loads, engaged=" +
                                (engine.host().prefill_sweep_engaged() ? "true" : "false"));
        harness.assert_that("H: the short prefill still produced a token",
                            !short_reply.token_ids.empty(),
                            ids_to_string(short_reply.token_ids));
    }

    // =========================================================================
    // G. strip_thinking — pure text, no model
    // =========================================================================
    std::printf("\n[G] strip_thinking\n");
    {
        const std::string marker =
            engine.tokenizer().decode({engine.tokenizer().thinking_end_token_id()});
        harness.assert_that("G: the tokenizer names a thinking-end marker",
                            !marker.empty(), std::to_string(marker.size()) + " bytes");

        const std::string with_thinking = std::string("a reasoning block") + marker + "the reply";
        harness.assert_that("G: text after the marker is returned",
                            V4Engine::strip_thinking(with_thinking, engine.tokenizer()) ==
                                "the reply",
                            "suffix");

        // The last marker wins: a reply that mentions an earlier block keeps the
        // text after the final one, which is what the reference CLI did.
        const std::string twice =
            std::string("first") + marker + "second" + marker + "third";
        harness.assert_that("G: the last marker wins",
                            V4Engine::strip_thinking(twice, engine.tokenizer()) == "third",
                            "third");

        harness.assert_that("G: text without a marker is returned unchanged",
                            V4Engine::strip_thinking("plain reply", engine.tokenizer()) ==
                                "plain reply",
                            "unchanged");
    }

    // =========================================================================
    // The generated text, printed because the gate must not pretend to judge it
    // =========================================================================
    std::printf("\n[S] The generated text (human judgement, not asserted)\n");
    {
        const V4Reply shown = engine.chat({user_message(kUserTurn)}, prompt_options,
                                          generation_options(24), sampling);
        std::printf("  prompt   : %s\n", kUserTurn);
        std::printf("  reply    : %s\n", shown.text.c_str());
        std::printf("  tokens   : %zu, stop: %s, TTFT %.0f ms, %.1f tok/s\n",
                    shown.token_ids.size(), aeon::text::stop_reason_name(shown.stop_reason),
                    shown.ttft_ms, shown.decode_tokens_per_second);
        std::printf("  context  : %u tokens, prompt %u\n", kContext, shown.prompt_tokens);
    }

    stage("the whole gate", gate_start);
    std::printf("\n--------------------------------------------------------------------------------\n");
    std::printf("  P4: %u checks, %u failures\n", harness.checks, harness.failures);
    std::printf("--------------------------------------------------------------------------------\n");
    return harness.failures == 0 ? 0 : 1;
}
