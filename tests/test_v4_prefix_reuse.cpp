// -----------------------------------------------------------------------------
// Gate — prefix reuse: does a continuation equal a from-scratch run?
//
// The engine keeps the model state of one live conversation resident, and each
// turn prefills only what the turn added. The requirement is *correctness first*:
// a conversation reached turn-by-turn must produce the same generated token ids
// as the same prompt prefilled in one shot, and a reuse that cannot be proven must
// replay rather than serve stale state.
//
// What the gate asserts, section by section:
//
//   A. R2 — a continuation's ids equal a from-scratch run's, at token granularity.
//      The from-scratch run shares no reuse code, so it is an independent
//      reference. The first-token distributions are compared as a *diagnostic*
//      (bit-equality and max abs diff), with the top-1/top-2 margin printed so a
//      near-tie is distinguishable from a state defect.
//   B. R1 — only the addition is prefilled: the counters say so. Wall time is
//      printed, never asserted.
//   C. R3 — a rejected reuse (diverged, no tail) replays, is correct, and leaves a
//      record the *next* turn can reuse again.
//   D. R4 — every non-token input invalidates, through the real encoder.
//   E. R5 — the context bound is an up-front refusal that leaves the session
//      intact.
//   F. R6 — deterministic and inspectable; a direct host reset is caught as stale.
//   G. Boundaries — a wrapped ring and a live compressor partial at the reuse
//      boundary.
//   H. Hygiene — the supply invariants hold after the run.
//
// The from-scratch run is the independent oracle. A mutation pass (skip a decode
// feed, feed the unfed last token, start the tail one token early, reuse after an
// epoch bump) must fail this gate.
//
// COST. The host assembly is ~8 s; it is built once and reused. Generations are
// kept short. The whole gate is a few minutes.
// -----------------------------------------------------------------------------

#include "platform/device.hpp"
#include "test_device.hpp"

#include "architecture/deepseek_v4/runtime/v4_engine.hpp"
#include "architecture/deepseek_v4/runtime/v4_graph.hpp"
#include "architecture/deepseek_v4/runtime/v4_model_host.hpp"
#include "architecture/deepseek_v4/text/dsv4_computation_key.hpp"
#include "infrastructure/sampler.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace {

using aeon::core::V4Engine;
using aeon::core::V4EngineOptions;
using aeon::core::V4Reply;
using aeon::core::SamplerConfig;
using aeon::session::ReuseVerdict;
using aeon::text::Dsv4PromptMessage;
using aeon::text::Dsv4PromptOptions;
using aeon::text::Dsv4Role;
using aeon::text::Dsv4ThinkingMode;
using aeon::text::Dsv4ToolDefinition;
using aeon::text::GenerationOptions;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

// The context the gate runs at. A thinking-mode turn with a declared tool renders
// a ~260-token system block, so 256 would refuse the R4 tools case; 512 fits every
// section and still wraps the real 128-token window inside it.
constexpr uint32_t kContext = 512;

// New tokens per generation for the correctness sections. More than one so a
// decode-position error is visible; small enough to stay cheap.
constexpr uint32_t kNew = 3;

// The key the token-level turns are built under. All turns of one session share
// it, so a mismatch would be a KeyChanged, which is not what these sections test.
const char* const kKey = "prefix-reuse-gate";

struct Harness {
    uint32_t checks{0};
    uint32_t failures{0};

    bool assert_that(const char* label, bool ok, const std::string& detail) {
        std::printf("  %-60s %-30s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
        ++checks;
        if (!ok) ++failures;
        return ok;
    }
};

Harness harness;

std::string ids_to_string(const std::vector<uint32_t>& ids, size_t limit = 10) {
    std::string out = "[";
    for (size_t i = 0; i < ids.size() && i < limit; ++i) {
        if (i) out += ", ";
        out += std::to_string(ids[i]);
    }
    if (ids.size() > limit) out += ", ...";
    return out + "]";
}

Dsv4PromptMessage user_message(const std::string& text) {
    Dsv4PromptMessage message;
    message.role = Dsv4Role::User;
    message.content = text;
    return message;
}

SamplerConfig greedy_config(const V4Engine& engine) {
    SamplerConfig sampling = engine.policy().to_sampler_config(engine.options().seed);
    sampling.temperature = 0.0f;
    sampling.top_p = 1.0f;
    return sampling;
}

GenerationOptions gen_options(uint32_t new_tokens) {
    GenerationOptions options;
    options.max_new_tokens = new_tokens;
    options.context_limit = kContext;
    // Greedy would stop on the first token for these prompts, which would leave the
    // decode path unexercised — and the decode path is where a continuation differs
    // from a from-scratch prefill. Emitting the full count makes the comparison a
    // real one.
    options.stop_on_eos = false;
    return options;
}

// The head's logits after the last forward the engine ran, read back to host fp16.
std::vector<double> copy_logits(V4Engine& engine) {
    const uint32_t vocab = engine.sampler().vocab();
    std::vector<half> host_logits(vocab);
    CHECK_HIP(hipMemcpyAsync(host_logits.data(), engine.graph().logits(),
                             static_cast<size_t>(vocab) * sizeof(half),
                             hipMemcpyDeviceToHost, engine.host().streams().compute));
    CHECK_HIP(hipStreamSynchronize(engine.host().streams().compute));
    std::vector<double> out(vocab);
    for (uint32_t i = 0; i < vocab; ++i) out[i] = static_cast<double>(host_logits[i]);
    return out;
}

// The top-1/top-2 gap in one position's logits, so a near-tie can be told from a
// real divergence.
void print_margin(const std::vector<double>& logits) {
    double best = -1e30, second = -1e30;
    uint32_t best_id = 0, second_id = 0;
    for (uint32_t i = 0; i < logits.size(); ++i) {
        if (logits[i] > best) {
            second = best; second_id = best_id;
            best = logits[i]; best_id = i;
        } else if (logits[i] > second) {
            second = logits[i]; second_id = i;
        }
    }
    std::printf("  [note] top-1 id %u (%.4f), top-2 id %u (%.4f), margin %.4f\n",
                best_id, best, second_id, second, best - second);
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  prefix reuse: a continuation vs a from-scratch run\n");
    std::printf("================================================================================\n");
    aeon::test::select_test_device(true);

    V4EngineOptions engine_options;
    engine_options.model_dir = kModelDir;
    engine_options.runtime.context_size = kContext;
    // The swept prefill streams whole layers and is the subject of its own gate;
    // this gate is about the reuse decision, so it runs with the sweep off.
    engine_options.runtime.prefill_sweep = false;
    engine_options.seed = 1234;

    V4Engine engine;
    engine.initialize(engine_options);

    const SamplerConfig greedy = greedy_config(engine);

    Dsv4PromptOptions base_options;
    base_options.thinking_mode = Dsv4ThinkingMode::Chat;

    const std::string p1_text = "What is the capital of France?";
    const std::vector<uint32_t> P1 =
        engine.encoder().encode_tokens({user_message(p1_text)}, base_options);
    const std::vector<uint32_t> suffix = engine.tokenizer().encode(" Why is that?");

    auto invariants = [&](const char* label) {
        const bool ok = engine.host().registry().invariants_hold() &&
                        engine.host().outstanding_expert_leases() == 0 &&
                        engine.host().staging_in_use_slots() == 0;
        harness.assert_that(label, ok,
                            ok ? "clean" : "registry/lease/staging invariant broken");
    };

    // =========================================================================
    // A + B. R2 correctness, and R1 counters
    // =========================================================================
    std::printf("\n[A] R2 — a continuation equals a from-scratch run\n");
    engine.end_session();
    const V4Reply r1 = engine.generate(P1, gen_options(kNew), greedy, true, kKey);
    std::vector<uint32_t> P2 = P1;
    P2.insert(P2.end(), r1.token_ids.begin(), r1.token_ids.end());
    P2.insert(P2.end(), suffix.begin(), suffix.end());
    // The last generated token is never fed, so the resident state covers
    // `P1 ++ R1[:-1]`.
    const uint32_t record_size =
        static_cast<uint32_t>(P1.size()) + static_cast<uint32_t>(r1.token_ids.size()) - 1;

    const V4Reply cont = engine.generate(P2, gen_options(kNew), greedy, true, kKey);
    const V4Reply cold = engine.generate(P2, gen_options(kNew), greedy, false, kKey);

    harness.assert_that("A: continuation ids == from-scratch ids",
                        cont.token_ids == cold.token_ids,
                        ids_to_string(cont.token_ids) + " vs " + ids_to_string(cold.token_ids));
    if (cont.token_ids != cold.token_ids) {
        size_t divergence = 0;
        while (divergence < cont.token_ids.size() && divergence < cold.token_ids.size() &&
               cont.token_ids[divergence] == cold.token_ids[divergence]) {
            ++divergence;
        }
        std::printf("  [note] first divergent token index %zu\n", divergence);
    }

    // The distribution diagnostic: both runs at one new token, the logits read
    // back and compared. Bit-equality is printed, never asserted (R2 is stated at
    // token granularity).
    {
        engine.end_session();
        (void)engine.generate(P1, gen_options(kNew), greedy, true, kKey);
        const V4Reply warm1 = engine.generate(P2, gen_options(1), greedy, true, kKey);
        const std::vector<double> warm_logits = copy_logits(engine);
        const V4Reply cold1 = engine.generate(P2, gen_options(1), greedy, false, kKey);
        const std::vector<double> cold_logits = copy_logits(engine);
        uint32_t differing = 0;
        double max_abs = 0.0;
        for (size_t i = 0; i < warm_logits.size(); ++i) {
            if (warm_logits[i] != cold_logits[i]) ++differing;
            max_abs = std::max(max_abs, std::fabs(warm_logits[i] - cold_logits[i]));
        }
        std::printf("  [note] first-token logits: %u of %zu bits differ, max abs %.5f\n",
                    differing, warm_logits.size(), max_abs);
        if (warm1.token_ids != cold1.token_ids) print_margin(warm_logits);
    }

    std::printf("\n[B] R1 — only the addition is prefilled\n");
    harness.assert_that("B: the continuation did not replay",
                        cont.reuse_verdict == ReuseVerdict::Reused,
                        aeon::session::PrefixRecord::verdict_name(cont.reuse_verdict));
    harness.assert_that("B: reused == the resident state's size",
                        cont.reused_tokens == record_size,
                        std::to_string(cont.reused_tokens) + " == " +
                            std::to_string(record_size));
    harness.assert_that("B: reused + prefilled == prompt size",
                        cont.reused_tokens + cont.prefilled_tokens == P2.size(),
                        std::to_string(cont.reused_tokens) + " + " +
                            std::to_string(cont.prefilled_tokens) + " == " +
                            std::to_string(P2.size()));
    std::printf("  [note] TTFT: continuation %.1f ms, from-scratch %.1f ms\n",
                cont.ttft_ms, cold.ttft_ms);
    invariants("H: invariants after A/B");

    // =========================================================================
    // C. R3 — a rejected reuse is correct and not stale
    // =========================================================================
    std::printf("\n[C] R3 — a rejected reuse replays correctly\n");
    {
        const std::function<void(const char*, std::function<std::vector<uint32_t>(
            const std::vector<uint32_t>&)>)> rejected =
            [&](const char* label,
                std::function<std::vector<uint32_t>(const std::vector<uint32_t>&)> make_prompt) {
                engine.end_session();
                const V4Reply rr = engine.generate(P1, gen_options(kNew), greedy, true, kKey);
                std::vector<uint32_t> record = P1;
                record.insert(record.end(), rr.token_ids.begin(), rr.token_ids.end() - 1);

                const std::vector<uint32_t> prompt = make_prompt(record);
                const V4Reply warm = engine.generate(prompt, gen_options(1), greedy, true, kKey);
                const V4Reply cold_run = engine.generate(prompt, gen_options(1), greedy, false, kKey);

                std::vector<uint32_t> extension = prompt;
                extension.push_back(cold_run.token_ids.front());
                const V4Reply follow =
                    engine.generate(extension, gen_options(1), greedy, true, kKey);

                const bool rejected_as_expected =
                    warm.reuse_verdict == ReuseVerdict::Diverged ||
                    warm.reuse_verdict == ReuseVerdict::NoTail;
                const bool ok = rejected_as_expected && warm.reused_tokens == 0 &&
                                warm.token_ids == cold_run.token_ids &&
                                follow.reuse_verdict == ReuseVerdict::Reused;
                char detail[160];
                std::snprintf(detail, sizeof(detail), "%s, reused=%u, follow=%s",
                              aeon::session::PrefixRecord::verdict_name(warm.reuse_verdict),
                              warm.reused_tokens,
                              aeon::session::PrefixRecord::verdict_name(follow.reuse_verdict));
                harness.assert_that(label, ok, detail);
            };

        rejected("C: a changed id mid-prefix replays", [](const std::vector<uint32_t>& record) {
            std::vector<uint32_t> prompt = record;
            prompt[prompt.size() / 2] = prompt[prompt.size() / 2] + 1u;
            prompt.push_back(200u);  // longer than the record, so this is a divergence
            return prompt;
        });
        rejected("C: a changed id at position 0 replays", [](const std::vector<uint32_t>& record) {
            std::vector<uint32_t> prompt = record;
            prompt.front() = prompt.front() + 1u;
            prompt.push_back(200u);
            return prompt;
        });
        rejected("C: a changed last recorded id replays",
                 [](const std::vector<uint32_t>& record) {
                     std::vector<uint32_t> prompt = record;
                     prompt.back() = prompt.back() + 1u;
                     prompt.push_back(200u);
                     return prompt;
                 });
        rejected("C: a prompt equal to the record is NoTail",
                 [](const std::vector<uint32_t>& record) { return record; });
        rejected("C: a prompt shorter than the record is NoTail",
                 [](const std::vector<uint32_t>& record) {
                     std::vector<uint32_t> prompt(record.begin(), record.end() - 1);
                     return prompt;
                 });
    }
    invariants("H: invariants after C");

    // =========================================================================
    // D. R4 — every non-token input invalidates
    // =========================================================================
    std::printf("\n[D] R4 — non-token inputs invalidate reuse\n");
    {
        Dsv4PromptMessage system_message;
        system_message.role = Dsv4Role::System;
        system_message.content = "";

        Dsv4PromptMessage user = user_message("Hi");
        const std::vector<Dsv4PromptMessage> base_messages = {system_message, user};
        Dsv4PromptOptions d_base;
        d_base.thinking_mode = Dsv4ThinkingMode::Thinking;
        d_base.drop_thinking = false;  // so a flip is meaningful

        const std::string base_key = aeon::text::dsv4_computation_key(base_messages, d_base);
        const std::vector<uint32_t> base_tokens =
            engine.encoder().encode_tokens(base_messages, d_base);

        const auto check_flip = [&](const char* label,
                                    const std::vector<Dsv4PromptMessage>& messages,
                                    const Dsv4PromptOptions& options) {
            engine.end_session();
            (void)engine.chat(base_messages, d_base, gen_options(1), greedy, true);

            const V4Reply warm = engine.chat(messages, options, gen_options(1), greedy, true);
            const V4Reply cold_run = engine.chat(messages, options, gen_options(1), greedy, false);

            const bool key_caught =
                aeon::text::dsv4_computation_key(messages, options) != base_key;
            const bool tokens_caught = engine.encoder().encode_tokens(messages, options) != base_tokens;

            const bool ok = warm.reuse_verdict != ReuseVerdict::Reused &&
                            warm.token_ids == cold_run.token_ids;
            char detail[192];
            std::snprintf(detail, sizeof(detail), "%s; key=%s tokens=%s",
                          aeon::session::PrefixRecord::verdict_name(warm.reuse_verdict),
                          key_caught ? "caught" : "same",
                          tokens_caught ? "caught" : "same");
            harness.assert_that(label, ok, detail);
        };

        {
            Dsv4PromptOptions options = d_base;
            options.thinking_mode = Dsv4ThinkingMode::Chat;
            check_flip("D: thinking_mode change invalidates", base_messages, options);
        }
        {
            Dsv4PromptOptions options = d_base;
            options.drop_thinking = true;
            check_flip("D: drop_thinking change invalidates", base_messages, options);
        }
        {
            Dsv4PromptOptions options = d_base;
            options.reasoning_effort = "high";
            check_flip("D: reasoning_effort change invalidates", base_messages, options);
        }
        {
            std::vector<Dsv4PromptMessage> messages = base_messages;
            Dsv4ToolDefinition tool;
            tool.function_json =
                "{\"name\": \"get_weather\", \"description\": \"d\", \"parameters\": {}}";
            messages.front().tools.push_back(std::move(tool));
            check_flip("D: declared tools invalidate", messages, d_base);
        }
        {
            std::vector<Dsv4PromptMessage> messages = base_messages;
            messages.front().response_format_json = "{\"type\": \"json_object\"}";
            check_flip("D: response_format change invalidates", messages, d_base);
        }
    }
    invariants("H: invariants after D");

    // =========================================================================
    // E. R5 — the bound is an up-front refusal; the session survives
    // =========================================================================
    std::printf("\n[E] R5 — the context bound is refused up front\n");
    {
        engine.end_session();
        (void)engine.generate(P1, gen_options(1), greedy, true, kKey);

        const std::vector<uint32_t> too_long(kContext + 1, 5u);
        bool refused = false;
        try {
            (void)engine.generate(too_long, gen_options(1), greedy, true, kKey);
        } catch (const std::exception&) {
            refused = true;
        }
        harness.assert_that("E: an over-capacity prompt is refused", refused, "threw");

        std::vector<uint32_t> extension = P1;
        extension.push_back(11u);
        const V4Reply follow = engine.generate(extension, gen_options(1), greedy, true, kKey);
        harness.assert_that("E: the session survived the refusal",
                            follow.reuse_verdict == ReuseVerdict::Reused,
                            aeon::session::PrefixRecord::verdict_name(follow.reuse_verdict));
    }

    // =========================================================================
    // F. R6 — deterministic and inspectable
    // =========================================================================
    std::printf("\n[F] R6 — deterministic and inspectable\n");
    {
        const auto sequence = [&]() {
            engine.end_session();
            const V4Reply a = engine.generate(P1, gen_options(kNew), greedy, true, kKey);
            std::vector<uint32_t> prompt = P1;
            prompt.insert(prompt.end(), a.token_ids.begin(), a.token_ids.end());
            prompt.insert(prompt.end(), suffix.begin(), suffix.end());
            return engine.generate(prompt, gen_options(kNew), greedy, true, kKey).token_ids;
        };
        const std::vector<uint32_t> first = sequence();
        const std::vector<uint32_t> second = sequence();
        harness.assert_that("F: the A+B sequence is reproducible", first == second,
                            ids_to_string(first));
    }
    {
        engine.end_session();
        (void)engine.generate(P1, gen_options(1), greedy, true, kKey);
        engine.host().reset_generation_state();  // a direct reset, outside the contract
        std::vector<uint32_t> extension = P1;
        extension.push_back(13u);
        const V4Reply stale = engine.generate(extension, gen_options(1), greedy, true, kKey);
        const V4Reply cold_run = engine.generate(extension, gen_options(1), greedy, false, kKey);
        harness.assert_that("F: a direct host reset is caught as StaleState",
                            stale.reuse_verdict == ReuseVerdict::StaleState &&
                                stale.reused_tokens == 0 &&
                                stale.token_ids == cold_run.token_ids,
                            aeon::session::PrefixRecord::verdict_name(stale.reuse_verdict));
    }
    invariants("H: invariants after F");

    // =========================================================================
    // G. Boundaries — a wrapped ring and a live compressor partial
    // =========================================================================
    std::printf("\n[G] boundary — wrapped ring and live compressor partial\n");
    {
        std::string long_text;
        for (int i = 0; i < 40; ++i) long_text += "The quick brown fox jumps over the lazy dog. ";
        std::vector<uint32_t> big =
            engine.encoder().encode_tokens({user_message(long_text)}, base_options);
        while (big.size() > 200) big.pop_back();
        while (big.size() < 140) big.push_back(100u);
        // The fed length must not be a multiple of 4 (the ratio-4 compressor
        // period) or of 128 (the ring), so the partial state is live at the reuse
        // boundary.
        while (((big.size() + kNew - 1) % 4 == 0) || ((big.size() + kNew - 1) % 128 == 0)) {
            big.push_back(100u);
        }

        engine.end_session();
        const V4Reply b1 = engine.generate(big, gen_options(kNew), greedy, true, kKey);
        std::vector<uint32_t> b2 = big;
        b2.insert(b2.end(), b1.token_ids.begin(), b1.token_ids.end());
        b2.insert(b2.end(), suffix.begin(), suffix.end());

        const V4Reply bc = engine.generate(b2, gen_options(kNew), greedy, true, kKey);
        const V4Reply bcc = engine.generate(b2, gen_options(kNew), greedy, false, kKey);
        harness.assert_that("G: reuse holds across the wrapped ring",
                            bc.reuse_verdict == ReuseVerdict::Reused &&
                                bc.token_ids == bcc.token_ids,
                            aeon::session::PrefixRecord::verdict_name(bc.reuse_verdict));
        std::printf("  [note] |P1| = %zu, record fed length = %zu\n", big.size(),
                    static_cast<size_t>(big.size()) + kNew - 1);
    }

    // =========================================================================
    // I. Exact-token history through the encoder — the shippable path
    // =========================================================================
    std::printf("\n[I] exact-token history through the encoder\n");
    {
        Dsv4PromptOptions chat;
        chat.thinking_mode = Dsv4ThinkingMode::Chat;

        // Turn 1 generated from a text-built prompt.
        engine.end_session();
        const std::vector<uint32_t> prompt1 =
            engine.encoder().encode_tokens({user_message(p1_text)}, chat);
        const V4Reply r1i = engine.generate(prompt1, gen_options(kNew), greedy, true, kKey);

        // Turn 2 builds the assistant body from the ids the engine produced (minus
        // the trailing EOS the template re-adds) instead of the reply text. This is
        // what makes the prefix exact rather than a BPE round-trip.
        Dsv4PromptMessage assistant;
        assistant.role = Dsv4Role::Assistant;
        assistant.content = r1i.text;  // readable only; `encode_tokens` uses the ids
        assistant.preencoded_ids = r1i.token_ids;
        if (!assistant.preencoded_ids.empty() &&
            assistant.preencoded_ids.back() == engine.tokenizer().eos_token_id()) {
            assistant.preencoded_ids.pop_back();
        }

        const std::vector<uint32_t> prompt2 = engine.encoder().encode_tokens(
            {user_message(p1_text), assistant, user_message("Why is that?")}, chat);

        const V4Reply warm = engine.generate(prompt2, gen_options(kNew), greedy, true, kKey);
        const V4Reply cold_run = engine.generate(prompt2, gen_options(kNew), greedy, false, kKey);

        harness.assert_that("I: id-history continuation reuses",
                            warm.reuse_verdict == ReuseVerdict::Reused,
                            aeon::session::PrefixRecord::verdict_name(warm.reuse_verdict));
        harness.assert_that("I: id-history continuation ids == cold",
                            warm.token_ids == cold_run.token_ids,
                            ids_to_string(warm.token_ids));
        std::printf("  [note] reused=%u prefilled=%u prompt=%u\n",
                    warm.reused_tokens, warm.prefilled_tokens, warm.prompt_tokens);
    }

    // =========================================================================
    // H. Hygiene
    // =========================================================================
    std::printf("\n[H] hygiene\n");
    invariants("H: registry, leases and staging are clean");

    std::printf("\n================================================================================\n");
    std::printf("  %u checks, %u failures\n", harness.checks, harness.failures);
    std::printf("================================================================================\n");
    return harness.failures == 0 ? 0 : 1;
}
