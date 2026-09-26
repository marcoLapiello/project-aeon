// -----------------------------------------------------------------------------
// aeon_chat — text in, text out, driven by the rewritten graph.
//
// This was a **legacy** target until P4: it drove `V4Pipeline`, the pre-rewrite
// graph (since deleted). The composition plan lists re-binding it as agreement
// **G5** and part of phase **P4**, and the reason
// is the acceptance criterion — "one command takes a conversation and returns
// text" — which cannot be met by a binary that is not in the default build.
//
// The CLI is now thin on purpose. It does four things the engine should not: parse
// arguments, assemble messages, choose what to print, and exit. Everything else —
// rendering the conversation, building the model, running the forward pass,
// sampling, detokenizing — is `V4Engine`'s, and this file names none of it.
//
// Two flags are accepted and deliberately inert, named here so their inertness is
// a documented fact rather than a surprise:
//
//   * `--deterministic-experts` — the **rewrite always** accumulates the routed
//     experts in the fixed slot order with a single fp32 rounding (plan §8, "one
//     accumulation"); the pre-rewrite graph had a selectable atomic path and that
//     is what the flag used to choose. It therefore changes nothing here, and a
//     green run cannot be produced by turning it on.
//
// The supply-telemetry flags (`--supply-telemetry`, `--run-id`) are **restored**
// here: the sink is wired into the host (`V4ModelHost::initialize` opens it,
// `free` flushes it) and the engine labels each dispatch's phase. Only
// supply-mediated traffic is captured — the Hot and Warm startup preloads read
// directly, so the Warmup phase is empty by construction.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_engine.hpp"
#include "platform/rdna3/device.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model_dir{"models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon"};
    std::string tokenizer_path;
    std::string system_prompt;
    std::string prompt;
    uint32_t context_size{4096};
    uint32_t max_new_tokens{256};
    uint64_t warm_gib{0};
    bool preload_warm_host{true};
    bool enable_warm_refill{true};
    bool deterministic_experts{false};  // accepted; see the header note
    bool thinking_mode{false};
    bool until_eos{false};
    bool diagnostic{false};
    bool verbose{false};
    bool greedy{false};
    bool has_temperature{false};
    float temperature{1.0f};
    float top_p{1.0f};
    uint64_t seed{0};
    std::string supply_telemetry_path;
    std::string run_id{"unnamed"};
    uint32_t max_hot_slots{0};
    std::string dump_logits_path;
    uint64_t demotion_queue{0};
    bool profile_routing{false};
    bool validate_registry{false};
    uint32_t prefill_window{4096};
    uint32_t prefill_chunk{64};
    uint32_t prefill_sweep_min_tokens{0};
    uint32_t staging_blocks{2};
};

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " --prompt <text> [options]\n"
        << "Options:\n"
        << "  --model-dir <path>       Native Aeon model directory\n"
        << "  --tokenizer <path>       Native tokenizer artifact (default: <model-dir>/tokenizer.aeon)\n"
        << "  --system <text>          Optional system message\n"
        << "  --prompt <text>          One user prompt (required)\n"
        << "  --thinking               Use explicit DSV4 thinking mode\n"
        << "  --max-new-tokens <n>     Maximum generated tokens (default: 256)\n"
        << "  --until-eos              Generate until EOS or context capacity\n"
        << "  --context-size <n>       Context capacity in tokens (default: 4096)\n"
        << "  --greedy                 Take the argmax instead of sampling\n"
        << "  --temperature <value>    Sampling temperature (default: the artifact's own)\n"
        << "  --top-p <value>          Nucleus threshold in (0, 1] (default: the artifact's own)\n"
        << "  --seed <n>               Sampling seed (default: 0)\n"
        << "  --warm-gib <n>           Warm host allocation in GiB (default: 0)\n"
        << "  --supply-telemetry <p>   Write supply telemetry JSONL to path <p> (default: off)\n"
        << "  --run-id <id>            Run identifier stamped on telemetry rows (default: unnamed)\n"
        << "  --max-hot-slots <n>      Cap the Hot VRAM expert pool at <n> slots (0 = derived)\n"
        << "  --dump-logits <path>     Append each position's raw fp16 logits to <path>\n"
        << "  --demotion-queue <n>     Demotion-queue capacity (0 = derived from warm refill)\n"
        << "  --prefill-window <n>     Layer-major prefill window W in tokens (0 = the whole prompt)\n"
        << "  --prefill-chunk <n>      Body chunk C in tokens, 1..64 (default: 64)\n"
        << "  --staging-blocks <n>     Swept-prefill staging arena in layer-blocks, 1.. (default: 2).\n"
        << "                           A memory budget: each block is E x payload bytes of pinned\n"
        << "                           host RAM (3.44 GiB at E=256), and the prefetch depth is\n"
        << "                           derived from it. Does not change throughput.\n"
        << "  --prefill-sweep-min-tokens <n>\n"
        << "                           Prompt length at or above which the sweep supplies\n"
        << "                           experts; below it, the routed cache (0 = 3E/4)\n"
        << "  --profile-routing        Print the decode routing reuse-distance (ideal-LRU) curve\n"
        << "  --validate-registry      Audit the registry after every expert operation (slow; debug)\n"
        << "  --no-warm-preload        Allocate Warm capacity without startup payload reads\n"
        << "  --no-warm-refill         Disable asynchronous Hot-to-Warm refill\n"
        << "  --deterministic-experts  Accepted and inert; the rewrite always uses the fixed-order\n"
        << "                           fp32 accumulator (plan section 8, \"one accumulation\")\n"
        << "  --verbose                Print the memory budget report and the residency summary\n"
        << "  --diagnostic             Print the rendered prompt, token ids, stop reason and timings\n"
        << "  --help                   Show this help\n";
}

uint64_t parse_unsigned(const std::string& value, const char* option) {
    size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &consumed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
    }
    if (consumed != value.size()) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
    }
    return static_cast<uint64_t>(parsed);
}

float parse_float(const std::string& value, const char* option) {
    size_t consumed = 0;
    float parsed = 0.0f;
    try {
        parsed = std::stof(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
    }
    if (consumed != value.size()) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
    }
    return parsed;
}

std::string require_value(int argc, char** argv, int& index, const char* option) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + option);
    }
    ++index;
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--model-dir") {
            options.model_dir = require_value(argc, argv, index, "--model-dir");
        } else if (argument == "--tokenizer") {
            options.tokenizer_path = require_value(argc, argv, index, "--tokenizer");
        } else if (argument == "--system") {
            options.system_prompt = require_value(argc, argv, index, "--system");
        } else if (argument == "--prompt") {
            options.prompt = require_value(argc, argv, index, "--prompt");
        } else if (argument == "--thinking") {
            options.thinking_mode = true;
        } else if (argument == "--until-eos") {
            options.until_eos = true;
        } else if (argument == "--greedy") {
            options.greedy = true;
        } else if (argument == "--max-new-tokens") {
            options.max_new_tokens = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--max-new-tokens"), "--max-new-tokens"));
        } else if (argument == "--context-size") {
            options.context_size = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--context-size"), "--context-size"));
        } else if (argument == "--warm-gib") {
            options.warm_gib = parse_unsigned(
                require_value(argc, argv, index, "--warm-gib"), "--warm-gib");
        } else if (argument == "--temperature") {
            options.temperature = parse_float(
                require_value(argc, argv, index, "--temperature"), "--temperature");
            options.has_temperature = true;
        } else if (argument == "--top-p") {
            options.top_p = parse_float(require_value(argc, argv, index, "--top-p"), "--top-p");
        } else if (argument == "--seed") {
            options.seed = parse_unsigned(require_value(argc, argv, index, "--seed"), "--seed");
        } else if (argument == "--supply-telemetry") {
            options.supply_telemetry_path =
                require_value(argc, argv, index, "--supply-telemetry");
        } else if (argument == "--run-id") {
            options.run_id = require_value(argc, argv, index, "--run-id");
        } else if (argument == "--max-hot-slots") {
            options.max_hot_slots = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--max-hot-slots"), "--max-hot-slots"));
        } else if (argument == "--dump-logits") {
            options.dump_logits_path = require_value(argc, argv, index, "--dump-logits");
        } else if (argument == "--demotion-queue") {
            options.demotion_queue = parse_unsigned(
                require_value(argc, argv, index, "--demotion-queue"), "--demotion-queue");
        } else if (argument == "--prefill-window") {
            options.prefill_window = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--prefill-window"), "--prefill-window"));
        } else if (argument == "--prefill-chunk") {
            options.prefill_chunk = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--prefill-chunk"), "--prefill-chunk"));
        } else if (argument == "--prefill-sweep-min-tokens") {
            options.prefill_sweep_min_tokens = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--prefill-sweep-min-tokens"),
                "--prefill-sweep-min-tokens"));
        } else if (argument == "--staging-blocks") {
            options.staging_blocks = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--staging-blocks"), "--staging-blocks"));
        } else if (argument == "--profile-routing") {
            options.profile_routing = true;
        } else if (argument == "--validate-registry") {
            options.validate_registry = true;
        } else if (argument == "--no-warm-preload") {
            options.preload_warm_host = false;
        } else if (argument == "--no-warm-refill") {
            options.enable_warm_refill = false;
        } else if (argument == "--deterministic-experts") {
            options.deterministic_experts = true;
        } else if (argument == "--verbose") {
            options.verbose = true;
        } else if (argument == "--diagnostic") {
            options.diagnostic = true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }

    if (options.prompt.empty()) {
        throw std::runtime_error("--prompt is required and must not be empty");
    }
    if (options.until_eos && options.max_new_tokens != 256) {
        throw std::runtime_error("--until-eos cannot be combined with --max-new-tokens");
    }
    if (options.context_size == 0 ||
        (!options.until_eos && options.max_new_tokens == 0)) {
        throw std::runtime_error(
            "context-size must be positive; max-new-tokens must be positive unless --until-eos");
    }
    if (options.tokenizer_path.empty()) {
        options.tokenizer_path =
            (std::filesystem::path(options.model_dir) / "tokenizer.aeon").string();
    }
    return options;
}

// The [Invariants] line is unconditional, not gated behind `--diagnostic`: the
// gate scripts grep it, and a run that leaked a lease or broke a registry
// invariant is not a valid measurement. Its format is therefore stable.
void print_invariants(const aeon::core::V4Engine& engine) {
    std::cout << "[Invariants] registry.invariants_hold="
              << (engine.host().registry().invariants_hold() ? "true" : "false")
              << " outstanding_leases=" << engine.host().outstanding_expert_leases()
              << " forced_drains=" << engine.host().forced_drains()
              << " staging_in_use=" << engine.host().staging_in_use_slots()
              << " demotion_queue=" << engine.host().demotion_queue_capacity()
              << "\n";
}

// Per-layer outcome distribution: the fraction of layer dispatches answered
// entirely from Hot, from Hot+Warm with no Cold, and with at least one Cold. A
// layer waits on its slowest of six fetches, so this distribution — not the cold
// request rate — is what a flat throughput under a shrinking miss rate reflects.
void print_layer_outcomes(const aeon::core::V4Engine& engine) {
    for (const bool prefill : {false, true}) {
        const uint64_t total = engine.host().layer_dispatches(prefill);
        if (total == 0) continue;
        const auto pct = [total](uint64_t part) {
            return 100.0 * static_cast<double>(part) / static_cast<double>(total);
        };
        std::cout << "[Layer outcomes] phase=" << (prefill ? "prefill" : "decode")
                  << " dispatches=" << total
                  << std::fixed << std::setprecision(1)
                  << " all_hot=" << pct(engine.host().layer_outcome_count(prefill, 0)) << "%"
                  << " warm_no_cold=" << pct(engine.host().layer_outcome_count(prefill, 1)) << "%"
                  << " has_cold=" << pct(engine.host().layer_outcome_count(prefill, 2)) << "%\n";
    }
}

// Decode routing reuse-distance curve (routing study Phase 1): the ideal-LRU hit
// rate at each capacity next to the measured Hot hit rate and the Belady-OPT
// bound. A gap between measured and ideal-LRU means the recency policy is leaving
// locality uncaptured; the ideal-LRU-to-OPT gap is the headroom only a non-recency
// policy could reach.
void print_routing_reuse(const aeon::core::V4Engine& engine) {
    if (!engine.host().routing_reuse_enabled()) return;
    const auto curve = engine.host().routing_reuse_curve();
    std::cout << "[Routing reuse] decode requests=" << curve.observed
              << " compulsory=" << curve.compulsory
              << " measured_hot_hit=" << std::fixed << std::setprecision(1)
              << curve.measured_hit_rate * 100.0 << "%\n";
    for (size_t i = 0; i < curve.capacities.size(); ++i) {
        std::cout << "    capacity=" << curve.capacities[i]
                  << " ideal_lru_hit=" << curve.ideal_lru_hit_rate[i] * 100.0 << "%"
                  << " opt_hit=" << curve.opt_hit_rate[i] * 100.0 << "%\n";
    }
}

std::vector<aeon::text::Dsv4PromptMessage> build_messages(const Options& options) {
    std::vector<aeon::text::Dsv4PromptMessage> messages;
    if (!options.system_prompt.empty()) {
        aeon::text::Dsv4PromptMessage system_message;
        system_message.role = aeon::text::Dsv4Role::System;
        system_message.content = options.system_prompt;
        messages.push_back(std::move(system_message));
    }
    aeon::text::Dsv4PromptMessage user_message;
    user_message.role = aeon::text::Dsv4Role::User;
    user_message.content = options.prompt;
    messages.push_back(std::move(user_message));
    return messages;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        aeon::core::select_compute_device(true);

        aeon::core::V4EngineOptions engine_options;
        engine_options.model_dir = options.model_dir;
        engine_options.tokenizer_path = options.tokenizer_path;
        engine_options.seed = options.seed;
        engine_options.verbose = options.verbose;
        engine_options.runtime.context_size = options.context_size;
        engine_options.runtime.warm_host_bytes =
            options.warm_gib * 1024ULL * 1024ULL * 1024ULL;
        engine_options.runtime.preload_warm_host = options.preload_warm_host;
        engine_options.runtime.enable_warm_refill = options.enable_warm_refill;
        engine_options.runtime.supply_telemetry_path = options.supply_telemetry_path;
        engine_options.runtime.run_id = options.run_id;
        engine_options.runtime.max_hot_vram_slots = options.max_hot_slots;
        engine_options.dump_logits_path = options.dump_logits_path;
        engine_options.runtime.demotion_queue_capacity = options.demotion_queue;
        engine_options.runtime.profile_routing_reuse = options.profile_routing;
        engine_options.runtime.validate_registry_each_request = options.validate_registry;
        engine_options.runtime.prefill_window = options.prefill_window;
        engine_options.runtime.prefill_chunk = options.prefill_chunk;
        engine_options.runtime.prefill_sweep_min_tokens = options.prefill_sweep_min_tokens;
        engine_options.runtime.prefill_sweep_staging_blocks = options.staging_blocks;

        aeon::core::V4Engine engine;
        engine.initialize(engine_options);

        // The context window is the flag most worth tuning by hand, and its cost is
        // the Hot expert pool: attention state is a per-layer VRAM charge, so a
        // larger context leaves fewer Hot slots and sends more expert fetches to
        // Cold. The budget report is the authority on where that trade sits, so it
        // is printed verbatim rather than summarised.
        if (options.verbose) {
            std::cout << engine.host().budget().to_string();
            // The prefill workspace is derived from `--prefill-window` and
            // `--prefill-chunk` and allocated at load, so its three buffers are
            // reported together: the two VRAM ones (carry, batch scratch) and the
            // pinned host staging arena, which are separate budgets and must never
            // be read as one figure.
            std::cout << "[Prefill workspace] window W=" << engine.host().prefill_window_tokens()
                      << " chunk C=" << engine.host().prefill_chunk_tokens()
                      << " carry=" << (engine.host().prefill_carry_bytes() / (1024 * 1024))
                      << " MiB batch_scratch="
                      << (engine.host().prefill_batch_scratch_bytes() / (1024 * 1024))
                      << " MiB decode_scratch="
                      << (engine.host().decode_scratch_bytes() / (1024 * 1024))
                      << " MiB pinned_staging=" << (engine.host().staging_bytes() / (1024 * 1024))
                      << " MiB\n";
            // A cap is worth reporting explicitly, because the requested value and
            // the applied one differ whenever the cap is below 6 (floored) or above
            // the derived size (no effect) — the gate reads the applied figure.
            if (options.max_hot_slots > 0) {
                std::cout << "[Hot cap] requested " << options.max_hot_slots
                          << " slots, applied " << engine.host().budget().hot_vram_slots
                          << " slots\n";
            }
        }

        aeon::text::Dsv4PromptOptions prompt_options;
        prompt_options.thinking_mode = options.thinking_mode
            ? aeon::text::Dsv4ThinkingMode::Thinking
            : aeon::text::Dsv4ThinkingMode::Chat;

        // The artifact's own policy unless the caller overrode a field of it.
        aeon::core::V4SamplerConfig sampling =
            engine.policy().to_sampler_config(options.seed);
        if (options.greedy) {
            sampling.temperature = 0.0f;
        } else if (options.has_temperature) {
            sampling.temperature = options.temperature;
        }
        sampling.top_p = options.top_p;

        aeon::text::GenerationOptions generation_options;
        if (options.until_eos) {
            // The context bound is what stops the run when EOS never arrives; the
            // engine refuses a prompt that leaves no room at all.
            generation_options.max_new_tokens = options.context_size;
        } else {
            generation_options.max_new_tokens = options.max_new_tokens;
        }
        generation_options.eos_token_id = engine.tokenizer().eos_token_id();
        generation_options.context_limit = options.context_size;
        generation_options.thinking_mode = options.thinking_mode;

        const auto messages = build_messages(options);
        const aeon::core::V4Reply reply =
            engine.chat(messages, prompt_options, generation_options, sampling);

        const std::string visible =
            aeon::core::V4Engine::strip_thinking(reply.text, engine.tokenizer());

        print_invariants(engine);

        // The sweep's work, **after** the run: it reads zero before one. The window is
        // a *bound*, so `layer_loads / layers` is the number of layer-major passes the
        // run actually took (greater than one once the prompt exceeds `W`), and
        // `experts_streamed` against `layers x experts_per_layer` is whether each pass
        // really loaded whole layers. Reported on the verbose path, next to the
        // headline timings, because it is what explains them.
        if (options.verbose) {
            const auto& sweep = engine.host().prefill_sweep();
            const uint32_t layers = engine.host().num_layers();
            const uint32_t per_layer = engine.host().registry().experts_per_layer;
            std::cout << "[Prefill sweep] layer_loads=" << sweep.layer_loads()
                      << " experts_streamed=" << sweep.experts_streamed()
                      << " layers=" << layers
                      << " experts_per_layer=" << per_layer
                      << " passes=" << (layers ? sweep.layer_loads() / layers : 0)
                      << " lookahead=" << engine.host().sweep_lookahead_depth()
                      << " load_ms=" << (engine.host().sweep_load_ns() / 1000000)
                      << " io_ms=" << (engine.host().sweep_io_ns() / 1000000)
                      << " submit_ms=" << (engine.host().direct_io_submit_ns() / 1000000)
                      << " sqes=" << engine.host().direct_io_requests_submitted()
                      << " nvme_gib="
                      << (engine.host().supply_bytes_from_nvme() / (1024.0 * 1024 * 1024))
                      << "\n";
        }

        if (options.diagnostic) {
            print_routing_reuse(engine);
            print_layer_outcomes(engine);
            std::cout << "Rendered prompt: "
                      << engine.encoder().encode(messages, prompt_options) << "\n";
            std::cout << "Prompt tokens: " << reply.prompt_tokens << "\n";
            std::cout << "Generated tokens: " << reply.token_ids.size() << "\n";
            std::cout << "Stop reason: "
                      << aeon::text::stop_reason_name(reply.stop_reason) << "\n";
            std::cout << std::fixed << std::setprecision(2)
                      << "TTFT: " << reply.ttft_ms << " ms\n"
                      << "Decode throughput: " << reply.decode_tokens_per_second
                      << " tokens/sec\n";
            if (!engine.policy_loaded()) {
                std::cout << "Note: no generation_config.json; using the deterministic default\n";
            }
            if (options.deterministic_experts) {
                std::cout << "Note: --deterministic-experts is inert; the rewrite always "
                             "accumulates in fixed fp32 order\n";
            }
            // The reply is decoded with special tokens intact, so the thinking
            // markers survive into `reply.text`. Two distinct outcomes need two
            // distinct reports, and the difference is not "does `reply.text`
            // differ from the stripped view": when the token cap cuts the model
            // off before it closes its reasoning there is no marker at all, and
            // `strip_thinking` then returns the text unchanged — the same shape a
            // non-thinking reply has. The observable is the *closing marker*,
            // which exists exactly when the model finished thinking.
            const std::string thinking_end =
                engine.tokenizer().decode({engine.tokenizer().thinking_end_token_id()});
            const bool thinking_finished = !thinking_end.empty() &&
                reply.text.find(thinking_end) != std::string::npos;
            if (thinking_finished) {
                std::cout << "--- Reply, thinking included ---\n"
                          << reply.text << "\n"
                          << "--- Visible reply ---\n";
            } else if (options.thinking_mode) {
                std::cout << "--- Reply, thinking requested but never closed (the "
                             "generation hit the token cap mid-reasoning) ---\n"
                          << reply.text << "\n"
                          << "--- End of truncated reply ---\n";
            }
        }
        std::cout << visible << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[AeonChat] error: " << error.what() << std::endl;
        return 1;
    }
}
