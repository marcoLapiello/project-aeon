#include "core/device.hpp"
#include "core/memory_budget.hpp"
#include "core/v4_pipeline.hpp"
#include "text/dsv4_chat_formatter.hpp"
#include "text/dsv4_tokenizer.hpp"
#include "text/text_generation.hpp"

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
    std::string prompt;
    uint32_t layers{43};
    uint32_t context_size{4096};
    uint32_t max_new_tokens{256};
    uint64_t warm_gib{0};
    bool preload_warm_host{true};
    bool enable_warm_refill{true};
    std::string supply_telemetry_path;
    std::string supply_telemetry_run_id{"aeon-chat"};
    bool thinking_mode{false};
    bool until_eos{false};
    bool diagnostic{false};
};

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " --prompt <text> [options]\n"
        << "Options:\n"
        << "  --model-dir <path>       Native Aeon model directory\n"
        << "  --tokenizer <path>       Native tokenizer artifact (default: <model-dir>/tokenizer.aeon)\n"
        << "  --prompt <text>          One user prompt\n"
        << "  --thinking               Use explicit DSV4 thinking mode\n"
        << "  --max-new-tokens <count> Maximum generated tokens (default: 256)\n"
        << "  --until-eos              Generate until EOS or context capacity\n"
        << "  --layers <count>         Pipeline layers (default: 43)\n"
        << "  --context-size <count>   KV-cache/context capacity (default: 4096)\n"
        << "  --warm-gib <count>       Warm host allocation in GiB (default: 0)\n"
        << "  --no-warm-preload        Allocate Warm capacity without startup payload reads\n"
        << "  --no-warm-refill         Disable asynchronous Hot-to-Warm refill for A/B control\n"
        << "  --supply-telemetry <path> Write phase/source supply telemetry JSONL\n"
        << "  --run-id <id>            Supply telemetry run identifier\n"
        << "  --diagnostic             Print rendered prompt, IDs, and timings\n"
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
        } else if (argument == "--prompt") {
            options.prompt = require_value(argc, argv, index, "--prompt");
        } else if (argument == "--thinking") {
            options.thinking_mode = true;
        } else if (argument == "--until-eos") {
            options.until_eos = true;
        } else if (argument == "--max-new-tokens") {
            options.max_new_tokens = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--max-new-tokens"), "--max-new-tokens"
            ));
        } else if (argument == "--layers") {
            options.layers = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--layers"), "--layers"
            ));
        } else if (argument == "--context-size") {
            options.context_size = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--context-size"), "--context-size"
            ));
        } else if (argument == "--warm-gib") {
            options.warm_gib = parse_unsigned(
                require_value(argc, argv, index, "--warm-gib"), "--warm-gib"
            );
        } else if (argument == "--no-warm-preload") {
            options.preload_warm_host = false;
        } else if (argument == "--no-warm-refill") {
            options.enable_warm_refill = false;
        } else if (argument == "--supply-telemetry") {
            options.supply_telemetry_path = require_value(
                argc, argv, index, "--supply-telemetry");
        } else if (argument == "--run-id") {
            options.supply_telemetry_run_id = require_value(argc, argv, index, "--run-id");
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
    if (options.layers == 0 || options.context_size == 0 ||
        (!options.until_eos && options.max_new_tokens == 0)) {
        throw std::runtime_error("layers and context-size must be positive; max-new-tokens must be positive unless --until-eos is used");
    }
    if (options.tokenizer_path.empty()) {
        options.tokenizer_path = (std::filesystem::path(options.model_dir) / "tokenizer.aeon").string();
    }
    return options;
}

void print_ids(const char* label, const std::vector<uint32_t>& ids) {
    std::cout << label << " [";
    for (size_t index = 0; index < ids.size(); ++index) {
        std::cout << ids[index] << (index + 1 < ids.size() ? ", " : "");
    }
    std::cout << "]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        aeon::text::Dsv4Tokenizer tokenizer;
        tokenizer.load(options.tokenizer_path);
        aeon::text::Dsv4ChatFormatter formatter(tokenizer);
        const auto formatted = formatter.format(
            {{aeon::text::Dsv4MessageRole::User, options.prompt, ""}},
            options.thinking_mode
                ? aeon::text::Dsv4ThinkingMode::Thinking
                : aeon::text::Dsv4ThinkingMode::Chat
        );

        aeon::core::select_compute_device(true);
        aeon::core::AeonRuntimeConfig runtime_config;
        runtime_config.context_size = options.context_size;
        runtime_config.warm_host_bytes = options.warm_gib * 1024ULL * 1024ULL * 1024ULL;
        runtime_config.preload_warm_host = options.preload_warm_host;
        runtime_config.enable_warm_refill = options.enable_warm_refill;

        aeon::core::V4Pipeline pipeline;
        pipeline.init_dynamic_global(
            options.model_dir, runtime_config, options.layers);
        if (!options.supply_telemetry_path.empty()) {
            pipeline.enable_supply_telemetry(
                options.supply_telemetry_path, options.supply_telemetry_run_id);
        }

        aeon::text::GenerationOptions generation_options;
        if (options.until_eos) {
            if (formatted.token_ids.size() >= options.context_size) {
                throw std::runtime_error("formatted prompt leaves no room for generation");
            }
            generation_options.max_new_tokens =
                options.context_size - static_cast<uint32_t>(formatted.token_ids.size()) + 1;
        } else {
            generation_options.max_new_tokens = options.max_new_tokens;
        }
        generation_options.eos_token_id = tokenizer.eos_token_id();
        generation_options.context_limit = options.context_size;
        generation_options.thinking_mode = options.thinking_mode;

        double ttft_ms = 0.0;
        double tok_per_sec = 0.0;
        const auto generation = pipeline.generate_until_stop(
            formatted.token_ids,
            generation_options,
            &ttft_ms,
            &tok_per_sec
        );

        std::vector<uint32_t> response_ids = generation.token_ids;
        if (!response_ids.empty() && response_ids.back() == tokenizer.eos_token_id()) {
            response_ids.pop_back();
        }
        std::string response = tokenizer.decode(response_ids);
        if (options.thinking_mode) {
            const std::string thinking_end = tokenizer.decode({tokenizer.thinking_end_token_id()});
            const size_t thinking_end_position = response.rfind(thinking_end);
            if (thinking_end_position != std::string::npos) {
                response = response.substr(thinking_end_position + thinking_end.size());
            }
        }

        if (options.diagnostic) {
            std::cout << "Rendered prompt: " << formatted.rendered_text << "\n";
            print_ids("Prompt IDs:", formatted.token_ids);
            print_ids("Generated IDs:", generation.token_ids);
            std::cout << "Stop reason: " << aeon::text::stop_reason_name(generation.stop_reason) << "\n";
            std::cout << std::fixed << std::setprecision(2)
                      << "TTFT: " << ttft_ms << " ms\n"
                      << "Decode throughput: " << tok_per_sec << " tokens/sec\n";
        }
        std::cout << response << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[AeonChat] error: " << error.what() << std::endl;
        return 1;
    }
}