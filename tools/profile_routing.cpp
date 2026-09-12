#include "core/device.hpp"
#include "core/routing_profile.hpp"
#include "core/v4_pipeline.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model_dir{"models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon"};
    std::string input_path;
    std::string output_dir;
    std::string corpus_id;
    std::string dataset{"profile"};
    std::string mode{"dynamic"};
    uint32_t layers{43};
    uint32_t context_size{4096};
    uint64_t warm_gib{0};
    uint32_t vram_slots{8};
    bool regenerate_summary{false};
};

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " --input <prompts.jsonl> --output-dir <directory> [options]\n"
        << "       " << executable << " --output-dir <directory> --regenerate-summary\n"
        << "Options:\n"
        << "  --model-dir <path>       Native Aeon model directory\n"
        << "  --input <path>           Tokenized JSONL prompt corpus\n"
        << "  --output-dir <path>      Persistent routing profile directory\n"
        << "  --corpus-id <id>         Stable corpus identity for incremental runs\n"
        << "  --dataset <profile|validation>  Dataset label (default: profile)\n"
        << "  --mode <dynamic|aeon>    Pipeline initialization mode (default: dynamic)\n"
        << "  --layers <count>         Layer count (default: 43)\n"
        << "  --context-size <count>  Maximum context size (default: 4096)\n"
        << "  --warm-gib <count>       Warm host allocation in GiB (default: 0)\n"
        << "  --vram-slots <count>     Fixed VRAM slots in aeon mode (default: 8)\n"
        << "  --regenerate-summary     Rebuild summary.csv from existing state.bin only\n"
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
        }
        if (argument == "--regenerate-summary") {
            options.regenerate_summary = true;
        } else if (argument == "--model-dir") {
            options.model_dir = require_value(argc, argv, index, "--model-dir");
        } else if (argument == "--input") {
            options.input_path = require_value(argc, argv, index, "--input");
        } else if (argument == "--output-dir") {
            options.output_dir = require_value(argc, argv, index, "--output-dir");
        } else if (argument == "--corpus-id") {
            options.corpus_id = require_value(argc, argv, index, "--corpus-id");
        } else if (argument == "--dataset") {
            options.dataset = require_value(argc, argv, index, "--dataset");
        } else if (argument == "--mode") {
            options.mode = require_value(argc, argv, index, "--mode");
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
        } else if (argument == "--vram-slots") {
            options.vram_slots = static_cast<uint32_t>(parse_unsigned(
                require_value(argc, argv, index, "--vram-slots"), "--vram-slots"
            ));
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }

    if (options.output_dir.empty()) {
        throw std::runtime_error("--output-dir is required");
    }
    if (options.regenerate_summary) {
        return options;
    }
    if (options.input_path.empty()) {
        throw std::runtime_error("--input is required");
    }
    if (options.layers == 0 || options.context_size == 0 || options.vram_slots == 0) {
        throw std::runtime_error("layers, context-size, and vram-slots must be positive");
    }
    if (options.mode != "dynamic" && options.mode != "aeon") {
        throw std::runtime_error("--mode must be dynamic or aeon");
    }
    if (options.dataset != "profile" && options.dataset != "validation") {
        throw std::runtime_error("--dataset must be profile or validation");
    }
    if (options.corpus_id.empty()) {
        options.corpus_id = std::filesystem::absolute(options.input_path).string();
    }
    return options;
}

void validate_prompt(const aeon::core::RoutingPrompt& prompt, uint32_t context_size) {
    constexpr uint32_t vocab_size = 129280;
    if (prompt.tokens.size() + prompt.max_new_tokens > context_size) {
        throw std::runtime_error("prompt exceeds configured context size");
    }
    for (uint32_t token : prompt.tokens) {
        if (token >= vocab_size) {
            throw std::runtime_error("prompt contains a token outside the model vocabulary");
        }
    }
}

void validate_counter(
    const aeon::core::RoutingCounter& counter,
    const aeon::core::RoutingPrompt& prompt,
    uint32_t layers
) {
    const uint64_t expected_prefill = static_cast<uint64_t>(prompt.tokens.size()) * aeon::core::RoutingCounter::kTopK;
    const uint64_t expected_decode = static_cast<uint64_t>(prompt.max_new_tokens - 1) * aeon::core::RoutingCounter::kTopK;
    if (counter.num_layers() != layers) {
        throw std::runtime_error("routing counter layer count does not match profiler configuration");
    }
    for (uint32_t layer_id = 0; layer_id < layers; ++layer_id) {
        if (counter.total_selections(aeon::core::RoutingPhase::Prefill, layer_id) != expected_prefill ||
            counter.total_selections(aeon::core::RoutingPhase::Decode, layer_id) != expected_decode) {
            throw std::runtime_error("routing counter did not capture six valid selections per processed token");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.regenerate_summary) {
            aeon::core::RoutingProfileStore::regenerate_summary(options.output_dir);
            std::cout << "[RoutingProfile] regenerated summary="
                      << (std::filesystem::path(options.output_dir) / "summary.csv") << '\n';
            return 0;
        }
        const auto prompts = aeon::core::load_routing_prompts(options.input_path);

        aeon::core::RoutingProfileRunConfig run_config;
    #ifdef AEON_GIT_COMMIT
        run_config.aeon_git_commit = AEON_GIT_COMMIT;
    #endif
        run_config.model_dir = std::filesystem::absolute(options.model_dir).string();
        run_config.input_path = std::filesystem::absolute(options.input_path).string();
        run_config.corpus_id = options.corpus_id;
        run_config.dataset = options.dataset;
        run_config.mode = options.mode;
        run_config.num_layers = options.layers;
        run_config.context_size = options.context_size;
        run_config.warm_gib = options.warm_gib;
        run_config.vram_slots = options.vram_slots;
        run_config.input_prompt_count = prompts.size();
        run_config.model_config_hash = aeon::core::routing_profile_detail::fnv1a_file(
            std::filesystem::path(options.model_dir) / "config.json"
        );
        const auto artifact = aeon::core::make_current_swizzled_artifact_spec();
        run_config.model_index_hash = aeon::core::routing_profile_detail::fnv1a_file(
            std::filesystem::path(options.model_dir) / artifact.index_filename
        );

        aeon::core::RoutingProfileStore store(options.output_dir, run_config);
        std::vector<aeon::core::RoutingPrompt> pending;
        pending.reserve(prompts.size());
        for (const auto& prompt : prompts) {
            if (!store.is_completed(prompt)) {
                pending.push_back(prompt);
            }
        }

        std::cout << "[RoutingProfile] prompts=" << prompts.size()
                  << " pending=" << pending.size()
                  << " completed=" << store.completed_count()
                  << " output=" << options.output_dir << '\n';
        if (pending.empty()) {
            return 0;
        }

        aeon::core::select_compute_device(true);
        aeon::core::V4Pipeline pipeline;
        if (options.mode == "dynamic") {
            aeon::core::AeonRuntimeConfig runtime_config;
            runtime_config.context_size = options.context_size;
            runtime_config.warm_host_bytes = static_cast<size_t>(options.warm_gib) * 1024ULL * 1024ULL * 1024ULL;
            pipeline.init_dynamic_global(options.model_dir, runtime_config, options.layers);
        } else {
            pipeline.init_aeon(options.model_dir, options.layers, options.vram_slots, options.context_size, true);
        }
        pipeline.enable_routing_counter();

        for (const auto& prompt : pending) {
            try {
                validate_prompt(prompt, options.context_size);
                pipeline.reset_routing_counter();
                const auto generated = pipeline.generate(prompt.tokens, prompt.max_new_tokens);
                if (generated.size() != prompt.max_new_tokens) {
                    throw std::runtime_error("pipeline returned an unexpected number of generated tokens");
                }
                const auto* counter = pipeline.routing_counter();
                if (counter == nullptr) {
                    throw std::runtime_error("routing counter was not enabled");
                }
                validate_counter(*counter, prompt, options.layers);
                store.add_prompt(prompt, *counter);
                store.checkpoint();
                std::cout << "[RoutingProfile] completed id=" << prompt.id
                          << " total_completed=" << store.completed_count() << '\n';
            } catch (const std::exception& error) {
                store.mark_failed(prompt, error.what());
                store.checkpoint();
                std::cerr << "[RoutingProfile] failed id=" << prompt.id << ": " << error.what() << '\n';
            }
        }

        std::cout << "[RoutingProfile] completed=" << store.completed_count()
                  << " failed=" << store.failed_count() << '\n';
        return store.failed_count() == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[RoutingProfile] error: " << error.what() << '\n';
        return 1;
    }
}