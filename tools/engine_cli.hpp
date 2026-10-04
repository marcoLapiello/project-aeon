#pragma once

// -----------------------------------------------------------------------------
// The engine-setup flags shared by `aeon_chat` and `aeon_serve`.
//
// Only the flags that configure how the engine is built and loaded live here; the
// development-only knobs (registry audits, telemetry, logit dumps, prompt and
// sampling flags) stay in `aeon_chat`. The point is that the serving binary does
// not carry a second, drifting copy of the option parsing for the fields the two
// binaries genuinely share.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/runtime/v4_engine.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace aeon::tools {

struct EngineCli {
    std::string model_dir{"models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon"};
    uint32_t context_size{4096};
    uint64_t warm_gib{0};
    bool preload_warm_host{true};
    uint32_t prefill_window{4096};
    uint32_t prefill_chunk{64};
    uint32_t prefill_sweep_min_tokens{0};
    uint32_t staging_blocks{2};
    bool verbose{false};

    aeon::core::V4EngineOptions to_engine_options() const {
        aeon::core::V4EngineOptions options;
        options.model_dir = model_dir;
        options.verbose = verbose;
        options.runtime.context_size = context_size;
        options.runtime.warm_host_bytes = warm_gib * 1024ULL * 1024ULL * 1024ULL;
        options.runtime.preload_warm_host = preload_warm_host;
        options.runtime.prefill_window = prefill_window;
        options.runtime.prefill_chunk = prefill_chunk;
        options.runtime.prefill_sweep_min_tokens = prefill_sweep_min_tokens;
        options.runtime.prefill_sweep_staging_blocks = staging_blocks;
        return options;
    }
};

inline uint64_t engine_cli_unsigned(const std::string& value, const char* option) {
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

inline std::string engine_cli_value(int argc, char** argv, int& index, const char* option) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + option);
    }
    ++index;
    return argv[index];
}

// Consumes the shared engine flags in `argv[index]`. Returns true when the flag was
// one of ours (and `index` was advanced); false leaves `index` untouched for the
// caller to handle its own flags.
inline bool parse_engine_flag(EngineCli& cli, int argc, char** argv, int& index) {
    const std::string argument = argv[index];
    if (argument == "--model-dir") {
        cli.model_dir = engine_cli_value(argc, argv, index, "--model-dir");
    } else if (argument == "--context-size") {
        cli.context_size = static_cast<uint32_t>(engine_cli_unsigned(
            engine_cli_value(argc, argv, index, "--context-size"), "--context-size"));
    } else if (argument == "--warm-gib") {
        cli.warm_gib = engine_cli_unsigned(
            engine_cli_value(argc, argv, index, "--warm-gib"), "--warm-gib");
    } else if (argument == "--no-warm-preload") {
        cli.preload_warm_host = false;
    } else if (argument == "--prefill-window") {
        cli.prefill_window = static_cast<uint32_t>(engine_cli_unsigned(
            engine_cli_value(argc, argv, index, "--prefill-window"), "--prefill-window"));
    } else if (argument == "--prefill-chunk") {
        cli.prefill_chunk = static_cast<uint32_t>(engine_cli_unsigned(
            engine_cli_value(argc, argv, index, "--prefill-chunk"), "--prefill-chunk"));
    } else if (argument == "--prefill-sweep-min-tokens") {
        cli.prefill_sweep_min_tokens = static_cast<uint32_t>(engine_cli_unsigned(
            engine_cli_value(argc, argv, index, "--prefill-sweep-min-tokens"),
            "--prefill-sweep-min-tokens"));
    } else if (argument == "--staging-blocks") {
        cli.staging_blocks = static_cast<uint32_t>(engine_cli_unsigned(
            engine_cli_value(argc, argv, index, "--staging-blocks"), "--staging-blocks"));
    } else if (argument == "--verbose") {
        cli.verbose = true;
    } else {
        return false;
    }
    return true;
}

}  // namespace aeon::tools
