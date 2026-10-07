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
#include "infrastructure/parallel/parallel_topology.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
    // The parallel topology flags: device ids in stage-major order, the two degrees,
    // and the fraction of each device's total VRAM the budget may plan against.
    std::vector<int> device_ids;
    uint32_t tensor_parallel{1};
    uint32_t pipeline_parallel{1};
    double gpu_memory_utilization{0.95};
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
        options.runtime.gpu_memory_utilization = gpu_memory_utilization;
        options.runtime.parallel.device_ids = device_ids;
        options.runtime.parallel.tensor_parallel = tensor_parallel;
        options.runtime.parallel.pipeline_parallel = pipeline_parallel;
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

inline double engine_cli_double(const std::string& value, const char* option) {
    size_t consumed = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
    }
    if (consumed != value.size()) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
    }
    return parsed;
}

inline std::string engine_cli_value(int argc, char** argv, int& index, const char* option) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + option);
    }
    ++index;
    return argv[index];
}

// The parallel-topology half of the shared engine flags, split out so a caller that
// parses its own flags can still reuse exactly this parsing (and this validation)
// without duplicating it. `parse_engine_flag` calls it too.
inline bool parse_engine_parallel_flag(EngineCli& cli, int argc, char** argv, int& index) {
    const std::string argument = argv[index];
    if (argument == "--device-ids") {
        cli.device_ids =
            aeon::core::parse_device_ids(engine_cli_value(argc, argv, index, "--device-ids"));
    } else if (argument == "--tensor-parallel") {
        cli.tensor_parallel = static_cast<uint32_t>(engine_cli_unsigned(
            engine_cli_value(argc, argv, index, "--tensor-parallel"), "--tensor-parallel"));
    } else if (argument == "--pipeline-parallel") {
        cli.pipeline_parallel = static_cast<uint32_t>(engine_cli_unsigned(
            engine_cli_value(argc, argv, index, "--pipeline-parallel"), "--pipeline-parallel"));
    } else if (argument == "--gpu-memory-utilization") {
        const std::string value =
            engine_cli_value(argc, argv, index, "--gpu-memory-utilization");
        const double parsed = engine_cli_double(value, "--gpu-memory-utilization");
        // The ceiling is deliberately below 1.0: what `total` includes but a budget
        // must not plan against — a display's scanout, the runtime's own pools — is
        // never zero, and `0.99` is the largest fraction that still leaves the small
        // margin a driver needs. Above it is a bug, not a preference.
        if (!(parsed > 0.0 && parsed <= 0.99)) {
            throw std::runtime_error(
                "invalid value for --gpu-memory-utilization: " + value +
                " (must be in (0, 0.99])");
        }
        cli.gpu_memory_utilization = parsed;
    } else {
        return false;
    }
    return true;
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
    } else if (parse_engine_parallel_flag(cli, argc, argv, index)) {
        // consumed by the parallel-topology half
    } else if (argument == "--verbose") {
        cli.verbose = true;
    } else {
        return false;
    }
    return true;
}

}  // namespace aeon::tools
