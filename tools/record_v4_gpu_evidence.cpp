#include "architecture/deepseek_v4/core/v4_pipeline.hpp"
#include "platform/rdna3/device.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kVocabularySize = 129280;
constexpr size_t kReportedTopK = 8;

struct Options {
    std::string model_dir{"models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon"};
    std::string output_path{"build/correctness/v4_gpu_evidence.json"};
    std::vector<uint32_t> token_ids{1, 101, 2054, 300};
    uint64_t warm_host_gib{35};
    size_t context_size{16};
    size_t requested_batch_size{16};
    bool include_logits{false};
};

struct LogitSnapshot {
    std::vector<half> half_values;
    std::vector<float> values;
    std::vector<size_t> top_indices;
    uint64_t fnv1a64{1469598103934665603ULL};
};

void check_hip(hipError_t result, const char* operation) {
    if (result != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(result));
    }
}

void write_json_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u00" << "0123456789abcdef"[(character >> 4) & 0xf]
                           << "0123456789abcdef"[character & 0xf];
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    output << '"';
}

uint64_t parse_unsigned(const std::string& value, const char* option) {
    size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &consumed, 10);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("invalid value for ") + option + ": " + value);
    }
    if (consumed != value.size()) {
        throw std::invalid_argument(std::string("invalid value for ") + option + ": " + value);
    }
    return static_cast<uint64_t>(parsed);
}

std::vector<uint32_t> parse_token_ids(const std::string& value) {
    std::vector<uint32_t> token_ids;
    size_t start = 0;
    while (start < value.size()) {
        const size_t separator = value.find(',', start);
        const size_t end = separator == std::string::npos ? value.size() : separator;
        if (end == start) throw std::invalid_argument("token list contains an empty item");
        const uint64_t token = parse_unsigned(value.substr(start, end - start), "--tokens");
        if (token > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("token ID exceeds uint32 range");
        }
        token_ids.push_back(static_cast<uint32_t>(token));
        if (separator == std::string::npos) break;
        start = separator + 1;
    }
    if (token_ids.empty()) throw std::invalid_argument("--tokens must not be empty");
    return token_ids;
}

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "Options:\n"
        << "  --model-dir <path>       Native Aeon model directory\n"
        << "  --output <path>          Evidence JSON path (default: build/correctness/v4_gpu_evidence.json)\n"
        << "  --tokens <a,b,c>         Exact input token IDs\n"
        << "  --warm-gib <count>       Warm host allocation in GiB (default: 35)\n"
        << "  --context-size <count>   Runtime context capacity (default: 16)\n"
        << "  --batch-size <count>     Requested prefill batch size (default: 16)\n"
        << "  --include-logits         Write the complete final logit vectors\n"
        << "  --help                   Show this help\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (argument == "--include-logits") {
            options.include_logits = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--model-dir") {
            options.model_dir = value;
        } else if (argument == "--output") {
            options.output_path = value;
        } else if (argument == "--tokens") {
            options.token_ids = parse_token_ids(value);
        } else if (argument == "--warm-gib") {
            options.warm_host_gib = parse_unsigned(value, "--warm-gib");
        } else if (argument == "--context-size") {
            options.context_size = static_cast<size_t>(parse_unsigned(value, "--context-size"));
        } else if (argument == "--batch-size") {
            options.requested_batch_size = static_cast<size_t>(parse_unsigned(value, "--batch-size"));
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.context_size == 0) throw std::invalid_argument("--context-size must be positive");
    if (options.requested_batch_size == 0) throw std::invalid_argument("--batch-size must be positive");
    if (options.token_ids.size() + 1 > options.context_size) {
        throw std::invalid_argument("input tokens plus one continuation exceed --context-size");
    }
    return options;
}

LogitSnapshot capture_logits(aeon::core::V4Pipeline& pipeline) {
    LogitSnapshot snapshot;
    snapshot.half_values.resize(kVocabularySize);
    check_hip(
        hipMemcpy(
            snapshot.half_values.data(),
            pipeline.scratch.d_logits,
            snapshot.half_values.size() * sizeof(half),
            hipMemcpyDeviceToHost),
        "copy logits");

    snapshot.values.resize(snapshot.half_values.size());
    for (size_t index = 0; index < snapshot.half_values.size(); ++index) {
        snapshot.values[index] = __half2float(snapshot.half_values[index]);
        if (!std::isfinite(snapshot.values[index])) {
            throw std::runtime_error("GPU logits contain a non-finite value");
        }
        const auto* bytes = reinterpret_cast<const uint8_t*>(&snapshot.half_values[index]);
        for (size_t byte = 0; byte < sizeof(half); ++byte) {
            snapshot.fnv1a64 ^= bytes[byte];
            snapshot.fnv1a64 *= 1099511628211ULL;
        }
    }

    snapshot.top_indices.resize(snapshot.values.size());
    std::iota(snapshot.top_indices.begin(), snapshot.top_indices.end(), 0);
    const auto compare_logits = [&snapshot](size_t left, size_t right) {
        if (snapshot.values[left] != snapshot.values[right]) {
            return snapshot.values[left] > snapshot.values[right];
        }
        return left < right;
    };
    const size_t top_count = std::min(kReportedTopK, snapshot.top_indices.size());
    std::partial_sort(
        snapshot.top_indices.begin(),
        snapshot.top_indices.begin() + static_cast<std::ptrdiff_t>(top_count),
        snapshot.top_indices.end(),
        compare_logits);
    snapshot.top_indices.resize(top_count);
    return snapshot;
}

const char* execution_path_name(aeon::core::V4PrefillExecutionPath path) {
    switch (path) {
        case aeon::core::V4PrefillExecutionPath::Batched: return "Batched";
        case aeon::core::V4PrefillExecutionPath::SerializedFallback: return "SerializedFallback";
    }
    return "Unknown";
}

void write_token_array(std::ostream& output, const std::vector<uint32_t>& values) {
    output << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        output << values[index];
    }
    output << ']';
}

void write_logit_snapshot(
    std::ostream& output,
    const LogitSnapshot& snapshot,
    bool include_logits
) {
    output << "{\n      \"vocabulary_size\":" << snapshot.values.size()
           << ",\n      \"finite_count\":" << snapshot.values.size()
           << ",\n      \"fnv1a64_half_bits\":" << snapshot.fnv1a64
           << ",\n      \"top_k\":[";
    for (size_t index = 0; index < snapshot.top_indices.size(); ++index) {
        if (index != 0) output << ',';
        const size_t token = snapshot.top_indices[index];
        output << "{\"token\":" << token << ",\"logit\":"
               << snapshot.values[token] << '}';
    }
    output << ']';
    if (include_logits) {
        output << ",\n      \"logits\":[";
        for (size_t index = 0; index < snapshot.values.size(); ++index) {
            if (index != 0) output << ',';
            output << snapshot.values[index];
        }
        output << ']';
    }
    output << "\n    }";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        aeon::core::select_compute_device(true);

        int device_index = 0;
        check_hip(hipGetDevice(&device_index), "get HIP device");
        hipDeviceProp_t device_properties{};
        check_hip(hipGetDeviceProperties(&device_properties, device_index), "get HIP device properties");

        aeon::core::AeonRuntimeConfig runtime_config;
        runtime_config.context_size = static_cast<uint32_t>(options.context_size);
        runtime_config.warm_host_bytes = options.warm_host_gib * 1024ULL * 1024ULL * 1024ULL;
        runtime_config.deterministic_expert_accumulation = true;

        aeon::core::V4Pipeline pipeline;
        pipeline.initialize(options.model_dir, runtime_config);

        const auto prefill_start = std::chrono::steady_clock::now();
        const auto prefill_result = pipeline.prefill_batched(
            std::span<const uint32_t>(options.token_ids),
            0,
            options.requested_batch_size,
            true);
        const auto prefill_end = std::chrono::steady_clock::now();
        const LogitSnapshot prefill_logits = capture_logits(pipeline);

        const auto decode_start = std::chrono::steady_clock::now();
        const uint32_t continuation_token = pipeline.step(
            prefill_result.next_token,
            static_cast<uint32_t>(options.token_ids.size()),
            aeon::core::RoutingPhase::Decode);
        const auto decode_end = std::chrono::steady_clock::now();
        const LogitSnapshot decode_logits = capture_logits(pipeline);

        const double prefill_ms = std::chrono::duration<double, std::milli>(
            prefill_end - prefill_start).count();
        const double decode_ms = std::chrono::duration<double, std::milli>(
            decode_end - decode_start).count();

        std::ofstream file;
        std::ostream* output = &std::cout;
        if (options.output_path != "-") {
            const std::filesystem::path output_path(options.output_path);
            if (!output_path.parent_path().empty()) {
                std::filesystem::create_directories(output_path.parent_path());
            }
            file.open(output_path);
            if (!file.is_open()) throw std::runtime_error("failed to open output: " + options.output_path);
            output = &file;
        }

        *output << std::setprecision(9)
                << "{\n  \"stage\":\"gpu_full_model_evidence\",\n  \"hardware\":{\n"
                << "    \"name\":";
        write_json_string(*output, device_properties.name);
        *output << ",\n    \"gcn_arch\":";
        write_json_string(*output, device_properties.gcnArchName);
        *output << "\n  },\n  \"runtime\":{\n    \"model_dir\":";
        write_json_string(*output, std::filesystem::absolute(options.model_dir).string());
        *output << ",\n    \"context_size\":" << options.context_size
                << ",\n    \"warm_host_gib\":" << options.warm_host_gib
                << ",\n    \"requested_batch_size\":" << options.requested_batch_size
                << ",\n    \"deterministic_expert_accumulation\":true\n  },\n  \"input_token_ids\":";
        write_token_array(*output, options.token_ids);
        *output << ",\n  \"prefill\":{\n    \"token_count\":" << prefill_result.token_count
                << ",\n    \"execution_path\":";
        write_json_string(*output, execution_path_name(prefill_result.execution_path));
        *output << ",\n    \"next_token\":" << prefill_result.next_token
                << ",\n    \"sequence_length\":" << options.token_ids.size()
                << ",\n    \"elapsed_ms\":" << prefill_ms << ",\n    \"logits\":";
        write_logit_snapshot(*output, prefill_logits, options.include_logits);
        *output << "\n  },\n  \"decode\":{\n    \"input_token\":" << prefill_result.next_token
                << ",\n    \"position\":" << options.token_ids.size()
                << ",\n    \"next_token\":" << continuation_token
                << ",\n    \"sequence_length\":" << (options.token_ids.size() + 1)
                << ",\n    \"elapsed_ms\":" << decode_ms << ",\n    \"logits\":";
        write_logit_snapshot(*output, decode_logits, options.include_logits);
        *output << "\n  }\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "record_v4_gpu_evidence: " << error.what() << '\n';
        return 1;
    }
}