#pragma once

#include "core/routing_counter.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace aeon::core {

struct RoutingPrompt {
    std::string id;
    std::vector<uint32_t> tokens;
    uint32_t max_new_tokens{0};
};

namespace routing_profile_detail {

inline void skip_whitespace(std::string_view input, size_t& cursor) {
    while (cursor < input.size()) {
        const unsigned char ch = static_cast<unsigned char>(input[cursor]);
        if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
            break;
        }
        ++cursor;
    }
}

inline void expect(std::string_view input, size_t& cursor, char expected) {
    skip_whitespace(input, cursor);
    if (cursor >= input.size() || input[cursor] != expected) {
        throw std::runtime_error("invalid JSONL record: expected '" + std::string(1, expected) + "'");
    }
    ++cursor;
}

inline uint32_t parse_uint32(std::string_view input, size_t& cursor) {
    skip_whitespace(input, cursor);
    const size_t begin = cursor;
    while (cursor < input.size() && input[cursor] >= '0' && input[cursor] <= '9') {
        ++cursor;
    }
    if (begin == cursor) {
        throw std::runtime_error("invalid JSONL record: expected unsigned integer");
    }

    uint32_t value = 0;
    const char* first = input.data() + begin;
    const char* last = input.data() + cursor;
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last) {
        throw std::runtime_error("invalid JSONL record: integer is out of range");
    }
    return value;
}

inline void append_utf8(std::string& output, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

inline std::string parse_string(std::string_view input, size_t& cursor) {
    skip_whitespace(input, cursor);
    if (cursor >= input.size() || input[cursor] != '"') {
        throw std::runtime_error("invalid JSONL record: expected string");
    }
    ++cursor;

    std::string value;
    while (cursor < input.size()) {
        const unsigned char ch = static_cast<unsigned char>(input[cursor++]);
        if (ch == '"') {
            return value;
        }
        if (ch < 0x20) {
            throw std::runtime_error("invalid JSONL record: control character in string");
        }
        if (ch != '\\') {
            value.push_back(static_cast<char>(ch));
            continue;
        }
        if (cursor >= input.size()) {
            throw std::runtime_error("invalid JSONL record: incomplete escape");
        }
        const char escaped = input[cursor++];
        switch (escaped) {
        case '"': value.push_back('"'); break;
        case '\\': value.push_back('\\'); break;
        case '/': value.push_back('/'); break;
        case 'b': value.push_back('\b'); break;
        case 'f': value.push_back('\f'); break;
        case 'n': value.push_back('\n'); break;
        case 'r': value.push_back('\r'); break;
        case 't': value.push_back('\t'); break;
        case 'u': {
            if (cursor + 4 > input.size()) {
                throw std::runtime_error("invalid JSONL record: incomplete unicode escape");
            }
            uint32_t codepoint = 0;
            for (size_t i = 0; i < 4; ++i) {
                const char digit = input[cursor++];
                codepoint <<= 4;
                if (digit >= '0' && digit <= '9') {
                    codepoint |= static_cast<uint32_t>(digit - '0');
                } else if (digit >= 'a' && digit <= 'f') {
                    codepoint |= static_cast<uint32_t>(digit - 'a' + 10);
                } else if (digit >= 'A' && digit <= 'F') {
                    codepoint |= static_cast<uint32_t>(digit - 'A' + 10);
                } else {
                    throw std::runtime_error("invalid JSONL record: malformed unicode escape");
                }
            }
            if (codepoint >= 0xd800 && codepoint <= 0xdfff) {
                throw std::runtime_error("invalid JSONL record: surrogate escape is unsupported");
            }
            append_utf8(value, codepoint);
            break;
        }
        default:
            throw std::runtime_error("invalid JSONL record: unsupported escape");
        }
    }
    throw std::runtime_error("invalid JSONL record: unterminated string");
}

inline std::vector<uint32_t> parse_token_array(std::string_view input, size_t& cursor) {
    expect(input, cursor, '[');
    skip_whitespace(input, cursor);
    std::vector<uint32_t> tokens;
    if (cursor < input.size() && input[cursor] == ']') {
        ++cursor;
        return tokens;
    }

    while (true) {
        tokens.push_back(parse_uint32(input, cursor));
        skip_whitespace(input, cursor);
        if (cursor >= input.size()) {
            throw std::runtime_error("invalid JSONL record: unterminated token array");
        }
        if (input[cursor] == ']') {
            ++cursor;
            return tokens;
        }
        expect(input, cursor, ',');
    }
}

inline std::string json_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (unsigned char ch : value) {
        switch (ch) {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\b': escaped += "\\b"; break;
        case '\f': escaped += "\\f"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (ch < 0x20) {
                std::ostringstream code;
                code << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                     << static_cast<unsigned int>(ch);
                escaped += code.str();
            } else {
                escaped.push_back(static_cast<char>(ch));
            }
            break;
        }
    }
    return escaped;
}

inline void write_string(std::ostream& output, std::string_view value) {
    const uint64_t size = value.size();
    output.write(reinterpret_cast<const char*>(&size), sizeof(size));
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!output) {
        throw std::runtime_error("failed to write routing profile state");
    }
}

inline std::string read_string(std::istream& input) {
    uint64_t size = 0;
    input.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!input || size > 16 * 1024 * 1024) {
        throw std::runtime_error("invalid routing profile state string");
    }
    std::string value(size, '\0');
    input.read(value.data(), static_cast<std::streamsize>(size));
    if (!input) {
        throw std::runtime_error("truncated routing profile state");
    }
    return value;
}

template <typename Value>
inline void write_value(std::ostream& output, const Value& value) {
    static_assert(std::is_trivially_copyable_v<Value>);
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error("failed to write routing profile state");
    }
}

template <typename Value>
inline Value read_value(std::istream& input) {
    static_assert(std::is_trivially_copyable_v<Value>);
    Value value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("truncated routing profile state");
    }
    return value;
}

inline void rename_atomic(const std::filesystem::path& temporary, const std::filesystem::path& target) {
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("failed to publish routing profile file " + target.string() + ": " + error.message());
    }
}

inline void write_text_atomic(const std::filesystem::path& path, std::string_view content) {
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("failed to open routing profile file " + temporary);
        }
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        output.flush();
        if (!output) {
            throw std::runtime_error("failed to write routing profile file " + temporary);
        }
    }
    rename_atomic(temporary, path);
}

template <typename Writer>
inline void write_binary_atomic(const std::filesystem::path& path, Writer&& writer) {
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("failed to open routing profile file " + temporary);
        }
        writer(output);
        output.flush();
        if (!output) {
            throw std::runtime_error("failed to write routing profile file " + temporary);
        }
    }
    rename_atomic(temporary, path);
}

inline uint64_t fnv1a_bytes(const void* data, size_t size, uint64_t hash = 1469598103934665603ULL) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline uint64_t fnv1a_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return 0;
    }
    uint64_t hash = 1469598103934665603ULL;
    std::array<char, 8192> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            hash = fnv1a_bytes(buffer.data(), static_cast<size_t>(count), hash);
        }
    }
    return hash;
}

inline std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open routing profile file " + path.string());
    }
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

inline std::string extract_json_string_field(std::string_view object, std::string_view field) {
    const std::string marker = "\"" + std::string(field) + "\"";
    const size_t field_pos = object.find(marker);
    if (field_pos == std::string_view::npos) {
        throw std::runtime_error("routing profile metadata is missing " + std::string(field));
    }
    size_t cursor = field_pos + marker.size();
    expect(object, cursor, ':');
    return parse_string(object, cursor);
}

} // namespace routing_profile_detail

inline RoutingPrompt parse_routing_prompt(std::string_view line, size_t line_number = 0) {
    using namespace routing_profile_detail;

    size_t cursor = 0;
    expect(line, cursor, '{');
    bool have_id = false;
    bool have_tokens = false;
    bool have_max_new_tokens = false;
    RoutingPrompt prompt;

    skip_whitespace(line, cursor);
    if (cursor < line.size() && line[cursor] == '}') {
        throw std::runtime_error("invalid JSONL record at line " + std::to_string(line_number) + ": empty object");
    }

    while (true) {
        const std::string field = parse_string(line, cursor);
        expect(line, cursor, ':');
        if (field == "id") {
            if (have_id) {
                throw std::runtime_error("invalid JSONL record: duplicate id field");
            }
            prompt.id = parse_string(line, cursor);
            have_id = true;
        } else if (field == "tokens") {
            if (have_tokens) {
                throw std::runtime_error("invalid JSONL record: duplicate tokens field");
            }
            prompt.tokens = parse_token_array(line, cursor);
            have_tokens = true;
        } else if (field == "max_new_tokens") {
            if (have_max_new_tokens) {
                throw std::runtime_error("invalid JSONL record: duplicate max_new_tokens field");
            }
            prompt.max_new_tokens = parse_uint32(line, cursor);
            have_max_new_tokens = true;
        } else {
            throw std::runtime_error("invalid JSONL record: unknown field '" + field + "'");
        }

        skip_whitespace(line, cursor);
        if (cursor >= line.size()) {
            throw std::runtime_error("invalid JSONL record: unterminated object");
        }
        if (line[cursor] == '}') {
            ++cursor;
            break;
        }
        expect(line, cursor, ',');
    }

    skip_whitespace(line, cursor);
    if (cursor != line.size()) {
        throw std::runtime_error("invalid JSONL record: trailing data");
    }
    if (!have_id || prompt.id.empty() || !have_tokens || prompt.tokens.empty() ||
        !have_max_new_tokens || prompt.max_new_tokens == 0) {
        throw std::runtime_error("invalid JSONL record: id, non-empty tokens, and positive max_new_tokens are required");
    }
    return prompt;
}

inline std::vector<RoutingPrompt> load_routing_prompts(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open routing prompt corpus " + path.string());
    }

    std::vector<RoutingPrompt> prompts;
    std::unordered_set<std::string> ids;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        bool only_whitespace = true;
        for (char ch : line) {
            if (ch != ' ' && ch != '\t' && ch != '\r') {
                only_whitespace = false;
                break;
            }
        }
        if (only_whitespace) {
            continue;
        }
        RoutingPrompt prompt = parse_routing_prompt(line, line_number);
        if (!ids.insert(prompt.id).second) {
            throw std::runtime_error("duplicate prompt id in corpus: " + prompt.id);
        }
        prompts.push_back(std::move(prompt));
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading routing prompt corpus " + path.string());
    }
    return prompts;
}

inline uint64_t routing_prompt_signature(const RoutingPrompt& prompt) {
    using namespace routing_profile_detail;
    uint64_t hash = 1469598103934665603ULL;
    hash = fnv1a_bytes(prompt.id.data(), prompt.id.size(), hash);
    hash = fnv1a_bytes(&prompt.max_new_tokens, sizeof(prompt.max_new_tokens), hash);
    for (uint32_t token : prompt.tokens) {
        hash = fnv1a_bytes(&token, sizeof(token), hash);
    }
    return hash;
}

struct RoutingProfileRunConfig {
    std::string aeon_git_commit{"unknown"};
    std::string model_dir;
    std::string input_path;
    std::string corpus_id;
    std::string dataset{"profile"};
    std::string mode{"dynamic"};
    uint32_t num_layers{43};
    uint32_t context_size{4096};
    uint64_t warm_gib{0};
    uint32_t vram_slots{8};
    uint64_t model_config_hash{0};
    uint64_t model_index_hash{0};
    uint64_t input_prompt_count{0};

    std::string compatibility_fingerprint() const {
        std::ostringstream fingerprint;
        fingerprint << "routing-profile-v1"
                    << "|aeon_git_commit=" << aeon_git_commit
                    << "|model_dir=" << model_dir
                    << "|corpus_id=" << corpus_id
                    << "|dataset=" << dataset
                    << "|mode=" << mode
                    << "|layers=" << num_layers
                    << "|experts=" << RoutingCounter::kExpertCount
                    << "|top_k=" << RoutingCounter::kTopK
                    << "|context_size=" << context_size
                    << "|warm_gib=" << warm_gib
                    << "|vram_slots=" << vram_slots
                    << "|model_config_hash=" << model_config_hash
                    << "|model_index_hash=" << model_index_hash
                    << "|phase_rules=prefill-and-decode-v1";
        return fingerprint.str();
    }
};

class RoutingProfileAggregate {
public:
    struct RankingRow {
        RoutingPhase phase;
        uint32_t layer_id;
        uint32_t rank;
        uint32_t expert_id;
        uint64_t selection_count;
        uint64_t total_selections;
        double probability;
        double cumulative_probability;
    };

    explicit RoutingProfileAggregate(uint32_t num_layers)
        : num_layers_(num_layers),
          counts_(static_cast<size_t>(RoutingCounter::kPhaseCount) * num_layers * RoutingCounter::kExpertCount, 0) {}

    void merge(const RoutingCounter& counter) {
        if (counter.num_layers() != num_layers_) {
            throw std::runtime_error("routing counter layer count does not match profile aggregate");
        }
        for (uint32_t phase = 0; phase < RoutingCounter::kPhaseCount; ++phase) {
            for (uint32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
                for (uint32_t expert_id = 0; expert_id < RoutingCounter::kExpertCount; ++expert_id) {
                    counts_[index(static_cast<RoutingPhase>(phase), layer_id, expert_id)] +=
                        counter.selection_count(static_cast<RoutingPhase>(phase), layer_id, expert_id);
                }
            }
        }
    }

    uint64_t selection_count(RoutingPhase phase, uint32_t layer_id, uint32_t expert_id) const {
        if (static_cast<uint32_t>(phase) >= RoutingCounter::kPhaseCount ||
            layer_id >= num_layers_ || expert_id >= RoutingCounter::kExpertCount) {
            return 0;
        }
        return counts_[index(phase, layer_id, expert_id)];
    }

    uint64_t total_selections(RoutingPhase phase, uint32_t layer_id) const {
        uint64_t total = 0;
        for (uint32_t expert_id = 0; expert_id < RoutingCounter::kExpertCount; ++expert_id) {
            total += selection_count(phase, layer_id, expert_id);
        }
        return total;
    }

    const std::vector<uint64_t>& raw_counts() const {
        return counts_;
    }

    void set_raw_counts(std::vector<uint64_t> counts) {
        if (counts.size() != counts_.size()) {
            throw std::runtime_error("routing profile state has an invalid count array size");
        }
        counts_ = std::move(counts);
    }

    uint32_t num_layers() const {
        return num_layers_;
    }

    static const char* phase_name(RoutingPhase phase) {
        return phase == RoutingPhase::Prefill ? "prefill" : "decode";
    }

    std::vector<RankingRow> ranking_rows() const {
        std::vector<RankingRow> rows;
        rows.reserve(static_cast<size_t>(RoutingCounter::kPhaseCount) * num_layers_ * RoutingCounter::kExpertCount);
        for (uint32_t phase_id = 0; phase_id < RoutingCounter::kPhaseCount; ++phase_id) {
            const auto phase = static_cast<RoutingPhase>(phase_id);
            for (uint32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
                std::vector<uint32_t> expert_ids(RoutingCounter::kExpertCount);
                for (uint32_t expert_id = 0; expert_id < RoutingCounter::kExpertCount; ++expert_id) {
                    expert_ids[expert_id] = expert_id;
                }
                std::sort(expert_ids.begin(), expert_ids.end(), [&](uint32_t left, uint32_t right) {
                    const uint64_t left_count = selection_count(phase, layer_id, left);
                    const uint64_t right_count = selection_count(phase, layer_id, right);
                    if (left_count != right_count) {
                        return left_count > right_count;
                    }
                    return left < right;
                });

                const uint64_t total = total_selections(phase, layer_id);
                double cumulative = 0.0;
                for (uint32_t rank = 0; rank < RoutingCounter::kExpertCount; ++rank) {
                    const uint32_t expert_id = expert_ids[rank];
                    const uint64_t count = selection_count(phase, layer_id, expert_id);
                    const double probability = total == 0 ? 0.0 : static_cast<double>(count) / total;
                    cumulative += probability;
                    rows.push_back({
                        phase,
                        layer_id,
                        rank + 1,
                        expert_id,
                        count,
                        total,
                        probability,
                        cumulative,
                    });
                }
            }
        }
        return rows;
    }

    std::string counts_csv() const {
        std::ostringstream output;
        output << "phase,layer_id,expert_id,selection_count\n";
        for (uint32_t phase_id = 0; phase_id < RoutingCounter::kPhaseCount; ++phase_id) {
            const auto phase = static_cast<RoutingPhase>(phase_id);
            for (uint32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
                for (uint32_t expert_id = 0; expert_id < RoutingCounter::kExpertCount; ++expert_id) {
                    output << phase_name(phase) << ',' << layer_id << ',' << expert_id << ','
                           << selection_count(phase, layer_id, expert_id) << '\n';
                }
            }
        }
        return output.str();
    }

    std::string ranking_csv() const {
        std::ostringstream output;
        output << "phase,layer_id,rank,expert_id,selection_count,total_selections,probability,cumulative_probability\n";
        output << std::setprecision(15);
        for (const RankingRow& row : ranking_rows()) {
            output << phase_name(row.phase) << ',' << row.layer_id << ',' << row.rank << ','
                   << row.expert_id << ',' << row.selection_count << ',' << row.total_selections << ','
                   << row.probability << ',' << row.cumulative_probability << '\n';
        }
        return output.str();
    }

private:
    size_t index(RoutingPhase phase, uint32_t layer_id, uint32_t expert_id) const {
        return (static_cast<size_t>(phase) * num_layers_ + layer_id) * RoutingCounter::kExpertCount + expert_id;
    }

    uint32_t num_layers_;
    std::vector<uint64_t> counts_;
};

class RoutingProfileStore {
public:
    RoutingProfileStore(std::filesystem::path output_dir, RoutingProfileRunConfig config)
        : output_dir_(std::move(output_dir)),
          config_(std::move(config)),
          aggregate_(config_.num_layers) {
        std::filesystem::create_directories(output_dir_);
        const auto metadata_path = output_dir_ / "metadata.json";
        const auto state_path = output_dir_ / "state.bin";
        if (std::filesystem::exists(metadata_path)) {
            const std::string metadata = routing_profile_detail::read_text(metadata_path);
            const std::string existing_fingerprint = routing_profile_detail::extract_json_string_field(
                metadata, "compatibility_fingerprint"
            );
            if (existing_fingerprint != config_.compatibility_fingerprint()) {
                throw std::runtime_error("routing profile compatibility fingerprint mismatch; use a new output directory");
            }
            if (!std::filesystem::exists(state_path)) {
                throw std::runtime_error("routing profile metadata exists without state.bin");
            }
            load_state(state_path);
        }
        checkpoint();
    }

    bool is_completed(const RoutingPrompt& prompt) const {
        const auto iterator = completed_.find(prompt.id);
        if (iterator == completed_.end()) {
            return false;
        }
        if (iterator->second != routing_prompt_signature(prompt)) {
            throw std::runtime_error("prompt id was already profiled with different content: " + prompt.id);
        }
        return true;
    }

    void add_prompt(const RoutingPrompt& prompt, const RoutingCounter& counter) {
        if (is_completed(prompt)) {
            throw std::runtime_error("prompt was already profiled: " + prompt.id);
        }
        aggregate_.merge(counter);
        completed_[prompt.id] = routing_prompt_signature(prompt);
        failed_.erase(prompt.id);
    }

    void mark_failed(const RoutingPrompt& prompt, std::string reason) {
        failed_[prompt.id] = std::move(reason);
    }

    void checkpoint() const {
        write_state();
        routing_profile_detail::write_text_atomic(output_dir_ / "counts.csv", aggregate_.counts_csv());
        routing_profile_detail::write_text_atomic(output_dir_ / "ranking.csv", aggregate_.ranking_csv());
        routing_profile_detail::write_text_atomic(output_dir_ / "progress.json", progress_json());
        routing_profile_detail::write_text_atomic(output_dir_ / "metadata.json", metadata_json());
    }

    const RoutingProfileAggregate& aggregate() const {
        return aggregate_;
    }

    size_t completed_count() const {
        return completed_.size();
    }

    size_t failed_count() const {
        return failed_.size();
    }

private:
    void load_state(const std::filesystem::path& state_path) {
        std::ifstream input(state_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("failed to open routing profile state " + state_path.string());
        }

        char magic[8]{};
        input.read(magic, sizeof(magic));
        if (!input || std::string(magic, sizeof(magic)) != "AEONRP01") {
            throw std::runtime_error("invalid routing profile state magic");
        }
        const uint32_t version = routing_profile_detail::read_value<uint32_t>(input);
        const uint32_t layers = routing_profile_detail::read_value<uint32_t>(input);
        const uint64_t count_size = routing_profile_detail::read_value<uint64_t>(input);
        if (version != 1 || layers != config_.num_layers || count_size != aggregate_.raw_counts().size()) {
            throw std::runtime_error("routing profile state is incompatible with the requested run");
        }

        std::vector<uint64_t> counts(count_size);
        for (uint64_t& count : counts) {
            count = routing_profile_detail::read_value<uint64_t>(input);
        }
        aggregate_.set_raw_counts(std::move(counts));

        const uint64_t completed_size = routing_profile_detail::read_value<uint64_t>(input);
        for (uint64_t i = 0; i < completed_size; ++i) {
            const std::string id = routing_profile_detail::read_string(input);
            const uint64_t signature = routing_profile_detail::read_value<uint64_t>(input);
            completed_[id] = signature;
        }

        const uint64_t failed_size = routing_profile_detail::read_value<uint64_t>(input);
        for (uint64_t i = 0; i < failed_size; ++i) {
            const std::string id = routing_profile_detail::read_string(input);
            failed_[id] = routing_profile_detail::read_string(input);
        }
    }

    void write_state() const {
        routing_profile_detail::write_binary_atomic(output_dir_ / "state.bin", [&](std::ostream& output) {
            output.write("AEONRP01", 8);
            routing_profile_detail::write_value<uint32_t>(output, 1);
            routing_profile_detail::write_value<uint32_t>(output, config_.num_layers);
            routing_profile_detail::write_value<uint64_t>(output, aggregate_.raw_counts().size());
            for (uint64_t count : aggregate_.raw_counts()) {
                routing_profile_detail::write_value<uint64_t>(output, count);
            }

            std::vector<std::pair<std::string, uint64_t>> completed(completed_.begin(), completed_.end());
            std::sort(completed.begin(), completed.end(), [](const auto& left, const auto& right) {
                return left.first < right.first;
            });
            routing_profile_detail::write_value<uint64_t>(output, completed.size());
            for (const auto& [id, signature] : completed) {
                routing_profile_detail::write_string(output, id);
                routing_profile_detail::write_value<uint64_t>(output, signature);
            }

            std::vector<std::pair<std::string, std::string>> failed(failed_.begin(), failed_.end());
            std::sort(failed.begin(), failed.end(), [](const auto& left, const auto& right) {
                return left.first < right.first;
            });
            routing_profile_detail::write_value<uint64_t>(output, failed.size());
            for (const auto& [id, reason] : failed) {
                routing_profile_detail::write_string(output, id);
                routing_profile_detail::write_string(output, reason);
            }
        });
    }

    std::string progress_json() const {
        std::vector<std::string> completed;
        completed.reserve(completed_.size());
        for (const auto& [id, signature] : completed_) {
            (void)signature;
            completed.push_back(id);
        }
        std::sort(completed.begin(), completed.end());

        std::vector<std::pair<std::string, std::string>> failed(failed_.begin(), failed_.end());
        std::sort(failed.begin(), failed.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });

        std::ostringstream output;
        output << "{\n  \"completed\": [";
        for (size_t i = 0; i < completed.size(); ++i) {
            if (i != 0) output << ", ";
            output << '"' << routing_profile_detail::json_escape(completed[i]) << '"';
        }
        output << "],\n  \"failed\": [";
        for (size_t i = 0; i < failed.size(); ++i) {
            if (i != 0) output << ", ";
            output << "{\"id\":\"" << routing_profile_detail::json_escape(failed[i].first)
                   << "\",\"error\":\"" << routing_profile_detail::json_escape(failed[i].second) << "\"}";
        }
        output << "]\n}\n";
        return output.str();
    }

    std::string metadata_json() const {
        std::ostringstream output;
        output << "{\n"
               << "  \"format_version\": 1,\n"
               << "  \"profiler_version\": \"routing-profile-v1\",\n"
               << "  \"aeon_git_commit\": \"" << routing_profile_detail::json_escape(config_.aeon_git_commit) << "\",\n"
               << "  \"compatibility_fingerprint\": \""
               << routing_profile_detail::json_escape(config_.compatibility_fingerprint()) << "\",\n"
               << "  \"model_dir\": \"" << routing_profile_detail::json_escape(config_.model_dir) << "\",\n"
               << "  \"input_path\": \"" << routing_profile_detail::json_escape(config_.input_path) << "\",\n"
               << "  \"corpus_id\": \"" << routing_profile_detail::json_escape(config_.corpus_id) << "\",\n"
               << "  \"dataset\": \"" << routing_profile_detail::json_escape(config_.dataset) << "\",\n"
               << "  \"mode\": \"" << routing_profile_detail::json_escape(config_.mode) << "\",\n"
               << "  \"num_layers\": " << config_.num_layers << ",\n"
               << "  \"expert_count\": " << RoutingCounter::kExpertCount << ",\n"
               << "  \"top_k\": " << RoutingCounter::kTopK << ",\n"
               << "  \"context_size\": " << config_.context_size << ",\n"
               << "  \"warm_gib\": " << config_.warm_gib << ",\n"
               << "  \"vram_slots\": " << config_.vram_slots << ",\n"
               << "  \"model_config_hash\": " << config_.model_config_hash << ",\n"
               << "  \"model_index_hash\": " << config_.model_index_hash << ",\n"
               << "  \"input_prompt_count\": " << config_.input_prompt_count << ",\n"
               << "  \"completed_prompts\": " << completed_.size() << ",\n"
               << "  \"failed_prompts\": " << failed_.size() << "\n"
               << "}\n";
        return output.str();
    }

    std::filesystem::path output_dir_;
    RoutingProfileRunConfig config_;
    RoutingProfileAggregate aggregate_;
    std::unordered_map<std::string, uint64_t> completed_;
    std::unordered_map<std::string, std::string> failed_;
};

} // namespace aeon::core