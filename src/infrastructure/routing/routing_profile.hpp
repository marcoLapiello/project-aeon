#pragma once

// -----------------------------------------------------------------------------
// Routing profiles: the prompt parser, the run config, the aggregate, and the
// persistent store. The JSONL / atomic-file / hashing helpers they use live in
// `routing_profile_json.hpp` (namespace `routing_profile_detail`), included here.
// -----------------------------------------------------------------------------

#include "infrastructure/routing/routing_counter.hpp"
#include "infrastructure/routing/routing_profile_json.hpp"

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
    uint32_t num_layers{0};
    uint32_t context_size{4096};
    uint64_t warm_gib{0};
    uint64_t model_config_hash{0};
    uint64_t model_index_hash{0};
    uint64_t input_prompt_count{0};

    std::string compatibility_fingerprint() const {
        std::ostringstream fingerprint;
        fingerprint << "routing-profile-v2"
                    << "|aeon_git_commit=" << aeon_git_commit
                    << "|model_dir=" << model_dir
                    << "|corpus_id=" << corpus_id
                    << "|dataset=" << dataset
                    << "|layers=" << num_layers
                    << "|experts=" << RoutingCounter::kExpertCount
                    << "|top_k=" << RoutingCounter::kTopK
                    << "|context_size=" << context_size
                    << "|warm_gib=" << warm_gib
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

    std::string summary_csv() const {
        constexpr std::array<uint32_t, 5> coverage_cutoffs{4, 8, 12, 16, 32};
        const auto rows = ranking_rows();

        std::ostringstream output;
        output << "phase,layer_id,total_selections,top1_expert,top1_probability,"
                  "top4_coverage,top8_coverage,top12_coverage,top16_coverage,"
                  "top32_coverage,top12_experts\n";
        output << std::setprecision(15);
        for (uint32_t phase_id = 0; phase_id < RoutingCounter::kPhaseCount; ++phase_id) {
            const auto phase = static_cast<RoutingPhase>(phase_id);
            for (uint32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
                const size_t base = (static_cast<size_t>(phase_id) * num_layers_ + layer_id) *
                                    RoutingCounter::kExpertCount;
                output << phase_name(phase) << ',' << layer_id << ','
                       << rows[base].total_selections << ','
                       << rows[base].expert_id << ',' << rows[base].probability;
                for (uint32_t cutoff : coverage_cutoffs) {
                    output << ',' << rows[base + cutoff - 1].cumulative_probability;
                }
                output << ',';
                for (uint32_t rank = 0; rank < 12; ++rank) {
                    if (rank != 0) {
                        output << ';';
                    }
                    output << rows[base + rank].expert_id;
                }
                output << '\n';
            }
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
                if (config_.num_layers == 0) {
                        throw std::invalid_argument("routing profile requires a positive model layer count");
                }
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

    static void regenerate_summary(const std::filesystem::path& output_dir) {
        const auto state_path = output_dir / "state.bin";
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
        const uint64_t expected_count_size = static_cast<uint64_t>(RoutingCounter::kPhaseCount) * layers *
                                              RoutingCounter::kExpertCount;
        if (version != 1 || layers == 0 || count_size != expected_count_size) {
            throw std::runtime_error("routing profile state is incompatible with summary regeneration");
        }

        RoutingProfileAggregate aggregate(layers);
        std::vector<uint64_t> counts(static_cast<size_t>(count_size));
        for (uint64_t& count : counts) {
            count = routing_profile_detail::read_value<uint64_t>(input);
        }
        aggregate.set_raw_counts(std::move(counts));
        routing_profile_detail::write_text_atomic(output_dir / "summary.csv", aggregate.summary_csv());
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
        routing_profile_detail::write_text_atomic(output_dir_ / "summary.csv", aggregate_.summary_csv());
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
               << "  \"profiler_version\": \"routing-profile-v2\",\n"
               << "  \"aeon_git_commit\": \"" << routing_profile_detail::json_escape(config_.aeon_git_commit) << "\",\n"
               << "  \"compatibility_fingerprint\": \""
               << routing_profile_detail::json_escape(config_.compatibility_fingerprint()) << "\",\n"
               << "  \"model_dir\": \"" << routing_profile_detail::json_escape(config_.model_dir) << "\",\n"
               << "  \"input_path\": \"" << routing_profile_detail::json_escape(config_.input_path) << "\",\n"
               << "  \"corpus_id\": \"" << routing_profile_detail::json_escape(config_.corpus_id) << "\",\n"
               << "  \"dataset\": \"" << routing_profile_detail::json_escape(config_.dataset) << "\",\n"
               << "  \"num_layers\": " << config_.num_layers << ",\n"
               << "  \"expert_count\": " << RoutingCounter::kExpertCount << ",\n"
               << "  \"top_k\": " << RoutingCounter::kTopK << ",\n"
               << "  \"context_size\": " << config_.context_size << ",\n"
               << "  \"warm_gib\": " << config_.warm_gib << ",\n"
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