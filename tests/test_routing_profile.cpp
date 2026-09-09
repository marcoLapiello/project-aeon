#include "core/routing_profile.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    const auto root = std::filesystem::temp_directory_path() / "aeon-routing-profile-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    const auto corpus_path = root / "prompts.jsonl";
    {
        std::ofstream corpus(corpus_path);
        corpus << R"({"id":"prompt_001","tokens":[1,100,256],"max_new_tokens":4})" << '\n';
        corpus << R"({"id":"prompt_002","tokens":[2,101],"max_new_tokens":2})" << '\n';
    }

    const auto prompts = aeon::core::load_routing_prompts(corpus_path);
    assert(prompts.size() == 2);
    assert(prompts[0].id == "prompt_001");
    assert(prompts[0].tokens.size() == 3);

    aeon::core::RoutingProfileRunConfig config;
    config.model_dir = "model";
    config.input_path = corpus_path.string();
    config.corpus_id = "test-corpus";
    config.mode = "aeon";
    config.num_layers = 1;
    config.context_size = 64;
    config.vram_slots = 8;

    aeon::core::RoutingProfileStore store(root / "run", config);
    aeon::core::RoutingCounter first_counter(1);
    const int32_t first_ids[] = {5, 2, 5, 7, 9, 11};
    first_counter.record(aeon::core::RoutingPhase::Prefill, 0, 0, first_ids);
    first_counter.record(aeon::core::RoutingPhase::Decode, 0, 1, first_ids);
    store.add_prompt(prompts[0], first_counter);
    store.checkpoint();

    aeon::core::RoutingProfileStore resumed(root / "run", config);
    assert(resumed.completed_count() == 1);
    assert(resumed.is_completed(prompts[0]));
    assert(resumed.aggregate().selection_count(aeon::core::RoutingPhase::Prefill, 0, 5) == 2);

    aeon::core::RoutingCounter second_counter(1);
    const int32_t second_ids[] = {3, 4, 6, 8, 10, 12};
    second_counter.record(aeon::core::RoutingPhase::Prefill, 0, 0, second_ids);
    second_counter.record(aeon::core::RoutingPhase::Decode, 0, 1, second_ids);
    resumed.add_prompt(prompts[1], second_counter);
    resumed.checkpoint();

    aeon::core::RoutingProfileStore final_store(root / "run", config);
    assert(final_store.completed_count() == 2);
    assert(final_store.aggregate().total_selections(aeon::core::RoutingPhase::Prefill, 0) == 12);
    assert(final_store.aggregate().total_selections(aeon::core::RoutingPhase::Decode, 0) == 12);

    std::ifstream ranking(root / "run" / "ranking.csv");
    size_t ranking_lines = 0;
    std::string line;
    while (std::getline(ranking, line)) {
        ++ranking_lines;
    }
    assert(ranking_lines == 1 + 2 * 256);

    std::ifstream summary(root / "run" / "summary.csv");
    size_t summary_lines = 0;
    while (std::getline(summary, line)) {
        ++summary_lines;
    }
    assert(summary_lines == 1 + 2);
    summary.close();
    std::filesystem::remove(root / "run" / "summary.csv");
    aeon::core::RoutingProfileStore::regenerate_summary(root / "run");
    assert(std::filesystem::exists(root / "run" / "summary.csv"));

    bool duplicate_rejected = false;
    {
        std::ofstream duplicate_corpus(root / "duplicate.jsonl");
        duplicate_corpus << R"({"id":"same","tokens":[1],"max_new_tokens":1})" << '\n';
        duplicate_corpus << R"({"id":"same","tokens":[2],"max_new_tokens":1})" << '\n';
    }
    try {
        (void)aeon::core::load_routing_prompts(root / "duplicate.jsonl");
    } catch (const std::exception&) {
        duplicate_rejected = true;
    }
    assert(duplicate_rejected);

    std::cout << "[SUCCESS] Routing profile parsing, ranking, checkpoint, and resume validated.\n";
    std::filesystem::remove_all(root);
    return 0;
}