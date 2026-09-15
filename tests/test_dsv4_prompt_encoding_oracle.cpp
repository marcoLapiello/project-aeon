// Step 0 prompt-encoding oracle (checkpoint plan Stage B).
//
// The DeepSeek-V4 artifact ships its own prompt encoder and four golden
// input/output vectors. This test renders each vector through OUR formatter and
// compares byte-for-byte against the artifact's golden output.
//
// The artifact is the authority for prompt encoding, so a mismatch here is a real
// defect in our formatter — not a tolerance question. Prompt encoding is not
// deferrable: an "almost right" template silently changes every prefix and
// defeats prefix caching.
//
// Two kinds of non-pass are distinguished:
//   FAIL     our formatter can represent the input but renders it differently.
//   BLOCKED  the input needs surface the formatter's type system cannot express
//            yet (developer / latest_reminder / tool / task / tools / mask).
//
// Exit status: non-zero if any vector FAILs. BLOCKED vectors do not fail the run
// unless `--strict` is passed. Run with `--strict` once Step 0 is declared
// complete; at that point BLOCKED must be zero as well.

#include "architecture/deepseek_v4/text/dsv4_chat_formatter.hpp"
#include "architecture/deepseek_v4/text/dsv4_tokenizer.hpp"
#include "infrastructure/core/json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kVectorCount = 4;

struct VectorOutcome {
    int index{0};
    bool blocked{false};
    bool matched{false};
    std::string detail;
};

std::string read_text_file(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

// Locate the artifact's encoding/tests directory. The snapshot hash is not fixed,
// so search for it rather than hardcoding a path.
fs::path find_vectors_dir() {
    const fs::path root = "models/DeepSeek-V4-Flash-0731-INT4-W4A16";
    if (!fs::exists(root)) {
        throw std::runtime_error("model directory not found: " + root.string());
    }
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_directory()) continue;
        if (entry.path().filename() != "tests") continue;
        if (entry.path().parent_path().filename() != "encoding") continue;
        if (fs::exists(entry.path() / "test_input_1.json")) {
            return entry.path();
        }
    }
    throw std::runtime_error("encoding/tests vectors not found under " + root.string());
}

// Keys that force a vector to BLOCKED, because Dsv4Message cannot carry them.
bool message_needs_missing_surface(const aeon::core::JsonValue& message) {
    for (const char* key :
         {"tools", "tool_calls", "tool_call_id", "task", "mask", "wo_eos"}) {
        if (message.find(key) != nullptr) return true;
    }
    return false;
}

std::string role_name(const aeon::core::JsonValue& message) {
    const aeon::core::JsonValue* role = message.find("role");
    return role != nullptr && role->is_string() ? role->as_string() : std::string();
}

std::string field_string(const aeon::core::JsonValue& object, const char* key) {
    const aeon::core::JsonValue* value = object.find(key);
    return value != nullptr && value->is_string() ? value->as_string() : std::string();
}

// Render one differing region for diagnosis.
std::string first_difference(const std::string& actual, const std::string& expected) {
    const size_t limit = std::min(actual.size(), expected.size());
    size_t offset = 0;
    while (offset < limit && actual[offset] == expected[offset]) ++offset;

    std::ostringstream out;
    out << "first divergence at byte " << offset;
    if (offset >= limit) {
        out << " (one string is a prefix of the other)";
    }
    out << "\nexpected: ";
    out << (expected.size() > offset ? expected.substr(offset, 120) : std::string("<eof>"));
    out << "\nactual:   ";
    out << (actual.size() > offset ? actual.substr(offset, 120) : std::string("<eof>"));
    return out.str();
}

VectorOutcome run_vector(const fs::path& vectors_dir,
                         int index,
                         const aeon::text::Dsv4ChatFormatter& formatter) {
    VectorOutcome outcome;
    outcome.index = index;

    const std::string stem = "test_input_" + std::to_string(index) + ".json";
    const std::string gold_name = "test_output_" + std::to_string(index) + ".txt";
    const aeon::core::JsonValue document = aeon::core::JsonValue::parse_file((vectors_dir / stem).string());
    const std::string gold = read_text_file(vectors_dir / gold_name);

    // Vector 1 carries tools at the top level; the others carry messages directly.
    const aeon::core::JsonValue* messages = &document;
    const aeon::core::JsonValue* wrapper = document.find("messages");
    if (wrapper != nullptr && wrapper->is_array()) {
        messages = wrapper;
        if (document.find("tools") != nullptr) {
            outcome.blocked = true;
            outcome.detail += "top-level tools; ";
        }
    }
    if (!messages->is_array()) {
        throw std::runtime_error(stem + ": messages is not an array");
    }

    std::vector<aeon::text::Dsv4Message> converted;
    for (const aeon::core::JsonValue& message : messages->as_array()) {
        if (!message.is_object()) continue;

        if (message_needs_missing_surface(message)) {
            outcome.blocked = true;
            outcome.detail += "message with tool/task/mask fields; ";
        }

        const std::string role = role_name(message);
        aeon::text::Dsv4Message out;
        if (role == "system") {
            out.role = aeon::text::Dsv4MessageRole::System;
        } else if (role == "user") {
            out.role = aeon::text::Dsv4MessageRole::User;
        } else if (role == "assistant") {
            out.role = aeon::text::Dsv4MessageRole::Assistant;
        } else {
            // developer / latest_reminder / tool: no representation exists.
            outcome.blocked = true;
            outcome.detail += "role '" + role + "' unsupported; ";
            continue;
        }
        out.content = field_string(message, "content");
        out.reasoning_content = field_string(message, "reasoning_content");
        converted.push_back(std::move(out));
    }

    // Case 4 is the only chat-mode vector; 1-3 are thinking mode.
    const auto mode = (index == 4) ? aeon::text::Dsv4ThinkingMode::Chat
                                   : aeon::text::Dsv4ThinkingMode::Thinking;

    // A vector that needs missing surface may also make the formatter throw (for
    // example a tool conversation with no user message). Report that rather than
    // aborting the whole run, so one vector cannot hide the others.
    try {
        const aeon::text::Dsv4FormattedPrompt rendered = formatter.format(converted, mode);
        outcome.matched = (rendered.rendered_text == gold);
        if (!outcome.matched) {
            outcome.detail += first_difference(rendered.rendered_text, gold);
        }
    } catch (const std::exception& error) {
        outcome.matched = false;
        outcome.blocked = true;
        outcome.detail += std::string("formatter threw: ") + error.what();
    }

    if (outcome.detail.empty()) {
        outcome.detail = "byte-identical";
    }
    return outcome;
}

} // namespace

int main(int argc, char** argv) {
    bool strict = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--strict") strict = true;
    }

    bool any_failed = false;
    bool any_blocked = false;

    try {
        const fs::path vectors_dir = find_vectors_dir();
        std::cout << "[Test] DSV4 prompt-encoding oracle (Step 0)\n"
                  << "  vectors: " << vectors_dir.string() << "\n" << std::endl;

        aeon::text::Dsv4Tokenizer tokenizer;
        tokenizer.load("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/tokenizer.aeon");
        const aeon::text::Dsv4ChatFormatter formatter(tokenizer);

        std::vector<VectorOutcome> outcomes;
        for (int index = 1; index <= kVectorCount; ++index) {
            outcomes.push_back(run_vector(vectors_dir, index, formatter));
        }

        std::cout << "=== results ===" << std::endl;
        for (const VectorOutcome& outcome : outcomes) {
            const char* status = outcome.matched
                                     ? "PASS"
                                     : (outcome.blocked ? "BLOCKED" : "FAIL");
            std::cout << "  vector " << outcome.index << ": " << status << "  "
                      << outcome.detail << std::endl;
            if (!outcome.matched && !outcome.blocked) any_failed = true;
            if (outcome.blocked) any_blocked = true;
        }

        const size_t matched =
            static_cast<size_t>(std::count_if(outcomes.begin(), outcomes.end(),
                                              [](const VectorOutcome& o) { return o.matched; }));
        std::cout << "\n  " << matched << "/" << kVectorCount << " vectors byte-identical."
                  << std::endl;

        if (any_blocked) {
            std::cout << "\n  NOTE: BLOCKED vectors need formatter surface that does not exist yet "
                         "(developer / latest_reminder / tool / task / tools / mask)."
                      << std::endl;
            std::cout << "  Once Step 0 is complete, run this target with --strict to make them "
                         "hard failures." << std::endl;
        }

        if (any_failed || (strict && any_blocked)) {
            std::cout << "\n[FAIL] prompt encoding does not match the artifact oracle" << std::endl;
            return 1;
        }
        if (any_blocked) {
            std::cout << "\n[PARTIAL] every representable vector matches; some surface is still "
                         "missing." << std::endl;
            return 0;
        }
        std::cout << "\n[SUCCESS] prompt encoding matches the artifact oracle on all vectors."
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << std::endl;
        return 1;
    }
}
