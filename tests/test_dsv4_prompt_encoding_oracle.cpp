// Step 0 prompt-encoding oracle (checkpoint plan Stage B).
//
// The DeepSeek-V4 artifact ships its own prompt encoder and four golden
// input/output vectors. This test renders each vector through our encoder and
// compares byte-for-byte against the artifact's golden output.
//
// The artifact is the authority for prompt encoding, so a mismatch is a real
// defect, not a tolerance question. Prompt encoding is not deferrable: an "almost
// right" template silently changes every prefix and defeats prefix caching.
//
// The inputs are parsed with the prompt JSON codec rather than the shared
// `core/json.hpp`, because tool schemas must be re-emitted with their original key
// order and with integers distinguished from reals — exactly what the encoder has
// to reproduce.

#include "architecture/deepseek_v4/text/dsv4_prompt_encoder.hpp"
#include "architecture/deepseek_v4/text/dsv4_prompt_json.hpp"
#include "architecture/deepseek_v4/text/dsv4_tokenizer.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using aeon::text::Dsv4PromptEncoder;
using aeon::text::Dsv4PromptMessage;
using aeon::text::Dsv4PromptOptions;
using aeon::text::Dsv4Role;
using aeon::text::Dsv4ThinkingMode;
using aeon::text::Dsv4ToolCall;
using aeon::text::Dsv4ToolDefinition;
using aeon::text::PromptJson;

namespace {

constexpr int kVectorCount = 4;

std::string read_text_file(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

// The snapshot hash is not fixed, so locate encoding/tests by search.
fs::path find_vectors_dir() {
    const fs::path root = "models/DeepSeek-V4-Flash-0731-INT4-W4A16";
    if (!fs::exists(root)) throw std::runtime_error("model directory not found: " + root.string());
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_directory()) continue;
        if (entry.path().filename() != "tests") continue;
        if (entry.path().parent_path().filename() != "encoding") continue;
        if (fs::exists(entry.path() / "test_input_1.json")) return entry.path();
    }
    throw std::runtime_error("encoding/tests vectors not found under " + root.string());
}

std::string string_field(const PromptJson& object, const char* key) {
    const PromptJson* value = object.find(key);
    return value != nullptr && value->is_string() ? value->text : std::string();
}

bool bool_field(const PromptJson& object, const char* key) {
    const PromptJson* value = object.find(key);
    if (value == nullptr) return false;
    if (value->kind == PromptJson::Kind::Bool) return value->boolean;
    if (value->kind == PromptJson::Kind::Int) return value->integer != 0;
    return false;
}

Dsv4Role parse_role(const std::string& role) {
    if (role == "system") return Dsv4Role::System;
    if (role == "developer") return Dsv4Role::Developer;
    if (role == "user") return Dsv4Role::User;
    if (role == "assistant") return Dsv4Role::Assistant;
    if (role == "latest_reminder") return Dsv4Role::LatestReminder;
    if (role == "tool") return Dsv4Role::Tool;
    throw std::runtime_error("unknown role: " + role);
}

Dsv4PromptMessage convert_message(const PromptJson& node) {
    Dsv4PromptMessage message;
    message.role = parse_role(string_field(node, "role"));
    message.content = string_field(node, "content");
    message.reasoning_content = string_field(node, "reasoning_content");
    message.tool_call_id = string_field(node, "tool_call_id");
    message.task = string_field(node, "task");
    message.wo_eos = bool_field(node, "wo_eos");
    message.mask = bool_field(node, "mask");

    if (const PromptJson* tools = node.find("tools"); tools != nullptr && tools->is_array()) {
        for (const PromptJson& tool : tools->array) {
            const PromptJson* function = tool.find("function");
            if (function == nullptr) continue;
            Dsv4ToolDefinition definition;
            // Keep the function object serialized in reference form.
            definition.function_json = function->to_python_json();
            message.tools.push_back(std::move(definition));
        }
    }

    if (const PromptJson* calls = node.find("tool_calls");
        calls != nullptr && calls->is_array()) {
        for (const PromptJson& call : calls->array) {
            const PromptJson* function = call.find("function");
            if (function == nullptr) continue;
            aeon::text::Dsv4ToolCall converted;
            converted.id = string_field(call, "id");
            converted.name = string_field(*function, "name");
            converted.arguments = string_field(*function, "arguments");
            message.tool_calls.push_back(std::move(converted));
        }
    }

    return message;
}

std::string first_difference(const std::string& actual, const std::string& expected) {
    const size_t limit = std::min(actual.size(), expected.size());
    size_t offset = 0;
    while (offset < limit && actual[offset] == expected[offset]) ++offset;

    std::ostringstream out;
    out << "first divergence at byte " << offset;
    if (offset >= limit) out << " (one string is a prefix of the other)";
    out << "\nexpected: "
        << (expected.size() > offset ? expected.substr(offset, 160) : std::string("<eof>"));
    out << "\nactual:   "
        << (actual.size() > offset ? actual.substr(offset, 160) : std::string("<eof>"));
    return out.str();
}

} // namespace

int main() {
    try {
        const fs::path vectors_dir = find_vectors_dir();
        std::cout << "[Test] DSV4 prompt-encoding oracle (Step 0)\n"
                  << "  vectors: " << vectors_dir.string() << "\n" << std::endl;

        aeon::text::Dsv4Tokenizer tokenizer;
        tokenizer.load("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/tokenizer.aeon");
        const Dsv4PromptEncoder encoder(tokenizer);

        size_t matched = 0;
        bool any_failed = false;

        for (int index = 1; index <= kVectorCount; ++index) {
            const std::string stem = "test_input_" + std::to_string(index) + ".json";
            const std::string gold_name = "test_output_" + std::to_string(index) + ".txt";
            const PromptJson document =
                PromptJson::parse(read_text_file(vectors_dir / stem));
            const std::string gold = read_text_file(vectors_dir / gold_name);

            const PromptJson* messages = &document;
            const PromptJson* wrapper = document.find("messages");
            if (wrapper != nullptr && wrapper->is_array()) messages = wrapper;
            if (!messages->is_array()) throw std::runtime_error(stem + ": messages is not an array");

            std::vector<Dsv4PromptMessage> converted;
            for (const PromptJson& node : messages->array) {
                converted.push_back(convert_message(node));
            }

            // The reference test suite attaches the document's top-level `tools`
            // to the first message before encoding (test_case_1 does exactly
            // `messages[0]["tools"] = td["tools"]`). Mirror that here.
            if (const PromptJson* top_tools = document.find("tools");
                top_tools != nullptr && top_tools->is_array() && !converted.empty()) {
                for (const PromptJson& tool : top_tools->array) {
                    const PromptJson* function = tool.find("function");
                    if (function == nullptr) continue;
                    Dsv4ToolDefinition definition;
                    definition.function_json = function->to_python_json();
                    converted.front().tools.push_back(std::move(definition));
                }
            }

            // Vector 4 is the only chat-mode vector; 1-3 are thinking mode.
            Dsv4PromptOptions options;
            options.thinking_mode =
                (index == 4) ? Dsv4ThinkingMode::Chat : Dsv4ThinkingMode::Thinking;

            std::string rendered;
            try {
                rendered = encoder.encode(converted, options);
            } catch (const std::exception& error) {
                std::cout << "  vector " << index << ": FAIL  encoder threw: " << error.what()
                          << std::endl;
                any_failed = true;
                continue;
            }

            if (rendered == gold) {
                ++matched;
                std::cout << "  vector " << index << ": PASS  byte-identical" << std::endl;
            } else {
                any_failed = true;
                std::cout << "  vector " << index << ": FAIL\n"
                          << first_difference(rendered, gold) << std::endl;
            }
        }

        std::cout << "\n  " << matched << "/" << kVectorCount << " vectors byte-identical."
                  << std::endl;

        if (any_failed) {
            std::cout << "\n[FAIL] prompt encoding does not match the artifact oracle" << std::endl;
            return 1;
        }
        std::cout << "\n[SUCCESS] prompt encoding matches the artifact oracle on all vectors."
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << std::endl;
        return 1;
    }
}
