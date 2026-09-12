#include "architecture/deepseek_v4/text/dsv4_chat_formatter.hpp"

#include <cassert>
#include <iostream>

int main() {
    aeon::text::Dsv4Tokenizer tokenizer;
    tokenizer.load("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/tokenizer.aeon");
    aeon::text::Dsv4ChatFormatter formatter(tokenizer);

    const auto bos = tokenizer.decode({tokenizer.bos_token_id()});
    const auto user = tokenizer.decode({tokenizer.user_token_id()});
    const auto assistant = tokenizer.decode({tokenizer.assistant_token_id()});
    const auto thinking_start = tokenizer.decode({tokenizer.thinking_start_token_id()});
    const auto thinking_end = tokenizer.decode({tokenizer.thinking_end_token_id()});
    const auto eos = tokenizer.decode({tokenizer.eos_token_id()});

    const std::vector<aeon::text::Dsv4Message> one_user = {
        {aeon::text::Dsv4MessageRole::User, "Hello", ""},
    };
    const auto chat = formatter.format(one_user, aeon::text::Dsv4ThinkingMode::Chat);
    assert(chat.rendered_text == bos + user + "Hello" + assistant + thinking_end);
    assert(chat.token_ids == std::vector<uint32_t>({0, 128803, 19923, 128804, 128822}));

    const auto thinking = formatter.format(one_user, aeon::text::Dsv4ThinkingMode::Thinking);
    assert(thinking.rendered_text == bos + user + "Hello" + assistant + thinking_start);
    assert(thinking.token_ids == std::vector<uint32_t>({0, 128803, 19923, 128804, 128821}));

    const std::vector<aeon::text::Dsv4Message> multi_turn = {
        {aeon::text::Dsv4MessageRole::System, "You are helpful.", ""},
        {aeon::text::Dsv4MessageRole::User, "Hello", ""},
        {aeon::text::Dsv4MessageRole::Assistant, "Hi there!", "old reasoning"},
        {aeon::text::Dsv4MessageRole::User, "What next?", ""},
    };
    const auto multi = formatter.format(multi_turn, aeon::text::Dsv4ThinkingMode::Thinking);
    assert(multi.rendered_text ==
        bos + "You are helpful." + user + "Hello" + assistant + thinking_end +
        "Hi there!" + eos + user + "What next?" + assistant + thinking_start);

    const auto preserved = formatter.format(multi_turn, aeon::text::Dsv4ThinkingMode::Thinking, false);
    assert(preserved.rendered_text ==
        bos + "You are helpful." + user + "Hello" + assistant + thinking_start +
        "old reasoning" + thinking_end + "Hi there!" + eos +
        user + "What next?" + assistant + thinking_start);

    bool rejected_empty = false;
    try {
        (void)formatter.format({}, aeon::text::Dsv4ThinkingMode::Chat);
    } catch (const std::invalid_argument&) {
        rejected_empty = true;
    }
    assert(rejected_empty);

    std::cout << "[PASS] DSV4 chat and thinking formatter contract" << std::endl;
    return 0;
}