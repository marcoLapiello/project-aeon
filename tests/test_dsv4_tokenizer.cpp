#include "text/dsv4_tokenizer.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main() {
    aeon::text::Dsv4Tokenizer tokenizer;
    tokenizer.load("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/tokenizer.aeon");

    assert(tokenizer.vocab_size() == 129280);
    assert(tokenizer.bos_token_id() == 0);
    assert(tokenizer.eos_token_id() == 1);
    assert(tokenizer.user_token_id() == 128803);
    assert(tokenizer.assistant_token_id() == 128804);
    assert(tokenizer.thinking_start_token_id() == 128821);
    assert(tokenizer.thinking_end_token_id() == 128822);
    assert(tokenizer.token_id("<｜User｜>") == 128803);
    assert(tokenizer.token_id("<｜Assistant｜>") == 128804);
    assert(tokenizer.token_id("<think>") == 128821);
    assert(tokenizer.token_id("</think>") == 128822);

    const std::vector<std::pair<std::string, std::vector<uint32_t>>> cases = {
        {"Hello", {19923}},
        {"Hello world", {19923, 2058}},
        {"hello, world!", {33310, 14, 2058, 3}},
        {" leading", {6646}},
        {"trailing ", {19660, 7400, 223}},
        {"\nnext line", {201, 6695, 2562}},
        {"1234", {6895, 22}},
        {"Café", {37, 2797, 619}},
        {"中文测试", {21134, 10251}},
        {"<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>",
            {0, 128803, 19923, 128804, 128822}},
    };

    for (const auto& [text, expected] : cases) {
        const auto actual = tokenizer.encode(text);
        assert(actual == expected);
        assert(tokenizer.decode(actual) == text);
    }

    assert(tokenizer.encode("<｜User｜> literal") == std::vector<uint32_t>({128803, 39248}));
    assert(tokenizer.decode({0, 1}, true).empty());

    std::cout << "[PASS] DSV4 tokenizer artifact, BPE, ByteLevel, and added tokens" << std::endl;
    return 0;
}