#include "infrastructure/text/text_generation.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
    {
        uint32_t calls = 0;
        aeon::text::GenerationOptions options;
        options.max_new_tokens = 4;
        options.eos_token_id = 99;
        const auto result = aeon::text::generate_token_ids(
            {10, 11}, options,
            [&calls](uint32_t, uint32_t, bool) {
                ++calls;
                return calls == 2 ? 50u : (calls == 3 ? 99u : 42u);
            }
        );
        assert(result.token_ids == std::vector<uint32_t>({50, 99}));
        assert(result.stop_reason == aeon::text::StopReason::Eos);
        assert(calls == 3);
    }

    {
        uint32_t calls = 0;
        aeon::text::GenerationOptions options;
        options.max_new_tokens = 4;
        options.eos_token_id = 99;
        const auto result = aeon::text::generate_token_ids(
            {10, 11}, options,
            [&calls](uint32_t, uint32_t, bool) {
                ++calls;
                return 99u;
            }
        );
        assert(result.token_ids == std::vector<uint32_t>({99}));
        assert(result.stop_reason == aeon::text::StopReason::Eos);
        assert(calls == 2);
    }

    {
        uint32_t calls = 0;
        aeon::text::GenerationOptions options;
        options.max_new_tokens = 3;
        options.eos_token_id = 99;
        const auto result = aeon::text::generate_token_ids(
            {10, 11}, options,
            [&calls](uint32_t, uint32_t, bool) {
                ++calls;
                return 40u + calls;
            }
        );
        assert(result.token_ids == std::vector<uint32_t>({42, 43, 44}));
        assert(result.stop_reason == aeon::text::StopReason::MaxNewTokens);
        assert(calls == 4);
    }

    {
        aeon::text::GenerationOptions options;
        options.max_new_tokens = 5;
        options.context_limit = 4;
        const auto result = aeon::text::generate_token_ids(
            {10, 11}, options,
            [](uint32_t, uint32_t, bool) { return 7u; }
        );
        assert(result.token_ids == std::vector<uint32_t>({7, 7}));
        assert(result.stop_reason == aeon::text::StopReason::ContextLimit);
    }

    bool rejected_empty = false;
    try {
        (void)aeon::text::generate_token_ids({}, {}, [](uint32_t, uint32_t, bool) { return 0u; });
    } catch (const std::invalid_argument&) {
        rejected_empty = true;
    }
    assert(rejected_empty);

    bool rejected_zero_limit = false;
    try {
        aeon::text::GenerationOptions options;
        options.max_new_tokens = 0;
        (void)aeon::text::generate_token_ids({1}, options, [](uint32_t, uint32_t, bool) { return 0u; });
    } catch (const std::invalid_argument&) {
        rejected_zero_limit = true;
    }
    assert(rejected_zero_limit);

    assert(std::string(aeon::text::stop_reason_name(aeon::text::StopReason::Eos)) == "eos");
    assert(std::string(aeon::text::stop_reason_name(aeon::text::StopReason::ContextLimit)) == "context_limit");
    std::cout << "[PASS] EOS-aware generation loop and stop reasons" << std::endl;
    return 0;
}