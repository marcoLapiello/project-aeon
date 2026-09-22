#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace aeon::text {

enum class StopReason {
    Eos,
    MaxNewTokens,
    ContextLimit,
    Error,
};

struct GenerationOptions {
    uint32_t max_new_tokens{256};
    uint32_t eos_token_id{1};
    uint32_t context_limit{0};
    bool stop_on_eos{true};
    bool thinking_mode{false};
};

struct GenerationResult {
    std::vector<uint32_t> token_ids;
    StopReason stop_reason{StopReason::Error};
};

// The step the decode loop calls: one token in, the next token out. `prefill` is
// true for a prompt token and false for a generated one.
using TokenStep = std::function<uint32_t(uint32_t token_id, uint32_t position, bool prefill)>;

// The prefill as **one unit**: the whole prompt in, the first generated token out.
//
// The per-token shape above and the windowed shape below are two mechanisms for
// the same step — one `forward_token` per prompt token, or one layer-major
// `forward_window` over the whole prompt (expert-streaming plan Step 6 D-a). The
// loop therefore takes the *step* and not the mechanism, which is what keeps it
// from growing a second decode loop for the swept path.
using PromptPrefill = std::function<uint32_t(const std::vector<uint32_t>& prompt)>;

GenerationResult generate_token_ids(
    const std::vector<uint32_t>& prompt,
    const GenerationOptions& options,
    const PromptPrefill& prefill,
    const TokenStep& decode
);

// The per-token shape: prefill walks the prompt one token at a time through
// `step`, then decode continues through the same `step` with `prefill = false`.
// Equivalent to passing a `PromptPrefill` that performs that walk, and kept as its
// own overload because it is the contract `test_text_generation` pins.
GenerationResult generate_token_ids(
    const std::vector<uint32_t>& prompt,
    const GenerationOptions& options,
    const TokenStep& step
);

const char* stop_reason_name(StopReason reason);

} // namespace aeon::text