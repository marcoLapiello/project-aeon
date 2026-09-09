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

using TokenStep = std::function<uint32_t(uint32_t token_id, uint32_t position, bool prefill)>;

GenerationResult generate_token_ids(
    const std::vector<uint32_t>& prompt,
    const GenerationOptions& options,
    const TokenStep& step
);

const char* stop_reason_name(StopReason reason);

} // namespace aeon::text