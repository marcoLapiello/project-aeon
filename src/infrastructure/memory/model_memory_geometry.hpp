#pragma once

// -----------------------------------------------------------------------------
// The model's memory geometry: what the budget policy needs that only an
// architecture knows.
//
// The budget engine owns the policy — the VRAM/host tier sizing, the feasibility
// gate, the max-viable-context search — and reads nothing model-specific. The
// architecture supplies this struct: four scalar counts, the residual carry per
// token, and the attention-state cost as a **function of context**. The engine
// calls that function at the context it is evaluating (and inside its binary
// search) and never sees a layer spec.
//
// This is the seam that lets the memory budget leave the model tree: the policy
// is engine code, the per-layer attention layout is model code, and this struct is
// the only thing that crosses between them.
// -----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <functional>

namespace aeon::core {

// The attention-state cost of one context, broken down by the owner of each term.
// Filled by the architecture's geometry function; held here because it is the unit
// the engine's policy reasons about.
struct AttentionStateMemory {
    size_t local_kv_bytes{0};
    size_t compressed_kv_bytes{0};
    size_t compressor_state_bytes{0};
    size_t indexer_state_bytes{0};
    size_t metadata_bytes{0};
    size_t layer_state_bytes{0};
    size_t rope_bytes{0};

    size_t total_bytes() const {
        return layer_state_bytes + rope_bytes;
    }
};

// The architecture-supplied geometry. Every field is a plain number except the
// attention function, which is the one thing the engine cannot derive itself.
struct ModelMemoryGeometry {
    int64_t num_hidden_layers{0};
    int64_t routed_experts{0};
    int64_t experts_per_tok{0};
    int64_t max_position_embeddings{0};

    // The residual carry per token: `hc_dim * (2 + 4)` for V4. The engine scales it
    // by the configured window, so it never needs to know what the residual is.
    size_t prefill_carry_bytes_per_token{0};

    // Attention state as a function of context length. Must be pure and callable
    // with any context the engine probes (the max-viable-context search halves the
    // range down from `max_position_embeddings`).
    std::function<AttentionStateMemory(uint32_t context_size)> attention_state_memory;
};

} // namespace aeon::core
