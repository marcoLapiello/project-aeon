#pragma once

// -----------------------------------------------------------------------------
// The DeepSeek-V4 memory geometry: the architecture's half of the budget seam.
//
// The budget engine (`infrastructure/core/memory_budget_engine.hpp`) owns the
// policy and reads a neutral `ModelMemoryGeometry`. This header builds that struct
// from the V4 config: the four counts, the residual carry per token, and — the one
// thing only the architecture can supply — the attention-state cost as a function
// of context, walked over the resolved layer specs and their per-layer layout.
//
// It is deliberately the *only* place the budget meets `V4LayerStateLayout` and
// `V4ModelSpec`, so the engine can stay model-free.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/spec/config.hpp"
#include "architecture/deepseek_v4/layer/v4_layer_state.hpp"
#include "architecture/deepseek_v4/spec/v4_model_spec.hpp"
#include "infrastructure/core/model_memory_geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace aeon::core {

inline ModelMemoryGeometry make_v4_memory_geometry(const DeepSeekV4Config& config) {
    // Resolve once, not per probe: the geometry function is called many times by
    // the max-viable-context search, and resolving the layer specs validates the
    // whole config each time.
    const std::vector<V4LayerSpec> layer_specs = V4ModelSpec::resolve_layers(config);
    const uint32_t rope_half = static_cast<uint32_t>(config.qk_rope_head_dim / 2);

    ModelMemoryGeometry geometry;
    geometry.num_hidden_layers = config.num_hidden_layers;
    geometry.routed_experts = config.n_routed_experts;
    geometry.experts_per_tok = config.num_experts_per_tok;
    geometry.max_position_embeddings = config.max_position_embeddings;
    geometry.prefill_carry_bytes_per_token =
        static_cast<size_t>(config.hc_mult) * static_cast<size_t>(config.hidden_size) *
        (sizeof(uint16_t) + sizeof(float));
    geometry.attention_state_memory =
        [layer_specs, rope_half](uint32_t context_size) {
            if (context_size == 0) {
                throw std::invalid_argument(
                    "make_v4_memory_geometry: attention context cannot be 0");
            }
            AttentionStateMemory memory;
            for (const auto& layer_spec : layer_specs) {
                const auto layout = V4LayerStateLayout::from_spec(layer_spec, context_size);
                memory.local_kv_bytes += layout.local_cache_bytes();
                memory.compressed_kv_bytes += layout.compressed_cache_bytes();
                memory.compressor_state_bytes += layout.compressor_state_bytes();
                memory.indexer_state_bytes +=
                    layout.indexer_cache_bytes() + layout.indexer_workspace_bytes();
                memory.metadata_bytes +=
                    layout.local_metadata_bytes() + layout.compressed_metadata_bytes();
                memory.layer_state_bytes += layout.total_device_bytes();
            }
            memory.rope_bytes =
                static_cast<size_t>(context_size) * rope_half * sizeof(float) * 4;
            return memory;
        };
    return geometry;
}

} // namespace aeon::core
