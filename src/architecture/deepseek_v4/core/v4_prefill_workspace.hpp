#pragma once

// -----------------------------------------------------------------------------
// The layer-major prefill working set.
//
// Two buffers, and the distinction between them is the whole reason they are
// separate: the **chunk workspace** holds per-op temporaries for the rows in flight
// and is recycled as layers advance, while the **carry** holds the residual being
// transformed and must survive all 43 layers of a pass.
//
// This struct owns both, plus the configured window/chunk it was sized for. It holds
// no reference to the host: `allocate` and `ensure_prefill_carry` take the layer
// vector and the model config they size themselves from, so the workspace can be
// read, allocated and freed without the assembly around it.
//
// Of the two buffers, only the carry is engine-owned: it is the neutral
// `infrastructure/core/prefill_carry.hpp` (`PrefillCarry`), sized by the model's
// residual width. The chunk workspace is the model's own `V4LayerBodyBatchScratch`
// — its layout is written in the model's dimensions and its worst case is taken
// across the model's layers, so it stays here.
// -----------------------------------------------------------------------------

#include "infrastructure/core/prefill_carry.hpp"
#include "infrastructure/core/runtime_config.hpp"
#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body_batch.hpp"
#include "infrastructure/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::core {

class V4PrefillWorkspace {
public:
    V4PrefillWorkspace() = default;
    ~V4PrefillWorkspace() { free(); }
    V4PrefillWorkspace(const V4PrefillWorkspace&) = delete;
    V4PrefillWorkspace& operator=(const V4PrefillWorkspace&) = delete;

    // The per-layer chunk workspace. `V4LayerBodyBatchScratch` sizes itself from a
    // layer's own capacities (the indexer's candidate scores and top-k), so left to
    // itself it would be re-allocated whenever the layer changes. When the window
    // workspace was allocated at load for the worst case across the layers
    // (`allocate`), that one buffer already covers every layer and this is a no-op
    // — which is what makes the layer-major pass allocation-free.
    void ensure_batch_scratch(V4Layer& layer, uint32_t layer_id, uint32_t count) {
        if (prefill_workspace_ready_ && count <= batch_scratch_.token_count()) return;
        if (batch_scratch_count_ == count && batch_scratch_layer_ == layer_id) return;
        batch_scratch_.allocate(layer, count);
        batch_scratch_layer_ = layer_id;
        batch_scratch_count_ = count;
    }

    V4LayerBodyBatchScratch& batch_scratch() noexcept { return batch_scratch_; }
    const V4LayerBodyBatchScratch& batch_scratch() const noexcept { return batch_scratch_; }

    // ---- the prefill workspace, derived from the knobs ----------------------
    //
    // The residual carry and the batch scratch are functions of the configured
    // window `W` and chunk `C`, so both are **derived and allocated once, at load**,
    // rather than grown lazily on the first window. Two reasons, and the second is
    // the one that matters:
    //
    //   * a window is a known size, so an allocation inside it can only fail after
    //     work has begun — and a mid-prefill failure has no clean recovery;
    //   * the budget has to know the figure up front (`vram_prefill_carry_bytes`),
    //     or the Hot pool is sized against VRAM the workspace then takes.
    //
    // The batch scratch covers all 43 layers, so its allocation is the worst-case
    // layout, not any one layer's. Its exact size is checked against the budget's
    // allowance here, so a configuration that would overrun fails with a named
    // message instead of quietly shrinking the expert pool.
    void allocate(uint32_t window_tokens, uint32_t chunk_tokens,
                  const std::vector<V4Layer>& layers,
                  const DeepSeekV4Config& config) {
        if (window_tokens == 0 || chunk_tokens == 0) {
            throw std::invalid_argument(
                "V4PrefillWorkspace: the prefill window and chunk must be positive");
        }
        if (chunk_tokens > V4LayerBodyBatchScratch::kMaxTokens) {
            throw std::invalid_argument(
                "V4PrefillWorkspace: prefill chunk " + std::to_string(chunk_tokens) +
                " exceeds the body's row cap of " +
                std::to_string(V4LayerBodyBatchScratch::kMaxTokens));
        }

        ensure_prefill_carry(window_tokens, config);

        // Worst case across the layers: the composed row-set, the indexer's
        // candidate scores and its top-k are each the maximum any layer needs.
        uint32_t max_compressed_capacity = 0;
        uint32_t max_index_topk = 0;
        uint32_t max_local_capacity = 0;
        for (const auto& layer : layers) {
            const auto& layout = layer.state_layout();
            max_compressed_capacity = std::max(max_compressed_capacity, layout.compressed_capacity);
            max_index_topk = std::max(max_index_topk, layout.index_topk);
            max_local_capacity = std::max(max_local_capacity, layout.local_capacity);
        }
        batch_scratch_.allocate_capacity(max_compressed_capacity, max_index_topk,
                                         max_local_capacity, chunk_tokens);
        const size_t allowance = batch_scratch_allowance_bytes(chunk_tokens);
        if (batch_scratch_.bytes() > allowance) {
            throw std::runtime_error(
                "V4PrefillWorkspace: the prefill batch scratch is " +
                std::to_string(batch_scratch_.bytes() / (1024 * 1024)) +
                " MiB but the budget allows " +
                std::to_string(allowance / (1024 * 1024)) +
                " MiB for chunk " + std::to_string(chunk_tokens) +
                " — raise BATCH_SCRATCH_BYTES_PER_ROW or lower prefill_chunk");
        }

        prefill_window_tokens_ = window_tokens;
        prefill_chunk_tokens_ = chunk_tokens;
        prefill_workspace_ready_ = true;
    }

    // What was allocated, for the report and the gates.
    uint32_t prefill_window_tokens() const noexcept { return prefill_window_tokens_; }
    uint32_t prefill_chunk_tokens() const noexcept { return prefill_chunk_tokens_; }
    size_t prefill_carry_bytes() const noexcept { return carry_.bytes(); }
    size_t prefill_batch_scratch_bytes() const noexcept { return batch_scratch_.bytes(); }

    // The residual carry: one window's worth of per-token residual, both fp16 and
    // fp32, held in VRAM for the whole layer-major pass. Grows only, so a window
    // of a given size is allocated once and reused by every later pass that fits.
    //
    // The buffer is the neutral `PrefillCarry`; the model supplies only its width
    // (`hc_mult × hidden_size`), so the growth policy and the sizing live in the
    // engine and no model type reaches them.
    void ensure_prefill_carry(uint32_t tokens, const DeepSeekV4Config& config) {
        carry_.ensure(tokens, carry_dim(config));
    }

    half* prefill_carry_half() noexcept { return carry_.half_ptr(); }
    float* prefill_carry() noexcept { return carry_.float_ptr(); }
    uint32_t prefill_carry_tokens() const noexcept { return carry_.tokens(); }

    void free() noexcept {
        batch_scratch_.free();
        batch_scratch_layer_ = UINT32_MAX;
        batch_scratch_count_ = 0;
        carry_.free();
        prefill_window_tokens_ = 0;
        prefill_chunk_tokens_ = 0;
        prefill_workspace_ready_ = false;
    }

private:
    // The residual width the carry is sized by: the HC streams times the hidden
    // size. A model property, so it is resolved here and passed as a scalar.
    static uint32_t carry_dim(const DeepSeekV4Config& config) noexcept {
        return static_cast<uint32_t>(config.hc_mult) *
               static_cast<uint32_t>(config.hidden_size);
    }

    V4LayerBodyBatchScratch batch_scratch_;
    uint32_t batch_scratch_layer_{UINT32_MAX};
    uint32_t batch_scratch_count_{0};
    PrefillCarry carry_;
    // The configured knobs and whether the workspace was allocated at load for them.
    uint32_t prefill_window_tokens_{0};
    uint32_t prefill_chunk_tokens_{0};
    bool prefill_workspace_ready_{false};
};

} // namespace aeon::core
