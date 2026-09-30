#pragma once

// -----------------------------------------------------------------------------
// G4 (model) orchestration: the grouped WMMA expert pair over a whole chunk.
//
// Given the chunk's routing (ids and weights, by draw) this runs the token->expert
// permutation, dispatches the grouped expert kernels in batches of at most the weight
// table's length, and writes each token's `moe_accum` through the same slot-ordered
// fixed-order reduce the per-token path uses. The only arithmetic difference from the
// per-token GEMV path is the summation order *inside* one expert's GEMM.
//
// Split out of the executor so it can be driven with any weight source — the tiered
// pool in production, synthetic payloads in a gate — which is what lets the grouped
// batch be compared against the per-token path on identical experts.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/kernels/moe_grouped_dispatch.hpp"
#include "infrastructure/hip_check.hpp"
#include "platform/ops/expert_permutation.hpp"
#include "platform/ops/moe_accumulate.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace aeon::core {

// Scratch the grouped chunk orchestration needs. Owned by the caller and allocated
// once, not per chunk.
struct MoeGroupedBatchScratch {
    int* perm_offsets{nullptr};    // [expert_count + 1]
    int* perm_tokens{nullptr};     // [token_count * slots]
    int* perm_draws{nullptr};      // [token_count * slots]
    float* perm_weights{nullptr};  // [token_count * slots]
    int* batch_offsets{nullptr};   // [kAeonSwizzledMaxExperts + 1]
    half* chunk_hidden{nullptr};   // [kAeonSwizzledMaxExperts, hidden_tokens, intermediate]
    float* chunk_contrib{nullptr}; // [token_count * slots, hidden]

    int intermediate{0};
    int hidden{0};
    // Row capacity of `chunk_hidden` per expert; a chunk whose widest expert exceeds
    // it is refused rather than overflowing.
    int hidden_tokens{0};
};

// `fill(expert_local, j, count, w13, w2)` sets weight-table position `j` for local
// expert `expert_local`, which owns `count` draws. `count` is passed so a filler can
// tell an expert with no draws (whose weights are never read) from one that has draws
// and must have weights supplied.
template <class Filler>
void run_moe_grouped_expert_batch(
    const half* batch_input,
    int input_stride,
    const int* batch_ids,
    const float* batch_weights,
    uint32_t token_count,
    int slots,
    int expert_count,
    float swiglu_limit,
    const MoeGroupedBatchScratch& scratch,
    Filler&& fill,
    half* batch_output,
    int output_stride,
    hipStream_t stream = 0
) {
    if (token_count == 0) {
        return;
    }
    kernel::dispatch_expert_permutation(
        batch_ids, static_cast<int>(token_count), slots, expert_count,
        scratch.perm_offsets, scratch.perm_tokens, scratch.perm_draws, batch_weights,
        scratch.perm_weights, stream);

    // The per-expert bounds are needed on the host to batch the experts and to size the
    // intermediate. The read is small next to the traffic it schedules.
    std::vector<int> offsets(static_cast<size_t>(expert_count) + 1);
    CHECK_HIP(hipMemcpyAsync(offsets.data(), scratch.perm_offsets,
                             offsets.size() * sizeof(int), hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    int widest = 1;
    for (int expert = 0; expert < expert_count; ++expert) {
        widest = std::max(widest, offsets[static_cast<size_t>(expert) + 1] -
                                      offsets[static_cast<size_t>(expert)]);
    }
    if (widest > scratch.hidden_tokens) {
        throw std::invalid_argument(
            "run_moe_grouped_expert_batch: the chunk's widest expert exceeds the scratch");
    }

    const int per_launch = kernel::kAeonSwizzledMaxExperts;
    for (int base = 0; base < expert_count; base += per_launch) {
        const int batch_count = std::min(per_launch, expert_count - base);
        // A batch with no draws at all is skipped; a batch's survivors are packed to
        // the front, so a sparse tail does not widen the dispatch.
        if (offsets[static_cast<size_t>(base)] ==
            offsets[static_cast<size_t>(base) + batch_count]) {
            continue;
        }
        const int first_draw = offsets[static_cast<size_t>(base)];

        kernel::SwizzledW13ExpertPtrs w13{};
        kernel::SwizzledW2ExpertPtrs w2{};
        std::vector<int> rebased(static_cast<size_t>(batch_count) + 1);
        for (int j = 0; j < batch_count; ++j) {
            const int expert = base + j;
            rebased[static_cast<size_t>(j)] =
                offsets[static_cast<size_t>(expert)] - first_draw;
            const int count = offsets[static_cast<size_t>(expert) + 1] -
                              offsets[static_cast<size_t>(expert)];
            fill(expert, j, count, w13, w2);
        }
        rebased[static_cast<size_t>(batch_count)] =
            offsets[static_cast<size_t>(base + batch_count)] - first_draw;

        CHECK_HIP(hipMemcpyAsync(scratch.batch_offsets, rebased.data(),
                                 rebased.size() * sizeof(int), hipMemcpyHostToDevice,
                                 stream));

        kernel::dispatch_aeon_moe_grouped_w13_swiglu_wmma<
            4, 4, 8, kernel::kMoeGroupedMTiles>(
            batch_input, scratch.batch_offsets, scratch.perm_tokens + first_draw, w13,
            scratch.chunk_hidden, batch_count, widest, scratch.intermediate,
            scratch.hidden, swiglu_limit, input_stride, stream);
        kernel::dispatch_aeon_moe_grouped_w2_wmma<4, 8, 4, kernel::kMoeGroupedMTiles>(
            scratch.chunk_hidden, scratch.batch_offsets, scratch.perm_draws + first_draw,
            scratch.perm_weights + first_draw, w2, scratch.chunk_contrib, batch_count,
            widest, scratch.hidden, scratch.intermediate, stream);
    }

    // Per token, the slot-ordered fixed-order reduce, folding in the shared expert the
    // body already wrote into that token's accumulator. Token `t` owns draws
    // `[t*slots, (t+1)*slots)`, which are contiguous rows of the by-draw contributions.
    constexpr int kThreads = 256;
    for (uint32_t token = 0; token < token_count; ++token) {
        half* accum = batch_output + static_cast<size_t>(token) * output_stride;
        kernel::moe_accumulate_fixed_order_kernel
            <<<(scratch.hidden + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                scratch.chunk_contrib +
                    static_cast<size_t>(token) * slots * scratch.hidden,
                slots, accum, accum, scratch.hidden);
    }
}

} // namespace aeon::core
