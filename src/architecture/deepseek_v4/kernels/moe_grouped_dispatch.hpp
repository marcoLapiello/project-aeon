#pragma once

// -----------------------------------------------------------------------------
// G4 (model) binding: the dispatcher that composes the grouped expert pair.
//
// This is the one place the three groups meet, and it is deliberate that the meeting
// happens at the top. The arch kernel, the weight feed and the model epilogue are each
// chosen here and nowhere else:
//
//   * `platform/rdna3/moe_grouped_ffn.hpp` — G2, the WMMA loop;
//   * `backend/swizzled_w4a16/kernels/swizzled_w4a16_feed.hpp` — G3, the weight decode;
//   * `kernels/moe_grouped_epilogue.hpp` — G4, the activation and scale.
//
// Nothing below this file names anything above it, which is what keeps a new format or
// a new architecture from reaching into the model tree. The public entry points keep
// the shapes the certified gates already call, so swapping the implementation behind
// them does not move what the gates measure.
//
// The tuning constants — waves per workgroup and the M window — are template
// parameters here rather than inside the kernel, because which shape is fastest is a
// property of the target architecture and the workload, not of the algorithm. The
// values the callers pass are from the measured sweep; changing them is a tuning
// decision made at this level.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/kernels/moe_grouped_epilogue.hpp"
#include "backend/swizzled_w4a16/kernels/swizzled_w4a16_feed.hpp"
#include "platform/moe_grouped_ffn.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace aeon::kernel {

// Default M window: how many token tiles share one dequantized slab. This is a tuning
// value the dispatcher owns — `4` (64 tokens) is the measured time optimum up to a
// 256-token chunk, past which more reuse buys bytes rather than speed. It is here
// rather than in the kernel because which shape is fastest is a property of the target
// and the workload, not of the algorithm.
inline constexpr int kMoeGroupedMTiles = 4;

// Gate/up plus the clamped SwiGLU, over a token->expert permutation.
template <int WAVES, int RPW, int LPR, int MTILES>
inline void dispatch_aeon_moe_grouped_w13_swiglu_wmma(
    const half* activation,
    const int* expert_offsets,
    const int* token_indices,
    const SwizzledW13ExpertPtrs& weights,
    half* expert_hidden,
    int expert_count,
    int expert_hidden_tokens,
    int N,
    int K,
    float swiglu_limit,
    hipStream_t stream = 0
) {
    using Feed = SwizzledW4A16Feed<RPW, LPR>;
    aeon::dispatch_moe_grouped_gate_up<WAVES, MTILES, Feed,
                                       Dsv4ClampedSwiGLUEpilogue>(
        activation, expert_offsets, token_indices, weights, expert_hidden, expert_count,
        expert_hidden_tokens, N, K, Dsv4ClampedSwiGLUEpilogue{swiglu_limit}, stream);
}

// The down projection, scaled by the routing weight and written per draw.
template <int WAVES, int RPW, int LPR, int MTILES>
inline void dispatch_aeon_moe_grouped_w2_wmma(
    const half* expert_hidden,
    const int* expert_offsets,
    const int* draw_indices,
    const float* draw_weights,
    const SwizzledW2ExpertPtrs& weights,
    float* contrib,
    int expert_count,
    int expert_hidden_tokens,
    int N,
    int K,
    hipStream_t stream = 0
) {
    using Feed = SwizzledW4A16Feed<RPW, LPR>;
    aeon::dispatch_moe_grouped_down<WAVES, MTILES, Feed,
                                    Dsv4RoutingWeightEpilogue>(
        expert_hidden, expert_offsets, draw_indices, draw_weights, weights, contrib,
        expert_count, expert_hidden_tokens, N, K, Dsv4RoutingWeightEpilogue{}, stream);
}

} // namespace aeon::kernel
