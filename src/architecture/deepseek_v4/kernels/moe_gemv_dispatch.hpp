#pragma once

// -----------------------------------------------------------------------------
// G4 (model) binding: the single-token GEMV expert pair, composed for DeepSeek-V4.
//
// The mirror of `moe_grouped_dispatch.hpp` for the per-token path (decode, and prefill
// below the grouped crossover). The G3 kernels (`aeon_moe_fused_w13.hpp`,
// `aeon_moe_fused_w2.hpp`) own the swizzled GEMV and the weight tables; here the
// model's activation is supplied and a model-named entry point is exposed, so a caller
// never names a raw activation at a G3 call site and a G3 file never names the model.
//
// This header is also what a G4 caller includes: it pulls in both G3 kernels, so the
// weight-table types come with it.
//
// The down projection carries no model math — its scale is the routing weight — so it
// is forwarded unchanged.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/kernels/moe_grouped_epilogue.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace aeon::kernel {

// Gate/up plus the model's clamped SwiGLU, per expert, one token.
template <int WAVES, int RPW, int LPR, int ITERS>
inline void dispatch_dsv4_moe_gemv_w13_swiglu(
    const half* activation,
    const SwizzledW13ExpertPtrs& weights,
    half* expert_hidden,
    float* output_f32,
    int output_dim,
    int expert_count,
    int N,
    int K,
    float swiglu_limit,
    hipStream_t stream = 0
) {
    dispatch_aeon_moe_fused_gate_up<WAVES, RPW, LPR, ITERS>(
        activation, weights, expert_hidden, output_f32, output_dim, expert_count, N, K,
        Dsv4ClampedSwiGLUEpilogue{swiglu_limit}, stream);
}

} // namespace aeon::kernel
