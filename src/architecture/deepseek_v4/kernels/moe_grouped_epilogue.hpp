#pragma once

// -----------------------------------------------------------------------------
// G4 (model): the DeepSeek-V4 epilogues for the grouped expert GEMM and the
// single-token GEMV expert pair.
//
// `platform/rdna3/moe_grouped_ffn.hpp` and the G3 `aeon_moe_fused_w13.hpp` both apply
// an injected `Epilogue`; this is the model's own. It carries the two model choices
// neither kernel may know:
//
//   * the clamped SwiGLU on the gate/up pair — `silu(min(gate, limit)) *
//     clamp(up, ±limit)`, with `swiglu_limit` a checkpoint config value;
//   * the routing-weight scale on the down projection.
//
// Both are trivial, and that is the point: because they are trivial the temptation is
// to inline them into the kernel. Keeping them here is what lets the same kernel serve
// a model with a different activation, and what keeps a G2 file free of `swiglu_limit`.
// -----------------------------------------------------------------------------

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>

namespace aeon::kernel {

// The clamped SwiGLU on the gate/up pair. Both the grouped kernel and the single-token
// GEMV pair take this as their injected `Epilogue`, so the activation is stated once.
struct Dsv4ClampedSwiGLUEpilogue {
    float limit;

    __device__ half gate_up(float gate, float up) const {
        gate = fminf(gate, limit);
        up = fminf(fmaxf(up, -limit), limit);
        return __float2half((gate / (1.0f + expf(-gate))) * up);
    }
};

// The down projection writes each contribution scaled by its routing weight. Separate
// from the gate/up epilogue so the down path carries no activation and no limit it
// would not use.
struct Dsv4RoutingWeightEpilogue {
    __device__ float down(float accumulator, float weight) const {
        return weight * accumulator;
    }
};

} // namespace aeon::kernel
