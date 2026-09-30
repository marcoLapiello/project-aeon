#pragma once

// -----------------------------------------------------------------------------
// G4: one dense projection `Y[T, N] = X[T, K] . W[N, K]^T` over `tokens` rows.
//
// Above `kDenseGemmMinTokens` rows the WMMA GEMM reads each weight once for the whole
// group; below it the per-token GEMV wins (measured: a 16-row, 64-wide launch is 3x
// slower on WMMA, a 37-row one 1.7x faster), and decode, which is always one row,
// keeps the exact GEMV it had. `x_stride` / `y_stride` are the row pitches, so a caller
// whose per-token buffers are padded reads and writes them in place.
// -----------------------------------------------------------------------------

#include "platform/dense_gemm.hpp"
#include "platform/ops/gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace aeon::core {

constexpr int kDenseGemmMinTokens = 32;

inline void project_dense(
    const __half* x, const __half* w, __half* y,
    int tokens, int in_dim, int out_dim, int x_stride, int y_stride, hipStream_t stream) {
    if (tokens >= kDenseGemmMinTokens) {
        dispatch_dense_gemm_fp16(x, w, y, tokens, in_dim, out_dim, x_stride, y_stride, stream);
        return;
    }
    for (int t = 0; t < tokens; ++t) {
        hipLaunchKernelGGL(
            kernel::gemv_fp16_kernel, dim3(out_dim, 1), dim3(32), 0, stream,
            x + static_cast<size_t>(t) * x_stride, w, y + static_cast<size_t>(t) * y_stride,
            in_dim, x_stride);
    }
}

} // namespace aeon::core
