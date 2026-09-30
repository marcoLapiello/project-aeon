#pragma once

// -----------------------------------------------------------------------------
// G2 (architecture) selector: the dense fp16 GEMM, resolved to the build's GPU.
//
// Same scheme as `platform/moe_grouped_ffn.hpp`: keyed on the build-visible
// `AEON_ARCH_*` macro so the host and device passes agree, and a new architecture is
// a new `platform/<arch>/dense_gemm.hpp` plus a branch here.
// -----------------------------------------------------------------------------

#if defined(AEON_ARCH_RDNA3)
#include "platform/rdna3/dense_gemm.hpp"
namespace aeon {
using rdna3::dispatch_dense_gemm_fp16;
} // namespace aeon
#else
#error "platform/dense_gemm.hpp: no dense GEMM implementation for this architecture"
#endif
