#pragma once

// -----------------------------------------------------------------------------
// G2 (architecture) selector: tiled causal attention, resolved to the build's GPU.
//
// Same scheme as `platform/dense_gemm.hpp` and `platform/moe_grouped_ffn.hpp`:
// keyed on the build-visible `AEON_ARCH_*` macro so the host and device passes
// agree, and a new architecture is a new `platform/<arch>/tiled_causal_attention.hpp`
// plus a branch here.
// -----------------------------------------------------------------------------

#if defined(AEON_ARCH_RDNA3)
#include "platform/rdna3/tiled_causal_attention.hpp"
namespace aeon {
using rdna3::CausalAttentionBlock;
using rdna3::dispatch_causal_attention_fp16;
using rdna3::dispatch_causal_attention_split_fp16;
} // namespace aeon
#else
#error "platform/tiled_causal_attention.hpp: no causal attention implementation for this architecture"
#endif
