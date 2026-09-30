#pragma once

// -----------------------------------------------------------------------------
// G2 (architecture) selector: the grouped gated-FFN GEMM, resolved to the build's GPU.
//
// A consumer (the G4 binding) calls `aeon::dispatch_moe_grouped_gate_up` / `_down` and
// names no architecture. The implementation, its tile geometry and its LDS layout are
// chosen here from the build-visible `AEON_ARCH_*` macro that CMake sets, so adding an
// architecture is a new `platform/<arch>/moe_grouped_ffn.hpp` plus a branch below — no
// change to the binding, the format feed or the model epilogue.
//
// Like `platform/dot2.hpp`, this keys on the build macro rather than `__gfx1100__`, so
// the host and device passes resolve to the same kernel.
// -----------------------------------------------------------------------------

#if defined(AEON_ARCH_RDNA3)
#include "platform/rdna3/moe_grouped_ffn.hpp"
namespace aeon {
using rdna3::dispatch_moe_grouped_gate_up;
using rdna3::dispatch_moe_grouped_down;
using rdna3::kGroupedKBlock;
} // namespace aeon
#elif defined(AEON_ARCH_RDNA4)
#include "platform/rdna4/moe_grouped_ffn.hpp"
namespace aeon {
using rdna4::dispatch_moe_grouped_gate_up;
using rdna4::dispatch_moe_grouped_down;
using rdna4::kGroupedKBlock;
} // namespace aeon
#else
#error "platform/moe_grouped_ffn.hpp: no grouped-FFN implementation for this architecture"
#endif
