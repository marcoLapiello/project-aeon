#pragma once

// -----------------------------------------------------------------------------
// G2 (architecture) selector: the half2 dot product, resolved to the build's GPU.
//
// A consumer (the G3 swizzled decoder, most notably) calls `aeon::fdot2` and names no
// architecture. Which implementation that resolves to is decided here, from the
// build-visible `AEON_ARCH_*` macro that CMake sets, so adding an architecture is a new
// `platform/<arch>/` file plus a branch below — no consumer changes and no edit to a
// format or model file.
//
// The selector must key on a macro that is the **same in the host and device passes**
// (see `AeonToolchain.cmake`), which is why it does not test `__gfx1100__`: that macro
// exists only in the device pass, so the two passes would instantiate different
// functions and fail to link.
// -----------------------------------------------------------------------------

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(AEON_ARCH_RDNA3)
#include "platform/rdna3/dot2.hpp"
namespace aeon {
using rdna3::fdot2;
} // namespace aeon
#elif defined(AEON_ARCH_RDNA4)
#include "platform/rdna4/dot2.hpp"
namespace aeon {
using rdna4::fdot2;
} // namespace aeon
#else
#error "platform/dot2.hpp: no dot2 implementation for this architecture"
#endif
