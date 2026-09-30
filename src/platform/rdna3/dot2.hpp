#pragma once

// -----------------------------------------------------------------------------
// G2 (architecture): the RDNA3 half2 dot-product primitive, `v_dot2_f32_f16`.
//
// Two fp16 pairs and an fp32 accumulator in one instruction. It is the inner
// operation of every W4A16 weight decode, but the instruction belongs to the
// architecture, not to the format: a G3 decoder calls this primitive instead of
// carrying an `#if __gfx11__` of its own. The portable fallback keeps a format gate
// buildable on a host pass and on architectures without the instruction, so a kernel
// that uses it degrades in speed rather than failing to compile.
// -----------------------------------------------------------------------------

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace aeon::rdna3 {

using half2v = _Float16 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float fdot2(half2 a, half2 b, float accumulator) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2v, a),
                                  __builtin_bit_cast(half2v, b),
                                  accumulator, false);
#else
    return accumulator + __half2float(a.x) * __half2float(b.x) +
           __half2float(a.y) * __half2float(b.y);
#endif
}

} // namespace aeon::rdna3
