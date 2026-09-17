#pragma once

// The one place `CHECK_HIP` is defined.
//
// It reached seven copies in `src/` alone, each wrapped in `#ifndef CHECK_HIP` so
// that whichever header is included first wins and the rest are inert. New code
// should include this file rather than add an eighth: the copies are textually the
// same, so the guarded macro stays compatible with all of them, and the older
// definitions can migrate here mechanically without touching behaviour.
//
// A macro rather than a function on purpose: the message carries `__FILE__` and
// `__LINE__`, which a call would collapse to the helper's own location.

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)
#endif
