#pragma once

// -----------------------------------------------------------------------------
// The layer-major prefill residual carry.
//
// One window's worth of per-token residual, held in VRAM in both fp16 and fp32
// for the whole layer-major pass (the fp16 half is what the layer body reads and
// writes; the fp32 half is the widened accumulator the HC pre-mix consumes). It
// must survive all layers of a pass, which is what distinguishes it from the
// chunk workspace that is recycled as layers advance.
//
// It is sized by a single scalar — the residual width, a model's `hc_mult ×
// hidden_size` — so it is pure buffer management and knows nothing of any model.
// The model that uses it supplies the width and the token count.
// -----------------------------------------------------------------------------

#include "infrastructure/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace aeon::core {

class PrefillCarry {
public:
    PrefillCarry() = default;
    ~PrefillCarry() { free(); }
    PrefillCarry(const PrefillCarry&) = delete;
    PrefillCarry& operator=(const PrefillCarry&) = delete;

    // Grows only: a carry of a given size is allocated once and reused by every
    // later pass that fits, so a steady-state workload never reallocates.
    void ensure(uint32_t tokens, uint32_t carry_dim) {
        if (tokens == 0) {
            throw std::invalid_argument("PrefillCarry: a prefill carry cannot be zero tokens");
        }
        if (tokens <= tokens_) return;

        free();
        CHECK_HIP(hipMalloc(&d_half_,
                            static_cast<size_t>(tokens) * carry_dim * sizeof(half)));
        CHECK_HIP(hipMalloc(&d_float_,
                            static_cast<size_t>(tokens) * carry_dim * sizeof(float)));
        tokens_ = tokens;
        carry_dim_ = carry_dim;
    }

    half* half_ptr() noexcept { return d_half_; }
    float* float_ptr() noexcept { return d_float_; }
    uint32_t tokens() const noexcept { return tokens_; }

    // The VRAM the carry occupies: both halves, `tokens × carry_dim` each.
    size_t bytes() const noexcept {
        return static_cast<size_t>(tokens_) * carry_dim_ *
               (sizeof(uint16_t) + sizeof(float));
    }

    void free() noexcept {
        if (d_half_) { (void)hipFree(d_half_); d_half_ = nullptr; }
        if (d_float_) { (void)hipFree(d_float_); d_float_ = nullptr; }
        tokens_ = 0;
        carry_dim_ = 0;
    }

private:
    half* d_half_{nullptr};
    float* d_float_{nullptr};
    uint32_t tokens_{0};
    uint32_t carry_dim_{0};
};

} // namespace aeon::core
