#pragma once

#include "core/aeon_loader.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

namespace aeon::core {

// Unified VRAM pool for S_hot routed INT4-W4A16 experts
// Shared dynamically across all transformer layers
class UnifiedVRAMExpertPool {
public:
    uint32_t num_slots{0};

    // Unified contiguous device allocations for num_slots experts:
    // W1: [num_slots, 2048, 512] uint32_t (4MB each) + [num_slots, 2048, 128] half (512KB each)
    // W2: [num_slots, 4096, 256] uint32_t (4MB each) + [num_slots, 4096, 64]  half (512KB each)
    // W3: [num_slots, 2048, 512] uint32_t (4MB each) + [num_slots, 2048, 128] half (512KB each)
    uint32_t* d_w1_packed{nullptr};
    half*     d_w1_scale{nullptr};
    uint32_t* d_w2_packed{nullptr};
    half*     d_w2_scale{nullptr};
    uint32_t* d_w3_packed{nullptr};
    half*     d_w3_scale{nullptr};

    // Sub-tensor byte sizes per single expert
    static constexpr size_t W1_PACKED_BYTES = 2048 * 512 * sizeof(uint32_t); // 4,194,304 B
    static constexpr size_t W1_SCALE_BYTES  = 2048 * 128 * sizeof(half);     // 524,288 B
    static constexpr size_t W2_PACKED_BYTES = 4096 * 256 * sizeof(uint32_t); // 4,194,304 B
    static constexpr size_t W2_SCALE_BYTES  = 4096 * 64 * sizeof(half);      // 524,288 B
    static constexpr size_t W3_PACKED_BYTES = 2048 * 512 * sizeof(uint32_t); // 4,194,304 B
    static constexpr size_t W3_SCALE_BYTES  = 2048 * 128 * sizeof(half);     // 524,288 B
    static constexpr size_t TOTAL_EXPERT_BYTES = W1_PACKED_BYTES + W1_SCALE_BYTES +
                                                 W2_PACKED_BYTES + W2_SCALE_BYTES +
                                                 W3_PACKED_BYTES + W3_SCALE_BYTES;

    UnifiedVRAMExpertPool() = default;

    explicit UnifiedVRAMExpertPool(uint32_t slots) {
        allocate(slots);
    }

    ~UnifiedVRAMExpertPool() {
        free();
    }

    UnifiedVRAMExpertPool(const UnifiedVRAMExpertPool&) = delete;
    UnifiedVRAMExpertPool& operator=(const UnifiedVRAMExpertPool&) = delete;

    UnifiedVRAMExpertPool(UnifiedVRAMExpertPool&& other) noexcept {
        move_from(std::move(other));
    }

    UnifiedVRAMExpertPool& operator=(UnifiedVRAMExpertPool&& other) noexcept {
        if (this != &other) {
            free();
            move_from(std::move(other));
        }
        return *this;
    }

    void allocate(uint32_t slots) {
        if (slots == 0) return;
        num_slots = slots;

        size_t total_w1_p = static_cast<size_t>(num_slots) * W1_PACKED_BYTES;
        size_t total_w1_s = static_cast<size_t>(num_slots) * W1_SCALE_BYTES;
        size_t total_w2_p = static_cast<size_t>(num_slots) * W2_PACKED_BYTES;
        size_t total_w2_s = static_cast<size_t>(num_slots) * W2_SCALE_BYTES;
        size_t total_w3_p = static_cast<size_t>(num_slots) * W3_PACKED_BYTES;
        size_t total_w3_s = static_cast<size_t>(num_slots) * W3_SCALE_BYTES;

        CHECK_HIP(hipMalloc(&d_w1_packed, total_w1_p));
        CHECK_HIP(hipMalloc(&d_w1_scale,  total_w1_s));
        CHECK_HIP(hipMalloc(&d_w2_packed, total_w2_p));
        CHECK_HIP(hipMalloc(&d_w2_scale,  total_w2_s));
        CHECK_HIP(hipMalloc(&d_w3_packed, total_w3_p));
        CHECK_HIP(hipMalloc(&d_w3_scale,  total_w3_s));
    }

    void free() {
        if (d_w1_packed) { (void)hipFree(d_w1_packed); d_w1_packed = nullptr; }
        if (d_w1_scale)  { (void)hipFree(d_w1_scale);  d_w1_scale = nullptr; }
        if (d_w2_packed) { (void)hipFree(d_w2_packed); d_w2_packed = nullptr; }
        if (d_w2_scale)  { (void)hipFree(d_w2_scale);  d_w2_scale = nullptr; }
        if (d_w3_packed) { (void)hipFree(d_w3_packed); d_w3_packed = nullptr; }
        if (d_w3_scale)  { (void)hipFree(d_w3_scale);  d_w3_scale = nullptr; }
        num_slots = 0;
    }

    // Pointers for a specific slot index
    uint32_t* get_w1_packed(uint32_t slot_idx) const {
        return d_w1_packed + static_cast<size_t>(slot_idx) * (2048 * 512);
    }
    half* get_w1_scale(uint32_t slot_idx) const {
        return d_w1_scale + static_cast<size_t>(slot_idx) * (2048 * 128);
    }
    uint32_t* get_w2_packed(uint32_t slot_idx) const {
        return d_w2_packed + static_cast<size_t>(slot_idx) * (4096 * 256);
    }
    half* get_w2_scale(uint32_t slot_idx) const {
        return d_w2_scale + static_cast<size_t>(slot_idx) * (4096 * 64);
    }
    uint32_t* get_w3_packed(uint32_t slot_idx) const {
        return d_w3_packed + static_cast<size_t>(slot_idx) * (2048 * 512);
    }
    half* get_w3_scale(uint32_t slot_idx) const {
        return d_w3_scale + static_cast<size_t>(slot_idx) * (2048 * 128);
    }

    // Stream an expert payload from contiguous host memory into this slot asynchronously
    void upload_from_host_expert(
        uint32_t slot_idx,
        const uint8_t* host_expert_payload,
        hipStream_t stream = 0
    ) {
        if (slot_idx >= num_slots) {
            throw std::runtime_error("UnifiedVRAMExpertPool: Invalid slot index " + std::to_string(slot_idx));
        }

        const uint8_t* p = host_expert_payload;
        CHECK_HIP(hipMemcpyAsync(get_w1_packed(slot_idx), p + AEON_W1_PACKED_OFFSET, W1_PACKED_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(get_w1_scale(slot_idx),  p + AEON_W1_SCALE_OFFSET,  W1_SCALE_BYTES,  hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(get_w2_packed(slot_idx), p + AEON_W2_PACKED_OFFSET, W2_PACKED_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(get_w2_scale(slot_idx),  p + AEON_W2_SCALE_OFFSET,  W2_SCALE_BYTES,  hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(get_w3_packed(slot_idx), p + AEON_W3_PACKED_OFFSET, W3_PACKED_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(get_w3_scale(slot_idx),  p + AEON_W3_SCALE_OFFSET,  W3_SCALE_BYTES,  hipMemcpyHostToDevice, stream));
    }

    // Stream an expert from individual host pointers (Safetensors host source) into this slot asynchronously
    void upload_from_pointers(
        uint32_t slot_idx,
        const uint32_t* w1_packed, const half* w1_scale,
        const uint32_t* w2_packed, const half* w2_scale,
        const uint32_t* w3_packed, const half* w3_scale,
        hipStream_t stream = 0
    ) {
        if (slot_idx >= num_slots) {
            throw std::runtime_error("UnifiedVRAMExpertPool: Invalid slot index " + std::to_string(slot_idx));
        }

        CHECK_HIP(hipMemcpyAsync(get_w1_packed(slot_idx), w1_packed, W1_PACKED_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(get_w1_scale(slot_idx),  w1_scale,  W1_SCALE_BYTES,  hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(get_w2_packed(slot_idx), w2_packed, W2_PACKED_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(get_w2_scale(slot_idx),  w2_scale,  W2_SCALE_BYTES,  hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(get_w3_packed(slot_idx), w3_packed, W3_PACKED_BYTES, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(get_w3_scale(slot_idx),  w3_scale,  W3_SCALE_BYTES,  hipMemcpyHostToDevice, stream));
    }

private:
    void move_from(UnifiedVRAMExpertPool&& other) {
        num_slots   = other.num_slots;
        d_w1_packed = other.d_w1_packed;
        d_w1_scale  = other.d_w1_scale;
        d_w2_packed = other.d_w2_packed;
        d_w2_scale  = other.d_w2_scale;
        d_w3_packed = other.d_w3_packed;
        d_w3_scale  = other.d_w3_scale;

        other.num_slots   = 0;
        other.d_w1_packed = nullptr;
        other.d_w1_scale  = nullptr;
        other.d_w2_packed = nullptr;
        other.d_w2_scale  = nullptr;
        other.d_w3_packed = nullptr;
        other.d_w3_scale  = nullptr;
    }
};

// Backwards-compatibility alias during refactoring
using GlobalVRAMExpertPool = UnifiedVRAMExpertPool;

} // namespace aeon::core
