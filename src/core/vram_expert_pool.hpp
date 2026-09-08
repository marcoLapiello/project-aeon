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
//
// Physical layout (Expert Review Step 2): one contiguous device allocation
// holding num_slots back-to-back expert regions, each AEON_EXPERT_BYTES and
// byte-identical to the .aeon host/staging layout. A full expert therefore
// uploads with a single hipMemcpyAsync instead of six per-sub-tensor copies:
// fewer API submissions, one sequential SDMA burst, and hardware-friendly
// prefetching on both the PCIe and NVMe paths.
class UnifiedVRAMExpertPool {
public:
    uint32_t num_slots{0};

    // Contiguous per-slot device storage: slot i spans
    // [d_experts + i * AEON_EXPERT_BYTES, + AEON_EXPERT_BYTES)
    uint8_t* d_experts{nullptr};

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
    static_assert(TOTAL_EXPERT_BYTES == AEON_EXPERT_BYTES,
                  "Per-slot device layout must match the .aeon expert container layout");

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
        CHECK_HIP(hipMalloc(&d_experts, static_cast<size_t>(num_slots) * TOTAL_EXPERT_BYTES));
    }

    void free() {
        if (d_experts) { (void)hipFree(d_experts); d_experts = nullptr; }
        num_slots = 0;
    }

    // Contiguous base pointer for a slot (start of its 13.5 MiB expert region)
    uint8_t* get_slot_base(uint32_t slot_idx) const {
        return d_experts + static_cast<size_t>(slot_idx) * TOTAL_EXPERT_BYTES;
    }

    // Sub-tensor views, derived from the contiguous slot base using the
    // canonical .aeon layout offsets.
    uint32_t* get_w1_packed(uint32_t slot_idx) const {
        return reinterpret_cast<uint32_t*>(get_slot_base(slot_idx) + AEON_W1_PACKED_OFFSET);
    }
    half* get_w1_scale(uint32_t slot_idx) const {
        return reinterpret_cast<half*>(get_slot_base(slot_idx) + AEON_W1_SCALE_OFFSET);
    }
    uint32_t* get_w2_packed(uint32_t slot_idx) const {
        return reinterpret_cast<uint32_t*>(get_slot_base(slot_idx) + AEON_W2_PACKED_OFFSET);
    }
    half* get_w2_scale(uint32_t slot_idx) const {
        return reinterpret_cast<half*>(get_slot_base(slot_idx) + AEON_W2_SCALE_OFFSET);
    }
    uint32_t* get_w3_packed(uint32_t slot_idx) const {
        return reinterpret_cast<uint32_t*>(get_slot_base(slot_idx) + AEON_W3_PACKED_OFFSET);
    }
    half* get_w3_scale(uint32_t slot_idx) const {
        return reinterpret_cast<half*>(get_slot_base(slot_idx) + AEON_W3_SCALE_OFFSET);
    }

    // Stream a complete expert from a contiguous host payload (.aeon layout)
    // into this slot with a single asynchronous copy.
    void upload_from_host_expert(
        uint32_t slot_idx,
        const uint8_t* host_expert_payload,
        hipStream_t stream = 0
    ) {
        if (slot_idx >= num_slots) {
            throw std::runtime_error("UnifiedVRAMExpertPool: Invalid slot index " + std::to_string(slot_idx));
        }
        CHECK_HIP(hipMemcpyAsync(get_slot_base(slot_idx), host_expert_payload,
                                 TOTAL_EXPERT_BYTES, hipMemcpyHostToDevice, stream));
    }

    void download_to_host_expert(
        uint32_t slot_idx,
        uint8_t* host_expert_payload,
        hipStream_t stream = 0
    ) const {
        if (slot_idx >= num_slots || host_expert_payload == nullptr) {
            throw std::runtime_error("UnifiedVRAMExpertPool: Invalid download destination");
        }
        CHECK_HIP(hipMemcpyAsync(host_expert_payload, get_slot_base(slot_idx),
                                 TOTAL_EXPERT_BYTES, hipMemcpyDeviceToHost, stream));
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
        num_slots  = other.num_slots;
        d_experts  = other.d_experts;

        other.num_slots  = 0;
        other.d_experts  = nullptr;
    }
};

// Backwards-compatibility alias during refactoring
using GlobalVRAMExpertPool = UnifiedVRAMExpertPool;

} // namespace aeon::core
