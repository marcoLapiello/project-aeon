#pragma once

#include "core/aeon_loader.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace aeon::core {

// Tier 2 Warm Pinned Host DDR Expert Pool
// Allocates page-locked (pinned) system memory via hipHostMalloc for ultra-fast PCIe SDMA (25 GB/s)
class HostExpertPool {
public:
    uint32_t num_slots{0};
    uint8_t* h_pinned_buffer{nullptr};

    HostExpertPool() = default;

    explicit HostExpertPool(uint32_t slots) {
        allocate(slots);
    }

    ~HostExpertPool() {
        free();
    }

    HostExpertPool(const HostExpertPool&) = delete;
    HostExpertPool& operator=(const HostExpertPool&) = delete;

    HostExpertPool(HostExpertPool&& other) noexcept {
        move_from(std::move(other));
    }

    HostExpertPool& operator=(HostExpertPool&& other) noexcept {
        if (this != &other) {
            free();
            move_from(std::move(other));
        }
        return *this;
    }

    void allocate(uint32_t slots) {
        if (slots == 0) return;
        num_slots = slots;

        size_t total_bytes = static_cast<size_t>(num_slots) * AEON_EXPERT_BYTES;

        // Use hipHostMalloc with hipHostMallocPortable for high-throughput PCIe DMA transfers
        hipError_t err = hipHostMalloc(reinterpret_cast<void**>(&h_pinned_buffer), total_bytes, hipHostMallocPortable);
        if (err != hipSuccess) {
            // Fallback to posix_memalign if hipHostMalloc fails due to OS ulimit
            std::cerr << "[HostExpertPool] Warning: hipHostMalloc failed (" << hipGetErrorString(err)
                      << "), attempting 4KB-aligned posix_memalign..." << std::endl;
            void* ptr = nullptr;
            int ret = posix_memalign(&ptr, AEON_SECTOR_SIZE, total_bytes);
            if (ret != 0 || ptr == nullptr) {
                throw std::runtime_error("HostExpertPool: Failed to allocate " +
                                         std::to_string(total_bytes / (1024 * 1024)) + " MB of host memory!");
            }
            h_pinned_buffer = static_cast<uint8_t*>(ptr);
        }
    }

    void free() {
        if (h_pinned_buffer) {
            (void)hipHostFree(h_pinned_buffer);
            h_pinned_buffer = nullptr;
        }
        num_slots = 0;
    }

    // Direct pointer to contiguous 14.15 MB expert payload at slot index
    uint8_t* get_expert_slot_ptr(uint32_t slot_idx) {
        if (slot_idx >= num_slots) {
            throw std::runtime_error("HostExpertPool: Invalid slot index " + std::to_string(slot_idx));
        }
        return h_pinned_buffer + static_cast<size_t>(slot_idx) * AEON_EXPERT_BYTES;
    }

    const uint8_t* get_expert_slot_ptr(uint32_t slot_idx) const {
        if (slot_idx >= num_slots) {
            throw std::runtime_error("HostExpertPool: Invalid slot index " + std::to_string(slot_idx));
        }
        return h_pinned_buffer + static_cast<size_t>(slot_idx) * AEON_EXPERT_BYTES;
    }

    // Sub-tensor pointers within a host slot
    const uint32_t* get_w1_packed(uint32_t slot_idx) const {
        return reinterpret_cast<const uint32_t*>(get_expert_slot_ptr(slot_idx) + AEON_W1_PACKED_OFFSET);
    }
    const half* get_w1_scale(uint32_t slot_idx) const {
        return reinterpret_cast<const half*>(get_expert_slot_ptr(slot_idx) + AEON_W1_SCALE_OFFSET);
    }
    const uint32_t* get_w2_packed(uint32_t slot_idx) const {
        return reinterpret_cast<const uint32_t*>(get_expert_slot_ptr(slot_idx) + AEON_W2_PACKED_OFFSET);
    }
    const half* get_w2_scale(uint32_t slot_idx) const {
        return reinterpret_cast<const half*>(get_expert_slot_ptr(slot_idx) + AEON_W2_SCALE_OFFSET);
    }
    const uint32_t* get_w3_packed(uint32_t slot_idx) const {
        return reinterpret_cast<const uint32_t*>(get_expert_slot_ptr(slot_idx) + AEON_W3_PACKED_OFFSET);
    }
    const half* get_w3_scale(uint32_t slot_idx) const {
        return reinterpret_cast<const half*>(get_expert_slot_ptr(slot_idx) + AEON_W3_SCALE_OFFSET);
    }

    // Load expert payload from disk/source into host slot
    void copy_from(uint32_t slot_idx, const uint8_t* src_payload) {
        uint8_t* dst = get_expert_slot_ptr(slot_idx);
        std::memcpy(dst, src_payload, AEON_EXPERT_BYTES);
    }

private:
    void move_from(HostExpertPool&& other) {
        num_slots = other.num_slots;
        h_pinned_buffer = other.h_pinned_buffer;

        other.num_slots = 0;
        other.h_pinned_buffer = nullptr;
    }
};

} // namespace aeon::core
