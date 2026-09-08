#pragma once

#include "core/aeon_loader.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace aeon::core {

// Tier 2 Warm Pinned Host DDR Expert Pool
// Allocates page-locked (pinned) system memory via hipHostMalloc for ultra-fast PCIe SDMA (25 GB/s)
class HostExpertPool {
public:
    static constexpr uint32_t SEGMENT_SLOTS = 64;

    uint32_t num_slots{0};
    // Compatibility view for callers that only need the first segment.
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
        free();
        if (slots == 0) return;
        num_slots = slots;

        const uint32_t segment_count = (num_slots + SEGMENT_SLOTS - 1) / SEGMENT_SLOTS;
        segments_.reserve(segment_count);
        bool reported_pinned_fallback = false;

        try {
            for (uint32_t segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
                Segment segment;
                segment.slot_count = std::min(SEGMENT_SLOTS,
                                              num_slots - segment_idx * SEGMENT_SLOTS);
                const size_t segment_bytes = static_cast<size_t>(segment.slot_count) * AEON_EXPERT_BYTES;

                hipError_t err = hipHostMalloc(reinterpret_cast<void**>(&segment.base),
                                                segment_bytes, hipHostMallocPortable);
                if (err != hipSuccess) {
                    if (!reported_pinned_fallback) {
                        std::cerr << "[HostExpertPool] hipHostMalloc failed (" << hipGetErrorString(err)
                                  << "); using sector-aligned host segments where pinning is unavailable."
                                  << std::endl;
                        reported_pinned_fallback = true;
                    }
                    void* ptr = nullptr;
                    const int ret = posix_memalign(&ptr, AEON_SECTOR_SIZE, segment_bytes);
                    if (ret != 0 || ptr == nullptr) {
                        throw std::runtime_error("HostExpertPool: Failed to allocate " +
                                                 std::to_string(segment_bytes / (1024 * 1024)) +
                                                 " MB host segment");
                    }
                    segment.base = static_cast<uint8_t*>(ptr);
                    segment.uses_hip_host_malloc = false;
                } else {
                    segment.uses_hip_host_malloc = true;
                }
                segments_.push_back(segment);
            }
        } catch (...) {
            free();
            throw;
        }

        h_pinned_buffer = segments_.front().base;
    }

    void free() {
        for (auto& segment : segments_) {
            if (!segment.base) continue;
            if (segment.uses_hip_host_malloc) {
                (void)hipHostFree(segment.base);
            } else {
                std::free(segment.base);
            }
        }
        segments_.clear();
        h_pinned_buffer = nullptr;
        num_slots = 0;
    }

    // Direct pointer to contiguous 14.15 MB expert payload at slot index
    uint8_t* get_expert_slot_ptr(uint32_t slot_idx) {
        const uint32_t segment_idx = slot_idx / SEGMENT_SLOTS;
        const uint32_t segment_slot = slot_idx % SEGMENT_SLOTS;
        if (slot_idx >= num_slots || segment_idx >= segments_.size() ||
            segment_slot >= segments_[segment_idx].slot_count) {
            throw std::runtime_error("HostExpertPool: Invalid slot index " + std::to_string(slot_idx));
        }
        return segments_[segment_idx].base + static_cast<size_t>(segment_slot) * AEON_EXPERT_BYTES;
    }

    const uint8_t* get_expert_slot_ptr(uint32_t slot_idx) const {
        const uint32_t segment_idx = slot_idx / SEGMENT_SLOTS;
        const uint32_t segment_slot = slot_idx % SEGMENT_SLOTS;
        if (slot_idx >= num_slots || segment_idx >= segments_.size() ||
            segment_slot >= segments_[segment_idx].slot_count) {
            throw std::runtime_error("HostExpertPool: Invalid slot index " + std::to_string(slot_idx));
        }
        return segments_[segment_idx].base + static_cast<size_t>(segment_slot) * AEON_EXPERT_BYTES;
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
    struct Segment {
        uint8_t* base{nullptr};
        uint32_t slot_count{0};
        bool uses_hip_host_malloc{false};
    };

    std::vector<Segment> segments_;

    void move_from(HostExpertPool&& other) {
        num_slots = other.num_slots;
        h_pinned_buffer = other.h_pinned_buffer;
        segments_ = std::move(other.segments_);

        other.num_slots = 0;
        other.h_pinned_buffer = nullptr;
        other.segments_.clear();
    }
};

} // namespace aeon::core
