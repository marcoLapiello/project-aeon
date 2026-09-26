#pragma once

#include "infrastructure/core/expert_format.hpp"
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
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

    HostExpertPool() = default;

    explicit HostExpertPool(uint32_t slots)
        : HostExpertPool(slots, make_current_swizzled_expert_format()) {
    }

    HostExpertPool(uint32_t slots, const ExpertFormatDescriptor& format) {
        allocate(slots, format);
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
        allocate(slots, make_current_swizzled_expert_format());
    }

    void allocate(uint32_t slots, const ExpertFormatDescriptor& format) {
        free();
        format.validate_payload();
        format_ = format;
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
                const size_t segment_bytes = static_cast<size_t>(segment.slot_count) * format_.payload_bytes;

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
                    const int ret = posix_memalign(&ptr, format_.sector_size, segment_bytes);
                    if (ret != 0 || ptr == nullptr) {
                        throw std::runtime_error("HostExpertPool: Failed to allocate " +
                                                 std::to_string(segment_bytes / (1024 * 1024)) +
                                                 " MB host segment");
                    }
                    segment.base = static_cast<uint8_t*>(ptr);
                    segment.uses_hip_host_malloc = false;
                    segment.uses_hip_host_register =
                        hipHostRegister(segment.base, segment_bytes, hipHostRegisterPortable) == hipSuccess;
                } else {
                    segment.uses_hip_host_malloc = true;
                }
                segments_.push_back(segment);
            }
        } catch (...) {
            free();
            throw;
        }

    }

    void free() {
        // A **view** (see `bind`) points at another owner's memory: dropping the view
        // must not free it, only forget it.
        if (!owns_memory_) {
            segments_.clear();
            num_slots = 0;
            return;
        }
        for (auto& segment : segments_) {
            if (!segment.base) continue;
            if (segment.uses_hip_host_malloc) {
                (void)hipHostFree(segment.base);
            } else {
                if (segment.uses_hip_host_register) {
                    (void)hipHostUnregister(segment.base);
                }
                std::free(segment.base);
            }
        }
        segments_.clear();
        num_slots = 0;
    }

    // Bind this pool as a **view** over `slots` payloads at `base`, which another
    // owner allocated (the shared `ExpertHostRegion`). The pool then owns no memory:
    // `free()` only forgets the view. `pinned` says whether the backing memory is
    // page-locked, because the supply chooses a direct upload only for a pinned slot.
    //
    // A view is one segment, not many: the backing region is already a single
    // allocation, so the per-64-slot segmentation the owning path uses (it exists to
    // degrade to unpinned memory in pieces) has nothing to do here.
    // A view is laid out in `SEGMENT_SLOTS`-sized segments, exactly like an owned
    // pool: `get_expert_slot_ptr`/`is_slot_pinned` resolve a slot by
    // `slot / SEGMENT_SLOTS`, so a single large segment would mis-index every slot
    // past the first. The segments share the backing region's memory; only their
    // `base` pointers differ.
    void bind(uint8_t* base, uint32_t slots, const ExpertFormatDescriptor& format, bool pinned) {
        free();
        format.validate_payload();
        format_ = format;
        owns_memory_ = false;
        if (slots == 0) return;
        num_slots = slots;
        const uint32_t segment_count = (slots + SEGMENT_SLOTS - 1) / SEGMENT_SLOTS;
        segments_.reserve(segment_count);
        for (uint32_t index = 0; index < segment_count; ++index) {
            Segment segment;
            segment.slot_count = std::min(SEGMENT_SLOTS, slots - index * SEGMENT_SLOTS);
            segment.base = base + static_cast<size_t>(index) * SEGMENT_SLOTS *
                                      format_.payload_bytes;
            segment.uses_hip_host_malloc = pinned;
            segment.uses_hip_host_register = false;
            segments_.push_back(segment);
        }
    }

    const ExpertFormatDescriptor& format() const noexcept {
        return format_;
    }

    size_t payload_bytes() const noexcept {
        return format_.payload_bytes;
    }

    // Direct pointer to one opaque expert payload at slot index.
    uint8_t* get_expert_slot_ptr(uint32_t slot_idx) {
        const uint32_t segment_idx = slot_idx / SEGMENT_SLOTS;
        const uint32_t segment_slot = slot_idx % SEGMENT_SLOTS;
        if (slot_idx >= num_slots || segment_idx >= segments_.size() ||
            segment_slot >= segments_[segment_idx].slot_count) {
            throw std::runtime_error("HostExpertPool: Invalid slot index " + std::to_string(slot_idx));
        }
        return segments_[segment_idx].base + static_cast<size_t>(segment_slot) * format_.payload_bytes;
    }

    const uint8_t* get_expert_slot_ptr(uint32_t slot_idx) const {
        const uint32_t segment_idx = slot_idx / SEGMENT_SLOTS;
        const uint32_t segment_slot = slot_idx % SEGMENT_SLOTS;
        if (slot_idx >= num_slots || segment_idx >= segments_.size() ||
            segment_slot >= segments_[segment_idx].slot_count) {
            throw std::runtime_error("HostExpertPool: Invalid slot index " + std::to_string(slot_idx));
        }
        return segments_[segment_idx].base + static_cast<size_t>(segment_slot) * format_.payload_bytes;
    }

    // True when the segment backing this slot is page-locked (hipHostMalloc),
    // i.e. safe for direct SDMA upload without staging through a bounce buffer.
    bool is_slot_pinned(uint32_t slot_idx) const {
        const uint32_t segment_idx = slot_idx / SEGMENT_SLOTS;
        if (slot_idx >= num_slots || segment_idx >= segments_.size()) {
            return false;
        }
         return segments_[segment_idx].uses_hip_host_malloc ||
             segments_[segment_idx].uses_hip_host_register;
    }

    uint32_t pinned_slot_count() const {
        uint32_t count = 0;
        for (const auto& segment : segments_) {
            if (segment.uses_hip_host_malloc || segment.uses_hip_host_register) {
                count += segment.slot_count;
            }
        }
        return count;
    }

    uint32_t unpinned_slot_count() const {
        return num_slots - pinned_slot_count();
    }

    size_t allocated_bytes() const {
        return static_cast<size_t>(num_slots) * format_.payload_bytes;
    }

private:
    struct Segment {
        uint8_t* base{nullptr};
        uint32_t slot_count{0};
        bool uses_hip_host_malloc{false};
        bool uses_hip_host_register{false};
    };

    std::vector<Segment> segments_;
    ExpertFormatDescriptor format_{make_current_swizzled_expert_format()};
    // False when this pool is a view over another owner's memory (`bind`); `free()`
    // then only forgets the view.
    bool owns_memory_{true};

    void move_from(HostExpertPool&& other) {
        num_slots = other.num_slots;
        segments_ = std::move(other.segments_);
        format_ = other.format_;
        owns_memory_ = other.owns_memory_;

        other.num_slots = 0;
        other.segments_.clear();
        other.owns_memory_ = true;
        other.format_ = make_current_swizzled_expert_format();
    }
};

} // namespace aeon::core
