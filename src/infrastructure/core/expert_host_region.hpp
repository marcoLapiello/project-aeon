#pragma once

#include "infrastructure/core/expert_format.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace aeon::core {

// -----------------------------------------------------------------------------
// The single pinned host region the Warm tier and the transport corridor share.
//
// Warm expert residency and the staging arena were two separate pinned
// allocations, so the host RAM a run actually held was `warm + staging` — a number
// the user could only reach by adding two settings together, and one they could
// overshoot without noticing. They are also, physically, the same thing: both hold
// expert payloads of the artifact's own width, 4 KiB-aligned, in the same layout.
// The only difference is **policy** — a Warm slot is a durable residence whose owner
// is the registry, a staging slot is a transient one whose owner is a transfer in
// flight.
//
// So they are one region, partitioned by a **movable boundary**:
//
//     [  Warm slots (0 .. warm_slots)  |  staging slots (the tail)  ]
//                                       ^ boundary
//
// `warm_slots` is the only number that defines the split, and the partition is
// re-cut at phase boundaries because the two consumers want opposite shapes:
//
//   * decode needs `12` staging slots (`2 x 6`, the double buffer) and wants every
//     other byte as Warm residency — which is where its NVMe hits are;
//   * a swept prefill needs `blocks x E` staging slots and gives the difference
//     back when the window ends.
//
// Whoever owns the region therefore hands each consumer a **view**: the Warm pool
// takes the head, the arena takes the tail, and moving the boundary is two pointer
// moves rather than a reallocation. That is the whole point — resizing two separate
// allocations against each other would re-pin gigabytes per phase and could never
// make the two sizes cancel exactly, because a pool frees in whole segments.
//
// The region is pinned once and never returned to the OS for the life of the run,
// so a boundary move commits no new pages and costs no re-pinning.
// -----------------------------------------------------------------------------
class ExpertHostRegion {
public:
    ExpertHostRegion() = default;

    ~ExpertHostRegion() { free(); }

    ExpertHostRegion(const ExpertHostRegion&) = delete;
    ExpertHostRegion& operator=(const ExpertHostRegion&) = delete;

    // Allocate `slots` payloads of pinned host memory. Mirrors the fallback the
    // pools used individually: `hipHostMalloc` first, and a sector-aligned
    // `posix_memalign` plus a registration attempt where pinning is unavailable, so
    // an unpinnable host still runs (slower, through a bounce buffer).
    void allocate(uint32_t slots, const ExpertFormatDescriptor& format) {
        free();
        format.validate_payload();
        format_ = format;
        if (slots == 0) return;
        slot_count_ = slots;

        const size_t bytes = static_cast<size_t>(slots) * format_.payload_bytes;
        hipError_t err = hipHostMalloc(
            reinterpret_cast<void**>(&base_), bytes, hipHostMallocPortable);
        if (err != hipSuccess) {
            std::cerr << "[ExpertHostRegion] hipHostMalloc failed (" << hipGetErrorString(err)
                      << "); using sector-aligned host memory where pinning is unavailable."
                      << std::endl;
            void* ptr = nullptr;
            const int ret = posix_memalign(&ptr, format_.sector_size, bytes);
            if (ret != 0 || ptr == nullptr) {
                base_ = nullptr;
                slot_count_ = 0;
                throw std::runtime_error(
                    "ExpertHostRegion: failed to allocate " +
                    std::to_string(bytes / (1024 * 1024)) + " MB of host memory");
            }
            base_ = static_cast<uint8_t*>(ptr);
            uses_hip_host_malloc_ = false;
            uses_hip_host_register_ =
                hipHostRegister(base_, bytes, hipHostRegisterPortable) == hipSuccess;
        } else {
            uses_hip_host_malloc_ = true;
        }
    }

    void free() {
        if (base_ == nullptr) {
            slot_count_ = 0;
            return;
        }
        if (uses_hip_host_malloc_) {
            (void)hipHostFree(base_);
        } else {
            if (uses_hip_host_register_) {
                (void)hipHostUnregister(base_);
            }
            std::free(base_);
        }
        base_ = nullptr;
        slot_count_ = 0;
        uses_hip_host_malloc_ = false;
        uses_hip_host_register_ = false;
    }

    uint8_t* base() const noexcept { return base_; }
    uint32_t slot_count() const noexcept { return slot_count_; }
    size_t payload_bytes() const noexcept { return format_.payload_bytes; }
    const ExpertFormatDescriptor& format() const noexcept { return format_; }

    bool is_pinned() const noexcept {
        return uses_hip_host_malloc_ || uses_hip_host_register_;
    }

    // A view onto `slots` payloads starting at `first_slot` of the region. Both
    // consumers are built from this: the pools index their own slots from 0, and the
    // offset is what makes the boundary a pointer move rather than a copy.
    uint8_t* slot_ptr(uint32_t first_slot) const {
        if (base_ == nullptr || first_slot > slot_count_) {
            throw std::runtime_error("ExpertHostRegion: slot view is out of range");
        }
        return base_ + static_cast<size_t>(first_slot) * format_.payload_bytes;
    }

private:
    uint8_t* base_{nullptr};
    uint32_t slot_count_{0};
    bool uses_hip_host_malloc_{false};
    bool uses_hip_host_register_{false};
    ExpertFormatDescriptor format_{make_current_swizzled_expert_format()};
};

} // namespace aeon::core
