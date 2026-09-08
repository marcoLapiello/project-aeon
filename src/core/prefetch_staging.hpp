#pragma once

#include "core/aeon_loader.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)
#endif

namespace aeon::core {

// Double-Buffered Pinned Host Memory Arena for Asynchronous SDMA Prefetching
// Allocates page-locked host memory to saturate PCIe 4.0 x16 (~25 GB/s) without page faults.
class PrefetchStagingArena {
public:
    static constexpr uint32_t EXPERTS_PER_HORIZON = 6;
    static constexpr uint32_t NUM_BUFFERS = 2; // Double-buffering: Buffer 0 & Buffer 1
    static constexpr uint32_t TOTAL_STAGING_SLOTS = NUM_BUFFERS * EXPERTS_PER_HORIZON; // 12 slots

    uint8_t* h_pinned_base{nullptr};
    bool is_allocated_{false};

    // HIP Events for tracking asynchronous SDMA transfer completion per staging slot
    hipEvent_t events[TOTAL_STAGING_SLOTS];

    PrefetchStagingArena() {
        allocate();
    }

    ~PrefetchStagingArena() {
        free();
    }

    PrefetchStagingArena(const PrefetchStagingArena&) = delete;
    PrefetchStagingArena& operator=(const PrefetchStagingArena&) = delete;

    PrefetchStagingArena(PrefetchStagingArena&& other) noexcept {
        move_from(std::move(other));
    }

    PrefetchStagingArena& operator=(PrefetchStagingArena&& other) noexcept {
        if (this != &other) {
            free();
            move_from(std::move(other));
        }
        return *this;
    }

    void allocate() {
        if (is_allocated_) return;

        size_t total_bytes = static_cast<size_t>(TOTAL_STAGING_SLOTS) * AEON_EXPERT_BYTES;

        // Allocate pinned host memory
        hipError_t err = hipHostMalloc(reinterpret_cast<void**>(&h_pinned_base), total_bytes, hipHostMallocPortable);
        if (err != hipSuccess) {
            std::cerr << "[PrefetchStagingArena] hipHostMalloc failed (" << hipGetErrorString(err)
                      << "), attempting 4KB-aligned posix_memalign..." << std::endl;
            void* ptr = nullptr;
            int ret = posix_memalign(&ptr, AEON_SECTOR_SIZE, total_bytes);
            if (ret != 0 || ptr == nullptr) {
                throw std::runtime_error("PrefetchStagingArena: Failed to allocate " +
                                         std::to_string(total_bytes / (1024 * 1024)) + " MB of pinned staging memory!");
            }
            h_pinned_base = static_cast<uint8_t*>(ptr);
        }

        // Create non-blocking HIP events for transfer synchronization
        for (uint32_t i = 0; i < TOTAL_STAGING_SLOTS; ++i) {
            CHECK_HIP(hipEventCreateWithFlags(&events[i], hipEventDisableTiming));
        }

        is_allocated_ = true;
    }

    void free() {
        if (is_allocated_) {
            for (uint32_t i = 0; i < TOTAL_STAGING_SLOTS; ++i) {
                if (events[i]) {
                    (void)hipEventDestroy(events[i]);
                    events[i] = nullptr;
                }
            }
            if (h_pinned_base) {
                (void)hipHostFree(h_pinned_base);
                h_pinned_base = nullptr;
            }
            is_allocated_ = false;
        }
    }

    // Direct pointer to contiguous 14.15 MB staging slot
    uint8_t* get_slot_ptr(uint32_t slot_idx) {
        if (slot_idx >= TOTAL_STAGING_SLOTS) {
            throw std::runtime_error("PrefetchStagingArena: Slot index " + std::to_string(slot_idx) + " out of bounds");
        }
        return h_pinned_base + static_cast<size_t>(slot_idx) * AEON_EXPERT_BYTES;
    }

    const uint8_t* get_slot_ptr(uint32_t slot_idx) const {
        if (slot_idx >= TOTAL_STAGING_SLOTS) {
            throw std::runtime_error("PrefetchStagingArena: Slot index " + std::to_string(slot_idx) + " out of bounds");
        }
        return h_pinned_base + static_cast<size_t>(slot_idx) * AEON_EXPERT_BYTES;
    }

    // Stage expert payload from source (mmap / HostExpertPool) into pinned buffer
    void stage_payload(uint32_t slot_idx, const uint8_t* src_payload) {
        uint8_t* dst = get_slot_ptr(slot_idx);
        std::memcpy(dst, src_payload, AEON_EXPERT_BYTES);
    }

private:
    void move_from(PrefetchStagingArena&& other) {
        h_pinned_base = other.h_pinned_base;
        is_allocated_ = other.is_allocated_;
        for (uint32_t i = 0; i < TOTAL_STAGING_SLOTS; ++i) {
            events[i] = other.events[i];
            other.events[i] = nullptr;
        }
        other.h_pinned_base = nullptr;
        other.is_allocated_ = false;
    }
};

} // namespace aeon::core
