#pragma once

#include "core/aeon_loader.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <cstdlib>
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
    enum class SlotState : uint8_t {
        AVAILABLE,
        IO_PENDING,
        IO_COMPLETE,
        GPU_TRANSFER_PENDING
    };

    static constexpr uint32_t EXPERTS_PER_HORIZON = 6;
    static constexpr uint32_t NUM_BUFFERS = 2; // Double-buffering: Buffer 0 & Buffer 1
    static constexpr uint32_t TOTAL_STAGING_SLOTS = NUM_BUFFERS * EXPERTS_PER_HORIZON; // 12 slots

    uint8_t* h_pinned_base{nullptr};
    bool is_allocated_{false};

    // HIP Events for tracking asynchronous SDMA transfer completion per staging slot
    hipEvent_t events[TOTAL_STAGING_SLOTS]{};
    std::array<SlotState, TOTAL_STAGING_SLOTS> slot_states{};

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
        uses_hip_host_malloc_ = false;

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
        } else {
            uses_hip_host_malloc_ = true;
        }

        // Create non-blocking HIP events for transfer synchronization
        for (uint32_t i = 0; i < TOTAL_STAGING_SLOTS; ++i) {
            CHECK_HIP(hipEventCreateWithFlags(&events[i], hipEventDisableTiming));
            slot_states[i] = SlotState::AVAILABLE;
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
                if (uses_hip_host_malloc_) {
                    (void)hipHostFree(h_pinned_base);
                } else {
                    std::free(h_pinned_base);
                }
                h_pinned_base = nullptr;
            }
            uses_hip_host_malloc_ = false;
            slot_states.fill(SlotState::AVAILABLE);
            is_allocated_ = false;
        }
    }

    SlotState slot_state(uint32_t slot_idx) const {
        validate_slot(slot_idx);
        return slot_states[slot_idx];
    }

    void begin_io(uint32_t slot_idx) {
        transition(slot_idx, SlotState::AVAILABLE, SlotState::IO_PENDING);
    }

    void complete_io(uint32_t slot_idx) {
        transition(slot_idx, SlotState::IO_PENDING, SlotState::IO_COMPLETE);
    }

    void begin_gpu_transfer(uint32_t slot_idx) {
        transition(slot_idx, SlotState::IO_COMPLETE, SlotState::GPU_TRANSFER_PENDING);
    }

    void release_after_gpu_transfer(uint32_t slot_idx) {
        transition(slot_idx, SlotState::GPU_TRANSFER_PENDING, SlotState::AVAILABLE);
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
        validate_slot(slot_idx);
        if (slot_states[slot_idx] != SlotState::AVAILABLE) {
            throw std::runtime_error("PrefetchStagingArena: staging slot is still in use");
        }
        uint8_t* dst = get_slot_ptr(slot_idx);
        std::memcpy(dst, src_payload, AEON_EXPERT_BYTES);
        slot_states[slot_idx] = SlotState::IO_COMPLETE;
    }

private:
    bool uses_hip_host_malloc_{false};

    void validate_slot(uint32_t slot_idx) const {
        if (slot_idx >= TOTAL_STAGING_SLOTS) {
            throw std::runtime_error("PrefetchStagingArena: Slot index " + std::to_string(slot_idx) + " out of bounds");
        }
    }

    void transition(uint32_t slot_idx, SlotState expected, SlotState next) {
        validate_slot(slot_idx);
        if (slot_states[slot_idx] != expected) {
            throw std::runtime_error("PrefetchStagingArena: invalid slot state transition");
        }
        slot_states[slot_idx] = next;
    }

    void move_from(PrefetchStagingArena&& other) {
        h_pinned_base = other.h_pinned_base;
        is_allocated_ = other.is_allocated_;
        uses_hip_host_malloc_ = other.uses_hip_host_malloc_;
        for (uint32_t i = 0; i < TOTAL_STAGING_SLOTS; ++i) {
            events[i] = other.events[i];
            slot_states[i] = other.slot_states[i];
            other.events[i] = nullptr;
        }
        other.h_pinned_base = nullptr;
        other.uses_hip_host_malloc_ = false;
        other.is_allocated_ = false;
        other.slot_states.fill(SlotState::AVAILABLE);
    }
};

} // namespace aeon::core
