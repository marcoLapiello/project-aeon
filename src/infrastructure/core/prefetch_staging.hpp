#pragma once

#include "infrastructure/core/expert_format.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// The arena's slot count is a **construction parameter**, not a compile-time fact
// (Step 6 / Step 0 D4). Decode's `C = 1` dispatch double-buffers one token — two
// banks of six, `TOTAL_STAGING_SLOTS` — but a layer-wide chunk dispatch issues up
// to `6C` transfers as one set, and each distinct expert's transfer needs its own
// slot for as long as it is in transit. Sizing that as `NUM_BUFFERS * 6` would
// silently collide the second distinct expert with the first. The default
// therefore stays the decode shape (12) so nothing on the certified path moves;
// a caller that dispatches a chunk constructs the arena large enough for its
// deduplicated set, and `V4ModelHost` sizes it from the configured prefill chunk.

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
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
    hipEvent_t* events{nullptr};
    std::vector<SlotState> slot_states{};

    PrefetchStagingArena() {
        allocate();
    }

    explicit PrefetchStagingArena(const ExpertFormatDescriptor& format,
                                  uint32_t slots = TOTAL_STAGING_SLOTS)
        : format_(format) {
        slot_count_ = slots == 0 ? TOTAL_STAGING_SLOTS : slots;
        allocate();
    }

    // Construct as a **view** over another owner's memory (`bind` below), with no
    // allocation of its own. The owner is the shared `ExpertHostRegion`, which is what
    // lets the corridor and the Warm tier be one pinned allocation cut by a movable
    // boundary.
    PrefetchStagingArena(uint8_t* base, uint32_t slots, const ExpertFormatDescriptor& format) {
        bind(base, slots, format);
    }

    // The number of staging slots this arena owns. A batch dispatcher must not
    // assign more distinct staging indices than this.
    uint32_t slot_count() const noexcept { return slot_count_; }

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
        if (slot_count_ == 0) slot_count_ = TOTAL_STAGING_SLOTS;

        format_.validate_payload();
        size_t total_bytes = static_cast<size_t>(slot_count_) * format_.payload_bytes;
        uses_hip_host_malloc_ = false;
        owns_memory_ = true;

        // Allocate pinned host memory
        hipError_t err = hipHostMalloc(reinterpret_cast<void**>(&h_pinned_base), total_bytes, hipHostMallocPortable);
        if (err != hipSuccess) {
            std::cerr << "[PrefetchStagingArena] hipHostMalloc failed (" << hipGetErrorString(err)
                      << "), attempting 4KB-aligned posix_memalign..." << std::endl;
            void* ptr = nullptr;
            int ret = posix_memalign(&ptr, format_.sector_size, total_bytes);
            if (ret != 0 || ptr == nullptr) {
                throw std::runtime_error("PrefetchStagingArena: Failed to allocate " +
                                         std::to_string(total_bytes / (1024 * 1024)) + " MB of pinned staging memory!");
            }
            h_pinned_base = static_cast<uint8_t*>(ptr);
            uses_hip_host_register_ =
                hipHostRegister(h_pinned_base, total_bytes, hipHostRegisterPortable) == hipSuccess;
        } else {
            uses_hip_host_malloc_ = true;
        }

        create_slot_events();
        is_allocated_ = true;
    }

    // Bind the arena as a **view** over `slots` payloads at `base`, which another
    // owner allocated (the shared `ExpertHostRegion`): the arena then owns no memory
    // and `free()` only forgets the view. The per-slot events and states are still
    // this object's, because they describe transfer progress rather than storage.
    //
    // This is what makes the corridor a *partition* of one region instead of a second
    // allocation: moving the Warm/staging boundary moves this base pointer and the
    // slot count, nothing is reallocated, and no page is re-pinned.
    void bind(uint8_t* base, uint32_t slots, const ExpertFormatDescriptor& format) {
        if (slots == 0) {
            throw std::invalid_argument("PrefetchStagingArena: cannot bind zero slots");
        }
        format.validate_payload();
        destroy_slot_events();
        format_ = format;
        h_pinned_base = base;
        slot_count_ = slots;
        owns_memory_ = false;
        uses_hip_host_malloc_ = false;
        uses_hip_host_register_ = false;
        create_slot_events();
        is_allocated_ = true;
    }


    // Re-size the arena in place. Only the **depth** changes: the payload width, the
    // sector size, and the format are the artifact's, not a policy. The depth is a
    // resource budget the corridor consumes, so a host that wants to give it more or
    // less room changes this and nothing else.
    //
    // The caller must have drained every slot and synchronized every stream that
    // could still reference a slot event, because this destroys and recreates them.
    // `V4ModelHost::resize_staging_slots` is the guarded entry point.
    void resize(uint32_t slots) {
        if (slots == 0) {
            throw std::invalid_argument("PrefetchStagingArena: cannot resize to zero slots");
        }
        if (slots == slot_count_ && is_allocated_) return;
        free();
        slot_count_ = slots;
        allocate();
    }

    void free() {
        if (is_allocated_) {
            destroy_slot_events();
            if (h_pinned_base && owns_memory_) {
                if (uses_hip_host_malloc_) {
                    (void)hipHostFree(h_pinned_base);
                } else {
                    if (uses_hip_host_register_) {
                        (void)hipHostUnregister(h_pinned_base);
                    }
                    std::free(h_pinned_base);
                }
            }
            h_pinned_base = nullptr;
            uses_hip_host_malloc_ = false;
            uses_hip_host_register_ = false;
            owns_memory_ = true;
            slot_states.assign(slot_count_, SlotState::AVAILABLE);
            available_since_.assign(slot_count_, std::chrono::steady_clock::now());
            is_allocated_ = false;
        }
    }

    SlotState slot_state(uint32_t slot_idx) const {
        validate_slot(slot_idx);
        return slot_states[slot_idx];
    }

    // Slots not currently AVAILABLE — i.e. a transfer is in flight through them.
    // A gate asserts this returns to 0: every slot must be handed back when its
    // transfer completes, or a long run leaks the arena and the next fetch stalls.
    uint32_t in_use_slots() const {
        uint32_t in_use = 0;
        for (uint32_t i = 0; i < slot_count_; ++i) {
            if (slot_states[i] != SlotState::AVAILABLE) {
                ++in_use;
            }
        }
        return in_use;
    }

    // The arena's occupancy **by pipeline stage**, which is what says whether the
    // corridor is actually filled rather than merely sized. `reading` is the slots a
    // read is landing in (`IO_PENDING`/`IO_COMPLETE`); `copying` is the slots a
    // copy-into-VRAM is draining (`GPU_TRANSFER_PENDING`). A healthy two-block
    // staging corridor shows both non-zero at once during a layer body; a parking
    // lot shows one of them pinned at `E` while the other is zero.
    struct StateCounts {
        uint32_t free{0};
        uint32_t reading{0};
        uint32_t copying{0};
        uint32_t in_use() const noexcept { return reading + copying; }
    };

    StateCounts state_counts() const {
        StateCounts counts;
        for (uint32_t i = 0; i < slot_count_; ++i) {
            switch (slot_states[i]) {
            case SlotState::AVAILABLE: ++counts.free; break;
            case SlotState::IO_PENDING:
            case SlotState::IO_COMPLETE: ++counts.reading; break;
            case SlotState::GPU_TRANSFER_PENDING: ++counts.copying; break;
            }
        }
        return counts;
    }

    void begin_io(uint32_t slot_idx) {
        transition(slot_idx, SlotState::AVAILABLE, SlotState::IO_PENDING);
    }

    // Claim a slot and mark its bytes **ready to copy**, without staging a payload
    // into it. The deferred Warm hand-off needs this: the payload is already pinned
    // host memory, so a copy through the arena would add a whole-payload memcpy for
    // nothing — the slot is held only for its completion event, and the copy runs
    // later out of the host slot (see `enqueue_expert_copy`).
    void mark_ready(uint32_t slot_idx) {
        transition(slot_idx, SlotState::AVAILABLE, SlotState::IO_COMPLETE);
    }

    void complete_io(uint32_t slot_idx) {
        transition(slot_idx, SlotState::IO_PENDING, SlotState::IO_COMPLETE);
    }

    // Claim a slot purely to borrow its completion event for a direct pinned
    // host-to-device upload; no payload is staged into the arena.
    void begin_direct_transfer(uint32_t slot_idx) {
        transition(slot_idx, SlotState::AVAILABLE, SlotState::GPU_TRANSFER_PENDING);
    }

    bool try_begin_direct_transfer(uint32_t& slot_idx) {
        if (!is_pinned()) return false;
        for (uint32_t candidate = 0; candidate < slot_count_; ++candidate) {
            if (slot_states[candidate] == SlotState::AVAILABLE) {
                begin_direct_transfer(candidate);
                slot_idx = candidate;
                return true;
            }
        }
        return false;
    }

    void begin_gpu_transfer(uint32_t slot_idx) {
        transition(slot_idx, SlotState::IO_COMPLETE, SlotState::GPU_TRANSFER_PENDING);
    }

    void release_after_gpu_transfer(uint32_t slot_idx) {
        transition(slot_idx, SlotState::GPU_TRANSFER_PENDING, SlotState::AVAILABLE);
        available_since_[slot_idx] = std::chrono::steady_clock::now();
    }

    // Release a slot **iff** a copy is still draining through it, and report whether
    // it did. This is the completion-driven form (plan R3): the reaper releases the
    // slot when its copy's own event fires, and a caller that would also release it
    // as a block (the sweep's boundary drain, the executor's `on_routed_consumed`)
    // finds it already `AVAILABLE` and skips. `release_after_gpu_transfer` cannot be
    // used for that because it throws on an unexpected state.
    bool release_if_copying(uint32_t slot_idx) {
        validate_slot(slot_idx);
        if (slot_states[slot_idx] != SlotState::GPU_TRANSFER_PENDING) return false;
        slot_states[slot_idx] = SlotState::AVAILABLE;
        available_since_[slot_idx] = std::chrono::steady_clock::now();
        return true;
    }

    void release_after_failure(uint32_t slot_idx) {
        validate_slot(slot_idx);
        slot_states[slot_idx] = SlotState::AVAILABLE;
        available_since_[slot_idx] = std::chrono::steady_clock::now();
    }

    uint64_t take_reuse_delay_ns(uint32_t slot_idx) {
        validate_slot(slot_idx);
        const auto now = std::chrono::steady_clock::now();
        if (available_since_[slot_idx].time_since_epoch().count() == 0) return 0;
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            now - available_since_[slot_idx]).count());
    }

    bool is_pinned() const {
        return uses_hip_host_malloc_ || uses_hip_host_register_;
    }

    const ExpertFormatDescriptor& format() const noexcept {
        return format_;
    }

    size_t payload_bytes() const noexcept {
        return format_.payload_bytes;
    }

    // Direct pointer to one opaque expert payload staging slot.
    uint8_t* get_slot_ptr(uint32_t slot_idx) {
        if (slot_idx >= slot_count_) {
            throw std::runtime_error("PrefetchStagingArena: Slot index " + std::to_string(slot_idx) + " out of bounds");
        }
        return h_pinned_base + static_cast<size_t>(slot_idx) * format_.payload_bytes;
    }

    const uint8_t* get_slot_ptr(uint32_t slot_idx) const {
        if (slot_idx >= slot_count_) {
            throw std::runtime_error("PrefetchStagingArena: Slot index " + std::to_string(slot_idx) + " out of bounds");
        }
        return h_pinned_base + static_cast<size_t>(slot_idx) * format_.payload_bytes;
    }

    // Stage expert payload from source (mmap / HostExpertPool) into pinned buffer
    void stage_payload(uint32_t slot_idx, const uint8_t* src_payload) {
        validate_slot(slot_idx);
        if (slot_states[slot_idx] != SlotState::AVAILABLE) {
            throw std::runtime_error("PrefetchStagingArena: staging slot is still in use");
        }
        uint8_t* dst = get_slot_ptr(slot_idx);
        std::memcpy(dst, src_payload, format_.payload_bytes);
        slot_states[slot_idx] = SlotState::IO_COMPLETE;
    }

private:
    uint32_t slot_count_{TOTAL_STAGING_SLOTS};
    bool uses_hip_host_malloc_{false};
    bool uses_hip_host_register_{false};
    // False when the arena is a view over another owner's memory (`bind`); `free()`
    // then destroys the events but frees no storage.
    bool owns_memory_{true};
    ExpertFormatDescriptor format_{make_current_swizzled_expert_format()};
    std::vector<std::chrono::steady_clock::time_point> available_since_{};

    // The per-slot events and states, shared by `allocate` and `bind` so the two
    // cannot set up a slot differently.
    void create_slot_events() {
        events = new hipEvent_t[slot_count_]();
        slot_states.assign(slot_count_, SlotState::AVAILABLE);
        available_since_.assign(slot_count_, std::chrono::steady_clock::now());
        for (uint32_t i = 0; i < slot_count_; ++i) {
            CHECK_HIP(hipEventCreateWithFlags(&events[i], hipEventDisableTiming));
        }
    }

    void destroy_slot_events() {
        if (events == nullptr) return;
        for (uint32_t i = 0; i < slot_count_; ++i) {
            if (events[i]) {
                (void)hipEventDestroy(events[i]);
                events[i] = nullptr;
            }
        }
        delete[] events;
        events = nullptr;
    }

    void validate_slot(uint32_t slot_idx) const {
        if (slot_idx >= slot_count_) {
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
        uses_hip_host_register_ = other.uses_hip_host_register_;
        format_ = other.format_;
        slot_count_ = other.slot_count_;
        events = other.events;
        slot_states = std::move(other.slot_states);
        available_since_ = std::move(other.available_since_);
        other.events = nullptr;
        other.h_pinned_base = nullptr;
        other.uses_hip_host_malloc_ = false;
        other.uses_hip_host_register_ = false;
        other.format_ = make_current_swizzled_expert_format();
        other.is_allocated_ = false;
        other.slot_count_ = TOTAL_STAGING_SLOTS;
        other.slot_states.clear();
        other.available_since_.clear();
    }
};

} // namespace aeon::core
