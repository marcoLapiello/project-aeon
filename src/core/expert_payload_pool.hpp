#pragma once

#include "core/expert_format.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

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

class ExpertPayloadPool {
public:
    uint32_t num_slots{0};
    uint8_t* d_experts{nullptr};

    ExpertPayloadPool() = default;

    explicit ExpertPayloadPool(uint32_t slots)
        : ExpertPayloadPool(slots, make_current_swizzled_expert_format()) {
    }

    ExpertPayloadPool(uint32_t slots, const ExpertFormatDescriptor& format) {
        allocate(slots, format);
    }

    ~ExpertPayloadPool() {
        free();
    }

    ExpertPayloadPool(const ExpertPayloadPool&) = delete;
    ExpertPayloadPool& operator=(const ExpertPayloadPool&) = delete;

    ExpertPayloadPool(ExpertPayloadPool&& other) noexcept {
        move_from(std::move(other));
    }

    ExpertPayloadPool& operator=(ExpertPayloadPool&& other) noexcept {
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
        CHECK_HIP(hipMalloc(
            &d_experts,
            static_cast<size_t>(num_slots) * format_.payload_bytes));
    }

    void free() {
        if (d_experts) { (void)hipFree(d_experts); d_experts = nullptr; }
        num_slots = 0;
        format_ = make_current_swizzled_expert_format();
    }

    const ExpertFormatDescriptor& format() const noexcept {
        return format_;
    }

    uint8_t* get_slot_base(uint32_t slot_idx) const {
        if (d_experts == nullptr || slot_idx >= num_slots) {
            throw std::runtime_error(
                "ExpertPayloadPool: Invalid slot index " + std::to_string(slot_idx));
        }
        return d_experts + static_cast<size_t>(slot_idx) * format_.payload_bytes;
    }

    void upload_from_host_expert(
        uint32_t slot_idx,
        const uint8_t* host_expert_payload,
        hipStream_t stream = 0
    ) {
        if (slot_idx >= num_slots || host_expert_payload == nullptr) {
            throw std::runtime_error("ExpertPayloadPool: Invalid upload source or slot index");
        }
        CHECK_HIP(hipMemcpyAsync(
            get_slot_base(slot_idx),
            host_expert_payload,
            format_.payload_bytes,
            hipMemcpyHostToDevice,
            stream));
    }

    void download_to_host_expert(
        uint32_t slot_idx,
        uint8_t* host_expert_payload,
        hipStream_t stream = 0
    ) const {
        if (slot_idx >= num_slots || host_expert_payload == nullptr) {
            throw std::runtime_error("ExpertPayloadPool: Invalid download destination");
        }
        CHECK_HIP(hipMemcpyAsync(
            host_expert_payload,
            get_slot_base(slot_idx),
            format_.payload_bytes,
            hipMemcpyDeviceToHost,
            stream));
    }

private:
    ExpertFormatDescriptor format_{make_current_swizzled_expert_format()};

    void move_from(ExpertPayloadPool&& other) noexcept {
        num_slots = other.num_slots;
        d_experts = other.d_experts;
        format_ = other.format_;

        other.num_slots = 0;
        other.d_experts = nullptr;
        other.format_ = make_current_swizzled_expert_format();
    }
};

} // namespace aeon::core