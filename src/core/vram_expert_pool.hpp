#pragma once

#include "core/expert_payload_pool.hpp"
#include "core/swizzled_expert_format.hpp"
#include <hip/hip_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace aeon::core {

// Unified VRAM pool for S_hot routed INT4-W4A16 experts
// Shared dynamically across all transformer layers
//
// Physical layout (Expert Review Step 2): one contiguous device allocation
// holding num_slots back-to-back opaque expert regions, each described by the
// selected ExpertFormatDescriptor and
// byte-identical to the .aeon host/staging layout. A full expert therefore
// uploads with a single hipMemcpyAsync instead of six per-sub-tensor copies:
// fewer API submissions, one sequential SDMA burst, and hardware-friendly
// prefetching on both the PCIe and NVMe paths.
class UnifiedVRAMExpertPool : public ExpertPayloadPool {
public:
    // Sub-tensor byte sizes per single expert
    static constexpr size_t W1_PACKED_BYTES = AEON_W1_PACKED_BYTES;
    static constexpr size_t W1_SCALE_BYTES  = AEON_W1_SCALE_BYTES;
    static constexpr size_t W2_PACKED_BYTES = AEON_W2_PACKED_BYTES;
    static constexpr size_t W2_SCALE_BYTES  = AEON_W2_SCALE_BYTES;
    static constexpr size_t W3_PACKED_BYTES = AEON_W3_PACKED_BYTES;
    static constexpr size_t W3_SCALE_BYTES  = AEON_W3_SCALE_BYTES;
    static constexpr size_t TOTAL_EXPERT_BYTES = W1_PACKED_BYTES + W1_SCALE_BYTES +
                                                 W2_PACKED_BYTES + W2_SCALE_BYTES +
                                                 W3_PACKED_BYTES + W3_SCALE_BYTES;
    static_assert(TOTAL_EXPERT_BYTES == AEON_EXPERT_BYTES,
                  "Per-slot device layout must match the .aeon expert container layout");

    using ExpertPayloadPool::ExpertPayloadPool;

    UnifiedVRAMExpertPool() = default;
    UnifiedVRAMExpertPool(UnifiedVRAMExpertPool&&) noexcept = default;
    UnifiedVRAMExpertPool& operator=(UnifiedVRAMExpertPool&&) noexcept = default;

    UnifiedVRAMExpertPool(const UnifiedVRAMExpertPool&) = delete;
    UnifiedVRAMExpertPool& operator=(const UnifiedVRAMExpertPool&) = delete;

    // Current-backend views, derived from the swizzled slot layout.
    uint32_t* get_w1_packed(uint32_t slot_idx) const {
        require_swizzled_layout();
        return reinterpret_cast<uint32_t*>(get_slot_base(slot_idx) + AEON_W1_PACKED_OFFSET);
    }
    half* get_w1_scale(uint32_t slot_idx) const {
        require_swizzled_layout();
        return reinterpret_cast<half*>(get_slot_base(slot_idx) + AEON_W1_SCALE_OFFSET);
    }
    uint32_t* get_w2_packed(uint32_t slot_idx) const {
        require_swizzled_layout();
        return reinterpret_cast<uint32_t*>(get_slot_base(slot_idx) + AEON_W2_PACKED_OFFSET);
    }
    half* get_w2_scale(uint32_t slot_idx) const {
        require_swizzled_layout();
        return reinterpret_cast<half*>(get_slot_base(slot_idx) + AEON_W2_SCALE_OFFSET);
    }
    uint32_t* get_w3_packed(uint32_t slot_idx) const {
        require_swizzled_layout();
        return reinterpret_cast<uint32_t*>(get_slot_base(slot_idx) + AEON_W3_PACKED_OFFSET);
    }
    half* get_w3_scale(uint32_t slot_idx) const {
        require_swizzled_layout();
        return reinterpret_cast<half*>(get_slot_base(slot_idx) + AEON_W3_SCALE_OFFSET);
    }

private:
    void require_swizzled_layout() const {
        const auto& expert_format = format();
        if (expert_format.kind != ExpertFormatKind::SWIZZLED_W4A16 ||
            expert_format.artifact_version != AEON_SWIZZLED_EXPERT_FORMAT_VERSION ||
            expert_format.payload_bytes != TOTAL_EXPERT_BYTES) {
            throw std::logic_error(
                "UnifiedVRAMExpertPool: swizzled views are unavailable for this expert format");
        }
    }
};

} // namespace aeon::core
