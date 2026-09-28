#pragma once

// The four device streams, owned by the host and borrowed by everything that
// enqueues work.
//
// One type rather than one per consumer, because the executor's correctness
// depends on *which* stream a copy lands on: its capacity fallback has to drain
// every stream that carries expert traffic, and draining the wrong set releases
// leases without freeing anything (see the second defect recorded in
// `core/v4_expert_executor.hpp`). Two definitions of "the streams" would be a way
// to hand it the wrong one without a compile error.
//
// The split is by traffic class, not by caller:
//
//   * `compute`     — every graph kernel, and where per-transfer staging events
//                     are awaited, so an upload bound for this token is ordered
//                     against the kernels that read it.
//   * `sdma`        — warm-tier host-to-device uploads.
//   * `sdma_cold`   — cold-tier uploads (io_uring stages into the arena; this
//                     stream carries the copy out of it).
//   * `demotion`    — eviction downloads out of VRAM, which the incoming uploads
//                     are ordered against.

#include <hip/hip_runtime.h>

#include <array>
#include <stdexcept>
#include <string>

namespace aeon::core {

struct V4DeviceStreams {
    hipStream_t compute{nullptr};
    hipStream_t sdma{nullptr};
    hipStream_t sdma_cold{nullptr};
    hipStream_t demotion{nullptr};

    // Non-blocking, so a copy on one stream never serialises another. All four or
    // none: a partially created set is destroyed before the throw, so a caller
    // that catches this cannot leak the streams that did succeed.
    static V4DeviceStreams create() {
        V4DeviceStreams streams;
        try {
            streams.compute = create_one("compute");
            streams.sdma = create_one("sdma");
            streams.sdma_cold = create_one("sdma_cold");
            streams.demotion = create_one("demotion");
        } catch (...) {
            streams.destroy();
            throw;
        }
        return streams;
    }

    void destroy() noexcept {
        for (hipStream_t* stream : {&compute, &sdma, &sdma_cold, &demotion}) {
            if (*stream != nullptr) {
                (void)hipStreamDestroy(*stream);
                *stream = nullptr;
            }
        }
    }

private:
    static hipStream_t create_one(const char* identity) {
        hipStream_t stream = nullptr;
        const hipError_t error = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        if (error != hipSuccess) {
            throw std::runtime_error(std::string("V4DeviceStreams: hipStreamCreateWithFlags(") +
                                     identity + "): " + hipGetErrorString(error));
        }
        return stream;
    }
};

} // namespace aeon::core
