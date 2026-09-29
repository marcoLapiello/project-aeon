#pragma once

// -----------------------------------------------------------------------------
// The supply's transfer counters: the instrumentation for the streaming path, as
// one small owned unit rather than ten loose members.
//
// These stay exactly where they are measured — the mutation sites are inside
// `dispatch`, `materialize` and `reap`. Only their *storage* lives here, so the
// supply declares one member instead of ten and the phase-slicing reset has a
// single obvious home.
//
// Fields are public on purpose: a counter increment should read like arithmetic
// at the site that measures it, not like an encapsulation ceremony.
// -----------------------------------------------------------------------------

#include <cstdint>

namespace aeon::core {

struct SupplyTransferCounters {
    // Time inside `io_uring_enter` for submissions, and the SQE count it submitted.
    // Split from the rest of `dispatch` because issuing a whole layer's reads at
    // once can block on the device queue rather than the CPU.
    uint64_t direct_io_submit_ns{0};
    uint64_t direct_io_requests_submitted{0};
    uint64_t direct_io_submit_calls{0};

    // The transfer split. `io_wait` vs `h2d_drain` says whether the exposed load is
    // disk-bound or PCIe-bound; `h2d_drain_calls` separates the copy from the
    // per-slot driver round-trips.
    uint64_t io_wait_ns{0};
    uint64_t h2d_enqueue_ns{0};
    uint64_t h2d_drain_ns{0};
    uint64_t h2d_drain_calls{0};

    // CPU time in `dispatch`'s per-request loop (registry reservation, the two
    // `O(catalog)` scans, the transfer record-keeping).
    uint64_t dispatch_cpu_ns{0};

    // Staging slots released by the completion path instead of at a boundary, and
    // copy enqueues issued by the non-blocking pump rather than `materialize`.
    uint64_t staging_released_on_completion{0};
    uint64_t copies_pumped{0};

    // Zeroes every counter so a caller can slice one phase (prefill, then decode)
    // without re-instantiating the supply. Counters only — no state is reset, and
    // the registry and arena are untouched.
    void reset() noexcept {
        direct_io_submit_ns = 0;
        direct_io_requests_submitted = 0;
        direct_io_submit_calls = 0;
        io_wait_ns = 0;
        h2d_enqueue_ns = 0;
        h2d_drain_ns = 0;
        h2d_drain_calls = 0;
        dispatch_cpu_ns = 0;
        staging_released_on_completion = 0;
        copies_pumped = 0;
    }
};

} // namespace aeon::core
