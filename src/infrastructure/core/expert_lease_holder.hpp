#pragma once

// -----------------------------------------------------------------------------
// The expert-lease contract: what a lifecycle needs from whoever holds leases.
//
// A dispatch leases Hot VRAM slots so an eviction cannot pick a slot a running
// kernel still reads. The lease bookkeeping and the drain-and-release policy live
// with the dispatch (they are entangled with victim selection), but the *contract*
// a phase lifecycle needs is just two operations: hand the leases back at a token
// boundary, and say how many are outstanding. That is model-independent, so it is
// declared here and the executor implements it.
//
// This is what lets the prefill controller depend on a lease holder rather than on
// the model's executor type.
// -----------------------------------------------------------------------------

#include <cstddef>

namespace aeon::core {

class ExpertLeaseHolder {
public:
    virtual ~ExpertLeaseHolder() = default;

    // Hand back every lease held for the current token. The caller guarantees a
    // compute-stream boundary, so no reader of a leased slot is in flight.
    virtual void release_leases() = 0;

    // How many leases are outstanding right now.
    virtual size_t outstanding_leases() const = 0;
};

} // namespace aeon::core
