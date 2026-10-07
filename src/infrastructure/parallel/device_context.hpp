#pragma once

// -----------------------------------------------------------------------------
// A device and the four streams that belong to it.
//
// One value type, created under its own `DeviceScope`, so a per-device object is
// built **on** its device: the stream handles a runtime creates are bound to the
// device current at creation, so creating them under the wrong one is a silent bug
// that only surfaces as an illegal address much later. The scope makes the binding
// explicit and restores the previous device on the way out, so building the next
// device's context starts from a known state.
//
// It is deliberately thin: the split is by device, not by traffic (that is
// `DeviceStreams`) and not by rank or stage (that is the topology and, later, the
// stage host).
// -----------------------------------------------------------------------------

#include "infrastructure/device_streams.hpp"
#include "platform/device.hpp"

namespace aeon::core {

struct DeviceContext {
    int device{-1};
    DeviceStreams streams;

    // Create the stream set on `device`, then restore whatever device was current.
    // The scope is what binds the streams to `device`; the restore is what lets a
    // caller create several contexts in a row without tracking the device itself.
    static DeviceContext create(int device) {
        DeviceContext context;
        context.device = device;
        {
            DeviceScope scope(device);
            context.streams = DeviceStreams::create();
        }
        return context;
    }

    void destroy() noexcept {
        streams.destroy();
        device = -1;
    }
};

} // namespace aeon::core
