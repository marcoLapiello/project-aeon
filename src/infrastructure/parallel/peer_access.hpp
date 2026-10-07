#pragma once

// -----------------------------------------------------------------------------
// Device-to-device peer access.
//
// A pipeline hands a token's residual from one device to the next, and the copy
// (`hipMemcpyPeerAsync`) is correct **with or without** peer access: without it the
// runtime stages the bytes through host memory, which is slower but produces the
// same result. That is why this is best-effort and never a refusal — a rig without a
// usable peer link still runs the same topology, and the code keeps no assumption
// about a specific interconnect.
//
// `hipDeviceEnablePeerAccess` is enabled **from the accessing device**: to let
// device B read device A's memory, the call is made while B is current.
// -----------------------------------------------------------------------------

#include "platform/device.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

namespace aeon::core {

// Enable peer access for every ordered pair in `devices` where the hardware allows
// it. Devices not present in the list, pairs the driver reports as inaccessible, and
// an already-enabled pair are all silently skipped.
inline void enable_peer_access(const std::vector<int>& devices, bool verbose = false) {
    for (int accessor : devices) {
        for (int peer : devices) {
            if (accessor == peer) continue;

            int can_access = 0;
            if (hipDeviceCanAccessPeer(&can_access, accessor, peer) != hipSuccess ||
                can_access == 0) {
                if (verbose) {
                    std::printf("[Peer] device %d cannot access device %d directly\n", accessor,
                                peer);
                }
                continue;
            }

            DeviceScope scope(accessor);
            // Best-effort: an already-enabled pair, an unsupported link, and a driver
            // that declines are all non-fatal — the copy still corrects itself.
            (void)hipDeviceEnablePeerAccess(peer, 0);
        }
    }
}

} // namespace aeon::core
