#pragma once

// -----------------------------------------------------------------------------
// Explicit device selection.
//
// The device is chosen by an **index the caller supplies**, never by a heuristic.
// A prior version inspected each device's PCI bus and skipped the one driving a
// display on one particular rig — a rule that is unknowable on another machine and
// silently wrong when the display moves. Portability here means the topology is
// expressed as data (`--device-ids`) and resolved against what the runtime actually
// sees.
//
// `DeviceScope` sets a device for a region and restores the previous one on exit,
// which is what lets a per-rank object (a stream set, a pool) be constructed under
// its own device without leaking that choice to the code that builds the next one.
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace aeon::core {

// Validate `index` against the visible devices, make it current, and (optionally)
// print the name and PCI address. Throws a named error rather than falling back:
// a caller that asked for a device that is not there wants to know.
inline int select_device(int index, bool verbose = true) {
    int device_count = 0;
    const hipError_t count_error = hipGetDeviceCount(&device_count);
    if (count_error != hipSuccess) {
        throw std::runtime_error(std::string("select_device: hipGetDeviceCount: ") +
                                 hipGetErrorString(count_error));
    }
    if (device_count <= 0) {
        throw std::runtime_error("select_device: no HIP devices are visible");
    }
    if (index < 0 || index >= device_count) {
        throw std::runtime_error(
            "select_device: device index " + std::to_string(index) +
            " is out of range for " + std::to_string(device_count) + " visible devices");
    }
    const hipError_t set_error = hipSetDevice(index);
    if (set_error != hipSuccess) {
        throw std::runtime_error("select_device: hipSetDevice(" + std::to_string(index) +
                                 "): " + hipGetErrorString(set_error));
    }
    if (verbose) {
        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, index) == hipSuccess) {
            std::cout << "[Aeon Device] Using GPU [" << index << "]: " << props.name
                      << " (PCI Bus 0x" << std::hex << props.pciBusID << std::dec << ")"
                      << std::endl;
        }
    }
    return index;
}

// RAII device scope: the current device is captured on construction, `index` is
// made current, and the previous one is restored on destruction. The restore is
// best-effort (a destructor cannot throw), but the set is validated so a mistaken
// index is a loud failure rather than a silent run on the wrong device.
class DeviceScope {
public:
    explicit DeviceScope(int index) {
        if (hipGetDevice(&previous_) != hipSuccess) previous_ = 0;
        const hipError_t error = hipSetDevice(index);
        if (error != hipSuccess) {
            throw std::runtime_error("DeviceScope: hipSetDevice(" + std::to_string(index) +
                                     "): " + hipGetErrorString(error));
        }
    }

    ~DeviceScope() { (void)hipSetDevice(previous_); }

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int previous_{0};
};

} // namespace aeon::core

