#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

namespace aeon::core {

// Automatically selects a high-performance compute GPU that is NOT driving a display.
inline int select_compute_device(bool verbose = true) {
    // If HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES is set, device 0 is already isolated
    const char* hip_vis = std::getenv("HIP_VISIBLE_DEVICES");
    const char* cuda_vis = std::getenv("CUDA_VISIBLE_DEVICES");
    if (hip_vis != nullptr || cuda_vis != nullptr) {
        (void)hipSetDevice(0);
        if (verbose) {
            std::cout << "[Aeon Device] Using device 0 under isolated visibility ("
                      << (hip_vis ? "HIP_VISIBLE_DEVICES" : "CUDA_VISIBLE_DEVICES")
                      << "=" << (hip_vis ? hip_vis : cuda_vis) << ")" << std::endl;
        }
        return 0;
    }

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count <= 0) {
        std::cerr << "[Aeon Device] Warning: Failed to get device count or no devices found." << std::endl;
        (void)hipSetDevice(0);
        return 0;
    }

    // Inspect each device to skip the display GPU.
    // On this host, PCI bus 0x46 is driving the active desktop monitor (DP-10).
    // Devices with PCI bus 0x43 or 0x63 are dedicated headless compute GPUs.
    int chosen_dev = 0;
    for (int dev = 0; dev < device_count; ++dev) {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, dev) == hipSuccess) {
            if (props.pciBusID != 0x46) {
                chosen_dev = dev;
                break;
            }
        }
    }

    (void)hipSetDevice(chosen_dev);
    if (verbose) {
        hipDeviceProp_t props;
        (void)hipGetDeviceProperties(&props, chosen_dev);
        std::cout << "[Aeon Device] Selected headless compute GPU [" << chosen_dev << "]: "
                  << props.name << " (PCI Bus 0x" << std::hex << props.pciBusID << std::dec
                  << ") — display GPU bypassed." << std::endl;
    }
    return chosen_dev;
}

} // namespace aeon::core
