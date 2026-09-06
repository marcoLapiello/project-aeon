#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void print_separator(char c = '=', int len = 80) {
    std::cout << std::string(len, c) << "\n";
}

int main() {
    print_separator('=');
    std::cout << "               Project Aeon — Hardware & Topology Discovery\n";
    print_separator('=');

    int driver_version = 0;
    int runtime_version = 0;
    (void)hipDriverGetVersion(&driver_version);
    (void)hipRuntimeGetVersion(&runtime_version);

    std::cout << "HIP Driver Version  : " << driver_version << "\n";
    std::cout << "HIP Runtime Version : " << runtime_version << "\n";

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess) {
        std::cerr << "Failed to query device count: " << hipGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "Total Devices Found : " << device_count << "\n";
    print_separator('-');

    if (device_count == 0) {
        std::cerr << "Error: No HIP-compatible GPUs detected.\n";
        return 1;
    }

    for (int i = 0; i < device_count; ++i) {
        hipDeviceProp_t prop;
        err = hipGetDeviceProperties(&prop, i);
        if (err != hipSuccess) {
            std::cerr << "Failed to get properties for device " << i << ": "
                      << hipGetErrorString(err) << "\n";
            continue;
        }

        std::cout << "GPU [" << i << "]: " << prop.name << "\n";
        std::cout << "  - GFX Target Arch       : " << prop.gcnArchName << "\n";
        std::cout << "  - Compute Units (CUs)   : " << prop.multiProcessorCount << "\n";
        std::cout << "  - Warp / Wavefront Size : " << prop.warpSize
                  << (prop.warpSize == 32 ? " (Wave32 - Verified)" : " (Wave64)") << "\n";
        std::cout << "  - Total Global VRAM     : "
                  << std::fixed << std::setprecision(2)
                  << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
        std::cout << "  - Shared Mem per Block  : " << (prop.sharedMemPerBlock / 1024.0) << " KB\n";
        std::cout << "  - Max Threads per Block : " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  - Max Clock Frequency   : " << (prop.clockRate / 1000) << " MHz\n";
        std::cout << "  - Memory Bus Width      : " << prop.memoryBusWidth << " bits\n";
        std::cout << "  - Memory Clock Rate     : " << (prop.memoryClockRate / 1000) << " MHz\n";
        std::cout << "  - PCIe Domain:Bus:Dev   : "
                  << std::hex << prop.pciDomainID << ":" << prop.pciBusID << ":" << prop.pciDeviceID
                  << std::dec << "\n";
        std::cout << "  - Managed / Pageable Mem: " << (prop.managedMemory ? "Yes" : "No") << "\n";
        std::cout << "  - Concurrent Kernels    : " << (prop.concurrentKernels ? "Yes" : "No") << "\n";
        std::cout << "  - Async Engine Count    : " << prop.asyncEngineCount << "\n";
        print_separator('-');
    }

    // P2P Access Matrix
    if (device_count > 1) {
        std::cout << "Peer-to-Peer (P2P) Direct Access Matrix:\n\n";
        std::cout << "        ";
        for (int j = 0; j < device_count; ++j) {
            std::cout << "GPU " << j << "   ";
        }
        std::cout << "\n";

        for (int i = 0; i < device_count; ++i) {
            std::cout << "GPU " << i << " : ";
            for (int j = 0; j < device_count; ++j) {
                if (i == j) {
                    std::cout << "  --    ";
                } else {
                    int can_access = 0;
                    (void)hipDeviceCanAccessPeer(&can_access, i, j);
                    std::cout << (can_access ? " YES    " : "  NO    ");
                }
            }
            std::cout << "\n";
        }
        print_separator('=');
    }

    return 0;
}
