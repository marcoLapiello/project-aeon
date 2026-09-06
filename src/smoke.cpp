#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess) {
        std::cerr << "HIP initialization error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "[Project Aeon] Toolchain and HIP runtime verified successfully." << std::endl;
    std::cout << "[Project Aeon] Detected " << device_count << " HIP device(s)." << std::endl;
    return 0;
}
