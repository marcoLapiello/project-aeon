#include "io/aligned_allocator.hpp"
#include "io/direct_io_reader.hpp"

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    std::cout << "====================================================================" << std::endl;
    std::cout << "  Project Aeon — 4KB Sector-Aligned Direct I/O Reader Validation" << std::endl;
    std::cout << "====================================================================" << std::endl;

    // 1. Verify Aligned Allocator
    std::cout << "[Test 1] Testing 4096-byte aligned memory allocator..." << std::endl;
    constexpr size_t TEST_SIZE = 128 * 1024 * 1024; // 128 MB test buffer
    aeon::io::AlignedBuffer buffer(TEST_SIZE);

    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer.data());
    std::cout << "  - Buffer address: 0x" << std::hex << addr << std::dec << "\n";
    std::cout << "  - Sector alignment (addr % 4096): " << (addr % 4096) << "\n";

    if ((addr % 4096) != 0) {
        std::cerr << "FAIL: Buffer address is not 4KB aligned!\n";
        return 1;
    }
    std::cout << "  - Allocator 4KB alignment: PASSED\n";

    // 2. Create a temporary test file with deterministic pattern
    const std::string test_path = "/tmp/aeon_direct_io_test.bin";
    std::cout << "\n[Test 2] Preparing temporary file (" << (TEST_SIZE / (1024 * 1024))
              << " MB) at " << test_path << "..." << std::endl;

    // Fill buffer with pattern: each 4KB block has its block index as uint32_t header
    auto* byte_ptr = static_cast<uint8_t*>(buffer.data());
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        byte_ptr[i] = static_cast<uint8_t>((i & 0xFF) ^ 0x5A);
    }

    int write_fd = open(test_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (write_fd < 0) {
        std::cerr << "Failed to open file for O_DIRECT write: " << strerror(errno) << "\n";
        return 1;
    }

    ssize_t written = write(write_fd, buffer.data(), TEST_SIZE);
    close(write_fd);

    if (written != static_cast<ssize_t>(TEST_SIZE)) {
        std::cerr << "Failed to write complete test file: " << strerror(errno) << "\n";
        return 1;
    }
    std::cout << "  - Wrote " << (written / (1024 * 1024)) << " MB via O_DIRECT successfully.\n";

    // 3. Clear buffer memory to ensure we are actually reading from disk
    std::memset(buffer.data(), 0, TEST_SIZE);

    // 4. Read back using aeon::io::DirectIOReader (io_uring + O_DIRECT)
    std::cout << "\n[Test 3] Reading via aeon::io::DirectIOReader (io_uring O_DIRECT)..." << std::endl;
    int read_fd = open(test_path.c_str(), O_RDONLY | O_DIRECT);
    if (read_fd < 0) {
        std::cerr << "Failed to open file for O_DIRECT read: " << strerror(errno) << "\n";
        unlink(test_path.c_str());
        return 1;
    }

    aeon::io::DirectIOReader reader(64);

    // Benchmark reading in chunks (e.g. 16 MB chunks, typical for streaming MoE sub-tensors)
    constexpr size_t CHUNK_SIZE = 16 * 1024 * 1024; // 16 MB
    const size_t num_chunks = TEST_SIZE / CHUNK_SIZE;

    auto start_time = std::chrono::high_resolution_clock::now();

    size_t total_bytes_read = 0;
    for (size_t c = 0; c < num_chunks; ++c) {
        uint64_t offset = c * CHUNK_SIZE;
        void* chunk_dest = static_cast<char*>(buffer.data()) + offset;
        size_t bytes = reader.read_direct(read_fd, chunk_dest, CHUNK_SIZE, offset);
        total_bytes_read += bytes;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    close(read_fd);

    // Clean up temporary file
    unlink(test_path.c_str());

    std::chrono::duration<double> duration = end_time - start_time;
    double gb_read = static_cast<double>(total_bytes_read) / (1024.0 * 1024.0 * 1024.0);
    double throughput_gb_s = gb_read / duration.count();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  - Total Bytes Read  : " << (total_bytes_read / (1024 * 1024)) << " MB\n";
    std::cout << "  - Elapsed Time      : " << (duration.count() * 1000.0) << " ms\n";
    std::cout << "  - Sustained Speed   : " << throughput_gb_s << " GB/s\n";

    // 5. Verify Data Integrity
    std::cout << "\n[Test 4] Verifying payload data integrity..." << std::endl;
    bool integrity_pass = true;
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>((i & 0xFF) ^ 0x5A);
        if (byte_ptr[i] != expected) {
            std::cerr << "FAIL: Byte mismatch at offset " << i << ": expected 0x"
                      << std::hex << (int)expected << " got 0x" << (int)byte_ptr[i] << std::dec << "\n";
            integrity_pass = false;
            break;
        }
    }

    if (integrity_pass) {
        std::cout << "  - All " << TEST_SIZE << " bytes match bit-for-bit with source pattern!\n";
        std::cout << "--------------------------------------------------------------------" << std::endl;
        std::cout << ">>> VERIFICATION PASSED: io_uring direct I/O works seamlessly! <<<" << std::endl;
        std::cout << "====================================================================" << std::endl;
        return 0;
    } else {
        std::cerr << ">>> VERIFICATION FAILED: Data corruption detected! <<<" << std::endl;
        return 1;
    }
}
