#include "core/aeon_loader.hpp"
#include "io/aligned_allocator.hpp"
#include "io/direct_io_reader.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    constexpr size_t EXPERT_COUNT = 6;
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    const std::array<std::pair<uint32_t, uint32_t>, EXPERT_COUNT> requests = {{
        {0, 0},
        {0, 255},
        {1, 17},
        {10, 42},
        {42, 128},
        {42, 255}
    }};

    std::cout << "Testing model-backed batched O_DIRECT expert reads..." << std::endl;

    aeon::core::AeonModelLoader loader;
    loader.open_model(model_dir);
    aeon::io::DirectIOReader reader(64);

    std::vector<aeon::io::AlignedBuffer> staging_buffers;
    staging_buffers.reserve(EXPERT_COUNT);
    std::array<aeon::core::AeonModelLoader::ExpertLocation, EXPERT_COUNT> locations{};
    size_t request_count = 0;
    uint64_t next_user_data = 1;

    for (size_t i = 0; i < EXPERT_COUNT; ++i) {
        locations[i] = loader.get_expert_location(requests[i].first, requests[i].second);
        assert(locations[i].byte_length == aeon::core::AEON_EXPERT_BYTES);
        staging_buffers.emplace_back(locations[i].byte_length);
        const size_t expert_requests = reader.submit_read_chunks(
            loader.expert_direct_fd(),
            staging_buffers[i].data(),
            locations[i].byte_length,
            locations[i].file_offset,
            next_user_data
        );
        request_count += expert_requests;
        next_user_data += expert_requests;
    }

    auto start = std::chrono::high_resolution_clock::now();
    const size_t submitted = reader.submit_pending_reads();
    const auto completions = reader.wait_for_completions(request_count);
    const auto end = std::chrono::high_resolution_clock::now();

    assert(submitted == request_count);
    size_t completed_bytes = 0;
    for (const auto& completion : completions) {
        assert(completion.user_data >= 1 && completion.user_data <= request_count);
        assert(completion.result > 0);
        assert(completion.result % static_cast<int32_t>(aeon::core::AEON_SECTOR_SIZE) == 0);
        completed_bytes += static_cast<size_t>(completion.result);
    }
    assert(completed_bytes == EXPERT_COUNT * aeon::core::AEON_EXPERT_BYTES);

    for (size_t i = 0; i < EXPERT_COUNT; ++i) {
        const uint8_t* expected = loader.get_expert_data(requests[i].first, requests[i].second);
        assert(std::memcmp(staging_buffers[i].data(), expected, locations[i].byte_length) == 0);
    }

    const double elapsed_seconds = std::chrono::duration<double>(end - start).count();
    const double gigabytes = static_cast<double>(EXPERT_COUNT * aeon::core::AEON_EXPERT_BYTES) /
                             (1024.0 * 1024.0 * 1024.0);
    const double throughput = gigabytes / elapsed_seconds;

    std::cout << std::fixed << std::setprecision(2)
              << "  - Experts read       : " << EXPERT_COUNT << " (" << request_count << " aligned reads)\n"
              << "  - Bytes read         : " << (gigabytes * 1024.0) << " MiB\n"
              << "  - Batch elapsed      : " << (elapsed_seconds * 1000.0) << " ms\n"
              << "  - Direct-I/O speed   : " << throughput << " GiB/s\n"
              << "  - Payload integrity  : PASSED\n"
              << ">>> MODEL-BACKED BATCH DIRECT I/O PASSED <<<" << std::endl;

    return 0;
}
