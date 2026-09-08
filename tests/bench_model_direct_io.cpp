#include "core/aeon_loader.hpp"
#include "io/aligned_allocator.hpp"
#include "io/direct_io_reader.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Request = std::pair<uint32_t, uint32_t>;

constexpr size_t EXPERT_BYTES = aeon::core::AEON_EXPERT_BYTES;

const std::array<Request, 6> CONTIGUOUS_REQUESTS = {{
    {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}
}};

const std::array<Request, 6> SCATTERED_REQUESTS = {{
    {0, 0}, {0, 255}, {1, 17}, {10, 42}, {42, 128}, {42, 255}
}};

double gib_per_second(size_t bytes, double elapsed_seconds) {
    return static_cast<double>(bytes) / elapsed_seconds / (1024.0 * 1024.0 * 1024.0);
}

void print_result(const std::string& label, size_t bytes, double elapsed_seconds) {
    std::cout << std::left << std::setw(32) << label
              << " " << std::right << std::fixed << std::setprecision(2)
              << std::setw(8) << gib_per_second(bytes, elapsed_seconds) << " GiB/s"
              << " (" << std::setprecision(2) << elapsed_seconds * 1000.0 << " ms)\n";
}

double run_batched(
    const aeon::core::AeonModelLoader& loader,
    const std::array<Request, 6>& requests,
    size_t batch_width,
    const std::string& label,
    bool force_async
) {
    aeon::io::DirectIOReader reader(64, force_async);
    std::vector<aeon::io::AlignedBuffer> buffers;
    buffers.reserve(requests.size());
    std::array<aeon::core::AeonModelLoader::ExpertLocation, 6> locations{};

    for (size_t i = 0; i < requests.size(); ++i) {
        locations[i] = loader.get_expert_location(requests[i].first, requests[i].second);
        buffers.emplace_back(locations[i].byte_length);
    }

    auto start = std::chrono::steady_clock::now();
    for (size_t first = 0; first < requests.size(); first += batch_width) {
        const size_t count = std::min(batch_width, requests.size() - first);
        for (size_t i = 0; i < count; ++i) {
            const size_t request_index = first + i;
            reader.submit_read(
                loader.expert_direct_fd(),
                buffers[request_index].data(),
                locations[request_index].byte_length,
                locations[request_index].file_offset,
                request_index + 1
            );
        }

        assert(reader.submit_pending_reads() == count);
        for (size_t i = 0; i < count; ++i) {
            const auto completion = reader.wait_for_completion();
            assert(completion.user_data >= first + 1);
            assert(completion.user_data <= first + count);
            assert(completion.result == static_cast<int32_t>(EXPERT_BYTES));
        }
    }
    auto end = std::chrono::steady_clock::now();

    for (size_t i = 0; i < requests.size(); ++i) {
        const uint8_t* expected = loader.get_expert_data(requests[i].first, requests[i].second);
        assert(std::memcmp(buffers[i].data(), expected, EXPERT_BYTES) == 0);
    }

    const double elapsed_seconds = std::chrono::duration<double>(end - start).count();
    print_result(label, requests.size() * EXPERT_BYTES, elapsed_seconds);
    return elapsed_seconds;
}

double run_coalesced(
    const aeon::core::AeonModelLoader& loader,
    bool force_async,
    const std::string& label
) {
    constexpr size_t REQUEST_BYTES = CONTIGUOUS_REQUESTS.size() * EXPERT_BYTES;
    aeon::io::DirectIOReader reader(2, force_async);
    aeon::io::AlignedBuffer buffer(REQUEST_BYTES);
    const auto location = loader.get_expert_location(0, 0);

    auto start = std::chrono::steady_clock::now();
    reader.submit_read(
        loader.expert_direct_fd(),
        buffer.data(),
        REQUEST_BYTES,
        location.file_offset,
        1
    );
    assert(reader.submit_pending_reads() == 1);
    const auto completion = reader.wait_for_completion();
    assert(completion.result == static_cast<int32_t>(REQUEST_BYTES));
    auto end = std::chrono::steady_clock::now();

    const uint8_t* expected = loader.get_expert_data(0, 0);
    assert(std::memcmp(buffer.data(), expected, REQUEST_BYTES) == 0);

    const double elapsed_seconds = std::chrono::duration<double>(end - start).count();
    print_result(label, REQUEST_BYTES, elapsed_seconds);
    return elapsed_seconds;
}

double run_chunked(
    const aeon::core::AeonModelLoader& loader,
    const std::array<Request, 6>& requests,
    size_t chunk_bytes,
    const std::string& label,
    bool force_async
) {
    aeon::io::DirectIOReader reader(64, force_async);
    std::vector<aeon::io::AlignedBuffer> buffers;
    buffers.reserve(requests.size());
    std::array<aeon::core::AeonModelLoader::ExpertLocation, 6> locations{};
    size_t request_count = 0;

    for (size_t i = 0; i < requests.size(); ++i) {
        locations[i] = loader.get_expert_location(requests[i].first, requests[i].second);
        buffers.emplace_back(locations[i].byte_length);
        request_count += (locations[i].byte_length + chunk_bytes - 1) / chunk_bytes;
    }

    auto start = std::chrono::steady_clock::now();
    uint64_t user_data = 1;
    for (size_t i = 0; i < requests.size(); ++i) {
        size_t offset = 0;
        while (offset < locations[i].byte_length) {
            const size_t bytes = std::min(chunk_bytes, locations[i].byte_length - offset);
            reader.submit_read(
                loader.expert_direct_fd(),
                static_cast<uint8_t*>(buffers[i].data()) + offset,
                bytes,
                locations[i].file_offset + offset,
                user_data++
            );
            offset += bytes;
        }
    }

    assert(reader.submit_pending_reads() == request_count);
    for (size_t i = 0; i < request_count; ++i) {
        const auto completion = reader.wait_for_completion();
        assert(completion.result > 0);
    }
    auto end = std::chrono::steady_clock::now();

    for (size_t i = 0; i < requests.size(); ++i) {
        const uint8_t* expected = loader.get_expert_data(requests[i].first, requests[i].second);
        assert(std::memcmp(buffers[i].data(), expected, EXPERT_BYTES) == 0);
    }

    const double elapsed_seconds = std::chrono::duration<double>(end - start).count();
    print_result(label, requests.size() * EXPERT_BYTES, elapsed_seconds);
    return elapsed_seconds;
}

} // namespace

int main() {
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    aeon::core::AeonModelLoader loader;
    loader.open_model(model_dir);

    std::cout << "Model-backed direct-I/O request-shape benchmark\n"
              << "Payload per request: " << std::fixed << std::setprecision(3)
              << (static_cast<double>(EXPERT_BYTES) / (1024.0 * 1024.0)) << " MiB / "
              << (static_cast<double>(EXPERT_BYTES) / 1000000.0) << " MB / "
              << (EXPERT_BYTES / aeon::core::AEON_SECTOR_SIZE) << " sectors\n\n"
              << std::left << std::setw(32) << "Workload" << " Throughput\n"
              << std::string(60, '-') << '\n';

    for (size_t batch_width : {size_t{1}, size_t{2}, size_t{3}, size_t{6}}) {
        run_batched(loader, CONTIGUOUS_REQUESTS, batch_width,
                    "contiguous batch width " + std::to_string(batch_width), false);
    }
    run_coalesced(loader, false, "sync coalesced 84.9 MiB (1 request)");
    run_chunked(loader, CONTIGUOUS_REQUESTS, 4 * 1024 * 1024,
                "sync 4 MiB subreads (24 requests)", false);

    std::cout << '\n';
    for (size_t batch_width : {size_t{1}, size_t{2}, size_t{3}, size_t{6}}) {
        run_batched(loader, SCATTERED_REQUESTS, batch_width,
                    "scattered batch width " + std::to_string(batch_width), false);
    }
    run_chunked(loader, SCATTERED_REQUESTS, 4 * 1024 * 1024,
                "sync scattered 4 MiB subreads (24 requests)", false);

    std::cout << "\nForced IOSQE_ASYNC\n"
              << std::string(60, '-') << '\n';
    run_batched(loader, CONTIGUOUS_REQUESTS, 6, "async contiguous 14.15 MB requests x6", true);
    run_coalesced(loader, true, "async coalesced 84.9 MiB (1 request)");
    run_chunked(loader, CONTIGUOUS_REQUESTS, 4 * 1024 * 1024,
                "async contiguous 4 MiB subreads (24 requests)", true);
    run_batched(loader, SCATTERED_REQUESTS, 6, "async scattered 14.15 MB requests x6", true);
    run_chunked(loader, SCATTERED_REQUESTS, 4 * 1024 * 1024,
                "async scattered 4 MiB subreads (24 requests)", true);

    return 0;
}
