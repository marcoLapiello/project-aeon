#pragma once

// -----------------------------------------------------------------------------
// The direct (`O_DIRECT`) expert-fragment I/O state.
//
// Three collaborators that are one lifecycle, grouped because both the host's
// load-time reads and the tiered supply's asynchronous dispatch share all three:
//
//   * the `io_uring` reader, built for the artifact's payload and sector size;
//   * the completion map the supply fills while dispatches are in flight;
//   * the monotonic user-data counter both the supply and the blocking preload
//     draw request ids from.
//
// None of it is model-specific: it moves fixed-size expert fragments between the
// artifact and device/host memory and knows only the container's format and file
// descriptor, both of which the callers resolve. It lives in `infrastructure/`
// because it is storage plumbing, not model logic.
//
// `read_blocking` is the load/restore path — whole expert payloads, read straight
// from the artifact and batched to the reader's own submission capacity. It is the
// only synchronous expert read; the supply does not use it. Its short final request
// is expected rather than an error: a payload is 3.375 of the 4 MiB chunks.
// -----------------------------------------------------------------------------

#include "infrastructure/core/aeon_loader.hpp"
#include "infrastructure/io/direct_io_reader.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aeon::core {

class ExpertDirectIO {
public:
    ExpertDirectIO() = default;
    ~ExpertDirectIO() { free(); }
    ExpertDirectIO(const ExpertDirectIO&) = delete;
    ExpertDirectIO& operator=(const ExpertDirectIO&) = delete;

    void initialize(uint32_t queue_depth, uint32_t sector_size) {
        reader_ = std::make_unique<aeon::io::DirectIOReader>(queue_depth, true, sector_size);
        completions_.clear();
        next_io_id_ = 1;
    }

    void free() noexcept {
        reader_.reset();
        completions_.clear();
        next_io_id_ = 1;
    }

    aeon::io::DirectIOReader* reader() noexcept { return reader_.get(); }

    std::unordered_map<uint64_t, aeon::io::DirectIOCompletion>* completions() noexcept {
        return &completions_;
    }
    uint64_t* next_id() noexcept { return &next_io_id_; }

    size_t submission_capacity() const noexcept {
        return reader_ ? reader_->submission_capacity() : 0;
    }

    // The chunks one expert payload occupies at the reader's chunk size. The batch
    // sizing for every blocking read is expressed in these, so it is derived here
    // once rather than re-computed beside each read.
    static size_t requests_per_fragment(const ExpertFormatDescriptor& format) noexcept {
        return (format.payload_bytes + aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES - 1) /
               aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
    }

    // A batched, blocking `O_DIRECT` read of whole expert payloads. The batch is
    // sized to the reader's own submission capacity, so the chunk count per request
    // is the artifact's and the short final request is expected rather than an error.
    void read_blocking(const AeonModelLoader& source,
                       const std::vector<std::pair<uint32_t, uint32_t>>& expert_ids,
                       const std::vector<uint8_t*>& destinations) {
        if (expert_ids.size() != destinations.size()) {
            throw std::invalid_argument("ExpertDirectIO: a direct expert read batch is mismatched");
        }
        if (expert_ids.empty()) return;

        const size_t requests_per_expert = requests_per_fragment(source.expert_format());
        const size_t max_batch = std::max<size_t>(
            1, reader_->submission_capacity() / requests_per_expert);

        struct ReadJob {
            uint64_t first_user_data{0};
            size_t request_count{0};
        };

        for (size_t start = 0; start < expert_ids.size(); start += max_batch) {
            const size_t end = std::min(expert_ids.size(), start + max_batch);
            std::vector<ReadJob> jobs;
            size_t total_requests = 0;

            for (size_t i = start; i < end; ++i) {
                const auto location = source.get_expert_location(
                    expert_ids[i].first, expert_ids[i].second);
                const uint64_t first_user_data = next_io_id_;
                const size_t request_count = reader_->submit_read_chunks(
                    source.expert_direct_fd(), destinations[i], location.byte_length,
                    location.file_offset, first_user_data);
                next_io_id_ += request_count;
                total_requests += request_count;
                jobs.push_back(ReadJob{first_user_data, request_count});
            }

            if (reader_->submit_pending_reads() != total_requests) {
                throw std::runtime_error(
                    "ExpertDirectIO: a direct expert batch submitted an unexpected request count");
            }

            std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> completions;
            completions.reserve(total_requests);
            for (size_t i = 0; i < total_requests; ++i) {
                const auto completion = reader_->wait_for_completion();
                completions.emplace(completion.user_data, completion);
            }

            for (const auto& job : jobs) {
                for (size_t chunk = 0; chunk < job.request_count; ++chunk) {
                    const auto it = completions.find(job.first_user_data + chunk);
                    if (it == completions.end() || it->second.result < 0) {
                        throw std::runtime_error(
                            "ExpertDirectIO: a direct expert read failed");
                    }
                    const size_t chunk_offset =
                        chunk * aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
                    const size_t expected = std::min(
                        aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES,
                        source.expert_format().payload_bytes - chunk_offset);
                    if (it->second.result != static_cast<int32_t>(expected)) {
                        throw std::runtime_error(
                            "ExpertDirectIO: a direct expert read returned a short payload");
                    }
                }
            }
        }
    }

private:
    std::unique_ptr<aeon::io::DirectIOReader> reader_;
    std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> completions_;
    uint64_t next_io_id_{1};
};

} // namespace aeon::core
