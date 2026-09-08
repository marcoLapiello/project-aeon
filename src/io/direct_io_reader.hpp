#pragma once

#include "aligned_allocator.hpp"

#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeon::io {

struct DirectIOCompletion {
    uint64_t user_data{0};
    int32_t result{0};
};

// Syscall wrappers for Linux io_uring
inline int sys_io_uring_setup(unsigned entries, struct io_uring_params *p) {
    return (int)syscall(__NR_io_uring_setup, entries, p);
}

inline int sys_io_uring_enter(int fd, unsigned to_submit, unsigned min_complete,
                              unsigned flags, sigset_t *sig) {
    return (int)syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, sig);
}

// Low-overhead C++20 Direct I/O Reader bypassing Linux Page Cache
class DirectIOReader {
public:
    static constexpr size_t DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024;

    explicit DirectIOReader(uint32_t queue_depth = 64, bool force_async = true)
        : queue_depth_(queue_depth), force_async_(force_async) {
        if (queue_depth_ == 0) {
            throw std::invalid_argument("DirectIOReader: queue depth must be greater than zero");
        }

        std::memset(&params_, 0, sizeof(params_));
        ring_fd_ = sys_io_uring_setup(queue_depth_, &params_);
        if (ring_fd_ < 0) {
            throw std::runtime_error("DirectIOReader: io_uring_setup failed: " + std::string(strerror(errno)));
        }

        // Map Submission Queue (SQ)
        uint32_t sq_ring_sz = params_.sq_off.array + params_.sq_entries * sizeof(uint32_t);
        sq_ptr_ = mmap(nullptr, sq_ring_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                       ring_fd_, IORING_OFF_SQ_RING);
        if (sq_ptr_ == MAP_FAILED) {
            close(ring_fd_);
            throw std::runtime_error("DirectIOReader: mmap SQ ring failed: " + std::string(strerror(errno)));
        }

        // Map SQ Entries
        uint32_t sqes_sz = params_.sq_entries * sizeof(struct io_uring_sqe);
        sqes_ = static_cast<struct io_uring_sqe*>(
            mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                 ring_fd_, IORING_OFF_SQES));
        if (sqes_ == MAP_FAILED) {
            munmap(sq_ptr_, sq_ring_sz);
            close(ring_fd_);
            throw std::runtime_error("DirectIOReader: mmap SQEs failed: " + std::string(strerror(errno)));
        }

        // Map Completion Queue (CQ)
        uint32_t cq_ring_sz = params_.cq_off.cqes + params_.cq_entries * sizeof(struct io_uring_cqe);
        cq_ptr_ = mmap(nullptr, cq_ring_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                       ring_fd_, IORING_OFF_CQ_RING);
        if (cq_ptr_ == MAP_FAILED) {
            munmap(sqes_, sqes_sz);
            munmap(sq_ptr_, sq_ring_sz);
            close(ring_fd_);
            throw std::runtime_error("DirectIOReader: mmap CQ ring failed: " + std::string(strerror(errno)));
        }

        // Bind ring field pointers
        auto sq_byte_ptr = static_cast<char*>(sq_ptr_);
        sring_head_ = reinterpret_cast<uint32_t*>(sq_byte_ptr + params_.sq_off.head);
        sring_tail_ = reinterpret_cast<uint32_t*>(sq_byte_ptr + params_.sq_off.tail);
        sring_mask_ = *reinterpret_cast<uint32_t*>(sq_byte_ptr + params_.sq_off.ring_mask);
        sring_array_ = reinterpret_cast<uint32_t*>(sq_byte_ptr + params_.sq_off.array);

        auto cq_byte_ptr = static_cast<char*>(cq_ptr_);
        cring_head_ = reinterpret_cast<uint32_t*>(cq_byte_ptr + params_.cq_off.head);
        cring_tail_ = reinterpret_cast<uint32_t*>(cq_byte_ptr + params_.cq_off.tail);
        cring_mask_ = *reinterpret_cast<uint32_t*>(cq_byte_ptr + params_.cq_off.ring_mask);
        cqes_ = reinterpret_cast<struct io_uring_cqe*>(cq_byte_ptr + params_.cq_off.cqes);
    }

    ~DirectIOReader() noexcept {
        if (cq_ptr_ && cq_ptr_ != MAP_FAILED) {
            uint32_t cq_ring_sz = params_.cq_off.cqes + params_.cq_entries * sizeof(struct io_uring_cqe);
            munmap(cq_ptr_, cq_ring_sz);
        }
        if (sqes_ && sqes_ != MAP_FAILED) {
            uint32_t sqes_sz = params_.sq_entries * sizeof(struct io_uring_sqe);
            munmap(sqes_, sqes_sz);
        }
        if (sq_ptr_ && sq_ptr_ != MAP_FAILED) {
            uint32_t sq_ring_sz = params_.sq_off.array + params_.sq_entries * sizeof(uint32_t);
            munmap(sq_ptr_, sq_ring_sz);
        }
        if (ring_fd_ >= 0) {
            close(ring_fd_);
        }
    }

    void submit_read(
        int fd,
        void* aligned_buf,
        size_t bytes,
        uint64_t file_offset,
        uint64_t user_data
    ) {
        validate_request(fd, aligned_buf, bytes, file_offset);

        uint32_t head = __atomic_load_n(sring_head_, __ATOMIC_ACQUIRE);
        uint32_t tail = __atomic_load_n(sring_tail_, __ATOMIC_ACQUIRE);
        if (tail - head >= params_.sq_entries) {
            throw std::runtime_error("DirectIOReader: submission queue is full; submit pending reads first");
        }

        uint32_t index = tail & sring_mask_;
        struct io_uring_sqe* sqe = &sqes_[index];
        std::memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ;
        if (force_async_) {
            sqe->flags |= IOSQE_ASYNC;
        }
        sqe->fd = fd;
        sqe->addr = reinterpret_cast<uint64_t>(aligned_buf);
        sqe->len = static_cast<uint32_t>(bytes);
        sqe->off = file_offset;
        sqe->user_data = user_data;

        sring_array_[index] = index;
        __atomic_store_n(sring_tail_, tail + 1, __ATOMIC_RELEASE);
    }

    size_t submit_read_chunks(
        int fd,
        void* aligned_buf,
        size_t bytes,
        uint64_t file_offset,
        uint64_t first_user_data,
        size_t chunk_bytes = DEFAULT_CHUNK_BYTES
    ) {
        if (chunk_bytes == 0 || (chunk_bytes % SECTOR_SIZE) != 0) {
            throw std::invalid_argument("DirectIOReader: chunk length must be a non-zero 4KB multiple");
        }

        size_t request_count = 0;
        size_t offset = 0;
        while (offset < bytes) {
            const size_t request_bytes = std::min(chunk_bytes, bytes - offset);
            submit_read(
                fd,
                static_cast<uint8_t*>(aligned_buf) + offset,
                request_bytes,
                file_offset + offset,
                first_user_data + request_count
            );
            offset += request_bytes;
            ++request_count;
        }
        return request_count;
    }

    size_t submit_pending_reads() {
        uint32_t head = __atomic_load_n(sring_head_, __ATOMIC_ACQUIRE);
        uint32_t tail = __atomic_load_n(sring_tail_, __ATOMIC_ACQUIRE);
        uint32_t pending = tail - head;
        size_t submitted = 0;

        while (pending > 0) {
            int ret = sys_io_uring_enter(ring_fd_, pending, 0, 0, nullptr);
            if (ret < 0) {
                throw std::runtime_error("DirectIOReader: io_uring_enter submit failed: " +
                                         std::string(strerror(errno)));
            }
            if (ret == 0) {
                throw std::runtime_error("DirectIOReader: io_uring_enter submitted zero requests");
            }
            submitted += static_cast<size_t>(ret);
            pending -= static_cast<uint32_t>(ret);
        }
        return submitted;
    }

    uint32_t submission_capacity() const noexcept {
        return params_.sq_entries;
    }

    DirectIOCompletion wait_for_completion() {
        for (;;) {
            uint32_t head = __atomic_load_n(cring_head_, __ATOMIC_ACQUIRE);
            uint32_t tail = __atomic_load_n(cring_tail_, __ATOMIC_ACQUIRE);
            if (head != tail) {
                struct io_uring_cqe* cqe = &cqes_[head & cring_mask_];
                DirectIOCompletion completion{
                    .user_data = cqe->user_data,
                    .result = cqe->res
                };
                __atomic_store_n(cring_head_, head + 1, __ATOMIC_RELEASE);
                return completion;
            }

            int ret = sys_io_uring_enter(ring_fd_, 0, 1, IORING_ENTER_GETEVENTS, nullptr);
            if (ret < 0) {
                throw std::runtime_error("DirectIOReader: io_uring_enter wait failed: " +
                                         std::string(strerror(errno)));
            }
        }
    }

    std::vector<DirectIOCompletion> wait_for_completions(size_t count) {
        std::vector<DirectIOCompletion> completions;
        completions.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            completions.push_back(wait_for_completion());
        }
        return completions;
    }

    // Read synchronously via O_DIRECT io_uring submission and completion.
    size_t read_direct(int fd, void* aligned_buf, size_t bytes, uint64_t file_offset) {
        submit_read(fd, aligned_buf, bytes, file_offset, 0xAE01);
        submit_pending_reads();
        DirectIOCompletion completion = wait_for_completion();
        if (completion.result < 0) {
            throw std::runtime_error("DirectIOReader: read error: " +
                                     std::string(strerror(-completion.result)));
        }
        return static_cast<size_t>(completion.result);
    }

private:
    static void validate_request(int fd, const void* aligned_buf, size_t bytes, uint64_t file_offset) {
        if (fd < 0) {
            throw std::invalid_argument("DirectIOReader: file descriptor must be valid");
        }
        if (aligned_buf == nullptr || (reinterpret_cast<uintptr_t>(aligned_buf) % SECTOR_SIZE) != 0) {
            throw std::invalid_argument("DirectIOReader: destination buffer is not 4KB sector aligned");
        }
        if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max() || (bytes % SECTOR_SIZE) != 0) {
            throw std::invalid_argument("DirectIOReader: read length must be a non-zero 4KB multiple");
        }
        if ((file_offset % SECTOR_SIZE) != 0) {
            throw std::invalid_argument("DirectIOReader: file offset is not 4KB sector aligned");
        }
    }

    uint32_t queue_depth_{64};
    bool force_async_{false};
    int ring_fd_{-1};
    struct io_uring_params params_;

    void* sq_ptr_{nullptr};
    void* cq_ptr_{nullptr};
    struct io_uring_sqe* sqes_{nullptr};

    uint32_t* sring_head_{nullptr};
    uint32_t* sring_tail_{nullptr};
    uint32_t sring_mask_{0};
    uint32_t* sring_array_{nullptr};

    uint32_t* cring_head_{nullptr};
    uint32_t* cring_tail_{nullptr};
    uint32_t cring_mask_{0};
    struct io_uring_cqe* cqes_{nullptr};
};

} // namespace aeon::io
