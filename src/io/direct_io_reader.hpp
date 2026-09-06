#pragma once

#include "aligned_allocator.hpp"

#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace aeon::io {

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
    explicit DirectIOReader(uint32_t queue_depth = 64) : queue_depth_(queue_depth) {
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

    // Read synchronously via O_DIRECT io_uring submission & completion
    // Both buffer and file offset MUST be sector-aligned (4096 bytes)
    size_t read_direct(int fd, void* aligned_buf, size_t bytes, uint64_t file_offset) {
        if (((uintptr_t)aligned_buf % SECTOR_SIZE) != 0) {
            throw std::invalid_argument("DirectIOReader: destination buffer is not 4KB sector aligned!");
        }
        if ((file_offset % SECTOR_SIZE) != 0) {
            throw std::invalid_argument("DirectIOReader: file offset is not 4KB sector aligned!");
        }

        uint32_t tail = *sring_tail_;
        uint32_t index = tail & sring_mask_;

        struct io_uring_sqe* sqe = &sqes_[index];
        std::memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ;
        sqe->fd = fd;
        sqe->addr = (uint64_t)aligned_buf;
        sqe->len = (uint32_t)bytes;
        sqe->off = file_offset;
        sqe->user_data = 0xAE01;

        sring_array_[index] = index;
        __atomic_store_n(sring_tail_, tail + 1, __ATOMIC_RELEASE);

        // Submit to kernel and wait for 1 completion
        int ret = sys_io_uring_enter(ring_fd_, 1, 1, IORING_ENTER_GETEVENTS, nullptr);
        if (ret < 0) {
            throw std::runtime_error("DirectIOReader: io_uring_enter failed: " + std::string(strerror(errno)));
        }

        // Harvest completion from CQ
        uint32_t head = __atomic_load_n(cring_head_, __ATOMIC_ACQUIRE);
        if (head == *cring_tail_) {
            throw std::runtime_error("DirectIOReader: no CQE available after io_uring_enter");
        }

        struct io_uring_cqe* cqe = &cqes_[head & cring_mask_];
        int res = cqe->res;
        __atomic_store_n(cring_head_, head + 1, __ATOMIC_RELEASE);

        if (res < 0) {
            throw std::runtime_error("DirectIOReader: read error: " + std::string(strerror(-res)));
        }
        return static_cast<size_t>(res);
    }

private:
    uint32_t queue_depth_{64};
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
