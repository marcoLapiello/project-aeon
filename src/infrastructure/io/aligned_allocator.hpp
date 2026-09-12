#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

namespace aeon::io {

constexpr size_t SECTOR_SIZE = 4096; // Standard 4KB NVMe logical sector boundary

// Allocate host memory strictly aligned to 4096-byte boundaries for O_DIRECT compliance
inline void* allocate_aligned(size_t bytes, size_t alignment = SECTOR_SIZE) {
    // Round size up to multiple of alignment
    size_t aligned_size = (bytes + alignment - 1) & ~(alignment - 1);
    void* ptr = nullptr;
    int res = posix_memalign(&ptr, alignment, aligned_size);
    if (res != 0 || !ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

// Free memory allocated with allocate_aligned
inline void free_aligned(void* ptr) noexcept {
    if (ptr) {
        std::free(ptr);
    }
}

// RAII Sector-Aligned Buffer
class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t size_bytes, size_t alignment = SECTOR_SIZE)
        : size_(size_bytes), alignment_(alignment) {
        data_ = allocate_aligned(size_bytes, alignment);
    }

    ~AlignedBuffer() noexcept {
        free_aligned(data_);
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            free_aligned(data_);
            data_ = other.data_;
            size_ = other.size_;
            alignment_ = other.alignment_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] size_t alignment() const noexcept { return alignment_; }

private:
    void* data_{nullptr};
    size_t size_{0};
    size_t alignment_{SECTOR_SIZE};
};

} // namespace aeon::io
