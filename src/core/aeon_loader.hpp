#pragma once

#include "core/safetensors_loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace aeon::core {

constexpr uint32_t AEON_SECTOR_SIZE = 4096;
constexpr size_t   AEON_EXPERT_BYTES = 14155776; // 3,456 sectors of 4096 bytes

// Layout offsets inside one contiguous 14,155,776-byte expert payload:
// [W1_packed (4MB)] [W1_scale (512KB)] [W2_packed (4MB)] [W2_scale (512KB)] [W3_packed (4MB)] [W3_scale (512KB)]
constexpr size_t AEON_W1_PACKED_OFFSET = 0;
constexpr size_t AEON_W1_SCALE_OFFSET  = 4194304;
constexpr size_t AEON_W2_PACKED_OFFSET = 4718592;
constexpr size_t AEON_W2_SCALE_OFFSET  = 8912896;
constexpr size_t AEON_W3_PACKED_OFFSET = 9437184;
constexpr size_t AEON_W3_SCALE_OFFSET  = 13631488;

class AeonModelLoader {
public:
    struct ExpertLocation {
        uint64_t file_offset{0};
        size_t byte_length{0};
    };

    AeonModelLoader() = default;
    ~AeonModelLoader() {
        close_all();
    }

    AeonModelLoader(const AeonModelLoader&) = delete;
    AeonModelLoader& operator=(const AeonModelLoader&) = delete;
    AeonModelLoader(AeonModelLoader&&) = default;
    AeonModelLoader& operator=(AeonModelLoader&&) = default;

    void open_model(const std::string& model_dir) {
        model_dir_ = model_dir;
        open_dense(model_dir + "/model_dense.aeon");
        open_experts(model_dir + "/model_experts.aeon", model_dir + "/model_experts.index");
    }

    // Dense tensor query interface matching SafetensorsLoader
    bool has_tensor(const std::string& name) const {
        return dense_tensors_.find(name) != dense_tensors_.end();
    }

    const LoadedTensor& get_tensor(const std::string& name) const {
        auto it = dense_tensors_.find(name);
        if (it == dense_tensors_.end()) {
            throw std::runtime_error("AeonModelLoader: Dense tensor not found: " + name);
        }
        return it->second;
    }

    template<typename T>
    const T* get_data_ptr(const std::string& name) const {
        return reinterpret_cast<const T*>(get_tensor(name).data);
    }

    size_t total_dense_tensors() const {
        return dense_tensors_.size();
    }

    ExpertLocation get_expert_location(uint32_t layer_id, uint32_t expert_id) const {
        if (layer_id >= num_layers_ || expert_id >= experts_per_layer_) {
            throw std::runtime_error("AeonModelLoader: Invalid expert index: L" + std::to_string(layer_id) +
                                     " E" + std::to_string(expert_id));
        }

        uint64_t slot_idx = static_cast<uint64_t>(layer_id) * experts_per_layer_ + expert_id;
        uint64_t offset = expert_offsets_[slot_idx];
        if ((offset % AEON_SECTOR_SIZE) != 0 ||
            offset > experts_file_size_ || AEON_EXPERT_BYTES > experts_file_size_ - offset) {
            throw std::runtime_error("AeonModelLoader: Expert location is invalid or not sector aligned");
        }
        return ExpertLocation{offset, AEON_EXPERT_BYTES};
    }

    // Expert access interface: provides zero-copy pointers to an expert's contiguous payload in Host DDR.
    const uint8_t* get_expert_data(uint32_t layer_id, uint32_t expert_id) const {
        ExpertLocation location = get_expert_location(layer_id, expert_id);
        return experts_mmap_base_ + location.file_offset;
    }

    int expert_direct_fd() const {
        if (experts_direct_fd_ < 0) {
            throw std::runtime_error("AeonModelLoader: Direct expert file descriptor is not open");
        }
        return experts_direct_fd_;
    }

    uint32_t num_layers() const { return num_layers_; }
    uint32_t experts_per_layer() const { return experts_per_layer_; }

    void close_all() {
        if (dense_mmap_base_ && dense_mmap_base_ != MAP_FAILED) {
            ::munmap(const_cast<uint8_t*>(dense_mmap_base_), dense_file_size_);
            dense_mmap_base_ = nullptr;
        }
        if (dense_fd_ >= 0) {
            ::close(dense_fd_);
            dense_fd_ = -1;
        }

        if (experts_mmap_base_ && experts_mmap_base_ != MAP_FAILED) {
            ::munmap(const_cast<uint8_t*>(experts_mmap_base_), experts_file_size_);
            experts_mmap_base_ = nullptr;
        }
        if (experts_fd_ >= 0) {
            ::close(experts_fd_);
            experts_fd_ = -1;
        }
        if (experts_direct_fd_ >= 0) {
            ::close(experts_direct_fd_);
            experts_direct_fd_ = -1;
        }

        dense_tensors_.clear();
        expert_offsets_.clear();
    }

private:
    void open_dense(const std::string& dense_path) {
        dense_fd_ = ::open(dense_path.c_str(), O_RDONLY);
        if (dense_fd_ < 0) {
            throw std::runtime_error("AeonModelLoader: Failed to open dense file: " + dense_path + " (" + strerror(errno) + ")");
        }

        struct stat st;
        if (fstat(dense_fd_, &st) != 0) {
            ::close(dense_fd_);
            throw std::runtime_error("AeonModelLoader: Failed to stat dense file: " + dense_path);
        }
        dense_file_size_ = st.st_size;

        void* mapped = ::mmap(nullptr, dense_file_size_, PROT_READ, MAP_SHARED, dense_fd_, 0);
        if (mapped == MAP_FAILED) {
            ::close(dense_fd_);
            throw std::runtime_error("AeonModelLoader: Failed to mmap dense file: " + dense_path + " (" + strerror(errno) + ")");
        }
        dense_mmap_base_ = static_cast<const uint8_t*>(mapped);

        // Header: Magic (12 bytes), Version (4 bytes), Count (4 bytes), DirLen (8 bytes) = 28 bytes
        if (dense_file_size_ < 28) {
            throw std::runtime_error("AeonModelLoader: Dense file too small for header: " + dense_path);
        }

        const char* magic = reinterpret_cast<const char*>(dense_mmap_base_);
        if (std::memcmp(magic, "AEON_DENSE\x00\x00", 12) != 0) {
            throw std::runtime_error("AeonModelLoader: Invalid dense magic in " + dense_path);
        }

        uint64_t dir_len = *reinterpret_cast<const uint64_t*>(dense_mmap_base_ + 20);

        if (28 + dir_len > dense_file_size_) {
            throw std::runtime_error("AeonModelLoader: Directory length exceeds file size!");
        }

        std::string dir_json(reinterpret_cast<const char*>(dense_mmap_base_ + 28), dir_len);

        // Calculate 4KB sector aligned data payload start
        size_t pre_data_len = 28 + dir_len;
        size_t pad_bytes = (AEON_SECTOR_SIZE - (pre_data_len % AEON_SECTOR_SIZE)) % AEON_SECTOR_SIZE;
        size_t data_start = pre_data_len + pad_bytes;

        // Fast minimal JSON parser for tensor directory array of objects
        // Directory schema: [ { "name": str, "offset": int, "size": int, "dtype": str, "shape": [...] }, ... ]
        parse_dense_directory(dir_json, dense_mmap_base_ + data_start);
        std::cout << "[AeonModelLoader] Loaded dense container: " << dense_tensors_.size()
                  << " tensors (data starts at 0x" << std::hex << data_start << std::dec << ")." << std::endl;
    }

    void open_experts(const std::string& experts_path, const std::string& index_path) {
        // 1. Read index table
        int idx_fd = ::open(index_path.c_str(), O_RDONLY);
        if (idx_fd < 0) {
            throw std::runtime_error("AeonModelLoader: Failed to open expert index: " + index_path);
        }

        struct stat st_idx;
        if (fstat(idx_fd, &st_idx) != 0) {
            ::close(idx_fd);
            throw std::runtime_error("AeonModelLoader: Failed to stat expert index!");
        }

        std::vector<uint8_t> idx_buf(st_idx.st_size);
        ssize_t rd = ::read(idx_fd, idx_buf.data(), idx_buf.size());
        ::close(idx_fd);
        if (rd != static_cast<ssize_t>(idx_buf.size())) {
            throw std::runtime_error("AeonModelLoader: Incomplete read of expert index!");
        }

        // Header: Magic (12 bytes), Version (4B), Layers (4B), ExpertsPerLayer (4B), ExpertBytes (8B) = 32 bytes
        if (idx_buf.size() < 32) {
            throw std::runtime_error("AeonModelLoader: Expert index too small for header!");
        }

        if (std::memcmp(idx_buf.data(), "AEON_EXPERTS", 12) != 0) {
            throw std::runtime_error("AeonModelLoader: Invalid expert index magic!");
        }

        num_layers_ = *reinterpret_cast<const uint32_t*>(idx_buf.data() + 16);
        experts_per_layer_ = *reinterpret_cast<const uint32_t*>(idx_buf.data() + 20);
        uint64_t expert_bytes = *reinterpret_cast<const uint64_t*>(idx_buf.data() + 24);

        if (expert_bytes != AEON_EXPERT_BYTES) {
            throw std::runtime_error("AeonModelLoader: Unexpected expert bytes in index!");
        }

        uint32_t total_experts = num_layers_ * experts_per_layer_;
        expert_offsets_.resize(total_experts);

        const uint64_t* entries = reinterpret_cast<const uint64_t*>(idx_buf.data() + 32);
        for (uint32_t i = 0; i < total_experts; ++i) {
            expert_offsets_[i] = entries[i * 2]; // (offset, size)
        }

        // 2. Mmap model_experts.aeon
        experts_fd_ = ::open(experts_path.c_str(), O_RDONLY);
        if (experts_fd_ < 0) {
            throw std::runtime_error("AeonModelLoader: Failed to open experts file: " + experts_path);
        }

        struct stat st_exp;
        if (fstat(experts_fd_, &st_exp) != 0) {
            ::close(experts_fd_);
            throw std::runtime_error("AeonModelLoader: Failed to stat experts file!");
        }
        experts_file_size_ = st_exp.st_size;

        void* mapped = ::mmap(nullptr, experts_file_size_, PROT_READ, MAP_SHARED, experts_fd_, 0);
        if (mapped == MAP_FAILED) {
            ::close(experts_fd_);
            experts_fd_ = -1;
            throw std::runtime_error("AeonModelLoader: Failed to mmap experts file: " + experts_path);
        }
        experts_mmap_base_ = static_cast<const uint8_t*>(mapped);

        experts_direct_fd_ = ::open(experts_path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (experts_direct_fd_ < 0) {
            ::munmap(const_cast<uint8_t*>(experts_mmap_base_), experts_file_size_);
            experts_mmap_base_ = nullptr;
            ::close(experts_fd_);
            experts_fd_ = -1;
            throw std::runtime_error("AeonModelLoader: Failed to open expert file for O_DIRECT: " +
                                     experts_path + " (" + strerror(errno) + ")");
        }

        std::cout << "[AeonModelLoader] Loaded experts container: " << num_layers_
                  << " layers x " << experts_per_layer_ << " experts ("
                  << (experts_file_size_ / (1024 * 1024 * 1024)) << " GB mmaped)." << std::endl;
    }

    void parse_dense_directory(const std::string& dir_json, const uint8_t* payload_base) {
        // Fast streaming parser for the known directory format:
        // [ { "name": "...", "offset": 123, "size": 456, "dtype": "...", "shape": [...] }, ... ]
        size_t pos = 0;
        while (pos < dir_json.size()) {
            size_t name_tag = dir_json.find("\"name\":", pos);
            if (name_tag == std::string::npos) break;

            size_t name_start = dir_json.find('"', name_tag + 7);
            if (name_start == std::string::npos) break;
            size_t name_end = dir_json.find('"', name_start + 1);
            if (name_end == std::string::npos) break;
            std::string name = dir_json.substr(name_start + 1, name_end - name_start - 1);

            size_t off_tag = dir_json.find("\"offset\":", name_end);
            if (off_tag == std::string::npos) break;
            size_t off_start = dir_json.find_first_of("0123456789", off_tag + 9);
            size_t off_end = dir_json.find_first_not_of("0123456789", off_start);
            int64_t offset = std::stoll(dir_json.substr(off_start, off_end - off_start));

            size_t sz_tag = dir_json.find("\"size\":", off_end);
            if (sz_tag == std::string::npos) break;
            size_t sz_start = dir_json.find_first_of("0123456789", sz_tag + 7);
            size_t sz_end = dir_json.find_first_not_of("0123456789", sz_start);
            int64_t size = std::stoll(dir_json.substr(sz_start, sz_end - sz_start));

            size_t dt_tag = dir_json.find("\"dtype\":", sz_end);
            std::string dtype = "unknown";
            size_t next_scan = sz_end;
            if (dt_tag != std::string::npos && dt_tag < dir_json.find("\"name\":", sz_end)) {
                size_t dt_s = dir_json.find('"', dt_tag + 8);
                size_t dt_e = dir_json.find('"', dt_s + 1);
                if (dt_s != std::string::npos && dt_e != std::string::npos) {
                    dtype = dir_json.substr(dt_s + 1, dt_e - dt_s - 1);
                    next_scan = dt_e;
                }
            }

            LoadedTensor lt;
            lt.data = payload_base + offset;
            lt.byte_size = size;
            lt.dtype = dtype;

            dense_tensors_[name] = lt;
            pos = next_scan;
        }
    }

    std::string model_dir_;
    int dense_fd_{-1};
    size_t dense_file_size_{0};
    const uint8_t* dense_mmap_base_{nullptr};
    std::unordered_map<std::string, LoadedTensor> dense_tensors_;

    int experts_fd_{-1};
    int experts_direct_fd_{-1};
    size_t experts_file_size_{0};
    const uint8_t* experts_mmap_base_{nullptr};
    uint32_t num_layers_{0};
    uint32_t experts_per_layer_{0};
    std::vector<uint64_t> expert_offsets_;
};

} // namespace aeon::core
