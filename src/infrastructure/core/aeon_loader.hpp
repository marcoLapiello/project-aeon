#pragma once

#include "infrastructure/core/aeon_artifact.hpp"
#include "infrastructure/core/json.hpp"
#include "infrastructure/core/loaded_tensor.hpp"
#include "infrastructure/core/model_manifest.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace aeon::core {

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
        const std::string manifest_path = model_dir + "/model_manifest.json";
        if (std::filesystem::exists(manifest_path)) {
            open_model(model_dir, AeonModelManifest::load_from_json(manifest_path));
            return;
        }
        open_model(model_dir, make_current_swizzled_artifact_spec());
    }

    void open_model(const std::string& model_dir, const AeonArtifactSpec& artifact) {
        if (artifact.dense_filename.empty() || artifact.experts_filename.empty() ||
            artifact.index_filename.empty() || artifact.expected_dense_version == 0 ||
            artifact.expected_expert_version == 0 ||
            artifact.expert_sector_size == 0 ||
            (artifact.expert_sector_size & (artifact.expert_sector_size - 1)) != 0) {
            throw std::invalid_argument("AeonModelLoader: artifact specification is incomplete");
        }
        open_model_files(model_dir, artifact);
    }

    void open_model(const std::string& model_dir, const AeonModelManifest& manifest) {
        manifest.validate();
        const auto& backend = ExpertBackendRegistry::resolve(manifest.weight_backend);
        if (backend.format_kind != manifest.artifact.expert_format_kind) {
            throw std::invalid_argument(
                "AeonModelLoader: manifest backend does not match its artifact format");
        }
        open_model(model_dir, manifest.artifact);
        try {
            backend.validate_format(expert_format_);
            manifest.validate_loaded(expert_format_, dense_file_size_);
        } catch (...) {
            close_all();
            throw;
        }
    }

    uint32_t expert_format_version() const {
        return expert_format_version_;
    }

    const ExpertFormatDescriptor& expert_format() const noexcept {
        return expert_format_;
    }

    size_t dense_file_size() const noexcept {
        return dense_file_size_;
    }

    const std::string& model_dir() const noexcept {
        return model_dir_;
    }

    std::vector<std::pair<std::string, LoadedTensor>> dense_tensor_inventory() const {
        std::vector<std::pair<std::string, LoadedTensor>> inventory;
        inventory.reserve(dense_tensors_.size());
        for (const auto& entry : dense_tensors_) inventory.push_back(entry);
        std::sort(inventory.begin(), inventory.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
        return inventory;
    }

private:
    void open_model_files(
        const std::string& model_dir,
        const AeonArtifactSpec& artifact
    ) {
        model_dir_ = model_dir;
        open_dense(model_dir + "/" + artifact.dense_filename, artifact.expected_dense_version);
        open_experts(model_dir + "/" + artifact.experts_filename,
                     model_dir + "/" + artifact.index_filename,
                     artifact.expected_expert_version,
                     artifact.expert_format_kind,
                     artifact.expert_sector_size,
                     artifact.expected_expert_payload_bytes);
    }

public:

    // Dense tensor query interface for the native model container.
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
        if (expert_format_.payload_bytes == 0 ||
            (offset % expert_format_.sector_size) != 0 ||
            offset > experts_file_size_ ||
            expert_format_.payload_bytes > experts_file_size_ - offset) {
            throw std::runtime_error("AeonModelLoader: Expert location is invalid or not sector aligned");
        }
        return ExpertLocation{offset, expert_format_.payload_bytes};
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

    // Drop this process's residency of the dense mapping, keeping one named
    // tensor's span.
    //
    // The mapping is `MAP_SHARED` and read-only, so the pages it faulted in are
    // **clean and file-backed**: releasing them returns the memory to the page
    // cache's free pool without a writeback, and the file stays the backing store,
    // so any later read simply faults the page in again from disk. Nothing is
    // lost and no pointer is invalidated — `data` stays valid for every tensor.
    //
    // The reason to do it at all: after the device has its copies, those pages are
    // dead weight competing for host RAM with the pinned Warm pool, which cannot be
    // reclaimed. Only one dense tensor is read per token (`embed.weight`, by
    // `V4Graph::embed_token`), so that span is the only one worth keeping resident.
    //
    // `MADV_DONTNEED` operates on whole pages, so the kept span is widened outward
    // to page boundaries — a tensor sharing a page with a released neighbour keeps
    // that page. Returns the bytes whose residency was dropped.
    size_t release_dense_pages_except(const std::string& keep_tensor) const {
        if (!dense_mmap_base_ || dense_mmap_base_ == MAP_FAILED) return 0;
        const auto it = dense_tensors_.find(keep_tensor);
        if (it == dense_tensors_.end()) {
            throw std::runtime_error(
                "AeonModelLoader: cannot keep an unknown tensor resident: " + keep_tensor);
        }

        const uint8_t* base = dense_mmap_base_;
        const size_t total = dense_file_size_;
        const long page_size = ::sysconf(_SC_PAGESIZE);
        const size_t alignment = page_size > 0 ? static_cast<size_t>(page_size) : 4096;

        const size_t keep_begin = static_cast<size_t>(it->second.data - base);
        const size_t keep_end = keep_begin + static_cast<size_t>(it->second.byte_size);
        if (keep_end > total) {
            throw std::runtime_error(
                "AeonModelLoader: the kept tensor extends beyond the dense container");
        }
        const size_t keep_first_page = (keep_begin / alignment) * alignment;
        const size_t keep_last_page = ((keep_end + alignment - 1) / alignment) * alignment;

        return drop_residency(base, keep_first_page) +
               drop_residency(base + keep_last_page, total - keep_last_page);
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
        expert_format_version_ = 0;
        expert_format_ = {};
    }

private:
    void open_dense(const std::string& dense_path, uint32_t expected_version) {
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

        const uint32_t dense_version = *reinterpret_cast<const uint32_t*>(dense_mmap_base_ + 12);
        if (dense_version != expected_version) {
            throw std::runtime_error(
                "AeonModelLoader: Unexpected dense format version " +
                std::to_string(dense_version) + ", expected " +
                std::to_string(expected_version));
        }

        const uint32_t tensor_count = *reinterpret_cast<const uint32_t*>(dense_mmap_base_ + 16);
        const uint64_t dir_len = *reinterpret_cast<const uint64_t*>(dense_mmap_base_ + 20);

        if (dir_len > dense_file_size_ - 28) {
            throw std::runtime_error("AeonModelLoader: Directory length exceeds file size!");
        }

        std::string dir_json(reinterpret_cast<const char*>(dense_mmap_base_ + 28), dir_len);

        // Calculate 4KB sector aligned data payload start
        size_t pre_data_len = 28 + dir_len;
        size_t pad_bytes = (AEON_SECTOR_SIZE - (pre_data_len % AEON_SECTOR_SIZE)) % AEON_SECTOR_SIZE;
        const size_t data_start = pre_data_len + pad_bytes;
        if (data_start > dense_file_size_) {
            throw std::runtime_error("AeonModelLoader: Dense payload starts beyond file size");
        }

        const size_t parsed_tensor_count = parse_dense_directory(
            dir_json,
            dense_mmap_base_ + data_start,
            dense_file_size_ - data_start);
        if (parsed_tensor_count != tensor_count) {
            throw std::runtime_error(
                "AeonModelLoader: Dense directory count does not match its header");
        }
        std::cout << "[AeonModelLoader] Loaded dense container: " << dense_tensors_.size()
                  << " tensors (data starts at 0x" << std::hex << data_start << std::dec << ")." << std::endl;
    }

    void open_experts(
        const std::string& experts_path,
        const std::string& index_path,
        uint32_t expected_version,
        ExpertFormatKind expected_kind,
        uint32_t expert_sector_size,
        size_t expected_payload_bytes
    ) {
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

        const uint32_t index_version = *reinterpret_cast<const uint32_t*>(idx_buf.data() + 12);
        if (index_version != expected_version) {
            throw std::runtime_error(
                "AeonModelLoader: Unexpected expert index version " +
                std::to_string(index_version) + ", expected " +
                std::to_string(expected_version));
        }

        num_layers_ = *reinterpret_cast<const uint32_t*>(idx_buf.data() + 16);
        experts_per_layer_ = *reinterpret_cast<const uint32_t*>(idx_buf.data() + 20);
        uint64_t expert_bytes = *reinterpret_cast<const uint64_t*>(idx_buf.data() + 24);

        if (expected_payload_bytes != 0 && expert_bytes != expected_payload_bytes) {
            throw std::runtime_error(
                "AeonModelLoader: Expert payload size does not match the artifact specification");
        }

        if (expert_bytes > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("AeonModelLoader: Expert payload is too large for this platform");
        }
        ExpertFormatDescriptor format{
            expected_kind,
            index_version,
            expert_sector_size,
            num_layers_,
            experts_per_layer_,
            static_cast<size_t>(expert_bytes)
        };
        format.validate_catalog();

        const uint64_t total_experts = format.total_experts();
        if (total_experts > (std::numeric_limits<size_t>::max() - 32) / 16) {
            throw std::runtime_error("AeonModelLoader: Expert catalog is too large");
        }
        const size_t required_index_bytes = 32 + static_cast<size_t>(total_experts) * 16;
        if (idx_buf.size() < required_index_bytes) {
            throw std::runtime_error("AeonModelLoader: Expert index has incomplete entries!");
        }
        expert_format_version_ = index_version;
        expert_format_ = format;
        expert_offsets_.resize(total_experts);

        const uint64_t* entries = reinterpret_cast<const uint64_t*>(idx_buf.data() + 32);
        for (uint64_t i = 0; i < total_experts; ++i) {
            const uint64_t offset = entries[i * 2];
            const uint64_t byte_length = entries[i * 2 + 1];
            if (byte_length != expert_bytes || (offset % format.sector_size) != 0) {
                throw std::runtime_error(
                    "AeonModelLoader: Expert index contains an invalid payload entry");
            }
            expert_offsets_[static_cast<size_t>(i)] = offset;
        }

        // 2. Mmap the swizzled expert payload.
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

    size_t parse_dense_directory(
        const std::string& dir_json,
        const uint8_t* payload_base,
        size_t payload_size
    ) {
        const JsonValue directory = JsonValue::parse(dir_json);
        const auto& entries = directory.as_array();
        for (const auto& entry : entries) {
            const std::string& name = entry.at("name").as_string();
            const int64_t offset_value = entry.at("offset").as_int64();
            const int64_t size_value = entry.at("size").as_int64();
            if (offset_value < 0 || size_value < 0) {
                throw std::runtime_error("AeonModelLoader: Dense directory contains a negative range");
            }
            const auto offset = static_cast<uint64_t>(offset_value);
            const auto size = static_cast<uint64_t>(size_value);
            if (offset > payload_size || size > payload_size - offset ||
                offset > std::numeric_limits<size_t>::max() ||
                size > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("AeonModelLoader: Dense tensor range exceeds payload");
            }

            LoadedTensor tensor;
            tensor.data = payload_base + static_cast<size_t>(offset);
            tensor.byte_size = static_cast<int64_t>(size);
            tensor.dtype = entry.at("dtype").as_string();
            for (const auto& dimension : entry.at("shape").as_array()) {
                const int64_t value = dimension.as_int64();
                if (value < 0) throw std::runtime_error("AeonModelLoader: Dense tensor shape is negative");
                tensor.shape.push_back(value);
            }
            if (!dense_tensors_.emplace(name, std::move(tensor)).second) {
                throw std::runtime_error("AeonModelLoader: Duplicate dense tensor: " + name);
            }
        }
        return entries.size();
    }

    // Advice is a hint: a failure (or a zero-length range) is not an error, so the
    // caller learns the true released size rather than an assumption about it.
    static size_t drop_residency(const uint8_t* address, size_t length) {
        if (length == 0) return 0;
        return ::madvise(const_cast<uint8_t*>(address), length, MADV_DONTNEED) == 0 ? length : 0;
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
    uint32_t expert_format_version_{0};
    ExpertFormatDescriptor expert_format_{};
    std::vector<uint64_t> expert_offsets_;
};

} // namespace aeon::core
