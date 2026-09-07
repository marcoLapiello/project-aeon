#pragma once

#include "core/safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace aeon::core {

struct LoadedTensor {
    const uint8_t* data{nullptr};
    int64_t byte_size{0};
    std::string dtype;
    std::vector<int64_t> shape;
};

class SafetensorsLoader {
public:
    struct MmapShard {
        std::string filepath;
        int fd{-1};
        size_t file_size{0};
        const uint8_t* mmap_base{nullptr};
        SafetensorsHeader header;
    };

    SafetensorsLoader() = default;

    ~SafetensorsLoader() {
        close_all();
    }

    // Disable copy, allow move
    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;
    SafetensorsLoader(SafetensorsLoader&&) = default;
    SafetensorsLoader& operator=(SafetensorsLoader&&) = default;

    void open_shard(const std::string& filepath) {
        int fd = ::open(filepath.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("SafetensorsLoader: Failed to open file: " + filepath + " (" + strerror(errno) + ")");
        }

        struct stat st;
        if (fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("SafetensorsLoader: Failed to stat file: " + filepath);
        }

        size_t file_size = st.st_size;
        void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("SafetensorsLoader: Failed to mmap file: " + filepath + " (" + strerror(errno) + ")");
        }

        // Parse header
        SafetensorsHeader hdr = SafetensorsHeader::parse_file(filepath);

        auto shard = std::make_unique<MmapShard>();
        shard->filepath = filepath;
        shard->fd = fd;
        shard->file_size = file_size;
        shard->mmap_base = static_cast<const uint8_t*>(mapped);
        shard->header = std::move(hdr);

        // Index all tensors from this shard
        for (const auto& [name, item] : shard->header.tensors) {
            LoadedTensor lt;
            lt.data = shard->mmap_base + shard->header.data_base_offset + item.offset_begin;
            lt.byte_size = item.byte_size();
            lt.dtype = item.dtype;
            lt.shape = item.shape;

            tensor_index_[name] = lt;
        }

        shards_.push_back(std::move(shard));
    }

    bool has_tensor(const std::string& name) const {
        return tensor_index_.find(name) != tensor_index_.end();
    }

    const LoadedTensor& get_tensor(const std::string& name) const {
        auto it = tensor_index_.find(name);
        if (it == tensor_index_.end()) {
            throw std::runtime_error("SafetensorsLoader: Tensor not found: " + name);
        }
        return it->second;
    }

    template<typename T>
    const T* get_data_ptr(const std::string& name) const {
        return reinterpret_cast<const T*>(get_tensor(name).data);
    }

    size_t total_tensors() const {
        return tensor_index_.size();
    }

    void close_all() {
        for (auto& s : shards_) {
            if (s->mmap_base && s->mmap_base != MAP_FAILED) {
                ::munmap(const_cast<uint8_t*>(s->mmap_base), s->file_size);
                s->mmap_base = nullptr;
            }
            if (s->fd >= 0) {
                ::close(s->fd);
                s->fd = -1;
            }
        }
        shards_.clear();
        tensor_index_.clear();
    }

private:
    std::vector<std::unique_ptr<MmapShard>> shards_;
    std::unordered_map<std::string, LoadedTensor> tensor_index_;
};

} // namespace aeon::core
