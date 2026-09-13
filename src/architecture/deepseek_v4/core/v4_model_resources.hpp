#pragma once

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "infrastructure/core/aeon_loader.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace aeon::core {

class V4ModelResources {
public:
    kernel::RopeTable rope_table;
    kernel::RopeTable compressed_rope_table;

    const half* host_embed_table{nullptr};
    float* d_hc_head_fn{nullptr};
    float* d_hc_head_base{nullptr};
    float* d_hc_head_scale{nullptr};
    half* d_lm_head{nullptr};
    half* d_final_norm{nullptr};
    float* d_cos_cache{nullptr};
    float* d_sin_cache{nullptr};
    float* d_compressed_cos_cache{nullptr};
    float* d_compressed_sin_cache{nullptr};

    V4ModelResources() = default;

    ~V4ModelResources() {
        free();
    }

    V4ModelResources(const V4ModelResources&) = delete;
    V4ModelResources& operator=(const V4ModelResources&) = delete;

    V4ModelResources(V4ModelResources&& other) noexcept {
        move_from(std::move(other));
    }

    V4ModelResources& operator=(V4ModelResources&& other) noexcept {
        if (this != &other) {
            free();
            move_from(std::move(other));
        }
        return *this;
    }

    void initialize(const AeonModelLoader& loader, uint32_t max_seq_len) {
        initialize(loader, max_seq_len, DeepSeekV4Config{});
    }

    void initialize(
        const AeonModelLoader& loader,
        uint32_t max_seq_len,
        const DeepSeekV4Config& config
    ) {
        free();

        rope_table.init(max_seq_len, config.rope_theta, 1.0f);
        compressed_rope_table.init(
            max_seq_len,
            config.compress_rope_theta,
            config.rope_scaling.factor,
            config.rope_scaling.beta_fast,
            config.rope_scaling.beta_slow,
            static_cast<uint32_t>(config.rope_scaling.original_max_position_embeddings));
        allocate_rope_cache(rope_table, &d_cos_cache, &d_sin_cache, "main");
        allocate_rope_cache(
            compressed_rope_table,
            &d_compressed_cos_cache,
            &d_compressed_sin_cache,
            "compressed");

        host_embed_table = loader.get_data_ptr<half>("embed.weight");

        const auto& head_t = loader.get_tensor("head.weight");
        check_hip(hipMalloc(&d_lm_head, head_t.byte_size), "hipMalloc(LM head)");
        check_hip(
            hipMemcpy(d_lm_head, head_t.data, head_t.byte_size, hipMemcpyHostToDevice),
            "hipMemcpy(LM head)"
        );

        const auto& fn_t = loader.get_tensor("hc_head_fn");
        const auto& base_t = loader.get_tensor("hc_head_base");
        const auto& scale_t = loader.get_tensor("hc_head_scale");
        check_hip(hipMalloc(&d_hc_head_fn, fn_t.byte_size), "hipMalloc(HC head function)");
        check_hip(hipMalloc(&d_hc_head_base, base_t.byte_size), "hipMalloc(HC head base)");
        check_hip(hipMalloc(&d_hc_head_scale, scale_t.byte_size), "hipMalloc(HC head scale)");
        check_hip(
            hipMemcpy(d_hc_head_fn, fn_t.data, fn_t.byte_size, hipMemcpyHostToDevice),
            "hipMemcpy(HC head function)"
        );
        check_hip(
            hipMemcpy(d_hc_head_base, base_t.data, base_t.byte_size, hipMemcpyHostToDevice),
            "hipMemcpy(HC head base)"
        );
        check_hip(
            hipMemcpy(d_hc_head_scale, scale_t.data, scale_t.byte_size, hipMemcpyHostToDevice),
            "hipMemcpy(HC head scale)"
        );

        const auto& norm_t = loader.get_tensor("norm.weight");
        check_hip(hipMalloc(&d_final_norm, norm_t.byte_size), "hipMalloc(final norm)");
        check_hip(
            hipMemcpy(d_final_norm, norm_t.data, norm_t.byte_size, hipMemcpyHostToDevice),
            "hipMemcpy(final norm)"
        );
    }

    void free() noexcept {
        if (d_cos_cache) { (void)hipFree(d_cos_cache); d_cos_cache = nullptr; }
        if (d_sin_cache) { (void)hipFree(d_sin_cache); d_sin_cache = nullptr; }
        if (d_compressed_cos_cache) { (void)hipFree(d_compressed_cos_cache); d_compressed_cos_cache = nullptr; }
        if (d_compressed_sin_cache) { (void)hipFree(d_compressed_sin_cache); d_compressed_sin_cache = nullptr; }
        if (d_lm_head) { (void)hipFree(d_lm_head); d_lm_head = nullptr; }
        if (d_hc_head_fn) { (void)hipFree(d_hc_head_fn); d_hc_head_fn = nullptr; }
        if (d_hc_head_base) { (void)hipFree(d_hc_head_base); d_hc_head_base = nullptr; }
        if (d_hc_head_scale) { (void)hipFree(d_hc_head_scale); d_hc_head_scale = nullptr; }
        if (d_final_norm) { (void)hipFree(d_final_norm); d_final_norm = nullptr; }
        host_embed_table = nullptr;
    }

private:
    static void allocate_rope_cache(
        const kernel::RopeTable& table,
        float** cosine_cache,
        float** sine_cache,
        const char* identity
    ) {
        const size_t bytes = static_cast<size_t>(table.max_seq_len) * table.half_rope * sizeof(float);
        check_hip(
            hipMalloc(cosine_cache, bytes),
            (std::string("hipMalloc(") + identity + " cosine cache)").c_str());
        check_hip(
            hipMalloc(sine_cache, bytes),
            (std::string("hipMalloc(") + identity + " sine cache)").c_str());
        check_hip(
            hipMemcpy(*cosine_cache, table.cos_cache.data(), bytes, hipMemcpyHostToDevice),
            (std::string("hipMemcpy(") + identity + " cosine cache)").c_str());
        check_hip(
            hipMemcpy(*sine_cache, table.sin_cache.data(), bytes, hipMemcpyHostToDevice),
            (std::string("hipMemcpy(") + identity + " sine cache)").c_str());
    }

    static void check_hip(hipError_t error, const char* operation) {
        if (error != hipSuccess) {
            throw std::runtime_error(
                std::string("V4ModelResources: ") + operation + ": " + hipGetErrorString(error));
        }
    }

    void move_from(V4ModelResources&& other) noexcept {
        rope_table = std::move(other.rope_table);
        compressed_rope_table = std::move(other.compressed_rope_table);
        host_embed_table = other.host_embed_table;
        d_hc_head_fn = other.d_hc_head_fn;
        d_hc_head_base = other.d_hc_head_base;
        d_hc_head_scale = other.d_hc_head_scale;
        d_lm_head = other.d_lm_head;
        d_final_norm = other.d_final_norm;
        d_cos_cache = other.d_cos_cache;
        d_sin_cache = other.d_sin_cache;
        d_compressed_cos_cache = other.d_compressed_cos_cache;
        d_compressed_sin_cache = other.d_compressed_sin_cache;

        other.host_embed_table = nullptr;
        other.d_hc_head_fn = nullptr;
        other.d_hc_head_base = nullptr;
        other.d_hc_head_scale = nullptr;
        other.d_lm_head = nullptr;
        other.d_final_norm = nullptr;
        other.d_cos_cache = nullptr;
        other.d_sin_cache = nullptr;
        other.d_compressed_cos_cache = nullptr;
        other.d_compressed_sin_cache = nullptr;
    }
};

} // namespace aeon::core
