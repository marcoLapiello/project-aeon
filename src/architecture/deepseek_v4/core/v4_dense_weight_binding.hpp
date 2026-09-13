#pragma once

#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "infrastructure/core/aeon_loader.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)
#endif

namespace aeon::core {

class V4DenseWeightBinding {
public:
    half* d_attn_norm{nullptr};
    half* d_wq_a{nullptr};
    half* d_q_norm{nullptr};
    half* d_wq_b{nullptr};
    half* d_wkv{nullptr};
    half* d_kv_norm{nullptr};
    float* d_attn_sink{nullptr};
    half* d_wo_a{nullptr};
    half* d_wo_b{nullptr};

    float* d_compressor_ape{nullptr};
    half* d_compressor_norm{nullptr};
    half* d_compressor_wgate{nullptr};
    half* d_compressor_wkv{nullptr};

    float* d_indexer_compressor_ape{nullptr};
    half* d_indexer_compressor_norm{nullptr};
    half* d_indexer_compressor_wgate{nullptr};
    half* d_indexer_compressor_wkv{nullptr};
    half* d_indexer_weights_proj{nullptr};
    half* d_indexer_wq_b{nullptr};

    float* d_hc_attn_fn{nullptr};
    float* d_hc_attn_base{nullptr};
    float* d_hc_attn_scale{nullptr};

    half* d_ffn_norm{nullptr};
    float* d_hc_ffn_fn{nullptr};
    float* d_hc_ffn_base{nullptr};
    float* d_hc_ffn_scale{nullptr};

    half* d_shared_w1{nullptr};
    half* d_shared_w2{nullptr};
    half* d_shared_w3{nullptr};

    int64_t* d_tid2eid{nullptr};
    const int64_t* host_tid2eid{nullptr};
    half* d_gate_weight{nullptr};
    float* d_gate_bias{nullptr};

    V4DenseWeightBinding() = default;

    ~V4DenseWeightBinding() {
        free_dense_weights();
    }

    V4DenseWeightBinding(const V4DenseWeightBinding&) = delete;
    V4DenseWeightBinding& operator=(const V4DenseWeightBinding&) = delete;

    V4DenseWeightBinding(V4DenseWeightBinding&& other) noexcept {
        move_from(std::move(other));
    }

    V4DenseWeightBinding& operator=(V4DenseWeightBinding&& other) noexcept {
        if (this != &other) {
            free_dense_weights();
            move_from(std::move(other));
        }
        return *this;
    }

    template<typename LoaderT>
    void bind_dense_weights(const V4LayerSpec& layer_spec, const LoaderT& loader) {
        free_dense_weights();
        const int layer_id = static_cast<int>(layer_spec.layer_id);
        const bool is_hash_layer = layer_spec.layer_id < 3;
        const std::string prefix = "layers." + std::to_string(layer_id) + ".";

        upload_tensor(loader, prefix + "attn_norm.weight", &d_attn_norm);
        upload_tensor(loader, prefix + "attn.wq_a.weight", &d_wq_a);
        upload_tensor(loader, prefix + "attn.q_norm.weight", &d_q_norm);
        upload_tensor(loader, prefix + "attn.wq_b.weight", &d_wq_b);
        upload_tensor(loader, prefix + "attn.wkv.weight", &d_wkv);
        upload_tensor(loader, prefix + "attn.kv_norm.weight", &d_kv_norm);
        upload_tensor(loader, prefix + "attn.attn_sink", &d_attn_sink);
        upload_tensor(loader, prefix + "attn.wo_a.weight", &d_wo_a);
        upload_tensor(loader, prefix + "attn.wo_b.weight", &d_wo_b);

        if (layer_spec.attention_kind == V4AttentionKind::CSA ||
            layer_spec.attention_kind == V4AttentionKind::HCA) {
            upload_tensor(loader, prefix + "attn.compressor.ape", &d_compressor_ape);
            upload_tensor(loader, prefix + "attn.compressor.norm.weight", &d_compressor_norm);
            upload_tensor(loader, prefix + "attn.compressor.wgate.weight", &d_compressor_wgate);
            upload_tensor(loader, prefix + "attn.compressor.wkv.weight", &d_compressor_wkv);
        }
        if (layer_spec.attention_kind == V4AttentionKind::CSA) {
            upload_tensor(loader, prefix + "attn.indexer.compressor.ape", &d_indexer_compressor_ape);
            upload_tensor(loader, prefix + "attn.indexer.compressor.norm.weight", &d_indexer_compressor_norm);
            upload_tensor(loader, prefix + "attn.indexer.compressor.wgate.weight", &d_indexer_compressor_wgate);
            upload_tensor(loader, prefix + "attn.indexer.compressor.wkv.weight", &d_indexer_compressor_wkv);
            upload_tensor(loader, prefix + "attn.indexer.weights_proj.weight", &d_indexer_weights_proj);
            upload_tensor(loader, prefix + "attn.indexer.wq_b.weight", &d_indexer_wq_b);
        }

        upload_tensor(loader, prefix + "hc_attn_fn", &d_hc_attn_fn);
        upload_tensor(loader, prefix + "hc_attn_base", &d_hc_attn_base);
        upload_tensor(loader, prefix + "hc_attn_scale", &d_hc_attn_scale);

        upload_tensor(loader, prefix + "ffn_norm.weight", &d_ffn_norm);
        upload_tensor(loader, prefix + "hc_ffn_fn", &d_hc_ffn_fn);
        upload_tensor(loader, prefix + "hc_ffn_base", &d_hc_ffn_base);
        upload_tensor(loader, prefix + "hc_ffn_scale", &d_hc_ffn_scale);

        upload_tensor(loader, prefix + "ffn.shared_experts.w1.weight", &d_shared_w1);
        upload_tensor(loader, prefix + "ffn.shared_experts.w2.weight", &d_shared_w2);
        upload_tensor(loader, prefix + "ffn.shared_experts.w3.weight", &d_shared_w3);

        if (is_hash_layer) {
            upload_tensor(loader, prefix + "ffn.gate.tid2eid", &d_tid2eid);
            if (loader.has_tensor(prefix + "ffn.gate.tid2eid")) {
                host_tid2eid = reinterpret_cast<const int64_t*>(
                    loader.get_tensor(prefix + "ffn.gate.tid2eid").data);
            }
        }
        upload_tensor(loader, prefix + "ffn.gate.weight", &d_gate_weight);
        if (!is_hash_layer) {
            upload_tensor(loader, prefix + "ffn.gate.bias", &d_gate_bias);
        }
    }

    void free_dense_weights() {
        if (d_attn_norm) { (void)hipFree(d_attn_norm); d_attn_norm = nullptr; }
        if (d_wq_a) { (void)hipFree(d_wq_a); d_wq_a = nullptr; }
        if (d_q_norm) { (void)hipFree(d_q_norm); d_q_norm = nullptr; }
        if (d_wq_b) { (void)hipFree(d_wq_b); d_wq_b = nullptr; }
        if (d_wkv) { (void)hipFree(d_wkv); d_wkv = nullptr; }
        if (d_kv_norm) { (void)hipFree(d_kv_norm); d_kv_norm = nullptr; }
        if (d_attn_sink) { (void)hipFree(d_attn_sink); d_attn_sink = nullptr; }
        if (d_wo_a) { (void)hipFree(d_wo_a); d_wo_a = nullptr; }
        if (d_wo_b) { (void)hipFree(d_wo_b); d_wo_b = nullptr; }

        if (d_compressor_ape) { (void)hipFree(d_compressor_ape); d_compressor_ape = nullptr; }
        if (d_compressor_norm) { (void)hipFree(d_compressor_norm); d_compressor_norm = nullptr; }
        if (d_compressor_wgate) { (void)hipFree(d_compressor_wgate); d_compressor_wgate = nullptr; }
        if (d_compressor_wkv) { (void)hipFree(d_compressor_wkv); d_compressor_wkv = nullptr; }
        if (d_indexer_compressor_ape) { (void)hipFree(d_indexer_compressor_ape); d_indexer_compressor_ape = nullptr; }
        if (d_indexer_compressor_norm) { (void)hipFree(d_indexer_compressor_norm); d_indexer_compressor_norm = nullptr; }
        if (d_indexer_compressor_wgate) { (void)hipFree(d_indexer_compressor_wgate); d_indexer_compressor_wgate = nullptr; }
        if (d_indexer_compressor_wkv) { (void)hipFree(d_indexer_compressor_wkv); d_indexer_compressor_wkv = nullptr; }
        if (d_indexer_weights_proj) { (void)hipFree(d_indexer_weights_proj); d_indexer_weights_proj = nullptr; }
        if (d_indexer_wq_b) { (void)hipFree(d_indexer_wq_b); d_indexer_wq_b = nullptr; }
        if (d_hc_attn_fn) { (void)hipFree(d_hc_attn_fn); d_hc_attn_fn = nullptr; }
        if (d_hc_attn_base) { (void)hipFree(d_hc_attn_base); d_hc_attn_base = nullptr; }
        if (d_hc_attn_scale) { (void)hipFree(d_hc_attn_scale); d_hc_attn_scale = nullptr; }

        if (d_ffn_norm) { (void)hipFree(d_ffn_norm); d_ffn_norm = nullptr; }
        if (d_hc_ffn_fn) { (void)hipFree(d_hc_ffn_fn); d_hc_ffn_fn = nullptr; }
        if (d_hc_ffn_base) { (void)hipFree(d_hc_ffn_base); d_hc_ffn_base = nullptr; }
        if (d_hc_ffn_scale) { (void)hipFree(d_hc_ffn_scale); d_hc_ffn_scale = nullptr; }

        if (d_shared_w1) { (void)hipFree(d_shared_w1); d_shared_w1 = nullptr; }
        if (d_shared_w2) { (void)hipFree(d_shared_w2); d_shared_w2 = nullptr; }
        if (d_shared_w3) { (void)hipFree(d_shared_w3); d_shared_w3 = nullptr; }

        if (d_tid2eid) { (void)hipFree(d_tid2eid); d_tid2eid = nullptr; }
        if (d_gate_weight) { (void)hipFree(d_gate_weight); d_gate_weight = nullptr; }
        if (d_gate_bias) { (void)hipFree(d_gate_bias); d_gate_bias = nullptr; }
        host_tid2eid = nullptr;
    }

private:
    template<typename LoaderT, typename T>
    void upload_tensor(const LoaderT& loader, const std::string& name, T** device_ptr) {
        if (!loader.has_tensor(name)) {
            throw std::runtime_error("V4DenseWeightBinding: missing required tensor " + name);
        }
        const auto& tensor = loader.get_tensor(name);
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(device_ptr), tensor.byte_size));
        CHECK_HIP(hipMemcpy(*device_ptr, tensor.data, tensor.byte_size, hipMemcpyHostToDevice));
    }

    void move_from(V4DenseWeightBinding&& other) noexcept {
        d_attn_norm = other.d_attn_norm; other.d_attn_norm = nullptr;
        d_wq_a = other.d_wq_a; other.d_wq_a = nullptr;
        d_q_norm = other.d_q_norm; other.d_q_norm = nullptr;
        d_wq_b = other.d_wq_b; other.d_wq_b = nullptr;
        d_wkv = other.d_wkv; other.d_wkv = nullptr;
        d_kv_norm = other.d_kv_norm; other.d_kv_norm = nullptr;
        d_attn_sink = other.d_attn_sink; other.d_attn_sink = nullptr;
        d_wo_a = other.d_wo_a; other.d_wo_a = nullptr;
        d_wo_b = other.d_wo_b; other.d_wo_b = nullptr;

        d_compressor_ape = other.d_compressor_ape; other.d_compressor_ape = nullptr;
        d_compressor_norm = other.d_compressor_norm; other.d_compressor_norm = nullptr;
        d_compressor_wgate = other.d_compressor_wgate; other.d_compressor_wgate = nullptr;
        d_compressor_wkv = other.d_compressor_wkv; other.d_compressor_wkv = nullptr;
        d_indexer_compressor_ape = other.d_indexer_compressor_ape; other.d_indexer_compressor_ape = nullptr;
        d_indexer_compressor_norm = other.d_indexer_compressor_norm; other.d_indexer_compressor_norm = nullptr;
        d_indexer_compressor_wgate = other.d_indexer_compressor_wgate; other.d_indexer_compressor_wgate = nullptr;
        d_indexer_compressor_wkv = other.d_indexer_compressor_wkv; other.d_indexer_compressor_wkv = nullptr;
        d_indexer_weights_proj = other.d_indexer_weights_proj; other.d_indexer_weights_proj = nullptr;
        d_indexer_wq_b = other.d_indexer_wq_b; other.d_indexer_wq_b = nullptr;

        d_hc_attn_fn = other.d_hc_attn_fn; other.d_hc_attn_fn = nullptr;
        d_hc_attn_base = other.d_hc_attn_base; other.d_hc_attn_base = nullptr;
        d_hc_attn_scale = other.d_hc_attn_scale; other.d_hc_attn_scale = nullptr;

        d_ffn_norm = other.d_ffn_norm; other.d_ffn_norm = nullptr;
        d_hc_ffn_fn = other.d_hc_ffn_fn; other.d_hc_ffn_fn = nullptr;
        d_hc_ffn_base = other.d_hc_ffn_base; other.d_hc_ffn_base = nullptr;
        d_hc_ffn_scale = other.d_hc_ffn_scale; other.d_hc_ffn_scale = nullptr;

        d_shared_w1 = other.d_shared_w1; other.d_shared_w1 = nullptr;
        d_shared_w2 = other.d_shared_w2; other.d_shared_w2 = nullptr;
        d_shared_w3 = other.d_shared_w3; other.d_shared_w3 = nullptr;

        d_tid2eid = other.d_tid2eid; other.d_tid2eid = nullptr;
        host_tid2eid = other.host_tid2eid; other.host_tid2eid = nullptr;
        d_gate_weight = other.d_gate_weight; other.d_gate_weight = nullptr;
        d_gate_bias = other.d_gate_bias; other.d_gate_bias = nullptr;
    }
};

} // namespace aeon::core