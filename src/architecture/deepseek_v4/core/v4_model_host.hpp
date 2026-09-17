#pragma once

// -----------------------------------------------------------------------------
// The model host — what is resident, in the order the assembly has to happen.
//
// This is the component the composition plan calls **G1**, and it is the largest
// thing the graph rewrite was missing. It is deliberately *not* the forward pass:
// the graph (`core/v4_graph.hpp`) owns the order of operations, and this owns the
// objects they read and write. The split is what keeps the graph small enough to
// read — nothing in this file knows what a token is, and nothing in the graph
// knows how a tensor got into VRAM.
//
// It builds steps 1–9 of the plan's assembly (composition plan §5.1):
//
//    1. select the compute device and create the four shared streams
//    2. open the `.aeon` container and resolve its weight backend
//    3. load the architecture contract (`config.json`)
//    4. resolve the 43 layer specs from the compression schedule
//    5. validate every required dense tensor against the contract
//    6. evaluate the memory budget and refuse an infeasible configuration
//    7. build the RoPE tables and upload the model-level tensors
//    8. allocate the activation scratch
//    9. construct the 43 layers (dense weights + attention state in VRAM)
//
// Steps 10–15 — the expert pools, the registry, the staging arena, the supply and
// the routed-expert executor — are **not here yet**, because nothing in P1 reads
// an expert. The head stage touches `embed.weight`, `hc_head_fn` / `base` /
// `scale`, `norm.weight` and `head.weight`, and nothing else. Those objects land
// with the 43-layer driver (P2), which is the first thing that needs them, and
// they will hang off this same object so that the graph still never learns which
// tier answered a request.
//
// What is *not* re-implemented here. Every object below already exists and is
// used as it stands: `AeonModelLoader`, `DeepSeekV4Config`, `V4ModelSpec`,
// `V4ModelContract`, `MemoryBudgetEngine`, `V4ModelResources`, `V4Layer`,
// `PipelineScratchBuffers`, `V4DeviceStreams`, `V4LayerBodyTables`. The pre-rewrite
// `V4Pipeline::initialize` performs the same sequence, and this is that sequence
// lifted into the rewrite rather than a second design of it — with one difference
// that matters: the pre-rewrite version builds and runs the old graph inline,
// while this exposes the same objects to be driven by whichever graph the caller
// has. That is what makes it shareable with `AEON_ENABLE_LEGACY_V4_GRAPH` instead
// of a fork of it.
//
// Verification note (plan Part V, second rule). A host is not an op, so there is
// no arithmetic oracle for it. Its gate is that the assembly's own invariants
// hold and that something above it produces oracle-checked numbers: the layers
// and the head stage are read back and compared, and the budget's own accounting
// is cross-checked against what was actually allocated. "It constructed without
// throwing" is not the evidence.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/memory_budget.hpp"
#include "architecture/deepseek_v4/core/v4_device_streams.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/core/v4_model_contract.hpp"
#include "architecture/deepseek_v4/core/v4_model_resources.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline_scratch.hpp"
#include "infrastructure/backend_registry/expert_backend.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "infrastructure/hip_check.hpp"
#include "platform/rdna3/device.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aeon::core {

class V4ModelHost {
public:
    V4ModelHost() = default;

    ~V4ModelHost() {
        free();
    }

    V4ModelHost(const V4ModelHost&) = delete;
    V4ModelHost& operator=(const V4ModelHost&) = delete;

    // The whole assembly, in order, refusing at the first step that cannot hold.
    //
    // The refusal at step 6 is load-bearing rather than informative: a
    // configuration that does not fit is rejected before anything is uploaded,
    // because the alternative — uploading a backbone that does not fit and
    // discovering it as an allocation failure three layers in — is the failure
    // mode the budget engine exists to prevent.
    void initialize(const std::string& model_dir,
                    const AeonRuntimeConfig& runtime_cfg,
                    bool verbose = false) {
        free();
        verbose_ = verbose;

        select_compute_device(verbose_);
        streams_ = V4DeviceStreams::create();

        loader_.open_model(model_dir);
        const auto& expert_format = loader_.expert_format();
        const auto& backend = ExpertBackendRegistry::resolve(expert_format);
        if (!backend.supports_v4_pipeline) {
            throw std::runtime_error(
                "V4ModelHost: the selected artifact requires a different weight backend");
        }

        config_ = DeepSeekV4Config::load_from_json(model_dir + "/config.json");
        layer_specs_ = V4ModelSpec::resolve_layers(config_);
        V4ModelContract::validate(config_, loader_);

        budget_ = MemoryBudgetEngine::evaluate(
            runtime_cfg, config_, loader_.dense_file_size(), expert_format);
        if (!budget_.is_feasible) {
            throw std::runtime_error(
                "V4ModelHost: the memory budget rejected this configuration: " +
                budget_.rejection_reason);
        }

        context_capacity_ = runtime_cfg.context_size;
        if (context_capacity_ == 0) {
            throw std::invalid_argument("V4ModelHost: the context cannot be zero tokens");
        }

        // The RoPE tables and the model-level tensors are built for the declared
        // context, so `context_capacity_` is also every layer's `max_seq_len`.
        // Growing the context later means rebuilding both, which is why the layer
        // state's own refusal (trap 40) is the thing a caller meets rather than a
        // silently clamped position.
        resources_.initialize(loader_, context_capacity_, config_);
        scratch_.allocate();

        const int32_t num_layers = config_.num_hidden_layers;
        layers_.resize(static_cast<size_t>(num_layers));
        for (int32_t layer = 0; layer < num_layers; ++layer) {
            layers_[static_cast<size_t>(layer)].init_with_loader(
                layer_specs_[static_cast<size_t>(layer)], loader_, context_capacity_);
        }

        if (verbose_) {
            std::printf(
                "[Host] %d layers (%u Sliding, %u CSA, %u HCA), context %u tokens, "
                "%u hot + %u warm expert slots\n",
                num_layers, count_kind(V4AttentionKind::Sliding),
                count_kind(V4AttentionKind::CSA), count_kind(V4AttentionKind::HCA),
                context_capacity_, budget_.hot_vram_slots, budget_.warm_host_slots);
        }
    }

    void free() noexcept {
        // The layers free their own dense weights and attention state, the
        // resources and the scratch free theirs, and every `free()` nulls what it
        // released — so this is safe to call twice, once explicitly and once from
        // the destructor.
        layers_.clear();
        resources_.free();
        scratch_.free();
        streams_.destroy();
        loader_.close_all();
        layer_specs_.clear();
        budget_ = MemoryBudgetReport{};
        context_capacity_ = 0;
    }

    // --- the parts the graph drives -----------------------------------------

    uint32_t num_layers() const noexcept {
        return static_cast<uint32_t>(layers_.size());
    }

    uint32_t context_capacity() const noexcept { return context_capacity_; }

    const DeepSeekV4Config& config() const noexcept { return config_; }

    V4ModelResources& resources() noexcept { return resources_; }
    const V4ModelResources& resources() const noexcept { return resources_; }

    PipelineScratchBuffers& scratch() noexcept { return scratch_; }
    const PipelineScratchBuffers& scratch() const noexcept { return scratch_; }

    const V4DeviceStreams& streams() const noexcept { return streams_; }

    V4Layer& layer(uint32_t layer_id) { return layers_.at(layer_id); }
    const V4Layer& layer(uint32_t layer_id) const { return layers_.at(layer_id); }

    const std::vector<V4LayerSpec>& layer_specs() const noexcept { return layer_specs_; }

    const MemoryBudgetReport& budget() const noexcept { return budget_; }

    // Host read access, for building an oracle: `embed.weight` is reached through
    // `resources().host_embed_table` (a pointer into the container), while
    // `head.weight`, `norm.weight` and the three `hc_head` tensors are device-only
    // and have to come from the container's own host mapping.
    const AeonModelLoader& loader() const noexcept { return loader_; }

    // The two RoPE bases, in the shape the layer body consumes. Sliding layers use
    // the plain base (theta 10000) and compressed layers the YaRN-on-compressed one
    // (theta 160000, factor 16) — plan 2.3, trap 7. Handed over as a plain struct
    // so the body keeps no dependency on the resources object.
    V4LayerBodyTables tables() const noexcept {
        V4LayerBodyTables result;
        result.sliding_cos = resources_.d_cos_cache;
        result.sliding_sin = resources_.d_sin_cache;
        result.compressed_cos = resources_.d_compressed_cos_cache;
        result.compressed_sin = resources_.d_compressed_sin_cache;
        return result;
    }

    // The start of a new sequence. Every layer's ring sentinels, counters and
    // committed-entry positions go back to what a freshly allocated layer holds,
    // so a second conversation cannot read the first one's context.
    void reset_generation_state() {
        for (auto& layer : layers_) {
            layer.reset_generation_state();
        }
    }

private:
    uint32_t count_kind(V4AttentionKind kind) const noexcept {
        uint32_t count = 0;
        for (const auto& spec : layer_specs_) {
            if (spec.attention_kind == kind) ++count;
        }
        return count;
    }

    bool verbose_{false};
    AeonModelLoader loader_;
    DeepSeekV4Config config_;
    std::vector<V4LayerSpec> layer_specs_;
    MemoryBudgetReport budget_;
    uint32_t context_capacity_{0};

    V4DeviceStreams streams_;
    V4ModelResources resources_;
    PipelineScratchBuffers scratch_;
    std::vector<V4Layer> layers_;
};

} // namespace aeon::core
