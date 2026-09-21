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
// It builds all fifteen steps of the plan's assembly (composition plan §5.1):
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
//   10. allocate the Hot VRAM expert pool and the staging arena
//   11. initialise the expert registry (Hot residents, Warm capacity)
//   12. preload the Hot slots, then the Warm pool, with batched `O_DIRECT` reads
//   13. configure the tiered supply on the four streams
//   14. construct the routed-expert scratch and the production executor
//   15. (nothing left — the graph above this file drives it)
//
// The expert half is a **lift**, not a second design: `V4Pipeline::initialize`
// performs the same sequence (composition plan §6.1, G1). The difference is that
// the historical version built and ran its own graph inline, while this hands the
// same objects to whichever graph the caller drives.
//
// The **streaming system enters the graph only here**. `V4Graph` never learns
// which tier answered a request: it calls the `V4RoutedExpertExecutor` the body
// declares, and that executor is this object's. That is the whole reason the
// storage half can be correct while the graph is numerics-only.
//
// What is *not* re-implemented here. Every object below already exists and is
// used as it stands: `AeonModelLoader`, `DeepSeekV4Config`, `V4ModelSpec`,
// `V4ModelContract`, `MemoryBudgetEngine`, `V4ModelResources`, `V4Layer`,
// `PipelineScratchBuffers`, `V4DeviceStreams`, `V4LayerBodyTables`. This is that
// assembly lifted into the rewrite rather than a second design of it.
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
#include "architecture/deepseek_v4/core/v4_expert_executor.hpp"
#include "architecture/deepseek_v4/core/v4_expert_supply.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/core/v4_model_contract.hpp"
#include "architecture/deepseek_v4/core/v4_model_resources.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "architecture/deepseek_v4/core/v4_pipeline_scratch.hpp"
#include "backend/swizzled_w4a16/core/vram_expert_pool.hpp"
#include "infrastructure/backend_registry/expert_backend.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "infrastructure/core/expert_registry.hpp"
#include "infrastructure/core/host_expert_pool.hpp"
#include "infrastructure/core/prefetch_staging.hpp"
#include "infrastructure/core/supply_telemetry.hpp"
#include "infrastructure/hip_check.hpp"
#include "infrastructure/io/aligned_allocator.hpp"
#include "infrastructure/io/direct_io_reader.hpp"
#include "platform/rdna3/device.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

        // The budget is sized against the bytes the graph actually **uploads**, not
        // the container's file size: `embed.weight` stays host-side and the unused
        // `mtp.*` draft head is never read, and together those are 1.957 GiB the
        // old file-size reservation over-counted. Reserving them cost 148 Hot
        // expert slots (675 instead of 823 at context 256). The contract is the
        // authority on the uploaded set — the same table that validates the
        // artifact — so the budget cannot drift from what is placed on the device.
        budget_ = MemoryBudgetEngine::evaluate(
            runtime_cfg, config_, V4ModelContract::uploaded_dense_bytes(config_), expert_format);
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

        // Every dense tensor the contract enumerates has been uploaded by this
        // point — `resources_` binds the model-level ones and the loop above binds
        // the per-layer ones — and the only dense read left on the request path is
        // `embed.weight` in `embed_token`. The pages the uploads faulted in are
        // therefore dead weight that competes for host RAM with the **pinned** Warm
        // pool, which the kernel cannot reclaim or swap.
        //
        // Released here, before the Warm preload rather than after it, because the
        // preload is where host pressure peaks: it fills up to `warm_host_bytes` of
        // unevictable memory while the released pages would still be resident. The
        // step touches only the experts container, so nothing below re-reads dense.
        if (runtime_cfg.release_dense_pages_after_upload) {
            last_released_dense_bytes_ = loader_.release_dense_pages_except("embed.weight");
            // Reported here rather than with the residency summary below, so the
            // figure appears before the preload instead of after it.
            if (verbose_ && last_released_dense_bytes_ > 0) {
                std::printf(
                    "[Host] Released %.2f GiB of dense-container page cache after upload "
                    "(only embed.weight stays resident)\n",
                    static_cast<double>(last_released_dense_bytes_) / (1024.0 * 1024.0 * 1024.0));
            }
        }

        // The telemetry sink is opened before the supply is built, so a request
        // recorded during initialization cannot be dropped. (`enable_jsonl` opens
        // in truncate mode and clears the summaries, so "open early" means "start
        // from a clean slate and capture everything after it" rather than "start
        // clean and hope nothing has happened yet".)
        if (!runtime_cfg.supply_telemetry_path.empty()) {
            telemetry_.enable_jsonl(runtime_cfg.supply_telemetry_path, runtime_cfg.run_id);
            if (verbose_) {
                std::printf("[Host] Supply telemetry -> %s (run_id=%s)\n",
                            runtime_cfg.supply_telemetry_path.c_str(),
                            runtime_cfg.run_id.c_str());
            }
        }

        initialize_experts(runtime_cfg);

        if (verbose_) {
            std::printf(
                "[Host] %d layers (%u Sliding, %u CSA, %u HCA), context %u tokens, "
                "%u hot + %u warm expert slots%s\n",
                num_layers, count_kind(V4AttentionKind::Sliding),
                count_kind(V4AttentionKind::CSA), count_kind(V4AttentionKind::HCA),
                context_capacity_, budget_.hot_vram_slots, budget_.warm_host_slots,
                executor_ ? "" : " (expert tier not built: no Hot VRAM slot)");
        }
    }

    void free() noexcept {
        // Flush the telemetry sink first, while the pools it observes still exist,
        // so the final summary captures the peak occupancy of the run being torn
        // down rather than a post-free zero.
        telemetry_.disable();

        // Teardown is the construction order reversed, and the two references that
        // matter are dropped first: the executor borrows the supply, the pool, the
        // staging arena and the registry, and the supply borrows the reader and the
        // pools. Freeing in any other order would leave a live object pointing at
        // freed storage.
        executor_.reset();
        supply_.clear();
        expert_scratch_.free();
        vram_pool_.free();
        host_pool_.free();
        staging_.reset();
        io_reader_.reset();
        completions_.clear();
        next_io_id_ = 1;

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

    // The routed-expert supply the layer body drives. It is the *interface* type
    // on purpose: the graph must not be able to tell which tier answered, and it
    // must not acquire a dependency on the storage system to run a layer.
    V4RoutedExpertExecutor& executor() {
        if (!executor_) {
            throw std::logic_error(
                "V4ModelHost: the expert tier was not built — the memory budget left no "
                "Hot VRAM slot, so the graph cannot run");
        }
        return *executor_;
    }

    bool experts_ready() const noexcept { return executor_ != nullptr; }

    // The token boundary. The graph calls this once per token, after the head and
    // before the caller reads the logits back — a lease grants no ordering, so it
    // must be held for as long as compute reading that slot may be in flight
    // (trap 41), and the logits readback is the compute-stream boundary that makes
    // the release safe. Exposed here rather than on the seam because releasing is a
    // property of the concrete tiered executor, not of the interface the body sees.
    void release_expert_leases() {
        if (executor_) executor_->release_leases();
    }

    // Diagnostics, for an assembly gate: the registry's residency claims and the
    // pool it made them against. Not used by the graph.
    const ExpertRegistry& registry() const noexcept { return registry_; }
    UnifiedVRAMExpertPool& vram_pool() noexcept { return vram_pool_; }
    const SupplyTelemetry& telemetry() const noexcept { return telemetry_; }

    // Tell the telemetry which phase the next dispatch belongs to. The caller is
    // the generation loop, which already knows whether the token it is about to
    // advance is a prompt token (prefill) or a generated one (decode); this only
    // forwards that fact, it does not derive it. A no-op when the sink is off.
    void set_supply_phase(bool prefill) {
        telemetry_.set_phase(prefill ? RoutingPhase::Prefill : RoutingPhase::Decode);
    }

    // One generated token's worth of decode accounting. A no-op when the sink is off.
    void record_supply_decode_token() {
        telemetry_.record_decode_token();
    }

    // Dense-container page-cache residency dropped after the uploads, in bytes (0
    // when the runtime was configured not to release). Reported rather than
    // inferred: `madvise` is a hint, so the only figure worth printing is this one.
    size_t released_dense_bytes() const noexcept { return last_released_dense_bytes_; }

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

    // Steps 10–15 of §5.1 — the expert tier, in the order the loop needs it.
    //
    // Skipped, not faked, when the budget left no Hot VRAM slot. The dense graph
    // is still constructible (the head stage reads no expert), but the graph is
    // not runnable, and `executor()` refuses rather than hand back an executor
    // whose pool is empty and whose first dispatch would throw.
    void initialize_experts(const AeonRuntimeConfig& runtime_cfg) {
        if (budget_.hot_vram_slots == 0) return;

        const auto& format = loader_.expert_format();
        const uint32_t warm_slots = budget_.warm_host_slots;

        // 10 — the Hot VRAM pool, the staging arena, and the direct reader. The
        // arena and the reader are built for the *artifact's* format, not the
        // backend's default, so a payload's staging slot is the artifact's own
        // `payload_bytes` wide.
        vram_pool_.allocate(budget_.hot_vram_slots, format);
        staging_ = std::make_unique<PrefetchStagingArena>(format);
        io_reader_ = std::make_unique<aeon::io::DirectIOReader>(
            64, true, format.sector_size);

        // 11 — the registry. It is what decides residency for every request, and
        // it saturates VRAM at construction: every Hot slot is owned from the
        // first token on, so the production steady state (a cold miss must evict a
        // resident) is the only state that exists.
        registry_.init(static_cast<uint32_t>(config_.num_hidden_layers),
                       static_cast<uint32_t>(config_.n_routed_experts),
                       budget_.hot_vram_slots, warm_slots,
                       runtime_cfg.preload_warm_host);

        // 12 — Hot, then Warm, filled by batched `O_DIRECT` reads. This is the
        // only step with real mass; the byte-exactness of every route it uses is
        // the item-21 gate's subject, so nothing here re-checks it.
        preload_hot_experts(format);
        if (warm_slots > 0) {
            host_pool_.allocate(warm_slots, format);
            if (runtime_cfg.preload_warm_host) {
                preload_warm_experts(format);
            }
        }

        // 13 — the tiered supply, on the four shared streams.
        supply_.configure(
            &loader_,
            &vram_pool_,
            warm_slots > 0 ? &host_pool_ : nullptr,
            &registry_,
            staging_.get(),
            &telemetry_,
            io_reader_.get(),
            &completions_,
            &next_io_id_,
            streams_.compute, streams_.sdma, streams_.sdma_cold, streams_.demotion,
            format.payload_bytes,
            runtime_cfg.enable_warm_refill
                ? V4ExpertSupplyCoordinator::DEFAULT_DEMOTION_QUEUE_CAPACITY : 0);

        // 14 — the routed-expert scratch and the production executor. The executor
        // borrows the four streams the host owns, so its capacity fallback drains
        // exactly the set that carries expert traffic.
        expert_scratch_.allocate();
        executor_ = std::make_unique<V4TieredExpertExecutor>(
            supply_, vram_pool_, *staging_, registry_, expert_scratch_, streams_,
            config_.swiglu_limit);
    }

    size_t direct_requests_per_expert(const ExpertFormatDescriptor& format) const noexcept {
        return (format.payload_bytes + aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES - 1) /
               aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
    }

    // Every Hot resident, read straight from the artifact and uploaded once. The
    // physical slots come from the registry's own `vram_slots` rather than being
    // re-derived from the round-robin order.
    void preload_hot_experts(const ExpertFormatDescriptor& format) {
        const size_t batch = std::max<size_t>(
            1, io_reader_->submission_capacity() / direct_requests_per_expert(format));
        for (uint32_t start = 0; start < budget_.hot_vram_slots;
             start += static_cast<uint32_t>(batch)) {
            const uint32_t end = std::min<uint32_t>(
                budget_.hot_vram_slots, start + static_cast<uint32_t>(batch));

            std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
            std::vector<aeon::io::AlignedBuffer> buffers;
            for (uint32_t slot = start; slot < end; ++slot) {
                const int32_t gid = registry_.vram_slots[slot];
                if (gid < 0) continue;
                const auto& entry = registry_.catalog[static_cast<size_t>(gid)];
                expert_ids.emplace_back(entry.layer_id, entry.expert_id);
                buffers.emplace_back(format.payload_bytes, format.sector_size);
            }

            std::vector<uint8_t*> destinations;
            destinations.reserve(buffers.size());
            for (auto& buffer : buffers) {
                destinations.push_back(static_cast<uint8_t*>(buffer.data()));
            }
            read_experts_direct_blocking(expert_ids, destinations);

            size_t index = 0;
            for (uint32_t slot = start; slot < end; ++slot) {
                if (registry_.vram_slots[slot] < 0) continue;
                vram_pool_.upload_from_host_expert(
                    slot, destinations[index++], streams_.compute);
            }
            CHECK_HIP(hipStreamSynchronize(streams_.compute));
        }
    }

    // The Warm pool is filled in place — the registry chose the owning experts at
    // construction, so this writes their payloads into the pinned slots it
    // reserved rather than deciding residency again.
    void preload_warm_experts(const ExpertFormatDescriptor& format) {
        const size_t batch = std::max<size_t>(
            1, io_reader_->submission_capacity() / direct_requests_per_expert(format));
        for (uint32_t start = 0; start < budget_.warm_host_slots;
             start += static_cast<uint32_t>(batch)) {
            const uint32_t end = std::min<uint32_t>(
                budget_.warm_host_slots, start + static_cast<uint32_t>(batch));

            std::vector<std::pair<uint32_t, uint32_t>> expert_ids;
            std::vector<uint8_t*> destinations;
            for (uint32_t slot = start; slot < end; ++slot) {
                const int32_t gid = registry_.host_slots[slot];
                if (gid < 0) continue;
                const auto& entry = registry_.catalog[static_cast<size_t>(gid)];
                expert_ids.emplace_back(entry.layer_id, entry.expert_id);
                destinations.push_back(host_pool_.get_expert_slot_ptr(slot));
            }
            read_experts_direct_blocking(expert_ids, destinations);
        }
    }

    // A batched, blocking `O_DIRECT` read of whole expert payloads. The batch is
    // sized to the reader's own submission capacity, so the chunk count per
    // request is the artifact's (a payload is 3.375 of the 4 MiB chunks) and the
    // short final request is expected rather than an error.
    void read_experts_direct_blocking(
        const std::vector<std::pair<uint32_t, uint32_t>>& expert_ids,
        const std::vector<uint8_t*>& destinations) {
        if (expert_ids.size() != destinations.size()) {
            throw std::invalid_argument("V4ModelHost: a direct expert read batch is mismatched");
        }
        if (expert_ids.empty()) return;

        const size_t requests_per_expert = direct_requests_per_expert(loader_.expert_format());
        const size_t max_batch = std::max<size_t>(
            1, io_reader_->submission_capacity() / requests_per_expert);

        struct ReadJob {
            uint64_t first_user_data{0};
            size_t request_count{0};
        };

        for (size_t start = 0; start < expert_ids.size(); start += max_batch) {
            const size_t end = std::min(expert_ids.size(), start + max_batch);
            std::vector<ReadJob> jobs;
            size_t total_requests = 0;

            for (size_t i = start; i < end; ++i) {
                const auto location = loader_.get_expert_location(
                    expert_ids[i].first, expert_ids[i].second);
                const uint64_t first_user_data = next_io_id_;
                const size_t request_count = io_reader_->submit_read_chunks(
                    loader_.expert_direct_fd(), destinations[i], location.byte_length,
                    location.file_offset, first_user_data);
                next_io_id_ += request_count;
                total_requests += request_count;
                jobs.push_back(ReadJob{first_user_data, request_count});
            }

            if (io_reader_->submit_pending_reads() != total_requests) {
                throw std::runtime_error(
                    "V4ModelHost: a direct expert batch submitted an unexpected request count");
            }

            std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> completions;
            completions.reserve(total_requests);
            for (size_t i = 0; i < total_requests; ++i) {
                const auto completion = io_reader_->wait_for_completion();
                completions.emplace(completion.user_data, completion);
            }

            for (const auto& job : jobs) {
                for (size_t chunk = 0; chunk < job.request_count; ++chunk) {
                    const auto it = completions.find(job.first_user_data + chunk);
                    if (it == completions.end() || it->second.result < 0) {
                        throw std::runtime_error(
                            "V4ModelHost: a direct expert read failed");
                    }
                    const size_t chunk_offset =
                        chunk * aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;
                    const size_t expected = std::min(
                        aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES,
                        loader_.expert_format().payload_bytes - chunk_offset);
                    if (it->second.result != static_cast<int32_t>(expected)) {
                        throw std::runtime_error(
                            "V4ModelHost: a direct expert read returned a short payload");
                    }
                }
            }
        }
    }

    bool verbose_{false};
    AeonModelLoader loader_;
    DeepSeekV4Config config_;
    std::vector<V4LayerSpec> layer_specs_;
    MemoryBudgetReport budget_;
    uint32_t context_capacity_{0};
    size_t last_released_dense_bytes_{0};

    V4DeviceStreams streams_;
    V4ModelResources resources_;
    PipelineScratchBuffers scratch_;
    std::vector<V4Layer> layers_;

    UnifiedVRAMExpertPool vram_pool_;
    HostExpertPool host_pool_;
    ExpertRegistry registry_;
    std::unique_ptr<PrefetchStagingArena> staging_;
    SupplyTelemetry telemetry_;
    std::unique_ptr<aeon::io::DirectIOReader> io_reader_;
    std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> completions_;
    uint64_t next_io_id_{1};
    V4ExpertSupplyCoordinator supply_;
    V4RoutedExpertScratch expert_scratch_;
    std::unique_ptr<V4TieredExpertExecutor> executor_;
};

} // namespace aeon::core
