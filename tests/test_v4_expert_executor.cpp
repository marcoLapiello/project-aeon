// -----------------------------------------------------------------------------
// Item 23, first gate — the production routed-expert executor.
//
// `core/v4_expert_executor.hpp` is the implementation of the layer body's
// `V4RoutedExpertExecutor` seam that the *runtime* uses: the tiered supply system
// (index lookup, Hot/Warm/Cold promotion, prefetch, leases, staging) followed by the
// fused expert kernels. Until now that seam existed only inside test fixtures and
// inlined in the pre-rewrite `V4Pipeline`, so the production path had never run.
//
// -----------------------------------------------------------------------------
// What this gate certifies, and what it deliberately does not
//
// It certifies **delivery**: that the supply path hands the expert kernels the same
// bytes the artifact holds, for the expert the router actually selected, with the
// staging and lease protocol ordered correctly.
//
// It does **not** certify the expert arithmetic. That is Tier 1 item 14 (format,
// clamp, composed FFN against an independent decoder), item 15 (shared expert),
// item 16 (`moe_out` against an fp64 oracle on real `layers.0` experts) and the
// item-19b accumulation pair (`tests/test_v4_moe_accum_oracle.cpp`). Those own the
// dequantization, the clamped SwiGLU, the W2 reduction and the fp32 fixed-order
// combine. Re-certifying them here would be duplication; writing a *new* reference
// for them would be worse, because it would be a second oracle for an
// already-certified quantity.
//
// The instrument therefore holds the kernels fixed and removes only storage:
//
//   * **production** — the layer body with `V4TieredExpertExecutor`: six experts
//     resolved through `ExpertRegistry`, fetched by `TieredExpertSupply` (io_uring
//     `O_DIRECT` for the cold misses), uploaded into `UnifiedVRAMExpertPool` slots
//     on a side stream, and read by pointer from those slots behind a
//     `hipStreamWaitEvent` on the per-transfer staging event;
//   * **reference** — the same layer body, the same fused dispatches, the same
//     fixed-order accumulation, but each expert's payload copied straight from the
//     artifact's mmap into its own device buffer.
//
// Both runs drive the *same* driver function, so the residual chain, the token
// schedule and the release points are identical and only the payload source differs.
//
// The two must be **bit-identical**: `moe_out`, the layer output, the routed ids and
// the routed weights, on every step of a two-layer stack. Bit-identity, not a
// tolerance, is the right instrument because the only difference between them is
// where the bytes came from, and bytes either match or they do not.
//
// Non-vacuity is asserted in-gate by a **wrong-expert probe**: the reference is
// re-run with one slot pointed at a different expert, and the comparison must go
// red. Without that line, "bit-identical" would only show that two runs agree.
//
// -----------------------------------------------------------------------------
// Two defects this gate found in the executor, both worth recording
//
//   1. **Completed transfers were never reaped.** The registry publishes a slot when
//      the transfer's completion event is *queried*, and nothing else performs that
//      query. Without a reap the finished operations stay pending forever, so once
//      the pool saturates no slot is ever reclaimable and the registry throws
//      "no reclaimable Hot VRAM slot is available" — which is exactly how this gate
//      failed on its first run, on the second token. The executor now reaps before
//      every dispatch, as the engine does between layers.
//   2. **The capacity fallback could not free anything.** Draining the compute
//      stream releases the readers, but a slot is only *reclaimable* once the copies
//      out of and into it have also completed — so the fallback must drain the
//      streams that carry expert traffic, not only the one that reads it. It did
//      not, and the release it performed was therefore decorative.
//
// -----------------------------------------------------------------------------
// The lease policy the gate exercises
//
// The pool is deliberately too small (8 slots) for the leases a token accumulates
// across the stack, so the executor's capacity fallback fires rather than being
// assumed away, and the bit-identity assertion is what shows the drain and release
// are value-neutral. The alternative — releasing leases while the expert kernels are
// still in flight — is a silent-corruption hazard; the reasoning is in the
// executor's header.
//
// Deliberately NOT covered here, named so it is not mistaken for coverage:
//   * the 43-layer stack and the model head (item 23's later gates);
//   * tier *quality* — whether the LRU chooses well; item 21 and the routing
//     placement study own that, and this gate only requires that whichever tier
//     answers delivers the artifact's bytes;
//   * streaming while the graph runs on other layers (checkpoint plan Stage D.2),
//     which has no caller until the graph is assembled.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/core/v4_expert_executor.hpp"
#include "architecture/deepseek_v4/core/v4_layer.hpp"
#include "architecture/deepseek_v4/core/v4_layer_body.hpp"
#include "architecture/deepseek_v4/core/v4_model_spec.hpp"
#include "architecture/deepseek_v4/kernels/v4_rope.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "infrastructure/core/supply_telemetry.hpp"
#include "infrastructure/io/direct_io_reader.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        std::fprintf(stderr, "HIP Error: %s at %s:%d\n", \
                     hipGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)
#endif

namespace {

using aeon::core::AEON_EXPERT_BYTES;
using aeon::core::ExpertRegistry;
using aeon::core::HostExpertPool;
using aeon::core::PrefetchStagingArena;
using aeon::core::SupplyTelemetry;
using aeon::core::UnifiedVRAMExpertPool;
using aeon::core::V4DeviceStreams;
using aeon::core::V4ExpertSupplyCoordinator;
using aeon::core::V4Layer;
using aeon::core::V4LayerBodyOutput;
using aeon::core::V4LayerBodyTables;
using aeon::core::V4NullLayerBodyObserver;
using aeon::core::V4RoutedExpertExecutor;
using aeon::core::V4RoutedExpertScratch;
using aeon::core::V4TieredExpertExecutor;

constexpr const char* kModelDir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";

// A two-layer stack, so leases accumulate across layers within one token and the
// executor's capacity fallback is genuinely reached. Layer 2 is CSA (ratio 4, so
// six tokens commit one compressed entry and the indexer runs); layer 3 is HCA
// (ratio 128, no indexer at all — trap 33). Between them the gate drives both
// compressed classes through the production executor.
constexpr uint32_t kLayerIds[2] = {2, 3};
constexpr uint32_t kStackDepth = 2;
constexpr uint32_t kContext = 256;
constexpr uint32_t kTokens = 6;

// Deliberately smaller than the `kStackDepth * 6` leases a token accumulates.
constexpr uint32_t kVramSlots = 8;
constexpr uint32_t kHostSlots = 8;
constexpr uint64_t kDemotionQueueCapacity = 4;

constexpr int kHidden = 4096;
constexpr int kHcDim = 4 * kHidden;
constexpr int kExperts = 6;

const uint32_t kTokenIds[kTokens] = {1000, 42, 7777, 1780, 90125, 130};

struct Harness {
    uint32_t checks{0};
    uint32_t failures{0};

    bool expect(const char* label, bool ok, const std::string& detail) {
        std::printf("  %-56s %-34s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
        ++checks;
        if (!ok) ++failures;
        return ok;
    }

    // ------------------------------------------------------------------ fixtures
    aeon::core::AeonModelLoader loader;
    UnifiedVRAMExpertPool vram_pool;
    HostExpertPool host_pool;
    PrefetchStagingArena staging;
    ExpertRegistry registry;
    SupplyTelemetry telemetry;
    aeon::io::DirectIOReader reader{64};
    std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> completions;
    uint64_t next_direct_io_id{1};
    V4ExpertSupplyCoordinator supply;
    V4RoutedExpertScratch expert_scratch;

    hipStream_t compute{nullptr};
    hipStream_t sdma{nullptr};
    hipStream_t sdma_cold{nullptr};
    hipStream_t demotion{nullptr};

    V4DeviceStreams streams() const {
        return V4DeviceStreams{compute, sdma, sdma_cold, demotion};
    }

    // One (token, layer) checkpoint pair.
    struct Traces {
        std::vector<int32_t> ids;
        std::vector<float> weights;
        std::vector<__half> moe_out;
        std::vector<float> res_out;
    };

    void configure_stack() {
        loader.open_model(kModelDir);
        const auto& format = loader.expert_format();

        vram_pool.allocate(kVramSlots, format);
        host_pool.allocate(kHostSlots, format);
        registry.init(format.num_layers, format.experts_per_layer, kVramSlots, kHostSlots,
                      /*preload_warm_host=*/false);

        CHECK_HIP(hipStreamCreateWithFlags(&compute, hipStreamNonBlocking));
        CHECK_HIP(hipStreamCreateWithFlags(&sdma, hipStreamNonBlocking));
        CHECK_HIP(hipStreamCreateWithFlags(&sdma_cold, hipStreamNonBlocking));
        CHECK_HIP(hipStreamCreateWithFlags(&demotion, hipStreamNonBlocking));

        supply.configure(
            &loader, &vram_pool, &host_pool, &registry, &staging, &telemetry, &reader,
            &completions, &next_direct_io_id, compute, sdma, sdma_cold, demotion,
            format.payload_bytes, kDemotionQueueCapacity);
        expert_scratch.allocate();
    }

    // ------------------------------------------------------------------ tables
    aeon::kernel::RopeTable sliding_rope;
    aeon::kernel::RopeTable compressed_rope;
    float* d_sliding_cos{nullptr};
    float* d_sliding_sin{nullptr};
    float* d_compressed_cos{nullptr};
    float* d_compressed_sin{nullptr};

    void upload_rope_tables(const aeon::core::DeepSeekV4Config& cfg) {
        sliding_rope.init(kContext, cfg.rope_theta, 1.0f);
        compressed_rope.init(
            kContext, cfg.compress_rope_theta, cfg.rope_scaling.factor,
            cfg.rope_scaling.beta_fast, cfg.rope_scaling.beta_slow,
            static_cast<uint32_t>(cfg.rope_scaling.original_max_position_embeddings));
        const size_t bytes =
            static_cast<size_t>(kContext) * sliding_rope.half_rope * sizeof(float);
        CHECK_HIP(hipMalloc(&d_sliding_cos, bytes));
        CHECK_HIP(hipMalloc(&d_sliding_sin, bytes));
        CHECK_HIP(hipMalloc(&d_compressed_cos, bytes));
        CHECK_HIP(hipMalloc(&d_compressed_sin, bytes));
        CHECK_HIP(hipMemcpy(d_sliding_cos, sliding_rope.cos_cache.data(), bytes,
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_sliding_sin, sliding_rope.sin_cache.data(), bytes,
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_compressed_cos, compressed_rope.cos_cache.data(), bytes,
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_compressed_sin, compressed_rope.sin_cache.data(), bytes,
                            hipMemcpyHostToDevice));
    }

    V4LayerBodyTables tables() const {
        V4LayerBodyTables view;
        view.sliding_cos = d_sliding_cos;
        view.sliding_sin = d_sliding_sin;
        view.compressed_cos = d_compressed_cos;
        view.compressed_sin = d_compressed_sin;
        return view;
    }
};

// -----------------------------------------------------------------------------
// The reference executor: the same two fused dispatches and the same fixed-order
// combine as the production path, with the storage system removed. Each expert's
// payload is copied straight out of the artifact's mmap into its own device buffer,
// so the only thing that can differ from the production run is how the bytes got
// there.
//
// `wrong_slot` / `wrong_expert` are the non-vacuity probe: pointing one slot at a
// different expert must change the result, or the bit-identity comparison is not
// evidence.
// -----------------------------------------------------------------------------
class DirectPayloadExecutor final : public V4RoutedExpertExecutor {
public:
    DirectPayloadExecutor(V4RoutedExpertScratch& scratch_in,
                          aeon::core::AeonModelLoader& loader_in,
                          hipStream_t stream_in,
                          uint32_t layer_id_in,
                          float swiglu_limit_in)
        : scratch(scratch_in),
          loader(loader_in),
          stream(stream_in),
          layer_id(layer_id_in),
          swiglu_limit(swiglu_limit_in) {}

    ~DirectPayloadExecutor() override {
        for (int k = 0; k < kExperts; ++k) {
            if (d_payload[k] != nullptr) {
                (void)hipFree(d_payload[k]);
                d_payload[k] = nullptr;
            }
        }
    }

    V4RoutedExpertScratch& scratch;
    aeon::core::AeonModelLoader& loader;
    hipStream_t stream{nullptr};
    uint32_t layer_id{0};
    float swiglu_limit{10.0f};
    int wrong_slot{-1};
    uint32_t wrong_expert{0};

    std::vector<int32_t> ids;
    uint8_t* d_payload[kExperts]{};

    void allocate() {
        for (int k = 0; k < kExperts; ++k) {
            CHECK_HIP(hipMalloc(&d_payload[k], AEON_EXPERT_BYTES));
        }
    }

    void on_routing_ready(uint32_t, uint32_t, const std::vector<int32_t>& routed_ids,
                          const std::vector<float>&) override {
        ids = routed_ids;
    }

    void accumulate_routed(uint32_t, uint32_t, const half* expert_input,
                           const float* expert_weights, half* moe_accum) override {
        aeon::kernel::SwizzledW13ExpertPtrs w13{};
        aeon::kernel::SwizzledW2ExpertPtrs w2{};
        for (int k = 0; k < kExperts; ++k) {
            const uint32_t expert = (k == wrong_slot) ? wrong_expert
                                                      : static_cast<uint32_t>(ids[k]);
            const uint8_t* source = loader.get_expert_data(layer_id, expert);
            if (source == nullptr) {
                std::fprintf(stderr, "reference: expert %u is not in the artifact\n", expert);
                std::exit(1);
            }
            CHECK_HIP(hipMemcpyAsync(d_payload[k], source, AEON_EXPERT_BYTES,
                                     hipMemcpyHostToDevice, stream));
            uint8_t* base = d_payload[k];
            w13.w1[k] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W1_PACKED_OFFSET);
            w13.s1[k] = reinterpret_cast<const __half*>(base + aeon::core::AEON_W1_SCALE_OFFSET);
            w13.w3[k] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W3_PACKED_OFFSET);
            w13.s3[k] = reinterpret_cast<const __half*>(base + aeon::core::AEON_W3_SCALE_OFFSET);
            w2.w2[k] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W2_PACKED_OFFSET);
            w2.s2[k] = reinterpret_cast<const __half*>(base + aeon::core::AEON_W2_SCALE_OFFSET);
        }

        aeon::kernel::dispatch_aeon_moe_fused_w13_swiglu<8, 4, 8, 16>(
            expert_input, w13, scratch.d_expert_hidden, nullptr, kHidden, kExperts, 2048,
            kHidden, swiglu_limit, stream);
        aeon::kernel::dispatch_aeon_moe_fused_w2_contrib<8, 8, 4, 16>(
            scratch.d_expert_hidden, w2, expert_weights, scratch.d_contrib, kExperts,
            kHidden, 2048, stream);
        constexpr int kThreads = 256;
        aeon::kernel::v4_moe_accumulate_fixed_order_kernel
            <<<(kHidden + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                scratch.d_contrib, kExperts, moe_accum, moe_accum, kHidden);
    }
};

// The reference needs one payload set per layer, while the driver takes one
// executor. This selects by **layer id** rather than by a depth the driver would
// have to maintain: every hook already receives `layer_id`, so deriving the
// selection from it removes a piece of state that can silently go stale. (It did:
// the first version indexed by a depth counter the driver never set, which routed
// layer 3's expert lookups into layer 2's container — and the gate reported it as a
// delivery mismatch, which is what a wrong-expert lookup is.)
class PerLayerReference final : public V4RoutedExpertExecutor {
public:
    std::vector<uint32_t> layer_ids;
    std::vector<V4RoutedExpertExecutor*> by_layer;

    V4RoutedExpertExecutor& select(uint32_t layer_id) const {
        for (size_t i = 0; i < layer_ids.size(); ++i) {
            if (layer_ids[i] == layer_id) return *by_layer[i];
        }
        std::fprintf(stderr, "reference adapter: unknown layer %u\n", layer_id);
        std::exit(1);
    }

    void on_routing_ready(uint32_t layer_id, uint32_t position,
                          const std::vector<int32_t>& ids,
                          const std::vector<float>& weights) override {
        select(layer_id).on_routing_ready(layer_id, position, ids, weights);
    }

    void accumulate_routed(uint32_t layer_id, uint32_t position,
                           const half* expert_input, const float* expert_weights,
                           half* moe_accum) override {
        select(layer_id).accumulate_routed(layer_id, position, expert_input,
                                           expert_weights, moe_accum);
    }
};

// -----------------------------------------------------------------------------
// One run of the stack: `kTokens` positions through `kStackDepth` layers, with the
// residual carried by the device itself — layer L's output is layer L+1's input and
// the last layer's output is the next token's input. Nothing re-seeds it between
// steps, so a delivery defect on any layer propagates and is visible downstream.
// -----------------------------------------------------------------------------
std::vector<Harness::Traces> run_stack(
    Harness& h,
    std::vector<V4Layer*>& layers,
    aeon::core::PipelineScratchBuffers& scratch,
    const V4LayerBodyTables& view,
    V4RoutedExpertExecutor& executor,
    V4TieredExpertExecutor* production
) {
    V4NullLayerBodyObserver observer;

    // A fixed residual, so both runs start from the identical state. It is a plain
    // pattern rather than a real embedding: this gate measures delivery, and the
    // embedding has its own step.
    std::vector<float> seed(kHcDim);
    for (int i = 0; i < kHcDim; ++i) {
        seed[static_cast<size_t>(i)] = static_cast<float>((i % 17) - 8) * 0.0125f;
    }
    CHECK_HIP(hipMemcpy(scratch.d_res_in, seed.data(), kHcDim * sizeof(float),
                        hipMemcpyHostToDevice));

    std::vector<Harness::Traces> traces;
    traces.reserve(kTokens * kStackDepth);
    for (uint32_t t = 0; t < kTokens; ++t) {
        for (uint32_t depth = 0; depth < kStackDepth; ++depth) {
            const V4LayerBodyOutput out = aeon::core::run_layer_body_decoding(
                *layers[depth], scratch, view, kTokenIds[t], t, h.compute, executor, observer);

            Harness::Traces trace;
            trace.ids = out.topk_indices;
            trace.weights = out.topk_weights;
            trace.moe_out.resize(kHidden);
            trace.res_out.resize(kHcDim);
            CHECK_HIP(hipMemcpyAsync(trace.moe_out.data(), scratch.d_moe_accum,
                                     kHidden * sizeof(__half), hipMemcpyDeviceToHost,
                                     h.compute));
            CHECK_HIP(hipMemcpyAsync(trace.res_out.data(), scratch.d_res_in,
                                     kHcDim * sizeof(float), hipMemcpyDeviceToHost,
                                     h.compute));
            CHECK_HIP(hipStreamSynchronize(h.compute));
            traces.push_back(std::move(trace));
        }
        // The token boundary. The caller has a compute-stream boundary here in the
        // real graph (sampling reads the logits back), which is what makes handing
        // the leases back safe.
        if (production != nullptr) {
            production->release_leases();
        }
    }
    return traces;
}

size_t differing_halves(const std::vector<__half>& a, const std::vector<__half>& b) {
    size_t differ = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (__half_as_ushort(a[i]) != __half_as_ushort(b[i])) ++differ;
    }
    return differ;
}

size_t differing_floats(const std::vector<float>& a, const std::vector<float>& b) {
    size_t differ = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++differ;
    }
    return differ;
}

} // namespace

int main() {
    std::printf("================================================================================"
                "================\n");
    std::printf("  Item 23 — the production routed-expert executor (delivery gate)\n");
    std::printf("================================================================================"
                "================\n");

    aeon::core::select_compute_device(true);
    Harness h;
    h.configure_stack();

    const aeon::core::DeepSeekV4Config cfg =
        aeon::core::DeepSeekV4Config::load_from_json(std::string(kModelDir) + "/config.json");
    const std::vector<aeon::core::V4LayerSpec> specs =
        aeon::core::V4ModelSpec::resolve_layers(cfg);
    h.upload_rope_tables(cfg);
    const V4LayerBodyTables view = h.tables();
    const auto& format = h.loader.expert_format();
    const size_t steps = static_cast<size_t>(kTokens) * kStackDepth;

    // ---------------------------------------------------------------------
    std::printf("\n[A] The fixture is the real artifact\n");
    // ---------------------------------------------------------------------
    h.expect("A: the expert container describes this model",
             format.num_layers == 43 && format.experts_per_layer == 256 &&
                 format.payload_bytes == AEON_EXPERT_BYTES,
             "layers=" + std::to_string(format.num_layers) + " experts=" +
                 std::to_string(format.experts_per_layer) + " payload=" +
                 std::to_string(format.payload_bytes) + "B");
    h.expect("A: the stack is CSA then HCA (both compressed classes run)",
             specs.at(kLayerIds[0]).attention_kind == aeon::core::V4AttentionKind::CSA &&
                 specs.at(kLayerIds[1]).attention_kind == aeon::core::V4AttentionKind::HCA,
             "layer " + std::to_string(kLayerIds[0]) + " ratio=" +
                 std::to_string(specs.at(kLayerIds[0]).compression_ratio) + ", layer " +
                 std::to_string(kLayerIds[1]) + " ratio=" +
                 std::to_string(specs.at(kLayerIds[1]).compression_ratio));
    {
        const uint8_t* expert_a = h.loader.get_expert_data(kLayerIds[0], 0);
        const uint8_t* expert_b = h.loader.get_expert_data(kLayerIds[0], 1);
        h.expect("A: distinct experts have distinct bytes (not vacuous)",
                 expert_a != nullptr && expert_b != nullptr &&
                     std::memcmp(expert_a, expert_b, AEON_EXPERT_BYTES) != 0,
                 "memcmp over the whole payload");
    }

    // ---------------------------------------------------------------------
    std::printf("\n[B] Production executor vs direct-payload reference\n");
    // ---------------------------------------------------------------------
    std::vector<std::unique_ptr<V4Layer>> production_layers;
    for (uint32_t depth = 0; depth < kStackDepth; ++depth) {
        auto layer = std::make_unique<V4Layer>();
        layer->init_with_loader(specs.at(kLayerIds[depth]), h.loader, kContext);
        production_layers.push_back(std::move(layer));
    }
    std::vector<V4Layer*> production_ptrs;
    for (auto& layer : production_layers) production_ptrs.push_back(layer.get());

    aeon::core::PipelineScratchBuffers production_scratch;
    production_scratch.allocate();
    V4TieredExpertExecutor production(h.supply, h.vram_pool, h.staging, h.registry,
                                      h.expert_scratch, h.streams(), cfg.swiglu_limit);

    const uint64_t cold_before = h.registry.misses_cold;
    const std::vector<Harness::Traces> production_trace = run_stack(
        h, production_ptrs, production_scratch, view, production, &production);

    h.expect("B: the supply actually answered from cold storage",
             h.registry.misses_cold > cold_before,
             "cold misses +" + std::to_string(h.registry.misses_cold - cold_before));
    h.expect("B: the capacity fallback fired rather than the registry throwing",
             production.forced_drains() > 0,
             "drains=" + std::to_string(production.forced_drains()) + " (pool=" +
                 std::to_string(kVramSlots) + " slots, " +
                 std::to_string(kStackDepth * kExperts) + " leases/token)");
    h.expect("B: every lease was handed back",
             production.outstanding_leases() == 0,
             "outstanding=" + std::to_string(production.outstanding_leases()));

    // The reference: same body, same kernels, storage removed.
    std::vector<std::unique_ptr<V4Layer>> reference_layers;
    for (uint32_t depth = 0; depth < kStackDepth; ++depth) {
        auto layer = std::make_unique<V4Layer>();
        layer->init_with_loader(specs.at(kLayerIds[depth]), h.loader, kContext);
        reference_layers.push_back(std::move(layer));
    }
    std::vector<V4Layer*> reference_ptrs;
    for (auto& layer : reference_layers) reference_ptrs.push_back(layer.get());

    std::vector<std::unique_ptr<DirectPayloadExecutor>> reference_executors;
    std::vector<V4RoutedExpertExecutor*> reference_as_base;
    for (uint32_t depth = 0; depth < kStackDepth; ++depth) {
        reference_executors.push_back(std::make_unique<DirectPayloadExecutor>(
            h.expert_scratch, h.loader, h.compute, kLayerIds[depth], cfg.swiglu_limit));
        reference_executors.back()->allocate();
    }
    for (auto& executor : reference_executors) reference_as_base.push_back(executor.get());

    PerLayerReference reference_adapter;
    reference_adapter.layer_ids.assign(std::begin(kLayerIds), std::end(kLayerIds));
    reference_adapter.by_layer = reference_as_base;

    aeon::core::PipelineScratchBuffers reference_scratch;
    reference_scratch.allocate();
    const std::vector<Harness::Traces> reference_trace = run_stack(
        h, reference_ptrs, reference_scratch, view, reference_adapter, nullptr);

    size_t id_diffs = 0;
    size_t weight_diffs = 0;
    size_t moe_diffs = 0;
    size_t res_diffs = 0;
    for (size_t s = 0; s < steps; ++s) {
        if (production_trace[s].ids != reference_trace[s].ids) ++id_diffs;
        if (production_trace[s].weights != reference_trace[s].weights) ++weight_diffs;
        moe_diffs += differing_halves(production_trace[s].moe_out, reference_trace[s].moe_out);
        res_diffs += differing_floats(production_trace[s].res_out, reference_trace[s].res_out);
    }

    h.expect("B: the routed expert set is identical on every step",
             id_diffs == 0, "differing steps=" + std::to_string(id_diffs) + " of " +
                 std::to_string(steps));
    h.expect("B: the routing weights are bit-identical",
             weight_diffs == 0, "differing steps=" + std::to_string(weight_diffs));
    h.expect("B: moe_out is bit-identical (delivery is transparent)",
             moe_diffs == 0, "differing elements=" + std::to_string(moe_diffs) + " of " +
                 std::to_string(steps * kHidden));
    h.expect("B: the chained residual carries no delivery delta",
             res_diffs == 0, "differing elements=" + std::to_string(res_diffs) + " of " +
                 std::to_string(steps * kHcDim));

    // ---------------------------------------------------------------------
    std::printf("\n[C] The comparison can fail\n");
    // ---------------------------------------------------------------------
    // Without this, "bit-identical" would only say that two runs of the same code
    // agree. One reference slot is pointed at a different expert; the expert bytes
    // change, so `moe_out` must change.
    {
        std::vector<std::unique_ptr<V4Layer>> probe_layers;
        for (uint32_t depth = 0; depth < kStackDepth; ++depth) {
            auto layer = std::make_unique<V4Layer>();
            layer->init_with_loader(specs.at(kLayerIds[depth]), h.loader, kContext);
            probe_layers.push_back(std::move(layer));
        }
        std::vector<V4Layer*> probe_ptrs;
        for (auto& layer : probe_layers) probe_ptrs.push_back(layer.get());

        std::vector<std::unique_ptr<DirectPayloadExecutor>> probe_executors;
        std::vector<V4RoutedExpertExecutor*> probe_as_base;
        for (uint32_t depth = 0; depth < kStackDepth; ++depth) {
            probe_executors.push_back(std::make_unique<DirectPayloadExecutor>(
                h.expert_scratch, h.loader, h.compute, kLayerIds[depth], cfg.swiglu_limit));
            probe_executors.back()->allocate();
        }
        probe_executors[0]->wrong_slot = 0;
        for (auto& executor : probe_executors) probe_as_base.push_back(executor.get());

        PerLayerReference probe_adapter;
        probe_adapter.layer_ids.assign(std::begin(kLayerIds), std::end(kLayerIds));
        probe_adapter.by_layer = probe_as_base;

        aeon::core::PipelineScratchBuffers probe_scratch;
        probe_scratch.allocate();
        const std::vector<Harness::Traces> probe_trace = run_stack(
            h, probe_ptrs, probe_scratch, view, probe_adapter, nullptr);

        size_t probe_diffs = 0;
        for (size_t s = 0; s < steps; ++s) {
            probe_diffs += differing_halves(probe_trace[s].moe_out,
                                            reference_trace[s].moe_out);
        }
        h.expect("C: substituting one expert's payload moves moe_out",
                 probe_diffs > 0, "differing elements=" + std::to_string(probe_diffs) +
                     " (the delivered bytes are read)");
    }

    // Settle so the final assertion describes a quiescent registry rather than one
    // mid-transfer: every operation must have completed and been retired, which is
    // what leaves all eight slots occupied rather than parked in a pending state.
    CHECK_HIP(hipStreamSynchronize(h.compute));
    CHECK_HIP(hipStreamSynchronize(h.sdma));
    CHECK_HIP(hipStreamSynchronize(h.sdma_cold));
    CHECK_HIP(hipStreamSynchronize(h.demotion));
    h.supply.reap_registry_transfers();
    h.expect("C: after settling, every VRAM slot is occupied and none is pending",
             h.registry.invariants_hold() &&
                 h.registry.published_hot_slots() == kVramSlots &&
                 h.registry.pending_transfer_count() == 0,
             "hot=" + std::to_string(h.registry.published_hot_slots()) + " warm=" +
                 std::to_string(h.registry.published_warm_slots()) + " pending=" +
                 std::to_string(h.registry.pending_transfer_count()));

    std::printf("\n[Item 23 — production routed-expert executor] %s — %u checks, %u failed\n",
                h.failures == 0 ? "PASS" : "FAIL", h.checks, h.failures);
    return h.failures == 0 ? 0 : 1;
}
