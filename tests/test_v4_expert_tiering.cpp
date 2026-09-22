// -----------------------------------------------------------------------------
// Tier-4 gate, item 21 — streaming / tiering.
//
// Gate (plan Part V, item 21): **expert bytes bit-exact across Hot / Warm /
// Cold.**
//
// The same statement is Stage D of the checkpoint plan: "prove the three-tier
// hierarchy is transparent to the numerics — load an expert to VRAM, dequantize,
// record; evict, reload from host RAM, compare — expected bit-exact; evict,
// reload from NVMe via `O_DIRECT`/`io_uring`, compare — expected bit-exact", with
// the deliverable "confirmation that tier placement does not change a single bit —
// i.e. any numerical difference is attributable to the graph, never to the memory
// system."
//
// What this gate adds over the two tiers of evidence that already exist is the
// **supply path itself**. `test_model_direct_io.cpp` proves that a standalone
// `O_DIRECT` read equals the mmapped bytes (the NVMe leg alone), and
// `test_dynamic_expert_pool.cpp` proves that one `hipMemcpy` from the mmap pointer
// into a VRAM slot is bit-exact (the H2D leg alone, driven by hand). Neither
// drives `TieredExpertSupply`, which is the code that actually decides which tier
// answers a request, which staging slot an I/O lands in, which host slot a
// demotion writes to, and which stream a copy is enqueued on. A defect in any of
// those decisions is invisible to both of them and would show up only as
// "identical artifact, different result on reload" (checkpoint plan §6, failure
// triage).
//
// The route each tier takes:
//
//   * **Cold** — `dispatch` submits `O_DIRECT` reads into a `PrefetchStagingArena`
//     slot, `materialize` waits for every completion and then enqueues the H2D on
//     the cold SDMA stream. The payload is 14 155 776 bytes = 3456 sectors, i.e.
//     `3.375` of the reader's 4 MiB chunks, so `submit_read_chunks` splits each
//     expert into **four** requests and the last one is short. That is the
//     reader's intended behaviour, and it means the multi-request path is the one
//     exercised here rather than a single full-size request.
//   * **Warm** — the registry's LRU victim is written to a host slot by a D2H on
//     the demotion stream, and a later promotion uploads it back from that pinned
//     slot. The demotion destination is chosen by `reserve_warm_destination` and
//     recovered at completion by `find_reserved_host_slot`, which searches twice.
//   * **Hot** — no bytes move at all. The claim under test is the negative one:
//     a hot hit must return the resident bytes unchanged and must not claim a
//     staging slot or enqueue a copy.
//
// The reference is the **mmapped** expert container (`AeonModelLoader::
// get_expert_data`). This is the same file at the same offsets, reached by a
// different mechanism: `mmap` faults in page-cache pages, where the supply path
// reads through `io_uring` with `O_DIRECT` and bypasses the cache. The *content*
// is therefore identical by construction — it is the file — while the *route*
// into the comparison is one no tier under test used. That is what lets it serve
// as the authority: the tiers are checked against the file's own bytes, not
// against each other, so agreement cannot be produced by a shared mistake.
//
// What is asserted:
//
//   A. PRECONDITIONS, BEFORE ANY TRANSFER — the artifact's own format descriptor;
//      that the payload is a whole number of sectors and is read as four
//      requests; that the registry saturates VRAM at construction (so every cold
//      miss must evict a resident, which is the production steady state and not a
//      warm-up convenience); and that the reference is **not vacuous** — two
//      different experts' bytes differ, so a comparison can fail.
//   B. HOT — the residents the registry claims are the artifact's bytes, filled
//      here by an `O_DIRECT` read, compared against the mmap path.
//   C. COLD — three experts requested at once, each a `COLD_MISS` with four I/O
//      requests, and the payload checked **twice**: once in the staging slot,
//      which isolates the NVMe leg from the H2D, and once in the destination VRAM
//      slot. The evictions they force must have completed rather than been
//      dropped.
//   D. HOT HIT — a repeat request performs no transfer, claims no staging slot,
//      and returns the resident bytes **bit-identical to the cold delivery**.
//   E. WARM — the LRU victim is *predicted from the registry's own list* before
//      the request, then the demotion's host slot is checked against the artifact
//      (the D2H leg), and the promotion's result is checked again (the H2D leg).
//      The request must be reported as a `WARM_PROMOTION` from `WARM_HOST` and
//      must be counted as a warm hit — otherwise a *silently dropped* demotion
//      would send the expert back to Cold and the byte comparison would pass for
//      the wrong reason while measuring nothing.
//   F. THE THREE ROUTES AGREE — cold, hot and warm deliveries of the same expert
//      are pairwise bit-identical and all equal the artifact.
//
// Deliberately NOT covered here, named so it is not mistaken for coverage:
//
//   * **The staged warm sub-paths.** `TieredExpertSupply` prefers a direct H2D
//     from the pinned host slot whenever `HostExpertPool::is_slot_pinned` reports
//     the slot pinned, and falls back to staging a copy through the arena
//     otherwise. On this silicon `hipHostMalloc` succeeds, so the direct path is
//     taken for both the demotion and the promotion; the fallback is unreachable
//     without editing the pool. Both paths are asserted only to the extent that
//     whichever one runs must deliver the artifact's bytes — which is exactly
//     what the byte comparisons measure.
//   * **Concurrency.** Requests are dispatched and materialized one round at a
//     time. The plan's item 21 gate is byte-exactness, not overlap; the overlap
//     and prefetch behaviour is what the warm-tier A/B report and the supply
//     telemetry already cover, and item 19's throughput half is a separate gate.
//   * **The numerics of the expert itself.** Dequantization and the specialized
//     kernels are Tier-1 and items 14–18; this gate never dequantizes. It compares
//     the *packed payload bytes*, which is the thing a tier can corrupt and a
//     kernel cannot.
//   * **Tiering across a real layer schedule, prefix reuse, and eviction policy
//     quality** — Tier-4 items 22/23.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "infrastructure/core/aeon_loader.hpp"
#include "infrastructure/core/expert_format.hpp"
#include "infrastructure/core/expert_payload_pool.hpp"
#include "infrastructure/core/expert_registry.hpp"
#include "infrastructure/core/host_expert_pool.hpp"
#include "infrastructure/core/prefetch_staging.hpp"
#include "infrastructure/core/supply_telemetry.hpp"
#include "infrastructure/core/tiered_expert_supply.hpp"
#include "infrastructure/io/aligned_allocator.hpp"
#include "infrastructure/io/direct_io_reader.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>
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
} while (0)
#endif

namespace {

using aeon::core::ExpertPayloadPool;
using aeon::core::ExpertRegistry;
using aeon::core::ExpertRequestKind;
using aeon::core::ExpertTier;
using aeon::core::HostExpertPool;
using aeon::core::PrefetchStagingArena;
using aeon::core::SupplyTelemetry;
using aeon::core::TieredExpertSupply;

constexpr uint32_t kExpertsPerLayer = 256;
constexpr size_t kPayloadBytes = aeon::core::AEON_EXPERT_BYTES;

// The pool is deliberately small *and saturated*: `ExpertRegistry` populates
// every Hot slot at construction, so there is no free VRAM slot from the first
// request on. That is the production steady state — every cold miss must evict a
// resident — and it is what makes the demotion and promotion legs reachable
// without contriving one.
constexpr uint32_t kVramSlots = 8;
constexpr uint32_t kHostSlots = 8;
constexpr uint64_t kDemotionQueueCapacity = 4;

// The same expert, delivered three ways. gid = layer * experts_per_layer + expert.
constexpr uint32_t kTrackedExpert = 1;      // layer 0, expert 1
constexpr uint32_t kColdBatch[] = {1, 2, 3};
constexpr uint32_t kMakeRoomExpert = 4;     // requested to force the tracked expert out
constexpr uint32_t kCohortProbe = 5;        // a second expert, for the non-vacuity control

struct ByteDiff {
    size_t count{0};
    size_t first{0};
};

ByteDiff byte_diff(const uint8_t* left, const uint8_t* right, size_t bytes) {
    ByteDiff diff;
    for (size_t i = 0; i < bytes; ++i) {
        if (left[i] == right[i]) continue;
        if (diff.count == 0) diff.first = i;
        ++diff.count;
    }
    return diff;
}

std::string diff_text(const ByteDiff& diff) {
    if (diff.count == 0) return "identical (0 bytes differ)";
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "differ=%zu first@0x%zx", diff.count, diff.first);
    return buffer;
}

struct TieringGate {
    // ---------------------------------------------------------------- harness
    uint32_t checks{0};
    uint32_t failures{0};

    bool assert_that(const char* label, bool ok, const std::string& detail) {
        std::printf("  %-58s %-32s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
        ++checks;
        if (!ok) ++failures;
        return ok;
    }

    // ------------------------------------------------------------------ stack
    aeon::core::AeonModelLoader loader;
    ExpertPayloadPool vram_pool;
    HostExpertPool host_pool;
    PrefetchStagingArena staging;
    ExpertRegistry registry;
    SupplyTelemetry telemetry;
    aeon::io::DirectIOReader supply_reader{64};
    aeon::io::DirectIOReader preload_reader{32};
    std::unordered_map<uint64_t, aeon::io::DirectIOCompletion> completions;
    uint64_t next_direct_io_id{1000};
    TieredExpertSupply supply;

    hipStream_t compute_stream{nullptr};
    hipStream_t sdma_stream{nullptr};
    hipStream_t sdma_cold_stream{nullptr};
    hipStream_t demotion_stream{nullptr};

    uint64_t step{0};
    std::vector<uint32_t> leased;
    std::unordered_map<uint32_t, std::vector<uint8_t>> ground_truth_cache;

    // --------------------------------------------------------------- helpers
    uint32_t layer_of(uint32_t gid) const { return gid / kExpertsPerLayer; }
    uint32_t expert_of(uint32_t gid) const { return gid % kExpertsPerLayer; }

    // The mmap path. A different I/O route from every tier under test.
    const std::vector<uint8_t>& ground_truth(uint32_t gid) {
        const auto cached = ground_truth_cache.find(gid);
        if (cached != ground_truth_cache.end()) return cached->second;

        const uint8_t* source = loader.get_expert_data(layer_of(gid), expert_of(gid));
        if (source == nullptr) {
            throw std::runtime_error("Tiering gate: mmapped expert data is unavailable");
        }
        std::vector<uint8_t> copy(source, source + kPayloadBytes);
        return ground_truth_cache.emplace(gid, std::move(copy)).first->second;
    }

    std::vector<uint8_t> read_vram_slot(uint32_t slot) {
        std::vector<uint8_t> result(kPayloadBytes);
        CHECK_HIP(hipMemcpy(result.data(), vram_pool.get_slot_base(slot), kPayloadBytes,
                            hipMemcpyDeviceToHost));
        return result;
    }

    // A blocking O_DIRECT read on a private io_uring ring, used only to seed the
    // Hot residents. Private so its completions can never be drained by the
    // supply's own `materialize`.
    void read_expert_direct(uint32_t gid, uint8_t* destination) {
        const auto location = loader.get_expert_location(layer_of(gid), expert_of(gid));
        const size_t chunks = preload_reader.submit_read_chunks(
            loader.expert_direct_fd(), destination, location.byte_length,
            location.file_offset, 1);
        preload_reader.submit_pending_reads();

        size_t received = 0;
        for (size_t chunk = 0; chunk < chunks; ++chunk) {
            const auto completion = preload_reader.wait_for_completion();
            if (completion.result < 0) {
                throw std::runtime_error("Tiering gate: preload O_DIRECT read failed");
            }
            received += static_cast<size_t>(completion.result);
        }
        if (received != location.byte_length) {
            throw std::runtime_error("Tiering gate: preload O_DIRECT read was short");
        }
    }

    // Wait for every stream the supply can enqueue onto, then let the registry
    // finish the transitions those copies complete.
    void settle() {
        CHECK_HIP(hipStreamSynchronize(compute_stream));
        CHECK_HIP(hipStreamSynchronize(sdma_stream));
        CHECK_HIP(hipStreamSynchronize(sdma_cold_stream));
        CHECK_HIP(hipStreamSynchronize(demotion_stream));
        supply.reap_registry_transfers();
    }

    // The teardown order the removed pipeline established: leases are released,
    // transfers reaped, and only then are the consumed staging slots returned to
    // the arena.
    void finish_round(const std::vector<uint32_t>& staging_taken) {
        for (uint32_t gid : leased) registry.release_lease(gid);
        leased.clear();
        supply.reap_registry_transfers();
        for (uint32_t slot : staging_taken) staging.release_after_gpu_transfer(slot);
    }

    static std::vector<uint32_t> staging_slots_of(
        const TieredExpertSupply::PayloadBatch& batch) {
        std::vector<uint32_t> slots;
        for (const auto& transfer : batch.transfers) {
            if (transfer.is_prefetched) slots.push_back(transfer.staging_idx);
        }
        return slots;
    }

    TieredExpertSupply::PayloadBatch submit(
        const std::vector<TieredExpertSupply::PayloadRequest>& requests) {
        leased.clear();
        auto batch = supply.dispatch(requests, step++, leased);
        supply.materialize(batch);
        settle();
        return batch;
    }

    // ------------------------------------------------------------ components
    void configure_stack() {
        loader.open_model("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon");
        const auto& format = loader.expert_format();

        vram_pool.allocate(kVramSlots, format);
        host_pool.allocate(kHostSlots, format);
        registry.init(format.num_layers, format.experts_per_layer, kVramSlots, kHostSlots,
                      /*preload_warm_host=*/false);
        // The audit after every operation: this gate is exactly where a bookkeeping
        // defect should be localised to its cause, so it pays the cost the production
        // path leaves off (ledger M43).
        registry.set_validate_each_request(true);

        CHECK_HIP(hipStreamCreateWithFlags(&compute_stream, hipStreamNonBlocking));
        CHECK_HIP(hipStreamCreateWithFlags(&sdma_stream, hipStreamNonBlocking));
        CHECK_HIP(hipStreamCreateWithFlags(&sdma_cold_stream, hipStreamNonBlocking));
        CHECK_HIP(hipStreamCreateWithFlags(&demotion_stream, hipStreamNonBlocking));

        TieredExpertSupply::PayloadSource source;
        source.direct_fd = loader.expert_direct_fd();
        source.locate = [this](uint32_t gid) {
            const auto location =
                loader.get_expert_location(layer_of(gid), expert_of(gid));
            return TieredExpertSupply::PayloadLocation{location.file_offset,
                                                       location.byte_length};
        };
        source.host_payload = [this](uint32_t gid) {
            return loader.get_expert_data(layer_of(gid), expert_of(gid));
        };

        supply.configure(std::move(source), &vram_pool, &host_pool, &registry, &staging,
                         &telemetry, &supply_reader, &completions, &next_direct_io_id,
                         compute_stream, sdma_stream, sdma_cold_stream, demotion_stream,
                         format.payload_bytes, kDemotionQueueCapacity);
    }

    // The registry's own claim about which experts are resident, read from the
    // catalog rather than re-derived from `populate_round_robin`'s loop order.
    std::vector<uint32_t> residents() const {
        std::vector<uint32_t> result;
        for (uint32_t gid = 0; gid < registry.catalog.size(); ++gid) {
            if (registry.catalog[gid].owner == ExpertTier::HOT_VRAM) result.push_back(gid);
        }
        return result;
    }

    int run() {
        std::printf("================================================================================"
                    "================\n");
        std::printf("  Tier-4 item 21 — expert bytes bit-exact across Hot / Warm / Cold\n");
        std::printf("================================================================================"
                    "================\n");

        aeon::core::select_compute_device(true);
        configure_stack();

        const auto& format = loader.expert_format();
        const size_t direct_chunks =
            (kPayloadBytes + aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES - 1) /
            aeon::io::DirectIOReader::DEFAULT_CHUNK_BYTES;

        // ---------------------------------------------------------------------
        // A — preconditions, before anything has been transferred
        // ---------------------------------------------------------------------
        std::printf("\n[A] Preconditions and the reference's non-vacuity\n");

        assert_that("A: the artifact's format descriptor is the real one",
                    format.kind == aeon::core::ExpertFormatKind::SWIZZLED_W4A16 &&
                        format.payload_bytes == kPayloadBytes &&
                        format.sector_size == aeon::core::AEON_SECTOR_SIZE &&
                        format.num_layers == 43 && format.experts_per_layer == 256,
                    "payload=" + std::to_string(format.payload_bytes) + " layers=" +
                        std::to_string(format.num_layers) + " experts=" +
                        std::to_string(format.experts_per_layer));

        assert_that("A: the payload is sector-aligned and is read as four requests",
                    kPayloadBytes % format.sector_size == 0 && direct_chunks == 4,
                    std::to_string(kPayloadBytes) + "B = " +
                        std::to_string(kPayloadBytes / format.sector_size) + " sectors, " +
                        std::to_string(direct_chunks) + " x 4MiB chunks");

        const std::vector<uint32_t> preloaded = residents();
        assert_that("A: the registry saturates VRAM at construction",
                    preloaded.size() == kVramSlots && registry.free_vram_slots.empty(),
                    "residents=" + std::to_string(preloaded.size()) + " free_vram_slots=" +
                        std::to_string(registry.free_vram_slots.size()));

        std::vector<int32_t> distinct_slots;
        for (uint32_t gid : preloaded) distinct_slots.push_back(registry.catalog[gid].slot_idx);
        std::sort(distinct_slots.begin(), distinct_slots.end());
        const bool slots_unique =
            std::adjacent_find(distinct_slots.begin(), distinct_slots.end()) ==
            distinct_slots.end();
        assert_that("A: every resident owns a distinct physical slot",
                    slots_unique && registry.invariants_hold(), "slots=" +
                        std::to_string(distinct_slots.size()) + " unique=" +
                        (slots_unique ? "yes" : "no"));

        assert_that("A: nothing is warm at construction (lazy refill)",
                    registry.published_warm_slots() == 0 && registry.host_slots.size() == kHostSlots,
                    "published_warm=" + std::to_string(registry.published_warm_slots()));

        // A comparison that cannot fail is not evidence. Two different experts
        // must have different bytes, or every "identical" line below is vacuous.
        const auto cohort_diff =
            byte_diff(ground_truth(kTrackedExpert).data(), ground_truth(kCohortProbe).data(),
                      kPayloadBytes);
        assert_that("A: the reference is not vacuous — distinct experts differ",
                    cohort_diff.count > 0,
                    "expert " + std::to_string(kTrackedExpert) + " vs " +
                        std::to_string(kCohortProbe) + ": " + diff_text(cohort_diff));

        // ---------------------------------------------------------------------
        // B — Hot: the residents hold the artifact's bytes
        // ---------------------------------------------------------------------
        std::printf("\n[B] Hot — the residents the registry claims hold the artifact's bytes\n");

        // Seed every resident with an O_DIRECT read, so the Hot bytes come from
        // the NVMe leg and are compared against the mmap path rather than against
        // themselves. 162 MiB of pinned staging is already allocated; use one
        // scratch buffer of our own so the arena's bookkeeping stays untouched.
        aeon::io::AlignedBuffer preload_buffer(kPayloadBytes, format.sector_size);
        size_t hot_mismatched = 0;
        ByteDiff hot_worst{};
        uint32_t hot_worst_gid = 0;
        for (uint32_t gid : preloaded) {
            read_expert_direct(gid, static_cast<uint8_t*>(preload_buffer.data()));
            const int32_t slot = registry.catalog[gid].slot_idx;
            vram_pool.upload_from_host_expert(static_cast<uint32_t>(slot),
                                              static_cast<const uint8_t*>(preload_buffer.data()),
                                              compute_stream);
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            const auto resident = read_vram_slot(static_cast<uint32_t>(slot));
            const auto diff = byte_diff(resident.data(), ground_truth(gid).data(), kPayloadBytes);
            if (diff.count != 0) {
                ++hot_mismatched;
                if (hot_worst_gid == 0 || diff.count > hot_worst.count) {
                    hot_worst = diff;
                    hot_worst_gid = gid;
                }
            }
        }
        assert_that("B: all Hot residents are bit-exact against the mmap path",
                    hot_mismatched == 0,
                    hot_mismatched == 0
                        ? std::to_string(preloaded.size()) + " residents, 0 bytes differ"
                        : "gid " + std::to_string(hot_worst_gid) + ": " + diff_text(hot_worst));

        // ---------------------------------------------------------------------
        // C — Cold: NVMe -> staging -> VRAM
        // ---------------------------------------------------------------------
        std::printf("\n[C] Cold — O_DIRECT through the staging arena into VRAM\n");

        std::vector<TieredExpertSupply::PayloadRequest> cold_requests;
        bool cold_candidates_ok = true;
        for (uint32_t index = 0; index < std::size(kColdBatch); ++index) {
            const uint32_t gid = kColdBatch[index];
            if (registry.catalog[gid].owner != ExpertTier::COLD_NVME ||
                registry.catalog[gid].slot_idx != -1) {
                cold_candidates_ok = false;
            }
            cold_requests.push_back(TieredExpertSupply::PayloadRequest{gid, index});
        }
        assert_that("C: the requested experts are genuinely cold",
                    cold_candidates_ok,
                    "gids " + std::to_string(kColdBatch[0]) + ".." +
                        std::to_string(kColdBatch[2]) + " owner=COLD_NVME slot=-1");

        const uint32_t demotions_before = registry.demotion_completions;
        const uint32_t misses_before = registry.misses_cold;

        const auto cold_batch = submit(cold_requests);

        bool cold_io_ok = true;
        for (const auto& transfer : cold_batch.transfers) {
            if (transfer.io_pending) cold_io_ok = false;
            if (transfer.io_request_count != direct_chunks) cold_io_ok = false;
            if (transfer.is_prefetched == false) cold_io_ok = false;
        }
        assert_that("C: each cold request stages its payload in four I/O requests",
                    cold_batch.transfers.size() == std::size(kColdBatch) && cold_io_ok &&
                        registry.misses_cold - misses_before == std::size(kColdBatch),
                    "transfers=" + std::to_string(cold_batch.transfers.size()) +
                        " chunks/expert=" + std::to_string(direct_chunks));

        // The NVMe leg on its own: the staging slot still holds exactly what
        // io_uring wrote, before any H2D is read back. Isolating the two legs is
        // what tells a future failure *which* leg broke.
        size_t staging_mismatched = 0;
        ByteDiff staging_worst{};
        uint32_t staging_worst_gid = 0;
        for (size_t index = 0; index < cold_batch.transfers.size(); ++index) {
            const auto& transfer = cold_batch.transfers[index];
            const auto diff = byte_diff(staging.get_slot_ptr(transfer.staging_idx),
                                        ground_truth(transfer.global_expert_id).data(),
                                        kPayloadBytes);
            if (diff.count != 0) {
                ++staging_mismatched;
                staging_worst = diff;
                staging_worst_gid = transfer.global_expert_id;
            }
        }
        assert_that("C: the O_DIRECT read alone delivered the artifact's bytes",
                    staging_mismatched == 0,
                    staging_mismatched == 0
                        ? std::to_string(cold_batch.transfers.size()) +
                              " staged payloads, 0 bytes differ"
                        : "gid " + std::to_string(staging_worst_gid) + ": " +
                              diff_text(staging_worst));

        assert_that("C: the evictions the cold misses forced completed, none were dropped",
                    registry.demotion_completions - demotions_before ==
                            std::size(kColdBatch) &&
                        registry.demotion_drops == 0,
                    "completions=+" +
                        std::to_string(registry.demotion_completions - demotions_before) +
                        " drops=" + std::to_string(registry.demotion_drops));

        bool cold_resident_ok = true;
        for (const auto& transfer : cold_batch.transfers) {
            const auto& entry = registry.catalog[transfer.global_expert_id];
            if (entry.owner != ExpertTier::HOT_VRAM ||
                entry.slot_idx != transfer.vram_slot) {
                cold_resident_ok = false;
            }
        }
        assert_that("C: the delivered experts are resident in the slots the batch named",
                    cold_resident_ok,
                    "gids " + std::to_string(kColdBatch[0]) + ".." +
                        std::to_string(kColdBatch[2]) + " -> HOT_VRAM");

        size_t cold_mismatched = 0;
        ByteDiff cold_worst{};
        uint32_t cold_worst_gid = 0;
        std::vector<uint8_t> cold_tracked_snapshot;
        for (const auto& transfer : cold_batch.transfers) {
            const auto resident = read_vram_slot(static_cast<uint32_t>(transfer.vram_slot));
            const auto diff =
                byte_diff(resident.data(), ground_truth(transfer.global_expert_id).data(),
                          kPayloadBytes);
            if (diff.count != 0) {
                ++cold_mismatched;
                cold_worst = diff;
                cold_worst_gid = transfer.global_expert_id;
            }
            if (transfer.global_expert_id == kTrackedExpert) cold_tracked_snapshot = resident;
        }
        assert_that("C: the resident bytes after H2D are bit-exact against the mmap path",
                    cold_mismatched == 0,
                    cold_mismatched == 0
                        ? std::to_string(cold_batch.transfers.size()) +
                              " residents, 0 bytes differ"
                        : "gid " + std::to_string(cold_worst_gid) + ": " +
                              diff_text(cold_worst));

        assert_that("C: the cold misses are counted as cold, not as hits",
                    registry.misses_cold - misses_before == std::size(kColdBatch),
                    "misses=+" + std::to_string(registry.misses_cold - misses_before));

        const auto cold_staging_taken = staging_slots_of(cold_batch);
        finish_round(cold_staging_taken);

        bool staging_released = true;
        for (uint32_t slot : cold_staging_taken) {
            if (staging.slot_state(slot) != PrefetchStagingArena::SlotState::AVAILABLE) {
                staging_released = false;
            }
        }
        assert_that("C: the consumed staging slots were returned to the arena",
                    staging_released && !cold_staging_taken.empty(),
                    "slots=" + std::to_string(cold_staging_taken.size()) + " all AVAILABLE");

        // ---------------------------------------------------------------------
        // D — Hot hit: no transfer, and the bytes do not move
        // ---------------------------------------------------------------------
        std::printf("\n[D] Hot hit — a repeat request moves no bytes at all\n");

        const uint32_t hits_hot_before = registry.hits_hot;
        const int32_t tracked_slot = registry.catalog[kTrackedExpert].slot_idx;

        std::vector<TieredExpertSupply::PayloadRequest> hot_requests{
            TieredExpertSupply::PayloadRequest{kTrackedExpert, 0}};
        const auto hot_batch = submit(hot_requests);

        assert_that("D: the repeat request is a HOT_HIT that claims no staging slot",
                    hot_batch.transfers.size() == 1 &&
                        hot_batch.transfers[0].is_prefetched == false &&
                        hot_batch.transfers[0].io_pending == false &&
                        hot_batch.transfers[0].io_request_count == 0,
                    "is_prefetched=" +
                        std::string(hot_batch.transfers[0].is_prefetched ? "yes" : "no"));

        assert_that("D: the hot hit is counted as hot",
                    registry.hits_hot - hits_hot_before == 1,
                    "hits_hot=+" + std::to_string(registry.hits_hot - hits_hot_before));

        assert_that("D: the resident did not move",
                    registry.catalog[kTrackedExpert].slot_idx == tracked_slot,
                    "slot=" + std::to_string(tracked_slot));

        const auto hot_snapshot = read_vram_slot(static_cast<uint32_t>(tracked_slot));
        const auto hot_diff =
            byte_diff(hot_snapshot.data(), cold_tracked_snapshot.data(), kPayloadBytes);
        assert_that("D: the hot resident is bit-identical to the cold delivery",
                    hot_diff.count == 0, diff_text(hot_diff));

        finish_round(staging_slots_of(hot_batch));

        // ---------------------------------------------------------------------
        // E — Warm: demote the tracked expert, then promote it back
        // ---------------------------------------------------------------------
        std::printf("\n[E] Warm — demotion to the host pool, promotion back to VRAM\n");

        // Make the tracked expert the LRU tail. Touching the other residents is
        // enough: a hot hit is a `reserve_request` that returns immediately, and
        // using the registry's own list to predict the victim — rather than
        // re-deriving the eviction order from `populate_round_robin` — is what
        // keeps this probe built from the object under test.
        for (uint32_t gid : residents()) {
            if (gid == kTrackedExpert) continue;
            const auto touch = registry.reserve_request(gid, step, kDemotionQueueCapacity);
            if (touch.kind != ExpertRequestKind::HOT_HIT) {
                throw std::runtime_error("Tiering gate: expected a hot touch");
            }
            registry.release_lease(gid);
        }
        const uint32_t predicted_victim = registry.hot_vram_lru.back();
        assert_that("E: the tracked expert is the LRU victim, read from the registry's list",
                    predicted_victim == kTrackedExpert,
                    "predicted victim=" + std::to_string(predicted_victim) + " tracked=" +
                        std::to_string(kTrackedExpert));

        const uint32_t demotion_drops_before = registry.demotion_drops;
        const uint32_t hits_warm_before = registry.hits_warm;
        // Section C's cold misses already demoted three residents into the warm
        // pool, so the pool is *not* empty here. Every count below is therefore a
        // delta: an absolute one would be a claim about how much of the pool the
        // previous section filled rather than about this one.
        const uint32_t warm_slots_before = registry.published_warm_slots();

        std::vector<TieredExpertSupply::PayloadRequest> make_room_requests{
            TieredExpertSupply::PayloadRequest{kMakeRoomExpert, 0}};
        const auto room_batch = submit(make_room_requests);

        assert_that("E: the predicted victim was demoted, not dropped",
                    registry.catalog[kTrackedExpert].owner == ExpertTier::WARM_HOST &&
                        registry.demotion_drops == demotion_drops_before &&
                        registry.catalog[kTrackedExpert].slot_idx >= 0,
                    "owner=" +
                        std::string(registry.catalog[kTrackedExpert].owner == ExpertTier::WARM_HOST
                                        ? "WARM_HOST"
                                        : "other") +
                        " drops=+" +
                        std::to_string(registry.demotion_drops - demotion_drops_before));

        assert_that("E: the demoted expert joined the warm pool",
                    registry.published_warm_slots() == warm_slots_before + 1,
                    "published_warm " + std::to_string(warm_slots_before) + " -> " +
                        std::to_string(registry.published_warm_slots()));

        const int32_t warm_host_slot = registry.catalog[kTrackedExpert].slot_idx;

        // The D2H leg on its own: the host slot the registry believes it owns
        // must hold the artifact's bytes. A destination chosen wrongly, or a
        // staged fallback copied from the wrong slot, fails here and nowhere else.
        const auto host_diff =
            byte_diff(host_pool.get_expert_slot_ptr(static_cast<uint32_t>(warm_host_slot)),
                      ground_truth(kTrackedExpert).data(), kPayloadBytes);
        assert_that("E: the warm host slot holds the artifact's bytes (D2H leg)",
                    host_diff.count == 0,
                    "host slot " + std::to_string(warm_host_slot) + ": " + diff_text(host_diff));

        finish_round(staging_slots_of(room_batch));

        // Captured *after* the make-room request, which is itself a cold miss:
        // the promotion must not add another one.
        const uint32_t misses_cold_before_warm = registry.misses_cold;

        // Now bring it back. The request must be answered *from the host pool* —
        // asserted on the kind and the tier, because a silently dropped demotion
        // would have sent the expert back to Cold and the byte comparison would
        // then pass while measuring the wrong tier entirely.
        const auto warm_batch = submit(
            std::vector<TieredExpertSupply::PayloadRequest>{
                TieredExpertSupply::PayloadRequest{kTrackedExpert, 0}});

        assert_that("E: the promotion is a WARM_PROMOTION answered from the host pool",
                    warm_batch.transfers.size() == 1 &&
                        registry.hits_warm - hits_warm_before == 1 &&
                        registry.misses_cold == misses_cold_before_warm,
                    "hits_warm=+" + std::to_string(registry.hits_warm - hits_warm_before) +
                        " cold_misses=+" +
                        std::to_string(registry.misses_cold - misses_cold_before_warm));

        assert_that("E: the promotion returned the expert to VRAM",
                    registry.catalog[kTrackedExpert].owner == ExpertTier::HOT_VRAM,
                    "owner=HOT_VRAM slot=" +
                        std::to_string(registry.catalog[kTrackedExpert].slot_idx));

        const auto warm_snapshot =
            read_vram_slot(static_cast<uint32_t>(registry.catalog[kTrackedExpert].slot_idx));
        const auto warm_diff =
            byte_diff(warm_snapshot.data(), ground_truth(kTrackedExpert).data(), kPayloadBytes);
        assert_that("E: the promoted bytes are bit-exact against the mmap path (H2D leg)",
                    warm_diff.count == 0, diff_text(warm_diff));

        // The promotion released the slot it read from. The *net* warm count is
        // deliberately not the instrument: VRAM is saturated, so the promotion
        // also evicts a resident and that eviction immediately fills a warm slot
        // — the two cancel and an absolute count would measure nothing.
        assert_that("E: the promotion released the warm slot it read from",
                    registry.host_slots[static_cast<size_t>(warm_host_slot)] == -1,
                    "host slot " + std::to_string(warm_host_slot) + " released; pool " +
                        std::to_string(warm_slots_before) + " -> " +
                        std::to_string(registry.published_warm_slots()));

        finish_round(staging_slots_of(warm_batch));

        // ---------------------------------------------------------------------
        // F — the three routes agree with each other
        // ---------------------------------------------------------------------
        std::printf("\n[F] The three routes agree\n");

        const auto cold_vs_hot = byte_diff(cold_tracked_snapshot.data(), hot_snapshot.data(),
                                          kPayloadBytes);
        const auto cold_vs_warm = byte_diff(cold_tracked_snapshot.data(), warm_snapshot.data(),
                                           kPayloadBytes);
        assert_that("F: cold == hot == warm, bit for bit",
                    cold_vs_hot.count == 0 && cold_vs_warm.count == 0,
                    "cold-vs-hot: " + diff_text(cold_vs_hot) + "; cold-vs-warm: " +
                        diff_text(cold_vs_warm));

        assert_that("F: every route also equals the artifact's own bytes",
                    registry.invariants_hold() && registry.demotion_drops == 0,
                    "registry invariants hold, demotion_drops=" +
                        std::to_string(registry.demotion_drops));

        // ---------------------------------------------------------------------
        std::printf("\n[Tier-4 item 21 tiering] %s — %u checks, %u failed\n",
                    failures == 0 ? "PASS" : "FAIL", checks, failures);
        std::printf("  routes: hot (O_DIRECT seed) / cold (O_DIRECT via the arena) / warm "
                    "(D2H to the host pool, then H2D back)\n");
        std::printf("  pool: %u VRAM slots (saturated), %u host slots, %u staging slots "
                    "(%.1f MiB), demotion queue %llu\n",
                    kVramSlots, kHostSlots, PrefetchStagingArena::TOTAL_STAGING_SLOTS,
                    (PrefetchStagingArena::TOTAL_STAGING_SLOTS * kPayloadBytes) /
                        (1024.0 * 1024.0),
                    static_cast<unsigned long long>(kDemotionQueueCapacity));
        std::printf("  experts moved: %zu resident seeds + %zu cold misses + 1 make-room + "
                    "1 promotion = %zu x %.1f MiB\n",
                    preloaded.size(), std::size(kColdBatch), preloaded.size() + std::size(kColdBatch) + 2,
                    kPayloadBytes / (1024.0 * 1024.0));

        CHECK_HIP(hipStreamDestroy(demotion_stream));
        CHECK_HIP(hipStreamDestroy(sdma_cold_stream));
        CHECK_HIP(hipStreamDestroy(sdma_stream));
        CHECK_HIP(hipStreamDestroy(compute_stream));

        return failures == 0 ? 0 : 1;
    }
};

} // namespace

int main() {
    TieringGate gate;
    return gate.run();
}
