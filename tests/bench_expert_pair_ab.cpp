// -----------------------------------------------------------------------------
// Expert-pair A/B: the single-token GEMV pair versus the grouped WMMA pair.
//
// ## What question this answers
//
// Step 1 of the kernel plan replaces the per-token expert GEMVs with a grouped
// WMMA GEMM on the theory that it is "the largest gain". That theory has a
// premise — that a chunk's tokens-per-expert is high enough for the M dimension to
// amortise the weight read — and the premise is not obvious: with 256 routed
// experts and 6 slots, a chunk of a few hundred tokens is a few hundred draws over
// a few hundred distinct experts, i.e. a handful of tokens each. So the plan's
// Step 6 asks for the chunk size to be chosen so the mean reaches 16, and this
// file measures whether it does and what that buys.
//
// ## The two quantities, and why they are not the same measurement
//
// **Weight bytes** is the quantity that matters in production, and it is computed
// analytically from the routing rather than timed. It has to be: in the runtime an
// expert arrives from Warm or NVMe, and the point of the tiering is that the read
// happens once per layer per window. A benchmark holding every payload resident
// measures the *other* thing.
//
// Three byte figures are printed per configuration, and the spread between them is
// the point:
//
//   * **GEMV** — one expert read per draw.
//   * **grouped, as implemented** — one expert read per M tile the expert spans. The
//     kernel gives each workgroup one (N tile, M tile) pair and loops K inside it, so
//     an expert holding 80 tokens is re-dequantized six times.
//   * **grouped, if the slab were reused across token tiles** — one expert read per
//     distinct expert, which is what the plan's Step 1 asks for.
//
// Reporting only the third would overstate the kernel; reporting only the second
// would hide the target. Both are shown, so the remaining work is a number rather
// than a claim.
//
// **Elapsed time** is the other half — principally ALU and issue efficiency — but it
// is not worthless, because the pool here is deliberately far larger than the 96 MiB
// Infinity Cache (128 experts, ~1.7 GiB), so both arms stream their weights from
// DRAM and the GEMV arm's re-reads are not free either. Neither number alone is the
// answer; reported together they bracket it.
//
// ## Correctness is checked, not assumed
//
// Both arms write the same `[draw, hidden]` fp32 contributions, with `draw =
// token * 6 + slot`, so they are compared elementwise at every configuration.
// Without that this would be a race between two programs that might not compute
// the same function — the usual way a speed comparison misleads.
//
// ## The routing is real
//
// Draws are sampled from the measured per-layer expert distribution in a routing
// profile, which is what makes the tokens-per-expert column meaningful. A uniform
// draw over 256 experts would flatten exactly the quantity under test. Expert ids
// are mapped into the resident pool by modulo; that changes which payload is read,
// never how many bytes, so the byte figures are exact and the kernel shapes are
// untouched.
//
// It does not assert. The numbers are the deliverable; the pair's numerics are
// `test_v4_grouped_wmma_oracle`'s subject.
//
// Usage: bench_expert_pair_ab [chunk_sizes...] [--profile <counts.csv>]
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_grouped_wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_HIP(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

namespace {

namespace kernel = aeon::kernel;

constexpr int kHidden = 4096;
constexpr int kIntermediate = 2048;
constexpr int kSlots = 6;
constexpr int kMaxExpertsPerDispatch = kernel::kAeonSwizzledMaxExperts;
constexpr float kLimit = 10.0f;

// Resident expert pool. Sized well past the 96 MiB Infinity Cache so both arms
// stream from DRAM; see the header.
constexpr int kPoolExperts = 128;

constexpr int kGemvWaves = 8;
constexpr int kGemvW13Rpw = 4, kGemvW13Lpr = 8;
constexpr int kGemvW2Rpw = 8, kGemvW2Lpr = 4;
constexpr int kGemvIterations = 16;

constexpr int kGroupedWaves = 4;
constexpr int kGroupedW13Rpw = 4, kGroupedW13Lpr = 8;
constexpr int kGroupedW2Rpw = 8, kGroupedW2Lpr = 4;
// M windows swept: 1, 2, 4 and 8 token tiles, i.e. 16 to 128 tokens per slab.
constexpr int kMTileSweep[] = {1, 2, 4, 8};

constexpr int kM = 16;

// Timed passes per arm. The reported figure is the best of these; see `time_arm`.
constexpr int kRepetitions = 3;

// A permutation of `tokens * 6` draws, expert-contiguous.
//
// `expert_offsets[e] .. expert_offsets[e+1]` bounds expert `e`'s draws;
// `token_indices` names the activation row each draw reads, and `draw_indices` the
// output row it writes. The two are different numbers by construction, so a kernel
// that used one where the other belongs cannot pass unnoticed.
struct Permutation {
    std::vector<int> expert_offsets;
    std::vector<int> token_indices;
    std::vector<int> draw_indices;
    std::vector<float> draw_weights;

    int draws() const { return static_cast<int>(token_indices.size()); }
    int distinct() const { return static_cast<int>(expert_offsets.size()) - 1; }
    int widest_expert() const {
        int widest = 0;
        for (int e = 0; e < distinct(); ++e) {
            widest = std::max(widest, expert_offsets[static_cast<size_t>(e + 1)] -
                                          expert_offsets[static_cast<size_t>(e)]);
        }
        return widest;
    }

    // How many times the grouped kernel actually reads a whole expert.
    //
    // The kernel walks K inside a fixed (N tile, M window) workgroup, so an expert
    // spanning `ceil(count / (16 * window_tiles))` windows has its weight slab
    // dequantized that many times. A window large enough to hold the expert makes it
    // one read, which is the target; the plan's per-K-tile reuse is exactly this
    // quantity reaching `distinct`. This is what keeps the ideal figure from
    // standing in for the measured one.
    long expert_reads(int window_tiles) const {
        const int window = 16 * window_tiles;
        long reads = 0;
        for (int e = 0; e < distinct(); ++e) {
            const int count = expert_offsets[static_cast<size_t>(e + 1)] -
                              expert_offsets[static_cast<size_t>(e)];
            reads += (count + window - 1) / window;
        }
        return reads;
    }
};

struct SplitMix {
    uint64_t state;
    uint64_t next() {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform01() {
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    }
};

// Reads a measured `counts.csv` and returns one layer's prefill distribution.
std::vector<double> load_layer_distribution(const std::string& path, int layer_id) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open routing profile: " + path);
    }

    std::vector<double> counts(256, 0.0);
    std::string line;
    bool header = true;
    while (std::getline(file, line)) {
        if (header) { header = false; continue; }
        std::stringstream row(line);
        std::string phase, layer, expert, count;
        if (!std::getline(row, phase, ',')) continue;
        if (!std::getline(row, layer, ',')) continue;
        if (!std::getline(row, expert, ',')) continue;
        if (!std::getline(row, count, ',')) continue;
        if (phase != "prefill" || std::stoi(layer) != layer_id) continue;
        counts[static_cast<size_t>(std::stoi(expert))] = std::stod(count);
    }

    const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
    if (total <= 0.0) {
        throw std::runtime_error("routing profile has no prefill counts for that layer");
    }
    for (double& value : counts) value /= total;
    return counts;
}

Permutation build_permutation(const std::vector<double>& distribution, int token_count,
                              SplitMix& rng) {
    std::vector<double> cumulative(distribution.size());
    std::partial_sum(distribution.begin(), distribution.end(), cumulative.begin());

    struct Draw {
        int expert;
        int token;
        int draw;
        float weight;
    };
    std::vector<Draw> draws;
    draws.reserve(static_cast<size_t>(token_count) * kSlots);

    for (int token = 0; token < token_count; ++token) {
        // A token's slots are distinct experts, as the router produces; duplicates
        // would inflate the distinct-expert byte count this benchmark is built on.
        int chosen[kSlots];
        int chosen_count = 0;
        while (chosen_count < kSlots) {
            const double target = rng.uniform01();
            const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), target);
            const int expert = static_cast<int>(
                std::min<size_t>(static_cast<size_t>(it - cumulative.begin()),
                                 distribution.size() - 1));
            bool duplicate = false;
            for (int i = 0; i < chosen_count; ++i) duplicate |= (chosen[i] == expert);
            if (!duplicate) chosen[chosen_count++] = expert;
        }
        for (int slot = 0; slot < kSlots; ++slot) {
            // Roughly what a trained router produces: one dominant slot, a tail.
            draws.push_back({chosen[slot], token, token * kSlots + slot,
                             0.55f / static_cast<float>(slot + 1) + 0.03f});
        }
    }

    std::stable_sort(draws.begin(), draws.end(),
                     [](const Draw& a, const Draw& b) { return a.expert < b.expert; });

    Permutation permutation;
    permutation.expert_offsets.push_back(0);
    for (size_t i = 0; i < draws.size(); ++i) {
        permutation.token_indices.push_back(draws[i].token);
        permutation.draw_indices.push_back(draws[i].draw);
        permutation.draw_weights.push_back(draws[i].weight);
        if (i + 1 < draws.size() && draws[i + 1].expert != draws[i].expert) {
            permutation.expert_offsets.push_back(static_cast<int>(i + 1));
        }
    }
    permutation.expert_offsets.push_back(static_cast<int>(draws.size()));
    return permutation;
}

struct Device {
    half* activation = nullptr;
    half* expert_hidden = nullptr;
    float* contrib_gemv = nullptr;
    float* contrib_grouped = nullptr;
    float* slot_weights = nullptr;   // [tokens * slots], by draw index
    std::vector<uint8_t*> payloads;
    std::vector<uint8_t*> host_payloads;

    ~Device() {
        if (activation) (void)hipFree(activation);
        if (expert_hidden) (void)hipFree(expert_hidden);
        if (contrib_gemv) (void)hipFree(contrib_gemv);
        if (contrib_grouped) (void)hipFree(contrib_grouped);
        if (slot_weights) (void)hipFree(slot_weights);
        for (uint8_t* p : payloads) (void)hipFree(p);
    }
};

// Both arms read expert `e`'s payload from the same pool slot, so a divergence in
// the comparison below cannot come from the arms having been handed different
// weights.
const uint8_t* payload_of(const Device& device, int expert) {
    return device.payloads[static_cast<size_t>(expert) % kPoolExperts];
}

void fill_w13_table(const Device& device, int first, int count,
                    kernel::SwizzledW13ExpertPtrs& table) {
    table = kernel::SwizzledW13ExpertPtrs{};
    for (int j = 0; j < count; ++j) {
        const uint8_t* base = payload_of(device, first + j);
        table.w1[j] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W1_PACKED_OFFSET);
        table.s1[j] = reinterpret_cast<const half*>(base + aeon::core::AEON_W1_SCALE_OFFSET);
        table.w3[j] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W3_PACKED_OFFSET);
        table.s3[j] = reinterpret_cast<const half*>(base + aeon::core::AEON_W3_SCALE_OFFSET);
    }
}

void fill_w2_table(const Device& device, int first, int count,
                   kernel::SwizzledW2ExpertPtrs& table) {
    table = kernel::SwizzledW2ExpertPtrs{};
    for (int j = 0; j < count; ++j) {
        const uint8_t* base = payload_of(device, first + j);
        table.w2[j] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W2_PACKED_OFFSET);
        table.s2[j] = reinterpret_cast<const half*>(base + aeon::core::AEON_W2_SCALE_OFFSET);
    }
}

// Which expert a permuted position belongs to. The permutation is
// expert-contiguous, so this is a walk of the offset array.
int expert_at(const Permutation& permutation, int position) {
    int expert = 0;
    while (permutation.expert_offsets[static_cast<size_t>(expert + 1)] <= position) ++expert;
    return expert;
}

// The GEMV pair over the chunk: per token, its six slot experts, dispatched
// exactly as `V4TieredExpertExecutor::accumulate_routed` does today.
void run_gemv_arm(const Device& device, const Permutation& permutation, int token_count) {
    std::vector<int> expert_of_draw(static_cast<size_t>(permutation.draws()), 0);
    std::vector<size_t> position_of_draw(static_cast<size_t>(permutation.draws()), 0);
    for (int position = 0; position < permutation.draws(); ++position) {
        const size_t draw = static_cast<size_t>(
            permutation.draw_indices[static_cast<size_t>(position)]);
        expert_of_draw[draw] = expert_at(permutation, position);
        position_of_draw[draw] = static_cast<size_t>(position);
    }

    for (int token = 0; token < token_count; ++token) {
        kernel::SwizzledW13ExpertPtrs w13{};
        kernel::SwizzledW2ExpertPtrs w2{};
        for (int slot = 0; slot < kSlots; ++slot) {
            const size_t draw = static_cast<size_t>(token) * kSlots + static_cast<size_t>(slot);
            const uint8_t* base = payload_of(device, expert_of_draw[draw]);
            w13.w1[slot] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W1_PACKED_OFFSET);
            w13.s1[slot] = reinterpret_cast<const half*>(base + aeon::core::AEON_W1_SCALE_OFFSET);
            w13.w3[slot] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W3_PACKED_OFFSET);
            w13.s3[slot] = reinterpret_cast<const half*>(base + aeon::core::AEON_W3_SCALE_OFFSET);
            w2.w2[slot] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W2_PACKED_OFFSET);
            w2.s2[slot] = reinterpret_cast<const half*>(base + aeon::core::AEON_W2_SCALE_OFFSET);
        }

        const half* token_activation =
            device.activation + static_cast<size_t>(token) * kHidden;
        float* token_contrib =
            device.contrib_gemv + static_cast<size_t>(token) * kSlots * kHidden;
        // The GEMV pair takes its routing weights as a device pointer, exactly as
        // the executor hands them over — a host array here would fault.
        const float* token_weights =
            device.slot_weights + static_cast<size_t>(token) * kSlots;

        kernel::dispatch_aeon_moe_fused_w13_swiglu<kGemvWaves, kGemvW13Rpw, kGemvW13Lpr,
                                                   kGemvIterations>(
            token_activation, w13, device.expert_hidden, nullptr, kHidden, kSlots,
            kIntermediate, kHidden, kLimit);
        kernel::dispatch_aeon_moe_fused_w2_contrib<kGemvWaves, kGemvW2Rpw, kGemvW2Lpr,
                                                   kGemvIterations>(
            device.expert_hidden, w2, token_weights, token_contrib, kSlots, kHidden,
            kIntermediate);
    }
}

// The grouped pair: distinct experts in batches of at most the kernel's maximum,
// one launch pair per batch. Templated on the M window so the same source measures
// several window sizes; the kernel is the same code in each case.
template <int MTILES>
void run_grouped_arm(const Device& device, const Permutation& permutation,
                     int expert_hidden_tokens) {
    const int distinct = permutation.distinct();

    for (int first = 0; first < distinct; first += kMaxExpertsPerDispatch) {
        const int count = std::min(kMaxExpertsPerDispatch, distinct - first);
        const int begin = permutation.expert_offsets[static_cast<size_t>(first)];
        const int end = permutation.expert_offsets[static_cast<size_t>(first + count)];

        kernel::SwizzledW13ExpertPtrs w13{};
        kernel::SwizzledW2ExpertPtrs w2{};
        fill_w13_table(device, first, count, w13);
        fill_w2_table(device, first, count, w2);

        // Rebased so expert 0 of the launch is the batch's first expert.
        std::vector<int> local_offsets(
            permutation.expert_offsets.begin() + first,
            permutation.expert_offsets.begin() + first + count + 1);
        for (int& value : local_offsets) value -= begin;

        const int batch_draws = end - begin;
        int* d_offsets = nullptr;
        int* d_tokens = nullptr;
        int* d_draws = nullptr;
        float* d_weights = nullptr;
        CHECK_HIP(hipMalloc(&d_offsets, local_offsets.size() * sizeof(int)));
        CHECK_HIP(hipMalloc(&d_tokens, batch_draws * sizeof(int)));
        CHECK_HIP(hipMalloc(&d_draws, batch_draws * sizeof(int)));
        CHECK_HIP(hipMalloc(&d_weights, batch_draws * sizeof(float)));
        CHECK_HIP(hipMemcpy(d_offsets, local_offsets.data(), local_offsets.size() * sizeof(int),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_tokens, permutation.token_indices.data() + begin,
                            batch_draws * sizeof(int), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_draws, permutation.draw_indices.data() + begin,
                            batch_draws * sizeof(int), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_weights, permutation.draw_weights.data() + begin,
                            batch_draws * sizeof(float), hipMemcpyHostToDevice));

        kernel::dispatch_aeon_moe_grouped_w13_swiglu_wmma<kGroupedWaves, kGroupedW13Rpw,
                                                          kGroupedW13Lpr, MTILES>(
            device.activation, d_offsets, d_tokens, w13, device.expert_hidden, count,
            expert_hidden_tokens, kIntermediate, kHidden, kLimit);
        kernel::dispatch_aeon_moe_grouped_w2_wmma<kGroupedWaves, kGroupedW2Rpw,
                                                  kGroupedW2Lpr, MTILES>(
            device.expert_hidden, d_offsets, d_draws, d_weights, w2,
            device.contrib_grouped, count, expert_hidden_tokens, kHidden, kIntermediate);

        CHECK_HIP(hipFree(d_offsets));
        CHECK_HIP(hipFree(d_tokens));
        CHECK_HIP(hipFree(d_draws));
        CHECK_HIP(hipFree(d_weights));
    }
}

// Fills one expert payload with *valid* content.
//
// This matters more than it looks. The packed and scale regions are separate, and
// arbitrary bytes in a scale region are arbitrary fp16 — including infinities and
// NaNs. A payload like that makes both arms return garbage whose difference is
// dominated by how each handles a non-finite operand, so the A/B would compare two
// implementations of undefined behaviour. Nibbles are free (any 4-bit pattern is a
// legal value in `[-8, 7]`), so only the scales need care: they are written as
// small positive powers of two, which are exact in fp16.
void fill_payload(uint8_t* payload, int expert_seed) {
    std::memset(payload, 0, aeon::core::AEON_SWIZZLED_EXPERT_BYTES);

    // Nibbles: a spread pattern, so no path is short-circuited by a zero operand.
    for (size_t i = 0; i < aeon::core::AEON_SWIZZLED_EXPERT_BYTES; ++i) {
        payload[i] = static_cast<uint8_t>((i * 31 + static_cast<size_t>(expert_seed) * 7) & 0xFF);
    }

    // Scales: overwrite each scale region with valid fp16 values. Exponent -11 to
    // -8 keeps the dequantized weights small enough that the 4096-term sums stay far
    // from fp16 range limits, so the comparison below measures the kernels'
    // arithmetic rather than saturation.
    const size_t scale_offsets[3] = {aeon::core::AEON_W1_SCALE_OFFSET,
                                     aeon::core::AEON_W2_SCALE_OFFSET,
                                     aeon::core::AEON_W3_SCALE_OFFSET};
    const size_t scale_bytes[3] = {aeon::core::AEON_W1_SCALE_BYTES,
                                   aeon::core::AEON_W2_SCALE_BYTES,
                                   aeon::core::AEON_W3_SCALE_BYTES};
    for (int region = 0; region < 3; ++region) {
        const int entries = static_cast<int>(scale_bytes[region] / sizeof(uint16_t));
        for (int i = 0; i < entries; ++i) {
            // 2^exponent for exponent in [-11, -8], cycling.
            const int exponent = -11 + (i % 4);
            const uint16_t bits = static_cast<uint16_t>((exponent + 15) << 10);
            std::memcpy(payload + scale_offsets[region] + static_cast<size_t>(i) * 2, &bits,
                        sizeof(bits));
        }
    }
}

// Times one arm over `repetitions` passes and returns the best, in ms. Best rather
// than mean: the interference here is host-side noise (other processes, clock
// ramping), which only ever adds, so the minimum is the closest estimate of what
// the kernel costs.
template <class Pass>
float time_arm(hipEvent_t start, hipEvent_t stop, int repetitions, Pass pass) {
    float best = 1e30f;
    for (int i = 0; i < repetitions; ++i) {
        CHECK_HIP(hipEventRecord(start));
        pass();
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));
        float elapsed = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&elapsed, start, stop));
        best = std::min(best, elapsed);
    }
    return best;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> chunk_sizes;
    std::string profile = "routing-profile/first-real-prompt/counts.csv";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile" && i + 1 < argc) {
            profile = argv[++i];
        } else {
            chunk_sizes.push_back(std::stoi(arg));
        }
    }
    // Positional sizes replace the default sweep rather than extending it: the
    // permutation depends on the RNG's position, so a run that silently measured
    // eight sizes instead of four would report different routing for the same T as
    // a run that measured four.
    if (chunk_sizes.empty()) {
        chunk_sizes = {16, 64, 256, 1024};
    }

    std::cout << "[Bench] Expert pair A/B: GEMV pair vs grouped WMMA pair\n";
    aeon::core::select_compute_device(true);

    const std::vector<double> distribution = load_layer_distribution(profile, 0);
    std::printf("  routing: %s (layer 0, prefill)\n", profile.c_str());

    const int max_tokens = *std::max_element(chunk_sizes.begin(), chunk_sizes.end());

    Device device;
    for (int e = 0; e < kPoolExperts; ++e) {
        auto* host = new uint8_t[aeon::core::AEON_SWIZZLED_EXPERT_BYTES];
        fill_payload(host, e);
        device.host_payloads.push_back(host);
        uint8_t* dev = nullptr;
        CHECK_HIP(hipMalloc(&dev, aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
        CHECK_HIP(hipMemcpy(dev, host, aeon::core::AEON_SWIZZLED_EXPERT_BYTES,
                            hipMemcpyHostToDevice));
        device.payloads.push_back(dev);
    }

    std::vector<half> host_activation(static_cast<size_t>(max_tokens) * kHidden);
    for (size_t i = 0; i < host_activation.size(); ++i) {
        host_activation[i] = __float2half(
            static_cast<float>(static_cast<int>(i % 17) - 8) * 0.0625f);
    }
    CHECK_HIP(hipMalloc(&device.activation, host_activation.size() * sizeof(half)));
    CHECK_HIP(hipMemcpy(device.activation, host_activation.data(),
                        host_activation.size() * sizeof(half), hipMemcpyHostToDevice));

    const size_t contrib_elements = static_cast<size_t>(max_tokens) * kSlots * kHidden;
    CHECK_HIP(hipMalloc(&device.contrib_gemv, contrib_elements * sizeof(float)));
    CHECK_HIP(hipMalloc(&device.contrib_grouped, contrib_elements * sizeof(float)));
    CHECK_HIP(hipMalloc(&device.slot_weights,
                        static_cast<size_t>(max_tokens) * kSlots * sizeof(float)));

    const double expert_bytes = static_cast<double>(aeon::core::AEON_SWIZZLED_EXPERT_BYTES);
    const double mib = 1024.0 * 1024.0;
    std::printf("  expert payload: %.2f MiB; resident pool %d experts = %.2f GiB\n",
                expert_bytes / mib, kPoolExperts,
                expert_bytes * kPoolExperts / (mib * 1024.0));

    std::printf("\n  %6s %8s %8s %9s %10s %10s %9s %9s %9s %7s\n", "T", "draws",
                "distinct", "tok/expt", "gemv ms", "group ms", "gemv GiB", "grp GiB",
                "speedup", "agrees");
    std::printf("  %s\n", std::string(100, '-').c_str());

    SplitMix rng{0x5EED1234ull};
    for (int token_count : chunk_sizes) {
        Permutation permutation = build_permutation(distribution, token_count, rng);
        const int draws = permutation.draws();
        const int distinct = permutation.distinct();

        // Routing weights by draw index, so the GEMV arm reads them the way the
        // executor does. Filled once per chunk, outside the timed region.
        std::vector<float> draw_weights_by_draw(static_cast<size_t>(draws));
        for (int position = 0; position < draws; ++position) {
            draw_weights_by_draw[static_cast<size_t>(
                permutation.draw_indices[static_cast<size_t>(position)])] =
                permutation.draw_weights[static_cast<size_t>(position)];
        }
        CHECK_HIP(hipMemcpy(device.slot_weights, draw_weights_by_draw.data(),
                            draw_weights_by_draw.size() * sizeof(float),
                            hipMemcpyHostToDevice));

        // Rows per expert block: the widest group rounded up to the M tile, with a
        // floor that also covers the GEMV arm's per-token slot layout.
        const int rows_per_expert =
            std::max(kMaxExpertsPerDispatch * kIntermediate,
                     ((permutation.widest_expert() + kM - 1) / kM) * kM);
        if (device.expert_hidden) {
            CHECK_HIP(hipFree(device.expert_hidden));
            device.expert_hidden = nullptr;
        }
        // Two layouts share the buffer, and they size differently: the GEMV arm
        // indexes `[slot, row]` with `row < kIntermediate`, while the grouped arm
        // indexes `[expert, token, intermediate]`. The floor covers the former and
        // the token count covers the latter.
        const size_t hidden_rows = std::max<size_t>(
            kMaxExpertsPerDispatch * kIntermediate,
            static_cast<size_t>(kMaxExpertsPerDispatch) * rows_per_expert * kIntermediate);
        CHECK_HIP(hipMalloc(&device.expert_hidden, hidden_rows * sizeof(half)));

        // One warm-up pass per arm, so neither pays alone for a cold allocator.
        run_gemv_arm(device, permutation, token_count);
        run_grouped_arm<4>(device, permutation, rows_per_expert);
        CHECK_HIP(hipDeviceSynchronize());

        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        CHECK_HIP(hipEventCreate(&start));
        CHECK_HIP(hipEventCreate(&stop));

        CHECK_HIP(hipMemset(device.contrib_gemv, 0, contrib_elements * sizeof(float)));
        const float gemv_ms = time_arm(start, stop, kRepetitions, [&] {
            run_gemv_arm(device, permutation, token_count);
        });

        // The baseline arm's output is the reference every window is checked against.
        const size_t live = static_cast<size_t>(draws) * kHidden;
        std::vector<float> host_gemv(live);
        CHECK_HIP(hipMemcpy(host_gemv.data(), device.contrib_gemv, live * sizeof(float),
                            hipMemcpyDeviceToHost));
        double scale = 0.0;
        for (float value : host_gemv) {
            scale = std::max(scale, std::abs(static_cast<double>(value)));
        }

        const double gemv_bytes = static_cast<double>(draws) * expert_bytes;

        std::vector<float> host_grouped(live);
        for (int window_tiles : kMTileSweep) {
            CHECK_HIP(hipMemset(device.contrib_grouped, 0, contrib_elements * sizeof(float)));
            float grouped_ms = 0.0f;
            switch (window_tiles) {
                case 1: grouped_ms = time_arm(start, stop, kRepetitions, [&] {
                            run_grouped_arm<1>(device, permutation, rows_per_expert); }); break;
                case 2: grouped_ms = time_arm(start, stop, kRepetitions, [&] {
                            run_grouped_arm<2>(device, permutation, rows_per_expert); }); break;
                case 4: grouped_ms = time_arm(start, stop, kRepetitions, [&] {
                            run_grouped_arm<4>(device, permutation, rows_per_expert); }); break;
                case 8: grouped_ms = time_arm(start, stop, kRepetitions, [&] {
                            run_grouped_arm<8>(device, permutation, rows_per_expert); }); break;
                default: throw std::logic_error("uninstantiated M window");
            }

            CHECK_HIP(hipMemcpy(host_grouped.data(), device.contrib_grouped,
                                live * sizeof(float), hipMemcpyDeviceToHost));
            double max_delta = 0.0;
            for (size_t i = 0; i < live; ++i) {
                max_delta = std::max(max_delta,
                                     std::abs(static_cast<double>(host_gemv[i]) -
                                              static_cast<double>(host_grouped[i])));
            }
            const bool agrees = max_delta <= 1e-3 * std::max(scale, 1.0);

            // The grouped arm reads each expert once per M window it spans; a window
            // large enough to hold the expert makes it one read, which is the target
            // the plan's per-K-tile reuse names.
            const double grouped_bytes =
                static_cast<double>(permutation.expert_reads(window_tiles)) * expert_bytes;
            const double ideal_bytes = static_cast<double>(distinct) * expert_bytes;

            std::printf("  %6d %8d %8d %9.1f %10.3f %10.3f %9.2f %9.2f %8.2fx %7s\n",
                        token_count, draws, distinct,
                        static_cast<double>(draws) / static_cast<double>(distinct), gemv_ms,
                        grouped_ms, gemv_bytes / (mib * 1024.0),
                        grouped_bytes / (mib * 1024.0), gemv_ms / grouped_ms,
                        agrees ? "yes" : "NO");
            std::printf("         window %d tiles (%d tokens): traffic %5.2fx vs GEMV,"
                        " %5.2fx vs the one-read target\n",
                        window_tiles, window_tiles * kM, gemv_bytes / grouped_bytes,
                        ideal_bytes / grouped_bytes);
            if (!agrees) {
                std::printf("      max |gemv - grouped| = %.3e (scale %.3e)\n", max_delta,
                            scale);
            }
        }

        CHECK_HIP(hipEventDestroy(start));
        CHECK_HIP(hipEventDestroy(stop));
    }

    for (uint8_t* host : device.host_payloads) delete[] host;
    std::cout << "\n  bytes are analytic weight traffic; times are DRAM-resident kernels\n";
    return 0;
}
