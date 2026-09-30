// -----------------------------------------------------------------------------
// Gate: the grouped chunk orchestration vs the per-token GEMV path, on identical
// experts and identical routing.
//
// When the grouped batch was wired into `run_layer_body_chunk`, the chunk logits
// diverged from serial by `max_abs 2.7e-1` — far larger than a summation reorder
// produces. That is either deep-network amplification of the certified per-layer
// grouped-vs-GEMV delta, or a defect in the new batching (permutation, per-batch
// rebasing, or the reduce). This gate separates the two: it runs `moe_out` through
// **both** paths on the same experts, activation and routing, and compares them at the
// executor's own output, where a summation reorder is `~1e-5` and a bug is not.
//
// The kernels are each oracle-certified (`test_v4_grouped_wmma_oracle`,
// `test_v4_expert_oracle`), so this is a **parity** check between two independent
// orchestrations, not a third oracle. Its job is to say whether the batching itself
// is correct, which is the one thing the kernel gates cannot.
//
// The fixture exercises the parts the wiring added: a chunk wider than one expert
// batch (more than `kAeonSwizzledMaxExperts` distinct experts, so the batching loop
// runs more than once), a strided activation (each token in its own 16-row tile, as
// the layer body stores it), an expert with several draws sharing one slab, and
// experts that receive no draws at all.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"
#include "architecture/deepseek_v4/moe/moe_grouped_batch.hpp"
#include "architecture/deepseek_v4/kernels/moe_gemv_dispatch.hpp"
#include "platform/ops/moe_accumulate.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
namespace core = aeon::core;

constexpr int kHidden = 4096;
constexpr int kIntermediate = 2048;
constexpr int kSlots = 6;
constexpr int kExperts = 32;      // local experts the permutation ranges over
constexpr int kTokens = 8;        // the chunk
constexpr int kMPad = 16;         // rows per token in the layer body's tile
constexpr float kLimit = 10.0f;

// The experts that actually receive draws. They span three `kAeonSwizzledMaxExperts`
// batches with gaps, so the batching loop sees empty experts between drawn ones — the
// layout the real 256-expert layer has, and which a dense fixture would not exercise.
constexpr int kDrawn[6] = {0, 1, 8, 9, 16, 17};

// Valid payload content: nibbles are free, scales are small positive powers of two so
// the sums stay far from fp16 range limits.
void fill_payload(uint8_t* payload, int seed) {
    std::memset(payload, 0, core::AEON_SWIZZLED_EXPERT_BYTES);
    for (size_t i = 0; i < core::AEON_SWIZZLED_EXPERT_BYTES; ++i) {
        payload[i] = static_cast<uint8_t>((i * 37 + static_cast<size_t>(seed) * 11) & 0xFF);
    }
    const size_t scale_offsets[3] = {core::AEON_W1_SCALE_OFFSET, core::AEON_W2_SCALE_OFFSET,
                                     core::AEON_W3_SCALE_OFFSET};
    const size_t scale_bytes[3] = {core::AEON_W1_SCALE_BYTES, core::AEON_W2_SCALE_BYTES,
                                   core::AEON_W3_SCALE_BYTES};
    for (int region = 0; region < 3; ++region) {
        const int entries = static_cast<int>(scale_bytes[region] / sizeof(uint16_t));
        for (int i = 0; i < entries; ++i) {
            const int exponent = -6 + (i % 4);
            const uint16_t bits = static_cast<uint16_t>((exponent + 15) << 10);
            std::memcpy(payload + scale_offsets[region] + static_cast<size_t>(i) * 2, &bits,
                        sizeof(bits));
        }
    }
}

void fill_w13(const uint8_t* base, int slot, kernel::SwizzledW13ExpertPtrs& table) {
    table.w1[slot] = reinterpret_cast<const uint4*>(base + core::AEON_W1_PACKED_OFFSET);
    table.s1[slot] = reinterpret_cast<const half*>(base + core::AEON_W1_SCALE_OFFSET);
    table.w3[slot] = reinterpret_cast<const uint4*>(base + core::AEON_W3_PACKED_OFFSET);
    table.s3[slot] = reinterpret_cast<const half*>(base + core::AEON_W3_SCALE_OFFSET);
}

void fill_w2(const uint8_t* base, int slot, kernel::SwizzledW2ExpertPtrs& table) {
    table.w2[slot] = reinterpret_cast<const uint4*>(base + core::AEON_W2_PACKED_OFFSET);
    table.s2[slot] = reinterpret_cast<const half*>(base + core::AEON_W2_SCALE_OFFSET);
}

struct SplitMix {
    uint64_t state;
    uint64_t next() {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

} // namespace

int main() {
    std::cout << "[Gate] grouped chunk orchestration vs the per-token GEMV path\n";
    core::select_compute_device(true);
    SplitMix rng{0xC0FFEEull};

    // Routing: six distinct drawn experts per token, all from `kDrawn`, so experts
    // outside that set get no draws and sit between drawn ones in the batching order.
    std::vector<int> ids(static_cast<size_t>(kTokens) * kSlots);
    std::vector<float> weights(static_cast<size_t>(kTokens) * kSlots);
    for (int t = 0; t < kTokens; ++t) {
        for (int k = 0; k < kSlots; ++k) {
            ids[static_cast<size_t>(t) * kSlots + k] = kDrawn[(k + t) % kSlots];
            weights[static_cast<size_t>(t) * kSlots + k] =
                0.6f / static_cast<float>(k + 1) + 0.02f;
        }
    }

    // Host payloads and their device copies, for the drawn experts only. Expert e is
    // payload e, so both arms read identical weights by construction; an expert with
    // no draws is never read, so its payload is not allocated.
    std::vector<uint8_t*> d_payload(kExperts, nullptr);
    std::vector<uint8_t> payload(core::AEON_SWIZZLED_EXPERT_BYTES);
    for (int i = 0; i < kSlots; ++i) {
        const int e = kDrawn[i];
        fill_payload(payload.data(), e);
        CHECK_HIP(hipMalloc(&d_payload[e], core::AEON_SWIZZLED_EXPERT_BYTES));
        CHECK_HIP(hipMemcpy(d_payload[e], payload.data(), core::AEON_SWIZZLED_EXPERT_BYTES,
                            hipMemcpyHostToDevice));
    }

    // Activation: each token in its own 16-row tile, as the layer body stores it, so
    // the grouped arm is read with a row stride rather than a compact batch.
    const int stride = kMPad * kHidden;
    std::vector<half> host_activation(static_cast<size_t>(kTokens) * stride);
    for (int t = 0; t < kTokens; ++t) {
        for (int k = 0; k < kHidden; ++k) {
            host_activation[static_cast<size_t>(t) * stride + k] =
                __float2half(static_cast<float>(rng.uniform() - 0.5) * 1.0f);
        }
        for (int pad = kHidden; pad < stride; ++pad) {
            host_activation[static_cast<size_t>(t) * stride + pad] = __float2half(0.0f);
        }
    }

    half* d_activation = nullptr;
    int* d_ids = nullptr;
    float* d_weights = nullptr;
    CHECK_HIP(hipMalloc(&d_activation, host_activation.size() * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_ids, ids.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_weights, weights.size() * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_activation, host_activation.data(),
                        host_activation.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_weights, weights.data(), weights.size() * sizeof(float),
                        hipMemcpyHostToDevice));

    // ---- Grouped arm: the production orchestration ------------------------
    constexpr int kHiddenTokens = 16;   // an expert takes at most `kTokens` draws
    core::MoeGroupedBatchScratch grouped_scratch{};
    grouped_scratch.intermediate = kIntermediate;
    grouped_scratch.hidden = kHidden;
    grouped_scratch.hidden_tokens = kHiddenTokens;
    CHECK_HIP(hipMalloc(&grouped_scratch.perm_offsets, (kExperts + 1) * sizeof(int)));
    CHECK_HIP(hipMalloc(&grouped_scratch.perm_tokens, ids.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&grouped_scratch.perm_draws, ids.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&grouped_scratch.perm_weights, weights.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&grouped_scratch.batch_offsets,
                        (kernel::kAeonSwizzledMaxExperts + 1) * sizeof(int)));
    CHECK_HIP(hipMalloc(&grouped_scratch.chunk_hidden,
                        static_cast<size_t>(kernel::kAeonSwizzledMaxExperts) * kHiddenTokens *
                            kIntermediate * sizeof(half)));
    CHECK_HIP(hipMalloc(&grouped_scratch.chunk_contrib,
                        static_cast<size_t>(kTokens) * kSlots * kHidden * sizeof(float)));
    half* d_grouped_out = nullptr;
    CHECK_HIP(hipMalloc(&d_grouped_out, static_cast<size_t>(kTokens) * stride * sizeof(half)));
    CHECK_HIP(hipMemset(d_grouped_out, 0,
                        static_cast<size_t>(kTokens) * stride * sizeof(half)));

    core::run_moe_grouped_expert_batch(
        d_activation, stride, d_ids, d_weights, kTokens, kSlots, kExperts, kLimit,
        grouped_scratch,
        [&](int expert, int j, int count, kernel::SwizzledW13ExpertPtrs& w13,
            kernel::SwizzledW2ExpertPtrs& w2) {
            if (count == 0) return;
            fill_w13(d_payload[expert], j, w13);
            fill_w2(d_payload[expert], j, w2);
        },
        d_grouped_out, stride);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    // ---- Per-token arm: the certified GEMV path ---------------------------
    half* d_expert_hidden = nullptr;
    float* d_contrib = nullptr;
    half* d_gemv_out = nullptr;
    CHECK_HIP(hipMalloc(&d_expert_hidden,
                        static_cast<size_t>(kernel::kAeonSwizzledMaxExperts) * kIntermediate *
                            sizeof(half)));
    CHECK_HIP(hipMalloc(&d_contrib, static_cast<size_t>(kSlots) * kHidden * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_gemv_out, static_cast<size_t>(kTokens) * stride * sizeof(half)));
    CHECK_HIP(hipMemset(d_gemv_out, 0, static_cast<size_t>(kTokens) * stride * sizeof(half)));

    for (int t = 0; t < kTokens; ++t) {
        kernel::SwizzledW13ExpertPtrs w13{};
        kernel::SwizzledW2ExpertPtrs w2{};
        for (int k = 0; k < kSlots; ++k) {
            const int expert = ids[static_cast<size_t>(t) * kSlots + k];
            fill_w13(d_payload[expert], k, w13);
            fill_w2(d_payload[expert], k, w2);
        }
        const half* token_activation =
            d_activation + static_cast<size_t>(t) * stride;
        const float* token_weights =
            d_weights + static_cast<size_t>(t) * kSlots;
        half* token_out = d_gemv_out + static_cast<size_t>(t) * stride;
        kernel::dispatch_dsv4_moe_gemv_w13_swiglu<8, 4, 8, 16>(
            token_activation, w13, d_expert_hidden, nullptr, kIntermediate, kSlots,
            kIntermediate, kHidden, kLimit);
        kernel::dispatch_aeon_moe_fused_w2_contrib<8, 8, 4, 16>(
            d_expert_hidden, w2, token_weights, d_contrib, kSlots, kHidden, kIntermediate);
        constexpr int kThreads = 256;
        kernel::moe_accumulate_fixed_order_kernel
            <<<(kHidden + kThreads - 1) / kThreads, kThreads>>>(
                d_contrib, kSlots, nullptr, token_out, kHidden);
    }
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    // ---- Compare ----------------------------------------------------------
    std::vector<half> grouped_out(static_cast<size_t>(kTokens) * stride);
    std::vector<half> gemv_out(static_cast<size_t>(kTokens) * stride);
    CHECK_HIP(hipMemcpy(grouped_out.data(), d_grouped_out,
                        grouped_out.size() * sizeof(half), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(gemv_out.data(), d_gemv_out, gemv_out.size() * sizeof(half),
                        hipMemcpyDeviceToHost));

    double max_abs = 0.0;
    double max_rel = 0.0;
    double peak_gemv = 0.0;
    double peak_grouped = 0.0;
    size_t differing = 0;
    for (int t = 0; t < kTokens; ++t) {
        for (int i = 0; i < kHidden; ++i) {
            const double a = __half2float(
                gemv_out[static_cast<size_t>(t) * stride + i]);
            const double b = __half2float(
                grouped_out[static_cast<size_t>(t) * stride + i]);
            const double abs = std::fabs(a - b);
            const double rel = abs / std::max(1e-3, std::fabs(a));
            max_abs = std::max(max_abs, abs);
            max_rel = std::max(max_rel, rel);
            peak_gemv = std::max(peak_gemv, std::fabs(a));
            peak_grouped = std::max(peak_grouped, std::fabs(b));
            if (a != b) ++differing;
        }
    }
    const size_t total = static_cast<size_t>(kTokens) * kHidden;
    std::printf("  %-46s max_abs=%.3e max_rel=%.3e, %zu/%zu differ\n",
                "grouped batch vs per-token GEMV", max_abs, max_rel, differing, total);
    // If either arm is ~0 the comparison is vacuous, which a bit-identical pass would
    // otherwise hide.
    std::printf("  %-46s peak |gemv|=%.3e peak |grouped|=%.3e\n", "signal present (non-vacuous)",
                peak_gemv, peak_grouped);

    const bool non_vacuous = peak_gemv > 1e-3 && peak_grouped > 1e-3;
    const bool ok = non_vacuous && max_rel <= 1e-3;
    std::printf("  %-46s %s\n", "non-vacuous and within a reorder band", ok ? "PASS" : "FAIL");

    for (int i = 0; i < kSlots; ++i) CHECK_HIP(hipFree(d_payload[kDrawn[i]]));
    CHECK_HIP(hipFree(d_activation));
    CHECK_HIP(hipFree(d_ids));
    CHECK_HIP(hipFree(d_weights));
    CHECK_HIP(hipFree(grouped_scratch.perm_offsets));
    CHECK_HIP(hipFree(grouped_scratch.perm_tokens));
    CHECK_HIP(hipFree(grouped_scratch.perm_draws));
    CHECK_HIP(hipFree(grouped_scratch.perm_weights));
    CHECK_HIP(hipFree(grouped_scratch.batch_offsets));
    CHECK_HIP(hipFree(grouped_scratch.chunk_hidden));
    CHECK_HIP(hipFree(grouped_scratch.chunk_contrib));
    CHECK_HIP(hipFree(d_grouped_out));
    CHECK_HIP(hipFree(d_expert_hidden));
    CHECK_HIP(hipFree(d_contrib));
    CHECK_HIP(hipFree(d_gemv_out));

    if (!ok) {
        std::cout << "[FAIL] the grouped batch orchestration is not a reorder of the GEMV path\n";
        return 1;
    }
    std::cout << "[PASS] the grouped batch matches the per-token GEMV path\n";
    return 0;
}
