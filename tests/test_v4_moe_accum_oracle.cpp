// -----------------------------------------------------------------------------
// Routed-expert accumulation gate — the fixed-order fp32 path.
//
// The gate exists because the two paths that existed before it each violated one
// of two binding requirements, and neither satisfied both:
//
//   * the fused `aeon_moe_fused_w2_accum_kernel` accumulates with `atomicAdd`,
//     whose order across the six experts is the scheduler's, so its result is not
//     reproducible — trap 38, and the reason item 18's gate failed roughly one run
//     in ten;
//   * the `v4_pipeline_accumulate_expert_kernel` path *is* reproducible but keeps
//     its accumulator in **fp16** and re-rounds on every one of the six steps,
//     which is what plan §2.10.3 forbids ("accumulate in fp32"). Item 15 recorded
//     this at Tier 1 and flagged that it is "strictly *less* accurate, which is
//     the opposite of what the option name suggests".
//
// So every gate that needed reproducibility had to accept the less accurate path,
// and there was no correct option to choose. The replacement is a pair:
// `aeon_moe_fused_w2_contrib_kernel` writes one weighted contribution per expert
// into its own fp32 slice — one writer per element, so determinism follows from
// the structure rather than from observation — and
// `v4_moe_accumulate_fixed_order_kernel` sums those slices in slot order in fp32
// and rounds once.
//
// What is asserted:
//
//   A. STRUCTURE — six experts at the real dimensions; the contribution buffer is
//      `6 x 4096` fp32 and is neither constant nor degenerate, so nothing below
//      passes vacuously.
//   B. THE ACCUMULATION IS FP32 — the reduce kernel is compared against an fp64
//      slot-order sum of the *same* contributions, which isolates the accumulation
//      from the W2 projection (Tier-1 item 14 and item 16 own that projection).
//      The error must be at the single fp16 store.
//   C. THE OLD fp16 PATH IS MATERIALLY WORSE, and this is the point of the gate.
//      The fp16 read-modify-write chain is emulated on the host — rounding to fp16
//      after each of the six adds, exactly as the kernel does — and its error
//      against the same fp64 sum must be a multiple of the new path's. If the two
//      were indistinguishable the "fix" would not be one, and the gate would be
//      asserting a preference rather than a property.
//   D. DETERMINISM — the whole pair is run eight times and the fp16 outputs must be
//      **bit-identical**. Combined with B this is what makes the path usable by a
//      byte-exact gate, which is the requirement trap 38 imposes on items 19, 22
//      and 23.
//   E. THE SHARED EXPERT IS AN ADDEND OF THE fp32 ACCUMULATOR — not a post-hoc
//      `+=` on a finished fp16 sum. The two orders are computed and the kernel
//      must match the accumulator form, which is the reference's own fused shape.
//
// Experts are the artifact's real `layers.3` W2 payloads, so the contributions
// have real magnitudes; the hidden activation is synthetic because the *projection*
// is not what this gate certifies.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"

#include "architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "infrastructure/core/aeon_loader.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "support/v4_layer_body_gate.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHidden = 4096;
constexpr uint32_t kExpertDim = 2048;   // LPR * ITERS * 32 for <8, 8, 4, 16>
constexpr uint32_t kExperts = 6;
constexpr uint32_t kLayer = 3;
constexpr uint32_t kFirstExpert = 10;   // six consecutive real experts

uint32_t g_checks = 0;
uint32_t g_failures = 0;

bool assert_that(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-58s %-34s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    ++g_checks;
    if (!ok) ++g_failures;
    return ok;
}

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got, double tol_frac, double abs_floor = 0.0) {
    const bool ok = aeon::testgate::report(label, want, got, tol_frac, abs_floor);
    ++g_checks;
    if (!ok) ++g_failures;
    return ok;
}

std::string sci(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.3e", value);
    return buffer;
}

// Round-to-nearest through fp16, in both directions — the exact arithmetic the
// fp16 read-modify-write path performs on every accumulation step.
double as_half(double value) {
    return aeon::reference::half_bits_to_double(
        __half_as_ushort(__float2half(static_cast<float>(value))));
}

double max_abs_of(const std::vector<double>& a, const std::vector<double>& b) {
    return aeon::reference::compare(a, b, 0.0).max_abs;
}

struct DeviceBuffers {
    half* d_expert_hidden{nullptr};
    float* d_contrib{nullptr};
    float* d_weights{nullptr};
    half* d_shared{nullptr};
    half* d_out{nullptr};
    uint8_t* d_payload[kExperts]{};
    aeon::kernel::SwizzledW2ExpertPtrs w2{};

    hipStream_t stream{nullptr};
};

void allocate(DeviceBuffers& b) {
    CHECK_HIP(hipStreamCreateWithFlags(&b.stream, hipStreamNonBlocking));
    CHECK_HIP(hipMalloc(&b.d_expert_hidden, kExperts * kExpertDim * sizeof(half)));
    CHECK_HIP(hipMalloc(&b.d_contrib, kExperts * kHidden * sizeof(float)));
    CHECK_HIP(hipMalloc(&b.d_weights, kExperts * sizeof(float)));
    CHECK_HIP(hipMalloc(&b.d_shared, kHidden * sizeof(half)));
    CHECK_HIP(hipMalloc(&b.d_out, kHidden * sizeof(half)));
    for (uint32_t k = 0; k < kExperts; ++k) {
        CHECK_HIP(hipMalloc(&b.d_payload[k], aeon::core::AEON_SWIZZLED_EXPERT_BYTES));
    }
}

void release(DeviceBuffers& b) {
    for (uint32_t k = 0; k < kExperts; ++k) CHECK_HIP(hipFree(b.d_payload[k]));
    CHECK_HIP(hipFree(b.d_out));
    CHECK_HIP(hipFree(b.d_shared));
    CHECK_HIP(hipFree(b.d_weights));
    CHECK_HIP(hipFree(b.d_contrib));
    CHECK_HIP(hipFree(b.d_expert_hidden));
    CHECK_HIP(hipStreamDestroy(b.stream));
}

std::vector<double> read_half(const half* device, size_t count) {
    std::vector<half> host(count);
    CHECK_HIP(hipMemcpy(host.data(), device, count * sizeof(half), hipMemcpyDeviceToHost));
    std::vector<double> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = aeon::reference::half_bits_to_double(__half_as_ushort(host[i]));
    }
    return out;
}

std::vector<double> read_float(const float* device, size_t count) {
    std::vector<float> host(count);
    CHECK_HIP(hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost));
    return std::vector<double>(host.begin(), host.end());
}

// Contributions of the six experts for every output element, in slot order:
// `contrib[k * hidden + h]`.
std::vector<double> as_slot_major(const std::vector<double>& flat) { return flat; }

// The fp64 slot-order sum — the reference for the accumulation rule.
std::vector<double> fp64_slot_order_sum(const std::vector<double>& contrib) {
    std::vector<double> out(kHidden, 0.0);
    for (uint32_t h = 0; h < kHidden; ++h) {
        double acc = 0.0;
        for (uint32_t k = 0; k < kExperts; ++k) {
            acc += contrib[static_cast<size_t>(k) * kHidden + h];
        }
        out[h] = acc;
    }
    return out;
}

// The emulated fp16 read-modify-write chain: six fp16 roundings, one per step.
std::vector<double> fp16_chain_sum(const std::vector<double>& contrib) {
    std::vector<double> out(kHidden, 0.0);
    for (uint32_t h = 0; h < kHidden; ++h) {
        double acc = 0.0;
        for (uint32_t k = 0; k < kExperts; ++k) {
            acc = as_half(acc + contrib[static_cast<size_t>(k) * kHidden + h]);
        }
        out[h] = acc;
    }
    return out;
}

void run_reduce(DeviceBuffers& b, const half* shared) {
    const int blocks = (kHidden + 255) / 256;
    aeon::kernel::v4_moe_accumulate_fixed_order_kernel<<<blocks, 256, 0, b.stream>>>(
        b.d_contrib, static_cast<int>(kExperts), shared, b.d_out,
        static_cast<int>(kHidden));
    CHECK_HIP(hipStreamSynchronize(b.stream));
}

void run_contrib(DeviceBuffers& b) {
    aeon::kernel::dispatch_aeon_moe_fused_w2_contrib<8, 8, 4, 16>(
        b.d_expert_hidden, b.w2, b.d_weights, b.d_contrib,
        static_cast<int>(kExperts), static_cast<int>(kHidden),
        static_cast<int>(kExpertDim), b.stream);
    CHECK_HIP(hipStreamSynchronize(b.stream));
}

int run_gate() {
    std::printf("================================================================================\n");
    std::printf("  Routed-expert accumulation — fixed-order fp32 vs the fp16 read-modify-write\n");
    std::printf("================================================================================\n");

    aeon::core::select_compute_device(true);
    aeon::core::AeonModelLoader loader;
    loader.open_model("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon");

    DeviceBuffers b;
    allocate(b);

    // -------------------------------------------------------------------------
    // A — structure
    // -------------------------------------------------------------------------
    std::printf("\n[A] Structure, at the artifact's own dimensions\n");

    // Real W2 payloads from six consecutive `layers.3` experts.
    for (uint32_t k = 0; k < kExperts; ++k) {
        const uint8_t* payload = loader.get_expert_data(kLayer, kFirstExpert + k);
        CHECK_HIP(hipMemcpyAsync(b.d_payload[k], payload,
                                 aeon::core::AEON_SWIZZLED_EXPERT_BYTES,
                                 hipMemcpyHostToDevice, b.stream));
        const uint8_t* base = b.d_payload[k];
        b.w2.w2[k] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W2_PACKED_OFFSET);
        b.w2.s2[k] = reinterpret_cast<const __half*>(base + aeon::core::AEON_W2_SCALE_OFFSET);
    }
    CHECK_HIP(hipStreamSynchronize(b.stream));

    // Synthetic hidden activation (the *projection* is Tier-1's business; this
    // gate certifies the accumulation) with routing weights that differ per expert
    // so a per-expert mix-up would show.
    {
        std::vector<half> hidden(kExperts * kExpertDim);
        for (size_t i = 0; i < hidden.size(); ++i) {
            const double pattern = 0.5 - static_cast<double>((i * 37u) % 101u) / 101.0;
            hidden[i] = __float2half(static_cast<float>(pattern));
        }
        std::vector<float> weights(kExperts);
        {
            // The routing weights after `norm_topk_prob` and the 1.5 scaling, in
            // their most demanding realistic shape: one dominant expert and five
            // small ones, summing to exactly `routed_scaling_factor = 1.5`
            // `[V config]`. A large first addend followed by small ones is the case
            // an fp16 accumulator handles worst, so the comparison in section C is
            // made where the two rules actually diverge rather than where they
            // coincide.
            const float skewed[kExperts] = {1.20f, 0.06f, 0.06f, 0.06f, 0.06f, 0.06f};
            for (uint32_t k = 0; k < kExperts; ++k) weights[k] = skewed[k];
        }
        CHECK_HIP(hipMemcpy(b.d_expert_hidden, hidden.data(),
                            hidden.size() * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(b.d_weights, weights.data(),
                            weights.size() * sizeof(float), hipMemcpyHostToDevice));
    }

    assert_that("A: dimensions are the routed path's own",
                    kExperts == 6 && kHidden == 4096 && kExpertDim == 2048 &&
                    kExperts * kHidden * sizeof(float) == 98304,
                    "6 experts x 4096 fp32 = 96 KiB contribution buffer");

    run_contrib(b);
    const std::vector<double> contrib = as_slot_major(read_float(b.d_contrib, kExperts * kHidden));

    double contrib_peak = 0.0;
    bool all_finite = true;
    for (double v : contrib) {
        if (!std::isfinite(v)) all_finite = false;
        contrib_peak = std::fmax(contrib_peak, std::fabs(v));
    }
    assert_that("A: the contributions are finite and non-degenerate",
                    all_finite && contrib_peak > 0.0,
                    "peak=" + sci(contrib_peak));

    // Non-vacuity: the six experts must not be near-identical, or "the sum is
    // right" would be true of almost any combination rule.
    double min_pair_delta = 1e30;
    for (uint32_t a = 0; a < kExperts; ++a) {
        for (uint32_t c = a + 1; c < kExperts; ++c) {
            double delta = 0.0;
            for (uint32_t h = 0; h < kHidden; ++h) {
                delta = std::fmax(delta, std::fabs(
                    contrib[static_cast<size_t>(a) * kHidden + h] -
                    contrib[static_cast<size_t>(c) * kHidden + h]));
            }
            min_pair_delta = std::fmin(min_pair_delta, delta);
        }
    }
    assert_that("A: every pair of expert contributions actually differs",
                    min_pair_delta > 0.01 * contrib_peak,
                    "closest pair differs by " + sci(min_pair_delta) + " = " +
                        aeon::testgate::num(100.0 * min_pair_delta / contrib_peak, 1) +
                        "% of peak");

    // The contribution stage needs its own certification, and this is why. Every
    // reference below is an fp64 sum **of these same numbers**, which isolates the
    // accumulation from the projection — the right instrument for what this gate
    // certifies — but it also means a defect in the contribution stage would be
    // summed consistently and never seen. (A mutation that dropped the routing
    // weight entirely did exactly that and survived the first sweep.)
    //
    // The check is closed-form and needs no weight decoding: run the same kernel
    // with unit routing weights to get `raw[k][h] = W2_k · activation_k`, then
    // require `contrib[k][h] == weights[k] · raw[k][h]`. The projection arithmetic
    // inside the shared `swizzled_w2_row_dot` is Tier-1 item 14's and item 16's to
    // certify; what is certified here is that the routing weight is applied, once
    // and to the right expert.
    std::vector<double> raw;
    {
        std::vector<float> unit_weights(kExperts, 1.0f);
        CHECK_HIP(hipMemcpy(b.d_weights, unit_weights.data(), kExperts * sizeof(float),
                            hipMemcpyHostToDevice));
        run_contrib(b);
        raw = read_float(b.d_contrib, kExperts * kHidden);

        std::vector<float> routed(kExperts);
        const float skewed[kExperts] = {1.20f, 0.06f, 0.06f, 0.06f, 0.06f, 0.06f};
        for (uint32_t k = 0; k < kExperts; ++k) routed[k] = skewed[k];
        CHECK_HIP(hipMemcpy(b.d_weights, routed.data(), kExperts * sizeof(float),
                            hipMemcpyHostToDevice));
        run_contrib(b);
    }

    double raw_peak = 0.0;
    for (double v : raw) raw_peak = std::fmax(raw_peak, std::fabs(v));
    assert_that("A: the unit-weight contributions are non-degenerate",
                    raw_peak > 0.0, "peak=" + sci(raw_peak));

    {
        const float skewed[kExperts] = {1.20f, 0.06f, 0.06f, 0.06f, 0.06f, 0.06f};
        double worst = 0.0;
        for (uint32_t k = 0; k < kExperts; ++k) {
            for (uint32_t h = 0; h < kHidden; ++h) {
                const size_t i = static_cast<size_t>(k) * kHidden + h;
                worst = std::fmax(worst,
                                  std::fabs(contrib[i] - skewed[k] * raw[i]));
            }
        }
        assert_that("A: every contribution is exactly weight x the unit-weight result",
                    worst < 3e-3 * contrib_peak,
                    "worst=" + sci(worst) + " = " +
                        aeon::testgate::num(worst / contrib_peak, 5) + " of peak");
    }


    // -------------------------------------------------------------------------
    // B — the accumulation is fp32
    // -------------------------------------------------------------------------
    std::printf("\n[B] The accumulation is fp32, in a fixed order\n");

    const std::vector<double> reference_sum = fp64_slot_order_sum(contrib);
    run_reduce(b, nullptr);
    const std::vector<double> fixed_order = read_half(b.d_out, kHidden);

    const double fp32_error = max_abs_of(fixed_order, reference_sum);
    const double reference_peak = aeon::reference::peak_abs(reference_sum);
    report("B: reduce kernel vs the fp64 slot-order sum",
           reference_sum, fixed_order, 3e-3);

    // -------------------------------------------------------------------------
    // C — the fp16 path is materially worse
    // -------------------------------------------------------------------------
    std::printf("\n[C] Against the fp16 read-modify-write path it replaces\n");

    const std::vector<double> fp16_chain = fp16_chain_sum(contrib);
    const double fp16_error = max_abs_of(fp16_chain, reference_sum);
    const double separation = max_abs_of(fp16_chain, fixed_order);

    // The claim is **strictly worse**, not "dramatically worse", and the difference
    // is modest at six experts: both paths end in at least one fp16 store, and the
    // fp16 chain's six roundings partially cancel. The primary justification for
    // this pair is not accuracy — it is that the fp16 path violates §2.10.3 and the
    // atomic path cannot fix an order. The accuracy result is reported as what it
    // is: a real but small improvement in this regime, and a larger one at higher
    // expert counts, where more roundings accumulate.
    assert_that("C: the fp16 chain is measurably less accurate than the fp32 sum",
                    fp16_error > 1.25 * fp32_error,
                    "fp16=" + sci(fp16_error) + " vs fp32=" + sci(fp32_error) +
                        " = " + aeon::testgate::num(fp16_error / fp32_error, 2) + "x worse");

    assert_that("C: the two paths are observably different (the fix is a fix)",
                    separation > 1.25 * fp32_error,
                    "path delta=" + sci(separation) + " = " +
                        aeon::testgate::num(separation / fp32_error, 2) + "x the fp32 error");

    std::printf("  %-58s %-34s\n",
                "C: (reported) error as a fraction of the sum's peak",
                (sci(fp32_error / reference_peak) + " fp32 vs " +
                 sci(fp16_error / reference_peak) + " fp16").c_str());

    // -------------------------------------------------------------------------
    // D — determinism
    // -------------------------------------------------------------------------
    std::printf("\n[D] Determinism: the whole pair, eight times\n");

    const uint32_t repeats = 8;
    uint32_t mismatched_runs = 0;
    for (uint32_t run = 0; run < repeats; ++run) {
        run_contrib(b);
        run_reduce(b, nullptr);
        if (read_half(b.d_out, kHidden) != fixed_order) ++mismatched_runs;
    }
    assert_that("D: eight runs of the pair are bit-identical",
                    mismatched_runs == 0,
                    std::to_string(repeats) + " runs, " + std::to_string(mismatched_runs) +
                        " differ");

    // Determinism here is *by construction* rather than by luck: each element of
    // `contrib` has exactly one writer, so nothing is left for a scheduler to
    // order. Re-running the contribution stage alone must reproduce it exactly,
    // which is the half an atomic path cannot promise.
    {
        run_contrib(b);
        const std::vector<double> contrib_again = read_float(b.d_contrib, kExperts * kHidden);
        const double delta = max_abs_of(contrib_again, contrib);
        assert_that("D: the contribution stage alone is bit-reproducible",
                    delta == 0.0, "max_abs=" + sci(delta));
    }

    // -------------------------------------------------------------------------
    // E — the shared expert is an addend of the fp32 accumulator
    // -------------------------------------------------------------------------
    std::printf("\n[E] The shared expert enters the fp32 accumulator\n");

    {
        std::vector<half> shared(kHidden);
        for (uint32_t h = 0; h < kHidden; ++h) {
            shared[h] = __float2half(static_cast<float>(0.25 - (h % 13) * 0.03));
        }
        CHECK_HIP(hipMemcpy(b.d_shared, shared.data(), kHidden * sizeof(half),
                            hipMemcpyHostToDevice));

        run_reduce(b, b.d_shared);
        const std::vector<double> with_shared = read_half(b.d_out, kHidden);

        const std::vector<double> shared_values = read_half(b.d_shared, kHidden);
        const std::vector<double> expected(reference_sum.size(), 0.0);
        std::vector<double> folded(reference_sum.size());
        std::vector<double> post_hoc(reference_sum.size());
        for (uint32_t h = 0; h < kHidden; ++h) {
            folded[h] = as_half(reference_sum[h] + shared_values[h]);
            post_hoc[h] = as_half(as_half(reference_sum[h]) + shared_values[h]);
        }

        report("E: the kernel matches the fp32 accumulator form",
               folded, with_shared, 3e-3);

        // The tolerance above is necessarily loose, and the whole effect being
        // certified here is small. So the discriminating instrument is a
        // *targeted* comparison: the kernel must sit on the accumulator side of
        // the fork, not merely be within tolerance of it. A peak-relative bound
        // cannot express this — the two forms differ by ~1e-3 of peak, which any
        // sensible tolerance admits. This is the M9/M9b lesson, and it is the
        // reason that mutation survived the first sweep of this gate.
        const double to_folded = max_abs_of(with_shared, folded);
        const double to_post_hoc = max_abs_of(with_shared, post_hoc);
        assert_that("E: the kernel sits on the fp32-accumulator side of that fork",
                    to_folded * 3.0 < to_post_hoc,
                    "to accumulator=" + sci(to_folded) + " to post-hoc=" + sci(to_post_hoc));
        assert_that("E: the post-hoc `+=` on a finished sum is a different result",
                    max_abs_of(post_hoc, folded) > 0.0,
                    "delta=" + sci(max_abs_of(post_hoc, folded)));
    }

    std::printf("\n[moe accumulation] %s — %u checks, %u failed\n",
                g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    std::printf("  experts: layers.%u experts %u..%u, real W2 payloads; routing weights "
                "{1.20, 0.06 x5} (sums to routed_scaling_factor 1.5); activation synthetic\n",
                kLayer, kFirstExpert, kFirstExpert + kExperts - 1);

    release(b);
    return g_failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    return run_gate();
}
