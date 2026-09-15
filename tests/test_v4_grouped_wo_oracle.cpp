// -----------------------------------------------------------------------------
// Tier-1 gate: grouped output projection (Step 2.5) — versus an independent
// fp64 reference.
//
// The graph is
//
//     attn_out [T, 64, 512]  --view-->  o [T, 8, 4096]
//     z = einsum("tgd,grd->tgr", o, wo_a)      -> [T, 8192]
//     attn_proj = wo_b @ z                     -> [T, 4096]
//
// `wo_a` is the checkpoint tensor stored flat `[8192, 4096]`, which is a view of
// `[G, R, D]` = `[8, 1024, 4096]` [V sglang models/deepseek_v4.py:3346]. The
// whole point of the gate is that this contraction is **per group**: group g
// reduces the 4096-wide slice belonging to its own 8 contiguous heads against
// its own `[1024, 4096]` weight block, and nothing else.
//
// That structure is what a flat implementation would destroy, and a flat
// implementation still produces output of plausible magnitude. So the gate does
// not settle for a closeness check:
//
//   * it computes the **interleaved** weight reading as well and shows the
//     difference is large (if the two readings agreed, the layout would not be
//     load-bearing and the pass would mean nothing);
//   * it asserts **group isolation** directly — perturbing group 0 must leave
//     groups 1..7 bit-identical, which a shared/global treatment cannot satisfy;
//   * it asserts the reduction reaches the whole 4096-wide block, by perturbing
//     the last element of group 0 and requiring group 0 to move.
//
// Then `wo_b` is certified both in isolation (fed the kernel's own `z`) and as
// part of the end-to-end chain.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
#include "architecture/deepseek_v4/kernels/v4_gemv.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
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

using aeon::reference::ErrorStats;

constexpr size_t kTokens   = 2;
constexpr size_t kHeads    = 64;
constexpr size_t kHeadDim  = 512;
constexpr size_t kGroups   = 8;
constexpr size_t kRank     = 1024;
constexpr size_t kGroupDim = (kHeads / kGroups) * kHeadDim; // 4096
constexpr size_t kO        = kHeads * kHeadDim;             // 32768
constexpr size_t kZ        = kGroups * kRank;               // 8192
constexpr size_t kHidden   = 4096;

// fp16 inputs, one 4096-term dot accumulated in fp32 against an fp64 oracle.
// Stated once, as a fraction of the vector's own peak magnitude.
constexpr double kTol = 3e-3;

std::vector<double> widen(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(__half2float(v[i]));
    return out;
}

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got, double tol = kTol) {
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-52s SIZE MISMATCH                 FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(s.max_rel) && s.max_rel <= tol;
    std::printf("  %-52s max_abs=%.3e max_rel=%.3e  %s\n",
                label, s.max_abs, s.max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-52s %-24s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

// Exact element-wise agreement, used where the expected answer is "bit-identical"
// (a group that must not have been touched at all).
bool bit_identical(const std::vector<__half>& a, const std::vector<__half>& b,
                   size_t from, size_t count, std::string& detail) {
    size_t diff = 0;
    for (size_t i = from; i < from + count; ++i) {
        if (__half2float(a[i]) != __half2float(b[i])) ++diff;
    }
    detail = diff == 0 ? "bit-identical"
                       : (std::to_string(diff) + " of " + std::to_string(count) + " differ");
    return diff == 0;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: grouped output projection (wo_a + wo_b)\n";
    aeon::core::select_compute_device(true);

    bool ok = true;

    aeon::reference::Rng gen(0x5A0A0DEDull);

    // --- Inputs -------------------------------------------------------------
    // `o` is a token-major [T, 64, 512] attention-output buffer, which is also
    // its [T, 8, 4096] grouped view — the same memory, no copy.
    const size_t o_count = kTokens * kO;
    const size_t w_count = kZ * kGroupDim;   // wo_a [8192, 4096]
    const size_t b_count = kHidden * kZ;     // wo_b [4096, 8192]

    std::vector<__half> h_o(o_count);
    for (size_t i = 0; i < o_count; ++i) {
        h_o[i] = __float2half(static_cast<float>(gen.symmetric(1.0)));
    }
    // Weights at a smaller scale: the group dot is 4096 terms, so unit-scale
    // weights would push the output to ~64 and waste fp16 headroom.
    std::vector<__half> h_w(w_count);
    for (size_t i = 0; i < w_count; ++i) {
        h_w[i] = __float2half(static_cast<float>(gen.symmetric(0.02)));
    }
    std::vector<__half> h_b(b_count);
    for (size_t i = 0; i < b_count; ++i) {
        h_b[i] = __float2half(static_cast<float>(gen.symmetric(0.02)));
    }

    const std::vector<double> o = widen(h_o);

    auto w_at = [&](size_t i) { return static_cast<double>(__half2float(h_w[i])); };
    auto b_at = [&](size_t i) { return static_cast<double>(__half2float(h_b[i])); };
    auto b_at2 = [&](size_t out_i, size_t in_i) {
        return static_cast<double>(__half2float(h_b[out_i * kZ + in_i]));
    };

    // --- Device -------------------------------------------------------------
    __half* d_o = nullptr;
    __half* d_w = nullptr;
    __half* d_b = nullptr;
    __half* d_z = nullptr;
    __half* d_y = nullptr;
    CHECK_HIP(hipMalloc(&d_o, o_count * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_w, w_count * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_b, b_count * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_z, kTokens * kZ * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_y, kTokens * kHidden * sizeof(__half)));
    CHECK_HIP(hipMemcpy(d_w, h_w.data(), w_count * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.data(), b_count * sizeof(__half), hipMemcpyHostToDevice));

    // Runs the whole chain on the device from the current `d_o` and returns
    // (z, attn_proj) as host halves.
    auto run_chain = [&](std::vector<__half>& z_out, std::vector<__half>& y_out) {
        aeon::kernel::v4_grouped_wo_a_wave32_kernel<<<
            dim3(static_cast<unsigned>(kRank), static_cast<unsigned>(kGroups),
                 static_cast<unsigned>(kTokens)),
            dim3(32)>>>(d_o, d_w, d_z, static_cast<int>(kTokens));
        CHECK_HIP(hipGetLastError());

        aeon::kernel::v4_gemv_fp16_kernel<<<
            dim3(static_cast<unsigned>(kHidden), static_cast<unsigned>(kTokens)),
            dim3(32)>>>(d_z, d_b, d_y, static_cast<int>(kZ));
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());

        z_out.assign(kTokens * kZ, __half{});
        y_out.assign(kTokens * kHidden, __half{});
        CHECK_HIP(hipMemcpy(z_out.data(), d_z, z_out.size() * sizeof(__half),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(y_out.data(), d_y, y_out.size() * sizeof(__half),
                            hipMemcpyDeviceToHost));
    };

    const std::vector<double> want_z =
        aeon::reference::grouped_wo_a(kTokens, kGroups, kRank, kGroupDim, o, w_at);

    std::vector<__half> h_z, h_y;
    CHECK_HIP(hipMemcpy(d_o, h_o.data(), o_count * sizeof(__half), hipMemcpyHostToDevice));
    run_chain(h_z, h_y);

    // =======================================================================
    // A. wo_a — the grouped low-rank projection
    // =======================================================================
    const std::vector<double> got_z = widen(h_z);
    ok &= report("z matches einsum(tgd,grd->tgr)", want_z, got_z);

    // (1) The weight layout is load-bearing. Read the same bytes as an
    //     interleaved [R, G, D] and the answer must move materially — otherwise
    //     "group-major" would be a claim nothing here can falsify.
    {
        const std::vector<double> interleaved = aeon::reference::grouped_wo_a(
            kTokens, kGroups, kRank, kGroupDim, o, w_at, /*contiguous_blocks=*/false);
        const double d = aeon::reference::compare(want_z, interleaved, 1.0).max_rel;
        ok &= check("group-major weight layout is load-bearing", d > 1e-1,
                    "interleaved reading differs by " + std::to_string(d));
    }

    // (2) Group isolation. Perturbing group 0 — its 8 contiguous heads — must
    //     leave groups 1..7 untouched. A shared weight block, a global reduction
    //     over the whole 32768-wide row, or a wrong group stride all violate it.
    {
        std::vector<__half> h_o2 = h_o;
        for (size_t t = 0; t < kTokens; ++t) {
            for (size_t d = 0; d < kGroupDim; ++d) {
                const size_t i = t * kO + 0 * kGroupDim + d;
                h_o2[i] = __float2half(__half2float(h_o2[i]) * 100.0f + 1.0f);
            }
        }
        CHECK_HIP(hipMemcpy(d_o, h_o2.data(), o_count * sizeof(__half),
                            hipMemcpyHostToDevice));
        std::vector<__half> h_z2, h_y2;
        run_chain(h_z2, h_y2);

        const size_t per_group = kRank; // one group's slice per token
        size_t changed_others = 0;
        for (size_t t = 0; t < kTokens; ++t) {
            for (size_t g = 1; g < kGroups; ++g) {
                const size_t from = t * kZ + g * kRank;
                for (size_t i = from; i < from + per_group; ++i) {
                    if (__half2float(h_z[i]) != __half2float(h_z2[i])) ++changed_others;
                }
            }
        }
        ok &= check("perturbing group 0 leaves groups 1..7 untouched",
                    changed_others == 0,
                    std::to_string(changed_others) + " of " +
                        std::to_string((kGroups - 1) * kRank * kTokens) + " elements moved");

        std::string detail;
        const bool zero_moved = !bit_identical(h_z, h_z2, 0, per_group, detail);
        ok &= check("...and does change group 0", zero_moved, "group 0 " + detail);
    }

    // (3) The reduction reaches the whole 4096-wide block, not just its head.
    //     Perturb the very last element of group 0 (head 7, dim 511).
    {
        std::vector<__half> h_o3 = h_o;
        const size_t last = 0 * kO + 0 * kGroupDim + (kGroupDim - 1);
        h_o3[last] = __float2half(1.0f);
        CHECK_HIP(hipMemcpy(d_o, h_o3.data(), o_count * sizeof(__half),
                            hipMemcpyHostToDevice));
        std::vector<__half> h_z3, h_y3;
        run_chain(h_z3, h_y3);

        std::string detail;
        const bool moved = !bit_identical(h_z, h_z3, 0, kRank, detail);
        ok &= check("group 0 depends on its last head (extent = 4096)",
                    moved, "group 0 " + detail);
    }

    // Restore the unperturbed input for the wo_b section.
    CHECK_HIP(hipMemcpy(d_o, h_o.data(), o_count * sizeof(__half), hipMemcpyHostToDevice));
    run_chain(h_z, h_y);

    // =======================================================================
    // B. wo_b — 8192 -> 4096
    // =======================================================================
    //
    // Two comparisons, because they answer different questions:
    //   * fed the kernel's own `z`, this isolates wo_b (the einsum error is
    //     already measured above and must not leak into this one);
    //   * fed the fp64 oracle `z`, it certifies the chain end to end.
    {
        std::vector<double> want_y(kTokens * kHidden, 0.0);
        for (size_t t = 0; t < kTokens; ++t) {
            const std::vector<double> z_t(got_z.begin() + static_cast<long>(t * kZ),
                                          got_z.begin() + static_cast<long>((t + 1) * kZ));
            const std::vector<double> y_t = aeon::reference::matvec(kHidden, kZ, z_t, b_at2);
            for (size_t i = 0; i < kHidden; ++i) want_y[t * kHidden + i] = y_t[i];
        }
        ok &= report("wo_b: attn_proj vs oracle (isolated)", want_y, widen(h_y));

        std::vector<double> chain(kTokens * kHidden, 0.0);
        for (size_t t = 0; t < kTokens; ++t) {
            const std::vector<double> z_t(want_z.begin() + static_cast<long>(t * kZ),
                                          want_z.begin() + static_cast<long>((t + 1) * kZ));
            const std::vector<double> y_t = aeon::reference::matvec(kHidden, kZ, z_t, b_at2);
            for (size_t i = 0; i < kHidden; ++i) chain[t * kHidden + i] = y_t[i];
        }
        ok &= report("wo_b: attn_proj vs oracle (full chain)", chain, widen(h_y));

        // The layout the kernel reads from must be the same one the checkpoint
        // stores: `wo_b` is `[out, in]` row-major with no transpose anywhere.
        // A transposed read is a plausible, and plausible-looking, error.
        std::vector<double> transposed(kTokens * kHidden, 0.0);
        for (size_t t = 0; t < kTokens; ++t) {
            const std::vector<double> z_t(got_z.begin() + static_cast<long>(t * kZ),
                                          got_z.begin() + static_cast<long>((t + 1) * kZ));
            for (size_t o_i = 0; o_i < kHidden; ++o_i) {
                double acc = 0.0;
                for (size_t i = 0; i < kZ; ++i) acc += b_at(i * kHidden + o_i) * z_t[i];
                transposed[t * kHidden + o_i] = acc;
            }
        }
        const double d = aeon::reference::compare(want_y, transposed, 1.0).max_rel;
        ok &= check("wo_b weight orientation is load-bearing", d > 1e-1,
                    "transposed read differs by " + std::to_string(d));
    }

    // =======================================================================
    // C. Shapes
    // =======================================================================
    ok &= check("low-rank width is G·R = 8192", kZ == 8192, "8192");
    ok &= check("final width is the hidden size", kHidden == 4096, "4096");
    ok &= check("group_dim is 8 heads × 512", kGroupDim == 4096, "4096");
    ok &= check("z has T·G·R elements", h_z.size() == kTokens * kZ,
                std::to_string(h_z.size()) + " elements");
    ok &= check("attn_proj has T·H elements", h_y.size() == kTokens * kHidden,
                std::to_string(h_y.size()) + " elements");

    CHECK_HIP(hipFree(d_o));
    CHECK_HIP(hipFree(d_w));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_z));
    CHECK_HIP(hipFree(d_y));

    std::cout << (ok ? "\n[SUCCESS] Grouped output projection gate passed.\n"
                     : "\n[FAILURE] Grouped output projection gate failed.\n");
    return ok ? 0 : 1;
}
