// -----------------------------------------------------------------------------
// Tier-1 gate: RoPE (Step 2.3), two bases — versus an independent fp64 reference.
//
// This gate carries more weight than the RMSNorm one, because RoPE has four
// recorded traps that all produce output which "looks like" attention:
//
//   trap 27 — the *last* 64 dims rotate, not the first.
//   trap 7  — two bases (theta 10000 plain / 160000 YaRN), never one.
//   trap 8  — inverse RoPE exists and is a separate operation.
//   and the interleave is GPT-J (adjacent pairs), not NeoX.
//
// So the gate does not merely compare numbers; it *discriminates*. It asserts
// that the two bases are distinguishable, that the nope region is untouched, and
// that forward∘inverse is the identity. A gate that only checked "the numbers
// are close" would pass with the wrong 64 dims rotated.
//
// On tolerance at large positions. The kernel stores its tables as fp32, and
// computes `angle = pos * freq` in fp32; the oracle does both in fp64. At
// pos ≈ 65537 the fp32 angle has an ulp of 2^-7 ≈ 7.8e-3, so the two tables
// legitimately differ by up to ~1e-2 in cos/sin before any kernel error is
// involved. Those lines are therefore given an explicit `5e-2` bound *and*
// labelled, rather than being silently loosened. The tight checks live at small
// positions, where fp32 is exact to ~1e-7 and any real defect is visible. A
// wrong base or a missing YaRN factor changes cos/sin by O(1) and fails even the
// loose bound by two orders of magnitude.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/v4_rope.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
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
using aeon::reference::RopeClass;
using aeon::reference::RopeSpec;
using aeon::reference::RopeTableRef;

constexpr uint32_t kHeadDim   = 512;
constexpr uint32_t kRotaryDim = 64;
constexpr uint32_t kHalfRope  = kRotaryDim / 2;         // 32
constexpr uint32_t kNopeDim   = kHeadDim - kRotaryDim;  // 448

constexpr int kTokens = 8;
constexpr int kHeads  = 4;

// 65536 = original_max_position_embeddings, so 65537 is the first position past
// the YaRN regime boundary. The plan asks for a check beyond it.
constexpr uint32_t kFarPos  = 65537;
constexpr uint32_t kTableLen = kFarPos + 1;

// Tolerance sets, named so the intent is visible at the call site.
constexpr double kTightAbs = 1e-5,  kTightRel = 1e-5;   // fp64 oracle vs fp32 table
constexpr double kFp16Abs  = 1e-3,  kFp16Rel  = 2e-3;   // one fp16 ulp of O(1)
constexpr double kFp32AngleAbs = 5e-2, kFp32AngleRel = 5e-2; // see file header
constexpr double kIdentityAbs  = 2e-3, kIdentityRel  = 3e-3; // ulp of row scale

std::vector<double> widen(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<double>(__half2float(v[i]));
    }
    return out;
}

bool report(const char* label, const char* note,
            const ErrorStats& s, double tol_abs, double tol_rel) {
    if (s.size_mismatch) {
        std::printf("  %-38s SIZE MISMATCH                             FAIL\n", label);
        return false;
    }
    const bool pass = aeon::reference::within(s, tol_abs, tol_rel);
    std::printf("  %-38s max_abs=%.3e max_rel=%.3e  %s%s\n",
                label, s.max_abs, s.max_rel,
                pass ? "PASS" : "FAIL",
                (note && *note) ? note : "");
    return pass;
}

bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-38s %-46s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

// Concatenates the cos then sin caches of a kernel RopeTable into a flat vector,
// so one comparison covers both.
std::vector<double> widen_table(const aeon::kernel::RopeTable& t) {
    std::vector<double> out(t.cos_cache.size() + t.sin_cache.size());
    for (size_t i = 0; i < t.cos_cache.size(); ++i) {
        out[i] = static_cast<double>(t.cos_cache[i]);
    }
    for (size_t i = 0; i < t.sin_cache.size(); ++i) {
        out[t.cos_cache.size() + i] = static_cast<double>(t.sin_cache[i]);
    }
    return out;
}

std::vector<double> flatten_table(const RopeTableRef& t) {
    std::vector<double> out(t.cos.size() + t.sin.size());
    std::copy(t.cos.begin(), t.cos.end(), out.begin());
    std::copy(t.sin.begin(), t.sin.end(), out.begin() + t.cos.size());
    return out;
}

// Extracts one position's 2 * half values (cos then sin) from a flattened
// table. Both halves must be compared together — checking only cos would leave
// a sign error in the sine completely invisible.
std::vector<double> table_row_at(const std::vector<double>& flat, size_t cos_size,
                                 uint32_t position, uint32_t half) {
    std::vector<double> out(2 * half);
    const size_t base = static_cast<size_t>(position) * half;
    std::copy(flat.begin() + base, flat.begin() + base + half, out.begin());
    std::copy(flat.begin() + cos_size + base, flat.begin() + cos_size + base + half,
              out.begin() + half);
    return out;
}

// Applies the oracle rotation to a [tokens, heads, head_dim] buffer, one head
// row at a time. `position` is the token's absolute position in the sequence.
std::vector<double> oracle_rotate(const std::vector<__half>& src,
                                  int tokens, int heads,
                                  const RopeTableRef& table,
                                  bool inverse) {
    std::vector<double> out = widen(src);
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const size_t base = (static_cast<size_t>(t) * heads + h) * kHeadDim;
            std::vector<double> row(out.begin() + base, out.begin() + base + kHeadDim);
            aeon::reference::rope_apply_tail(row, table, static_cast<uint32_t>(t), inverse);
            std::copy(row.begin(), row.end(), out.begin() + base);
        }
    }
    return out;
}

// Same, for a single-position [heads, head_dim] buffer.
std::vector<double> oracle_rotate_at(const std::vector<__half>& src,
                                     int heads, uint32_t position,
                                     const RopeTableRef& table,
                                     bool inverse) {
    std::vector<double> out = widen(src);
    for (int h = 0; h < heads; ++h) {
        const size_t base = static_cast<size_t>(h) * kHeadDim;
        std::vector<double> row(out.begin() + base, out.begin() + base + kHeadDim);
        aeon::reference::rope_apply_tail(row, table, position, inverse);
        std::copy(row.begin(), row.end(), out.begin() + base);
    }
    return out;
}

// Bitwise check that the nope region [0, 448) of every head row is untouched.
// This is what makes trap 27 impossible to pass silently.
bool nope_region_intact(const std::vector<__half>& before,
                        const std::vector<__half>& after,
                        int tokens, int heads) {
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const size_t base = (static_cast<size_t>(t) * heads + h) * kHeadDim;
            for (uint32_t d = 0; d < kNopeDim; ++d) {
                if (__half_as_ushort(before[base + d]) != __half_as_ushort(after[base + d])) {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: RoPE (two bases, tail, GPT-J) vs independent fp64 reference\n";
    aeon::core::select_compute_device(true);

    bool ok = true;

    // -----------------------------------------------------------------------
    // A. Oracle self-checks (host only, no kernel involved).
    // -----------------------------------------------------------------------
    const RopeSpec spec_sliding = aeon::reference::rope_spec_for(RopeClass::Sliding);
    const RopeSpec spec_compressed = aeon::reference::rope_spec_for(RopeClass::Compressed);

    const RopeTableRef ref_sliding = aeon::reference::rope_table(spec_sliding, kTableLen);
    const RopeTableRef ref_compressed = aeon::reference::rope_table(spec_compressed, kTableLen);

    aeon::reference::Rng rng(0x0A0FE123ull);

    // A1. forward ∘ inverse = identity in fp64. If this fails the reference is
    //     wrong and nothing below means anything.
    {
        const std::vector<double> row = rng.vector_filled(kHeadDim, 1.0);
        std::vector<double> round = row;
        aeon::reference::rope_apply_tail(round, ref_compressed, 7, false);
        aeon::reference::rope_apply_tail(round, ref_compressed, 7, true);
        ok &= report("oracle: fwd∘inv = identity (fp64)", "", 
                     aeon::reference::compare(row, round), 1e-12, 1e-12);
    }

    // A2. The rotation preserves the L2 norm of the rotated tail.
    {
        const std::vector<double> row = rng.vector_filled(kHeadDim, 1.0);
        std::vector<double> rot = row;
        aeon::reference::rope_apply_tail(rot, ref_compressed, 1234, false);

        auto tail_norm = [](const std::vector<double>& v) {
            double sum = 0.0;
            for (size_t i = kNopeDim; i < kNopeDim + kRotaryDim; ++i) sum += v[i] * v[i];
            return std::sqrt(sum);
        };
        const std::vector<double> a{tail_norm(row)};
        const std::vector<double> b{tail_norm(rot)};
        ok &= report("oracle: tail L2 norm preserved", "",
                     aeon::reference::compare(a, b), 1e-12, 1e-12);
    }

    // A3. The two classes must actually differ, or every "two bases" claim below
    //     is vacuous and this gate cannot tell them apart.
    {
        double max_diff = 0.0;
        for (size_t i = 0; i < ref_sliding.cos.size(); ++i) {
            max_diff = std::fmax(max_diff, std::fabs(ref_sliding.cos[i] - ref_compressed.cos[i]));
        }
        ok &= check("oracle: the two bases differ",
                    max_diff > 0.1,
                    "max|cos_sliding - cos_compressed| = " + std::to_string(max_diff));
    }

    // -----------------------------------------------------------------------
    // B. Kernel tables versus the reference tables.
    // -----------------------------------------------------------------------
    aeon::kernel::RopeTable kern_sliding;
    kern_sliding.init(kTableLen, 10000.0f, 1.0f, 32.0f, 1.0f, 65536);
    aeon::kernel::RopeTable kern_compressed;
    kern_compressed.init(kTableLen, 160000.0f, 16.0f, 32.0f, 1.0f, 65536);

    const std::vector<double> kern_sliding_flat = widen_table(kern_sliding);
    const std::vector<double> kern_compressed_flat = widen_table(kern_compressed);
    const std::vector<double> ref_sliding_flat = flatten_table(ref_sliding);
    const std::vector<double> ref_compressed_flat = flatten_table(ref_compressed);

    // The tables are cos/sin: bounded by 1, and crossing zero at regular
    // intervals. At a crossing the relative error carries no information, so
    // these lines are judged on absolute error only — which is the right unit
    // for a table whose error comes from fp32 angle representation, not from a
    // proportional rounding. `rel_floor_fraction = 1.0` makes the denominator
    // the table's own scale for every element.
    constexpr double kAbsOnly = 1.0;

    // B1/B2. Position 1: fp32 is exact here to ~1e-7, so this is the tight,
    //        load-bearing frequency check — theta, the YaRN ramp, the base.
    ok &= report("table @pos1 (sliding)", "",
                 aeon::reference::compare(
                     table_row_at(ref_sliding_flat, ref_sliding.cos.size(), 1, kHalfRope),
                     table_row_at(kern_sliding_flat, ref_sliding.cos.size(), 1, kHalfRope),
                     kAbsOnly),
                 kTightAbs, kTightRel);
    ok &= report("table @pos1 (compressed)", "",
                 aeon::reference::compare(
                     table_row_at(ref_compressed_flat, ref_compressed.cos.size(), 1, kHalfRope),
                     table_row_at(kern_compressed_flat, ref_compressed.cos.size(), 1, kHalfRope),
                     kAbsOnly),
                 kTightAbs, kTightRel);

    // B3/B4. The whole table: a complete worst-case bound over every position,
    //        which already includes the ones past the YaRN boundary.
    ok &= report("table all positions (sliding) [fp32 angle]", "",
                 aeon::reference::compare(ref_sliding_flat, kern_sliding_flat, kAbsOnly),
                 kFp32AngleAbs, kFp32AngleRel);
    ok &= report("table all positions (compressed) [fp32 angle]", "",
                 aeon::reference::compare(ref_compressed_flat, kern_compressed_flat, kAbsOnly),
                 kFp32AngleAbs, kFp32AngleRel);

    // B5/B6. The plan's named position: the first one past
    //        original_max_position_embeddings = 65536.
    ok &= report("table @pos65537 (sliding) [fp32 angle]", "",
                 aeon::reference::compare(
                     table_row_at(ref_sliding_flat, ref_sliding.cos.size(), kFarPos, kHalfRope),
                     table_row_at(kern_sliding_flat, ref_sliding.cos.size(), kFarPos, kHalfRope),
                     kAbsOnly),
                 kFp32AngleAbs, kFp32AngleRel);
    ok &= report("table @pos65537 (compressed) [fp32 angle]", "",
                 aeon::reference::compare(
                     table_row_at(ref_compressed_flat, ref_compressed.cos.size(), kFarPos, kHalfRope),
                     table_row_at(kern_compressed_flat, ref_compressed.cos.size(), kFarPos, kHalfRope),
                     kAbsOnly),
                 kFp32AngleAbs, kFp32AngleRel);

    // -----------------------------------------------------------------------
    // C. Kernel rotation versus the oracle rotation.
    // -----------------------------------------------------------------------
    const size_t vec_count = static_cast<size_t>(kTokens) * kHeads * kHeadDim;
    const size_t single_count = static_cast<size_t>(kHeads) * kHeadDim;

    aeon::reference::Rng data_rng(0x0B0A7E45ull);
    std::vector<__half> h_vec(vec_count);
    {
        const std::vector<double> src = data_rng.vector_filled(vec_count, 1.0);
        for (size_t i = 0; i < vec_count; ++i) {
            h_vec[i] = __float2half(static_cast<float>(src[i]));
        }
    }
    const std::vector<__half> h_vec_orig = h_vec;

    __half* d_vec = nullptr;
    CHECK_HIP(hipMalloc(&d_vec, vec_count * sizeof(__half)));
    __half* d_single = nullptr;
    CHECK_HIP(hipMalloc(&d_single, single_count * sizeof(__half)));

    // The reference table lives in host memory; copy it to the device once.
    float* d_tab_cos = nullptr;
    float* d_tab_sin = nullptr;
    CHECK_HIP(hipMalloc(&d_tab_cos, static_cast<size_t>(kTableLen) * kHalfRope * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_tab_sin, static_cast<size_t>(kTableLen) * kHalfRope * sizeof(float)));

    auto load_tables = [&](const aeon::kernel::RopeTable& t) {
        CHECK_HIP(hipMemcpy(d_tab_cos, t.cos_cache.data(),
                            static_cast<size_t>(kTableLen) * kHalfRope * sizeof(float),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_tab_sin, t.sin_cache.data(),
                            static_cast<size_t>(kTableLen) * kHalfRope * sizeof(float),
                            hipMemcpyHostToDevice));
    };

    auto run_forward = [&](const std::vector<__half>& host) {
        CHECK_HIP(hipMemcpy(d_vec, host.data(), vec_count * sizeof(__half), hipMemcpyHostToDevice));
        aeon::kernel::v4_forward_rope_wave32_kernel<<<dim3(kHeads, kTokens), 32>>>(
            d_vec, d_tab_cos, d_tab_sin, kHeads, kHeadDim, kNopeDim, kHalfRope);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<__half> out(vec_count);
        CHECK_HIP(hipMemcpy(out.data(), d_vec, vec_count * sizeof(__half), hipMemcpyDeviceToHost));
        return out;
    };
    auto run_inverse = [&](const std::vector<__half>& host) {
        CHECK_HIP(hipMemcpy(d_vec, host.data(), vec_count * sizeof(__half), hipMemcpyHostToDevice));
        aeon::kernel::v4_inverse_rope_wave32_kernel<<<dim3(kHeads, kTokens), 32>>>(
            d_vec, d_tab_cos, d_tab_sin, kHeads, kHeadDim, kNopeDim, kHalfRope);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<__half> out(vec_count);
        CHECK_HIP(hipMemcpy(out.data(), d_vec, vec_count * sizeof(__half), hipMemcpyDeviceToHost));
        return out;
    };

    // --- Sliding class (theta 10000, plain) ---
    load_tables(kern_sliding);
    {
        const std::vector<__half> got = run_forward(h_vec);
        ok &= report("forward [T=8,H=4] (sliding)", "",
                     aeon::reference::compare(oracle_rotate(h_vec, kTokens, kHeads, ref_sliding, false),
                                              widen(got)),
                     kFp16Abs, kFp16Rel);
        // Trap 27: the first 448 dims of every head must be bit-identical.
        ok &= check("forward: nope region untouched (sliding)",
                    nope_region_intact(h_vec_orig, got, kTokens, kHeads),
                    "dims [0," + std::to_string(kNopeDim) + ") bit-identical");
    }
    {
        const std::vector<__half> got = run_inverse(h_vec);
        ok &= report("inverse [T=8,H=4] (sliding)", "",
                     aeon::reference::compare(oracle_rotate(h_vec, kTokens, kHeads, ref_sliding, true),
                                              widen(got)),
                     kFp16Abs, kFp16Rel);
    }

    // --- Compressed class (theta 160000, YaRN factor 16) ---
    load_tables(kern_compressed);
    {
        const std::vector<__half> got = run_forward(h_vec);
        ok &= report("forward [T=8,H=4] (compressed)", "",
                     aeon::reference::compare(oracle_rotate(h_vec, kTokens, kHeads, ref_compressed, false),
                                              widen(got)),
                     kFp16Abs, kFp16Rel);
        ok &= check("forward: nope region untouched (compressed)",
                    nope_region_intact(h_vec_orig, got, kTokens, kHeads),
                    "dims [0," + std::to_string(kNopeDim) + ") bit-identical");
    }
    {
        const std::vector<__half> got = run_inverse(h_vec);
        ok &= report("inverse [T=8,H=4] (compressed)", "",
                     aeon::reference::compare(oracle_rotate(h_vec, kTokens, kHeads, ref_compressed, true),
                                              widen(got)),
                     kFp16Abs, kFp16Rel);
    }

    // --- Single-position kernel, small and far position ---
    {
        std::vector<__half> h_single(h_vec.begin(), h_vec.begin() + single_count);
        const std::vector<__half> h_single_orig = h_single;

        CHECK_HIP(hipMemcpy(d_single, h_single.data(), single_count * sizeof(__half),
                            hipMemcpyHostToDevice));
        aeon::kernel::v4_forward_rope_at_pos_wave32_kernel<<<kHeads, 32>>>(
            d_single, d_tab_cos, d_tab_sin, 1, kHeads, kHeadDim, kNopeDim, kHalfRope);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_single.data(), d_single, single_count * sizeof(__half),
                            hipMemcpyDeviceToHost));
        ok &= report("forward @pos1 (at_pos, compressed)", "",
                     aeon::reference::compare(
                         oracle_rotate_at(h_single_orig, kHeads, 1, ref_compressed, false),
                         widen(h_single)),
                     kFp16Abs, kFp16Rel);

        CHECK_HIP(hipMemcpy(d_single, h_single_orig.data(), single_count * sizeof(__half),
                            hipMemcpyHostToDevice));
        aeon::kernel::v4_forward_rope_at_pos_wave32_kernel<<<kHeads, 32>>>(
            d_single, d_tab_cos, d_tab_sin, static_cast<int>(kFarPos),
            kHeads, kHeadDim, kNopeDim, kHalfRope);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_single.data(), d_single, single_count * sizeof(__half),
                            hipMemcpyDeviceToHost));
        ok &= report("forward @pos65537 (at_pos) [fp32 angle]", "",
                     aeon::reference::compare(
                         oracle_rotate_at(h_single_orig, kHeads, kFarPos, ref_compressed, false),
                         widen(h_single)),
                     kFp32AngleAbs, kFp32AngleRel);
    }

    // -----------------------------------------------------------------------
    // D. Device round trip: forward then inverse must return the input.
    // -----------------------------------------------------------------------
    {
        load_tables(kern_compressed);
        CHECK_HIP(hipMemcpy(d_vec, h_vec_orig.data(), vec_count * sizeof(__half),
                            hipMemcpyHostToDevice));
        aeon::kernel::v4_forward_rope_wave32_kernel<<<dim3(kHeads, kTokens), 32>>>(
            d_vec, d_tab_cos, d_tab_sin, kHeads, kHeadDim, kNopeDim, kHalfRope);
        CHECK_HIP(hipGetLastError());
        aeon::kernel::v4_inverse_rope_wave32_kernel<<<dim3(kHeads, kTokens), 32>>>(
            d_vec, d_tab_cos, d_tab_sin, kHeads, kHeadDim, kNopeDim, kHalfRope);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<__half> out(vec_count);
        CHECK_HIP(hipMemcpy(out.data(), d_vec, vec_count * sizeof(__half), hipMemcpyDeviceToHost));
        // Judged against the row scale, not per element. The rotation mixes each
        // (x0, x1) pair, so an element that is small next to its partner is
        // returned with an absolute error set by the *larger* operand — about
        // one fp16 ulp of the row's magnitude, independent of its own size. The
        // reference has exactly the same property (it casts back to the input
        // dtype after the rotation); a per-element relative test would flag it
        // as a failure of the kernel, which it is not.
        ok &= report("device fwd then inv = input (± ulp of row)", "",
                     aeon::reference::compare(widen(h_vec_orig), widen(out), 1.0),
                     kIdentityAbs, kIdentityRel);
    }

    CHECK_HIP(hipFree(d_vec));
    CHECK_HIP(hipFree(d_single));
    CHECK_HIP(hipFree(d_tab_cos));
    CHECK_HIP(hipFree(d_tab_sin));

    std::cout << (ok ? "[SUCCESS] RoPE gate passed.\n"
                     : "[FAILURE] RoPE gate failed.\n");
    return ok ? 0 : 1;
}
