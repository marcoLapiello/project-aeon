// -----------------------------------------------------------------------------
// Tier-1 gate: compressor + APE (Step 2.4.2) — versus an independent fp64
// reference.
//
// The compressor is the second op the audit found missing entirely (the APE
// term). It runs as two kernels — `v4_save_compressor_state_kernel` (per token)
// and `v4_materialize_compressed_entry_kernel` (per boundary) — so the gate
// exercises both, at both ratio classes.
//
// What it proves beyond closeness:
//
//   * APE IS A `score` TERM ONLY. `partial_kv` is compared **exactly** against a
//     plain widening of the input, so any leakage of APE into the kv branch is
//     caught bit-for-bit. `partial_score` is compared against `score + ape`.
//   * THE APE ROW IS `position % ratio`, i.e. periodic with period `ratio`.
//     Positions `p` and `p + ratio` must produce identical rows.
//   * THE WINDOW IS `(1 + overlap)·ratio`, with `overlap = (ratio == 4)`. Asserted
//     by comparing against oracle runs at a *different* window length and
//     requiring them to differ.
//   * THE TWO-SEGMENT OVERLAP MAPS CORRECTLY. For ratio 4 the window's second
//     half reads the second segment (`offset >= head_dim`). The gate asserts the
//     segments are populated from the right offsets by feeding the two segments
//     distinguishable data and checking the result uses both.
//   * THE BOUNDARY IS `(pos + 1) % ratio == 0`, and the RoPE position is the
//     window start `(pos / ratio)·ratio` — checked equal to the plan's
//     `pos + 1 − ratio` at every boundary.
//   * K = V: the kernel writes the same value to `compressed_key` and
//     `compressed_value`.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention.hpp"
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
using aeon::reference::RopeTableRef;

constexpr size_t kHeadDim = 512;
constexpr size_t kNopeDim = 448;
constexpr size_t kRopeDim = 64;
constexpr double kEps = 1e-6;

constexpr double kExactTol = 1e-6;  // fp32 widening / one fp32 add
constexpr double kFp16Tol  = 3e-3;  // fraction of peak; fp16 store

std::vector<double> widen(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(__half2float(v[i]));
    return out;
}

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got, double tol) {
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-52s SIZE MISMATCH                    FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(s.max_rel) && s.max_rel <= tol;
    std::printf("  %-52s max_abs=%.3e  max_rel=%.3e  %s\n",
                label, s.max_abs, s.max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-52s %-26s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

// One ratio class under test.
struct Spec {
    const char* name;
    int ratio;
    int coeff;          // 1 + overlap; overlap = (ratio == 4)
    int window;         // coeff * ratio
    int capacity;       // ring capacity (>= window, no wrap in these tests)
};

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: compressor + APE vs independent fp64 reference\n";
    aeon::core::select_compute_device(true);

    bool ok = true;

    // Ratio 4 (CSA, overlap, coeff 2) and ratio 128 (HCA, no overlap, coeff 1).
    const Spec specs[] = {
        {"ratio 4   (CSA, overlap)",  4, 2,   8,  16},
        {"ratio 128 (HCA, no overlap)", 128, 1, 128, 128},
    };

    // RoPE table large enough for every boundary position used below.
    const uint32_t table_len = 4096;
    const aeon::reference::RopeSpec rope_spec =
        aeon::reference::rope_spec_for(RopeClass::Compressed);
    const RopeTableRef rope_ref = aeon::reference::rope_table(rope_spec, table_len);
    aeon::kernel::RopeTable rope_kern;
    rope_kern.init(table_len, 160000.0f, 16.0f, 32.0f, 1.0f, 65536);

    float* d_cos = nullptr;
    float* d_sin = nullptr;
    CHECK_HIP(hipMalloc(&d_cos, static_cast<size_t>(table_len) * 32 * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_sin, static_cast<size_t>(table_len) * 32 * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_cos, rope_kern.cos_cache.data(),
                        static_cast<size_t>(table_len) * 32 * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sin, rope_kern.sin_cache.data(),
                        static_cast<size_t>(table_len) * 32 * sizeof(float), hipMemcpyHostToDevice));

    for (const Spec& spec : specs) {
        std::cout << "\n  --- " << spec.name << " (ratio " << spec.ratio
                  << ", coeff " << spec.coeff << ", window " << spec.window << ") ---\n";

        const size_t width = static_cast<size_t>(spec.coeff) * kHeadDim;
        aeon::reference::Rng gen(0x0C0A9E55ull + static_cast<unsigned>(spec.ratio));

        // ===================================================================
        // A. save: APE on score, not kv; row = position % ratio
        // ===================================================================
        const std::vector<double> ape_d = gen.vector_filled(
            static_cast<size_t>(spec.ratio) * width, 1.5);
        std::vector<float> h_ape(ape_d.size());
        for (size_t i = 0; i < h_ape.size(); ++i) h_ape[i] = static_cast<float>(ape_d[i]);

        // A position past `ratio`, so that using `position` instead of
        // `position % ratio` as the row index would read out of bounds and fail.
        const int64_t position = static_cast<int64_t>(spec.ratio) * 3 + (spec.ratio - 1);

        std::vector<__half> h_kv(width), h_score(width);
        for (size_t i = 0; i < width; ++i) {
            h_kv[i] = __float2half(static_cast<float>(gen.symmetric(1.5)));
            h_score[i] = __float2half(static_cast<float>(gen.symmetric(1.5)));
        }

        __half *d_kv = nullptr, *d_score = nullptr;
        float *d_ape = nullptr, *d_pkv = nullptr, *d_pscore = nullptr;
        int64_t* d_positions = nullptr;
        CHECK_HIP(hipMalloc(&d_kv, width * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_score, width * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_ape, h_ape.size() * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_pkv, static_cast<size_t>(spec.capacity) * width * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_pscore, static_cast<size_t>(spec.capacity) * width * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_positions, static_cast<size_t>(spec.capacity) * sizeof(int64_t)));

        CHECK_HIP(hipMemcpy(d_kv, h_kv.data(), width * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_score, h_score.data(), width * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_ape, h_ape.data(), h_ape.size() * sizeof(float), hipMemcpyHostToDevice));

        auto run_save = [&](int64_t pos) {
            aeon::kernel::v4_save_compressor_state_kernel<<<dim3(1), dim3(256)>>>(
                d_kv, d_score, d_pkv, d_pscore, d_positions, d_ape, pos, spec.ratio,
                spec.capacity, static_cast<int>(width));
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipDeviceSynchronize());
        };

        run_save(position);

        const int slot = static_cast<int>(position % spec.capacity);
        const size_t row_off = static_cast<size_t>(slot) * width;

        std::vector<float> h_pkv(static_cast<size_t>(spec.capacity) * width);
        std::vector<float> h_pscore(static_cast<size_t>(spec.capacity) * width);
        CHECK_HIP(hipMemcpy(h_pkv.data(), d_pkv, h_pkv.size() * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_pscore.data(), d_pscore, h_pscore.size() * sizeof(float), hipMemcpyDeviceToHost));

        std::vector<double> pkv_row(h_pkv.begin() + row_off, h_pkv.begin() + row_off + width);
        std::vector<double> pscore_row(h_pscore.begin() + row_off, h_pscore.begin() + row_off + width);

        // partial_kv must be a *plain widening* of the input — no APE. Exact.
        ok &= report("save: partial_kv == kv (no APE leakage)",
                     widen(h_kv), pkv_row, kExactTol);

        // partial_score must be score + ape[position % ratio]. Exact modulo the
        // kernel's fp32 add versus the oracle's fp64 add.
        ok &= report("save: partial_score == score + ape[pos%ratio]",
                     aeon::reference::compressor_ape_apply(
                         widen(h_score), ape_d, position, spec.ratio, width),
                     pscore_row, kExactTol);

        // Discriminating: the APE row is periodic with period `ratio`.
        {
            run_save(position + spec.ratio);
            const int slot2 = static_cast<int>((position + spec.ratio) % spec.capacity);
            const size_t row2 = static_cast<size_t>(slot2) * width;
            std::vector<float> h2(h_pscore.size());
            CHECK_HIP(hipMemcpy(h2.data(), d_pscore, h2.size() * sizeof(float), hipMemcpyDeviceToHost));
            std::vector<double> row2_D(h2.begin() + row2, h2.begin() + row2 + width);
            const ErrorStats s = aeon::reference::compare(pscore_row, row2_D, 1.0);
            ok &= check("APE row is periodic with period ratio", s.max_abs == 0.0,
                        "pos and pos+ratio agree exactly");

            // And a position one step over must use a *different* row.
            run_save(position + 1);
            const int slot3 = static_cast<int>((position + 1) % spec.capacity);
            const size_t row3 = static_cast<size_t>(slot3) * width;
            CHECK_HIP(hipMemcpy(h2.data(), d_pscore, h2.size() * sizeof(float), hipMemcpyDeviceToHost));
            std::vector<double> row3_D(h2.begin() + row3, h2.begin() + row3 + width);
            ok &= check("a different residue uses a different APE row",
                        aeon::reference::compare(pscore_row, row3_D, 1.0).max_rel > 1e-2,
                        "max_rel between residues = " +
                            std::to_string(aeon::reference::compare(pscore_row, row3_D, 1.0).max_rel));
        }

        // ===================================================================
        // B. materialize: window, segments, per-dim softmax, norm, RoPE
        // ===================================================================
        // Boundary just past a window so every source position is >= 0.
        const int64_t boundary = static_cast<int64_t>(spec.window) * 2 - 1;
        if ((boundary + 1) % spec.ratio != 0) {
            std::printf("    (boundary setup error)\n");
            return 1;
        }

        // Build the ring: one save per window token, with per-token data so the
        // segments are distinguishable.
        std::vector<std::vector<__half>> win_kv(spec.window), win_score(spec.window);
        for (int o = 0; o < spec.window; ++o) {
            win_kv[o].resize(width);
            win_score[o].resize(width);
            for (size_t i = 0; i < width; ++i) {
                // Segment 0 values around +1, segment 1 around -1, so the two
                // halves are clearly not interchangeable.
                const double seg = (i < kHeadDim) ? 1.0 : -1.0;
                win_kv[o][i] = __float2half(static_cast<float>(seg + gen.symmetric(0.2)));
                win_score[o][i] = __float2half(static_cast<float>(gen.symmetric(0.8)));
            }
        }
        for (int o = 0; o < spec.window; ++o) {
            const int64_t p = boundary - spec.window + 1 + o;
            CHECK_HIP(hipMemcpy(d_kv, win_kv[o].data(), width * sizeof(__half), hipMemcpyHostToDevice));
            CHECK_HIP(hipMemcpy(d_score, win_score[o].data(), width * sizeof(__half), hipMemcpyHostToDevice));
            run_save(p);
        }

        // Norm weight and output.
        std::vector<__half> h_norm(kHeadDim);
        for (size_t i = 0; i < kHeadDim; ++i) {
            h_norm[i] = __float2half(static_cast<float>(0.8 + gen.symmetric(0.3)));
        }
        __half *d_norm = nullptr, *d_ckey = nullptr, *d_cvalue = nullptr;
        int64_t* d_cpos = nullptr;
        CHECK_HIP(hipMalloc(&d_norm, kHeadDim * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_ckey, kHeadDim * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_cvalue, kHeadDim * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_cpos, sizeof(int64_t)));
        CHECK_HIP(hipMemcpy(d_norm, h_norm.data(), kHeadDim * sizeof(__half), hipMemcpyHostToDevice));

        auto run_materialize = [&](int64_t bpos) {
            // One thread per head dimension: `dimension = threadIdx.x` and
            // `dimension < head_dim` guards the writes, so blockDim must be at
            // least head_dim. Launching fewer threads silently leaves the upper
            // dimensions untouched — which is exactly how this gate failed on
            // its first run.
            aeon::kernel::v4_materialize_compressed_entry_kernel<<<dim3(1), dim3(kHeadDim)>>>(
                d_pkv, d_pscore, d_positions, d_norm, d_ckey, d_cvalue, d_cpos,
                d_cos, d_sin, bpos, spec.ratio, spec.capacity,
                static_cast<int>(kHeadDim), static_cast<int>(width), 0,
                static_cast<int>(kNopeDim), static_cast<int>(kRopeDim),
                static_cast<float>(kEps));
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipDeviceSynchronize());
        };

        run_materialize(boundary);

        std::vector<__half> h_ckey(kHeadDim), h_cvalue(kHeadDim);
        CHECK_HIP(hipMemcpy(h_ckey.data(), d_ckey, kHeadDim * sizeof(__half), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_cvalue.data(), d_cvalue, kHeadDim * sizeof(__half), hipMemcpyDeviceToHost));

        // --- Oracle ------------------------------------------------------
        // The window entries, chronological, with APE already applied.
        std::vector<std::vector<double>> o_kv(spec.window), o_score(spec.window);
        for (int o = 0; o < spec.window; ++o) {
            const int64_t p = boundary - spec.window + 1 + o;
            o_kv[o] = widen(win_kv[o]);
            o_score[o] = aeon::reference::compressor_ape_apply(
                widen(win_score[o]), ape_d, p, spec.ratio, width);
        }
        const std::vector<double> raw = aeon::reference::compressor_raw(
            o_kv, o_score, kHeadDim, spec.ratio);
        std::vector<double> want = aeon::reference::rmsnorm(raw, widen(h_norm), kEps);

        const int64_t rope_pos = aeon::reference::compressor_rope_position(boundary, spec.ratio);
        aeon::reference::rope_apply_tail(want, rope_ref, static_cast<uint32_t>(rope_pos), false);

        ok &= report("materialize: compressed row (norm + RoPE)",
                     want, widen(h_ckey), kFp16Tol);

        // K = V.
        ok &= check("compressed_key == compressed_value (K = V)",
                    aeon::reference::compare(widen(h_ckey), widen(h_cvalue), 1e-9).max_abs == 0.0,
                    "byte-identical key and value");

        // The non-RoPE part must be untouched by the rotation.
        {
            const std::vector<double> ckey_d = widen(h_ckey);
            const std::vector<double> pre_rope = aeon::reference::rmsnorm(raw, widen(h_norm), kEps);
            const std::vector<double> nope_want(pre_rope.begin(), pre_rope.begin() + kNopeDim);
            const std::vector<double> nope_got(ckey_d.begin(), ckey_d.begin() + kNopeDim);
            ok &= report("materialize: nope region unrotated", nope_want, nope_got, kFp16Tol);
        }

        // --- Window length is load-bearing: a wrong window changes the answer.
        //     Drops HALF the window rather than one entry: with 128 window
        //     positions a single missing entry moves the weighted average by
        //     only ~4e-3, which is real but too small to be a clean signal at
        //     any window size. Half is unambiguous for both classes.
        {
            const size_t keep = o_kv.size() / 2;
            std::vector<std::vector<double>> short_kv(o_kv.begin(), o_kv.begin() + keep);
            std::vector<std::vector<double>> short_score(o_score.begin(), o_score.begin() + keep);
            const std::vector<double> raw_short = aeon::reference::compressor_raw(
                short_kv, short_score, kHeadDim, spec.ratio);
            std::vector<double> want_short =
                aeon::reference::rmsnorm(raw_short, widen(h_norm), kEps);
            aeon::reference::rope_apply_tail(want_short, rope_ref,
                                             static_cast<uint32_t>(rope_pos), false);
            const double d = aeon::reference::compare(want_short, widen(h_ckey), 1.0).max_rel;
            ok &= check("window length is load-bearing", d > 1e-2,
                        "half-window would differ by " + std::to_string(d));
        }

        // --- The two segments are both used (ratio 4 only).
        if (spec.coeff == 2) {
            // Zero the second segment's kv and score: the result must change,
            // proving the second half is actually read.
            std::vector<std::vector<double>> no_seg2_kv = o_kv;
            std::vector<std::vector<double>> no_seg2_score = o_score;
            for (int o = 0; o < spec.window; ++o) {
                for (size_t d = 0; d < kHeadDim; ++d) {
                    no_seg2_kv[o][kHeadDim + d] = 0.0;
                    no_seg2_score[o][kHeadDim + d] = -1000.0;
                }
            }
            const std::vector<double> raw_ns = aeon::reference::compressor_raw(
                no_seg2_kv, no_seg2_score, kHeadDim, spec.ratio);
            std::vector<double> want_ns = aeon::reference::rmsnorm(raw_ns, widen(h_norm), kEps);
            aeon::reference::rope_apply_tail(want_ns, rope_ref,
                                             static_cast<uint32_t>(rope_pos), false);
            const double d = aeon::reference::compare(want_ns, widen(h_ckey), 1.0).max_rel;
            ok &= check("second (overlap) segment is read", d > 1e-2,
                        "zeroing segment 2 would differ by " + std::to_string(d));
        }

        // --- RoPE position is the window start.
        {
            std::vector<double> wrong_pos = aeon::reference::rmsnorm(raw, widen(h_norm), kEps);
            const uint32_t wrong = static_cast<uint32_t>(
                aeon::reference::compressor_rope_position(boundary + 1, spec.ratio));
            aeon::reference::rope_apply_tail(wrong_pos, rope_ref, wrong, false);
            const double d = aeon::reference::compare(wrong_pos, widen(h_ckey), 1.0).max_rel;
            ok &= check("RoPE position is the window start", d > 1e-2,
                        "pos+1 would differ by " + std::to_string(d));
        }

        // --- Boundary alignment: the plan's `pos + 1 - ratio` == our formula.
        {
            bool all_equal = true;
            for (int64_t b = spec.ratio - 1; b <= 4 * spec.ratio; b += spec.ratio) {
                const int64_t plan_form = b + 1 - spec.ratio;
                const int64_t ours = aeon::reference::compressor_rope_position(b, spec.ratio);
                if (plan_form != ours) all_equal = false;
                // And the boundary condition itself holds at `b`.
                if ((b + 1) % spec.ratio != 0) all_equal = false;
            }
            ok &= check("rope pos == plan's pos+1-ratio at boundaries", all_equal,
                        "checked 5 consecutive boundaries");
        }

        // --- Truncated window: positions before 0 are skipped.
        //
        // Self-contained: it must write its *own* rows into the ring, because
        // the materialization guard `partial_positions[slot] == source_position`
        // rejects any slot holding a different token's state. The first run of
        // this gate reused slots left over from part A and the kernel — quite
        // correctly — found nothing to reduce.
        {
            const int64_t small_boundary = spec.ratio - 1; // window starts negative
            const int64_t first = small_boundary - spec.window + 1;

            std::vector<std::vector<__half>> t_kv, t_score;
            std::vector<int64_t> t_pos;
            for (int o = 0; o < spec.window; ++o) {
                const int64_t p = first + o;
                if (p < 0) continue;
                std::vector<__half> kv(width), sc(width);
                for (size_t i = 0; i < width; ++i) {
                    kv[i] = __float2half(static_cast<float>(gen.symmetric(1.0)));
                    sc[i] = __float2half(static_cast<float>(gen.symmetric(1.0)));
                }
                CHECK_HIP(hipMemcpy(d_kv, kv.data(), width * sizeof(__half), hipMemcpyHostToDevice));
                CHECK_HIP(hipMemcpy(d_score, sc.data(), width * sizeof(__half), hipMemcpyHostToDevice));
                run_save(p);
                t_kv.push_back(std::move(kv));
                t_score.push_back(std::move(sc));
                t_pos.push_back(p);
            }

            run_materialize(small_boundary);
            std::vector<__half> h_small(kHeadDim);
            CHECK_HIP(hipMemcpy(h_small.data(), d_ckey, kHeadDim * sizeof(__half),
                                hipMemcpyDeviceToHost));

            // Oracle over the same entries, in the same `segment = o/ratio`
            // mapping. Leading offsets whose position is negative are marked
            // invalid, exactly as the kernel's `if (source_position < 0) continue`
            // does — for ratio 4 that means the whole first segment is absent and
            // the surviving tokens arrive through the second segment.
            std::vector<std::vector<double>> s_kv(spec.window, std::vector<double>(width, 0.0));
            std::vector<std::vector<double>> s_score(spec.window, std::vector<double>(width, 0.0));
            std::vector<bool> s_valid(spec.window, false);
            for (size_t k = 0; k < t_pos.size(); ++k) {
                const size_t o = static_cast<size_t>(t_pos[k] - first);
                s_kv[o] = widen(t_kv[k]);
                s_score[o] = aeon::reference::compressor_ape_apply(
                    widen(t_score[k]), ape_d, t_pos[k], spec.ratio, width);
                s_valid[o] = true;
            }
            const std::vector<double> raw_small = aeon::reference::compressor_raw(
                s_kv, s_score, kHeadDim, spec.ratio, &s_valid);
            std::vector<double> want_small =
                aeon::reference::rmsnorm(raw_small, widen(h_norm), kEps);
            const int64_t rp = aeon::reference::compressor_rope_position(small_boundary, spec.ratio);
            aeon::reference::rope_apply_tail(want_small, rope_ref, static_cast<uint32_t>(rp), false);

            ok &= report("materialize: truncated window (early position)",
                         want_small, widen(h_small), kFp16Tol);

            // How many leading offsets were never written? This is a structural
            // fact, not a threshold. The first entry can only be emitted at
            // `pos = ratio - 1`, while the window is `coeff·ratio` long:
            //
            //   ratio 4   (coeff 2): window 8, first entry at pos 3, start = -4
            //                        -> exactly `ratio` leading offsets absent,
            //                           i.e. the whole OLDER segment.
            //   ratio 128 (coeff 1): window 128, first entry at pos 127,
            //                        start = 0 -> nothing absent.
            {
                size_t absent = 0;
                for (int o = 0; o < spec.window; ++o) {
                    if (first + o < 0) ++absent;
                }
                const size_t expected = spec.coeff == 2 ? static_cast<size_t>(spec.ratio) : 0;
                ok &= check("leading offsets absent on the first entry",
                            absent == expected,
                            std::to_string(absent) + " of " + std::to_string(spec.window) +
                                " absent (expected " + std::to_string(expected) + ")");
            }

            // The skipped offsets must genuinely be excluded, not merely
            // zero-weighted. Only meaningful when something *is* skipped — for
            // ratio 128 the first entry is already a full window, and asserting
            // a difference there would be asserting a false property.
            if (first < 0) {
                std::vector<bool> all_valid(spec.window, true);
                const std::vector<double> raw_all = aeon::reference::compressor_raw(
                    s_kv, s_score, kHeadDim, spec.ratio, &all_valid);
                std::vector<double> want_all =
                    aeon::reference::rmsnorm(raw_all, widen(h_norm), kEps);
                aeon::reference::rope_apply_tail(want_all, rope_ref,
                                                 static_cast<uint32_t>(rp), false);
                const double d = aeon::reference::compare(want_all, widen(h_small), 1.0).max_rel;
                ok &= check("skipped negative positions are excluded", d > 1e-2,
                            "including them would differ by " + std::to_string(d));
            }
        }

        CHECK_HIP(hipFree(d_kv));
        CHECK_HIP(hipFree(d_score));
        CHECK_HIP(hipFree(d_ape));
        CHECK_HIP(hipFree(d_pkv));
        CHECK_HIP(hipFree(d_pscore));
        CHECK_HIP(hipFree(d_positions));
        CHECK_HIP(hipFree(d_norm));
        CHECK_HIP(hipFree(d_ckey));
        CHECK_HIP(hipFree(d_cvalue));
        CHECK_HIP(hipFree(d_cpos));
    }

    // =======================================================================
    // Oracle self-checks
    // =======================================================================
    {
        // A single-entry window is a passthrough: the weighted sum is the value.
        {
            std::vector<std::vector<double>> kv{{0.5, -1.5}};
            std::vector<std::vector<double>> sc{{3.0, 3.0}};
            const std::vector<double> raw = aeon::reference::compressor_raw(kv, sc, 2, 4);
            ok &= report("oracle: one-entry window is a passthrough",
                         std::vector<double>{0.5, -1.5}, raw, 1e-12);
        }

        // Equal scores give equal weights, so the result is the mean.
        {
            std::vector<std::vector<double>> kv{{1.0, 0.0}, {3.0, 0.0}, {5.0, 0.0}};
            std::vector<std::vector<double>> sc{{2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}};
            const std::vector<double> raw = aeon::reference::compressor_raw(kv, sc, 1, 4);
            ok &= report("oracle: equal scores -> mean", std::vector<double>{3.0}, raw, 1e-12);
        }

        // A dominant score selects its own value.
        {
            std::vector<std::vector<double>> kv{{1.0}, {9.0}};
            std::vector<std::vector<double>> sc{{0.0}, {50.0}};
            const std::vector<double> raw = aeon::reference::compressor_raw(kv, sc, 1, 4);
            ok &= report("oracle: dominant score wins", std::vector<double>{9.0}, raw, 1e-9);
        }
    }

    CHECK_HIP(hipFree(d_cos));
    CHECK_HIP(hipFree(d_sin));

    std::cout << (ok ? "\n[SUCCESS] Compressor gate passed.\n"
                     : "\n[FAILURE] Compressor gate failed.\n");
    return ok ? 0 : 1;
}
