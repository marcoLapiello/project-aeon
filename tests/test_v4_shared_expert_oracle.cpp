// -----------------------------------------------------------------------------
// Tier-1 gate: shared expert (Step 2.10.4) — versus an independent fp64 oracle.
//
// The shared expert is the simplest op in the graph, and that is exactly why the
// gate is short. It is structurally the same activation as the routed expert, so
// the two things worth certifying are the ones that make it *different*:
//
//   A. STRUCTURE   — unquantized fp16, row-major `[out, in]`, exactly one shared
//                    expert, and no routing: it fires on every token.
//   B. NUMERICS    — all three projections and the composed FFN, on synthetic
//                    weights and then on the artifact's own tensors.
//   C. THE CLAMP   — the shared path uses the *same* `activation_clamp` as the
//                    routed path. Re-checked here on the real weights, because a
//                    shared expert that used a symmetric clamp would look fine on
//                    small activations.
//   D. THE COMBINE — the shared contribution enters the MoE output **exactly
//                    once**. Measured on the device, because "once" is the kind of
//                    property that is right by accident until it is not.
//
// The clamped-SwiGLU *rule* itself is certified by item 14's gate, against the
// asymmetric/symmetric/no-clamp fork. This gate reuses `clamped_swiglu` rather
// than re-deriving it: duplicating the rule here would let the two copies drift,
// and re-proving it would add no information.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "architecture/deepseek_v4/core/config.hpp"
#include "architecture/deepseek_v4/kernels/v4_gemv.hpp"
#include "architecture/deepseek_v4/kernels/v4_pipeline_ops.hpp"
#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"
#include "backend/swizzled_w4a16/core/swizzled_expert_format.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w13.hpp"
#include "backend/swizzled_w4a16/kernels/aeon_moe_fused_w2.hpp"
#include "infrastructure/core/aeon_loader.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

using aeon::reference::ClampMode;
using aeon::reference::ErrorStats;

constexpr int kHidden = 4096;
constexpr int kIntermediate = 2048;
constexpr double kLimit = 10.0;

// fp16 weights, fp32 accumulation, one fp16 store at the end of each matmul.
constexpr double kGemvTol = 3e-3;
constexpr double kFfnTol = 4e-3;

std::vector<double> widen(const std::vector<__half>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(__half2float(v[i]));
    return out;
}

std::vector<__half> to_half(const std::vector<double>& v) {
    std::vector<__half> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = __float2half(static_cast<float>(v[i]));
    return out;
}

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got, double tol) {
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-54s SIZE MISMATCH                 FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(s.max_rel) && s.max_rel <= tol;
    std::printf("  %-54s max_abs=%.3e max_rel=%.3e  %s\n",
                label, s.max_abs, s.max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

bool check(const char* label, bool ok, const std::string& detail) {
    std::printf("  %-54s %-22s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 primitive: shared expert (dense fp16 FFN)\n";
    aeon::core::select_compute_device(true);

    bool ok = true;
    aeon::reference::Rng gen(0x5A4E2D17ull);

    // ---------------------------------------------------------------------
    // Device buffers for the three projections.
    // ---------------------------------------------------------------------
    __half* d_x = nullptr;
    __half* d_w1 = nullptr;
    __half* d_w2 = nullptr;
    __half* d_w3 = nullptr;
    __half* d_gate = nullptr;
    __half* d_up = nullptr;
    __half* d_hidden = nullptr;
    __half* d_out = nullptr;

    CHECK_HIP(hipMalloc(&d_x, kHidden * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_w1, static_cast<size_t>(kIntermediate) * kHidden * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_w3, static_cast<size_t>(kIntermediate) * kHidden * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_w2, static_cast<size_t>(kHidden) * kIntermediate * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_gate, kIntermediate * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_up, kIntermediate * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_hidden, kIntermediate * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_out, kHidden * sizeof(__half)));

    // Runs the full shared expert for one token and returns the four stages.
    struct SharedStages {
        std::vector<__half> gate, up, hidden, out;
    };
    auto run_shared = [&]() {
        hipLaunchKernelGGL(aeon::kernel::v4_gemv_fp16_vec8_kernel,
                           dim3(kIntermediate), dim3(32), 0, 0,
                           d_x, d_w1, d_gate, kHidden);
        CHECK_HIP(hipGetLastError());
        hipLaunchKernelGGL(aeon::kernel::v4_gemv_fp16_vec8_kernel,
                           dim3(kIntermediate), dim3(32), 0, 0,
                           d_x, d_w3, d_up, kHidden);
        CHECK_HIP(hipGetLastError());
        hipLaunchKernelGGL(aeon::kernel::v4_pipeline_swiglu_clamp_kernel,
                           dim3((kIntermediate + 255) / 256), dim3(256), 0, 0,
                           d_gate, d_up, d_hidden, kIntermediate,
                           static_cast<float>(kLimit));
        CHECK_HIP(hipGetLastError());
        hipLaunchKernelGGL(aeon::kernel::v4_gemv_fp16_vec8_kernel,
                           dim3(kHidden), dim3(32), 0, 0,
                           d_hidden, d_w2, d_out, kIntermediate);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());

        SharedStages stages;
        stages.gate.assign(kIntermediate, __half{});
        stages.up.assign(kIntermediate, __half{});
        stages.hidden.assign(kIntermediate, __half{});
        stages.out.assign(kHidden, __half{});
        CHECK_HIP(hipMemcpy(stages.gate.data(), d_gate, kIntermediate * sizeof(__half),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(stages.up.data(), d_up, kIntermediate * sizeof(__half),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(stages.hidden.data(), d_hidden, kIntermediate * sizeof(__half),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(stages.out.data(), d_out, kHidden * sizeof(__half),
                            hipMemcpyDeviceToHost));
        return stages;
    };

    // =====================================================================
    // 0. Oracle self-check — closed form, on the activation rule
    // =====================================================================
    // `dense_ffn` could be wrong in a way that a kernel comparison cannot see
    // (both read the same weights). Pin it to arithmetic that can be worked out
    // by hand: with w1 = e_0 (a single 1 in the first column) the gate is x[0],
    // with w3 the same the up is x[0], and w2 = e_0ᵀ makes the output a single
    // value depending only on x[0].
    std::cout << "\n--- 0. oracle self-check (closed form) ---\n";
    {
        const size_t inter = 2, hid = 3;
        std::vector<double> w1(inter * hid, 0.0), w3(inter * hid, 0.0), w2(hid * inter, 0.0);
        w1[0] = 1.0; // gate[0] = x[0]
        w3[0] = 1.0; // up[0]   = x[0]
        w2[0] = 1.0; // out[0]  = hidden[0]
        const std::vector<double> x = {0.5, 0.0, 0.0};

        const std::vector<double> out =
            aeon::reference::dense_ffn(inter, hid, x, w1, w3, w2, kLimit);
        // hidden[0] = silu(clamp(0.5, max=10)) * clamp(0.5) = silu(0.5)*0.5
        const double expected = (0.5 / (1.0 + std::exp(-0.5))) * 0.5;
        const double error = std::fabs(out[0] - expected);
        ok &= check("dense_ffn matches a hand-computed value",
                    error < 1e-15 && out[1] == 0.0 && out[2] == 0.0,
                    "out[0] = " + std::to_string(out[0]) + " vs " + std::to_string(expected));
    }

    // =====================================================================
    // A. Structure
    // =====================================================================
    std::cout << "\n--- A. structure ---\n";
    {
        const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
        const auto config = aeon::core::DeepSeekV4Config::load_from_json(
            model_dir + "/config.json");

        ok &= check("n_shared_experts == 1", config.n_shared_experts == 1,
                    std::to_string(config.n_shared_experts));
        ok &= check("shared width is moe_intermediate_size (2048)",
                    config.moe_intermediate_size == kIntermediate,
                    std::to_string(config.moe_intermediate_size));

        aeon::core::AeonModelLoader loader;
        loader.open_model(model_dir);

        struct TensorCase { const char* name; int rows; int cols; };
        const TensorCase cases[] = {
            {"layers.3.ffn.shared_experts.w1.weight", kIntermediate, kHidden},
            {"layers.3.ffn.shared_experts.w3.weight", kIntermediate, kHidden},
            {"layers.3.ffn.shared_experts.w2.weight", kHidden, kIntermediate},
        };
        bool shapes_ok = true, dtype_ok = true;
        std::string detail;
        for (const TensorCase& c : cases) {
            const auto& t = loader.get_tensor(c.name);
            if (t.shape.size() != 2 || t.shape[0] != c.rows || t.shape[1] != c.cols) {
                shapes_ok = false;
                detail = c.name;
            }
            if (t.dtype != "F16") { dtype_ok = false; detail = c.name; }
            // Unquantized means "not one expert payload per expert": check the
            // byte count equals rows*cols*2 exactly.
            if (t.byte_size != static_cast<int64_t>(c.rows) * c.cols * 2) {
                dtype_ok = false;
                detail = std::string("byte size of ") + c.name;
            }
        }
        ok &= check("shared expert is F16 [2048,4096]/[4096,2048], not quantized",
                    shapes_ok && dtype_ok, detail);

        // No routing: the shared weights exist on every layer, and have no
        // per-expert structure. Checked on a hash layer, a ratio-4 layer and a
        // ratio-128 layer so the claim is not an accident of the sampled layer.
        bool everywhere = true;
        for (int layer : {0, 2, 3, 10, 42}) {
            const std::string prefix =
                "layers." + std::to_string(layer) + ".ffn.shared_experts.";
            if (!loader.has_tensor(prefix + "w1.weight") ||
                !loader.has_tensor(prefix + "w2.weight") ||
                !loader.has_tensor(prefix + "w3.weight")) {
                everywhere = false;
            }
        }
        ok &= check("shared expert weights exist on every sampled layer",
                    everywhere, "layers 0, 2, 3, 10, 42");
    }

    // =====================================================================
    // B. Numerics on synthetic weights
    // =====================================================================
    std::cout << "\n--- B. numerics (synthetic fp16 weights) ---\n";
    {
        std::vector<double> x_d = gen.vector_filled(kHidden, 1.0);
        std::vector<double> w1_d(static_cast<size_t>(kIntermediate) * kHidden);
        std::vector<double> w3_d(static_cast<size_t>(kIntermediate) * kHidden);
        std::vector<double> w2_d(static_cast<size_t>(kHidden) * kIntermediate);
        for (double& v : w1_d) v = gen.symmetric(0.03);
        for (double& v : w3_d) v = gen.symmetric(0.03);
        for (double& v : w2_d) v = gen.symmetric(0.03);

        const std::vector<__half> x_h = to_half(x_d);
        const std::vector<__half> w1_h = to_half(w1_d);
        const std::vector<__half> w3_h = to_half(w3_d);
        const std::vector<__half> w2_h = to_half(w2_d);

        // Compare against the oracle fed the *widened* fp16 inputs, so the delta
        // is the kernel's rounding, not the fixture's.
        const std::vector<double> xw = widen(x_h);
        const std::vector<double> w1w = widen(w1_h);
        const std::vector<double> w3w = widen(w3_h);
        const std::vector<double> w2w = widen(w2_h);

        CHECK_HIP(hipMemcpy(d_x, x_h.data(), kHidden * sizeof(__half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w1, w1_h.data(), w1_h.size() * sizeof(__half),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w3, w3_h.data(), w3_h.size() * sizeof(__half),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w2, w2_h.data(), w2_h.size() * sizeof(__half),
                            hipMemcpyHostToDevice));

        const SharedStages s = run_shared();

        const std::vector<double> want_gate = aeon::reference::matvec(
            kIntermediate, kHidden, xw,
            [&](size_t o, size_t i) { return w1w[o * kHidden + i]; });
        const std::vector<double> want_up = aeon::reference::matvec(
            kIntermediate, kHidden, xw,
            [&](size_t o, size_t i) { return w3w[o * kHidden + i]; });
        ok &= report("w1 projection vs oracle", want_gate, widen(s.gate), kGemvTol);
        ok &= report("w3 projection vs oracle", want_up, widen(s.up), kGemvTol);

        std::vector<double> want_hidden(kIntermediate);
        for (int i = 0; i < kIntermediate; ++i) {
            want_hidden[i] = aeon::reference::clamped_swiglu(
                want_gate[i], want_up[i], kLimit, ClampMode::Asymmetric);
        }
        ok &= report("hidden (clamped SwiGLU) vs oracle",
                     want_hidden, widen(s.hidden), kFfnTol);

        const std::vector<double> want_out = aeon::reference::dense_ffn(
            kIntermediate, kHidden, xw, w1w, w3w, w2w, kLimit);
        ok &= report("composed shared FFN vs oracle", want_out, widen(s.out), kFfnTol);

        // The activation must be doing something: a zeroed-gate implementation
        // would produce all-zero output and would pass any comparison with a
        // relative tolerance (dividing by a peak of zero). Measured absolutely.
        {
            double peak = 0.0;
            for (double v : want_out) peak = std::fmax(peak, std::fabs(v));
            ok &= check("output is not degenerate", peak > 1e-3,
                        "peak |out| = " + std::to_string(peak));
        }
    }

    // =====================================================================
    // C. Real artifact weights
    // =====================================================================
    std::cout << "\n--- C. real artifact shared expert (layers.3) ---\n";
    {
        const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
        aeon::core::AeonModelLoader loader;
        loader.open_model(model_dir);

        const auto& t1 = loader.get_tensor("layers.3.ffn.shared_experts.w1.weight");
        const auto& t3 = loader.get_tensor("layers.3.ffn.shared_experts.w3.weight");
        const auto& t2 = loader.get_tensor("layers.3.ffn.shared_experts.w2.weight");

        CHECK_HIP(hipMemcpy(d_w1, t1.data, static_cast<size_t>(t1.byte_size),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w3, t3.data, static_cast<size_t>(t3.byte_size),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w2, t2.data, static_cast<size_t>(t2.byte_size),
                            hipMemcpyHostToDevice));

        // Real activations are unit-RMS (this tensor consumes an RMSNorm output),
        // so the scale below is representative rather than arbitrary.
        std::vector<double> x_d(static_cast<size_t>(kHidden));
        for (int i = 0; i < kHidden; ++i) x_d[static_cast<size_t>(i)] = gen.symmetric(1.6);
        const std::vector<__half> x_h = to_half(x_d);
        const std::vector<double> xw = widen(x_h);
        CHECK_HIP(hipMemcpy(d_x, x_h.data(), kHidden * sizeof(__half), hipMemcpyHostToDevice));

        // Widen the real fp16 weights for the oracle.
        auto widen_fp16_tensor = [](const uint8_t* data, size_t count) {
            std::vector<double> out(count);
            for (size_t i = 0; i < count; ++i) {
                uint16_t bits = 0;
                std::memcpy(&bits, data + i * 2, sizeof(bits));
                out[i] = aeon::reference::half_bits_to_double(bits);
            }
            return out;
        };
        const std::vector<double> w1w =
            widen_fp16_tensor(t1.data, static_cast<size_t>(kIntermediate) * kHidden);
        const std::vector<double> w3w =
            widen_fp16_tensor(t3.data, static_cast<size_t>(kIntermediate) * kHidden);
        const std::vector<double> w2w =
            widen_fp16_tensor(t2.data, static_cast<size_t>(kHidden) * kIntermediate);

        const SharedStages s = run_shared();

        const std::vector<double> want_gate = aeon::reference::matvec(
            kIntermediate, kHidden, xw,
            [&](size_t o, size_t i) { return w1w[o * kHidden + i]; });
        const std::vector<double> want_up = aeon::reference::matvec(
            kIntermediate, kHidden, xw,
            [&](size_t o, size_t i) { return w3w[o * kHidden + i]; });
        ok &= report("real w1 projection vs oracle", want_gate, widen(s.gate), kGemvTol);
        ok &= report("real w3 projection vs oracle", want_up, widen(s.up), kGemvTol);

        std::vector<double> want_hidden(kIntermediate);
        for (int i = 0; i < kIntermediate; ++i) {
            want_hidden[i] = aeon::reference::clamped_swiglu(
                want_gate[i], want_up[i], kLimit, ClampMode::Asymmetric);
        }
        ok &= report("real hidden (clamped SwiGLU) vs oracle",
                     want_hidden, widen(s.hidden), kFfnTol);

        const std::vector<double> want_out = aeon::reference::dense_ffn(
            kIntermediate, kHidden, xw, w1w, w3w, w2w, kLimit);
        ok &= report("real composed shared FFN vs oracle", want_out, widen(s.out), kFfnTol);

        // =================================================================
        // C.1 The clamp rule on real weights
        // =================================================================
        // The shared path takes the same `activation_clamp` as the routed path
        // [V model.py:1016-1021]. Measure how far the clamp actually moves the
        // real data, and whether the asymmetry is reachable here at all.
        //
        // The first run of this gate asserted "the clamp is engaged" at unit-RMS
        // activation and failed: with these real weights the pre-activations stay
        // well inside +-10 there, so the clamp is genuinely inert. That is a
        // measurement about the data, not a defect, and reporting it as a failure
        // was the same mistake as asserting a mechanism fires without checking
        // that it can. So the scale is swept instead: the clamp is asserted to
        // engage only at a scale where it demonstrably does.
        const double scales[2] = {1.0, 8.0};
        for (int case_index = 0; case_index < 2; ++case_index) {
            const double scale = scales[case_index];
            std::vector<double> x_case(static_cast<size_t>(kHidden));
            for (int i = 0; i < kHidden; ++i) {
                x_case[static_cast<size_t>(i)] = x_d[static_cast<size_t>(i)] * scale;
            }
            const std::vector<__half> x_case_h = to_half(x_case);
            const std::vector<double> x_case_w = widen(x_case_h);
            CHECK_HIP(hipMemcpy(d_x, x_case_h.data(), kHidden * sizeof(__half),
                                hipMemcpyHostToDevice));
            const SharedStages sc = run_shared();

            const std::vector<double> g = aeon::reference::matvec(
                kIntermediate, kHidden, x_case_w,
                [&](size_t o, size_t i) { return w1w[o * kHidden + i]; });
            const std::vector<double> u = aeon::reference::matvec(
                kIntermediate, kHidden, x_case_w,
                [&](size_t o, size_t i) { return w3w[o * kHidden + i]; });

            size_t gate_above = 0, up_outside = 0, gate_below = 0;
            double gate_peak = 0.0, up_peak = 0.0;
            for (int i = 0; i < kIntermediate; ++i) {
                if (g[i] > kLimit) ++gate_above;
                if (std::fabs(u[i]) > kLimit) ++up_outside;
                if (g[i] < -kLimit) ++gate_below;
                gate_peak = std::fmax(gate_peak, std::fabs(g[i]));
                up_peak = std::fmax(up_peak, std::fabs(u[i]));
            }
            std::vector<double> want_h(kIntermediate), no_clamp(kIntermediate),
                symmetric(kIntermediate);
            for (int i = 0; i < kIntermediate; ++i) {
                want_h[i] = aeon::reference::clamped_swiglu(g[i], u[i], kLimit,
                                                            ClampMode::Asymmetric);
                no_clamp[i] = aeon::reference::clamped_swiglu(g[i], u[i], kLimit,
                                                              ClampMode::None);
                symmetric[i] = aeon::reference::clamped_swiglu(g[i], u[i], kLimit,
                                                               ClampMode::Symmetric);
            }
            const double no_clamp_delta =
                aeon::reference::compare(want_h, no_clamp, 1.0).max_abs;
            const double symmetric_delta =
                aeon::reference::compare(want_h, symmetric, 1.0).max_abs;

            ok &= report(case_index == 0 ? "real shared FFN vs oracle (scale 1x)"
                                         : "real shared FFN vs oracle (scale 8x)",
                         aeon::reference::dense_ffn(kIntermediate, kHidden, x_case_w,
                                                    w1w, w3w, w2w, kLimit),
                         widen(sc.out), kFfnTol);

            const std::string engagement =
                (case_index == 0 ? "1x: " : "8x: ") + std::to_string(gate_above) +
                " gate>+10, " + std::to_string(up_outside) + " |up|>10, " +
                std::to_string(gate_below) + " gate<-10; peaks " +
                std::to_string(gate_peak) + "/" + std::to_string(up_peak) +
                "; no-clamp delta " + std::to_string(no_clamp_delta);

            if (case_index == 0) {
                std::printf("  %-54s %-22s %s\n", "clamp engagement at 1x measured",
                            no_clamp_delta == 0.0 ? "clamp inert" : "clamp fires",
                            engagement.c_str());
            } else {
                ok &= check("the clamp is engaged at 8x on real weights",
                            no_clamp_delta > 1e-3, engagement);
                if (gate_below > 0) {
                    ok &= check("the asymmetric edge is reachable on real weights",
                                symmetric_delta > 1e-5,
                                "symmetric delta = " + std::to_string(symmetric_delta));
                } else {
                    std::printf("  %-54s %-22s %s\n",
                                "asymmetric edge not reached even at 8x",
                                "n/a (item 14 covers it)",
                                (std::to_string(gate_below) + " gate below -10").c_str());
                }
            }
        }

        // Restore the 1x activation for the combine section below, which uses
        // `d_out` as the shared contribution.
        CHECK_HIP(hipMemcpy(d_x, x_h.data(), kHidden * sizeof(__half),
                            hipMemcpyHostToDevice));
        (void)run_shared();

        // Sanity: the real weights must not be degenerate.
        size_t nonzero = 0;
        for (double v : w1w) {
            if (v != 0.0) ++nonzero;
        }
        ok &= check("real shared weights are non-degenerate",
                    nonzero > w1w.size() / 2,
                    std::to_string(nonzero * 100 / w1w.size()) + "% of w1 non-zero");
    }

    // =====================================================================
    // D. The combine: shared enters the MoE output exactly once
    // =====================================================================
    std::cout << "\n--- D. combine (shared enters exactly once) ---\n";
    {
        // Drive the MoE accumulation kernel the way the pipeline does, and show
        // that its output is `routed + shared` and not `routed`, `2*shared`, or
        // `routed + shared` twice.
        //
        // We cannot run real routed experts here (they need the expert pool), so
        // this uses a single synthetic expert via the same kernel. What is being
        // certified is the arithmetic of the combine, not the expert contents.
        constexpr int kExperts = 6;
        constexpr int kWaves = 8, kRpw = 8, kLpr = 4, kIters = 16;

        // A synthetic swizzled expert payload, encoded by the oracle. `shuffle`
        // varies it per expert so the sum is not a multiple of one term.
        const size_t payload_bytes = aeon::core::AEON_SWIZZLED_EXPERT_BYTES;
        std::vector<std::vector<uint8_t>> payloads(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            payloads[static_cast<size_t>(e)].assign(payload_bytes, 0);
            std::vector<double> w2(static_cast<size_t>(kHidden) * kIntermediate);
            for (double& v : w2) v = gen.symmetric(0.02);
            aeon::reference::swizzled_encode(payloads[static_cast<size_t>(e)].data(),
                                             aeon::reference::SwizzledKind::W2, w2);
        }

        __half* d_hidden_many = nullptr;
        __half* d_payloads = nullptr;
        float* d_f32 = nullptr;
        __half* d_combined = nullptr;
        __half* d_routed = nullptr;
        __half* d_shared = nullptr;
        float* d_weights6 = nullptr;
        int* d_counters = nullptr;

        const size_t accum_blocks = static_cast<size_t>(kHidden / (kWaves * kRpw));
        CHECK_HIP(hipMalloc(&d_hidden_many,
                            static_cast<size_t>(kExperts) * kIntermediate * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_payloads, kExperts * payload_bytes));
        CHECK_HIP(hipMalloc(&d_f32, kHidden * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_combined, kHidden * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_routed, kHidden * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_shared, kHidden * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_weights6, kExperts * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_counters, accum_blocks * sizeof(int)));

        aeon::kernel::SwizzledW2ExpertPtrs weights{};
        uint8_t* payload_base = reinterpret_cast<uint8_t*>(d_payloads);
        for (int e = 0; e < kExperts; ++e) {
            CHECK_HIP(hipMemcpy(payload_base + static_cast<size_t>(e) * aeon::core::AEON_EXPERT_BYTES,
                                payloads[static_cast<size_t>(e)].data(), payload_bytes,
                                hipMemcpyHostToDevice));
            const uint8_t* base = payload_base + static_cast<size_t>(e) * aeon::core::AEON_EXPERT_BYTES;
            weights.w2[e] = reinterpret_cast<const uint4*>(base + aeon::core::AEON_W2_PACKED_OFFSET);
            weights.s2[e] = reinterpret_cast<const half*>(base + aeon::core::AEON_W2_SCALE_OFFSET);
        }

        // Hidden activations, one row per expert.
        std::vector<double> hidden_d(static_cast<size_t>(kExperts) * kIntermediate);
        for (double& v : hidden_d) v = gen.symmetric(2.0);
        const std::vector<__half> hidden_h = to_half(hidden_d);
        CHECK_HIP(hipMemcpy(d_hidden_many, hidden_h.data(), hidden_h.size() * sizeof(__half),
                            hipMemcpyHostToDevice));

        const float weights6[kExperts] = {0.31f, 0.27f, 0.19f, 0.11f, 0.07f, 0.05f};
        CHECK_HIP(hipMemcpy(d_weights6, weights6, sizeof(weights6), hipMemcpyHostToDevice));

        // The shared expert output, produced by the real shared-expert kernels
        // rather than fabricated, so the combine is tested end to end.
        // (`d_out` still holds it from section C.)
        CHECK_HIP(hipMemcpy(d_shared, d_out, kHidden * sizeof(__half),
                            hipMemcpyDeviceToDevice));

        auto run_accum = [&](const __half* initial, __half* destination) {
            CHECK_HIP(hipMemset(d_f32, 0, kHidden * sizeof(float)));
            CHECK_HIP(hipMemset(d_counters, 0, accum_blocks * sizeof(int)));
            aeon::kernel::dispatch_aeon_moe_fused_w2_accum<kWaves, kRpw, kLpr, kIters>(
                d_hidden_many, weights, d_weights6, initial, d_f32, destination,
                d_counters, kExperts, kHidden, kIntermediate);
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipDeviceSynchronize());
        };

        run_accum(nullptr, d_routed);
        run_accum(d_shared, d_combined);

        std::vector<__half> routed_h(kHidden), combined_h(kHidden);
        CHECK_HIP(hipMemcpy(routed_h.data(), d_routed, kHidden * sizeof(__half),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(combined_h.data(), d_combined, kHidden * sizeof(__half),
                            hipMemcpyDeviceToHost));
        const std::vector<__half> shared_h = [&] {
            std::vector<__half> v(kHidden);
            CHECK_HIP(hipMemcpy(v.data(), d_shared, kHidden * sizeof(__half),
                                hipMemcpyDeviceToHost));
            return v;
        }();

        // The oracle answer: combined == routed + shared in double, from the
        // device's own routed and shared values.
        const std::vector<double> routed_d = widen(routed_h);
        const std::vector<double> shared_d = widen(shared_h);
        std::vector<double> want_combined(kHidden);
        for (size_t i = 0; i < static_cast<size_t>(kHidden); ++i) {
            want_combined[i] = routed_d[i] + shared_d[i];
        }
        ok &= report("combined MoE output == routed + shared",
                     want_combined, widen(combined_h), 2e-3);

        // Shared must not be applied twice, and must not be absent. Both are
        // checked, because either alone would pass a one-sided test.
        std::vector<double> want_twice(kHidden);
        for (size_t i = 0; i < static_cast<size_t>(kHidden); ++i) {
            want_twice[i] = routed_d[i] + 2.0 * shared_d[i];
        }
        const double to_twice =
            aeon::reference::compare(want_twice, widen(combined_h), 1.0).max_rel;
        const double to_routed_only =
            aeon::reference::compare(routed_d, widen(combined_h), 1.0).max_rel;
        const double shared_peak =
            aeon::reference::compare(shared_d, std::vector<double>(kHidden, 0.0), 1.0).max_rel;

        ok &= check("shared is not applied twice", to_twice > 1e-3,
                    "distance from routed+2*shared = " + std::to_string(to_twice));
        ok &= check("shared is not omitted", to_routed_only > 1e-3,
                    "distance from routed-only = " + std::to_string(to_routed_only) +
                        " (shared peak " + std::to_string(shared_peak) + ")");

        CHECK_HIP(hipFree(d_hidden_many));
        CHECK_HIP(hipFree(d_payloads));
        CHECK_HIP(hipFree(d_f32));
        CHECK_HIP(hipFree(d_combined));
        CHECK_HIP(hipFree(d_routed));
        CHECK_HIP(hipFree(d_shared));
        CHECK_HIP(hipFree(d_weights6));
        CHECK_HIP(hipFree(d_counters));
    }

    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_w1));
    CHECK_HIP(hipFree(d_w2));
    CHECK_HIP(hipFree(d_w3));
    CHECK_HIP(hipFree(d_gate));
    CHECK_HIP(hipFree(d_up));
    CHECK_HIP(hipFree(d_hidden));
    CHECK_HIP(hipFree(d_out));

    std::cout << (ok ? "\n[SUCCESS] Shared expert gate passed.\n"
                     : "\n[FAILURE] Shared expert gate failed.\n");
    return ok ? 0 : 1;
}
