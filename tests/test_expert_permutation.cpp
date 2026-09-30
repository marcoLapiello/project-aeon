// -----------------------------------------------------------------------------
// Gate: the device token -> expert permutation, versus independent invariants.
//
// The grouped expert pair (`test_v4_grouped_wmma_oracle`) is certified against an
// fp64 reference with a hand-built permutation, so it proves the kernels honour a
// permutation. Where the permutation itself comes from was, until this gate, not
// proved anywhere. That seam fails quietly: an off-by-one in the offsets, or a
// swapped `token`/`draw`, still yields a permutation — just not the right one — and
// the grouped kernels would compute a different function with it, without a fault.
//
// The checks below are the definition of the permutation rather than a second copy
// of the kernel's algorithm, so none of them is circular:
//
//   * offsets are non-decreasing, start at 0 and end at the draw count;
//   * every draw sits under its own expert, and its token is `draw / slots`;
//   * `draw_indices` is a bijection onto `0 .. draws-1` (nothing dropped, nothing
//     duplicated);
//   * placement is in ascending draw order within each expert, so the permutation
//     is a function of the input and not of a race;
//   * each expert's width equals the independently counted histogram.
//
// The shapes stress what a single fixture would miss: an expert with no draws, an
// expert wider than one 16-row M tile, a one-token chunk, a `draws = 0` chunk, and a
// small expert count where every expert appears.
// -----------------------------------------------------------------------------

#include "platform/rdna3/device.hpp"
#include "platform/ops/expert_permutation.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// Small deterministic generator; the exact stream does not matter, only that it is
// skewed enough to leave experts empty and to make some experts very wide.
struct Lcg {
    uint64_t state;
    uint32_t next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(state >> 33);
    }
};

// One token's `slots` routed experts, skewing most draws onto a small hot prefix so
// the distribution has both empty experts and very wide ones.
std::vector<int> make_ids(int token_count, int slots, int expert_count, uint64_t seed) {
    const int hot = std::max(1, expert_count / 16);
    Lcg rng{seed};
    std::vector<int> ids;
    ids.reserve(static_cast<size_t>(token_count) * slots);
    for (int token = 0; token < token_count; ++token) {
        int chosen[64];
        int found = 0;
        while (found < slots) {
            const int expert = (rng.next() % 100 < 70)
                ? static_cast<int>(rng.next() % hot)
                : static_cast<int>(rng.next() % expert_count);
            bool duplicate = false;
            for (int i = 0; i < found; ++i) {
                if (chosen[i] == expert) { duplicate = true; break; }
            }
            if (!duplicate) chosen[found++] = expert;
        }
        for (int slot = 0; slot < slots; ++slot) ids.push_back(chosen[slot]);
    }
    return ids;
}

bool check_case(const char* label, int token_count, int slots, int expert_count,
                uint64_t seed) {
    const int draws = token_count * slots;
    const std::vector<int> ids = make_ids(token_count, slots, expert_count, seed);

    int* d_ids = nullptr;
    int* d_offsets = nullptr;
    int* d_tokens = nullptr;
    int* d_draws = nullptr;
    CHECK_HIP(hipMalloc(&d_ids, std::max<int>(1, draws) * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_offsets, (expert_count + 1) * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_tokens, std::max<int>(1, draws) * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_draws, std::max<int>(1, draws) * sizeof(int)));
    if (draws > 0) {
        CHECK_HIP(hipMemcpy(d_ids, ids.data(), draws * sizeof(int),
                            hipMemcpyHostToDevice));
    }

    kernel::dispatch_expert_permutation(d_ids, token_count, slots, expert_count,
                                        d_offsets, d_tokens, d_draws);
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<int> offsets(expert_count + 1);
    std::vector<int> token_of(draws);
    std::vector<int> draw_of(draws);
    CHECK_HIP(hipMemcpy(offsets.data(), d_offsets, offsets.size() * sizeof(int),
                        hipMemcpyDeviceToHost));
    if (draws > 0) {
        CHECK_HIP(hipMemcpy(token_of.data(), d_tokens, draws * sizeof(int),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(draw_of.data(), d_draws, draws * sizeof(int),
                            hipMemcpyDeviceToHost));
    }

    // Independent histogram, counted straight from the ids.
    std::vector<int> histogram(expert_count, 0);
    for (int draw = 0; draw < draws; ++draw) {
        if (ids[draw] >= 0 && ids[draw] < expert_count) ++histogram[ids[draw]];
    }

    int failures = 0;
    auto fail = [&](const char* what) {
        std::printf("    %-56s FAIL\n", what);
        ++failures;
    };

    // 1. offsets: monotone, anchored at 0 and at the draw count, widths = histogram.
    if (offsets[0] != 0 || offsets[expert_count] != draws) {
        fail("offsets anchored at 0 and at the draw count");
    }
    for (int expert = 0; expert < expert_count; ++expert) {
        if (offsets[expert + 1] < offsets[expert]) {
            fail("offsets are non-decreasing");
            break;
        }
        if (offsets[expert + 1] - offsets[expert] != histogram[expert]) {
            fail("expert width equals the counted histogram");
            break;
        }
    }

    // 2. every draw is under its own expert, with token = draw / slots.
    for (int expert = 0; expert < expert_count; ++expert) {
        bool mismatch = false;
        for (int position = offsets[expert]; position < offsets[expert + 1]; ++position) {
            const int draw = draw_of[position];
            if (draw < 0 || draw >= draws || ids[draw] != expert ||
                token_of[position] != draw / slots) {
                mismatch = true;
                break;
            }
        }
        if (mismatch) { fail("each draw is under its own expert, token = draw/slots"); break; }
    }

    // 3. `draw_indices` is a bijection: every draw placed exactly once.
    std::vector<char> seen(draws, 0);
    bool bijection = true;
    for (int position = 0; position < draws; ++position) {
        const int draw = draw_of[position];
        if (draw < 0 || draw >= draws || seen[draw]) { bijection = false; break; }
        seen[draw] = 1;
    }
    if (!bijection) fail("draw_indices is a bijection onto 0..draws-1");

    // 4. ascending draw order within each expert, i.e. deterministic placement.
    for (int expert = 0; expert < expert_count; ++expert) {
        bool ordered = true;
        for (int position = offsets[expert] + 1; position < offsets[expert + 1]; ++position) {
            if (draw_of[position] <= draw_of[position - 1]) { ordered = false; break; }
        }
        if (!ordered) { fail("placement is in ascending draw order"); break; }
    }

    int widest = 0;
    int present = 0;
    for (int expert = 0; expert < expert_count; ++expert) {
        const int width = offsets[expert + 1] - offsets[expert];
        widest = std::max(widest, width);
        if (width > 0) ++present;
    }
    const char* verdict = failures == 0 ? "PASS" : "FAIL";
    std::printf("  %-40s T=%-5d draws=%-6d experts=%-4d present=%-4d widest=%-4d %s\n",
                label, token_count, draws, expert_count, present, widest, verdict);

    CHECK_HIP(hipFree(d_ids));
    CHECK_HIP(hipFree(d_offsets));
    CHECK_HIP(hipFree(d_tokens));
    CHECK_HIP(hipFree(d_draws));
    return failures == 0;
}

} // namespace

int main() {
    std::cout << "[Gate] token -> expert permutation, versus independent invariants\n";
    aeon::core::select_compute_device(true);

    bool ok = true;
    // A one-token chunk: the minimum the grouped path ever dispatches.
    ok &= check_case("single token", 1, 6, 256, 0x5EED0001ull);
    // A few tokens over many experts: mostly empty experts, some sharing.
    ok &= check_case("few tokens", 7, 6, 256, 0x5EED0002ull);
    // A chunk wide enough that hot experts exceed one 16-row M tile.
    ok &= check_case("wide chunk, skewed", 300, 6, 256, 0x5EED0003ull);
    // Small expert count: every expert present.
    ok &= check_case("small expert count", 64, 6, 8, 0x5EED0004ull);
    // Degenerate: no draws at all.
    ok &= check_case("no draws", 0, 6, 256, 0x5EED0005ull);

    if (!ok) {
        std::cout << "[FAIL] token -> expert permutation violates its invariants\n";
        return 1;
    }
    std::cout << "[PASS] token -> expert permutation holds every invariant\n";
    return 0;
}
