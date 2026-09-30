// -----------------------------------------------------------------------------
// Gate: the tiled causal attention *formulation*, before any tiled kernel exists.
//
// The per-query attention kernel sits on the dominant GPU term, and the way to
// make it fast is to give a whole query tile one shared key row-set and mask each
// query into it, instead of composing a private row-set per query. That
// reformulation is the entire semantic risk of the step — a wrong mask, or a
// window boundary off by one, changes the answer with no crash and no NaN — so it
// is pinned here, in fp64, before a kernel is built on it.
//
// Two independent implementations of the same quantity are compared:
//
//   A (current)  each query attends the **contiguous sub-range** of the union that
//                is exactly its own window, via `attention_scores_sink` — the
//                shipped reference, already certified against the scalar
//                `v4_sliding_window_attn_wave32_kernel` by
//                `test_v4_attention_sink_oracle`.
//   B (tiled)    each query attends the **whole union**, with every key outside its
//                window excluded by a mask. Written here from the masking rule, not
//                derived from A.
//
// Agreement is therefore evidence about the formulation, not self-consistency. The
// tiled kernel will be gated against A.
//
// What this deliberately does NOT test: the sink-in-the-max robustness property
// (unobservable in the output, as `test_v4_attention_sink_oracle` records), and the
// summation **order** across the union. A tile sums the union in key order while the
// per-query path sums only its window, so the two agree to fp64 rounding rather than
// bit-for-bit; the kernel's own comparison is at the greedy bar for the same reason.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/reference/dsv4_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

using aeon::reference::ErrorStats;

// Small shapes: this is a semantics gate, not a performance one. The head width
// stays real (512) because the scale is `1/sqrt(head_dim)` and a wrong scale is one
// of the things the window boundary is compared against.
constexpr size_t  kHeads     = 4;
constexpr size_t  kHeadDim   = 512;
constexpr int64_t kWindow    = 8;    // small, so the tile's queries have different windows
constexpr size_t  kTile      = 16;   // queries in the tile
constexpr size_t  kKeys      = 40;   // key rows in the sequence
constexpr double  kScale     = 0.04419417382415922; // 1 / sqrt(512), the full head
constexpr double  kExactTol  = 1e-12; // fp64 sum of the same terms, different order

// The union one tile needs: the earliest window-start among its queries through the
// last query. This is the bound the workspace must hold, and it is asserted below
// rather than assumed.
int64_t window_start(int64_t position) {
    return position >= kWindow - 1 ? position - (kWindow - 1) : 0;
}

// B — the tiled formulation: one shared union, a per-query mask. Written from the
// rule the kernel will use, `window_start(query) <= key_position <= query`.
std::vector<double> tiled_masked_attention(
    const std::vector<double>& query, size_t num_heads, size_t head_dim,
    const std::vector<double>& keys, const std::vector<int64_t>& key_positions,
    int64_t query_position, const std::vector<double>& sink, double scale) {
    std::vector<double> out(num_heads * head_dim, 0.0);
    const size_t num_keys = key_positions.size();

    for (size_t h = 0; h < num_heads; ++h) {
        const double* qh = query.data() + h * head_dim;

        std::vector<double> score(num_keys);
        std::vector<char> keep(num_keys, 0);
        double m = sink[h];
        for (size_t j = 0; j < num_keys; ++j) {
            const int64_t position = key_positions[j];
            const bool valid = position >= window_start(query_position) &&
                               position <= query_position;
            keep[j] = valid ? 1 : 0;
            if (!valid) continue;
            const double* kj = keys.data() + j * head_dim;
            double dot = 0.0;
            for (size_t d = 0; d < head_dim; ++d) dot += qh[d] * kj[d];
            score[j] = dot * scale;
            m = std::fmax(m, score[j]);
        }

        // Masked keys are skipped entirely, not exponentiated to zero: `exp(-inf)`
        // would be correct but the skip is what the kernel does, so skipping keeps
        // the two implementations comparable in the terms they actually add.
        double l = std::exp(sink[h] - m);
        std::vector<double> p(num_keys, 0.0);
        for (size_t j = 0; j < num_keys; ++j) {
            if (!keep[j]) continue;
            p[j] = std::exp(score[j] - m);
            l += p[j];
        }
        const double denom = std::fmax(l, 1e-30);

        double* oh = out.data() + h * head_dim;
        for (size_t d = 0; d < head_dim; ++d) {
            double acc = 0.0;
            for (size_t j = 0; j < num_keys; ++j) {
                if (!keep[j]) continue;
                acc += p[j] * keys[j * head_dim + d];
            }
            oh[d] = acc / denom;
        }
    }
    return out;
}

bool report(const char* label, const std::vector<double>& want,
            const std::vector<double>& got) {
    const ErrorStats s = aeon::reference::compare(want, got, 1.0);
    if (s.size_mismatch) {
        std::printf("  %-52s SIZE MISMATCH                       FAIL\n", label);
        return false;
    }
    const bool pass = std::isfinite(s.max_abs) && s.max_abs <= kExactTol;
    std::printf("  %-52s max_abs=%.3e  (peak=%.3e)  %s\n",
                label, s.max_abs, aeon::reference::peak_abs(want),
                pass ? "PASS" : "FAIL");
    return pass;
}

} // namespace

int main() {
    std::cout << "[Gate] Tier-1 formulation: tiled union + causal mask == per-query window\n";
    bool ok = true;

    aeon::reference::Rng gen(0x71EDCA71ull);

    // The key sequence: one shared KV head, as the model has.
    std::vector<double> keys(kKeys * kHeadDim);
    std::vector<int64_t> key_positions(kKeys);
    for (size_t j = 0; j < kKeys; ++j) {
        key_positions[j] = static_cast<int64_t>(j);
        for (size_t d = 0; d < kHeadDim; ++d) {
            keys[j * kHeadDim + d] = gen.symmetric(1.0);
        }
    }

    // Per-head sink logits, both signs, as the shipped gate uses.
    std::vector<double> sink(kHeads);
    for (size_t h = 0; h < kHeads; ++h) {
        sink[h] = (static_cast<int>(h % 7) - 3) * 0.9 + 0.25;
    }

    // The tile is the last `kTile` keys; its queries run over the union below.
    const int64_t tile_first = static_cast<int64_t>(kKeys) - static_cast<int64_t>(kTile);
    const int64_t union_first = window_start(tile_first);
    const size_t union_count = static_cast<size_t>(key_positions.back() - union_first + 1);

    // The union must be a contiguous run of the sequence, and must not be the first
    // `kTile` rows of it (which would make every query's window identical and the
    // gate vacuous).
    std::printf("--- A. the union is the tile's real working set ---\n");
    ok &= [&] {
        bool contiguous = true;
        for (size_t j = 0; j < union_count; ++j) {
            if (key_positions[static_cast<size_t>(union_first) + j] !=
                union_first + static_cast<int64_t>(j)) {
                contiguous = false;
            }
        }
        std::printf("  %-52s first=%lld count=%zu  %s\n",
                    "the union starts at the earliest window", (long long)union_first,
                    union_count, contiguous ? "PASS" : "FAIL");
        return contiguous;
    }();

    ok &= [&] {
        // `W + tile - 1` is the bound the workspace must size `composed_rows` to.
        const size_t bound = static_cast<size_t>(kWindow) + kTile - 1;
        const bool within = union_count <= bound;
        std::printf("  %-52s %zu <= W+tile-1 = %zu  %s\n",
                    "the union is bounded by W + tile - 1", union_count, bound,
                    within ? "PASS" : "FAIL");
        return within;
    }();

    ok &= [&] {
        // Only the tile's first query spans the whole union; every later query drops
        // the keys older than its own window, so the mask is genuinely exercised.
        size_t spanning = 0;
        for (size_t r = 0; r < kTile; ++r) {
            const int64_t q = tile_first + static_cast<int64_t>(r);
            if (window_start(q) == union_first) ++spanning;
        }
        const bool non_vacuous = spanning < kTile;
        std::printf("  %-52s %zu of %zu queries drop the oldest keys  %s\n",
                    "the mask excludes keys for most queries", kTile - spanning, kTile,
                    non_vacuous ? "PASS" : "FAIL");
        return non_vacuous;
    }();

    // --- B. the reformulation itself -----------------------------------------
    std::printf("--- B. masked union == per-query window (fp64) ---\n");
    // The union is the contiguous run of the sequence at [union_first, +union_count),
    // not the first `union_count` rows — the tile sits at the end of the sequence.
    std::vector<double> union_keys(keys.begin() + union_first * kHeadDim,
                                   keys.begin() + (union_first + union_count) * kHeadDim);
    std::vector<int64_t> union_positions(key_positions.begin() + union_first,
                                         key_positions.begin() + union_first + union_count);

    for (size_t r = 0; r < kTile; ++r) {
        const int64_t query_position = tile_first + static_cast<int64_t>(r);

        std::vector<double> query(kHeads * kHeadDim);
        for (double& value : query) value = gen.symmetric(1.0);

        // A — the contiguous sub-range of the union that is this query's window.
        const size_t local_first = static_cast<size_t>(window_start(query_position) - union_first);
        const size_t local_count = static_cast<size_t>(query_position - union_first + 1) - local_first;
        std::vector<double> local_keys(union_keys.begin() + local_first * kHeadDim,
                                       union_keys.begin() + (local_first + local_count) * kHeadDim);
        const std::vector<double> want = aeon::reference::attention_scores_sink(
            query, kHeads, kHeadDim, local_keys, local_count, sink, kScale);

        // B — the whole union with the mask.
        const std::vector<double> got = tiled_masked_attention(
            query, kHeads, kHeadDim, union_keys, union_positions, query_position,
            sink, kScale);

        char label[96];
        std::snprintf(label, sizeof(label), "query %2lld  (window %zu of %zu keys)",
                      (long long)query_position, local_count, union_count);
        ok &= report(label, want, got);
    }

    std::cout << (ok ? "[Tier-1 tiled formulation] PASS\n" : "[Tier-1 tiled formulation] FAIL\n");
    return ok ? 0 : 1;
}
