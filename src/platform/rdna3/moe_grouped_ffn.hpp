#pragma once

// -----------------------------------------------------------------------------
// G2 (architecture): grouped gated-FFN GEMM on the Wave32 WMMA unit.
//
// Two kernels: a gate/up projection pair with one fused two-operand activation, and
// a down projection with a fused per-draw scale. Both run one GEMM per expert over a
// token->expert permutation, so a weight byte is read once no matter how many tokens
// want that expert — the whole reason the grouped shape exists.
//
// This file is architecture-specific but names no format and no model. What it reads
// comes from two injected policies:
//
//   * `Feed` (a G3 concern) fills a K-tile of one expert's weights into LDS in the
//     `[K][N]` order `wmma_load_b_from_lds` expects, and states how many experts a
//     launch may address (`kMaxExperts`). The swizzled W4A16 layout is one such feed;
//     another format supplies its own without touching this file.
//   * `Epilogue` (a G4 concern) is the activation applied to the gate/up pair, and the
//     scaling applied to a down-projection result. The clamp, or its absence, is the
//     model's choice, not the kernel's.
//
// Keeping the three apart without splitting the kernel is the point: the dequantized
// slab is the expensive operand, and its reuse across token tiles only pays while the
// staging and the MMAs stay fused in registers and LDS. A policy is a compile-time
// type, so the fusion survives — a runtime call per tile would not.
//
// ## Tiling and the loop nest
//
//   * M tile = 16 tokens, one WMMA tile, matching the instruction.
//   * N tile = 16 output rows per wave, so one wave produces a whole 16x16 tile and
//     needs no cross-wave reduction.
//   * K moves in `kGroupedKBlock`-wide blocks, the granularity the feed fills LDS at.
//   * The nest is `K outer, M inner`: the slab for one K block is held across every
//     token tile of the window, so it is dequantized once per (K block, M window)
//     rather than once per (K block, M tile). `MTILES` is how many token tiles a
//     window holds. The bound on it is register pressure (two live f32_vec8 per tile),
//     not LDS — the numbers are measured in the A/B benchmark, not argued here.
//
// The slab is single-buffered. Double-buffering it (stage block n+1 while block n
// multiplies) was built and measured slower: the slab costs LDS per wave, and the
// occupancy lost outweighs the latency hidden (`6.2 → 8.8 ms` on the gate half).
//
// Partially filled M tiles (an expert whose token count is not a multiple of 16) are
// padded by repeating a resident token's row and masked at the store, rather than
// given a separate GEMV path the permutation can absorb.
//
// Two indexings travel through both kernels and are not the same: a **token** names a
// row of the chunk's activations, and a **draw** — a (token, slot) pair — names an
// output row. The gate reads activations by token; the down projection writes
// contributions by draw, one writer per element, so a higher-level reduce still sees a
// single contribution per slot.
// -----------------------------------------------------------------------------

#include "platform/rdna3/wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace aeon::rdna3 {

// K per LDS fill. With a 32-wide quantization group this is the block one workgroup
// fills with exactly one int4 load per thread, but the size is stated as a constant of
// this layer so a feed is written against it rather than against any one format's
// group width.
inline constexpr int kGroupedKBlock = 64;

// Gate/up projections for one expert group, one fused two-operand activation.
//
// `expert_offsets` and `token_indices` are the permutation: expert e owns
// `expert_offsets[e] .. expert_offsets[e+1]` of `token_indices`, and each entry is a
// row of `activation`. `Epilogue::gate_up(gate, up)` produces the stored value.
template <int WAVES, int MTILES, class Feed, class Epilogue>
__global__ __launch_bounds__(WAVES * 32)
void moe_grouped_gate_up_kernel(
    const half* __restrict__ activation,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ token_indices,
    typename Feed::GateUpWeights weights,
    half* __restrict__ expert_hidden,
    int expert_count,
    int expert_hidden_tokens,
    int N,
    int K,
    int activation_stride,
    Epilogue epilogue
) {
    constexpr int kNTile = WAVES * 16;
    constexpr int kM = 16;
    constexpr int kMWindow = MTILES * kM;

    const int expert = blockIdx.y;
    if (expert >= expert_count) {
        return;
    }

    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int lane_axis = wmma_lane_axis(lane);
    const int lane_parity = wmma_lane_parity(lane);
    // The LDS slab holds the whole workgroup's `kNTile` output rows and is filled
    // cooperatively; each wave then reads its own 16-row slice out of it.
    const int n_base = blockIdx.x * kNTile;
    const int n_wave = wave * kM;

    __shared__ half gate_slab[kGroupedKBlock * kNTile];
    __shared__ half up_slab[kGroupedKBlock * kNTile];

    const int first = expert_offsets[expert];
    const int count = expert_offsets[expert + 1] - first;
    // An expert with no draws (an expert of the layer the chunk never routed to) has
    // `first == expert_offsets[expert + 1]`, so `token_indices[first]` would read the
    // next expert's first entry or past the end. Guard it; the padding row is unused
    // when `count == 0` because the M loop below does not run.
    const int pad_row = count > 0 ? token_indices[first] : 0;

    const int iterations = (K / 32) / Feed::kLpr;

    for (int m_window = 0; m_window < count; m_window += kMWindow) {
        // A padded token tile repeats `pad_row`, which is always in range, so the A
        // fragment load never touches memory past the group's token list.
        const half* row_ptr[MTILES];
        #pragma unroll
        for (int tile = 0; tile < MTILES; ++tile) {
            const int row_in_group = m_window + tile * kM + lane_axis;
            const int activation_row =
                row_in_group < count ? token_indices[first + row_in_group] : pad_row;
            // `activation_stride` is the row pitch, `K` for a compact `[T, H]` batch
            // and larger when the caller's workspace pads each token to its own tile.
            // A stride instead of a gather keeps the copy out of the hot path.
            row_ptr[tile] = activation +
                            static_cast<size_t>(activation_row) * activation_stride;
        }

        f32_vec8 gate_accumulator[MTILES];
        f32_vec8 up_accumulator[MTILES];
        #pragma unroll
        for (int tile = 0; tile < MTILES; ++tile) {
            gate_accumulator[tile] = wmma_zero_accumulator();
            up_accumulator[tile] = wmma_zero_accumulator();
        }

        for (int k_base = 0; k_base < K; k_base += kGroupedKBlock) {
            // Once per K block, and the enclosing M loop is what makes that possible.
            Feed::fill_gate_up_slab(weights, expert, gate_slab, up_slab, kNTile, n_base,
                                    k_base, iterations);
            __syncthreads();

            #pragma unroll
            for (int k_sub = 0; k_sub < kGroupedKBlock / kWmmaTileK; ++k_sub) {
                const int k_offset = k_base + k_sub * kWmmaTileK;
                const f16_vec16 gate_fragment = wmma_load_b_from_lds(
                    gate_slab + k_sub * kWmmaTileK * kNTile, kNTile, n_wave + lane_axis);
                const f16_vec16 up_fragment = wmma_load_b_from_lds(
                    up_slab + k_sub * kWmmaTileK * kNTile, kNTile, n_wave + lane_axis);
                // The A fragment is per tile; the two B fragments above are shared by
                // every tile of the window, which is the reuse this nest buys.
                #pragma unroll
                for (int tile = 0; tile < MTILES; ++tile) {
                    const f16_vec16 a_fragment =
                        wmma_load_a_row(row_ptr[tile] + k_offset);
                    gate_accumulator[tile] =
                        wmma_mma(a_fragment, gate_fragment, gate_accumulator[tile]);
                    up_accumulator[tile] =
                        wmma_mma(a_fragment, up_fragment, up_accumulator[tile]);
                }
            }
            __syncthreads();
        }

        const int n = n_base + n_wave + lane_axis;
        #pragma unroll
        for (int tile = 0; tile < MTILES; ++tile) {
            #pragma unroll
            for (int slot = 0; slot < 8; ++slot) {
                const int m = m_window + tile * kM + 2 * slot + lane_parity;
                if (m < count) {
                    expert_hidden[(static_cast<size_t>(expert) * expert_hidden_tokens + m) * N +
                                  n] =
                        epilogue.gate_up(gate_accumulator[tile][slot],
                                         up_accumulator[tile][slot]);
                }
            }
        }
    }
}

// The down projection: `[tokens, in] -> [tokens, out]` per expert, the result passed
// through `Epilogue::down` and written to the draw's own output row.
//
// The output is indexed by **draw**, not token, so every element has exactly one
// writer — the property a slot-ordered summation downstream depends on. Row `m` of
// `expert_hidden` is where the gate kernel left this expert's `m`-th token, so the A
// rows are read by group position and only the store consults `draw_indices`.
template <int WAVES, int MTILES, class Feed, class Epilogue>
__global__ __launch_bounds__(WAVES * 32)
void moe_grouped_down_kernel(
    const half* __restrict__ expert_hidden,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ draw_indices,
    const float* __restrict__ draw_weights,
    typename Feed::DownWeights weights,
    float* __restrict__ contrib,
    int expert_count,
    int expert_hidden_tokens,
    int N,
    int K,
    Epilogue epilogue
) {
    constexpr int kNTile = WAVES * 16;
    constexpr int kM = 16;
    constexpr int kMWindow = MTILES * kM;

    const int expert = blockIdx.y;
    if (expert >= expert_count) {
        return;
    }

    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int lane_axis = wmma_lane_axis(lane);
    const int lane_parity = wmma_lane_parity(lane);
    const int n_base = blockIdx.x * kNTile;
    const int n_wave = wave * kM;

    __shared__ half down_slab[kGroupedKBlock * kNTile];

    const int first = expert_offsets[expert];
    const int count = expert_offsets[expert + 1] - first;
    const int iterations = (K / 32) / Feed::kLpr;

    for (int m_window = 0; m_window < count; m_window += kMWindow) {
        const half* row_ptr[MTILES];
        #pragma unroll
        for (int tile = 0; tile < MTILES; ++tile) {
            const int row_in_group = m_window + tile * kM + lane_axis;
            // The gate kernel writes only the group's first `count` rows, so the
            // padding rows of the last M tile have no defined value to read. Row 0
            // always exists and its products are discarded by the store below.
            const int m = row_in_group < count ? row_in_group : 0;
            row_ptr[tile] = expert_hidden +
                            (static_cast<size_t>(expert) * expert_hidden_tokens + m) * K;
        }

        f32_vec8 accumulator[MTILES];
        #pragma unroll
        for (int tile = 0; tile < MTILES; ++tile) {
            accumulator[tile] = wmma_zero_accumulator();
        }

        for (int k_base = 0; k_base < K; k_base += kGroupedKBlock) {
            Feed::fill_down_slab(weights, expert, down_slab, kNTile, n_base, k_base,
                                 iterations);
            __syncthreads();

            #pragma unroll
            for (int k_sub = 0; k_sub < kGroupedKBlock / kWmmaTileK; ++k_sub) {
                const int k_offset = k_base + k_sub * kWmmaTileK;
                const f16_vec16 b_fragment = wmma_load_b_from_lds(
                    down_slab + k_sub * kWmmaTileK * kNTile, kNTile, n_wave + lane_axis);
                #pragma unroll
                for (int tile = 0; tile < MTILES; ++tile) {
                    const f16_vec16 a_fragment =
                        wmma_load_a_row(row_ptr[tile] + k_offset);
                    accumulator[tile] =
                        wmma_mma(a_fragment, b_fragment, accumulator[tile]);
                }
            }
            __syncthreads();
        }

        const int n = n_base + n_wave + lane_axis;
        #pragma unroll
        for (int tile = 0; tile < MTILES; ++tile) {
            #pragma unroll
            for (int slot = 0; slot < 8; ++slot) {
                const int row = m_window + tile * kM + 2 * slot + lane_parity;
                if (row < count) {
                    const int position = first + row;
                    contrib[static_cast<size_t>(draw_indices[position]) * N + n] =
                        epilogue.down(accumulator[tile][slot], draw_weights[position]);
                }
            }
        }
    }
}

template <int WAVES, int MTILES, class Feed, class Epilogue>
inline void dispatch_moe_grouped_gate_up(
    const half* activation,
    const int* expert_offsets,
    const int* token_indices,
    const typename Feed::GateUpWeights& weights,
    half* expert_hidden,
    int expert_count,
    int expert_hidden_tokens,
    int N,
    int K,
    int activation_stride,
    Epilogue epilogue,
    hipStream_t stream = 0
) {
    static_assert(WAVES > 0 && MTILES > 0, "WAVES and MTILES must be positive");
    if (expert_count <= 0 || expert_count > Feed::kMaxExperts) {
        throw std::invalid_argument("dispatch_moe_grouped_gate_up: expert count out of range");
    }
    if (N <= 0 || N % (WAVES * 16) != 0 || K <= 0 || K % kGroupedKBlock != 0 ||
        (K / 32) % Feed::kLpr != 0 || activation_stride < K) {
        throw std::invalid_argument("dispatch_moe_grouped_gate_up: incompatible N/K shape");
    }

    const dim3 block(WAVES * 32);
    const dim3 grid(N / (WAVES * 16), expert_count);
    moe_grouped_gate_up_kernel<WAVES, MTILES, Feed, Epilogue><<<grid, block, 0, stream>>>(
        activation, expert_offsets, token_indices, weights, expert_hidden, expert_count,
        expert_hidden_tokens, N, K, activation_stride, epilogue);
}

template <int WAVES, int MTILES, class Feed, class Epilogue>
inline void dispatch_moe_grouped_down(
    const half* expert_hidden,
    const int* expert_offsets,
    const int* draw_indices,
    const float* draw_weights,
    const typename Feed::DownWeights& weights,
    float* contrib,
    int expert_count,
    int expert_hidden_tokens,
    int N,
    int K,
    Epilogue epilogue,
    hipStream_t stream = 0
) {
    static_assert(WAVES > 0 && MTILES > 0, "WAVES and MTILES must be positive");
    if (expert_count <= 0 || expert_count > Feed::kMaxExperts) {
        throw std::invalid_argument("dispatch_moe_grouped_down: expert count out of range");
    }
    if (N <= 0 || N % (WAVES * 16) != 0 || K <= 0 || K % kGroupedKBlock != 0 ||
        (K / 32) % Feed::kLpr != 0 || contrib == nullptr) {
        throw std::invalid_argument("dispatch_moe_grouped_down: incompatible N/K shape");
    }

    const dim3 block(WAVES * 32);
    const dim3 grid(N / (WAVES * 16), expert_count);
    moe_grouped_down_kernel<WAVES, MTILES, Feed, Epilogue><<<grid, block, 0, stream>>>(
        expert_hidden, expert_offsets, draw_indices, draw_weights, weights, contrib,
        expert_count, expert_hidden_tokens, N, K, epilogue);
}

} // namespace aeon::rdna3
