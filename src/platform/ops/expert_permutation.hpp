#pragma once

// -----------------------------------------------------------------------------
// Model-agnostic primitive: the token -> expert permutation the grouped expert
// kernels consume.
//
// The grouped expert pair (`architecture/deepseek_v4/kernels/moe_grouped_dispatch.hpp`,
// composed from the G2 kernel and the G3 feed) runs one GEMM per expert with that
// expert's tokens as the M dimension, so it needs the chunk's `(token, slot)` draws
// gathered expert-contiguous: for each expert, the list of activation rows that routed
// to it and the output row each one writes.
//
// A counting sort builds that permutation, because the keys are expert ids in a dense
// range `[0, expert_count)` — histogram, prefix-sum the histogram into offsets, then
// place each draw at its expert's running cursor. The keys need no ordering by value,
// which is what makes a comparison sort (and its `O(d log d)` cost) unnecessary.
//
// Two properties the grouped pair relies on, and how they are kept:
//
//   * **expert order is ascending.** Expert `e` owns
//     `expert_offsets[e] .. expert_offsets[e+1)`. A caller wanting to hand one launch
//     a batch of experts rebases the offsets by `expert_offsets[first]`, and the
//     draws of a contiguous expert range are themselves contiguous — an absent expert
//     has zero width, not a missing slot, so it cannot break the range apart.
//   * **placement is deterministic.** The scatter visits draws in index order, so a
//     given input always yields the same `token_indices`. A byte-reproducible token
//     restore depends on the whole routed path being reproducible, and a race that
//     reordered the permutation would undo it.
//
// `token_indices` and `draw_indices` are deliberately two arrays and not one. A draw
// is a `(token, slot)` pair; the gate half reads activations by token while the output
// projection writes contributions by draw, so a kernel handed the wrong one scatters
// output rows across draws. Keeping both makes that confusion impossible to express.
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace aeon::kernel {

// One `int` of shared memory per expert in the histogram stage.
inline constexpr int kMaxPermutationExperts = 1024;

// Histogram and exclusive prefix sum, in one block: the offsets are a function of the
// whole histogram, so the scan has to see every count before any of them is written.
// The histogram itself is over shared memory with `atomicAdd`, which is safe here
// because only this one block exists.
__global__ void expert_permutation_offsets_kernel(
    const int* __restrict__ topk_indices,
    int draws,
    int expert_count,
    int* __restrict__ expert_offsets
) {
    extern __shared__ int histogram[];

    for (int expert = threadIdx.x; expert < expert_count; expert += blockDim.x) {
        histogram[expert] = 0;
    }
    __syncthreads();

    for (int draw = threadIdx.x; draw < draws; draw += blockDim.x) {
        const int expert = topk_indices[draw];
        // The router emits ids in `[0, expert_count)`; an id outside it would write
        // past the histogram, so it is dropped rather than trusted. The precondition
        // belongs to the caller, and the permutation gate asserts it.
        if (expert >= 0 && expert < expert_count) {
            atomicAdd(&histogram[expert], 1);
        }
    }
    __syncthreads();

    // One thread scans. `expert_count` is a few hundred, so the serial walk costs
    // nothing against the alternative's cross-thread communication.
    if (threadIdx.x == 0) {
        int running = 0;
        for (int expert = 0; expert < expert_count; ++expert) {
            expert_offsets[expert] = running;
            running += histogram[expert];
        }
        expert_offsets[expert_count] = running;
    }
}

// Places each draw in its expert's range. One block per expert, and within a block a
// single thread walks the draws in index order — parallelising the walk would let two
// threads claim positions out of order, and the deterministic placement above is worth
// more than the width, since the walk is one cached pass over the ids.
__global__ void expert_permutation_scatter_kernel(
    const int* __restrict__ topk_indices,
    const int* __restrict__ expert_offsets,
    int draws,
    int slots,
    int* __restrict__ token_indices,
    int* __restrict__ draw_indices
) {
    const int expert = blockIdx.x;
    int position = expert_offsets[expert];
    const int end = expert_offsets[expert + 1];

    for (int draw = 0; draw < draws && position < end; ++draw) {
        if (topk_indices[draw] == expert) {
            token_indices[position] = draw / slots;
            draw_indices[position] = draw;
            ++position;
        }
    }
}

// Reorders a per-draw array into per-position order: `out[p] = weights[draw_indices[p]]`.
//
// The grouped down kernel reads the routing weight by permutation **position**
// (`draw_weights[position]`), while the layer body holds the weights by **draw**
// (`weights[token * slots + slot]`). This is the one-line bridge between the two. Kept
// as a separate kernel rather than folding the index into the grouped kernel so that
// kernel's contract stays "both arrays are per position", as its oracle assumes.
__global__ void expert_permutation_gather_weights_kernel(
    const float* __restrict__ weights,
    const int* __restrict__ draw_indices,
    int draws,
    float* __restrict__ out
) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position < draws) {
        out[position] = weights[draw_indices[position]];
    }
}

// `topk_indices` is `[token_count * slots]`, row-major by token, so draw
// `token * slots + slot` selects `token`'s `slot`-th expert. `expert_offsets` is
// `[expert_count + 1]`; `token_indices` and `draw_indices` are `[token_count * slots]`.
// When `weights_by_draw`/`weights_by_position` are supplied, the weights are reordered
// into position order in the same dispatch.
inline void dispatch_expert_permutation(
    const int* topk_indices,
    int token_count,
    int slots,
    int expert_count,
    int* expert_offsets,
    int* token_indices,
    int* draw_indices,
    const float* weights_by_draw = nullptr,
    float* weights_by_position = nullptr,
    hipStream_t stream = 0
) {
    if (topk_indices == nullptr || expert_offsets == nullptr ||
        token_indices == nullptr || draw_indices == nullptr) {
        throw std::invalid_argument("dispatch_expert_permutation: null buffer");
    }
    if (token_count < 0 || slots <= 0) {
        throw std::invalid_argument("dispatch_expert_permutation: bad draw shape");
    }
    if (expert_count <= 0 || expert_count > kMaxPermutationExperts) {
        throw std::invalid_argument(
            "dispatch_expert_permutation: expert count out of range");
    }
    if ((weights_by_draw == nullptr) != (weights_by_position == nullptr)) {
        throw std::invalid_argument(
            "dispatch_expert_permutation: weights need both a source and a destination");
    }

    const int draws = token_count * slots;
    constexpr int kOffsetThreads = 256;
    expert_permutation_offsets_kernel
        <<<1, kOffsetThreads, expert_count * static_cast<int>(sizeof(int)), stream>>>(
            topk_indices, draws, expert_count, expert_offsets);
    expert_permutation_scatter_kernel<<<expert_count, 1, 0, stream>>>(
        topk_indices, expert_offsets, draws, slots, token_indices, draw_indices);
    if (weights_by_draw != nullptr && draws > 0) {
        const int blocks = (draws + kOffsetThreads - 1) / kOffsetThreads;
        expert_permutation_gather_weights_kernel<<<blocks, kOffsetThreads, 0, stream>>>(
            weights_by_draw, draw_indices, draws, weights_by_position);
    }
}

} // namespace aeon::kernel
