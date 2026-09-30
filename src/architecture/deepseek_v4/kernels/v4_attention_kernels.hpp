#pragma once

// The DeepSeek-V4 attention kernels by concern: the sliding-window, cached
// sliding-window and cached (local + compressed) attention, the compressor state
// save and compressed-entry materialization, and the indexer score path. Split
// out of `v4_attention.hpp`, which includes this header for its callers.

#include "architecture/deepseek_v4/kernels/v4_attention_config.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>

namespace aeon::kernel {

// Causal sliding-window attention kernel with attention sink (Wave32).
// Q: [num_tokens, 64, 512]
// K: [num_tokens, 512] (Single KV head shared across all 64 Q heads)
// Out: [num_tokens, 64, 512]
// sink: [64] (float per head)
// Window size: W = 128
__global__ void __launch_bounds__(32) v4_sliding_window_attn_wave32_kernel(
    const __half* __restrict__ q,       // [T, 64, 512]
    const __half* __restrict__ k,       // [T, 512]
    const float*  __restrict__ attn_sink,// [64]
    __half*       __restrict__ out,     // [T, 64, 512]
    int total_tokens,
    int window_size,                    // 128
    float scale                         // 1.0f / sqrt(512)
) {
    int head = blockIdx.x;              // 0..63
    int token = blockIdx.y;             // 0..total_tokens-1
    int lane = threadIdx.x;             // 0..31

    // LDS storage for up to 128 attention scores in the sliding window
    __shared__ float lds_scores[DSV4_SLIDING_WINDOW];

    int j_start = max(0, token - window_size + 1);
    int num_keys = token - j_start + 1;

    const __half* q_ptr = q + token * (DSV4_NUM_HEADS * DSV4_HEAD_DIM) + head * DSV4_HEAD_DIM;

    // Phase 1: Compute scaled dot products Q_i * K_j for each key j in sliding window
    for (int step = 0; step < num_keys; ++step) {
        int j = j_start + step;
        const __half* k_ptr = k + j * DSV4_HEAD_DIM;

        float dot = 0.0f;
        // Each of the 32 threads computes 512 / 32 = 16 elements
        #pragma unroll 4
        for (int d = lane * 16; d < (lane + 1) * 16; ++d) {
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor(dot, offset, 32);
        }

        if (lane == 0) {
            lds_scores[step] = dot * scale;
        }
    }
    __syncthreads();

    // Phase 2: Softmax with Attention Sink
    // Find maximum among all key scores and the head attention sink
    float max_score = attn_sink[head];
    for (int step = 0; step < num_keys; ++step) {
        max_score = fmaxf(max_score, lds_scores[step]);
    }

    // Compute denominator: sum of exp(score - max) + exp(sink - max)
    float sink_weight = expf(attn_sink[head] - max_score);
    float sum_exp = sink_weight;

    for (int step = 0; step < num_keys; ++step) {
        float p = expf(lds_scores[step] - max_score);
        lds_scores[step] = p; // Store unnormalized exp
        sum_exp += p;
    }

    float inv_sum = 1.0f / fmaxf(sum_exp, 1e-30f);
    __syncthreads();

    // Phase 3: Weighted sum of Value vectors (V = K in DeepSeek MLA)
    // Note: The attention sink contributes only to the denominator, absorbing probability mass!
    __half* out_ptr = out + token * (DSV4_NUM_HEADS * DSV4_HEAD_DIM) + head * DSV4_HEAD_DIM;

    #pragma unroll 4
    for (int d = lane * 16; d < (lane + 1) * 16; ++d) {
        float acc = 0.0f;
        for (int step = 0; step < num_keys; ++step) {
            int j = j_start + step;
            float weight = lds_scores[step] * inv_sum;
            acc += weight * __half2float(k[j * DSV4_HEAD_DIM + d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}

// Autoregressive sliding-window attention with persistent KV cache.
__global__ void __launch_bounds__(32) v4_cached_sliding_window_attn_wave32_kernel(
    const __half* __restrict__ q,          // [64, 512]
    const __half* __restrict__ key_cache,  // [window_size, 512]
    const __half* __restrict__ value_cache,// [window_size, 512]
    const int64_t* __restrict__ positions, // [window_size]
    const float*  __restrict__ attn_sink,  // [64]
    __half*       __restrict__ out,        // [64, 512]
    int current_pos,                       // sequence index (0, 1, 2, ...)
    int window_size,                       // 128
    float scale                            // 1.0f / sqrt(512)
) {
    int head = blockIdx.x;                // 0..63
    int lane = threadIdx.x;               // 0..31

    __shared__ float lds_scores[DSV4_SLIDING_WINDOW];

    const int j_start = max(0, current_pos - window_size + 1);
    const __half* q_ptr = q + head * DSV4_HEAD_DIM;
    constexpr int kLaneElements = static_cast<int>(DSV4_HEAD_DIM) / 32;

    // The query is loop-invariant, so it is read once into registers. The old loop
    // re-read it from L1 for every one of the `window_size` slots — `128` redundant
    // `q` walks per head — and hoisting is what makes the staging loop below a
    // key-only stream.
    float q_reg[kLaneElements];
    #pragma unroll
    for (int i = 0; i < kLaneElements; ++i) {
        q_reg[i] = __half2float(q_ptr[lane * kLaneElements + i]);
    }

    // Phase 1: Dot products with valid cached ring slots.
    for (int slot = 0; slot < window_size; ++slot) {
        const int64_t key_position = positions[slot];
        const bool valid = key_position >= static_cast<int64_t>(j_start) &&
                           key_position <= static_cast<int64_t>(current_pos);
        const __half* k_ptr = key_cache + slot * DSV4_HEAD_DIM;

        float dot = 0.0f;
        if (valid) {
            #pragma unroll
            for (int i = 0; i < kLaneElements; ++i) {
                dot += q_reg[i] * __half2float(k_ptr[lane * kLaneElements + i]);
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_xor(dot, offset, 32);
            }
        }

        if (lane == 0) {
            lds_scores[slot] = valid ? dot * scale : -INFINITY;
        }
    }
    __syncthreads();

    // Phase 2: Softmax with attention sink, lane-strided.
    //
    // Only the two order-exact-or-cheap halves are parallelised. The max is a tree
    // reduction — `fmaxf` is associative and exact, so the result is unchanged. The
    // `expf` is strided across the lanes instead of issued in lockstep by all of
    // them, which is where the `window_size`-fold transcendental redundancy goes.
    // The denominator is deliberately left a single ascending walk: its terms sit at
    // array positions that depend on the caller's row count (decode passes the full
    // ring, a chunk passes its composed row count), so a tree sum would regroup them
    // and break bit-identity against the per-token path. A sequential `fadd` is
    // cheap; the `expf` that fed it was not.
    float thread_max = attn_sink[head];
    for (int slot = lane; slot < window_size; slot += 32) {
        thread_max = fmaxf(thread_max, lds_scores[slot]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_xor(thread_max, offset, 32));
    }
    const float max_score = thread_max;

    for (int slot = lane; slot < window_size; slot += 32) {
        lds_scores[slot] = expf(lds_scores[slot] - max_score);
    }
    __syncthreads();

    // A masked slot holds `-INFINITY`, so its `expf` is exactly `0`.
    float sum_exp = expf(attn_sink[head] - max_score);
    for (int slot = 0; slot < window_size; ++slot) {
        sum_exp += lds_scores[slot];
    }
    const float inv_sum = 1.0f / fmaxf(sum_exp, 1e-30f);
    __syncthreads();

    // Phase 3: Weighted sum of value vectors.
    __half* out_ptr = out + head * DSV4_HEAD_DIM;

    #pragma unroll 4
    for (int d = lane * 16; d < (lane + 1) * 16; ++d) {
        float acc = 0.0f;
        for (int slot = 0; slot < window_size; ++slot) {
            const float weight = lds_scores[slot] * inv_sum;
            acc += weight * __half2float(value_cache[slot * DSV4_HEAD_DIM + d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}

// Save one compressor or indexer partial row with its APE-adjusted score.
// The state is a position-addressed ring. The following materialization kernel
// runs on the same stream, so no device-side barrier is required between them.
__global__ void v4_save_compressor_state_kernel(
    const __half* __restrict__ kv,
    const __half* __restrict__ score,
    float* __restrict__ partial_kv,
    float* __restrict__ partial_score,
    int64_t* __restrict__ partial_positions,
    const float* __restrict__ ape,
    int64_t position,
    int ratio,
    int partial_capacity,
    int width
) {
    const int slot = static_cast<int>(position % partial_capacity);
    if (threadIdx.x == 0) {
        partial_positions[slot] = position;
    }

    const size_t row_offset = static_cast<size_t>(slot) * static_cast<size_t>(width);
    const size_t ape_offset = static_cast<size_t>(position % ratio) * static_cast<size_t>(width);
    for (int dimension = threadIdx.x; dimension < width; dimension += blockDim.x) {
        partial_kv[row_offset + static_cast<size_t>(dimension)] = __half2float(kv[dimension]);
        partial_score[row_offset + static_cast<size_t>(dimension)] =
            __half2float(score[dimension]) + ape[ape_offset + static_cast<size_t>(dimension)];
    }
}

// Materialize a completed C4/C128 compressed entry. The reduction is
// intentionally simple and float32: each output dimension independently
// softmaxes the compressor scores over the causal window, then block 0
// performs the small RMS reduction before the normalized row is stored.
__global__ void v4_materialize_compressed_entry_kernel(
    const float* __restrict__ partial_kv,
    const float* __restrict__ partial_score,
    const int64_t* __restrict__ partial_positions,
    const __half* __restrict__ norm,
    __half* __restrict__ compressed_key,
    __half* __restrict__ compressed_value,
    int64_t* __restrict__ compressed_positions,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int64_t boundary_position,
    int ratio,
    int partial_capacity,
    int head_dim,
    int width,
    int compressed_index,
    int nope_dim,
    int rope_dim,
    float rms_eps
) {
    __shared__ float raw[DSV4_HEAD_DIM];
    __shared__ float inverse_rms;

    const int dimension = threadIdx.x;
    const int coefficient = width / head_dim;
    const int window = coefficient * ratio;
    const int64_t rope_position = (boundary_position / ratio) * ratio;

    if (dimension < head_dim) {
        float maximum = -3.402823466e+38F;
        bool has_value = false;
        for (int offset = 0; offset < window; ++offset) {
            const int64_t source_position = boundary_position - window + 1 + offset;
            if (source_position < 0) continue;
            const int slot = static_cast<int>(source_position % partial_capacity);
            if (partial_positions[slot] != source_position) continue;
            const int segment = offset / ratio;
            maximum = fmaxf(
                maximum,
                partial_score[static_cast<size_t>(slot) * static_cast<size_t>(width) +
                              static_cast<size_t>(segment * head_dim + dimension)]);
            has_value = true;
        }

        float compressed = 0.0f;
        if (has_value) {
            float denominator = 0.0f;
            for (int offset = 0; offset < window; ++offset) {
                const int64_t source_position = boundary_position - window + 1 + offset;
                if (source_position < 0) continue;
                const int slot = static_cast<int>(source_position % partial_capacity);
                if (partial_positions[slot] != source_position) continue;
                const int segment = offset / ratio;
                const float score = partial_score[
                    static_cast<size_t>(slot) * static_cast<size_t>(width) +
                    static_cast<size_t>(segment * head_dim + dimension)];
                denominator += expf(score - maximum);
            }
            if (denominator > 0.0f) {
                for (int offset = 0; offset < window; ++offset) {
                    const int64_t source_position = boundary_position - window + 1 + offset;
                    if (source_position < 0) continue;
                    const int slot = static_cast<int>(source_position % partial_capacity);
                    if (partial_positions[slot] != source_position) continue;
                    const int segment = offset / ratio;
                    const size_t source_offset =
                        static_cast<size_t>(slot) * static_cast<size_t>(width) +
                        static_cast<size_t>(segment * head_dim + dimension);
                    const float score = partial_score[source_offset];
                    compressed += expf(score - maximum) / denominator * partial_kv[source_offset];
                }
            }
        }
        raw[dimension] = compressed;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum_sq = 0.0f;
        for (int index = 0; index < head_dim; ++index) sum_sq += raw[index] * raw[index];
        inverse_rms = rsqrtf(sum_sq / static_cast<float>(head_dim) + rms_eps);
        compressed_positions[compressed_index] = boundary_position;
    }
    __syncthreads();

    if (dimension < head_dim) {
        float value = raw[dimension] * inverse_rms * __half2float(norm[dimension]);
        if (dimension >= nope_dim && dimension < nope_dim + rope_dim) {
            const int pair = (dimension - nope_dim) / 2;
            const float cosine = cos_cache[rope_position * (rope_dim / 2) + pair];
            const float sine = sin_cache[rope_position * (rope_dim / 2) + pair];
            const float partner = raw[dimension + ((dimension - nope_dim) % 2 == 0 ? 1 : -1)] *
                                  inverse_rms * __half2float(norm[dimension + ((dimension - nope_dim) % 2 == 0 ? 1 : -1)]);
            value = ((dimension - nope_dim) % 2 == 0)
                ? value * cosine - partner * sine
                : value * cosine + partner * sine;
        }
        const size_t output_offset = static_cast<size_t>(compressed_index) * static_cast<size_t>(head_dim) +
                                     static_cast<size_t>(dimension);
        compressed_key[output_offset] = __float2half(value);
        compressed_value[output_offset] = __float2half(value);
    }
}

// Float32 lightning-indexer score path. One thread owns one compressed
// candidate.
//
//   score[c] = Σ_h w[h] · relu( q[h] · k[c] ) · softmax_scale · head_scale
//
// THE RELU IS ON THE PER-HEAD DOT, BEFORE THE WEIGHTING — not on the sum, and not
// after the weight. A missing ReLU still yields a plausible attention score, so
// `tests/test_v4_indexer_oracle.cpp` is what catches it (a missing ReLU gave 65 of
// 512 wrong top-k indices). Reference `[V sglang .../dsv4/indexer.py:119-124]`:
//   `score = bmm(kv, q.T); score = F.relu(score); score = score * weight;
//    score = score.sum(dim=2)`
__global__ void v4_indexer_scores_kernel(
    const __half* __restrict__ query,
    const float* __restrict__ weights,
    const __half* __restrict__ key_cache,
    float* __restrict__ scores,
    int candidate_count,
    int num_heads,
    int head_dim,
    float softmax_scale,
    float head_scale
) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    float score = 0.0f;
    const __half* key = key_cache + static_cast<size_t>(candidate) * static_cast<size_t>(head_dim);
    for (int head = 0; head < num_heads; ++head) {
        float dot = 0.0f;
        const size_t head_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
        for (int dimension = 0; dimension < head_dim; ++dimension) {
            dot += __half2float(query[head_offset + static_cast<size_t>(dimension)]) *
                   __half2float(key[static_cast<size_t>(dimension)]);
        }
        // ReLU here, per head, before the weight.
        score += fmaxf(dot, 0.0f) * weights[head] * softmax_scale * head_scale;
    }
    scores[candidate] = score;
}

// Assembles one query's local row-set for the compressed classes: the pre-chunk ring
// rows still inside the query's window, then the chunk's own rows up to and including
// the query. **Rows are ordered by ring slot (`position mod capacity`), not by
// position**, because that is the order decode's kernel iterates its ring in — so
// composing in that order is what makes `chunk ≡ serial` an equality rather than a
// tolerance. For a wrapped window the two orders are a rotation of each other, which
// is a small floating-point difference in the softmax and a hard gate failure.
//
// The host used to build this row-set with two `hipMemcpyAsync` per ring slot per
// query (~`256` submissions per token, the largest single cost in the prefill). The
// row-set has a closed form, so it needs no loop: **row `r` is slot `r`**, holding
// position `first + ((r - first) mod capacity)`. One block per row reads the ring or
// the chunk buffer by that formula.
//
// `first` is the window's first position; `rows` is its length (at most `capacity`,
// counted down from `query_position`). `start_position` is the chunk's first position:
// a position below it lives in the ring (the chunk has not written it), one at or
// above it in the chunk buffer. Every row emitted has `position <= query_position`, so
// the chunk index it reads is one the chunk has already written.
__global__ void v4_compose_local_rows_kernel(
    const __half* __restrict__ ring_keys,   // [capacity, head_dim]
    const __half* __restrict__ chunk_keys,  // [chunk_count, head_dim]
    __half* __restrict__ out_keys,          // [rows, head_dim]
    int64_t* __restrict__ out_positions,    // [rows]
    int64_t first,
    int64_t start_position,
    int capacity,
    int head_dim
) {
    const int row = blockIdx.x;
    // Slot `row`'s position inside the window, for a window that starts at `first`.
    int64_t offset = (static_cast<int64_t>(row) - first) % capacity;
    if (offset < 0) offset += capacity;
    const int64_t position = first + offset;
    const __half* source = position < start_position
        ? ring_keys + static_cast<size_t>(row) * static_cast<size_t>(head_dim)
        : chunk_keys + static_cast<size_t>(position - start_position) *
                           static_cast<size_t>(head_dim);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out_keys[static_cast<size_t>(row) * static_cast<size_t>(head_dim) +
                 static_cast<size_t>(d)] = source[d];
    }
    if (threadIdx.x == 0) {
        out_positions[row] = position;
    }
}

// Gather a position-contiguous row range for a whole query tile: output row `r` holds
// position `first + r`, read from the ring (slot `position % capacity`) when it predates
// the chunk and from the chunk buffer otherwise.
//
// This is the tile form of `v4_compose_local_rows_kernel`. That kernel composes **one
// query's** window and deliberately orders rows by ring slot, so the chunk and the
// decode path add the same `exp` terms in the same sequence and stay bit-identical. A
// tile instead masks by position — every query reads the same union and keeps only its
// own window — so it can use position order directly, and each row is labelled with its
// position for the mask.
__global__ void v4_compose_union_rows_kernel(
    const __half* __restrict__ ring_keys,   // [capacity, head_dim]
    const __half* __restrict__ chunk_keys,  // [chunk_count, head_dim]
    __half* __restrict__ out_keys,          // [rows, head_dim]
    int64_t* __restrict__ out_positions,    // [rows]
    int64_t first,
    int64_t start_position,
    int capacity,
    int head_dim
) {
    const int row = blockIdx.x;
    const int64_t position = first + row;
    const int64_t slot = position % capacity;
    const __half* source = position < start_position
        ? ring_keys + static_cast<size_t>(slot) * static_cast<size_t>(head_dim)
        : chunk_keys + static_cast<size_t>(position - start_position) *
                           static_cast<size_t>(head_dim);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out_keys[static_cast<size_t>(row) * static_cast<size_t>(head_dim) +
                 static_cast<size_t>(d)] = source[d];
    }
    if (threadIdx.x == 0) {
        out_positions[row] = position;
    }
}

// The indexer's candidate selection, on the device: the top `topk` candidates by
// score, in descending order, ties broken to the **lower index** — the same rule the
// host used to implement in `select_indexer_topk`, which this replaces.
// Why a kernel and not the host sort: the host version copied the scores back,
// synchronized, sorted on the CPU, copied the indices forward, and synchronized
// again. Those two `hipStreamSynchronize` calls are the reason the attention phase
// spends its time with the GPU idle — the CPU cannot enqueue the next token's work
// until the whole queue behind it has drained. The selection is unchanged; only
// where it runs is.
//
// The degenerate branch is kept exact: `candidate_count <= topk` selects every
// candidate in ascending index order and pads the tail with `-1`, with no sort.
// A CSA layer commits one entry per `ratio` tokens, so at short contexts this is
// every token, and it is a pure index ramp.
//
// One block per row; `selected` is a `ceil(candidate_count/8)`-byte dynamic-shared
// bitmap so a chosen candidate is excluded without touching the caller's scores.
__global__ void v4_indexer_topk_kernel(
    const float* __restrict__ scores,
    int32_t* __restrict__ topk_indices,
    int candidate_count,
    int topk
) {
    extern __shared__ unsigned char selected[];

    const int tid = threadIdx.x;

    if (candidate_count <= topk) {
        for (int i = tid; i < topk; i += blockDim.x) {
            topk_indices[i] = (i < candidate_count) ? i : -1;
        }
        return;
    }

    const int mask_bytes = (candidate_count + 7) >> 3;
    for (int i = tid; i < mask_bytes; i += blockDim.x) {
        selected[i] = 0;
    }
    __syncthreads();

    // Iterative max-extraction, `topk` rounds. Each round reduces over the
    // unselected candidates; the comparison takes the higher score, and on an exact
    // tie the lower index, so the order matches a descending stable sort exactly and
    // does not depend on the reduction tree.
    __shared__ int s_best_index[64];
    __shared__ float s_best_score[64];

    for (int k = 0; k < topk; ++k) {
        float best_score = -3.402823466e+38F;
        int best_index = -1;
        for (int i = tid; i < candidate_count; i += blockDim.x) {
            if ((selected[i >> 3] >> (i & 7)) & 1) continue;
            const float s = scores[i];
            if (s > best_score || (s == best_score && best_index >= 0 && i < best_index)) {
                best_score = s;
                best_index = i;
            }
        }

        // Block reduction, applying the same (score, lower-index) rule at each step.
        const int warp = tid >> 5;
        const int lane = tid & 31;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            const float other_score = __shfl_xor(best_score, offset);
            const int other_index = __shfl_xor(best_index, offset);
            if (other_score > best_score ||
                (other_score == best_score && other_index >= 0 &&
                 (best_index < 0 || other_index < best_index))) {
                best_score = other_score;
                best_index = other_index;
            }
        }
        if (lane == 0) {
            s_best_score[warp] = best_score;
            s_best_index[warp] = best_index;
        }
        __syncthreads();

        if (tid == 0) {
            const int warps = (blockDim.x + 31) >> 5;
            float block_score = -3.402823466e+38F;
            int block_index = -1;
            for (int w = 0; w < warps; ++w) {
                if (s_best_score[w] > block_score ||
                    (s_best_score[w] == block_score && s_best_index[w] >= 0 &&
                     (block_index < 0 || s_best_index[w] < block_index))) {
                    block_score = s_best_score[w];
                    block_index = s_best_index[w];
                }
            }
            topk_indices[k] = block_index;
            selected[block_index >> 3] |= static_cast<unsigned char>(1u << (block_index & 7));
        }
        __syncthreads();
    }
}

// Serial local-plus-compressed attention for C4A and C128A. The bounded
// shared score array covers the 128-token local ring plus a 512-entry top-k.
__global__ void __launch_bounds__(32) v4_cached_compressed_attention_wave32_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ local_key_cache,
    const __half* __restrict__ local_value_cache,
    const int64_t* __restrict__ local_positions,
    const float* __restrict__ attn_sink,
    const __half* __restrict__ compressed_key_cache,
    const __half* __restrict__ compressed_value_cache,
    const int64_t* __restrict__ compressed_positions,
    const int32_t* __restrict__ topk_indices,
    __half* __restrict__ out,
    int64_t current_position,
    int local_capacity,
    int compressed_count,
    int topk_count,
    bool uses_indexer,
    float scale
) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    __shared__ float scores[DSV4_MAX_ATTENTION_KEYS];

    const int local_start = max(0, static_cast<int>(current_position) - local_capacity + 1);
    const __half* query = q + static_cast<size_t>(head) * DSV4_HEAD_DIM;
    const int compressed_slots = uses_indexer ? topk_count : compressed_count;
    constexpr int kLaneElements = static_cast<int>(DSV4_HEAD_DIM) / 32;

    // Same loop-invariant hoist as the sliding kernel: one query walk into
    // registers instead of one per key slot, for up to `local + topk` slots.
    float q_reg[kLaneElements];
    #pragma unroll
    for (int i = 0; i < kLaneElements; ++i) {
        q_reg[i] = __half2float(query[lane + i * 32]);
    }

    for (int slot = 0; slot < local_capacity; ++slot) {
        const int64_t key_position = local_positions[slot];
        const bool valid = key_position >= local_start && key_position <= current_position;
        float dot = 0.0f;
        if (valid) {
            const __half* key = local_key_cache + static_cast<size_t>(slot) * DSV4_HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < kLaneElements; ++i) {
                dot += q_reg[i] * __half2float(key[lane + i * 32]);
            }
            for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_xor(dot, offset, 32);
        }
        if (lane == 0) scores[slot] = valid ? dot * scale : -3.402823466e+38F;
    }

    for (int index = 0; index < compressed_slots; ++index) {
        const int compressed_index = uses_indexer ? topk_indices[index] : index;
        bool valid = compressed_index >= 0 && compressed_index < compressed_count;
        if (valid) valid = compressed_positions[compressed_index] <= current_position;
        float dot = 0.0f;
        if (valid) {
            const __half* key = compressed_key_cache + static_cast<size_t>(compressed_index) * DSV4_HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < kLaneElements; ++i) {
                dot += q_reg[i] * __half2float(key[lane + i * 32]);
            }
            for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_xor(dot, offset, 32);
        }
        if (lane == 0) scores[local_capacity + index] = valid ? dot * scale : -3.402823466e+38F;
    }
    __syncthreads();

    // Softmax, lane-strided on the same principle as the sliding kernel: the max is
    // an exact tree, the `expf` is strided off the redundant lockstep issue, and the
    // denominator keeps its ascending walk so the result stays bit-identical. That
    // last point is load-bearing here — `scores[]` packs the local block then the
    // compressed block at `local_capacity`, so the caller's row count shifts every
    // compressed term's position; a tree sum would regroup them and diverge from the
    // per-token path, which the chunk equality gates would see.
    const int total_keys = local_capacity + compressed_slots;
    float thread_max = attn_sink[head];
    for (int index = lane; index < total_keys; index += 32) {
        thread_max = fmaxf(thread_max, scores[index]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_xor(thread_max, offset, 32));
    }
    const float maximum = thread_max;

    for (int index = lane; index < total_keys; index += 32) {
        scores[index] = expf(scores[index] - maximum);
    }
    __syncthreads();

    float denominator = expf(attn_sink[head] - maximum);
    for (int index = 0; index < total_keys; ++index) {
        denominator += scores[index];
    }
    const float inverse_denominator = 1.0f / fmaxf(denominator, 1e-30f);
    __syncthreads();

    __half* output = out + static_cast<size_t>(head) * DSV4_HEAD_DIM;
    for (int dimension = lane; dimension < static_cast<int>(DSV4_HEAD_DIM); dimension += 32) {
        float value = 0.0f;
        for (int slot = 0; slot < local_capacity; ++slot) {
            if (scores[slot] == 0.0f) continue;
            value += scores[slot] * inverse_denominator *
                     __half2float(local_value_cache[static_cast<size_t>(slot) * DSV4_HEAD_DIM + dimension]);
        }
        for (int index = 0; index < compressed_slots; ++index) {
            const int compressed_index = uses_indexer ? topk_indices[index] : index;
            if (compressed_index < 0 || compressed_index >= compressed_count) continue;
            if (scores[local_capacity + index] == 0.0f) continue;
            value += scores[local_capacity + index] * inverse_denominator *
                     __half2float(compressed_value_cache[
                         static_cast<size_t>(compressed_index) * DSV4_HEAD_DIM + dimension]);
        }
        output[dimension] = __float2half(value);
    }
}

} // namespace aeon::kernel
