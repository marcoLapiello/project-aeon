#pragma once

// -----------------------------------------------------------------------------
// G2 (RDNA3): causal attention with **heads as the WMMA M axis**.
//
// Every query head attends the same key/value rows (one shared KV head), so a block
// serves `(query, 16 heads)`: `S = Q[16 heads] · Kᵀ` and `O = P · V` are both WMMA
// tiles, each key row is read once per 16 heads instead of once per head, and the
// per-query window or index selection needs no query tiling. The valid keys are first
// compacted (ordered, deterministic) so masked union rows and `-1` selections cost
// nothing, then consumed by an online softmax in blocks of `WARPS × 16` keys.
//
// Same contract as `dispatch_causal_attention_split_fp16`; it falls back to that kernel
// for any shape this one does not instantiate.
// -----------------------------------------------------------------------------

#include "platform/rdna3/tiled_causal_attention.hpp"

namespace aeon::rdna3 {

constexpr int kHeadGroupRows = kWmmaTileM;        // heads per block
constexpr int kHeadGroupDimRound = 64;            // V dims transposed through LDS per round
constexpr int kHeadGroupWarps = 8;

template <int HEAD_DIM, int WARPS>
struct HeadGroupAttentionLayout {
    static constexpr int kThreads = WARPS * kCausalAttentionLanes;
    static constexpr int kBlockKeys = WARPS * kWmmaTileN;
    static constexpr int kDimsPerWarp = HEAD_DIM / WARPS;
    static constexpr int kQPitch = HEAD_DIM + 8;          // pad: spreads the A-fragment rows over banks
    static constexpr int kPPitch = kBlockKeys + 8;
    static constexpr size_t kQBytes = size_t(kHeadGroupRows) * kQPitch * 2;
    static constexpr size_t kSBytes = size_t(kHeadGroupRows) * kBlockKeys * 4;
    static constexpr size_t kPBytes = size_t(kHeadGroupRows) * kPPitch * 2;
    static constexpr size_t kVtBytes = size_t(WARPS) * kHeadGroupDimRound * kWmmaTileK * 2;
    static constexpr size_t kStatBytes = (3 * kHeadGroupRows + WARPS + 1) * 4;
    static constexpr size_t kFixedBytes = kQBytes + kSBytes + kPBytes + kVtBytes + kStatBytes;
    static_assert(HEAD_DIM % (WARPS * kHeadGroupDimRound) == 0, "dims per warp must be whole rounds");
};

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ void head_group_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

template <int HEAD_DIM, int WARPS>
__global__ void __launch_bounds__(WARPS * kCausalAttentionLanes)
causal_attention_head_group_fp16_kernel(
    const __half* __restrict__ q, int q_stride,
    const __half* __restrict__ keys0, int key_stride0,
    const __half* __restrict__ values0, int value_stride0,
    const int64_t* __restrict__ positions0, int rows0, int window0,
    const __half* __restrict__ keys1, int key_stride1,
    const __half* __restrict__ values1, int value_stride1,
    const int64_t* __restrict__ positions1, int rows1, int window1,
    const int32_t* __restrict__ per_query_keys, int per_query_count,
    int64_t query_position_base, int64_t query_position_stride,
    __half* __restrict__ out, int out_stride,
    int num_heads, const float* __restrict__ bias, float scale) {
    using L = HeadGroupAttentionLayout<HEAD_DIM, WARPS>;
    constexpr int KB = L::kBlockKeys;
    constexpr int ROUNDS = L::kDimsPerWarp / kHeadGroupDimRound;
    constexpr int TILES_PER_ROUND = kHeadGroupDimRound / kWmmaTileN;
    constexpr int ROW_THREADS = L::kThreads / kHeadGroupRows;
    constexpr int COLS_PER_THREAD = KB / ROW_THREADS;

    extern __shared__ __align__(16) unsigned char head_group_smem[];
    unsigned char* smem = head_group_smem;
    _Float16* q_lds = reinterpret_cast<_Float16*>(smem);
    float* s_lds = reinterpret_cast<float*>(smem + L::kQBytes);
    _Float16* p_lds = reinterpret_cast<_Float16*>(smem + L::kQBytes + L::kSBytes);
    uint16_t* vt_all = reinterpret_cast<uint16_t*>(smem + L::kQBytes + L::kSBytes + L::kPBytes);
    float* row_max = reinterpret_cast<float*>(smem + L::kQBytes + L::kSBytes + L::kPBytes + L::kVtBytes);
    float* row_sum = row_max + kHeadGroupRows;
    float* row_alpha = row_sum + kHeadGroupRows;
    int* warp_count = reinterpret_cast<int*>(row_alpha + kHeadGroupRows);   // [WARPS + 1]
    int* key_list = reinterpret_cast<int*>(smem + L::kFixedBytes);

    const int head0 = blockIdx.x * kHeadGroupRows;
    const int query = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & (kCausalAttentionLanes - 1);
    const int warp = tid / kCausalAttentionLanes;
    const int axis = wmma_lane_axis(lane);
    const int parity = wmma_lane_parity(lane);

    const int64_t query_position =
        query_position_base + static_cast<int64_t>(query) * query_position_stride;
    const int64_t window_first0 = window0 > 0
        ? (query_position >= window0 - 1 ? query_position - (window0 - 1) : 0) : 0;
    const int64_t window_first1 = window1 > 0
        ? (query_position >= window1 - 1 ? query_position - (window1 - 1) : 0) : 0;

    // Q tile into LDS; heads past `num_heads` are zero rows whose output is never written.
    constexpr int kChunksPerRow = HEAD_DIM / 8;
    for (int c = tid; c < kHeadGroupRows * kChunksPerRow; c += L::kThreads) {
        const int r = c / kChunksPerRow;
        const int d = (c % kChunksPerRow) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (head0 + r < num_heads) {
            v = *reinterpret_cast<const uint4*>(
                q + static_cast<size_t>(query) * q_stride +
                static_cast<size_t>(head0 + r) * HEAD_DIM + d);
        }
        *reinterpret_cast<uint4*>(q_lds + r * L::kQPitch + d) = v;
    }
    if (tid < kHeadGroupRows) {
        const float b = (bias != nullptr && head0 + tid < num_heads) ? bias[head0 + tid] : -INFINITY;
        row_max[tid] = b;
        row_sum[tid] = b == -INFINITY ? 0.0f : 1.0f;
    }

    // Ordered compaction of the valid keys: block-0 row `r` is `r`, block-1 row `b` is
    // `rows0 + b`. Order is candidate order, so the result is deterministic.
    const int rows1_effective = per_query_keys != nullptr ? per_query_count : rows1;
    const int candidates = rows0 + rows1_effective;
    int n_valid = 0;
    for (int base = 0; base < candidates; base += L::kThreads) {
        const int k = base + tid;
        int encoded = -1;
        if (k < rows0) {
            const int64_t p = positions0[k];
            if (p >= window_first0 && p <= query_position) encoded = k;
        } else if (k < candidates) {
            const int j = k - rows0;
            const int row = per_query_keys != nullptr
                ? per_query_keys[static_cast<size_t>(query) * per_query_count + j] : j;
            if (row >= 0 && row < rows1) {
                const int64_t p = positions1[row];
                if (p >= window_first1 && p <= query_position) encoded = rows0 + row;
            }
        }
        const unsigned long long mask = __ballot(encoded >= 0);
        if (lane == 0) warp_count[warp] = __popcll(mask);
        __syncthreads();
        int offset = n_valid;
        int total = 0;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const int c = warp_count[w];
            if (w < warp) offset += c;
            total += c;
        }
        if (encoded >= 0) key_list[offset + __popcll(mask & __lanemask_lt())] = encoded;
        n_valid += total;
        __syncthreads();
    }

    auto key_row = [&](int encoded) {
        return encoded < rows0 ? keys0 + static_cast<size_t>(encoded) * key_stride0
                               : keys1 + static_cast<size_t>(encoded - rows0) * key_stride1;
    };
    auto value_row = [&](int encoded) {
        return encoded < rows0 ? values0 + static_cast<size_t>(encoded) * value_stride0
                               : values1 + static_cast<size_t>(encoded - rows0) * value_stride1;
    };

    f32_vec8 acc_o[ROUNDS * TILES_PER_ROUND];
    #pragma unroll
    for (int t = 0; t < ROUNDS * TILES_PER_ROUND; ++t) acc_o[t] = wmma_zero_accumulator();

    uint16_t* vt = vt_all + warp * kHeadGroupDimRound * kWmmaTileK;
    const int dim_base = warp * L::kDimsPerWarp;
    const int softmax_row = tid / ROW_THREADS;
    const int softmax_sub = tid % ROW_THREADS;

    for (int kb = 0; kb < n_valid; kb += KB) {
        // S = Q · Kᵀ: this warp's 16 keys.
        const int tile_first = kb + warp * kWmmaTileN;
        if (tile_first < n_valid) {
            const int kk = tile_first + axis;
            const __half* krow = key_row(key_list[min(kk, n_valid - 1)]);
            f32_vec8 acc = wmma_zero_accumulator();
            #pragma unroll 8
            for (int kd = 0; kd < HEAD_DIM; kd += kWmmaTileK) {
                f16_vec16 a;
                __builtin_memcpy(&a, q_lds + axis * L::kQPitch + kd, sizeof(a));
                const f16_vec16 b = wmma_load_a_row(krow + kd);
                acc = wmma_mma(a, b, acc);
            }
            const bool valid = kk < n_valid;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                s_lds[(2 * i + parity) * KB + warp * kWmmaTileN + axis] =
                    valid ? acc[i] * scale : -INFINITY;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                s_lds[(2 * i + parity) * KB + warp * kWmmaTileN + axis] = -INFINITY;
            }
        }
        __syncthreads();

        // Online softmax over the block, one row per `ROW_THREADS` contiguous lanes.
        {
            const float* s_row = s_lds + softmax_row * KB;
            float s[COLS_PER_THREAD];
            float block_max = -INFINITY;
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                s[j] = s_row[softmax_sub + j * ROW_THREADS];
                block_max = fmaxf(block_max, s[j]);
            }
            #pragma unroll
            for (int offset = ROW_THREADS / 2; offset > 0; offset /= 2) {
                block_max = fmaxf(block_max, __shfl_xor(block_max, offset, kCausalAttentionLanes));
            }
            const float m_old = row_max[softmax_row];
            const float m_new = fmaxf(m_old, block_max);
            float block_sum = 0.0f;
            _Float16* p_row = p_lds + softmax_row * L::kPPitch;
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                const _Float16 p = static_cast<_Float16>(s[j] == -INFINITY ? 0.0f : __expf(s[j] - m_new));
                p_row[softmax_sub + j * ROW_THREADS] = p;
                block_sum += static_cast<float>(p);
            }
            #pragma unroll
            for (int offset = ROW_THREADS / 2; offset > 0; offset /= 2) {
                block_sum += __shfl_xor(block_sum, offset, kCausalAttentionLanes);
            }
            if (softmax_sub == 0) {
                const float alpha = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
                row_alpha[softmax_row] = alpha;
                row_sum[softmax_row] = row_sum[softmax_row] * alpha + block_sum;
                row_max[softmax_row] = m_new;
            }
        }
        __syncthreads();

        // O = O·alpha + P · V over this warp's dims.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float alpha = row_alpha[2 * i + parity];
            #pragma unroll
            for (int t = 0; t < ROUNDS * TILES_PER_ROUND; ++t) acc_o[t][i] *= alpha;
        }
        #pragma unroll 1
        for (int ks = 0; ks < WARPS; ++ks) {
            const int step_first = kb + ks * kWmmaTileK;
            if (step_first >= n_valid) break;
            f16_vec16 a;
            __builtin_memcpy(&a, p_lds + axis * L::kPPitch + ks * kWmmaTileK, sizeof(a));
            const int kk = step_first + axis;
            // Padding keys carry P = 0, but their V must be finite: 0·NaN is NaN.
            const __half* vrow = kk < n_valid ? value_row(key_list[kk]) : nullptr;
            #pragma unroll
            for (int r = 0; r < ROUNDS; ++r) {
                const int d0 = dim_base + r * kHeadGroupDimRound + parity * (kHeadGroupDimRound / 2);
                #pragma unroll
                for (int c = 0; c < kHeadGroupDimRound / 2 / 8; ++c) {
                    uint4 v = make_uint4(0, 0, 0, 0);
                    if (vrow != nullptr) v = *reinterpret_cast<const uint4*>(vrow + d0 + c * 8);
                    uint16_t bits[8];
                    __builtin_memcpy(bits, &v, sizeof(bits));
                    const int dl = parity * (kHeadGroupDimRound / 2) + c * 8;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) vt[(dl + e) * kWmmaTileK + axis] = bits[e];
                }
                head_group_wave_sync();
                #pragma unroll
                for (int t = 0; t < TILES_PER_ROUND; ++t) {
                    f16_vec16 b;
                    __builtin_memcpy(&b, vt + (t * kWmmaTileN + axis) * kWmmaTileK, sizeof(b));
                    acc_o[r * TILES_PER_ROUND + t] = wmma_mma(a, b, acc_o[r * TILES_PER_ROUND + t]);
                }
                head_group_wave_sync();
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = 2 * i + parity;
        if (head0 + m >= num_heads) continue;
        const float inv = 1.0f / fmaxf(row_sum[m], 1e-30f);
        __half* out_row = out + static_cast<size_t>(query) * out_stride +
                          static_cast<size_t>(head0 + m) * HEAD_DIM + dim_base + axis;
        #pragma unroll
        for (int t = 0; t < ROUNDS * TILES_PER_ROUND; ++t) {
            const int d = (t / TILES_PER_ROUND) * kHeadGroupDimRound + (t % TILES_PER_ROUND) * kWmmaTileN;
            out_row[d] = __float2half(acc_o[t][i] * inv);
        }
    }
}

#endif  // gfx11 device pass, or any host pass

template <int WARPS = kHeadGroupWarps>
inline void dispatch_causal_attention_head_group_fp16(
    const __half* q, int q_stride,
    const CausalAttentionBlock& block0,
    const CausalAttentionBlock& block1,
    const int32_t* per_query_keys, int per_query_count,
    int64_t query_position_base, int64_t query_position_stride,
    __half* out, int out_stride,
    int count, int num_heads, int head_dim,
    const float* bias, float scale, hipStream_t stream) {
    constexpr int HEAD_DIM = 512;
    using L = HeadGroupAttentionLayout<HEAD_DIM, WARPS>;
    auto aligned = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    const int rows1_effective = per_query_keys != nullptr ? per_query_count : block1.rows;
    const size_t shared =
        L::kFixedBytes + static_cast<size_t>(block0.rows + rows1_effective) * sizeof(int);
    const bool fits = head_dim == HEAD_DIM && shared <= 64 * 1024 &&
        aligned(q) && q_stride % 8 == 0 &&
        aligned(block0.keys) && aligned(block0.values) &&
        block0.key_stride % 8 == 0 && block0.value_stride % 8 == 0 &&
        (rows1_effective == 0 || (aligned(block1.keys) && aligned(block1.values) &&
                                  block1.key_stride % 8 == 0 && block1.value_stride % 8 == 0));
    if (!fits || count <= 0 || num_heads <= 0 || block0.rows + rows1_effective <= 0) {
        dispatch_causal_attention_split_fp16<>(
            q, q_stride, block0, block1, per_query_keys, per_query_count,
            query_position_base, query_position_stride, out, out_stride,
            count, num_heads, head_dim, bias, scale, stream);
        return;
    }
    if (q_stride < num_heads * head_dim || out_stride < num_heads * head_dim) {
        throw std::invalid_argument("dispatch_causal_attention_head_group_fp16: row pitch too small");
    }
    if (block0.positions == nullptr ||
        (rows1_effective > 0 && block1.positions == nullptr)) {
        throw std::invalid_argument("dispatch_causal_attention_head_group_fp16: null positions");
    }
    if (per_query_keys != nullptr && per_query_count <= 0) {
        throw std::invalid_argument("dispatch_causal_attention_head_group_fp16: empty selection");
    }

    const __half* keys1 = block1.keys != nullptr ? block1.keys : block0.keys;
    const __half* values1 = block1.values != nullptr ? block1.values : block0.values;
    const int64_t* positions1 = block1.positions != nullptr ? block1.positions : block0.positions;

    const dim3 grid((num_heads + kHeadGroupRows - 1) / kHeadGroupRows, count);
    causal_attention_head_group_fp16_kernel<HEAD_DIM, WARPS>
        <<<grid, L::kThreads, shared, stream>>>(
        q, q_stride,
        block0.keys, block0.key_stride, block0.values, block0.value_stride,
        block0.positions, block0.rows, block0.window,
        keys1, block1.key_stride, values1, block1.value_stride,
        positions1, block1.rows, block1.window,
        per_query_keys, per_query_count,
        query_position_base, query_position_stride, out, out_stride,
        num_heads, bias, scale);
}

} // namespace aeon::rdna3
