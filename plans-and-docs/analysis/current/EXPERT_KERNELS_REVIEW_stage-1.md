# Expert Kernels Review: Stage 1

Good — this is exactly the information I needed, and it changes my plan significantly. Let me give you the analysis first, because it determines everything, then the code.

## Roofline: Where You Actually Are

| Quantity | Value |
|---|---:|
| Bytes read per routed GEMV (`W1`, `N=2048`, `K=4096`) | `4096 · 2048 / 2 + 2048 · 128 · 2 = 4.72 MB` |
| Measured | `13.33 µs → 354 GB/s` |
| RX 7900 XTX peak VRAM bandwidth | `960 GB/s` (384-bit, 20 Gbps) |
| Efficiency | `≈ 37%` |
| Expert bytes per layer (`6 × 14.16 MB`) | `84.9 MB` |
| Per token (`43` layers) | `3.65 GB` |
| Hard floor at `960 GB/s` | `3.8 ms/token → ~263 tok/s` |
| Your current routed GEMM | `11 ms/token` |

There are two separate wins on the table, and they are roughly equal in size:

1. **Bandwidth efficiency.** You are at `37%` of peak on a kernel that is `100%` memory-bound. Getting to `75–80%` is realistic and worth roughly `5 ms/token`.
2. **Dispatch count.** You issue `18` GEMV dispatches per layer, or `774` per token. On `ROCm 7.x`, a back-to-back HIP kernel launch on the same stream has `~2–4 µs` of end-to-end latency. `774 × 2.5 µs ≈ 1.9 ms/token` is pure dispatch, and it also means the memory system is repeatedly drained and refilled because each `13.3 µs` kernel spends its first `~2 µs` ramping up. Note that your own numbers agree: a `4.72 MB` read at even `700 GB/s` is `6.7 µs`, yet you measure `13.3 µs`; roughly half the kernel is ramp/tail.

The fix for dispatch count is not a faster GEMV; it is one kernel for all six experts. I am collapsing `18` dispatches to `2`.

## What I Am Changing

### A. New On-Disk Weight Swizzle

This is the single biggest lever, and you confirmed you can repack. It has two changes:

1. **Wave-consumption ordering.** The packed words are stored in the exact order a wave32 consumes them, so every iteration is one `global_load_dwordx4` per lane over a fully contiguous `512 B` span, and each lane owns its own output row. There is no cross-lane reduction in the inner loop and no strided row access. Your current `[out, in]` row-major layout forces either strided loads or a wave-wide reduction per row; that accounts for most of the missing `60%`.
2. **Nibble interleave.** The pattern `[0, 2, 4, 6, 1, 3, 5, 7]` is interleaved within each `uint32` (AWQ order). This makes `(w >> 4t) & 0x000F000F` yield a `half2` of consecutive `K` elements, so dequantization takes `3 VALU ops per 2 weights` with zero shuffling, and the activation is a naturally aligned `half2`.

### B. Fused Expert Kernels

`W1‖W3 → SwiGLU` is fused across all six experts in one launch, and `W2 → weighted accumulate → fp16` is fused in a second launch. The `fp32 → fp16` finalize is folded in via a `last-block-arrives` trick, with no third dispatch and no separate zeroing dispatch because kernel 1 zeroes the accumulator.

## Arithmetic Check

- `~20 VALU ops per 16 B loaded = 1.25 op/byte`.
- At `960 GB/s`, that is `1.2 Tops/s` against roughly `15 Tops/s` of wave32 VALU throughput on `96 CUs`.
- **Conclusion:** Comfortable; there is no risk of becoming compute-bound.

## 1. Layout Spec + Offline Repacker

```cpp
// ============================================================================
// aeon_w4a16_swizzle.hpp  -- Aeon W4A16 kernel-oriented weight layout (gfx1100)
//
// LAYOUT CONTRACT  (per weight tensor, per expert)
// ------------------------------------------------
// Source (current Aeon format):
//   packed : [N, K/8] uint32, 8 sequential nibbles/word, weight k at bits 4*(k%8)
//   scale  : [N, K/32] fp16, symmetric, value = (nibble - 8) * scale
//
// Destination (swizzled), parameterised by (RPW, LPR) with RPW*LPR == 32:
//   RPW = rows  per wave (each lane owns exactly one output row)
//   LPR = lanes per row  (K-slices; reduced with LPR/2 shfl_xor steps at the end)
//   GROUPS = K/32          (one fp16 scale per group of 32 weights)
//   ITERS  = GROUPS / LPR  (loop trip count; must divide evenly)
//   NB     = N / RPW       (row-blocks)
//
//   lane l = r*LPR + s      (r in [0,RPW), s in [0,LPR))
//   at iteration i, lane l handles  row = b*RPW + r,  group = i*LPR + s
//
//   packed_sw[ (((b*ITERS + i)*32 + l) * 4) + w ]   // w in [0,4): uint4 per group
//   scale_sw [   (b*ITERS + i)*32 + l           ]
//
//   => a wave reads 32*16 = 512 B fully contiguous per iteration (dwordx4/lane)
//      and 32*2 = 64 B of scales, also contiguous.
//
// NIBBLE INTERLEAVE (within each uint32 word):
//   dst word w, nibble p  <-  src word w, nibble PERM[p], PERM = {0,2,4,6,1,3,5,7}
//   => (word >> 4*t) & 0x000F000F  is a half2 of source elements (2t, 2t+1).
//
// Element count and total bytes are IDENTICAL to the source layout, so the
// 14,155,776-byte expert payload size and all pool/IO plumbing are unchanged.
// Only the byte ORDER inside each of the six sub-tensors changes.
// ============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <cassert>

namespace aeon {

// Chosen tiles. Both give ITERS == 16 and 512 B/wave/iteration.
//   W1 / W3 : N=2048, K=4096, GROUPS=128 -> RPW=4,  LPR=8, ITERS=16, NB=512
//   W2      : N=4096, K=2048, GROUPS=64  -> RPW=8,  LPR=4, ITERS=16, NB=512
struct SwizzleCfg {
    int rpw, lpr;
};

static constexpr SwizzleCfg kCfgW13{4, 8};
static constexpr SwizzleCfg kCfgW2{8, 4};

static constexpr int kNibblePerm[8] = {0, 2, 4, 6, 1, 3, 5, 7};

// Permute the 8 nibbles of one word into AWQ-style extraction order.
inline uint32_t permute_word(uint32_t src) {
    uint32_t dst = 0;
    for (int p = 0; p < 8; ++p) {
        const uint32_t nib = (src >> (4 * kNibblePerm[p])) & 0xFu;
        dst |= nib << (4 * p);
    }
    return dst;
}

// Host-side repack. dst buffers have the SAME element counts as src.
// scale_t is whatever 16-bit type you use (__half / uint16_t); it is copied opaquely.
template <typename scale_t>
void swizzle_w4a16(
    const uint32_t* src_packed,
    const scale_t* src_scale,
    uint32_t* dst_packed,
    scale_t* dst_scale,
    int N,
    int K,
    SwizzleCfg cfg) {
    const int RPW = cfg.rpw, LPR = cfg.lpr;
    assert(RPW * LPR == 32);
    const int GROUPS = K / 32;
    assert(GROUPS % LPR == 0 && N % RPW == 0 && K % 32 == 0);
    const int ITERS = GROUPS / LPR;
    const int NB    = N / RPW;
    const int SRC_WORDS_PER_ROW = K / 8;

    for (int b = 0; b < NB; ++b)
        for (int i = 0; i < ITERS; ++i)
            for (int r = 0; r < RPW; ++r)
                for (int s = 0; s < LPR; ++s) {
                    const int n = b * RPW + r;
                    const int g = i * LPR + s;
                    const int l = r * LPR + s;
                    const size_t dst_u4 = (size_t)(b * ITERS + i) * 32 + l;   // uint4 index
                    const size_t src_w0 = (size_t)n * SRC_WORDS_PER_ROW + (size_t)g * 4;
                    for (int w = 0; w < 4; ++w)
                        dst_packed[dst_u4 * 4 + w] = permute_word(src_packed[src_w0 + w]);
                    dst_scale[dst_u4] = src_scale[(size_t)n * GROUPS + g];
                }
}

// Exact inverse -- use it in the converter to assert round-trip bit identity.
template <typename scale_t>
void unswizzle_w4a16(
    const uint32_t* sw_packed,
    const scale_t* sw_scale,
    uint32_t* out_packed,
    scale_t* out_scale,
    int N,
    int K,
    SwizzleCfg cfg) {
    const int RPW = cfg.rpw, LPR = cfg.lpr;
    const int GROUPS = K / 32, ITERS = GROUPS / LPR, NB = N / RPW;
    const int SRC_WORDS_PER_ROW = K / 8;
    int inv[8];
    for (int p = 0; p < 8; ++p)
        inv[kNibblePerm[p]] = p;

    for (int b = 0; b < NB; ++b)
        for (int i = 0; i < ITERS; ++i)
            for (int r = 0; r < RPW; ++r)
                for (int s = 0; s < LPR; ++s) {
                    const int n = b * RPW + r, g = i * LPR + s, l = r * LPR + s;
                    const size_t su4 = (size_t)(b * ITERS + i) * 32 + l;
                    const size_t dw0 = (size_t)n * SRC_WORDS_PER_ROW + (size_t)g * 4;
                    for (int w = 0; w < 4; ++w) {
                        uint32_t v = sw_packed[su4 * 4 + w], o = 0;
                        for (int p = 0; p < 8; ++p)
                            o |= ((v >> (4 * p)) & 0xFu) << (4 * inv[p]);
                        out_packed[dw0 + w] = o;
                    }
                    out_scale[(size_t)n * GROUPS + g] = sw_scale[su4];
                }
}

} // namespace aeon
```

The equivalent for convert_safetensors_to_aeon.py, vectorised so it runs in a second:

```py
import numpy as np

PERM = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)

def permute_words(w):                       # w: uint32 array, any shape
    out = np.zeros_like(w)
    for p in range(8):
        out |= (((w >> np.uint32(4 * int(PERM[p]))) & np.uint32(0xF))
                << np.uint32(4 * p)).astype(np.uint32)
    return out

def swizzle_w4a16(packed, scale, rpw, lpr):
    """packed: [N, K//8] uint32 ; scale: [N, K//32] float16 -> (packed_sw, scale_sw) flat"""
    N, wpr = packed.shape
    K = wpr * 8
    G, ITERS, NB = K // 32, (K // 32) // lpr, N // rpw
    assert rpw * lpr == 32 and (K // 32) % lpr == 0 and N % rpw == 0

    p = permute_words(packed).reshape(NB, rpw, ITERS, lpr, 4)   # b r i s w
    p = p.transpose(0, 2, 1, 3, 4)                              # b i r s w  (l = r*lpr+s)
    s = scale.reshape(NB, rpw, ITERS, lpr).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(p).ravel(), np.ascontiguousarray(s).ravel()

# W1/W3: swizzle_w4a16(p, s, 4, 8)   |   W2: swizzle_w4a16(p, s, 8, 4)
```

Note the transpose is the entire swizzle — byte count, sector count and the 14,155,776-byte payload are unchanged, so your loader, host pool, VRAM pool and direct-I/O index all keep working. Only the converter and the kernels need to agree on (rpw, lpr) per sub-tensor. Bump a version field in the Aeon header so an old checkpoint can't be silently fed to the new kernel.

```cpp
// ============================================================================
// aeon_moe_fused.hpp -- fused 6-expert MoE FFN for gfx1100 / wave32
//   dispatch 1: W1 & W3 (all experts) -> clamped SwiGLU -> h[E][2048]
//               (also zeroes the fp32 output accumulator)
//   dispatch 2: W2 (all experts) -> weighted accumulate -> fp16 out[4096]
//               (fp32->fp16 finalize folded in via last-block-arrives)
// Requires weights in the aeon_w4a16_swizzle.hpp layout.
// ============================================================================
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "aeon_w4a16_swizzle.hpp"

namespace aeon {

typedef _Float16 half2v __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float fdot2(half2 a, half2 b, float c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2v, a),
                                  __builtin_bit_cast(half2v, b), c, false);
#else
    return c + __half2float(a.x) * __half2float(b.x)
             + __half2float(a.y) * __half2float(b.y);
#endif
}

// nibbles p=t and p=t+4 -> source elements 2t and 2t+1, as an exact half2.
// (0x6400 = fp16 1024.0; 1024+n is exactly representable for n in [0,15].)
__device__ __forceinline__ half2 unpack2(uint32_t w, int t) {
    uint32_t b = ((w >> (4 * t)) & 0x000F000Fu) | 0x64006400u;
    return __hsub2(*reinterpret_cast<half2*>(&b), __float2half2_rn(1024.0f));
}

// ---- expert pointer bundle, passed by value (no per-layer pointer upload) ---
#define AEON_MAX_TOPK 8
struct MoEExpertPtrs {
    const uint4* w1[AEON_MAX_TOPK]; const half* s1[AEON_MAX_TOPK];
    const uint4* w3[AEON_MAX_TOPK]; const half* s3[AEON_MAX_TOPK];
    const uint4* w2[AEON_MAX_TOPK]; const half* s2[AEON_MAX_TOPK];
};

// --------------------------------------------------------------------------
// Core per-wave W4A16 GEMV over the swizzled layout.
// Each lane accumulates one (row, k-slice) partial; caller reduces over LPR.
// Two independent weight streams (A and B) share one activation stream, which
// is what lets W1 and W3 fuse for free.
// --------------------------------------------------------------------------
template <int RPW, int LPR, int ITERS, bool DUAL>
__device__ __forceinline__ void gemv_tile(
    const uint4* __restrict__ wa, const half* __restrict__ sa,
    const uint4* __restrict__ wb, const half* __restrict__ sb,
    const half* __restrict__ sx, int lane, float& accA, float& accB)
{
    const int s = lane % LPR;
    accA = 0.f; accB = 0.f;

    uint4 pa = wa[lane], pb;  half va = sa[lane], vb;
    if (DUAL) { pb = wb[lane]; vb = sb[lane]; }

    #pragma unroll 1
    for (int i = 0; i < ITERS; ++i) {
        // ---- prefetch i+1 (keeps >=2 dwordx4 in flight per lane) ----
        uint4 na, nb; half wa_s, wb_s;
        const int nx = (i + 1) * 32 + lane;
        if (i + 1 < ITERS) {
            na = wa[nx]; wa_s = sa[nx];
            if (DUAL) { nb = wb[nx]; wb_s = sb[nx]; }
        }
        // ---- compute i ----
        const half* a = sx + (i * LPR + s) * 32;
        const uint32_t* ua = reinterpret_cast<const uint32_t*>(&pa);
        const uint32_t* ub = reinterpret_cast<const uint32_t*>(&pb);
        float dA = 0.f, dB = 0.f;
        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const half2* av = reinterpret_cast<const half2*>(a + wi * 8);
            const uint32_t bA = ua[wi];
            const uint32_t bB = DUAL ? ub[wi] : 0u;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const half2 x = av[t];
                dA = fdot2(unpack2(bA, t), x, dA);
                if (DUAL) dB = fdot2(unpack2(bB, t), x, dB);
            }
        }
        accA += __half2float(va) * dA;
        if (DUAL) accB += __half2float(vb) * dB;
        pa = na; va = wa_s;
        if (DUAL) { pb = nb; vb = wb_s; }
    }
}

template <int LPR>
__device__ __forceinline__ float reduce_slices(float v) {
    #pragma unroll
    for (int m = 1; m < LPR; m <<= 1) v += __shfl_xor(v, m, 32);
    return v;
}

// >>> REPLACE WITH YOUR EXACT FORMULA from v4_pipeline_ops.hpp <<<
// Placeholder = SiLU(clamp(gate,-L,L)) * up. If your production op is the
// gpt-oss style (up+1)*g*sigmoid(alpha*g) with an up-clamp, swap it in here;
// the surrounding kernel is unaffected.
__device__ __forceinline__ float swiglu_clamped(float g, float u, float L) {
    g = fminf(fmaxf(g, -L), L);
    return (g / (1.0f + __expf(-g))) * u;
}

// ==========================  DISPATCH 1  ===================================
// grid  = dim3(N13 / (WAVES*RPW13), E)      block = WAVES*32
template <int WAVES, int RPW, int LPR, int ITERS, int K>
__global__ __launch_bounds__(WAVES * 32)
void aeon_moe_w13_swiglu_kernel(const half* __restrict__ x,   // [K]
                                MoEExpertPtrs P,
                                half* __restrict__ h,          // [E][N]
                                float* __restrict__ out_f32,   // [K] (zeroed here)
                                int N, int out_dim, float limit)
{
    __shared__ half sx[K];

    // cooperative zero of the fp32 accumulator consumed by dispatch 2
    if (blockIdx.y == 0) {
        for (int t = blockIdx.x * (WAVES * 32) + threadIdx.x; t < out_dim;
                 t += gridDim.x * (WAVES * 32))
            out_f32[t] = 0.f;
    }

    { // stage activations (8 KB) with uint4 loads
        const uint4* g = reinterpret_cast<const uint4*>(x);
        uint4* d = reinterpret_cast<uint4*>(sx);
        for (int t = threadIdx.x; t < K / 8; t += WAVES * 32) d[t] = g[t];
    }
    __syncthreads();

    const int e    = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int b    = blockIdx.x * WAVES + wave;      // row-block
    const size_t off = (size_t)b * ITERS * 32;

    float g, u;
    gemv_tile<RPW, LPR, ITERS, true>(P.w1[e] + off, P.s1[e] + off,
                                     P.w3[e] + off, P.s3[e] + off,
                                     sx, lane, g, u);
    g = reduce_slices<LPR>(g);
    u = reduce_slices<LPR>(u);

    if ((lane % LPR) == 0) {
        const int row = b * RPW + (lane / LPR);
        h[(size_t)e * N + row] = __float2half(swiglu_clamped(g, u, limit));
    }
}

// ==========================  DISPATCH 2  ===================================
// grid = dim3(N2 / (WAVES*RPW), E)   block = WAVES*32
// Accumulates topk_w[e] * (h_e @ W2_e^T) into out_f32, then the last expert
// block for each row-block converts that slice to fp16 -- no extra dispatch.
template <int WAVES, int RPW, int LPR, int ITERS, int K>
__global__ __launch_bounds__(WAVES * 32)
void aeon_moe_w2_accum_kernel(const half* __restrict__ h,        // [E][K]
                              MoEExpertPtrs P,
                              const float* __restrict__ topk_w,  // [E]
                              float* __restrict__ out_f32,       // [N]
                              half*  __restrict__ out_f16,       // [N]
                              int* __restrict__ counters,        // [gridDim.x], zeroed once
                              int N, int E)
{
    __shared__ half sx[K];
    __shared__ int  s_last;

    const int e = blockIdx.y;
    {
        const uint4* g = reinterpret_cast<const uint4*>(h + (size_t)e * K);
        uint4* d = reinterpret_cast<uint4*>(sx);
        for (int t = threadIdx.x; t < K / 8; t += WAVES * 32) d[t] = g[t];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int b    = blockIdx.x * WAVES + wave;
    const size_t off = (size_t)b * ITERS * 32;

    float acc, dummy;
    gemv_tile<RPW, LPR, ITERS, false>(P.w2[e] + off, P.s2[e] + off,
                                      nullptr, nullptr, sx, lane, acc, dummy);
    acc = reduce_slices<LPR>(acc);

    const float wgt = topk_w[e];
    if ((lane % LPR) == 0) {
        const int row = b * RPW + (lane / LPR);
        atomicAdd(&out_f32[row], wgt * acc);
    }

    // ---- fold the fp32->fp16 finalize into the last arriving expert block ----
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) s_last = (atomicAdd(&counters[blockIdx.x], 1) == E - 1);
    __syncthreads();
    if (s_last) {
        const int base = blockIdx.x * (WAVES * RPW);
        for (int t = threadIdx.x; t < WAVES * RPW; t += WAVES * 32)
            out_f16[base + t] = __float2half(out_f32[base + t]);
        __syncthreads();
        if (threadIdx.x == 0) counters[blockIdx.x] = 0;   // rearm for next layer
    }
}

// ============================== host side ==================================
// DeepSeek-V4-Flash routed FFN: H=4096, I=2048, E=6.
//   W1/W3 : N=2048 K=4096  cfg(4,8)  ITERS=16  grid.x = 2048/(8*4) = 64
//   W2    : N=4096 K=2048  cfg(8,4)  ITERS=16  grid.x = 4096/(8*8) = 64
// -> 384 workgroups / 3072 waves per dispatch on 96 CUs. Two dispatches/layer.
inline void aeon_moe_ffn_fused(const half* d_x, const MoEExpertPtrs& P,
                               const float* d_topk_w, half* d_h,
                               float* d_out_f32, half* d_out_f16,
                               int* d_counters, int E, float swiglu_limit,
                               hipStream_t stream)
{
    constexpr int WAVES = 8, H = 4096, I = 2048;
    hipLaunchKernelGGL((aeon_moe_w13_swiglu_kernel<WAVES, 4, 8, 16, H>),
        dim3(I / (WAVES * 4), E), dim3(WAVES * 32), 0, stream,
        d_x, P, d_h, d_out_f32, I, H, swiglu_limit);

    hipLaunchKernelGGL((aeon_moe_w2_accum_kernel<WAVES, 8, 4, 16, I>),
        dim3(H / (WAVES * 8), E), dim3(WAVES * 32), 0, stream,
        d_h, P, d_topk_w, d_out_f32, d_out_f16, d_counters, H, E);
}

} // namespace aeon
```

## Integration Notes, Briefly

- `d_counters` is `int[64]`, allocated once and `hipMemset` to `0` once at init — the kernel rearms it, so there is no per-layer memset.
- `d_h` is `half[8][2048]` (max top-k), allocated once in your scratch.
- `MoEExpertPtrs` is filled on the host from your six `UnifiedVRAMExpertPool::get_*` accessors after prefetch resolution, and passed by value — `288 B` of kernarg, no extra `hipMemcpyAsync` per layer.
- `topk_w` must be the FP32 normalized top-6 weights already on device (you have them from the router).
- This assumes all 6 experts are resident when the launch happens; it does not change your streaming logic.

## Expected Effect

| Metric | Now | Expected |
|---|---:|---:|
| Dispatches/layer | 18 | 2 |
| Dispatches/token | 774 | 86 |
| Effective BW | 354 GB/s | 700–800 GB/s (target) |
| Routed GEMM/token | 11 ms | 4.5–5.5 ms |

I want to be explicit: I cannot run this on your hardware, so the bandwidth figure is a design target derived from the roofline, not a measurement. The correctness path, however, is exactly checkable — `unswizzle_w4a16` round-trips bit-identically, and `test_w4a16_wmma`'s CPU reference should give max error `0` against the new kernels since the dequant math `((n-8)*scale, fp32 accumulate)` is unchanged.

## What I Need Back Before Stage 2

1. Your exact clamped-SwiGLU formula — I placeholdered it and it is the one thing here that can silently change your logits.
2. A `rocprof` run of these two kernels (`--stats`) plus `--kernel-trace`, so we can see achieved bandwidth and whether we are latency- or occupancy-limited. If you also give me `MemUnitStalled`/`FetchSize`, I can tune `WAVES` and the prefetch depth precisely.
3. Confirmation that top-k is always exactly 6 and experts are never duplicated in the selection.

## Stage 2 Priorities

Stage 2, in priority order once those land:

1. The MLA attention rewrite — your `[64,512]` q against a `[pos,512]` latent with `v_head_dim=512` and a `128`-window is currently `2.62 µs/token`, which at `43` layers is `113 µs/token` and not your problem, so I would rather spend stage 2 on:
2. The LM head, which you note reads `1.06 GB/token` — that is `1.1 ms/token` at peak and is almost certainly your second-largest compute cost after the experts, and it is a trivially fixable FP16 GEMV over `[129280, 4096]`; and
3. Folding the router GEMV + top-6 + the shared expert into a single dispatch.

Also worth flagging as a correctness item independent of performance: the YaRN factor=16 vs the 1.0 actually passed to `RopeTable::init` will corrupt any context beyond `65536`, and the `M_PAD=16` prefill path never engages because prefill runs `M=1` per token — a real batched prefill would cut your `1156 ms` TTFT by roughly the batch factor.