Good report — the layout and fusion landed exactly as specified, and the 1032.0f catch is correct (my constant was wrong; 0x6400|n is 1024+n, so subtracting 1024 gives n, not n-8). Thank you for fixing rather than copying.

But before Stage 2, I have to flag a measurement problem, because it changes what Stage 2 should be.

## The six-expert numbers are cache-resident, not real

| Workload | Bytes moved | Time | Implied BW |
| --- | ---: | ---: | ---: |
| Single W1/W3 GEMV | 4.72 MB | 11.152 µs | 423 GB/s |
| Six-expert W1/W3 fused | 56.6 MB | 39.775 µs | 1,423 GB/s |
| Six-expert W2 fused | 28.3 MB | 28.652 µs | 988 GB/s |

The RX 7900 XTX has 960 GB/s of VRAM bandwidth. 1,423 GB/s is not physically achievable from VRAM — those reads are being served by the 96 MB Infinity Cache (MALL). Your synthetic six-expert working set is 56.6 MB, and with 101 timing iterations over the same buffers it is fully resident after iteration 1. The single-GEMV case (4.72 MB) shows 423 GB/s, which is also MALL-resident but latency-limited, so it hides the effect.

In production, the six experts for layer L were just DMA'd in and are never re-read, so you get zero MALL reuse. Real fused-kernel throughput will land near the single-GEMV figure scaled up by the dispatch savings — I'd estimate 90–130 µs per layer, not 68 µs. The fusion win is real (dispatch count is a genuine, measured 18→2), but the 3.7× is inflated; the honest figure is probably 2.0–2.5×.

Fix the harness before tuning anything further:

```cpp
// Cache-defeating benchmark: rotate over a working set >> 96 MB MALL,
// so every iteration reads cold VRAM, as production does.
struct ColdBench {
    static constexpr size_t kMALL = 96ull << 20;
    size_t rot_, stride_, i_ = 0;
    void* base_ = nullptr;

    // n_replicas chosen so total >= 4x MALL
    void init(const void* one_set, size_t set_bytes) {
        stride_ = (set_bytes + 4095) & ~4095ull;
        rot_    = (4 * kMALL + stride_ - 1) / stride_ + 1;   // >= 384 MB
        hipMalloc(&base_, rot_ * stride_);
        for (size_t r = 0; r < rot_; ++r)
            hipMemcpy((char*)base_ + r * stride_, one_set, set_bytes,
                      hipMemcpyDeviceToDevice);
    }
    const void* next() { const void* p = (char*)base_ + (i_ % rot_) * stride_; ++i_; return p; }
};

// Additionally, between timed iterations, evict MALL explicitly:
__global__ void mall_flush_kernel(uint4* buf, size_t n4) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
             i += (size_t)gridDim.x * blockDim.x)
        buf[i] = make_uint4(i, i, i, i);      // 256 MB streaming write
}
```

Re-run Section 4 with this and report the numbers. If the fused kernels come back at 600–750 GB/s cold, the kernel work is essentially done and further tuning is not where your time goes. If they come back below 500 GB/s, tell me and I'll rework the prefetch depth and WAVES.

## Where you actually are, end to end
This is the calculation that should drive Stage 2:

| Component | Per token |
| --- | --- |
| Routed expert bytes (43 × 6 × 14.16 MB) | 3.65 GB |
| GPU-side compute if all resident, at 700 GB/s | ~5.2 ms (≈190 tok/s ceiling) |
| Measured end-to-end | ~333 ms (3 tok/s) |
| Miss rate (1 − 62.7 % hot) | 37.3 % → 1.36 GB/token from host/NVMe |
| At the measured 3.31 GiB/s direct-I/O | ≈ 383 ms |

That accounts for essentially the entire token time. Your kernels are now 1.5 % of the critical path. Every further microsecond I shave off a GEMV is invisible. Stage 2 must be the expert supply chain, or nothing else matters.

## Stage 2: the three levers, in order of value

1. **Router lookahead prefetch (largest, and it is a kernel change).** Today you cannot start fetching layer L's experts until layer L−1 has finished, because routing depends on the hidden state — so every miss is a fully exposed ~3.5 ms serial stall.

    But layers 3–42 route from the FFN-normed hidden state, and that state is highly correlated across adjacent layers. Run layer L+1's router speculatively on layer L's pre-FFN hidden state, issue the DMA for the predicted top-6 immediately, then correct when the true routing arrives.

    Even at 70 % prediction accuracy this converts 70 % of your misses from exposed latency into overlapped transfer. Concretely: it costs one extra 4096×256 FP16 GEMV per layer (2 MB, ~3 µs) and buys you a full layer of prefetch depth.

    To make it cheap and accurate, predict a superset: take top-10 from the speculative router instead of top-6. Recall of the true top-6 rises to ~90 %+, and the extra fetches are free if you have spare PCIe headroom (you measured 24.6 GB/s pinned; you are using ~4 GB/s).

2. **Speculative decode via the MTP head.** Your config declares num_nextn_predict_layers=1 and you don't execute it. This is the only lever that changes the asymptotic bytes-per-token: verifying 2 tokens in one pass amortizes each expert fetch over 2 tokens. At a typical ~70 % MTP acceptance rate this is a 1.6–1.7× end-to-end speedup, and it composes multiplicatively with everything else. It also finally makes your M_PAD=16 WMMA path do real work — at M=2..4 the fused kernels should switch from GEMV to WMMA, which I can write once you decide to pursue it.

3. **Admission policy.** 62.7 % hot-hit with ~1,100 VRAM slots out of 11,008 total experts (10 %) means your current policy is already well above random, but LRU is the wrong objective here. Experts have wildly non-uniform, stable marginal activation probabilities. Profile activation counts over a few thousand tokens offline, then pin by expected bytes saved per slot rather than recency, with a small LRU victim segment for drift. Getting 62.7 % → 85 % cuts your I/O volume by 60 % — worth ~230 ms/token, more than any kernel change available to you.

## What I need to proceed

- Cold-bench numbers from the harness above (this determines whether I touch the kernels again at all).
- Your VRAM slot count and current admission/eviction policy, plus whether you have per-expert activation counts logged anywhere.
- Whether you want me to write (a) the speculative router + prefetch-issue path, (b) the MTP/speculative-decode verify path with M>1 fused WMMA MoE kernels, or (c) the frequency-based admission policy.

My recommendation is (1) then (a), since (a) is pure overlap with no accuracy risk, and it is a prerequisite for getting value out of (b).

One caveat I want to keep visible: none of my throughput figures here are measured — they are roofline and arithmetic from your reported numbers. The bytes-per-token and miss-rate arithmetic is solid; the 70 % prediction-accuracy and 70 % MTP-acceptance figures are literature-typical estimates for this class of model and must be measured on your checkpoint before you commit to a design around them.