# Expert Kernels Review - Stage 1 Follow-up Answers

## Cache-hit and capacity correction

The `62.7%` hot-hit value in the previous answers came from one 43-layer `.aeon` benchmark with a 35 GiB Warm tier. It was a workload-specific observation, not a calibrated inference statistic. The answers did not include enough run metadata to make it independently auditable, and it should not be used as a general model hit-rate assumption.

The controlled full-model measurements in the performance ledger are closer to a `59.3%` Hot hit rate with `664` Hot VRAM slots, with other workloads measuring `56.7%` and `59.2%`. The `~1,100` VRAM-slot figure is not an Aeon configuration; it appears to be the reviewer's approximation of `10%` of `11,008` total experts. Our documented full-model runs use approximately `664` Hot slots. The hit-rate and capacity claims therefore need to remain workload-specific until they are measured over a representative inference corpus.

## Cold rotating-buffer measurement

**Date:** 2026-09-10

**Target:** RX 7900 XTX (`gfx1100`), Wave32, ROCm 7.2.2

**Purpose:** Re-measure the Stage 1 fused paths after reducing repeated-buffer reuse from the original benchmark. The benchmark-only harness now rotates through independent device weight replicas whose total footprint exceeds `4 x 96 MiB`, the estimated MALL capacity. Production kernels and runtime code are unchanged.

**Method:**

- Each timed iteration selects the next complete six-expert weight replica.
- W1/W3 uses nine replicas and rotates through a `486 MiB` working set. One six-expert set contains `56.623 MB` of packed weights and scales.
- W2 also allocates nine replicas and `486 MiB` in total because each replica contains both source-layout weights for the current baseline and swizzled weights for the fused path. Each timed path reads its own representation; the active fused W2 rotation is `243 MiB` and contains `28.312 MB` per six-expert set.
- Timing uses HIP events around 100 kernel-only iterations. Initialization and device-to-device replica copies are outside the timed regions.
- This is a cache-defeating working-set measurement, not a hardware guarantee that every individual load misses MALL. No explicit cache-flush kernel was used.

### Results

| Workload | Warm baseline | Warm fused | Cold baseline | Cold fused | Cold fusion speedup | Cold fused weight bandwidth | Output check |
|---|---:|---:|---:|---:|---:|---:|---:|
| Six-expert W1/W3 + SwiGLU | `95.674 us` | `37.358 us` | `150.042 us` | `103.987 us` | `1.443x` | `544.519 GB/s` | warm max diff `0.016` |
| Six-expert W2 + accumulation | `80.767 us` | `24.423 us` | `78.103 us` | `56.897 us` | `1.373x` | `497.593 GB/s` | warm max diff `0.002` |

The fused W1/W3 path slowed from `37.358 us` with repeated buffers to `103.987 us` with the rotating footprint. The fused W2 path slowed from `24.423 us` to `56.897 us`. The cold effective bandwidths are well below the earlier cache-resident W1/W3 implication of more than `1 TB/s`, supporting the reviewer's observation that the original six-expert measurements benefited materially from cache reuse.

Fusion remains beneficial, but the cold harness measures approximately `1.44x` for W1/W3 and `1.37x` for W2 rather than the original `3.7x` cache-resident ratios. These are still isolated synthetic-kernel results; they do not establish a 43-layer end-to-end speedup or predict the exact production latency after expert transfers and dispatch waits.

**Validation:** `cmake --build build --target bench_aeon_moe_fused_w13 bench_aeon_moe_fused_w2 -j2` passed, followed by both rebuilt silicon benchmarks.