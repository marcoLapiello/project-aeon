# Host Memory Pressure Investigation — why a large Warm tier does not finish loading

**Status: OPEN.** Started `2026-09-17` on branch `rewrite/graph-v2`. This document records the
investigation, the hypotheses that were tested and **refuted**, and the one measurement that
survived — so that a future session does not repeat the refuted work.

This is an execution record, not a plan. When the cause is found, this document gets a conclusion and
the residual work moves to the composition plan; when it is not, its open section is the handover.

---

## 1. Symptom

With the rebuilt graph (post-P4) and a large Warm tier, `aeon_chat` **never finishes
`initialize()`**. It does not fail, does not throw, and does not print its budget report — it keeps
running while the host memory pressure grows. Observed twice, both times with `--context-size 32768`
and `--diagnostic --verbose`:

| Configuration | Outcome |
| :--- | :--- |
| `--warm-gib 45` | interrupted by hand; `read_bytes` climbing past 33 GB; process `VmSwap` 234 MiB; system free 437 MiB |
| `--warm-gib 45` (after the page release, §4) | interrupted by hand; `read_bytes` climbing past 11 GB; process `VmSwap` 0; system free 764 MiB |
| `--warm-gib 40` | **completed**, but only when the host was freshly booted (see §5) |
| `--warm-gib 8` | completed, `initialize()` ≈ 2 minutes |

The budget does not reject these configurations: the host cap is `total_ram − 10 GiB = 52.62 GB`, so
`45 GiB` is admitted. **The refusal is not in the budget, it is in the machine.**

---

## 2. What is measured, and what it means

All figures from `free`, `/proc/<pid>/status` (`VmRSS`, `VmSwap`) and `/proc/<pid>/io`, plus
`rocm-smi --showpids` for per-process VRAM.

| Fact | Value | Source |
| :--- | ---: | :--- |
| Host RAM total | `62.62 GiB` | `free` |
| Budget host cap | `52.62 GB` (`total − 10 GiB`) | `--verbose` report |
| Warm pool, `--warm-gib 40` | `3022` slots = `39.84 GB` **pinned** | `--verbose` report |
| Process RSS at `40 GiB` | `42.74 GiB`, `VmSwap 0` | `/proc/<pid>/status` |
| Process RSS at `45 GiB` during preload | `55.2 GiB` (`57,863,412 kB`) | `/proc/<pid>/status` |
| Dense container mapping | `14.66 GiB` total | `model_dense.aeon` size |
| Dense uploaded to VRAM | `12.71 GiB` (`13,643,885,660 B`) | `V4ModelContract::uploaded_dense_bytes` |
| Live dense mapping after init | `0.99 GiB` (`embed.weight`) | `embed_token` |
| Hot preload staging batch | `16` slots × `13.5 MiB` = `217 MiB` | `sq_entries 64 / 4 requests per expert` |
| VRAM held | `23.75 GiB` = `99.06%` of the card | `rocm-smi --showpids` |

**The shape of the problem.** At `--warm-gib 45` the process RSS during preload is `55.2 GiB`, and
the components that must coexist are:

$$14.66\ \text{(dense map)} + 45 \times 0.93\ \text{(pinned)} \approx 56.5\ \text{GiB} \quad\text{against}\quad 62.62\ \text{GiB total}$$

Leaving `≈6 GiB` for the kernel, the GPU driver, the graph's own allocations, and the 4 GiB of swap
already held by other work. That is not enough, and the kernel responds by reclaiming the dense pages
and swapping process memory.

**Two properties make it worse than the arithmetic suggests:**

1. **The Warm pool is pinned** (`hipHostMalloc`, `HostExpertPool`). Pinned memory is neither
   reclaimable nor swappable, so it does not participate in the kernel's memory balancing at all. It
   is a hard claim, and everything else must fit around it.
2. **Swap is not reclaimed eagerly on this host.** `free` reported `3,942 MiB` swap used at the start
   of one of these runs and `6,695 MiB` shortly after, while the machine was otherwise idle. Each
   attempted large-Warm run therefore starts from a worse baseline than the last, and confirms the
   same wall again.

---

## 3. Hypotheses tested and **refuted**

Each of these was a plausible cause, was investigated, and is **not** the cause. Recorded so the work
is not repeated.

| Hypothesis | Method | Verdict |
| :--- | :--- | :--- |
| `embed.weight` duplicates `head.weight` | read both tensors' shapes and the gate | **Refuted.** Distinct `[129280, 4096]` F16 matrices, `tie_word_embeddings = false`; `test_v4_graph_head.cpp` asserts they differ **from the bytes**, not from config. `head.weight` *is* uploaded (`d_lm_head`), so a `[129280, 4096]` matrix already lives in VRAM without difficulty. |
| Keeping `embed.weight` host-side is forced by construction | read `V4ModelContract` and `V4ModelResources` | **Refuted — it is our decision.** It is a VRAM-for-latency trade worth `1,059,061,760 / 14,155,776 ≈ 74` Hot expert slots, against a per-token cost of `4 × 4096 × 2 B = 32,768 B` ≈ `1.3 µs` at 25 GB/s. Normal LLM servers keep the table on device; this engine trades it for slots, deliberately. |
| The dense mmap's page cache consumes the host budget | compute `RSS − pinned` from `/proc` | **Refuted as the *cause*, and fixed anyway (§4).** The mapping is `14.66 GiB`, of which only `0.99 GiB` is read per token; the rest is dead after init. Releasing it frees `13.68 GiB` deterministically — and `45 GiB` still does not load. |
| The Warm preload re-reads through the mmap, polluting page cache | read `preload_warm_experts` → `read_experts_direct_blocking` | **Refuted.** It reads through `expert_direct_fd()` with `O_DIRECT`, which by construction bypasses the page cache. |
| The Hot VRAM pool keeps a host-side mirror | read `ExpertPayloadPool` | **Refuted.** Plain `hipMalloc`; no host allocation. |
| The Hot-preload staging batches are large | read `submission_capacity()` and `direct_requests_per_expert()` | **Refuted.** `sq_entries = 64`, `4` chunk requests per expert, so the batch is `16` slots = `217 MiB`. |

---

## 4. The change that was kept (a real improvement, not the fix)

**`AeonModelLoader::release_dense_pages_except(keep_tensor)`**, driven by
`AeonRuntimeConfig::release_dense_pages_after_upload` (default `true`), called from
`V4ModelHost::initialize` **after the layer/dense uploads and before the Warm preload**.

It `madvise(MADV_DONTNEED)`s the page-aligned complement of one tensor's span in the dense mapping.
The mapping is `MAP_SHARED` and read-only, so its pages are clean and file-backed: releasing them
returns memory to the kernel without a writeback, the file remains the backing store, and no
`LoadedTensor::data` pointer is invalidated. Verified on silicon:

```
[Host] Released 13.68 GiB of dense-container page cache after upload (only embed.weight stays resident)
```

`13.68 GiB = 15,744,996,344 − 1,059,061,760`, i.e. the container minus `embed.weight`, exactly as
designed. **Zero VRAM cost**, no output change.

**Why keep it although it is not the fix:** the kernel may reclaim those pages anyway, but *when* it
does is up to the reclaim heuristics. Releasing deterministically makes the host footprint
independent of that timing — and this footprint matters precisely because the Warm pool is pinned and
therefore cannot be reclaimed at all.

**Corrected reasoning.** A previous comment in `memory_budget.hpp` justified the `10 GiB` reserve with
"the ~13 GiB of resident dense weights that the embedding lookup and every oracle read touch through
the mmap". That was wrong twice: only `0.99 GiB` is touched, and those pages are now released. The
comment is corrected; the reserve's size is unchanged, because §5 shows it is not the binding
constraint anyway.

---

## 5. What is still open

**The cause of the `45 GiB` failure is not identified.** The dense pages were the leading hypothesis
and are now excluded. The remaining candidates, none of them measured yet:

1. **Pinned-vs-total arithmetic with insufficient margin.** `45 GiB` of pins plus `0.99 GiB` of live
   map plus the graph and driver is `≈50 GiB` of a `62.62 GiB` machine, and the swap history of §2
   means the *available* figure at start-up is worse than the total suggests. This is the simplest
   explanation and the cheapest to test.
2. **`hipHostMalloc` costing more than it reports.** Pinned memory is accounted by the driver, not by
   `/proc`; RSS may understate the true claim. `HIP` and `GTT` accounting on AMD is not visible in
   `free`.
3. **Fragmentation or slab pressure** from allocating thousands of 13.5 MiB pinned segments (the
   pool allocates in `SEGMENT_SLOTS = 64` chunks of `864 MiB`).
4. **Swap behaviour under a pinned workload**, which is a hard-to-predict interaction and which §2
   shows is already degraded before the run starts.

**The measurement that would decide it** was attempted twice and did not complete:
`--warm-gib 45 --no-warm-preload` isolates allocation from I/O — if allocation alone stalls, the
problem is the pinned claim; if it completes and only the preload stalls, the problem is the preload
path. It was interrupted both times. **It should be run on a freshly booted host**, with the peak RSS
sampled correctly.

**Two traps found while attempting that measurement, both about the instrument:**

- Sampling the wrong PID. `cmd & ; p=$!` captures the PID of the subshell, not of `aeon_chat`, which
  is why one run reported a peak RSS of `3 MiB`. Sampling must resolve the binary's own PID
  (`pgrep -x aeon_chat`).
- Reading `free` while the process is dying. A killed process releasing tens of GiB of pinned memory
  takes several seconds, during which `free` shows memory still held. A reading taken immediately
  after `kill` is not the settled state.

**Recommendations.**

1. On a freshly booted host, run `--warm-gib 45 --no-warm-preload` and measure peak RSS. That
   splits the problem in two.
2. If allocation is the wall, derive the Warm cap from *available* memory rather than from
   `total − 10 GiB`: the reserve does not account for the pinned pool's true cost, and §2 shows the
   machine's own baseline is both non-zero and non-constant.
3. Consider the honest ceiling: `≈40 GiB` Warm on a `62.62 GiB` host with this graph. If a larger
   Warm tier is genuinely wanted, the lever is the dense mapping (now released) and the `mtp.*` head,
   not the budget literal.
4. Do **not** revisit the refuted hypotheses of §3 without new evidence.

---

## 6. References

- `src/infrastructure/core/aeon_loader.hpp` — `release_dense_pages_except`, `drop_residency`
- `src/architecture/deepseek_v4/core/memory_budget.hpp` — `release_dense_pages_after_upload`,
  `HOST_RAM_RESERVED_BYTES`
- `src/architecture/deepseek_v4/core/v4_model_host.hpp` — the release call site, `preload_warm_experts`,
  `read_experts_direct_blocking`, `preload_hot_experts`
- `src/infrastructure/core/host_expert_pool.hpp` — the pinned Warm allocation
- `src/architecture/deepseek_v4/core/v4_model_contract.hpp` — `uploaded_dense_bytes`
- [PERFORMANCE_LEDGER](../../status/PERFORMANCE_LEDGER.md) **M28**
