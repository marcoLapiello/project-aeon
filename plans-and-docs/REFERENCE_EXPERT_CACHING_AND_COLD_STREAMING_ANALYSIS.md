# Project Aeon: Reference Analysis of Expert Caching and Cold Streaming

## Purpose and Scope

This document records a read-only investigation of the local FreeToken, Colibri, and DwarfStar (DS4) reference checkouts. The investigation focused on expert identity, cache admission and eviction, cold-pool reads, asynchronous I/O, host-to-device promotion, prefetch, in-flight ownership, and the measurements that are relevant to Aeon's VRAM/Warm-RAM/NVMe design.

The references are comparative source material only. They are not Aeon build or runtime dependencies. Paths and line locations below refer to the checkout revisions recorded in the provenance table; line locations are approximate for generated or amalgamated files.

## Provenance

| Reference | Local checkout | Revision | Scope caveat |
| --- | --- | --- | --- |
| FreeToken | `../../aeon-references/freetoken` | `3d919e9bd94fc5454bdb50e09659648443e30f5e` | The GPU LRU implementation is supplied by the pinned external `flashlib==0.3.0` dependency. |
| Colibri | `../../aeon-references/colibri` | `fd93c41aa6ae2c7d1cc1a1e2d6b79dbe6d341708` | DeepSeek V4 is an amalgamated `c/deepseek_v4.c`; its unit list is generated and source line locations can move. |
| DwarfStar (DS4) | `../../aeon-references/ds4` | `6289c516273979173abbc062209a81dd3706b804` | The checkout is shallow with one visible grafted commit; hotlist generation data is not present locally. |

No files were modified in any reference checkout during this investigation.

## Executive Comparison

| Capability | FreeToken | Colibri | DS4 |
| --- | --- | --- | --- |
| Runtime cold reads during decode | Generally no; expert banks are loaded into host memory first | Yes; generic path supports direct I/O and `io_uring` | Yes; pthread `pread` workers, with optional `O_DIRECT` on ROCm |
| Primary GPU cache policy | Unified timestamp LRU | LRU with leases, in-flight protection, pins, and pilot reservations | LRU on ROCm; decayed route hotness plus recency on Metal |
| True bounded Warm-RAM cache | No; host banks are predominantly fully materialized | Yes; working sets and RAM caches are bounded | No managed Warm tier; mmap/page cache and staging are separate mechanisms |
| Single-flight cold load | No general disk-read future table | Strong in the V4 hot-store path | Strong for selected/pending load sets and batch deduplication |
| Expert-union deduplication | Shared miss list for fused copies | Unique routed unions, generally in blocks of 64 | Unique selected IDs mapped back to token selections |
| Disk batching | Startup shard reads with background lookahead | Expert/tensor-level `io_uring` batches, up to 64 expert loads and about 512 reads | Batched pthread read jobs |
| Predictive prefetch | Recent-activity heuristic only | Next-layer pilot prediction, advisory hints, and coupling data | Hotlist seeding, early pending loads, and layer page-in |
| Physical frequency placement | Host-bank layout only | Manifest/shard layout; no general frequency reorder | No; hotlists change admission order, not GGUF offsets |
| GPU promotion | Host gather/copy into unified GPU slots | Optional GPU mirror/refill | Pinned staging plus asynchronous H2D on ROCm; shared buffers on Metal |

The strongest direct comparison for Aeon is a combination of Colibri's generic aligned, batched cold-read path with Colibri V4's lease and single-flight protocol. DS4 contributes useful DeepSeek-specific selected-set deduplication, profile-derived admission seeds, and resident/missing execution splitting. FreeToken contributes a practical hybrid GPU/CPU overflow path and a mature double-buffered prefill design, but its cold storage is primarily a startup concern rather than a request-time NVMe tier.

## FreeToken

### Architecture and data flow

FreeToken uses a host-resident expert-bank design with a bounded unified GPU slot cache:

```text
checkpoint / FTW / Safetensors
    -> per-layer HostBank tensors
    -> optional PINNED or LOCKED host residency
    -> unified GPU expert slots
    -> routed GEMM
```

`Engine._init_offload_moe_cache()` in `python/freetoken/engine/engine.py:509` constructs `ExpertBanks` and `OffloadMoeCache`, attaches the cache to every `OffloadMoELayer`, and optionally creates a CPU executor. Decode routing in `python/freetoken/layers/moe.py:255` and `:397` calls `ensure_experts()`, which maps `(layer, expert)` IDs to GPU slots, copies missing experts, and then executes the expert GEMM.

Prefill follows a different path. `python/freetoken/layers/moe.py:341` and `python/freetoken/moe/offload_cache.py:606` materialize a complete expert layer into the first `num_experts` slots or into one of two borrowed full-layer buffer regions. This is not individual request-time cold streaming.

Hybrid decode sends cache hits and a capped subset of misses to the GPU, while overflow misses are submitted to the persistent CPU executor before GPU PCIe fetch and GEMM. The owning path is `python/freetoken/layers/moe.py:292`.

### Expert identity, residency, and admission

The expert identity and bank layout are defined in:

- `python/freetoken/moe/expert_pieces.py:27`, `:40`, `:71`, and `:122` for layer/expert mapping and packed source information.
- `python/freetoken/moe/expert_banks.py:31`, `:70`, and `:275` for bank construction and loading.
- `python/freetoken/checkpoint/ftw.py:200`, `:357`, and `:424` for FTW reading and bank loading.
- `python/freetoken/moe/host_banks.py:43`, `:78`, `:246`, and `:268` for `PINNED`, `LOCKED`, and `PAGEABLE` host residency.

`OffloadMoeCache` in `python/freetoken/moe/offload_cache.py:104` maintains a unified slot pool. `slot_for_id[layer, expert]` maps an expert to a GPU slot and `id_of_slot[slot]` records the reverse flat ID. All layers and bank types share the same slot and eviction-index arrays.

GPU admission is delegated through `python/freetoken/moe/offload_kernels.py:19` to `flashlib.kernels.slot_cache.lru_ensure`; LRU is the only exposed cache policy. The hybrid path is implemented by `ensure_experts_hybrid()` at `offload_kernels.py:43` and the `_ensure_experts_hybrid_kernel()` around `:290`.

The hybrid selection defaults to the most recently active missing experts, with lower expert ID as a tie-break. `FREETOKEN_HYBRID_FETCH=lowest_id` restores an older routing-blind choice. Overflow misses remain nonresident and are rewritten to `-1`, causing the CPU GEMV path to process them. `tests/moe/test_hybrid_fetch.py:89` compares the GPU and CPU state transitions.

### Prefill overlap and transfers

The two-buffer prefill path is owned by:

- `OffloadMoeCache.begin_prefill()` at `python/freetoken/moe/offload_cache.py:645`.
- `prefetch_prefill_layer()` at `:668`.
- `_prefetch_split()` at `:746`.
- `wait_prefill_layer()` at `:818`.
- `release_prefill_layer()` at `:832`.

The borrowed regions occupy `[0, 2 * num_experts)`. Reusing a region invalidates its old slot ownership and zeros its usage, making those slots first eviction candidates. A row is considered a hit for prefill D2D only when its slot is at least `2 * num_experts`; experts in volatile buffer slots are reloaded from host.

Decode movement uses `copy_missing()` at `offload_cache.py:1011` and the JIT implementation in `python/freetoken/kernel/csrc/jit/fast_index_copy.cuh:475`. The fused path shares one miss list across banks and launches one multi-bank kernel instead of one kernel per bank.

Prefill uses a dedicated copy stream, ready and release events, and a begin-of-prefill fence. With hit-D2D enabled, resident rows are gathered cache-to-buffer on the compute stream and missing contiguous runs are submitted with one `cudaMemcpyBatchAsync` call. The binding is in `python/freetoken/kernel/batch_memcpy.py:29` and `python/freetoken/kernel/csrc/jit/batch_memcpy.cuh:16`.

### Cold loading and CPU fallback

Original Safetensors loading is implemented by `_read_shard_odirect_parallel()` and `iter_expert_tensors_parallel()` in `python/freetoken/models/weight.py:40` and `:77`, with model-specific readers such as `python/freetoken/models/qwen3_5_moe/weight.py:658` and `:867`. The reader uses multithreaded `O_DIRECT`/`preadv` for complete relevant shards and queues the next two shards from a background thread. FTW uses aligned 4096-byte entries and either threaded direct reads or an mmap plus `MADV_SEQUENTIAL` fallback.

This means that FreeToken's runtime miss usually reads from a fully loaded `HostBank`, not from NVMe. `HostBank` defaults to lazy anonymous mmap, fills it, and then calls `cudaHostRegister`; `cudaHostAlloc` is selectable with `FREETOKEN_BANK_CUDA_ALLOC`. `PINNED` banks feed the GPU movement path, while `LOCKED` and `PAGEABLE` banks are CPU-executor paths.

The CPU fallback is implemented in `python/freetoken/moe/cpu_executor.py:144`, `:493`, `:505`, `:543`, and `:589`. Persistent task descriptors are keyed by `(layer_id, batch_size)`, and the C++ executor processes one current task/generation at a time before consuming the next task.

### Scheduling, tests, and tradeoffs

The main scheduling paths are `python/freetoken/scheduler/scheduler.py:197` (`overlap_loop`), `python/freetoken/scheduler/prefill.py:251`, and `python/freetoken/scheduler/decode.py:32`.

Relevant tests include:

- `tests/moe/test_offload.py:128` and `:243` for two-buffer prefill and borrowed-slot invalidation.
- `tests/moe/test_hybrid_fetch.py:29`, `:89`, and `:119` for hybrid selection and state parity.
- `tests/moe/test_prefill_hit_d2d.py:63`, `:83`, and `:117` for batched H2D and hit/miss extremes.
- `tests/moe/test_fused_copy.py:43` for fused versus legacy copy behavior.
- `tests/moe/test_cpu_moe.py:41` and `:442` for CPU/GPU parity and flag-handshake behavior.

Useful command-line controls are defined near `python/freetoken/server/args.py:552`: `--expert-load`, `--moe-cache-size`, `--moe-cache-rate`, `--moe-cache-auto`, `--moe-cache-policy lru`, `--moe-cpu-threads`, `--moe-cpu-layers`, `--moe-hybrid-max-fetch`, `--disable-moe-prefill-overlap`, and `--moe-prefill-hit-d2d`.

Source comments record approximately 7x scaling for an 8-thread single-shard read, approximately 31 GB/s for the host-to-device gather path, a roughly 22% end-to-end regression for an unfavorable CUDA batch-copy threshold, and roughly 6 ms per step of callback overhead for a 75-layer model before flag-handshake optimization. These are source-embedded claims, not measurements reproduced for this analysis.

The principal tradeoff is simplicity versus tier depth. FreeToken eagerly materializes the full expert bank in host memory and makes runtime GPU misses cheap and predictable. It does not provide bounded Warm RAM plus request-time NVMe promotion. It also has no persistent frequency-ranked hotlist, routing-history prefetch queue, or disk-backed miss coalescer. Prefill overlap consumes `2 * num_experts` unified cache slots and requires pinned host banks.

## Colibri

Colibri has two relevant designs: a generic GLM/Colibri path with the strongest direct cold-streaming implementation, and a DeepSeek V4 path with the strongest lease and single-flight correctness model.

### Generic path: working set, RAM cache, and cold reads

The generic flow is:

```text
routed (layer, expert) union
    -> per-layer working-set slots
    -> direct or buffered tensor reads
    -> expert computation
    -> bounded RAM cache promotion
    -> optional VRAM mirror promotion
```

The generic RAM slot is `ESlot` in `c/colibri.c:390`. It contains the expert ID, quantized tensors, slabs, usage state, and `in_flight` protection. `eslot_lru_victim()` is near `:413`.

`expert_load_impl()` at `c/colibri.c:2786` resolves the gate, up, and down tensor records for `model.layers.<L>.mlp.experts.<E>`, reads them into a slot, and constructs quantized-tensor views. Contiguous weight payloads use aligned `O_DIRECT` in the path near `:2910`; scales remain buffered because they are small. Noncontiguous layouts fall back to multiple buffered reads.

The residency index and publication protocol are at `c/colibri.c:5066`: `ecache_indexed()`, `ecache_publish()`, `ecache_reserve()`, and `ecache_hide()`. A victim is not eligible while in flight or reserved for pilot work. A freed slab-less slot is reused while live slabs remain below the configured capacity, preventing accidental cache growth.

The generic Linux `io_uring` implementation is concentrated around `c/colibri.c:3338` (`uring_load_add()`), `:3446` (`uring_reap()`), `:3478` (`uring_wait_load()`), and `:3486` (`uring_finalize_load()`). The wrapper in `c/uring.h:20` uses a single-owner ring, and `coli_uring_prep_read()` near `:89` sets `IOSQE_ASYNC` so regular-file reads are dispatched through io-wq rather than serializing during submission.

Under `URING=1`, up to 64 expert loads and 512 individual reads can be assembled into one ring batch. The generic prefill path constructs a unique expert union and processes it in blocks of 64, reusing one loaded expert for all batch rows that selected it.

### Generic asynchronous pipe and prefetch

The optional pthread path is owned by `PipePool` around `c/colibri.c:3549`. A generation-tagged cursor is published with a release-store; workers claim jobs with CAS and read layer/expert fields only after winning the claim. `pipe_wait()` is the correctness barrier immediately before an expert is consumed.

Route-aware prefetch is implemented through `pilot_prefetch()` around `c/colibri.c:6690`, with real future-layer loads in `pilot_realload()` near `:6414` and batched URING loads in `pilot_uring_batch()` near `:6485`.

The pilot modes have distinct behavior:

- `PILOT_REAL` predicts and performs a real future-layer RAM load.
- Plain `PILOT` issues advisory `POSIX_FADV_WILLNEED` hints.
- `PILOT_TWO` adds a shared-expert correction.
- `COUPLE` uses an offline layer-coupling file.

Pilot reservations use `eid = -(expert + 2)` so other workers can see that an expert is already reserved before the unlocked disk read begins. This prevents duplicate pilot work but is not a guarantee that the load will complete before demand arrives.

History-ranked admission and replacement are implemented near `c/colibri.c:9721` (`pin_load()`) and `:8377` (`repin_pass_limit()`). These can pin frequently used experts in RAM and feed optional GPU tiers. The generic demand path does not provide the same cross-thread arbitrary-demand single-flight table as the V4 hot store.

### V4 path: leases and single-flight loading

The DeepSeek V4 expert record and slot are defined around `c/deepseek_v4.c:7255` as `V4ExpertRecord`, `V4ExpertSlot`, and `V4ExpertStoreState`. The base store starts around `:7603`; its `lookup()` is synchronous and `prefetch()` is near `:7741`.

The production hot wrapper begins around `c/deepseek_v4.c:8525`. `lookup_hot()` releases the mutex while reading from disk and implements single-flight loading. `v4_read_expert_record()` near `:8205` uses aligned direct windows for contiguous payloads and buffered fallback reads.

The key V4 invariant is an indexed loading entry. `slot_by_expert[layer * experts_per_layer + expert]` indexes both published and loading records. `loading_expert` reserves a slot before the unlocked read. A second lookup waits on `load_ready` and retries rather than issuing a duplicate read.

Active leases increment `references` and `active_leases`. Eviction considers only unreferenced slots, and `destroy()` asserts that active leases are zero. This is the clearest reusable contract for Aeon expert handles: a promotion may be asynchronous, but a resident buffer cannot be recycled while a consumer owns a lease.

V4 uses intrusive per-partition LRU with optional history-ranked pins. Prefill can set `pool_layer` so the active layer borrows unused slabs from other partitions while preserving a per-layer decode reserve.

The V4 GPU mirror is attached through `coli_v4_gpu_expert_attach_async()` around `c/deepseek_v4.c:10386`. The caller must retain the RAM lease until the asynchronous device stream has drained. Route-aware prefill-bank refill begins near `:10615`.

### Generic storage topology and GPU promotion

Safetensor indexing, shard descriptors, direct-I/O twins, and mirror support are in `c/st.h:140`, with multi-directory indexing near `:462`, `st_prefetch()` near `:774`, and persistent mmap support near `:1005`.

Multi-SSD routing is deterministic on `(layer, expert)`. Mirror files are accepted only after size and header validation. `COLI_DISK_WEIGHTS` or startup bandwidth probing determines weighted replica cuts, while `COLI_MODEL_DIRS` can split distinct shards across drives.

Normal generic RAM-to-VRAM promotion uses `coli_cuda_tensor_upload()`, which synchronously copies weights and scales. Asynchronous streams and pinned staging are used more strongly in expert-group execution than in ordinary resident-weight upload. The V4 async mirror path is therefore the more relevant Colibri example for Aeon.

### Tests, benchmarks, and measured evidence

Relevant tests include:

- `c/tests/test_uring.c:31` for layout, completion, routed mirror reads, primary fallback, drive accounting, and pilot publication.
- `c/tests/test_eslot_inflight.c:14` for in-flight GPU-borrowed slots and pilot reservations.
- `c/tests/test_tier.c:9` for hysteresis, saturated counters, heat decay, and LFRU tie-breaking.
- `c/tests/test_deepseek_v4.c:621` for V4 single-flight loading, distinct-expert overlap, lease clearing, indexed LRU eviction, and statistics.
- `c/tests/test_st_mirror.c:64` and `c/tests/test_st_pread.c:25` for mirrors and short-read behavior.
- `c/iobench.c:1` for parallel random expert-sized reads in buffered and direct modes.

Documented, not independently reproduced, measurements include:

- Approximately 11 GB of expert reads per cold token in one generic GLM benchmark, producing about `0.05-0.1 tok/s` on a 1 GB/s WSL2/VHDX disk.
- `O_DIRECT` scaling from 72 MB/s at queue depth 1 to 207 MB/s at queue depth 16 in `docs/glm53-flash.md:91`.
- Decode improvement from about 44 s/token cold to 20 s/token with a warm cache in the same document.
- Next-layer pilot recall of 71.6% versus 41.3% for previous-token reuse, with about an 11 percentage-point hit-rate improvement for `PILOT_REAL` in `docs/tuning.md:184`.
- DeepSeek V4 expert-bank refill at roughly 6 GB/s and routed-read contribution near 75% of decode time at a 6% VRAM hit rate in `docs/deepseek-v4.md:249`.

Colibri also records important negative evidence: hint-only prefetch can be net negative on saturated disks, and transient full-layer prefetch was worse on a 3.3k-token prompt than route-aware incremental refill.

### Tradeoffs

Colibri's generic path is the closest reference for Aeon's cold tier because it combines bounded residency, aligned direct I/O, multiple storage drives, batching, and speculative prefetch. However, it still uses CPU buffers between disk and GPU; `io_uring` hides submission and completion latency but does not itself provide GPU DMA.

The V4 path has the better ownership model but not the generic path's normal `io_uring` implementation. Its fast path depends on a contiguous `[scales][weights]` layout; packed checkpoints that scatter the three matrices and scales incur more reads. More cache is not automatically better: after cold misses fall, NUMA placement, upload pressure, and CPU matmul bandwidth can dominate.

## DwarfStar (DS4)

DS4 streams expert slices from GGUF tensor offsets. It has distinct Metal and ROCm implementations but no persistent Aeon-style Warm-RAM expert registry.

### Expert source indexing and cache data flow

`ds4.c:model_open` around `:2532` opens the GGUF file read-only and creates a file-backed mmap. `graph_stream_expert_table_make` around `:4818` converts each layer into a streaming descriptor containing the model map, layer ID, expert count, absolute tensor offsets, and per-expert byte sizes.

The runtime then follows this pattern:

```text
router-selected expert IDs
    -> cache lookup by model/layer/expert/source geometry
    -> resident buffer or address-table hit
    -> slot allocation or reuse on miss
    -> gate/up/down slice reads
    -> device-visible promotion
    -> cache installation and routed execution
```

The cache key includes model identity, layer, expert, absolute tensor offsets, and byte sizes. This protects against accidentally reusing a slot across incompatible model layouts.

Metal cache entries are defined in `ds4_metal.m` around `:954` as `ds4_gpu_stream_expert_cache_entry`, with layer, expert, source offsets, byte sizes, `last_used`, `use_count`, `inflight_seq`, and slab metadata. ROCm entries are defined in `rocm/ds4_rocm_runtime.cuh` around `:115` as `cuda_stream_resident_expert`, with global resident vectors and maps near `:300`.

### Metal cache and cold reads

Metal cache lookup and installation are around `ds4_metal.m:15949` (`ds4_gpu_stream_expert_cache_peek`), `:16094` (`ds4_gpu_stream_expert_cache_get_protected` and install logic), and `:16707` (`ds4_gpu_stream_expert_cache_load_selected_missing_with_source`). Batched prefill deduplication is around `:17285`.

Metal uses file reads directly into `MTLStorageModeShared` expert buffers, so the normal miss path has no separate H2D copy. `ds4_gpu_stream_expert_pread_into()` around `:13290` performs robust `pread` loops. The pthread pool starts around `:13469`; the default is nine read threads, capped at 18 by `DS4_METAL_STREAMING_EXPERT_PREAD_THREADS`.

Each missing expert creates gate, up, and down tasks. The selected-missing loader batches these tasks, deduplicates repeated IDs through `source_slots`, and installs completed entries. `ds4_gpu_stream_expert_pending_load_finish()` around `:16353` joins the pending read batch. Early loading is enabled by default unless disabled by `DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD`.

The split path commits resident work first, waits for the cold loads, and then runs missing experts. It is considered worthwhile when at least three experts are missing and some are resident.

Metal uses command-buffer sequence epochs (`batch_seq`, `owned_seq`, `pending_max_seq`, and `done_seq`) plus per-entry `inflight_seq` to prevent reuse while kernels still reference a buffer. `ds4_gpu_stream_expert_cache_mark_entries_inflight()` and `ds4_gpu_stream_expert_cache_wait_inflight()` protect the lifetime; recursive waits are rejected to avoid service-thread deadlock.

The Metal cache has one global slab size class of approximately `2 * gate_bytes + down_bytes`, established by `ds4_gpu_stream_expert_cache_note_expert_size()` around `:12949`. Mixed-size routed layers bypass that slab cache and use mapped views. Eviction in `ds4_gpu_stream_expert_cache_prune_layer()` around `:15231` and `prune_global()` around `:15884` selects the lowest route-hotness entry, with oldest `last_used` as tie-break. Current selected IDs, protected entries, and in-flight entries cannot be evicted.

Route hotness is incremented by `ds4_gpu_stream_expert_cache_note_selected_hotness()` around `:14259` and halved every 16 decode tokens. Reusable buffers are selected globally by `take_reusable()` around `:15305` and its batch form around `:15417`.

### ROCm cache, direct reads, and H2D promotion

ROCm selected loading is `cuda_stream_selected_load()` around `rocm/ds4_rocm_runtime.cuh:4054`; batch preparation is `cuda_stream_batch_selected_prepare_from_host()` around `:3122`; full-layer loading is `cuda_stream_layer_expert_cache_load()` around `:2562`. The shared API bridge is in `rocm/ds4_rocm_current_api_compat.cuh` around `:204`, `:228`, `:251`, and `:299`.

`cuda_stream_resident_alloc()` around `:1653` allocates fixed-size device slots from approximately 1 GiB slabs, with dedicated allocations for mixed sizes. `cuda_stream_resident_evict_one()` around `:1489` is LRU-based and excludes experts selected by the current request. `DS4_ROCM_STREAM_EVICT_PAST_LAYERS_FIRST` can prefer old layers, and `DS4_ROCM_STREAM_FREE_RESERVE_GB` defaults to a 16 GiB free-memory reserve.

ROCm opens an optional `O_DIRECT` descriptor through `ds4_gpu_set_model_fd()` around `:6313`. `cuda_stream_read_job_run()` around `:1952` aligns offset and read size to the filesystem/device block size, with buffered `pread` fallback. The minimum direct alignment is 512 bytes.

Pinned host staging is allocated by the read-job preparation path. `cuda_stream_read_jobs_start()` around `:2247` submits one active job set to a persistent worker pool. The default is 16 workers, capped by `DS4_ROCM_STREAM_READ_WORKERS` and a compile-time 24-job expert-tensor limit. Each worker performs the file read, enqueues `cudaMemcpyAsync` on its own nonblocking upload stream, synchronizes that stream, and then marks the job complete. The pinned buffer is staging, not residency.

ROCm batch preparation deduplicates all `n_tokens * n_selected` IDs into unique experts and creates a compact pair-to-unique map. HIP/CUDA events protect selected-cache reuse, upload readiness, and batch-cache reuse. `cuda_stream_resident_reclaim_wait()` waits for selected and batch reuse events before eviction or free.

The main limitation is that each worker serializes its own read and H2D upload. Multiple workers overlap across experts, but disk scheduling and DMA are not independent queues in the way Aeon's cold-read and SDMA streams are intended to be.

### Hotlists, prefetch, and physical placement

`ds4_streaming_hotlist.inc` is generated from expert profiles and sorted by hits/weight. It contains 6,884 unique `(layer, expert)` pairs for the Pro table and 6,436 for the Flash table; `ds4_streaming_hotlist_glm52.inc` contains 6,501 pairs across 75 routed layers. The arrays contain layer/expert pairs only; original hit counts and weights are not embedded.

Runtime consumption is in `ds4.c:21596` (`metal_graph_streaming_expert_hotlist_load_default`), `:21642` (`metal_graph_streaming_expert_preload_count`), and `:21498` (`metal_graph_streaming_expert_hotlist_load_file`). Pro and Flash seed ranked built-in lists unless `--ssd-streaming-cold` disables hotlist seeding. GLM defaults to demand-fill unless an explicit preload count or automatic cap is provided. Custom lists use `DS4_ROCM_STREAMING_EXPERT_HOTLIST` or `DS4_METAL_STREAMING_EXPERT_HOTLIST`.

Profile generation is in `ds4.c:1465` (`ds4_expert_profile_record`), `:1619` (`ds4_expert_profile_write_hotlist_file`), and `:1683` (`ds4_expert_profile_close`). It records selected IDs, router weights, adjacent overlap/Jaccard, and simulated LRU hit rates at capacities from 1 through 384. The runtime hotlist changes admission/preload order, not the physical GGUF layout: experts remain at `tensor_base_offset + expert_id * per_expert_bytes`.

Optional prefill page-in is implemented by `metal_graph_stream_pread_range()` around `ds4.c:19359`, `metal_graph_stream_prefill_layer_pagein_start()` around `:19809`, and `metal_graph_stream_prepare_start_if_needed()` around `:20051`. The layer helper can run ahead by up to four layers using `pread`, `F_RDADVISE`, and `POSIX_MADV_WILLNEED`.

ROCm full-layer preloading is owned by `rocm_graph_stream_layer_expert_load_start()` around `ds4.c:18981` and `rocm_graph_stream_layer_expert_load_ready()` around `:19044`. It activates for prefill batches of at least 1,024 tokens, uses up to eight selected-token rows for later cache seeding, and uses temporary storage that can flush the dynamic resident cache when released.

### Tests, benchmarks, and measured evidence

Relevant tests include:

- `tests/test_metal_ssd_experts.c` for route steps, small cache budgets, eviction, repeated batch sizes, exact output parity, cache bounds, mapping replacement, and mapping lifetime.
- `tests/test_ssd_cache.c` for automatic budget sizing, context reductions, percentage overrides, expert caps, and one-expert fallback planning.
- The SSD section of `QA_BEFORE_RELEASES.md` for cold mode, one-slot mode, oversized hints, exact logits, and memory-pressure checks.

The checked-in QA record reports these M5 Max SSD-streaming medians:

- GLM initial 2K: prefill `111.73 -> 121.28 t/s`, generation `6.11 -> 11.85 t/s`.
- GLM 1K append: prefill `83.33 -> 104.43 t/s`, generation `7.11 -> 14.89 t/s`.
- DeepSeek initial 8K: prefill `285.12 -> 296.59 t/s`, generation `7.21 -> 7.91 t/s`.
- DeepSeek 4K append: prefill `257.07 -> 259.44 t/s`, generation `9.42 -> 11.58 t/s`.

The same QA record reports exact logit equality across compared frontiers, approximately 31 seconds per token in a one-slot direct-read case, and an oversized hint reduced to 0.36 tok/s. Selected-address batching reduced a 16-token short append from 30.8 to 2.9 seconds, with generation improving from 4.09 to 5.11 tok/s while preserving logits and text.

No comparable checked-in ROCm cold-streaming throughput table was found; the available ROCm benchmark document covers resident/prefill measurements rather than cold expert streaming.

### Tradeoffs

DS4 supplies useful patterns for model/source identity validation, bounded global residency, route-informed admission, selected-set deduplication, asynchronous reads, and resident/missing execution splitting. It does not implement a complete three-tier cold-pool design:

- There is no durable Warm-RAM expert registry.
- There is no `io_uring` ownership model.
- There is no DMA queue independent of host read workers.
- There is no physically frequency-ordered expert container.
- Global cache policy differs by backend: Metal uses decayed route hotness plus recency, while ROCm primarily uses LRU and optional past-layer preference.

Tiny budgets preserve correctness through mapped views or fallback paths, but the one-slot measurement demonstrates that capacity correctness can become unusably slow. Hotlists are useful admission seeds, not proof that per-token routing can be predicted.

## Cross-Repository Design Findings for Aeon

### 1. Separate storage residency from compute residency

FreeToken shows the convenience of fully materialized host banks, while Colibri and DS4 show the need for bounded runtime working sets when the model exceeds host memory or when cold reads must happen during execution. Aeon's explicit Hot/Warm/Cold registry should remain the owning abstraction rather than treating mmap or the OS page cache as an implicit Warm tier.

### 2. Use single-flight ownership for every asynchronous promotion

The most reusable Colibri V4 invariant is:

```text
lookup -> indexed loading marker -> one physical read -> load-ready signal
       -> resident publication -> lease-protected consumption
```

A second request for the same `(layer, expert)` must wait on the existing load rather than submit another disk read. A resident entry must remain protected until the GPU event associated with its consumer has completed. This should apply across Cold-to-Warm, Warm-to-Hot, and any Hot eviction/demotion transition.

### 3. Deduplicate routed unions before I/O

Colibri generic and DS4 both construct a unique expert union before loading. This is especially important for prefill, where many token rows select the same expert. The request should carry a unique expert list plus a compact mapping back to token selections; the storage layer should never see one independent read per token occurrence.

### 4. Keep disk scheduling and DMA scheduling independently observable

Colibri's `io_uring` path improves queue depth and completion handling, but its data still travels through CPU buffers. DS4 overlaps multiple worker-local read/H2D sequences, while Aeon already separates a cold I/O stream from the ordinary SDMA stream. Future measurements should report at least:

- disk submission-to-completion latency,
- Warm promotion latency,
- H2D enqueue-to-ready latency,
- time spent waiting for a lease,
- physical bytes read versus logical expert bytes requested,
- number of duplicate or coalesced requests.

Without those counters, logical hit rate can conceal a saturated or serialized physical path.

### 5. Treat prefetch as a hypothesis with a rejection criterion

DS4 hotlists and Colibri pilot prefetch demonstrate two different uses of historical information:

- admission ordering for experts known to be historically frequent,
- speculative loading of future-layer experts.

Colibri documents useful pilot recall but also warns that advisory prefetch can be net negative on a saturated disk. Aeon should measure prediction coverage, false-positive bytes, queue interference, and end-to-end latency before enabling a new prefetch policy. A hotlist must not be treated as evidence that arbitrary future token routing is predictable.

### 6. Frequency-informed placement is distinct from runtime LRU

DS4's profile-derived hotlists change admission order without changing physical expert offsets. Aeon's planned contiguous `.aeon` repack can go further by placing high-frequency experts in favorable physical regions, but the benefit must be measured as physical read behavior, not only cache hit rate. If physical reads are already random or the path is just-in-time latency-bound, logical reordering alone will not solve the stall.

### 7. CPU fallback is a valid pressure valve

FreeToken's hybrid path is the clearest operational example: GPU cache hits and a bandwidth-limited subset of misses remain on the GPU, while overflow misses execute on a persistent CPU path. This is aligned with Aeon's stated Colibri-trap mitigation: when an expert is absent from both Hot and Warm tiers, compute fallback may be preferable to stalling the GPU on a synchronous Cold read. The tradeoff is that CPU/GPU numerical parity and scheduler fairness must be tested explicitly.

### 8. More capacity is not automatically more throughput

Colibri's measurements show that after cold misses decrease, RAM/VRAM residency, NUMA placement, upload pressure, and compute bandwidth can dominate. Aeon's M17 result similarly showed that the current LRU already captures available short-term token locality. Increasing capacity without changing the request schedule will not remove just-in-time wait time.

## Recommended Aeon Follow-Up Measurements

Before adopting a reference mechanism, record the following on the current DeepSeek-V4 pipeline:

1. **Cold request coalescing:** count unique requested experts, physical reads, duplicate requests, and single-flight waits per layer and per token.
2. **Tier transition timeline:** record timestamps for Cold read submission, Cold completion, Warm publication, Hot upload submission, Hot readiness, and GPU consumption.
3. **Prefetch quality:** measure coverage, lead time, false positives, wasted bytes, and whether prefetch increases demand-read queue latency.
4. **Placement effect:** compare the current `.aeon` layout with frequency-ordered contiguous placement using physical read offsets, queue depth, throughput, and token parity.
5. **Lease correctness:** stress eviction and promotion while kernels are in flight; assert that no slot is recycled before its associated HIP event completes.
6. **CPU fallback boundary:** measure whether routing a cold miss to CPU is faster than waiting for a Cold read under realistic contention, while checking bit-exact or tolerance-bounded output parity.
7. **Physical versus logical bandwidth:** report SSD device bytes and model-backed bytes separately so cache-layer metrics do not overstate storage throughput.

## Bottom Line

FreeToken demonstrates robust host/GPU offload and CPU fallback but treats disk loading mostly as startup work. Colibri demonstrates the most complete bounded cold-streaming architecture, especially when its generic `io_uring` path is combined with the V4 hot store's leases and single-flight loading. DS4 demonstrates DeepSeek-specific cache identity, selected-set deduplication, profile-derived admission seeds, and backend-specific asynchronous promotion, but it does not provide a full Warm-RAM tier or independent disk/DMA scheduling.

For Aeon, the highest-value reusable design is therefore not a direct copy of one repository. It is a combination of Colibri's batched direct I/O and ownership protocol, DS4's DeepSeek-aware selected-set handling and hotlist discipline, and FreeToken's hybrid CPU/GPU overflow strategy, all measured against Aeon's existing three-tier registry and physical RDNA3 execution path.
