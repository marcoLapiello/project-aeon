# Multi-GPU / Parallel Topology — What We Are Building

**Status:** requirements. This document states *what* is to be built and *why it matters* — outcomes, not procedure. It owns the requirements and the scope boundary; it does not own the implementation sequence (that lives in an execution plan) or measured results (those live in the Performance & Accuracy Ledger).

**Scope of this document:** running one model instance across **more than one GPU on a single host** — the parallel *layout* (how dense weights, attention/KV state, and routed experts are spread across devices) and the rules that layout must satisfy. It is a G1–G4 concern; it is **orthogonal to the G5 server** and does not require request concurrency.

**See also:** [Prefill Supply and Multi-GPU Scaling](../../analysis/historical/PREFILL_SUPPLY_AND_MULTIGPU_SCALING_ANALYSIS.md) §7 — the one worked multi-GPU analysis, whose §9.5 pre-questions this document answers and whose remaining questions it inherits. [Project vision](../../reference/strategy/PROJECT_AEON_VISION.md) §5 — the earlier TP-vs-PP rationale, written under premises this document revises. The deferred row in [PROJECT_STATUS.md](../../status/PROJECT_STATUS.md) §3. The testbed in the [Performance & Accuracy Ledger](../../status/PERFORMANCE_LEDGER.md): 4× RX 7900 XTX, 24 GB each, PCIe 4.0 x16, P2P enabled, one 6.33 GB/s NVMe, 64 GB DDR4.

---

## 1. What we are building

A single model instance that runs on the host's GPUs **together** and makes **single-stream** inference faster, without waiting for concurrency to pay off.

The three levers, in the order they matter:

- **Capacity.** The routed store is `145.12 GiB`; one 24 GB card holds ~`660` experts (`~8.7 GiB`), roughly `6%` of it. Across four cards the aggregate hot pool grows roughly fourfold once the dense backbone and KV are accounted, so a far larger share of the model is resident and Cold-tier traffic falls.
- **Supply.** With more of the model resident, the per-layer Cold traffic is lower, which permits a **wider lookahead window** during a swept prefill.
- **Parallel H2D.** Each rank has its own PCIe link, so expert bytes reach VRAM in parallel — bounded by host DDR rather than by one link alone.

The workload remains the one the engine is built for — **one live conversation** — and the benefit is claimed for it directly, not for a concurrent one.

---

## 2. Requirements

**R1 — The parallel topology is runtime configuration.**
The engine's parallel layout — the tensor degree, the pipeline degree, and their combination — is a runtime parameter set, not a compile-time choice. No component branches on a specific degree; the topology is data the engine reads, and every configuration runs the same code with the same invariants.

**R2 — The single-device path is preserved exactly.**
At the degenerate topology (one device) the engine reproduces today's behaviour bit-for-bit and allocates no additional memory. Adding multi-GPU support must not perturb the one-GPU result on any axis: numerics, allocation, or the supply path. "Preserved exactly" is a claim about the *computation*; which device is chosen is a separate concern, now governed by R12 (explicit index) rather than the display-bypass heuristic.

**R3 — Partitioning causes no read amplification.**
For any supported topology, the bytes read from the Cold tier to make a tensor resident once across all ranks equal that tensor's own size, read once. No rank reads or transforms bytes belonging to another rank. The split happens at or before the H2D boundary, never by having every rank read the whole tensor.

**R4 — The partition structure lives in the stored artifact.**
Weight partitioning is expressed in the weight format, not applied at runtime. Any supported topology is served from **one unmodified artifact** — no per-rank artifact, no conversion pass, no temporary SSD copy. A rank's slice of every tensor is contiguous and 4 KiB aligned, so it remains `O_DIRECT`-clean.

**R5 — The artifact declares its maximum decomposition, rig-independently.**
The artifact declares the deepest decomposition it carries, and a supported topology degree must evenly divide that maximum. The format names no device count, PCI bus topology, or GPU model; the same artifact serves any rig up to the declared maximum.

**R6 — The logical expert remains one unit.**
Cache, residency, byte accounting, and supply scheduling treat an expert as one whole unit no matter how its pieces are distributed across ranks. Partial residency of an expert is never observable, useful, or representable, and the registry's slot arithmetic stays topology-agnostic.

**R7 — Tensor and pipeline partitioning are independent, composable axes.**
Neither mechanism is subordinate to nor excludes the other; a topology is their combination, with the single-device case as the degenerate configuration. Supporting one must not preclude or complicate the other.

**R8 — Attention and KV placement follows the attention class.**
Head-partitioned attention state is sharded across ranks; a shared latent that cannot be split by head — DeepSeek-V4's compressed MQA, whose key/value latent is common to all 64 heads — is replicated. The engine may assume neither: correctness holds for both placements, and the choice is derived from the topology together with the architecture, never hard-coded.

**R9 — Cross-rank communication preserves numerical equivalence.**
Any collective introduced by partitioning (an all-reduce or equivalent) leaves the result within the project's existing FP16/BF16 tolerance relative to the single-device result. No correctness gate may depend on a cross-device reduction order.

**R10 — Zero runtime dependency and rig-agnosticism are preserved.**
Multi-GPU support introduces no Python/PyTorch runtime dependency and bakes no specific device count, PCI topology, or GPU model into engine logic. Hardware probing informs configuration; it never becomes a compiled assumption (AGENTS.md rules 1 and 6).

**R11 — The benefit is single-stream.**
The design must improve single-stream inference on its own — higher hit rate, lower per-layer Cold traffic, wider lookahead — and its correctness and its gain must not be contingent on serving more than one request at a time.

**R12 — Device selection is explicit and by index.**
The set of devices the engine uses is a runtime parameter: a list of HIP device indices, in the established form other engines expose (e.g. `--device-ids 0,1,2,3`, or `1,3` for a subset). The engine does not *infer* its device from a display-driving heuristic or a hard-coded PCI identifier; the visible-device environment variable remains the other way to constrain it. The selected device count and the parallel topology must agree — the topology is realised across exactly the selected devices — and the default preserves a valid single-device run.

**R13 — The VRAM budget is a user-set utilization fraction.**
The engine takes a utilization target in `0.0–0.99` — e.g. `0.95` means every selected GPU may occupy up to 95% of its own total VRAM — applied uniformly across the devices, mirroring the reference engines' `gpu_memory_utilization`. Every per-device allowance (the Hot pool, the workspaces, the KV state) derives from that fraction against that device's total, replacing the current fixed hidden headroom. The budget stays a **refusal, not a best effort**: a configuration that does not fit under the fraction is rejected with a named reason rather than silently over-committing, and the recommended default is one value for every rig rather than a per-machine constant.

---

## 3. In scope

- The parallel layout as runtime configuration: tensor degree, pipeline degree, and their combination (R1, R2, R7).
- Explicit device selection by index and a user-set VRAM utilization fraction, replacing the display-bypass heuristic and the fixed hidden headroom (R12, R13).
- A **partition-aware weight format** and the converter that emits it, so one artifact serves any supported degree (R4, R5).
- Per-rank expert supply from the existing tiers, with the expert kept as one logical unit across the distribution (R3, R6).
- The **device abstraction** the layout requires — per-rank streams, allocations, and device selection, threaded through the supply and graph (R1, R10).
- Attention/KV placement that follows the attention class (R8), and the cross-rank communication tensor partitioning needs (R9).

---

## 4. Out of scope

- **Concurrency and serving.** Request queues, batching, and multiple live sessions are G5 concerns; multi-GPU is pursued on the single-stream path and does not depend on them (R11).
- **Multi-node / multi-host.** One host, PCIe; inter-node transport is not this revision's subject.
- **NVLink-class interconnect assumptions.** The design assumes PCIe links and P2P DMA only.
- **Multi-NVMe striping.** A separate storage capability; it composes with this work (it is the lever that turns parallel H2D into throughput) but is not required by it.
- **A change to the numerics or the KV precision.** Partitioning is a layout concern; it does not re-open attention, quantization, or the fp8/E4M3 KV gate.
- **A specific topology as the chosen one.** This document requires that both axes be *representable*; it does not select the deployment.

---

## 5. What this buys, and what it honestly does not

It buys the largest single lever available below the disk limit: **aggregate VRAM**. With the dense backbone sharded rather than duplicated, four 24 GB cards hold a materially larger expert hot pool, so the fraction of the `145.12 GiB` store kept off-disk rises well above today's, per-layer Cold traffic falls, and the prefill lookahead can widen. It also makes the per-rank H2D links and the per-rank GPU compute available to the one stream.

It does **not** delete the capacity wall (`4 × 24 GB = 96 GB < 145.12 GiB`), so some experts remain on NVMe and the tiered supply stays. It does **not** scale throughput by itself: the single NVMe remains the Cold-tier ceiling, and parallel H2D is bounded by host DDR — the path crosses DDR twice per byte because there is no GPUDirect-Storage equivalent on this platform. And with a shared-latent attention (DeepSeek-V4), KV is replicated rather than distributed, so KV is not part of the capacity win.

---

## 6. Deferred, and named so it is not an implicit "later"

| Deferred | Why | Revisits when |
| :--- | :--- | :--- |
| Multi-NVMe striping (one drive per rank) | a storage capability, not a layout one; the layout must merely not preclude it | aggregate NVMe bandwidth is the measured bottleneck |
| Expert-parallel placement (whole experts per rank) | a valid alternative to tensor-sharding experts with a different NVMe and warm-tier profile; the layout must admit it, not implement it | the tensor-sharded supply's Cold traffic is measured and found wanting |
| Hybrid pipeline × tensor / expert topologies beyond the plain combination | the axes must compose first | a plain topology is working and a rig needs the hybrid |
| Overlapping collectives with expert supply | an optimisation on top of a correct layout | correctness and the naive layout are green |
| Multi-node execution | a networking concern | a single host is saturated |

---

## 7. Feasibility

The routed-expert partition structure was the gating decision, and it is settled: **an expert split into independently-swizzled contiguous shards is feasible and numerically exact.** The format is the enabler — each shard is a valid swizzle domain at a smaller shape, and a rank's slice of every tensor is a contiguous, 4 KiB-aligned byte range — so R3, R4 and R5 hold by construction rather than by runtime work.

Five measurements on `gfx1100` back the expert-format half, all graded against an independent source-layout reference that shares no code with the swizzle, the feed, or the kernels:

| Property | Result |
| :--- | :--- |
| The `W2` down-projection consumed as separately-swizzled K-shards, summed | equals the whole-expert result to `4.2e-7` relative — fp32 rounding, ~2400× inside the `1e-3` ε |
| The grouped-WMMA feed given a shard instead of the whole | **bit-identical** across all 8 shards × 4 group pairs × 2 row tiles |
| The full expert pair (W13 N-shard → activation → W2 K-shard) under an 8-way partition | equals the whole-expert result to `9.1e-6` relative |
| The full expert pair under a real routing permutation — two experts, lumpy token counts, non-identity token order — for **every** degree `{1, 2, 4, 8}` | equals the whole-expert result to `6.7e-5`–`1.3e-4` relative (and the reference to `4e-3`) |
| The `W13` intermediate under sharding | **bit-identical** to the whole intermediate |

Three measurements back the attention/KV half (R8), through the real `v4_sliding_window_attn_wave32_kernel` and `v4_grouped_wo_a_wave32_kernel`:

| Property | Result |
| :--- | :--- |
| A rank's attention head block versus the whole's slice, for `{2, 4, 8}` | **bit-identical** — attention is head-local and every rank reads the same replicated latent |
| A rank's grouped `W_o_a` `z` columns versus the whole's slice, for `{2, 4, 8}` | **bit-identical** — the group (8 heads) is the coupling unit, so rank blocks are group-aligned |
| The group-mixing `W_o_b` per-rank partials summed versus the whole | exactly equal in double — the row-parallel reduction point (R9's shape) |

`W2` is the only routed tensor whose shard changes the swizzle's *iteration stride* rather than only its row-block count, so it was the case that could silently amplify or corrupt; it does neither. A rank's block may also be an arbitrary contiguous group of shards — at `TP = 2` one rank owns intermediate `[0, 1024)` as a single swizzled matrix — so the supported degrees are checked directly, not extrapolated from the finest. On the attention side, the two pieces of per-head state beyond the query — the head's output slice and its **attention sink** — shard with the head, while the shared 512-dim latent replicates; the gate's own first attempt failed until the sink was offset with the head block, which is exactly the class of silent error this verification exists to catch.

**All feasibility questions are now answered.** The remaining work is engineering, not correctness: threading a per-rank device context through the streams, allocations, and supply path so that the existing single-device logic follows the configured topology, and producing the partition-aware artifact the format defines.
