# Startup GPU Idle and Warm-Preload Memory Spike — two observations to investigate

**Status: OPEN, not investigated.** Written `2026-10-01` on branch `main`. This document only
**records** two observations so a future session can pick them up. No root cause is claimed and no
code was changed for either.

Both were noticed while working on the [kernel improvement plan](../../execution/active/KERNELS_IMPROVEMENT.md)
(prefill throughput). Both are on the **startup / supply** path, not the prefill compute path the
plan owns, which is why they are parked here rather than acted on.

---

## 1. GPU sits at very low utilization / ~50 W for hundreds of seconds

**Reported, from realtime observation.** During a run, GPU utilization and power are very low
(≈`50 W`) for **many hundreds of seconds**, and the reporter states this is **not** the Warm preload —
the preload completes as quickly as it always has, and the low-utilization stretch comes *after* it.

**Why it is recorded rather than diagnosed.** The one instrument available to a sampled check is
`/sys/class/drm/card0/device/gpu_busy_percent`. A 0.5 s sampling showed a **≈`11 s` 0 % window at
process start**, then the sweep load, then prefill at `85–98 %`. That startup window matches the
Warm preload and is **not** the stretch the reporter sees. The two are different windows:

| Window | Where | Seen by |
| :--- | :--- | :--- |
| `≈11 s` at 0 % | process start, before layer 0 | sampled `gpu_busy_percent`; present on every binary including the pre-session baseline `ac6493b` |
| hundreds of seconds at ~50 W | reported to come *after* the preload | realtime monitor only; **not** caught by the sampled probe |

So the instrument a later session should reach for is **not** a coarse `gpu_busy_percent` poll but a
per-process, continuous one (`rocm-smi`/`amd-smi` per-PID busy, or a sampling that runs for the whole
prefill, not just the first tens of seconds). The `≈11 s` startup window and the long stretch must be
told apart before either is explained.

**Candidates to test (none measured):**
- A **supply stall** during prefill — expert H2D/NVMe waited on while the compute queue drains (the
  decode profile already shows the MoE region is mostly *wait*; the same could expose on a prefill
  stretch). Cross-check against the supply telemetry (`--supply-telemetry`): `io_ms`, `load_ms`,
  `h2d_enqueue_to_ready`.
- A **sweep/lease barrier** — `forward_window` does one `hipStreamSynchronize` per layer boundary
  (`release_expert_leases`); if any layer's set is slow to land, the GPU empties for that whole layer.
- A **host-side stall** not visible in the phase profile — a `hipStreamSynchronize` draining a long
  queue, or a host allocation/`mmap` stall between layers.

**Do not** run a second probe process concurrently while chasing this: see §3.

---

## 2. Host RSS overshoots during the first Warm preload

**Reported.** With `--warm-gib 35` and the usual `5–7 GiB` of system usage, the expected host ceiling
is `≈42 GiB`. During the **first** preload the process reaches **`49 GiB`**, then falls back to the
normal `40–42 GiB`. A transient `≈7 GiB` above the steady state is a signal that something is held
longer than it should be.

**Why it matters.** The Warm pool is **pinned** (`hipHostMalloc`) — neither reclaimable nor
swappable — so any transient that competes with it is a hard double-claim, and the host is already
near its cap ("`98 %` of the allowance" per the plan) with a `45 GiB` Warm tier known not to load.
This is the same resource as the [host memory pressure investigation](HOST_MEMORY_PRESSURE_INVESTIGATION.md),
seen from a different angle: that document asks why a *large* Warm does not finish loading; this one
asks why a *fitted* Warm transiently overshoots its own footprint.

**Candidates to test (none measured) — the release ordering is the first thing to read:**
- **Dense-page release timing.** `release_dense_pages_after_upload` (`V4ModelHost::initialize`)
  `madvise`s the dense container *before* the Warm preload, precisely so the preload meets the least
  pressure. If the release is partial at the moment of peak — or the uploads fault more pages in after
  it — `13.68 GiB` of clean file-backed pages coexist with the filling pinned pool. Verify the release
  completes (its `[Host] Released … GiB` line) **before** the first large preload read.
- **A transient copy in the preload path.** A staging buffer, a full-file read, or a non-`O_DIRECT`
  read that lands in page cache and is not `madvise`d would show exactly this shape (spike then fall).
- **Allocation-before-free in Warm growth.** If the pool is regrown rather than filled in place, the
  old region and the new one briefly coexist.
- **First-run-only.** The reporter saw it on the **first** preload; confirm whether a second run in the
  same process (or a warm page cache) still spikes, which would separate "transient allocation" from
  "cold file pages".

---

## 3. Do not measure this with a concurrent probe

An attempt to characterise both symptoms by sampling RSS + `gpu_busy_percent` at `0.2 s` in a shell
loop **pushed the host into memory pressure on its own**, before `aeon_chat` was even doing real work,
and had to be interrupted. The engine sits close enough to the host cap that a second consumer (or a
tight sampling loop that forks `/proc` reads, or a parallel build) is enough to tip it. Any future
investigation must:

- run **one** `aeon_chat` at a time, nothing else;
- prefer **in-process instrumentation** (an interval timer inside the engine, or the existing phase
  profiler / supply telemetry) over an **external** polling loop;
- sample long enough to cover the whole prefill, not only the first tens of seconds, since the
  unreproduced window is the long one.
