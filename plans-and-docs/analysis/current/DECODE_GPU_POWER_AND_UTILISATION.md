# Decode GPU power and utilisation — observations

Date `2026-10-02`. Observation: in a `--until-eos` decode run (`5706` tokens, `3.73 tok/s`) the GPU sits at `~3.1 GHz`, `95–100%` busy and `185–205 W`, while decode is supply-bound and the kernels are idle most of the time. Prefill in the same run drew only `~85 W` at `~33%`. Goal: record what was measured and what is still open.

## Measured

### Reproduced

A clean `1000`-token decode (no `--supply-telemetry`, no `--phase-profile`) on the compute GPU (`0000:63:00.0`, sample every `5 s`): `95–100%` use, `175–203 W`, `3.05–3.11 GHz`, flat for the whole decode. So neither diagnostic flag is needed to see it.

### Kernels are busy only ~20% of the time

`rocprofv3 --kernel-trace`, first `60` decode tokens, `211,969` kernels: total kernel time `3.5 s` of a `39.8 s` wall. Per-second busy fraction after the prefill: `17–25%` early, `~20%` steady.

Top kernels by decode time (of `3.5 s`): `gemv_fp16_kernel` (`1,150 ms`, `19,942` calls), `v4_indexer_scores_kernel` (`358 ms`), `gemv_fp16_vec8_kernel` (`329 ms`), `moe_grouped_gate_up_kernel` (`258 ms`), `v4_grouped_wo_a_wave32_kernel` (`236 ms`), `aeon_moe_fused_gate_up_kernel` (`198 ms`), `rmsnorm_wave32_kernel` (`179 ms`), `__amd_rocclr_copyBuffer` (`150 ms`, `79,060` calls). The `gemv` count is itself a finding: decode runs a per-expert GEMV path (`~20k` GEMV launches in `60` tokens) rather than the grouped kernel.

### Supply traffic

From the run's telemetry (`decode`, `5705` tokens):

| source tier | requests | bytes | h2d enqueue→ready | gpu readiness wait | nvme wait |
| :--- | ---: | ---: | ---: | ---: | ---: |
| hot | 776,423 | `0` | `0` | `0` | `0` |
| warm | 501,218 | `6,607.9 GiB` | `3,434 s` | `2,741 s` | `0` |
| cold | 194,249 | `2,560.9 GiB` | `797 s` | `621 s` | `400 s` |

`9,169 GiB` over `5705` tokens is `1.6 GiB/token`. The warm H2D rate over the whole decode is `~6.4 GB/s` against a measured link of `~28 GB/s`.

### The link alone raises the clocks

`barrier_test` (copies only, no compute kernels), `14 s` per mode, sampled every `1.5 s`:

| mode | link | use | power | sclk |
| :--- | :--- | ---: | ---: | ---: |
| `256 MiB` H2D back to back | `28.1 GB/s` | `100%` | `134 W` | `3,121 MHz` |
| same, waited on by `hipStreamWaitEvent` + tiny kernel | `28.0 GB/s` | `100%` | `135 W` | `3,120 MHz` |
| same, waited on by `hipEventSynchronize` + tiny kernel | `28.0 GB/s` | `100%` | `135 W` | `3,120 MHz` |
| `HSA_ENABLE_SDMA=0` (blit kernels instead of SDMA) | `28.0 GB/s` | `100%` | `143 W` | `3,116 MHz` |
| light: `14 MiB` per copy `+ 20 ms` sleep | `0.7 GB/s` | `0–5%` | `36–37 W` | `~90 MHz` |

`duty_test`, copy only, a `14 MiB` copy every `N` µs:

| period | use | power | sclk |
| --- | ---: | ---: | ---: |
| `5000 µs` | `15%` | `38 W` | `177 MHz` |
| `2000 µs` | `35%` | `42 W` | `1,292 MHz` |
| `1000 µs` | `68%` | `98 W` | `2,896 MHz` |
| continuous (`barrier_test`) | `100%` | `134 W` | `3,121 MHz` |

So a stream of host-to-device copies, with no kernels at all, reproduces `100% use / ~3.1 GHz / 134 W`, and a ~50% copy duty cycle already reaches ~`2.9 GHz`. `rocm-smi`'s use, power and clock read the **whole device**, copies included; at `3.73 tok/s` a `76 ms` token carries ~`16 ms` of transfer, and the D2H/H2D mix keeps the device clocked up between kernels.

No spin-wait or polling loop was found in the kernels (searched for atomics/volatile/`s_sleep` patterns).

## Read

- The high use/power/clock is explained by **data movement, not by kernel compute**: the copies alone produce the same state.
- The relevant question is **why `1.6 GiB/token` is needed at `6.4 GB/s`** — ~`360` expert requests per token, of which `~45%` are served from warm (host DDR) and `~17%` from NVMe. That is a supply/placement cost (registry churn, demotions, re-reads), not a kernel cost.
- Kernel compute is `~20%` busy, in line with the phase profile's `~21%` from the kernel plan.
- Prefill reads lower because NVMe wait (`load_wait`) leaves long idle gaps, not because its compute is cheaper per byte.

## Open — next time

1. **Cap the clock and A/B the throughput.** Decode is supply-bound, so a lower `sclk` cap should cost no tok/s. `rocm-smi --setperflevel low` / `--setsclk` against tokens/sec and power. This is the cheapest actionable test.
2. **Attribute the `1.6 GiB/token`.** Read the transfer events (they are in the telemetry JSONL) to separate first-read from re-read, and check the `45%` warm share against expert reuse — the same experts are being re-fetched.
3. **Find why decode uses the per-expert GEMV** (`~20k` launches / `60` tokens) instead of the grouped path, and whether the launch rate itself keeps the device busy.
4. **Make the diagnostics bounded** (they are separate from this issue but were the RAM growth): `--supply-telemetry` buffers every transfer event and writes at shutdown (`supply_telemetry.hpp::flush`), and `--phase-profile` holds two GPU events per region per layer per token until `resolve()`. Flush telemetry in batches; resolve profiler events periodically.
5. **Instrumentation for the next run:** sample `rocm-smi` use/power/sclk next to the kernel trace, so a busy-copy phase is distinguishable from a busy-compute phase in one log.

## Artifacts

In `/tmp/aeon-run/` (not in the repo): `clean.csv` (RAM/GPU series), `prof/` (`rocprofv3` kernel and copy traces), `barrier_test.cpp`, `duty_test.cpp`, `run_monitored.sh`, `portfolio.jsonl` (`408 MB` telemetry).
