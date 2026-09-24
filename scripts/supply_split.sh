#!/usr/bin/env bash
#
# Supply-chain hot-path analysis, Step 1 — the exposed-load split.
#
# Runs `bench_supply_split` twice at a modest matrix of prompt lengths: once with no
# Warm (maximal NVMe signal, the disk/H2D legs at full size) and once at the
# production Warm shape (35 GiB), because any conclusion we act on must hold there.
#
# The model is loaded once per process, so the run is short compared to `prefill_ab`
# — that bench reloads the artifact per arm. Lengths default to 64 / 256 / 512: the
# first is below the sweep gate and exercises the routed bank, the other two the
# sweep. The bench prints which strategy engaged, so read the table with that in mind.
#
# Usage: scripts/supply_split.sh [len ...]
#   AEON_MODEL_DIR     model directory
#   AEON_SPLIT_BIN     benchmark binary
#   AEON_SPLIT_DECODE  decode tokens per length (default 64)
#   AEON_WARM_GIB      if set, only this Warm size is run (skip the 0/35 pair)
#
# The two Warm sizes are two separate processes on purpose: the Warm residency at the
# start of a run is part of the configuration, so it must not carry between them.

set -euo pipefail

export LC_ALL=C
export LC_NUMERIC=C

cd "$(dirname "$0")/.."

MODEL_DIR="${AEON_MODEL_DIR:-models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon}"
BIN="${AEON_SPLIT_BIN:-build/bin/bench_supply_split}"
SIZES=("$@")
if [ "${#SIZES[@]}" -eq 0 ]; then SIZES=(64 256 512); fi
if [ -n "${AEON_WARM_GIB:-}" ]; then
  WARMS=("$AEON_WARM_GIB")
else
  WARMS=(0 35)
fi

if [ ! -x "$BIN" ]; then
  echo "[supply-split] $BIN not found — configure with -DAEON_BUILD_BENCHMARKS=ON" >&2
  exit 1
fi

echo "[supply-split] sizes: ${SIZES[*]}  warm: ${WARMS[*]} GiB  decode: ${AEON_SPLIT_DECODE:-64} tok"

for warm in "${WARMS[@]}"; do
  echo
  echo "##############################################################################"
  echo "#  Warm = ${warm} GiB"
  echo "##############################################################################"
  "$BIN" "$MODEL_DIR" "$warm" "" "${SIZES[@]}"
done
