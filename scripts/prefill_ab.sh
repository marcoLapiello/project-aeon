#!/usr/bin/env bash
#
# Step 6 outcome 5 — the prefill A/B.
#
# Measures the certified serial prefill (one `forward_token` per prompt token)
# against the swept layer-major window (one `forward_window` over the whole prompt),
# at several prompt lengths, with a pristine host per arm per length.
#
# The point of the matrix rather than one number: the sweep's byte cost is very
# nearly constant in the prompt length (one model read, ~148 GiB) while serial's
# grows linearly, so the two arms cross over and a single point says nothing about
# the strategy.
#
# Usage: scripts/prefill_ab.sh [n_tokens ...]
#   AEON_WARM_GIB   Warm host allocation in GiB (default 0)
#   AEON_MODEL_DIR  model directory
#   AEON_AB_BIN     benchmark binary
#
# Each arm is a separate process, so neither arm sees the other's Warm residency.

set -euo pipefail

# The tabulation below is float arithmetic in `python3`, so force a C numeric
# locale: under a comma-decimal locale `printf %6.2f` rejects `51.246`.
export LC_ALL=C

export LC_NUMERIC=C

cd "$(dirname "$0")/.."

MODEL_DIR="${AEON_MODEL_DIR:-models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon}"
WARM_GIB="${AEON_WARM_GIB:-0}"
BIN="${AEON_AB_BIN:-build/bin/bench_prefill_ab}"
SIZES=("$@")
if [ "${#SIZES[@]}" -eq 0 ]; then SIZES=(128 512); fi

if [ ! -x "$BIN" ]; then
  echo "[prefill-ab] $BIN not found — configure with -DAEON_BUILD_BENCHMARKS=ON" >&2
  exit 1
fi

echo "[prefill-ab] warm=${WARM_GIB} GiB, sizes: ${SIZES[*]}"

declare -A ARM_LINE
for n in "${SIZES[@]}"; do
  for arm in serial swept routed; do
    echo "[prefill-ab] $arm N=$n ..." >&2
    line=$("$BIN" "$arm" "$n" "$MODEL_DIR" "$WARM_GIB" | grep '^RESULT')
    echo "$line"
    ARM_LINE[$n,$arm]="$line"
  done
done

field() { sed -n "s/.*\b$2=\([0-9.]*\).*/\1/p" <<<"$1"; }

echo
echo "=============================================================================================="
echo "  Prefill A/B — serial vs swept vs routed (warm ${WARM_GIB} GiB)"
echo "=============================================================================================="
printf '%-6s | %-24s | %-24s | %-24s\n' "N" "serial (per-token)" "swept (whole-layer)" "routed (bank)"
printf -- '-------+--------------------------+--------------------------+--------------------------\n'
for n in "${SIZES[@]}"; do
  s="${ARM_LINE[$n,serial]}"; w="${ARM_LINE[$n,swept]}"; r="${ARM_LINE[$n,routed]}"
  stps=$(field "$s" tok_per_s); sn=$(field "$s" nvme_gib)
  wtps=$(field "$w" tok_per_s); wn=$(field "$w" nvme_gib)
  rtps=$(field "$r" tok_per_s); rn=$(field "$r" nvme_gib)
  printf '%-6s | %6.2f tok/s %5.1f GiB  | %6.2f tok/s %5.1f GiB  | %6.2f tok/s %5.1f GiB\n' \
    "$n" "$stps" "$sn" "$wtps" "$wn" "$rtps" "$rn"
done
echo "=============================================================================================="
