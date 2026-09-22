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

declare -A SERIAL_SWEPT
for n in "${SIZES[@]}"; do
  for arm in serial swept; do
    echo "[prefill-ab] $arm N=$n ..." >&2
    line=$("$BIN" "$arm" "$n" "$MODEL_DIR" "$WARM_GIB" | grep '^RESULT')
    echo "$line"
    if [ "$arm" = serial ]; then SERIAL_SWEPT[$n,"serial"]="$line"; else SERIAL_SWEPT[$n,"swept"]="$line"; fi
  done
done

field() { sed -n "s/.*\b$2=\([0-9.]*\).*/\1/p" <<<"$1"; }

echo
echo "=============================================================================================="
echo "  Step 6 outcome 5 — prefill A/B (warm ${WARM_GIB} GiB)"
echo "=============================================================================================="
printf '%-7s | %-30s | %-30s | %s\n' "N" "serial (per-token)" "swept (layer-major)" "ratio"
printf -- '--------+--------------------------------+--------------------------------+--------\n'
for n in "${SIZES[@]}"; do
  s="${SERIAL_SWEPT[$n,"serial"]}"; w="${SERIAL_SWEPT[$n,"swept"]}"
  st=$(field "$s" seconds); stps=$(field "$s" tok_per_s); sn=$(field "$s" nvme_gib)
  wt=$(field "$w" seconds); wtps=$(field "$w" tok_per_s); wn=$(field "$w" nvme_gib)
  wl=$(field "$w" load_s); we=$(field "$w" swept)
  ratio=$(python3 -c "print(f'{$wtps/max($stps,1e-9):.2f}x')")
  printf '%-7s | %6.1fs %6.2f tok/s %6.1fG | %6.1fs %6.2f tok/s %6.1fG | %s\n' \
    "$n" "$st" "$stps" "$sn" "$wt" "$wtps" "$wn" "$ratio"
  printf '%-7s | %-30s | load %ss of %ss, swept=%s, %s G/1k tok\n' \
    "" "" "$wl" "$wt" "$we" "$(python3 -c "print(f'{$wn*1024/max($n,1):.1f}')")"
done
echo "=============================================================================================="
echo "  reference: colibri 32 tok/s (3324 tok / 103.9 s) on a 2x NVMe mirror; ~0.35 s/layer sweep"
echo "=============================================================================================="
