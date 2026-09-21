#!/usr/bin/env bash
#
# Step 5 — the demotion-queue A/B.
#
# `TieredExpertSupply::DEFAULT_DEMOTION_QUEUE_CAPACITY` is 2. One layer dispatches
# up to 6 experts and each may evict a victim, with the bookkeeping cleared once
# per layer, so two thirds of every eviction is dropped with `queue_pressure`. This
# A/B runs capacity 2 against 6 and asks whether the larger queue converts dropped
# demotions into Warm service.
#
# Assertions:
#   (i)   logical_bytes_from_warm rises (queue 6 > queue 2)
#   (ii)  bytes_from_nvme falls
#   (iii) demotion_drops falls
#   (iv)  logits byte-identical between the two arms
#
# The applied capacity is read back from the [Invariants] line, so a mis-wired
# knob fails loudly rather than silently producing two identical runs.
#
# Usage: scripts/expert_demotion_queue_ab.sh [model_dir] [out_dir]

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_DIR="${1:-models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon}"
OUT_DIR="${2:-$(mktemp -d /tmp/aeon-demotion-ab.XXXXXX)}"
BIN="${AEON_CHAT_BIN:-build/bin/aeon_chat}"
PROMPT="${AEON_PROMPT:-Write a detailed essay about the history of computing, from the abacus to modern GPUs.}"
CONTEXT="${AEON_CONTEXT:-32768}"
TOKENS="${AEON_MAX_NEW_TOKENS:-24}"
WARM_GIB="${AEON_WARM_GIB:-40}"

mkdir -p "$OUT_DIR"
echo "[demotion-ab] model=$MODEL_DIR out=$OUT_DIR context=$CONTEXT tokens=$TOKENS warm=${WARM_GIB}GiB"

common=(
  --model-dir "$MODEL_DIR"
  --prompt "$PROMPT"
  --greedy
  --context-size "$CONTEXT"
  --max-new-tokens "$TOKENS"
  --diagnostic
  --warm-gib "$WARM_GIB"
)

run() {
  local name="$1"; shift
  echo "[demotion-ab] run $name ..."
  "$BIN" "${common[@]}" "$@" \
    --supply-telemetry "$OUT_DIR/$name.telemetry.jsonl" \
    --dump-logits "$OUT_DIR/$name.logits.bin" \
    > "$OUT_DIR/$name.log" 2>&1
}

run q2 --demotion-queue 2
run q6 --demotion-queue 6

# Read a summed telemetry field across every phase_summary row.
sum_field() {
  python3 - "$1" "$2" <<'PY'
import json,sys
path,field=sys.argv[1],sys.argv[2]
total=0
for line in open(path):
    r=json.loads(line)
    if r.get("record_type")=="phase_summary":
        total+=r.get(field,0)
print(total)
PY
}

applied_q2=$(sed -n 's/.*demotion_queue=\([0-9]*\).*/\1/p' "$OUT_DIR/q2.log" | head -1)
applied_q6=$(sed -n 's/.*demotion_queue=\([0-9]*\).*/\1/p' "$OUT_DIR/q6.log" | head -1)
warm2=$(sum_field "$OUT_DIR/q2.telemetry.jsonl" logical_bytes_from_warm)
warm6=$(sum_field "$OUT_DIR/q6.telemetry.jsonl" logical_bytes_from_warm)
nvme2=$(sum_field "$OUT_DIR/q2.telemetry.jsonl" bytes_from_nvme)
nvme6=$(sum_field "$OUT_DIR/q6.telemetry.jsonl" bytes_from_nvme)
drops2=$(sum_field "$OUT_DIR/q2.telemetry.jsonl" demotion_drops)
drops6=$(sum_field "$OUT_DIR/q6.telemetry.jsonl" demotion_drops)

fail=0
check() {
  local label="$1"; shift
  if "$@"; then echo "  PASS  $label"; else echo "  FAIL  $label"; fail=1; fi
}

echo "[demotion-ab] applied capacities: q2=$applied_q2 q6=$applied_q6"
check "knob applied (q2=2, q6=6)" test "${applied_q2:-x}" = "2" -a "${applied_q6:-x}" = "6"
check "(i) logical_bytes_from_warm rises" test "$warm6" -gt "$warm2"
check "(ii) bytes_from_nvme falls" test "$nvme6" -lt "$nvme2"
check "(iii) demotion_drops falls" test "$drops6" -lt "$drops2"
check "(iv) logits byte-identical across arms" cmp -s "$OUT_DIR/q2.logits.bin" "$OUT_DIR/q6.logits.bin"

cat <<EOF
[demotion-ab] ladder (q2 -> q6):
  logical_bytes_from_warm : $warm2 -> $warm6
  bytes_from_nvme         : $nvme2 -> $nvme6
  demotion_drops          : $drops2 -> $drops6
EOF

echo "[demotion-ab] artifacts in $OUT_DIR"
if [ "$fail" -ne 0 ]; then echo "[demotion-ab] GATE FAILED"; exit 1; fi
echo "[demotion-ab] GATE PASSED"
