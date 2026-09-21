#!/usr/bin/env bash
#
# Step 3 — the tier-invariance gate.
#
# Two runs, same context, same prompt, both greedy. They differ in exactly one
# thing: which tier answers the non-Hot misses.
#
#   A: --warm-gib 0   -> Hot and Cold only
#   B: --warm-gib 40  -> Hot and Warm
#
# Same context means the same Hot pool, so the variable under test is the
# answering tier. The claim is not "the logits match an all-resident run" (that
# needs 156 GB and is not realizable); it is that the tier that answered did not
# change the number.
#
# Assertions:
#   (i)   identical generated token ids
#   (ii)  byte-identical fp16 logits at every position   (cmp)
#   (iii) registry.invariants_hold()
#   (iv)  outstanding_leases == 0
#   (v)   logical_bytes_from_warm == 0 in A and > 0 in B
#
# Usage: scripts/expert_tier_invariance.sh [model_dir] [out_dir]

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_DIR="${1:-models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon}"
OUT_DIR="${2:-$(mktemp -d /tmp/aeon-tier-invariance.XXXXXX)}"
BIN="${AEON_CHAT_BIN:-build/bin/aeon_chat}"
PROMPT="${AEON_PROMPT:-What is the capital of France?}"
CONTEXT="${AEON_CONTEXT:-32768}"
TOKENS="${AEON_MAX_NEW_TOKENS:-8}"

mkdir -p "$OUT_DIR"
echo "[tier-invariance] model=$MODEL_DIR out=$OUT_DIR context=$CONTEXT tokens=$TOKENS"

common=(
  --model-dir "$MODEL_DIR"
  --prompt "$PROMPT"
  --greedy
  --context-size "$CONTEXT"
  --max-new-tokens "$TOKENS"
  --diagnostic
)

run() {
  local name="$1"; shift
  echo "[tier-invariance] run $name ..."
  "$BIN" "${common[@]}" "$@" > "$OUT_DIR/$name.log" 2>&1
}

run A --warm-gib 0 \
  --supply-telemetry "$OUT_DIR/A.telemetry.jsonl" \
  --dump-logits "$OUT_DIR/A.logits.bin"

run B --warm-gib 40 \
  --supply-telemetry "$OUT_DIR/B.telemetry.jsonl" \
  --dump-logits "$OUT_DIR/B.logits.bin"

fail=0
check() {
  local label="$1"; shift
  if "$@"; then
    echo "  PASS  $label"
  else
    echo "  FAIL  $label"
    fail=1
  fi
}

echo "[tier-invariance] assertions:"

# (i) identical generated token ids
tokens_a=$(grep -m1 '^Generated IDs:' "$OUT_DIR/A.log" || true)
tokens_b=$(grep -m1 '^Generated IDs:' "$OUT_DIR/B.log" || true)
check "(i) identical generated token ids" test "$tokens_a" = "$tokens_b"

# (ii) byte-identical logits at every position
check "(ii) byte-identical fp16 logits" cmp -s "$OUT_DIR/A.logits.bin" "$OUT_DIR/B.logits.bin"

# (iii)+(iv) registry invariants and lease release, from the stable report line
check "(iii) invariants_hold in A" grep -q 'registry.invariants_hold=true' "$OUT_DIR/A.log"
check "(iii) invariants_hold in B" grep -q 'registry.invariants_hold=true' "$OUT_DIR/B.log"
check "(iv) zero outstanding leases in A" grep -q 'outstanding_leases=0' "$OUT_DIR/A.log"
check "(iv) zero outstanding leases in B" grep -q 'outstanding_leases=0' "$OUT_DIR/B.log"

# (v) cold-only in A, Warm actually answering in B
warm_a=$(python3 - "$OUT_DIR/A.telemetry.jsonl" <<'PY'
import json,sys
total=0
for line in open(sys.argv[1]):
    r=json.loads(line)
    if r.get("record_type")=="phase_summary":
        total+=r.get("logical_bytes_from_warm",0)
print(total)
PY
)
warm_b=$(python3 - "$OUT_DIR/B.telemetry.jsonl" <<'PY'
import json,sys
total=0
for line in open(sys.argv[1]):
    r=json.loads(line)
    if r.get("record_type")=="phase_summary":
        total+=r.get("logical_bytes_from_warm",0)
print(total)
PY
)
check "(v) logical_bytes_from_warm == 0 in A" test "$warm_a" -eq 0
check "(v) logical_bytes_from_warm > 0 in B" test "$warm_b" -gt 0
echo "  (v) logical_bytes_from_warm: A=$warm_a B=$warm_b"

echo "[tier-invariance] artifacts in $OUT_DIR"
if [ "$fail" -ne 0 ]; then
  echo "[tier-invariance] GATE FAILED"
  exit 1
fi
echo "[tier-invariance] GATE PASSED"
