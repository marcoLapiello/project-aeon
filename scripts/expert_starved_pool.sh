#!/usr/bin/env bash
#
# Step 4 — the starved-pool gate.
#
# Re-runs Run A from the tier-invariance gate at `--max-hot-slots 12` (two
# layers' worth). Every layer now forces a drain, which is the only configuration
# that exercises the emergency valve, the eviction-under-lease-pressure path, and
# the asynchronous-demotion hazard.
#
# Assertions:
#   (i)   forced_drains > 0                       (the starved path actually ran)
#   (ii)  staging_in_use == 0                     (arena drained at the end)
#   (iii) logits byte-identical to the uncapped Run A   (still numerically exact)
#   (iv)  registry.invariants_hold()
#
# Usage: scripts/expert_starved_pool.sh [model_dir] [out_dir]

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_DIR="${1:-models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon}"
OUT_DIR="${2:-$(mktemp -d /tmp/aeon-starved-pool.XXXXXX)}"
BIN="${AEON_CHAT_BIN:-build/bin/aeon_chat}"
PROMPT="${AEON_PROMPT:-What is the capital of France?}"
CONTEXT="${AEON_CONTEXT:-32768}"
TOKENS="${AEON_MAX_NEW_TOKENS:-8}"
CAP="${AEON_HOT_CAP:-12}"

mkdir -p "$OUT_DIR"
echo "[starved-pool] model=$MODEL_DIR out=$OUT_DIR context=$CONTEXT tokens=$TOKENS cap=$CAP"

common=(
  --model-dir "$MODEL_DIR"
  --prompt "$PROMPT"
  --greedy
  --context-size "$CONTEXT"
  --max-new-tokens "$TOKENS"
  --diagnostic
)

# The uncapped reference: same as tier-invariance Run A.
echo "[starved-pool] reference (uncapped) ..."
"$BIN" "${common[@]}" --warm-gib 0 --dump-logits "$OUT_DIR/ref.logits.bin" \
  > "$OUT_DIR/ref.log" 2>&1

# The starved run: Hot capped at two layers, Warm off so every miss drains.
echo "[starved-pool] starved (cap $CAP) ..."
"$BIN" "${common[@]}" --warm-gib 0 --max-hot-slots "$CAP" \
  --supply-telemetry "$OUT_DIR/starved.telemetry.jsonl" \
  --dump-logits "$OUT_DIR/starved.logits.bin" \
  > "$OUT_DIR/starved.log" 2>&1

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

drains=$(sed -n 's/.*forced_drains=\([0-9]*\).*/\1/p' "$OUT_DIR/starved.log" | head -1)
staging=$(sed -n 's/.*staging_in_use=\([0-9]*\).*/\1/p' "$OUT_DIR/starved.log" | head -1)

echo "[starved-pool] assertions:"
check "(i) forced_drains > 0" test "${drains:-0}" -gt 0
check "(ii) staging_in_use == 0" test "${staging:-1}" -eq 0
check "(iii) logits byte-identical to uncapped reference" cmp -s "$OUT_DIR/ref.logits.bin" "$OUT_DIR/starved.logits.bin"
check "(iv) invariants_hold" grep -q 'registry.invariants_hold=true' "$OUT_DIR/starved.log"
check "(iv) zero outstanding leases" grep -q 'outstanding_leases=0' "$OUT_DIR/starved.log"
echo "  forced_drains=$drains staging_in_use=$staging"

echo "[starved-pool] artifacts in $OUT_DIR"
if [ "$fail" -ne 0 ]; then
  echo "[starved-pool] GATE FAILED"
  exit 1
fi
echo "[starved-pool] GATE PASSED"
