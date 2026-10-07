#!/usr/bin/env bash
#
# The topology-equivalence gate.
#
# Runs the same two prompts through `aeon_chat` (a short one and the long corpus
# prompt, which takes the swept prefill path) for a given flag set and compares the
# result against a stored baseline. It is the instrument every topology change is
# measured with: multi-GPU work must not move a single bit at a fixed residency
# (`--exact`), or must stay within a reduction tolerance where a collective reorders
# the sum (`--tolerance`).
#
# Two prompts, because they take different paths: the short one is the per-token
# prefill and decode, and the corpus one is long enough to engage the swept prefill
# and the supply switch.
#
# Modes:
#   --capture            write the baseline (both cases) into the baseline directory
#   --exact              byte-compare the fp16 logits and the generated token ids
#   --tolerance <t>      max relative logit error <= t, via an inline Python compare
#                        (offline only; the token ids must still match)
# It always prints the first divergent token when the ids differ.
#
# Usage:
#   scripts/topology_equivalence.sh --capture
#   scripts/topology_equivalence.sh --exact
#   scripts/topology_equivalence.sh --tolerance 1e-3 --tensor-parallel 2
#
# Flags after `--` are passed to `aeon_chat` verbatim (e.g. `-- --tensor-parallel 2`).
#
# The baseline is captured on `main` and is **not** committed: it is a large,
# machine-specific artifact (`build/topology-baseline/`). Record the commit it was
# taken at here when it is refreshed:
#   baseline commit: 8c5067c

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="exact"
TOLERANCE="1e-3"
CAPTURE=0
BASELINE="${AEON_TOPOLOGY_BASELINE:-build/topology-baseline}"
MODEL_DIR="${AEON_MODEL_DIR:-models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon}"
BIN="${AEON_CHAT_BIN:-build/bin/aeon_chat}"
DEVICE_IDS="${AEON_DEVICE_IDS:-0}"
MAX_HOT_SLOTS=""
CONTEXT="${AEON_CONTEXT:-32768}"
TOKENS="${AEON_MAX_NEW_TOKENS:-64}"
PROMPT_SHORT="${AEON_PROMPT:-What is the capital of France?}"
CORPUS="${AEON_CORPUS:-profiling-prompts/prefill-corpus.txt}"
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --exact) MODE="exact"; shift ;;
    --tolerance) MODE="tolerance"; TOLERANCE="$2"; shift 2 ;;
    --capture) CAPTURE=1; shift ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --device-ids) DEVICE_IDS="$2"; shift 2 ;;
    --max-hot-slots) MAX_HOT_SLOTS="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "[topology-equiv] unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ ! -x "$BIN" ]; then
  echo "[topology-equiv] $BIN not found — build the aeon_chat target" >&2
  exit 1
fi

common=(
  --model-dir "$MODEL_DIR"
  --greedy
  --context-size "$CONTEXT"
  --max-new-tokens "$TOKENS"
  --device-ids "$DEVICE_IDS"
)
if [ -n "$MAX_HOT_SLOTS" ]; then common+=(--max-hot-slots "$MAX_HOT_SLOTS"); fi
if [ ${#EXTRA[@]} -gt 0 ]; then common+=("${EXTRA[@]}"); fi

# Run one case, dumping the per-position logits. The greedy token of each position
# is recovered from the logits (argmax over the row) during the compare, so no
# separate token-id channel is needed and the check is against the numbers the run
# actually produced.
run_case() {
  local name="$1" prompt="$2" dir="$3"
  mkdir -p "$dir"
  "$BIN" "${common[@]}" --prompt "$prompt" --dump-logits "$dir/$name.logits.bin" \
    > "$dir/$name.log" 2>&1
}

corpus_prompt="$(cat "$CORPUS")"

# The vocab width, needed to read a position's row out of the dumped logits.
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[topology-equiv] missing $MODEL_DIR/config.json" >&2; exit 1
fi
VOCAB="$(python3 -c "import json;print(json.load(open('$MODEL_DIR/config.json'))['vocab_size'])")"

if [ "$CAPTURE" -eq 1 ]; then
  echo "[topology-equiv] capturing baseline -> $BASELINE (device-ids=$DEVICE_IDS)"
  rm -rf "$BASELINE"
  run_case short "$PROMPT_SHORT" "$BASELINE"
  run_case corpus "$corpus_prompt" "$BASELINE"
  echo "[topology-equiv] baseline captured (commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown))"
  exit 0
fi

if [ ! -d "$BASELINE" ]; then
  echo "[topology-equiv] no baseline at $BASELINE — run with --capture first" >&2
  exit 1
fi

TMP="$(mktemp -d /tmp/aeon-topology.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

echo "[topology-equiv] comparing (mode=$MODE device-ids=$DEVICE_IDS) against $BASELINE"

fail=0

# Compare one case. A single Python pass reads both logits files, finds the max
# relative error, and the first position whose greedy token (argmax over the row)
# differs. Prints `<worst> <first_div> <ta> <tb>`; `first_div` is -1 when the greedy
# tokens match.
compare_case() {
  local name="$1"
  local prompt
  if [ "$name" = corpus ]; then prompt="$corpus_prompt"; else prompt="$PROMPT_SHORT"; fi
  run_case "$name" "$prompt" "$TMP"

  local summary
  summary="$(python3 - "$VOCAB" "$BASELINE/$name.logits.bin" "$TMP/$name.logits.bin" <<'PY'
import struct, sys
vocab = int(sys.argv[1])
def load(p):
    data = open(p, "rb").read()
    return data, [v[0] for v in struct.iter_unpack("<e", data)]
da, a = load(sys.argv[2])
db, b = load(sys.argv[3])
if len(a) != len(b):
    print("SIZE", len(a), len(b)); sys.exit(0)
rows = len(a) // vocab
worst = 0.0
first_div = -1
first_pair = (0, 0)
for r in range(rows):
    base = r * vocab
    ra = a[base:base + vocab]
    rb = b[base:base + vocab]
    for i in range(vocab):
        x = ra[i]; y = rb[i]
        d = abs(y - x) / max(1.0, abs(x))
        if d > worst: worst = d
    ta = max(range(vocab), key=lambda i: ra[i])
    tb = max(range(vocab), key=lambda i: rb[i])
    if ta != tb and first_div < 0:
        first_div = r; first_pair = (ta, tb)
print(worst, first_div, first_pair[0], first_pair[1])
PY
)"

  local worst div ta tb
  worst="$(echo "$summary" | awk '{print $1}')"
  div="$(echo "$summary" | awk '{print $2}')"
  ta="$(echo "$summary" | awk '{print $3}')"
  tb="$(echo "$summary" | awk '{print $4}')"

  if [ "$worst" = "SIZE" ]; then
    echo "  FAIL  $name: logits element counts differ ($div vs $ta)"
    fail=1
    return
  fi
  if [ "$div" != "-1" ]; then
    echo "  FAIL  $name: greedy tokens diverge at position $div (baseline $ta, current $tb)"
    fail=1
  else
    echo "  PASS  $name: greedy tokens identical"
  fi

  if [ "$MODE" = "exact" ]; then
    if cmp -s "$BASELINE/$name.logits.bin" "$TMP/$name.logits.bin"; then
      echo "  PASS  $name: fp16 logits byte-identical"
    else
      echo "  FAIL  $name: fp16 logits differ (byte compare)"
      fail=1
    fi
  else
    if python3 -c "import sys; sys.exit(0 if float('$worst') <= $TOLERANCE else 1)"; then
      echo "  PASS  $name: max relative logit error $worst <= $TOLERANCE"
    else
      echo "  FAIL  $name: max relative logit error $worst > $TOLERANCE"
      fail=1
    fi
  fi
}

compare_case short
compare_case corpus

if [ "$fail" -ne 0 ]; then
  echo "[topology-equiv] GATE FAILED"
  exit 1
fi
echo "[topology-equiv] GATE PASSED"
