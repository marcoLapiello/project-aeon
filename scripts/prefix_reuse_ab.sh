#!/usr/bin/env bash
#
# The prefix-reuse A/B.
#
# Runs one 3-turn conversation through `aeon_chat` twice: once letting each turn
# reuse the resident prefix, once with `--no-prefix-reuse`. The two arms are the
# same script and the same prompts, so the only difference is whether turn N+1
# recomputes the whole conversation or only its addition.
#
# What it reports, per turn: the verdict, the reused and freshly prefilled token
# counts, and TTFT. The claim under test is that the continuation's TTFT is
# governed by the *added* length, while the from-scratch arm's grows with the
# accumulated length — so both columns are printed and neither is asserted here.
#
# Usage: scripts/prefix_reuse_ab.sh
#   AEON_MODEL_DIR   model directory
#   AEON_CONTEXT     context capacity in tokens (default 4096)
#   AEON_MAX_NEW     generated tokens per turn (default 32)
#
# Two model loads, one per arm, so neither arm inherits the other's residency.

set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

MODEL_DIR="${AEON_MODEL_DIR:-models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon}"
CONTEXT="${AEON_CONTEXT:-4096}"
MAX_NEW="${AEON_MAX_NEW:-32}"
BIN="build/bin/aeon_chat"

if [ ! -x "$BIN" ]; then
  echo "[prefix-reuse-ab] $BIN not found — build the aeon_chat target" >&2
  exit 1
fi

# A multi-turn script whose first turn is long enough that the accumulated
# conversation is a meaningful prefix by turn 3.
PROMPT="I am building a small offline inference engine for a mixture-of-experts language model on a consumer GPU. The model's experts do not all fit in VRAM, so I stream them from NVMe on demand. Please summarise, in a few sentences, the main trade-offs between keeping more experts resident in VRAM and keeping a larger context window resident instead."
FOLLOW1="Thanks. Now imagine the conversation continues for many turns like this one. What changes about the cost of each new turn if I keep the model state resident between turns instead of recomputing the whole conversation every time?"
FOLLOW2="Good. In one short paragraph, what is the single most important correctness check I should run before trusting that optimisation?"

run_arm() {
  local label="$1"; shift
  "$BIN" \
    --model-dir "$MODEL_DIR" \
    --context-size "$CONTEXT" \
    --max-new-tokens "$MAX_NEW" \
    --greedy \
    --prompt "$PROMPT" \
    --follow-up "$FOLLOW1" \
    --follow-up "$FOLLOW2" \
    "$@" 2>/dev/null | grep '^\[Reuse\]' | sed "s/^/$label /"
}

echo "[prefix-reuse-ab] context=${CONTEXT} max_new=${MAX_NEW}"

WITH_REUSE="$(run_arm reuse)"
WITHOUT_REUSE="$(run_arm noreuse --no-prefix-reuse)"

echo
echo "$WITH_REUSE"
echo "$WITHOUT_REUSE"

python3 - "$WITH_REUSE" "$WITHOUT_REUSE" <<'PY'
import re, sys

def parse(blob):
    rows = []
    for line in blob.splitlines():
        if not line.strip():
            continue
        arm = line.split()[0]
        fields = dict(re.findall(r'(\w+)=([0-9A-Za-z_.]+)', line))
        rows.append((arm, fields))
    return rows

rows = parse(sys.argv[1]) + parse(sys.argv[2])
if not rows:
    print("[prefix-reuse-ab] no [Reuse] lines captured", file=sys.stderr)
    raise SystemExit(1)

print()
print("=" * 88)
print("  Prefix reuse A/B — continuation vs from-scratch")
print("=" * 88)
print(f"{'arm':<9} {'turn':>4} {'verdict':<13} {'reused':>7} {'prefilled':>10} {'prompt':>7} {'ttft_ms':>10}")
print("-" * 88)
for arm, f in rows:
    print(f"{arm:<9} {f.get('turn','?'):>4} {f.get('verdict','?'):<13} "
          f"{f.get('reused','?'):>7} {f.get('prefilled','?'):>10} "
          f"{f.get('prompt','?'):>7} {f.get('ttft_ms','?'):>10}")
print("=" * 88)
PY
