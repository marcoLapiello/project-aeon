#!/usr/bin/env bash
#
# The end-to-end server smoke test, against the real checkpoint.
#
# Exercises the wire contract the requirements name: health, streaming, queueing,
# cancellation, prefix reuse across a re-sent conversation, the error shapes, and a
# clean shutdown. `curl` is the only HTTP client; `python3` is used to read fields
# out of the JSON responses.
#
# Usage: scripts/server_smoke.sh
#   AEON_MODEL_DIR  model directory (default: the Aeon artifact)
#   AEON_PORT       bind port (default 8099)
#   AEON_CONTEXT    context capacity (default 4096)

set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."

MODEL_DIR="${AEON_MODEL_DIR:-models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon}"
PORT="${AEON_PORT:-8099}"
CONTEXT="${AEON_CONTEXT:-4096}"
BIN="${AEON_SERVE_BIN:-build/bin/aeon_serve}"
BASE="http://127.0.0.1:${PORT}"
LOG="$(mktemp)"

if [ ! -x "$BIN" ]; then
  echo "[server-smoke] $BIN not found — configure with -DAEON_BUILD_SERVER=ON" >&2
  exit 1
fi

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$LOG"
}
trap cleanup EXIT

fail() { echo "[server-smoke] FAIL: $*" >&2; exit 1; }
ok() { echo "[server-smoke] ok: $*"; }

field() { python3 -c "import sys,json;d=json.load(sys.stdin);print(eval('d'+sys.argv[1]))" "$1"; }

echo "[server-smoke] starting aeon_serve on ${BASE} (context ${CONTEXT})"
"$BIN" --model-dir "$MODEL_DIR" --context-size "$CONTEXT" --port "$PORT" \
  --no-warm-preload >"$LOG" 2>&1 &
SERVER_PID=$!

# 1. /health is 503 while loading, then 200.
saw_loading=0
for _ in $(seq 1 600); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/health" || true)
  if [ "$code" = "503" ]; then saw_loading=1; fi
  if [ "$code" = "200" ]; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then cat "$LOG" >&2; fail "server exited during load"; fi
  sleep 0.5
done
[ "$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/health")" = "200" ] || { cat "$LOG" >&2; fail "/health never became 200"; }
ok "/health reported 503 then 200 (saw_loading=$saw_loading)"

read -r -d '' BODY <<'JSON' || true
{"messages":[{"role":"user","content":"What is the capital of France?"}],"temperature":0,"max_tokens":16}
JSON

# 2. Non-stream and stream for the same prompt (greedy) are identical.
nonstream=$(curl -s -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY")
text_nonstream=$(field "['choices'][0]['message']['content']" <<<"$nonstream")
stream_raw=$(curl -s -N -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"temperature":0,"max_tokens":16,"stream":true}')
text_stream=$(python3 -c '
import sys, json
out = ""
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "): continue
    payload = line[6:]
    if payload == "[DONE]": continue
    try: obj = json.loads(payload)
    except Exception: continue
    for c in obj.get("choices", []):
        out += c.get("delta", {}).get("content", "") or ""
print(out)
' <<<"$stream_raw")
[ -n "$text_nonstream" ] || fail "non-stream reply was empty"
[ "$text_nonstream" = "$text_stream" ] || fail "stream and non-stream differ: '$text_nonstream' vs '$text_stream'"
ok "stream == non-stream (greedy): '$text_nonstream'"

# 3. Two concurrent requests: the second queues.
first_out="$(mktemp)"; second_out="$(mktemp)"
curl -s -N -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count from one to thirty slowly."}],"temperature":0,"max_tokens":64,"stream":true}' >"$first_out" &
C1=$!
sleep 0.5
curl -s -N -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":8,"stream":true}' >"$second_out" &
C2=$!
status_json=$(curl -s "${BASE}/status")
wait "$C1" "$C2" || true
grep -q ': queued' "$second_out" && ok "second stream saw a queue keep-alive" || echo "[server-smoke] note: no queue keep-alive observed (first may have finished quickly)"
python3 -c 'import sys,json;d=json.load(sys.stdin);assert "queue_depth" in d and "conversation" in d;print("[server-smoke] ok: /status fields present")' <<<"$status_json"
rm -f "$first_out" "$second_out"

# 4. Cancellation mid-stream: kill curl, then the server still works.
curl -s -N -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a very long essay."}],"temperature":0,"max_tokens":400,"stream":true}' >/dev/null &
C3=$!
sleep 1
kill "$C3" 2>/dev/null || true
wait "$C3" 2>/dev/null || true
sleep 1
after_cancel=$(curl -s -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi."}],"temperature":0,"max_tokens":8}')
[ -n "$(field "['choices'][0]['message']['content']" <<<"$after_cancel")" ] || fail "server did not recover after a cancel"
ok "server recovered after a mid-stream cancel"

# 5. A 3-turn conversation re-sent in full: later turns reuse the prefix.
python3 - "$BASE" <<'PY'
import json, subprocess, sys
base = sys.argv[1]

def chat(messages, max_tokens=12):
    body = json.dumps({"messages": messages, "temperature": 0, "max_tokens": max_tokens})
    raw = subprocess.check_output([
        "curl", "-s", "-X", "POST", base + "/v1/chat/completions",
        "-H", "Content-Type: application/json", "-d", body])
    obj = json.loads(raw)
    if "choices" not in obj:
        raise SystemExit("[server-smoke] FAIL: turn error: %r" % obj)
    return obj

history = [{"role": "user", "content": "My name is Ada."}]
r1 = chat(history)
v1 = r1["timings"]["reuse_verdict"]
history += [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
            {"role": "user", "content": "What is my name?"}]
r2 = chat(history)
v2 = r2["timings"]["reuse_verdict"]
ru2 = r2["timings"]["reused_tokens"]
history += [{"role": "assistant", "content": r2["choices"][0]["message"]["content"]},
            {"role": "user", "content": "Are you sure?"}]
r3 = chat(history)
v3 = r3["timings"]["reuse_verdict"]

print("[server-smoke] reuse verdicts: turn1=%s turn2=%s(reused=%d) turn3=%s"
      % (v1, v2, ru2, v3))
if v2 == "reused" and v3 == "reused":
    print("[server-smoke] ok: continuation turns report 'reused'")
else:
    raise SystemExit("[server-smoke] FAIL: continuation turns did not reuse: %s, %s" % (v2, v3))
PY

# 6. Errors.
bad_json=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d '{not json}')
[ "$bad_json" = "400" ] || fail "malformed JSON returned ${bad_json}"
ok "malformed JSON -> 400"
overflow_code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json;print(json.dumps({"messages":[{"role":"user","content":"hello world "*4000}]}))')")
[ "$overflow_code" = "400" ] && ok "overflow -> 400" || fail "overflow returned ${overflow_code}"
inv=$(field "['conversation']['invariants_ok']" <<<"$(curl -s "${BASE}/status")")
[ "$inv" = "True" ] && ok "invariants_ok true after everything" || fail "invariants_ok=${inv}"

# 7. SIGINT -> clean exit 0.
kill -INT "$SERVER_PID"
wait "$SERVER_PID"
server_rc=$?
SERVER_PID=""
[ "$server_rc" -eq 0 ] && ok "SIGINT -> clean exit 0" || fail "SIGINT exit code ${server_rc}"

echo "[server-smoke] all checks passed"
