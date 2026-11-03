#!/usr/bin/env bash
# Usage: variant.sh NAME MODEL_FILE [PORT]
# Scores one artifact against the standard harnesses: wikitext, arc_easy, gsm8k, mmlu (20/subject),
# HumanEval+/MBPP+ and BFCL single-turn. Each harness runs against its own server so a failed
# harness does not take the others down.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EV=$ROOT/.scratch/evals
NAME=$1
MODEL=$2
PORT=${3:-18098}
RES=$EV/results/variants/$NAME
mkdir -p "$RES"
cp "$ROOT/target/release/micro-ai" "/tmp/micro-ai-variant-$NAME"
"/tmp/micro-ai-variant-$NAME" serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" > "$RES/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
"$ROOT/.scratch/venv-eval/bin/python" - "$PORT" "$RES/smoke.json" <<'EOF'
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
for _ in range(600):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
    except Exception:
        time.sleep(0.5)
else:
    sys.exit("server never became healthy")
body = {"model": "minicpm-v-4.7-1b", "temperature": 0, "max_tokens": 128,
        "messages": [{"role": "user", "content": "Explain in three sentences why the sky is blue, then give 17*23."}]}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
open(out, "w").write(json.dumps(r, indent=2))
print(r["choices"][0]["message"]["content"])
EOF
[ $? -eq 0 ] || exit 1
export OUT=$RES
bash "$ROOT/evals/lm_eval.sh" "$PORT" "" wikitext arc_easy
bash "$ROOT/evals/lm_eval.sh" "$PORT" "" gsm8k
bash "$ROOT/evals/lm_eval.sh" "$PORT" 20 mmlu
bash "$ROOT/evals/evalplus.sh" "$PORT" "" humaneval mbpp
BFCL_PROJECT_ROOT=$RES/bfcl bash "$ROOT/evals/bfcl.sh" "$PORT" "" fc single_turn
echo "VARIANT $NAME DONE"
