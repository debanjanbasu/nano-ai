#!/usr/bin/env bash
# Baseline quality run for the current artifact: one dedicated server, harnesses in sequence.
# Full: arc_easy, arc_challenge (25-shot), gsm8k (5-shot), wikitext, HumanEval+/MBPP+, BFCL single-turn (fc).
# Subsets (full runs take 5-13 h): mmlu 5-shot at 20 per subject, hellaswag 10-shot at 500.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EV=$ROOT/.scratch/evals
PORT=18097
MODEL=${MODEL:-$ROOT/models/minicpm-v-4.7-1b/minicpm-v-4.7-1b-tern4.bin}
cp "$ROOT/target/release/micro-ai" /tmp/micro-ai-baseline
/tmp/micro-ai-baseline serve --host 127.0.0.1 --port $PORT --model "$MODEL" > "$EV/results/baseline_server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
python3 - <<EOF
import time, urllib.request
for _ in range(600):
    try:
        urllib.request.urlopen("http://127.0.0.1:$PORT/health", timeout=2); break
    except Exception:
        time.sleep(0.5)
EOF
cd "$EV"
mkdir -p results
bash "$ROOT/evals/lm_eval.sh" $PORT "" arc_easy gsm8k wikitext arc_challenge
bash "$ROOT/evals/lm_eval.sh" $PORT 20 mmlu
bash "$ROOT/evals/lm_eval.sh" $PORT 500 hellaswag
bash "$ROOT/evals/evalplus.sh" $PORT "" humaneval mbpp
bash "$ROOT/evals/bfcl.sh" $PORT "" fc single_turn
echo "BASELINE DONE"
