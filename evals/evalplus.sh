#!/usr/bin/env bash
# Usage: evalplus.sh PORT [LIMIT] [DATASET...]   (datasets: humaneval mbpp; LIMIT = first N problems)
# EvalPlus 0.3.1 (own venv: bfcl-eval pins tree-sitter 0.21) with the OpenAI chat backend, greedy, n=1.
# Datasets are built offline from the cached HF parquet (build_evalplus_data.py).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EV=$ROOT/.scratch/evals
PY=$ROOT/.scratch/venv-evalplus/bin
PORT=${1:?PORT}
LIMIT=${2:-}
shift $(( $# < 2 ? $# : 2 ))
OUT=${OUT:-$EV/results/evalplus${LIMIT:+_limit$LIMIT}}
mkdir -p "$OUT"
[ -s "$EV/data/MbppPlus.jsonl" ] || "$PY/python" "$ROOT/evals/build_evalplus_data.py" "$EV/data"
# macOS rejects evalplus' RLIMIT_AS cap ("current limit exceeds maximum limit"); -1 skips it.
export OPENAI_API_KEY=none EVALPLUS_MAX_MEMORY_BYTES=-1
for ds in ${*:-humaneval mbpp}; do
  case $ds in humaneval) f=HumanEvalPlus ;; mbpp) f=MbppPlus ;; *) echo "unknown dataset $ds" >&2; exit 2 ;; esac
  data=$EV/data/$f.jsonl
  # evaluate requires a sample for every problem, so a limited run uses a truncated dataset.
  if [ -n "$LIMIT" ]; then head -n "$LIMIT" "$data" > "$OUT/$f.jsonl"; data=$OUT/$f.jsonl; fi
  export HUMANEVAL_OVERRIDE_PATH=$data MBPP_OVERRIDE_PATH=$data
  "$PY/evalplus.codegen" --model minicpm-v-4.7-1b --dataset "$ds" --backend openai \
    --base-url "http://127.0.0.1:$PORT/v1" --greedy --root "$OUT" 2>&1 | tee "$OUT/${ds}_codegen.log"
  "$PY/evalplus.evaluate" --dataset "$ds" \
    --samples "$OUT/$ds/minicpm-v-4.7-1b_openai_temp_0.0.jsonl" 2>&1 | tee "$OUT/${ds}_evaluate.log"
done
