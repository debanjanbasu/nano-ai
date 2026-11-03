#!/usr/bin/env bash
# Usage: bfcl.sh PORT [LIMIT] [MODE] [CATEGORIES]
#   MODE: xml (default; official chat template rendered client-side, POST /v1/completions)
#         fc  (POST /v1/chat/completions with `tools`; needs server tool-call support)
#   CATEGORIES: bfcl test categories/groups, comma-separated (default single_turn,multi_turn;
#   web_search needs internet and memory is agentic, both left out). LIMIT = first N entries per category.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EV=$ROOT/.scratch/evals
PORT=${1:?PORT}
LIMIT=${2:-}
MODE=${3:-xml}
CATS=${4:-single_turn,multi_turn}
case $MODE in xml) MODEL=minicpm-v-4.7-1b-xml ;; fc) MODEL=minicpm-v-4.7-1b-FC ;; *) echo "MODE must be xml or fc" >&2; exit 2 ;; esac
export BFCL_PROJECT_ROOT=${BFCL_PROJECT_ROOT:-$EV/results/bfcl${LIMIT:+_limit$LIMIT}}
export HF_HOME=$ROOT/.scratch/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LOCAL_SERVER_ENDPOINT=127.0.0.1 LOCAL_SERVER_PORT=$PORT                     # xml mode
export OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1 OPENAI_API_KEY=none               # fc mode
mkdir -p "$BFCL_PROJECT_ROOT"
PY=$ROOT/.scratch/venv-eval/bin/python
RUN=(--test-category "$CATS")
if [ -n "$LIMIT" ]; then
  "$PY" - "$CATS" "$LIMIT" > "$BFCL_PROJECT_ROOT/test_case_ids_to_generate.json" <<'EOF'
import json, sys
from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument
cats, n = sys.argv[1].split(","), int(sys.argv[2])
print(json.dumps({c: [e["id"] for e in load_dataset_entry(c)[:n]] for c in parse_test_category_argument(cats)}))
EOF
  RUN=(--run-ids)
fi
# The server decodes one request at a time, so one thread.
"$PY" "$ROOT/evals/bfcl_run.py" generate --model "$MODEL" "${RUN[@]}" --num-threads 1 --temperature 0 \
  --skip-server-setup --local-model-path "$ROOT/.scratch/tokenizer" 2>&1 | tee "$BFCL_PROJECT_ROOT/generate_$MODE.log"
"$PY" "$ROOT/evals/bfcl_run.py" evaluate --model "$MODEL" --test-category "$CATS" ${LIMIT:+--partial-eval} \
  2>&1 | tee "$BFCL_PROJECT_ROOT/evaluate_$MODE.log"
