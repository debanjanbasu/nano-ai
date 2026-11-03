#!/usr/bin/env bash
# Usage: lm_eval.sh PORT [LIMIT] [TASK...]
# Runs lm-eval (local-completions, token-id prompts) against a micro-ai server, offline.
# Few-shot = Open LLM Leaderboard v1 settings; arc_easy and wikitext use the lm-eval defaults (0-shot).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EV=$ROOT/.scratch/evals
PORT=${1:?PORT}
LIMIT=${2:-}
shift $(( $# < 2 ? $# : 2 ))
TASKS=${*:-mmlu arc_easy arc_challenge hellaswag gsm8k wikitext}
export HF_HOME=$ROOT/.scratch/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-$EV/results/lm_eval}
mkdir -p "$OUT"
# The server runs one request at a time; batch_size=16 sends 16 token-id prompts per call.
MODEL_ARGS="model=minicpm-v-4.7-1b,base_url=http://127.0.0.1:$PORT/v1/completions,tokenizer=$ROOT/.scratch/tokenizer,tokenizer_backend=huggingface,tokenized_requests=True,num_concurrent=1,max_retries=3,timeout=3600,max_length=4096"
for t in $TASKS; do
  case $t in
    mmlu) shots=5 ;; arc_challenge) shots=25 ;; hellaswag) shots=10 ;; gsm8k) shots=5 ;; *) shots=0 ;;
  esac
  "$ROOT/.scratch/venv-eval/bin/python" "$ROOT/evals/offline_datasets.py" lm_eval run \
    --model local-completions --model_args "$MODEL_ARGS" --batch_size 16 \
    --tasks "$t" --num_fewshot "$shots" \
    --output_path "$OUT/$t" --log_samples \
    --use_cache "$OUT/cache/$t${LIMIT:+_limit$LIMIT}" \
    ${LIMIT:+--limit "$LIMIT"} 2>&1 | tee "$OUT/$t${LIMIT:+_limit$LIMIT}.log"
done
