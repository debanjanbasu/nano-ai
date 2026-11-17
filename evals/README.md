# Quality harnesses

Standard benchmarks against a running `nano-ai serve` on `127.0.0.1`:

| Harness | Script | Task |
| --- | --- | --- |
| lm-eval | `lm_eval.sh` | wikitext, arc_easy, arc_challenge, gsm8k, mmlu, hellaswag |
| EvalPlus | `evalplus.sh` | HumanEval+, MBPP+ |
| BFCL | `bfcl.sh` | tool calling (`fc` = OpenAI `tools`, `xml` = MiniCPM-V 4.7 chat template) |

`baseline.sh` runs the whole gate for the shipped artifact (full runs take 5-13 h; it uses
`--limit 20` for MMLU and `--limit 500` for HellaSwag).

`smoke_toolcall.py` checks the tool-calling path end to end against a running server: OpenAI
`tools` (non-streaming and streaming), the tool-result follow-up turn, a coding-agent tool
choice, `tool_choice=required` / named `tool_choice`, and a `stop` sequence.

```bash
python3 evals/smoke_toolcall.py http://127.0.0.1:18093/v1
```

## Prerequisites

The Python venvs and the dataset caches are gitignored and live under `.scratch/`:

```bash
python3 -m venv .scratch/venv-eval && .scratch/venv-eval/bin/pip install lm-eval bfcl-eval
python3 -m venv .scratch/venv-evalplus && .scratch/venv-evalplus/bin/pip install evalplus==0.3.1
```

`HF_HOME=.scratch/hf` must hold the cached snapshots (wikitext, arc, gsm8k, mmlu, hellaswag,
evalplus/humanevalplus, evalplus/mbppplus, the BFCL datasets) and `.scratch/tokenizer` the
MiniCPM-V 4.7 tokenizer. `offline_datasets.py` maps `load_dataset("org/name")` onto those
parquet files because `datasets` refuses hub repo ids in offline mode.

## Measured scores (shipped `int4-qat-awq` artifact, M4 Pro)

| Task | Metric | Score |
| --- | --- | --- |
| wikitext | word perplexity | 22.26 |
| arc_easy (0-shot) | acc / acc_norm | 0.652 / 0.588 |
| gsm8k (5-shot) | flexible / strict | 0.594 / 0.556 |
| mmlu (5-shot, 20/subject) | acc | 0.557 |
| HumanEval+ / MBPP+ | pass@1 | 0.384 / 0.405 |
| BFCL (single turn, `fc`) | overall acc | 19.54% |

BF16 reference on the same server: wikitext 21.17, arc_easy 0.666 / 0.600, gsm8k 0.656 / 0.640, BFCL 16.14%.
