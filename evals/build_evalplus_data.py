"""Build evalplus HUMANEVAL_OVERRIDE_PATH / MBPP_OVERRIDE_PATH jsonl files from the cached HF parquet.

The HF export folds base and plus tests into one `inputs = [...]` list, base first. The base part is the
longest prefix whose argument lists occur in the original tests: HumanEval asserts (data/HumanEval.jsonl.gz
from the `human-eval` wheel) or MBPP `test_list`.
Usage: python build_evalplus_data.py OUT_DIR
"""

import ast
import glob
import gzip
import json
import math
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.path.join(ROOT, ".scratch", "evals")
HUB = os.path.join(ROOT, ".scratch", "hf", "hub")
NS = {"inf": math.inf, "nan": math.nan}


def parquet(repo):
    return pd.read_parquet(glob.glob(f"{HUB}/datasets--evalplus--{repo}/snapshots/*/data/*.parquet")[0])


def literal(test, name):
    return eval(re.search(rf"^\s*{name} = (.*)$", test, re.M).group(1), dict(NS))  # noqa: S307 - trusted dataset


def atol(test):
    # HumanEval/32 checks `_poly(...) <= tol` instead of calling assertion(); evalplus' oracle uses atol for it.
    m = re.search(r"^\s*assertion\(.*, ([\d.e+-]+)\)\s*$", test, re.M) or re.search(r"<= ([\d.e+-]+)\s*$", test, re.M)
    return float(m.group(1)) if m else 0.0


def call_args(test, fn):
    """Positional args of the first `fn(...)` call in each assert, and the number of such asserts."""
    out, n = [], 0
    for node in ast.walk(ast.parse(test)):
        call = isinstance(node, ast.Assert) and next(
            (c for c in ast.walk(node) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == fn), None)
        if call:
            n += 1
            try:
                out.append([eval(ast.unparse(a), dict(NS)) for a in call.args])  # noqa: S307 - trusted dataset
            except NameError:  # args built from loop variables; such asserts are not in the HF inputs
                pass
    return out, n


def key(x):
    return json.dumps(x, default=str)  # tuples -> lists, as in the official release


def split(task_id, inputs, base_args, n_asserts):
    seen = {key(a) for a in base_args}
    k = 0
    while k < len(inputs) and key(list(inputs[k])) in seen:
        k += 1
    if k != n_asserts:
        print(f"note {task_id}: base={k} of {len(inputs)} inputs, original asserts={n_asserts}", file=sys.stderr)
    return json.loads(key(inputs[:k])), json.loads(key(inputs[k:]))


def main(out):
    os.makedirs(out, exist_ok=True)
    orig = {t["task_id"]: t for t in map(json.loads, gzip.open(os.path.join(SCRATCH, "data", "HumanEval.jsonl.gz")))}
    with open(os.path.join(out, "HumanEvalPlus.jsonl"), "w") as f:
        for r in parquet("humanevalplus").itertuples():
            base, plus = split(r.task_id, literal(r.test, "inputs"), *call_args(orig[r.task_id]["test"], "candidate"))
            f.write(json.dumps({"task_id": r.task_id, "prompt": r.prompt, "entry_point": r.entry_point,
                                "canonical_solution": r.canonical_solution, "contract": "",
                                "base_input": base, "plus_input": plus, "atol": atol(r.test)}) + "\n")
    with open(os.path.join(out, "MbppPlus.jsonl"), "w") as f:
        for r in parquet("mbppplus").itertuples():
            tests = list(r.test_list)
            entry = re.search(r"assertion\((\w+)\(\*inp\)", r.test).group(1)
            calls = [call_args(t, entry) for t in tests]
            base, plus = split(r.task_id, literal(r.test, "inputs"), [a for c, _ in calls for a in c], len(tests))
            # Official MBPP+ prompt: description plus the first test, in a docstring.
            prompt = f'"""\n{r.prompt}\n{tests[0]}\n"""\n'
            f.write(json.dumps({"task_id": f"Mbpp/{r.task_id}", "prompt": prompt, "entry_point": entry,
                                "canonical_solution": r.code, "contract": "",
                                "base_input": base, "plus_input": plus, "atol": atol(r.test)}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
