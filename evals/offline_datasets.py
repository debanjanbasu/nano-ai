"""Run a Python CLI entry point with datasets.load_dataset served from the local HF hub snapshots.

datasets>=5 refuses hub repo ids in offline mode unless an Arrow cache already exists, so map
`load_dataset("org/name", config)` to the cached parquet files of that repo's snapshot.
Usage: python offline_datasets.py <module> [args...]   (e.g. lm_eval run ...)
"""

import glob
import os
import re
import runpy
import sys

import datasets

_load = datasets.load_dataset
_HUB = os.path.join(os.environ["HF_HOME"], "hub")


def load_dataset(path, name=None, *args, data_files=None, revision=None, **kwargs):
    # One snapshot per cached repo; hellaswag/wikitext only exist as refs/convert/parquet.
    snaps = glob.glob(os.path.join(_HUB, f"datasets--{str(path).replace('/', '--')}", "snapshots", "*"))
    if len(snaps) != 1 or data_files is not None:
        return _load(path, name, *args, data_files=data_files, revision=revision, **kwargs)
    root = os.path.join(snaps[0], name or "default")
    files = {}
    for f in sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)):
        rel = os.path.relpath(f, root)
        # Layouts: <split>-00000-of-00001.parquet (main) or <split>/0000.parquet (refs/convert/parquet).
        split = rel.split(os.sep)[0] if os.sep in rel else re.sub(r"-\d{5}-of-\d{5}\.parquet$", "", rel)
        files.setdefault(split, []).append(f)
    if not files:
        raise FileNotFoundError(f"no cached parquet for {path} config {name} under {root}")
    return _load("parquet", *args, data_files=files, **kwargs)


datasets.load_dataset = load_dataset
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
