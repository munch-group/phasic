"""Capture the Plan A bit-identity goldens on MASTER code (run before any
Batch 1 source change). Writes tests/pytest/data/fd_domain_steps_golden.json.

One process per cell (a cell's failure or memory cannot affect the others) is
driven by capture_golden.sh; this script handles one cell index, or merges."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "..", "..", "..", "tests", "pytest")
sys.path.insert(0, TESTS)
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import phasic  # noqa: E402,F401
import _fd_golden_cells as C  # noqa: E402

PART = os.path.join(HERE, "golden_parts")

if sys.argv[1] == "merge":
    import subprocess
    rec = {}
    for f in sorted(os.listdir(PART)):
        rec.update(json.load(open(os.path.join(PART, f))))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=HERE).stdout.strip()
    out = {"platform": C.platform_tag(), "source_commit": commit,
           "note": "captured on master code before any Plan A source change",
           "n_cells": len(rec), "grads": rec}
    os.makedirs(os.path.dirname(C.GOLDEN_PATH), exist_ok=True)
    json.dump(out, open(C.GOLDEN_PATH, "w"), indent=1, sort_keys=True)
    print("wrote", C.GOLDEN_PATH, len(rec), "entries")
    sys.exit(0)

k = int(sys.argv[1])
name, theta = C.CELLS[k]
os.makedirs(PART, exist_ok=True)
rec = {}
for mode in C.modes_for(name):
    g = C.grad_in_mode(name, theta, mode)
    assert all(map(lambda v: v == v, g.ravel())), (name, theta, mode, g)  # no NaN
    rec[C.key(name, theta, mode)] = C.to_hex(g)
    print(name, theta, mode, g.ravel(), flush=True)
json.dump(rec, open(os.path.join(PART, f"{k:03d}.json"), "w"))
