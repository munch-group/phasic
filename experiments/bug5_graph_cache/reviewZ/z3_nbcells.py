"""Read-only dump of the notebook cells the plan says should switch to cache_key
(never executes or edits notebooks)."""
import json, sys
ROOT = "/Users/kmt/phasic/docs/pages/"
targets = {
    "tutorial/state-space.ipynb": [33, 34, 35, 36, 38, 45],
    "tutorial/python_trace-and-jax-caching.ipynb": [8, 10, 12],
}
for nb, cells in targets.items():
    d = json.load(open(ROOT + nb))
    for i in cells:
        c = d["cells"][i]
        src = "".join(c["source"])
        print(f"===== {nb} cell {i} ({c['cell_type']})")
        print(src[:1800])
