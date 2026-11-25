"""reviewAC p2: print (read-only) the notebook cells the plan converts to cache_key."""
import json, sys
base = "/Users/kmt/phasic/docs/pages/"
want = {
    "tutorial/python_trace-and-jax-caching.ipynb": [2, 3, 4, 5, 6, 8, 10, 12],
    "tutorial/state-space.ipynb": [34, 35, 36, 37, 38, 45],
}
for nb, cells in want.items():
    d = json.load(open(base + nb))
    for i in cells:
        c = d["cells"][i]
        print(f"===== {nb} cell {i} ({c['cell_type']})")
        print("".join(c["source"])[:2500])
