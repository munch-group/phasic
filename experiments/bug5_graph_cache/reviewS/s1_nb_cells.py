"""Print selected notebook cells (read-only JSON read)."""
import json, sys
root = "/Users/kmt/phasic/docs/pages/tutorial/"
for nb, cells in [("python_trace-and-jax-caching.ipynb", range(2, 13)),
                  ("trace-and-jax-caching-c-path.ipynb", [8, 10, 12]),
                  ("state-space.ipynb", range(28, 46))]:
    d = json.load(open(root + nb, encoding="utf-8"))
    for i in cells:
        c = d["cells"][i]
        if c["cell_type"] != "code":
            continue
        print(f"===== {nb} cell {i}")
        print("".join(c["source"]))
