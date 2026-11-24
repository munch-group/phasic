"""reviewJ p3: print (read-only, never executed) the source of notebook cells the
plan tells the user to switch to cache_key=..., plus the cells defining the
callbacks they use, to check whether the advised call would be keyable under D3.
Reads the MAIN tree's notebooks (the plan's §5 scan source)."""
import json, sys, re
base = "/Users/kmt/phasic/docs/pages/"
targets = {
    "tutorial/python_trace-and-jax-caching.ipynb": None,
    "tutorial/trace-and-jax-caching-c-path.ipynb": None,
    "tutorial/state-space.ipynb": None,
}
pat = re.compile(r"graph_cache|def [a-z_]+\(state|lru_cache|partial\(|import |@with_ipv|@callback|StateIndexer\(|Graph\(")
for nb, _ in targets.items():
    cells = json.load(open(base + nb))["cells"]
    print("=" * 30, nb, len(cells))
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if pat.search(src):
            print(f"--- cell {i}")
            print(src[:1500])
