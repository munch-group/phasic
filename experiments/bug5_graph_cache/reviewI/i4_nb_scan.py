"""Review I probe 4: read-only scan of the main tree's notebooks (sources only,
0-based cell index) for graph-cache API names, to check plan section 5."""
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
pat = re.compile(r"graph_cache|GraphCache|PHASIC_DISABLE_GRAPH_CACHE|save_graph|load_graph|hash_callback|get_or_build|clear_all_graph_caches|PHASIC_DISABLE_CACHE")
for nb in sorted(root.rglob("*.ipynb")):
    if ".ipynb_checkpoints" in nb.parts or "_site" in nb.parts or "_freeze" in nb.parts:
        continue
    try:
        cells = json.loads(nb.read_text())["cells"]
    except Exception as e:
        print(nb, "unreadable", e); continue
    hits = []
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        for line in src.splitlines():
            if pat.search(line):
                hits.append((i, c["cell_type"][0], line.strip()[:110]))
    if hits:
        print("==", nb.relative_to(root))
        for h in hits:
            print("  ", h)
