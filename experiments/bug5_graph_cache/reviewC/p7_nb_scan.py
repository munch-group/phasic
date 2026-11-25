"""Review C p7: read-only scan of notebook JSON in the MAIN tree for
graph-cache API uses (source cells only; outputs reported separately)."""
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
pat = re.compile(r"graph_cache|GraphCache|DISABLE_GRAPH_CACHE|DISABLE_CACHE|save_graph|load_graph|clear_caches|clear_all_graph_caches|hash_callback|get_or_build")
for nb in sorted(root.rglob("*.ipynb")):
    if ".ipynb_checkpoints" in nb.parts or "_site" in nb.parts or ".pixi" in nb.parts:
        continue
    try:
        cells = json.loads(nb.read_text())["cells"]
    except Exception as e:
        print(nb, "ERR", e); continue
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        for line in src.splitlines():
            if pat.search(line):
                print(f"{nb.relative_to(root)} cell {i} [{c['cell_type']}]: {line.strip()[:110]}")
        outs = "".join("".join(o.get("text", [])) if isinstance(o.get("text"), list) else str(o.get("text", "")) for o in c.get("outputs", []))
        if pat.search(outs):
            print(f"{nb.relative_to(root)} cell {i} [output]")
