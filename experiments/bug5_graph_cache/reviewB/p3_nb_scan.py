"""reviewB p3: read-only scan of notebook JSON (main tree) for graph-cache
references, to compare against plan §5. Never executes or edits notebooks."""
import json, pathlib, re, sys

root = pathlib.Path(sys.argv[1])
pat = re.compile(r"graph_cache|GraphCache|PHASIC_DISABLE_GRAPH_CACHE|hash_callback|"
                 r"load_graph|save_graph|get_or_build|clear_caches\(|clear_all_graph_caches|"
                 r"PHASIC_DISABLE_CACHE|cache_enabled|cache=True")
for nb in sorted(root.rglob("*.ipynb")):
    if ".ipynb_checkpoints" in nb.parts or "_freeze" in nb.parts or "_site" in nb.parts:
        continue
    try:
        cells = json.loads(nb.read_text())["cells"]
    except Exception as e:
        print(nb, "unreadable", e); continue
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        for ln in src.splitlines():
            if pat.search(ln):
                kind = c.get("cell_type")
                print(f"{nb.relative_to(root)} cell {i} [{kind}]: {ln.strip()[:110]}")
        if c.get("cell_type") == "code":
            outs = json.dumps(c.get("outputs", []))
            if pat.search(outs):
                print(f"{nb.relative_to(root)} cell {i} [OUTPUT match]")
