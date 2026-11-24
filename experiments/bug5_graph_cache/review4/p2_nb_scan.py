"""Read-only scan of /Users/kmt/phasic/docs notebooks for graph-cache API uses."""
import json, pathlib, re
root = pathlib.Path("/Users/kmt/phasic/docs")
pat = re.compile(r"graph_cache|GraphCache|save_graph|load_graph|get_or_build|hash_callback|"
                 r"PHASIC_DISABLE_GRAPH_CACHE|is_graph_cache_disabled|PHASIC_CALLBACK_VERSION|"
                 r"clear_all_graph_caches|print_graph_cache_info|get_graph_cache_stats|clear_from_cache")
for nb in sorted(root.rglob("*.ipynb")):
    if ".ipynb_checkpoints" in nb.parts or "_site" in nb.parts or "_freeze" in nb.parts:
        continue
    try:
        cells = json.loads(nb.read_text())["cells"]
    except Exception as e:
        print(nb, "UNREADABLE", e); continue
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        for ln in src.splitlines():
            if pat.search(ln):
                print(f"{nb.relative_to(root)} cell {i} [{c['cell_type'][:4]}]: {ln.strip()[:110]}")
# also non-notebook docs sources
for f in sorted(list(root.rglob("*.qmd")) + list(root.rglob("*.py")) + list(root.rglob("*.md"))):
    if "_site" in f.parts or "_freeze" in f.parts or "api" in f.parts:
        continue
    for j, ln in enumerate(f.read_text(errors="replace").splitlines()):
        if pat.search(ln):
            print(f"{f.relative_to(root)}:{j+1}: {ln.strip()[:110]}")
