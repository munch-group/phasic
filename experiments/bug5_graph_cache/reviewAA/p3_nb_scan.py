"""reviewAA p3: read-only scan of notebooks for graph-cache API use (source
only, 0-based cell indices).  Usage: python p3_nb_scan.py <docs dir>"""
import json, re, sys, pathlib
root = pathlib.Path(sys.argv[1])
pat = re.compile(r"graph_cache|GraphCache|save_graph|load_graph|PHASIC_DISABLE_GRAPH_CACHE|clear_all_graph_caches|hash_callback|get_or_build")
for nb in sorted(root.rglob("*.ipynb")):
    if ".ipynb_checkpoints" in nb.parts or "_freeze" in nb.parts or "_site" in nb.parts:
        continue
    try:
        cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
    except Exception as ex:
        print(nb, "unreadable", ex); continue
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        for line in src.splitlines():
            if pat.search(line):
                print(f"{nb.relative_to(root)} cell {i} [{c.get('cell_type')}]: {line.strip()[:110]}")
