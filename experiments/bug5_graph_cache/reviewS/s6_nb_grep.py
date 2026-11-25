"""reviewS s6: read-only grep of notebook CODE cell sources for removed graph-cache API."""
import json, glob, os, re
pat = re.compile(r"hash_callback|get_or_build|load_graph|save_graph|is_graph_cache_disabled|PHASIC_DISABLE_GRAPH_CACHE|graph_cache\s*=|_deserialize_value|callback_hash|PHASIC_CALLBACK_VERSION")
for p in sorted(glob.glob("/Users/kmt/phasic/docs/**/*.ipynb", recursive=True)):
    try: d = json.load(open(p, encoding="utf-8"))
    except Exception: continue
    for i, c in enumerate(d.get("cells", [])):
        if c.get("cell_type") != "code": continue
        for ln in "".join(c["source"]).splitlines():
            if pat.search(ln):
                print(f"{os.path.relpath(p,'/Users/kmt/phasic/docs')}[{i}]: {ln.strip()[:100]}")
