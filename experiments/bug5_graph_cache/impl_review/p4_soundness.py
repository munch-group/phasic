import os, sys, json
import numpy as np
import phasic
from phasic import Graph, Property, StateIndexer, with_ipv, GraphCacheKeyError, GraphCacheLoadError
from phasic.callback_hash import cache_entry_key
from phasic import graph_cache as gcm

def mk():
    return lambda x: x
f, g = mk(), mk()
assert f is not g and f != g

@with_ipv([3])
def cb_tbl(state, table=None, sel=None, **kw):
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], float(table[sel]) * n]]

def attempt(label, mk_kw):
    try:
        ka = cache_entry_key("c", cb_tbl, **mk_kw(f))
        kb = cache_entry_key("c", cb_tbl, **mk_kw(g))
    except GraphCacheKeyError as e:
        print("RESULT", label, "refused:", str(e)[:90]); return
    ga = Graph(cb_tbl, **mk_kw(f)); gb = Graph(cb_tbl, **mk_kw(g))
    print("RESULT", label, "same_key=", ka == kb, "E_a=", ga.expectation(), "E_b=", gb.expectation())

# the plan's own refused case, for reference
attempt("bare function keys", lambda s: dict(table={f: 1.0, g: 5.0}, sel=s))
attempt("tuple-wrapped function keys", lambda s: dict(table={(f,): 1.0, (g,): 5.0}, sel=(s,)))
# wrappers the identity predicate does not descend into
attempt("Property-wrapped function keys", lambda s: dict(table={Property(s, max_value=1): 1.0, Property(g if s is f else f, max_value=1): 5.0} if False else {Property(f, max_value=1): 1.0, Property(g, max_value=1): 5.0}, sel=Property(s, max_value=1)))
attempt("slice-wrapped function keys", lambda s: dict(table={slice(f): 1.0, slice(g): 5.0}, sel=slice(s)))

# interpretation 3: float field inside an indexer
try:
    ix = StateIndexer(lineage=[Property("n", max_value=3.0)])
    print("RESULT float-field indexer constructs; state_length", ix.state_length)
    cache_entry_key("c", cb_tbl, indexer=ix)
    print("RESULT float-field indexer keyed")
except GraphCacheKeyError as e:
    print("RESULT float-field indexer refused:", str(e)[:120])
except Exception as e:
    print("RESULT float-field indexer construct error", repr(e)[:120])

# interpretation 8: corrupted entry whose header is destroyed
@with_ipv([3])
def cb8(state, **kw):
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], float(n)]]
g8 = Graph(cb8, cache_key="i8")
p = g8._graph_cache_entry
for label, content in (("empty file", b""), ("header bytes overwritten", None)):
    Graph(cb8, cache_key="i8") if not os.path.exists(p) else None
    if content is None:
        with open(p, "r+b") as fh:
            fh.write(b"\xff\xfe\x00\x00")
    else:
        open(p, "wb").write(content)
    try:
        Graph(cb8, cache_key="i8"); print("RESULT", label, "no error?!")
    except GraphCacheLoadError as e:
        msg = str(e)
        r = phasic.clear_all_graph_caches()
        survived = os.path.exists(p)
        try:
            Graph(cb8, cache_key="i8"); again = "ok"
        except GraphCacheLoadError:
            again = "GraphCacheLoadError again"
        print("RESULT", label, "| msg tail:", msg[-110:], "| clear_all:", r, "survived:", survived, "| retry:", again)
    if os.path.exists(p): os.unlink(p)

# interpretation 6: legacy sweep and a foreign MD5-named JSON
legacy = gcm.Path(gcm.DEFAULT_CACHE_DIR)
legacy.mkdir(parents=True, exist_ok=True)
foreign = legacy / ("0123456789abcdef0123456789abcdef.json")
foreign.write_text(json.dumps({"user": "data"}))
cur_foreign = gcm.GraphCache(create=True).cache_dir / "fedcba9876543210fedcba9876543210.json"
cur_foreign.write_text(json.dumps({"user": "data"}))
print("RESULT legacy dir differs from current:", legacy.resolve() != gcm.GraphCache().cache_dir)
r = phasic.clear_all_graph_caches()
print("RESULT foreign in LEGACY dir survives:", foreign.exists(), "| foreign in CURRENT dir survives:", cur_foreign.exists(), r)
for x in (foreign, cur_foreign):
    if x.exists(): x.unlink()
