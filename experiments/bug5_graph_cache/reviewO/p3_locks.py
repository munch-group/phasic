"""reviewO p3: a cache hit is from_serialized(serialize()); compare the
C-level locks (param_length lock, edge-mode lock) of a fresh callback build
with the round-tripped graph by attempting the same follow-up operation on
each. graph_cache=False: master's cache not involved."""
import numpy as np
from phasic import Graph

def coal(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2])]

def coal_const(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2)]

def attempt(label, fn):
    try:
        fn()
        return f"{label}: ok"
    except Exception as e:
        return f"{label}: {type(e).__name__}: {str(e)[:90]}"

for name, cb, kw in (("param theta_dim=None", coal, {}),
                     ("param theta_dim=1", coal, {"theta_dim": 1}),
                     ("param theta_dim=3", coal, {"theta_dim": 3}),
                     ("constant", coal_const, {})):
    g = Graph(cb, ipv=[4], graph_cache=False, **kw)
    h = Graph(Graph.from_serialized(g.serialize()))  # what the hit path builds
    print(f"--- {name}: param_length fresh/hit {g.param_length()}/{h.param_length()}")
    for gg, tag in ((g, "fresh"), (h, "hit  ")):
        v = gg.find_or_create_vertex([7])
        w = gg.find_or_create_vertex([8])
        r1 = attempt("constant edge", lambda: v.add_edge(w, 1.0))
        r2 = attempt("2-coef edge", lambda: w.add_edge(v, [1.0, 2.0]))
        print(f"  {tag} {r1} | {r2}")
    try:
        sf, sh = g.serialize(), h.serialize()
    except Exception as e:
        print("  serialize after edits:", type(e).__name__, str(e)[:80])
