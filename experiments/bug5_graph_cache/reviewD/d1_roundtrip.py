"""Review D probe 1: does serialize()->from_serialized() (the cache hit path)
give the same graph state as a fresh callback build? Unmodified master code.
Run with HOME / PHASIC_CACHE_DIR in scratch; no graph cache used (round trip
done in memory, through the same JSON encode/decode the cache uses)."""
import os, json
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import numpy as np, phasic
from phasic.graph_cache import _serialize_value, _deserialize_value

def via_cache_format(g):
    d = json.loads(json.dumps(_serialize_value(g.serialize())))
    return phasic.Graph.from_serialized(_deserialize_value(d))

def describe(tag, g):
    out = {}
    out["n"] = g.vertices_length()
    out["param_length"] = g.param_length()
    out["parameterized"] = g.parameterized()
    out["dyn"] = g.dyn_ordering
    w = []
    for v in g.vertices():
        for e in v.edges():
            w.append(round(e.weight(), 6))
    out["weights_sorted"] = sorted(w)
    try:
        out["E_before_update"] = round(float(g.expectation()), 6)
    except Exception as ex:
        out["E_before_update"] = f"ERR {type(ex).__name__}: {ex}"
    print(tag, out)
    return out

# (a) 3-tuple with a non-zero vestigial weight and coefficients
def cb3(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], 7.0, [n * (n - 1) / 2]]]
# (b) 2-tuple (state, [coeffs])
def cb2(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], [n * (n - 1) / 2]]]
# (c) constant
def cbc(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], n * (n - 1) / 2]]

for name, cb in [("3tuple", cb3), ("2tuple", cb2), ("const", cbc)]:
    g = phasic.Graph(cb, ipv=[[[5], 1.0]], graph_cache=False)
    h = via_cache_format(g)
    a = describe(name + " fresh", g)
    b = describe(name + " hit  ", h)
    if a["parameterized"]:
        g.update_weights([2.0]); h.update_weights([2.0])
        print(name, "after update E fresh/hit", g.expectation(), h.expectation())
    print(name, "SAME" if a == b else "DIFFERENT")
