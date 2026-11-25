"""Does a cache HIT (master's restore path = from_serialized(serialize()) +
attribute copy) equal a FRESH build in everything a user can observe, for
graph kinds reachable through Graph(callback, ...)? The plan relies on
'the current restore sequence (:1689-1723) runs' plus dyn_ordering.
Run with HOME redirected (master's cache is on by default)."""
import os, sys, tempfile
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import numpy as np
import phasic
from phasic import Graph


def summary(g):
    out = {}
    out["n"] = g.vertices_length()
    out["parameterized"] = g.parameterized()
    out["param_length"] = g.param_length()
    w = []
    for v in g:
        for e in v.edges():
            w.append(round(float(e.weight()), 12))
    out["weights"] = w
    try:
        out["E_before_update"] = float(g.expectation())
    except Exception as ex:
        out["E_before_update"] = f"raised {type(ex).__name__}: {str(ex)[:80]}"
    return out


def cb3(state, **kw):
    # 3-tuple form (state, weight, coeffs): weight documented as vestigial
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], 2.5, [n * (n - 1) / 2])]


def cb2(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2])]


def cbc(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2)]


for name, cb in [("3-tuple", cb3), ("2-tuple coeffs", cb2), ("constant", cbc)]:
    fresh = Graph(cb, ipv=[5], graph_cache=False)
    rt = Graph.from_serialized(fresh.serialize())
    Graph(cb, ipv=[5], graph_cache=True)          # warm
    hit = Graph(cb, ipv=[5], graph_cache=True)    # master hit path
    sf, sr, sh = summary(fresh), summary(rt), summary(hit)
    print(name)
    for k in sf:
        flag = "" if sf[k] == sh[k] else "   <-- HIT DIFFERS FROM FRESH"
        print(f"   {k}: fresh={sf[k]} roundtrip={sr[k]} hit={sh[k]}{flag}")
    if sf["parameterized"]:
        for g in (fresh, hit):
            g.update_weights([1.0])
        print("   E after update_weights([1]): fresh", fresh.expectation(), "hit", hit.expectation())

phasic.GraphCache().clear()
