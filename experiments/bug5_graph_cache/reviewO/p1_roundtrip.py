"""reviewO p1: does Graph.from_serialized(g.serialize()) (what a cache hit
returns) behave like a fresh build, for callback graphs a user can write?
Run with HOME / PHASIC_CACHE_DIR pointing at a scratch dir.
Uses graph_cache=False so master's cache is not involved."""
import os, numpy as np
assert "phasic_cache" not in os.environ.get("PHASIC_CACHE_DIR", "") or True
import phasic
from phasic import Graph

N = 4

def coal_3tuple(state, w=7.0, **kw):
    # 3-tuple form (state, weight, coeffs): weight documented as vestigial
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], w, [n * (n - 1) / 2])]

def coal_aux(state, **kw):
    # 3 coefficients, theta_dim=2 (auxiliary coefficient, formula/callback mode)
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2, 0.0, float(n)])]

def coal_const(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2)]

def edges_of(g):
    out = []
    for v in g:
        out.append((tuple(v.state()), [(tuple(e.to().state()), e.weight()) for e in v.edges()]))
    return out

def compare(name, g):
    h = Graph.from_serialized(g.serialize())
    print(f"--- {name}")
    print("  param_length fresh/hit:", g.param_length(), h.param_length(),
          " parameterized:", g.parameterized(), h.parameterized())
    print("  vertices:", g.vertices_length(), h.vertices_length())
    ef, eh = edges_of(g), edges_of(h)
    print("  edge weights equal before update_weights:", ef == eh)
    if ef != eh:
        print("   fresh:", ef[:3]); print("   hit:  ", eh[:3])
    for nm, gg in (("fresh", g), ("hit", h)):
        try:
            print(f"  {nm} expectation (no update_weights):", gg.expectation())
        except Exception as e:
            print(f"  {nm} expectation raised {type(e).__name__}: {e}")
    return h

compare("3-tuple weight 7.0", Graph(coal_3tuple, ipv=[N], graph_cache=False))
g = Graph(coal_aux, ipv=[N], theta_dim=2, graph_cache=False)
h = compare("aux coefficient, theta_dim=2", g)
for nm, gg in (("fresh", g), ("hit", h)):
    try:
        gg.update_weights([1.0, 1.0], lambda t, c: c[0] * t[0] + c[2])
        print(f"  {nm} callback-mode expectation:", gg.expectation())
    except Exception as e:
        print(f"  {nm} callback update raised {type(e).__name__}: {e}")
compare("constant", Graph(coal_const, ipv=[N], graph_cache=False))
