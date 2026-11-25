"""reviewX x1: does a serialize()/from_serialized() round trip (the cache-hit
construction path) reproduce the fresh build's per-vertex EDGE ORDER and
derived results (expectation bits, seeded sample())?

Run with HOME redirected; writes no cache files (no graph_cache use)."""
import os, sys, random
import numpy as np
import phasic
from phasic import Graph

def edges_of(g):
    out = []
    for v in g.vertices():
        row = []
        for e in v.edges():
            row.append((tuple(e.to().state()), float(e.weight()), e.coefficients_length()))
        out.append((tuple(v.state()), row))
    return out

def report(name, cb, ipv, theta=None, **kw):
    g = Graph(cb, ipv=ipv, graph_cache=False, **kw)
    h = Graph.from_serialized(g.serialize())
    if theta is not None:
        g.update_weights(theta); h.update_weights(theta)
    eg, eh = edges_of(g), edges_of(h)
    same_order = eg == eh
    same_multiset = [ (s, sorted(r)) for s, r in eg ] == [ (s, sorted(r)) for s, r in eh ]
    print(f"[{name}] vertices {g.vertices_length()} vs {h.vertices_length()}")
    print(f"[{name}] per-vertex edge ORDER identical: {same_order}; edge multiset identical: {same_multiset}")
    if not same_order:
        for (s, r1), (_, r2) in zip(eg, eh):
            if r1 != r2:
                print(f"   first differing vertex {s}: fresh {r1}  hit {r2}")
                break
    try:
        a, b = g.expectation(), h.expectation()
        print(f"[{name}] expectation fresh {a!r} hit {b!r} bit-equal {a == b}")
    except Exception as e:
        print(f"[{name}] expectation error {type(e).__name__}: {e}")
    try:
        random.seed(7); np.random.seed(7); s1 = np.asarray(g.sample(2000))
        random.seed(7); np.random.seed(7); s2 = np.asarray(h.sample(2000))
        print(f"[{name}] seeded sample(2000) identical: {np.array_equal(s1, s2)}; means {s1.mean():.6f} {s2.mean():.6f}")
    except Exception as e:
        print(f"[{name}] sample error {type(e).__name__}: {e}")

# 1. constant-rate model, two transitions per vertex (baseline: should match)
def cb_const(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2), ([n - 1 if n > 2 else 1], 0.5)] if False else [([n - 1], n * (n - 1) / 2)]

# 2. parameterized model where one vertex's list mixes a coefficient edge and a
#    constant (empty-coefficient) edge, via 3-tuples
def cb_mixed(state, **kw):
    a, b = state
    out = []
    if a > 0:
        out.append(([a - 1, b + 1], 0.0, []))      # constant-ish edge (empty coeffs) FIRST
    if b > 0:
        out.append(([a + 1 if a < 3 else a, b - 1], 0.0, [1.0 * b]))  # parameterized SECOND
    if a == 0 and b == 0:
        return []
    return out

# 3. parameterized model, per-vertex format differs: some vertices return
#    (state, weight) and others (state, [coeffs])
def cb_pervertex(state, **kw):
    n, k = state
    if n <= 1:
        return []
    if n % 2 == 0:
        return [([n - 1, k], float(n)), ([n - 1, k + 1], 1.0)]
    return [([n - 1, k], [float(n)]), ([n - 1, k + 1], [1.0])]

for name, cb, ipv, theta in [("const", cb_const, [5], None),
                             ("mixed3", cb_mixed, [[[3, 0], 1.0]], [2.0]),
                             ("pervertex", cb_pervertex, [[[6, 0], 1.0]], [2.0])]:
    try:
        report(name, cb, ipv, theta)
    except Exception as e:
        print(f"[{name}] build/roundtrip error {type(e).__name__}: {e}")
