"""reviewM m1: does a cache HIT (master's hit path, serialize->JSON->from_serialized
-> super().__init__) equal a fresh build?  Run with HOME=scratch.
Uses master's graph_cache=True to produce a real hit (cache dir under scratch HOME)."""
import os, sys, numpy as np
assert os.environ["HOME"].startswith("/private/tmp/claude-501/"), os.environ["HOME"]
import phasic
from phasic import Graph

def edges_of(g):
    out = []
    for v in [g.starting_vertex()] + list(g.vertices()):
        for e in v.edges():
            out.append((tuple(v.state()), tuple(e.to().state()), float(e.weight()),
                        tuple(e.coefficients()) if hasattr(e, "coefficients") else None))
    return out

def summary(g):
    d = {}
    for name in ("vertices_length", "param_length", "parameterized"):
        try:
            d[name] = getattr(g, name)()
        except Exception as ex:
            d[name] = f"ERR {type(ex).__name__}: {ex}"
    try:
        d["expectation"] = float(g.expectation())
    except Exception as ex:
        d["expectation"] = f"ERR {type(ex).__name__}: {str(ex)[:80]}"
    d["edges"] = edges_of(g)
    return d

def compare(label, make):
    phasic.GraphCache().clear() if hasattr(phasic, "GraphCache") else None
    try:
        fresh = make(False)
    except Exception as ex:
        print(f"== {label}: fresh build raises {type(ex).__name__}: {str(ex)[:100]}"); return
    make(True)            # writes the entry
    hit = make(True)      # hit (master logs "Loaded graph from cache")
    a, b = summary(fresh), summary(hit)
    same = a == b
    print(f"== {label}: identical={same}")
    if not same:
        for k in a:
            if a[k] != b[k]:
                print(f"   {k}: fresh={a[k] if k!='edges' else a[k][:6]}")
                print(f"   {k}: hit  ={b[k] if k!='edges' else b[k][:6]}")

# A: constant rates, explicit theta_dim=3
def cbA(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], n * (n - 1) / 2]]
compare("A const graph theta_dim=3",
        lambda c: Graph(cbA, ipv=[[[4], 1.0]], theta_dim=3, graph_cache=c))

# B: 3-tuple parameterized with nonzero vestigial weight
def cbB(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], 7.0, [n * (n - 1) / 2]]]
compare("B 3-tuple w=7", lambda c: Graph(cbB, ipv=[[[4], 1.0]], graph_cache=c))

# C: duplicate child edges
def cbC(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], 1.0], [[n - 1], 2.0]]
compare("C duplicate edges", lambda c: Graph(cbC, ipv=[[[4], 1.0]], graph_cache=c))

# D: theta_dim=2, 3 coefficients (aux coefficient)
def cbD(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2, 1.0, 5.0]]]
compare("D theta_dim=2 with 3 coeffs",
        lambda c: Graph(cbD, ipv=[[[4], 1.0]], theta_dim=2, graph_cache=c))

# E: parameterized with an all-zero coefficient edge
def cbE(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2, 0.0]], [[0], [0.0, 0.0]]] if n == 3 else [[[n - 1], [n * (n - 1) / 2, 0.0]]]
compare("E zero-coefficient edge", lambda c: Graph(cbE, ipv=[[[4], 1.0]], graph_cache=c))

# F: ipv with two starting states, one prob 0
def cbF(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], float(n)]]
compare("F ipv with zero-prob start", lambda c: Graph(cbF, ipv=[[[4], 1.0], [[3], 0.0]], graph_cache=c))

# G: parameterized graph where update_weights was never called: expectation default
def cbG(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [float(n)]]]
compare("G param 2-tuple", lambda c: Graph(cbG, ipv=[[[4], 1.0]], graph_cache=c))
