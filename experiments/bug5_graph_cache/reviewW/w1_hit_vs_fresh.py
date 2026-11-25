"""reviewW w1: does a master cache HIT reproduce a fresh build? (HOME redirected by caller)

Run: HOME=<scratch> PHASIC_CACHE_DIR unset; graph_cache=True on master code.
Compares param_length/parameterized/edge lists/expectation of a fresh
(uncached) build vs the hit returned on the second cached call.
"""
import os, sys, shutil
from pathlib import Path
assert "phasic_cache" not in os.environ.get("HOME", "") and "/scratchpad/" in os.environ["HOME"], os.environ["HOME"]
os.environ.pop("PHASIC_CACHE_DIR", None)
import numpy as np
import phasic
from phasic import Graph

def edges_of(g):
    out = []
    for v in g.vertices():
        es = []
        for e in v.edges():
            es.append((tuple(e.to().state()), round(e.weight(), 12), e.coefficients_length()))
        for e in v.parameterized_edges():
            es.append(("P", tuple(e.to().state()), tuple(e.edge_state(e.coefficients_length()))))
        out.append((tuple(v.state()), es))
    return out

def describe(g):
    d = dict(param_length=g.param_length(), parameterized=g.parameterized(),
             n=g.vertices_length(), dyn=g.dyn_ordering,
             is_discrete=getattr(g, "is_discrete", None), was_dph=g.get_was_dph())
    return d

def run(name, cb, ipv, theta=None, **kw):
    try:
        _run(name, cb, ipv, theta, **kw)
    except Exception as ex:
        print(f"--- {name}: fresh/cached build raised {type(ex).__name__}: {str(ex)[:120]}")

def _run(name, cb, ipv, theta=None, **kw):
    hits = {"n": 0}
    def spy(state, **k):
        hits["n"] += 1
        return cb(state, **k)
    spy.__code__ = cb.__code__.replace() if False else spy.__code__
    fresh = Graph(cb, ipv, graph_cache=False, **kw)
    Graph(cb, ipv, graph_cache=True, **kw)
    hit = Graph(cb, ipv, graph_cache=True, **kw)
    df, dh = describe(fresh), describe(hit)
    ef, eh = edges_of(fresh), edges_of(hit)
    res = {}
    if theta is not None:
        for g in (fresh, hit):
            try:
                g.update_weights(theta)
                res.setdefault("exp", []).append(round(float(g.expectation()), 12))
            except Exception as ex:
                res.setdefault("exp", []).append(f"{type(ex).__name__}: {str(ex)[:90]}")
    print(f"--- {name}")
    print("  fresh:", df)
    print("  hit  :", dh)
    print("  edges equal:", ef == eh)
    if ef != eh:
        for a, b in zip(ef, eh):
            if a != b:
                print("   first diff fresh:", a); print("   first diff hit  :", b); break
    if res:
        print("  update_weights+expectation fresh/hit:", res["exp"])

# (a) constant-only callback with theta_dim=2
def cb_a(state, **kw):
    n = state[0]
    return [([n - 1], n * (n - 1) / 2.0)] if n > 1 else []
run("a constant edges, no theta_dim", cb_a, [4], theta=[1.0])

# (b) parameterized 3 coeffs, theta_dim=2 (decoupled)
def cb_b(state, **kw):
    n = state[0]
    return [([n - 1], [n * (n - 1) / 2.0, 0.0, 1.0])] if n > 1 else []
run("b 3 coeffs, theta_dim=2", cb_b, [4], theta=[1.0, 2.0], theta_dim=2)

# (c) ordinary parameterized
def cb_c(state, **kw):
    n = state[0]
    return [([n - 1], [n * (n - 1) / 2.0])] if n > 1 else []
run("c parameterized", cb_c, [4], theta=[2.0])

# (d) parallel edges to the same target, parameterized
def cb_d(state, **kw):
    n = state[0]
    return [([n - 1], [1.0, 0.0]), ([n - 1], [0.0, 1.0])] if n > 1 else []
run("d parallel parameterized edges", cb_d, [4], theta=[1.0, 2.0])

# (e) parallel constant edges
def cb_e(state, **kw):
    n = state[0]
    return [([n - 1], 1.0), ([n - 1], 2.0)] if n > 1 else []
run("e parallel constant edges", cb_e, [4], theta=None)

# (f) multi-start ipv with probabilities, parameterized
def cb_f(state, **kw):
    n = state[0]
    return [([n - 1], [float(n)])] if n > 1 else []
run("f multi-start ipv", cb_f, [[[4], 0.25], [[3], 0.5]], theta=[1.5])

# (g) all-zero coefficient edge
def cb_g(state, **kw):
    n = state[0]
    return [([n - 1], [float(n), 0.0]), ([0 if n > 1 else 9], [0.0, 0.0])] if n > 1 else []
run("g zero-coefficient edge", cb_g, [4], theta=[1.0, 1.0])
