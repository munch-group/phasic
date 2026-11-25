"""reviewJ p1: on MASTER code, compare a graph-cache HIT with a fresh build,
structurally (vertex states, every edge's target/weight/coefficients,
param_length, parameterized flag) for graph shapes ordinary callbacks make.
Run with HOME redirected to a scratch dir (the real cache is never touched).
"""
import os, sys, shutil, json
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import numpy as np
import phasic
from phasic import Graph

calls = {"n": 0}

def snapshot(g):
    out = {"n": g.vertices_length(), "state_length": g.state_length(),
           "param_length": int(g.param_length()), "parameterized": bool(g.parameterized()),
           "dyn": bool(g.dyn_ordering)}
    verts = []
    for i in range(g.vertices_length()):
        v = g.vertex_at(i)
        es = []
        for e in v.edges():
            cl = e.coefficients_length()
            es.append(("E", int(e.to().index()), float(e.weight()), int(cl)))
        for e in v.parameterized_edges():
            cl = e.coefficients_length()
            es.append(("P", int(e.to().index()), float(e.weight()), int(cl),
                       tuple(float(x) for x in e.edge_state(cl))))
        verts.append((tuple(int(x) for x in v.state()), sorted(es)))
    out["verts"] = verts
    return out

def diff(a, b):
    d = []
    for k in a:
        if k == "verts":
            continue
        if a[k] != b[k]:
            d.append((k, a[k], b[k]))
    if len(a["verts"]) != len(b["verts"]):
        d.append(("nverts", len(a["verts"]), len(b["verts"])))
    else:
        for i, (x, y) in enumerate(zip(a["verts"], b["verts"])):
            if x != y:
                d.append(("vertex", i, x, y))
                if len(d) > 4:
                    break
    return d

def case(name, cb, ipv, **kw):
    calls["n"] = 0
    fresh = Graph(cb, ipv, graph_cache=False, **kw)
    Graph(cb, ipv, graph_cache=True, **kw)  # writes
    n_before = calls["n"]
    hit = Graph(cb, ipv, graph_cache=True, **kw)
    was_hit = calls["n"] == n_before
    a, b = snapshot(fresh), snapshot(hit)
    d = diff(a, b)
    extra = ""
    try:
        if a["parameterized"]:
            th = [1.5] * max(1, a["param_length"])
            fresh.update_weights(th); hit.update_weights(th)
        ef, eh = fresh.expectation(), hit.expectation()
        extra = f"E fresh={ef:.12g} hit={eh:.12g}"
    except Exception as e:
        extra = f"E error {type(e).__name__}: {e}"
    print(f"[{name}] hit={was_hit} diffs={d if d else 'none'} {extra}")

# A constant coalescent
def coal(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2)]
case("A const coalescent", coal, [([5], 1.0)])

# B parameterized, coeff len 1
def coalp(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2])]
case("B param len1", coalp, [([5], 1.0)])

# C coeff len 3, theta_dim 2
def coal3(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2, 0.0, 7.0])]
case("C coeff3 theta_dim2", coal3, [([5], 1.0)], theta_dim=2)

# D duplicate parameterized edges to the same target
def dupp(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [1.0, 0.0]), ([n - 1], [0.0, 2.0])]
case("D dup param edges", dupp, [([4], 1.0)])

# E duplicate constant edges
def dupc(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], 1.0), ([n - 1], 2.0)]
case("E dup const edges", dupc, [([4], 1.0)])

# F defective ipv, two start states
case("F defective ipv", coal, [([5], 0.3), ([3], 0.5)])

# G zero-weight constant edge alongside a real one
def zero(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    out = [([n - 1], 1.0)]
    if n > 2:
        out.append(([n - 2], 0.0))
    return out
case("G zero weight edge", zero, [([5], 1.0)])

# H parameterized with an all-zero coefficient edge
def zc(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    out = [([n - 1], [1.0, 1.0])]
    if n > 2:
        out.append(([n - 2], [0.0, 0.0]))
    return out
case("H all-zero coeff edge", zc, [([5], 1.0)])

# I edge back to the starting state vector? (state [5] revisited from [4]: a cycle)
def cyc(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    out = [([n - 1], [1.0, 0.0])]
    if n < 5:
        out.append(([n + 1], [0.0, 1.0]))
    return out
case("I cyclic param", cyc, [([5], 1.0)])

# J parameterized coalescent where start state also reached via an edge
case("J ipv split param", coalp, [([5], 0.5), ([4], 0.5)])

# K constant callback with theta_dim given
# L start-only graph (absorbing immediately)
def dead(state, **kw):
    calls["n"] += 1
    return []
try:
    case("L start-only", dead, [([5], 1.0)])
except Exception as e:
    print("[L start-only] error", type(e).__name__, e)
# N 3-tuple format
def three(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], 9.0, [n * 1.0, 2.0])]
case("N 3-tuple", three, [([5], 1.0)])
# O coeff len 1 but theta_dim 3
