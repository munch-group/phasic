"""reviewL l1: does a (master) cache hit equal a fresh build beyond serialize()?

Run with HOME and PHASIC_CACHE_DIR pointing at a scratch dir.
Compares param_length(), parameterized(), per-edge weights/coefficients,
expectation() before and after update_weights, for several callback kinds.
"""
import os, sys, json
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import numpy as np
import phasic
from phasic import Graph, with_ipv


def edges_of(g):
    out = []
    for i in range(g.vertices_length()):
        v = g.vertex_at(i)
        row = []
        for e in v.edges():
            row.append(("E", e.to().index(), float(e.weight()), e.coefficients_length()))
        for e in v.parameterized_edges():
            cl = e.coefficients_length()
            row.append(("P", e.to().index(), float(e.weight()), cl,
                        tuple(e.edge_state(cl)) if cl else ()))
        out.append(row)
    return out


def describe(g, theta):
    d = {
        "n": g.vertices_length(),
        "param_length": int(g.param_length()),
        "parameterized": bool(g.parameterized()),
        "edges": edges_of(g),
    }
    try:
        d["E_before"] = float(g.expectation())
    except Exception as e:
        d["E_before"] = "ERR " + type(e).__name__ + ": " + str(e)[:80]
    if theta is not None:
        try:
            g.update_weights(theta)
            d["E_after"] = float(g.expectation())
            d["edges_after"] = edges_of(g)
        except Exception as e:
            d["E_after"] = "ERR " + type(e).__name__ + ": " + str(e)[:80]
    return d


N = 4

def cb_const(state, **kw):
    n = state[0]
    return [([n - 1], n * (n - 1) / 2)] if n > 1 else []

def cb_param(state, **kw):
    n = state[0]
    return [([n - 1], [n * (n - 1) / 2, 0.0])] if n > 1 else []

def cb_3tuple(state, **kw):
    n = state[0]
    return [([n - 1], 7.0, [n * (n - 1) / 2])] if n > 1 else []

cases = [
    ("const", cb_const, dict(), None),
    ("const_theta_dim3", cb_const, dict(theta_dim=3), None),
    ("param", cb_param, dict(), [2.0, 1.0]),
    ("param_theta_dim1", cb_param, dict(theta_dim=1), [2.0]),
    ("3tuple_w7", cb_3tuple, dict(), [2.0]),
]

res = {}
for name, cb, kw, theta in cases:
    phasic.GraphCache().clear()
    try:
        fresh = Graph(cb, ipv=[N], graph_cache=False, **kw)
    except Exception as e:
        print(name, "fresh build failed:", type(e).__name__, e)
        continue
    Graph(cb, ipv=[N], graph_cache=True, **kw)  # write
    hit = Graph(cb, ipv=[N], graph_cache=True, **kw)  # master hit
    a, b = describe(fresh, theta), describe(hit, theta)
    diffs = [k for k in a if a[k] != b[k]]
    print(f"{name}: diffs={diffs}")
    for k in diffs:
        print(f"   fresh {k} = {a[k]}")
        print(f"   hit   {k} = {b[k]}")
phasic.GraphCache().clear()
