"""reviewAB p1: does a master-path cache HIT behave like a fresh build?

Run with HOME and PHASIC_CACHE_DIR redirected into the scratchpad.  Uses
master's graph_cache=True path (the plan keeps the same restore sequence)
and compares against graph_cache=False builds.
"""
import os, sys, shutil
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import numpy as np
import phasic
from phasic import Graph

def cb_theta_lt_coeffs(state, **kw):
    # 2 coefficients, theta_dim=1 -> second coefficient is auxiliary
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], [n * (n - 1) / 2, 7.0]]]

def cb_dup_param(state, **kw):
    # two distinct events lead to the same child state (duplicate edges)
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], [1.0, 0.0]], [[n - 1], [0.0, float(n)]]]

def cb_dup_const(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], 1.0], [[n - 1], float(n)]]

def describe(g, theta):
    out = {}
    out["V"] = g.vertices_length()
    out["param_length"] = g.param_length()
    out["parameterized"] = g.parameterized()
    out["edges"] = [len(g.vertex_at(i).edges()) for i in range(g.vertices_length())]
    try:
        if theta is not None:
            g.update_weights(theta)
        out["E"] = float(g.expectation())
        out["pdf"] = [float(x) for x in g.pdf([0.3, 1.0])]
    except Exception as e:
        out["err"] = repr(e)[:200]
    return out

cases = [
    ("theta_lt_coeffs", cb_theta_lt_coeffs, dict(theta_dim=1), [2.0]),
    ("dup_param", cb_dup_param, {}, [1.0, 3.0]),
    ("dup_const", cb_dup_const, {}, None),
]
for name, cb, kw, theta in cases:
    fresh = Graph(cb, ipv=[5], graph_cache=False, **kw)
    _ = Graph(cb, ipv=[5], graph_cache=True, **kw)  # writes entry
    hit = Graph(cb, ipv=[5], graph_cache=True, **kw)  # should hit
    df, dh = describe(fresh, theta), describe(hit, theta)
    print(name, "SAME" if df == dh else "DIFF")
    if df != dh:
        print("  fresh", df)
        print("  hit  ", dh)
