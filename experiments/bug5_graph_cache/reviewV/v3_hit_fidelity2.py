"""reviewV v3: master hit path vs fresh build for (a) theta_dim larger
than the coefficient length, (b) start edges to several vertices with
parameterized edges, (c) coefficient vectors with zeros.  Scratch HOME."""
import os
assert os.environ["HOME"].startswith("/private/tmp/"), os.environ["HOME"]
import numpy as np
import phasic
from phasic import Graph

def cb2(state, **kw):
    k = state[0]
    if k <= 1:
        return []
    return [([k - 1], [k * (k - 1) / 2, 0.0])]

def cbz(state, **kw):
    k, m = state
    out = []
    if k > 1:
        out.append(([k - 1, m], [0.0, k * 1.0, 0.0]))
    if m < 2 and k > 1:
        out.append(([k, m + 1], [1.0, 0.0, 0.5]))
    return out

def dump(g):
    out = []
    for v in g.vertices():
        for e in v.edges():
            out.append((tuple(v.state()), tuple(e.to().state()), e.weight(), e.coefficients_length()))
    return out

cases = [
    ("theta_dim=3 > coeff 2", phasic.with_ipv([5])(cb2), dict(theta_dim=3), [2.0, 0.0, 0.0]),
    ("multi-start ipv", phasic.with_ipv([[[4], 0.5], [[3], 0.5]])(cb2), {}, [2.0, 0.0]),
    ("zeros in coeffs", phasic.with_ipv([4, 0])(cbz), {}, [1.0, 2.0, 3.0]),
]
for name, w, kw, theta in cases:
    try:
        g = Graph(w, graph_cache=False, **kw)
        Graph(w, graph_cache=True, **kw)
        h = Graph(w, graph_cache=True, **kw)
    except Exception as ex:
        print(f"== {name}: build ERR {type(ex).__name__}: {str(ex)[:200]}")
        continue
    print(f"== {name}")
    print("  param_length", g.param_length(), h.param_length(), " parameterized", g.parameterized(), h.parameterized())
    print("  edges equal:", dump(g) == dump(h))
    for gg in (g, h):
        try:
            gg.update_weights(theta)
        except Exception as ex:
            print("  update_weights ERR", type(ex).__name__, str(ex)[:150])
    try:
        print("  E fresh/hit", g.expectation(), h.expectation(), " var", g.variance(), h.variance())
    except Exception as ex:
        print("  moments ERR", type(ex).__name__, str(ex)[:150])
