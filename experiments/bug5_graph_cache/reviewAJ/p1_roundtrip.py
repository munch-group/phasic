"""reviewAJ p1: does a hit (from_serialized + copy ctor, the hit path) equal a fresh build?
Run with HOME pointing at a scratch dir. No cache files are written (graph_cache=False)."""
import os, sys
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import numpy as np
import phasic
from phasic import Graph, with_ipv

def hit_like(g):
    cached = Graph.from_serialized(g.serialize())
    return Graph(cached)  # same super().__init__(cached_graph) as the hit path

def edges(g):
    out = []
    for v in g:
        for e in v.edges():
            out.append((tuple(v.state()), tuple(e.to().state()), round(e.weight(), 12)))
    return sorted(out)

# (a) parameterized 2-tuple form
def cb_param(state, n=4.0):
    k = state[0]
    if k <= 1:
        return []
    return [([k - 1], [k * (k - 1) / 2])]

# (b) 3-tuple form with vestigial nonzero weight
def cb_three(state, n=4.0):
    k = state[0]
    if k <= 1:
        return []
    return [([k - 1], 7.0, [k * (k - 1) / 2])]

# (c) parallel edges to the same target from one callback call
def cb_parallel(state):
    k = state[0]
    if k <= 1:
        return []
    return [([k - 1], 1.0), ([k - 1], 2.0)]

# (d) constant
def cb_const(state):
    k = state[0]
    if k <= 1:
        return []
    return [([k - 1], float(k))]

for name, cb, kw in [("param", cb_param, {}), ("three", cb_three, {}),
                     ("parallel", cb_parallel, {}), ("const", cb_const, {})]:
    try:
        g = Graph(cb, ipv=[[[4], 1.0]], graph_cache=False, **kw)
        h = hit_like(g)
        same_edges = edges(g) == edges(h)
        print(name, "vertices", g.vertices_length(), h.vertices_length(), "edges equal:", same_edges)
        if not same_edges:
            print("  fresh:", edges(g))
            print("  hit  :", edges(h))
        for label, gg in (("fresh", g), ("hit", h)):
            try:
                print("  ", label, "expectation", gg.expectation(), "param", gg.parameterized(), gg.param_length())
            except Exception as e:
                print("  ", label, "expectation raised", type(e).__name__, e)
        if g.parameterized():
            g.update_weights([2.0]); h.update_weights([2.0])
            print("   after update_weights([2]):", g.expectation(), h.expectation(), edges(g) == edges(h))
    except Exception as e:
        print(name, "raised", type(e).__name__, e)
