"""reviewV v1: does a graph rebuilt the way a cache hit rebuilds it
(serialize -> JSON-able -> from_serialized -> Graph(super().__init__(g)))
match a fresh build on state the plan's test 3 does not compare?
Uses master code; no cache file is written (graph_cache=False)."""
import os, sys, json
assert os.environ["HOME"].startswith("/private/tmp/"), os.environ["HOME"]
import numpy as np
import phasic
from phasic import Graph
from phasic.graph_cache import _serialize_value

def cb(state, n=4):
    if len(state) == 0:
        return [([n], 1.0)]
    k = state[0]
    if k <= 1:
        return []
    return [([k - 1], [k * (k - 1) / 2, 0.0])]

def cbc(state, n=4):  # constant edges
    if len(state) == 0:
        return [([n], 1.0)]
    k = state[0]
    if k <= 1:
        return []
    return [([k - 1], k * (k - 1) / 2)]

def rebuild(g):
    data = g.serialize()
    enc = json.loads(json.dumps(_serialize_value(data)))
    from phasic.graph_cache import GraphCache
    dec = phasic.graph_cache._deserialize_value(enc) if hasattr(phasic.graph_cache, "_deserialize_value") else enc
    g2 = Graph.from_serialized(dec)
    h = Graph.__new__(Graph)
    phasic._Graph.__init__(h, g2)
    return h, g2

def edge_dump(g):
    out = []
    for v in g.vertices():
        for e in v.edges():
            out.append((tuple(v.state()), tuple(e.to().state()), e.weight(), e.coefficients_length()))
    return out

for name, f in [("param", cb), ("const", cbc)]:
    w = phasic.with_ipv([4])(f)
    g = Graph(w, graph_cache=False)
    Graph(w, graph_cache=True)   # writes master entry in scratch HOME
    calls = []
    h = Graph(w, graph_cache=True)   # master hit path
    print("hit? callback attr", h._callback is w)
    print(f"== {name}")
    for attr in ["vertices_length", "param_length", "parameterized", "get_was_dph", "state_length"]:
        try:
            print(f"  {attr}: fresh={getattr(g, attr)()} hit={getattr(h, attr)()}")
        except Exception as ex:
            print(f"  {attr}: ERR {type(ex).__name__}: {ex}")
    ef, eh = edge_dump(g), edge_dump(h)
    print("  edges equal:", ef == eh)
    if ef != eh:
        print("  fresh:", ef[:4]); print("  hit:  ", eh[:4])
    try:
        print("  expectation fresh/hit (no theta):", g.expectation(), h.expectation())
    except Exception as ex:
        print("  expectation ERR", type(ex).__name__, ex)
    if name == "param":
        for gg in (g, h):
            gg.update_weights([2.0, 0.0])
        print("  expectation after theta:", g.expectation(), h.expectation())
    # add a constant edge after build (edge-mode lock)
    for lab, gg in (("fresh", g), ("hit", h)):
        try:
            v = gg.find_or_create_vertex([7])
            v.add_edge(gg.find_or_create_vertex([1]), 1.0)
            print(f"  add const edge on {lab}: ok")
        except Exception as ex:
            print(f"  add const edge on {lab}: {type(ex).__name__}: {str(ex)[:120]}")
