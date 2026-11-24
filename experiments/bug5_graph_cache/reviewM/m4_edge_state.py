"""reviewM m4: per-edge C state of a cache HIT vs a fresh build (master hit path).
Compares coefficients_length per edge, parameterized(), param_length(), and the
behaviour of follow-up operations users perform on a built graph:
update_weights, adding a parameterized / constant edge, discretize, serialize.
HOME=scratch."""
import os, numpy as np
assert os.environ["HOME"].startswith("/private/tmp/claude-501/"), os.environ["HOME"]
import phasic
from phasic import Graph

def per_edge(g):
    return [(tuple(v.state()), tuple(e.to().state()), e.coefficients_length(), float(e.weight()))
            for v in [g.starting_vertex()] + list(g.vertices()) for e in v.edges()]

def tryit(f):
    try:
        r = f()
        return r if not isinstance(r, float) else round(r, 12)
    except Exception as ex:
        return f"ERR {type(ex).__name__}: {str(ex)[:90]}"

def follow_ups(g):
    out = {}
    out["parameterized"] = tryit(g.parameterized)
    out["param_length"] = tryit(g.param_length)
    out["edges"] = per_edge(g)
    out["update_weights([2.0])+E"] = tryit(lambda: (g.update_weights([2.0]), float(g.expectation()))[1])
    def add_param():
        v = [v for v in g.vertices() if v.state()[0] == 2][0]
        w = g.find_or_create_vertex([7])
        v.add_edge_parameterized(w, 0.0, [1.0])
        return "ok"
    out["add_edge_parameterized"] = tryit(add_param)
    def add_const():
        v = [v for v in g.vertices() if v.state()[0] == 3][0]
        w = g.find_or_create_vertex([8])
        v.add_edge(w, 0.5)
        return "ok"
    out["add_edge const"] = tryit(add_const)
    return out

def cb_const(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], n * (n - 1) / 2]]

def cb_param(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2]]]

for label, cb in (("constant", cb_const), ("parameterized", cb_param)):
    phasic.GraphCache().clear()
    fresh = Graph(cb, ipv=[[[4], 1.0]], graph_cache=False)
    Graph(cb, ipv=[[[4], 1.0]], graph_cache=True)
    hit = Graph(cb, ipv=[[[4], 1.0]], graph_cache=True)
    a, b = follow_ups(fresh), follow_ups(hit)
    print(f"== {label}: identical={a == b}")
    for k in a:
        mark = "" if a[k] == b[k] else "   <-- DIFFERS"
        print(f"  {k}: fresh={a[k]}{mark}")
        if a[k] != b[k]:
            print(f"  {k}: hit  ={b[k]}")
