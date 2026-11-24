"""reviewJ p2: on MASTER code, does a graph-cache HIT keep the C-side locks of a
fresh build (param_length lock, edge-mode lock)? Probed by post-construction
edits that a fresh build accepts or rejects. HOME redirected to scratch.
"""
import os
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import numpy as np
from phasic import Graph

calls = {"n": 0}

def coalp(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2, 1.0])]

def coalc(state, **kw):
    calls["n"] += 1
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2)]

def attempt(g, what):
    try:
        if what == "param3":
            v = g.find_or_create_vertex([9]); g.vertex_at(1).add_edge(v, [1.0, 2.0, 3.0])
        elif what == "param2":
            v = g.find_or_create_vertex([9]); g.vertex_at(1).add_edge(v, [1.0, 2.0])
        elif what == "const":
            v = g.find_or_create_vertex([9]); g.vertex_at(1).add_edge(v, 2.0)
        elif what == "set_param_length3":
            g.set_param_length(3)
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:90]}"

def pair(name, cb, ipv, **kw):
    out = {}
    for what in ["param3", "param2", "const", "set_param_length3"]:
        fresh = Graph(cb, ipv, graph_cache=False, **kw)
        Graph(cb, ipv, graph_cache=True, **kw)
        n0 = calls["n"]
        hit = Graph(cb, ipv, graph_cache=True, **kw)
        assert calls["n"] == n0, "not a hit"
        a, b = attempt(fresh, what), attempt(hit, what)
        flag = "SAME" if a == b else "DIFFERS"
        print(f"[{name}] {what}: fresh -> {a} | hit -> {b}  [{flag}]")

pair("param len2", coalp, [([5], 1.0)])
pair("param len2 theta_dim2", coalp, [([5], 1.0)], theta_dim=2)
pair("constant", coalc, [([5], 1.0)])
