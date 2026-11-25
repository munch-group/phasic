# reviewAD p2: does serialize -> JSON -> from_serialized (the hit path) preserve per-vertex edge
# order and give bit-identical results for a callback mixing constant and parameterized edges?
import json, numpy as np
import phasic
from phasic import Graph
from phasic.graph_cache import _serialize_value, _deserialize_value

def cb(state):
    n = state[0]
    if n <= 1:
        return []
    out = []
    # constant edge FIRST, then parameterized ones (mixed per vertex)
    out.append([[n - 1], 0.0, []])           # constant edge? (empty coefficient list)
    if n >= 3:
        out.append([[n - 2], 0.0, [0.3 * n, 0.1]])
    return out

def edges_of(g):
    rows = []
    for v in g.vertices():
        rows.append((tuple(v.state()), [(tuple(e.to().state()), e.weight(), tuple(e.edge_state(e.coefficients_length())) if hasattr(e,'edge_state') else None) for e in v.edges()]))
    return rows

for variant in ("mixed",):
    try:
        g = Graph(cb, ipv=[8], graph_cache=False)
    except Exception as ex:
        print("build failed:", type(ex).__name__, ex); raise
    print("parameterized:", g.parameterized(), "param_length:", g.param_length())
    g.update_weights([1.0, 2.0])
    d = g.serialize()
    enc = json.dumps(_serialize_value(d), separators=(",", ":"))
    h = Graph.from_serialized(_deserialize_value(json.loads(enc)))
    h = Graph(h)
    h.update_weights([1.0, 2.0])
    e1, e2 = edges_of(g), edges_of(h)
    same_order = e1 == e2
    print("edge lists identical (order, weights):", same_order)
    if not same_order:
        for a, b in zip(e1, e2):
            if a != b:
                print(" fresh:", a); print(" hit  :", b); break
    x1, x2 = g.expectation(), h.expectation()
    print("expectation fresh/hit:", repr(x1), repr(x2), "bit-equal:", x1 == x2)
    m1, m2 = g.moments(3), h.moments(3)
    print("moments bit-equal:", list(m1) == list(m2), list(m1), list(m2))
