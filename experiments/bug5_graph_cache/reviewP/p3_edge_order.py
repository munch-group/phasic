"""Probe: does a graph rebuilt by from_serialized(serialize()) (the cache-hit
path) keep each vertex's out-edge ORDER of the fresh build?  serialize()
splits edges into param_edges / constant_edges, so a test comparing
serialize() arrays cannot see an order change.  Order matters for sample()
streams and for floating-point summation order."""
import random
import numpy as np
import phasic
from phasic import Graph

def cb(state):
    n = state[0]
    if n <= 1:
        return []
    out = []
    # interleave: parameterized, constant, parameterized
    out.append(([n - 1], 0.0, [n * (n - 1) / 2]))
    if n >= 3:
        out.append(([1], 0.0, [0.3 * n]))       # parameterized edge
        out.append(([n - 2], 0.0, [0.7]))       # parameterized edge
    return out
# (mixing constant and parameterized edges is rejected by the builder,
#  measured; so only same-kind edges are compared)

g = Graph(cb, ipv=[[[6],0.5],[[5],0.5]], graph_cache=False)
h = Graph(Graph.from_serialized(g.serialize()))

def edge_lists(gr):
    res = []
    for i in range(gr.vertices_length()):
        v = gr.vertex_at(i)
        es = []
        for e in v.edges():
            try:
                c = list(np.asarray(e.edge_state(), dtype=float)) if hasattr(e, "edge_state") else None
            except Exception:
                c = None
            es.append((list(e.to().state()), round(float(e.weight()), 6), c))
        res.append((list(v.state()), es))
    return res

eg, eh = edge_lists(g), edge_lists(h)
print("vertices", g.vertices_length(), h.vertices_length())
print("same vertex order:", [s for s, _ in eg] == [s for s, _ in eh])
diff = [(s, a, b) for (s, a), (_, b) in zip(eg, eh) if a != b]
print("vertices whose out-edge list (order/content) differs:", len(diff))
for s, a, b in diff[:3]:
    print("  state", s)
    print("    fresh:", a)
    print("    hit  :", b)
g.update_weights([1.0]); h.update_weights([1.0])
print("expectation fresh/hit:", g.expectation(), h.expectation())
for gr in (g, h):
    random.seed(1); np.random.seed(1)
print("serialize arrays equal:",
      all(np.array_equal(np.asarray(g.serialize()[k]), np.asarray(h.serialize()[k]))
          for k in ("states", "param_edges", "constant_edges", "start_edges")))
random.seed(1); np.random.seed(1)
sg = g.sample(5)
random.seed(1); np.random.seed(1)
sh = h.sample(5)
print("sample(5) fresh:", np.round(sg, 6))
print("sample(5) hit  :", np.round(sh, 6))
