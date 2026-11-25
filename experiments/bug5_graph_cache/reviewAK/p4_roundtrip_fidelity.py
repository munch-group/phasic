# reviewAK p4: does a JSON round trip (the cache's load path) reproduce the fresh graph exactly,
# including per-vertex edge ORDER, for constant, parameterized and mixed callbacks?
import json, numpy as np
import phasic
from phasic import Graph, with_ipv
from phasic.graph_cache import _serialize_value, _deserialize_value

def edges_of(g):
    out = []
    for i in range(g.vertices_length()):
        v = g.vertex_at(i)
        es = [(tuple(e.to().state()), float(e.weight()), e.coefficients_length()) for e in v.edges()]
        es += [("P", tuple(e.to().state()), tuple(e.edge_state(e.coefficients_length()))) for e in v.parameterized_edges()]
        out.append((tuple(v.state()), es))
    return out

def rt(g):
    d = g.serialize()
    s = json.dumps(_serialize_value(d), separators=(",", ":"))
    return Graph.from_serialized(_deserialize_value(json.loads(s)))

def cb_const(state):
    n, m = state
    out = []
    if n > 1: out.append([[n - 1, m + 1], n * (n - 1) / 2 + 0.1 * m])
    if m > 0: out.append([[n, m - 1], 0.37 * m])
    if n > 0 and m < 3: out.append([[n - 1, m], 0.11 * n])
    return out

def cb_param(state):
    n, m = state
    out = []
    if n > 1: out.append([[n - 1, m + 1], [n * (n - 1) / 2, 0.1 * m]])
    if m > 0: out.append([[n, m - 1], [0.0, 0.37 * m]])
    if n > 0 and m < 3: out.append([[n - 1, m], [0.11 * n, 0.0]])
    return out

def main():
 for name, cb in [("const", cb_const), ("param", cb_param)]:
     g = Graph(with_ipv([[[6, 0], 0.7], [[5, 1], 0.3]])(cb))
     g2 = rt(g)
     same_edges = edges_of(g) == edges_of(g2)
     d1, d2 = g.serialize(), g2.serialize()
     same_ser = all((np.array_equal(np.asarray(d1[k]), np.asarray(d2[k])) and np.asarray(d1[k]).dtype == np.asarray(d2[k]).dtype) if isinstance(d1[k], np.ndarray) else d1[k] == d2[k] for k in d1)
     diffkeys = [k for k in d1 if not (np.array_equal(np.asarray(d1[k]), np.asarray(d2[k])) if isinstance(d1[k], np.ndarray) else d1[k] == d2[k])]
     if name == "param":
         g.update_weights([1.3, 0.7]); g2.update_weights([1.3, 0.7])
     e1, e2 = g.expectation(), g2.expectation()
     p1 = np.asarray(g.pdf(np.array([0.3, 1.0, 2.5]))); p2 = np.asarray(g2.pdf(np.array([0.3, 1.0, 2.5])))
     print(name, "V", g.vertices_length(), g2.vertices_length(), "edge order identical:", same_edges,
           "serialize identical:", same_ser, diffkeys, "E bitwise:", e1 == e2, "pdf bitwise:", bool(np.array_equal(p1, p2)),
           "param_length", g.param_length(), g2.param_length(), "parameterized", g.parameterized(), g2.parameterized())
     if not same_edges:
         for (s1, a), (s2, b) in zip(edges_of(g), edges_of(g2)):
             if (s1, a) != (s2, b):
                 print("  first diff:", s1, a, "|", s2, b); break

if __name__ == "__main__":
    main()
