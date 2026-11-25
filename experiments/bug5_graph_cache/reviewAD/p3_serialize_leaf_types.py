# reviewAD p3: leaf types in Graph.serialize() output (what a strict "plain JSON + ndarray"
# _serialize_value must accept) for a linear callback graph and a formula-mode graph.
import numpy as np
from collections import Counter
from phasic import Graph

def walk(x, path, out):
    t = type(x)
    if t is dict:
        for k, v in x.items():
            if type(k) is not str: out[("KEY", type(k).__name__)] += 1
            walk(v, f"{path}[{k!r}]", out)
    elif t in (list, tuple):
        if t is tuple: out[("tuple", path)] += 1
        for i, v in enumerate(x): walk(v, f"{path}[{i}]", out)
    elif t is np.ndarray:
        out[("ndarray", str(x.dtype))] += 1
    else:
        out[(t.__module__ + "." + t.__name__, path if t.__module__ != "builtins" else "")] += 1

def cb(state, rate=1.0):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2 * rate, 0.0]]]

g = Graph(cb, ipv=[6], graph_cache=False)
c = Counter(); walk(g.serialize(), "", c)
print("LINEAR callback graph leaf types:"); [print("  ", k, v) for k, v in sorted(c.items(), key=str)]

g2 = Graph(cb, ipv=[6], graph_cache=False)
for meth in ("set_weight_formula", "weight_formula"):
    if hasattr(g2, meth): print("formula setter:", meth)
try:
    g2.set_weight_formula("t0*c0 + t1*c1")
    c2 = Counter(); walk(g2.serialize(), "", c2)
    print("FORMULA graph leaf types:"); [print("  ", k, v) for k, v in sorted(c2.items(), key=str)]
except Exception as e:
    print("formula path:", type(e).__name__, e)
