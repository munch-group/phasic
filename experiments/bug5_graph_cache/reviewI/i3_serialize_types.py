"""Review I probe 3: which Python/numpy types appear in Graph.serialize() output
(constant, parameterized, formula, discretized). Decides whether a save-side
encoder limited to 'plain JSON values plus the ndarray tag' must still map
numpy scalars. No cache is used."""
import collections, numpy as np, phasic
from phasic import Graph, with_ipv

@with_ipv([5])
def coal(state, **kw):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], [n * (n - 1) / 2]]]

@with_ipv([4])
def coal_const(state, **kw):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], n * (n - 1) / 2]]

def walk(x, path, out):
    t = type(x)
    if isinstance(x, dict):
        for k, v in x.items():
            walk(v, f"{path}.{k}", out)
    elif isinstance(x, (list, tuple)):
        out[(t.__name__, path)] += 1
        for v in x[:3]:
            walk(v, path + "[]", out)
    else:
        out[(f"{t.__module__}.{t.__name__}", path)] += 1

graphs = {"param": Graph(coal), "const": Graph(coal_const)}
g = Graph(coal); g.set_weight_formula("t0 * c0") if hasattr(g, "set_weight_formula") else None
graphs["formula?"] = g
try:
    gd = Graph(coal); gd.update_weights([1.0]); graphs["discretized"] = gd.discretize(0.1)
except Exception as e:
    print("discretize failed:", type(e).__name__, e)
for name, gr in graphs.items():
    out = collections.Counter()
    walk(gr.serialize(), name, out)
    odd = {k: v for k, v in out.items() if not k[0].startswith(("builtins.", "numpy.ndarray")) and k[0] not in ("list", "tuple")}
    print(name, "weight_mode", getattr(gr, "_weight_mode", None), "non-builtin scalar types:", odd or "none")
