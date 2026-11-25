"""reviewAC p5: leaf types in Graph.serialize() output (what the edited
_serialize_value must encode). No cache use."""
import os, numpy as np
assert "scratchpad" in os.environ["HOME"]
from phasic import Graph, with_ipv
from collections import Counter

def leaves(x, path, out):
    if isinstance(x, dict):
        for k, v in x.items():
            out[("key", type(k).__name__)] += 1
            leaves(v, f"{path}.{k}", out)
    elif isinstance(x, (list, tuple)):
        out[(type(x).__name__, "container")] += 1
        for v in x:
            leaves(v, path, out)
    elif isinstance(x, np.ndarray):
        out[("ndarray", str(x.dtype))] += 1
    else:
        out[(type(x).__name__, path.split(".")[1] if "." in path else path)] += 1

@with_ipv([4])
def cb(state, **kw):
    n = state[0]
    return [([n - 1], [n * (n - 1) / 2.0])] if n > 1 else []

g = Graph(cb, graph_cache=False)
for name, gg in [("param", g), ("discretized", g.discretize(0.1))]:
    out = Counter(); leaves(gg.serialize(), "", out); print(name, dict(out))
g2 = Graph(cb, graph_cache=False)
g2.weight_formula = "c0*t0"
out = Counter(); leaves(g2.serialize(), "", out); print("formula", dict(out))
