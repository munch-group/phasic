"""Review F p7: which Python types occur in Graph.serialize() output for
callback-built graphs (what the edited _serialize_value must encode)."""
from collections import Counter
import numpy as np
from phasic import Graph


def walk(x, c, path="root"):
    c[type(x).__module__ + "." + type(x).__name__] += 1
    if isinstance(x, dict):
        for k, v in x.items():
            walk(v, c, f"{path}.{k}")
    elif isinstance(x, (list, tuple)):
        for v in x:
            walk(v, c, path)


def const_cb(state, **kw):
    n = state[0]
    return [] if n <= 1 else [([n - 1], n * (n - 1) / 2)]


def param_cb(state, **kw):
    n = state[0]
    return [] if n <= 1 else [([n - 1], [n * (n - 1) / 2, 0.0])]


for label, g in [("constant", Graph(const_cb, ipv=[5], graph_cache=False)),
                 ("parameterized", Graph(param_cb, ipv=[5], graph_cache=False)),
                 ("theta_dim=1 w/ 2 coeffs", Graph(param_cb, ipv=[5], theta_dim=1, graph_cache=False))]:
    c = Counter()
    walk(g.serialize(), c)
    print(f"{label:26s}", dict(c))
