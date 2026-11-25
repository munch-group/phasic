"""reviewH p5: leaf types in Graph.serialize() output (what the edited
_serialize_value must accept), for a constant, a parameterized and a
formula-mode callback graph (test 15 saves a formula entry by hand)."""
import os
assert "scratchpad" in os.environ["HOME"]
from collections import Counter
import numpy as np
from phasic import Graph


def leaves(x, path, out):
    if isinstance(x, dict):
        for k, v in x.items():
            leaves(v, f"{path}.{k}", out)
    elif isinstance(x, (list, tuple)):
        out[(path, type(x).__name__)] += 1
        for v in x:
            leaves(v, path + "[]", out)
    elif isinstance(x, np.ndarray):
        out[(path, f"ndarray[{x.dtype}]")] += 1
    else:
        out[(path, type(x).__module__ + "." + type(x).__name__)] += 1


def cb_c(state, **kw):
    n = state[0]
    return [] if n <= 1 else [([n - 1], n * (n - 1) / 2)]


def cb_p(state, **kw):
    n = state[0]
    return [] if n <= 1 else [([n - 1], [n * (n - 1) / 2, 1.0])]


for name, g in [("constant", Graph(cb_c, ipv=[4])), ("param", Graph(cb_p, ipv=[4]))]:
    out = Counter()
    leaves(g.serialize(), name, out)
    for k, v in sorted(out.items()):
        print(k, v)

g = Graph(cb_p, ipv=[4])
try:
    g.weight_formula = "t0*c0 + c1"
    out = Counter()
    leaves(g.serialize(), "formula", out)
    for k, v in sorted(out.items()):
        if not k[0].startswith("formula.") or "tape" in k[0] or "weight" in k[0]:
            print(k, v)
except Exception as e:
    print("formula:", type(e).__name__, e)
