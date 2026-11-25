# reviewAN p5: leaf types in serialize() output (linear and formula mode): does the
# planned "plain JSON values + ndarray tag only" _serialize_value accept them?
import os
assert os.environ["HOME"].startswith("/private/tmp/")
import numpy as np, phasic
from phasic import Graph
def cb(state):
    n = state[0]
    if n <= 1: return []
    return [([n - 1], [n * (n - 1) / 2, 1.0])]
def leaves(x, path, out):
    if isinstance(x, dict):
        for k, v in x.items(): leaves(v, f"{path}.{k}", out)
    elif type(x) in (list, tuple):
        for i, v in enumerate(x[:3]): leaves(v, f"{path}[{i}]", out)
    else:
        out.setdefault(type(x).__name__, []).append(path)
for mode in ["linear", "formula"]:
    g = Graph(cb, ipv=[5])
    if mode == "formula":
        g.weight_formula = "t0*c0 + t1*c1"
    s = g.serialize()
    out = {}
    leaves(s, "s", out)
    print(mode, {k: v[:4] for k, v in out.items()})
