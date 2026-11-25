# Types of every top-level serialize() field for callback-built graphs
# (does any field hold a numpy SCALAR, which a "plain JSON + ndarray only"
# encoder would reject?).
import numpy as np
from phasic import Graph

def cbp(state, **kw):
    n = state[0]
    return [] if n <= 1 else [([n - 1], [n * (n - 1) / 2])]
def cbc(state, **kw):
    n = state[0]
    return [] if n <= 1 else [([n - 1], n * (n - 1) / 2)]
for name, g in [("param", Graph(cbp, ipv=[5])), ("param theta_dim=np.int64(1)", Graph(cbp, ipv=[5], theta_dim=np.int64(1))),
                ("constant", Graph(cbc, ipv=[5]))]:
    d = g.serialize()
    print(name, {k: type(v).__name__ for k, v in d.items()})
