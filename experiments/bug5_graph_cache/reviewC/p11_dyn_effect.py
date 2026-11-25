"""Review C p11: effect of dyn_ordering (the flag a hit inherits from the
writer, p9) on results and time; two-island-like cyclic model, no cache."""
import time, numpy as np
from phasic import Graph, with_ipv
n = 7
@with_ipv([[[n, 0], 1.0]])
def two_island(state, **kw):
    a, b = int(state[0]), int(state[1])
    out = []
    if a >= 2: out.append([[a - 1, b], [a * (a - 1) / 2, 0.0]])
    if b >= 2: out.append([[a, b - 1], [b * (b - 1) / 2, 0.0]])
    if a >= 1 and a + b > 1: out.append([[a - 1, b + 1], [0.0, a]])
    if b >= 1 and a + b > 1: out.append([[a + 1, b - 1], [0.0, b]])
    return out
res = {}
for flag in (False, True):
    g = Graph(two_island, graph_cache=False)
    g.dyn_ordering = flag
    g.update_weights([1.0, 0.3])
    t = time.time(); m = np.asarray(g.moments(3)); dt = time.time() - t
    res[flag] = m
    print("dyn_ordering", flag, "moments", m, f"{dt:.3f}s")
print("max rel diff:", np.max(np.abs(res[True] - res[False]) / np.abs(res[False])))
