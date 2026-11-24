"""Review A: leaf types in serialize() of callback-built graphs (what save must encode)."""
import os, collections
import numpy as np
assert "scratchpad" in os.environ["HOME"]
from phasic import Graph, with_ipv, StateIndexer, Property
from phasic.graph_cache import _serialize_value
def walk(x, acc, path="root"):
    if isinstance(x, dict):
        for k, v in x.items(): walk(v, acc, f"{path}.{k}")
    elif isinstance(x, (list, tuple)):
        for v in x: walk(v, acc, path + "[]")
    else:
        t = type(x).__name__ + (f"[{x.dtype}]" if isinstance(x, np.ndarray) else "")
        acc[t].add(path)
@with_ipv([4, 0, 0, 0])
def coal(state, indexer=None):
    t = []
    for i in range(state.size):
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2: continue
            if not same and (state[i] < 1 or state[j] < 1): continue
            new = state.copy(); new[i] -= 1; new[j] -= 1; new[min(i + j + 1, 3)] += 1
            t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return t
@with_ipv([4, 0, 0, 0])
def coal_const(state):
    return [[s, float(c[0])] for s, c in coal(state)] if False else [[np.array(s), 1.0] for s, c in [(x[0], x[1]) for x in coal.__wrapped__(state)]]
idx = StateIndexer(lineages=[Property("d", min_value=1, max_value=4)])
for name, g in [("param", Graph(coal)), ("param+indexer+theta_dim", Graph(coal, indexer=idx, theta_dim=1)),
                ("constant", Graph(coal_const))]:
    acc = collections.defaultdict(set); d = g.serialize(); walk(d, acc)
    print(name, {k: sorted(v)[:3] for k, v in acc.items()})
    try: _serialize_value(d); print("   _serialize_value ok")
    except Exception as e: print("   _serialize_value FAILS:", type(e).__name__, e)
