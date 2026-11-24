"""Review A: PHASIC_DYN_ORDERING at build time is carried by serialize() into a hit,
but is not a key input. Simulates hit path: from_serialized + Graph(_Graph) copy."""
import os
import numpy as np
assert "scratchpad" in os.environ["HOME"]
from phasic import Graph, with_ipv
n = 30
@with_ipv([n] + [0] * (n - 1))
def coal(state):
    t = []
    for i in range(state.size):
        if state[i] == 0: continue
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2: continue
            if not same and (state[i] < 1 or state[j] < 1): continue
            new = state.copy(); new[i] -= 1; new[j] -= 1; new[i + j + 1] += 1
            t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return t
os.environ["PHASIC_DYN_ORDERING"] = "1"
g_writer = Graph(coal)                      # process A builds (env set) and saves
data = g_writer.serialize()
del os.environ["PHASIC_DYN_ORDERING"]       # process B: env not set
cached = Graph.from_serialized(data)
hit = Graph(cached)                         # what super().__init__(cached_graph) does
fresh = Graph(coal)
print("writer.dyn_ordering =", g_writer.dyn_ordering, "| serialized:", data["dyn_ordering"])
print("hit (from_serialized+copy).dyn_ordering =", hit.dyn_ordering, "| fresh build in B:", fresh.dyn_ordering)
for g in (hit, fresh): g.update_weights([1.0])
eh, ef = hit.expectation(), fresh.expectation()
print("expectation hit=%r fresh=%r bit-identical=%s" % (eh, ef, eh == ef))
mh, mf = np.asarray(hit.moments(4)), np.asarray(fresh.moments(4))
print("moments(4) max rel diff = %.3e" % float(np.max(np.abs(mh - mf) / np.abs(mf))))
