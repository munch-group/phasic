"""Review 6: is the STRUCTURE (edge order) of a reward-transformed graph
address-dependent, not only its last-digit values?"""
import random, hashlib
import numpy as np
from phasic import Graph, with_ipv
N = 14
def cb(state):
    out = []
    for i in range(state.size):
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2: continue
            if not same and (state[i] < 1 or state[j] < 1): continue
            new = state.copy(); new[i] -= 1; new[j] -= 1; new[i + j + 1] += 1
            out.append([new, float(state[i] * (state[j] - same) / (1 + same)) * 0.7])
    return out
f = with_ipv([N] + [0] * (N - 1))(cb)
random.seed(2); keep = []; structs = set(); orders = set()
for k in range(16):
    for _ in range(random.randint(1, 400)): keep.append(np.empty(random.randint(1, 5000)))
    g = Graph(f); nv = g.vertices_length()
    rew = np.array([float((v * 7919) % 13) / 5.0 + 0.1 for v in range(nv)])
    gt = g.reward_transform(rew)
    order = []
    for i in range(gt.vertices_length()):
        v = gt.vertex_at(i)
        order.append((tuple(v.state()), tuple(tuple(e.to().state()) for e in v.edges())))
    orders.add(hashlib.sha1(repr(order).encode()).hexdigest())
    s = gt.serialize()
    structs.add(hashlib.sha1(repr({k2: np.asarray(v2).tobytes() if hasattr(v2,'__len__') and not isinstance(v2,str) else v2 for k2, v2 in sorted(s.items())}).encode()).hexdigest())
    if k % 3 == 2: keep = keep[len(keep)//2:]
print(f"reward_transform: distinct edge-order patterns {len(orders)}, distinct serialize() outputs {len(structs)} over 16 instances")
