"""Review 6 probe: does the address-order (R14) non-determinism reproduce on
master, which paths does it touch, and is the parameterized (already
index-ordered) path immune?  In-process heap perturbation: random-size
native allocations (numpy) kept alive between graph builds.
"""
import sys, random, struct
import numpy as np
from phasic import Graph, with_ipv

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
K = int(sys.argv[2]) if len(sys.argv) > 2 else 12

def make_cb(param):
    def cb(state):
        out = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy(); new[i] -= 1; new[j] -= 1; new[i + j + 1] += 1
                w = state[i] * (state[j] - same) / (1 + same)
                out.append([new, [w]] if param else [new, float(w) * 0.7])
        # add a back-migration-like cycle to make the graph cyclic & non-trivial
        return out
    return with_ipv([N] + [0] * (N - 1))(cb)

def bits(x):
    a = np.atleast_1d(np.asarray(x, dtype=float))
    return tuple(struct.pack('<d', float(v)).hex() for v in a)

random.seed(1)
keep = []
res = {k: set() for k in ['const_expect', 'const_var', 'const_rt_expect',
                          'const_moments3', 'param_expect', 'param_var']}
for k in range(K):
    for _ in range(random.randint(1, 400)):
        keep.append(np.empty(random.randint(1, 5000)))
    g = Graph(make_cb(False))
    nv = g.vertices_length()
    rew = np.array([float((v * 7919) % 13) / 5.0 + 0.1 for v in range(nv)])
    res['const_expect'].add(bits(g.expectation()))
    res['const_var'].add(bits(g.variance()))
    res['const_moments3'].add(bits(g.moments(3)))
    gt = g.reward_transform(rew)
    res['const_rt_expect'].add(bits(gt.expectation()))
    for _ in range(random.randint(1, 400)):
        keep.append(np.empty(random.randint(1, 5000)))
    gp = Graph(make_cb(True))
    gp.update_weights([0.7])
    res['param_expect'].add(bits(gp.expectation()))
    res['param_var'].add(bits(gp.variance()))
    if k % 3 == 2:
        keep = keep[len(keep) // 2:]

print(f"N={N} vertices={nv} instances={K}")
for key, s in res.items():
    print(f"  {key:18s} distinct bit patterns: {len(s)}")
