# Probe: are constant-model read-outs bit-identical across instances, and does
# heap layout (allocation interleaving) change them? Modest sizes only.
import os, struct, sys
os.environ.setdefault("PHASIC_CACHE_DIR", os.path.join(os.path.dirname(__file__), "cache"))
import numpy as np
import phasic

def two_island(n):
    # constant-rate structured coalescent-ish model with cycles (migration)
    def cb(state):
        a, b = int(state[0]), int(state[1])
        out = []
        if a + b <= 1:
            return out
        if a >= 2: out.append(([a-1, b], a*(a-1)/2 * 1.0))
        if b >= 2: out.append(([a, b-1], b*(b-1)/2 * 0.7))
        if a >= 1: out.append(([a-1, b+1], a * 0.31))
        if b >= 1: out.append(([a+1, b-1], b * 0.17))
        return out
    return phasic.Graph(cb, ipv=[([n, 0], 1.0)])

def bits(x):
    return struct.pack('<d', float(x)).hex()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
res = {}
junk = []
for trial in range(6):
    # perturb heap layout between builds
    junk.append([np.empty(np.random.randint(1, 5000)) for _ in range(50)])
    g = two_island(N)
    e = g.expectation(); v = g.variance()
    rw = np.arange(g.vertices_length(), dtype=float) % 3
    er = g.expectation(rw)
    res.setdefault('E', set()).add(bits(e))
    res.setdefault('V', set()).add(bits(v))
    res.setdefault('E_rw', set()).add(bits(er))
    try:
        rt = g.reward_transform(rw)
        res.setdefault('rt_E', set()).add(bits(rt.expectation()))
    except Exception as ex:
        res['rt_E_err'] = {repr(ex)[:80]}
print("vertices", g.vertices_length())
for k, s in res.items():
    print(k, len(s), sorted(s)[:3])
