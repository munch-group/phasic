"""Review 4 probe: per-call cost of a serialize -> from_serialized hand-off
(the R13 route) versus building a graph, at increasing vertex counts.
Coalescent model (the svgd-basics builder). Parameterized, theta_dim=1."""
import sys, time, json
import numpy as np
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from phasic import Graph, with_ipv

def coal(n):
    @with_ipv([n] + [0] * (n - 1))
    def cb(state):
        tr = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2: continue
                if not same and (state[i] < 1 or state[j] < 1): continue
                new = state.copy(); new[i] -= 1; new[j] -= 1; new[i + j + 1] += 1
                tr.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return tr
    return Graph(cb)

for n in [int(a) for a in sys.argv[1:]]:
    t0 = time.perf_counter(); g = coal(n); tb = time.perf_counter() - t0
    V = g.vertices_length()
    t0 = time.perf_counter(); d = g.serialize(); ts = time.perf_counter() - t0
    js = json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()})
    t0 = time.perf_counter(); g2 = Graph.from_serialized(d); td = time.perf_counter() - t0
    g.update_weights([1.0]); g2.update_weights([1.0])
    e1, e2 = g.expectation(), g2.expectation()
    print(f"n={n} V={V} build={tb:.2f}s serialize={ts:.2f}s from_serialized={td:.2f}s json={len(js)/1e6:.1f}MB expect_equal={e1==e2} ({e1!r} vs {e2!r})", flush=True)
