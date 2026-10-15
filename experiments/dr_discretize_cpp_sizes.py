"""De-risk (discretize C++ plan, phase -1): graph sizes and build times."""
import sys, time
import numpy as np
from phasic import Graph, with_ipv

def coal(n, param=True):
    @with_ipv([n] + [0] * (n - 1))
    def cb(state):
        out = []
        for i in range(state.size):
            if state[i] == 0:
                continue
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and state[j] < 1:
                    continue
                k = i + j + 1
                if k >= state.size:
                    continue
                new = state.copy(); new[i] -= 1; new[j] -= 1; new[k] += 1
                r = state[i] * (state[j] - same) / (1 + same)
                out.append([new, [r] if param else r])
        return out
    return Graph(cb, graph_cache=False)

if __name__ == '__main__':
    for n in [int(a) for a in sys.argv[1:]]:
        t = time.perf_counter(); g = coal(n); dt = time.perf_counter() - t
        print(f"coalescent n={n}: {g.vertices_length()} vertices, built in {dt:.1f}s", flush=True)
