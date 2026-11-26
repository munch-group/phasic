"""Review 6: does the pybind callback builder leak the whole C graph?"""
import gc, resource, sys, os
import numpy as np
from phasic import Graph, with_ipv
import psutil
N = 32
def cb(state):
    out = []
    for i in range(state.size):
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2: continue
            if not same and (state[i] < 1 or state[j] < 1): continue
            new = state.copy(); new[i] -= 1; new[j] -= 1; new[i + j + 1] += 1
            out.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return out
f = with_ipv([N] + [0] * (N - 1))(cb)
memo = {}
def m(state):
    k = state.tobytes()
    if k not in memo: memo[k] = cb(state)
    return memo[k]
fm = with_ipv([N] + [0] * (N - 1))(m)
p = psutil.Process()
for k in range(12):
    g = Graph(fm); nv = g.vertices_length(); del g; gc.collect()
    print(f"iter {k} vertices={nv} rss={p.memory_info().rss/1e6:.0f} MB", flush=True)
