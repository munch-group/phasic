"""Review 6 probe: construction cost split (user callback vs everything else),
structure-JSON size, and the per-call string copy+hash cost proxy.

Usage: python probe_construct.py N  (coalescent with N samples)
"""
import sys, time, os, resource, hashlib
import numpy as np
from phasic import Graph, with_ipv

N = int(sys.argv[1])

user_t = [0.0]
ncalls = [0]
memo = {}

def raw_cb(state):
    transitions = []
    for i in range(state.size):
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[i + j + 1] += 1
            transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return transitions

def timed_cb(state):
    t0 = time.perf_counter()
    r = raw_cb(state)
    user_t[0] += time.perf_counter() - t0
    ncalls[0] += 1
    memo[state.tobytes()] = r
    return r

def memo_cb(state):
    t0 = time.perf_counter()
    r = memo[state.tobytes()]
    user_t[0] += time.perf_counter() - t0
    return r

ipv = [N] + [0] * (N - 1)

t0 = time.perf_counter()
g = Graph(with_ipv(ipv)(timed_cb))
tot = time.perf_counter() - t0
nv = g.vertices_length()
print(f"N={N} vertices={nv} construct_total={tot:.3f}s user_callback={user_t[0]:.3f}s "
      f"({100*user_t[0]/tot:.1f}%) calls={ncalls[0]}")

user_t[0] = 0.0
t0 = time.perf_counter()
g2 = Graph(with_ipv(ipv)(memo_cb))
tot2 = time.perf_counter() - t0
print(f"memoized rebuild: total={tot2:.3f}s lookup={user_t[0]:.3f}s "
      f"non-callback (wrapper+pybind+native)={tot2-user_t[0]:.3f}s")

t0 = time.perf_counter()
s = g.serialize_json() if hasattr(g, 'serialize_json') else None
ts = time.perf_counter() - t0
if s is None:
    from phasic.ffi_wrappers import _ensure_json_string
    t0 = time.perf_counter(); s = _ensure_json_string(g.serialize()); ts = time.perf_counter() - t0
print(f"structure JSON: {len(s)/1e6:.2f} MB, produce={ts:.3f}s, bytes/vertex={len(s)/nv:.0f}")

# proxy for per-call std::string copy + unordered_map hash of the full key
b = s.encode()
t0 = time.perf_counter()
for _ in range(20):
    c = bytes(bytearray(b))
    hashlib.blake2b(c, digest_size=8).digest()
tc = (time.perf_counter() - t0) / 20
print(f"copy+hash proxy per call: {tc*1e3:.2f} ms")
print(f"maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.0f} MB")
