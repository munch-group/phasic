# Probe: size of the FFI structure JSON (XLA attribute; copied+hashed per FFI
# call) per vertex for a parameterized two-island model; extrapolate.
import os, sys, time
os.environ.setdefault("PHASIC_CACHE_DIR", os.path.join(os.path.dirname(__file__), "cache"))
import numpy as np
import phasic
from phasic.ffi_wrappers import _serialize_graph_structure

def two_island(n):
    def cb(state):
        a, b = int(state[0]), int(state[1])
        out = []
        if a + b <= 1:
            return out
        if a >= 2: out.append(([a-1, b], [a*(a-1)/2, 0, 0]))
        if b >= 2: out.append(([a, b-1], [0, b*(b-1)/2, 0]))
        if a >= 1: out.append(([a-1, b+1], [0, 0, a]))
        if b >= 1: out.append(([a+1, b-1], [0, 0, b]))
        return out
    return phasic.Graph(cb, ipv=[([n, 0], 1.0)])

for N in (40, 80, 160):
    g = two_island(N)
    nv = g.vertices_length()
    ne = sum(len(g.vertex_at(i).edges()) for i in range(nv))
    t = time.time(); s = _serialize_graph_structure(g); dt = time.time() - t
    print(f"N={N} vertices={nv} edges={ne} json_bytes={len(s)} "
          f"bytes/edge={len(s)/max(ne,1):.1f} serialize_s={dt:.2f}")
