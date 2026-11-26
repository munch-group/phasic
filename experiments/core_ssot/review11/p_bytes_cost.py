"""Review 11 probe: cost of a bytes (JSON) hand-off into a Python Graph.

Builds a synthetic constant graph (2-D state, ~3 out-edges per vertex),
then times serialize -> json.dumps -> json.loads -> numpy dict ->
Graph.from_serialized, and the Graph(g) wrap / copy used today.
Usage: p_bytes_cost.py N
"""
import json, sys, time, resource
import numpy as np
import phasic

N = int(sys.argv[1])
side = int(np.ceil(np.sqrt(N)))

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS bytes

t0 = time.perf_counter()
g = phasic.Graph(2)
start = g.starting_vertex()
vs = {}
def V(i, j):
    k = (i, j)
    v = vs.get(k)
    if v is None:
        v = g.find_or_create_vertex([i + 1, j + 1])
        vs[k] = v
    return v
start.add_edge(V(0, 0), 1.0)
for i in range(side):
    for j in range(side):
        if i * side + j >= N:
            break
        v = V(i, j)
        if i + 1 < side:
            v.add_edge(V(i + 1, j), 1.0 + i)
        if j + 1 < side:
            v.add_edge(V(i, j + 1), 2.0 + j)
        if i + 1 == side or j + 1 == side:
            v.add_edge(g.find_or_create_vertex([0, 0]), 0.5)
t_build = time.perf_counter() - t0
nv = g.vertices_length()
print(f"N_req={N} vertices={nv} python_build_s={t_build:.2f} peak_rss_mb={rss_mb():.0f}", flush=True)

t0 = time.perf_counter(); d = g.serialize(); t_ser = time.perf_counter() - t0
from phasic.ffi_wrappers import _make_json_serializable  # noqa
t0 = time.perf_counter(); s = json.dumps(_make_json_serializable(d)); t_dump = time.perf_counter() - t0
print(f"serialize_s={t_ser:.2f} json_dumps_s={t_dump:.2f} json_MB={len(s)/1e6:.1f} B_per_vertex={len(s)/nv:.0f} peak_rss_mb={rss_mb():.0f}", flush=True)

t0 = time.perf_counter()
dj = json.loads(s)
for k in ('states', 'vertex_indices', 'edges', 'constant_edges', 'start_edges', 'param_edges', 'start_param_edges', 'is_aux'):
    if k in dj:
        dj[k] = np.asarray(dj[k])
t_load = time.perf_counter() - t0
print(f"json_loads_np_s={t_load:.2f} peak_rss_mb={rss_mb():.0f}", flush=True)

t0 = time.perf_counter(); g2 = phasic.Graph.from_serialized(dj); t_fs = time.perf_counter() - t0
print(f"from_serialized_s={t_fs:.2f} peak_rss_mb={rss_mb():.0f}", flush=True)
print(f"roundtrip_equal_vertices={g2.vertices_length()==nv}", flush=True)

t0 = time.perf_counter(); g3 = phasic.Graph(g); t_wrap = time.perf_counter() - t0
print(f"Graph(g)_wrap_s={t_wrap:.6f}", flush=True)
print(f"type_today_is_raw_base={type(g).__mro__[1].__name__}", flush=True)
