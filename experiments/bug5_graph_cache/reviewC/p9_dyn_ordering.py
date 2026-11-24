"""Review C p9: does a cache hit carry the WRITER's PHASIC_DYN_ORDERING
(phasic.c:2996, serialized at _graph_serialize.py:222, restored :698)?
Master code, graph_cache=True path = the hit path v2.7 keeps.
Run as: p9 write|read  (HOME and PHASIC_CACHE_DIR point to scratch)."""
import os, sys
import phasic
from phasic import Graph, with_ipv

@with_ipv([6])
def cb(state, **kw):
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], [n * (n - 1) / 2]]]

mode = sys.argv[1]
print(mode, "PHASIC_DYN_ORDERING env:", os.environ.get("PHASIC_DYN_ORDERING"))
g = Graph(cb, graph_cache=True)
fresh = Graph(cb, graph_cache=False)
print(mode, "cache-path graph dyn_ordering:", g.dyn_ordering, "| uncached build dyn_ordering:", fresh.dyn_ordering)
