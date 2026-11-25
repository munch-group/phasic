import os, json, errno
import numpy as np
import phasic
from phasic import Graph, with_ipv, GraphCacheKeyError
from phasic import graph_cache as gcm
from phasic.callback_hash import cache_entry_key
g = Graph(3)
print("RESULT Graph instance callable:", callable(g))
try:
    Graph(g, cache_key="x"); print("RESULT Graph(graph, cache_key) accepted?!")
except Exception as e:
    print("RESULT Graph(graph, cache_key) ->", type(e).__name__, str(e)[:80])

@with_ipv([3])
def cb(state, **kw):
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], float(n)]]

# bool vs int vs float kwargs: distinct keys although the callback sees the same float
ks = {repr(v): cache_entry_key("k", cb, a=v) for v in (True, 1, 1.0, np.float64(1.0), np.int64(1))}
print("RESULT distinct keys for True/1/1.0/np.float64/np.int64:", len(set(ks.values())))

# the identity predicate vs content-hashed keys: StateIndexer key refused although it hashes by content
from phasic import StateIndexer, Property
ix = StateIndexer(lineage=[Property("n", max_value=3)])
try:
    cache_entry_key("k", cb, d={ix: 1.0}); print("RESULT indexer dict key admitted")
except GraphCacheKeyError as e:
    print("RESULT indexer dict key refused")
try:
    import math
    cache_entry_key("k", cb, d={math.sqrt: 1.0}); print("RESULT builtin dict key admitted")
except GraphCacheKeyError:
    print("RESULT builtin dict key refused")

# OSError message of a non-OSError-with-errno-None rewrap
e = gcm._reraise_oserror(OSError("plain"), "msg")
print("RESULT rewrap of errno-less OSError:", type(e).__name__, str(e))

# save: zero-length tmp left after failure in serialize? (serialize happens before tmp creation)
d = gcm.GraphCache(create=True).cache_dir
print("RESULT partials after run:", [p for p in os.listdir(d) if p.endswith(".partial")])
