"""reviewE e2: on master, compare a graph-cache HIT against a fresh build attribute by
attribute (Python __dict__, C-level flags, edge lists in order). HOME is a scratch dir."""
import os, sys, numpy as np
import phasic
from phasic import Graph, StateIndexer, Property
print("cache dir:", phasic.graph_cache.GraphCache().cache_dir)
assert "phasic_cache" in str(phasic.graph_cache.GraphCache().cache_dir) and os.environ["HOME"].startswith("/private/tmp")
ix = StateIndexer(lineage=[Property('n', max_value=6)])

CALLS=[0]
def cb(state, rate=1.0, indexer=None, **kw):
    CALLS[0]+=1
    n = state[0]
    if n <= 1: return []
    return [([n-1], [n*(n-1)/2*rate, 0.5*n])]

def snap(g):
    d = {}
    for k, v in sorted(vars(g).items()):
        if k in ("_callback",): v = getattr(v, "__name__", type(v).__name__)
        d[k] = repr(v)[:80]
    d["param_length"] = g.param_length()
    d["was_dph"] = g.get_was_dph()
    d["dyn_ordering"] = g.dyn_ordering
    d["state_length"] = g.state_length()
    edges = []
    for v in g:
        edges.append((tuple(v.state()), [(tuple(e.to().state()), e.weight(), tuple(e.edge_state(g.param_length()) if hasattr(e,'edge_state') else ())) for e in v.edges()]))
    d["edges"] = edges
    return d

kw = dict(rate=2.0, theta_dim=2, indexer=ix)
g1 = Graph(cb, ipv=[6], graph_cache=True, **kw)
c=CALLS[0]
g2 = Graph(cb, ipv=[6], graph_cache=True, **kw)
print('callback calls fresh/hit:', c, CALLS[0]-c)
print('has edge_state:', hasattr(next(iter(g1)).edges()[0] if next(iter(g1)).edges() else None,'edge_state'))
s1, s2 = snap(g1), snap(g2)
diff = [k for k in sorted(set(s1) | set(s2)) if s1.get(k) != s2.get(k)]
print("keys only in fresh:", sorted(set(s1) - set(s2)))
print("keys only in hit:", sorted(set(s2) - set(s1)))
for k in diff:
    print("DIFF", k, "\n  fresh:", str(s1.get(k))[:200], "\n  hit:  ", str(s2.get(k))[:200])
g1.update_weights([1.0, 0.3]); g2.update_weights([1.0, 0.3])
print("expectation fresh/hit:", g1.expectation(), g2.expectation())
phasic.graph_cache.GraphCache().clear()
