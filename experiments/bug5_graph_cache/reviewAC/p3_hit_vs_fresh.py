"""reviewAC p3: on MASTER code (the restore sequence the plan keeps), compare a
cache HIT with a fresh build, for shapes of callback the plan does not test.
HOME and PHASIC_CACHE_DIR point into the scratchpad (set by the caller)."""
import os, sys, numpy as np
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import phasic
from phasic import Graph, with_ipv, StateIndexer, Property

def describe(g, theta):
    out = {"n": g.vertices_length(), "param_length": g.param_length(),
           "parameterized": g.parameterized()}
    edges = []
    for v in g:
        edges.append((tuple(v.state()), tuple((tuple(e.to().state()), round(e.weight(), 12)) for e in v.edges())))
    out["edges_hash"] = hash(tuple(edges))
    if theta is not None:
        g.update_weights(theta)
    try:
        out["E"] = float(g.expectation())
    except Exception as e:
        out["E"] = f"ERR {type(e).__name__}: {e}"[:120]
    return out

def compare(name, cb, ipv, theta, **kw):
    fresh = Graph(cb, ipv=ipv, graph_cache=False, **kw)
    Graph(cb, ipv=ipv, graph_cache=True, **kw)         # writes entry
    calls = []
    hit = Graph(cb, ipv=ipv, graph_cache=True, **kw)   # should hit
    a, b = describe(fresh, theta), describe(hit, theta)
    diffs = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    print(f"{name:40} {'SAME' if not diffs else 'DIFF ' + repr(diffs)}")

# 1 constant graph with an explicit theta_dim kwarg
def const_cb(state, **kw):
    if state[0] > 1:
        return [([state[0] - 1], state[0] * 1.0)]
    return []
# invalid on fresh build: compare("constant + theta_dim=2", const_cb, [4], None, theta_dim=2)

# 2 parameterized, theta_dim > coefficient length
def par_cb(state, **kw):
    if state[0] > 1:
        return [([state[0] - 1], [state[0] * 1.0, 0.5])]
    return []
# invalid (too few coeffs): compare("param coeff2, theta_dim=3", par_cb, [4], [1.0, 2.0, 3.0], theta_dim=3)
compare("param coeff2, no theta_dim", par_cb, [4], [1.0, 2.0])

# 3 two transitions to the same child state
def dup_cb(state, **kw):
    if state[0] > 1:
        return [([state[0] - 1], [1.0, 0.0]), ([state[0] - 1], [0.0, 2.0])]
    return []
compare("duplicate edges same target", dup_cb, [4], [1.0, 3.0])

# 4 an all-zero coefficient edge
def zero_cb(state, **kw):
    if state[0] > 1:
        return [([state[0] - 1], [0.0, 0.0]), ([0], [1.0, 1.0])] if state[0] == 3 else [([state[0] - 1], [1.0, 1.0])]
    return []
compare("all-zero coefficient edge", zero_cb, [4], [1.0, 3.0])

# 5 multi-state ipv (start edges)
def ms_cb(state, **kw):
    if state[0] > 1:
        return [([state[0] - 1], [state[0] * 1.0])]
    return []
compare("ipv with two start states", ms_cb, [[[4], 0.25], [[3], 0.75]], [2.0])
compare("ipv sub-stochastic (defect 0.2)", ms_cb, [[[4], 0.5], [[3], 0.3]], [2.0])
