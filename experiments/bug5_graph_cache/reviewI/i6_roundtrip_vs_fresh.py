"""Review I probe 6: does a graph restored the way a v3.4 hit restores it
(Graph.from_serialized(json round trip of serialize()) then copy-construct)
equal a fresh build for theta_dim variants? Uses master's encoder/decoder
functions directly; no cache directory is touched."""
import json, numpy as np
from phasic import Graph, with_ipv
from phasic.graph_cache import _serialize_value, _deserialize_value

@with_ipv([4])
def const_cb(state, **kw):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], n * (n - 1) / 2]]

@with_ipv([4])
def param_cb(state, **kw):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], [n * (n - 1) / 2]]]

@with_ipv([4])
def param2_cb(state, **kw):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], [n * (n - 1) / 2, 1.0]]]

def restore(g):
    blob = json.dumps(_serialize_value(g.serialize()))
    c = Graph.from_serialized(_deserialize_value(json.loads(blob)))
    class H(Graph):
        pass
    h = Graph.__new__(Graph)
    super(Graph, h).__init__(c)          # what the hit path does: super().__init__(cached_graph)
    h._weight_mode = getattr(c, '_weight_mode', 'linear'); h._weight_formula_tape = None; h.is_discrete = c.is_discrete
    return h

def summary(g):
    out = {"V": g.vertices_length(), "param_length": g.param_length()}
    s = g.serialize()
    out["ser_keys"] = sorted(s)
    return out, s

cases = [("const", const_cb, {}), ("const theta_dim=2", const_cb, {"theta_dim": 2}),
         ("param", param_cb, {}), ("param theta_dim=3", param_cb, {"theta_dim": 3}),
         ("param2", param2_cb, {}), ("param2 theta_dim=1", param2_cb, {"theta_dim": 1}), ("param theta_dim=1", param_cb, {"theta_dim": 1})]
for name, cb, kw in cases:
    try:
        f = Graph(cb, graph_cache=False, **kw)
        h = restore(f)
        sf, ser_f = summary(f); sh, ser_h = summary(h)
        diffs = [k for k in ser_f if k in ser_h and not np.array_equal(np.asarray(ser_f[k], dtype=object), np.asarray(ser_h[k], dtype=object))]
        print(f"{name:22s} fresh {sf['V']}v pl={sf['param_length']}  restored {sh['V']}v pl={sh['param_length']}  "
              f"serialize fields differing: {diffs or 'none'}  keys equal: {sf['ser_keys'] == sh['ser_keys']}")
    except Exception as e:
        print(f"{name:22s} ERROR {type(e).__name__}: {e}")
