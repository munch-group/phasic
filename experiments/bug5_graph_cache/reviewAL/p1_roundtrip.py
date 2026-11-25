"""reviewAL p1: does a master-path cache hit (same restore sequence the plan keeps)
give identical serialize() output / param_length to a fresh build, for callbacks
whose theta_dim is given explicitly?  HOME is redirected by the caller."""
import os, sys, json
import numpy as np
assert os.environ["HOME"].startswith("/private/tmp/"), os.environ["HOME"]
import phasic
from phasic import Graph

def cmp_ser(a, b):
    sa, sb = a.serialize(), b.serialize()
    diffs = []
    for k in sorted(set(sa) | set(sb)):
        va, vb = sa.get(k, "<absent>"), sb.get(k, "<absent>")
        try:
            eq = (np.array_equal(np.asarray(va), np.asarray(vb))
                  if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray) else va == vb)
        except Exception as e:
            eq = f"cmp-error {e}"
        if eq is not True:
            diffs.append((k, str(va)[:80], str(vb)[:80]))
    return diffs

def const_cb(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], n * (n - 1) / 2.0]]

def param_cb(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], [n * (n - 1) / 2.0, 0.0]]]

cases = [
    ("const, theta_dim=None", const_cb, {}),
    ("param(len2), theta_dim=None", param_cb, {}),
    ("param(len2), theta_dim=2", param_cb, {"theta_dim": 2}),
    ("param(len2), theta_dim=1", param_cb, {"theta_dim": 1}),
]
for label, cb, kw in cases:
    fresh = Graph(cb, ipv=[5], graph_cache=False, **kw)
    w = Graph(cb, ipv=[5], graph_cache=True, tag=label, **kw)   # write
    h = Graph(cb, ipv=[5], graph_cache=True, tag=label, **kw)   # hit (master path)
    try:
        pf, ph = fresh.param_length(), h.param_length()
    except Exception as e:
        pf = ph = f"err {e}"
    print(f"{label}: param_length fresh={pf} hit={ph}; serialize diffs vs fresh: {cmp_ser(fresh, h)}")
