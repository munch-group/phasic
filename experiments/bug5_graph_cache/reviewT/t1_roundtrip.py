"""reviewT t1: does Graph.from_serialized(g.serialize()) (the hit path) match a
fresh build on state that the plan's hit path does not restore?
Checks param_length / parameterized / theta_dim cases. No cache used."""
import os, sys
assert os.environ["HOME"].startswith("/private/tmp/"), os.environ["HOME"]
import numpy as np
import phasic
from phasic import Graph


def const_cb(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], n * (n - 1) / 2]]


def par_cb(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], [n * (n - 1) / 2, 0.0]]]


def par3_cb(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], [n * (n - 1) / 2, 1.0, 0.5]]]


def show(tag, g):
    out = {}
    for name in ("vertices_length", "param_length", "parameterized", "get_was_dph"):
        try:
            out[name] = getattr(g, name)()
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {e}"
    out["dyn"] = getattr(g, "dyn_ordering", None)
    print(tag, out)
    return out


cases = [
    ("const", const_cb, {}),
    ("const theta_dim=2", const_cb, {"theta_dim": 2}),
    ("par2", par_cb, {}),
    ("par3 theta_dim=2", par3_cb, {"theta_dim": 2}),
]
for name, cb, kw in cases:
    try:
        g = Graph(cb, [5], **kw)
    except Exception as e:
        print(name, "fresh build failed", type(e).__name__, e)
        continue
    h = Graph(Graph.from_serialized(g.serialize()))
    a = show(f"{name:20s} fresh:", g)
    b = show(f"{name:20s} hit  :", h)
    print("   differ:", {k for k in a if a[k] != b[k]})
    # behaviour probes
    for theta in ([2.0], [2.0, 3.0], [2.0, 3.0, 4.0]):
        res = []
        for gg in (g, h):
            try:
                gg.update_weights(theta)
                res.append(round(float(gg.expectation()), 6))
            except Exception as e:
                res.append(f"ERR {type(e).__name__}")
        print(f"   update_weights({theta}) -> expectation fresh/hit: {res}")
