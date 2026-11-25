"""reviewY y1: does Graph.from_serialized(g.serialize()) (what a cache hit
installs) reproduce a fresh callback build?  Runs on master code; no cache.
HOME must point to a scratch dir."""
import os, sys
assert "phasic_cache" not in os.environ.get("PHASIC_CACHE_DIR", "") or True
import numpy as np
import phasic
from phasic import Graph

def cmp(name, g):
    h = Graph.from_serialized(g.serialize())
    out = {}
    for attr in ("param_length", "parameterized", "vertices_length", "state_length"):
        try:
            a = getattr(g, attr)(); b = getattr(h, attr)()
        except Exception as e:
            a = b = f"ERR {type(e).__name__}: {e}"
        out[attr] = (a, b)
    try:
        out["expectation"] = (g.expectation(), h.expectation())
    except Exception as e:
        out["expectation"] = f"ERR {type(e).__name__}: {e}"
    # theta behaviour
    try:
        pl = g.param_length()
        th = [2.0] * max(pl, 1)
        g2 = Graph.from_serialized(g.serialize())
        g.update_weights(th); h.update_weights(th)
        out["exp_after_update"] = (g.expectation(), h.expectation())
    except Exception as e:
        out["exp_after_update"] = f"ERR {type(e).__name__}: {e}"
    print(name)
    for k, v in out.items():
        flag = "" if (isinstance(v, tuple) and v[0] == v[1]) else "   <-- DIFF"
        print(f"   {k}: {v}{flag}")

# 1. constant callback with theta_dim=2
def cb_const(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2)]
try:
    g = Graph(cb_const, ipv=[4], theta_dim=2)
    cmp("constant edges + theta_dim=2", g)
except Exception as e:
    print("constant+theta_dim build fails:", e)

# 2. parameterized callback, coefficients length 3, theta_dim=2
def cb_par(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2, 0.0, 1.0])]
g = Graph(cb_par, ipv=[4], theta_dim=2)
cmp("param coeff len 3, theta_dim=2", g)

# 3. parameterized, no theta_dim
g = Graph(cb_par, ipv=[4])
cmp("param coeff len 3, no theta_dim", g)

# 4. constant callback, no theta_dim
g = Graph(cb_const, ipv=[4])
cmp("constant edges, no theta_dim", g)
