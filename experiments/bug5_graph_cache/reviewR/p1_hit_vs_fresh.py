"""reviewR p1: does a cache HIT (master's restore sequence, which the plan
keeps, §2.1 'hit: the current restore sequence runs') equal a fresh build?
Compares graph-level properties beyond serialize() arrays.
Run with HOME and PHASIC_CACHE_DIR redirected to a scratch dir."""
import os, sys, numpy as np
assert "scratchpad" in os.environ["HOME"], os.environ["HOME"]
import phasic
from phasic import Graph

def cb_const(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2.0)]

def cb_param(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2.0, 1.0])]

def props(g):
    out = {}
    for name in ["vertices_length", "param_length", "parameterized"]:
        try:
            out[name] = getattr(g, name)()
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {e}"
    out["dyn_ordering"] = getattr(g, "dyn_ordering", None)
    try:
        w = [e.weight() for v in g for e in v.edges()]
        out["weights"] = w
    except Exception as e:
        out["weights"] = f"ERR {e}"
    try:
        out["coeff_len"] = [e.coefficients_length() for v in g for e in v.edges()]
    except Exception as e:
        out["coeff_len"] = f"ERR {e}"
    return out

cases = [

    ("const", cb_const, dict()),
    ("param theta_dim=1 (decoupled)", cb_param, dict(theta_dim=1)),
    ("param", cb_param, dict()),
]
for label, cb, kw in cases:
    fresh = Graph(cb, [6], graph_cache=False, **kw)
    w = Graph(cb, [6], graph_cache=True, **kw)   # writes
    hit = Graph(cb, [6], graph_cache=True, **kw)  # loads
    pf, ph = props(fresh), props(hit)
    diff = {k: (pf[k], ph[k]) for k in pf if pf[k] != ph[k]}
    print(label, "DIFF" if diff else "same", diff)
    for g, tag in [(fresh, "fresh"), (hit, "hit")]:
        try:
            if g.parameterized() or kw.get("theta_dim"):
                th = [1.0] * (kw.get("theta_dim") or g.param_length())
                g.update_weights(th)
            print("   ", tag, "expectation", g.expectation())
        except Exception as e:
            print("   ", tag, "update/expectation ERR", type(e).__name__, str(e)[:150])
