"""reviewAF p4: does a cache HIT (master's restore sequence, which the plan keeps) equal a fresh
build for callback graphs with an explicit theta_dim?  Uses master's cache under a scratch HOME
(DEFAULT_CACHE_DIR monkeypatched into the scratchpad); files are deleted at the end.
"""
import os, shutil, tempfile, pathlib
assert os.environ["HOME"].startswith("/private/tmp/")
import numpy as np
import phasic
import phasic.graph_cache as gc
from phasic import Graph

tmp = pathlib.Path(tempfile.mkdtemp(dir=os.environ["HOME"]))
gc.DEFAULT_CACHE_DIR = tmp / "graphs"

def const_cb(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2)]

def param_cb(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2, 1.0])]

def mixed_cb(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    if n % 2:
        return [([n - 1], n * 1.0)]
    return [([n - 1], [n * (n - 1) / 2])]

def describe(g):
    d = {"V": g.vertices_length(), "param_length": g.param_length(), "parameterized": g.parameterized(),
         "was_dph": g.get_was_dph(), "is_discrete": g.is_discrete, "dyn": g.dyn_ordering}
    return d

def probe(label, cb, **kw):
    for f in tmp.glob("graphs/*.json"):
        f.unlink()
    try:
        fresh = Graph(cb, ipv=[5], graph_cache=False, **kw)
    except Exception as e:
        print(f"{label}: fresh build raises {type(e).__name__}: {e}")
        return
    try:
        Graph(cb, ipv=[5], graph_cache=True, **kw)      # miss -> save
        files = list(tmp.glob("graphs/*.json"))
        hit = Graph(cb, ipv=[5], graph_cache=True, **kw)  # hit
    except Exception as e:
        print(f"{label}: cached path raises {type(e).__name__}: {e}")
        return
    df, dh = describe(fresh), describe(hit)
    print(f"{label}: entry files={len(files)}")
    print(f"   fresh {df}")
    print(f"   hit   {dh}   {'SAME' if df == dh else 'DIFFERENT'}")
    for theta in ([2.0], [2.0, 3.0], [2.0, 3.0, 4.0]):
        out = []
        for g in (fresh, hit):
            try:
                g.update_weights(theta)
                out.append(round(float(g.expectation()), 10))
            except Exception as e:
                out.append(f"{type(e).__name__}")
        print(f"   update_weights({theta}) -> expectation fresh={out[0]} hit={out[1]} {'SAME' if out[0] == out[1] else 'DIFFERENT'}")

probe("A const_cb theta_dim=2", const_cb, theta_dim=2)
probe("B param_cb (2 coeffs) theta_dim=None", param_cb)
probe("C param_cb (2 coeffs) theta_dim=1", param_cb, theta_dim=1)
probe("D mixed_cb theta_dim=None", mixed_cb)
probe("E const_cb theta_dim=None", const_cb)
shutil.rmtree(tmp)
print("cleaned", not tmp.exists())
