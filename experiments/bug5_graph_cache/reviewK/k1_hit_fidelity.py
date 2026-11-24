"""reviewK k1: does a cache HIT (master's load path: from_serialized +
_Graph copy-constructor, exactly what v3.6 keeps) equal a fresh build?
Run with HOME and PHASIC_CACHE_DIR pointing at a scratch dir."""
import os, sys, tempfile, shutil
assert 'scratchpad' in os.environ['HOME'], os.environ['HOME']
import numpy as np
import phasic
from phasic import Graph
from phasic._graph_serialize import from_serialized
from phasic import phasic_pybind


def via_hit(g):
    """Replicate the hit: serialize -> from_serialized -> _Graph(copy)."""
    data = g.serialize()
    cached = Graph.from_serialized(data)
    h = Graph.__new__(Graph)
    phasic_pybind.Graph.__init__(h, cached)
    h._weight_mode = getattr(cached, '_weight_mode', 'linear')
    h._weight_formula = None; h._weight_formula_tape = None
    h._weight_callback = None
    h.is_discrete = cached.is_discrete
    return h


def describe(tag, g):
    out = dict(n=g.vertices_length(), pl=g.param_length(),
               par=g.parameterized())
    try:
        out['E'] = float(g.expectation())
    except Exception as e:
        out['E'] = f'ERR {type(e).__name__}: {e}'[:90]
    return out


def cb_const(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], n * (n - 1) / 2]]


def cb_param3(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], [n * (n - 1) / 2, 1.0, 0.5]]]


def cb_param2(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], [n * (n - 1) / 2, 0.0]]]


cases = [
    ("constant graph, no theta_dim", cb_const, dict()),
    ("constant graph, theta_dim=3", cb_const, dict(theta_dim=3)),
    ("3 coeffs, theta_dim=2 (aux coeff)", cb_param3, dict(theta_dim=2)),
    ("2 coeffs, no theta_dim", cb_param2, dict()),
]
for tag, cb, kw in cases:
    try:
        f = Graph(cb, ipv=[5], **kw)
    except Exception as e:
        print(tag, 'fresh build ERR', type(e).__name__, e); continue
    try:
        h = via_hit(f)
    except Exception as e:
        print(tag, 'hit ERR', type(e).__name__, e); continue
    df, dh = describe('fresh', f), describe('hit', h)
    print(f"{tag}:\n   fresh {df}\n   hit   {dh}\n   SAME={df == dh}")
    for theta in ([1.0, 2.0, 3.0], [1.0, 2.0], [1.0]):
        rf = rh = None
        for name, g in (('fresh', f), ('hit', h)):
            try:
                g.update_weights(theta)
                r = float(g.expectation())
            except Exception as e:
                r = f'ERR {type(e).__name__}'
            if name == 'fresh': rf = r
            else: rh = r
        print(f"   update_weights({theta}): fresh={rf} hit={rh} "
              f"{'DIFF' if rf != rh else ''}")
