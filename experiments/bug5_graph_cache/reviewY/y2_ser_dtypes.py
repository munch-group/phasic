"""reviewY y2: dtype/shape of every serialize() field, and whether master's
ndarray JSON tag ({'__type__':'ndarray','__data__':tolist()}, no dtype, no
shape) round-trips them. No cache use."""
import json
import numpy as np
from phasic import Graph
from phasic.graph_cache import _serialize_value, _deserialize_value

def report(name, g):
    d = g.serialize()
    enc = json.loads(json.dumps(_serialize_value(d)))
    dec = _deserialize_value(enc)
    print(name)
    for k, v in d.items():
        w = dec[k]
        if isinstance(v, np.ndarray):
            same = (isinstance(w, np.ndarray) and v.dtype == w.dtype and v.shape == w.shape)
            print(f"   {k}: {v.dtype} {v.shape} -> {getattr(w,'dtype',type(w))} {getattr(w,'shape','')}"
                  + ("" if same else "   <-- dtype/shape changed"))
        else:
            print(f"   {k}: {type(v).__name__} {v!r:.40} -> {type(w).__name__}")
    try:
        h = Graph.from_serialized(dec)
        print("   from_serialized OK; expectation", g.expectation(), h.expectation())
    except Exception as e:
        print("   from_serialized FAILED:", type(e).__name__, e)

def cb_par(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2])]

def cb_const(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2)]

report("parameterized", Graph(cb_par, ipv=[4]))
report("constant", Graph(cb_const, ipv=[4]))
