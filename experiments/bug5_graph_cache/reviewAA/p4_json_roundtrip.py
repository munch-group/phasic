"""reviewAA p4: JSON round trip of serialize() through master's
_serialize_value -> json -> ndarray-tag decode (the format the plan keeps),
for graph shapes with empty edge blocks.  Any failure would become a
permanent GraphCacheLoadError on every hit under the plan's policy."""
import json
import numpy as np
import phasic
phasic.configure(graph_cache=False)
from phasic import Graph
from phasic.graph_cache import _serialize_value


def decode(v):  # restricted decoder: plain JSON + ndarray tag
    if isinstance(v, list):
        return [decode(x) for x in v]
    if isinstance(v, dict):
        if v.get("__type__") == "ndarray":
            return np.array(decode(v["__data__"]))
        return {k: decode(x) for k, x in v.items()}
    return v


def check(name, g):
    d = g.serialize()
    s = json.dumps(_serialize_value(d), separators=(",", ":"))
    back = decode(json.loads(s))
    try:
        h = Graph.from_serialized(back)
    except Exception as ex:
        print(f"[{name}] from_serialized RAISES {type(ex).__name__}: {str(ex)[:200]}")
        return
    same = all(np.array_equal(np.asarray(d[k]), np.asarray(h.serialize()[k]))
               for k in d if isinstance(d[k], np.ndarray))
    print(f"[{name}] ok n={h.vertices_length()} param_length {g.param_length()}/{h.param_length()} "
          f"parameterized {g.parameterized()}/{h.parameterized()} arrays equal: {same}")
    try:
        print(f"   expectation fresh/hit: {g.expectation()} / {h.expectation()}")
    except Exception as ex:
        print(f"   expectation raises {type(ex).__name__}: {str(ex)[:100]}")


def cb_absorb(state, **kw):  # every state absorbing: no edges at all
    return []
check("absorbing_only", Graph(cb_absorb, ipv=[([0], 1.0)]))

def cb_const(state, **kw):
    return [([state[0] + 1], 1.5)] if state[0] < 3 else []
check("constant", Graph(cb_const, ipv=[([0], 1.0)]))

def cb_par(state, **kw):
    return [([state[0] + 1], [1.0, 2.0])] if state[0] < 3 else []
g = Graph(cb_par, ipv=[([0], 1.0)])
g.update_weights([1.0, 1.0])
check("param", g)

def cb_sub(state, **kw):  # defective ipv (sum < 1)
    return [([state[0] + 1], 1.0)] if state[0] < 1 else []
check("defective_ipv", Graph(cb_sub, ipv=[([0], 0.5)]))

def cb_multi(state, **kw):
    return [([state[0] + 1, state[1]], 1.0)] if state[0] < 1 else []
check("two_start_states", Graph(cb_multi, ipv=[([0, 0], 0.5), ([0, 1], 0.5)]))
