"""Review D probe 8: feasibility of plan test 15's formula-mode entry: can a
graph with weight_formula set be encoded with the entry encoder
(_serialize_value + json) and decoded with plain JSON + the ndarray tag only,
then rebuilt by from_serialized? Also counts _set_weight_tape calls on a
simulated hit (from_serialized + restore on a second object). In memory only."""
import os, json
assert "scratchpad" in os.environ["HOME"]
import numpy as np, phasic
from phasic.graph_cache import _serialize_value

def cb(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2, 1.0]]]
g = phasic.Graph(cb, ipv=[[[4], 1.0]], graph_cache=False, theta_dim=1)
g.weight_formula = "t0*c0 + c1"
enc = json.loads(json.dumps(_serialize_value(g.serialize())))
tags = set()
def dec(v):
    if isinstance(v, list): return [dec(x) for x in v]
    if isinstance(v, dict):
        if "__type__" in v:
            tags.add(v["__type__"]); assert v["__type__"] == "ndarray"; return np.array(v["__data__"])
        return {k: dec(x) for k, x in v.items()}
    return v
data = dec(enc)
print("tags seen:", tags, "| tape keys:", sorted(data.get("weight_formula_tape", {}).keys()))
calls = []
orig = phasic.Graph._set_weight_tape
def spy(self, *a):
    calls.append(type(self).__name__); return orig(self, *a)
phasic.Graph._set_weight_tape = spy
h = phasic.Graph.from_serialized(data)
print("_set_weight_tape calls during from_serialized:", len(calls), "weight_mode:", h._weight_mode)
g.update_weights([2.0]); h.update_weights([2.0])
print("E fresh-with-formula / round-trip:", g.expectation(), h.expectation())
