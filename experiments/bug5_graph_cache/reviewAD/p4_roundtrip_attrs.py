# reviewAD p4: fresh build vs hit-path reconstruction (serialize -> JSON -> from_serialized ->
# Graph(cached)) for C-level / Python-level attributes a caller can observe.
import json, numpy as np
from phasic import Graph
from phasic.graph_cache import _serialize_value, _deserialize_value

def const_cb(state):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], n * (n - 1) / 2]]

def par_cb(state):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2, 0.0, 1.0]]]   # 3 coefficients

def hit_like(g):
    enc = json.dumps(_serialize_value(g.serialize()), separators=(",", ":"))
    c = Graph.from_serialized(_deserialize_value(json.loads(enc)))
    return Graph(c)   # the hit path does super().__init__(cached_graph)

def obs(g):
    out = {}
    for m in ("param_length", "parameterized", "vertices_length", "get_was_dph"):
        try: out[m] = getattr(g, m)()
        except Exception as e: out[m] = f"{type(e).__name__}"
    out["is_discrete"] = getattr(g, "is_discrete", None)
    return out

cases = [("const, no theta_dim", const_cb, {}), ("const, theta_dim=2", const_cb, {"theta_dim": 2}),
         ("param 3 coeffs, no theta_dim", par_cb, {}), ("param 3 coeffs, theta_dim=2", par_cb, {"theta_dim": 2})]
for label, cb, kw in cases:
    try:
        f = Graph(cb, ipv=[6], graph_cache=False, **kw)
    except Exception as e:
        print(f"{label:32s} build error {type(e).__name__}: {e}"); continue
    h = hit_like(f)
    of, oh = obs(f), obs(h)
    print(f"{label:32s} fresh={of}\n{'':32s} hit  ={oh}  SAME={of==oh}")
    try:
        if f.parameterized():
            f.update_weights([1.0, 2.0] if kw.get("theta_dim") == 2 else [1.0, 2.0, 3.0])
            h.update_weights([1.0, 2.0] if kw.get("theta_dim") == 2 else [1.0, 2.0, 3.0])
        print(f"{'':32s} expectation fresh={f.expectation()!r} hit={h.expectation()!r}")
    except Exception as e:
        print(f"{'':32s} eval: fresh/hit raised {type(e).__name__}: {e}")
