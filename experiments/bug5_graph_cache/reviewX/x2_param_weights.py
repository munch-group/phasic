"""reviewX x2: fresh build vs serialize()/from_serialized() (the hit path):
edge weights / attributes BEFORE update_weights, for parameterized callbacks
in the 2-tuple and 3-tuple forms (3-tuple weight 'vestigial'), with and without
theta_dim larger/smaller than the coefficient length."""
import numpy as np
from phasic import Graph

def snap(g):
    out = []
    for v in g.vertices():
        es = v.parameterized_edges() if g.parameterized() else v.edges()
        out.append([(tuple(e.to().state()), float(e.weight()),
                     tuple(e.edge_state(e.coefficients_length())) if hasattr(e, 'edge_state') else ())
                    for e in es])
    return out

def cb2(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [([n - 1], [n * (n - 1) / 2, 1.0])]

def cb3(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [([n - 1], 7.5, [n * (n - 1) / 2, 1.0])]

for name, cb, kw in [("2tuple", cb2, {}), ("3tuple", cb3, {}),
                     ("2tuple_td1", cb2, {"theta_dim": 1}),
                     ("2tuple_td3", cb2, {"theta_dim": 3})]:
    try:
        g = Graph(cb, ipv=[4], graph_cache=False, **kw)
        h = Graph.from_serialized(g.serialize())
        print(f"[{name}] param_length fresh {g.param_length()} hit {h.param_length()}; "
              f"parameterized {g.parameterized()} {h.parameterized()}")
        print(f"[{name}] edges identical before update_weights: {snap(g) == snap(h)}")
        if snap(g) != snap(h):
            print("   fresh", snap(g)[:3]); print("   hit  ", snap(h)[:3])
        for attr in ("dyn_ordering",):
            print(f"[{name}] {attr} {getattr(g, attr)} {getattr(h, attr)}")
        try:
            a = g.expectation(); b = h.expectation()
            print(f"[{name}] expectation before update_weights fresh {a!r} hit {b!r}")
        except Exception as e:
            print(f"[{name}] expectation before update_weights error {type(e).__name__}: {str(e)[:120]}")
        th = [1.0] * max(1, g.param_length())
        for G in (g, h):
            try: G.update_weights(th)
            except Exception as e: print(f"[{name}] update_weights error {type(e).__name__}: {str(e)[:120]}")
        try:
            print(f"[{name}] expectation after theta=1 fresh {g.expectation()!r} hit {h.expectation()!r}")
        except Exception as e:
            print(f"[{name}] expectation after error {type(e).__name__}: {str(e)[:120]}")
    except Exception as e:
        print(f"[{name}] error {type(e).__name__}: {str(e)[:200]}")
