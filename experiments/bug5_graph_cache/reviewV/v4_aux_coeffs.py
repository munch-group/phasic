"""reviewV v4: theta_dim < coefficient length (documented 'auxiliary
coefficients' for callback / formula weight modes, Graph.__init__
docstring). Does a graph that went through serialize -> from_serialized
(what every cache hit does, master and the plan alike) keep the
auxiliary coefficients?  Master hit path, scratch HOME."""
import os
assert os.environ["HOME"].startswith("/private/tmp/"), os.environ["HOME"]
import numpy as np
import phasic
from phasic import Graph

def cb(state, **kw):
    k = state[0]
    if k <= 1:
        return []
    # theta_dim=1: coefficient 0 is multiplied by theta, coefficient 1 is aux data
    return [([k - 1], [k * (k - 1) / 2, float(k)])]

w = phasic.with_ipv([5])(cb)
g = Graph(w, theta_dim=1, graph_cache=False)
Graph(w, theta_dim=1, graph_cache=True)
h = Graph(w, theta_dim=1, graph_cache=True)
print("param_length fresh/hit:", g.param_length(), h.param_length())
cl = lambda gg: sorted({e.coefficients_length() for v in gg.vertices() for e in v.edges()})
print("coefficient lengths fresh/hit:", cl(g), cl(h))
s_g, s_h = g.serialize(), h.serialize()
print("serialize param_edges shape fresh/hit:", s_g["param_edges"].shape, s_h["param_edges"].shape)

def wcb(theta, coeffs):
    # uses the auxiliary coefficient
    return coeffs[0] * theta[0] + (coeffs[1] if len(coeffs) > 1 else 0.0)

for lab, gg in (("fresh", g), ("hit", h)):
    try:
        gg.update_weights([2.0], wcb)
        ws = [round(e.weight(), 6) for v in gg.vertices() for e in v.edges()]
        print(f"  {lab}: weights {ws}  E={gg.expectation():.6f}")
    except Exception as ex:
        print(f"  {lab}: ERR {type(ex).__name__}: {str(ex)[:200]}")

for lab, gg in (("fresh", g), ("hit", h)):
    try:
        gg.weight_formula = "t0*c0 + c1"
        gg.update_weights([2.0])
        print(f"  formula {lab}: E={gg.expectation():.6f}")
    except Exception as ex:
        print(f"  formula {lab}: ERR {type(ex).__name__}: {str(ex)[:200]}")
