# reviewAK p5: is expectation() of a constant graph bit-identical between two fresh builds,
# and between a fresh build and its JSON round trip (the cache load path)?
import json, numpy as np, sys
from phasic import Graph, with_ipv
from phasic.graph_cache import _serialize_value, _deserialize_value
sys.path.insert(0, "experiments/bug5_graph_cache/reviewAK")
from p4_roundtrip_fidelity import cb_const, rt
def fresh():
    return Graph(with_ipv([[[6, 0], 0.7], [[5, 1], 0.3]])(cb_const), graph_cache=False)
a, b = fresh(), fresh()
c = rt(fresh())
vals = {k: g.expectation() for k, g in [("fresh_a", a), ("fresh_b", b), ("rt", c)]}
vals2 = {k: g.expectation() for k, g in [("fresh_a", a), ("fresh_b", b), ("rt", c)]}
for k in vals: print(k, repr(vals[k]), "repeat:", repr(vals2[k]))
print("dyn", a.dyn_ordering, c.dyn_ordering)
for k in ["fresh_b", "rt"]:
    print(k, "vs fresh_a rel diff", abs(vals[k] - vals["fresh_a"]) / vals["fresh_a"])
