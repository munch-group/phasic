# Does a from_serialized round trip (what a cache hit returns) behave like a
# fresh build BEFORE update_weights, for callbacks returning 3-tuples
# (state, weight, coeffs) with a nonzero "vestigial" weight?
import numpy as np
from phasic import Graph, with_ipv

def cb3(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], 2.5, [n * (n - 1) / 2])]

def cb2(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2])]

for name, cb in [("3-tuple", cb3), ("2-tuple", cb2)]:
    g = Graph(cb, ipv=[5])
    h = Graph.from_serialized(g.serialize())
    from phasic import _Graph
    hit = Graph.__new__(Graph)  # mimic the hit path: FIRST base __init__ from the loaded graph
    _Graph.__init__(hit, h)
    hit._weight_mode='linear'; hit._weight_formula_tape=None
    def ex(x):
        try:
            return x.expectation()
        except Exception as e:
            return f"ERR {type(e).__name__}: {e}"
    print(name, "fresh E[T] pre-update:", ex(g), " roundtrip:", ex(h), " hit-like:", ex(hit))
    g.update_weights([1.0]); h.update_weights([1.0])
    print(name, "fresh E[T] post-update:", g.expectation(), " roundtrip:", h.expectation())
