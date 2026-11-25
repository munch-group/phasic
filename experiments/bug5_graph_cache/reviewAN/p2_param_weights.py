# reviewAN p2: does a serialize()/from_serialized round trip (the cache hit path:
# load -> Graph.from_serialized -> Graph.__init__ super().__init__(cached)) reproduce
# the writer graph's EDGE WEIGHTS before update_weights? serialize() omits them.
import os
assert os.environ["HOME"].startswith("/private/tmp/")
import numpy as np, phasic
from phasic import Graph, with_ipv
from phasic import _Graph

def cb3(state):          # 3-tuple form: (state, w, coeffs) -- w documented "vestigial"
    n = state[0]
    if n <= 1: return []
    return [([n - 1], 7.0, [n * (n - 1) / 2, 1.0])]

def cb2(state):          # 2-tuple parameterized form
    n = state[0]
    if n <= 1: return []
    return [([n - 1], [n * (n - 1) / 2, 1.0])]

def cbc(state):          # constant
    n = state[0]
    if n <= 1: return []
    return [([n - 1], n * (n - 1) / 2)]

def weights(g):
    out = []
    for i in range(g.vertices_length()):
        v = g.vertex_at(i)
        out.append([round(e.weight(), 12) for e in v.edges()])
    return out

for name, cb in [("3-tuple", cb3), ("2-tuple", cb2), ("constant", cbc)]:
    fresh = Graph(cb, ipv=[5])
    rt = Graph.from_serialized(fresh.serialize())
    hit = Graph(rt)             # what the hit path does: super().__init__(cached_graph)
    wf, wh = weights(fresh), weights(hit)
    print(name, "parameterized:", fresh.parameterized(), hit.parameterized())
    print("  fresh weights:", wf)
    print("  hit   weights:", wh)
    print("  equal:", wf == wh)
    for g, lab in [(fresh, "fresh"), (hit, "hit")]:
        try:
            print("  ", lab, "expectation() w/o update_weights:", g.expectation())
        except Exception as e:
            print("  ", lab, "expectation() raised", type(e).__name__, str(e)[:120])
    if fresh.parameterized():
        fresh.update_weights([1.0, 0.5]); hit.update_weights([1.0, 0.5])
        print("   after update_weights equal:", weights(fresh) == weights(hit), fresh.expectation(), hit.expectation())
