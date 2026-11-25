"""reviewW w2: is a pybind-backed Graph still usable when Graph.__init__ raises
AFTER super().__init__ (the plan's e.graph-on-save-failure design)?"""
import os
assert "/scratchpad/" in os.environ["HOME"]
import phasic
from phasic import Graph

class Boom(Exception):
    pass

class G2(Graph):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        e = Boom("save failed")
        e.graph = self
        raise e

def cb(state, **kw):
    n = state[0]
    return [([n - 1], [n * (n - 1) / 2.0])] if n > 1 else []

try:
    G2(cb, [5], graph_cache=False)
except Boom as e:
    g = e.graph
    print("type", type(g).__name__, "vertices", g.vertices_length())
    g.update_weights([1.0])
    print("expectation", g.expectation())
    ref = Graph(cb, [5], graph_cache=False); ref.update_weights([1.0])
    print("ref expectation", ref.expectation())
    s = g.serialize()
    print("serialize keys", sorted(s)[:4])
    del e
    import gc; gc.collect()
    print("after gc vertices", g.vertices_length())
