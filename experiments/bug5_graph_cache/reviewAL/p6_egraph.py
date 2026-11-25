"""reviewAL p6: is a pybind Graph whose Python __init__ raised AFTER super().__init__
usable when recovered from the exception (the plan's e.graph on a failed save)?"""
import os, gc
assert os.environ["HOME"].startswith("/private/tmp/")
from phasic import Graph

def cb(state, **kw):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], n * (n - 1) / 2.0]]

class G2(Graph):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        e = OSError(13, "simulated save failure")
        e.graph = self
        e.add_note("built graph attached as e.graph")
        raise e

try:
    G2(cb, ipv=[6], graph_cache=False)
except OSError as e:
    g = e.graph
    ref = Graph(cb, ipv=[6], graph_cache=False)
    print("recovered type", type(g).__name__, "vertices", g.vertices_length(),
          "expectation", g.expectation(), "fresh", ref.expectation())
    s1, s2 = g.serialize(), ref.serialize()
    import numpy as np
    print("serialize equal:", all(np.array_equal(np.asarray(s1[k]), np.asarray(s2[k])) for k in s1))
    print("notes:", e.__notes__)
del g
gc.collect()
print("exit clean")
