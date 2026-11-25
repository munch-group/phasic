# reviewAN p4: is a pybind-backed Graph usable after its __init__ raised (e.graph pattern)?
import os, gc
assert os.environ["HOME"].startswith("/private/tmp/")
import numpy as np, phasic
from phasic import Graph
def cb(state):
    n = state[0]
    if n <= 1: return []
    return [([n - 1], n * (n - 1) / 2)]
class G(Graph):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        e = KeyboardInterrupt("simulated during save")
        e.graph = self
        raise e
try:
    G(cb, ipv=[6])
except KeyboardInterrupt as e:
    g = e.graph
gc.collect()
ref = Graph(cb, ipv=[6])
print("usable:", g.vertices_length(), g.expectation(), "ref:", ref.vertices_length(), ref.expectation())
s1, s2 = g.serialize(), ref.serialize()
print("serialize equal:", all((np.array_equal(s1[k], s2[k]) if isinstance(s1[k], np.ndarray) else s1[k] == s2[k]) for k in s1))
