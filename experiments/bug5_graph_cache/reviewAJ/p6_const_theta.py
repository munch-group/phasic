"""reviewAJ p6: constant callback with theta_dim given: fresh vs hit-path (no cache files)."""
import os
assert "scratchpad" in os.environ["HOME"]
from phasic import Graph
def cb(state):
    k = state[0]
    return [] if k <= 1 else [([k - 1], float(k))]
for td in (None, 2):
    kw = {} if td is None else {"theta_dim": td}
    g = Graph(cb, ipv=[[[4], 1.0]], **kw)
    h = Graph(Graph.from_serialized(g.serialize()))
    print("theta_dim", td, "param_length fresh", g.param_length(), "hit", h.param_length(),
          "parameterized", g.parameterized(), h.parameterized(), "E", g.expectation(), h.expectation())
    for gg, nm in ((g, "fresh"), (h, "hit")):
        try:
            gg.update_weights([1.0, 2.0]); print("   ", nm, "update_weights([1,2]) ok, E", gg.expectation())
        except Exception as e:
            print("   ", nm, "update_weights([1,2]) raised", type(e).__name__, str(e)[:90])
