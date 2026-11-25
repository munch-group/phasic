"""reviewM m2: theta_dim < coefficient length (aux coefficients, documented for
callback/formula weight modes in Graph.__init__'s docstring). Does a cache hit keep
the aux coefficients?  HOME=scratch; uses master's graph_cache=True for a real hit."""
import os, numpy as np
assert os.environ["HOME"].startswith("/private/tmp/claude-501/"), os.environ["HOME"]
import phasic
from phasic import Graph

def cbD(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [[[n - 1], [n * (n - 1) / 2, 1.0, 5.0]]]

phasic.GraphCache().clear()
mk = lambda c: Graph(cbD, ipv=[[[4], 1.0]], theta_dim=2, graph_cache=c)
fresh = mk(False)
mk(True)
hit = mk(True)
e = next(iter(fresh.vertices())).edges()[0]
print("edge methods:", [m for m in dir(e) if "coef" in m.lower()])
for name, g in (("fresh", fresh), ("hit", hit)):
    v = [v for v in g.vertices() if v.state()[0] == 4][0]
    ed = v.edges()[0]
    print(name, "param_length", g.param_length(), "coefficients_length", ed.coefficients_length(),
          "coefficients", list(ed.coefficients()) if hasattr(ed, "coefficients") else None)
    ser = g.serialize()
    print(name, "serialize param_edges shape", ser["param_edges"].shape)
    # callback weight mode reading the aux coefficient c[2]
    try:
        g.update_weights([1.0, 1.0], callback=lambda th, c: c[0] * th[0] + c[2])
        print(name, "callback-mode expectation", float(g.expectation()))
    except Exception as ex:
        print(name, "callback-mode ERR", type(ex).__name__, str(ex)[:120])
