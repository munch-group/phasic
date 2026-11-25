"""reviewAJ p5: hit-path fidelity for theta_dim != coefficient length (no cache files written)."""
import os
assert "scratchpad" in os.environ["HOME"]
import numpy as np
from phasic import Graph

def cb2(state):
    k = state[0]
    if k <= 1:
        return []
    return [([k - 1], [k * (k - 1) / 2, 1.0])]   # 2 coefficients

def hit_like(g):
    return Graph(Graph.from_serialized(g.serialize()))

for td in (None, 1, 2, 3):
    kw = {} if td is None else {"theta_dim": td}
    try:
        g = Graph(cb2, ipv=[[[4], 1.0]], **kw)
    except Exception as e:
        print("theta_dim", td, "fresh build raised", type(e).__name__, str(e)[:100]); continue
    try:
        h = hit_like(g)
    except Exception as e:
        print("theta_dim", td, "round trip raised", type(e).__name__, str(e)[:100]); continue
    print("theta_dim", td, "param_length fresh", g.param_length(), "hit", h.param_length())
    for theta in ([2.0], [2.0, 0.5], [2.0, 0.5, 9.0]):
        res = []
        for gg in (g, h):
            try:
                gg.update_weights(theta); res.append(round(gg.expectation(), 10))
            except Exception as e:
                res.append(type(e).__name__)
        print("   theta", theta, "fresh", res[0], "hit", res[1], "SAME" if res[0] == res[1] else "DIFFERENT")
