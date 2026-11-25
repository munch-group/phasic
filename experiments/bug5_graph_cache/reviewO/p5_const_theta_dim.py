"""reviewO p5: constant-rate callback built with an explicit theta_dim:
fresh build vs Graph(Graph.from_serialized(g.serialize())) (= the hit path).
graph_cache=False: master's cache not involved."""
from phasic import Graph

def coal_const(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], n * (n - 1) / 2)]

def coal_param(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [([n - 1], [n * (n - 1) / 2, 0.0])]

for name, cb, td in (
                     ("param 2 coefs, theta_dim=1", coal_param, 1),):
    g = Graph(cb, ipv=[4], theta_dim=td, graph_cache=False)
    h = Graph(Graph.from_serialized(g.serialize()))
    print(f"--- {name}")
    print("  param_length fresh/hit:", g.param_length(), h.param_length(),
          "| parameterized:", g.parameterized(), h.parameterized())
    for gg, tag in ((g, "fresh"), (h, "hit  ")):
        for theta in ([2.0], [2.0, 3.0]):
            try:
                gg.update_weights(theta)
                r = f"E={gg.expectation():.6g}"
            except Exception as e:
                r = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"  {tag} update_weights({theta}) -> {r}")
