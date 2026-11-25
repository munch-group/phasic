"""reviewX x5: theta_dim passed with a CONSTANT-weight callback: does the
hit-path construction (serialize -> from_serialized) reproduce the fresh
build's param_length / parameterized() / update_weights behaviour?"""
from phasic import Graph
def cb(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [([n - 1], n * (n - 1) / 2)]
for td in (None, 1, 2, 3):
    kw = {} if td is None else {"theta_dim": td}
    try:
        g = Graph(cb, ipv=[4], graph_cache=False, **kw)
    except Exception as e:
        print(f"theta_dim={td}: fresh build error {type(e).__name__}: {str(e)[:100]}"); continue
    h = Graph.from_serialized(g.serialize())
    print(f"theta_dim={td}: param_length fresh {g.param_length()} hit {h.param_length()}; "
          f"parameterized {g.parameterized()} {h.parameterized()}")
    for name, G in (("fresh", g), ("hit", h)):
        try:
            G.update_weights([1.0] * (td or 1)); r = "ok"
        except Exception as e:
            r = f"{type(e).__name__}: {str(e)[:90]}"
        print(f"   update_weights({[1.0]*(td or 1)}) on {name}: {r}; E[T]={G.expectation()}")
