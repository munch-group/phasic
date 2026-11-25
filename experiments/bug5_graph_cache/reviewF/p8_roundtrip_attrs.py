"""Review F p8: attributes of a fresh callback build vs Graph.from_serialized
(the object a cache hit copies from), for constant / parameterized / theta_dim."""
from phasic import Graph


def const_cb(state, **kw):
    n = state[0]
    return [] if n <= 1 else [([n - 1], n * (n - 1) / 2)]


def param_cb(state, **kw):
    n = state[0]
    return [] if n <= 1 else [([n - 1], [n * (n - 1) / 2, 1.0])]


def attrs(g):
    return dict(param_length=g.param_length(), parameterized=g.parameterized(),
                was_dph=g.get_was_dph(), is_discrete=g.is_discrete,
                weight_mode=g._weight_mode, n=g.vertices_length(),
                states=[list(g.vertex_at(i).state()) for i in range(g.vertices_length())])


for label, mk in [("constant", lambda: Graph(const_cb, ipv=[4], graph_cache=False)),
                  ("param 2 coeffs", lambda: Graph(param_cb, ipv=[4], graph_cache=False)),
                  ("param theta_dim=1", lambda: Graph(param_cb, ipv=[4], theta_dim=1, graph_cache=False))]:
    g = mk()
    h = Graph.from_serialized(g.serialize())
    a, b = attrs(g), attrs(h)
    diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    print(f"{label:18s} differences fresh vs from_serialized: {diff or 'none'}")
