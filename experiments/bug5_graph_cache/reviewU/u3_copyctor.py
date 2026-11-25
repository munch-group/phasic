# reviewU u3: the hit path does super().__init__(cached_graph) (pybind copy
# constructor). Does that copy keep param_length, dyn_ordering, was_dph,
# edge lists of the from_serialized graph?  No cache files written.
import json, numpy as np, phasic
from phasic import Graph
from phasic.graph_cache import _serialize_value
def decode(v):
    if isinstance(v, dict):
        if v.get('__type__') == 'ndarray': return np.array(v['__data__'])
        return {k: decode(x) for k, x in v.items()}
    if isinstance(v, list): return [decode(x) for x in v]
    return v
@phasic.with_ipv([[[5], 1.0]])
def acb(state):
    n = state[0]
    if n <= 1: return []
    return [[[n - 1], [n * (n - 1) / 2, 0.1 * n, 7.0]]]
for td in (None, 2):
    g = Graph(acb) if td is None else Graph(acb, theta_dim=td)
    c = Graph.from_serialized(decode(json.loads(json.dumps(_serialize_value(g.serialize())))))
    h = Graph(c)  # what super().__init__(cached_graph) produces
    for name, x in (("fresh", g), ("from_serialized", c), ("copy", h)):
        print(td, name, "param_length", x.param_length(), "parameterized", x.parameterized(),
              "dyn", x.dyn_ordering, "was_dph", x.get_was_dph(), "n", x.vertices_length(),
              "edges", sum(len(list(v.edges())) for v in x.vertices()))
    try:
        for name, x in (("fresh", g), ("copy", h)):
            x.update_weights([1.0, 2.0]); print(td, name, "E", x.expectation())
    except Exception as e:
        print(td, "update_weights failed", type(e).__name__, e)
