"""Review C p10: a with_ipv-wrapped functools.partial builds (no cache), and
its __wrapped__ has no __name__ (plan 2.6 callback_name)."""
from functools import partial
from phasic import Graph, with_ipv
def model(state, k=1.0, **kw):
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], [k * n]]]
cb = with_ipv([5])(partial(model, k=2.0))
g = Graph(cb, graph_cache=False)
print("built vertices:", g.vertices_length())
print("__wrapped__ type:", type(cb.__wrapped__).__name__, "has __name__:", hasattr(cb.__wrapped__, "__name__"))
