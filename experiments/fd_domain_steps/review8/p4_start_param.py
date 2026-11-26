import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from phasic import Graph
def build():
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([1])
    s.add_edge(a, [1.0, 0.0, 0.0])      # parameterized start edge
    s.add_edge(b, [0.0, 0.0, 1.0])
    a.add_edge(b, [0.0, 1.0, 0.0])
    b.add_edge(c, [0.0, 1.0, 0.0])
    return g
g = build()
print("serialize start_param_edges", g.serialize()['start_param_edges'])
t = jnp.array([0.5, 1.0, 2.0])
builders = {
 "pmf_and_moments": lambda: Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=3),
 "pmf_from_graph": lambda: Graph.pmf_from_graph(g, theta_dim=3),
}
for name, mk in builders.items():
    try:
        m = mk()
    except Exception as e:
        print(name, "build RAISE", type(e).__name__, str(e)[:200]); continue
    for th in ([0.5, 1.0, 0.5], [5e-8, 1.0, 1.0], [-5e-8, 1.0, 1.0], [0.0, 1.0, 1.0]):
        th = jnp.array(th)
        def L(p):
            out = m(p, t)
            return jnp.sum(out[0]) if isinstance(out, tuple) else jnp.sum(out)
        try:
            v = L(th); print(name, th, "fwd", float(v))
        except Exception as e:
            print(name, th, "fwd RAISE", type(e).__name__, str(e)[:160])
        try:
            gr = jax.grad(L)(th); print(name, th, "grad", np.asarray(gr))
        except Exception as e:
            print(name, th, "grad RAISE", type(e).__name__, str(e)[:160])
