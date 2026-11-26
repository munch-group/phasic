import jax, jax.numpy as jnp, numpy as np
from phasic import Graph
g = Graph(1)
s = g.starting_vertex()
a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([1])
s.add_edge(a, 1.0)
a.add_edge(b, [1.0, 0.0])
b.add_edge(c, [0.0, 1.0])
t = jnp.array([0.5, 1.0, 2.0])
for rw in (None, jnp.array([0.0, 1.0, 2.0, 0.0])):
    for emg in (False, True):
        try:
            m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, exact_moment_grad=emg)
            kw = {} if rw is None else {"rewards": rw}
            for th in ([1.0, 1.0], [1.0, -5e-8], [1.0, 5e-8]):
                th = jnp.array(th)
                try:
                    out = m(th, t, **kw)
                    gr = jax.grad(lambda p: jnp.sum(m(p, t, **kw)[1]))(th)
                    print("rewards" if rw is not None else "plain", "emg", emg, th, "fwd ok", np.asarray(out[1]), "grad", np.asarray(gr))
                except Exception as e:
                    print("rewards" if rw is not None else "plain", "emg", emg, th, "RAISE", type(e).__name__, str(e)[:200])
        except Exception as e:
            print("build RAISE", type(e).__name__, str(e)[:200])
