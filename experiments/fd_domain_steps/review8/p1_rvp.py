import jax; jax.config.update("jax_enable_x64", True)
import jax, jax.numpy as jnp, numpy as np
from phasic import Graph
g = Graph(1)
s = g.starting_vertex()
a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([1]); d=g.find_or_create_vertex([4])
s.add_edge(a, 1.0)
a.add_edge(b, [1.0, 0.0])
a.add_edge(d, [0.0, 1.0])
d.add_edge(c, [1.0, 0.0])
b.add_edge(c, [0.0, 1.0])
nv = g.vertices_length()
rw = np.zeros(nv); rw[d.index()] = 1.0
print("nv", nv)
for th in ([1.0, 1.0], [1.0, -5e-8], [1.0, 5e-8], [1.0, 0.0]):
    th = jnp.array(th)
    try:
        v = g.reward_visit_probability(rw, theta=th)
        print(th, "fwd", float(v))
    except Exception as e:
        print(th, "fwd RAISE", type(e).__name__, str(e)[:200])
    try:
        gr = jax.grad(lambda p: g.reward_visit_probability(rw, theta=p))(th)
        print(th, "grad", np.asarray(gr))
    except Exception as e:
        print(th, "grad RAISE", type(e).__name__, str(e)[:200])
# exact: p = t1/(t0+t1)
for th in ([1.0, 5e-8],):
    t0,t1=th; print("exact grad", -t1/(t0+t1)**2, t0/(t0+t1)**2)
