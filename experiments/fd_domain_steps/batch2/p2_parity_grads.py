"""Per-particle gradients of the parity model: declared 'positive' vs undeclared."""
import random
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from phasic import Graph


def build_graph():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0); v2.add_edge(v1, [1.0])
    return g


np.random.seed(0); random.seed(0)
gd = build_graph(); gd.update_weights([3.0])
data = jnp.asarray(np.array(gd.sample(80)))
md = Graph.pmf_and_moments_from_graph(build_graph(), nr_moments=2, param_domains=["positive"])
mu = Graph.pmf_and_moments_from_graph(build_graph(), nr_moments=2)
L = lambda m: (lambda th: jnp.sum(jnp.log(m(th, data)[0])))
th = jnp.asarray([[0.3], [1.0], [2.8], [3.0], [10.0], [50.0], [300.0]])
for name, f in (("eager", lambda m: jnp.stack([jax.grad(L(m))(t) for t in th])),
                ("jitvmap", lambda m: jax.jit(jax.vmap(jax.grad(L(m))))(th))):
    gdd = np.asarray(f(md))[:, 0]; guu = np.asarray(f(mu))[:, 0]
    for t, a, b in zip(np.asarray(th)[:, 0], gdd, guu):
        print(f"{name:7s} theta={t:7.2f} declared={a: .10e} undeclared={b: .10e} rel={abs(a-b)/abs(b):.2e}")
