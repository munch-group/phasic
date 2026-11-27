"""Particles after k SVGD iterations: declared vs undeclared model (direct SVGD)."""
import random, logging
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from phasic import Graph, SVGD


def build_graph():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0); v2.add_edge(v1, [1.0])
    return g


np.random.seed(0); random.seed(0)
gd = build_graph(); gd.update_weights([3.0])
data = np.array(gd.sample(80))
prior = lambda phi: -0.5 * jnp.sum((phi / 10.0) ** 2)
for k in (1, 2, 5):
    out = {}
    for name, doms in (("declared", ["positive"]), ("undeclared", None)):
        m = Graph.pmf_and_moments_from_graph(build_graph(), nr_moments=2, param_domains=doms)
        s = SVGD(model=m, observed_data=data, prior=prior, n_particles=20, n_iterations=k,
                 learning_rate=0.05, positive_params=True, theta_dim=1, seed=42,
                 progress=False, parallel='vmap')
        s.optimize()
        out[name] = np.asarray(s.particles)[:, 0]
        if k == 1:
            print(name, "preconditioner:", type(getattr(s, "_preconditioner", None)).__name__,
                  getattr(s, "preconditioner", None))
    d = np.abs(out["declared"] - out["undeclared"])
    print(f"k={k}: max |dphi|={d.max():.3e}  argmax={d.argmax()} "
          f"decl={out['declared'][d.argmax()]:.6f} undecl={out['undeclared'][d.argmax()]:.6f}")
