"""Batch 2 gate regression: test_svgd_api_parity::test_graph_svgd_and_direct_svgd_agree.

Same data and settings; Graph.svgd declared, Graph.svgd with domains forced
off (undeclared = Batch 1 / master behaviour), and direct SVGD.
"""
import random
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import phasic
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


def shared_prior(phi):
    return -0.5 * jnp.sum((phi / 10.0) ** 2)


kw = dict(observed_data=data, prior=shared_prior, n_particles=20, n_iterations=150,
          learning_rate=0.05, bandwidth='median_per_dim', positive_params=True,
          theta_dim=1, seed=42, verbose=False, progress=False, parallel='vmap')

a = build_graph().svgd(**kw)
print("Graph.svgd declared  ", a.model._param_domains, float(a.theta_mean[0]), float(a.theta_std[0]))
orig = phasic._svgd_fd_domains
phasic._svgd_fd_domains = lambda *x, **k: None
u = build_graph().svgd(**kw)
phasic._svgd_fd_domains = orig
print("Graph.svgd undeclared", u.model._param_domains, float(u.theta_mean[0]), float(u.theta_std[0]))
m = Graph.pmf_and_moments_from_graph(build_graph(), nr_moments=2)
b = SVGD(model=m, **kw); b.optimize()
print("SVGD direct          ", float(b.theta_mean[0]), float(b.theta_std[0]))
for name, s in (("declared", a), ("undeclared", u), ("direct", b)):
    print(name, "init phi[:5]", np.asarray(s.theta_init)[:5, 0])
