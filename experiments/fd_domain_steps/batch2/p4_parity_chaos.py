"""Is the parity fit chaotic? Undeclared (master) model only: perturb the
initial particles by 1e-9 and compare the posterior std; also the declared vs
undeclared divergence over iterations."""
import random
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


def fit(doms, k, init=None):
    m = Graph.pmf_and_moments_from_graph(build_graph(), nr_moments=2, param_domains=doms)
    kw = dict(theta_init=init) if init is not None else dict(n_particles=20)
    s = SVGD(model=m, observed_data=data, prior=prior, n_iterations=k, learning_rate=0.05,
             positive_params=True, theta_dim=1, seed=42, progress=False, parallel='vmap', **kw)
    s.optimize()
    return s


base = fit(None, 1)
init = np.asarray(base.theta_init)
for k in (10, 25, 50, 100, 150):
    a = fit(["positive"], k, init); b = fit(None, k, init)
    print(f"declared vs undeclared k={k:3d}: max|dphi|={np.max(np.abs(np.asarray(a.particles)-np.asarray(b.particles))):.2e} "
          f"std {float(a.theta_std[0]):.4f} vs {float(b.theta_std[0]):.4f}", flush=True)
rng = np.random.default_rng(0)
for j in range(4):
    pert = init + rng.normal(scale=1e-9, size=init.shape)
    s = fit(None, 150, pert)
    print(f"undeclared, init + 1e-9 noise #{j}: mean {float(s.theta_mean[0]):.4f} std {float(s.theta_std[0]):.4f}", flush=True)
