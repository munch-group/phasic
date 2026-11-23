"""A6 (master): fixed=[(1, 0.0)] under positive_params=True -- the theta the MODEL receives during optimize()."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, SVGD, GaussPrior
g = Graph(1); s = g.starting_vertex()
v3 = g.find_or_create_vertex([3]); v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
s.add_edge(v3, 1.0); v3.add_edge(v2, [1.0, 0.0]); v2.add_edge(v1, [1.0, 1.0])   # second rate = t0 + t1
base = Graph.pmf_and_moments_from_graph(g, theta_dim=2)
for val in (0.0, 1e-12, 0.25):
    seen = []
    def spy(theta, times, rewards=None, _s=seen):
        jax.debug.callback(lambda t: _s.append(np.asarray(t).copy()), theta)
        return base(theta, times, rewards=rewards)
    sv = SVGD(spy, jnp.array([0.5, 1.0, 2.0]), prior=[GaussPrior(mean=1.0, std=1.0, lower=0.0), None], theta_dim=2,
              fixed=[(1, val)], n_particles=4, n_iterations=2, progress=False, jit=False, precompile=False)
    sv.optimize()
    a = np.array([x for x in seen if x.ndim == 1])
    print(f'fixed=[(1, {val})]: theta[1] received by the model: min {a[:,1].min()!r} max {a[:,1].max()!r}')
