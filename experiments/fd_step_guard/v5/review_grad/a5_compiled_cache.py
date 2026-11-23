# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""A5 (master): SVGD._compiled_cache key = (id(model), theta_dim, obs_shape, nr_moments, regularization, rewards).
Same model object, same-shape data, DIFFERENT prior / positive_params / observed values (the FreeEpochModel.fit and
"rebind the model later" situations): does the second SVGD reuse the first one's compiled gradient?"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, SVGD, GaussPrior
g = Graph(1); s = g.starting_vertex()
v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
s.add_edge(v2, 1.0); v2.add_edge(v1, [1.0])
m = Graph.pmf_and_moments_from_graph(g, theta_dim=1)
kw = dict(theta_dim=1, n_particles=8, n_iterations=400, progress=False, seed=1, learning_rate=0.3)
obsA = jnp.array([0.2, 0.2, 0.2, 0.2])     # MLE rate 5
obsB = jnp.array([2., 2., 2., 2.])         # MLE rate 0.5
def fit(obs, order):
    sv = SVGD(m, obs, prior=GaussPrior(mean=1.5, std=3.0, lower=0.0), **kw); sv.optimize()
    print(f'   {order}: data mean {float(obs.mean()):5.1f} -> theta_mean {float(np.asarray(sv.theta_mean)[0]):.4f}  cache entries {len(SVGD._compiled_cache)}')
print('A then B on the SAME model object (same data shape):'); fit(obsA, 'A'); fit(obsB, 'B')
SVGD._compiled_cache.clear()
print('B alone (cache cleared):'); fit(obsB, 'B')
