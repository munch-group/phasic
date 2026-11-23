"""A1 (master): where the domain is unknown / decided late / differs from what the transform gives.
 1. single BetaPrior (not a list): transform used for the MODEL vs transform given to the PRIOR
 2. fixed value 0.0 (and 1e-12) under positive_params: theta the model sees
 3. [GaussPrior, BetaPrior] + exposure on the Beta slot: theta the model sees (outer exposure wrapper)
 4. tied slave under a BetaPrior master: per-dim transform of the slave slot at SVGD level
"""
import warnings, logging
warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, SVGD, BetaPrior, GaussPrior
from phasic.svgd import _sigmoid

def g2():
    g = Graph(1); s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3]); v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0); v3.add_edge(v2, [1.0, 0.0]); v2.add_edge(v1, [0.0, 1.0])
    return g
def g1():
    g = Graph(1); s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0); v2.add_edge(v1, [1.0])
    return g
obs = jnp.array([0.5, 1.0, 2.0])

print('## 1. single BetaPrior, theta_dim=1, positive_params=True')
m = Graph.pmf_and_moments_from_graph(g1(), theta_dim=1)
p = BetaPrior(ci=(0.05, 0.5))
sv = SVGD(m, obs, prior=p, theta_dim=1, n_particles=4, n_iterations=1, progress=False)
phi = jnp.array([5.0])
print('   model sees param_transform(5.0) =', float(sv.param_transform(phi)[0]),
      '| prior._transform is sigmoid:', p._transform is _sigmoid, '-> prior evaluates at', float(p._transform(phi)[0]))

print('## 2. fixed values under positive_params (theta the model sees = param_transform(fixed_values))')
m2 = Graph.pmf_and_moments_from_graph(g2(), theta_dim=2)
for val in (0.0, 1e-12, 1e-20, 0.3):
    try:
        sv = SVGD(m2, obs, prior=[GaussPrior(ci=[0.5, 2], lower=0.0), None], theta_dim=2, fixed=[(1, val)],
                  n_particles=4, n_iterations=1, progress=False)
        print(f'   fixed={val!r}: phi={float(sv.fixed_values[1])!r} -> theta seen by model = {float(sv.param_transform(sv.fixed_values)[1])!r}')
    except Exception as e:
        print(f'   fixed={val!r}: {type(e).__name__}: {str(e)[:150]}')

print('## 3. [Gauss, Beta] + exposure on the Beta slot: thetas reaching the model')
seen = []
base = Graph.pmf_and_moments_from_graph(g2(), theta_dim=2)
def spy(theta, times, rewards=None):
    jax.debug.callback(lambda t: seen.append(np.asarray(t).copy()), theta)
    return base(theta, times, rewards=rewards)
try:
    sv = SVGD(spy, obs, prior=[GaussPrior(ci=[0.5, 2], lower=0.0), BetaPrior(ci=(0.2, 0.8))], theta_dim=2,
              exposure=jnp.array([1.0, 5.0, 50.0]), exposure_param_index=1,
              n_particles=4, n_iterations=1, progress=False, jit=False, precompile=False)
    sv.optimize()
    a = np.array([s for s in seen if s.ndim == 1])
    print('   max theta[1] seen by the model:', a[:, 1].max(), '(declared (0,1) at SVGD level; > 1 inside the model)')
except Exception as e:
    print('   ', type(e).__name__, str(e)[:300])
