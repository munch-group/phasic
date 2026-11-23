"""Shared model builders for the FD-site experiments (no source changes)."""
import sys, warnings
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
warnings.filterwarnings('ignore')
from test_notebook_model_recovery import coalescent_indexed, two_island_graph
from phasic import Graph, GaussPrior

EPOCHS = np.array([0.0, 0.15])
OBS = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0]]

def joint():
    g, idx = coalescent_indexed(4)
    J = g.joint_prob_graph(idx, reward_limit=3, discrete=False)
    oi = J._map_joint_observations_to_indices(OBS)
    return J, oi

def _ll(model):
    def f(theta):
        out = model(theta)
        p = out[0] if isinstance(out, tuple) else out
        return jnp.sum(jnp.log(p))
    return f

def site_joint_index(baked=True):
    J, oi = joint()
    m = Graph.pmf_from_graph_joint_index(J, theta_dim=2,
                                         observed_indices=oi if baked else None)
    vi = jnp.asarray(oi)
    def f(theta):
        p, _ = m(theta, vi)
        return jnp.sum(jnp.log(p))
    return f

def site_daisy(exposure=False, efg=False):
    J, oi = joint()
    kw = {}
    if exposure:
        kw = dict(exposure_arr=np.array([1.0, 2.0, 5.0, 1.0, 2.0, 5.0]), exposure_param_index=1)
    model, td, prior, fixed = J._daisy_chain_svgd_model(
        observed_indices=oi, epoch_starts=EPOCHS, user_prior=None,
        user_fixed=None, exact_final_grad=efg, **kw)
    return _ll(model)

def island():
    return two_island_graph(4)

TIMES = jnp.array([0.3, 0.8, 1.5, 2.5])

def site_pmf_from_graph():
    m = Graph.pmf_from_graph(island(), theta_dim=2)
    return lambda th: jnp.sum(jnp.log(m(th, TIMES)))

def site_pmf_and_moments(exact=False):
    m = Graph.pmf_and_moments_from_graph(island(), nr_moments=2, theta_dim=2,
                                         exact_moment_grad=exact)
    def f(th):
        p, mom = m(th, TIMES)
        return jnp.sum(jnp.log(p))
    return f

def site_pmf_and_moments_mom(exact=False):
    m = Graph.pmf_and_moments_from_graph(island(), nr_moments=2, theta_dim=2,
                                         exact_moment_grad=exact)
    def f(th):
        p, mom = m(th, TIMES)
        return jnp.sum(mom)
    return f

def site_moments_from_graph():
    m = Graph.moments_from_graph(island(), nr_moments=2)
    return lambda th: jnp.sum(m(th))

def site_rvp():
    g = island()
    r = np.zeros(g.vertices_length()); r[2] = 1.0; r[3] = 1.0
    return lambda th: g.reward_visit_probability(r, theta=th)
