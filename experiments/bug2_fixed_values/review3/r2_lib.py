"""Review 3: (A) which fixed values round-trip exactly today; (B) real model at a
rate of exactly 0 / tiny; (C) prototype of the planned fix by runtime patching
(no source edit) to test plan test 7's bit-identity claim and the fixed-at-0 use case."""
import warnings, random
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph, GaussPrior, SVGD

def full_graph():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); c = g.find_or_create_vertex([2]); z = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0)
    a.add_edge(z, [1.0, 0.0])      # theta0
    a.add_edge(c, [0.0, 1.0])      # theta1  (the rate switched off)
    c.add_edge(z, [2.0, 0.0])      # 2*theta0
    return g
def reduced_graph():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); z = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(z, [1.0])
    return g
def full_model(): return Graph.pmf_and_moments_from_graph(full_graph(), nr_moments=2, theta_dim=2)
def red_model(): return Graph.pmf_and_moments_from_graph(reduced_graph(), nr_moments=2, theta_dim=1)

def apply_fix(s, fixed):
    """Runtime prototype of plan section 2 (wrapper + 0.0 placeholder)."""
    mask = np.zeros(s.theta_dim, bool); ft = np.zeros(s.theta_dim)
    for i, v in fixed: mask[i] = True; ft[i] = v
    base = s.param_transform; m = jnp.asarray(mask); f = jnp.asarray(ft)
    def with_fixed(phi):
        th = base(phi) if base is not None else phi
        if phi.shape[-1] != m.shape[0]:   # learnable-only vector (preconditioner): plan routes this to _free_transform
            return th
        return jnp.where(m, f, th)
    if getattr(base, '_per_dim_fns', None) is not None: with_fixed._per_dim_fns = base._per_dim_fns
    s._free_transform = base; s.param_transform = with_fixed
    s.fixed_values = jnp.where(m, 0.0, jnp.asarray(s.fixed_values))
    s.theta_init = jnp.where(m[None, :], 0.0, s.theta_init)
    return s

random.seed(3); np.random.seed(3)
data = np.random.exponential(1 / 1.5, size=40)
def fit(v, patched, n_iter=15, n_particles=8, model=None, theta_dim=2, **kw):
    kw.setdefault('prior', [GaussPrior(ci=[0.5, 5], lower=0.0), None])
    fx = [(1, v)] if theta_dim == 2 else None
    s = SVGD(model=model or full_model(), observed_data=data, theta_dim=theta_dim, fixed=fx,
             n_particles=n_particles, n_iterations=n_iter, progress=False, seed=1, **kw)
    if patched: apply_fix(s, fx)
    s.optimize()
    return s, np.asarray(s.get_results()['particles'])

