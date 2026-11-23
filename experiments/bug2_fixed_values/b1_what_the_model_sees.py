"""Bug 2 scoping on unmodified code: what value does the MODEL receive for a
fixed parameter, and what do the results report?"""
import sys, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
import phasic
from phasic import Graph, GaussPrior, BetaPrior, SVGD
print('phasic from', phasic.__file__)
def two():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); b.add_edge(c, [0.0, 1.0]); return g
base = Graph.pmf_and_moments_from_graph(two(), nr_moments=2, theta_dim=2)
def make_spy():
    # a FRESH model object per fit: reusing one object across fits picks up
    # the previous fit's compiled gradient (bug 1) and contaminates this table
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.asarray(t).copy()), theta)
        return jnp.ones_like(times), jnp.zeros(2)
    return spy, seen
d = np.array([0.5, 1.0, 1.5, 2.0])
def run(label, fixed, **kw):
    spy, seen = make_spy()
    kw.setdefault('prior', [GaussPrior(ci=[0.5, 5], lower=0.0 if kw.get('positive_params', True) else None), None])
    try:
        s = SVGD(model=spy, observed_data=d, theta_dim=2, fixed=fixed, n_particles=4,
                 n_iterations=1, progress=False, seed=1, **kw)
        s.optimize()
        vals = sorted({float(np.atleast_2d(x)[0, 1]) for x in seen})
        tm = float(np.asarray(s.get_results()['theta_mean'])[1])
        print(f'  {label}: model saw theta[1] = {vals[:2]}; theta_mean[1] = {tm!r}')
    except Exception as e:
        print(f'  {label}: {type(e).__name__}: {str(e)[:160]}')
print('## positive_params=True (default), tuple format')
for v in (0.0, 1e-12, 1e-10, 1e-9, 1e-8, 0.3, 2.0, 50.0, 800.0, -0.5):
    run(f'fixed=(1, {v:g})', [(1, v)])
print('## binary mask format [0, 1] (documented as "fix at 1.0")')
run('fixed=[0, 1]', [0, 1])
print('## custom param_transform = exp (positive_params=False)')
for v in (2.0, 0.0):
    run(f'fixed=(1, {v:g}), param_transform=exp', [(1, v)], positive_params=False,
        param_transform=lambda phi: jnp.exp(phi), prior=[GaussPrior(ci=[0.5, 5]), None])
print('## positive_params=False, no transform')
for v in (0.0, -0.5, 2.0):
    run(f'fixed=(1, {v:g})', [(1, v)], positive_params=False, prior=[GaussPrior(ci=[0.5, 5]), None])
print('## per-dimension transforms: BetaPrior on slot 0, slot 1 fixed')
run('fixed=(1, 0.25) with [BetaPrior, None]', [(1, 0.25)], prior=[BetaPrior(ci=(0.05, 0.95)), None])
