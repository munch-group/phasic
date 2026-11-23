"""Bug 3 scoping on unmodified code: for each way of giving a BetaPrior, what
transform does the MODEL get, what transform does the PRIOR use, and do the
results stay in (0, 1)?  Fresh model object per fit (bug 1)."""
import sys, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph, GaussPrior, BetaPrior, SVGD
from phasic.mcmc import MCMC
print('phasic from', phasic.__file__)
d = np.array([0.5, 1.0, 1.5, 2.0])
def make_spy(dim):
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.ones_like(times), jnp.zeros(2)
    return spy, seen
def beta(): return BetaPrior(ci=(0.05, 0.95))
def report(label, s, seen, priors, dim):
    phi = jnp.full((dim,), 5.0)
    pt = None if s.param_transform is None else np.asarray(s.param_transform(phi))
    pr = [None if (p is None or getattr(p, '_transform', None) is None) else float(np.asarray(p._transform(jnp.array([5.0])))[0]) for p in priors]
    allv = np.concatenate(seen) if seen else np.zeros((0, dim))
    rng = (allv.min(axis=0).round(4).tolist(), allv.max(axis=0).round(4).tolist()) if len(allv) else None
    print(f'  {label}: param_transform(phi=5) = {None if pt is None else pt.round(4).tolist()}; prior._transform(5) = {pr}; model saw theta in {rng}')
print('## SVGD')
for label, mk, dim, kw in [
    ('single BetaPrior, dim 1', lambda: beta(), 1, {}),
    ('single BetaPrior, dim 2', lambda: beta(), 2, {}),
    ('[BetaPrior]', lambda: [beta()], 1, {}),
    ('[BetaPrior, BetaPrior]', lambda: [beta(), beta()], 2, {}),
    ('[GaussPrior, BetaPrior]', lambda: [GaussPrior(ci=[0.5, 5], lower=0.0), beta()], 2, {}),
    ('single BetaPrior, positive_params=False', lambda: beta(), 1, dict(positive_params=False)),
    ('[BetaPrior], positive_params=False', lambda: [beta()], 1, dict(positive_params=False)),
]:
    spy, seen = make_spy(dim); pr = mk()
    try:
        s = SVGD(model=spy, observed_data=d, theta_dim=dim, prior=pr, n_particles=8, n_iterations=3,
                 progress=False, seed=1, learning_rate=0.5, **kw)
        s.optimize(); report(label, s, seen, pr if isinstance(pr, list) else [pr], dim)
    except Exception as e:
        print(f'  {label}: {type(e).__name__}: {str(e)[:170]}')
print('## MCMC')
for label, mk, dim in [('single BetaPrior, dim 1', lambda: beta(), 1), ('[BetaPrior]', lambda: [beta()], 1),
                       ('[GaussPrior, BetaPrior]', lambda: [GaussPrior(ci=[0.5, 5], lower=0.0), beta()], 2)]:
    pr = mk(); seen = []
    def lp(theta, seen=seen):
        jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return 0.0 * jnp.sum(theta)
    try:
        m = MCMC(log_prob_fn=lp, prior=pr, theta_dim=dim, n_samples=400, n_chains=2, burn_in=50,
                 proposal_scale=1.0, positive_params=True, verbose=False, progress=False, seed=0)
        m.run(); r = m.get_results()
        allv = np.concatenate(seen)
        P = np.asarray(r['particles'])
        print(f'  {label}: model saw theta in ({allv.min(axis=0).round(4).tolist()}, {allv.max(axis=0).round(4).tolist()}); '
              f'samples min {P.min(axis=0).round(4).tolist()} max {P.max(axis=0).round(4).tolist()} mean {P.mean(axis=0).round(4).tolist()} (Beta(ci 0.05-0.95) mean 0.5)')
    except Exception as e:
        print(f'  {label}: {type(e).__name__}: {str(e)[:170]}')
