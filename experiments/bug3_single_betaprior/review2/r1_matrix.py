"""Attack 1: run SVGD and MCMC through the prototype for the full matrix."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
import proto
warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
import phasic
from phasic import Graph, GaussPrior, BetaPrior, SVGD
from phasic.mcmc import MCMC
import phasic.svgd as S
print('patched:', hasattr(S, '_resolve_param_transform'), '| PROTO_OFF =', os.environ.get('PROTO_OFF'))
d = np.array([0.5, 1.0, 1.5, 2.0])
def make_spy(dim):
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.ones_like(times) * (1.0 + 0.0 * jnp.sum(theta)), jnp.zeros(2) + 0.0 * jnp.sum(theta)
    return spy, seen
def beta(): return BetaPrior(ci=(0.05, 0.95))
def gauss(): return GaussPrior(ci=[0.5, 5], lower=0.0)
def rng(seen, dim):
    if not seen: return None
    a = np.concatenate(seen); return (a.min(axis=0).round(5).tolist(), a.max(axis=0).round(5).tolist())
cases = [
 ('single Beta dim1', beta, 1, {}),
 ('single Beta dim2', beta, 2, {}),
 ('[Beta]', lambda: [beta()], 1, {}),
 ('[Gauss, Beta]', lambda: [gauss(), beta()], 2, {}),
 ('[None, Beta] fixed slot0=5.0', lambda: [None, beta()], 2, dict(fixed=[(0, 5.0)])),
 ('single Beta dim2 fixed slot0=5.0', beta, 2, dict(fixed=[(0, 5.0)])),
 ('single Beta dim2 fixed slot0=0.25', beta, 2, dict(fixed=[(0, 0.25)])),
 ('single Gauss dim2', gauss, 2, {}),
 ('no prior dim2', lambda: None, 2, {}),
 ('custom transform exp, Gauss(lower=None)', lambda: GaussPrior(ci=[0.5, 5]), 1, dict(positive_params=False, param_transform=jnp.exp)),
 ('custom transform exp, single Beta', beta, 1, dict(positive_params=False, param_transform=jnp.exp)),
 ('custom transform exp, [Gauss(None), Beta]', lambda: [GaussPrior(ci=[0.5, 5]), beta()], 2, dict(positive_params=False, param_transform=jnp.exp)),
 ('positive_params=False single Beta', beta, 1, dict(positive_params=False)),
 ('positive_params=False [Beta]', lambda: [beta()], 1, dict(positive_params=False)),
 ('single Beta, theta_init given, theta_dim=None', beta, None, dict(theta_init=np.zeros((8, 2)))),
 ('[Beta,Beta], theta_init given, theta_dim=None', lambda: [beta(), beta()], None, dict(theta_init=np.zeros((8, 2)))),
]
print('## SVGD')
for label, mk, dim, kw in cases:
    for precond in (None, 'none'):
        spy, seen = make_spy(dim or 2); pr = mk()
        try:
            s = SVGD(model=spy, observed_data=d, theta_dim=dim, prior=pr, n_particles=8, n_iterations=3,
                     progress=False, seed=1, learning_rate=0.5, preconditioner=precond, verbose=False, **kw)
            s.optimize(); r = s.get_results()
            pt = s.param_transform
            print(f'  {label} [precond={precond}]: OK; model saw {rng(seen, dim)}; has _per_dim_fns={getattr(pt, "_per_dim_fns", None) is not None}; '
                  f'theta_mean={np.asarray(r["theta_mean"]).round(4).tolist()}; init theta range '
                  f'{np.asarray(jnp.array([pt(p) for p in s.theta_init])).min(axis=0).round(4).tolist() if pt is not None else None}')
        except Exception as e:
            print(f'  {label} [precond={precond}]: {type(e).__name__}: {str(e)[:200]}')
print('## MCMC (log_prob_fn mode, sequential)')
for label, mk, dim, kw in cases:
    pr = mk(); seen = []
    def lp(theta, seen=seen):
        seen.append(np.atleast_2d(np.asarray(theta)).copy()); return 0.0 * jnp.sum(theta)
    try:
        kw2 = dict(kw); kw2.setdefault('positive_params', True)
        if 'theta_init' in kw2: kw2['theta_init'] = np.zeros((2, 2))
        m = MCMC(log_prob_fn=lp, prior=pr, theta_dim=dim, n_samples=300, n_chains=2, burn_in=50,
                 proposal_scale=1.0, verbose=False, progress=False, seed=0, **kw2)
        init = None if m.param_transform is None else np.asarray(jnp.array([m.param_transform(p) for p in m.theta_init])).round(4).tolist()
        m.run(); r = m.get_results(); P = np.asarray(r['particles'])
        print(f'  {label}: OK; init(theta)={init}; model saw {rng(seen, dim)}; samples min {P.min(axis=0).round(4).tolist()} max {P.max(axis=0).round(4).tolist()} mean {P.mean(axis=0).round(4).tolist()}')
    except Exception as e:
        print(f'  {label}: {type(e).__name__}: {str(e)[:200]}')
print('## MCMC model mode, parallel=vmap (jit) and sequential')
for label, mk, dim, kw in cases[:6]:
    for par in ('vmap', None):
        spy, seen = make_spy(dim); pr = mk()
        try:
            m = MCMC(model=spy, observed_data=d, prior=pr, theta_dim=dim, n_samples=200, n_chains=3, burn_in=50,
                     proposal_scale=1.0, verbose=False, progress=False, seed=0, parallel=par, **kw)
            m.run(); r = m.get_results(); P = np.asarray(r['particles'])
            print(f'  {label} [parallel={m.parallel}]: OK; model saw {rng(seen, dim)}; samples min {P.min(axis=0).round(4).tolist()} max {P.max(axis=0).round(4).tolist()}; rhat {np.asarray(r["rhat"]).round(2).tolist()}')
        except Exception as e:
            print(f'  {label} [parallel={par}]: {type(e).__name__}: {str(e)[:200]}')
