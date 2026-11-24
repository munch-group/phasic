"""(a) binary mask under a custom transform / MCMC; (b) LRT shared-fixed-value guard under the prototype fix."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from r2_lib import *
from phasic.mcmc import MCMC
import phasic.model_selection as ms
d = np.array([0.5, 1.0, 1.5, 2.0])
def spy_pair(mcmc=False):
    seen = []
    rec = lambda t: seen.append(np.asarray(t).reshape(-1, 2)[:, 1].copy())
    if mcmc:
        def spy(theta, data): jax.debug.callback(rec, theta); return jnp.ones_like(data)
    else:
        def spy(theta, times, rewards=None): jax.debug.callback(rec, theta); return jnp.ones_like(times), jnp.zeros(2)
    return spy, seen
print('## (a) binary mask [0,1] ("fix at exactly 1.0")')
for label, kw in (('svgd positive', dict(prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None])),
                  ('svgd exp transform', dict(positive_params=False, param_transform=lambda p: jnp.exp(p), prior=[GaussPrior(ci=[0.5, 5]), None])),
                  ('svgd no transform', dict(positive_params=False, prior=[GaussPrior(ci=[0.5, 5]), None]))):
    spy, seen = spy_pair()
    s = SVGD(model=spy, observed_data=d, theta_dim=2, fixed=[0, 1], n_particles=4, n_iterations=1, progress=False, seed=1, **kw); s.optimize()
    print(f'  {label}: model saw {sorted({float(x) for a in seen for x in a})}')
for label, kw in (('mcmc positive', dict(prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None])),
                  ('mcmc exp transform', dict(positive_params=False, param_transform=lambda p: jnp.exp(p), prior=[GaussPrior(ci=[0.5, 5]), None]))):
    spy, seen = spy_pair(True)
    try:
        m = MCMC(model=spy, observed_data=d, theta_dim=2, fixed=[0, 1], n_samples=10, n_chains=2, burn_in=2, verbose=False, seed=1, **kw); m.run()
        print(f'  {label}: model saw {sorted({float(x) for a in seen for x in a})}')
    except Exception as e: print(f'  {label}: RAISED {type(e).__name__}: {str(e)[:150]}')
print('## (b) LRT guard "fixed in both models but at different values" (model_selection._check_nested_fixed reads fixed_values)')
def mk(fixed, patched):
    s = SVGD(model=full_model(), observed_data=data, theta_dim=2, fixed=fixed, n_particles=4, n_iterations=1, progress=False, seed=1,
             prior=[None if i in dict(fixed) else GaussPrior(ci=[0.5, 5], lower=0.0) for i in range(2)])
    return apply_fix(s, fixed) if patched else s
for patched in (False, True):
    full = mk([(1, 2.0)], patched); nested = mk([(1, 3.0)], patched)
    # escalate nested to a strict superset, exactly as tests/pytest/inference/test_model_selection.py:551-557 does
    nested.fixed_mask = jnp.asarray([1.0, 1.0]); nv = np.asarray(nested.fixed_values).copy(); nv[0] = 0.5; nested.fixed_values = jnp.asarray(nv)
    try:
        ms._check_nested_fixed(full, nested, True); print(f'  patched={patched}: NO ERROR (mismatch 2.0 vs 3.0 not detected)')
    except Exception as e: print(f'  patched={patched}: RAISED {type(e).__name__}: {str(e)[:120]}')
