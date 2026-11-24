"""Review 3: minimal versions of plan tests 1, 2, 5, 6 on the UNMODIFIED build.
Prints PASS/FAIL per row (FAIL today = the test has power)."""
import warnings, math
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
import phasic
from phasic import GaussPrior, BetaPrior, SVGD
from phasic.mcmc import MCMC
print('phasic from', phasic.__file__)
d = np.array([0.5, 1.0, 1.5, 2.0])

def make_spy(mcmc=False):
    seen = []
    def rec(t): seen.append(np.asarray(t).copy())
    if mcmc:
        def spy(theta, data):
            jax.debug.callback(rec, theta)
            return jnp.ones_like(data) * (1.0 + 0.0 * theta[0])
    else:
        def spy(theta, times, rewards=None):
            jax.debug.callback(rec, theta)
            return jnp.ones_like(times), jnp.zeros(2)
    return spy, seen

def slot(seen):
    return sorted({float(x.reshape(-1, x.shape[-1])[i, 1]) for x in seen for i in range(x.reshape(-1, x.shape[-1]).shape[0])})

def bit(a, v):
    a = np.asarray(a, dtype=np.float64).ravel()
    return bool(np.all(a == v)) and not (v == 0 and np.any(np.signbit(a)))

def svgd_row(label, v, n_particles=4, **kw):
    spy, seen = make_spy()
    pos = kw.get('positive_params', True)
    kw.setdefault('prior', [GaussPrior(ci=[0.5, 5], lower=0.0 if pos else None), None])
    try:
        s = SVGD(model=spy, observed_data=d, theta_dim=2, fixed=[(1, v)], n_particles=n_particles,
                 n_iterations=2, progress=False, seed=1, **kw)
        s.optimize(return_history=True)
        r = s.get_results()
        checks = dict(
            model=all(x == v for x in slot(seen)) and len(seen) > 0,
            theta_mean=bit(np.asarray(r['theta_mean'])[1], v),
            attr_mean=bit(np.asarray(s.theta_mean)[1], v),
            particles=bit(np.asarray(r['particles'])[:, 1], v),
            hpd=bit(r['hpd_lower'][1], v) and bit(r['hpd_upper'][1], v),
            std0=float(np.asarray(r['theta_std'])[1]) == 0.0,
        )
        if r.get('history') is not None:
            checks['history'] = bit(np.asarray(r['history'])[..., 1], v)
        bad = [k for k, ok in checks.items() if not ok]
        print(f'  {label:42s} {"PASS" if not bad else "FAIL"}  failing={bad}  model_saw={slot(seen)[:2]} n_evals={len(seen)}')
    except Exception as e:
        print(f'  {label:42s} RAISED {type(e).__name__}: {str(e)[:140]}')

def mcmc_row(label, v, **kw):
    spy, seen = make_spy(mcmc=True)
    pos = kw.get('positive_params', True)
    kw.setdefault('prior', [GaussPrior(ci=[0.5, 5], lower=0.0 if pos else None), None])
    try:
        m = MCMC(model=spy, observed_data=d, theta_dim=2, fixed=[(1, v)], n_samples=30, n_chains=2,
                 burn_in=5, verbose=False, seed=1, **kw)
        m.run()
        r = m.get_results()
        checks = dict(
            model=all(x == v for x in slot(seen)) and len(seen) > 0,
            theta_mean=bit(np.asarray(r['theta_mean'])[1], v),
            particles=bit(np.asarray(r['particles'])[:, 1], v),
            hpd=bit(r['hpd_lower'][1], v) and bit(r['hpd_upper'][1], v),
            std0=float(np.asarray(r['theta_std'])[1]) == 0.0,
        )
        bad = [k for k, ok in checks.items() if not ok]
        print(f'  {label:42s} {"PASS" if not bad else "FAIL"}  failing={bad}  model_saw={slot(seen)[:2]} n_evals={len(seen)}')
    except Exception as e:
        print(f'  {label:42s} RAISED {type(e).__name__}: {str(e)[:140]}')

VALS = (0.0, 1e-12, 1e-10, 1e-9, 1e-8, 0.3, 2.0, 50.0, 800.0, 1e6)
print('## TEST 1  SVGD positive_params')
for v in VALS: svgd_row(f'svgd fixed=(1,{v:g})', v)
print('## TEST 1b same, n_particles=7 (mean of constants)')
for v in (0.3, 2.0, 50.0): svgd_row(f'svgd n=7 fixed=(1,{v:g})', v, n_particles=7)
print('## TEST 2  custom transform exp / BetaPrior list')
for v in (2.0, 0.0, 0.3):
    svgd_row(f'svgd exp fixed=(1,{v:g})', v, positive_params=False,
             param_transform=lambda phi: jnp.exp(phi), prior=[GaussPrior(ci=[0.5, 5]), None])
for v in (0.25, 0.5, 0.3):
    svgd_row(f'svgd [Beta,None] fixed=(1,{v:g})', v, prior=[BetaPrior(ci=(0.05, 0.95)), None])
print('## TEST 3  positive_params=False, no transform (guard)')
for v in (0.0, -0.5, 0.3):
    svgd_row(f'svgd nopos fixed=(1,{v:g})', v, positive_params=False, prior=[GaussPrior(ci=[0.5, 5]), None])
print('## TEST 5  MCMC')
for v in VALS: mcmc_row(f'mcmc fixed=(1,{v:g})', v)
for v in (2.0, 0.0):
    mcmc_row(f'mcmc exp fixed=(1,{v:g})', v, positive_params=False,
             param_transform=lambda phi: jnp.exp(phi), prior=[GaussPrior(ci=[0.5, 5]), None])
mcmc_row('mcmc [Beta,None] fixed=(1,0.25)', 0.25, prior=[BetaPrior(ci=(0.05, 0.95)), None])
mcmc_row('mcmc nopos fixed=(1,-0.5)', -0.5, positive_params=False, prior=[GaussPrior(ci=[0.5, 5]), None])
print('## TEST 6  validation (expect raises AFTER fix)')
for cls, name in ((SVGD, 'svgd'), (MCMC, 'mcmc')):
    for v in (-0.5, float('nan'), float('inf'), -float('inf')):
        spy, seen = make_spy(mcmc=(cls is MCMC))
        try:
            kw = dict(n_particles=4, n_iterations=1, progress=False) if cls is SVGD else dict(n_samples=5, n_chains=2, burn_in=1, verbose=False)
            cls(model=spy, observed_data=d, theta_dim=2, fixed=[(1, v)], seed=1,
                prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None], **kw)
            print(f'  {name} fixed=(1,{v}) constructed: NO ERROR (test would FAIL today)')
        except Exception as e:
            print(f'  {name} fixed=(1,{v}) RAISED {type(e).__name__}: {str(e)[:100]}')
print('## mean-of-constants check (numpy / jax), is mean(full(n, v)) == v and std == 0?')
for v in (0.3, 0.1, 1e-8, 1.2e-4, 0.7):
    badn = [n for n in range(2, 201) if not (np.full(n, v).mean() == v and np.full(n, v).std() == 0.0)]
    badj = [n for n in range(2, 201) if not (float(jnp.mean(jnp.full(n, v))) == v and float(jnp.std(jnp.full(n, v))) == 0.0)]
    print(f'  v={v:g}: numpy fails for {len(badn)} of 199 n (e.g. {badn[:6]}); jax fails for {len(badj)} (e.g. {badj[:6]})')
