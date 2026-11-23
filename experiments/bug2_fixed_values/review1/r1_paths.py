"""Review 1 (diagnosis completeness): paths by which a fixed value reaches a
model, on UNMODIFIED code. Fresh spy model per fit unless a section says
otherwise (section E3 deliberately reuses one)."""
import sys, warnings, math
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import phasic
from phasic import GaussPrior, SVGD
from phasic.mcmc import MCMC
print('phasic from', phasic.__file__)
d = np.array([0.5, 1.0, 1.5, 2.0])

def make_spy(dim=2):
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t))[0].copy()), theta)
        return jnp.ones_like(times), jnp.zeros(2)
    return spy, seen
def P(dim, free, pos=True):
    return [GaussPrior(ci=[0.5, 5], lower=0.0 if pos else None) if i in free else None for i in range(dim)]
def vals(seen, k): return sorted({float(x[k]) for x in seen})

print('\n## E1 reproduce plan rows (positive_params)')
for v in (0.0, 1e-10, 800.0, 1e6):
    spy, seen = make_spy()
    s = SVGD(model=spy, observed_data=d, theta_dim=2, fixed=[(1, v)], prior=P(2, {0}),
             n_particles=4, n_iterations=1, progress=False, seed=1); s.optimize()
    print(f'  fixed=(1,{v:g}) -> model saw {vals(seen,1)}')

print('\n## E2 how often is the softplus round trip NOT bit-exact (pure math, as in svgd.py:6172 + transform)')
rng = np.random.default_rng(0)
for lo, hi in ((1e-8, 1e-3), (1e-3, 1.0), (1.0, 30.0), (30.0, 700.0)):
    v = np.exp(rng.uniform(np.log(lo), np.log(hi), 2000))
    phi = np.array([float(jnp.log(jnp.exp(x) - 1)) for x in v[:400]])
    back = np.asarray(jnp.maximum(jax.nn.softplus(jnp.asarray(phi)), 1e-9))
    rel = np.abs(back - v[:400]) / v[:400]
    print(f'  v in [{lo:g},{hi:g}]: not bit-exact {np.mean(back != v[:400])*100:.0f}%  max rel err {rel.max():.2e}')

print('\n## E3 SAME model object, two fits with different fixed values / sets (bug-1 cache is live)')
spy, seen = make_spy()
s1 = SVGD(model=spy, observed_data=d, theta_dim=2, fixed=[(1, 2.0)], prior=P(2, {0}),
          n_particles=4, n_iterations=2, progress=False, seed=1); s1.optimize()
print('  fit1 fixed=(1,2.0): saw', vals(seen, 1)); seen.clear()
s2 = SVGD(model=spy, observed_data=d, theta_dim=2, fixed=[(1, 5.0)], prior=P(2, {0}),
          n_particles=4, n_iterations=2, progress=False, seed=1)
seen.clear(); s2.optimize()
print('  fit2 fixed=(1,5.0), same model object, during optimize: saw', vals(seen, 1),
      '| cache entries', len(SVGD._compiled_cache))
spy3, seen3 = make_spy(3)
a = SVGD(model=spy3, observed_data=d, theta_dim=3, fixed=[(2, 2.0)], prior=P(3, {0, 1}),
         n_particles=4, n_iterations=2, progress=False, seed=1); a.optimize(); seen3.clear()
b = SVGD(model=spy3, observed_data=d, theta_dim=3, fixed=[(2, 2.0), (1, 7.0)], prior=P(3, {0}),
         n_particles=4, n_iterations=2, progress=False, seed=1)
seen3.clear(); b.optimize()
print('  nested fit on same model, extra fixed (1,7.0): theta[1] seen', vals(seen3, 1), ' theta[2] seen', vals(seen3, 2))

print('\n## E4 user-supplied theta_init + fixed')
spy, seen = make_spy()
ti = np.full((4, 2), 3.0); ti[:, 0] = [0.5, 1.0, 1.5, 2.0]
s = SVGD(model=spy, observed_data=d, theta_init=ti, theta_dim=2, n_particles=4, fixed=[(1, 0.25)], prior=P(2, {0}),
         n_iterations=1, progress=False, seed=1)
print('  after __init__ (smoke call): theta[1] seen', vals(seen, 1), '(softplus(3.0)=%.6f)' % float(jax.nn.softplus(3.0)))
seen.clear(); s.optimize(return_history=True)
r = s.get_results()
print('  during optimize: theta[1] seen', vals(seen, 1))
print('  history[0][:,1] =', np.asarray(r['history'])[0][:, 1], ' history[-1][:,1] =', np.asarray(r['history'])[-1][:, 1])

print('\n## E5 MCMC')
for v in (0.0, 1e-10, 0.3, 800.0):
    spy, seen = make_spy()
    try:
        m = MCMC(model=spy, observed_data=d, theta_dim=2, fixed=[(1, v)], prior=P(2, {0}),
                 n_samples=30, n_chains=2, burn_in=5, progress=False, seed=1, verbose=False); m.run()
        r = m.get_results()
        print(f'  fixed=(1,{v:g}) -> model saw {vals(seen,1)}; theta_mean[1]={float(r["theta_mean"][1])!r} theta_std[1]={float(r["theta_std"][1])!r}')
    except Exception as e:
        print(f'  fixed=(1,{v:g}) -> {type(e).__name__}: {str(e)[:150]}')

print('\n## E6 is mean/std of a constant column bit-exact? (plan test 1: theta_mean == v, theta_std == 0)')
for v in (0.3, 0.1, 1e-10, 1.2e-4, 2.0):
    bad_j = [n for n in range(2, 201) if float(jnp.full((n,), v).mean()) != v]
    bad_n = [n for n in range(2, 201) if float(np.full((n,), v).mean()) != v]
    sd_j = [n for n in range(2, 201) if float(jnp.full((n,), v).std()) != 0.0]
    sd_n = [n for n in range(2, 2001) if float(np.full((n,), v).std()) != 0.0]
    print(f'  v={v:g}: jnp.mean != v for {len(bad_j)}/199 sizes (e.g. {bad_j[:5]}); np.mean != v for {len(bad_n)}/199 (e.g. {bad_n[:5]}); '
          f'jnp.std != 0 for {len(sd_j)}/199; np.std != 0 for {len(sd_n)}/1999')

print('\n## E7 plot_pairwise_hdr re-inverts the constrained mean with _inverse_softplus (floor 1e-6)')
for v in (1e-8, 0.0, 2.0):
    spy, seen = make_spy(3)
    s = SVGD(model=spy, observed_data=d, theta_dim=3, fixed=[(2, v)], prior=P(3, {0, 1}),
             n_particles=6, n_iterations=1, progress=False, seed=1); s.optimize()
    during = vals(seen, 2); seen.clear()
    try:
        s.plot_pairwise_hdr(n=4, center_map=False)
        print(f'  fixed=(2,{v:g}): fit saw {during}; plot_pairwise_hdr evaluated the model at theta[2] = {vals(seen, 2)}')
    except Exception as e:
        print(f'  fixed=(2,{v:g}): fit saw {during}; plot -> {type(e).__name__}: {str(e)[:200]}')

print('\n## E8 model_selection._check_nested_fixed reads fixed_values in PHI space')
from phasic.model_selection import _check_nested_fixed
def fit(fixed, free):
    spy, _ = make_spy(3)
    return SVGD(model=spy, observed_data=d, theta_dim=3, fixed=fixed, prior=P(3, free),
                n_particles=4, n_iterations=1, progress=False, seed=1)
def chk(label, full, nested):
    try: _check_nested_fixed(full, nested, True); print(f'  {label}: ACCEPTED')
    except Exception as e: print(f'  {label}: rejected ({str(e)[:90]})')
chk('today, shared slot 2 at 2.0 vs 3.0', fit([(2, 2.0)], {0, 1}), fit([(2, 3.0), (1, 1.0)], {0}))
chk('today, shared slot 2 at 800 vs 900 (both phi=inf)', fit([(2, 800.0)], {0, 1}), fit([(2, 900.0), (1, 1.0)], {0}))
chk('today, shared slot 2 at 1e-8 vs 1.00000001e-8 (rel 1e-8)', fit([(2, 1e-8)], {0, 1}), fit([(2, 1.00000001e-8), (1, 1.0)], {0}))
chk('today, shared slot 2 at 1e-3 vs 1.000001e-3 (rel 1e-6)', fit([(2, 1e-3)], {0, 1}), fit([(2, 1.000001e-3), (1, 1.0)], {0}))
f, n = fit([(2, 2.0)], {0, 1}), fit([(2, 3.0), (1, 1.0)], {0})
f.fixed_values = jnp.zeros(3); n.fixed_values = jnp.zeros(3)       # plan step 3: placeholder 0.0
chk('EMULATED plan step 3 (fixed_values := placeholder 0.0), 2.0 vs 3.0', f, n)
