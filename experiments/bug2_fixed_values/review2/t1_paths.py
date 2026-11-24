"""Run a fit through each SVGD path with the prototype wrapper; report what the
model saw at the fixed slot and whether anything broke."""
import sys, warnings, traceback
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/review2')
import numpy as np, jax, jax.numpy as jnp
import proto
from phasic import Graph, GaussPrior, BetaPrior, SVGD
d = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 0.8, 1.2, 0.6])
def fit(label, mode, fixed, v_slot, post=True, theta_dim=2, graph=None, **kw):
    proto.install(mode); proto.clear()
    g = graph or proto.graph3()
    base = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=theta_dim)
    spy, seen = proto.spy_model(base)
    pos = kw.get('positive_params', True)
    kw.setdefault('prior', [None if i == v_slot else GaussPrior(ci=[0.5, 5], lower=0.0 if pos else None) for i in range(theta_dim)])
    try:
        s = SVGD(model=spy, observed_data=d, theta_dim=theta_dim, fixed=fixed, n_particles=6,
                 n_iterations=3, progress=False, seed=1, **kw)
        s.optimize(return_history=True)
        vals = proto.slot_values(seen, v_slot)
        r = s.get_results()
        out = [f'saw={vals[:3]}{"..." if len(vals)>3 else ""}',
               f'mean={float(np.asarray(r["theta_mean"])[v_slot])!r}',
               f'std={float(np.asarray(r["theta_std"])[v_slot])!r}',
               f'hist={sorted(set(np.asarray(r["history"])[..., v_slot].ravel().tolist()))[:2]}',
               f'free_finite={bool(np.all(np.isfinite(np.asarray(r["particles"]))))}']
        print(f'  [{mode}] {label}: ' + '; '.join(out))
        shapes = sorted(set(proto.CALLS)); print(f'      wrapper input shapes: {shapes}')
        if post:
            for name, fn in [('map_from_particles', lambda: s.map_estimate_from_particles()[0]),
                             ('map_with_opt', lambda: s.map_estimate_with_optimization(n_steps=3)[0]),
                             ('loglik', lambda: s.log_likelihood()),
                             ('loglik_refine', lambda: s.log_likelihood(refine=True)),
                             ('hpd', lambda: [np.asarray(x)[v_slot] for x in s.hpd_intervals()]),
                             ('theta_mean_attr', lambda: float(np.asarray(s.theta_mean)[v_slot]))]:
                n0 = len(seen)
                try:
                    res = fn()
                    saw = proto.slot_values(seen[n0:], v_slot) if len(seen) > n0 else []
                    print(f'      {name}: {res}  model_saw={saw[:3]}')
                except Exception as e:
                    print(f'      {name}: RAISED {type(e).__name__}: {str(e)[:200]}')
        return s
    except Exception as e:
        print(f'  [{mode}] {label}: RAISED {type(e).__name__}: {str(e)[:300]}')
        return None
    finally:
        proto.uninstall()

print('## A. default softplus, jacobian preconditioner (default)')
for v in (0.0, 1e-12, 0.3, 800.0):
    fit(f'fixed=(1,{v:g})', 'plan', [(1, v)], 1, post=(v == 0.3))
print('## A2. same with STRICT wrapper (what breaks if the implementer forgets a call site)')
fit('fixed=(1,0.3) strict', 'strict', [(1, 0.3)], 1, post=False)
print('## A3. NAIVE wrapper, no shape check (silent misbehaviour?)')
fit('fixed=(1,0.3) naive', 'naive', [(1, 0.3)], 1, post=False)
print('## B. preconditioners')
for pc in ('fisher', 'none'):
    fit(f'preconditioner={pc}', 'plan', [(1, 0.3)], 1, post=False, preconditioner=pc)
    fit(f'preconditioner={pc} strict', 'strict', [(1, 0.3)], 1, post=False, preconditioner=pc)
print('## C. regularization > 0')
fit('regularization=1', 'plan', [(1, 0.3)], 1, post=False, regularization=1.0, nr_moments=2)
print('## D. composite transform [Beta, None] and [None, Beta]')
fit('[Beta, None] fixed=(1,0.25)', 'plan', [(1, 0.25)], 1, prior=[BetaPrior(ci=(0.05, 0.95)), None])
fit('[None, Beta] fixed=(0,0.25)', 'plan', [(0, 0.25)], 0, prior=[None, BetaPrior(ci=(0.05, 0.95))], post=False)
fit('[None, Beta] fixed=(0,0.25) strict', 'strict', [(0, 0.25)], 0, prior=[None, BetaPrior(ci=(0.05, 0.95))], post=False)
print('## E. custom transform exp')
fit('exp fixed=(1,2.0)', 'plan', [(1, 2.0)], 1, positive_params=False, param_transform=lambda p: jnp.exp(p),
    prior=[GaussPrior(ci=[0.5, 5]), None])
fit('exp fixed=(1,0.0)', 'plan', [(1, 0.0)], 1, positive_params=False, param_transform=lambda p: jnp.exp(p),
    prior=[GaussPrior(ci=[0.5, 5]), None], post=False)
print('## F. single (broadcast) prior, fixed slot')
fit('single GaussPrior fixed=(1,0.3)', 'plan', [(1, 0.3)], 1, prior=GaussPrior(ci=[0.5, 5], lower=0.0), post=False)
print('## G. binary mask')
fit('mask [0,1]', 'plan', [0, 1], 1, post=False)
print('## H. no transform (positive_params=False)')
fit('no transform fixed=(1,0.3)', 'plan', [(1, 0.3)], 1, positive_params=False, post=False)
